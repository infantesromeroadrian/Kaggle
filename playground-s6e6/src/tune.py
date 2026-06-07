"""Optuna hyperparameter search over the frozen LightGBM baseline (BL-011).

Cycle C5 (POC). Searches LightGBM hyperparameters to beat the frozen baseline
(0.96448 +/- 0.00029 balanced_accuracy) on the IDENTICAL frozen 5 folds, with a
statistically defensible margin. The target is 0.97; if the search cannot reach
it defensibly, this module reports that honestly rather than overfitting the CV.

Two-speed design (the central engineering decision, justified by a measured
benchmark — see docs/c5-poc/tuning-report.md):
  - SEARCH speed: each Optuna trial runs the 5-fold CV with CPU multi-threaded
    LightGBM (num_threads=-1) + per-fold MedianPruner + early stopping on the
    boosting rounds. This is ~30% faster per fold than single-thread and prunes
    bad trials after 1-2 folds, so 150 trials fit the wall-clock budget. Search
    does NOT need bit-exact determinism — it only needs to RANK configs.
  - REPORT number: the best trial is RE-EVALUATED in the deterministic regime
    (num_threads=1, deterministic=True, force_row_wise=True, seed=42) — exactly
    the regime the baseline was measured in — so the reported delta is a like-for
    -like comparison and is reproducible bit-for-bit.

Reproducibility of the SEARCH itself: TPESampler(seed=42) + a seeded study make
the sequence of sampled trials deterministic given the same objective values.
MedianPruner stops trials whose intermediate (per-fold) score lags the running
median, so we never pay for the remaining folds of a clearly-losing trial.

Statistics (for the statistical review): the best config's 5 per-fold scores are
compared to the baseline's per-fold scores PAIRED by fold. We reuse
feature_eval.attach_paired_stats: d_k = best_k - baseline_k, paired_diff_mean =
mean(d), paired_diff_sem = std(d, ddof=1)/sqrt(5), and the delta is called
defensible iff paired_diff_mean > 2 * paired_diff_sem (the same conservative
2-sigma-on-5-folds rule the C3 ablation used).

Leakage: tuning reads TRAIN folds only; sample weights are computed on each
fold's train slice only; load_test is imported ONLY to build the final
submission AFTER the best config is fixed (never inside the CV loop).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Final

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from src.cv_utils import compute_sample_weights, load_folds
from src.data_loader import ID_COLUMN, TARGET, load_test, load_train
from src.feature_eval import (
    FeatureConfig,
    N_SPLITS,
    SEED,
    attach_paired_stats,
    build_model_frame,
    evaluate_config,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Layout + study contract.
# --------------------------------------------------------------------------- #
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RESULTS_DIR: Final[Path] = _REPO_ROOT / "docs" / "c5-poc"
STUDY_PATH: Final[Path] = RESULTS_DIR / "tuning_results.json"
DATA_DIR_OUT: Final[Path] = _REPO_ROOT / "data"
SUBMISSION_PATH: Final[Path] = DATA_DIR_OUT / "submission_c5_optuna.csv"
MLFLOW_DIR: Final[Path] = _REPO_ROOT / "mlruns"
MLFLOW_EXPERIMENT: Final[str] = "s6e6-c5-optuna"

# Budget. A measured trial costs ~1-3 min in the search regime (see the
# tuning-report cost section): one fold of 462k rows is ~16s multi-threaded, and
# the bounded estimator cap below stops a low-lr trial from training thousands of
# trees. MedianPruner abandons losing trials after 1-2 folds. So under the 45-min
# wall-clock cap roughly 30-60 trials complete; we request 60 and let the timeout
# bind first. 150 was the original spec but is NOT reachable on this host without
# subsampling the data (which would break baseline comparability) — documented in
# the report. Both n_trials AND timeout are honoured (Optuna stops at whichever
# hits first); the JSON records how many actually ran.
DEFAULT_N_TRIALS: Final[int] = 40
DEFAULT_TIMEOUT_S: Final[int] = 40 * 60

# Defensibility multiplier, identical to feature_eval._SIGMA_K (a ~2-sigma paired
# signal on 5 folds). Re-stated so the report cites a single threshold.
_SIGMA_K: Final[float] = 2.0

# Hard cap on boosting rounds; early stopping picks the real count per fold. 500
# brackets the baseline's 300 with head-room for a lower learning rate, while
# capping the worst-case per-fold wall-clock (an uncapped low-lr trial trained
# ~2000 trees and cost ~5 min/trial — too slow for the budget).
_MAX_ESTIMATORS: Final[int] = 400
_EARLY_STOPPING_ROUNDS: Final[int] = 30

# Thread count for the SEARCH regime. NOT -1: a measured benchmark on this host
# under realistic background load (MCP servers etc.) showed num_threads=-1 is
# CATASTROPHIC (130s/fold from thread over-subscription on 22 logical cores),
# while num_threads=4 is the sweet spot (43s/fold) and 8 is in between (67s). So
# we pin a modest thread count that keeps LightGBM's histogram reduction cheap
# and leaves headroom for the OS. The final reported number still runs at
# num_threads=1 (deterministic regime) regardless of this value.
_SEARCH_THREADS: Final[int] = 4

# Fixed model params shared by every trial (objective + classes + reproducibility
# knobs). Only the suggested hyperparameters vary across trials.
_FIXED_PARAMS: Final[dict[str, object]] = {
    "objective": "multiclass",
    "num_class": 3,
    "random_state": SEED,
    "verbose": -1,
}


def suggest_params(trial: Any) -> dict[str, object]:
    """Sample one LightGBM hyperparameter configuration from the search space.

    Search space (the BL-011 spec). Ranges are chosen around the baseline
    (num_leaves 63, lr 0.05, 300 trees, 0.8 subsample/colsample, reg_lambda 1.0)
    so the baseline itself is reachable, while opening room to trade depth for
    regularisation:
      - num_leaves [15, 255]: model capacity (log scale).
      - max_depth {-1, 4..16}: -1 = unbounded (the baseline); bounded depths let
        the search find a shallower, better-regularised tree.
      - learning_rate [0.03, 0.2] (log): the lower bound is raised from the
        textbook 5e-3 so a low-lr trial cannot blow the per-fold wall-clock
        budget by training the full _MAX_ESTIMATORS cap; 0.03 still allows a
        slower, better-generalising schedule than the baseline's 0.05.
      - n_estimators: NOT sampled directly — capped at _MAX_ESTIMATORS and chosen
        per fold by early stopping (the principled way to size the ensemble).
      - min_child_samples [5, 300] (log): leaf-size floor; the main overfitting
        guard on 462k-row folds.
      - bagging_fraction [0.5, 1.0] + bagging_freq=1: row subsampling
        (LightGBM 'subsample').
      - feature_fraction [0.5, 1.0]: column subsampling ('colsample_bytree').
      - reg_alpha / reg_lambda [1e-3, 10.0] (log): L1 / L2 leaf-weight penalties.
      - min_split_gain [0, 0.5]: minimum loss reduction to split (complexity
        control).

    Args:
        trial: The Optuna trial to sample from.

    Returns:
        A dict of LightGBM kwargs (search-space params only; merged with
        _FIXED_PARAMS and the speed/determinism block by the caller).
    """
    max_depth = trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12, 16])
    return {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": max_depth,
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
    }


def _search_cv(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: dict[str, np.ndarray],
    cat_idx: list[int],
    params: dict[str, object],
    trial: Any | None = None,
) -> tuple[list[float], int]:
    """Run multi-threaded CV for one config, reporting to Optuna for pruning.

    SEARCH regime: num_threads=-1 (fast, non-deterministic ordering is fine for
    ranking), early stopping on an internal validation = the fold's own val slice
    (this is the standard LightGBM CV early-stopping; the val slice is the same
    one scored, so the rounds are chosen to maximise THIS fold's val metric — a
    mild optimism that applies EQUALLY to every trial and is removed in the final
    deterministic re-evaluation that uses the baseline's fixed 300 rounds).

    After each fold we report the running mean to the trial and check
    trial.should_prune(), so a clearly-losing config is abandoned early.

    Args:
        x: Full design matrix (engineered/raw columns already selected).
        y: Integer-encoded target aligned to x rows (positional).
        folds: Frozen fold index dict.
        cat_idx: Positional indices of categorical columns in x.
        params: Full LightGBM kwargs for this trial (fixed + suggested + speed).
        trial: Optuna trial for intermediate reporting/pruning, or None.

    Returns:
        (fold_scores, mean_best_iteration) — the per-fold balanced_accuracy and
        the rounded mean best_iteration_ across folds (the early-stopped ensemble
        size, logged for transparency).

    Raises:
        optuna.TrialPruned: If the trial is pruned by the configured pruner.
    """
    import optuna  # local import: optuna is a search-only dependency

    fold_scores: list[float] = []
    best_iters: list[int] = []
    for k in range(N_SPLITS):
        tr_idx = folds[f"fold{k}_train"]
        va_idx = folds[f"fold{k}_val"]
        x_tr, y_tr = x.iloc[tr_idx], y[tr_idx]
        x_va, y_va = x.iloc[va_idx], y[va_idx]

        weights = compute_sample_weights(y_tr)
        model = lgb.LGBMClassifier(n_estimators=_MAX_ESTIMATORS, **params)
        model.fit(
            x_tr,
            y_tr,
            sample_weight=weights,
            categorical_feature=cat_idx,
            eval_set=[(x_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        pred = model.predict(x_va)
        score = float(balanced_accuracy_score(y_va, pred))
        fold_scores.append(score)
        best_iters.append(int(model.best_iteration_ or _MAX_ESTIMATORS))

        if trial is not None:
            trial.report(float(np.mean(fold_scores)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return fold_scores, int(round(float(np.mean(best_iters))))


def _objective_factory(
    x: pd.DataFrame, y: np.ndarray, folds: dict[str, np.ndarray], cat_idx: list[int]
):
    """Build the Optuna objective closure bound to the data and folds.

    Returns a function trial -> mean balanced_accuracy (the quantity to MAXIMISE),
    which also records per-trial diagnostics (per-fold scores, mean best
    iteration) as trial user attributes and, if MLflow is available, as a nested
    child run.

    Args:
        x: Design matrix.
        y: Integer-encoded target.
        folds: Frozen folds.
        cat_idx: Categorical column indices.

    Returns:
        The objective callable for study.optimize.
    """

    def objective(trial: Any) -> float:
        suggested = suggest_params(trial)
        # Speed block for the search: a pinned modest thread count (see
        # _SEARCH_THREADS — -1 over-subscribes on this host), no determinism pins.
        params = {
            **_FIXED_PARAMS,
            **suggested,
            "num_threads": _SEARCH_THREADS,
            "n_jobs": _SEARCH_THREADS,
        }
        fold_scores, mean_best_iter = _search_cv(x, y, folds, cat_idx, params, trial)
        mean = float(np.mean(fold_scores))
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("std", float(np.std(fold_scores)))
        trial.set_user_attr("mean_best_iteration", mean_best_iter)
        _log_trial_mlflow(trial, suggested, fold_scores, mean, mean_best_iter)
        return mean

    return objective


def _log_trial_mlflow(
    trial: Any,
    params: dict[str, object],
    fold_scores: list[float],
    mean: float,
    mean_best_iter: int,
) -> None:
    """Best-effort nested MLflow child run for one trial (optional).

    Args:
        trial: The Optuna trial (for its number).
        params: The sampled hyperparameters.
        fold_scores: Per-fold balanced_accuracy.
        mean: Mean balanced_accuracy.
        mean_best_iter: Mean early-stopped boosting rounds.
    """
    try:
        import mlflow  # noqa: PLC0415
    except ImportError:
        return
    try:
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params({f"lgbm_{k}": v for k, v in params.items()})
            mlflow.log_metric("balanced_accuracy_mean", mean)
            mlflow.log_metric("mean_best_iteration", mean_best_iter)
            for k, s in enumerate(fold_scores):
                mlflow.log_metric("balanced_accuracy_fold", s, step=k)
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("MLflow trial log failed (%s); continuing", exc)


def deterministic_eval(
    df: pd.DataFrame,
    y: np.ndarray,
    folds: dict[str, np.ndarray],
    best_params: dict[str, object],
):
    """Re-evaluate the best config in the deterministic baseline regime.

    Builds the SAME raw-feature matrix as the baseline (FeatureConfig with no
    engineered features), then runs feature_eval.evaluate_config — which pins
    deterministic=True / force_row_wise=True / num_threads=1 / n_estimators from
    the params. We inject the best hyperparameters by monkey-patching the params
    used inside evaluate_config via a config-bound _lgbm_params override.

    We do NOT reuse the early-stopped per-fold round counts here: to be a strict
    like-for-like with the baseline we fix n_estimators to the rounded mean best
    iteration from the search, so every fold trains the same-size deterministic
    ensemble. This removes the per-fold early-stopping optimism present in the
    search regime.

    Args:
        df: Raw training frame.
        y: Integer-encoded target.
        folds: Frozen folds.
        best_params: The best trial's suggested hyperparameters PLUS an
            'n_estimators' key (the fixed deterministic ensemble size).

    Returns:
        A feature_eval.ConfigResult with the deterministic per-fold scores.
    """
    # Determinism block, identical to the baseline's _lgbm_params determinism.
    det_params = {
        **_FIXED_PARAMS,
        **best_params,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "n_jobs": 1,
    }

    # evaluate_config calls module-level _lgbm_params(); to reuse its proven CV
    # loop verbatim with our tuned params we temporarily bind a params provider.
    # This keeps a SINGLE CV implementation (no copy) — the comparison stays exact.
    import src.feature_eval as fe

    original = fe._lgbm_params
    fe._lgbm_params = lambda: det_params  # type: ignore[assignment]
    try:
        result = evaluate_config(df, y, folds, FeatureConfig(name="best_optuna"))
    finally:
        fe._lgbm_params = original  # type: ignore[assignment]
    return result


def build_submission(df_train: pd.DataFrame, y: np.ndarray, best_params: dict[str, object]) -> Path:
    """Train the best config on ALL of train and predict test -> submission CSV.

    The CV estimate sizes and validates the model; the deliverable model is fit
    on the FULL training set (no held-out fold) with the deterministic params and
    the fixed n_estimators chosen by the search. Predictions are hard labels
    decoded back to GALAXY/STAR/QSO via the same LabelEncoder fitted on train.

    Leakage note: load_test is called HERE and only here, AFTER the config is
    frozen. Test never enters CV, weighting, or tuning.

    Args:
        df_train: Raw training frame.
        y: Integer-encoded train target.
        best_params: Best hyperparameters + n_estimators (deterministic size).

    Returns:
        Path to the written submission CSV.
    """
    encoder = LabelEncoder()
    encoder.fit(df_train[TARGET].to_numpy())

    x_train, cat = build_model_frame(df_train, FeatureConfig(name="best_optuna"))
    cat_idx = [list(x_train.columns).index(c) for c in cat]
    weights = compute_sample_weights(y)

    det_params = {
        **_FIXED_PARAMS,
        **best_params,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "n_jobs": 1,
    }
    model = lgb.LGBMClassifier(**det_params)
    model.fit(x_train, y, sample_weight=weights, categorical_feature=cat_idx)

    df_test = load_test()
    x_test, _ = build_model_frame(df_test, FeatureConfig(name="best_optuna"))
    pred_int = model.predict(x_test)
    pred_labels = encoder.inverse_transform(pred_int)

    submission = pd.DataFrame({ID_COLUMN: df_test[ID_COLUMN].to_numpy(), TARGET: pred_labels})
    DATA_DIR_OUT.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    logger.info("Wrote submission %s (%d rows)", SUBMISSION_PATH, len(submission))
    return SUBMISSION_PATH


def run(
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    make_submission: bool = True,
) -> dict[str, object]:
    """Run the full Optuna study and the deterministic re-evaluation of the best.

    Steps:
      1. Load train, encode target, load frozen folds, build the raw matrix once.
      2. Run the TPE study (seed 42) with MedianPruner under an n_trials/timeout
         budget, maximising mean balanced_accuracy in the fast search regime.
      3. Re-evaluate the best trial deterministically (baseline regime) for the
         reported number, fixing n_estimators to the search's mean best iteration.
      4. Compute the paired-vs-baseline statistics (delta, SEM_diff, defensible).
      5. Optionally build the submission from the best config.
      6. Persist docs/c5-poc/tuning_results.json.

    Args:
        n_trials: Max Optuna trials.
        timeout_s: Max wall-clock seconds for the study (whichever hits first).
        make_submission: If True, train on all train and write the submission CSV.

    Returns:
        JSON-serialisable summary dict (also written to STUDY_PATH).
    """
    import optuna

    started = time.time()
    df = load_train()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].to_numpy())
    folds = load_folds()

    # Build the raw model matrix ONCE (every trial shares the same columns; only
    # hyperparameters vary). Categorical handling matches the baseline.
    x, cat = build_model_frame(df, FeatureConfig(name="baseline_raw"))
    cat_idx = [list(x.columns).index(c) for c in cat]

    # Baseline per-fold scores (deterministic) for the paired comparison. This is
    # the SAME config/regime baseline.py reports; recomputed here so the study is
    # self-contained and the paired deltas use matched folds.
    baseline_result = evaluate_config(df, y, folds, FeatureConfig(name="baseline_raw"))
    logger.info(
        "baseline (paired reference): mean=%.6f std=%.6f",
        baseline_result.mean,
        baseline_result.std,
    )

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner, study_name="s6e6-c5-optuna"
    )

    parent_run_id = _start_parent_mlflow()
    objective = _objective_factory(x, y, folds, cat_idx)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)
    _end_parent_mlflow()

    best = study.best_trial
    best_suggested = dict(best.params)
    mean_best_iter = int(best.user_attrs["mean_best_iteration"])
    logger.info(
        "best trial #%d search_mean=%.6f params=%s n_estimators(fixed)=%d",
        best.number,
        best.value,
        best_suggested,
        mean_best_iter,
    )

    # Deterministic re-evaluation of the best, with n_estimators fixed to the
    # search's mean best iteration so it is comparable to the baseline regime.
    det_params = {**best_suggested, "n_estimators": mean_best_iter}
    det_result = deterministic_eval(df, y, folds, det_params)
    attach_paired_stats(det_result, baseline_result.fold_scores)

    submission_path: str | None = None
    if make_submission:
        submission_path = str(build_submission(df, y, det_params))

    summary: dict[str, object] = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "metric": "balanced_accuracy",
        "sigma_k": _SIGMA_K,
        "n_trials_requested": n_trials,
        "n_trials_completed": len([t for t in study.trials if t.state.is_finished()]),
        "n_trials_pruned": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        ),
        "timeout_s": timeout_s,
        "elapsed_seconds": round(time.time() - started, 1),
        "baseline": {
            "fold_scores": baseline_result.fold_scores,
            "mean": baseline_result.mean,
            "std": baseline_result.std,
        },
        "best_trial_number": best.number,
        "best_search_mean": best.value,
        "best_params_suggested": best_suggested,
        "best_n_estimators_fixed": mean_best_iter,
        "best_deterministic": {
            "fold_scores": det_result.fold_scores,
            "mean": det_result.mean,
            "std": det_result.std,
            "paired_diff_mean": det_result.paired_diff_mean,
            "paired_diff_sem": det_result.paired_diff_sem,
            "defensible": det_result.defensible,
        },
        "beats_baseline_point": det_result.mean > baseline_result.mean,
        "reaches_target_0_97": det_result.mean >= 0.97,
        "mlflow_parent_run_id": parent_run_id,
        "submission_path": submission_path,
        "class_order": list(encoder.classes_),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", STUDY_PATH)
    return summary


def _start_parent_mlflow() -> str | None:
    """Open the parent MLflow run for the study (best-effort). Returns its id."""
    # MLflow 3.x rejects the local file store unless this opt-out is set; we use a
    # local mlruns/ dir by design (single-host POC), so set it here. setdefault so
    # an explicit caller override still wins.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow  # noqa: PLC0415
    except ImportError:
        logger.warning("mlflow not installed; skipping study tracking (JSON is the record)")
        return None
    try:
        mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        run = mlflow.start_run(run_name="optuna_study")
        return run.info.run_id
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("MLflow parent run failed (%s); continuing", exc)
        return None


def _end_parent_mlflow() -> None:
    """Close the parent MLflow run if one is open (best-effort)."""
    try:
        import mlflow  # noqa: PLC0415

        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:  # pragma: no cover - environment-dependent
        pass


def main() -> None:
    """CLI entry point: run the study under the configured budget and report."""
    parser = argparse.ArgumentParser(description="Optuna LightGBM HPO (BL-011).")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--no-submission",
        action="store_true",
        help="Skip building the test submission CSV (CV-only run).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    np.random.seed(SEED)
    summary = run(
        n_trials=args.n_trials,
        timeout_s=args.timeout_s,
        make_submission=not args.no_submission,
    )
    bd = summary["best_deterministic"]
    print(
        f"best trial #{summary['best_trial_number']}: "
        f"det_mean={bd['mean']:.6f} std={bd['std']:.6f} "
        f"delta={bd['paired_diff_mean']:+.6f} sem={bd['paired_diff_sem']:.6f} "
        f"defensible={bd['defensible']} reaches_0.97={summary['reaches_target_0_97']}"
    )


if __name__ == "__main__":
    main()
