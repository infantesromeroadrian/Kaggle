"""Optuna hyperparameter search for the C6 XGBoost base learner (BL-026).

Cycle C6 (Build), ensemble step 2. Tunes XGBoost to PARITY with the C5 tuned
LightGBM on the IDENTICAL frozen 5 folds, so the re-blend pits comparably-strong
members. Same contract as src.tune_catboost and src.tune: same folds, same
balanced train-slice weights, same raw feature matrix (categoricals handled
natively via enable_categorical), same TPESampler(seed=42) + MedianPruner engine,
same paired-defensibility threshold.

COST WARNING (the central budget decision): XGBoost runs on CPU (4 threads — the
C5 sweet spot; -1 over-subscribes the 22 logical cores and is catastrophic). A
measured fold with depth 8 / 800-iter cap costs ~35s, i.e. ~175s/trial for the
full 5 folds. That is ~10x CatBoost's GPU cost. So this tuner runs FEWER trials by
design: DEFAULT_N_TRIALS=15 with an aggressive MedianPruner (abandons a losing
trial after 1-2 folds) under a ~35-min timeout. TPE around a region that already
reaches parity on a single benchmark fold (0.96492 vs LightGBM 0.96487) does not
need 50 trials; it needs to confirm and slightly refine. The JSON records exactly
how many trials completed so the report is honest about the reduced budget.

Two-speed design (identical to src.tune / src.tune_catboost):
  - SEARCH: cap n_estimators at _MAX_ESTIMATORS, early-stop on the fold's own val
    slice, MedianPruner prunes losers. Determinism not required for ranking.
  - REPORT: re-evaluate the best with n_estimators fixed to the search's mean best
    iteration (no early stopping) for the like-for-like reported per-fold scores.

Leakage: TRAIN folds only; weights on each fold's train slice only; the val slice
is the early-stopping eval set + the scored hold-out, never fitted. No load_test.

Output: docs/c6-build/xgboost_tuning.json — best params + deterministic per-fold
scores + fixed n_estimators, consumed by ensemble.xgboost_params.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from src.cv_utils import compute_sample_weights, load_folds
from src.data_loader import TARGET, load_train
from src.ensemble import build_feature_frame

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Reproducibility + layout.
# --------------------------------------------------------------------------- #
SEED: Final[int] = 42
N_SPLITS: Final[int] = 5
N_CLASSES: Final[int] = 3

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RESULTS_DIR: Final[Path] = _REPO_ROOT / "docs" / "c6-build"
STUDY_PATH: Final[Path] = RESULTS_DIR / "xgboost_tuning.json"
MLFLOW_DIR: Final[Path] = _REPO_ROOT / "mlruns"
MLFLOW_EXPERIMENT: Final[str] = "s6e6-c6-xgboost-tuning"

# Budget — REDUCED on purpose (see module docstring): CPU XGBoost is ~10x CatBoost
# GPU per trial. 15 trials + a 35-min timeout + aggressive pruning is the envelope.
DEFAULT_N_TRIALS: Final[int] = 15
DEFAULT_TIMEOUT_S: Final[int] = 35 * 60

# Hard cap on boosting rounds; early stopping picks the real count per fold.
_MAX_ESTIMATORS: Final[int] = 800
_EARLY_STOPPING_ROUNDS: Final[int] = 30

# Thread count — the C5 sweet spot. NOT -1 (over-subscription is catastrophic here).
_CPU_THREADS: Final[int] = 4

# Defensibility multiplier, identical across cycles.
_SIGMA_K: Final[float] = 2.0

# Fixed params shared by every trial: the CPU hist MultiClass regime with native
# categoricals. Only the suggested hyperparameters vary across trials.
_FIXED_PARAMS: Final[dict[str, object]] = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_metric": "mlogloss",
    "random_state": SEED,
    "n_jobs": _CPU_THREADS,
    "verbosity": 0,
}


def suggest_params(trial: Any) -> dict[str, object]:
    """Sample one XGBoost hyperparameter configuration (BL-026 search space).

    Ranges per the task brief, centred on the benchmark config that already reached
    parity (depth 8, lr 0.05, subsample/colsample 0.8):
      - max_depth [4, 10]: level-wise tree depth.
      - learning_rate [0.02, 0.2] (log): slower schedules allowed within budget.
      - min_child_weight [1, 10] (log): minimum child Hessian sum; the leaf-size
        analogue, an overfit guard on 462k-row folds.
      - subsample [0.5, 1.0]: row subsampling per tree.
      - colsample_bytree [0.5, 1.0]: column subsampling per tree.
      - reg_alpha / reg_lambda [1e-3, 10] (log): L1 / L2 weight penalties.
      - gamma [0, 5]: minimum loss reduction to split (complexity control).
      - n_estimators: NOT sampled — capped at _MAX_ESTIMATORS and chosen per fold by
        early stopping.

    Args:
        trial: The Optuna trial to sample from.

    Returns:
        A dict of XGBoost kwargs (search-space only; merged with _FIXED_PARAMS and
        the estimator cap by the caller).
    """
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def _cv_scores(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: dict[str, np.ndarray],
    params: dict[str, object],
    *,
    n_estimators: int | None,
    early_stop: bool,
    trial: Any | None = None,
) -> tuple[list[float], int]:
    """Run 5-fold CV for one XGBoost config; report to Optuna for pruning.

    Two regimes, exactly as src.tune_catboost._cv_scores:
      - SEARCH (early_stop=True, n_estimators=None): cap at _MAX_ESTIMATORS, early-
        stop on the fold's own val slice, report+prune after each fold.
      - REPORT (early_stop=False, n_estimators=fixed): train exactly n_estimators
        rounds per fold, no early stopping, for the like-for-like reported number.

    The val slice is the early-stopping eval set AND the scored hold-out; it never
    enters the fit beyond round selection — the same uniform optimism every trial
    shares, removed in the REPORT regime.

    Args:
        x: Full design matrix (categoricals as 'category').
        y: Integer-encoded target aligned to x rows.
        folds: Frozen fold index dict.
        params: Full XGBoost kwargs (fixed + suggested).
        n_estimators: Fixed round count for REPORT, or None for SEARCH.
        early_stop: Whether to early-stop on the val slice (SEARCH only).
        trial: Optuna trial for reporting/pruning, or None.

    Returns:
        (fold_scores, mean_best_iteration).

    Raises:
        optuna.TrialPruned: If the trial is pruned.
    """
    import optuna
    import xgboost as xgb

    cap = _MAX_ESTIMATORS if n_estimators is None else n_estimators
    fold_scores: list[float] = []
    best_iters: list[int] = []

    for k in range(N_SPLITS):
        tr_idx = folds[f"fold{k}_train"]
        va_idx = folds[f"fold{k}_val"]
        x_tr, y_tr = x.iloc[tr_idx], y[tr_idx]
        x_va, y_va = x.iloc[va_idx], y[va_idx]

        w_tr = compute_sample_weights(y_tr)

        # early_stopping_rounds is a CONSTRUCTOR arg in XGBoost 2.x; we only set it
        # in the SEARCH regime, where an eval_set is supplied. In REPORT we omit it
        # (no eval_set) so the model trains the full fixed n_estimators.
        ctor = dict(params)
        if early_stop:
            ctor["early_stopping_rounds"] = _EARLY_STOPPING_ROUNDS
        model = xgb.XGBClassifier(n_estimators=cap, **ctor)

        if early_stop:
            model.fit(
                x_tr, y_tr, sample_weight=w_tr, eval_set=[(x_va, y_va)], verbose=False
            )
            best_iter = int(model.best_iteration) + 1  # best_iteration is 0-based
        else:
            model.fit(x_tr, y_tr, sample_weight=w_tr)
            best_iter = cap

        pred = model.predict(x_va)
        score = float(balanced_accuracy_score(y_va, pred))
        fold_scores.append(score)
        best_iters.append(best_iter)

        if trial is not None:
            trial.report(float(np.mean(fold_scores)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return fold_scores, int(round(float(np.mean(best_iters))))


def run(
    *, n_trials: int = DEFAULT_N_TRIALS, timeout_s: int = DEFAULT_TIMEOUT_S
) -> dict[str, object]:
    """Run the XGBoost Optuna study and the deterministic re-evaluation of the best.

    Steps mirror src.tune_catboost.run: load data + folds + matrix once, run the
    pruned TPE study under the (reduced) budget, re-evaluate the best with
    n_estimators fixed to the mean best iteration, persist the JSON.

    Args:
        n_trials: Max Optuna trials (default 15 — see the cost warning).
        timeout_s: Max wall-clock seconds (whichever hits first).

    Returns:
        JSON-serialisable summary dict (also written to STUDY_PATH).
    """
    import optuna

    started = time.time()
    df = load_train()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].to_numpy())
    folds = load_folds()
    x, _categorical = build_feature_frame(df)  # XGBoost reads 'category' dtype natively

    sampler = optuna.samplers.TPESampler(seed=SEED)
    # Aggressive pruning: fewer startup trials than CatBoost because the budget is
    # tighter, so we start pruning sooner.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner, study_name="s6e6-c6-xgboost"
    )

    parent_run_id = _start_parent_mlflow()

    def objective(trial: Any) -> float:
        suggested = suggest_params(trial)
        params = {**_FIXED_PARAMS, **suggested}
        fold_scores, mean_best_iter = _cv_scores(
            x, y, folds, params, n_estimators=None, early_stop=True, trial=trial
        )
        mean = float(np.mean(fold_scores))
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("mean_best_iteration", mean_best_iter)
        _log_trial_mlflow(trial, suggested, fold_scores, mean, mean_best_iter)
        return mean

    study.optimize(objective, n_trials=n_trials, timeout=timeout_s, show_progress_bar=False)
    _end_parent_mlflow()

    best = study.best_trial
    best_suggested = dict(best.params)
    mean_best_iter = int(best.user_attrs["mean_best_iteration"])
    logger.info(
        "best XGBoost trial #%d search_mean=%.6f n_estimators(fixed)=%d params=%s",
        best.number,
        best.value,
        mean_best_iter,
        best_suggested,
    )

    report_params = {**_FIXED_PARAMS, **best_suggested}
    report_scores, _ = _cv_scores(
        x, y, folds, report_params, n_estimators=mean_best_iter, early_stop=False
    )

    summary: dict[str, object] = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "metric": "balanced_accuracy",
        "sigma_k": _SIGMA_K,
        "model": "xgboost",
        "device": "CPU",
        "cpu_threads": _CPU_THREADS,
        "n_trials_requested": n_trials,
        "n_trials_completed": len([t for t in study.trials if t.state.is_finished()]),
        "n_trials_pruned": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        ),
        "timeout_s": timeout_s,
        "elapsed_seconds": round(time.time() - started, 1),
        "best_trial_number": best.number,
        "best_search_mean": best.value,
        "best_params": best_suggested,
        "best_n_estimators_fixed": mean_best_iter,
        "report_fold_scores": report_scores,
        "report_mean": float(np.mean(report_scores)),
        "report_std": float(np.std(report_scores)),
        "mlflow_parent_run_id": parent_run_id,
        "class_order": list(encoder.classes_),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STUDY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", STUDY_PATH)
    return summary


def _start_parent_mlflow() -> str | None:
    """Open the parent MLflow run for the study (best-effort). Returns its id."""
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow  # noqa: PLC0415
    except ImportError:
        logger.warning("mlflow not installed; skipping study tracking (JSON is the record)")
        return None
    try:
        mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        return mlflow.start_run(run_name="xgboost_study").info.run_id
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("MLflow parent run failed (%s); continuing", exc)
        return None


def _log_trial_mlflow(
    trial: Any,
    params: dict[str, object],
    fold_scores: list[float],
    mean: float,
    mean_best_iter: int,
) -> None:
    """Best-effort nested MLflow child run for one trial (optional)."""
    try:
        import mlflow  # noqa: PLC0415
    except ImportError:
        return
    try:
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params({f"xgb_{k}": v for k, v in params.items()})
            mlflow.log_metric("balanced_accuracy_mean", mean)
            mlflow.log_metric("mean_best_iteration", mean_best_iter)
            for k, s in enumerate(fold_scores):
                mlflow.log_metric("balanced_accuracy_fold", s, step=k)
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("MLflow trial log failed (%s); continuing", exc)


def _end_parent_mlflow() -> None:
    """Close the parent MLflow run if one is open (best-effort)."""
    try:
        import mlflow  # noqa: PLC0415

        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:  # pragma: no cover - environment-dependent
        pass


def main() -> None:
    """CLI entry point: run the XGBoost study under the configured budget."""
    parser = argparse.ArgumentParser(description="Optuna XGBoost HPO (BL-026).")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    np.random.seed(SEED)
    summary = run(n_trials=args.n_trials, timeout_s=args.timeout_s)
    print(
        f"XGBoost best #{summary['best_trial_number']}: "
        f"report_mean={summary['report_mean']:.6f} std={summary['report_std']:.6f} "
        f"n_estimators={summary['best_n_estimators_fixed']} "
        f"(completed {summary['n_trials_completed']}, pruned {summary['n_trials_pruned']})"
    )


if __name__ == "__main__":
    main()
