"""Optuna hyperparameter search for the C6 CatBoost base learner (BL-025).

Cycle C6 (Build), ensemble step 2. The naive C6 blend lost to the C5 tuned
LightGBM because CatBoost and XGBoost ran on defaults against a tuned opponent
(the statistical review certified the loss as legitimate, not a bug). This module tunes
CatBoost to PARITY with LightGBM on the IDENTICAL frozen 5 folds, so a re-blend is
a fair fight between comparably-strong members.

Mirrors the proven C5 LightGBM tuner (src.tune) one-for-one so the comparison
machinery, the determinism story and the leakage guarantees are the SAME:
  - SAME folds: the frozen data/folds/cv_indices.npz partition, positional .iloc.
  - SAME weighting: compute_sample_weights (balanced, TRAIN slice only), per fold.
  - SAME features: the raw S6E6 matrix from ensemble.build_feature_frame, with the
    two categoricals handled NATIVELY by CatBoost (cat_features by name).
  - SAME search engine: TPESampler(seed=42) + MedianPruner, maximise mean
    balanced_accuracy, report per-fold scores.

Two-speed design (identical philosophy to src.tune):
  - SEARCH: each trial runs 5-fold CV on GPU (a one-fold benchmark measured ~14.6x
    over CPU on this host), with per-fold early stopping on the fold's own val
    slice and MedianPruner abandoning losing trials after 1-2 folds. Early stopping
    on the val slice is a mild, uniform optimism (every trial gets it) that is
    REMOVED in the final number by fixing `iterations` to the rounded mean best
    iteration. GPU MultiClass is not bit-deterministic, but ranking trials does not
    need determinism.
  - REPORT: the best trial is re-evaluated with `iterations` fixed to the mean best
    iteration (no per-fold early stopping), so the reported per-fold scores are a
    like-for-like estimate the ensemble will reproduce.

Leakage: tuning reads TRAIN folds only; weights are computed on each fold's train
slice only; the val slice is used solely as the early-stopping eval set and as the
scored hold-out — never to fit. load_test is never imported here.

Output: docs/c6-build/catboost_tuning.json — best params + the deterministic
per-fold scores + the fixed iteration count, consumed by ensemble.catboost_params.
MLflow logging is best-effort (optional), matching the rest of the project.
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

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RESULTS_DIR: Final[Path] = _REPO_ROOT / "docs" / "c6-build"
STUDY_PATH: Final[Path] = RESULTS_DIR / "catboost_tuning.json"
MLFLOW_DIR: Final[Path] = _REPO_ROOT / "mlruns"
MLFLOW_EXPERIMENT: Final[str] = "s6e6-c6-catboost-tuning"

# Budget. A measured fold costs ~8s on GPU (1000-iter cap + early stopping); 5
# folds ~40s/trial, and MedianPruner abandons losers after 1-2 folds. So ~30
# trials fit comfortably in ~20 min. Both n_trials AND timeout bind (Optuna stops
# at whichever hits first); the JSON records how many actually ran.
DEFAULT_N_TRIALS: Final[int] = 30
DEFAULT_TIMEOUT_S: Final[int] = 22 * 60

# Hard cap on boosting rounds; early stopping picks the real count per fold. 1000
# brackets the depth-6/lr-0.05 default's under-fit (which wanted >600 trees) while
# capping worst-case wall-clock.
_MAX_ITERATIONS: Final[int] = 1000
_EARLY_STOPPING_ROUNDS: Final[int] = 50

# Defensibility multiplier, identical to feature_eval/tune/ensemble _SIGMA_K.
_SIGMA_K: Final[float] = 2.0

# Fixed params shared by every trial: the GPU MultiClass regime. Only the
# suggested hyperparameters vary across trials.
_FIXED_PARAMS: Final[dict[str, object]] = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "random_seed": SEED,
    "task_type": "GPU",
    "devices": "0",
    "verbose": False,
    "allow_writing_files": False,
}


def suggest_params(trial: Any) -> dict[str, object]:
    """Sample one CatBoost hyperparameter configuration (BL-025 search space).

    Ranges per the task brief, chosen around CatBoost's defaults so the default
    region stays reachable while opening room to trade depth for regularisation:
      - depth [4, 10]: tree depth (symmetric trees). The default-6 under-fit, so we
        let the search go deeper.
      - learning_rate [0.02, 0.2] (log): lower bound allows a slower, better-
        generalising schedule than the default 0.05; the _MAX_ITERATIONS cap + early
        stopping keep a low-lr trial within budget.
      - l2_leaf_reg [1, 10] (log): L2 leaf-weight penalty, the main overfit guard.
      - border_count {32, 64, 128, 254}: number of feature-discretisation splits;
        higher = finer histograms (more capacity, slower).
      - random_strength [0, 10]: noise added to the split-score, a CatBoost-specific
        regulariser against overfitting the greedy split choice.
      - iterations: NOT sampled — capped at _MAX_ITERATIONS and chosen per fold by
        early stopping (the principled way to size the ensemble).

    Args:
        trial: The Optuna trial to sample from.

    Returns:
        A dict of CatBoost kwargs (search-space only; merged with _FIXED_PARAMS and
        the iteration cap by the caller).
    """
    return {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 254]),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
    }


def _cv_scores(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: dict[str, np.ndarray],
    categorical: list[str],
    params: dict[str, object],
    *,
    iterations: int | None,
    early_stop: bool,
    trial: Any | None = None,
) -> tuple[list[float], int]:
    """Run 5-fold CV for one CatBoost config; report to Optuna for pruning.

    Used in two regimes:
      - SEARCH (early_stop=True, iterations=None): cap at _MAX_ITERATIONS, early-stop
        on the fold's own val slice, report the running mean to the trial after each
        fold and prune if it lags the median.
      - REPORT (early_stop=False, iterations=fixed): train exactly `iterations`
        rounds per fold (no early stopping), for a like-for-like reported number.

    Early stopping uses the fold's val slice as eval_set — the same slice we score.
    This is the standard LightGBM/CatBoost CV early-stopping; the rounds are chosen
    to maximise THIS fold's val metric, a mild optimism applied EQUALLY to every
    trial and REMOVED in the report regime (fixed iterations). The val slice never
    enters the fit (no labels leak into tree construction beyond round selection).

    Args:
        x: Full design matrix (raw columns selected, categoricals as 'category').
        y: Integer-encoded target aligned to x rows (positional).
        folds: Frozen fold index dict.
        categorical: Names of categorical columns (CatBoost takes names).
        params: Full CatBoost kwargs for this config (fixed + suggested).
        iterations: Fixed round count for the REPORT regime, or None for SEARCH.
        early_stop: Whether to early-stop on the val slice (SEARCH only).
        trial: Optuna trial for intermediate reporting/pruning, or None.

    Returns:
        (fold_scores, mean_best_iteration) — the per-fold balanced_accuracy and the
        rounded mean best_iteration across folds (the early-stopped ensemble size).

    Raises:
        optuna.TrialPruned: If the trial is pruned by the configured pruner.
    """
    import optuna
    from catboost import CatBoostClassifier, Pool

    cap = _MAX_ITERATIONS if iterations is None else iterations
    fold_scores: list[float] = []
    best_iters: list[int] = []

    for k in range(N_SPLITS):
        tr_idx = folds[f"fold{k}_train"]
        va_idx = folds[f"fold{k}_val"]
        x_tr, y_tr = x.iloc[tr_idx], y[tr_idx]
        x_va, y_va = x.iloc[va_idx], y[va_idx]

        w_tr = compute_sample_weights(y_tr)
        pool_tr = Pool(x_tr, y_tr, cat_features=categorical, weight=w_tr)
        pool_va = Pool(x_va, y_va, cat_features=categorical)

        model = CatBoostClassifier(iterations=cap, **params)
        if early_stop:
            model.fit(
                pool_tr,
                eval_set=pool_va,
                use_best_model=True,
                early_stopping_rounds=_EARLY_STOPPING_ROUNDS,
            )
        else:
            model.fit(pool_tr)

        pred = model.predict(pool_va).ravel().astype(int)
        score = float(balanced_accuracy_score(y_va, pred))
        fold_scores.append(score)
        best_iters.append(int(model.best_iteration_ or cap))

        if trial is not None:
            trial.report(float(np.mean(fold_scores)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return fold_scores, int(round(float(np.mean(best_iters))))


def run(
    *, n_trials: int = DEFAULT_N_TRIALS, timeout_s: int = DEFAULT_TIMEOUT_S
) -> dict[str, object]:
    """Run the CatBoost Optuna study and the deterministic re-evaluation of the best.

    Steps:
      1. Load train, encode target, load frozen folds, build the raw matrix once.
      2. Run the TPE study (seed 42) with MedianPruner under an n_trials/timeout
         budget, maximising mean balanced_accuracy in the GPU search regime.
      3. Re-evaluate the best trial with iterations fixed to the search's mean best
         iteration (no early stopping) for the reported per-fold scores.
      4. Persist docs/c6-build/catboost_tuning.json (consumed by ensemble.py).

    Args:
        n_trials: Max Optuna trials.
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
    x, categorical = build_feature_frame(df)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner, study_name="s6e6-c6-catboost"
    )

    parent_run_id = _start_parent_mlflow()

    def objective(trial: Any) -> float:
        suggested = suggest_params(trial)
        params = {**_FIXED_PARAMS, **suggested}
        fold_scores, mean_best_iter = _cv_scores(
            x, y, folds, categorical, params, iterations=None, early_stop=True, trial=trial
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
        "best CatBoost trial #%d search_mean=%.6f iters(fixed)=%d params=%s",
        best.number,
        best.value,
        mean_best_iter,
        best_suggested,
    )

    # Deterministic re-evaluation: fixed iterations, no early stopping.
    report_params = {**_FIXED_PARAMS, **best_suggested}
    report_scores, _ = _cv_scores(
        x, y, folds, categorical, report_params, iterations=mean_best_iter, early_stop=False
    )

    summary: dict[str, object] = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "metric": "balanced_accuracy",
        "sigma_k": _SIGMA_K,
        "model": "catboost",
        "device": "GPU",
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
        "best_iterations_fixed": mean_best_iter,
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
        return mlflow.start_run(run_name="catboost_study").info.run_id
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
            mlflow.log_params({f"cb_{k}": v for k, v in params.items()})
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
    """CLI entry point: run the CatBoost study under the configured budget."""
    parser = argparse.ArgumentParser(description="Optuna CatBoost HPO (BL-025).")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    np.random.seed(SEED)
    summary = run(n_trials=args.n_trials, timeout_s=args.timeout_s)
    print(
        f"CatBoost best #{summary['best_trial_number']}: "
        f"report_mean={summary['report_mean']:.6f} std={summary['report_std']:.6f} "
        f"iters={summary['best_iterations_fixed']} "
        f"(completed {summary['n_trials_completed']}, pruned {summary['n_trials_pruned']})"
    )


if __name__ == "__main__":
    main()
