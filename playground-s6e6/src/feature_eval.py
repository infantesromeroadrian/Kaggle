"""Feature-ablation cross-validation harness for Playground S6E6 (BL-006).

Cycle C3 (Feature & Hypothesis). Measures whether the engineered features from
src.features actually move balanced_accuracy versus the frozen C2 baseline, with
a statistically defensible (paired-by-fold) comparison.

Design contract — fairness of the comparison:
  - SAME folds. Every config is evaluated on the IDENTICAL frozen partition from
    data/folds/cv_indices.npz (positional .iloc), so per-fold deltas are paired.
  - SAME model. The LightGBM MODEL hyperparameters, the balanced sample weights
    and the native categorical handling are identical to the C2 baseline config
    (see src.eda._lgbm_params / _compute_sample_weights / _prepare_model_frame).
    On top of those the harness ADDS determinism flags (deterministic /
    force_row_wise / num_threads=1 / n_jobs=1) that eda.py does not set; these are
    proven NOT to alter the solution because `baseline_raw` reproduces the C2
    score to 5 decimals (0.964479 vs 0.96448). The ONLY thing that varies between
    configs is the set of feature COLUMNS fed to the model. We re-declare the
    config here (rather than import private eda helpers) so the harness has a
    stable public contract.
  - DETERMINISM. On top of the baseline params we pin deterministic=True,
    force_row_wise=True and a fixed num_threads so two runs of this harness
    produce identical numbers (the exploratory eda.py did not need this; a
    statistical comparison does). seed = 42 everywhere.

Statistics (for the statistical review):
  For each config we report the 5 per-fold balanced_accuracy values, their mean
  and std, and a PAIRED comparison versus the baseline: per fold k,
  d_k = score_config[k] - score_baseline[k]; we report mean(d), and the standard
  error of the mean of the paired differences SEM_diff = std(d, ddof=1)/sqrt(5).
  A delta is called defensible iff mean(d) > 2 * SEM_diff (a ~2-sigma paired
  signal on 5 folds — deliberately conservative given n=5).

Outputs:
  - docs/c3-feature/feature_eval_results.json — the canonical machine-readable
    results (per-fold scores + paired stats). This file is the source of truth
    for the statistical review and for the report; it is written regardless of MLflow.
  - MLflow runs (one per config) IF mlflow is importable. MLflow is OPTIONAL: it
    is not in the project lockfile, so the harness degrades gracefully to a
    warning and the JSON when mlflow is absent. The real-execution requirement is
    satisfied by the JSON, not by MLflow.

This module trains on TRAIN ONLY (it never imports load_test). Feature columns
come exclusively from src.features, whose row-wise purity guarantees no leakage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Final

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from src.cv_utils import compute_sample_weights, load_folds
from src.data_loader import TARGET, load_train
from src.features import (
    ALL_ENGINEERED,
    COLOR_DEFINITIONS,
    REDSHIFT_LOG_COL,
    add_features,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Reproducibility and layout.
# --------------------------------------------------------------------------- #
SEED: Final[int] = 42
N_SPLITS: Final[int] = 5

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
# FOLDS_PATH and load_folds now live in src.cv_utils (BL-019) — imported above so
# eda, feature_eval and the C5 tuner all resolve the SAME frozen partition.
RESULTS_DIR: Final[Path] = _REPO_ROOT / "docs" / "c3-feature"
RESULTS_PATH: Final[Path] = RESULTS_DIR / "feature_eval_results.json"
MLFLOW_DIR: Final[Path] = _REPO_ROOT / "mlruns"
MLFLOW_EXPERIMENT: Final[str] = "s6e6-c3-feature-ablation"

# Defensibility multiplier: delta is defensible iff mean(d) > _SIGMA_K * SEM_diff.
# 2.0 == a ~2-sigma paired signal. Kept as a constant so the report and the gate
# cite the same threshold.
_SIGMA_K: Final[float] = 2.0

# Raw model inputs (the C2 baseline feature set), split by dtype handling.
_RAW_NUMERIC: Final[tuple[str, ...]] = (
    "alpha",
    "delta",
    "u",
    "g",
    "r",
    "i",
    "z",
    "redshift",
)
_RAW_CATEGORICAL: Final[tuple[str, ...]] = ("spectral_type", "galaxy_population")
_RAW_FEATURES: Final[tuple[str, ...]] = _RAW_NUMERIC + _RAW_CATEGORICAL

# All four colour names, as a convenience for config declarations.
_COLORS: Final[tuple[str, ...]] = tuple(COLOR_DEFINITIONS)


def _lgbm_params() -> dict[str, object]:
    """Return the baseline LightGBM hyperparameters PLUS determinism flags.

    The hyperparameters reproduce the C2 baseline config exactly (src.eda
    _lgbm_params): multiclass / 3 classes / 300 trees / lr 0.05 / 63 leaves /
    unbounded depth / 0.8 row+col subsample / reg_lambda 1.0 / seed 42.

    Added ON TOP (not present in the exploratory eda config) so the statistical
    comparison is reproducible to the last digit:
      - deterministic=True + force_row_wise=True: disable the non-deterministic
        multi-threaded histogram path; LightGBM then produces identical splits
        across runs for fixed data+seed.
      - num_threads fixed: the histogram reduction order depends on thread count,
        so we pin it. We use a single thread for guaranteed determinism; the
        trade-off is wall-clock, accepted because the comparison must be exact.

    Returns:
        Keyword args for lgb.LGBMClassifier.
    """
    return {
        "objective": "multiclass",
        "num_class": 3,
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": SEED,
        # Determinism block (additions vs the exploratory baseline).
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "n_jobs": 1,
        "verbose": -1,
    }


@dataclass(frozen=True)
class FeatureConfig:
    """One evaluation configuration: which columns the model sees.

    Attributes:
        name: Stable identifier (used as the MLflow run name and JSON key).
        engineered: Engineered features (subset of ALL_ENGINEERED) to ADD on top
            of the raw frame before column selection.
        drop: Raw or engineered columns to REMOVE from the model matrix (used by
            the ablation configs to knock out one feature at a time).
    """

    name: str
    engineered: tuple[str, ...] = ()
    drop: tuple[str, ...] = ()


@dataclass
class ConfigResult:
    """Per-config CV outcome plus its paired comparison to the baseline.

    Attributes:
        name: Config identifier.
        features: The exact ordered feature columns fed to the model.
        categorical: Which of those columns were treated as categorical.
        fold_scores: balanced_accuracy per fold (length N_SPLITS).
        mean: Mean of fold_scores.
        std: Population std of fold_scores (ddof=0), matching the C2 baseline's
            reported ± convention.
        paired_diff_mean: mean_k(score_k - baseline_k). None for the baseline.
        paired_diff_sem: SEM of the paired differences (std ddof=1 / sqrt(n)).
            None for the baseline.
        defensible: True iff paired_diff_mean > _SIGMA_K * paired_diff_sem.
            None for the baseline.
    """

    name: str
    features: list[str]
    categorical: list[str]
    fold_scores: list[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0
    paired_diff_mean: float | None = None
    paired_diff_sem: float | None = None
    defensible: bool | None = None


def build_model_frame(df: pd.DataFrame, config: FeatureConfig) -> tuple[pd.DataFrame, list[str]]:
    """Build the model design matrix for a config (engineered + raw, minus drops).

    Steps:
      1. Append the config's engineered features via the pure add_features.
      2. Select the column set: raw features + engineered, minus `drop`.
      3. Cast the surviving categorical columns to pandas `category` dtype so
         LightGBM uses native categorical handling (matching the baseline).

    No target, no scaling, no imputation (LightGBM is scale-invariant; the data
    has no NaN). The engineered columns are produced row-wise, so this whole
    function is leakage-free.

    Args:
        df: The raw training frame (with id and class still present).
        config: The FeatureConfig describing engineered adds and drops.

    Returns:
        (X, categorical_columns) where X has exactly the model columns in a
        deterministic order and categorical_columns lists the category-dtype
        columns within X.

    Raises:
        KeyError: If a `drop` names a column not in the assembled feature set.
    """
    enriched = add_features(df, include=config.engineered) if config.engineered else df.copy()

    # Deterministic column order: raw numeric, raw categorical, then engineered
    # in ALL_ENGINEERED order. Fixed order => reproducible models and SHAP-able.
    ordered = list(_RAW_FEATURES) + [c for c in ALL_ENGINEERED if c in config.engineered]

    drop_set = set(config.drop)
    unknown_drop = drop_set - set(ordered)
    if unknown_drop:
        raise KeyError(
            f"config {config.name!r}: cannot drop unknown column(s) {sorted(unknown_drop)}"
        )
    selected = [c for c in ordered if c not in drop_set]

    x = enriched[selected].copy()
    categorical = [c for c in _RAW_CATEGORICAL if c in selected]
    for col in categorical:
        x[col] = x[col].astype("category")
    return x, categorical


def evaluate_config(
    df: pd.DataFrame,
    y: np.ndarray,
    folds: dict[str, np.ndarray],
    config: FeatureConfig,
) -> ConfigResult:
    """Run frozen-fold CV for one config and return per-fold balanced_accuracy.

    For each fold k: build the model matrix (engineered features added on the
    FULL frame is safe because they are row-wise pure — no fit, no leakage),
    slice train/val by the frozen positional indices, fit LightGBM with balanced
    weights, predict val, record balanced_accuracy.

    Args:
        df: Raw training frame.
        y: Integer-encoded target aligned to df rows (positional).
        folds: Frozen fold index dict (fold{k}_train / fold{k}_val).
        config: The configuration to evaluate.

    Returns:
        A ConfigResult with fold_scores/mean/std filled (paired stats added later
        by attach_paired_stats once the baseline is known).

    Raises:
        KeyError: If a fold key is missing.
    """
    x, categorical = build_model_frame(df, config)
    cat_idx = [list(x.columns).index(c) for c in categorical]

    fold_scores: list[float] = []
    for k in range(N_SPLITS):
        tr_idx = folds[f"fold{k}_train"]
        va_idx = folds[f"fold{k}_val"]

        x_tr, y_tr = x.iloc[tr_idx], y[tr_idx]
        x_va, y_va = x.iloc[va_idx], y[va_idx]

        weights = compute_sample_weights(y_tr)
        model = lgb.LGBMClassifier(**_lgbm_params())
        model.fit(x_tr, y_tr, sample_weight=weights, categorical_feature=cat_idx)

        pred = model.predict(x_va)
        score = float(balanced_accuracy_score(y_va, pred))
        fold_scores.append(score)
        logger.info("config=%s fold=%d balanced_accuracy=%.6f", config.name, k, score)

    return ConfigResult(
        name=config.name,
        features=list(x.columns),
        categorical=categorical,
        fold_scores=fold_scores,
        mean=float(np.mean(fold_scores)),
        std=float(np.std(fold_scores)),  # ddof=0 to match the C2 baseline ± report
    )


def attach_paired_stats(result: ConfigResult, baseline_scores: list[float]) -> None:
    """Fill the paired-difference statistics on `result` in place.

    Computes d_k = result.fold_scores[k] - baseline_scores[k] for each fold, then
    paired_diff_mean = mean(d), paired_diff_sem = std(d, ddof=1)/sqrt(n), and
    defensible = paired_diff_mean > _SIGMA_K * paired_diff_sem. The baseline
    config (whose scores ARE baseline_scores) is left with None stats (a config
    is not compared to itself).

    Args:
        result: The ConfigResult to annotate (mutated in place).
        baseline_scores: The baseline config's per-fold scores (length N_SPLITS).

    Raises:
        ValueError: If the two score vectors differ in length (folds mismatch).
    """
    if len(result.fold_scores) != len(baseline_scores):
        raise ValueError("fold count mismatch between config and baseline")

    # The baseline compared to itself carries no informative delta -> leave None.
    if result.fold_scores == baseline_scores:
        return

    diffs = np.asarray(result.fold_scores) - np.asarray(baseline_scores)
    mean_d = float(np.mean(diffs))
    # ddof=1: SEM of the paired differences (sample std over the 5 folds).
    sem_d = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    result.paired_diff_mean = mean_d
    result.paired_diff_sem = sem_d
    result.defensible = bool(mean_d > _SIGMA_K * sem_d)


def default_configs() -> list[FeatureConfig]:
    """Return the BL-006 comparison plan as a list of FeatureConfig.

    Plan:
      - baseline_raw: raw features only (fidelity check, must reproduce ~0.96448).
      - colors: + the four adjacent-band colours (the hypothesised gain).
      - colors_redshift_log: + colours AND the redshift log control (the control
        for monotonic-transform invariance; expected ~0 delta vs colors).
      - ablations on the best feature set (raw + colours): knock out one of
        r / alpha / delta / galaxy_population at a time, to test whether each
        carries marginal signal once colours are present.

    Returns:
        The ordered list of configs to evaluate.
    """
    configs = [
        FeatureConfig(name="baseline_raw"),
        FeatureConfig(name="colors", engineered=_COLORS),
        FeatureConfig(name="colors_redshift_log", engineered=_COLORS + (REDSHIFT_LOG_COL,)),
    ]
    # Ablations operate on the (raw + colours) set — the expected best config.
    for col in ("r", "alpha", "delta", "galaxy_population"):
        configs.append(
            FeatureConfig(name=f"colors_drop_{col}", engineered=_COLORS, drop=(col,))
        )
    return configs


def _try_log_mlflow(results: list[ConfigResult]) -> bool:
    """Best-effort MLflow logging; returns True if it ran, False if skipped.

    MLflow is OPTIONAL (not in the lockfile). If importable, we log each config as
    a run to a local file store under mlruns/: params = the feature columns and
    the determinism flags; metrics = mean/std balanced_accuracy and per-fold
    scores (metric step = fold index) and the paired delta. If MLflow is absent
    or errors, we log a warning and continue — the JSON remains the source of
    truth.

    Args:
        results: All evaluated configs.

    Returns:
        True if MLflow logging completed, False if it was skipped.
    """
    try:
        import mlflow  # noqa: PLC0415  (intentional optional import)
    except ImportError:
        logger.warning("mlflow not installed; skipping MLflow logging (JSON is the record)")
        return False

    try:
        mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        params = _lgbm_params()
        for res in results:
            with mlflow.start_run(run_name=res.name):
                mlflow.log_params(
                    {f"lgbm_{k}": v for k, v in params.items()}
                    | {
                        "features": ",".join(res.features),
                        "n_features": len(res.features),
                        "categorical": ",".join(res.categorical),
                        "seed": SEED,
                    }
                )
                mlflow.log_metric("balanced_accuracy_mean", res.mean)
                mlflow.log_metric("balanced_accuracy_std", res.std)
                for k, score in enumerate(res.fold_scores):
                    mlflow.log_metric("balanced_accuracy_fold", score, step=k)
                if res.paired_diff_mean is not None:
                    mlflow.log_metric("paired_diff_mean", res.paired_diff_mean)
                    mlflow.log_metric("paired_diff_sem", res.paired_diff_sem)
                    mlflow.log_metric("defensible", float(bool(res.defensible)))
        logger.info("MLflow runs logged under %s (experiment %s)", MLFLOW_DIR, MLFLOW_EXPERIMENT)
        return True
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("MLflow logging failed (%s); continuing with JSON record", exc)
        return False


def run(configs: list[FeatureConfig] | None = None) -> dict[str, object]:
    """Execute the full ablation and persist the canonical results JSON.

    Steps:
      1. Load train; label-encode the target.
      2. Load frozen folds.
      3. Evaluate every config on the identical folds.
      4. Attach paired-vs-baseline stats (baseline = the 'baseline_raw' config).
      5. Persist docs/c3-feature/feature_eval_results.json and, if available,
         MLflow runs.

    Args:
        configs: Configs to run. None uses default_configs().

    Returns:
        The JSON-serialisable results dict (also written to RESULTS_PATH).

    Raises:
        RuntimeError: If no config named 'baseline_raw' is present (the paired
            comparison has no reference otherwise).
    """
    configs = configs if configs is not None else default_configs()
    if not any(c.name == "baseline_raw" for c in configs):
        raise RuntimeError("a config named 'baseline_raw' is required as the paired reference")

    df = load_train()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].to_numpy())

    folds = load_folds()

    started = time.time()
    results: list[ConfigResult] = []
    for config in configs:
        t0 = time.time()
        res = evaluate_config(df, y, folds, config)
        logger.info(
            "config=%s mean=%.6f std=%.6f (%.1fs)",
            res.name,
            res.mean,
            res.std,
            time.time() - t0,
        )
        results.append(res)

    baseline = next(r for r in results if r.name == "baseline_raw")
    for res in results:
        attach_paired_stats(res, baseline.fold_scores)

    mlflow_logged = _try_log_mlflow(results)

    summary: dict[str, object] = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "metric": "balanced_accuracy",
        "sigma_k": _SIGMA_K,
        "lgbm_params": _lgbm_params(),
        "baseline_config": "baseline_raw",
        "class_order": list(encoder.classes_),
        "mlflow_logged": mlflow_logged,
        "elapsed_seconds": round(time.time() - started, 1),
        "results": [asdict(r) for r in results],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d configs)", RESULTS_PATH, len(results))
    return summary


def main() -> None:
    """CLI entry point: configure logging and run the full ablation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    np.random.seed(SEED)
    summary = run()
    # Compact stdout digest so a caller sees the headline without parsing the JSON.
    for res in summary["results"]:  # type: ignore[index]
        delta = res["paired_diff_mean"]
        sem = res["paired_diff_sem"]
        tail = "" if delta is None else f" delta={delta:+.6f} sem={sem:.6f} defensible={res['defensible']}"
        print(f"{res['name']:>22s}  mean={res['mean']:.6f} std={res['std']:.6f}{tail}")


if __name__ == "__main__":
    main()
