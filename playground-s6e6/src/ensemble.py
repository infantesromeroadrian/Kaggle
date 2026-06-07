"""Soft-voting ensemble of LightGBM + CatBoost + XGBoost for S6E6 (BL-024).

Cycle C6 (Build). The C5 POC fixed a single tuned LightGBM (deterministic CV mean
0.96487 balanced_accuracy, public LB 0.96521). This module asks the only question
that justifies the extra complexity of an ensemble: does averaging the class
PROBABILITIES of three gradient-boosting families with different inductive biases
beat that single model BY A DEFENSIBLE MARGIN on the IDENTICAL frozen folds?

Why these three base learners (different errors are what a blend averages out):
  - LightGBM: leaf-wise (best-first) tree growth. The C5 champion, re-fit here
    with its tuned params so the blend includes the model we must beat.
  - CatBoost: ordered boosting + symmetric (oblivious) trees + native categorical
    handling. A genuinely different tree topology and a different overfitting
    control than LightGBM, so its per-row errors are only partly correlated.
  - XGBoost: level-wise (depth-first) tree growth with native categorical support.
    A third decorrelated view of the same features.
A soft-voting blend (mean of the per-class probabilities, then argmax) only helps
if the models' errors are not identical; it cannot help a perfectly-correlated
trio, which is precisely why we MEASURE the paired delta instead of assuming gain.

Comparison contract (the whole point — for the statistical review):
  - SAME folds. Every model is trained and scored on the frozen partition from
    data/folds/cv_indices.npz (positional .iloc), the exact folds the C5 LightGBM
    was measured on. So the blend's 5 per-fold scores are PAIRED with LightGBM's.
  - SAME weighting. compute_sample_weights (balanced, train-slice only) is applied
    to every model's fit, identical to C5. No model sees val labels.
  - PAIRED REFERENCE = the TUNED LightGBM, not the untuned C2 baseline. The bar to
    clear is C5's best_deterministic.fold_scores (mean 0.96487). We reuse
    feature_eval.attach_paired_stats: d_k = blend_k - lgbm_k, delta = mean(d),
    SEM_diff = std(d, ddof=1)/sqrt(5), defensible iff delta > 2 * SEM_diff (the
    same conservative 2-sigma-on-5-folds rule C3 and C5 used).
  - FIDELITY GUARD. Our re-fitted LightGBM per-fold scores must reproduce the C5
    best_deterministic.fold_scores to ~5 decimals; if they do not, the comparison
    is comparing two different LightGBMs and is void. run() asserts this.

OOF (out-of-fold) leakage invariant:
  For each model and each fold k, we fit on fold k's TRAIN rows only and predict
  fold k's VAL rows; the val predictions are written into an OOF probability matrix
  at the val positions. Every row's OOF prediction therefore comes from a model
  that NEVER saw that row in training. The blend and all per-fold metrics are
  computed on these OOF probabilities. Test is loaded ONLY in build_submission,
  AFTER the CV verdict, never inside the CV loop.

Compute (the C5 wall-clock lesson + a measured C6 benchmark):
  - LightGBM: CPU, num_threads=1, deterministic regime (bit-for-bit comparable to
    C5; that is why the fidelity guard can be tight).
  - CatBoost: GPU. A one-fold benchmark on this host (RTX 2000 Ada) measured GPU at
    ~14.6x the CPU speed (2.4s vs 34.5s for 300 trees depth 6); GPU is the only way
    CatBoost fits the budget here. Trade-off: CatBoost's GPU MultiClass path is not
    bit-deterministic run-to-run, so its per-fold scores carry small variance. This
    does NOT corrupt the paired comparison: the LightGBM reference is fixed, and the
    blend is reported with the same mean±std convention LightGBM lived under. Pinned
    random_seed=42 keeps the variance small.
  - XGBoost: CPU hist, nthread=4 (the C5 sweet spot; -1 over-subscribes the 22
    logical cores and is catastrophic, per the tuning report).

MLflow tracking (C6 is a BUILD cycle: tracking is first-class here): a parent run
plus one child run per base model and one for the blend, logging params, per-fold
scores, mean/std and the paired delta. MLflow stays OPTIONAL via a graceful
import guard (matching feature_eval): the canonical record is the JSON, so the
harness runs and the verdict stands even with mlflow absent.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from src.cv_utils import compute_sample_weights, load_folds
from src.data_loader import ID_COLUMN, TARGET, load_test, load_train

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Reproducibility and layout.
# --------------------------------------------------------------------------- #
SEED: Final[int] = 42
N_SPLITS: Final[int] = 5
N_CLASSES: Final[int] = 3

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RESULTS_DIR: Final[Path] = _REPO_ROOT / "docs" / "c6-build"
RESULTS_PATH: Final[Path] = RESULTS_DIR / "ensemble_results.json"
DATA_DIR_OUT: Final[Path] = _REPO_ROOT / "data"
SUBMISSION_PATH: Final[Path] = DATA_DIR_OUT / "submission_c6_ensemble_tuned.csv"
MLFLOW_DIR: Final[Path] = _REPO_ROOT / "mlruns"
MLFLOW_EXPERIMENT: Final[str] = "s6e6-c6-ensemble"

# C5 tuning record: the source of the LightGBM tuned params AND the per-fold
# scores the blend must beat. Read at run time so the reference is never re-typed.
TUNING_RESULTS_PATH: Final[Path] = _REPO_ROOT / "docs" / "c5-poc" / "tuning_results.json"

# C6 tuning records for the two non-LightGBM base learners (BL-025 / BL-026). When
# present, catboost_params/xgboost_params load the tuned hyperparameters from these
# files (single source of truth, same pattern as the LightGBM C5 record); when
# absent, both fall back to the sensible-default factories so the harness still runs
# end-to-end (e.g. in a fresh checkout before the studies have been executed).
CATBOOST_TUNING_PATH: Final[Path] = _REPO_ROOT / "docs" / "c6-build" / "catboost_tuning.json"
XGBOOST_TUNING_PATH: Final[Path] = _REPO_ROOT / "docs" / "c6-build" / "xgboost_tuning.json"

# Defensibility multiplier — identical to feature_eval._SIGMA_K / tune._SIGMA_K so
# C3, C5 and C6 all cite one threshold. A delta is defensible iff it exceeds this
# many standard errors of the paired difference.
_SIGMA_K: Final[float] = 2.0

# Tolerance for the LightGBM fidelity guard. Our per-fold LightGBM scores must
# match the C5 best_deterministic scores to within this absolute tolerance; a
# larger gap means we are not re-fitting the same model and the paired comparison
# is invalid. 1e-4 allows for trivial library-build drift while catching a real
# parameter or regime mismatch.
_FIDELITY_ATOL: Final[float] = 1e-4

# Feature columns. RAW only (the C5 champion used raw features; C3 ablations found
# no defensible engineered gain), split by dtype handling.
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

# Thread count for the CPU models. NOT -1: the C5 tuning report measured -1 as
# catastrophic on this 22-logical-core host (thread over-subscription under MCP
# background load); 4 is the sweet spot.
_CPU_THREADS: Final[int] = 4


def load_lgbm_tuned_params() -> dict[str, object]:
    """Load the C5-tuned LightGBM hyperparameters from the tuning results JSON.

    Returns the best suggested hyperparameters merged with the fixed deterministic
    n_estimators, so the model rebuilt here is the SAME one C5 reported. Reading
    from the JSON (rather than hard-coding) keeps a single source of truth: if the
    C5 record changes, the ensemble's LightGBM follows automatically.

    Returns:
        LightGBM kwargs: the tuned search-space params plus 'n_estimators'.

    Raises:
        FileNotFoundError: If the C5 tuning JSON is absent (run src/tune.py first).
        KeyError: If the JSON lacks the expected best-params keys.
    """
    if not TUNING_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"C5 tuning results not found at {TUNING_RESULTS_PATH}; run src/tune.py first."
        )
    record = json.loads(TUNING_RESULTS_PATH.read_text(encoding="utf-8"))
    params = dict(record["best_params_suggested"])
    params["n_estimators"] = int(record["best_n_estimators_fixed"])
    return params


def load_lgbm_reference_scores() -> list[float]:
    """Load the C5 tuned-LightGBM per-fold scores that the blend must beat.

    These are the deterministic per-fold balanced_accuracy values from the C5
    study (best_deterministic.fold_scores, mean 0.96487). They are the PAIRED
    reference for the whole C6 comparison.

    Returns:
        The 5 per-fold balanced_accuracy values (one per frozen fold).

    Raises:
        FileNotFoundError: If the C5 tuning JSON is absent.
        KeyError: If the JSON lacks the best_deterministic.fold_scores entry.
    """
    record = json.loads(TUNING_RESULTS_PATH.read_text(encoding="utf-8"))
    return [float(s) for s in record["best_deterministic"]["fold_scores"]]


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select the raw model columns and cast categoricals to pandas 'category'.

    The same column set and order is used for every base model and for train/test,
    so the three models and the submission all see identical inputs. Casting the
    two categorical columns to the pandas 'category' dtype lets LightGBM and
    XGBoost use native categorical handling; CatBoost takes the same columns by
    name via cat_features.

    No scaling, no imputation: GBDTs are scale-invariant and the data has no NaN
    (enforced by the C2 schema validator). This function is row-wise and reads no
    target, so it cannot leak.

    Args:
        df: A validated train or test frame (raw columns present).

    Returns:
        (X, categorical_columns) with X holding exactly the model columns in a
        fixed order and categorical_columns naming the category-dtype columns.
    """
    x = df[list(_RAW_FEATURES)].copy()
    categorical = list(_RAW_CATEGORICAL)
    for col in categorical:
        x[col] = x[col].astype("category")
    return x, categorical


# --------------------------------------------------------------------------- #
# Base-model factories. Each returns a fresh, unfitted estimator-like object with
# a uniform .fit(X, y, sample_weight)/.predict_proba(X) contract used by the OOF
# loop. We wrap CatBoost/XGBoost minimally so the loop stays model-agnostic.
# --------------------------------------------------------------------------- #


def lgbm_params() -> dict[str, object]:
    """Full LightGBM kwargs: tuned params + the deterministic regime + objective.

    The determinism block (deterministic / force_row_wise / single thread) pins the
    model to the exact regime C5's best_deterministic was measured in, which is what
    makes the fidelity guard (reproduce C5 to ~5 decimals) meaningful.

    Returns:
        Keyword args for lgb.LGBMClassifier.
    """
    return {
        "objective": "multiclass",
        "num_class": N_CLASSES,
        **load_lgbm_tuned_params(),
        "random_state": SEED,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "n_jobs": 1,
        "verbose": -1,
    }


def catboost_params() -> dict[str, object]:
    """Full CatBoost kwargs for the GPU MultiClass model.

    Loads the BL-025 tuned hyperparameters (depth / learning_rate / l2_leaf_reg /
    border_count / random_strength + the fixed iteration count) from
    CATBOOST_TUNING_PATH when it exists, so the ensemble uses the SAME model the
    Optuna study reported — single source of truth, no re-typed numbers. Falls back
    to sensible defaults (600 trees, depth 6, lr 0.05) when the tuning record is
    absent, so a fresh checkout still runs.

    GPU because a one-fold benchmark measured ~14.6x over CPU on this host. The
    balanced class weighting is applied via per-sample weights in the fit
    (compute_sample_weights), identical to every other model — NOT a CatBoost-
    specific class_weights scheme.

    Returns:
        Keyword args for catboost.CatBoostClassifier.
    """
    base: dict[str, object] = {
        "loss_function": "MultiClass",
        "random_seed": SEED,
        "task_type": "GPU",
        "devices": "0",
        "thread_count": _CPU_THREADS,
        "verbose": False,
        "allow_writing_files": False,
    }
    if CATBOOST_TUNING_PATH.is_file():
        record = json.loads(CATBOOST_TUNING_PATH.read_text(encoding="utf-8"))
        tuned = dict(record["best_params"])
        tuned["iterations"] = int(record["best_iterations_fixed"])
        return {**base, **tuned}
    # Fallback defaults (pre-tuning).
    return {**base, "iterations": 600, "depth": 6, "learning_rate": 0.05}


def xgboost_params() -> dict[str, object]:
    """Full XGBoost kwargs for the CPU hist MultiClass model with native categoricals.

    Loads the BL-026 tuned hyperparameters (max_depth / learning_rate /
    min_child_weight / subsample / colsample_bytree / reg_alpha / reg_lambda / gamma
    + the fixed n_estimators) from XGBOOST_TUNING_PATH when it exists; falls back to
    sensible defaults (600 level-wise trees, depth 6, lr 0.05) otherwise.
    enable_categorical=True consumes the pandas 'category' columns directly (no
    manual OrdinalEncoder).

    Returns:
        Keyword args for xgboost.XGBClassifier.
    """
    base: dict[str, object] = {
        "objective": "multi:softprob",
        "num_class": N_CLASSES,
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": SEED,
        "n_jobs": _CPU_THREADS,
        "verbosity": 0,
    }
    if XGBOOST_TUNING_PATH.is_file():
        record = json.loads(XGBOOST_TUNING_PATH.read_text(encoding="utf-8"))
        tuned = dict(record["best_params"])
        tuned["n_estimators"] = int(record["best_n_estimators_fixed"])
        return {**base, **tuned}
    # Fallback defaults (pre-tuning).
    return {**base, "n_estimators": 600, "max_depth": 6, "learning_rate": 0.05}


def fit_predict_lgbm(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    x_pred: pd.DataFrame,
    cat_idx: list[int],
) -> np.ndarray:
    """Fit LightGBM on one fold's train slice and return val/test class probabilities.

    Args:
        x_tr: Train design matrix for the fold.
        y_tr: Integer-encoded train labels.
        w_tr: Balanced per-sample train weights.
        x_pred: Rows to predict (a fold's val slice, or the full test set).
        cat_idx: Positional indices of categorical columns in the design matrix.

    Returns:
        An (len(x_pred), N_CLASSES) float array of class probabilities.
    """
    import lightgbm as lgb  # local import keeps module import cheap for tests

    model = lgb.LGBMClassifier(**lgbm_params())
    model.fit(x_tr, y_tr, sample_weight=w_tr, categorical_feature=cat_idx)
    return model.predict_proba(x_pred)


def fit_predict_catboost(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    x_pred: pd.DataFrame,
    categorical: list[str],
) -> np.ndarray:
    """Fit CatBoost (GPU) on one fold's train slice and return class probabilities.

    Args:
        x_tr: Train design matrix.
        y_tr: Integer-encoded train labels.
        w_tr: Balanced per-sample train weights.
        x_pred: Rows to predict.
        categorical: Names of categorical columns (CatBoost takes names, not idx).

    Returns:
        An (len(x_pred), N_CLASSES) float array of class probabilities.
    """
    from catboost import CatBoostClassifier, Pool

    pool_tr = Pool(x_tr, y_tr, cat_features=categorical, weight=w_tr)
    model = CatBoostClassifier(**catboost_params())
    model.fit(pool_tr)
    pool_pred = Pool(x_pred, cat_features=categorical)
    # CatBoost returns columns in sorted class-label order (0,1,2) == the
    # LabelEncoder order, matching the other two models' column convention.
    return model.predict_proba(pool_pred)


def fit_predict_xgboost(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    x_pred: pd.DataFrame,
) -> np.ndarray:
    """Fit XGBoost (CPU hist) on one fold's train slice and return class probabilities.

    XGBoost reads the pandas 'category' columns natively via enable_categorical, so
    no encoder is needed and the input frame is identical to the other models'.

    Args:
        x_tr: Train design matrix (categorical columns are 'category' dtype).
        y_tr: Integer-encoded train labels.
        w_tr: Balanced per-sample train weights.
        x_pred: Rows to predict.

    Returns:
        An (len(x_pred), N_CLASSES) float array of class probabilities.
    """
    import xgboost as xgb

    model = xgb.XGBClassifier(**xgboost_params())
    model.fit(x_tr, y_tr, sample_weight=w_tr)
    return model.predict_proba(x_pred)


@dataclass
class ModelOOF:
    """Out-of-fold result for one base model.

    Attributes:
        name: Model identifier ('lightgbm' / 'catboost' / 'xgboost').
        oof_proba: (N, N_CLASSES) OOF probability matrix; row i is filled by the
            fold whose val slice contains i (so model never trained on row i).
        fold_scores: per-fold balanced_accuracy of this model alone (argmax of its
            own val probabilities), length N_SPLITS.
    """

    name: str
    oof_proba: np.ndarray
    fold_scores: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        """Mean per-fold balanced_accuracy of this model alone."""
        return float(np.mean(self.fold_scores))

    @property
    def std(self) -> float:
        """Population std (ddof=0) of the per-fold scores, matching the C5 ± report."""
        return float(np.std(self.fold_scores))


def compute_oof(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: dict[str, np.ndarray],
    name: str,
    predict_fn: Callable[[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame], np.ndarray],
) -> ModelOOF:
    """Build the OOF probability matrix and per-fold scores for one base model.

    For each fold k: slice train/val by the frozen positional indices, compute the
    balanced train weights on the TRAIN slice only, fit + predict via predict_fn,
    write the val probabilities into the OOF matrix at the val positions, and score
    that fold's val slice (argmax) with balanced_accuracy.

    The OOF matrix is initialised to NaN and every row is written exactly once (the
    folds partition the data), so a leftover NaN would signal a coverage bug; we
    assert full coverage at the end.

    Args:
        x: Full design matrix (all rows, model columns selected).
        y: Integer-encoded target aligned to x rows (positional).
        folds: Frozen fold index dict (fold{k}_train / fold{k}_val).
        name: Model identifier for logging and the result.
        predict_fn: Closure (x_tr, y_tr, w_tr, x_va) -> val probabilities. The
            categorical handling is baked into the closure by the caller, so this
            loop stays model-agnostic.

    Returns:
        A ModelOOF with the filled OOF matrix and the 5 per-fold scores.

    Raises:
        AssertionError: If any OOF row was left unwritten (fold coverage bug).
    """
    oof = np.full((len(x), N_CLASSES), np.nan, dtype=np.float64)
    fold_scores: list[float] = []

    for k in range(N_SPLITS):
        tr_idx = folds[f"fold{k}_train"]
        va_idx = folds[f"fold{k}_val"]
        x_tr, y_tr = x.iloc[tr_idx], y[tr_idx]
        x_va, y_va = x.iloc[va_idx], y[va_idx]

        # Weights from the TRAIN slice only — never val. This is the leakage-safe
        # equivalent of class_weight='balanced' for this fold.
        w_tr = compute_sample_weights(y_tr)

        t0 = time.time()
        proba_va = predict_fn(x_tr, y_tr, w_tr, x_va)
        oof[va_idx] = proba_va

        score = float(balanced_accuracy_score(y_va, proba_va.argmax(axis=1)))
        fold_scores.append(score)
        logger.info(
            "model=%s fold=%d balanced_accuracy=%.6f (%.1fs)",
            name,
            k,
            score,
            time.time() - t0,
        )

    # Folds partition the rows: every row must have been written exactly once.
    assert not np.isnan(oof).any(), f"{name}: OOF has unwritten rows (fold coverage bug)"
    return ModelOOF(name=name, oof_proba=oof, fold_scores=fold_scores)


def blend_probabilities(
    oof_matrices: list[np.ndarray], weights: list[float] | None = None
) -> np.ndarray:
    """Soft-voting blend: (weighted) mean of per-class probability matrices.

    Equal weights (weights=None) are the default and the headline: averaging the
    probabilities is the simplest sound blend and does not overfit a weight vector
    to the CV. Optional weights are supported ONLY so a justified, separately
    audited weight search can reuse this primitive; they are NOT used by default.

    Args:
        oof_matrices: List of (N, N_CLASSES) probability matrices, one per model.
            All must share the same shape and the same column (class) order.
        weights: Optional non-negative weights, one per matrix. None => equal.
            Need not sum to 1; they are normalised here. argmax is invariant to a
            positive global scale, but we normalise so the output is a proper
            probability (sums to 1 per row) for logging/threshold use.

    Returns:
        An (N, N_CLASSES) blended probability matrix whose rows sum to 1.

    Raises:
        ValueError: If matrices differ in shape, if fewer than one matrix is given,
            or if weights are the wrong length / not all non-negative / sum to 0.
    """
    if not oof_matrices:
        raise ValueError("blend needs at least one probability matrix")
    shape = oof_matrices[0].shape
    for m in oof_matrices:
        if m.shape != shape:
            raise ValueError(f"probability matrices must share shape; got {shape} and {m.shape}")

    if weights is None:
        weights = [1.0] * len(oof_matrices)
    if len(weights) != len(oof_matrices):
        raise ValueError(f"got {len(weights)} weights for {len(oof_matrices)} matrices")
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < 0.0):
        raise ValueError("blend weights must be non-negative")
    total = float(w.sum())
    if total == 0.0:
        raise ValueError("blend weights sum to zero")
    w = w / total

    # Weighted mean across models. Each input row already sums to ~1, so the convex
    # combination also sums to ~1; we renormalise to absorb float error.
    stacked = np.stack(oof_matrices, axis=0)  # (n_models, N, N_CLASSES)
    blended = np.tensordot(w, stacked, axes=([0], [0]))  # (N, N_CLASSES)
    row_sums = blended.sum(axis=1, keepdims=True)
    return blended / row_sums


def score_blend_per_fold(
    blended_oof: np.ndarray, y: np.ndarray, folds: dict[str, np.ndarray]
) -> list[float]:
    """Per-fold balanced_accuracy of a blended OOF matrix, on the SAME folds.

    Scoring fold-by-fold (rather than once over all OOF rows) is what makes the
    blend's 5 values PAIRED with the LightGBM reference's 5 values: fold k's blend
    score and fold k's LightGBM score are computed on the identical val slice.

    Args:
        blended_oof: (N, N_CLASSES) blended OOF probabilities.
        y: Integer-encoded target (positional).
        folds: Frozen fold index dict.

    Returns:
        The 5 per-fold balanced_accuracy values of argmax(blended_oof).
    """
    pred = blended_oof.argmax(axis=1)
    scores: list[float] = []
    for k in range(N_SPLITS):
        va_idx = folds[f"fold{k}_val"]
        scores.append(float(balanced_accuracy_score(y[va_idx], pred[va_idx])))
    return scores


# --------------------------------------------------------------------------- #
# Nested-CV blend-weight optimisation.
#
# WHY nested: optimising blend weights on the SAME folds we then report is CV
# overfitting — the weights would be fit to maximise the very numbers we quote, an
# optimism the statistical review rejects. The honest protocol is nested cross-validation:
# for each held-out fold k, choose the weights on the OTHER N-1 folds' OOF rows
# only, then score those weights on fold k (which played no part in choosing them).
# Aggregating the 5 held-out scores gives a per-fold vector that is paired with the
# LightGBM reference AND free of weight-selection leakage.
# --------------------------------------------------------------------------- #

# Candidate weight grid for the three models (lightgbm, catboost, xgboost). A
# coarse simplex grid is enough: the response surface is smooth and we only need a
# better-than-equal point, not a precise optimum. A grid (not a continuous
# optimiser) keeps the search deterministic and auditable. The default _GRID_STEPS
# is divisible by 3 so the EQUAL-weight vector (n_steps/3 per model) is an EXACT
# grid point — important for the tie-break that prefers equal weights on a flat
# objective (otherwise it could never return exactly equal).
_GRID_STEPS: Final[int] = 12  # 91 lattice points on the 2-simplex; includes 4/4/4.


def _simplex_grid(n_models: int, n_steps: int = _GRID_STEPS) -> list[tuple[float, ...]]:
    """Enumerate all weight vectors of length n_models on an n_steps-spaced simplex.

    Each vector is non-negative and sums to 1.0. The grid is the set of integer
    compositions of n_steps into n_models parts, divided by n_steps. For n_models=3
    and n_steps=12 this is C(12+2, 2) = 91 candidates — cheap to score exhaustively.

    Args:
        n_models: Number of models (length of each weight vector).
        n_steps: Number of grid intervals; a multiple of n_models makes equal
            weights an exact grid point.

    Returns:
        A list of weight tuples, each summing to 1.0.
    """

    def recurse(remaining_models: int, remaining_steps: int) -> list[tuple[int, ...]]:
        if remaining_models == 1:
            return [(remaining_steps,)]
        out: list[tuple[int, ...]] = []
        for take in range(remaining_steps + 1):
            for tail in recurse(remaining_models - 1, remaining_steps - take):
                out.append((take, *tail))
        return out

    return [tuple(v / n_steps for v in combo) for combo in recurse(n_models, n_steps)]


def _best_weights_on_folds(
    oof_matrices: list[np.ndarray],
    y: np.ndarray,
    folds: dict[str, np.ndarray],
    train_fold_ids: list[int],
    grid: list[tuple[float, ...]],
) -> tuple[float, ...]:
    """Pick the weight vector maximising blend balanced_accuracy on given folds.

    Scores every grid candidate on the UNION of the train_fold_ids' val rows (the
    inner objective of the nested CV) and returns the argmax weight vector. Ties are
    broken toward the candidate closest to equal weights (the simplest), so a flat
    region does not silently prefer an extreme corner.

    Args:
        oof_matrices: Per-model OOF probability matrices (all (N, N_CLASSES)).
        y: Integer-encoded target (positional).
        folds: Frozen fold index dict.
        train_fold_ids: The inner-training fold indices (everything except the
            held-out fold) whose val rows the weights are selected on.
        grid: Candidate weight vectors (from _simplex_grid).

    Returns:
        The best weight tuple for these folds.
    """
    inner_rows = np.concatenate([folds[f"fold{k}_val"] for k in train_fold_ids])
    y_inner = y[inner_rows]
    equal = tuple([1.0 / len(oof_matrices)] * len(oof_matrices))

    best_w = equal
    best_score = -1.0
    best_dist = float("inf")
    for w in grid:
        blended = blend_probabilities([m[inner_rows] for m in oof_matrices], weights=list(w))
        score = float(balanced_accuracy_score(y_inner, blended.argmax(axis=1)))
        dist = float(np.sum((np.asarray(w) - np.asarray(equal)) ** 2))
        # Strictly better score wins; on a tie, prefer the weights nearest to equal.
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and dist < best_dist):
            best_score, best_w, best_dist = score, w, dist
    return best_w


@dataclass
class NestedWeightResult:
    """Outcome of nested-CV blend-weight optimisation.

    Attributes:
        fold_scores: Held-out balanced_accuracy per fold (length N_SPLITS), each
            computed with weights chosen WITHOUT that fold — the leakage-free,
            paired-with-reference vector to report.
        per_fold_weights: The weight tuple selected for each held-out fold (the
            weights fit on the other N-1 folds). Logged for transparency.
    """

    fold_scores: list[float]
    per_fold_weights: list[tuple[float, ...]]

    @property
    def mean(self) -> float:
        """Mean of the held-out per-fold scores."""
        return float(np.mean(self.fold_scores))

    @property
    def std(self) -> float:
        """Population std (ddof=0) of the held-out per-fold scores."""
        return float(np.std(self.fold_scores))


def nested_cv_weight_blend(
    oof_matrices: list[np.ndarray],
    y: np.ndarray,
    folds: dict[str, np.ndarray],
    n_steps: int = _GRID_STEPS,
) -> NestedWeightResult:
    """Score a weight-optimised blend with leakage-free nested cross-validation.

    For each held-out fold k:
      1. Select blend weights on the OTHER N-1 folds' OOF val rows (inner objective).
      2. Apply those weights to fold k's OOF rows and score balanced_accuracy.
    Fold k never participates in choosing its own weights, so the resulting 5
    per-fold scores are an honest, paired-with-reference estimate of a
    weight-optimised blend — NOT the optimistic same-fold number.

    Args:
        oof_matrices: Per-model OOF probability matrices (all (N, N_CLASSES)).
        y: Integer-encoded target (positional).
        folds: Frozen fold index dict.
        n_steps: Simplex grid resolution for the weight search.

    Returns:
        A NestedWeightResult with the held-out per-fold scores and the per-fold
        selected weights.
    """
    grid = _simplex_grid(len(oof_matrices), n_steps=n_steps)
    fold_scores: list[float] = []
    per_fold_weights: list[tuple[float, ...]] = []

    for k in range(N_SPLITS):
        train_fold_ids = [j for j in range(N_SPLITS) if j != k]
        w = _best_weights_on_folds(oof_matrices, y, folds, train_fold_ids, grid)

        va_idx = folds[f"fold{k}_val"]
        blended_k = blend_probabilities([m[va_idx] for m in oof_matrices], weights=list(w))
        score = float(balanced_accuracy_score(y[va_idx], blended_k.argmax(axis=1)))
        fold_scores.append(score)
        per_fold_weights.append(w)
        logger.info("nested-weight fold=%d weights=%s held_out_bal_acc=%.6f", k, w, score)

    return NestedWeightResult(fold_scores=fold_scores, per_fold_weights=per_fold_weights)


@dataclass
class PairedDelta:
    """Paired comparison of a candidate's per-fold scores vs the LightGBM reference.

    Attributes:
        diff_mean: mean_k(candidate_k - reference_k).
        diff_sem: std(diff, ddof=1)/sqrt(n) — the SEM of the paired differences.
        defensible: True iff diff_mean > _SIGMA_K * diff_sem (a ~2-sigma paired
            signal on 5 folds).
    """

    diff_mean: float
    diff_sem: float
    defensible: bool


def paired_delta(candidate_scores: list[float], reference_scores: list[float]) -> PairedDelta:
    """Compute the paired-difference statistics of candidate vs reference.

    This is the same arithmetic as feature_eval.attach_paired_stats, exposed as a
    small pure function so it can be unit-tested directly and reused for every
    candidate (each base model and the blend) against the LightGBM reference.

    Args:
        candidate_scores: The candidate's per-fold scores (length N_SPLITS).
        reference_scores: The LightGBM reference per-fold scores (length N_SPLITS).

    Returns:
        A PairedDelta with diff_mean, diff_sem and the defensibility flag.

    Raises:
        ValueError: If the two score vectors differ in length (fold mismatch).
    """
    if len(candidate_scores) != len(reference_scores):
        raise ValueError("fold count mismatch between candidate and reference")
    diffs = np.asarray(candidate_scores, dtype=np.float64) - np.asarray(
        reference_scores, dtype=np.float64
    )
    diff_mean = float(np.mean(diffs))
    diff_sem = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    return PairedDelta(
        diff_mean=diff_mean,
        diff_sem=diff_sem,
        defensible=bool(diff_mean > _SIGMA_K * diff_sem),
    )


def check_lgbm_fidelity(lgbm_scores: list[float], reference_scores: list[float]) -> bool:
    """Assert our re-fitted LightGBM reproduces the C5 per-fold scores.

    If the re-fitted LightGBM's per-fold scores do not match the C5
    best_deterministic scores to within _FIDELITY_ATOL, we are not blending the
    model we claim to beat, and the whole paired comparison is void. This guard
    makes that failure loud instead of silent.

    Args:
        lgbm_scores: Our re-fitted LightGBM per-fold scores.
        reference_scores: The C5 best_deterministic.fold_scores.

    Returns:
        True if every fold matches within tolerance.

    Raises:
        AssertionError: On any per-fold mismatch beyond _FIDELITY_ATOL.
    """
    a = np.asarray(lgbm_scores, dtype=np.float64)
    b = np.asarray(reference_scores, dtype=np.float64)
    ok = bool(np.allclose(a, b, atol=_FIDELITY_ATOL))
    assert ok, (
        "LightGBM fidelity check FAILED: re-fitted per-fold scores do not match the "
        f"C5 reference within atol={_FIDELITY_ATOL}.\n  refit:     {lgbm_scores}\n  "
        f"reference: {reference_scores}"
    )
    return ok


def build_submission(
    df_train: pd.DataFrame,
    y: np.ndarray,
    encoder: LabelEncoder,
    weights: list[float] | None = None,
) -> Path:
    """Train all three models on ALL of train, blend test probabilities, write CSV.

    The CV estimate validates the blend; the deliverable refits each base model on
    the FULL training set (no held-out fold) and blends their TEST probabilities,
    then decodes argmax back to GALAXY/STAR/QSO via the train-fitted LabelEncoder.

    Leakage note: load_test is called HERE and only here, AFTER the CV verdict.
    Test never enters CV, weighting or any per-fold fit.

    Args:
        df_train: Raw training frame (id + class present).
        y: Integer-encoded train target (positional, aligned to df_train).
        encoder: LabelEncoder already fitted on the train target (for inverse).
        weights: Optional blend weights (None => equal). Pass through to
            blend_probabilities so the submission uses the SAME blend the CV scored.

    Returns:
        Path to the written submission CSV.
    """
    x_train, categorical = build_feature_frame(df_train)
    cat_idx = [list(x_train.columns).index(c) for c in categorical]
    w_full = compute_sample_weights(y)

    df_test = load_test()
    x_test, _ = build_feature_frame(df_test)

    # Each model trained on the FULL train set, predicting the test set once.
    proba_lgbm = fit_predict_lgbm(x_train, y, w_full, x_test, cat_idx)
    proba_cat = fit_predict_catboost(x_train, y, w_full, x_test, categorical)
    proba_xgb = fit_predict_xgboost(x_train, y, w_full, x_test)

    blended = blend_probabilities([proba_lgbm, proba_cat, proba_xgb], weights=weights)
    pred_labels = encoder.inverse_transform(blended.argmax(axis=1))

    submission = pd.DataFrame(
        {ID_COLUMN: df_test[ID_COLUMN].to_numpy(), TARGET: pred_labels}
    )
    DATA_DIR_OUT.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    logger.info("Wrote submission %s (%d rows)", SUBMISSION_PATH, len(submission))
    return SUBMISSION_PATH


def _log_mlflow(summary: dict[str, object]) -> bool:
    """Best-effort MLflow logging of the ensemble run; True if it ran, else False.

    Logs a parent run with the seed/threads/GPU decision, then one child run per
    base model and one for the blend, each carrying per-fold scores, mean/std and
    (for non-reference candidates) the paired delta vs LightGBM. MLflow is optional;
    on absence/error we warn and continue — the JSON is the canonical record.

    Args:
        summary: The assembled results dict (the same one written to JSON).

    Returns:
        True if MLflow logging completed, False if skipped.
    """
    try:
        import mlflow  # noqa: PLC0415
    except ImportError:
        logger.warning("mlflow not installed; skipping MLflow logging (JSON is the record)")
        return False

    try:
        import os

        # MLflow 3.x refuses a local file store unless this opt-out is set; the POC
        # uses a local mlruns/ dir by design (single host). setdefault so an
        # explicit caller override still wins.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        models = summary["models"]  # type: ignore[index]
        with mlflow.start_run(run_name="ensemble"):
            mlflow.log_params(
                {
                    "seed": SEED,
                    "n_splits": N_SPLITS,
                    "cpu_threads": _CPU_THREADS,
                    "catboost_device": "GPU",
                    "blend": "equal_weight_soft_voting",
                    "reference": "c5_lightgbm_tuned",
                }
            )
            for entry in models:  # type: ignore[attr-defined]
                with mlflow.start_run(run_name=entry["name"], nested=True):
                    mlflow.log_metric("balanced_accuracy_mean", entry["mean"])
                    mlflow.log_metric("balanced_accuracy_std", entry["std"])
                    for k, s in enumerate(entry["fold_scores"]):
                        mlflow.log_metric("balanced_accuracy_fold", s, step=k)
                    if entry.get("paired_diff_mean") is not None:
                        mlflow.log_metric("paired_diff_mean", entry["paired_diff_mean"])
                        mlflow.log_metric("paired_diff_sem", entry["paired_diff_sem"])
                        mlflow.log_metric("defensible", float(bool(entry["defensible"])))
        logger.info("MLflow runs logged under %s (experiment %s)", MLFLOW_DIR, MLFLOW_EXPERIMENT)
        return True
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("MLflow logging failed (%s); continuing with JSON record", exc)
        return False


def run(*, make_submission: bool = True) -> dict[str, object]:
    """Run the full OOF ensemble evaluation and persist the canonical results JSON.

    Steps:
      1. Load train; label-encode the target; load frozen folds; build the raw
         feature matrix once.
      2. Compute OOF probabilities + per-fold scores for LightGBM, CatBoost, XGBoost.
      3. FIDELITY GUARD: assert our LightGBM reproduces the C5 reference per-fold.
      4. Equal-weight blend the three OOF matrices; score it per fold.
      5. Paired delta of the blend (and of each base model) vs the LightGBM
         reference; decide if the blend beats LightGBM DEFENSIBLY.
      6. Build the submission ONLY if the blend wins defensibly (no complexity
         without measured gain).
      7. Persist docs/c6-build/ensemble_results.json (+ optional MLflow runs).

    Args:
        make_submission: If True AND the blend wins defensibly, train on all of
            train and write the submission CSV. A non-defensible blend never
            produces a submission (we recommend the C5 LightGBM instead).

    Returns:
        The JSON-serialisable summary dict (also written to RESULTS_PATH).
    """
    started = time.time()
    df = load_train()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].to_numpy())
    folds = load_folds()

    x, categorical = build_feature_frame(df)
    cat_idx = [list(x.columns).index(c) for c in categorical]

    reference_scores = load_lgbm_reference_scores()

    # Per-model OOF. The categorical handling is baked into each closure so the
    # generic compute_oof loop stays model-agnostic.
    lgbm_oof = compute_oof(
        x, y, folds, "lightgbm",
        lambda xt, yt, wt, xp: fit_predict_lgbm(xt, yt, wt, xp, cat_idx),
    )
    cat_oof = compute_oof(
        x, y, folds, "catboost",
        lambda xt, yt, wt, xp: fit_predict_catboost(xt, yt, wt, xp, categorical),
    )
    xgb_oof = compute_oof(
        x, y, folds, "xgboost",
        lambda xt, yt, wt, xp: fit_predict_xgboost(xt, yt, wt, xp),
    )

    # FIDELITY: our LightGBM must equal the C5 reference, else the comparison is void.
    fidelity_ok = check_lgbm_fidelity(lgbm_oof.fold_scores, reference_scores)

    # Equal-weight soft-voting blend (the headline; no weight overfitting).
    blended_oof = blend_probabilities(
        [lgbm_oof.oof_proba, cat_oof.oof_proba, xgb_oof.oof_proba]
    )
    blend_scores = score_blend_per_fold(blended_oof, y, folds)
    blend_mean = float(np.mean(blend_scores))
    blend_std = float(np.std(blend_scores))

    blend_delta = paired_delta(blend_scores, reference_scores)
    cat_delta = paired_delta(cat_oof.fold_scores, reference_scores)
    xgb_delta = paired_delta(xgb_oof.fold_scores, reference_scores)

    models_record = [
        {
            "name": "lightgbm",  # the paired reference itself
            "fold_scores": lgbm_oof.fold_scores,
            "mean": lgbm_oof.mean,
            "std": lgbm_oof.std,
            "paired_diff_mean": None,
            "paired_diff_sem": None,
            "defensible": None,
        },
        {
            "name": "catboost",
            "fold_scores": cat_oof.fold_scores,
            "mean": cat_oof.mean,
            "std": cat_oof.std,
            "paired_diff_mean": cat_delta.diff_mean,
            "paired_diff_sem": cat_delta.diff_sem,
            "defensible": cat_delta.defensible,
        },
        {
            "name": "xgboost",
            "fold_scores": xgb_oof.fold_scores,
            "mean": xgb_oof.mean,
            "std": xgb_oof.std,
            "paired_diff_mean": xgb_delta.diff_mean,
            "paired_diff_sem": xgb_delta.diff_sem,
            "defensible": xgb_delta.defensible,
        },
        {
            "name": "blend_equal",
            "fold_scores": blend_scores,
            "mean": blend_mean,
            "std": blend_std,
            "paired_diff_mean": blend_delta.diff_mean,
            "paired_diff_sem": blend_delta.diff_sem,
            "defensible": blend_delta.defensible,
        },
    ]

    oof_matrices = [lgbm_oof.oof_proba, cat_oof.oof_proba, xgb_oof.oof_proba]

    # If the equal-weight blend does not already win defensibly, try a weight-
    # optimised blend — but ONLY via nested CV (weights chosen without the fold they
    # are scored on), so the reported number carries no weight-selection leakage.
    # We always run it here for transparency (it is cheap: scoring 91 grid points on
    # precomputed OOF probabilities), and report both blends.
    nested = nested_cv_weight_blend(oof_matrices, y, folds)
    nested_delta = paired_delta(nested.fold_scores, reference_scores)
    models_record.append(
        {
            "name": "blend_nested_weights",
            "fold_scores": nested.fold_scores,
            "mean": nested.mean,
            "std": nested.std,
            "paired_diff_mean": nested_delta.diff_mean,
            "paired_diff_sem": nested_delta.diff_sem,
            "defensible": nested_delta.defensible,
            "per_fold_weights": [list(w) for w in nested.per_fold_weights],
        }
    )

    # Choose the blend to (potentially) ship: prefer the equal-weight blend (simpler)
    # when it is defensible; else the nested-weight blend if IT is defensible. The
    # nested per-fold weights are not a single deployable vector, so for the
    # submission we re-select ONE weight vector on the FULL OOF (all 5 folds) — this
    # is sound because the submission model is trained on ALL of train (there is no
    # held-out fold to leak), and the CV verdict that GATES the submission already
    # came from the leakage-free nested scores.
    shipped_blend: str | None = None
    ship_weights: list[float] | None = None
    if blend_delta.defensible:
        shipped_blend, ship_weights = "equal", None
    elif nested_delta.defensible:
        full_weights = _best_weights_on_folds(
            oof_matrices, y, folds, list(range(N_SPLITS)), _simplex_grid(len(oof_matrices))
        )
        shipped_blend, ship_weights = "nested_weights", list(full_weights)

    submission_path: str | None = None
    if make_submission and shipped_blend is not None:
        submission_path = str(build_submission(df, y, encoder, weights=ship_weights))

    summary: dict[str, object] = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "metric": "balanced_accuracy",
        "sigma_k": _SIGMA_K,
        "cpu_threads": _CPU_THREADS,
        "catboost_device": "GPU",
        "catboost_tuned": CATBOOST_TUNING_PATH.is_file(),
        "xgboost_tuned": XGBOOST_TUNING_PATH.is_file(),
        "blend": "soft_voting",
        "reference": "c5_lightgbm_tuned",
        "reference_scores": reference_scores,
        "lgbm_fidelity_ok": fidelity_ok,
        "class_order": list(encoder.classes_),
        "models": models_record,
        "blend_equal_beats_lightgbm_point": blend_mean > float(np.mean(reference_scores)),
        "blend_equal_beats_lightgbm_defensibly": blend_delta.defensible,
        "blend_nested_beats_lightgbm_defensibly": nested_delta.defensible,
        "shipped_blend": shipped_blend,
        "shipped_full_train_weights": ship_weights,
        "reaches_target_0_97": max(blend_mean, nested.mean) >= 0.97,
        "submission_path": submission_path,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    summary["mlflow_logged"] = _log_mlflow(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", RESULTS_PATH)
    return summary


def main() -> None:
    """CLI entry point: run the ensemble evaluation and print a headline digest."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    np.random.seed(SEED)
    summary = run()
    for entry in summary["models"]:  # type: ignore[index]
        delta = entry["paired_diff_mean"]
        tail = (
            ""
            if delta is None
            else f" delta={delta:+.6f} sem={entry['paired_diff_sem']:.6f} "
            f"defensible={entry['defensible']}"
        )
        print(f"{entry['name']:>20s}  mean={entry['mean']:.6f} std={entry['std']:.6f}{tail}")
    print(
        f"equal blend defensible: {summary['blend_equal_beats_lightgbm_defensibly']} | "
        f"nested-weight blend defensible: {summary['blend_nested_beats_lightgbm_defensibly']} | "
        f"shipped: {summary['shipped_blend']} | reaches 0.97: {summary['reaches_target_0_97']} | "
        f"submission: {summary['submission_path']}"
    )


if __name__ == "__main__":
    main()
