"""Exploratory Data Analysis (train-only) + SHAP baseline for Playground S6E6.

Cycle C2 (Data) deliverable BL-004 — the FINAL step of C2, which closes the
cycle. This script characterises the TRAINING data only (stellar classification:
GALAXY / QSO / STAR) and produces an exploratory "ceiling signal" via a quick
LightGBM 5-fold cross-validation over the FROZEN folds (BL-005), plus a SHAP
feature-importance baseline.

Hard scope boundaries (enforced by construction, not just by convention):
  - TRAIN ONLY. This module never imports load_test and never reads test.csv.
    Touching test here would leak knowledge of the holdout into C2/C3 decisions.
  - FOLDS ONLY FROM cv_indices.npz. We do not re-split, re-seed, or invent any
    partition. Every CV number is measured on the identical frozen partition the
    rest of the pipeline uses, so the numbers are comparable downstream.
  - FEATURE ENGINEERING IS MOTIVATED, NOT IMPLEMENTED. Colour indices (u-g, g-r,
    r-i, i-z) and redshift transforms are discussed in the report as C3 work.
    This script trains on the RAW features only — it produces a ceiling signal,
    NOT the formal baseline (that is BL-008 in C5).

What this is and is NOT:
  - IS: an exploratory characterisation + a SHAP-ranked feature signal to hand
    off to C3 (feature engineering) and the eventual modelling cycles.
  - IS NOT: the official baseline model. The official metric is
    balanced_accuracy; we report it here only as a ceiling signal so C3 knows
    which raw features carry the discriminative load.

Determinism: every stochastic step (LightGBM, the SHAP subsample) is pinned to
SEED = 42. Two runs produce identical numbers and identical figures.

Side effects:
  - Writes PNG figures under docs/c2-data/figures/.
  - Emits structured log records (no emojis). The caller owns log configuration;
    main() attaches a basic handler so the script is runnable standalone.
  - Does NOT mutate any data file and does NOT write to data/.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import matplotlib

# Use a non-interactive backend: this script runs headless (no display) and only
# writes PNGs to disk. Must be set before importing pyplot.
matplotlib.use("Agg")

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats as scipy_stats
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.preprocessing import LabelEncoder

from src.cv_utils import compute_sample_weights, load_folds
from src.data_loader import TARGET, load_train

# --------------------------------------------------------------------------- #
# Reproducibility and layout constants.
# --------------------------------------------------------------------------- #
SEED: Final[int] = 42

# Numeric and categorical feature groups. `id` is an identity column (dropped
# before any analysis — it is not a feature). These lists are the single source
# of truth for which columns are treated as what throughout the script.
NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "alpha",
    "delta",
    "u",
    "g",
    "r",
    "i",
    "z",
    "redshift",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("spectral_type", "galaxy_population")
ALL_FEATURES: Final[tuple[str, ...]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# The five SDSS photometric bands, in wavelength order (ultraviolet -> infrared).
# Used for the band-correlation analysis and to motivate colour-index features.
PHOTOMETRIC_BANDS: Final[tuple[str, ...]] = ("u", "g", "r", "i", "z")

# Class order used consistently in every per-class report (confusion matrix,
# recall vector, SHAP-per-class). Fixed so columns/rows never silently reorder
# between runs or between figures.
CLASS_ORDER: Final[tuple[str, ...]] = ("GALAXY", "QSO", "STAR")

# Output directory for figures. Resolved relative to the repo root (two levels
# up from this file: src/ -> repo root) so the script is location-independent.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
FIGURES_DIR: Final[Path] = _REPO_ROOT / "docs" / "c2-data" / "figures"

# Frozen CV folds (BL-005) are loaded via src.cv_utils.load_folds (BL-019), which
# owns FOLDS_PATH. N_SPLITS is re-stated here for the local CV-loop range.
N_SPLITS: Final[int] = 5

# SHAP is O(n) in trees but the explanation step over 577k rows is needlessly
# slow for an exploratory signal. We subsample the SHAP background/explanation
# set (the MODEL still trains on the full fold). 4000 rows is enough for a stable
# mean(|SHAP|) ranking; the subsample is seeded so it is reproducible.
SHAP_SAMPLE_SIZE: Final[int] = 4000

# Normality test switch point. Shapiro-Wilk is appropriate for small n; for the
# large per-class slices here we use D'Agostino-Pearson (normaltest), which is
# valid at large n. We still cap the sample fed to the test because Shapiro is
# undefined above ~5000 and even normaltest is dominated by trivial deviations
# at n>1e5 — so we report the statistic on a seeded 5000-row subsample purely as
# a descriptive shape signal, NOT as a hard accept/reject of normality.
_NORMALITY_SAMPLE: Final[int] = 5000

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data quality + univariate statistics.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSummary:
    """Per-feature univariate statistics with downstream interpretation hooks.

    Attributes:
        name: Feature column name.
        mean / median / std: Central tendency and dispersion.
        skewness / kurtosis: Distribution shape. |skew| > 1 flags a transform
            candidate; Fisher kurtosis (excess) >> 0 flags heavy tails / outliers.
        nan_fraction: Fraction of missing values (expected 0.0 here).
        normality_stat / normality_p: D'Agostino-Pearson statistic and p-value on
            a seeded subsample, as a descriptive shape signal (see _NORMALITY_*).
    """

    name: str
    mean: float
    median: float
    std: float
    skewness: float
    kurtosis: float
    nan_fraction: float
    normality_stat: float
    normality_p: float


def compute_quality_report(df: pd.DataFrame) -> dict[str, object]:
    """Build the dataset-level data-quality summary.

    Args:
        df: The loaded training frame (including id and class).

    Returns:
        Dict with shape, total NaN count, duplicate-row count, the target class
        distribution (counts + proportions), and the imbalance ratio
        (majority / minority). All values are plain Python types so the dict is
        JSON-serialisable for the report.
    """
    class_counts = df[TARGET].value_counts()
    class_props = (class_counts / len(df)).round(6)
    majority = int(class_counts.max())
    minority = int(class_counts.min())

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "total_nan": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "class_proportions": {k: float(v) for k, v in class_props.items()},
        "imbalance_ratio_majority_over_minority": round(majority / minority, 4),
    }


def _normality_signal(series: pd.Series, rng: np.random.Generator) -> tuple[float, float]:
    """Return a D'Agostino-Pearson (statistic, p-value) on a seeded subsample.

    We subsample because the test is dominated by trivial deviations at n>1e5,
    and we want a comparable shape signal across features, not a hard hypothesis
    test. NaN-free input is assumed (the dataset has none); we drop defensively.

    Args:
        series: A numeric feature column.
        rng: Seeded NumPy Generator for the subsample (reproducibility).

    Returns:
        (statistic, p_value). Returns (nan, nan) if the column has < 8 valid
        points (normaltest requires at least 8).
    """
    values = series.dropna().to_numpy()
    if values.size > _NORMALITY_SAMPLE:
        values = rng.choice(values, size=_NORMALITY_SAMPLE, replace=False)
    if values.size < 8:
        return float("nan"), float("nan")
    stat, pvalue = scipy_stats.normaltest(values)
    return float(stat), float(pvalue)


def compute_univariate(df: pd.DataFrame) -> list[FeatureSummary]:
    """Compute univariate statistics for every numeric feature.

    Reports the mandated statistics (mean, median, std, skewness, kurtosis,
    NaN fraction) plus a normality shape signal. Categorical features are
    summarised separately via value_counts in the cross-tab analysis.

    Args:
        df: The loaded training frame.

    Returns:
        One FeatureSummary per numeric feature, in NUMERIC_FEATURES order.
    """
    rng = np.random.default_rng(SEED)
    summaries: list[FeatureSummary] = []
    for col in NUMERIC_FEATURES:
        series = df[col]
        stat, pvalue = _normality_signal(series, rng)
        summaries.append(
            FeatureSummary(
                name=col,
                mean=float(series.mean()),
                median=float(series.median()),
                std=float(series.std()),
                # Fisher-Pearson skewness; bias-corrected via pandas default.
                skewness=float(series.skew()),
                # pandas .kurt() returns EXCESS (Fisher) kurtosis: 0 == normal.
                kurtosis=float(series.kurt()),
                nan_fraction=float(series.isna().mean()),
                normality_stat=stat,
                normality_p=pvalue,
            )
        )
    return summaries


def compute_redshift_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise redshift per class — the dominant separability signal.

    redshift is the highest-MI feature (context: MI/H = 0.584). Quantifying its
    per-class location and spread tells C3 whether a raw split or a transform
    (log1p, binning) is warranted.

    Args:
        df: The loaded training frame.

    Returns:
        A DataFrame indexed by class (CLASS_ORDER) with columns
        [count, mean, median, std, q05, q95]. q05/q95 bracket the bulk of each
        class to expose overlap between classes.
    """
    grouped = df.groupby(TARGET, observed=True)["redshift"]
    table = pd.DataFrame(
        {
            "count": grouped.count(),
            "mean": grouped.mean(),
            "median": grouped.median(),
            "std": grouped.std(),
            "q05": grouped.quantile(0.05),
            "q95": grouped.quantile(0.95),
        }
    )
    # Reindex to the canonical class order so the table never reorders by chance.
    return table.reindex(list(CLASS_ORDER))


def compute_band_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix among the five SDSS photometric bands.

    The bands u,g,r,i,z are expected to be highly correlated (a bright object is
    bright in every band), which is exactly why their DIFFERENCES (colours) are
    the physically meaningful, lower-collinearity features. This matrix is the
    evidence that motivates colour-index engineering in C3.

    Args:
        df: The loaded training frame.

    Returns:
        A 5x5 Pearson correlation DataFrame over PHOTOMETRIC_BANDS.
    """
    return df[list(PHOTOMETRIC_BANDS)].corr(method="pearson")


def compute_categorical_crosstabs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Row-normalised cross-tabulations of each categorical feature vs class.

    Row normalisation answers "given this category, how do the classes split?",
    which is the form that matters for predictiveness (e.g. spectral_type == M
    -> mostly GALAXY). Cramer's V (computed in the report layer) quantifies the
    overall association strength.

    Args:
        df: The loaded training frame.

    Returns:
        {feature_name: crosstab DataFrame} for each categorical feature, with
        class columns in CLASS_ORDER and proportions per row.
    """
    crosstabs: dict[str, pd.DataFrame] = {}
    for col in CATEGORICAL_FEATURES:
        ct = pd.crosstab(df[col], df[TARGET], normalize="index")
        # Ensure class columns appear in canonical order even if a category is
        # missing a class (reindex fills absent classes with 0, not NaN drift).
        ct = ct.reindex(columns=list(CLASS_ORDER), fill_value=0.0)
        crosstabs[col] = ct.round(4)
    return crosstabs


def cramers_v(confusion: pd.DataFrame) -> float:
    """Compute bias-corrected Cramer's V for a categorical-vs-categorical table.

    Cramer's V in [0, 1] measures association strength between two nominal
    variables (here: a categorical feature vs the class target). We use the
    bias-corrected variant (Bergsma 2013) so the statistic is not inflated for
    small tables / large samples.

    Args:
        confusion: A contingency table of raw COUNTS (not proportions).

    Returns:
        Bias-corrected Cramer's V. Returns 0.0 for a degenerate (single
        row/col) table where association is undefined.
    """
    chi2 = scipy_stats.chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    # Bias correction (Bergsma): subtract the expected chi2 under independence.
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


# --------------------------------------------------------------------------- #
# Figures.
# --------------------------------------------------------------------------- #
def _save_fig(fig: plt.Figure, name: str) -> Path:
    """Save a figure to FIGURES_DIR as PNG and close it to free memory.

    Args:
        fig: The matplotlib Figure to persist.
        name: File stem (without extension).

    Returns:
        The path written.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote figure %s", path)
    return path


def plot_target_distribution(df: pd.DataFrame) -> Path:
    """Bar plot of the class distribution (absolute counts + class imbalance)."""
    counts = df[TARGET].value_counts().reindex(list(CLASS_ORDER))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.to_numpy(), ax=ax, hue=counts.index,
                palette="viridis", legend=False)
    ax.set_title("Target class distribution (train)")
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    for i, v in enumerate(counts.to_numpy()):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    return _save_fig(fig, "target_distribution")


def plot_numeric_univariate(df: pd.DataFrame) -> Path:
    """Grid of histogram+KDE for every numeric feature (distribution shapes)."""
    n = len(NUMERIC_FEATURES)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.ravel()
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.histplot(df[col], bins=80, kde=True, ax=ax, color="steelblue")
        ax.set_title(col)
        ax.set_xlabel("")
    # Hide any unused subplot cells so the grid is clean.
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Univariate distributions — numeric features (train)", y=1.01)
    return _save_fig(fig, "numeric_univariate")


def plot_numeric_boxplots(df: pd.DataFrame) -> Path:
    """Grid of boxplots per numeric feature (outlier / spread overview)."""
    n = len(NUMERIC_FEATURES)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.ravel()
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(y=df[col], ax=ax, color="lightcoral")
        ax.set_title(col)
        ax.set_ylabel("")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Boxplots — numeric features (train)", y=1.01)
    return _save_fig(fig, "numeric_boxplots")


def plot_redshift_by_class(df: pd.DataFrame) -> Path:
    """Redshift distribution per class — the headline separability figure.

    Two panels: a clipped histogram (linear, to show STAR/GALAXY near zero) and
    a log1p-scaled violin (to show the QSO tail without it dominating the axis).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: clipped linear histogram. redshift has a long QSO tail (up to ~7);
    # clipping the x-axis at the 99th pct keeps the STAR/GALAXY structure legible
    # without discarding any data from the underlying statistics.
    clip = float(df["redshift"].quantile(0.99))
    for cls in CLASS_ORDER:
        sub = df.loc[df[TARGET] == cls, "redshift"]
        sns.histplot(sub.clip(upper=clip), bins=80, ax=axes[0], label=cls,
                     stat="density", element="step", fill=False)
    axes[0].set_title(f"redshift by class (x clipped at p99={clip:.2f})")
    axes[0].set_xlabel("redshift")
    axes[0].legend()

    # Right: log1p violin to compress the QSO tail. log1p is safe because the
    # contract floor is -0.01; we shift by the observed min so the transform is
    # defined for the tiny negative-noise values.
    shift = float(df["redshift"].min())
    logz = np.log1p(df["redshift"] - shift)
    plot_df = pd.DataFrame({"class": df[TARGET].to_numpy(), "log1p_redshift": logz})
    sns.violinplot(data=plot_df, x="class", y="log1p_redshift",
                   order=list(CLASS_ORDER), ax=axes[1], hue="class",
                   palette="viridis", legend=False)
    axes[1].set_title("log1p(redshift) by class")
    return _save_fig(fig, "redshift_by_class")


def plot_band_correlation_heatmap(corr: pd.DataFrame) -> Path:
    """Heatmap of the photometric-band Pearson correlation matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="rocket_r", vmin=0, vmax=1,
                square=True, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Photometric band correlation (u,g,r,i,z)")
    return _save_fig(fig, "band_correlation_heatmap")


def plot_feature_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Heatmap of the full numeric-feature Spearman correlation matrix.

    Spearman (rank) rather than Pearson here because several features (redshift)
    are heavily skewed; rank correlation captures monotonic association without
    being dominated by the tail.
    """
    corr = df[list(NUMERIC_FEATURES)].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                square=True, ax=ax, cbar_kws={"label": "Spearman rho"})
    ax.set_title("Numeric feature correlation (Spearman)")
    return _save_fig(fig, "feature_correlation_heatmap")


def plot_categorical_crosstabs(crosstabs: dict[str, pd.DataFrame]) -> Path:
    """Stacked-bar visual of each categorical feature's class composition."""
    fig, axes = plt.subplots(1, len(crosstabs), figsize=(6 * len(crosstabs), 4.5))
    if len(crosstabs) == 1:
        axes = [axes]
    for ax, (name, ct) in zip(axes, crosstabs.items()):
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.set_title(f"{name} -> class composition")
        ax.set_ylabel("proportion")
        ax.set_xlabel(name)
        ax.legend(title="class", bbox_to_anchor=(1.0, 1.0))
        ax.tick_params(axis="x", rotation=30)
    return _save_fig(fig, "categorical_crosstabs")


# --------------------------------------------------------------------------- #
# LightGBM quick CV (ceiling signal) + SHAP baseline.
# --------------------------------------------------------------------------- #
@dataclass
class CVResult:
    """Aggregated cross-validation outcome over the frozen folds.

    Attributes:
        fold_balanced_accuracy: balanced_accuracy per fold (length N_SPLITS).
        mean_balanced_accuracy / std_balanced_accuracy: aggregate of the above.
        confusion: confusion matrix summed across all out-of-fold predictions,
            rows/cols in CLASS_ORDER (true x predicted).
        per_class_recall: out-of-fold recall per class (CLASS_ORDER), i.e. the
            diagonal of the row-normalised confusion. balanced_accuracy is the
            unweighted mean of these.
    """

    fold_balanced_accuracy: list[float] = field(default_factory=list)
    mean_balanced_accuracy: float = 0.0
    std_balanced_accuracy: float = 0.0
    confusion: np.ndarray | None = None
    per_class_recall: dict[str, float] = field(default_factory=dict)


def _prepare_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Build the model design matrix X and encoded target y from the raw frame.

    Categorical features are cast to pandas `category` dtype so LightGBM uses its
    NATIVE categorical handling (optimal split search over category subsets) —
    this is the correct, leakage-free treatment for nominal features and avoids
    imposing a false ordinal order. The target is label-encoded; we return the
    encoder so predictions can be mapped back to class names.

    No scaling/imputation is applied: LightGBM is scale-invariant and the data
    has no missing values (verified upstream). RAW features only — colours and
    transforms are deliberately left for C3.

    Args:
        df: The loaded training frame (with id and class).

    Returns:
        (X, y, label_encoder) where X has ALL_FEATURES columns (categoricals as
        category dtype), y is the integer-encoded target, and label_encoder maps
        integers back to class strings.
    """
    x = df[list(ALL_FEATURES)].copy()
    for col in CATEGORICAL_FEATURES:
        x[col] = x[col].astype("category")

    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].to_numpy())
    return x, y, encoder


def _lgbm_params() -> dict[str, object]:
    """Return the quick-CV LightGBM hyperparameters (deliberately modest).

    These are NOT tuned — this is a ceiling signal, not a baseline. A shallow,
    moderately-sized GBM is enough to reveal which raw features carry signal.
    multiclass objective with 3 classes; deterministic via random_state and
    single-threaded determinism flags left at LightGBM defaults (its CPU
    histogram algorithm is deterministic for fixed data + seed + n_jobs).

    Returns:
        Keyword args for lgb.LGBMClassifier.
    """
    return {
        "objective": "multiclass",
        "num_class": len(CLASS_ORDER),
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }


def run_cv(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: dict[str, np.ndarray],
    encoder: LabelEncoder,
) -> tuple[CVResult, lgb.LGBMClassifier, pd.DataFrame, np.ndarray]:
    """Train LightGBM on each frozen fold and aggregate balanced_accuracy.

    For every frozen fold k: fit on fold{k}_train (with balancing sample
    weights), predict fold{k}_val, and record the per-fold balanced_accuracy.
    Out-of-fold predictions are accumulated to form ONE aggregate confusion
    matrix and per-class recall over the whole train set (each row predicted
    exactly once, since the folds partition the rows).

    The LAST fold's fitted model and its training slice are returned so the
    caller can run SHAP on a representative model without retraining.

    Args:
        x: Design matrix (categoricals as category dtype).
        y: Integer-encoded target.
        folds: The frozen fold index dict (positional .iloc indices).
        encoder: Fitted LabelEncoder (for mapping class ids to names).

    Returns:
        (CVResult, last_fold_model, last_fold_train_X, last_fold_train_y).

    Raises:
        KeyError: If a required fold key is missing from `folds`.
    """
    categorical_idx = [list(ALL_FEATURES).index(c) for c in CATEGORICAL_FEATURES]
    fold_scores: list[float] = []

    # Out-of-fold prediction buffer: every row gets exactly one prediction.
    oof_pred = np.full(len(y), -1, dtype=np.int64)

    last_model: lgb.LGBMClassifier | None = None
    last_train_x: pd.DataFrame | None = None
    last_train_y: np.ndarray | None = None

    for k in range(N_SPLITS):
        tr_idx = folds[f"fold{k}_train"]
        va_idx = folds[f"fold{k}_val"]

        x_tr, y_tr = x.iloc[tr_idx], y[tr_idx]
        x_va, y_va = x.iloc[va_idx], y[va_idx]

        weights = compute_sample_weights(y_tr)
        model = lgb.LGBMClassifier(**_lgbm_params())
        model.fit(
            x_tr,
            y_tr,
            sample_weight=weights,
            categorical_feature=categorical_idx,
        )

        pred = model.predict(x_va)
        oof_pred[va_idx] = pred
        score = balanced_accuracy_score(y_va, pred)
        fold_scores.append(float(score))
        logger.info("fold %d balanced_accuracy=%.5f", k, score)

        last_model, last_train_x, last_train_y = model, x_tr, y_tr

    # Sanity: the folds must have covered every row exactly once.
    if (oof_pred < 0).any():
        raise RuntimeError("out-of-fold predictions incomplete: folds did not cover all rows")

    # Aggregate confusion + per-class recall over ALL out-of-fold predictions.
    name_to_int = {name: i for i, name in enumerate(encoder.classes_)}
    ordered_int = [name_to_int[c] for c in CLASS_ORDER]
    confusion = confusion_matrix(y, oof_pred, labels=ordered_int)
    recalls = recall_score(y, oof_pred, labels=ordered_int, average=None, zero_division=0)

    result = CVResult(
        fold_balanced_accuracy=fold_scores,
        mean_balanced_accuracy=float(np.mean(fold_scores)),
        std_balanced_accuracy=float(np.std(fold_scores)),
        confusion=confusion,
        per_class_recall={c: float(r) for c, r in zip(CLASS_ORDER, recalls)},
    )
    assert last_model is not None and last_train_x is not None and last_train_y is not None
    return result, last_model, last_train_x, last_train_y


def compute_shap(
    model: lgb.LGBMClassifier,
    x_train: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute the SHAP feature-importance ranking via TreeExplainer.

    Uses TreeExplainer (exact, fast for GBMs) on a seeded subsample of the
    model's own training slice. We report:
      1. Global ranking: mean(|SHAP|) per feature, averaged across all classes
         and samples — the headline feature-importance signal.
      2. Per-class ranking: mean(|SHAP|) per feature for each class, so C3 sees
         WHICH features push toward each class (e.g. redshift toward QSO).

    Args:
        model: A fitted LightGBM multiclass model.
        x_train: The training slice the model was fit on (category dtype intact).

    Returns:
        (global_ranking, per_class_ranking) where:
          - global_ranking is a Series indexed by feature, sorted descending.
          - per_class_ranking is a DataFrame (features x CLASS_ORDER) of mean(
            |SHAP|) per class.
    """
    rng = np.random.default_rng(SEED)
    n = min(SHAP_SAMPLE_SIZE, len(x_train))
    sample_pos = rng.choice(len(x_train), size=n, replace=False)
    x_sample = x_train.iloc[sample_pos]
    logger.info("Computing SHAP on %d sampled rows", n)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)

    # LightGBM multiclass SHAP shape normalisation. Depending on shap version the
    # output is either a list of (n, n_features) arrays (one per class) or a
    # single (n, n_features, n_classes) array. We normalise to a list per class.
    if isinstance(shap_values, list):
        per_class_arrays = shap_values
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes) -> list over the class axis.
            per_class_arrays = [arr[:, :, c] for c in range(arr.shape[2])]
        else:  # pragma: no cover - defensive; multiclass should never be 2D
            per_class_arrays = [arr]

    feature_names = list(x_train.columns)

    # Per-class mean(|SHAP|).
    per_class = {}
    # explainer.classes_ ordering follows the model's class order (0..n-1), which
    # equals encoder order. We map back to CLASS_ORDER names via the model later;
    # here the model trained on LabelEncoder ints 0,1,2 == sorted class names.
    sorted_class_names = sorted(CLASS_ORDER)  # LabelEncoder sorts alphabetically
    for cls_name, arr in zip(sorted_class_names, per_class_arrays):
        per_class[cls_name] = np.abs(arr).mean(axis=0)
    per_class_df = pd.DataFrame(per_class, index=feature_names)
    # Reorder columns to the canonical CLASS_ORDER for consistent reporting.
    per_class_df = per_class_df.reindex(columns=list(CLASS_ORDER))

    # Global ranking = mean across classes of the per-class mean(|SHAP|).
    global_ranking = per_class_df.mean(axis=1).sort_values(ascending=False)

    return global_ranking, per_class_df


def plot_shap_importance(global_ranking: pd.Series, per_class: pd.DataFrame) -> Path:
    """Bar plot of global mean(|SHAP|) plus a per-class breakdown panel."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    order = global_ranking.index
    sns.barplot(x=global_ranking.to_numpy(), y=list(order), ax=axes[0],
                hue=list(order), palette="mako", legend=False)
    axes[0].set_title("Global feature importance (mean |SHAP|)")
    axes[0].set_xlabel("mean(|SHAP|)")

    per_class.reindex(order).plot(kind="barh", ax=axes[1], colormap="viridis")
    axes[1].set_title("Per-class mean(|SHAP|)")
    axes[1].set_xlabel("mean(|SHAP|)")
    axes[1].invert_yaxis()
    axes[1].legend(title="class")
    return _save_fig(fig, "shap_importance")


def plot_confusion(result: CVResult) -> Path:
    """Heatmap of the aggregate out-of-fold confusion matrix (row-normalised)."""
    assert result.confusion is not None
    cm = result.confusion.astype(np.float64)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)  # row-normalise: recall view
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="Blues",
                xticklabels=list(CLASS_ORDER), yticklabels=list(CLASS_ORDER), ax=ax)
    ax.set_title("Out-of-fold confusion (row-normalised = recall)")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    return _save_fig(fig, "confusion_matrix")


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def main() -> dict[str, object]:
    """Run the full EDA + SHAP baseline and return a JSON-serialisable summary.

    Steps (all train-only):
      1. Load train, build data-quality report.
      2. Univariate stats + redshift-by-class + band corr + categorical crosstabs.
      3. Render all figures to docs/c2-data/figures/.
      4. LightGBM quick CV over frozen folds -> balanced_accuracy + confusion.
      5. SHAP TreeExplainer ranking (global + per-class) on the last fold model.

    Returns:
        A nested dict with every numeric result, suitable for logging or for the
        report layer to read back. Side effect: figures written to disk.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    np.random.seed(SEED)

    logger.info("Loading train (train-only; test is never touched)")
    df = load_train()

    quality = compute_quality_report(df)
    logger.info("Data quality: %s", json.dumps(quality))

    univariate = compute_univariate(df)
    redshift_table = compute_redshift_by_class(df)
    band_corr = compute_band_correlations(df)
    crosstabs = compute_categorical_crosstabs(df)

    # Cramer's V per categorical feature vs class (association strength).
    cramers = {
        col: round(cramers_v(pd.crosstab(df[col], df[TARGET])), 4)
        for col in CATEGORICAL_FEATURES
    }
    logger.info("Cramer's V vs class: %s", cramers)

    # Figures.
    plot_target_distribution(df)
    plot_numeric_univariate(df)
    plot_numeric_boxplots(df)
    plot_redshift_by_class(df)
    plot_band_correlation_heatmap(band_corr)
    plot_feature_correlation_heatmap(df)
    plot_categorical_crosstabs(crosstabs)

    # Modelling: quick CV + SHAP.
    x, y, encoder = _prepare_model_frame(df)
    folds = load_folds()
    cv_result, last_model, last_train_x, _ = run_cv(x, y, folds, encoder)
    logger.info(
        "CV balanced_accuracy = %.5f +/- %.5f (folds=%s)",
        cv_result.mean_balanced_accuracy,
        cv_result.std_balanced_accuracy,
        [round(s, 5) for s in cv_result.fold_balanced_accuracy],
    )
    logger.info("Per-class out-of-fold recall: %s", cv_result.per_class_recall)

    global_shap, per_class_shap = compute_shap(last_model, last_train_x)
    logger.info("SHAP global ranking:\n%s", global_shap.to_string())
    plot_shap_importance(global_shap, per_class_shap)
    plot_confusion(cv_result)

    # Assemble the machine-readable summary.
    summary: dict[str, object] = {
        "quality": quality,
        "univariate": [vars(s) for s in univariate],
        "redshift_by_class": redshift_table.round(4).to_dict(orient="index"),
        "band_correlations": band_corr.round(4).to_dict(),
        "categorical_crosstabs": {k: v.to_dict(orient="index") for k, v in crosstabs.items()},
        "cramers_v": cramers,
        "cv": {
            "fold_balanced_accuracy": cv_result.fold_balanced_accuracy,
            "mean_balanced_accuracy": cv_result.mean_balanced_accuracy,
            "std_balanced_accuracy": cv_result.std_balanced_accuracy,
            "per_class_recall": cv_result.per_class_recall,
            "confusion_rows_true_cols_pred_CLASS_ORDER": (
                cv_result.confusion.tolist() if cv_result.confusion is not None else None
            ),
            "class_order": list(CLASS_ORDER),
        },
        "shap_global_ranking": global_shap.round(6).to_dict(),
        "shap_per_class": per_class_shap.round(6).to_dict(orient="index"),
    }
    logger.info("BL-004 EDA + SHAP baseline complete")
    return summary


if __name__ == "__main__":
    result_summary = main()
    # Emit the full machine-readable summary to stdout so a caller (or the report
    # build) can capture it. Figures are already on disk.
    print(json.dumps(result_summary, indent=2))
