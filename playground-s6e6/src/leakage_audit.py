"""Leakage audit for Kaggle Playground S6E6 (stellar classification).

Cycle C2 (Data) deliverable BL-003. This is the BLOCKING gate that must sign
off before any modelling decision (C3) is taken on this dataset. It quantifies,
per feature, whether the feature carries legitimate signal or is a target proxy
/ leak, and emits a USE | DROP | USE-WITH-ADR verdict for each.

Scope (intentionally narrow — C2 audit only):
  - NO modelling, NO feature engineering, NO imputation (there are 0 NaN here,
    confirmed by BL-002). This module only MEASURES.
  - It is a script that prints a structured, reproducible report to stdout via
    the logger. The human-readable verdict table lives in
    docs/c2-data/leakage-audit.md, authored from this script's output.

What it checks (BL-003, in order):
  1. Mutual information MI(feature; class) and Cramer's V for categoricals,
     each compared against H(class). MI -> H(class) means the feature alone
     determines the target == target proxy == DROP-or-ADR.
  2. Normalised cross-tabs P(class | value) for the two categorical features,
     to detect any value that DETERMINISTICALLY predicts a class.
  3. redshift vs class: confirm the physical ordering STAR < GALAXY < QSO so the
     strong signal is justified as physics, not leakage.
  4. Adversarial validation train(0) vs test(1): a binary classifier under CV.
     AUC >> 0.5 == covariate shift; documented, NON-blocking.
  5. Duplicates: (a) exact feature-row duplicates within train; (b) train
     feature-rows also present in test (cross-split leakage). `id` is excluded
     from the row signature (it is a trivial index, not a feature).
  6. id: confirm MI(id; class) ~ 0 (trivial index, no signal).

Statistical rigor (per math-critic skill):
  - MI is estimated on a fixed-size stratified SAMPLE for the numeric estimator
    (k-NN based) because it is O(n log n) per feature and the discrete-target
    estimator on 577k rows is wasteful; the sample size is reported. Categorical
    MI is computed EXACTLY on the full frame from the contingency table (closed
    form, cheap), so the headline spectral_type / galaxy_population numbers are
    not sampled.
  - Cramer's V uses the bias-corrected form (Bergsma 2013) and is reported with
    the small/medium/large interpretation bands (0.1 / 0.3 / 0.5).

Side effects:
  - Reads train.csv and test.csv via src.data_loader (schema-validated).
  - Emits log records; no files written. Determinism via SEED.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.data_loader import (
    ID_COLUMN,
    N_SPLITS,
    SEED,
    TARGET,
    load_test,
    load_train,
)

logger = logging.getLogger(__name__)

# SEED is imported from data_loader (single source of truth). This audit's
# adversarial StratifiedKFold and the frozen folds (make_folds.py) MUST share
# one seed; importing it here prevents a silent divergence where the two were
# pinned to different values.

# Sample size for the k-NN-based numeric MI estimator. 80k rows is ample for a
# stable MI estimate on 9 features (the estimator's variance is dominated by k,
# not by n once n >> 10k) while keeping the audit fast. Categorical MI is exact
# and uses the full frame regardless of this cap.
_MI_SAMPLE_SIZE: Final[int] = 80_000

# Numeric feature columns (SDSS photometry + astrometry + redshift).
_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "alpha",
    "delta",
    "u",
    "g",
    "r",
    "i",
    "z",
    "redshift",
)
# Categorical feature columns.
_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("spectral_type", "galaxy_population")

# Cramer's V interpretation bands (math-critic skill section 10).
_V_SMALL: Final[float] = 0.1
_V_MEDIUM: Final[float] = 0.3
_V_LARGE: Final[float] = 0.5


@dataclass(frozen=True)
class FeatureMI:
    """Mutual-information / association summary for one feature vs the target.

    Attributes:
        feature: Column name.
        mi_nats: MI(feature; class) in nats (natural log base), >= 0.
        mi_fraction_of_h: mi_nats / H(class) in [0, ~1]. 1.0 == perfect target
            proxy (the feature alone resolves the entire label entropy).
        cramers_v: Bias-corrected Cramer's V in [0, 1], or None for numeric
            features (V is defined for categorical x categorical only).
    """

    feature: str
    mi_nats: float
    mi_fraction_of_h: float
    cramers_v: float | None


def entropy_nats(labels: pd.Series) -> float:
    """Compute Shannon entropy H of a categorical label series, in nats.

    Args:
        labels: Series of discrete class labels.

    Returns:
        H(labels) using natural log. For this dataset's 3-class target the
        value is ~0.8798 nats (== 1.2694 bits).

    Notes:
        Empty inputs return 0.0. Zero-probability classes are skipped (0*log0=0).
    """
    counts = labels.value_counts().to_numpy(dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def cramers_v_bias_corrected(x: pd.Series, y: pd.Series) -> float:
    """Compute the bias-corrected Cramer's V between two categorical series.

    Uses Bergsma's (2013) correction, which removes the upward bias of the naive
    estimator on small/sparse tables. On large samples (577k rows here) the
    correction is negligible, but we apply it for correctness regardless.

    Args:
        x: First categorical series.
        y: Second categorical series (typically the class target).

    Returns:
        V in [0, 1]. 0 == independent, 1 == one variable perfectly determines
        the other. Interpretation bands: 0.1 small / 0.3 medium / 0.5 large.

    Edge cases:
        A degenerate table (one row or one column) yields V = 0.0, since no
        association can be measured against a constant.
    """
    from scipy.stats import chi2_contingency

    table = pd.crosstab(x, y).to_numpy(dtype=np.float64)
    n = table.sum()
    if n == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0

    chi2 = float(chi2_contingency(table, correction=False)[0])
    phi2 = chi2 / n
    r, k = table.shape

    # Bergsma bias correction.
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def categorical_mi_nats(x: pd.Series, y: pd.Series) -> float:
    """Compute the EXACT mutual information MI(x; y) for two categoricals, nats.

    Closed-form plug-in estimate from the joint contingency table. Exact (not
    sampled) and cheap, so the headline categorical numbers are computed on the
    full 577k-row frame.

    Args:
        x: First categorical series (a feature).
        y: Second categorical series (the class target).

    Returns:
        MI(x; y) in nats, >= 0. Compare against H(y): MI == H(y) means x
        determines y exactly (target proxy).

    Notes:
        MI = sum_{i,j} p(i,j) * log( p(i,j) / (p(i) p(j)) ), over cells with
        p(i,j) > 0. Equivalent to H(y) - H(y|x); we compute it from the joint
        directly for numerical symmetry.
    """
    joint = pd.crosstab(x, y).to_numpy(dtype=np.float64)
    n = joint.sum()
    if n == 0:
        return 0.0
    p_xy = joint / n
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    # Outer product of marginals; guard the log against zero cells.
    denom = p_x @ p_y
    mask = p_xy > 0
    mi = float((p_xy[mask] * np.log(p_xy[mask] / denom[mask])).sum())
    # MI is non-negative; clamp tiny negative float noise.
    return max(0.0, mi)


def numeric_mi_nats(
    features: pd.DataFrame, target: pd.Series, sample_size: int = _MI_SAMPLE_SIZE
) -> dict[str, float]:
    """Estimate MI(feature; class) in nats for each numeric feature, on a sample.

    Uses sklearn's k-NN-based estimator (Kraskov/Ross), appropriate for a
    continuous feature vs a discrete target. Runs on a stratified sample of
    `sample_size` rows for speed; the estimator's variance is k-dominated, so
    the sample estimate is stable for n >> 10k.

    Args:
        features: DataFrame of numeric feature columns only.
        target: Discrete class target aligned to `features`.
        sample_size: Stratified sample size. Capped at len(features).

    Returns:
        Mapping {feature_name: mi_nats}. sklearn returns nats (natural log).

    Notes:
        We stratify the sample by class to preserve the 65/20/14 proportions.
        Caveat on the stratified subsample: a class whose `frac * n_class < 0.5`
        could round down to 0 sampled rows and drop out of the estimate. In THIS
        dataset that cannot happen at the default cap — the rarest class (STAR)
        has ~82.7k rows and frac ~= 0.139, so it keeps ~11.5k samples. The
        "rare-class not under-represented" guarantee is therefore scoped to the
        current dataset / default sample_size, not a property of arbitrary
        inputs. For a genuinely tiny class, raise sample_size or skip sampling.
    """
    n = len(features)
    take = min(sample_size, n)
    if take < n:
        # Stratified subsample preserving class proportions.
        frac = take / n
        idx = (
            target.groupby(target, observed=True)
            .sample(frac=frac, random_state=SEED)
            .index
        )
        x_s = features.loc[idx]
        y_s = target.loc[idx]
    else:
        x_s, y_s = features, target

    codes = y_s.astype("category").cat.codes.to_numpy()
    mi = mutual_info_classif(
        x_s.to_numpy(dtype=np.float64),
        codes,
        discrete_features=False,
        random_state=SEED,
    )
    logger.info("Numeric MI estimated on %d sampled rows (of %d)", len(x_s), n)
    return {col: float(m) for col, m in zip(features.columns, mi, strict=True)}


def id_mi_nats(ids: pd.Series, target: pd.Series, n_bins: int = 256) -> float:
    """Estimate MI(id; class) by binning the id index, in nats.

    A monotone integer index carries no class signal; binning it and measuring
    MI confirms MI ~ 0. We bin rather than treat id as 577k-level categorical
    (which would spuriously inflate plug-in MI from finite-sample noise).

    Args:
        ids: The id column.
        target: The class target.
        n_bins: Number of equal-width bins over the id range.

    Returns:
        MI(binned id; class) in nats. Expected ~ 0.
    """
    binned = pd.cut(ids, bins=n_bins, labels=False, include_lowest=True)
    return categorical_mi_nats(binned.astype("category"), target)


def adversarial_validation_auc(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> float:
    """Run train-vs-test adversarial validation; return out-of-fold mean AUC.

    Builds a binary classifier to distinguish train (label 0) from test
    (label 1) using only the shared feature columns. If the classifier can tell
    them apart (AUC >> 0.5), the two splits differ in distribution == covariate
    shift. This is DOCUMENTED, not blocking: it informs C3 validation design,
    it is not leakage of the target.

    Args:
        train_df: Schema-validated train frame (with `class`, dropped here).
        test_df: Schema-validated test frame (no `class`).

    Returns:
        Mean out-of-fold ROC AUC over N_SPLITS stratified folds. ~0.5 == no
        shift, ~1.0 == fully separable.

    Notes:
        Categoricals are ordinal-encoded and numerics standardised inside a
        single design matrix. We use a linear model (LogisticRegression) on
        purpose: a strong nonlinear model could overfit and report spurious
        separability; a linear AUC is a conservative shift signal. id is
        excluded — it is an arbitrary index and would trivially separate the
        two id ranges, which is not a real distributional shift.
    """
    shared = [c for c in train_df.columns if c not in {TARGET, ID_COLUMN}]
    x_tr = train_df[shared].copy()
    x_te = test_df[shared].copy()
    x_tr["__is_test__"] = 0
    x_te["__is_test__"] = 1
    combined = pd.concat([x_tr, x_te], ignore_index=True)

    y = combined.pop("__is_test__").to_numpy()

    cat_cols = [c for c in _CATEGORICAL_FEATURES if c in combined.columns]
    num_cols = [c for c in combined.columns if c not in cat_cols]

    design_parts: list[np.ndarray] = []
    if num_cols:
        scaled = StandardScaler().fit_transform(combined[num_cols].to_numpy())
        design_parts.append(scaled)
    if cat_cols:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit_transform(combined[cat_cols])
        design_parts.append(enc)
    design = np.hstack(design_parts)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    proba = cross_val_predict(
        clf, design, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    auc = float(roc_auc_score(y, proba))
    logger.info("Adversarial validation OOF AUC = %.4f", auc)
    return auc


@dataclass(frozen=True)
class DuplicateReport:
    """Duplicate accounting across and within splits.

    Attributes:
        train_internal_dups: Number of train rows that are exact feature-row
            duplicates of an earlier train row (excludes id; the first
            occurrence of each group is NOT counted).
        train_internal_dup_groups: Number of distinct duplicated feature-rows.
        cross_split_dups: Number of DISTINCT feature-rows present in BOTH splits
            (cross-split leakage). This is a set-membership count: a feature-row
            that occurs k times in train and m times in test contributes 1, not
            k*m. 0 is the only safe value.
    """

    train_internal_dups: int
    train_internal_dup_groups: int
    cross_split_dups: int


def duplicate_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> DuplicateReport:
    """Count internal train duplicates and train/test cross-split overlaps.

    The row signature is ALL feature columns excluding `id` (an index, not a
    feature) and excluding `class` (present only in train). Two rows are
    "the same observation" iff every feature value matches.

    Args:
        train_df: Schema-validated train frame.
        test_df: Schema-validated test frame.

    Returns:
        DuplicateReport with the three counts.

    Notes:
        Cross-split detection is a SET-MEMBERSHIP test, not a join: both sides
        are deduplicated to their distinct feature-rows BEFORE the inner merge,
        so a row repeated in either split counts once. A raw inner merge would
        multiply multiplicities (k train copies x m test copies = k*m rows),
        over-reporting leakage on any split that contains internal duplicates.
        With continuous photometry to full float precision, a coincidental exact
        match across 9+ columns is astronomically unlikely, so any hit is a
        genuine shared observation == leakage.
    """
    feature_cols = [c for c in train_df.columns if c not in {ID_COLUMN, TARGET}]

    dup_mask = train_df.duplicated(subset=feature_cols, keep="first")
    train_internal_dups = int(dup_mask.sum())
    train_internal_dup_groups = int(
        train_df.loc[train_df.duplicated(subset=feature_cols, keep=False)]
        .groupby(feature_cols, observed=True)
        .ngroups
    ) if train_internal_dups > 0 else 0

    # Cross-split: count DISTINCT feature-rows shared by both splits. Dedup each
    # side first so this is a set-intersection size, not a multiplicity product.
    tr_keys = train_df[feature_cols].drop_duplicates()
    te_keys = test_df[feature_cols].drop_duplicates()
    cross_split_dups = int(len(tr_keys.merge(te_keys, on=feature_cols, how="inner")))

    logger.info(
        "Duplicates -> train internal: %d rows (%d groups); cross-split: %d",
        train_internal_dups,
        train_internal_dup_groups,
        cross_split_dups,
    )
    return DuplicateReport(
        train_internal_dups=train_internal_dups,
        train_internal_dup_groups=train_internal_dup_groups,
        cross_split_dups=cross_split_dups,
    )


def run_audit() -> dict[str, object]:
    """Execute the full BL-003 leakage audit and return a structured result.

    Returns:
        A dict with every measured quantity (H(class), per-feature MI and
        Cramer's V, cross-tabs, adversarial AUC, duplicate counts). The verdict
        table in docs/c2-data/leakage-audit.md is authored from these numbers.

    Side effects:
        Loads train/test via the schema-validated loader and emits log records.
    """
    logger.info("Loading schema-validated train/test splits")
    train_df = load_train()
    test_df = load_test()

    target = train_df[TARGET]
    h_nats = entropy_nats(target)
    h_bits = h_nats / np.log(2)
    logger.info("H(class) = %.6f nats = %.6f bits", h_nats, h_bits)

    # --- Numeric features: k-NN MI on a stratified sample. ---
    numeric_mi = numeric_mi_nats(train_df[list(_NUMERIC_FEATURES)], target)

    # --- Categorical features: EXACT MI + bias-corrected Cramer's V, full frame. ---
    cat_mi: dict[str, float] = {}
    cat_v: dict[str, float] = {}
    cat_crosstabs: dict[str, pd.DataFrame] = {}
    for col in _CATEGORICAL_FEATURES:
        cat_mi[col] = categorical_mi_nats(train_df[col], target)
        cat_v[col] = cramers_v_bias_corrected(train_df[col], target)
        cat_crosstabs[col] = pd.crosstab(train_df[col], target, normalize="index")

    # --- id sanity: MI ~ 0. ---
    id_mi = id_mi_nats(train_df[ID_COLUMN], target)

    # --- redshift physical ordering. ---
    redshift_by_class = (
        train_df.groupby(TARGET)["redshift"].agg(["median", "mean", "min", "max"])
    )

    # --- Adversarial validation. ---
    adv_auc = adversarial_validation_auc(train_df, test_df)

    # --- Duplicates. ---
    dups = duplicate_report(train_df, test_df)

    # Assemble per-feature MI summaries.
    summaries: list[FeatureMI] = []
    for col in _NUMERIC_FEATURES:
        mi = numeric_mi[col]
        summaries.append(FeatureMI(col, mi, mi / h_nats, None))
    for col in _CATEGORICAL_FEATURES:
        mi = cat_mi[col]
        summaries.append(FeatureMI(col, mi, mi / h_nats, cat_v[col]))
    summaries.append(FeatureMI(ID_COLUMN, id_mi, id_mi / h_nats, None))

    return {
        "h_nats": h_nats,
        "h_bits": h_bits,
        "summaries": summaries,
        "cat_crosstabs": cat_crosstabs,
        "redshift_by_class": redshift_by_class,
        "adversarial_auc": adv_auc,
        "duplicates": dups,
        "mi_sample_size": min(_MI_SAMPLE_SIZE, len(train_df)),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }


def _interpret_v(v: float | None) -> str:
    """Map a Cramer's V value to its effect-size band label.

    Args:
        v: Cramer's V in [0, 1], or None for non-categorical features.

    Returns:
        One of "n/a", "negligible", "small", "medium", "large".
    """
    if v is None:
        return "n/a"
    if v < _V_SMALL:
        return "negligible"
    if v < _V_MEDIUM:
        return "small"
    if v < _V_LARGE:
        return "medium"
    return "large"


def _log_report(result: dict[str, object]) -> None:
    """Emit the audit result as a structured, human-readable log block.

    Args:
        result: The dict returned by run_audit().

    Returns:
        None. Writes to the module logger at INFO level.
    """
    h_nats = float(result["h_nats"])
    logger.info("=" * 68)
    logger.info("BL-003 LEAKAGE AUDIT — S6E6 stellar classification")
    logger.info("=" * 68)
    logger.info(
        "rows: train=%d test=%d | MI sample=%d",
        result["n_train"],
        result["n_test"],
        result["mi_sample_size"],
    )
    logger.info("H(class) = %.6f nats = %.6f bits", h_nats, result["h_bits"])
    logger.info("-" * 68)
    logger.info("%-18s %10s %10s %10s %12s", "feature", "MI(nats)", "MI/H", "CramersV", "V-band")
    for s in result["summaries"]:  # type: ignore[union-attr]
        v_str = f"{s.cramers_v:.4f}" if s.cramers_v is not None else "—"
        logger.info(
            "%-18s %10.4f %10.4f %10s %12s",
            s.feature,
            s.mi_nats,
            s.mi_fraction_of_h,
            v_str,
            _interpret_v(s.cramers_v),
        )
    logger.info("-" * 68)
    for col, ct in result["cat_crosstabs"].items():  # type: ignore[union-attr]
        logger.info("P(class | %s):\n%s", col, (ct * 100).round(2).to_string())
    logger.info("-" * 68)
    logger.info(
        "redshift by class:\n%s",
        result["redshift_by_class"].round(4).to_string(),  # type: ignore[union-attr]
    )
    logger.info("adversarial validation OOF AUC = %.4f", result["adversarial_auc"])
    dups = result["duplicates"]
    logger.info(
        "duplicates -> train internal: %d (%d groups) | cross-split: %d",
        dups.train_internal_dups,  # type: ignore[union-attr]
        dups.train_internal_dup_groups,  # type: ignore[union-attr]
        dups.cross_split_dups,  # type: ignore[union-attr]
    )
    logger.info("=" * 68)


def main() -> None:
    """CLI entry point: configure logging, run the audit, print the report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = run_audit()
    _log_report(result)


if __name__ == "__main__":
    main()
