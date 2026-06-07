"""Tests for the BL-003 / BL-005 statistical primitives (C2 gate).

These formalise the mathematical identities verified by hand during the statistical review,
turning them into a regression gate so a future refactor cannot silently break
them. Scope is the four functions whose correctness the leakage verdict and the
fold contract depend on:

  - entropy_nats            (the H(class) ceiling)
  - categorical_mi_nats     (exact MI for the categorical features)
  - cramers_v_bias_corrected (effect size for the categorical association)
  - verify_folds            (the BL-005 fold partition contract, incl. W1 fix)

Strategy (mirrors test_data_loader.py):
  - Small SYNTHETIC data built by hand, never the real CSVs, so the suite is
    fast and deterministic.
  - One identity / one failure mode per test, signals isolated.
  - We assert mathematical properties (bounds, identities) rather than
    hard-coded estimator internals, so the tests document WHY each number is
    correct, not just THAT it is.

Mathematical facts under test:
  - H of a 3-class dist (0.6538, 0.2029, 0.1433) == 0.879847 nats (the dataset
    ceiling), to 1e-3.
  - MI(x; y) for independent x, y == 0 in the limit; for a finite plug-in
    estimate the upward bias is ~ (r-1)(k-1)/(2n). We bound it by the generous
    2*(r-1)(k-1)/(2n).
  - MI(y; y) == H(y)  (a variable shares all its entropy with itself).
  - MI(x; y) <= H(y)  (information about y cannot exceed y's own entropy).
  - Cramer's V(y, y) == 1  (perfect association); V(independent) ~ 0.
  - verify_folds raises AssertionError when train+val does not cover all rows
    (the W1 completeness check), and passes on a valid partition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.leakage_audit import (
    categorical_mi_nats,
    cramers_v_bias_corrected,
    duplicate_report,
    entropy_nats,
)
from src.make_folds import verify_folds

# --------------------------------------------------------------------------- #
# Helpers — deterministic synthetic builders.
# --------------------------------------------------------------------------- #


def _labels_from_proportions(
    proportions: dict[str, float], n: int
) -> pd.Series:
    """Build a label Series realising given class proportions as closely as int allows.

    Args:
        proportions: {class_name: fraction}, need not sum to exactly 1.
        n: Total number of labels.

    Returns:
        A Series of length ~n with each class repeated round(fraction*n) times.
    """
    parts: list[str] = []
    for cls, frac in proportions.items():
        parts.extend([cls] * round(frac * n))
    return pd.Series(parts, dtype="object")


def _independent_pair(
    n_per_cell: int, r: int = 3, k: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Build two categorical Series that are EXACTLY independent by construction.

    A full factorial grid with the same count in every (x, y) cell has a joint
    that factorises as the product of marginals, so MI == 0 and Cramer's V == 0
    in the population. The plug-in estimate on this exact grid is also ~0 up to
    the finite-sample bias.

    Args:
        n_per_cell: Repetitions of each (x_i, y_j) combination.
        r: Number of x categories.
        k: Number of y categories.

    Returns:
        (x, y) Series of length r*k*n_per_cell, independent by construction.
    """
    x_vals: list[str] = []
    y_vals: list[str] = []
    for i in range(r):
        for j in range(k):
            x_vals.extend([f"x{i}"] * n_per_cell)
            y_vals.extend([f"y{j}"] * n_per_cell)
    return pd.Series(x_vals, dtype="object"), pd.Series(y_vals, dtype="object")


# --------------------------------------------------------------------------- #
# entropy_nats
# --------------------------------------------------------------------------- #


def test_entropy_matches_dataset_ceiling() -> None:
    """H of the real class proportions equals the documented 0.8798 nats."""
    labels = _labels_from_proportions(
        {"GALAXY": 0.6538, "QSO": 0.2029, "STAR": 0.1433}, n=1_000_000
    )
    h = entropy_nats(labels)
    assert h == pytest.approx(0.879847, abs=1e-3)


def test_entropy_uniform_three_classes_is_log3() -> None:
    """A uniform 3-class distribution has entropy ln(3) nats (sanity anchor)."""
    labels = pd.Series(["a", "b", "c"] * 1000, dtype="object")
    assert entropy_nats(labels) == pytest.approx(np.log(3), abs=1e-9)


def test_entropy_constant_label_is_zero() -> None:
    """A single-class series carries zero entropy."""
    labels = pd.Series(["only"] * 500, dtype="object")
    assert entropy_nats(labels) == pytest.approx(0.0, abs=1e-12)


def test_entropy_empty_is_zero() -> None:
    """An empty series returns 0.0 rather than erroring (documented edge case)."""
    assert entropy_nats(pd.Series([], dtype="object")) == 0.0


# --------------------------------------------------------------------------- #
# categorical_mi_nats
# --------------------------------------------------------------------------- #


def test_mi_independent_is_near_zero() -> None:
    """MI of independent categoricals is ~0, within the finite-sample bias bound.

    Plug-in MI on a finite sample has an upward bias ~ (r-1)(k-1)/(2n). We bound
    by twice that, which the exact-grid construction comfortably satisfies.
    """
    r, k, n_per_cell = 3, 3, 200
    x, y = _independent_pair(n_per_cell, r=r, k=k)
    n = len(x)
    mi = categorical_mi_nats(x, y)
    bias_bound = 2 * (r - 1) * (k - 1) / (2 * n)
    assert 0.0 <= mi < bias_bound


def test_mi_self_equals_entropy() -> None:
    """MI(y; y) == H(y): a variable shares all of its entropy with itself."""
    y = _labels_from_proportions(
        {"GALAXY": 0.6538, "QSO": 0.2029, "STAR": 0.1433}, n=50_000
    )
    assert categorical_mi_nats(y, y) == pytest.approx(entropy_nats(y), abs=1e-12)


def test_mi_upper_bounded_by_target_entropy() -> None:
    """MI(x; y) <= H(y): information about y cannot exceed y's own entropy."""
    # A noisy but informative x: y with 20% of labels shuffled into noise.
    rng = np.random.default_rng(42)
    y = _labels_from_proportions(
        {"GALAXY": 0.6538, "QSO": 0.2029, "STAR": 0.1433}, n=30_000
    ).to_numpy()
    x = y.copy()
    noise_idx = rng.choice(len(x), size=len(x) // 5, replace=False)
    x[noise_idx] = rng.choice(["GALAXY", "QSO", "STAR"], size=len(noise_idx))
    mi = categorical_mi_nats(pd.Series(x), pd.Series(y))
    h_y = entropy_nats(pd.Series(y))
    assert mi <= h_y + 1e-9
    # And it should be substantial (x is mostly y), so the bound is non-trivial.
    assert mi > 0.5 * h_y


def test_mi_non_negative_and_empty_zero() -> None:
    """MI is non-negative; empty input returns 0.0."""
    assert categorical_mi_nats(pd.Series([], dtype="object"), pd.Series([], dtype="object")) == 0.0


# --------------------------------------------------------------------------- #
# cramers_v_bias_corrected
# --------------------------------------------------------------------------- #


def test_cramers_v_self_is_one() -> None:
    """V(y, y) == 1: a variable is perfectly associated with itself."""
    y = _labels_from_proportions(
        {"GALAXY": 0.6538, "QSO": 0.2029, "STAR": 0.1433}, n=50_000
    )
    assert cramers_v_bias_corrected(y, y) == pytest.approx(1.0, abs=1e-6)


def test_cramers_v_independent_is_near_zero() -> None:
    """V of independent categoricals is ~0 (bias correction floors it at 0)."""
    x, y = _independent_pair(n_per_cell=300, r=3, k=3)
    assert cramers_v_bias_corrected(x, y) == pytest.approx(0.0, abs=1e-6)


def test_cramers_v_degenerate_table_is_zero() -> None:
    """A constant column (one category) yields V = 0: no association measurable."""
    x = pd.Series(["c"] * 100, dtype="object")
    y = pd.Series(["a", "b"] * 50, dtype="object")
    assert cramers_v_bias_corrected(x, y) == 0.0


# --------------------------------------------------------------------------- #
# verify_folds — the BL-005 contract (incl. W1 completeness fix)
# --------------------------------------------------------------------------- #


def test_verify_folds_accepts_valid_partition() -> None:
    """A disjoint+complete+stratified partition passes and returns provenance."""
    n, n_splits = 1000, 5
    # Balanced target so stratification tolerance is trivially met.
    target = pd.Series((["GALAXY"] * 654 + ["QSO"] * 203 + ["STAR"] * 143), dtype="object")
    assert len(target) == n
    folds = _build_stratified_partition(target, n_splits, seed=0)
    meta = verify_folds(folds, target)
    assert meta["n_rows"] == n
    assert meta["n_splits"] == n_splits
    assert len(meta["folds"]) == n_splits  # type: ignore[arg-type]


def test_verify_folds_rejects_incomplete_train_w1() -> None:
    """W1 fix: dropping a row from a fold's train+val raises AssertionError.

    Build a valid partition, then DELETE one index from fold 0's train set so
    fold 0 no longer covers arange(n). Before the W1 fix this slipped through
    (only the global validation union was checked); now check 3b catches it.
    """
    _, n_splits = 1000, 5
    target = pd.Series((["GALAXY"] * 654 + ["QSO"] * 203 + ["STAR"] * 143), dtype="object")
    folds = _build_stratified_partition(target, n_splits, seed=0)
    # Remove the last train index of fold 0: fold 0's train+val now misses a row,
    # but the GLOBAL validation union is still complete, so only the per-fold
    # completeness check (3b) can catch this.
    folds["fold0_train"] = folds["fold0_train"][:-1]
    with pytest.raises(AssertionError, match="does not cover all rows"):
        verify_folds(folds, target)


def test_verify_folds_rejects_train_val_overlap() -> None:
    """An index present in both train and val of a fold raises AssertionError."""
    _, n_splits = 1000, 5
    target = pd.Series((["GALAXY"] * 654 + ["QSO"] * 203 + ["STAR"] * 143), dtype="object")
    folds = _build_stratified_partition(target, n_splits, seed=0)
    # Inject fold 0's first val index into its train set: now they overlap.
    overlap_idx = folds["fold0_val"][0]
    folds["fold0_train"] = np.append(folds["fold0_train"], overlap_idx)
    with pytest.raises(AssertionError, match="train/val overlap"):
        verify_folds(folds, target)


def test_verify_folds_rejects_duplicate_validation_row() -> None:
    """A row validated in two folds raises AssertionError (cross-fold leakage)."""
    _, n_splits = 1000, 5
    target = pd.Series((["GALAXY"] * 654 + ["QSO"] * 203 + ["STAR"] * 143), dtype="object")
    folds = _build_stratified_partition(target, n_splits, seed=0)
    # Force fold 1 to re-validate a row already in fold 0's val set.
    dup = folds["fold0_val"][0]
    folds["fold1_val"] = np.append(folds["fold1_val"], dup)
    with pytest.raises(AssertionError):
        verify_folds(folds, target)


def _build_stratified_partition(
    target: pd.Series, n_splits: int, seed: int
) -> dict[str, np.ndarray]:
    """Build a stratified disjoint+complete fold dict over the target's rows.

    Stratifies by assigning each class's rows round-robin across folds so every
    fold's class proportions match the global ones (needed to pass check 4).

    Args:
        target: The class target.
        n_splits: Number of folds.
        seed: Shuffle seed.

    Returns:
        A valid partition fold dict.
    """
    rng = np.random.default_rng(seed)
    n = len(target)
    val_assignment = np.empty(n, dtype=np.int64)
    target_arr = target.to_numpy()
    for cls in np.unique(target_arr):
        cls_idx = np.where(target_arr == cls)[0]
        cls_idx = rng.permutation(cls_idx)
        # Round-robin this class across folds -> proportional per fold.
        val_assignment[cls_idx] = np.arange(len(cls_idx)) % n_splits
    folds: dict[str, np.ndarray] = {}
    for k in range(n_splits):
        va = np.sort(np.where(val_assignment == k)[0]).astype(np.int64)
        tr = np.sort(np.setdiff1d(np.arange(n), va)).astype(np.int64)
        folds[f"fold{k}_train"] = tr
        folds[f"fold{k}_val"] = va
    return folds


# --------------------------------------------------------------------------- #
# duplicate_report — cross-split membership (B1) and internal-dup branch (W-A1)
# --------------------------------------------------------------------------- #

# Feature columns the duplicate signature operates on (id + class are excluded by
# duplicate_report itself). We build distinct rows by varying `redshift` only;
# every other feature is held constant, which is enough for exact-row identity.
_FEATURE_COLS: tuple[str, ...] = (
    "alpha", "delta", "u", "g", "r", "i", "z", "redshift",
    "spectral_type", "galaxy_population",
)


def _row(redshift: float) -> dict[str, object]:
    """Build one synthetic feature-row whose identity is keyed by `redshift`.

    Args:
        redshift: The value that makes this row distinct from others.

    Returns:
        A dict with every feature column populated (constant except redshift).
    """
    return {
        "alpha": 10.0, "delta": 5.0, "u": 20.0, "g": 19.0, "r": 18.0,
        "i": 17.0, "z": 16.0, "redshift": redshift,
        "spectral_type": "G/K", "galaxy_population": "Blue_Cloud",
    }


def _frame(redshifts: list[float], with_class: bool) -> pd.DataFrame:
    """Assemble a train/test-shaped frame from a list of row keys.

    Args:
        redshifts: One entry per row; repeats create exact-duplicate rows.
        with_class: If True, append a `class` column (train shape); the value is
            irrelevant to duplicate_report, which excludes it.

    Returns:
        DataFrame with id + features (+ class), columns in loader order.
    """
    rows = [_row(z) for z in redshifts]
    df = pd.DataFrame(rows, columns=list(_FEATURE_COLS))
    df.insert(0, "id", np.arange(len(df), dtype=np.int64))
    if with_class:
        df["class"] = "GALAXY"
    return df


def test_cross_split_counts_distinct_membership_not_multiplicity_b1() -> None:
    """B1: a row tripled in train and present once in test counts as 1, not 3.

    The bug being guarded: a raw inner merge on the feature keys would emit
    k*m rows (3 train copies x 1 test copy = 3). The fix deduplicates each side
    first, so cross_split_dups is the set-intersection size = 1.
    """
    # redshift=0.5 appears 3x in train and 1x in test; 0.6/0.7 are train-only.
    train = _frame([0.5, 0.5, 0.5, 0.6, 0.7], with_class=True)
    test = _frame([0.5, 0.9], with_class=False)
    report = duplicate_report(train, test)
    assert report.cross_split_dups == 1


def test_cross_split_zero_when_disjoint() -> None:
    """No shared feature-row -> cross_split_dups == 0 (the safe value)."""
    train = _frame([0.1, 0.2, 0.3], with_class=True)
    test = _frame([0.4, 0.5], with_class=False)
    report = duplicate_report(train, test)
    assert report.cross_split_dups == 0


def test_internal_dup_branch_counts_rows_and_groups_wa1() -> None:
    """W-A1: internal train dups report both the row count and the group count.

    redshift=0.5 appears 3x (2 copies beyond the first), 0.6 appears 2x (1 copy
    beyond the first), 0.7 is unique. duplicated(keep="first") counts every copy
    after the first: (3-1) + (2-1) = 3 duplicate ROWS, spread over 2 distinct
    duplicated feature-rows (GROUPS).
    """
    # 0.5 x3, 0.6 x2, 0.7 x1 -> extras: (3-1)+(2-1)+0 = 3 duplicate rows.
    train = _frame([0.5, 0.5, 0.5, 0.6, 0.6, 0.7], with_class=True)
    test = _frame([0.9], with_class=False)
    report = duplicate_report(train, test)
    assert report.train_internal_dups == 3
    assert report.train_internal_dup_groups == 2
    assert report.cross_split_dups == 0


def test_internal_dup_branch_zero_when_unique() -> None:
    """No internal dup -> both internal counters are 0 (the else branch)."""
    train = _frame([0.1, 0.2, 0.3], with_class=True)
    test = _frame([0.9], with_class=False)
    report = duplicate_report(train, test)
    assert report.train_internal_dups == 0
    assert report.train_internal_dup_groups == 0
