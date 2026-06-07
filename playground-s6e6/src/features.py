"""Row-wise feature engineering for Kaggle Playground S6E6 (stellar classification).

Cycle C3 (Feature & Hypothesis) deliverable BL-006. Turns the RAW SDSS features
(photometric magnitudes u,g,r,i,z + redshift) into engineered features that
expose the discriminative structure a gradient-boosted tree approximates only
inefficiently from the raw inputs.

LEAKAGE INVARIANT (enforced by construction, audited in tests)
--------------------------------------------------------------
Every function here is ROW-WISE PURE: the output for row i is a function of the
RAW VALUES OF ROW i ONLY, plus compile-time constants passed as keyword
arguments. No function:
  - reads the target `class`,
  - aggregates across rows (no mean/min/max/quantile fitted on the data),
  - depends on the train/val/test split.
Because of this, a feature computed here CANNOT carry information from other
rows (no train->val leakage) and is split-invariant: computing it before or
after the CV split yields byte-identical columns. This is WHY no `fit` step
exists and WHY the redshift floor is a PARAMETER (a known physical constant),
never `df['redshift'].min()` (which would be a fitted, leaking statistic).

The colours u-g, g-r, r-i, i-z are differences of ADJACENT bands in wavelength
order (u<g<r<i<z). Physically they encode the spectral slope of an object; a
single tree split on a raw magnitude cannot reproduce a linear combination of
two magnitudes without many axis-aligned splits, so handing the tree the colour
directly is the expected source of gain (to be confirmed by ablation, not
assumed — see the redshift-log control which is expected to move the metric by
~0 under a tree's monotonic-transform invariance).

Determinism / purity contract for every public function:
  - inputs:  a pandas DataFrame containing the required raw columns.
  - output:  a NEW DataFrame (the input is never mutated) with the requested
             engineered columns appended; original columns are preserved.
  - side effects: NONE.
  - idempotent: calling twice with the same `include` yields the same frame
             (re-adding an existing column overwrites it with the identical
             value, so repeated application is a fixpoint).
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Feature catalogue.
# --------------------------------------------------------------------------- #
# Adjacent-band colour indices in SDSS wavelength order (u < g < r < i < z).
# Each colour is the magnitude DIFFERENCE of two neighbouring bands. We pin the
# ordering and the exact (left, right) band pair per colour so the arithmetic is
# unambiguous and testable: colour = mag_left - mag_right.
COLOR_DEFINITIONS: Final[dict[str, tuple[str, str]]] = {
    "u_g": ("u", "g"),
    "g_r": ("g", "r"),
    "r_i": ("r", "i"),
    "i_z": ("i", "z"),
}

# Name of the redshift log-transform column (kept as a module constant so the
# eval harness and the tests reference the same string, never a typo'd literal).
REDSHIFT_LOG_COL: Final[str] = "redshift_log"

# Physical floor of the redshift contract (see data_loader: redshift min_value is
# -0.01; measurement noise near zero produces small negatives). Used as the
# DEFAULT shift so log1p is defined on the whole observed support WITHOUT reading
# the data. Passing a constant (not df.min()) is what keeps the transform
# row-wise pure and leakage-free.
REDSHIFT_FLOOR: Final[float] = -0.01

# The full set of feature names this module can add, for callers that want to
# request "everything" or validate an `include` request.
ALL_ENGINEERED: Final[tuple[str, ...]] = tuple(COLOR_DEFINITIONS) + (REDSHIFT_LOG_COL,)


def add_colors(df: pd.DataFrame, *, colors: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Append SDSS adjacent-band colour indices to a copy of `df`.

    A colour is the difference of two adjacent photometric magnitudes, e.g.
    ``u_g = u - g``. Differencing removes the shared "overall brightness"
    component (a near-constant offset across all five bands) and isolates the
    spectral slope, which is the class-discriminative part. See COLOR_DEFINITIONS
    for the exact band pairs.

    Args:
        df: Frame containing the raw magnitude columns u, g, r, i, z.
        colors: Subset of COLOR_DEFINITIONS keys to add. None (default) adds all
            four. An empty tuple adds nothing (returns a copy unchanged).

    Returns:
        A NEW DataFrame: all original columns plus the requested colour columns.
        The input frame is not mutated.

    Raises:
        KeyError: If a required magnitude column is missing, or if `colors`
            contains a name not in COLOR_DEFINITIONS.

    Edge cases:
        - Negative or zero magnitudes are fine: a colour is a plain subtraction,
          defined for any real inputs (no log, no division).
        - Calling twice is idempotent: the second call overwrites each colour
          column with the identical value.
    """
    requested = tuple(COLOR_DEFINITIONS) if colors is None else colors
    unknown = set(requested) - set(COLOR_DEFINITIONS)
    if unknown:
        raise KeyError(
            f"unknown colour(s) {sorted(unknown)}; valid: {sorted(COLOR_DEFINITIONS)}"
        )

    out = df.copy()
    for name in requested:
        left, right = COLOR_DEFINITIONS[name]
        # Plain difference: mag_left - mag_right. Both columns are required; a
        # missing band raises KeyError loudly rather than producing NaN silently.
        out[name] = out[left] - out[right]
    return out


def add_redshift_log(
    df: pd.DataFrame,
    *,
    floor: float = REDSHIFT_FLOOR,
    column: str = "redshift",
) -> pd.DataFrame:
    """Append ``log1p(redshift - floor)`` to a copy of `df`.

    This is the CONTROL feature for the monotonic-invariance hypothesis. A
    decision tree is invariant to any strictly-monotonic transform of a SINGLE
    variable (it splits on order statistics, which log1p preserves), so adding
    this column ALONE is expected to move balanced_accuracy by ~0. We include it
    precisely to CONFIRM that empirically, not to gain points.

    The shift by `floor` makes the argument of log1p strictly > 0 on the whole
    observed redshift support (min ~ -0.00997), so the transform is finite
    everywhere. `floor` is a PARAMETER defaulting to the contract's physical floor
    (REDSHIFT_FLOOR = -0.01); it is NEVER computed from the data, which is what
    keeps this function row-wise pure and leakage-free.

    Args:
        df: Frame containing the redshift column.
        floor: Constant subtracted before log1p. Must be strictly below the
            smallest redshift value so the log argument stays positive. Defaults
            to the contract floor -0.01.
        column: Name of the redshift source column (override only for testing).

    Returns:
        A NEW DataFrame with the REDSHIFT_LOG_COL column appended. Input not
        mutated.

    Raises:
        KeyError: If the redshift column is absent.
        ValueError: If `floor` is not strictly below the minimum observed value
            in `column` (would make log1p receive a non-positive argument). This
            check inspects only `column` to validate the parameter; it does not
            DERIVE the floor from the data, so purity is preserved.

    Edge cases:
        - redshift exactly at `floor` would give log1p(0) = 0, which is fine; the
          guard only rejects values strictly BELOW floor (negative argument).
    """
    if column not in df.columns:
        raise KeyError(f"redshift column {column!r} not found in frame")

    shifted = df[column].to_numpy(dtype=np.float64) - floor
    # Validate the PARAMETER against the data without deriving it from the data:
    # if any shifted value is < 0 the chosen floor is wrong for this input.
    if np.any(shifted < 0.0):
        raise ValueError(
            f"floor={floor} is not below min({column})="
            f"{df[column].min()}; log1p would receive a negative argument"
        )

    out = df.copy()
    out[REDSHIFT_LOG_COL] = np.log1p(shifted)
    return out


def add_features(
    df: pd.DataFrame,
    *,
    include: tuple[str, ...] = ALL_ENGINEERED,
    redshift_floor: float = REDSHIFT_FLOOR,
) -> pd.DataFrame:
    """Composable entry point: append the requested engineered features.

    Dispatches to add_colors / add_redshift_log based on `include`. This is the
    function the eval harness calls; the per-feature functions remain public so
    tests can exercise each in isolation.

    Args:
        df: Frame with the raw u,g,r,i,z and redshift columns.
        include: Names of engineered features to add. Any subset of
            ALL_ENGINEERED (the four colours + REDSHIFT_LOG_COL). Order is
            irrelevant; duplicates are harmless (idempotent). Empty tuple returns
            a plain copy.
        redshift_floor: Forwarded to add_redshift_log when REDSHIFT_LOG_COL is in
            `include`.

    Returns:
        A NEW DataFrame with all original columns plus the requested engineered
        columns. Input not mutated.

    Raises:
        KeyError: If `include` names an unknown feature, or a required raw column
            is missing.
        ValueError: Propagated from add_redshift_log on a bad floor.
    """
    unknown = set(include) - set(ALL_ENGINEERED)
    if unknown:
        raise KeyError(
            f"unknown feature(s) {sorted(unknown)}; valid: {sorted(ALL_ENGINEERED)}"
        )

    out = df.copy()

    colour_names = tuple(name for name in include if name in COLOR_DEFINITIONS)
    if colour_names:
        out = add_colors(out, colors=colour_names)

    if REDSHIFT_LOG_COL in include:
        out = add_redshift_log(out, floor=redshift_floor)

    return out
