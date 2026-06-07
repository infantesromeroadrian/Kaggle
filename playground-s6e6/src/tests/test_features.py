"""Tests for the S6E6 row-wise feature engineering module (BL-006).

Strategy:
  - Pure synthetic frames (no real data needed) so the suite runs in any
    checkout, including a worktree without data/. Values are chosen to make the
    expected arithmetic obvious and exact.
  - One behaviour per test (testing skill: small, focused). Colour arithmetic is
    parametrized over every COLOR_DEFINITIONS entry so a new colour is covered
    automatically.
  - We assert the module's CONTRACT: exact arithmetic, non-mutation of the input,
    idempotence, output shape, the row-wise/leakage invariant (split-invariance),
    negative-redshift handling, and every documented error path. Target: 100%
    coverage of src.features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    ALL_ENGINEERED,
    COLOR_DEFINITIONS,
    REDSHIFT_FLOOR,
    REDSHIFT_LOG_COL,
    add_colors,
    add_features,
    add_redshift_log,
)


def _frame() -> pd.DataFrame:
    """Build a small synthetic frame with the raw columns the module needs.

    The magnitudes are spread so every adjacent-band difference is a distinct,
    easily-checked integer-ish value; redshift includes a small NEGATIVE value
    (mirroring the real -0.00997 noise floor) plus a zero and positive values so
    the log transform exercises its full domain.

    Returns:
        A 4-row DataFrame with u, g, r, i, z, redshift and a decoy `class` column
        (present to prove the feature functions ignore the target entirely).
    """
    return pd.DataFrame(
        {
            "u": np.array([20.0, 18.5, 25.0, 19.0]),
            "g": np.array([19.0, 17.0, 24.0, 18.0]),
            "r": np.array([18.0, 16.5, 23.0, 17.5]),
            "i": np.array([17.5, 16.0, 22.5, 17.0]),
            "z": np.array([17.0, 15.5, 22.0, 16.5]),
            "redshift": np.array([-0.005, 0.0, 1.5, 0.3]),
            "class": ["GALAXY", "STAR", "QSO", "GALAXY"],
        }
    )


# --------------------------------------------------------------------------- #
# Colour arithmetic.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,pair", list(COLOR_DEFINITIONS.items()))
def test_color_arithmetic_is_exact(name: str, pair: tuple[str, str]) -> None:
    """Each colour equals the exact difference of its two source magnitudes."""
    df = _frame()
    left, right = pair
    out = add_colors(df, colors=(name,))
    expected = df[left] - df[right]
    pd.testing.assert_series_equal(out[name], expected, check_names=False)


def test_add_colors_default_adds_all_four() -> None:
    """With colors=None every colour in the catalogue is produced."""
    out = add_colors(_frame())
    for name in COLOR_DEFINITIONS:
        assert name in out.columns


def test_add_colors_empty_tuple_is_noop_copy() -> None:
    """An empty colour request returns an unchanged COPY (no new columns)."""
    df = _frame()
    out = add_colors(df, colors=())
    pd.testing.assert_frame_equal(out, df)
    assert out is not df


def test_add_colors_unknown_name_raises_keyerror() -> None:
    """Requesting a colour not in the catalogue fails loudly."""
    with pytest.raises(KeyError, match="unknown colour"):
        add_colors(_frame(), colors=("u_z",))


def test_add_colors_missing_band_raises_keyerror() -> None:
    """A missing source magnitude column raises (no silent NaN)."""
    df = _frame().drop(columns=["g"])
    with pytest.raises(KeyError):
        add_colors(df, colors=("u_g",))


def test_add_colors_does_not_mutate_input() -> None:
    """add_colors never mutates the caller's frame."""
    df = _frame()
    before = df.copy(deep=True)
    add_colors(df)
    pd.testing.assert_frame_equal(df, before)


def test_add_colors_is_idempotent() -> None:
    """Applying add_colors twice yields an identical frame (fixpoint)."""
    df = _frame()
    once = add_colors(df)
    twice = add_colors(once)
    pd.testing.assert_frame_equal(once, twice)


# --------------------------------------------------------------------------- #
# Redshift log transform.
# --------------------------------------------------------------------------- #
def test_redshift_log_matches_log1p_shifted() -> None:
    """redshift_log == log1p(redshift - floor) elementwise."""
    df = _frame()
    out = add_redshift_log(df)
    expected = np.log1p(df["redshift"].to_numpy() - REDSHIFT_FLOOR)
    np.testing.assert_allclose(out[REDSHIFT_LOG_COL].to_numpy(), expected)


def test_redshift_log_handles_negative_redshift() -> None:
    """A small negative redshift (noise floor) yields a finite, real log value."""
    df = _frame()  # contains redshift = -0.005, below 0 but above the -0.01 floor
    out = add_redshift_log(df)
    assert np.isfinite(out[REDSHIFT_LOG_COL].to_numpy()).all()


def test_redshift_log_value_at_floor_is_zero() -> None:
    """redshift exactly at the floor maps to log1p(0) = 0 (boundary is valid)."""
    df = pd.DataFrame({"redshift": np.array([REDSHIFT_FLOOR])})
    out = add_redshift_log(df)
    assert out[REDSHIFT_LOG_COL].iloc[0] == pytest.approx(0.0)


def test_redshift_log_floor_too_high_raises_valueerror() -> None:
    """A floor not strictly below the min redshift is rejected (negative log arg)."""
    df = _frame()  # min redshift is -0.005
    with pytest.raises(ValueError, match="negative argument"):
        add_redshift_log(df, floor=0.0)


def test_redshift_log_missing_column_raises_keyerror() -> None:
    """Absent redshift column raises rather than silently no-op."""
    df = _frame().drop(columns=["redshift"])
    with pytest.raises(KeyError, match="redshift"):
        add_redshift_log(df)


def test_redshift_log_custom_column_name() -> None:
    """The source column name is overridable (used for isolated testing)."""
    df = pd.DataFrame({"z_spec": np.array([0.0, 0.5])})
    out = add_redshift_log(df, column="z_spec")
    expected = np.log1p(np.array([0.0, 0.5]) - REDSHIFT_FLOOR)
    np.testing.assert_allclose(out[REDSHIFT_LOG_COL].to_numpy(), expected)


def test_redshift_log_does_not_mutate_input() -> None:
    """add_redshift_log never mutates the caller's frame."""
    df = _frame()
    before = df.copy(deep=True)
    add_redshift_log(df)
    pd.testing.assert_frame_equal(df, before)


def test_redshift_log_is_idempotent() -> None:
    """Applying add_redshift_log twice yields an identical frame."""
    df = _frame()
    once = add_redshift_log(df)
    twice = add_redshift_log(once)
    pd.testing.assert_frame_equal(once, twice)


# --------------------------------------------------------------------------- #
# Composable entry point: add_features.
# --------------------------------------------------------------------------- #
def test_add_features_default_adds_everything() -> None:
    """The default include adds all four colours and the redshift log column."""
    out = add_features(_frame())
    for name in ALL_ENGINEERED:
        assert name in out.columns


def test_add_features_subset_colours_only() -> None:
    """Requesting only colours adds no redshift_log column."""
    out = add_features(_frame(), include=("u_g", "g_r"))
    assert "u_g" in out.columns and "g_r" in out.columns
    assert REDSHIFT_LOG_COL not in out.columns
    assert "r_i" not in out.columns


def test_add_features_redshift_log_only() -> None:
    """Requesting only the redshift log adds no colour columns."""
    out = add_features(_frame(), include=(REDSHIFT_LOG_COL,))
    assert REDSHIFT_LOG_COL in out.columns
    assert not any(c in out.columns for c in COLOR_DEFINITIONS)


def test_add_features_empty_include_is_copy() -> None:
    """An empty include returns an unchanged copy."""
    df = _frame()
    out = add_features(df, include=())
    pd.testing.assert_frame_equal(out, df)
    assert out is not df


def test_add_features_unknown_feature_raises_keyerror() -> None:
    """An unknown engineered-feature name is rejected."""
    with pytest.raises(KeyError, match="unknown feature"):
        add_features(_frame(), include=("not_a_feature",))


def test_add_features_forwards_redshift_floor() -> None:
    """A bad redshift_floor propagates the ValueError from add_redshift_log."""
    with pytest.raises(ValueError, match="negative argument"):
        add_features(_frame(), include=(REDSHIFT_LOG_COL,), redshift_floor=0.0)


def test_add_features_does_not_mutate_input() -> None:
    """add_features never mutates the caller's frame."""
    df = _frame()
    before = df.copy(deep=True)
    add_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_add_features_is_idempotent() -> None:
    """Applying add_features twice yields an identical frame (fixpoint)."""
    df = _frame()
    once = add_features(df)
    twice = add_features(once)
    pd.testing.assert_frame_equal(once, twice)


def test_add_features_shape_is_original_plus_engineered() -> None:
    """Output has exactly the original columns plus the engineered ones."""
    df = _frame()
    out = add_features(df)
    assert out.shape[0] == df.shape[0]
    assert out.shape[1] == df.shape[1] + len(ALL_ENGINEERED)
    # Original columns are preserved unchanged.
    pd.testing.assert_frame_equal(out[df.columns], df)


# --------------------------------------------------------------------------- #
# Leakage invariant: row-wise purity == split-invariance.
# --------------------------------------------------------------------------- #
def test_features_are_split_invariant() -> None:
    """Engineering before vs after a row split yields identical columns.

    This is the operational test of the row-wise/leakage invariant: because each
    feature depends only on its own row (plus constants), splitting the frame and
    engineering each part must equal engineering the whole and then splitting. If
    any feature aggregated across rows, this equality would break.
    """
    df = _frame()
    whole = add_features(df)

    top, bottom = df.iloc[:2].copy(), df.iloc[2:].copy()
    recombined = pd.concat([add_features(top), add_features(bottom)])

    pd.testing.assert_frame_equal(whole, recombined)


def test_features_ignore_target_column() -> None:
    """Changing the target leaves every engineered feature unchanged.

    The functions must never read `class`; permuting it must not perturb a single
    engineered value. This guards the 'no target leakage' half of the invariant.
    """
    df = _frame()
    base = add_features(df)

    perturbed = df.copy()
    perturbed["class"] = perturbed["class"].iloc[::-1].to_numpy()
    after = add_features(perturbed)

    engineered_cols = list(ALL_ENGINEERED)
    pd.testing.assert_frame_equal(base[engineered_cols], after[engineered_cols])
