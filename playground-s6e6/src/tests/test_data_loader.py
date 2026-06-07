"""Tests for the S6E6 data loader and schema validator (BL-002).

Strategy:
  - Unit tests build a tiny SYNTHETIC valid frame and mutate one thing at a
    time, asserting the validator raises SchemaError. One failure mode per
    test keeps signals isolated (testing skill: small, focused tests).
  - The synthetic frame is the single source of truth for "valid"; every
    negative test derives from it, so a contract change only needs the fixture
    updated.
  - Integration tests load the REAL competition CSVs when present. They are
    skipped (not failed) when the data directory is absent, so the unit suite
    stays runnable in a worktree that has no data/ checkout.

We test the public behaviour (validate_schema raising / loaders returning
correct shapes), not private implementation details.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src import data_loader as dl
from src.data_loader import (
    DATA_DIR,
    N_SPLITS,
    SchemaError,
    cv_splitter,
    load_test,
    load_train,
    validate_schema,
)

# Real data is only present in the primary checkout, not in worktrees.
_REAL_DATA_AVAILABLE = (DATA_DIR / "train.csv").is_file() and (
    DATA_DIR / "test.csv"
).is_file()
_requires_real_data = pytest.mark.skipif(
    not _REAL_DATA_AVAILABLE,
    reason=f"real competition CSVs not found under {DATA_DIR}",
)


def _valid_train_frame() -> pd.DataFrame:
    """Build a small, schema-valid training frame (3 rows, one per class).

    Dtypes are set explicitly to mirror what load_train produces: int64 id,
    float64 numerics, str/object categoricals. Values sit inside every sanity
    range and use only allowed categorical members.

    Returns:
        A DataFrame that validate_schema(df, "train") accepts.
    """
    return pd.DataFrame(
        {
            "id": pd.Series([0, 1, 2], dtype="int64"),
            "alpha": pd.Series([147.7, 127.9, 179.8], dtype="float64"),
            "delta": pd.Series([16.9, 32.3, 35.3], dtype="float64"),
            "u": pd.Series([25.4, 20.7, 21.0], dtype="float64"),
            "g": pd.Series([21.9, 19.0, 21.1], dtype="float64"),
            "r": pd.Series([20.3, 17.6, 21.2], dtype="float64"),
            "i": pd.Series([19.2, 17.2, 20.6], dtype="float64"),
            "z": pd.Series([18.6, 16.8, 20.6], dtype="float64"),
            "redshift": pd.Series([0.41, 0.16, 2.82], dtype="float64"),
            "spectral_type": pd.Series(["M", "M", "O/B"], dtype="object"),
            "galaxy_population": pd.Series(
                ["Red_Sequence", "Red_Sequence", "Blue_Cloud"], dtype="object"
            ),
            "class": pd.Series(["GALAXY", "GALAXY", "QSO"], dtype="object"),
        }
    )


@pytest.fixture
def valid_train() -> pd.DataFrame:
    """Provide a fresh valid training frame per test (no cross-test mutation)."""
    return _valid_train_frame()


# --------------------------------------------------------------------------- #
# Happy paths.
# --------------------------------------------------------------------------- #
def test_valid_train_frame_passes(valid_train: pd.DataFrame) -> None:
    """A contract-conformant train frame validates without raising."""
    # Returns None on success; absence of exception is the assertion.
    assert validate_schema(valid_train, "train") is None


def test_valid_test_frame_passes(valid_train: pd.DataFrame) -> None:
    """A test frame (train minus `class`) validates under the test contract."""
    test_frame = valid_train.drop(columns=["class"])
    assert validate_schema(test_frame, "test") is None


def test_unknown_split_raises_keyerror(valid_train: pd.DataFrame) -> None:
    """An unrecognised split name is a programming error, not a data error."""
    with pytest.raises(KeyError, match="Unknown split"):
        validate_schema(valid_train, "holdout")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Negative: structural violations (count, names).
# --------------------------------------------------------------------------- #
def test_renamed_column_raises(valid_train: pd.DataFrame) -> None:
    """Renaming a column trips the names/order check."""
    df = valid_train.rename(columns={"redshift": "z_shift"})
    with pytest.raises(SchemaError, match="column names/order mismatch"):
        validate_schema(df, "train")


def test_extra_column_raises(valid_train: pd.DataFrame) -> None:
    """An unexpected extra column trips the count and names checks."""
    df = valid_train.copy()
    df["unexpected"] = 1.0
    with pytest.raises(SchemaError, match="column count mismatch"):
        validate_schema(df, "train")


def test_missing_column_raises(valid_train: pd.DataFrame) -> None:
    """Dropping a required column trips the count check."""
    df = valid_train.drop(columns=["u"])
    with pytest.raises(SchemaError, match="column count mismatch"):
        validate_schema(df, "train")


def test_reordered_columns_raises(valid_train: pd.DataFrame) -> None:
    """Same columns in the wrong order trips the order check (count is fine)."""
    cols = list(valid_train.columns)
    cols[1], cols[2] = cols[2], cols[1]  # swap alpha and delta
    df = valid_train[cols]
    with pytest.raises(SchemaError, match="names/order mismatch"):
        validate_schema(df, "train")


# --------------------------------------------------------------------------- #
# Negative: dtype violations.
# --------------------------------------------------------------------------- #
def test_wrong_dtype_numeric_as_text_raises(valid_train: pd.DataFrame) -> None:
    """A numeric column delivered as text trips the dtype-family check."""
    df = valid_train.copy()
    df["redshift"] = df["redshift"].astype("object")
    with pytest.raises(SchemaError, match="does not match expected family"):
        validate_schema(df, "train")


def test_wrong_dtype_id_as_float_raises(valid_train: pd.DataFrame) -> None:
    """`id` must be integer; a float id trips the dtype-family check."""
    df = valid_train.copy()
    df["id"] = df["id"].astype("float64")
    with pytest.raises(SchemaError, match="column 'id'.*expected family 'int'"):
        validate_schema(df, "train")


# --------------------------------------------------------------------------- #
# Negative: NaN threshold.
# --------------------------------------------------------------------------- #
def test_nan_injected_in_numeric_raises(valid_train: pd.DataFrame) -> None:
    """A NaN in a numeric column (default threshold 0.0) fails loudly.

    Injecting NaN also promotes redshift's dtype, so we expect the NaN-threshold
    violation to be present in the message.
    """
    df = valid_train.copy()
    df.loc[0, "redshift"] = np.nan
    with pytest.raises(SchemaError, match="NaN fraction"):
        validate_schema(df, "train")


def test_nan_injected_in_categorical_raises(valid_train: pd.DataFrame) -> None:
    """A NaN in a categorical column (default threshold 0.0) fails loudly."""
    df = valid_train.copy()
    df.loc[1, "galaxy_population"] = None
    with pytest.raises(SchemaError, match="column 'galaxy_population'.*NaN fraction"):
        validate_schema(df, "train")


# --------------------------------------------------------------------------- #
# Negative: numeric range sanity.
# --------------------------------------------------------------------------- #
def test_alpha_above_max_raises(valid_train: pd.DataFrame) -> None:
    """alpha > 360 degrees is physically impossible and must fail."""
    df = valid_train.copy()
    df.loc[0, "alpha"] = 400.0
    with pytest.raises(SchemaError, match=r"column 'alpha'.*above max 360"):
        validate_schema(df, "train")


def test_delta_below_min_raises(valid_train: pd.DataFrame) -> None:
    """delta < -90 degrees is physically impossible and must fail."""
    df = valid_train.copy()
    df.loc[2, "delta"] = -91.0
    with pytest.raises(SchemaError, match=r"column 'delta'.*below min -90"):
        validate_schema(df, "train")


def test_redshift_below_floor_raises(valid_train: pd.DataFrame) -> None:
    """redshift below the -0.01 measurement-noise floor must fail."""
    df = valid_train.copy()
    df.loc[0, "redshift"] = -0.5
    with pytest.raises(SchemaError, match=r"column 'redshift'.*below min -0.01"):
        validate_schema(df, "train")


def test_magnitude_above_max_raises(valid_train: pd.DataFrame) -> None:
    """A corrupt sentinel magnitude (e.g. 9999) trips the upper bound."""
    df = valid_train.copy()
    df.loc[1, "g"] = 9999.0
    with pytest.raises(SchemaError, match=r"column 'g'.*above max 40"):
        validate_schema(df, "train")


# --------------------------------------------------------------------------- #
# Negative: categorical membership.
# --------------------------------------------------------------------------- #
def test_unexpected_class_value_raises(valid_train: pd.DataFrame) -> None:
    """A class outside {GALAXY, STAR, QSO} must fail."""
    df = valid_train.copy()
    df.loc[0, "class"] = "NEBULA"
    with pytest.raises(SchemaError, match=r"column 'class'.*not in allowed set"):
        validate_schema(df, "train")


def test_unexpected_spectral_type_raises(valid_train: pd.DataFrame) -> None:
    """A spectral type outside the 4-member enum must fail."""
    df = valid_train.copy()
    df.loc[2, "spectral_type"] = "Z9"
    with pytest.raises(SchemaError, match=r"column 'spectral_type'.*not in allowed"):
        validate_schema(df, "train")


def test_multiple_violations_all_reported(valid_train: pd.DataFrame) -> None:
    """The validator accumulates and reports every violation, not just one."""
    df = valid_train.copy()
    df.loc[0, "alpha"] = 400.0  # range
    df.loc[0, "class"] = "NEBULA"  # membership
    with pytest.raises(SchemaError) as exc_info:
        validate_schema(df, "train")
    message = str(exc_info.value)
    assert "alpha" in message
    assert "class" in message


# --------------------------------------------------------------------------- #
# Loader error handling and logging.
# --------------------------------------------------------------------------- #
def test_load_missing_file_raises(tmp_path) -> None:
    """Loading from a directory without the CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="train.csv"):
        load_train(data_dir=tmp_path)


def test_valid_frame_logs_pass(
    valid_train: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    """A passing validation emits an INFO log (structured logging, no print)."""
    with caplog.at_level(logging.INFO, logger="src.data_loader"):
        validate_schema(valid_train, "train")
    assert any("Schema validation passed" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# Cross-validation splitter object (instantiated, NOT executed here).
# --------------------------------------------------------------------------- #
def test_cv_splitter_is_configured() -> None:
    """The exposed splitter is a StratifiedKFold with the frozen config."""
    assert isinstance(cv_splitter, StratifiedKFold)
    assert cv_splitter.n_splits == N_SPLITS == 5
    assert cv_splitter.shuffle is True
    assert cv_splitter.random_state == dl.SEED == 42


# --------------------------------------------------------------------------- #
# Integration: real competition data (skipped when absent).
# --------------------------------------------------------------------------- #
@_requires_real_data
def test_load_real_train_passes() -> None:
    """The real train.csv loads and validates against the train contract."""
    df = load_train()
    assert df.shape[1] == 12
    assert df.shape[0] > 500_000
    assert list(df.columns)[-1] == "class"
    assert set(df["class"].unique()) == {"GALAXY", "STAR", "QSO"}


@_requires_real_data
def test_load_real_test_passes() -> None:
    """The real test.csv loads and validates against the test contract."""
    df = load_test()
    assert df.shape[1] == 11
    assert df.shape[0] > 200_000
    assert "class" not in df.columns


@_requires_real_data
def test_real_train_idempotent_load() -> None:
    """Loading the same CSV twice yields identical frames (deterministic)."""
    a = load_train()
    b = load_train()
    pd.testing.assert_frame_equal(a, b)
