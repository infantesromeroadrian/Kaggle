"""Data loader and schema validation for Kaggle Playground S6E6 (stellar classification).

Cycle C2 (Data) deliverable BL-002. Loads the raw competition CSVs with an
EXPLICIT dtype contract (no silent type inference) and validates each frame
against a per-column schema that fails loudly on any contract violation.

Scope (intentionally narrow — C2 only):
  - NO EDA, NO modelling, NO feature engineering (those belong to C3).
  - NO Airflow/Spark/dbt: the data is local CSV and fits comfortably in
    pandas on the target host (RTX 2000 Ada workstation).

The StratifiedKFold splitter is INSTANTIATED and exposed here but never
executed. Freezing the actual fold indices is BL-005 (the leakage audit step);
doing it here would pre-empt that gate.

Side effects on import:
  - Sets the global RNG seed (random + numpy) to SEED for reproducibility.
  - Configures a module-level logger (no handler attached: the application
    owns handler/level configuration; we only emit records).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------------------------------- #
# Reproducibility — seed the global RNGs at import time.
#
# We seed random + numpy only. No torch/tf here: this module touches neither.
# StratifiedKFold receives its own random_state below, so its determinism does
# not depend on this global seed — but we set it so any incidental randomness
# in downstream import-time code inherits a fixed seed.
# --------------------------------------------------------------------------- #
SEED: Final[int] = 42
random.seed(SEED)
np.random.seed(SEED)

logger = logging.getLogger(__name__)

# Split identifiers used throughout the module.
Split = Literal["train", "test"]

# Default data directory (the raw competition CSVs). The CSVs are gitignored
# (download them via the Kaggle API — see README). Resolved relative to this
# file so the loader works from any checkout. Overridable per call so it stays
# reusable from tests and from other projects without env coupling.
DATA_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "data"

# Pandas reports text columns as "object" (pandas <=2.x) or "str"/"string"
# (pandas >=3.x / Arrow-backed). We accept any of these as "the column is
# textual" rather than pinning one runtime's spelling. This is a deliberate
# robustness choice, not laxity: numeric/int contracts remain strict.
_TEXT_DTYPE_NAMES: Final[frozenset[str]] = frozenset({"object", "str", "string"})


class SchemaError(ValueError):
    """Raised when a DataFrame violates the declared schema contract.

    Subclasses ValueError so callers that only catch ValueError still work,
    while allowing precise `except SchemaError` handling upstream.
    """


@dataclass(frozen=True)
class ColumnSpec:
    """Contract for a single column.

    Attributes:
        name: Exact column name expected in the CSV.
        dtype: Expected dtype family — "int", "float", or "text". We compare
            against a family rather than a concrete numpy dtype so the contract
            survives platform/pandas-version differences (int64 vs int32, the
            str/object spelling drift). Numeric families are still enforced.
        nan_threshold: Maximum FRACTION of NaN allowed in this column, in
            [0.0, 1.0]. Default 0.0 means "no NaN tolerated". Configurable per
            column precisely so a column that is legitimately sparse (e.g. a
            label that only exists for one class) can be admitted without
            silencing the signal — the threshold is explicit and auditable.
        min_value: Inclusive lower sanity bound for numeric columns, or None.
        max_value: Inclusive upper sanity bound for numeric columns, or None.
        allowed: Closed set of permitted values for categorical columns, or
            None to skip the membership check.
    """

    name: str
    dtype: Literal["int", "float", "text"]
    nan_threshold: float = 0.0
    min_value: float | None = None
    max_value: float | None = None
    allowed: frozenset[str] | None = None


# --------------------------------------------------------------------------- #
# Column contracts.
#
# Ranges are physical sanity bounds, NOT the observed min/max:
#   - alpha (right ascension): [0, 360] degrees by definition.
#   - delta (declination): [-90, 90] degrees by definition. Observed range in
#     this dataset is ~[-18, 79]; the wider bound is the physical envelope.
#   - u/g/r/i/z: SDSS apparent magnitudes. Observed includes a slightly
#     negative u (~-0.14) from photometric noise, which is legitimate. We use
#     a wide [-10, 40] envelope so genuine photometric outliers do not trip the
#     validator while still catching obviously corrupt values (e.g. 9999).
#   - redshift: cosmological z. Physically >= 0; the pipeline sees a -0.01
#     floor (measurement noise near zero), so the contract floor is -0.01 per
#     the BL-002 spec, NOT 0.
#
# Categorical `allowed` sets are derived from the ACTUAL observed train values
# (4 spectral types, 2 galaxy populations, 3 classes). Encoding them as closed
# sets means an unexpected category in test/future data fails loudly.
# --------------------------------------------------------------------------- #
_MAG_MIN: Final[float] = -10.0
_MAG_MAX: Final[float] = 40.0

_SPECTRAL_TYPES: Final[frozenset[str]] = frozenset({"A/F", "G/K", "M", "O/B"})
_GALAXY_POPULATIONS: Final[frozenset[str]] = frozenset({"Blue_Cloud", "Red_Sequence"})
_CLASSES: Final[frozenset[str]] = frozenset({"GALAXY", "STAR", "QSO"})

# Feature/identity columns shared by train and test, in CSV order.
_SHARED_SPECS: Final[tuple[ColumnSpec, ...]] = (
    ColumnSpec("id", "int"),
    ColumnSpec("alpha", "float", min_value=0.0, max_value=360.0),
    ColumnSpec("delta", "float", min_value=-90.0, max_value=90.0),
    ColumnSpec("u", "float", min_value=_MAG_MIN, max_value=_MAG_MAX),
    ColumnSpec("g", "float", min_value=_MAG_MIN, max_value=_MAG_MAX),
    ColumnSpec("r", "float", min_value=_MAG_MIN, max_value=_MAG_MAX),
    ColumnSpec("i", "float", min_value=_MAG_MIN, max_value=_MAG_MAX),
    ColumnSpec("z", "float", min_value=_MAG_MIN, max_value=_MAG_MAX),
    ColumnSpec("redshift", "float", min_value=-0.01, max_value=None),
    ColumnSpec("spectral_type", "text", allowed=_SPECTRAL_TYPES),
    ColumnSpec("galaxy_population", "text", allowed=_GALAXY_POPULATIONS),
)

# train adds the supervised target `class`.
_TRAIN_SPECS: Final[tuple[ColumnSpec, ...]] = _SHARED_SPECS + (
    ColumnSpec("class", "text", allowed=_CLASSES),
)
_TEST_SPECS: Final[tuple[ColumnSpec, ...]] = _SHARED_SPECS

_SCHEMAS: Final[dict[Split, tuple[ColumnSpec, ...]]] = {
    "train": _TRAIN_SPECS,
    "test": _TEST_SPECS,
}

# Target column name and the metric-relevant identity column.
TARGET: Final[str] = "class"
ID_COLUMN: Final[str] = "id"

# Cross-validation splitter for the multiclass target. 5 folds, stratified to
# preserve the GALAXY/QSO/STAR proportions (65/20/14), shuffled with a fixed
# seed. We EXPOSE the object; we do NOT call .split() (that is BL-005).
N_SPLITS: Final[int] = 5
cv_splitter: Final[StratifiedKFold] = StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=SEED
)


def _read_dtypes(specs: tuple[ColumnSpec, ...]) -> dict[str, type]:
    """Build the explicit pandas read_csv dtype map from a schema.

    We pass concrete Python types to read_csv so numeric columns are parsed as
    int/float and text columns as str — never inferred. NaN handling for
    numeric columns still applies (pandas promotes int-with-NaN to float), which
    the validator catches via the NaN-threshold check.

    Args:
        specs: The column contract for a split.

    Returns:
        Mapping {column_name: python_type} suitable for read_csv(dtype=...).
    """
    family_to_type: dict[str, type] = {"int": int, "float": float, "text": str}
    return {spec.name: family_to_type[spec.dtype] for spec in specs}


def _load(split: Split, data_dir: Path) -> pd.DataFrame:
    """Load one split CSV with an explicit dtype contract, then validate it.

    Args:
        split: "train" or "test".
        data_dir: Directory containing train.csv / test.csv.

    Returns:
        Validated DataFrame with columns in contract order.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        SchemaError: If the loaded frame violates its contract.
    """
    specs = _SCHEMAS[split]
    csv_path = data_dir / f"{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected {split} CSV at {csv_path}, not found")

    logger.info("Loading %s split from %s", split, csv_path)
    # dtype= forces the contract; we do not rely on inference. We do NOT pass
    # usecols so that an unexpected extra column is caught by validate_schema
    # rather than silently dropped.
    frame = pd.read_csv(csv_path, dtype=_read_dtypes(specs))
    logger.info("Loaded %s: %d rows, %d columns", split, len(frame), frame.shape[1])

    validate_schema(frame, split)
    return frame


def load_train(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and validate the training split.

    The returned frame retains `id` (needed only for traceability — it is NOT a
    feature and must be dropped before modelling in C3) and the `class` target.

    Args:
        data_dir: Directory containing train.csv. Defaults to DATA_DIR.

    Returns:
        Validated training DataFrame (12 columns).

    Raises:
        FileNotFoundError: If train.csv is missing.
        SchemaError: On any contract violation.
    """
    return _load("train", data_dir)


def load_test(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and validate the test split.

    The returned frame retains `id`, which is required to build the submission
    file (id -> predicted class).

    Args:
        data_dir: Directory containing test.csv. Defaults to DATA_DIR.

    Returns:
        Validated test DataFrame (11 columns, no `class`).

    Raises:
        FileNotFoundError: If test.csv is missing.
        SchemaError: On any contract violation.
    """
    return _load("test", data_dir)


def _dtype_matches(series: pd.Series, family: str) -> bool:
    """Check whether a Series' dtype satisfies a contract family.

    Args:
        series: The column to check.
        family: "int", "float", or "text".

    Returns:
        True if the dtype is compatible with the family.

    Notes:
        - "text" accepts object/str/string (pandas-version drift, see
          _TEXT_DTYPE_NAMES).
        - "float" requires a floating dtype.
        - "int" requires an integer dtype. A column that is logically int but
          carries NaN will have been promoted to float by pandas; that is
          surfaced as a dtype mismatch here AND as a NaN-threshold failure,
          which is the louder of the two signals.
    """
    if family == "text":
        return series.dtype.name in _TEXT_DTYPE_NAMES
    if family == "float":
        return pd.api.types.is_float_dtype(series.dtype)
    if family == "int":
        return pd.api.types.is_integer_dtype(series.dtype)
    raise ValueError(f"Unknown dtype family: {family!r}")  # pragma: no cover


def validate_schema(df: pd.DataFrame, split: Split) -> None:
    """Validate a DataFrame against the declared schema for `split`, loudly.

    Checks, in order, accumulating ALL violations before raising so the caller
    sees every problem at once rather than fixing them one round-trip at a time:

      1. Column count matches the contract.
      2. Column names and order match the contract exactly.
      3. Each column's dtype matches its family (int / float / text).
      4. Per-column NaN fraction does not exceed its configured threshold
         (default 0.0 for every column in this dataset).
      5. Numeric sanity ranges (min_value / max_value, inclusive) hold.
      6. Categorical columns contain only their allowed values.

    Args:
        df: The DataFrame to validate.
        split: Which contract to apply ("train" or "test").

    Returns:
        None. Returns silently when the frame is valid.

    Raises:
        KeyError: If `split` is not a known split.
        SchemaError: If any contract violation is found. The message lists
            every violation discovered.

    Edge cases:
        - An extra/unexpected column is caught by checks 1 and 2.
        - A renamed column is caught by check 2.
        - An int column promoted to float by injected NaN is caught by both
          check 3 (dtype) and check 4 (NaN threshold).
    """
    if split not in _SCHEMAS:
        raise KeyError(f"Unknown split {split!r}; expected one of {tuple(_SCHEMAS)}")

    specs = _SCHEMAS[split]
    expected_names = [spec.name for spec in specs]
    violations: list[str] = []

    # 1. Column count.
    if df.shape[1] != len(specs):
        violations.append(
            f"column count mismatch: expected {len(specs)}, got {df.shape[1]} "
            f"(expected {expected_names}, got {list(df.columns)})"
        )

    # 2. Names and order. We compare the full ordered list so both a rename and
    # a reordering surface explicitly.
    if list(df.columns) != expected_names:
        violations.append(
            f"column names/order mismatch: expected {expected_names}, "
            f"got {list(df.columns)}"
        )

    # Per-column checks only run on columns that actually exist, so a missing
    # column does not mask downstream dtype/range checks for present columns.
    n_rows = len(df)
    for spec in specs:
        if spec.name not in df.columns:
            continue
        series = df[spec.name]

        # 3. Dtype family.
        if not _dtype_matches(series, spec.dtype):
            violations.append(
                f"column {spec.name!r}: dtype {series.dtype.name!r} does not "
                f"match expected family {spec.dtype!r}"
            )

        # 4. NaN threshold (fraction).
        if n_rows > 0:
            nan_frac = float(series.isna().mean())
            if nan_frac > spec.nan_threshold:
                violations.append(
                    f"column {spec.name!r}: NaN fraction {nan_frac:.6f} exceeds "
                    f"threshold {spec.nan_threshold:.6f}"
                )

        # 5. Numeric sanity ranges. Skip NaN entries so the range check reports
        # only genuine out-of-bound values, not nullness (covered by check 4).
        if spec.min_value is not None or spec.max_value is not None:
            numeric = pd.to_numeric(series, errors="coerce")
            valid = numeric.dropna()
            if spec.min_value is not None:
                below = int((valid < spec.min_value).sum())
                if below > 0:
                    violations.append(
                        f"column {spec.name!r}: {below} value(s) below "
                        f"min {spec.min_value} (observed min {valid.min()})"
                    )
            if spec.max_value is not None:
                above = int((valid > spec.max_value).sum())
                if above > 0:
                    violations.append(
                        f"column {spec.name!r}: {above} value(s) above "
                        f"max {spec.max_value} (observed max {valid.max()})"
                    )

        # 6. Categorical membership. Ignore NaN here (a sparse categorical is a
        # check-4 concern); only non-null values must be in the allowed set.
        if spec.allowed is not None:
            present = set(series.dropna().unique())
            unexpected = present - spec.allowed
            if unexpected:
                violations.append(
                    f"column {spec.name!r}: unexpected value(s) "
                    f"{sorted(unexpected)} not in allowed set "
                    f"{sorted(spec.allowed)}"
                )

    if violations:
        joined = "\n  - ".join(violations)
        message = f"Schema validation failed for split {split!r}:\n  - {joined}"
        logger.error(message)
        raise SchemaError(message)

    logger.info("Schema validation passed for split %r (%d rows)", split, n_rows)
