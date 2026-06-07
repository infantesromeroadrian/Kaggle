"""Shared cross-validation primitives for Kaggle Playground S6E6 (BL-019).

Cycle C5 (POC). Single source of truth for the two CV helpers that were copied
byte-for-byte between src.eda and src.feature_eval before this refactor:

  - compute_sample_weights: balanced (1/class_freq) per-sample weights, the
    sample_weight equivalent of class_weight='balanced'. The official metric is
    balanced_accuracy (unweighted mean per-class recall), so the model must
    weight minority classes up; applied to TRAIN rows only, per fold.
  - load_folds / FOLDS_PATH: load the frozen positional fold indices that
    make_folds.py (BL-005) materialised. Every cycle trains and validates on the
    IDENTICAL partition, which is what makes per-fold deltas paired and any
    leaderboard delta meaningful rather than split noise.

Why a dedicated module and not just leave the copies:
  Two byte-identical definitions in two files are a maintenance hazard — a fix to
  one silently desynchronises the comparison the whole C3/C5 statistics rest on
  (eda's baseline vs feature_eval's ablations vs the C5 tuner must all weight and
  fold IDENTICALLY). Centralising them makes that invariant structural, not a
  convention someone has to remember.

This module is intentionally tiny and dependency-light (numpy + the loader's
DATA_DIR only). It imports NOTHING from eda/feature_eval, so it sits cleanly at
the bottom of the import graph and both can depend on it without a cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from src.data_loader import DATA_DIR, N_SPLITS

# Frozen CV folds (BL-005). data/ is gitignored and the .npz is reproducible from
# SEED, so it is a build artifact: make_folds.py writes it under DATA_DIR. The
# loader's DATA_DIR points at the primary checkout, so consumers in a worktree
# read the same partition without copying anything.
FOLDS_PATH: Final[Path] = DATA_DIR / "folds" / "cv_indices.npz"


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Balanced per-sample weights (1/class_freq, normalised to mean 1.0).

    The official metric is balanced_accuracy (the unweighted mean of per-class
    recall), so the model must NOT optimise raw accuracy on the GALAXY-dominated
    data (65/20/14 GALAXY/QSO/STAR). Weighting each sample by 1 / class_frequency
    is the sample_weight equivalent of class_weight='balanced'; we normalise to
    mean 1.0 so the effective sample size (and thus the regularisation scale) is
    unchanged versus unweighted training.

    Leakage note: this MUST be called with one fold's TRAIN labels only, never the
    validation slice and never the full dataset — the class frequencies are a
    statistic of the training data, and computing them over val would leak the
    val class balance into the fit.

    Args:
        y: Integer-encoded labels for ONE fold's train slice.

    Returns:
        Float weights, same length as y, mean ~1.0. Minority-class rows get a
        weight > 1, the majority class < 1.

    Raises:
        ZeroDivisionError: If y is empty (no classes to weight).
    """
    classes, counts = np.unique(y, return_counts=True)
    freq = {c: n / len(y) for c, n in zip(classes, counts)}
    raw = np.array([1.0 / freq[c] for c in y], dtype=np.float64)
    return raw * (len(raw) / raw.sum())  # normalise to mean 1.0


def load_folds(path: Path = FOLDS_PATH) -> dict[str, np.ndarray]:
    """Load the frozen CV fold indices from the BL-005 .npz artifact.

    The indices are POSITIONAL (into the load_train() RangeIndex), so consumers
    apply them with .iloc against the same loader output.

    Args:
        path: Path to cv_indices.npz. Defaults to FOLDS_PATH.

    Returns:
        {fold{k}_train, fold{k}_val: int64 index array} for k in [0, N_SPLITS).

    Raises:
        FileNotFoundError: If the artifact is missing (folds not yet frozen).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen folds not found at {path}. Run src/make_folds.py (BL-005) first."
        )
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def fold_indices(folds: dict[str, np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the (train, val) positional index arrays for fold k.

    A thin accessor so callers iterate folds by integer index without re-spelling
    the f"fold{k}_train" key convention in every CV loop.

    Args:
        folds: The dict returned by load_folds().
        k: Fold index in [0, N_SPLITS).

    Returns:
        (train_idx, val_idx) int64 arrays for fold k.

    Raises:
        KeyError: If fold k is not present in `folds`.
    """
    return folds[f"fold{k}_train"], folds[f"fold{k}_val"]


__all__ = ["FOLDS_PATH", "N_SPLITS", "compute_sample_weights", "fold_indices", "load_folds"]
