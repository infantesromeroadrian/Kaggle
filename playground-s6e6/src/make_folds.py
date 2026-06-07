"""Freeze the cross-validation folds for Kaggle Playground S6E6.

Cycle C2 (Data) deliverable BL-005. Materialises the StratifiedKFold split that
data_loader.py only INSTANTIATES (never executes), and persists the per-fold
train/validation row indices to disk so that every downstream cycle (C3 feature
work, C5 POC, C6 build) trains and validates on the IDENTICAL partition. Without
frozen folds, two runs of "5-fold CV" are not comparable and any leaderboard
delta is noise.

Why persist indices rather than re-split each run:
  - StratifiedKFold(shuffle=True, random_state=42) IS reproducible from the
    seed, so the .npz is strictly derivable — it is NOT versioned in git (data/
    is gitignored). But materialising it once and loading it everywhere removes
    a whole class of "did someone pass a different seed / a different splitter"
    bugs, and lets non-Python tooling read the partition.
  - The contract checked here (no cross-fold validation overlap; class
    proportions preserved per fold) is the BL-005 gate. We assert it, not assume
    it.

Output:
  data/folds/cv_indices.npz — arrays:
    fold{k}_train, fold{k}_val for k in [0, N_SPLITS).
  data/folds/cv_indices_meta.json — seed, n_splits, n_rows, per-fold sizes and
    class proportions, for provenance.

The fold indices are POSITIONAL row indices into the train frame as returned by
load_train() (i.e. into a 0..n-1 RangeIndex), NOT `id` values. Consumers must
apply them with .iloc, against the same loader output.

Side effects:
  - Writes the two files above (creates data/folds/ if missing).
  - Emits log records. Determinism via the loader's SEED-pinned cv_splitter.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.data_loader import (
    DATA_DIR,
    N_SPLITS,
    SEED,
    TARGET,
    cv_splitter,
    load_train,
)

logger = logging.getLogger(__name__)

# Destination for the frozen fold indices. data/ is gitignored; the folds are
# reproducible from SEED, so we persist them as a build artifact, not in VCS.
FOLDS_DIR: Final[Path] = DATA_DIR / "folds"
INDICES_PATH: Final[Path] = FOLDS_DIR / "cv_indices.npz"
META_PATH: Final[Path] = FOLDS_DIR / "cv_indices_meta.json"

# Maximum absolute deviation of a fold's per-class proportion from the global
# proportion before the BL-005 stratification gate fails. 0.005 == 0.5 pp; with
# ~115k validation rows per fold the real jitter is far tighter, so this is a
# deliberately generous, non-brittle gate threshold.
_STRAT_TOL: Final[float] = 0.005


def _class_proportions(labels: pd.Series) -> dict[str, float]:
    """Return the class proportion map for a label slice.

    Args:
        labels: A slice of the class target.

    Returns:
        {class_name: fraction} summing to 1.0 (within float tolerance).
    """
    counts = Counter(labels)
    total = sum(counts.values())
    return {cls: counts[cls] / total for cls in sorted(counts)}


def build_folds() -> dict[str, np.ndarray]:
    """Materialise the StratifiedKFold partition into a dict of index arrays.

    Uses the SEED-pinned `cv_splitter` from data_loader (StratifiedKFold,
    5 folds, shuffle=True, random_state=42) on the loaded train target. Indices
    are positional (into the load_train() RangeIndex), to be applied with .iloc.

    Returns:
        Mapping with keys fold{k}_train / fold{k}_val -> int64 index arrays.

    Raises:
        ValueError: If a fold's validation set is empty (degenerate split).
    """
    train_df = load_train()
    target = train_df[TARGET]
    # StratifiedKFold needs only y for the stratification; X is a placeholder of
    # the right length. We pass the positional range so returned indices are
    # positional by construction.
    x_placeholder = np.zeros((len(target), 1))

    folds: dict[str, np.ndarray] = {}
    for k, (train_idx, val_idx) in enumerate(
        cv_splitter.split(x_placeholder, target.to_numpy())
    ):
        if len(val_idx) == 0:
            raise ValueError(f"fold {k}: empty validation set")
        folds[f"fold{k}_train"] = train_idx.astype(np.int64)
        folds[f"fold{k}_val"] = val_idx.astype(np.int64)
        logger.info(
            "fold %d: train=%d val=%d", k, len(train_idx), len(val_idx)
        )
    return folds


def verify_folds(folds: dict[str, np.ndarray], target: pd.Series) -> dict[str, object]:
    """Assert the BL-005 fold contract and return a provenance summary.

    Contract (all must hold, else the gate FAILS with AssertionError):
      1. The union of all validation indices == the full row set, exactly once
         (every row validated exactly once across the 5 folds).
      2. No row appears in two different folds' validation sets (no cross-fold
         leakage of a validation row).
      3. Within each fold, train and val indices are disjoint.
      3b. Within each fold, train and val TOGETHER cover arange(n) exactly —
         no row is silently dropped from the fold. Together with 3 this makes
         {train, val} an exact partition of the row set.
      4. Each fold's validation class proportions track the global 65/20/14
         within a tight tolerance (stratification actually held).

    Args:
        folds: The dict produced by build_folds().
        target: The class target aligned to the fold indices (positional).

    Returns:
        A provenance dict: per-fold sizes and validation class proportions, plus
        the global proportions, for the meta JSON.

    Raises:
        AssertionError: On any contract violation. This is the gate.
    """
    n = len(target)
    global_props = _class_proportions(target)

    all_val = np.concatenate([folds[f"fold{k}_val"] for k in range(N_SPLITS)])

    # 1 + 2: every row validated exactly once -> sorted union is 0..n-1 with no
    # repeats. A repeat would make len(unique) < len(all_val).
    assert len(all_val) == n, (
        f"validation indices cover {len(all_val)} rows, expected {n}"
    )
    assert len(np.unique(all_val)) == n, (
        "duplicate row across validation folds (cross-fold leakage)"
    )
    np.testing.assert_array_equal(
        np.sort(all_val), np.arange(n), err_msg="validation union != full row set"
    )

    per_fold: list[dict[str, object]] = []
    target_arr = target.to_numpy()
    for k in range(N_SPLITS):
        tr = folds[f"fold{k}_train"]
        va = folds[f"fold{k}_val"]

        # 3: train/val disjoint within the fold.
        assert np.intersect1d(tr, va).size == 0, f"fold {k}: train/val overlap"

        # 3b: train+val COVERS every row exactly once within the fold. The
        # docstring promises we ASSERT completeness, not assume it: a fold that
        # silently dropped rows (train+val != full set) would train/evaluate on
        # a subset and the per-fold metric would be measured on the wrong
        # denominator. Disjoint (check 3) + this completeness check together
        # mean {train, val} is an exact partition of arange(n).
        assert np.array_equal(
            np.sort(np.concatenate([tr, va])), np.arange(n)
        ), f"fold {k}: train+val does not cover all rows"

        # 4: stratification held. Tolerance is _STRAT_TOL (see module constant).
        val_props = _class_proportions(pd.Series(target_arr[va]))
        for cls, gp in global_props.items():
            assert abs(val_props.get(cls, 0.0) - gp) < _STRAT_TOL, (
                f"fold {k}: class {cls} proportion {val_props.get(cls, 0.0):.4f} "
                f"deviates from global {gp:.4f} beyond tolerance"
            )

        per_fold.append(
            {
                "fold": k,
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "val_class_proportions": {c: round(p, 6) for c, p in val_props.items()},
            }
        )
        logger.info(
            "fold %d verified: val props %s",
            k,
            {c: round(p, 4) for c, p in val_props.items()},
        )

    logger.info("All %d folds verified: disjoint, complete, stratified", N_SPLITS)
    return {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "n_rows": n,
        "global_class_proportions": {c: round(p, 6) for c, p in global_props.items()},
        "folds": per_fold,
    }


def save_folds(folds: dict[str, np.ndarray], meta: dict[str, object]) -> None:
    """Persist fold indices (.npz) and provenance (.json) to FOLDS_DIR.

    Args:
        folds: The dict of index arrays from build_folds().
        meta: The provenance summary from verify_folds().

    Side effects:
        Creates FOLDS_DIR if absent; writes cv_indices.npz and
        cv_indices_meta.json.
    """
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INDICES_PATH, **folds)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d arrays)", INDICES_PATH, len(folds))
    logger.info("Wrote %s", META_PATH)


def main() -> None:
    """CLI entry point: build, verify (the gate), and persist the frozen folds."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    train_df = load_train()
    target = train_df[TARGET]
    folds = build_folds()
    meta = verify_folds(folds, target)
    save_folds(folds, meta)
    logger.info("BL-005 freeze-folds complete and verified")


if __name__ == "__main__":
    main()
