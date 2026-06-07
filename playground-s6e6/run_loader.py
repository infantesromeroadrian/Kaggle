"""Smoke entrypoint: load + validate the real S6E6 CSVs and print a data profile.

Run from the project root:
    python run_loader.py

Not part of the test suite. Exists so the BL-002 loader can be exercised end to
end against the real competition data, emitting a per-column NaN profile and the
NaN-by-class cross-tab for the BL-003 leakage audit. Uses logging, not
print, for status; the profile table is the one intentional stdout artifact.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from src.data_loader import TARGET, load_test, load_train


def _nan_profile(df: pd.DataFrame, name: str) -> None:
    """Print per-column NaN count and percentage for a frame."""
    n = len(df)
    print(f"\n[{name}] shape={df.shape}")
    print(f"[{name}] per-column NaN:")
    for col in df.columns:
        cnt = int(df[col].isna().sum())
        print(f"  {col:18s} NaN={cnt:8d}  pct={100 * cnt / n:.4f}%")


def main() -> int:
    """Load both splits, validate, and print the data-quality profile."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    train = load_train()
    test = load_test()

    _nan_profile(train, "train")
    _nan_profile(test, "test")

    print("\n[train] class distribution (%):")
    print(train[TARGET].value_counts(normalize=True).mul(100).round(2).to_string())

    # NaN-by-class cross-tab for the two categorical features flagged in BL-002
    # as candidates for class-conditional sparsity. Reported regardless of
    # outcome so the leakage audit sees the evidence, not a masked result.
    print("\n[train] categorical NaN by class (galaxy_population / spectral_type):")
    for cls in sorted(train[TARGET].unique()):
        sub = train[train[TARGET] == cls]
        gp = sub["galaxy_population"].isna().mean() * 100
        st = sub["spectral_type"].isna().mean() * 100
        print(
            f"  class={cls:7s} n={len(sub):7d}  "
            f"galaxy_population NaN={gp:.3f}%  spectral_type NaN={st:.3f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
