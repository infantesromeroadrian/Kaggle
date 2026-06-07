"""Tests for the shared CV primitives extracted in BL-019 (src.cv_utils).

Strategy:
  - Pure synthetic / on-disk fixtures (no real competition data), so the suite
    runs in any checkout including a worktree without data/.
  - One behaviour per test. We assert the CONTRACT the rest of the C3/C5
    statistics depend on:
      * compute_sample_weights: mean exactly 1.0, inverse-frequency ordering
        (minority class up), correct length, and the leakage-relevant property
        that the weight of a class scales as 1/its frequency.
      * load_folds: round-trips an .npz, surfaces a missing artifact loudly.
      * fold_indices: returns the right (train, val) arrays and raises on a
        missing fold.
  - A regression guard pins compute_sample_weights to the exact arithmetic of
    the two byte-identical copies it replaced (eda + feature_eval), so the C2
    baseline number it underpins cannot silently drift after the refactor.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cv_utils import compute_sample_weights, fold_indices, load_folds


# --------------------------------------------------------------------------- #
# compute_sample_weights
# --------------------------------------------------------------------------- #
def test_weights_mean_is_one() -> None:
    """Normalisation invariant: weights average to 1.0 (effective N unchanged)."""
    y = np.array([0, 0, 0, 1, 2])
    w = compute_sample_weights(y)
    assert w.mean() == pytest.approx(1.0)


def test_weights_match_length_of_labels() -> None:
    """One weight per row, in row order."""
    y = np.array([0, 1, 1, 2, 2, 2])
    assert compute_sample_weights(y).shape == y.shape


def test_minority_class_is_upweighted() -> None:
    """The rarer the class, the larger its sample weight (inverse-frequency)."""
    # class 0 appears 6x, class 1 twice, class 2 once -> w0 < w1 < w2.
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2])
    w = compute_sample_weights(y)
    w0 = w[y == 0][0]
    w1 = w[y == 1][0]
    w2 = w[y == 2][0]
    assert w0 < w1 < w2


def test_weight_ratio_equals_inverse_frequency_ratio() -> None:
    """A class half as frequent gets exactly double the weight.

    This is the property that makes the weighting the sample_weight equivalent
    of class_weight='balanced': w_a / w_b == freq_b / freq_a.
    """
    # class 0: 8 rows, class 1: 4 rows -> freq ratio 2:1 -> weight ratio 1:2.
    y = np.array([0] * 8 + [1] * 4)
    w = compute_sample_weights(y)
    w0 = w[y == 0][0]
    w1 = w[y == 1][0]
    assert w1 / w0 == pytest.approx(2.0)


def test_balanced_input_gives_uniform_weights() -> None:
    """Perfectly balanced classes -> every weight is exactly 1.0."""
    y = np.array([0, 0, 1, 1, 2, 2])
    w = compute_sample_weights(y)
    np.testing.assert_allclose(w, np.ones_like(w, dtype=float))


def test_weights_regression_pins_exact_arithmetic() -> None:
    """Pin the exact values so the refactor cannot drift the C2 baseline.

    Reproduces the closed form of the original eda/feature_eval helper:
    raw_i = 1/freq(class_i); w = raw * (N / sum(raw)). For y below (6:3:1) the
    expected weights are computed by hand from that formula.
    """
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 2])  # N=10, counts 6/3/1
    w = compute_sample_weights(y)

    n = len(y)
    freq = {0: 6 / n, 1: 3 / n, 2: 1 / n}
    raw = np.array([1.0 / freq[c] for c in y])
    expected = raw * (n / raw.sum())
    np.testing.assert_allclose(w, expected, rtol=0, atol=1e-12)


def test_empty_labels_returns_empty() -> None:
    """An empty label slice produces an empty weight array (no crash).

    This pins the inherited contract of the original helper: numpy maps the
    degenerate 0/0 normalisation to an empty result rather than raising. The
    folds always have rows, so this path is never hit in practice; the test
    exists so a future "harden empty input" change is a conscious decision, not
    an accidental contract break.
    """
    with np.errstate(invalid="ignore"):  # 0/0 normalisation on the empty array
        w = compute_sample_weights(np.array([], dtype=int))
    assert w.shape == (0,)


# --------------------------------------------------------------------------- #
# load_folds
# --------------------------------------------------------------------------- #
def test_load_folds_round_trips_npz(tmp_path) -> None:
    """load_folds returns exactly the arrays written to the .npz."""
    path = tmp_path / "cv_indices.npz"
    written = {
        "fold0_train": np.array([0, 1, 2], dtype=np.int64),
        "fold0_val": np.array([3, 4], dtype=np.int64),
    }
    np.savez_compressed(path, **written)

    loaded = load_folds(path)
    assert set(loaded) == set(written)
    for key, arr in written.items():
        np.testing.assert_array_equal(loaded[key], arr)


def test_load_folds_missing_artifact_raises(tmp_path) -> None:
    """A missing .npz fails loudly with a remediation hint, not silently."""
    with pytest.raises(FileNotFoundError, match="make_folds"):
        load_folds(tmp_path / "does_not_exist.npz")


# --------------------------------------------------------------------------- #
# fold_indices
# --------------------------------------------------------------------------- #
def test_fold_indices_returns_train_and_val() -> None:
    """fold_indices unpacks the (train, val) pair for a given fold index."""
    folds = {
        "fold0_train": np.array([0, 1]),
        "fold0_val": np.array([2]),
        "fold1_train": np.array([2]),
        "fold1_val": np.array([0, 1]),
    }
    tr, va = fold_indices(folds, 1)
    np.testing.assert_array_equal(tr, np.array([2]))
    np.testing.assert_array_equal(va, np.array([0, 1]))


def test_fold_indices_unknown_fold_raises() -> None:
    """Requesting a fold that was never frozen fails loudly."""
    folds = {"fold0_train": np.array([0]), "fold0_val": np.array([1])}
    with pytest.raises(KeyError):
        fold_indices(folds, 9)
