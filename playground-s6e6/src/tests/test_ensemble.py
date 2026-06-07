"""Tests for the C6 soft-voting ensemble helpers (BL-024, src.ensemble).

Scope: the PURE logic that does NOT need the real 577k-row CSVs or a multi-minute
3-model CV. The heavy OOF run is exercised by executing the script (its evidence is
docs/c6-build/ensemble_results.json), not by unit tests. Here we pin the contracts
the statistical review and code review care about:

  - OOF is leakage-free: the loop fits only on a fold's TRAIN slice and writes only
    that fold's VAL positions; we assert with a spy model that records which rows it
    was trained on, so a single train/val overlap would fail the test.
  - The blend is the exact arithmetic mean of probabilities (and a proper weighted
    mean for the optional-weights path), and argmax of the blend matches a hand
    computation.
  - The submission id->class mapping is correct: the written CSV pairs each test id
    with the decoded label of that row's blended argmax, in id order, with the right
    header.
  - Shapes/coverage: OOF matrix is (N, 3), every row written exactly once.
  - The paired-delta arithmetic matches feature_eval's rule and the shared 2-sigma
    threshold, and the fidelity guard fires on a real mismatch.

We avoid importing lightgbm/catboost/xgboost in these tests: the OOF loop is
model-agnostic (it takes a predict_fn closure), so a tiny deterministic stub model
exercises the loop's leakage/coverage logic without any GPU or multi-minute fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import ensemble
from src.feature_eval import _SIGMA_K as FE_SIGMA_K


# --------------------------------------------------------------------------- #
# Fixtures: a tiny synthetic dataset and a frozen 3-fold partition. No CSVs.
# --------------------------------------------------------------------------- #
@pytest.fixture
def synth(monkeypatch) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    """Build a tiny (12-row, 3-fold) design matrix + target + frozen folds.

    We monkeypatch ensemble.N_SPLITS to 3 so the loop iterates our 3 folds. The
    folds partition the 12 rows into 4-row val slices, disjoint and exhaustive.
    """
    monkeypatch.setattr(ensemble, "N_SPLITS", 3)
    rng = np.random.default_rng(0)
    n = 12
    x = pd.DataFrame(
        {
            "alpha": rng.uniform(0, 360, n),
            "delta": rng.uniform(-90, 90, n),
            "u": rng.uniform(10, 25, n),
            "g": rng.uniform(10, 25, n),
            "r": rng.uniform(10, 25, n),
            "i": rng.uniform(10, 25, n),
            "z": rng.uniform(10, 25, n),
            "redshift": rng.uniform(0, 3, n),
            "spectral_type": pd.Categorical(rng.choice(["A/F", "G/K", "M"], n)),
            "galaxy_population": pd.Categorical(rng.choice(["Blue_Cloud", "Red_Sequence"], n)),
        }
    )
    # Three classes, each present in every fold's val so balanced_accuracy is defined.
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    folds = {
        "fold0_train": np.array([4, 5, 6, 7, 8, 9, 10, 11]),
        "fold0_val": np.array([0, 1, 2, 3]),
        "fold1_train": np.array([0, 1, 2, 3, 8, 9, 10, 11]),
        "fold1_val": np.array([4, 5, 6, 7]),
        "fold2_train": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
        "fold2_val": np.array([8, 9, 10, 11]),
    }
    return x, y, folds


# --------------------------------------------------------------------------- #
# OOF leakage + coverage.
# --------------------------------------------------------------------------- #
def test_compute_oof_no_leakage_and_full_coverage(synth) -> None:
    """OOF fits only on TRAIN rows of a fold and writes only that fold's VAL rows.

    A spy predict_fn records the size of every train slice it sees and the rows it
    is asked to predict, and returns a one-hot proba so the OOF matrix carries a
    detectable signature. We then assert:
      - each train slice excludes the rows being predicted (no train/val overlap);
      - the OOF matrix is fully written (no NaN) and shaped (N, 3).
    """
    x, y, folds = synth
    seen_predict_rows: list[int] = []

    def spy_predict_fn(x_tr, y_tr, w_tr, x_pred):
        # The fold's train and the rows we predict must be DISJOINT by index.
        train_index = set(x_tr.index)
        pred_index = set(x_pred.index)
        assert train_index.isdisjoint(pred_index), "OOF leakage: train overlaps predicted rows"
        # Weights come from the train slice only and align to it.
        assert len(w_tr) == len(x_tr)
        seen_predict_rows.extend(pred_index)
        # One-hot proba keyed on a stable per-row value so the matrix is recognisable.
        proba = np.zeros((len(x_pred), ensemble.N_CLASSES))
        proba[:, 0] = 1.0
        return proba

    result = ensemble.compute_oof(x, y, folds, "spy", spy_predict_fn)

    assert result.oof_proba.shape == (len(x), ensemble.N_CLASSES)
    assert not np.isnan(result.oof_proba).any()  # every row written
    # Every row predicted exactly once across folds (the partition is exhaustive).
    assert sorted(seen_predict_rows) == list(range(len(x)))
    assert len(result.fold_scores) == ensemble.N_SPLITS


def test_compute_oof_writes_each_row_once(synth) -> None:
    """Each OOF row is filled by its own fold; folds do not overwrite each other.

    The spy stamps each fold's val rows with a fold-specific probability vector;
    after the loop, every row must carry the stamp of the fold whose val contains
    it (so no fold clobbered another's rows).
    """
    x, y, folds = synth

    def stamping_predict_fn(x_tr, y_tr, w_tr, x_pred):
        # Stamp column 1 with a value unique per call (monotonic across folds).
        proba = np.zeros((len(x_pred), ensemble.N_CLASSES))
        stamp = stamping_predict_fn.calls  # type: ignore[attr-defined]
        proba[:, 1] = 1.0
        proba[:, 0] = stamp  # carries the fold order; argmax still col 1 so scores valid
        stamping_predict_fn.calls += 1  # type: ignore[attr-defined]
        return proba

    stamping_predict_fn.calls = 0  # type: ignore[attr-defined]
    result = ensemble.compute_oof(x, y, folds, "stamp", stamping_predict_fn)

    # Fold k's val rows must carry stamp k in column 0.
    for k in range(ensemble.N_SPLITS):
        va = folds[f"fold{k}_val"]
        assert np.allclose(result.oof_proba[va, 0], float(k)), f"fold {k} rows not stamped by fold {k}"


# --------------------------------------------------------------------------- #
# Blend arithmetic.
# --------------------------------------------------------------------------- #
def test_blend_equal_is_arithmetic_mean() -> None:
    """Equal-weight blend equals the elementwise mean of the input matrices."""
    a = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    b = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])
    c = np.array([[0.1, 0.1, 0.8], [0.3, 0.3, 0.4]])
    blended = ensemble.blend_probabilities([a, b, c])
    expected = (a + b + c) / 3.0
    # Inputs already sum to 1 per row, so the mean does too; no renormalisation drift.
    assert np.allclose(blended, expected)
    assert np.allclose(blended.sum(axis=1), 1.0)


def test_blend_rows_sum_to_one_after_renormalisation() -> None:
    """Even with slightly-off input rows, blended rows are renormalised to sum 1."""
    a = np.array([[0.6, 0.3, 0.11]])  # sums to 1.01
    b = np.array([[0.2, 0.5, 0.29]])  # sums to 0.99
    blended = ensemble.blend_probabilities([a, b])
    assert np.allclose(blended.sum(axis=1), 1.0)


def test_blend_weighted_is_convex_combination() -> None:
    """Weighted blend is the normalised weighted mean of the matrices."""
    a = np.array([[0.8, 0.1, 0.1]])
    b = np.array([[0.0, 0.5, 0.5]])
    blended = ensemble.blend_probabilities([a, b], weights=[3.0, 1.0])
    expected = (3.0 * a + 1.0 * b) / 4.0
    assert np.allclose(blended, expected)


def test_blend_argmax_changes_with_evidence() -> None:
    """Blending can flip the decision when models disagree — the point of voting."""
    # Model A says class 0, model B says class 1 strongly; blend should pick 1.
    a = np.array([[0.51, 0.49, 0.0]])
    b = np.array([[0.10, 0.90, 0.0]])
    blended = ensemble.blend_probabilities([a, b])
    assert blended.argmax(axis=1)[0] == 1


def test_blend_rejects_shape_mismatch() -> None:
    """Mismatched probability matrices are a programming error, not silent."""
    a = np.zeros((3, 3))
    b = np.zeros((2, 3))
    with pytest.raises(ValueError, match="share shape"):
        ensemble.blend_probabilities([a, b])


def test_blend_rejects_bad_weights() -> None:
    """Wrong-length, negative, or all-zero weights raise rather than mislead."""
    a = np.zeros((2, 3))
    b = np.zeros((2, 3))
    with pytest.raises(ValueError, match="weights for"):
        ensemble.blend_probabilities([a, b], weights=[1.0])
    with pytest.raises(ValueError, match="non-negative"):
        ensemble.blend_probabilities([a, b], weights=[1.0, -1.0])
    with pytest.raises(ValueError, match="sum to zero"):
        ensemble.blend_probabilities([a, b], weights=[0.0, 0.0])


def test_blend_rejects_empty() -> None:
    """Blending zero matrices is a contract violation."""
    with pytest.raises(ValueError, match="at least one"):
        ensemble.blend_probabilities([])


# --------------------------------------------------------------------------- #
# Per-fold scoring pairs with the reference.
# --------------------------------------------------------------------------- #
def test_score_blend_per_fold_matches_manual(synth) -> None:
    """Per-fold blend scores equal balanced_accuracy on each fold's val slice."""
    from sklearn.metrics import balanced_accuracy_score

    x, y, folds = synth
    # A blended OOF where argmax == true label everywhere => perfect per-fold score.
    perfect = np.zeros((len(x), ensemble.N_CLASSES))
    perfect[np.arange(len(x)), y] = 1.0
    scores = ensemble.score_blend_per_fold(perfect, y, folds)
    assert len(scores) == ensemble.N_SPLITS
    for k in range(ensemble.N_SPLITS):
        va = folds[f"fold{k}_val"]
        manual = balanced_accuracy_score(y[va], perfect[va].argmax(axis=1))
        assert scores[k] == pytest.approx(manual)
    assert all(s == pytest.approx(1.0) for s in scores)


# --------------------------------------------------------------------------- #
# Paired-delta statistics + fidelity guard.
# --------------------------------------------------------------------------- #
def test_paired_delta_matches_definition() -> None:
    """diff_mean/diff_sem follow the paired-difference definition (ddof=1 SEM)."""
    cand = [0.97, 0.96, 0.98, 0.965, 0.972]
    ref = [0.965, 0.964, 0.964, 0.964, 0.964]
    d = ensemble.paired_delta(cand, ref)
    diffs = np.array(cand) - np.array(ref)
    assert d.diff_mean == pytest.approx(float(np.mean(diffs)))
    assert d.diff_sem == pytest.approx(float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))))
    assert d.defensible == bool(d.diff_mean > 2.0 * d.diff_sem)


def test_paired_delta_defensible_only_above_two_sigma() -> None:
    """A delta within 2 SEM is NOT defensible; a large consistent delta is."""
    ref = [0.96, 0.96, 0.96, 0.96, 0.96]
    # Tiny noisy improvement: not defensible.
    noisy = [0.961, 0.959, 0.962, 0.958, 0.9605]
    assert ensemble.paired_delta(noisy, ref).defensible is False
    # Large consistent improvement: defensible.
    strong = [0.97, 0.971, 0.969, 0.972, 0.97]
    assert ensemble.paired_delta(strong, ref).defensible is True


def test_paired_delta_rejects_length_mismatch() -> None:
    """A fold-count mismatch is a setup error, raised loudly."""
    with pytest.raises(ValueError, match="fold count mismatch"):
        ensemble.paired_delta([0.9, 0.9], [0.9, 0.9, 0.9])


def test_sigma_k_matches_feature_eval() -> None:
    """The defensibility multiplier is the single shared 2-sigma rule across cycles."""
    assert ensemble._SIGMA_K == FE_SIGMA_K == 2.0


def test_fidelity_guard_passes_within_tolerance() -> None:
    """LightGBM scores matching the C5 reference within atol pass the guard."""
    ref = [0.9655, 0.9648, 0.9646, 0.9645, 0.9647]
    near = [r + 5e-6 for r in ref]  # within _FIDELITY_ATOL
    assert ensemble.check_lgbm_fidelity(near, ref) is True


def test_fidelity_guard_fires_on_mismatch() -> None:
    """A real LightGBM mismatch (different model) trips the guard loudly."""
    ref = [0.9655, 0.9648, 0.9646, 0.9645, 0.9647]
    wrong = [0.95, 0.95, 0.95, 0.95, 0.95]  # clearly a different model
    with pytest.raises(AssertionError, match="fidelity check FAILED"):
        ensemble.check_lgbm_fidelity(wrong, ref)


# --------------------------------------------------------------------------- #
# Feature frame + submission id<->class mapping.
# --------------------------------------------------------------------------- #
def test_build_feature_frame_columns_and_dtypes() -> None:
    """The design matrix has exactly the raw columns, with categoricals cast."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "alpha": [10.0, 20.0],
            "delta": [0.0, 1.0],
            "u": [18.0, 19.0],
            "g": [17.0, 18.0],
            "r": [16.0, 17.0],
            "i": [15.0, 16.0],
            "z": [14.0, 15.0],
            "redshift": [0.1, 0.5],
            "spectral_type": ["A/F", "M"],
            "galaxy_population": ["Blue_Cloud", "Red_Sequence"],
            "class": ["GALAXY", "STAR"],
        }
    )
    x, categorical = ensemble.build_feature_frame(df)
    assert list(x.columns) == list(ensemble._RAW_FEATURES)
    assert categorical == list(ensemble._RAW_CATEGORICAL)
    for col in categorical:
        assert isinstance(x[col].dtype, pd.CategoricalDtype)
    # id and class are NOT model columns.
    assert "id" not in x.columns
    assert "class" not in x.columns


def test_submission_maps_id_to_decoded_blend_argmax(monkeypatch, tmp_path) -> None:
    """build_submission writes id->class where class = decoded argmax of the blend.

    We stub the three fit_predict functions to return fixed test probabilities and
    stub load_test to a tiny frame, so the test verifies the WIRING (id alignment,
    blend, decode, header, order) without any real model fit. The encoder is fitted
    on a known class order so we can assert the exact decoded labels.
    """
    from sklearn.preprocessing import LabelEncoder

    # Test frame: 3 rows with known ids.
    df_test = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "alpha": [10.0, 20.0, 30.0],
            "delta": [0.0, 1.0, 2.0],
            "u": [18.0, 19.0, 20.0],
            "g": [17.0, 18.0, 19.0],
            "r": [16.0, 17.0, 18.0],
            "i": [15.0, 16.0, 17.0],
            "z": [14.0, 15.0, 16.0],
            "redshift": [0.1, 0.5, 1.0],
            "spectral_type": ["A/F", "M", "G/K"],
            "galaxy_population": ["Blue_Cloud", "Red_Sequence", "Blue_Cloud"],
        }
    )
    monkeypatch.setattr(ensemble, "load_test", lambda: df_test)

    # Encoder fitted on the canonical class order (alphabetical => GALAXY,QSO,STAR).
    encoder = LabelEncoder()
    encoder.fit(np.array(["GALAXY", "QSO", "STAR"]))

    # Each base model returns the SAME fixed proba so the blend equals it: row 0 ->
    # class 0 (GALAXY), row 1 -> class 2 (STAR), row 2 -> class 1 (QSO).
    fixed = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.2, 0.7, 0.1]])
    monkeypatch.setattr(ensemble, "fit_predict_lgbm", lambda *a, **k: fixed.copy())
    monkeypatch.setattr(ensemble, "fit_predict_catboost", lambda *a, **k: fixed.copy())
    monkeypatch.setattr(ensemble, "fit_predict_xgboost", lambda *a, **k: fixed.copy())
    # Redirect the submission output into tmp_path.
    out_path = tmp_path / "submission.csv"
    monkeypatch.setattr(ensemble, "SUBMISSION_PATH", out_path)
    monkeypatch.setattr(ensemble, "DATA_DIR_OUT", tmp_path)

    df_train = df_test.assign(class_=["GALAXY", "STAR", "QSO"]).rename(columns={"class_": "class"})
    y = encoder.transform(df_train["class"].to_numpy())

    path = ensemble.build_submission(df_train, y, encoder)
    written = pd.read_csv(path)

    assert list(written.columns) == [ensemble.ID_COLUMN, ensemble.TARGET]
    assert written[ensemble.ID_COLUMN].tolist() == [101, 102, 103]
    # Decoded argmax of `fixed`: [GALAXY, STAR, QSO].
    assert written[ensemble.TARGET].tolist() == ["GALAXY", "STAR", "QSO"]


def test_submission_only_one_class_per_row_no_nan(monkeypatch, tmp_path) -> None:
    """Every submission row has exactly one hard label from the allowed class set."""
    from sklearn.preprocessing import LabelEncoder

    df_test = pd.DataFrame(
        {
            "id": [1, 2],
            "alpha": [10.0, 20.0],
            "delta": [0.0, 1.0],
            "u": [18.0, 19.0],
            "g": [17.0, 18.0],
            "r": [16.0, 17.0],
            "i": [15.0, 16.0],
            "z": [14.0, 15.0],
            "redshift": [0.1, 0.5],
            "spectral_type": ["A/F", "M"],
            "galaxy_population": ["Blue_Cloud", "Red_Sequence"],
        }
    )
    monkeypatch.setattr(ensemble, "load_test", lambda: df_test)
    encoder = LabelEncoder()
    encoder.fit(np.array(["GALAXY", "QSO", "STAR"]))
    fixed = np.array([[0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
    monkeypatch.setattr(ensemble, "fit_predict_lgbm", lambda *a, **k: fixed.copy())
    monkeypatch.setattr(ensemble, "fit_predict_catboost", lambda *a, **k: fixed.copy())
    monkeypatch.setattr(ensemble, "fit_predict_xgboost", lambda *a, **k: fixed.copy())
    monkeypatch.setattr(ensemble, "SUBMISSION_PATH", tmp_path / "s.csv")
    monkeypatch.setattr(ensemble, "DATA_DIR_OUT", tmp_path)

    df_train = df_test.assign(**{"class": ["GALAXY", "STAR"]})
    y = encoder.transform(df_train["class"].to_numpy())
    written = pd.read_csv(ensemble.build_submission(df_train, y, encoder))

    assert written[ensemble.TARGET].notna().all()
    assert set(written[ensemble.TARGET]).issubset({"GALAXY", "QSO", "STAR"})


# --------------------------------------------------------------------------- #
# Nested-CV blend-weight optimisation (the leakage guard required by the statistical review).
# --------------------------------------------------------------------------- #
def test_simplex_grid_is_valid_simplex() -> None:
    """Every grid vector is non-negative and sums to 1; the count is the simplex size."""
    grid = ensemble._simplex_grid(3, n_steps=12)
    # C(12+2, 2) = 91 lattice points on the 2-simplex at n_steps=12.
    assert len(grid) == 91
    for w in grid:
        assert len(w) == 3
        assert all(v >= 0.0 for v in w)
        assert sum(w) == pytest.approx(1.0)
    # The equal-weight vector is an EXACT grid point (12 divisible by 3), and the
    # three corners are present.
    assert any(np.allclose(w, [1 / 3, 1 / 3, 1 / 3]) for w in grid)
    assert (1.0, 0.0, 0.0) in [tuple(round(v, 10) for v in w) for w in grid]


def test_best_weights_concentrate_on_the_strong_model(synth) -> None:
    """Weight selection finds an inner-optimal vector that dominates toward the best model.

    Model 0 is always correct; models 1 and 2 each point to a DIFFERENT wrong class.
    Many weight vectors are perfect on the inner rows (any with w0 strictly the
    largest), so the tie-break returns the inner-optimal point NEAREST to equal — not
    necessarily (1,0,0). The contract we assert is the one that matters: the selected
    weights make model 0 dominant (w0 > w1 and w0 > w2), and the inner blend scores
    perfectly under them.
    """
    from sklearn.metrics import balanced_accuracy_score

    x, y, folds = synth
    n = len(x)
    perfect = np.zeros((n, ensemble.N_CLASSES))
    perfect[np.arange(n), y] = 1.0
    wrong_a = np.zeros((n, ensemble.N_CLASSES))
    wrong_a[np.arange(n), (y + 1) % ensemble.N_CLASSES] = 1.0
    wrong_b = np.zeros((n, ensemble.N_CLASSES))
    wrong_b[np.arange(n), (y + 2) % ensemble.N_CLASSES] = 1.0
    mats = [perfect, wrong_a, wrong_b]

    grid = ensemble._simplex_grid(3)
    w = ensemble._best_weights_on_folds(mats, y, folds, [0, 1], grid)
    assert w[0] > w[1] and w[0] > w[2]  # strong model dominates
    # Under the selected weights the inner blend is perfect (the inner objective).
    inner = np.concatenate([folds["fold0_val"], folds["fold1_val"]])
    blended = ensemble.blend_probabilities([m[inner] for m in mats], weights=list(w))
    assert balanced_accuracy_score(y[inner], blended.argmax(axis=1)) == pytest.approx(1.0)


def test_nested_cv_weights_held_out_scoring(synth) -> None:
    """Nested CV reports held-out scores; per-fold weights were chosen without that fold.

    One always-correct model + two distinctly-wrong models: every fold's inner
    objective selects a model-0-dominant vector, and the held-out score is perfect
    because model 0 is perfect everywhere. Asserts the structure (5 scores, 5 weight
    tuples) and that the strong model dominates each selected vector.
    """
    x, y, folds = synth
    n = len(x)
    perfect = np.zeros((n, ensemble.N_CLASSES))
    perfect[np.arange(n), y] = 1.0
    wrong_a = np.zeros((n, ensemble.N_CLASSES))
    wrong_a[np.arange(n), (y + 1) % ensemble.N_CLASSES] = 1.0
    wrong_b = np.zeros((n, ensemble.N_CLASSES))
    wrong_b[np.arange(n), (y + 2) % ensemble.N_CLASSES] = 1.0

    result = ensemble.nested_cv_weight_blend([perfect, wrong_a, wrong_b], y, folds)
    assert len(result.fold_scores) == ensemble.N_SPLITS
    assert len(result.per_fold_weights) == ensemble.N_SPLITS
    for w in result.per_fold_weights:
        assert w[0] > w[1] and w[0] > w[2]  # strong model dominates each fold
    assert all(s == pytest.approx(1.0) for s in result.fold_scores)


def test_best_weights_ignore_held_out_fold(synth) -> None:
    """The selector's choice depends ONLY on the train folds, never the held-out one.

    This is the no-leakage guard required by the statistical review. We build OOF where model 1
    is the best on the INNER rows (folds 0,1) but model 2 is the best on the held-out
    fold (fold 2). A leakage-free selector, given train folds [0,1], must pick the
    model-1-dominant vector and be INDIFFERENT to fold 2's contents — so mutating
    fold 2's rows leaves the selected weights unchanged.
    """
    x, y, folds = synth
    n = len(x)
    inner = np.concatenate([folds["fold0_val"], folds["fold1_val"]])
    held = folds["fold2_val"]

    # Model 1 correct on inner, wrong on held; model 2 the reverse; model 0 wrong.
    m0 = np.zeros((n, ensemble.N_CLASSES)); m0[np.arange(n), (y + 1) % 3] = 1.0
    m1 = np.zeros((n, ensemble.N_CLASSES))
    m1[inner, y[inner]] = 1.0; m1[held, (y[held] + 1) % 3] = 1.0
    m2 = np.zeros((n, ensemble.N_CLASSES))
    m2[inner, (y[inner] + 1) % 3] = 1.0; m2[held, y[held]] = 1.0
    mats = [m0, m1, m2]

    grid = ensemble._simplex_grid(3)
    w_before = ensemble._best_weights_on_folds(mats, y, folds, [0, 1], grid)
    # Model 1 must dominate (it is the one correct on the inner rows).
    assert w_before[1] > w_before[0] and w_before[1] > w_before[2]

    # Now corrupt ONLY the held-out fold's rows for every model. If the selector
    # leaked fold 2, its choice would move; it must not.
    for m in mats:
        m[held] = 0.0
        m[held, (y[held] + 2) % 3] = 1.0
    w_after = ensemble._best_weights_on_folds(mats, y, folds, [0, 1], grid)
    assert w_after == w_before  # held-out contents are irrelevant to the inner choice


def test_nested_cv_ties_prefer_equal_weights(synth) -> None:
    """On a flat objective (all models identical), the selector prefers equal weights.

    If every model has the same OOF, every weight vector scores identically; the
    tie-break must return the equal-weight vector (the simplest), not an arbitrary
    corner. Equal weights are an exact grid point (n_steps=12 divisible by 3). This
    guards against silently shipping an extreme weight on noise.
    """
    x, y, folds = synth
    n = len(x)
    same = np.zeros((n, ensemble.N_CLASSES))
    same[np.arange(n), y] = 1.0
    grid = ensemble._simplex_grid(3)
    w = ensemble._best_weights_on_folds([same, same.copy(), same.copy()], y, folds, [0, 1], grid)
    assert np.allclose(w, [1 / 3, 1 / 3, 1 / 3])


# --------------------------------------------------------------------------- #
# Tuned-param loaders read the C6 study JSONs (single source of truth).
# --------------------------------------------------------------------------- #
def test_catboost_params_reads_tuning_json(monkeypatch, tmp_path) -> None:
    """catboost_params merges the tuned best_params + fixed iterations from JSON."""
    record = {
        "best_params": {"depth": 8, "learning_rate": 0.07, "l2_leaf_reg": 4.2},
        "best_iterations_fixed": 850,
    }
    path = tmp_path / "catboost_tuning.json"
    path.write_text(__import__("json").dumps(record), encoding="utf-8")
    monkeypatch.setattr(ensemble, "CATBOOST_TUNING_PATH", path)

    params = ensemble.catboost_params()
    assert params["depth"] == 8
    assert params["learning_rate"] == 0.07
    assert params["l2_leaf_reg"] == 4.2
    assert params["iterations"] == 850
    # The fixed GPU regime is still present.
    assert params["task_type"] == "GPU"
    assert params["loss_function"] == "MultiClass"


def test_catboost_params_falls_back_to_defaults(monkeypatch, tmp_path) -> None:
    """With no tuning JSON, catboost_params returns the sensible defaults."""
    monkeypatch.setattr(ensemble, "CATBOOST_TUNING_PATH", tmp_path / "absent.json")
    params = ensemble.catboost_params()
    assert params["iterations"] == 600
    assert params["depth"] == 6
    assert params["task_type"] == "GPU"


def test_xgboost_params_reads_tuning_json(monkeypatch, tmp_path) -> None:
    """xgboost_params merges the tuned best_params + fixed n_estimators from JSON."""
    record = {
        "best_params": {"max_depth": 9, "learning_rate": 0.06, "subsample": 0.8},
        "best_n_estimators_fixed": 720,
    }
    path = tmp_path / "xgboost_tuning.json"
    path.write_text(__import__("json").dumps(record), encoding="utf-8")
    monkeypatch.setattr(ensemble, "XGBOOST_TUNING_PATH", path)

    params = ensemble.xgboost_params()
    assert params["max_depth"] == 9
    assert params["learning_rate"] == 0.06
    assert params["subsample"] == 0.8
    assert params["n_estimators"] == 720
    assert params["enable_categorical"] is True
    assert params["objective"] == "multi:softprob"


def test_xgboost_params_falls_back_to_defaults(monkeypatch, tmp_path) -> None:
    """With no tuning JSON, xgboost_params returns the sensible defaults."""
    monkeypatch.setattr(ensemble, "XGBOOST_TUNING_PATH", tmp_path / "absent.json")
    params = ensemble.xgboost_params()
    assert params["n_estimators"] == 600
    assert params["max_depth"] == 6
    assert params["enable_categorical"] is True
