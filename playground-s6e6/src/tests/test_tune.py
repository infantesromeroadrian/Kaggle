"""Tests for the Optuna HPO module's pure logic (BL-011, src.tune).

Scope: the parts that do NOT need the real 577k-row CSVs or a multi-minute CV
run. The heavy CV / study orchestration is exercised by running the script
(docs/c5-poc/tuning_results.json is its evidence), not by unit tests.

What we assert here:
  - suggest_params samples EVERY documented hyperparameter, within the declared
    bounds, for a representative sweep of trials (using a real Optuna study with a
    fixed seed so the sweep is deterministic). This is the contract the statistical
    review cares about: the search space matches the BL-011 spec and the baseline is
    reachable (num_leaves can be 63-ish, lr can be 0.05-ish, etc.).
  - The defensibility threshold constant matches feature_eval's, so the report
    and the gate cite ONE rule.
"""

from __future__ import annotations

import optuna
import pytest

from src import tune
from src.feature_eval import _SIGMA_K as FE_SIGMA_K


def _sample_many(n: int = 64) -> list[dict[str, object]]:
    """Draw n parameter dicts from a seeded TPE study (deterministic sweep).

    We run a trivial study whose objective just returns 0.0; we only care about
    the SAMPLED params, not the optimisation. Seeding keeps the sweep stable.
    """
    sampled: list[dict[str, object]] = []

    def objective(trial: optuna.Trial) -> float:
        sampled.append(tune.suggest_params(trial))
        return 0.0

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n, show_progress_bar=False)
    return sampled


def test_suggest_params_has_every_spec_key() -> None:
    """Every BL-011 hyperparameter is present in a sampled config."""
    params = _sample_many(8)[0]
    expected = {
        "num_leaves",
        "max_depth",
        "learning_rate",
        "min_child_samples",
        "bagging_fraction",
        "bagging_freq",
        "feature_fraction",
        "reg_alpha",
        "reg_lambda",
        "min_split_gain",
    }
    assert expected <= set(params)


def test_suggest_params_within_declared_bounds() -> None:
    """Across a 64-trial sweep, every sample respects its declared range."""
    for p in _sample_many(64):
        assert 15 <= p["num_leaves"] <= 255
        assert p["max_depth"] in {-1, 4, 6, 8, 10, 12, 16}
        assert 0.03 <= p["learning_rate"] <= 0.2
        assert 5 <= p["min_child_samples"] <= 300
        assert 0.5 <= p["bagging_fraction"] <= 1.0
        assert p["bagging_freq"] == 1
        assert 0.5 <= p["feature_fraction"] <= 1.0
        assert 1e-3 <= p["reg_alpha"] <= 10.0
        assert 1e-3 <= p["reg_lambda"] <= 10.0
        assert 0.0 <= p["min_split_gain"] <= 0.5


def test_baseline_config_is_reachable() -> None:
    """The search space spans the baseline neighbourhood (so it cannot only do worse).

    The baseline is num_leaves 63, lr 0.05, depth -1. We check the sweep produces
    samples that bracket those values, i.e. the baseline region is not excluded by
    the bounds — a sanity check that we are tuning AROUND the baseline, not away.
    """
    sweep = _sample_many(64)
    leaves = [p["num_leaves"] for p in sweep]
    lrs = [p["learning_rate"] for p in sweep]
    depths = {p["max_depth"] for p in sweep}
    assert min(leaves) <= 63 <= max(leaves)
    assert min(lrs) <= 0.05 <= max(lrs)
    assert -1 in depths  # the baseline's unbounded depth is in the categorical set


def test_sigma_k_matches_feature_eval() -> None:
    """The defensibility multiplier is the single shared 2-sigma rule."""
    assert tune._SIGMA_K == FE_SIGMA_K == 2.0


def test_budget_defaults_are_sane() -> None:
    """Default budget matches the documented 40-min envelope and bounded cap."""
    assert tune.DEFAULT_N_TRIALS == 40
    assert tune.DEFAULT_TIMEOUT_S == 40 * 60
    assert tune._EARLY_STOPPING_ROUNDS == 30
    assert tune._MAX_ESTIMATORS == 400
    assert tune._SEARCH_THREADS == 4


# --------------------------------------------------------------------------- #
# deterministic_eval — the monkey-patch contract (code-critic regression guard)
# --------------------------------------------------------------------------- #
# deterministic_eval temporarily rebinds feature_eval._lgbm_params so the proven
# CV loop runs with the tuned params instead of the baseline ones. These tests
# pin two invariants that are otherwise only exercised by a multi-minute real run:
#   1. the params actually injected carry the best hyperparameters + the
#      determinism block + the fixed n_estimators (not the baseline params);
#   2. the global binding is ALWAYS restored afterwards, even if evaluate_config
#      raises — a leaked patch would silently corrupt every later baseline call.
def _spy_eval_capturing_injected_params(monkeypatch) -> dict[str, object]:
    """Stub evaluate_config so it records what _lgbm_params() yields while patched.

    Returns the dict the captured `_lgbm_params()` produced inside the patched
    region, so the caller can assert the injected params. Routes around the real
    462k-row CV entirely.
    """
    import src.feature_eval as fe

    captured: dict[str, object] = {}

    def fake_evaluate_config(df, y, folds, config):
        # While patched, fe._lgbm_params is the tuner's lambda: capture its output.
        captured.update(fe._lgbm_params())
        return fe.ConfigResult(
            name=config.name, features=[], categorical=[], fold_scores=[0.9] * 5
        )

    monkeypatch.setattr(tune, "evaluate_config", fake_evaluate_config)
    return captured


def test_deterministic_eval_injects_best_params_and_determinism(monkeypatch) -> None:
    """The patched _lgbm_params yields best params + determinism + fixed n_estimators."""
    captured = _spy_eval_capturing_injected_params(monkeypatch)
    best = {"num_leaves": 61, "learning_rate": 0.0784, "n_estimators": 399}

    tune.deterministic_eval(df=None, y=None, folds={}, best_params=best)

    # Tuned hyperparameters won, not the baseline 63 leaves / 0.05 lr / 300 trees.
    assert captured["num_leaves"] == 61
    assert captured["learning_rate"] == 0.0784
    assert captured["n_estimators"] == 399
    # Determinism block is pinned to the baseline regime (bit-for-bit comparable).
    assert captured["deterministic"] is True
    assert captured["force_row_wise"] is True
    assert captured["num_threads"] == 1
    assert captured["n_jobs"] == 1
    # Fixed model contract still present.
    assert captured["objective"] == "multiclass"
    assert captured["num_class"] == 3


def test_deterministic_eval_restores_global_binding(monkeypatch) -> None:
    """fe._lgbm_params is restored to the original after a successful eval."""
    import src.feature_eval as fe

    original = fe._lgbm_params
    _spy_eval_capturing_injected_params(monkeypatch)

    tune.deterministic_eval(df=None, y=None, folds={}, best_params={"n_estimators": 10})

    assert fe._lgbm_params is original


def test_deterministic_eval_restores_binding_on_exception(monkeypatch) -> None:
    """A raising evaluate_config must NOT leak the patched _lgbm_params globally."""
    import src.feature_eval as fe

    original = fe._lgbm_params

    def boom(df, y, folds, config):
        raise RuntimeError("CV blew up mid-eval")

    monkeypatch.setattr(tune, "evaluate_config", boom)

    with pytest.raises(RuntimeError, match="CV blew up"):
        tune.deterministic_eval(df=None, y=None, folds={}, best_params={"n_estimators": 10})

    # The finally-block must have reverted the global even though eval raised.
    assert fe._lgbm_params is original
