"""Tests for the C6 CatBoost / XGBoost Optuna search spaces (BL-025, BL-026).

Scope mirrors test_tune.py: the parts that do NOT need the real 577k-row CSVs or a
multi-minute GPU/CPU study. We assert the search-space CONTRACT the statistical review
cares about — every documented hyperparameter is sampled within its declared bounds over
a deterministic seeded sweep — and that the budget/threshold constants are sane and
shared. The heavy study itself is exercised by running the scripts (their evidence
is docs/c6-build/{catboost,xgboost}_tuning.json), not by unit tests.
"""

from __future__ import annotations

import optuna

from src import tune_catboost, tune_xgboost
from src.feature_eval import _SIGMA_K as FE_SIGMA_K


def _sample_many(suggest_fn, n: int = 64) -> list[dict[str, object]]:
    """Draw n parameter dicts from a seeded TPE study (deterministic sweep).

    The objective returns 0.0; we only care about the SAMPLED params. Seeding keeps
    the sweep stable so the bound assertions are reproducible.
    """
    sampled: list[dict[str, object]] = []

    def objective(trial: optuna.Trial) -> float:
        sampled.append(suggest_fn(trial))
        return 0.0

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n, show_progress_bar=False)
    return sampled


# --------------------------------------------------------------------------- #
# CatBoost (BL-025).
# --------------------------------------------------------------------------- #
def test_catboost_suggest_has_every_spec_key() -> None:
    """Every BL-025 CatBoost hyperparameter is present in a sampled config."""
    params = _sample_many(tune_catboost.suggest_params, 8)[0]
    expected = {"depth", "learning_rate", "l2_leaf_reg", "border_count", "random_strength"}
    assert expected <= set(params)


def test_catboost_suggest_within_bounds() -> None:
    """Across a 64-trial sweep, every CatBoost sample respects its declared range."""
    for p in _sample_many(tune_catboost.suggest_params, 64):
        assert 4 <= p["depth"] <= 10
        assert 0.02 <= p["learning_rate"] <= 0.2
        assert 1.0 <= p["l2_leaf_reg"] <= 10.0
        assert p["border_count"] in {32, 64, 128, 254}
        assert 0.0 <= p["random_strength"] <= 10.0


def test_catboost_budget_and_sigma() -> None:
    """CatBoost budget + the shared 2-sigma rule are as documented."""
    assert tune_catboost.DEFAULT_N_TRIALS == 30
    assert tune_catboost.DEFAULT_TIMEOUT_S == 22 * 60
    assert tune_catboost._MAX_ITERATIONS == 1000
    assert tune_catboost._EARLY_STOPPING_ROUNDS == 50
    assert tune_catboost._SIGMA_K == FE_SIGMA_K == 2.0
    # GPU regime is fixed.
    assert tune_catboost._FIXED_PARAMS["task_type"] == "GPU"
    assert tune_catboost._FIXED_PARAMS["loss_function"] == "MultiClass"


# --------------------------------------------------------------------------- #
# XGBoost (BL-026).
# --------------------------------------------------------------------------- #
def test_xgboost_suggest_has_every_spec_key() -> None:
    """Every BL-026 XGBoost hyperparameter is present in a sampled config."""
    params = _sample_many(tune_xgboost.suggest_params, 8)[0]
    expected = {
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "gamma",
    }
    assert expected <= set(params)


def test_xgboost_suggest_within_bounds() -> None:
    """Across a 64-trial sweep, every XGBoost sample respects its declared range."""
    for p in _sample_many(tune_xgboost.suggest_params, 64):
        assert 4 <= p["max_depth"] <= 10
        assert 0.02 <= p["learning_rate"] <= 0.2
        assert 1.0 <= p["min_child_weight"] <= 10.0
        assert 0.5 <= p["subsample"] <= 1.0
        assert 0.5 <= p["colsample_bytree"] <= 1.0
        assert 1e-3 <= p["reg_alpha"] <= 10.0
        assert 1e-3 <= p["reg_lambda"] <= 10.0
        assert 0.0 <= p["gamma"] <= 5.0


def test_xgboost_budget_is_reduced_for_cpu_cost() -> None:
    """XGBoost runs FEWER trials by design (CPU is ~10x CatBoost GPU per trial)."""
    assert tune_xgboost.DEFAULT_N_TRIALS == 15  # reduced vs CatBoost's 30
    assert tune_xgboost.DEFAULT_TIMEOUT_S == 35 * 60
    assert tune_xgboost._MAX_ESTIMATORS == 800
    assert tune_xgboost._EARLY_STOPPING_ROUNDS == 30
    assert tune_xgboost._CPU_THREADS == 4  # NOT -1 (over-subscription is catastrophic)
    assert tune_xgboost._SIGMA_K == FE_SIGMA_K == 2.0
