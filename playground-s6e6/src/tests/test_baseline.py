"""Tests for the formal baseline module's pure logic (BL-008, src.baseline).

Scope: the verification gate and the provenance hashing — the parts that decide
whether the C5 comparison is valid — without running the real multi-minute CV.
The actual 0.96448 reproduction is evidence in docs/c5-poc/baseline_results.json
(written by running the module), not a unit test.

We stub feature_eval.evaluate_config and load_train/load_folds so run_baseline
exercises ONLY its own logic: the drift computation, the gate decision, and the
summary assembly. MLflow is not installed-required here; _try_log_mlflow degrades
to None and we tolerate that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import baseline
from src.feature_eval import ConfigResult


def _fake_result(mean: float) -> ConfigResult:
    """A ConfigResult whose fold scores average to `mean` (for gate tests)."""
    scores = [mean] * 5
    return ConfigResult(
        name="baseline_raw",
        features=["alpha", "delta", "u", "g", "r", "i", "z", "redshift"],
        categorical=[],
        fold_scores=scores,
        mean=mean,
        std=0.0,
    )


@pytest.fixture
def stub_cv(monkeypatch, tmp_path):
    """Stub the heavy pieces so run_baseline runs in milliseconds.

    Patches load_train (a 3-row frame), the LabelEncoder target, load_folds, the
    sha256 (no real CSV), and routes outputs to a temp dir. Returns a setter that
    fixes the mean the stubbed evaluate_config will report.
    """
    df = pd.DataFrame(
        {"id": [1, 2, 3], "class": ["GALAXY", "STAR", "QSO"]}
    )
    monkeypatch.setattr(baseline, "load_train", lambda: df)
    monkeypatch.setattr(baseline, "load_folds", lambda: {})
    monkeypatch.setattr(baseline, "_sha256", lambda p: "deadbeef")
    monkeypatch.setattr(baseline, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(baseline, "RESULTS_PATH", tmp_path / "baseline_results.json")
    # Disable MLflow side effects regardless of install state.
    monkeypatch.setattr(baseline, "_try_log_mlflow", lambda summary: None)

    def set_mean(mean: float) -> None:
        monkeypatch.setattr(
            baseline, "evaluate_config", lambda df, y, folds, cfg: _fake_result(mean)
        )

    return set_mean


def test_gate_passes_within_tolerance(stub_cv) -> None:
    """A mean within _BASELINE_TOL of the canonical value passes the gate."""
    stub_cv(baseline.CANONICAL_BALANCED_ACCURACY + 5e-5)
    summary = baseline.run_baseline(enforce_gate=True)
    assert summary["gate_passed"] is True
    assert summary["drift_vs_canonical"] <= baseline._BASELINE_TOL


def test_gate_raises_on_drift(stub_cv) -> None:
    """A mean far from the canonical value raises (drift caught loudly)."""
    stub_cv(baseline.CANONICAL_BALANCED_ACCURACY - 0.01)
    with pytest.raises(RuntimeError, match="Baseline drift"):
        baseline.run_baseline(enforce_gate=True)


def test_no_gate_allows_drift_for_diagnostics(stub_cv) -> None:
    """enforce_gate=False records the drift but does not raise."""
    stub_cv(baseline.CANONICAL_BALANCED_ACCURACY - 0.01)
    summary = baseline.run_baseline(enforce_gate=False)
    assert summary["gate_passed"] is False
    assert summary["drift_vs_canonical"] == pytest.approx(0.01, abs=1e-6)


def test_summary_records_provenance_and_params(stub_cv) -> None:
    """The summary carries the params, provenance hashes and library versions."""
    stub_cv(baseline.CANONICAL_BALANCED_ACCURACY)
    summary = baseline.run_baseline(enforce_gate=True)
    assert summary["lgbm_params"]["objective"] == "multiclass"
    prov = summary["provenance"]
    assert prov["folds_sha256"] == "deadbeef"
    assert "lightgbm_version" in prov
    assert summary["class_order"] == ["GALAXY", "QSO", "STAR"]


def test_sha256_missing_file_returns_marker(tmp_path) -> None:
    """A missing file hashes to the 'missing' marker, not a crash."""
    assert baseline._sha256(tmp_path / "absent.csv") == "missing"


def test_sha256_hashes_real_file(tmp_path) -> None:
    """A present file returns a 64-hex-char digest."""
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    digest = baseline._sha256(f)
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
