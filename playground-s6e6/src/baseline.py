"""Formal LightGBM baseline with structured MLflow tracking (BL-008).

Cycle C5 (POC). This is the CANONICAL comparison point for the whole cycle: the
frozen baseline config evaluated on the frozen 5 folds, expected to reproduce
0.96448 +/- 0.00029 balanced_accuracy. The C5 tuner (BL-011) must beat THIS, by
a statistically defensible margin, on the IDENTICAL folds.

Design — why this is a thin wrapper, not a new CV loop:
  The fair-comparison machinery (frozen folds, balanced weights, native
  categorical handling, determinism flags, the LightGBM hyperparameters) already
  lives in src.feature_eval and is PROVEN to reproduce the C2 baseline to five
  decimals. Re-implementing it here would risk a subtle divergence that breaks
  the very comparability the baseline exists to provide. So baseline.py REUSES
  feature_eval.evaluate_config / build_model_frame / _lgbm_params verbatim and
  adds only two things on top:
    1. A verification gate: assert the reproduced mean is within tolerance of the
       canonical 0.96448, so a silent drift (dependency bump, accidental config
       edit) fails loudly instead of poisoning the C5 comparison.
    2. Structured MLflow tracking: a single run logging the exact params, the
       per-fold and aggregate balanced_accuracy, the dataset/folds provenance
       hashes, and this module's source as an artifact — the production-readiness
       requirement that the experiment be reproducible from the tracking record.

MLflow is optional in the lockfile (extra 'tracking'). If absent we still run the
CV and write the JSON record + enforce the gate; we only skip the MLflow logging.
The number is the contract, MLflow is the audit trail.

Train-only: never imports load_test. The baseline is a CV estimate; the
submission model (trained on all of train) is built by the tuner once the best
config is chosen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Final

import lightgbm as lgb
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

from src.cv_utils import FOLDS_PATH, load_folds
from src.data_loader import DATA_DIR, TARGET, load_train
from src.feature_eval import FeatureConfig, SEED, _lgbm_params, evaluate_config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Canonical baseline contract.
# --------------------------------------------------------------------------- #
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
RESULTS_DIR: Final[Path] = _REPO_ROOT / "docs" / "c5-poc"
RESULTS_PATH: Final[Path] = RESULTS_DIR / "baseline_results.json"
MLFLOW_DIR: Final[Path] = _REPO_ROOT / "mlruns"
MLFLOW_EXPERIMENT: Final[str] = "s6e6-c5-baseline"

# The frozen C2 baseline number this module must reproduce. Sourced from the
# C3 feature_eval_results.json (baseline_raw config). We assert the reproduced
# mean is within _BASELINE_TOL of it; the tolerance is generous relative to the
# reported +/-0.00029 std because the gate guards against a CONFIG/DEPENDENCY
# drift (which would move the mean by far more), not against fold jitter (there
# is none — same folds, deterministic fit).
CANONICAL_BALANCED_ACCURACY: Final[float] = 0.96448
_BASELINE_TOL: Final[float] = 1e-4


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file (dataset/folds provenance).

    Args:
        path: File to hash.

    Returns:
        Hex digest string, or "missing" if the file is absent (the data CSVs are
        gitignored and may not exist in a fresh worktree; we still record the
        attempt rather than crashing the tracking).
    """
    if not path.is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_baseline(*, enforce_gate: bool = True) -> dict[str, object]:
    """Reproduce the frozen baseline on the frozen folds and track it in MLflow.

    Steps:
      1. Load train, label-encode the target.
      2. Load the frozen folds (BL-005).
      3. Evaluate the baseline_raw config (raw features only) with the canonical
         deterministic LightGBM params via feature_eval.evaluate_config — the
         exact engine that produced the C2/C3 number.
      4. Verify mean balanced_accuracy is within tolerance of the canonical
         0.96448 (the gate). On failure, raise so the drift is caught here.
      5. Persist docs/c5-poc/baseline_results.json and, if MLflow is available, a
         structured run (params, per-fold + aggregate metrics, provenance, source
         artifact).

    Args:
        enforce_gate: If True (default), raise RuntimeError when the reproduced
            mean drifts from CANONICAL_BALANCED_ACCURACY by more than
            _BASELINE_TOL. Set False only for diagnostic runs.

    Returns:
        JSON-serialisable summary dict (also written to RESULTS_PATH).

    Raises:
        RuntimeError: If enforce_gate and the reproduced mean drifts beyond
            tolerance from the canonical baseline.
    """
    started = time.time()
    df = load_train()
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[TARGET].to_numpy())
    folds = load_folds()

    # The baseline IS the raw-feature config with the deterministic params — the
    # same object feature_eval evaluates as "baseline_raw".
    result = evaluate_config(df, y, folds, FeatureConfig(name="baseline_raw"))

    drift = abs(result.mean - CANONICAL_BALANCED_ACCURACY)
    gate_passed = drift <= _BASELINE_TOL
    logger.info(
        "baseline mean=%.6f std=%.6f drift_vs_canonical=%.6f gate_passed=%s",
        result.mean,
        result.std,
        drift,
        gate_passed,
    )
    if enforce_gate and not gate_passed:
        raise RuntimeError(
            f"Baseline drift: reproduced mean {result.mean:.6f} differs from "
            f"canonical {CANONICAL_BALANCED_ACCURACY} by {drift:.6f} "
            f"(> tolerance {_BASELINE_TOL}). The C5 comparison is invalid until "
            "this is resolved (check LightGBM version / config edits / folds)."
        )

    params = _lgbm_params()
    provenance = {
        "train_csv_sha256": _sha256(DATA_DIR / "train.csv"),
        "folds_sha256": _sha256(FOLDS_PATH),
        "lightgbm_version": lgb.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    summary: dict[str, object] = {
        "name": "baseline_lgbm_raw",
        "seed": SEED,
        "metric": "balanced_accuracy",
        "canonical_balanced_accuracy": CANONICAL_BALANCED_ACCURACY,
        "baseline_tolerance": _BASELINE_TOL,
        "gate_passed": gate_passed,
        "drift_vs_canonical": drift,
        "fold_scores": result.fold_scores,
        "mean": result.mean,
        "std": result.std,
        "features": result.features,
        "categorical": result.categorical,
        "lgbm_params": params,
        "provenance": provenance,
        "class_order": list(encoder.classes_),
        "elapsed_seconds": round(time.time() - started, 1),
    }

    summary["mlflow_run_id"] = _try_log_mlflow(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", RESULTS_PATH)
    return summary


def _try_log_mlflow(summary: dict[str, object]) -> str | None:
    """Best-effort structured MLflow logging; return the run id or None.

    Logs one run to the local file store under mlruns/: all LightGBM params
    (prefixed), the dataset/library provenance, the per-fold balanced_accuracy
    (metric step = fold index), the aggregate mean/std, and this module's source
    as an artifact so the run is self-describing. MLflow is optional (not in the
    base lockfile); if absent or erroring we log a warning and continue — the
    JSON record remains the contract.

    Args:
        summary: The baseline summary dict (already computed).

    Returns:
        The MLflow run id string, or None if logging was skipped.
    """
    # MLflow 3.x raises on the local file store unless this opt-out is set. We use
    # a local mlruns/ dir by design (single-host POC, no tracking server), so we
    # set it here rather than asking the caller to export it. See the MLflow 3
    # filesystem-backend deprecation note.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow  # noqa: PLC0415  (intentional optional import)
    except ImportError:
        logger.warning("mlflow not installed; skipping tracking (JSON is the record)")
        return None

    try:
        mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name="baseline_lgbm_raw") as run:
            params = summary["lgbm_params"]
            # Explicit guard (not assert): assert is stripped under `python -O`,
            # which would turn a malformed summary into a confusing AttributeError
            # deeper in the dict-comprehension instead of this clear message.
            if not isinstance(params, dict):
                raise TypeError(f"summary['lgbm_params'] must be a dict, got {type(params)}")
            mlflow.log_params({f"lgbm_{k}": v for k, v in params.items()})
            mlflow.log_params(
                {
                    "features": ",".join(summary["features"]),  # type: ignore[arg-type]
                    "n_features": len(summary["features"]),  # type: ignore[arg-type]
                    "categorical": ",".join(summary["categorical"]),  # type: ignore[arg-type]
                    "seed": summary["seed"],
                    "metric": summary["metric"],
                }
            )
            mlflow.set_tags(summary["provenance"])  # type: ignore[arg-type]
            mlflow.log_metric("balanced_accuracy_mean", summary["mean"])  # type: ignore[arg-type]
            mlflow.log_metric("balanced_accuracy_std", summary["std"])  # type: ignore[arg-type]
            mlflow.log_metric("drift_vs_canonical", summary["drift_vs_canonical"])  # type: ignore[arg-type]
            mlflow.log_metric("gate_passed", float(bool(summary["gate_passed"])))
            for k, score in enumerate(summary["fold_scores"]):  # type: ignore[arg-type]
                mlflow.log_metric("balanced_accuracy_fold", score, step=k)
            mlflow.log_artifact(str(Path(__file__).resolve()))
            run_id = run.info.run_id
        logger.info("MLflow baseline run %s under %s", run_id, MLFLOW_DIR)
        return run_id
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("MLflow logging failed (%s); continuing with JSON record", exc)
        return None


def main() -> None:
    """CLI entry point: run the baseline, print the headline, enforce the gate."""
    parser = argparse.ArgumentParser(description="Formal LightGBM baseline (BL-008).")
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Diagnostic mode: do not raise if the reproduced mean drifts.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    np.random.seed(SEED)
    summary = run_baseline(enforce_gate=not args.no_gate)
    folds_str = " ".join(f"{s:.6f}" for s in summary["fold_scores"])  # type: ignore[arg-type]
    print(
        f"baseline_lgbm_raw  mean={summary['mean']:.6f} std={summary['std']:.6f} "
        f"gate_passed={summary['gate_passed']}\n  folds: {folds_str}"
    )


if __name__ == "__main__":
    main()
