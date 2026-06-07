"""
Triagegeist — Training Pipeline

Multiclass classification (ESI 1-5) with:
  - LightGBM + XGBoost + CatBoost ensemble
  - Quadratic Weighted Kappa (QWK) as primary metric
  - Optimized class thresholds for ordinal nature
  - GPU acceleration

Metrics tracked: QWK, macro F1, per-class recall (especially ESI-1)

Run: uv run python src/train.py
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (
    DATA_DIR,
    MODELS_DIR,
    ASSETS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    ID_COL,
    TRAIN_FILE,
    TEST_FILE,
    ESI_LABELS,
)
from features import build_all, get_feature_cols

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

N_FOLDS = 10
ES_ROUNDS = 150


# ---------------------------------------------------------------------------
# GPU probes
# ---------------------------------------------------------------------------


def _probe_lgbm_gpu() -> bool:
    try:
        import lightgbm as lgb

        d = lgb.Dataset(
            np.random.rand(20, 4).astype(np.float32), label=np.random.randint(0, 5, 20), free_raw_data=False
        )
        lgb.train(
            {"device": "gpu", "objective": "multiclass", "num_class": 5, "verbose": -1, "metric": "multi_logloss"},
            d,
            num_boost_round=1,
        )
        return True
    except Exception:
        return False


def _probe_xgb_gpu() -> bool:
    try:
        import xgboost as xgb

        dm = xgb.DMatrix(np.random.rand(20, 4).astype(np.float32), label=np.random.randint(0, 5, 20))
        xgb.train(
            {"tree_method": "hist", "device": "cuda", "objective": "multi:softprob", "num_class": 5, "verbosity": 0},
            dm,
            num_boost_round=1,
        )
        return True
    except Exception:
        return False


def _probe_cb_gpu() -> bool:
    try:
        from catboost import CatBoostClassifier

        CatBoostClassifier(iterations=1, task_type="GPU", devices="0", verbose=0, loss_function="MultiClass").fit(
            np.random.rand(20, 4).astype(np.float32), np.random.randint(0, 5, 20)
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Model params
# ---------------------------------------------------------------------------


def _lgbm_params(gpu: bool) -> dict:
    p = {
        "objective": "multiclass",
        "num_class": 5,
        "metric": "multi_logloss",
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    if gpu:
        p["device"] = "gpu"
    return p


def _xgb_params(gpu: bool) -> dict:
    p = {
        "objective": "multi:softprob",
        "num_class": 5,
        "eval_metric": "mlogloss",
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 6,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "tree_method": "hist",
        "verbosity": 0,
        "early_stopping_rounds": ES_ROUNDS,
    }
    if gpu:
        p["device"] = "cuda"
    return p


def _cb_params(gpu: bool) -> dict:
    p = {
        "iterations": 2000,
        "learning_rate": 0.02,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "auto_class_weights": "Balanced",
        "eval_metric": "MultiClass",
        "random_seed": RANDOM_SEED,
        "verbose": 200,
        "early_stopping_rounds": ES_ROUNDS,
        "loss_function": "MultiClass",
    }
    if gpu:
        p["task_type"] = "GPU"
        p["devices"] = "0"
    return p


# ---------------------------------------------------------------------------
# Optimized thresholds for ordinal predictions
# ---------------------------------------------------------------------------


def optimize_thresholds(y_true: np.ndarray, oof_probs: np.ndarray) -> np.ndarray:
    """Find optimal argmax → class mapping by optimizing QWK on OOF probs."""
    # Start with simple argmax
    base_preds = np.argmax(oof_probs, axis=1)
    base_qwk = cohen_kappa_score(y_true, base_preds, weights="quadratic")
    logger.info("  Base argmax QWK: %.4f", base_qwk)

    # Try ordinal regression-style: use expected value
    expected = np.sum(oof_probs * np.arange(5), axis=1)

    def neg_qwk(thresholds: np.ndarray) -> float:
        t = np.sort(thresholds)
        preds = np.digitize(expected, t)
        return -cohen_kappa_score(y_true, preds, weights="quadratic")

    # Initial thresholds at equal spacing
    init = np.array([0.5, 1.5, 2.5, 3.5])
    result = minimize(neg_qwk, init, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-6})
    opt_thresholds = np.sort(result.x)
    opt_preds = np.digitize(expected, opt_thresholds)
    opt_qwk = cohen_kappa_score(y_true, opt_preds, weights="quadratic")
    logger.info("  Optimized thresholds QWK: %.4f (thresholds: %s)", opt_qwk, opt_thresholds.round(3))

    if opt_qwk > base_qwk:
        logger.info("  → Using optimized thresholds (+%.4f)", opt_qwk - base_qwk)
        return opt_thresholds
    else:
        logger.info("  → Keeping argmax (no improvement)")
        return np.array([])  # empty = use argmax


def apply_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Apply thresholds to probability predictions."""
    if len(thresholds) == 0:
        return np.argmax(probs, axis=1)
    expected = np.sum(probs * np.arange(5), axis=1)
    return np.digitize(expected, thresholds)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    name: str,
    model_type: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    fcols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train one model type with K-fold CV. Returns oof_probs, test_probs, fold_qwks."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_probs = np.zeros((len(X_train), 5))
    test_probs = np.zeros((len(X_test), 5))
    fold_qwks: list[float] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, y_tr = X_train.iloc[tr_idx][fcols], y_train[tr_idx]
        X_val, y_val = X_train.iloc[val_idx][fcols], y_train[val_idx]

        if model_type == "lgbm":
            import lightgbm as lgb

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False), lgb.log_evaluation(-1)],
            )
            best = model.best_iteration_

        elif model_type == "xgb":
            from xgboost import XGBClassifier

            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            best = model.best_iteration

        elif model_type == "cb":
            from catboost import CatBoostClassifier, Pool

            model = CatBoostClassifier(**params)
            model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_val, y_val), use_best_model=True)
            best = model.get_best_iteration()

        else:
            raise ValueError(f"Unknown model: {model_type}")

        val_prob = model.predict_proba(X_val)
        oof_probs[val_idx] = val_prob
        test_probs += model.predict_proba(X_test[fcols]) / N_FOLDS

        val_pred = np.argmax(val_prob, axis=1)
        qwk = cohen_kappa_score(y_val, val_pred, weights="quadratic")
        fold_qwks.append(qwk)

        if fold % 2 == 0 or fold == N_FOLDS:
            logger.info("  [%s] Fold %d/%d — QWK: %.4f | best: %d", name, fold, N_FOLDS, qwk, best)

    overall_qwk = cohen_kappa_score(y_train, np.argmax(oof_probs, axis=1), weights="quadratic")
    logger.info(
        "  [%s] OOF QWK: %.4f (mean fold: %.4f ± %.4f)\n", name, overall_qwk, np.mean(fold_qwks), np.std(fold_qwks)
    )
    return oof_probs, test_probs, fold_qwks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    logger.info("=" * 65)
    logger.info("Triagegeist — Training Pipeline (ESI 1-5 Multiclass)")
    logger.info("=" * 65)

    # GPU probes
    lgbm_gpu = _probe_lgbm_gpu()
    xgb_gpu = _probe_xgb_gpu()
    cb_gpu = _probe_cb_gpu()
    for n, g in [("LGBM", lgbm_gpu), ("XGB", xgb_gpu), ("CB", cb_gpu)]:
        logger.info("  %s → %s", n, "GPU ✓" if g else "CPU")

    # Build features
    logger.info("\nBuilding features...")
    X_train, X_test, fcols = build_all(use_nlp=True)
    y_train = (X_train[TARGET_COL] - 1).values.astype(int)  # 0-indexed
    test_ids = pd.read_csv(TEST_FILE)[ID_COL]
    logger.info("  Features: %d | Train: %s | Test: %s", len(fcols), X_train.shape, X_test.shape)

    # Train models
    logger.info("\n── LightGBM ──")
    lgbm_oof, lgbm_tp, lgbm_qwks = train_model("LGBM", "lgbm", _lgbm_params(lgbm_gpu), X_train, y_train, X_test, fcols)

    logger.info("── XGBoost ──")
    xgb_oof, xgb_tp, xgb_qwks = train_model("XGB", "xgb", _xgb_params(xgb_gpu), X_train, y_train, X_test, fcols)

    logger.info("── CatBoost ──")
    cb_oof, cb_tp, cb_qwks = train_model("CB", "cb", _cb_params(cb_gpu), X_train, y_train, X_test, fcols)

    # Ensemble — weighted average of probabilities
    logger.info("── Ensemble ──")
    qwks = {
        "lgbm": cohen_kappa_score(y_train, np.argmax(lgbm_oof, axis=1), weights="quadratic"),
        "xgb": cohen_kappa_score(y_train, np.argmax(xgb_oof, axis=1), weights="quadratic"),
        "cb": cohen_kappa_score(y_train, np.argmax(cb_oof, axis=1), weights="quadratic"),
    }
    scores = np.array(list(qwks.values()))
    w = np.exp(scores - scores.max())
    w /= w.sum()
    weights = dict(zip(qwks.keys(), w))

    ens_oof = weights["lgbm"] * lgbm_oof + weights["xgb"] * xgb_oof + weights["cb"] * cb_oof
    ens_test = weights["lgbm"] * lgbm_tp + weights["xgb"] * xgb_tp + weights["cb"] * cb_tp

    ens_preds = np.argmax(ens_oof, axis=1)
    ens_qwk = cohen_kappa_score(y_train, ens_preds, weights="quadratic")
    ens_f1 = f1_score(y_train, ens_preds, average="macro")
    logger.info("  Ensemble QWK: %.4f | Macro F1: %.4f", ens_qwk, ens_f1)
    for k, v in weights.items():
        logger.info("    %s: weight=%.3f, QWK=%.4f", k, v, qwks[k])

    # Optimize thresholds
    logger.info("\n── Threshold Optimization ──")
    opt_thresholds = optimize_thresholds(y_train, ens_oof)
    opt_preds = apply_thresholds(ens_oof, opt_thresholds)
    opt_qwk = cohen_kappa_score(y_train, opt_preds, weights="quadratic")
    opt_f1 = f1_score(y_train, opt_preds, average="macro")

    # Classification report
    logger.info("\n── Classification Report (OOF) ──")
    final_preds_oof = opt_preds if opt_qwk > ens_qwk else ens_preds
    final_qwk = max(opt_qwk, ens_qwk)
    target_names = [f"ESI-{i + 1}" for i in range(5)]
    print(classification_report(y_train, final_preds_oof, target_names=target_names))

    # Per-class recall (critical for ESI-1)
    logger.info("Per-class recall:")
    cm = confusion_matrix(y_train, final_preds_oof)
    for i in range(5):
        recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        logger.info("  ESI-%d: %.1f%% (%d/%d)", i + 1, recall * 100, cm[i, i], cm[i].sum())

    # Generate submission
    logger.info("\n── Submission ──")
    test_final = apply_thresholds(ens_test, opt_thresholds) if opt_qwk > ens_qwk else np.argmax(ens_test, axis=1)
    test_labels = test_final + 1  # back to 1-indexed

    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_labels})
    sub_path = DATA_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False)

    logger.info("  Submission distribution:")
    for level in sorted(submission[TARGET_COL].unique()):
        n = (submission[TARGET_COL] == level).sum()
        logger.info("    ESI-%d: %d (%.1f%%)", level, n, n / len(submission) * 100)

    # Save OOF predictions for later analysis
    oof_df = pd.DataFrame(ens_oof, columns=[f"prob_esi_{i + 1}" for i in range(5)])
    oof_df["true_label"] = y_train + 1
    oof_df["pred_label"] = final_preds_oof + 1
    oof_df.to_csv(DATA_DIR / "oof_predictions.csv", index=False)

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 65)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 65)
    logger.info("  LGBM QWK    : %.4f", qwks["lgbm"])
    logger.info("  XGB QWK     : %.4f", qwks["xgb"])
    logger.info("  CB QWK      : %.4f", qwks["cb"])
    logger.info("  ─────────────────────")
    logger.info("  Ensemble QWK: %.4f", ens_qwk)
    logger.info("  Optimized   : %.4f", final_qwk)
    logger.info("  Macro F1    : %.4f", f1_score(y_train, final_preds_oof, average="macro"))
    logger.info("  ─────────────────────")
    logger.info("  Baseline    : ~0.71 (provided notebook)")
    logger.info("  Improvement : +%.4f", final_qwk - 0.71)
    logger.info("  ─────────────────────")
    logger.info("  Submission  : %s", sub_path)
    logger.info("  OOF preds   : %s", DATA_DIR / "oof_predictions.csv")
    logger.info("  Time        : %.1f min", elapsed / 60)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
