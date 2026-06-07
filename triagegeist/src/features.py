"""
Triagegeist — Feature Engineering Pipeline

Two-stage design:
  1. build_structured_features() — vitals, scores, history, interactions (fast, no ML)
  2. build_nlp_features() — sentence embeddings from chief complaints (needs GPU)

Both can be applied identically to train and test (no data leakage).
NLP embeddings are precomputed and cached to disk.

Run standalone:  uv run python src/features.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import (
    DATA_DIR,
    MODELS_DIR,
    TRAIN_FILE,
    TEST_FILE,
    CC_FILE,
    HISTORY_FILE,
    TARGET_COL,
    ID_COL,
    VITAL_COLS,
    SCORE_COLS,
    BODY_COLS,
    HX_COLS,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("features")

# Service columns for counting
SERVICE_COLS_HX = [
    "hx_hypertension",
    "hx_diabetes_type2",
    "hx_diabetes_type1",
    "hx_asthma",
    "hx_copd",
    "hx_heart_failure",
    "hx_atrial_fibrillation",
    "hx_ckd",
    "hx_liver_disease",
    "hx_malignancy",
]

CARDIOVASCULAR_HX = [
    "hx_hypertension",
    "hx_coronary_artery_disease",
    "hx_heart_failure",
    "hx_atrial_fibrillation",
    "hx_stroke_prior",
    "hx_peripheral_vascular_disease",
]
RESPIRATORY_HX = ["hx_asthma", "hx_copd"]
PSYCHIATRIC_HX = ["hx_depression", "hx_anxiety", "hx_substance_use_disorder"]
METABOLIC_HX = ["hx_diabetes_type2", "hx_diabetes_type1", "hx_obesity", "hx_hypothyroidism", "hx_hyperthyroidism"]

# Categorical columns for ordinal encoding
CAT_ENCODE_MAPS: dict[str, dict[str, int]] = {
    "arrival_mode": {"walk-in": 0, "brought_by_family": 1, "police": 2, "transfer": 3, "ambulance": 4, "helicopter": 5},
    "shift": {"morning": 0, "afternoon": 1, "evening": 2, "night": 3},
    "arrival_season": {"spring": 0, "summer": 1, "autumn": 2, "winter": 3},
    "age_group": {"pediatric": 0, "young_adult": 1, "middle_aged": 2, "elderly": 3},
    "sex": {"F": 0, "M": 1, "Other": 2},
    "mental_status_triage": {"alert": 0, "drowsy": 1, "confused": 2, "agitated": 3, "unresponsive": 4},
    "arrival_day": {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6},
}


# ---------------------------------------------------------------------------
# Stage 1: Structured features
# ---------------------------------------------------------------------------


def build_structured_features(
    df: pd.DataFrame,
    cc: pd.DataFrame,
    hist: pd.DataFrame,
    is_train: bool = True,
) -> pd.DataFrame:
    """Build all structured features. No data leakage — same transform for train/test."""
    out = df.copy()

    # ── 1. Merge external tables ──
    out = out.merge(hist, on=ID_COL, how="left")
    out = out.merge(cc[["patient_id", "chief_complaint_raw"]], on=ID_COL, how="left")

    # ── 2. Missingness indicators (clinically informative) ──
    out["bp_missing"] = out["systolic_bp"].isnull().astype(np.int8)
    out["rr_missing"] = out["respiratory_rate"].isnull().astype(np.int8)
    out["temp_missing"] = out["temperature_c"].isnull().astype(np.int8)
    out["pain_not_recorded"] = (out["pain_score"] == -1).astype(np.int8)
    out["any_vital_missing"] = (out["bp_missing"] + out["rr_missing"] + out["temp_missing"]).clip(0, 1).astype(np.int8)
    out["n_vitals_missing"] = (out["bp_missing"] + out["rr_missing"] + out["temp_missing"]).astype(np.int8)

    # ── 3. Handle pain_score -1 ──
    out.loc[out["pain_score"] == -1, "pain_score"] = np.nan
    out["pain_score"] = out.groupby("age_group")["pain_score"].transform(lambda x: x.fillna(x.median()))
    out["pain_score"] = out["pain_score"].fillna(out["pain_score"].median())

    # ── 4. Impute remaining vitals ──
    impute_cols = [
        "systolic_bp",
        "diastolic_bp",
        "mean_arterial_pressure",
        "pulse_pressure",
        "shock_index",
        "respiratory_rate",
        "temperature_c",
    ]
    for col in impute_cols:
        out[col] = out.groupby(["age_group", "shift"])[col].transform(lambda x: x.fillna(x.median()))
        out[col] = out[col].fillna(out[col].median())

    # ── 5. Derived clinical features ──

    # Demographics
    out["elderly"] = (out["age"] >= 65).astype(np.int8)
    out["pediatric"] = (out["age"] < 18).astype(np.int8)
    out["very_old"] = (out["age"] >= 80).astype(np.int8)

    # Vital-derived
    out["hr_rr_ratio"] = (out["heart_rate"] / out["respiratory_rate"].clip(lower=1)).astype(np.float32)
    out["pulse_pressure_ratio"] = (out["pulse_pressure"] / out["systolic_bp"].clip(lower=1)).astype(np.float32)
    out["map_deficit"] = (65 - out["mean_arterial_pressure"]).clip(lower=0).astype(np.float32)  # MAP<65 = shock
    out["hypertensive_crisis"] = (out["systolic_bp"] >= 180).astype(np.int8)
    out["hypotensive"] = (out["systolic_bp"] <= 90).astype(np.int8)
    out["tachycardic"] = (out["heart_rate"] >= 100).astype(np.int8)
    out["bradycardic"] = (out["heart_rate"] <= 50).astype(np.int8)
    out["tachypneic"] = (out["respiratory_rate"] >= 22).astype(np.int8)
    out["febrile"] = (out["temperature_c"] >= 38.0).astype(np.int8)
    out["hypothermic"] = (out["temperature_c"] <= 35.0).astype(np.int8)
    out["hypoxic"] = (out["spo2"] < 94).astype(np.int8)
    out["severe_hypoxia"] = (out["spo2"] < 88).astype(np.int8)

    # GCS categories
    out["gcs_severe"] = (out["gcs_total"] <= 8).astype(np.int8)
    out["gcs_moderate"] = ((out["gcs_total"] > 8) & (out["gcs_total"] <= 12)).astype(np.int8)
    out["gcs_mild"] = (out["gcs_total"] >= 13).astype(np.int8)

    # NEWS2 categories (standard thresholds)
    out["news2_low"] = (out["news2_score"] <= 4).astype(np.int8)
    out["news2_medium"] = ((out["news2_score"] >= 5) & (out["news2_score"] <= 6)).astype(np.int8)
    out["news2_high"] = (out["news2_score"] >= 7).astype(np.int8)
    out["news2_critical"] = (out["news2_score"] >= 12).astype(np.int8)

    # Pain categories
    out["pain_severe"] = (out["pain_score"] >= 7).astype(np.int8)
    out["pain_moderate"] = ((out["pain_score"] >= 4) & (out["pain_score"] < 7)).astype(np.int8)
    out["pain_none"] = (out["pain_score"] == 0).astype(np.int8)

    # Composite risk flags
    out["n_abnormal_vitals"] = (
        out["hypotensive"]
        + out["tachycardic"]
        + out["tachypneic"]
        + out["febrile"]
        + out["hypoxic"]
        + out["bradycardic"]
        + out["hypothermic"]
    ).astype(np.int8)

    out["critical_flag"] = (
        (out["gcs_total"] <= 8) | (out["spo2"] < 88) | (out["systolic_bp"] <= 70) | (out["news2_score"] >= 12)
    ).astype(np.int8)

    # ── 6. History composite scores ──
    out["cv_risk_score"] = out[CARDIOVASCULAR_HX].sum(axis=1).astype(np.int8)
    out["resp_risk_score"] = out[RESPIRATORY_HX].sum(axis=1).astype(np.int8)
    out["psych_risk_score"] = out[PSYCHIATRIC_HX].sum(axis=1).astype(np.int8)
    out["metabolic_risk_score"] = out[METABOLIC_HX].sum(axis=1).astype(np.int8)
    out["total_hx_count"] = out[HX_COLS].sum(axis=1).astype(np.int8)

    # History × vitals interactions
    out["elderly_high_news2"] = (out["elderly"] & (out["news2_score"] >= 7)).astype(np.int8)
    out["elderly_low_gcs"] = (out["elderly"] & (out["gcs_total"] <= 12)).astype(np.int8)
    out["cv_risk_x_shock"] = (out["cv_risk_score"] * out["shock_index"]).astype(np.float32)
    out["comorbid_x_news2"] = (out["num_comorbidities"] * out["news2_score"]).astype(np.float32)
    out["prior_visits_x_severity"] = (out["num_prior_ed_visits_12m"] * out["news2_score"]).astype(np.float32)

    # ── 7. Chief complaint system encoding ──
    cc_system_map: dict[str, int] = {
        "cardiac": 0,
        "neurological": 1,
        "respiratory": 2,
        "trauma": 3,
        "gastrointestinal": 4,
        "psychiatric": 5,
        "infectious": 6,
        "musculoskeletal": 7,
        "genitourinary": 8,
        "dermatological": 9,
        "ENT": 10,
        "endocrine": 11,
        "ophthalmological": 12,
        "other": 13,
    }
    out["cc_system_encoded"] = out["chief_complaint_system"].map(cc_system_map).fillna(13).astype(np.int8)

    # ── 8. Chief complaint text features (simple, no ML) ──
    cc_text = out["chief_complaint_raw"].fillna("")
    out["cc_word_count"] = cc_text.str.split().str.len().fillna(0).astype(np.int8)
    out["cc_char_count"] = cc_text.str.len().fillna(0).astype(np.int16)

    # High-risk keyword flags
    high_risk_keywords = [
        "chest pain",
        "dyspnea",
        "shortness of breath",
        "seizure",
        "unconscious",
        "unresponsive",
        "stroke",
        "cardiac arrest",
        "anaphylaxis",
        "sepsis",
        "hemorrhage",
        "bleeding",
        "fracture",
        "head injury",
        "overdose",
        "suicide",
        "suicidal",
        "ruptured",
        "acute",
        "severe",
    ]
    for kw in high_risk_keywords:
        col_name = f"cc_has_{kw.replace(' ', '_')}"
        out[col_name] = cc_text.str.contains(kw, case=False, na=False).astype(np.int8)

    out["cc_n_high_risk_keywords"] = sum(
        cc_text.str.contains(kw, case=False, na=False).astype(int) for kw in high_risk_keywords
    ).astype(np.int8)

    # Temporal keywords
    temporal_keywords = ["worsening", "sudden", "acute", "new onset", "progressive"]
    out["cc_n_temporal_keywords"] = sum(
        cc_text.str.contains(kw, case=False, na=False).astype(int) for kw in temporal_keywords
    ).astype(np.int8)

    # ── 9. Ordinal encode categoricals ──
    for col, mapping in CAT_ENCODE_MAPS.items():
        if col in out.columns:
            out[col] = out[col].astype(str).map(mapping).fillna(-1).astype(np.int8)

    # Language and insurance — frequency encode
    for col in ["language", "insurance_type", "transport_origin", "pain_location"]:
        if col in out.columns:
            freq = out[col].value_counts(normalize=True).to_dict()
            out[f"{col}_freq"] = out[col].map(freq).fillna(0).astype(np.float32)
            # Also ordinal
            cats = out[col].astype("category").cat.codes
            out[col] = cats.astype(np.int8)

    # Site and nurse — ordinal
    for col in ["site_id", "triage_nurse_id"]:
        if col in out.columns:
            out[col] = out[col].astype("category").cat.codes.astype(np.int8)

    # ── 10. Drop columns not useful for modeling ──
    drop_cols = ["patient_id", "chief_complaint_raw", "chief_complaint_system"]
    if is_train:
        drop_cols += ["disposition", "ed_los_hours"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")

    return out


# ---------------------------------------------------------------------------
# Stage 2: NLP embeddings from chief complaints
# ---------------------------------------------------------------------------


def build_nlp_embeddings(
    cc: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Generate sentence embeddings for chief complaints. Caches to disk."""
    if cache_path and cache_path.exists():
        logger.info("Loading cached NLP embeddings from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Generating NLP embeddings with %s...", model_name)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cuda")
    texts = cc["chief_complaint_raw"].fillna("unknown complaint").tolist()

    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)

    emb_dim = embeddings.shape[1]
    emb_cols = [f"cc_emb_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols, dtype=np.float32)
    emb_df.insert(0, ID_COL, cc[ID_COL].values)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        emb_df.to_parquet(cache_path, index=False)
        logger.info("Cached embeddings to %s (%s)", cache_path, emb_df.shape)

    return emb_df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all feature columns (exclude target and ID)."""
    exclude = {TARGET_COL, ID_COL, "patient_id"}
    return [c for c in df.columns if c not in exclude]


def build_all(
    train_path: Path = TRAIN_FILE,
    test_path: Path = TEST_FILE,
    cc_path: Path = CC_FILE,
    hist_path: Path = HISTORY_FILE,
    use_nlp: bool = True,
    nlp_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Full feature pipeline: structured + NLP. Returns train, test, feature_cols."""
    logger.info("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    cc = pd.read_csv(cc_path)
    hist = pd.read_csv(hist_path)

    logger.info("Building structured features...")
    train_ids = train[ID_COL].copy()
    test_ids = test[ID_COL].copy()

    X_train = build_structured_features(train, cc, hist, is_train=True)
    X_test = build_structured_features(test, cc, hist, is_train=False)

    if use_nlp:
        logger.info("Building NLP embeddings...")
        cache = MODELS_DIR / "cc_embeddings.parquet"
        emb_df = build_nlp_embeddings(cc, model_name=nlp_model, cache_path=cache)

        # Merge embeddings
        X_train.insert(0, ID_COL, train_ids)
        X_test.insert(0, ID_COL, test_ids)
        X_train = X_train.merge(emb_df, on=ID_COL, how="left")
        X_test = X_test.merge(emb_df, on=ID_COL, how="left")
        X_train = X_train.drop(columns=[ID_COL])
        X_test = X_test.drop(columns=[ID_COL])

        # Fill NaN embeddings
        emb_cols = [c for c in X_train.columns if c.startswith("cc_emb_")]
        X_train[emb_cols] = X_train[emb_cols].fillna(0)
        X_test[emb_cols] = X_test[emb_cols].fillna(0)

    feature_cols = get_feature_cols(X_train)
    logger.info(
        "Total features: %d (structured: %d, NLP: %d)",
        len(feature_cols),
        len([c for c in feature_cols if not c.startswith("cc_emb_")]),
        len([c for c in feature_cols if c.startswith("cc_emb_")]),
    )

    return X_train, X_test, feature_cols


# ---------------------------------------------------------------------------
# Standalone execution — precompute embeddings
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("Triagegeist — Feature Engineering Pipeline")
    logger.info("=" * 60)

    X_train, X_test, fcols = build_all(use_nlp=True)

    print(f"\n── Results ──")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Features: {len(fcols)}")
    print(f"  Target present: {TARGET_COL in X_train.columns}")

    # Show feature groups
    structured = [c for c in fcols if not c.startswith("cc_emb_")]
    nlp = [c for c in fcols if c.startswith("cc_emb_")]
    keyword = [c for c in fcols if c.startswith("cc_has_")]
    print(f"\n  Structured features: {len(structured)}")
    print(f"  NLP embedding dims: {len(nlp)}")
    print(f"  Keyword flags: {len(keyword)}")

    if TARGET_COL in X_train.columns:
        print(f"\n  Target distribution:")
        vc = X_train[TARGET_COL].value_counts().sort_index()
        for level, count in vc.items():
            print(f"    ESI-{level}: {count:,}")

    print("\n" + "=" * 60)
