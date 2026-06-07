#!/usr/bin/env python
# ---
# jupyter:
#   title: "Triagegeist — AI-Assisted Emergency Triage Acuity Prediction"
#   author: "Adrian Infantes"
# ---

# %% [markdown]
# # Triagegeist — AI-Assisted Emergency Triage Acuity Prediction
# ## Predicting ESI Acuity (1–5) Using Gradient Boosted Trees with Clinical NLP
#
# **Competition:** Triagegeist — AI in Emergency Triage
# **Host:** Laitinen-Fredriksson Foundation
# **Author:** Adrian Infantes ([GitHub](https://github.com/adrianinfantes))
#
# ---
#
# ### Clinical Motivation
#
# Emergency department (ED) triage is the critical first step in the patient care pathway.
# Triage nurses must rapidly assign a severity level — typically using the Emergency Severity
# Index (ESI, 1–5) — that determines treatment priority. This decision is made under extreme
# cognitive load, with incomplete information, and in chronically understaffed environments.
#
# Inter-rater variability in ESI assignment is well-documented in the literature. Undertriage
# of critically ill patients (assigning a lower severity than warranted) directly contributes
# to delayed care and adverse outcomes. This notebook demonstrates that an ensemble of gradient
# boosted decision trees, augmented with NLP-derived features from chief complaint text, can
# achieve near-perfect ESI prediction — offering a potential clinical decision support tool
# to reduce triage variability and catch missed high-acuity presentations.
#
# ### Approach Summary
#
# | Component | Method |
# |-----------|--------|
# | **Structured features** | 132 features: vitals, clinical scores, history composites, interaction terms, missingness indicators |
# | **NLP features** | 384-dim sentence embeddings from `all-MiniLM-L6-v2` on chief complaint text |
# | **Models** | LightGBM + XGBoost + CatBoost (weighted ensemble) |
# | **Metric** | Quadratic Weighted Kappa (QWK) |
# | **Result** | QWK = 0.9998 (baseline: ~0.71) |
# | **Explainability** | SHAP values (global + per-class), error analysis, fairness audit |

# %% [markdown]
# ## 1. Setup and Configuration

# %%
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, f1_score
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

SEED = 42
np.random.seed(SEED)

# Paths — Kaggle or local
if os.path.exists("/kaggle/input/datasets/adrininfantesromero/triagegeist-data/"):
    DATA_PATH = "/kaggle/input/datasets/adrininfantesromero/triagegeist-data/"
elif os.path.exists("/kaggle/input/triagegeist-data/"):
    DATA_PATH = "/kaggle/input/triagegeist-data/"
elif os.path.exists("/kaggle/input/triagegeist/"):
    DATA_PATH = "/kaggle/input/triagegeist/"
else:
    DATA_PATH = "../data/"

TARGET_COL = "triage_acuity"
ID_COL = "patient_id"
N_FOLDS = 10
ES_ROUNDS = 150

ESI_COLORS = {1: "#d32f2f", 2: "#f57c00", 3: "#fbc02d", 4: "#388e3c", 5: "#1976d2"}
ESI_LABELS = {1: "Immediate", 2: "Emergent", 3: "Urgent", 4: "Less Urgent", 5: "Non-Urgent"}

print(f"LightGBM {lgb.__version__} | XGBoost {xgb.__version__}")
print(f"Data path: {DATA_PATH}")

# %% [markdown]
# ## 2. Load Data

# %%
train = pd.read_csv(DATA_PATH + "train.csv")
test = pd.read_csv(DATA_PATH + "test.csv")
cc = pd.read_csv(DATA_PATH + "chief_complaints.csv")
hist = pd.read_csv(DATA_PATH + "patient_history.csv")

print(f"Train:  {train.shape[0]:,} x {train.shape[1]}")
print(f"Test:   {test.shape[0]:,} x {test.shape[1]}")
print(f"CC:     {cc.shape[0]:,} x {cc.shape[1]}")
print(f"Hist:   {hist.shape[0]:,} x {hist.shape[1]}")

# %% [markdown]
# ## 3. Exploratory Data Analysis

# %%
# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

vc = train[TARGET_COL].value_counts().sort_index()
labels = [f"ESI-{k}\n({v})" for k, v in ESI_LABELS.items()]
bars = axes[0].bar(labels, vc.values, color=[ESI_COLORS[i] for i in vc.index], edgecolor="white")
for bar, v in zip(bars, vc.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2, v + 200, f"{v:,}\n({v / len(train) * 100:.1f}%)", ha="center", fontsize=9
    )
axes[0].set_title("Triage Acuity Distribution", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Count")

# NEWS2 by acuity
train.boxplot(
    column="news2_score",
    by=TARGET_COL,
    ax=axes[1],
    boxprops=dict(color="steelblue"),
    medianprops=dict(color="crimson", linewidth=2),
)
axes[1].set_title("NEWS2 Score by Acuity", fontsize=13, fontweight="bold")
axes[1].set_xlabel("ESI Level")
axes[1].set_ylabel("NEWS2")
plt.suptitle("")
plt.tight_layout()
plt.show()

# %%
# Vital signs heatmap
vital_cols = [
    "systolic_bp",
    "heart_rate",
    "respiratory_rate",
    "temperature_c",
    "spo2",
    "gcs_total",
    "news2_score",
    "shock_index",
]
medians = train.groupby(TARGET_COL)[vital_cols].median()

fig, ax = plt.subplots(figsize=(12, 3.5))
sns.heatmap(medians.T, annot=True, fmt=".1f", cmap="RdYlGn_r", linewidths=0.5, ax=ax, cbar_kws={"label": "Median"})
ax.set_xlabel("ESI Level")
ax.set_title("Median Vital Signs & Scores by Triage Acuity", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Key clinical observations:**
# - **NEWS2** is the dominant predictor: ESI-1 patients have a median NEWS2 of ~14, while ESI-5 patients have ~0
# - **GCS** sharply separates ESI-1 (median 6.5) from ESI-2+ (median 12-15)
# - **SpO2**, **respiratory rate**, and **shock index** follow expected clinical gradients
# - ESI-1 (Immediate) accounts for only 4% of cases — class imbalance requiring balanced training

# %% [markdown]
# ## 4. Feature Engineering

# %%
SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]  # Not present in this dataset

HX_COLS = [c for c in hist.columns if c.startswith("hx_")]

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

CAT_ENCODE_MAPS = {
    "arrival_mode": {"walk-in": 0, "brought_by_family": 1, "police": 2, "transfer": 3, "ambulance": 4, "helicopter": 5},
    "shift": {"morning": 0, "afternoon": 1, "evening": 2, "night": 3},
    "arrival_season": {"spring": 0, "summer": 1, "autumn": 2, "winter": 3},
    "age_group": {"pediatric": 0, "young_adult": 1, "middle_aged": 2, "elderly": 3},
    "sex": {"F": 0, "M": 1, "Other": 2},
    "mental_status_triage": {"alert": 0, "drowsy": 1, "confused": 2, "agitated": 3, "unresponsive": 4},
    "arrival_day": {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6},
}


def build_features(df, cc_df, hist_df, is_train=True):
    """Build all structured + NLP features. No data leakage."""
    out = df.copy()

    # Merge
    out = out.merge(hist_df, on=ID_COL, how="left")
    out = out.merge(cc_df[[ID_COL, "chief_complaint_raw"]], on=ID_COL, how="left")

    # Missingness indicators
    out["bp_missing"] = out["systolic_bp"].isnull().astype(np.int8)
    out["rr_missing"] = out["respiratory_rate"].isnull().astype(np.int8)
    out["temp_missing"] = out["temperature_c"].isnull().astype(np.int8)
    out["pain_not_recorded"] = (out["pain_score"] == -1).astype(np.int8)
    out["n_vitals_missing"] = (out["bp_missing"] + out["rr_missing"] + out["temp_missing"]).astype(np.int8)

    # Handle pain_score
    out.loc[out["pain_score"] == -1, "pain_score"] = np.nan
    out["pain_score"] = out.groupby("age_group")["pain_score"].transform(lambda x: x.fillna(x.median()))
    out["pain_score"] = out["pain_score"].fillna(out["pain_score"].median())

    # Impute vitals
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

    # Demographics
    out["elderly"] = (out["age"] >= 65).astype(np.int8)
    out["pediatric"] = (out["age"] < 18).astype(np.int8)
    out["very_old"] = (out["age"] >= 80).astype(np.int8)

    # Vital-derived
    out["hr_rr_ratio"] = (out["heart_rate"] / out["respiratory_rate"].clip(lower=1)).astype(np.float32)
    out["pulse_pressure_ratio"] = (out["pulse_pressure"] / out["systolic_bp"].clip(lower=1)).astype(np.float32)
    out["map_deficit"] = (65 - out["mean_arterial_pressure"]).clip(lower=0).astype(np.float32)
    out["hypertensive_crisis"] = (out["systolic_bp"] >= 180).astype(np.int8)
    out["hypotensive"] = (out["systolic_bp"] <= 90).astype(np.int8)
    out["tachycardic"] = (out["heart_rate"] >= 100).astype(np.int8)
    out["bradycardic"] = (out["heart_rate"] <= 50).astype(np.int8)
    out["tachypneic"] = (out["respiratory_rate"] >= 22).astype(np.int8)
    out["febrile"] = (out["temperature_c"] >= 38.0).astype(np.int8)
    out["hypothermic"] = (out["temperature_c"] <= 35.0).astype(np.int8)
    out["hypoxic"] = (out["spo2"] < 94).astype(np.int8)
    out["severe_hypoxia"] = (out["spo2"] < 88).astype(np.int8)

    # Score categories
    out["gcs_severe"] = (out["gcs_total"] <= 8).astype(np.int8)
    out["gcs_moderate"] = ((out["gcs_total"] > 8) & (out["gcs_total"] <= 12)).astype(np.int8)
    out["news2_low"] = (out["news2_score"] <= 4).astype(np.int8)
    out["news2_medium"] = ((out["news2_score"] >= 5) & (out["news2_score"] <= 6)).astype(np.int8)
    out["news2_high"] = (out["news2_score"] >= 7).astype(np.int8)
    out["news2_critical"] = (out["news2_score"] >= 12).astype(np.int8)
    out["pain_severe"] = (out["pain_score"] >= 7).astype(np.int8)
    out["pain_moderate"] = ((out["pain_score"] >= 4) & (out["pain_score"] < 7)).astype(np.int8)

    # Composite flags
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

    # History composites
    out["cv_risk_score"] = out[CARDIOVASCULAR_HX].sum(axis=1).astype(np.int8)
    out["resp_risk_score"] = out[RESPIRATORY_HX].sum(axis=1).astype(np.int8)
    out["psych_risk_score"] = out[PSYCHIATRIC_HX].sum(axis=1).astype(np.int8)
    out["metabolic_risk_score"] = out[METABOLIC_HX].sum(axis=1).astype(np.int8)
    out["total_hx_count"] = out[HX_COLS].sum(axis=1).astype(np.int8)

    # Interactions
    out["elderly_high_news2"] = (out["elderly"] & (out["news2_score"] >= 7)).astype(np.int8)
    out["elderly_low_gcs"] = (out["elderly"] & (out["gcs_total"] <= 12)).astype(np.int8)
    out["cv_risk_x_shock"] = (out["cv_risk_score"] * out["shock_index"]).astype(np.float32)
    out["comorbid_x_news2"] = (out["num_comorbidities"] * out["news2_score"]).astype(np.float32)
    out["prior_visits_x_severity"] = (out["num_prior_ed_visits_12m"] * out["news2_score"]).astype(np.float32)

    # Chief complaint system encoding
    cc_sys_map = {
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
    out["cc_system_encoded"] = out["chief_complaint_system"].map(cc_sys_map).fillna(13).astype(np.int8)

    # Chief complaint text features
    cc_text = out["chief_complaint_raw"].fillna("")
    out["cc_word_count"] = cc_text.str.split().str.len().fillna(0).astype(np.int8)
    out["cc_char_count"] = cc_text.str.len().fillna(0).astype(np.int16)

    high_risk_kw = [
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
    kw_flags = pd.DataFrame(index=out.index)
    for kw in high_risk_kw:
        kw_flags[f"cc_has_{kw.replace(' ', '_')}"] = cc_text.str.contains(kw, case=False, na=False).astype(np.int8)
    out = pd.concat([out, kw_flags], axis=1)

    out["cc_n_high_risk_keywords"] = kw_flags.sum(axis=1).astype(np.int8)

    temporal_kw = ["worsening", "sudden", "acute", "new onset", "progressive"]
    out["cc_n_temporal_keywords"] = sum(
        cc_text.str.contains(kw, case=False, na=False).astype(int) for kw in temporal_kw
    ).astype(np.int8)

    # Ordinal encode categoricals
    for col, mapping in CAT_ENCODE_MAPS.items():
        if col in out.columns:
            out[col] = out[col].astype(str).map(mapping).fillna(-1).astype(np.int8)

    for col in ["language", "insurance_type", "transport_origin", "pain_location"]:
        if col in out.columns:
            freq = out[col].value_counts(normalize=True).to_dict()
            out[f"{col}_freq"] = out[col].map(freq).fillna(0).astype(np.float32)
            out[col] = out[col].astype("category").cat.codes.astype(np.int8)

    for col in ["site_id", "triage_nurse_id"]:
        if col in out.columns:
            out[col] = out[col].astype("category").cat.codes.astype(np.int8)

    # Drop non-feature columns
    drop_cols = [ID_COL, "chief_complaint_raw", "chief_complaint_system"]
    if is_train:
        drop_cols += ["disposition", "ed_los_hours"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return out


t0 = time.time()
X_train = build_features(train, cc, hist, is_train=True)
X_test = build_features(test, cc, hist, is_train=False)
print(f"Structured features built in {time.time() - t0:.1f}s")
print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# %% [markdown]
# ### 4b. NLP Features — Sentence Embeddings from Chief Complaints

# %%
from sentence_transformers import SentenceTransformer

t0 = time.time()
model_name = "sentence-transformers/all-MiniLM-L6-v2"
st_model = SentenceTransformer(model_name, device="cuda")

texts = cc["chief_complaint_raw"].fillna("unknown complaint").tolist()
embeddings = st_model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

emb_cols = [f"cc_emb_{i}" for i in range(embeddings.shape[1])]
emb_df = pd.DataFrame(embeddings, columns=emb_cols, dtype=np.float32)
emb_df[ID_COL] = cc[ID_COL].values

# Merge with train/test
train_ids = train[ID_COL].values
test_ids = test[ID_COL].values

X_train[ID_COL] = train_ids
X_test[ID_COL] = test_ids
X_train = X_train.merge(emb_df, on=ID_COL, how="left").drop(columns=[ID_COL])
X_test = X_test.merge(emb_df, on=ID_COL, how="left").drop(columns=[ID_COL])

emb_c = [c for c in X_train.columns if c.startswith("cc_emb_")]
X_train[emb_c] = X_train[emb_c].fillna(0)
X_test[emb_c] = X_test[emb_c].fillna(0)

print(f"NLP embeddings: {len(emb_c)} dims, generated in {time.time() - t0:.1f}s")
print(f"Total features: {X_train.shape[1] - 1} (structured: {X_train.shape[1] - 1 - len(emb_c)}, NLP: {len(emb_c)})")

# %%
y_train = (X_train[TARGET_COL] - 1).values.astype(int)
feature_cols = [c for c in X_train.columns if c != TARGET_COL]
print(f"Features: {len(feature_cols)} | Target classes: {np.unique(y_train)}")

# %% [markdown]
# ## 5. Model Training — GBDT Ensemble with 10-Fold CV
#
# We train three gradient boosted tree models and combine them via weighted averaging.
# All models use `class_weight='balanced'` to address the ESI-1 class imbalance (4%).


# %%
def train_model(name, model_type, params, X, y, X_te, fcols, n_folds=N_FOLDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X), 5))
    test_p = np.zeros((len(X_te), 5))
    qwks = []

    for fold, (tr_i, val_i) in enumerate(skf.split(X, y), 1):
        Xtr, ytr = X.iloc[tr_i][fcols], y[tr_i]
        Xv, yv = X.iloc[val_i][fcols], y[val_i]

        if model_type == "lgbm":
            m = lgb.LGBMClassifier(**params)
            m.fit(
                Xtr,
                ytr,
                eval_set=[(Xv, yv)],
                callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False), lgb.log_evaluation(-1)],
            )
        elif model_type == "xgb":
            m = xgb.XGBClassifier(**params)
            m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        elif model_type == "cb":
            m = CatBoostClassifier(**params)
            m.fit(Pool(Xtr, ytr), eval_set=Pool(Xv, yv), use_best_model=True)

        vp = m.predict_proba(Xv)
        oof[val_i] = vp
        test_p += m.predict_proba(X_te[fcols]) / n_folds
        qwk = cohen_kappa_score(yv, np.argmax(vp, axis=1), weights="quadratic")
        qwks.append(qwk)
        if fold % 5 == 0:
            print(f"  [{name}] Fold {fold}/{n_folds} — QWK: {qwk:.4f}")

    overall = cohen_kappa_score(y, np.argmax(oof, axis=1), weights="quadratic")
    print(f"  [{name}] OOF QWK: {overall:.4f} (mean: {np.mean(qwks):.4f} +/- {np.std(qwks):.4f})\n")
    return oof, test_p, overall


# %%
lgbm_params = {
    "objective": "multiclass",
    "num_class": 5,
    "metric": "multi_logloss",
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
    "device": "gpu",
}

xgb_params = {
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
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
    "device": "cuda",
    "verbosity": 0,
    "early_stopping_rounds": ES_ROUNDS,
}

cb_params = {
    "iterations": 2000,
    "learning_rate": 0.02,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "auto_class_weights": "Balanced",
    "eval_metric": "MultiClass",
    "random_seed": SEED,
    "verbose": 200,
    "early_stopping_rounds": ES_ROUNDS,
    "task_type": "GPU",
    "devices": "0",
    "loss_function": "MultiClass",
}

print("Training LightGBM...")
lgbm_oof, lgbm_tp, lgbm_qwk = train_model("LGBM", "lgbm", lgbm_params, X_train, y_train, X_test, feature_cols)

print("Training XGBoost...")
xgb_oof, xgb_tp, xgb_qwk = train_model("XGB", "xgb", xgb_params, X_train, y_train, X_test, feature_cols)

print("Training CatBoost...")
cb_oof, cb_tp, cb_qwk = train_model("CB", "cb", cb_params, X_train, y_train, X_test, feature_cols)

# %% [markdown]
# ### Ensemble & Threshold Optimization

# %%
# Weighted ensemble
scores = np.array([lgbm_qwk, xgb_qwk, cb_qwk])
w = np.exp(scores - scores.max())
w /= w.sum()
print(f"Ensemble weights: LGBM={w[0]:.3f} XGB={w[1]:.3f} CB={w[2]:.3f}")

ens_oof = w[0] * lgbm_oof + w[1] * xgb_oof + w[2] * cb_oof
ens_test = w[0] * lgbm_tp + w[1] * xgb_tp + w[2] * cb_tp
ens_qwk = cohen_kappa_score(y_train, np.argmax(ens_oof, axis=1), weights="quadratic")
print(f"Ensemble QWK: {ens_qwk:.4f}")

# Ordinal threshold optimization
expected = np.sum(ens_oof * np.arange(5), axis=1)


def neg_qwk(thresholds):
    t = np.sort(thresholds)
    preds = np.digitize(expected, t)
    return -cohen_kappa_score(y_train, preds, weights="quadratic")


result = minimize(neg_qwk, [0.5, 1.5, 2.5, 3.5], method="Nelder-Mead", options={"maxiter": 5000})
opt_t = np.sort(result.x)
opt_preds = np.digitize(expected, opt_t)
opt_qwk = cohen_kappa_score(y_train, opt_preds, weights="quadratic")
print(f"Optimized QWK: {opt_qwk:.4f} (thresholds: {opt_t.round(3)})")

use_opt = opt_qwk > ens_qwk
final_qwk = max(opt_qwk, ens_qwk)
final_oof_preds = opt_preds if use_opt else np.argmax(ens_oof, axis=1)
print(f"\nFinal OOF QWK: {final_qwk:.4f} (method: {'ordinal thresholds' if use_opt else 'argmax'})")

# %% [markdown]
# ## 6. Evaluation

# %%
target_names = [f"ESI-{i + 1}" for i in range(5)]
print("Classification Report (OOF):")
print(classification_report(y_train, final_oof_preds, target_names=target_names))

# %%
# Confusion matrix
cm = confusion_matrix(y_train, final_oof_preds)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm_pct,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Row %"},
)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
ax.set_title(f"Confusion Matrix — QWK = {final_qwk:.4f}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %%
# Per-class recall (safety-critical)
print("Per-class recall:")
for i in range(5):
    recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"  ESI-{i + 1}: {recall:.1%} ({cm[i, i]:,} / {cm[i].sum():,})")

# %% [markdown]
# ## 7. Explainability — SHAP Analysis
#
# We use SHAP (SHapley Additive exPlanations) to understand which features drive
# the model's triage decisions. This is essential for clinical trust and adoption.

# %%
import shap

# Train single model for SHAP
shap_model = lgb.LGBMClassifier(**{**lgbm_params, "n_estimators": 1500})
shap_model.fit(X_train[feature_cols], y_train)

sample_idx = np.random.choice(len(X_train), size=3000, replace=False)
X_shap = X_train.iloc[sample_idx]

explainer = shap.TreeExplainer(shap_model)
shap_raw = explainer.shap_values(X_shap[feature_cols])

# Handle format
if isinstance(shap_raw, list):
    shap_vals = shap_raw
else:
    shap_vals = [shap_raw[:, :, c] for c in range(shap_raw.shape[2])]

# Global importance
mean_shap = np.zeros(len(feature_cols))
for sv in shap_vals:
    mean_shap += np.abs(sv).mean(axis=0)
mean_shap /= 5

top30_idx = np.argsort(mean_shap)[::-1][:30]
top30_feat = [feature_cols[i] for i in top30_idx]
top30_val = mean_shap[top30_idx]

fig, ax = plt.subplots(figsize=(10, 9))
colors = ["#d32f2f" if not f.startswith("cc_emb_") else "#1976d2" for f in top30_feat]
ax.barh(top30_feat[::-1], top30_val[::-1], color=colors[::-1], edgecolor="white")
ax.set_title("Top 30 Features — Mean |SHAP| (All Classes)", fontsize=13, fontweight="bold")
ax.set_xlabel("Mean |SHAP value|")
from matplotlib.patches import Patch

ax.legend(
    handles=[Patch(facecolor="#d32f2f", label="Structured"), Patch(facecolor="#1976d2", label="NLP Embedding")],
    loc="lower right",
)
plt.tight_layout()
plt.show()

# %% [markdown]
# **SHAP confirms clinical expectations**: NEWS2, GCS, SpO2, respiratory rate, and shock index
# are the dominant predictors. NLP embeddings from chief complaint text contribute meaningful
# signal, particularly for differentiating ESI-2/3 where physiological scores overlap.

# %% [markdown]
# ## 8. Fairness Analysis
#
# We examine whether the model exhibits systematic bias across demographic groups.
# This is critical for clinical deployment — undertriage of vulnerable populations
# is a documented patient safety concern.

# %%
oof_df = pd.DataFrame({"true": y_train + 1, "pred": final_oof_preds + 1})
oof_df["correct"] = (oof_df["true"] == oof_df["pred"]).astype(int)
oof_df["undertriage"] = (oof_df["pred"] > oof_df["true"]).astype(int)

for col in ["sex", "age_group", "language", "insurance_type"]:
    oof_df[col] = train[col].values

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, col in enumerate(["sex", "age_group", "language", "insurance_type"]):
    ax = axes[idx // 2, idx % 2]
    g = (
        oof_df.groupby(col)
        .agg(accuracy=("correct", "mean"), undertriage=("undertriage", "mean"), n=("correct", "count"))
        .reset_index()
    )
    x = range(len(g))
    ax.bar([i - 0.17 for i in x], g["accuracy"] * 100, 0.35, label="Accuracy %", color="#1976d2", alpha=0.8)
    ax.bar([i + 0.17 for i in x], g["undertriage"] * 100, 0.35, label="Undertriage %", color="#d32f2f", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(g[col], rotation=45, ha="right", fontsize=9)
    ax.set_title(f"Fairness: {col.replace('_', ' ').title()}", fontweight="bold")
    ax.set_ylabel("Rate (%)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
plt.suptitle("Fairness Audit — Accuracy & Undertriage by Demographics", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Fairness findings**: Accuracy and undertriage rates are consistent across all demographic
# groups (gap < 0.1%). This is expected given the synthetic nature of the dataset, but the
# analysis framework itself is directly transferable to real-world clinical data where such
# disparities are documented.

# %% [markdown]
# ## 9. Generate Submission

# %%
if use_opt:
    test_expected = np.sum(ens_test * np.arange(5), axis=1)
    test_labels = np.digitize(test_expected, opt_t) + 1
else:
    test_labels = np.argmax(ens_test, axis=1) + 1

submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: test_labels})
print("Submission distribution:")
print(submission[TARGET_COL].value_counts().sort_index())
submission.to_csv("submission.csv", index=False)
print(f"\nsubmission.csv saved ({len(submission):,} rows)")

# %% [markdown]
# ## 10. Limitations and Future Work
#
# **1. Synthetic data limitations**: The dataset is synthetically generated, which produces
# cleaner separation between ESI classes than real clinical data. The QWK of 0.9998 would
# likely be substantially lower on real-world data where inter-rater disagreement, documentation
# variability, and edge cases are prevalent.
#
# **2. NEWS2 dominance**: NEWS2 alone achieves most of the predictive power. In a real
# deployment, the model's value would come from cases where NEWS2 is ambiguous (scores 4-6)
# or where chief complaint text contains critical information not captured by vital signs.
#
# **3. Missingness patterns**: In real EDs, vital sign missingness is strongly correlated
# with acuity (sicker patients may have vitals taken more urgently). Our imputation strategy
# should be validated against real missingness patterns.
#
# **4. Clinical integration**: A production triage support system would require real-time
# inference, integration with EHR systems, configurable alert thresholds, and extensive
# clinician-in-the-loop validation before any deployment.
#
# **5. Fairness on real data**: While this synthetic dataset shows no demographic bias,
# real-world triage data frequently exhibits systematic undertriage of elderly, non-English
# speaking, and uninsured patients. The fairness analysis framework presented here should
# be applied to real clinical cohorts.

# %% [markdown]
# ---
# **Citation**: Olaf Yunus Laitinen Imanov (2026). Triagegeist. https://kaggle.com/competitions/triagegeist, 2026. Kaggle.
