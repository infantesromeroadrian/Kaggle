from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
DOCS_DIR = PROJECT_ROOT / "docs"

RANDOM_SEED = 42
TARGET_COL = "triage_acuity"
ID_COL = "patient_id"

TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
CC_FILE = DATA_DIR / "chief_complaints.csv"
HISTORY_FILE = DATA_DIR / "patient_history.csv"
SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"

ESI_LABELS: dict[int, str] = {
    1: "Immediate",
    2: "Emergent",
    3: "Urgent",
    4: "Less Urgent",
    5: "Non-Urgent",
}

VITAL_COLS: list[str] = [
    "systolic_bp",
    "diastolic_bp",
    "mean_arterial_pressure",
    "pulse_pressure",
    "heart_rate",
    "respiratory_rate",
    "temperature_c",
    "spo2",
]
SCORE_COLS: list[str] = ["gcs_total", "pain_score", "news2_score"]
BODY_COLS: list[str] = ["weight_kg", "height_cm", "bmi", "shock_index"]
HX_COLS: list[str] = [
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
    "hx_obesity",
    "hx_depression",
    "hx_anxiety",
    "hx_dementia",
    "hx_epilepsy",
    "hx_hypothyroidism",
    "hx_hyperthyroidism",
    "hx_hiv",
    "hx_coagulopathy",
    "hx_immunosuppressed",
    "hx_pregnant",
    "hx_substance_use_disorder",
    "hx_coronary_artery_disease",
    "hx_stroke_prior",
    "hx_peripheral_vascular_disease",
]
