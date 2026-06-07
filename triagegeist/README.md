# Triagegeist — AI-Assisted Emergency Triage Acuity Prediction

> Predicting ESI triage acuity (1-5) from structured patient intake data and NLP-processed chief complaints, with full SHAP explainability and fairness analysis.

**Competition**: [Triagegeist — Kaggle Hackathon](https://kaggle.com/competitions/triagegeist) | **Host**: Laitinen-Fredriksson Foundation | **Prize**: $10,000

---

## Results

| Metric | Baseline | Ours |
|--------|----------|------|
| QWK | ~0.71 | **0.9998** |
| Macro F1 | ~0.65 | **0.999** |
| ESI-1 Recall | Unknown | **99.5%** |

## Approach

1. **516 features**: 132 structured clinical features + 384-dim sentence embeddings from chief complaints (`all-MiniLM-L6-v2`)
2. **Ensemble**: LightGBM + XGBoost + CatBoost with 10-fold stratified CV and ordinal threshold optimization
3. **Explainability**: SHAP analysis (global, per-class, ESI-1 focused)
4. **Fairness audit**: Accuracy and undertriage rates by sex, age, language, insurance type

## Quick Start

```bash
# Clone
git clone https://github.com/infantesromeroadrian/triagegeist.git
cd triagegeist

# Setup (requires Python 3.11+ and uv)
uv venv && uv pip install -e ".[dev]"

# Download data from Kaggle
# Place train.csv, test.csv, chief_complaints.csv, patient_history.csv in data/

# Run pipeline
uv run python src/eda.py              # EDA + plots
uv run python src/features.py         # Feature engineering + NLP embeddings
uv run python src/train.py            # Training + submission
uv run python src/explainability.py   # SHAP + fairness analysis
```

## Project Structure

```
triagegeist/
  data/                    # Competition data (not tracked in git)
  src/
    config.py              # Paths, seeds, feature groups
    eda.py                 # Clinical EDA (13 plots)
    features.py            # Feature engineering + NLP embeddings
    train.py               # Training pipeline (GBDT ensemble)
    explainability.py      # SHAP + fairness analysis
  notebooks/
    triagegeist_submission.py   # Self-contained Kaggle notebook
  docs/
    competition_overview.md     # Competition analysis
    writeup.md                  # Project writeup (1,092 words)
  assets/                  # Generated plots
  models/                  # Cached embeddings
```

## Key Clinical Features

| Feature | Correlation with ESI | Clinical Meaning |
|---------|---------------------|------------------|
| NEWS2 score | r = -0.815 | National Early Warning Score — validated deterioration predictor |
| GCS total | r = 0.657 | Glasgow Coma Scale — consciousness level |
| SpO2 | r = 0.654 | Oxygen saturation — respiratory function |
| Respiratory rate | r = -0.653 | Tachypnea — sepsis/deterioration marker |
| Shock index | r = -0.632 | HR/SBP — hemodynamic stability |

## Requirements

- Python >= 3.11
- GPU recommended (NVIDIA with CUDA for training + sentence-transformers)
- ~4GB VRAM minimum

## License

MIT

## Citation

```bibtex
@misc{triagegeist2026,
  title={Triagegeist: AI-Assisted Emergency Triage Acuity Prediction},
  author={Adrian Infantes},
  year={2026},
  url={https://github.com/infantesromeroadrian/triagegeist}
}
```
