# Triagegeist: AI-Assisted Emergency Triage Acuity Prediction

## Clinical Problem Statement

Emergency department triage is the critical first-touch decision that determines patient care priority. Triage nurses assign an Emergency Severity Index (ESI) level from 1 (immediate, life-threatening) to 5 (non-urgent) based on a rapid assessment of vital signs, chief complaint, and clinical gestalt. This decision occurs under extreme time pressure, cognitive load, and staffing constraints.

The clinical consequences of triage errors are asymmetric. **Undertriage** — assigning a lower severity than warranted — leads to delayed care for critically ill patients and is directly associated with adverse outcomes and preventable deaths. The literature documents significant inter-rater variability in ESI assignment, with disagreement rates of 15-30% between experienced triage nurses on identical patient presentations.

This project addresses the question: **can a machine learning system trained on structured intake data and free-text chief complaints reliably predict ESI acuity level, and can its decision process be made transparent enough for clinical trust?**

## Methodology

### Data and Features

We used the provided Triagegeist dataset (80,000 training / 20,000 test patients) with three linked tables: structured patient intake data (vitals, demographics, arrival context), patient medical history (25 binary comorbidity flags), and free-text chief complaints.

Our feature engineering pipeline produced **516 features** in two stages:

**Stage 1 — Structured clinical features (132 features):**
- **Missingness indicators**: Blood pressure, respiratory rate, temperature, and pain score missingness as binary flags. In real clinical practice, unmeasured vitals can indicate patient acuity (unconscious patients cannot self-report pain scores).
- **Clinical threshold flags**: Hypotension (SBP <= 90), tachycardia (HR >= 100), hypoxia (SpO2 < 94), tachypnea (RR >= 22), febrile (T >= 38C) — all standard clinical deterioration criteria.
- **Composite risk scores**: Cardiovascular risk (sum of 6 cardiac history flags), respiratory risk, psychiatric risk, and total comorbidity burden.
- **Clinical interaction terms**: Elderly patients with high NEWS2, cardiovascular risk multiplied by shock index, comorbidity burden scaled by NEWS2.
- **Chief complaint keyword flags**: 20 high-risk clinical keywords (chest pain, seizure, stroke, cardiac arrest, anaphylaxis, etc.) extracted via pattern matching.

**Stage 2 — NLP sentence embeddings (384 features):**
We encoded each patient's chief complaint text using `all-MiniLM-L6-v2`, a 384-dimensional sentence transformer. This captures semantic meaning that keyword matching cannot — for example, "thunderclap headache worsening with movement" maps to a different embedding region than "chronic headache review," enabling the model to distinguish emergent from non-urgent neurological presentations without explicit rule engineering.

### Model Architecture

We trained three gradient boosted tree models — LightGBM, XGBoost, and CatBoost — each with 10-fold stratified cross-validation and balanced class weighting to address the ESI-1 minority class (4% of cases). Final predictions use a probability-weighted ensemble with ordinal threshold optimization to exploit the ordered nature of ESI levels.

### Evaluation Metric

Quadratic Weighted Kappa (QWK) penalizes predictions that are far from the true label more heavily than near-misses, making it appropriate for ordinal classification where predicting ESI-1 as ESI-5 is far worse than predicting ESI-1 as ESI-2.

## Results

| Component | QWK |
|-----------|-----|
| LightGBM | 0.9998 |
| XGBoost | 0.9997 |
| CatBoost | 0.9992 |
| **Ensemble (optimized)** | **0.9998** |
| Baseline (provided notebook) | ~0.71 |

### Per-Class Performance

| ESI Level | Recall | Clinical Significance |
|-----------|--------|----------------------|
| ESI-1 (Immediate) | 99.5% | 17 of 3,222 patients misclassified as ESI-2 (undertriage) |
| ESI-2 (Emergent) | 99.9% | 14 of 13,439 patients misclassified |
| ESI-3 (Urgent) | 100.0% | Near-perfect |
| ESI-4 (Less Urgent) | 100.0% | Perfect |
| ESI-5 (Non-Urgent) | 100.0% | Perfect |

All 17 ESI-1 errors were classified as ESI-2 (one level off), not as lower acuity. No patient was dangerously undertriaged by more than one level.

### Feature Importance (SHAP Analysis)

SHAP analysis confirms clinical face validity:
1. **NEWS2 score** — by far the strongest predictor, consistent with its clinical role as a validated deterioration score
2. **GCS total** — separates ESI-1 (coma/obtunded) from higher-functioning patients
3. **SpO2** — respiratory compromise indicator
4. **Respiratory rate** — tachypnea is a qSOFA/sepsis criterion
5. **Shock index** — hemodynamic instability marker

NLP embeddings ranked in the top 30 features, confirming that chief complaint semantics add predictive value beyond vital signs alone.

### Fairness Audit

We assessed model accuracy and undertriage rates across four demographic axes: sex, age group, language, and insurance type. The maximum accuracy gap across any group was **0.09%** (insurance type). ESI-1 recall was slightly lower for uninsured patients (98.5% vs 99.5% overall) and elderly patients (99.3%), though sample sizes limit the statistical power of these comparisons.

While the synthetic nature of this dataset limits the external validity of these fairness findings, the analytical framework is directly transferable to real-world clinical cohorts where systematic undertriage of vulnerable populations is a documented patient safety concern.

## Limitations

1. **Synthetic data**: The near-perfect QWK reflects the clean separation in synthetic data. Real-world performance would be substantially lower due to documentation variability, inter-rater disagreement, and clinical edge cases. The model's true value would emerge in ambiguous presentations where NEWS2 scores are in the 4-6 range.

2. **NEWS2 dominance**: NEWS2 alone accounts for most predictive power. In a clinical deployment, the model's marginal value comes from cases where physiological scores are borderline and chief complaint text provides the differentiating signal.

3. **Temporal and institutional factors**: This model was trained on data from 5 Finnish hospital sites. Triage practices, patient populations, and disease epidemiology vary across institutions and geographies. Transfer learning and site-specific calibration would be required for deployment outside this context.

4. **Missing real-world validation**: No clinician-in-the-loop evaluation was performed. Before any clinical deployment, prospective validation against expert triage decisions, with particular attention to the model's performance on disagreement cases, would be essential.

## Reproducibility

- All code is contained in a single Kaggle notebook that runs end-to-end with GPU acceleration
- Random seed (42) is fixed for all stochastic operations
- No external data sources beyond the provided competition datasets
- All dependencies are standard Kaggle-available packages (LightGBM, XGBoost, CatBoost, sentence-transformers, SHAP)
- GitHub repository: [github.com/adrianinfantes/triagegeist](https://github.com/adrianinfantes/triagegeist)

## References

1. Gilboy N, et al. Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care. AHRQ Publication No. 12-0014. 2011.
2. Royal College of Physicians. National Early Warning Score (NEWS) 2. 2017.
3. Hinson JS, et al. Accuracy of emergency department triage using the Emergency Severity Index and independent predictors of under-triage and over-triage. Annals of Emergency Medicine. 2019.
4. Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.
5. Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
