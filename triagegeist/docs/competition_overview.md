# Triagegeist — AI in Emergency Triage

## Overview

**Host**: Laitinen-Fredriksson Foundation (Helsinki)
**Tipo**: Community Hackathon (NO es una competicion tipica de Kaggle con leaderboard automatico)
**Premio**: $10,000 total ($5K + $3K + $2K)
**Deadline**: 21 abril 2026
**Equipos**: 23 participantes / 335 inscritos
**Tags**: Health, Medicine, Classification, Tabular, NLP

---

## IMPORTANTE — Esto NO es una Competicion Normal de Kaggle

Triagegeist es un **Hackathon juzgado por panel humano**, no una competicion con leaderboard automatico. No hay metric submission automatica. Los jueces evaluan:

1. **Clinical Relevance** (25 pts) — ¿El problema es real y relevante?
2. **Technical Quality** (30 pts) — ¿El codigo es riguroso, reproducible, limpio?
3. **Documentation/Writeup** (20 pts) — ¿El writeup es claro y completo?
4. **Insight and Findings** (15 pts) — ¿Los hallazgos son significativos?
5. **Novelty and Impact** (10 pts) — ¿Hay algo nuevo? ¿Podria usarse en clinica?

**Total: 100 puntos**

---

## Que hay que entregar

1. **Kaggle Notebook** publico — debe correr end-to-end sin errores
2. **Project Writeup** (max 2,000 palabras) via Writeups tab con:
   - Clinical problem statement
   - Methodology
   - Results
   - Limitations
   - Reproducibility notes
3. **Cover Image** (560x280px)
4. **Project Link** — URL a demo, repo GitHub, etc.

---

## El Reto

> "Can AI meaningfully support triage decisions in the emergency department?"

Enfoques validos:
- Modelo que predice **ESI acuity level** (1-5) desde datos de intake
- Pipeline NLP que extrae y clasifica **chief complaints** para flaggear riesgo alto
- Sistema de decision support que detecta **riesgo de deterioro** en pacientes en espera
- Notebook analitico que identifica **sesgos sistematicos** en el triaje

---

## Datos Proporcionados

### train.csv (80,000 filas)
Target: `triage_acuity` (1-5, ESI — Emergency Severity Index)
- 1 = Immediate (critico)
- 2 = Emergent
- 3 = Urgent
- 4 = Less Urgent
- 5 = Non-Urgent

**Columnas principales** (37 features + target):
| Grupo | Features |
|-------|----------|
| IDs | patient_id, site_id, triage_nurse_id |
| Arrival | arrival_mode, arrival_hour, arrival_day, arrival_month, arrival_season, shift |
| Demographics | age, age_group, sex, language, insurance_type |
| Clinical context | transport_origin, pain_location, mental_status_triage, chief_complaint_system |
| Prior visits | num_prior_ed_visits_12m, num_prior_admissions_12m |
| Medications/comorbidities | num_active_medications, num_comorbidities |
| Vitales | systolic_bp, diastolic_bp, mean_arterial_pressure, pulse_pressure, heart_rate, respiratory_rate, temperature_c, spo2 |
| Scores | gcs_total, pain_score, news2_score |
| Body | weight_kg, height_cm, bmi, shock_index |
| Outcome | disposition (discharged/admitted/etc), ed_los_hours |

### test.csv (20,000 filas)
Mismas columnas excepto `triage_acuity`, `disposition`, `ed_los_hours`

### chief_complaints.csv (100,000 filas)
- patient_id, chief_complaint_raw (texto libre), chief_complaint_system

### patient_history.csv (100,000 filas)
- patient_id + 25 columnas binarias de historial medico:
  hx_hypertension, hx_diabetes_type2, hx_diabetes_type1, hx_asthma, hx_copd,
  hx_heart_failure, hx_atrial_fibrillation, hx_ckd, hx_liver_disease,
  hx_malignancy, hx_obesity, hx_depression, hx_anxiety, hx_dementia,
  hx_epilepsy, hx_hypothyroidism, hx_hyperthyroidism, hx_hiv,
  hx_coagulopathy, hx_immunosuppressed, hx_pregnant,
  hx_substance_use_disorder, hx_coronary_artery_disease, hx_stroke_prior,
  hx_peripheral_vascular_disease

### sample_submission.csv (20,000 filas)
- patient_id, triage_acuity (prediccion 1-5)

---

## Datos Externos Recomendados

- **MIMIC-IV-ED** — Datos de ED de Beth Israel Deaconess (PhysioNet, acceso con credenciales)
- **NHAMCS** — National Hospital Ambulatory Medical Care Survey (CDC, publico)
- **CTG/UCI** — Datasets fisiologicos del UCI ML Repository

---

## Baseline del Notebook Proporcionado

Pipeline del baseline:
1. Join train + patient_history + chief_complaints
2. Feature engineering (pain_not_recorded, missingness indicators, derived vitals, elderly/pediatric flags)
3. TF-IDF sobre chief_complaint_raw (top 80 features)
4. LightGBM multiclass con class_weight='balanced'
5. 5-fold StratifiedKFold
6. **Metrica baseline: QWK ~0.71** (Quadratic Weighted Kappa)

---

## Notas Clave para la Estrategia

1. **Es un hackathon, no una competicion AUC** — la calidad del writeup, la relevancia clinica y la novedad pesan tanto como el modelo
2. **30 pts por Technical Quality** — codigo limpio, reproducible, bien comentado es critico
3. **25 pts por Clinical Relevance** — hay que demostrar que entendemos el problema clinico real
4. **Solo 23 equipos** — con un buen trabajo, top 3 es alcanzable
5. **NLP del chief complaint** es un diferenciador — embeddings > TF-IDF
6. **Fairness/bias analysis** suma puntos de novedad — analizar si el modelo sub-triaja ciertas poblaciones
7. **SHAP/explicabilidad** es casi obligatorio para un contexto clinico

---

## Jueces

- Olaf Yunus Laitinen Imanov — Data Science Specialist, DTU
- OzanM. — Data Scientist
- Kutluk Atalay — Data Scientist
- S. Burde Dulger — Student

---

## Citation

Olaf Yunus Laitinen Imanov (2026). Triagegeist. https://kaggle.com/competitions/triagegeist, 2026. Kaggle.

---

## Links

- [Competicion](https://kaggle.com/competitions/triagegeist)
- [Overview](https://kaggle.com/competitions/triagegeist/overview)
- [Data](https://kaggle.com/competitions/triagegeist/data)
