# Requirements Document — Kaggle PGS6E6 "Predicting Stellar Class"

> Project type: PERSONAL (Kaggle competition).
> Cycle: C1 Discovery + Planning.
> Date: 2026-06-06. Deadline: 2026-06-30 (~24 days).

---

## Business Requirements

Note: "Business" here maps to competition objectives, since this is a personal Kaggle
project with `reward=Swag`. The real ROI is leaderboard rank, reproducible pipeline,
and reusable code-store assets — not revenue.

- **B1**: Achieve a public leaderboard `balanced_accuracy >= 0.97` by the submission
  deadline 2026-06-30, placing the solution in the top ~25% of the 850-team field
  (top-25% threshold estimated; refine in C2 once first public scores are observed).
- **B2**: Produce a fully reproducible end-to-end pipeline (fixed seed, DVC-versioned
  data, MLflow-registered models) so any submission score can be regenerated bit-for-bit
  with one command. Reproducibility is a hard business gate, not a nice-to-have.
- **B3**: Harvest at least 3 atomic, cross-project-reusable utilities into
  a shared utilities library (e.g. photometric-color feature transformer, balanced-accuracy
  threshold tuner, stratified-split validator) for future tabular competitions.
- **B4**: Zero leakage in the final submission. A leaderboard score inflated by a target
  proxy (notably the categorical `galaxy_population`, suspected GALAXY-only) is worth
  nothing and is treated as a business failure (B4 supersedes any short-term score gain
  from a leaky/near-leak feature).

## User Requirements

The sole user is the competitor (single-developer project). User stories framed accordingly.

- **U1**: As the competitor, I want a one-command submission generator
  (`make submit` or `python -m src.submit`) that reads `test.csv`, runs the registered
  best model, and writes a valid `submission.csv` with columns `id,class`, so that I can
  iterate fast without manual glue.
- **U2**: As the competitor, I want every experiment tracked in MLflow with the CV
  `balanced_accuracy` and per-class recall logged, so that I can compare runs and pick
  the model that maximizes balanced accuracy without QSO collapse.
- **U3**: As the competitor, I want an automated leakage audit report (per-feature
  mutual information vs `class`, plus a per-class value-distribution + informative-NaN
  check on the two categoricals `spectral_type` and `galaxy_population`) before any model
  trains, so that I never ship a target-proxy feature by accident.
- **U4**: As the competitor, I want a held-out local validation set whose CV score
  tracks the public leaderboard within +/- 0.005 balanced_accuracy, so that I can trust
  local CV and avoid burning daily submission quota on blind guesses.
- **U5**: As the competitor, I want a per-class confusion-matrix view after each model
  run, so that I can see immediately whether QSO (the minority class) is being predicted
  at acceptable recall.

## System Requirements

NFRs for a batch, offline, single-machine Kaggle pipeline. No serving endpoint exists.

- **S1** (Performance — batch): Full training of the final model on the train set
  (577,348 rows, ~100MB) shall complete in `<= 30 min` wall-clock on the RTX 2000 Ada 8GB
  GPU. If a single model exceeds this, it must be excluded from the per-fold tuning loop
  and run only once for the final fit.
- **S2** (Performance — inference): Generating predictions for the full `test.csv`
  (247,436 rows, ~41MB) shall complete in `<= 5 min` wall-clock (batch, offline). No
  real-time latency SLA exists (N/A — this is batch scoring).
- **S3** (Resource): Peak GPU memory shall stay `<= 8 GB` (RTX 2000 Ada VRAM ceiling).
  Peak host RAM `<= 24 GB`. Any pipeline step that OOMs is a blocking defect.
- **S4** (Reproducibility): Given a fixed global seed and the committed config, two runs
  of the full pipeline shall produce CV `balanced_accuracy` differing by `<= 0.0005`
  (tolerance for nondeterministic GPU reductions). Submission CSVs must be byte-identical
  when run on the same hardware/seed.
- **S5** (Output contract): `submission.csv` shall have exactly the columns `id,class`,
  one row per `test.csv` row, `class` values drawn only from `{GALAXY, STAR, QSO}`
  (exact label strings from train), UTF-8, no index column, RFC 4180. A schema validator
  must reject any malformed submission before it reaches disk.
- **S6** (Integration): The pipeline shall depend only on libraries installable in the
  project venv (LightGBM/XGBoost/CatBoost with GPU, scikit-learn, Optuna, pandas/polars,
  MLflow, DVC). No external network calls at inference time.

## ML-specific Requirements

- **M1** (Task & primary metric): The model shall perform 3-class classification over
  `{GALAXY, STAR, QSO}` from the column `class`, optimizing `balanced_accuracy`
  (macro-averaged recall). Target: `balanced_accuracy >= 0.97` on local stratified CV,
  with the public leaderboard expected within +/- 0.005 of CV (validated in C8).
  Stretch: `>= 0.98`. Rationale and leaderboard justification in `ml-problem-statement.md`.
- **M2** (Per-class fairness / no minority collapse): Real distribution is GALAXY 65.38%,
  QSO 20.29%, **STAR 14.33% (minority)**. Per-class recall shall satisfy
  `recall(STAR) >= 0.93` AND `recall(QSO) >= 0.94` AND `recall(GALAXY) >= 0.96` on local
  CV. The spread `max_recall - min_recall <= 0.06`. This prevents a high-accuracy model
  that silently sacrifices STAR or QSO. M2 is a hard gate: a model that beats M1 but
  violates M2 is rejected.
- **M3** (No leakage — blocking): The trivial row index `id` shall be dropped (not a
  feature). The two categoricals `spectral_type` and `galaxy_population` shall NOT be used
  as predictive features unless the C2 leakage audit proves they are not target proxies.
  Specific suspicion to clear: `galaxy_population` may be populated ONLY for `class=GALAXY`
  (an informative-NaN / near-leak), and `spectral_type` may map near-deterministically to
  a class. Default posture: both categoricals held out pending audit; any inclusion (or
  inclusion of an engineered NaN-indicator derived from them) requires an explicit ADR.
  Acceptance: leakage audit report shows no feature with near-deterministic mapping to
  `class`. (Note: the originally feared SDSS spurious IDs — obj_ID, run_ID, plate, MJD,
  fiber_ID, spec_obj_ID — do NOT exist in this dataset; threat degraded.)
- **M4** (Calibrated decisioning for balanced accuracy): The decision rule shall be tuned
  for balanced accuracy via class weights and/or per-class probability thresholds (not
  default argmax), with the tuning done ONLY on validation folds (never on test). Target:
  tuned decision rule improves CV balanced_accuracy by `>= 0.003` over naive argmax.
- **M5** (Reproducibility & versioning): Seed fixed (global `SEED=42`); train/test data
  versioned with DVC; every model + CV metrics + feature list registered in MLflow.
  Acceptance: `dvc status` clean and MLflow run ID recorded in the submission metadata.
- **M6** (Synthetic-data robustness): Dataset is synthetic (Playground Series, generated
  from a model trained on SDSS). The pipeline shall be validated for train/test
  distribution parity (adversarial-validation AUC `<= 0.55` between train and test;
  AUC `> 0.6` triggers a covariate-shift investigation in C2).

## Out of scope

Explicitly NOT done in this project, to prevent scope creep:

- No real-time / online serving endpoint, REST API, or latency-SLA service. Batch only.
- No deep-learning tabular models (TabNet, FT-Transformer, SAINT) in the MUST path.
  Gradient-boosted trees are the committed approach; DL tabular is a COULD-HAVE spike only.
- No external astronomical catalogs or the original UCI SDSS17 dataset joined in
  (would be leakage / against typical Playground rules — treat as forbidden unless the
  competition rules explicitly allow external data; default forbidden).
- No fairness work on protected human attributes (none exist; "fairness" here means
  per-class recall balance only, with STAR as the true minority — see M2).
- No model interpretability deliverable beyond SHAP feature importance and per-class
  confusion matrices (no SAE/circuit analysis — that is for LLM projects).
- No deployment, monitoring, or MLOps serving infra (C9-C12 collapse to "reproducible
  submission pipeline"; no production runtime exists).

## Stakeholders

| Rol | Nombre | Decision power | Estrategia | Concerns |
|-----|--------|----------------|------------|----------|
| Competitor / Product Owner / End user | Adrián Infantes | High (sole decision-maker) | N/A (self) | LB rank, reproducibility, zero leakage |
| C1 gate (requirements) | project planning | Blocking (sign-off) | — | Requirements completeness, quantified criteria |
| Data leakage auditor | leakage audit step | Blocking (C2) | — | ID leakage, train/test drift, duplicates |
| Statistical review gate | statistical validation | Blocking (C3/C5/C6/C8) | — | CV correctness, metric definition, no test peeking |
| Code review gate | code review | Blocking (between cycles) | — | Reproducibility, code quality, leakage in code |
| Competition host | Kaggle | High (external, fixed) | Keep satisfied | Rules compliance, submission format, deadline |
| Longevity gate | longevity review | Blocking (C8) | — | code-store reusability, naming, cohesion |

---

## C1 Exit Checklist (requirements-engineering skill)

- [x] Stakeholder register complete
- [x] Requirements at all 4 levels (Business / User / System / ML-specific)
- [x] ML Problem Statement complete (separate artifact — blocking, see `ml-problem-statement.md`)
- [x] Non-functional requirements explicit (S1-S6)
- [x] No vague requirements (all quantified: balanced_accuracy >= 0.97, recall(STAR) >= 0.93, recall(QSO) >= 0.94, etc.)
- [x] No implementation decisions encoded as requirements (algorithm choice lives in backlog/ADR, M1 is metric-not-algorithm)
- [x] Out of scope explicit
- [ ] Final sign-off (pending)
