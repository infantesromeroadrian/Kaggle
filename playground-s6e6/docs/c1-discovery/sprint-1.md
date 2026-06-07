# Sprint 1 — Kaggle PGS6E6 "Predicting Stellar Class"

> Date: 2026-06-06. Estimation: Planning Poker (Fibonacci).
> Single-developer project. Deadline 2026-06-30.

## Sprint 1 — 2026-06-06 to 2026-06-19 (2 weeks)

- **Duration**: 2 weeks (10 working days).
- **Capacity**: single-dev, ~4 focused hours/day x 10 days = ~40h gross.
  Minus ceremonies/review/imprevistos (~25%) -> ~30h net effective.
- **Velocity assumption**: first sprint, no history -> baseline **13 points** committed,
  with a stretch pull to 16 if the leakage audit (BL-003) resolves cleanly and early.
- **Lane allocation** (agile-ml skill): Data ~45%, Model ~35%, Eval ~20% this sprint
  (front-loaded on data correctness because the leakage audit gates everything).

## Sprint Goal

> "By the end of Sprint 1, a reproducible LightGBM baseline produces a valid `submission.csv`
> from cleaned, leakage-audited features, with local stratified-CV `balanced_accuracy`
> measured and per-class recall (STAR/QSO) logged — establishing the floor every later
> experiment must beat."

## Tickets

Estimates use Fibonacci with explicit uncertainty ranges (experiments ±100%,
baseline ±30%, deterministic infra ±20%). Each ticket maps to a backlog ID.

### TICKET-001 (= BL-002) — Reproducible data loading + schema validation
- Owner: data engineering
- Estimate: **3** (±20%, deterministic infra: 2-3)
- Type: infra / data
- Acceptance criteria (quantified):
  - [ ] Loader reads train.csv (577,348 rows) and test.csv (247,436 rows) with an explicit
        dtype contract; row counts asserted exactly.
  - [ ] `class` confirmed present in train, absent in test; `id` present in both.
  - [ ] Global `SEED=42` set; loader is deterministic (two loads -> identical hash).
  - [ ] Schema validator rejects any unexpected column or dtype with a clear error.
- Definition of Done:
  - [ ] tests >= 80% coverage on new code
  - [ ] code review approval logged
  - [ ] longevity review approval
  - [ ] PR merged to main
- Dependencies: BL-001 (done)

### TICKET-002 (= BL-003) — LEAKAGE AUDIT (BLOCKING)
- Owner: leakage auditor
- Estimate: **5** (±30%, has investigative uncertainty: 3-5)
- Type: eval (leakage)
- Acceptance criteria (quantified):
  - [ ] Per-class value distribution computed for `spectral_type` and `galaxy_population`.
  - [ ] Informative-NaN check: report whether `galaxy_population` is non-null ONLY for
        `class=GALAXY` (the GALAXY-only-proxy suspicion) — explicit YES/NO with counts.
  - [ ] Mutual information of every feature vs `class` ranked; any feature with MI implying
        near-deterministic mapping (>0.95 normalized) flagged as leak.
  - [ ] Adversarial validation train-vs-test: report AUC; flag if AUC > 0.6 (covariate shift).
  - [ ] Written verdict per categorical: USE / DROP / USE-ONLY-NaN-INDICATOR-WITH-ADR.
- Definition of Done:
  - [ ] statistical review approval (MI + adversarial-validation methodology correct)
  - [ ] code review approval logged
  - [ ] audit report written to docs/c2-data/leakage-audit.md
  - [ ] PR merged to main
- Dependencies: TICKET-001
- **BLOCKING**: TICKET-004 (feature eng) cannot finalize feature set until this clears.

### TICKET-003 (= BL-005) — Stratified K-fold split
- Owner: data engineering
- Estimate: **2** (±20%, deterministic: 2-3)
- Type: data
- Acceptance criteria (quantified):
  - [ ] 5-fold stratified split on `class`; each fold's class proportions within ±0.5%
        of the global 65.38/20.29/14.33.
  - [ ] Fold assignment frozen + persisted (reproducible from seed=42).
  - [ ] Zero duplicate rows shared across folds (verified).
- Definition of Done:
  - [ ] tests >= 80% coverage
  - [ ] code review approval logged
  - [ ] PR merged to main
- Dependencies: TICKET-001

### TICKET-004 (= BL-004) — EDA on TRAIN only
- Owner: data science
- Estimate: **3** (±30%: 2-5)
- Type: data
- Acceptance criteria (quantified):
  - [ ] Target distribution confirmed: GALAXY 65.38% / QSO 20.29% / STAR 14.33% (±0.1%).
  - [ ] redshift distribution by class plotted; STAR≈0 / GALAXY low+ / QSO high+ verified.
  - [ ] Photometric band + color distributions per class documented.
  - [ ] Missingness map for all 10 features (especially galaxy_population NaN pattern).
  - [ ] EDA touches TRAIN only (no test peeking — asserted in code).
- Definition of Done:
  - [ ] code review approval logged
  - [ ] EDA notebook/report written to docs/c2-data/eda.md
  - [ ] PR merged to main
- Dependencies: TICKET-001

### TICKET-005 (= BL-006) — Feature engineering (colors + cleared categoricals)
- Owner: ML engineering
- Estimate: **5** (±30%: 3-8, depends on TICKET-002 verdict)
- Type: model / data
- Acceptance criteria (quantified):
  - [ ] Photometric colors derived: u-g, g-r, r-i, i-z (4 features).
  - [ ] redshift transform(s) added (e.g. log1p, binning) with rationale.
  - [ ] Categorical encoding applied ONLY to features cleared by TICKET-002; leaky ones
        dropped (or NaN-indicator only, if ADR approved).
  - [ ] Feature transforms reversible / leak-free (fit on train folds only, no test fit).
  - [ ] Feature list documented and versioned.
- Definition of Done:
  - [ ] statistical review approval (transform correctness, no leakage)
  - [ ] code review approval logged
  - [ ] longevity review approval (color transformer is a code-store candidate)
  - [ ] PR merged to main
- Dependencies: TICKET-002, TICKET-004

### TICKET-006 (= BL-008) — Baseline LightGBM (GPU)
- Owner: ML engineering
- Estimate: **3** (±30%, baseline-known: 2-5)
- Type: model
- Acceptance criteria (quantified):
  - [ ] LightGBM (GPU) trained with class_weight=balanced across the 5 folds.
  - [ ] CV balanced_accuracy computed and logged to MLflow.
  - [ ] Per-class recall (GALAXY/QSO/STAR) + 3x3 confusion matrix logged per fold.
  - [ ] Single-model train time <= 30 min (S1) verified.
  - [ ] Run is reproducible from config + seed=42.
- Definition of Done:
  - [ ] statistical review approval (CV setup, no fold leakage, metric correct)
  - [ ] code review approval logged
  - [ ] MLflow run ID recorded
  - [ ] PR merged to main
- Dependencies: TICKET-003, TICKET-005

### TICKET-007 (= BL-009) — Submission pipeline + schema validator
- Owner: ML engineering
- Estimate: **3** (±20%, deterministic infra: 2-3)
- Type: infra
- Acceptance criteria (quantified):
  - [ ] One command produces submission.csv from the registered baseline model on test.csv.
  - [ ] submission.csv has exactly columns id,class; 247,436 rows; class in {GALAXY,STAR,QSO}.
  - [ ] Schema validator (S5) rejects any malformed submission before write.
  - [ ] Full test scoring time <= 5 min (S2) verified.
- Definition of Done:
  - [ ] tests >= 80% coverage (including the validator)
  - [ ] code review approval logged
  - [ ] PR merged to main
- Dependencies: TICKET-006

## Sprint 1 capacity check

| Ticket | Backlog | Owner | Fibonacci | Range |
|--------|---------|-------|-----------|-------|
| TICKET-001 | BL-002 | data engineering | 3 | 2-3 |
| TICKET-002 | BL-003 | leakage auditor | 5 | 3-5 |
| TICKET-003 | BL-005 | data engineering | 2 | 2-3 |
| TICKET-004 | BL-004 | data science | 3 | 2-5 |
| TICKET-005 | BL-006 | ML engineering | 5 | 3-8 |
| TICKET-006 | BL-008 | ML engineering | 3 | 2-5 |
| TICKET-007 | BL-009 | ML engineering | 3 | 2-3 |
| **Committed total** | | | **13 (MUST core: 001+002+003+006)** | |
| **Full sprint total** | | | **24** | 16-32 |

> Committed = the critical path to a measured baseline submission (TICKET-001,002,003,006 =
> 3+5+2+3 = 13 points = the baseline velocity). TICKET-004/005/007 are pulled in the same
> sprint if the leakage audit (002) resolves early; if 002 drags, 004/005/007 spill to a
> mid-sprint replan. Sprint stays FROZEN: scope change = freeze + replan, no mid-sprint adds.

## Kill criteria / time-boxes

- TICKET-002 (leakage audit): time-box 2 days. If the categoricals' status is still
  ambiguous after 2 days, escalate for a USE/DROP decision (an ambiguous
  near-leak defaults to DROP — B4 posture).
- No experiment tickets in Sprint 1 (all are baseline/infra/eval), so no ±100% research
  risk this sprint. The first true experiment (HPO, ensembles) lands in Sprint 2.

## Definition of Done (sprint-level)

- [ ] All committed (MUST core) tickets meet their per-ticket DoD.
- [ ] A measured CV balanced_accuracy baseline exists and is logged in MLflow.
- [ ] A valid submission.csv has been generated at least once (format-validated).
- [ ] Leakage audit verdict written (the gate for all later feature work).
- [ ] No leaky feature used without an ADR.
- [ ] Sprint Review + Retro held; velocity recorded for Sprint 2 planning.

## Risks for Sprint 1

| Risk | Mitigation |
|------|------------|
| Leakage audit (TICKET-002) reveals galaxy_population is a usable-but-risky signal | Default DROP; only USE with ADR + adversarial-validation evidence it transfers to test |
| Baseline already hits 0.97 trivially (categoricals leak) | That is a RED FLAG, not success — re-run baseline WITHOUT categoricals to get the honest floor |
| GPU LightGBM setup friction | TICKET-006 has a CPU fallback (dataset is small; CPU finishes in minutes — S1 still met) |
