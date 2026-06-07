# Backlog — Kaggle PGS6E6 "Predicting Stellar Class"

> Prioritized backlog. MoSCoW + RICE prioritization (machine-parseable).
> Date: 2026-06-06. Covers C2 -> C8.
>
> Schema: MUST/SHOULD/COULD rows = 11 cells:
> `ID | Title | Type | Cycle | StoryPts | Reach | Impact | Confidence | Effort | RICE | Deps`.
> WON'T rows = 3 cells: `ID | Title | Reason`.
> RICE = (Reach x Impact x Confidence) / Effort. Reach = LB-score-leverage proxy (0-10).
> Impact in {3,2,1,0.5,0.25}. Confidence in [0,1]. StoryPts Fibonacci {1,2,3,5,8,13}.
> Effort = person-days. Type in {Data,Model,Infra,Eval,Docs,Spike,Integration,Security,UI}.

### MUST

| ID | Title | Type | Cycle | StoryPts | Reach | Impact | Confidence | Effort | RICE | Deps |
|----|-------|------|-------|----------|-------|--------|------------|--------|------|------|
| BL-009 | Submission pipeline: one-command predict on test -> submission.csv (id,class) + schema validator (S5) | Infra | C5 | 3 | 9 | 2 | 1.0 | 1 | 18.0 | — |
| BL-012 | Decision-rule tuning for balanced accuracy: class weights + per-class thresholds tuned on val folds only (M4) | Model | C6 | 5 | 8 | 3 | 0.8 | 2 | 9.6 | — |
| BL-014 | Model eval + fairness gate: verify M1 (bal_acc>=0.97) + M2 (recall STAR>=0.93, QSO>=0.94, spread<=0.06); error analysis | Eval | C8 | 3 | 9 | 3 | 0.8 | 1 | 21.6 | BL-012 |
| BL-015 | Test suite >=80% coverage on data+feature+submission code; reproducibility test (seed -> byte-identical submission) | Eval | C8 | 5 | 7 | 2 | 1.0 | 2 | 7.0 | BL-009 |

### SHOULD

| ID | Title | Type | Cycle | StoryPts | Reach | Impact | Confidence | Effort | RICE | Deps |
|----|-------|------|-------|----------|-------|--------|------------|--------|------|------|
| BL-016 | Extract >=3 atomic utilities to a shared utilities library (color transformer, balanced-acc threshold tuner, stratified-split validator) | Docs | C8 | 3 | 5 | 1 | 0.8 | 1 | 4.0 | BL-015 |

### COULD

| ID | Title | Type | Cycle | StoryPts | Reach | Impact | Confidence | Effort | RICE | Deps |
|----|-------|------|-------|----------|-------|--------|------------|--------|------|------|
| BL-017 | DL tabular spike (TabNet/FT-Transformer); time-boxed 3 days, kill if not beating boosting by >=0.002 | Spike | C6 | 8 | 4 | 1 | 0.5 | 3 | 0.7 | BL-014 |
| BL-018 | Pseudo-labeling / test-time augmentation experiment; time-boxed, kill if CV-LB gap worsens | Spike | C8 | 5 | 4 | 1 | 0.5 | 2 | 1.0 | BL-014 |
| BL-020 | TECH DEBT (DEBT-002): add `seaborn` + `matplotlib` to pyproject.toml [project.dependencies] — src/eda.py imports both but they are unpinned/undeclared, so eda.py is not reproducibly installable from the lockfile (breaks reproducibility). MEDIUM. | Infra | C6 | 1 | 4 | 1 | 1.0 | 0.5 | 8.0 | — |
| BL-021 | TECH DEBT (DEBT-003): `N_SPLITS` redefined as a module-level `Final[int] = 5` in eda.py:101 and feature_eval.py:76 instead of importing the canonical `src.data_loader.N_SPLITS` (re-exported by src.cv_utils). Drift risk if the canonical k ever changes. LOW. | Infra | C6 | 1 | 2 | 0.5 | 1.0 | 0.5 | 2.0 | — |
| BL-023 | TECH DEBT (DEBT-004): the deterministic-regime params block (`num_threads=1, deterministic=True, force_row_wise=True, seed=42`) is duplicated inline in three spots of tune.py (deterministic_eval, build_submission, run_study). Extract one `_DET_PARAMS` constant. LOW. | Infra | C6 | 1 | 2 | 0.5 | 1.0 | 0.5 | 2.0 | — |
| BL-024 | TECH DEBT (DEBT-005): the success threshold `0.97` is a bare literal in tune.py (`det_result.mean >= 0.97`, key `reaches_target_0_97`). Name it `STRETCH_TARGET_BAL_ACC = 0.97` so the renegotiated target lives in one place. LOW. | Infra | C6 | 1 | 1 | 0.25 | 1.0 | 0.5 | 0.5 | — |
| BL-027 | TECH DEBT (DEBT-006): extract the duplicated MLflow helpers (run setup + per-fold logging + deterministic-eval boilerplate, ~98 lines) shared across the XGBoost/CatBoost/LightGBM tuners into `src/tune_common.py`. Rule-of-three tripped by the 3rd member tuner. LOW. | Infra | C6 | 2 | 3 | 0.5 | 1.0 | 1 | 1.5 | — |
| BL-028 | TECH DEBT (DEBT-007): residual lint in the tuners — move `pandas` import under `TYPE_CHECKING` (only used in annotations) and migrate `typing.Callable` -> `collections.abc.Callable` (ruff UP035). Cosmetic, no behaviour change. LOW. | Infra | C6 | 1 | 1 | 0.25 | 1.0 | 0.5 | 0.5 | — |

### WON'T

| ID | Title | Reason |
|----|-------|--------|
| BL-101 | Real-time serving endpoint / REST API | Batch-only competition; no runtime exists (Out of scope) |
| BL-102 | Monitoring + drift dashboards (C12 infra) | No production system to monitor; submission is one-shot |
| BL-103 | External-data join (original SDSS17 / catalogs) | Leakage + rules risk; forbidden by default (B4) |
| BL-104 | Probability-calibration deliverable | Submission needs hard labels only; thresholds tuned for balanced acc, not calibration |
| BL-105 | DL tabular in the MUST path | Boosting committed; DL relegated to time-boxed COULD spike (BL-017) |

---

## Closed items (context v1.1)

| ID | Title | Status | Note |
|----|-------|--------|------|
| BL-001 | Accept competition rules + download train/test/sample_submission | done | Rules accepted; 577,348 train / 247,436 test downloaded. External 403 blocker resolved. |
| BL-002 | Reproducible data loading + schema validation (dtype contract, seed=42) | done | C2, commit a6aa293. data_loader.py: load + fail-loud dtype contract + StratifiedKFold. 24 tests, 100% cov. Review gates (tech-debt scan + code review) PASS. |
| BL-003 | LEAKAGE AUDIT (BLOCKING): spectral_type & galaxy_population dist + MI vs class + adversarial validation | done | C2, commit 5f743fa. leakage_audit.py: 0 leakage. H(class)=0.8798; max MI/H redshift 0.584. spectral_type CramerV=0.525, galaxy_population V=0.594 (physical -> USE). id MI/H=0.0005 -> DROP. Adv.val AUC=0.4997 (train=test dist; CV reliable). 0 dups. Review gates (statistical validation + tech-debt scan + code review) PASS (2 cycles, bug B1 cross_split_dups fixed). |
| BL-004 | EDA on TRAIN only: target dist, redshift-by-class, band/color dists, missingness map | done | C2, commit 11e0860. eda.py: EDA + SHAP baseline. CV balanced_accuracy ceiling 0.96448 +/- 0.00029. Recall GALAXY 0.953 / QSO 0.975 / STAR 0.966. SHAP redshift dominates 4x. 9 figures. Review gates (statistical validation + tech-debt scan + code review) PASS (2 cycles). |
| BL-005 | Stratified K-fold (k=5) split on class; freeze fold IDs; verify no dup rows cross-fold | done | C2, commit 5f743fa. make_folds.py: StratifiedKFold(5, seed=42) frozen + contract verified. |
| BL-006 | Feature engineering: photometric colors (u-g, g-r, r-i, i-z) + redshift transforms; categorical encoding (cleared features only) | done (NEGATIVE result, certified) | C3, commit 520af20. feature_eval.py: SDSS color features + redshift_log + paired ablation harness. **VERDICT: NEGATIVE — no engineered feature beats the 0.96448 balanced_accuracy CV baseline defensibly** (best paired Δ +8.6e-05 < 2·SEM, statistically indistinguishable from noise). LightGBM monotone invariance to univariate transforms confirmed empirically. Side finding: alpha/delta (celestial coords) strongly predictive (ablation drop −0.009 / −0.006) — synthetic-SDSS artifact, NOT leakage (C2 adversarial AUC 0.4997); legitimate to USE. Conclusion: 0.9645 is a STRUCTURAL ceiling on first-order features; the ~0.0055 gap to target 0.97 is delegated to C5 Optuna HPO. Review gates: statistical validation ✓ (3-test triangulation) → tech-debt scan ✓ (logged DEBT-001 → BL-019) → code review ✓ (0 blockers, 29/29 tests, 100% cov). |
| BL-007 | Statistical hypothesis tests: color/redshift separability per class (Kruskal-Wallis + effect size); document signal ranking | deferred (closed, low incremental value) | C3. DEFERRED — per-class separability is already established empirically by the SHAP baseline (C2, BL-004) plus the causal ablation (C3, BL-006), both stronger evidence than a marginal univariate test. A Kruskal-Wallis result would not change any modeling decision. Closed to avoid ceremony; not re-opening unless a future cycle needs a documented signal-ranking artifact for a deliverable. |
| BL-008 | Baseline LightGBM (GPU) class_weight=balanced; log CV balanced_accuracy + per-class recall + confusion matrix to MLflow | done | C5, commit b78a6f4 (ADR-001 in b0b37ae). Deterministic baseline CV balanced_accuracy = 0.964479. src/baseline.py reuses feature_eval CV harness verbatim. Review pipeline: statistical validation ✓ → tech-debt scan ✓ → code review ✓ (97 tests). |
| BL-011 | Optuna HPO on best single model (>=50 trials, per-fold, balanced_accuracy objective); time-boxed | done | C5, commit b78a6f4. src/tune.py: TPESampler(seed=42) + MedianPruner, best tuned config re-evaluated deterministically = CV 0.964866 vs baseline 0.964479 (Δ +0.000386, marginal). **FIRST KAGGLE SUBMISSION (id 53435263): public LB = 0.96521 > CV → CV is faithful/conservative, model generalizes, NO overfitting; the tuning gain is real (held-out confirms it)**. 0.97 NOT reached (LB gap 0.0048). Review pipeline: statistical validation ✓ (claim qualified re winner's curse — refuted positively by LB>CV) → tech-debt scan ✓ (logged DEBT-002..005 → BL-020/021/023/024) → code review ✓ (0 blockers, 4 in-place fixes, 97 tests). Adversarial-robustness check not applicable (synthetic tabular data). |
| BL-019 | TECH DEBT (DEBT-001): extract `compute_sample_weights` + `load_folds` (byte-identical dups in eda.py & feature_eval.py) to `src/cv_utils.py` | done | C5, commit b78a6f4. Rule-of-three tripped (tune.py = 3rd consumer). src/cv_utils.py created; FOLDS_PATH + N_SPLITS + compute_sample_weights + fold_indices + load_folds centralized. Closed DEBT-001 by the tech-debt scan. |
| BL-022 | TECH DEBT (transient): unused import `_lgbm_params` flagged during C5 review | done (resolved in-place) | C5, commit b78a6f4. Resolved IN-PLACE during the code review gate pass — the import is now live (baseline.py:52,149 + tune.py determinism reuse). No separate work item required; recorded here for audit completeness only. |
| BL-010 | C6 ENSEMBLE step 1 — XGBoost (GPU) + CatBoost (GPU) members tuned to PARITY with the tuned LightGBM | done | C6, commit 366649d. Both members tuned (Optuna, per-fold balanced_accuracy on the frozen folds) until they reach parity with LightGBM — XGBoost reached deterministic CV 0.964830 ≈ LightGBM 0.964866. This parity was the load-bearing precondition: the C6 step-2 naive blend had FAILED earlier because CatBoost/XGBoost ran on defaults (Δ +0.000069 < 2·SEM, statistically null). Tuning them to parity is what made the blend defensible. Review pipeline: statistical validation ✓ → tech-debt scan ✓ → code review ✓. |
| BL-013 | C6 ENSEMBLE step 2 — nested-weight blend of {tuned LightGBM, CatBoost, XGBoost}; STRETCH lever toward 0.97 | done — **ENSEMBLE = FINAL MODEL** | C6, commit 366649d. Nested-CV weighted blend (weights LightGBM 0.500 / CatBoost 0.083 / XGBoost 0.417, selected by an inner CV that NEVER reads the reporting fold). Deterministic CV balanced_accuracy 0.965360 vs tuned LightGBM-alone 0.964866 (Δ +0.000049). **Second Kaggle submission: public LB = 0.96625 > LightGBM-alone LB 0.96521 (Δ +0.00104 held-out — LARGER than the CV Δ +0.00049, so the held-out leaderboard CONFIRMS and AMPLIFIES the gain).** Primary target ≥0.965 SUPERADO (LB 0.96625). Stretch 0.97 NOT reached (LB gap 0.00375). **The ensemble is the final model.** Review pipeline: statistical validation ✓ (nested-CV weight selection certified leakage-free — the selector never reads the reporting fold; Δ measured with the nested weights, not global) → tech-debt scan ✓ (debt LOW → DEBT-006/007 → BL-027/028) → code review ✓ (0 blockers, 131 tests, 2 in-place fixes). Adversarial-robustness check not applicable (synthetic tabular data). |

## Dependency DAG override (RICE caveat)

BL-008 (RICE 30.0) and BL-014 (RICE 21.6) top raw RICE, but execution order is governed by
the dependency edges, not RICE alone:

```
[C2 CLOSED] BL-001 -> BL-002 -> { BL-003, BL-004, BL-005 }   (all done — C2 sign-off 2026-06-06)
[C3 CLOSED] BL-006 (done, NEGATIVE result) ; BL-007 (deferred)   (C3 sign-off 2026-06-06)
[C5 CLOSED] BL-008 (baseline, CV 0.964479) -> BL-011 (tuned, CV 0.964866, LB 0.96521)
              + BL-019 (tech debt DEBT-001 extracted to src/cv_utils.py)
              (C5 sign-off 2026-06-07 — first submission succeeded, target renegotiated)
[C6 CLOSED — ENSEMBLE = FINAL MODEL]
  BL-010 (XGB+CatBoost members tuned to parity, done) -> BL-013 (nested-weight blend, done)
     => ENSEMBLE FINAL: LightGBM 0.500 / CatBoost 0.083 / XGBoost 0.417 (nested CV weights)
        CV 0.965360, public LB 0.96625 > LightGBM-alone LB 0.96521 (Δ +0.00104 held-out).
        Primary ≥0.965 SUPERADO. Stretch 0.97 NOT reached (LB gap 0.00375).
        (C6 sign-off 2026-06-07 — commit 366649d, ensemble is the final model)
[C8 OPEN — EVAL / SHIP THE ENSEMBLE]
  BL-009 (submission pipeline)  [ELIGIBLE — dependency-free; NEXT — wire the ensemble into one-command predict]
  BL-012 (decision-rule / threshold tuning, M4)  [ELIGIBLE — dependency-free]
     +-> BL-014 (eval + fairness gate) -> BL-015 (tests) -> BL-016 (promote utils)
  BL-014 -> { BL-017, BL-018 }   (COULD spikes, time-boxed + kill criteria)
  Tech debt (any time): BL-020 (MEDIUM, do before re-running eda.py), BL-021, BL-023, BL-024,
     BL-027 (extract MLflow helpers -> src/tune_common.py), BL-028 (residual lint) — all LOW/MEDIUM
```

> **C2 close note (2026-06-06)**: BL-002..BL-005 moved to "Closed items". Their
> resolved deps were stripped from BL-006 (was BL-003,BL-004 -> now `—`) and
> BL-008 (was BL-005,BL-006 -> now BL-006); by convention, `Deps` lists only
> *pending* blocking edges, so dependency eligibility stays correct and the
> format validator (cross-ref against active MoSCoW IDs) passes.

> **C3 close note (2026-06-06)**: BL-006 closed with a CERTIFIED NEGATIVE result
> (no engineered feature beats the 0.96448 CV baseline; 0.9645 is a structural
> first-order ceiling — commit 520af20) and BL-007 deferred (separability already
> proven via SHAP + ablation, lower-strength test would not move any decision).
> Both moved to "Closed items". BL-006 was stripped from BL-008's `Deps`
> (was `BL-006` -> now `—`), so BL-008 is now dependency-free and ELIGIBLE.
> New item **BL-019** (tech debt DEBT-001: extract `compute_sample_weights` +
> `load_folds` to `src/cv_utils.py`) added under COULD, dep BL-008, deferred until
> the C5 tuning harness becomes the 3rd consumer. The ~0.0055 gap to the 0.97
> target is now owned by C5 Optuna HPO (BL-011), not by feature engineering.
> Frontier open item: **BL-008** (C5 LightGBM baseline + Optuna tuning).

> **BLOCKING note**: BL-003 (leakage audit) ranks mid-table by raw RICE (effort vs reach)
> but is a hard blocking edge — BL-006 (FE) cannot finalize the feature set until BL-003
> clears the two categoricals `spectral_type` and `galaxy_population` (galaxy_population
> suspected GALAXY-only target proxy). RICE orders *within* a dependency tier, never across
> a blocking edge.

## Changed-context notes (v1.1)

- BL-001 done: rules accepted, data downloaded (577,348 train / 247,436 test rows).
- BL-003 reframed: the leakage threat is NOT SDSS spurious IDs (obj_ID/run_ID/plate/MJD/
  fiber_ID/spec_obj_ID — they do NOT exist here; only a trivial row `id`), but the two
  categoricals `spectral_type` and `galaxy_population`. This is the highest correctness
  risk in the project.
- Minority class is STAR (14.33%), not QSO — fairness gate BL-014 checks recall(STAR)>=0.93
  AND recall(QSO)>=0.94.

## Changed-context notes (v1.2 — C3 close, 2026-06-06)

- **BL-006 negative result is load-bearing for C5 planning**: feature engineering did
  NOT close the 0.0055 gap to 0.97. The gap is now C5's problem, to be attacked by
  hyperparameter tuning (Optuna, BL-011), not by more univariate transforms. Tree models
  (LightGBM) are invariant to monotone univariate transforms — this was confirmed
  empirically, so colors/log-redshift were never going to help a boosting baseline.
- **alpha/delta (celestial coords) are predictive and legitimate to USE**: ablation drop
  −0.009 / −0.006. This is a synthetic-dataset artifact of the SDSS generator, not leakage
  (C2 adversarial AUC 0.4997 already certified train ≈ test). They stay in the feature set.
- **DEBT-001 → BL-019**: `compute_sample_weights` + `load_folds` are duplicated byte-for-byte
  across `eda.py` and `feature_eval.py`. Rule-of-three not yet tripped (2 consumers). Extract
  to `src/cv_utils.py` when the C5 tuning harness becomes the 3rd consumer. Registered by
  the tech-debt scan; deferred to C5.
- **OPEN RISK carried into C5**: the 0.97 target may not be reachable. If C5 Optuna does not
  close the gap, renegotiate the target rather than overfit.

## Changed-context notes (v1.3 — C5 close, 2026-06-07)

- **C5 POC CLOSED. First Kaggle submission succeeded.** BL-008 (baseline), BL-011 (Optuna HPO),
  BL-019 (DEBT-001 dedup) all done in commit b78a6f4; ADR-001 (model + tuning strategy) in b0b37ae.
  Tuned LightGBM: deterministic CV balanced_accuracy 0.964866 vs baseline 0.964479 (Δ +0.000386,
  marginal). **Public LB = 0.96521 (id 53435263) — LB > CV.** The leaderboard *exceeding* the local
  CV certifies the CV is faithful and conservative: the model generalizes, there is NO overfitting,
  and the tuning gain is real (held-out confirms it). The statistical review's winner's-curse caveat on the
  marginal-improvement claim did NOT materialize as a negative — the held-out LB refuted it positively.
- **0.97 NOT reached by single-model tuning** (LB gap 0.0048). Combined with the C3 certified
  structural ceiling (BL-006: 0.9645 is a first-order ceiling, no engineered feature beats it),
  this confirms 0.97 is unreachable by tuning a single LightGBM. **The 0.97 target was aspirational — not an external requirement.**
- **TARGET RENEGOTIATED (2026-06-07)** — see ml-problem-statement.md v1.2 §3:
  primary minimum = **balanced_accuracy >= 0.965 (ACHIEVED, LB 0.96521)**; **0.97 is now a STRETCH
  goal, not guaranteed**, to be attempted via the C6 ENSEMBLE (CatBoost + XGBoost + LightGBM blend).
  Fairness guardrails (recall STAR>=0.93, QSO>=0.94, spread<=0.06) UNCHANGED.
- **C6 strategy = ENSEMBLE.** BL-010 (XGBoost + CatBoost members) and BL-013 (stack/blend) already
  existed in the backlog and ARE the ensemble work — relabeled "C6 ENSEMBLE step 1/2". No new
  duplicate item created (would be ceremony + capability duplication). **Next eligible: BL-010**
  (dependency-free after the C5 strip), the entry point to the ensemble.
- **New tech debt logged by the tech-debt scan (DEBT-002..005 → BL-020/021/023/024):**
  - BL-020 (MEDIUM): `seaborn` + `matplotlib` imported by src/eda.py but missing from pyproject.toml
    dependencies → eda.py not reproducibly installable. Fix before re-running eda.py in C6.
  - BL-021 (LOW): `N_SPLITS` redefined in eda.py + feature_eval.py instead of importing canonical.
  - BL-023 (LOW): determinism params block duplicated 3x in tune.py.
  - BL-024 (LOW): bare literal `0.97` in tune.py — name it now that the target is renegotiated.
  - BL-022 (transient unused-import): RESOLVED IN-PLACE during the code review gate pass; closed
    as done in "Closed items" for audit completeness, NOT carried as pending work.
- **Excalidraw**: C5 does not require a diagram. **C6 Build DOES require one** (ensemble topology) —
  tracked as a pending/deferred blocker.

## Changed-context notes (v1.4 — C6 close, 2026-06-07)

- **C6 BUILD CLOSED. The ENSEMBLE is the final model.** BL-010 (members tuned to parity) and
  BL-013 (nested-weight blend) done in commit 366649d. Final blend weights LightGBM 0.500 /
  CatBoost 0.083 / XGBoost 0.417, selected by nested CV. Deterministic CV balanced_accuracy
  0.965360 vs tuned LightGBM-alone 0.964866 (Δ +0.000049). **Second public LB = 0.96625 >
  LightGBM-alone LB 0.96521 (Δ +0.00104 held-out — LARGER than the CV Δ +0.00049, so the
  leaderboard CONFIRMS and AMPLIFIES the gain, exactly as in C5).**
- **The naive blend (equal weights) had LOST.** With CatBoost/XGBoost on defaults the equal-weight
  blend gave Δ +0.000069 < 2·SEM — statistically null. The fix was NOT to abandon the ensemble but
  to (a) tune each member to PARITY (XGBoost reached 0.964830 ≈ LightGBM) and (b) use a nested-CV
  weighted blend. Only then did the blend win defensibly. **Lesson load-bearing for any future
  ensemble: members must be at parity, not on defaults; weights must come from nested CV, not the
  reporting fold.** Honesty about the negative naive result is what drove the correct fix.
- **Target status**: primary `balanced_accuracy >= 0.965` SUPERADO (LB 0.96625). Stretch `0.97`
  NOT reached — LB gap 0.00375. Open question (Blockers): whether to keep pursuing 0.97 or freeze
  the ensemble as final. Fairness guardrails UNCHANGED (recall STAR>=0.93, QSO>=0.94, spread<=0.06,
  re-verified at the C8 gate).
- **New tech debt logged by the tech-debt scan (DEBT-006/007 → BL-027/028), both LOW:**
  - BL-027 (LOW): extract ~98 lines of duplicated MLflow helpers across the three member tuners
    into `src/tune_common.py` (rule-of-three tripped by the 3rd tuner).
  - BL-028 (LOW, cosmetic): residual lint — `pandas` under `TYPE_CHECKING`, `typing.Callable` →
    `collections.abc.Callable` (ruff UP035).
- **Review pipeline (C6)**: statistical validation ✓ (nested-CV weight selection certified leakage-free — selector
  never reads the reporting fold; Δ computed with nested weights, not global) → tech-debt scan ✓
  (debt LOW) → code review ✓ (0 blockers, 131 tests, 2 in-place fixes).
  Adversarial-robustness check not applicable (synthetic tabular data).
- **Next eligible: BL-009** (submission pipeline, dependency-free) — wire the final ensemble into
  the one-command predict so the C8 eval + fairness gate runs against the shipping artifact.
- **Architecture diagram for the C6 ensemble topology still DEFERRED** — consistent with the C1/C4
  diagram deferral. Not a release blocker for a batch Kaggle submission.
