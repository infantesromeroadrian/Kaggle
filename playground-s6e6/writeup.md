# Writeup — Kaggle Playground Series S6E6: Predicting Stellar Class

**Competition:** Playground Series Season 6 Episode 6  
**Task:** Multiclass classification — GALAXY / STAR / QSO (3 classes)  
**Official metric:** `balanced_accuracy` (unweighted mean of per-class recall, a.k.a. macro-recall)  
**Submission format:** CSV with columns `id,class` (hard labels, no probabilities)  
**Deadline:** 2026-06-30  
**Final public LB:** **0.96625** (ensemble, 2nd submission)  
**Pipeline:** Structured ML pipeline (phases C1-C6 complete, C8 pending)  
**Hardware:** RTX 2000 Ada 8 GB VRAM (consumer laptop GPU)

---

## 1. Overview

Astronomical observations are classified from synthetic SDSS-derived photometry into three
stellar object types. The dominant separating signal is `redshift`: stars sit near zero,
galaxies at intermediate values, quasars (QSOs) at high values with a long tail.

| Target | Value | Status |
|---|---|---|
| `balanced_accuracy >= 0.965` (primary, renegotiated at C5) | **LB 0.96625** | ACHIEVED |
| `balanced_accuracy >= 0.97` (stretch) | LB gap 0.00375 | NOT reached |
| CV-LB gap <= 0.005 | LB > CV on all 3 submissions | PASSED (CV faithful) |
| recall(STAR) >= 0.93, recall(QSO) >= 0.94 | C8 gate | pending |

**Public leaderboard (snapshot 2026-06-07):** rank **361 / 965 teams (top ~37%)**, score **0.96625**
balanced_accuracy. Submissions used: 2 (LightGBM-only 0.96521, then tuned ensemble 0.96625).

**Final model:** soft-voting blend of three gradient-boosted tree models. Weights selected by
nested cross-validation: LightGBM 0.500 / CatBoost 0.083 / XGBoost 0.417.  
**Submissions:** 2 total. LB improved monotonically (CV conservative throughout).

---

## 2. Repo Structure

```
playground-s6e6/
├── src/
│   ├── data_loader.py        # load_train / load_test with fail-loud schema validation
│   ├── leakage_audit.py      # MI, Cramer's V, adversarial validation (C2)
│   ├── make_folds.py         # freeze StratifiedKFold(5, seed=42) to data/folds/
│   ├── eda.py                # EDA + LightGBM quick ceiling + SHAP (C2)
│   ├── features.py           # SDSS color indices + redshift_log (C3, row-wise pure)
│   ├── feature_eval.py       # frozen-fold ablation harness + paired stats (C3)
│   ├── cv_utils.py           # shared: compute_sample_weights, load_folds (C5)
│   ├── baseline.py           # formal LightGBM baseline with MLflow (C5)
│   ├── tune.py               # Optuna LightGBM HPO, TPESampler+MedianPruner (C5)
│   ├── tune_catboost.py      # Optuna CatBoost HPO, GPU (C6)
│   ├── tune_xgboost.py       # Optuna XGBoost HPO, CPU 4 threads (C6)
│   ├── ensemble.py           # nested-CV weight blend of 3 tuned models (C6)
│   └── tests/
│       ├── test_data_loader.py
│       ├── test_features.py
│       ├── test_leakage_stats.py
│       ├── test_cv_utils.py
│       ├── test_baseline.py
│       ├── test_tune.py
│       ├── test_tune_base_learners.py
│       └── test_ensemble.py
├── data/
│   ├── train.csv             # 577,348 rows (gitignored)
│   ├── test.csv              # 247,436 rows (gitignored)
│   ├── sample_submission.csv
│   └── folds/
│       ├── cv_indices.npz    # frozen 5-fold indices, seed=42
│       └── cv_indices_meta.json
├── docs/
│   ├── c1-discovery/         # ml-problem-statement.md, backlog.md, feasibility.md
│   ├── c2-data/              # eda.md + figures/
│   ├── c3-feature/           # feature-engineering.md, feature_eval_results.json
│   ├── c4-design/            # ADR-001-model-and-tuning-strategy.md
│   ├── c5-poc/               # tuning-report.md, baseline_results.json, tuning_results.json
│   └── c6-build/             # ensemble-report.md, ensemble_results.json,
│                             # catboost_tuning.json, xgboost_tuning.json
├── mlruns/                   # MLflow local file store (gitignored)
├── pyproject.toml
└── writeup.md                # this file
```

---

## 3. Setup and Reproduce

All commands assume the repo root is the working directory.

### 3.1 Install dependencies

```bash
# Requires uv (https://github.com/astral-sh/uv) and Python >= 3.12
uv sync
```

To also install MLflow experiment tracking (optional, used by baseline/tuning scripts):

```bash
uv sync --extra tracking
```

Verify the environment:

```bash
uv run python -c "import lightgbm, catboost, xgboost, optuna; print('OK')"
```

### 3.2 Download competition data

The `KAGGLE_API_TOKEN` environment variable is set on the host but its value is corrupted.
Use `env -u` to unset it and fall back to the `~/.kaggle/kaggle.json` credentials file:

```bash
env -u KAGGLE_API_TOKEN kaggle competitions download -c playground-series-s6e6 -p data/
unzip data/playground-series-s6e6.zip -d data/
```

Expected files after extraction:

```
data/train.csv          (577,348 rows)
data/test.csv           (247,436 rows)
data/sample_submission.csv
```

### 3.3 Freeze cross-validation folds

Run once. Writes `data/folds/cv_indices.npz` and `data/folds/cv_indices_meta.json`.
Every subsequent script reads from this frozen file — never re-splits.

```bash
PYTHONPATH=. uv run python -m src.make_folds
```

Verify:

```bash
PYTHONPATH=. uv run python -c "
import numpy as np
d = np.load('data/folds/cv_indices.npz', allow_pickle=False)
print('folds:', len(d.files), '| fold sizes:', [len(d[k]) for k in d.files])
"
```

### 3.4 Run the full test suite

```bash
PYTHONPATH=. uv run pytest src/tests/ -v
```

Expected: 131 tests pass, 0 failures.

### 3.5 EDA and SHAP ceiling (C2, informational only)

```bash
PYTHONPATH=. uv run python -m src.eda
```

Writes figures to `docs/c2-data/figures/`. Prints a JSON summary to stdout.
Note: `seaborn` and `matplotlib` are not declared in `pyproject.toml` (open debt BL-020).
Install them manually if re-running:

```bash
uv add seaborn matplotlib
```

### 3.6 Formal LightGBM baseline (C5)

Reproduces `balanced_accuracy = 0.964479` exactly (drift tolerance 1e-4).

```bash
PYTHONPATH=. uv run python -m src.baseline
```

Result lands in `docs/c5-poc/baseline_results.json`. MLflow run logged to `mlruns/` if
`mlflow` is installed.

### 3.7 LightGBM Optuna HPO (C5)

Runs 40 trials with a 40-minute timeout (whichever binds first). Writes
`data/submission_c5_optuna.csv` (first Kaggle submission artifact).

```bash
PYTHONPATH=. uv run python -m src.tune
```

Optionally limit trials or timeout:

```bash
PYTHONPATH=. uv run python -m src.tune --n-trials 20 --timeout-s 1200
```

### 3.8 CatBoost Optuna HPO (C6)

Uses GPU. Default: 30 trials, 22-minute timeout.

```bash
PYTHONPATH=. uv run python -m src.tune_catboost
```

Best params written to `docs/c6-build/catboost_tuning.json`.

### 3.9 XGBoost Optuna HPO (C6)

Uses CPU, 4 threads. Default: 15 trials, 35-minute timeout.

```bash
PYTHONPATH=. uv run python -m src.tune_xgboost
```

Best params written to `docs/c6-build/xgboost_tuning.json`.

### 3.10 Nested-weight ensemble (C6 — final model)

Runs the full nested-CV blend evaluation and writes
`data/submission_c6_ensemble_tuned.csv` (the final submission artifact).

```bash
PYTHONPATH=. uv run python -m src.ensemble
```

---

## 4. Cycle-by-Cycle Narrative

### C1 Discovery

Defined the ML problem statement (`docs/c1-discovery/ml-problem-statement.md`) and a
MoSCoW+RICE backlog (`docs/c1-discovery/backlog.md`). Three key framing decisions:

1. Metric is `balanced_accuracy` (macro-recall), not plain accuracy — minority class STAR
   (14.33%) must not be collapsed.
2. Initial target set at `>= 0.97`, benchmarked against published SDSS notebooks that
   reach 0.97-0.98 plain accuracy; formally aspirational, re-validated against the first
   public LB in C5.
3. Leakage threat identified as the two categoricals (`spectral_type`, `galaxy_population`),
   NOT any spurious row ID — the dataset ships only a trivial sequential `id`.

### C2 Data

Four deliverables built and gated:

- `src/data_loader.py` — loads train/test with a fail-loud schema contract (dtypes, column
  names, row counts). 24 tests, 100% coverage.
- `src/leakage_audit.py` — mutual information, Cramér's V (Bergsma-corrected), adversarial
  validation. Result: adversarial AUC = 0.4997 (train and test indistinguishable), zero
  duplicates, zero leakage. Both categoricals are physically informative, not proxies.
- `src/make_folds.py` — StratifiedKFold(n_splits=5, shuffle=True, random_state=42). Folds
  frozen to `data/folds/cv_indices.npz` and reused positionally in every subsequent cycle.
- `src/eda.py` — univariate stats, per-class redshift distributions, band correlation
  heatmap, SHAP with TreeExplainer on the last fold.

**Baseline RAW (C2 ceiling signal):** CV `balanced_accuracy = 0.96448 ± 0.00029`.
SHAP shows `redshift` dominates by ~4x: per-class mean |SHAP| 1.864 vs 0.472 for the
next feature (`z` band). STAR separation is almost entirely driven by redshift near zero.

### C3 Feature and Hypothesis

Hypothesis: SDSS color indices (u-g, g-r, r-i, i-z) and a `log1p(redshift)` transform
will beat the raw baseline.

Result: certified negative. `src/features.py` implements all candidates row-wise pure;
`src/feature_eval.py` runs them on the identical frozen folds with paired statistics.

Best configuration (colors): mean paired delta = +0.000086 vs baseline, paired SEM =
0.000129. The delta is smaller than 1×SEM, far below the 2×SEM defensibility bar.

The `redshift_log` transform adds nothing on top of colors (delta +0.000048 over colors),
confirming LightGBM's invariance to monotone univariate transforms empirically.

Ablation finding: dropping `alpha` (right ascension) costs -0.009 balanced_accuracy;
dropping `delta` costs -0.006. These sky coordinates carry SDSS survey footprint signal.
Adversarial AUC 0.4997 confirms there is no covariate shift — they are safe to use.

**Conclusion:** 0.96448 is a structural first-order ceiling. The 0.0055 gap to 0.97 is
delegated to HPO in C5.

### C4 Design

ADR-001 (`docs/c4-design/ADR-001-model-and-tuning-strategy.md`) records four decisions:

1. Model family: LightGBM (GBT already at 0.9645; switching is premature).
2. Tuning engine: Optuna TPESampler(seed=42) + MedianPruner.
3. Feature set: RAW (C3 proved engineered features add nothing defensible).
4. Evaluation: the frozen folds throughout; no moving denominator across cycles.

### C5 POC

`src/cv_utils.py` extracted first (DEBT-001: `compute_sample_weights` and `load_folds`
duplicated across eda.py and feature_eval.py; tune.py became the third consumer).

`src/baseline.py` runs the formal baseline with MLflow tracking and a drift gate:
reproduces 0.964479 (drift vs canonical 0.96448 = 8.4e-7, well within 1e-4 tolerance).

`src/tune.py` ran Optuna: 18 trials completed + 1 pruned in 40 minutes.

**Best config (trial #16):**

| Param | Value |
|---|---|
| max_depth | 8 |
| num_leaves | 61 |
| learning_rate | 0.07840 |
| n_estimators | 399 |
| min_child_samples | 151 |
| bagging_fraction | 0.7065 |
| feature_fraction | 0.7106 |
| reg_alpha | 1.4997 |
| reg_lambda | 0.5615 |
| min_split_gain | 0.1987 |

Deterministic CV: `0.964866 ± 0.000367`. Paired delta vs baseline: +0.000386, all 5
fold deltas positive, paired t = 8.39 (defensible). The statistical review noted winner's curse
risk (60-78% of the delta plausibly selection noise).

**First Kaggle submission:** public LB = **0.96521**. LB > CV refuted the winner's curse
concern positively: the held-out leaderboard confirmed the gain is real. CV is faithful
and conservative. Target renegotiated: primary >= 0.965 (ACHIEVED); 0.97 demoted to stretch.

GPU decision: OpenCL GPU rejected — no speedup (17.3s/fold vs 16.4s CPU multi-thread)
AND gave a different score (0.965286 vs 0.965048), breaking like-for-like comparability.

### C6 Build

**Step 1: naive equal-weight blend — failure, correctly diagnosed.**

CatBoost and XGBoost on defaults scored 0.957229 and 0.962944 respectively. The
equal-weight blend gave CV 0.963518, a paired delta of +0.000069 vs LightGBM-alone — below
2×SEM, statistically null. The statistical review certified the failure as legitimate: members not
at parity cannot be blended into a defensible improvement.

**Step 2: tune members to parity, then blend with nested-CV weights.**

CatBoost tuned with GPU (17/30 trials, 22-minute timeout): CV `0.962662`.
XGBoost tuned with CPU 4 threads (15 trials, 4 pruned, 36 minutes): CV `0.964830`.
XGBoost reached effective parity with LightGBM (gap -0.000036).

Nested-CV weight selection: for each held-out fold k, inner folds {0..4}\{k} used to
select weights over a 91-point simplex grid. All 5 folds independently selected
**LightGBM 0.500 / CatBoost 0.083 / XGBoost 0.417** — weight stability across disjoint
inner sets is strong evidence the optimum is not overfit.

Blend (nested-CV weights): CV `0.965360 ± 0.000379`. Paired delta vs LightGBM-alone:
+0.000494, ~7×SEM, positive on all 5 folds. Defensible.

**Second Kaggle submission:** public LB = **0.96625**. Held-out delta vs C5 = +0.00104,
larger than the CV delta of +0.00049. The leaderboard confirmed and amplified the gain.

---

## 5. Final Model

**Architecture:** soft-voting blend (weighted average of predicted probabilities).

**Weights (nested-CV, leakage-free):**

| Model | Weight | CV balanced_accuracy |
|---|---|---|
| LightGBM (tuned) | 0.500 | 0.964866 |
| XGBoost (tuned) | 0.417 | 0.964830 |
| CatBoost (tuned) | 0.083 | 0.962662 |
| **Ensemble** | — | **0.965360** |

**LightGBM best params:**

| Param | Value |
|---|---|
| max_depth | 8 |
| num_leaves | 61 |
| learning_rate | 0.07840 |
| n_estimators | 399 |
| min_child_samples | 151 |
| bagging_fraction | 0.7065 |
| feature_fraction | 0.7106 |
| reg_alpha | 1.4997 |
| reg_lambda | 0.5615 |
| min_split_gain | 0.1987 |

**CatBoost best params:**

| Param | Value |
|---|---|
| depth | 7 |
| learning_rate | 0.14176 |
| l2_leaf_reg | 1.5697 |
| border_count | 254 |
| random_strength | 2.5704 |
| iterations | 999 |

**XGBoost best params:**

| Param | Value |
|---|---|
| max_depth | 6 |
| learning_rate | 0.09325 |
| min_child_weight | 1.7882 |
| subsample | 0.6750 |
| colsample_bytree | 0.7557 |
| reg_alpha | 0.001017 |
| reg_lambda | 0.056561 |
| gamma | 1.0646 |
| n_estimators | 798 |

All three models use native categorical handling for `spectral_type` and
`galaxy_population`. Sample weights are balanced (1/class_freq, normalized to mean 1)
applied to training rows only.

---

## 6. Results

| Model | CV balanced_accuracy | Public LB | Notes |
|---|---|---|---|
| RAW LightGBM (C2 ceiling) | 0.96448 ± 0.00029 | — | Untuned, 300 trees |
| Tuned LightGBM (C5, trial #16) | 0.96487 ± 0.00037 | **0.96521** | 1st submission |
| CatBoost tuned (C6 member) | 0.96266 ± 0.00029 | — | GPU, 999 iterations |
| XGBoost tuned (C6 member) | 0.96483 ± 0.00031 | — | CPU, 798 estimators |
| Equal-weight blend (C6 step 1) | 0.96494 ± 0.00034 | — | Not defensible (Δ < 2·SEM) |
| **Nested-weight ensemble (final)** | **0.96536 ± 0.00038** | **0.96625** | **2nd submission** |

CV is consistently conservative (LB > CV on both submitted models): the evaluation
protocol is trustworthy.

---

## 7. Key Decisions and Lessons

### Feature engineering is not the lever for tree models on this dataset

SDSS color indices (u-g, g-r, r-i, i-z) are the canonical photometric discriminators.
They add zero defensible signal (+0.000086, below 1×SEM) because LightGBM with 461k
training rows already recovers all relevant magnitude interactions internally.
`log1p(redshift)` is strictly null under LightGBM's monotone-transform invariance, confirmed
empirically: the delta is +0.000048 on top of colors, negative relative to baseline noise.

Lesson: on large tabular datasets with a strong tree baseline, invest in HPO before
investing in feature engineering. Ablate before adding.

### The 0.97 target was aspirational, not structural

The ceiling on first-order features was certified at 0.9645 in C3. Optuna could not close
the 0.0055 gap to 0.97 (the best single-model LB is 0.96521). The target was renegotiated
to 0.965 against the first real public LB score, exactly as the problem statement committed
to. This is not lowering the bar; it is honest calibration after held-out evidence.

### Winner's curse must be validated on held-out data, not dismissed

The statistical review flagged that 60-78% of the C5 Optuna delta (+0.000386) was plausibly
selection noise (18 trials on the same 5 folds). The correct response was not to ignore
the caveat or to abandon the model — it was to submit and let the Kaggle LB arbitrate.
LB 0.96521 > CV 0.96487 refuted the worst-case scenario positively.

### Ensemble members must reach parity before blending

The naive equal-weight blend of untuned CatBoost + XGBoost + LightGBM LOST
(delta +0.000069, below 2×SEM). The diagnosis was correct: CatBoost at 0.957229 and
XGBoost at 0.962944 dragged the blend below the champion. Tuning each member to parity
(XGBoost: 0.964830) and then applying nested-CV weights (CatBoost down-weighted to 0.083)
produced a defensible win (+0.000494, ~7×SEM).

### Nested cross-validation is necessary for blend weight selection

Selecting blend weights on the same folds that report the blend score is CV overfitting.
The nested protocol (inner selector never reads the held-out fold) was unit-tested with a
deliberate corruption test: `test_best_weights_ignore_held_out_fold` poisons only the
held-out fold and asserts the selected weights are unchanged.

### GPU for LightGBM on 10 features is not worthwhile

Benchmarked on one fold: GPU 17.3s vs CPU multi-thread 16.4s, AND the GPU produced a
different score (0.965286 vs 0.965048), breaking baseline comparability. CPU
deterministic mode (num_threads=1, deterministic=True, force_row_wise=True) was used
for all reported numbers. CatBoost GPU is the exception: 14.6x speedup, making 17
tuning trials feasible within 22 minutes.

### Thread count is the dominant cost knob for multi-model tuning

`num_threads=-1` (all 22 cores) caused severe over-subscription: 130s/fold vs 43s/fold
at 4 threads. All tuning runs pin `num_threads=4` for LightGBM and XGBoost.

---

## 8. Open Debt

Items tracked in `docs/c1-discovery/backlog.md`, all LOW/MEDIUM severity:

| ID | Description | Severity |
|---|---|---|
| BL-020 | `seaborn` + `matplotlib` missing from `pyproject.toml` dependencies; `src/eda.py` imports both but they are not declared — breaks reproducible install from the lockfile | MEDIUM |
| BL-021 | `N_SPLITS` redefined as a local `Final[int] = 5` in `eda.py` and `feature_eval.py` instead of importing `src.data_loader.N_SPLITS` (re-exported by `src.cv_utils`); drift risk if canonical k changes | LOW |
| BL-023 | Deterministic-regime params block (`num_threads=1, deterministic=True, force_row_wise=True, seed=42`) duplicated inline in three spots of `tune.py`; should be a single `_DET_PARAMS` constant | LOW |
| BL-024 | Bare literal `0.97` in `tune.py` (`det_result.mean >= 0.97`, key `reaches_target_0_97`); should be a named constant `STRETCH_TARGET_BAL_ACC = 0.97` | LOW |
| BL-027 | ~98 lines of MLflow helpers (run setup + per-fold logging + deterministic-eval boilerplate) duplicated across the three member tuners; extract to `src/tune_common.py` (rule-of-three tripped) | LOW |
| BL-028 | Residual lint: `pandas` import should move under `TYPE_CHECKING`; `typing.Callable` should migrate to `collections.abc.Callable` (ruff UP035). Cosmetic, no behaviour change | LOW |

Architecture diagrams for the discovery context (C1), model-selection flow (C4), and ensemble
topology (C6) are deferred. They are non-blocking for a batch Kaggle submission.

Pending C8 (eval and fairness gate): recall(STAR) >= 0.93, recall(QSO) >= 0.94, recall
spread <= 0.06, macro F1 >= 0.96, submission pipeline BL-009.

---

## 9. Reproduce the Submissions

### Submission 1 — tuned LightGBM (C5, public LB 0.96521)

```bash
# Prerequisite: folds frozen and data present (sections 3.2-3.3)
PYTHONPATH=. uv run python -m src.tune
# Output: data/submission_c5_optuna.csv
```

Upload:

```bash
env -u KAGGLE_API_TOKEN kaggle competitions submit \
  -c playground-series-s6e6 \
  -f data/submission_c5_optuna.csv \
  -m "C5 tuned LightGBM Optuna trial 16"
```

### Submission 2 — nested-weight ensemble (C6, public LB 0.96625, FINAL)

Step 1: tune CatBoost (GPU required, ~23 min):

```bash
PYTHONPATH=. uv run python -m src.tune_catboost
# Output: docs/c6-build/catboost_tuning.json
```

Step 2: tune XGBoost (CPU, ~36 min):

```bash
PYTHONPATH=. uv run python -m src.tune_xgboost
# Output: docs/c6-build/xgboost_tuning.json
```

Steps 1 and 2 can run in parallel (CatBoost uses GPU, XGBoost uses CPU):

```bash
PYTHONPATH=. uv run python -m src.tune_catboost &
PYTHONPATH=. uv run python -m src.tune_xgboost &
wait
```

Step 3: run the ensemble (reads both tuning JSONs and the frozen LightGBM params, ~8 min):

```bash
PYTHONPATH=. uv run python -m src.ensemble
# Output: data/submission_c6_ensemble_tuned.csv
```

Verify the submission file:

```bash
PYTHONPATH=. uv run python -c "
import pandas as pd
sub = pd.read_csv('data/submission_c6_ensemble_tuned.csv')
assert list(sub.columns) == ['id', 'class'], 'wrong columns'
assert sub['id'].nunique() == len(sub), 'duplicate ids'
assert sub['class'].isna().sum() == 0, 'nulls in class'
print('rows:', len(sub), '| classes:', sub['class'].value_counts().to_dict())
"
```

Upload:

```bash
env -u KAGGLE_API_TOKEN kaggle competitions submit \
  -c playground-series-s6e6 \
  -f data/submission_c6_ensemble_tuned.csv \
  -m "C6 nested-weight ensemble LightGBM 0.5/CatBoost 0.083/XGBoost 0.417"
```

---

## Appendix — Environment and Provenance

| Item | Value |
|---|---|
| Python | 3.12.13 |
| LightGBM | 4.6.0 |
| scikit-learn | 1.9.0 |
| NumPy | 2.4.6 |
| Optuna | >= 3.6 |
| CatBoost | >= 1.2, < 2 |
| XGBoost | >= 2.0, < 3 |
| Folds SHA-256 | `a0a19620...` (see `data/folds/cv_indices_meta.json`) |
| train.csv SHA-256 | `da2c5118...` (see `docs/c5-poc/baseline_results.json`) |
| seed | 42 (everywhere) |
| MLflow experiments | `s6e6-c5-baseline`, `s6e6-c5-optuna`, `s6e6-c6-ensemble`, `s6e6-c6-catboost-tuning`, `s6e6-c6-xgboost-tuning` |
