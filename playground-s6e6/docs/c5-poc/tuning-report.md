# C5 POC — LightGBM Tuning Report (BL-008 + BL-011)

Kaggle Playground S6E6 — stellar classification (GALAXY / STAR / QSO).
Metric: **balanced_accuracy** (macro-recall). Higher is better.

This report covers the formal baseline (BL-008) and the Optuna hyperparameter
search (BL-011). It is the artifact the statistical-validation review audits before the cycle gate.

---

## 1. Summary verdict

| Question | Answer |
|---|---|
| Does the baseline reproduce 0.96448 +/- 0.00029? | **Yes** — mean 0.964479, std 0.000288, drift 8.4e-7 vs canonical. Gate PASSED. |
| Does the best Optuna config beat the baseline POINT estimate? | **Yes** — det mean 0.964866 > 0.964479 (+0.000386). |
| Is the improvement statistically DEFENSIBLE in-sample? | **Yes, but marginal** — delta +0.000386 > 2*SEM_diff (9.2e-5); paired t = 8.39; all 5 fold deltas > 0. These statistics are conditional on selection (see next row). |
| Is the gain UNBIASED (selection-bias corrected)? | **Not established** — the best config was chosen by MAXIMIZING balanced_accuracy over ~18 trials on these SAME folds. Expected selection optimism for the max of 18 noisy estimates is +0.00024..+0.00030, vs observed +0.000386 → ~60-78% of the gain is plausibly capitalized noise; the unbiased residual (+0.00009..+0.00015) is comparable to the SEM itself. To be confirmed on held-out data (the Kaggle public LB is the unbiased arbiter). |
| Does it reach the 0.97 target? | **No** — 0.964866, gap of 0.00513 to 0.97. Reported honestly; not overfit to close the gap. |

> Overfitting guard: we report mean +/- std AND per-fold for every config, and the
> baseline comparison is PAIRED by fold. We do NOT chase a delta smaller than its
> own noise — a config is only called an improvement if its paired delta clears a
> 2-sigma band (see Section 5).

---

## 2. Baseline (BL-008) — the canonical comparison point

Reproduced by `src/baseline.py`, which reuses the proven CV engine in
`src/feature_eval.py` (frozen folds, balanced sample weights, native categorical
handling, determinism flags) and adds a verification gate + structured MLflow
tracking.

- **Config**: raw features only (the 8 numeric + 2 categorical columns; BL-006
  proved the engineered colour/log features add nothing defensible).
- **LightGBM**: multiclass, 3 classes, 300 trees, lr 0.05, 63 leaves, depth -1,
  subsample 0.8, colsample 0.8, reg_lambda 1.0, seed 42, deterministic=True,
  force_row_wise=True, num_threads=1.

| Fold | balanced_accuracy |
|---|---|
| 0 | 0.965048 |
| 1 | 0.964363 |
| 2 | 0.964385 |
| 3 | 0.964246 |
| 4 | 0.964354 |
| **mean** | **0.964479** |
| **std (ddof=0)** | **0.000288** |

Drift vs canonical 0.96448: **8.4e-7** (< 1e-4 tolerance) → gate PASSED.
Provenance: lightgbm 4.6.0, sklearn 1.9.0, numpy 2.4.6, python 3.12.13;
folds SHA-256 `a0a19620…`, train.csv SHA-256 `da2c5118…`.

---

## 3. Compute decision — GPU vs CPU (measured, not assumed)

Benchmarked on one real fold (462k train rows, 10 features) on the host
(RTX 2000 Ada 8GB, 22 CPU threads, LightGBM 4.6.0 OpenCL build):

| Mode | Time / fold | balanced_accuracy (fold 0) |
|---|---|---|
| CPU multi-thread (22 threads) | 16.4 s | 0.965048 |
| CPU deterministic (1 thread, deterministic+force_row_wise) | 23.6 s | 0.965048 |
| GPU (OpenCL, `device='gpu'`) | 17.3 s | 0.965286 (**different**) |

**Decision**:
- **GPU rejected.** It gave NO speedup (17.3s vs 16.4s CPU multi-thread) — a
  10-feature histogram GBM is too small to amortise the OpenCL transfer overhead
  on this laptop GPU — AND it produced a *different* score (0.965286), so it is
  not comparable to the baseline. Using it would corrupt the like-for-like
  comparison.
- **Search regime = CPU multi-thread (num_threads=-1).** Fastest, and on this
  fold it produced the SAME score as the deterministic single-thread run, so the
  ranking it yields is trustworthy. Non-bit-exact ordering across threads is
  irrelevant when the only goal is to RANK trials.
- **Reported-number regime = CPU deterministic (num_threads=1,
  deterministic=True, force_row_wise=True, seed=42).** The best trial is
  re-evaluated here so the reported delta is bit-for-bit reproducible and
  measured in the exact regime the baseline was.

---

## 4. Search space (BL-011) and budget

`src/tune.py`, Optuna `TPESampler(seed=42)` + `MedianPruner(n_startup_trials=10,
n_warmup_steps=1)`. Objective = mean balanced_accuracy over the 5 frozen folds.

| Hyperparameter | Range / set | Scale | Rationale |
|---|---|---|---|
| num_leaves | [15, 255] | log | model capacity; baseline 63 is inside |
| max_depth | {-1, 4, 6, 8, 10, 12, 16} | categorical | -1 = baseline; bounded depths allow shallower, better-regularised trees |
| learning_rate | [0.03, 0.2] | log | baseline 0.05 inside; floor raised from 5e-3 to bound per-fold wall-clock |
| n_estimators | <= 500, early-stopped | — | sized per fold by early stopping (patience 30) |
| min_child_samples | [5, 300] | log | leaf-size floor — main overfit guard on 462k rows |
| bagging_fraction (+freq=1) | [0.5, 1.0] | linear | row subsampling (baseline 0.8) |
| feature_fraction | [0.5, 1.0] | linear | column subsampling (baseline 0.8) |
| reg_alpha | [1e-3, 10.0] | log | L1 leaf-weight penalty |
| reg_lambda | [1e-3, 10.0] | log | L2 penalty (baseline 1.0) |
| min_split_gain | [0.0, 0.5] | linear | minimum loss reduction to split |

### Budget — honest accounting

- Spec asked for 150 trials. **Not reachable on this host inside 45 min without
  subsampling the data** — and subsampling would break baseline comparability
  (the data is the asset). A measured trial costs ~1-3 min (one fold ~16s
  multi-thread; the 500-tree cap bounds the worst case; MedianPruner abandons
  losing trials after 1-2 folds).
- Budget used: **n_trials=40 with a hard 40-min (2400s) timeout, whichever binds
  first** (Optuna honours both). The study JSON records `n_trials_completed` and
  `n_trials_pruned` so the actual count is auditable.
- **Actual study**: 18 trials completed + 1 pruned in the 40-min window
  (elapsed 2867s including the deterministic baseline reference, the best-config
  deterministic re-evaluation, and the submission fit). The 0.97 target was NOT
  reached within this budget.
- An uncapped early-stopping cap (2000 trees) made a single low-lr trial cost
  ~5 min — confirmed in a smoke run — hence the 400 cap + 0.03 lr floor.
- **Thread sizing is the dominant cost knob.** A re-benchmark under realistic
  background load showed `num_threads=-1` is catastrophic on this 22-thread host
  (130s/fold from over-subscription) versus `num_threads=4` (43s/fold). The
  search runs at `num_threads=4`; the first study attempt with -1 cost ~9
  min/trial and was abandoned. This is the single biggest lesson for C6: pin a
  modest thread count, do not trust -1 on a loaded workstation.

---

## 5. Statistical methodology

Per-fold paired comparison vs baseline, reusing `feature_eval.attach_paired_stats`:

- For fold k: `d_k = best_k - baseline_k` (same frozen fold → paired).
- `paired_diff_mean = mean(d)`.
- `paired_diff_sem = std(d, ddof=1) / sqrt(5)` — SEM of the paired differences.
- **Defensible iff `paired_diff_mean > 2 * paired_diff_sem`** (a conservative
  ~2-sigma band on n=5 folds; `_SIGMA_K = 2.0`, identical to the C3 ablation).

The reported best number uses the DETERMINISTIC regime with n_estimators fixed to
the search's rounded mean best-iteration, so every fold trains an identical-size
deterministic ensemble and the per-fold early-stopping optimism present in the
search regime is removed before the paired comparison.

### Numbers the statistical review must audit

1. **Baseline per-fold**: [0.965048, 0.964363, 0.964385, 0.964246, 0.964354];
   mean 0.9644792; std (ddof=0) 0.0002884.
2. **Best deterministic per-fold**: [0.9655840, 0.9647970, 0.9646577, 0.9645635,
   0.9647256]; mean 0.9648656; std (ddof=0) 0.0003674.
3. **Paired per-fold differences** d_k = best_k - baseline_k:
   [+0.000536, +0.000434, +0.000273, +0.000317, +0.000372] — **all 5 positive**.
4. **paired_diff_mean = +0.00038640**; **paired_diff_sem (std ddof=1 / sqrt 5) =
   0.00004605**; **2*SEM_diff = 0.00009211**.
5. **Defensible**: 0.00038640 > 0.00009211 → **True**. Paired t-statistic =
   mean(d)/SEM_diff = **8.39** (5 folds, one-sided — well past the 2-sigma rule).
6. **Beats baseline point**: 0.9648656 > 0.9644792 → True.
   **Reaches 0.97**: 0.9648656 >= 0.97 → **False** (gap 0.005134).

> Statistical caveats: (a) n=5 folds is small, so the SEM is itself
> noisy; the t=8.39 and the unanimous sign of the 5 deltas are the strength of
> the claim, not the absolute SEM. (b) The best config was SELECTED on these same
> 5 folds (selection-on-CV optimism); the deterministic re-evaluation removes the
> early-stopping optimism but NOT the model-selection optimism — the honest read
> is "a small, consistent, defensible CV gain", and the Kaggle LB will be the
> unbiased check. (c) The search regime is non-deterministic (multi-thread), but
> the REPORTED number is the single-thread deterministic re-eval, which is
> reproducible.

---

## 6. Best configuration

Best = Optuna **trial #16** (search_mean 0.965019, deterministic mean 0.964866).

| Param | Best value | Baseline |
|---|---|---|
| max_depth | 8 | -1 (unbounded) |
| num_leaves | 61 | 63 |
| learning_rate | 0.07840 | 0.05 |
| n_estimators | 399 (fixed = mean best-iteration) | 300 |
| min_child_samples | 151 | 20 (LightGBM default) |
| bagging_fraction | 0.7065 | 0.8 |
| feature_fraction | 0.7106 | 0.8 |
| reg_alpha | 1.4997 | 0.0 |
| reg_lambda | 0.5615 | 1.0 |
| min_split_gain | 0.1987 | 0.0 |

Interpretation: the search traded the baseline's unbounded depth for a
**depth-8, more strongly regularised** tree (much larger min_child_samples,
non-zero L1 + min_split_gain, lighter subsampling), with a slightly higher
learning rate and ~33% more (but early-stop-sized) trees. The direction is
"less overfitting, smoother decision surface" — consistent with a small,
unanimous per-fold gain rather than a single lucky fold.

Per-fold (deterministic): [0.965584, 0.964797, 0.964658, 0.964564, 0.964726],
mean 0.964866, std 0.000367. Beats the baseline on **every** fold.

---

## 7. Submission

The best config beats the baseline by a *defensible in-sample* margin (a small,
consistent paired CV gain; the selection-bias-corrected gain is NOT yet
established — see the summary table and Section 5, the Kaggle LB is the
unbiased arbiter), so `src/tune.py` trained it on ALL
of train and predicted test → **`data/submission_c5_optuna.csv`**
(247,435 rows; columns `id,class`; hard labels; 0 nulls, 0 duplicate ids).
Predicted class mix: GALAXY 63.2%, QSO 20.8%, STAR 16.0% (vs train 65.4/20.3/14.3
— the balanced weighting nudges minority recall up, as intended). NOT uploaded to
Kaggle. Leakage note:
`load_test` is called only AFTER the config is frozen; test never enters CV,
weighting, or tuning (adversarial AUC was 0.4997 in C2 — no train/test shift).

---

## 8. MLflow tracking

- Baseline: experiment `s6e6-c5-baseline`, run `baseline_lgbm_raw` (params,
  per-fold + aggregate metrics, provenance tags, source artifact).
- Study: experiment `s6e6-c5-optuna`, parent run `optuna_study`
  (id `36be6826284140eda231a623724aa03d`) + one nested child per trial. Local
  file store under `mlruns/` (single-host POC; `MLFLOW_ALLOW_FILE_STORE=true` set
  in-code for MLflow 3.x — without it MLflow 3 raises on the file backend).
- Baseline run id is recorded in `docs/c5-poc/baseline_results.json` (`mlflow_run_id`).
