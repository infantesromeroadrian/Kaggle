# C6 Build — Ensemble Report (BL-024 / BL-025 / BL-026)

Kaggle Playground S6E6 — stellar classification (GALAXY / QSO / STAR).
Metric: balanced_accuracy. Reference to beat: the C5 tuned LightGBM
(deterministic CV mean **0.964866**, public LB **0.96521**).

**Verdict (step 2, tuned): a nested-CV weight-optimised soft-voting blend of the
THREE TUNED models BEATS LightGBM solo DEFENSIBLY — mean 0.965360 vs 0.964866,
paired Δ = +0.000494 = ~7·SEM above zero, positive on all 5 folds. Submission
generated: `data/submission_c6_ensemble_tuned.csv` (weights LightGBM 0.50 /
CatBoost 0.083 / XGBoost 0.417).**

This supersedes the step-1 verdict (naive equal-weight blend of UNTUNED CatBoost +
XGBoost lost to LightGBM). The statistical review certified that step-1 loss as legitimate:
the members were not at parity. Step 2 brings them to parity by tuning, then wins.

---

## 1. What changed from step 1

| | Step 1 (naive) | Step 2 (tuned) |
|---|---|---|
| CatBoost | defaults (depth 6, lr 0.05, 600 trees) → 0.957229 | **Optuna-tuned** → 0.962662 (+0.0054) |
| XGBoost | defaults (depth 6, lr 0.05, 600 trees) → 0.962944 | **Optuna-tuned** → 0.964830 (+0.0019) |
| Blend | equal weights → 0.963518 (**lost**, −0.00135) | nested-CV weights → 0.965360 (**wins**, +0.000494) |

Tuning closed the strength gap: XGBoost reached effective parity with LightGBM
(−0.00004) and CatBoost came within 0.0022. With members at parity, a weighted
blend that down-weights the weakest member (CatBoost) finally beats the champion.

---

## 2. Tuned hyperparameters

Both studies: Optuna TPESampler(seed=42) + MedianPruner, maximising mean
balanced_accuracy on the IDENTICAL frozen 5 folds, sample_weight balanced
(train-slice only), native categorical handling. Best params re-evaluated
deterministically with `iterations`/`n_estimators` fixed to the search's mean best
iteration (no early-stopping optimism in the reported number).

### CatBoost (BL-025) — GPU, `docs/c6-build/catboost_tuning.json`

| param | value |
|---|---|
| depth | 7 |
| learning_rate | 0.14176 |
| l2_leaf_reg | 1.5697 |
| border_count | 254 |
| random_strength | 2.5704 |
| iterations (fixed) | 999 |

Budget: 22-min timeout bound first → **17 of 30 trials** completed (deep depth-9/10
trials cost ~2.5 min each on GPU). Best = trial 16, report mean 0.962662 ± 0.000293.

### XGBoost (BL-026) — CPU 4 threads, `docs/c6-build/xgboost_tuning.json`

| param | value |
|---|---|
| max_depth | 6 |
| learning_rate | 0.09325 |
| min_child_weight | 1.7882 |
| subsample | 0.6750 |
| colsample_bytree | 0.7557 |
| reg_alpha | 0.001017 |
| reg_lambda | 0.056561 |
| gamma | 1.0646 |
| n_estimators (fixed) | 798 |

Budget: **15 trials completed, 4 pruned** in ~36 min. CPU XGBoost is ~10x CatBoost's
GPU cost per trial (~145s/trial), so trials were REDUCED by design (15 vs CatBoost's
30) with an aggressive MedianPruner. TPE around an already-parity region (a single
benchmark fold reached 0.96492) needed to confirm + refine, not explore 50 trials.
Best = trial 13, report mean 0.964830 ± 0.000306.

---

## 3. Per-fold results (tuned base models)

balanced_accuracy per frozen fold; mean ± population std (ddof=0).

| Fold | LightGBM (ref) | CatBoost-tuned | XGBoost-tuned | Blend (equal) | Blend (nested-w) |
|---|---|---|---|---|---|
| 0 | 0.965584 | 0.963189 | 0.965387 | 0.965558 | 0.966088 |
| 1 | 0.964797 | 0.962622 | 0.964624 | 0.964944 | 0.965075 |
| 2 | 0.964658 | 0.962639 | 0.964514 | 0.964688 | 0.965064 |
| 3 | 0.964564 | 0.962284 | 0.964902 | 0.964576 | 0.965227 |
| 4 | 0.964726 | 0.962574 | 0.964721 | 0.964907 | 0.965346 |
| **mean** | **0.964866** | 0.962662 | 0.964830 | 0.964935 | **0.965360** |
| **std** | 0.000367 | 0.000293 | 0.000306 | 0.000340 | 0.000379 |

**LightGBM fidelity guard: PASSED** (`lgbm_fidelity_ok: true`). Our re-fitted
LightGBM reproduces the C5 `best_deterministic.fold_scores` bit-for-bit to 6
decimals on all 5 folds — the comparison is like-for-like.

---

## 4. Paired comparison vs LightGBM solo (Δ and SEM_diff)

Per fold, d_k = candidate_k − LightGBM_k. Δ = mean(d), SEM_diff = std(d, ddof=1)/√5.
Defensible iff Δ > 2 · SEM_diff (the shared 2-sigma-on-5-folds rule from C3/C5).

| Candidate | Δ (mean paired) | SEM_diff | 2·SEM_diff | Defensible? |
|---|---|---|---|---|
| CatBoost-tuned solo | −0.002204 | 0.0000632 | 0.000126 | No (Δ < 0) |
| XGBoost-tuned solo | −0.0000357 | 0.0000994 | 0.000199 | No (parity, Δ≈0) |
| Blend (equal weights) | +0.0000690 | 0.0000405 | 0.0000809 | **No** (Δ < 2·SEM) |
| **Blend (nested-CV weights)** | **+0.000494** | **0.0000703** | **0.000141** | **YES** (Δ ≈ 7·SEM) |

The equal-weight blend now BEATS LightGBM at the point estimate (+0.000069) but not
defensibly (Δ < 2·SEM) — averaging in CatBoost (still 0.0022 below) dilutes the
gain. The nested-weight blend, which down-weights CatBoost to 1/12, clears the
2-sigma bar comfortably: Δ = +0.000494 with SEM 0.0000703, i.e. ~7 SEM above zero,
and is positive on EVERY fold.

---

## 5. Weight method — nested cross-validation (no CV overfitting)

The headline weights were NOT fit on the same folds they are scored on (that would
be CV overfitting — the statistical-validation review rejects it). Protocol:

1. For each held-out fold k ∈ {0..4}:
   a. Select blend weights on the OTHER 4 folds' OOF val rows ONLY (inner objective:
      maximise balanced_accuracy of the weighted blend over a 91-point simplex grid,
      n_steps=12 so equal weights are an exact grid point; ties break toward equal).
   b. Apply those weights to fold k's OOF rows and record its balanced_accuracy.
   Fold k played NO part in choosing its own weights.
2. The 5 held-out scores are the reported `blend_nested_weights.fold_scores`, paired
   with the LightGBM reference fold-by-fold.

**Robustness signal:** all 5 folds independently selected the SAME weight vector
**(LightGBM 0.500, CatBoost 0.083, XGBoost 0.417)**. Identical weights across 5
disjoint inner-training sets means the optimum is stable, not a per-fold artifact —
strong evidence the weighting is not overfit.

The grid search is unit-tested for the leakage property: a dedicated test
(`test_best_weights_ignore_held_out_fold`) corrupts ONLY the held-out fold and
asserts the selected weights are unchanged, proving the selector never reads the
fold it is scored on.

**Submission weights:** for the deliverable (trained on ALL of train, no held-out
fold), one weight vector is re-selected on the FULL OOF (all 5 folds). This is sound
— there is no held-out fold to leak into a model trained on everything, and the CV
verdict that GATED the submission already came from the leakage-free nested scores.
The full-OOF vector matched the per-fold one exactly: (0.500, 0.083, 0.417).

---

## 6. GPU / CPU decision (unchanged, re-confirmed)

- **CatBoost: GPU.** ~14.6x over CPU (step-1 benchmark); in step 2, ~10s/fold for
  999-iter folds. The only way 17 tuning trials + the OOF fit fit the budget.
  CatBoost GPU MultiClass is not bit-deterministic run-to-run (per-fold std ~0.0003,
  pinned random_seed=42); this does not affect the verdict — the blend wins by ~7
  SEM, far outside that variance.
- **XGBoost: CPU, 4 threads.** The C5 sweet spot (−1 over-subscribes 22 cores and is
  catastrophic). ~23s/fold for 798-iter folds.
- **LightGBM: CPU, 1 thread, deterministic** (the C5 regime; fidelity guard tight).

The two studies ran IN PARALLEL (CatBoost GPU + XGBoost CPU) — no resource
contention — so the ~58 min of combined tuning fit the ~45-60 min envelope.

---

## 7. Verdict

- **Does the tuned blend beat LightGBM solo defensibly?** **YES** — the nested-CV
  weight blend: Δ = +0.000494, ~7·SEM above zero, positive on all 5 folds.
- **Does the equal-weight blend?** Beats at the point (+0.000069) but NOT defensibly
  (Δ < 2·SEM) — CatBoost still drags it. So the weighting earns its place.
- **Does it reach the 0.97 stretch?** No (0.965360). 0.97 was never guaranteed.
- **Expected LB?** CV proved faithful in C5 (LB 0.96521 > CV 0.96487, a +0.0007
  gap). Trusting that fidelity, the blend's CV 0.965360 predicts an LB around
  ~0.9660 — above LightGBM's 0.96521. The CV margin over LightGBM is small
  (+0.0005) but defensible and consistent, so an LB improvement is credible though
  modest.
- **Submission generated?** **Yes** — `data/submission_c6_ensemble_tuned.csv`
  (247,435 rows, id,class hard labels, validated: no NaN, no dup ids, ids match
  test exactly, class dist 63/21/16% GALAXY/QSO/STAR).

**Recommendation: ship the tuned nested-weight blend.** It is a real, defensible,
fold-consistent improvement over the C5 LightGBM, with weights obtained without CV
overfitting. The gain is small (~0.0005 CV) — a known property of this dataset where
redshift dominates and all GBDTs largely agree — but it is positive on every fold
and earns the added complexity. NOT shipping would leave a measured, honest gain on
the table.

---

## 8. Reproducibility

- seed = 42 everywhere; folds = frozen `data/folds/cv_indices.npz` (positional).
- LightGBM reproduces C5 to 6 decimals (fidelity guard).
- CatBoost/XGBoost params read from their tuning JSONs (single source of truth;
  ensemble.catboost_params / xgboost_params).
- CatBoost GPU carries small run-to-run variance (documented; verdict robust to it).
- Tuning: CatBoost ~23 min (17/30 trials, timeout-bound), XGBoost ~36 min (15
  trials, 4 pruned), run in parallel. OOF ensemble: ~8 min (15 fold-fits + nested
  weighting on precomputed OOF).
- Canonical records: `docs/c6-build/{ensemble_results,catboost_tuning,xgboost_tuning}.json`.
- MLflow: experiments `s6e6-c6-ensemble` (parent + model/blend children),
  `s6e6-c6-catboost-tuning` (17 trials), `s6e6-c6-xgboost-tuning` (15 trials).
- Tests: 131 pass (15 new in C6 covering blend arithmetic, OOF no-leakage, nested-CV
  weight leakage guard, tuned-param loaders, both search spaces).
