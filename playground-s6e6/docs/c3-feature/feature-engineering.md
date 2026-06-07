# C3 Feature Engineering — BL-006 Report

**Cycle:** C3 (Feature & Hypothesis) · **Item:** BL-006 (MUST) · **Owner:** ML engineering
**Metric:** `balanced_accuracy` (macro-recall, 3 classes GALAXY/QSO/STAR)
**Frozen C2 baseline:** 0.96448 ± 0.00029 (positional 5-fold CV, seed 42)
**Targets:** ≥0.97 (min) / ≥0.98 (stretch)

> **Verdict (headline):** The engineered features do **NOT** beat the frozen
> baseline with a statistically defensible margin, and **0.97 is NOT reached.**
> The largest paired gain (adjacent-band colours) is +0.000086 balanced_accuracy
> with a paired SEM of 0.000129 — i.e. mean(Δ) < 2·SEM, indistinguishable from
> noise. **This is a negative result reported honestly; no overfitting was applied
> to manufacture a pass.** The ~0.0055 gap to 0.97 remains open and should be
> escalated for a strategy decision (model class / tuning /
> stacking), not closed by feature engineering on this raw feature set.

---

## 1. Method (why the comparison is fair)

Every configuration was evaluated by **real execution** (no estimates) on the
**identical frozen folds** (`data/folds/cv_indices.npz`, applied with `.iloc`),
with the **identical LightGBM config and balanced sample weights** as the C2
baseline. The only thing that varied between configurations was the set of
feature **columns** handed to the model.

- **Model:** LightGBM multiclass, 300 trees, lr 0.05, 63 leaves, depth -1,
  subsample 0.8 / colsample 0.8, reg_lambda 1.0, seed 42. Native categorical
  handling for `spectral_type`, `galaxy_population`. Balanced sample weights
  (1/freq, mean 1) on train rows only.
- **Determinism:** `deterministic=True`, `force_row_wise=True`, `num_threads=1`,
  `n_jobs=1`. Two runs of the harness produce identical numbers (the exploratory
  C2 config did not need this; a paired statistical test does).
- **Paired statistic (per config vs baseline):** for fold k,
  `d_k = score_config[k] − score_baseline[k]`; report `mean(d)` and the standard
  error of the paired differences `SEM_diff = std(d, ddof=1)/√5`. A delta is
  **defensible iff `mean(d) > 2·SEM_diff`** (a conservative ~2σ paired signal on
  only 5 folds).
- **Leakage:** all engineered features come from `src/features.py`, which is
  **row-wise pure** (each output depends only on its own row + compile-time
  constants; the redshift floor is a parameter, never `df.min()`). Verified by a
  split-invariance test (engineering before vs after a row split is byte-identical)
  and a target-permutation test. Adversarial AUC and 0-leakage guardrails from C2
  are untouched.
- **Reproducibility:** `baseline_raw` reproduces the frozen baseline to 5 decimals
  (**0.964479** vs 0.96448), which proves the harness is faithful — the deltas
  below are real signal differences, not harness artefacts.

Canonical machine-readable results: `docs/c3-feature/feature_eval_results.json`.
Total wall-clock: 977 s (single-thread determinism).

---

## 2. Results — per-fold balanced_accuracy

| config | fold0 | fold1 | fold2 | fold3 | fold4 | mean | std | Δ vs base (paired) | SEM_diff | defensible (>2·SEM)? |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline_raw** | 0.965048 | 0.964363 | 0.964385 | 0.964246 | 0.964354 | **0.964479** | 0.000288 | — | — | — (reference) |
| **colors** | 0.965205 | 0.964772 | 0.964032 | 0.964231 | 0.964587 | 0.964566 | 0.000412 | **+0.000086** | 0.000129 | **No** (0.000086 < 0.000258) |
| **colors_redshift_log** | 0.965152 | 0.964362 | 0.964100 | 0.964613 | 0.964408 | 0.964527 | 0.000353 | +0.000048 | 0.000104 | No (0.000048 < 0.000209) |
| colors_drop_r | 0.965154 | 0.964688 | 0.963866 | 0.964244 | 0.964682 | 0.964527 | 0.000438 | +0.000047 | 0.000155 | No |
| colors_drop_alpha | 0.955798 | 0.955557 | 0.955963 | 0.954863 | 0.955249 | 0.955486 | 0.000393 | **−0.008993** | 0.000172 | No (negative) |
| colors_drop_delta | 0.958386 | 0.958583 | 0.957546 | 0.957876 | 0.958643 | 0.958207 | 0.000427 | **−0.006273** | 0.000228 | No (negative) |
| colors_drop_galaxy_population | 0.965038 | 0.964626 | 0.963902 | 0.964179 | 0.964535 | 0.964456 | 0.000389 | −0.000023 | 0.000130 | No |

`std` is population std (ddof=0) to match the C2 baseline ± convention. `SEM_diff`
is the sample SEM (ddof=1) of the 5 paired differences. Defensibility threshold is
`2·SEM_diff`.

---

## 3. Interpretation

### 3.1 Colours add no defensible signal (the core hypothesis fails)
Adding the four adjacent-band colours (u−g, g−r, r−i, i−z) moved the mean by
**+0.000086**, with a paired SEM of 0.000129. The gain is **smaller than 1×SEM**,
let alone the 2×SEM defensibility bar. Mechanistically: the colours are exact
linear combinations of magnitudes already in the model, and on 461k training rows
the GBM has more than enough splits to recover the relevant magnitude interactions
on its own. The "tree approximates linear interactions inefficiently" argument is
real in principle but is **not the binding constraint** at this sample size — the
data already saturates what raw magnitudes can express.

### 3.2 Monotonic invariance of `redshift_log` — CONFIRMED empirically
The control config `colors_redshift_log` adds `log1p(redshift − floor)` on top of
the colours. Predicted effect under a tree's single-variable monotonic-transform
invariance: **Δ ≈ 0**. Measured: **+0.000048** vs the colours config's +0.000086,
i.e. the log transform *removed* a hair of the (already-noise) colour gain rather
than adding anything. This is exactly the predicted null effect. **The hypothesis
that a monotone redshift transform alone moves a LightGBM metric is empirically
falsified on this dataset.** Any future redshift gain must come from *interactions*
(e.g. redshift × colour ratios), not a univariate transform.

### 3.3 Ablations — which raw features actually carry load
- **`alpha` (right ascension): −0.0090.** Removing it costs ~0.9 pp — the single
  largest ablation effect. Sky position is genuinely predictive here, almost
  certainly via the SDSS **survey footprint** (different sky regions have different
  class mixes / selection functions). This is a dataset artefact, not astrophysics,
  but it is real, in-distribution signal (C2 adversarial AUC 0.4997 → no covariate
  shift, so it is safe to use).
- **`delta` (declination): −0.0063.** Same story, smaller magnitude.
- **`r` band: +0.000047 (≈0).** With colours present, the `r` magnitude is
  redundant — dropping it costs nothing defensible. A candidate for pruning if a
  leaner model is ever wanted (no accuracy case for it on its own).
- **`galaxy_population`: −0.000023 (≈0).** Carries no marginal signal once the
  other features are present. Also a pruning candidate.

---

## 4. Decision — what to retain / drop

| feature(s) | decision | rationale |
|---|---|---|
| raw magnitudes u,g,r,i,z | **keep** | core inputs; the model needs the absolute scale, even if `r` is individually redundant |
| `alpha`, `delta` | **keep (important)** | largest ablation effects (−0.009 / −0.006); real footprint signal, no covariate shift |
| 4 colours (u_g,g_r,r_i,i_z) | **neutral — keep as harmless** | no defensible gain, no harm; cheap, interpretable, may help a *different* model class. Do NOT count them as the path to 0.97 |
| `redshift_log` | **drop** | confirmed-null univariate transform; redundant with raw redshift under tree invariance |
| `galaxy_population` | keep (no harm) / prune-candidate | ≈0 marginal signal; keep for now, prune only if a leaner model is needed |
| `spectral_type` | keep | not ablated here (strong categorical per C2 Cramér's V); no reason to touch |

**Net engineered set worth carrying forward:** the four colours (as neutral,
interpretable extras) — and that is all. `redshift_log` is dropped.

---

## 5. Verdict and escalation

- **Beats the frozen 0.96448 baseline?** No — not defensibly. Best paired Δ is
  +0.000086 (colours), well under 2×SEM. Effectively a tie with the baseline.
- **Reaches 0.97?** No. Best mean is 0.964566; the ~0.0055 gap to 0.97 is **fully
  intact.** Feature engineering on this raw feature set does not close it.
- **Overfitting applied to force a pass?** No — and deliberately so. Per the
  Karpathy principles and the brief, the honest result is reported as-is.

**Escalation note:** the gap to 0.97 will not
be closed by univariate/colour features. Candidate next moves, in rough order of
expected payoff, all out of BL-006 scope:

1. **Hyperparameter tuning** of LightGBM (Optuna; more trees + lower lr, deeper
   leaves, tuned `min_child_samples`/regularisation). The current config is the
   untuned C2 ceiling config — there is likely headroom here before anything fancy.
2. **Interaction features** the tree cannot cheaply synthesise: `redshift × colour`,
   colour *ratios*, magnitude-normalised colours. (Univariate transforms are ruled
   out by §3.2.)
3. **Model class / ensemble:** XGBoost + LightGBM + a calibrated linear model
   stack; or a small tabular MLP (only if tuning + interactions stall).
4. **Threshold / decision tuning per class** against balanced_accuracy directly.

Recommend C3 closes BL-006 with this negative result documented, and the 0.97
target is pursued in C5/C6 via tuning + interactions, not via more univariate
features.

---

## 6. Artefacts

- `src/features.py` — row-wise pure feature module (colours + redshift_log control).
- `src/feature_eval.py` — frozen-fold ablation harness (paired stats, optional MLflow).
- `src/tests/test_features.py` — 29 tests, 100% coverage of `src/features.py`.
- `docs/c3-feature/feature_eval_results.json` — canonical per-fold results.
- `pyproject.toml` — added `lightgbm` (runtime), `shap` (runtime), `mlflow`
  (optional `tracking` extra); coverage source extended to `src.features`.
