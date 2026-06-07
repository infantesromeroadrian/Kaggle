# ML Feasibility Check — Kaggle PGS6E6 "Predicting Stellar Class"

> C1 Discovery feasibility gate. Date: 2026-06-06.
> Verdict at bottom. Inputs: real columns + real target distribution (data downloaded).

---

## 1. Data sufficiency

| Dimension | Value | Assessment |
|---|---|---|
| Train rows | 577,348 | ABUNDANT. ~5.7x the original SDSS17 dataset (~100k). |
| Test rows | 247,436 | Large; batch scoring time is the only concern (S2 covers it). |
| Minority class absolute count | STAR = 14.33% x 577,348 ≈ **82,724 rows** | More than enough even for the minority. No few-shot regime. |
| QSO absolute count | 20.29% x 577,348 ≈ **117,144 rows** | Abundant. |
| GALAXY absolute count | 65.38% x 577,348 ≈ **377,480 rows** | Abundant. |
| Features | 10 usable (alpha, delta, u, g, r, i, z, redshift, spectral_type, galaxy_population) + `id` (drop) | Compact, low-dimensional. Tree boosting thrives here. |

**Verdict — data sufficiency: STRONG.** With ~82k rows even in the minority class and
only ~10 features, this is a comfortable tabular regime. The risk is NOT insufficient
data; it is leakage / metric mismatch (handled below).

## 2. Expected signal

| Feature | Expected role | Signal strength |
|---|---|---|
| `redshift` | Dominant separator. STAR≈0, GALAXY low+, QSO high+. | VERY HIGH — single most predictive feature in all SDSS classification literature. |
| `u,g,r,i,z` photometric bands | Magnitudes; differences (colors) separate object types. | HIGH after deriving colors (u-g, g-r, r-i, i-z). |
| `spectral_type` (categorical) | May be near-deterministic per class (e.g. stellar spectral types only for STAR). | HIGH but **leakage suspect** — audit before use (M3). |
| `galaxy_population` (categorical, e.g. 'Red_Sequence') | Suspected GALAXY-only → near-leak / informative-NaN. | HIGH but **leakage suspect** — likely a target proxy (M3). |
| `alpha`, `delta` (RA/Dec sky position) | Weak; sky position has minor selection-effect signal in synthetic data. | LOW-MEDIUM. Keep but expect little. |
| `id` | Trivial row index. | NONE — drop. |

**Verdict — signal: STRONG.** `redshift` alone gets a tree model far. The honest target
(0.97 balanced_accuracy) is reachable WITHOUT the two leakage-suspect categoricals; they
are upside-if-clean, not load-bearing. This is the crucial feasibility insight: we can hit
the target on physics features alone, so dropping the categoricals if they prove leaky
costs us little.

## 3. Metric feasibility (balanced_accuracy)

- Metric = macro-averaged recall over 3 classes. The minority (STAR, 14.33%) and QSO
  (20.29%) errors are weighted up.
- With ~82k STAR and ~117k QSO training rows, there is no data-scarcity reason for either
  to collapse. Collapse, if it happens, is a *decision-rule* problem (argmax favoring the
  majority), addressable by class weights and per-class thresholds (M4).
- Therefore `balanced_accuracy >= 0.97` is feasible **conditional on** tuning the decision
  rule for balance — which is exactly what M4 mandates and is cheap to implement.

**Verdict — metric: FEASIBLE with M4 in place.**

## 4. Hardware feasibility

| Resource | Requirement | RTX 2000 Ada 8GB | Assessment |
|---|---|---|---|
| Train 577k x ~14 features (after FE) boosting | < 30 min/model | LightGBM/XGBoost/CatBoost GPU handle this in seconds-to-minutes | COMFORTABLE |
| GPU memory | <= 8 GB | 577k x ~14 cols is < 100MB in memory; GPU histograms tiny | TRIVIAL headroom |
| Optuna HPO (50-100 trials x K folds) | overnight-feasible | each trial is fast; full sweep fits in hours | FEASIBLE within 24-day window |
| Test scoring 247k rows | < 5 min | milliseconds per tree-model predict | TRIVIAL |

**Verdict — hardware: NON-ISSUE.** This dataset is small for the RTX 2000 Ada. GPU is a
convenience, not a necessity; even CPU LightGBM would finish in minutes.

## 5. Schedule feasibility

- Deadline 2026-06-30, ~24 days from 2026-06-06. Sprint 1 = 2 weeks.
- Data already downloaded → C2 starts immediately, no external-blocker slip.
- Boosting on clean tabular data is a well-trodden path; the only research-risk item is
  the leakage audit outcome on the two categoricals (which may *remove* features, not add
  schedule).

**Verdict — schedule: FEASIBLE.** 24 days is ample for a boosting + ensemble pipeline on
a 10-feature dataset. The bottleneck is rigor (CV-LB gap discipline, leakage audit), not
compute or time.

## 6. Key risks (feasibility-impacting)

| Risk | Likelihood | Impact on feasibility | Mitigation |
|---|---|---|---|
| **`galaxy_population` is a GALAXY-only target proxy** | High | If used naively → inflated CV that does NOT transfer to LB; if its NaN-pattern itself is informative it may even be a legitimate (rules-permitted) signal. Either way must be audited. | C2 BLOCKING audit (TICKET-003). Default: drop until cleared. Target reachable without it. |
| **`spectral_type` near-deterministic per class** | Medium-High | Same as above. | Same audit. |
| **Minority (STAR) recall collapse under argmax** | Medium | Would miss M2 even at high accuracy. | Class weights + per-class thresholds (M4); balanced_accuracy as primary metric. |
| **CV-LB gap from synthetic quirks** | Medium | Could overfit generator artifacts. | Stratified K-fold + adversarial validation (M6); regularization. |
| **Over-engineering on noise** | Low-Medium | Wasted effort, possible overfit. | Karpathy "simplicity first": redshift + colors + clean categoricals is likely 90% of the score. |

## 7. Synthetic vs real consideration

The data is Playground-synthetic (a generative model trained on real SDSS). Implications:
- Separability may be slightly compressed vs real SDSS17 → target set at 0.97 (not 0.98+).
- Distribution parity between train/test is expected (same generator) — verify with
  adversarial validation (M6, AUC <= 0.55 acceptance).
- Generator may introduce spurious feature interactions; keep models regularized and
  prefer the physically-motivated features (redshift, colors).

---

## Overall Feasibility Verdict

**FEASIBLE — GREEN.**

- Data: strong (577k rows, ~82k even in minority).
- Signal: strong (redshift dominant; target reachable on physics features alone).
- Metric: feasible with M4 decision-rule tuning.
- Hardware: non-issue (dataset tiny for RTX 2000 Ada).
- Schedule: ample (24 days for a boosting pipeline).

**The single thing that could derail it is leakage discipline on the two categoricals.**
That is a *correctness* risk, not a feasibility risk — handled by the C2 BLOCKING audit
(TICKET-003). The honest 0.97 target stands. Proceed to C2.
