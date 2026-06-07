# BL-003 Leakage Audit + BL-005 Freeze Folds — C2 Gate Report

- **Competition:** Kaggle Playground S6E6 — stellar classification (GALAXY / QSO / STAR), multiclass, metric `balanced_accuracy`.
- **Cycle / Gate:** C2 Data — BLOCKING gate. C2 does not close and C3 does not start without this verdict.
- **Auditor:** leakage audit step (C2)
- **Date:** 2026-06-06
- **Data audited (real, full):** train 577,347 rows × 12 cols · test 247,435 rows × 11 cols. 0 NaN (confirmed BL-002).
- **Reproduce:**
  - `uv run python -m src.leakage_audit` (BL-003 analysis, stdout report)
  - `uv run python -m src.make_folds` (BL-005, writes `data/folds/cv_indices.npz`)
- **Seed:** 42 (global RNG + `StratifiedKFold` `random_state`).

---

## 0. Target entropy — the ceiling

```
H(class) = 0.879847 nats = 1.269351 bits
```

This is the reference for every MI verdict. A feature whose `MI(feature; class)`
approaches `H(class)` resolves the entire label entropy on its own — that is a
target proxy and triggers DROP-or-ADR under guardrail B4. The **maximum observed
MI/H is 0.584 (redshift)** — no feature comes close to the ceiling. **No target
proxy exists in this dataset.**

Class distribution (train): GALAXY 65.38% · QSO 20.29% · STAR 14.33%.

---

## 1. Mutual information & association per feature

MI for **numeric** features estimated with sklearn's k-NN estimator (Kraskov/Ross)
on a **stratified sample of 80,000 rows** (variance of the estimator is k-dominated,
so it is stable for n ≫ 10k; sample preserves 65/20/14). MI for **categorical**
features computed **EXACTLY on the full 577,347-row frame** from the contingency
table (closed form). Cramér's V is bias-corrected (Bergsma 2013), interpretation
interpretation bands: 0.1 small / 0.3 medium / 0.5 large.

| feature | type | MI (nats) | MI / H(class) | Cramér's V | V band | verdict |
|---|---|---:|---:|---:|---|---|
| redshift | numeric | 0.5140 | 0.5842 | — | n/a | **USE** |
| spectral_type | categorical | 0.3000 | 0.3409 | 0.5248 | large | **USE** |
| z | numeric | 0.2138 | 0.2430 | — | n/a | **USE** |
| galaxy_population | categorical | 0.1927 | 0.2190 | 0.5935 | large | **USE** |
| u | numeric | 0.1787 | 0.2031 | — | n/a | **USE** |
| g | numeric | 0.1765 | 0.2006 | — | n/a | **USE** |
| i | numeric | 0.1579 | 0.1795 | — | n/a | **USE** |
| alpha | numeric | 0.1260 | 0.1432 | — | n/a | **USE** |
| delta | numeric | 0.1175 | 0.1335 | — | n/a | **USE** |
| r | numeric | 0.0974 | 0.1107 | — | n/a | **USE** |
| id | identity | 0.0004 | 0.0005 | — | n/a | **DROP (not a feature)** |

**Reading.** Every feature sits well below the `MI/H = 1.0` proxy line. The two
strongest associations (`redshift`, `spectral_type`) are legitimate physical
signal, justified in sections 2–3. `id` carries effectively zero information
(MI/H = 0.0005) — it is a trivial index, confirmed below, and must be dropped
before modelling (the loader already documents it is not a feature).

---

## 2. Deterministic-prediction check — categorical cross-tabs

`P(class | value)` (%, row-normalised, full frame). A value that predicted a
class **deterministically** (≈100%) would be a leak. **None does.**

### spectral_type → class

| spectral_type | GALAXY | QSO | STAR | max P |
|---|---:|---:|---:|---:|
| A/F | 19.85 | 50.37 | 29.78 | 50.37 |
| G/K | 56.78 | 19.27 | 23.95 | 56.78 |
| M | 94.96 | 1.28 | 3.76 | **94.96** |
| O/B | 8.28 | 71.08 | 20.64 | 71.08 |

### galaxy_population → class

| galaxy_population | GALAXY | QSO | STAR | max P |
|---|---:|---:|---:|---:|
| Blue_Cloud | 34.51 | 42.00 | 23.49 | 42.00 |
| Red_Sequence | 90.28 | 2.78 | 6.94 | **90.28** |

**Reading.** The strongest single value is `spectral_type = M → GALAXY 94.96%`,
and `galaxy_population = Red_Sequence → GALAXY 90.28%`. Both are *strong* but
*non-deterministic*: each leaves a non-trivial residual mass across the other two
classes (M still has 3.76% STAR + 1.28% QSO). A determinism leak requires ≈100%
on a value; the gap to 100% here is real signal-vs-noise structure the model must
learn, not a copied label. **No categorical value is a target proxy.**

These features describe physical/photometric properties (stellar spectral class,
galaxy colour-magnitude population). In a real SDSS pipeline these are derived
from the *same observations* as the inputs, not from the label — there is no
temporal or target-derived construction path. The association is physics, not
leakage.

---

## 3. redshift — physical signal, not leakage

`redshift` by class (full frame):

| class | median | mean | min | max |
|---|---:|---:|---:|---:|
| STAR | 0.0565 | 0.0681 | −0.0100 | 5.4452 |
| GALAXY | 0.4820 | 0.5090 | −0.0099 | 6.8603 |
| QSO | 1.7989 | 1.8756 | 0.0001 | 7.0108 |

**Reading.** The ordering **STAR ≈ 0 < GALAXY (moderate) < QSO (high)** is exactly
the textbook astrophysics: stars are within the Milky Way (negligible cosmological
redshift), galaxies are at moderate cosmological distance, quasars are the most
distant/high-redshift objects. The slight negative floor (−0.01) is measurement
noise near zero, already admitted by the BL-002 schema contract. `redshift` is the
single most informative feature (MI/H = 0.584) **because the physics makes it so**,
not because it leaks the label. **Verdict: USE, justified as legitimate physical
signal.** The class distributions overlap heavily (max reaches 5.4–7.0 for all
three classes), so it is informative, not separating — consistent with a non-proxy.

---

## 4. Adversarial validation — train vs test distribution

Binary classifier train(0) vs test(1), LogisticRegression on standardised numerics
+ ordinal-encoded categoricals (linear model chosen deliberately: conservative
shift signal, no spurious nonlinear separability), 5-fold stratified OOF, `id`
excluded.

```
OOF ROC AUC = 0.4997
```

**Reading.** AUC ≈ 0.50 = the classifier cannot distinguish train from test —
**train and test are drawn from the same distribution. No covariate shift.** This
is the ideal outcome: CV scores on train will track the leaderboard. NON-blocking
in any case (covariate shift is documented, not gated), but here there is nothing
to document beyond "clean".

---

## 5. Duplicates

Row signature = all feature columns, **excluding `id`** (trivial index) and
`class` (train-only). Two rows are the same observation iff every feature matches.
Cross-split is a **set-membership count** (each side deduplicated before the
intersection), so a row repeated in either split counts once — not a join that
would multiply multiplicities.

| check | count | status |
|---|---:|---|
| train internal feature-row duplicates | 0 (0 groups) | clean |
| **cross-split (distinct feature-rows in both splits)** | **0** | **clean — no leakage** |

**Reading.** Zero cross-split overlap. There is no path for inflated metrics via
a test row memorised from train. With 9+ continuous photometric columns to full
float precision, a coincidental exact match would be astronomically improbable, so
0 is the genuine and only safe value. **PASS.**

---

## 6. id sanity

```
MI(binned id; class) = 0.0004 nats   (MI/H = 0.0005)
```

`id` (binned into 256 equal-width bins to avoid finite-sample plug-in inflation
from a 577k-level categorical) carries effectively zero class information. It is a
trivial monotone index. **Verdict: DROP before modelling** — not a leak, just not
a feature. The loader already flags `id` as identity-only.

---

## Per-feature verdict table (B4 guardrail)

> **Guardrail B4:** any model whose score depends on a leaking feature scores ZERO.
> A leak here would force DROP or an ADR before C3. **No leak found.**

| feature | MI/H | Cramér's V | verdict | justification |
|---|---:|---:|---|---|
| redshift | 0.5842 | — | **USE** | Most informative; physical ordering STAR<GALAXY<QSO. Not a proxy (MI/H ≪ 1, heavy overlap). |
| spectral_type | 0.3409 | 0.5248 | **USE** | Strong (large V) but non-deterministic (max P 94.96%); physical spectral class, not label-derived. |
| z | 0.2430 | — | **USE** | SDSS magnitude, moderate signal. |
| galaxy_population | 0.2190 | 0.5935 | **USE** | Strong (large V) but non-deterministic (max P 90.28%); physical colour-magnitude population. |
| u | 0.2031 | — | **USE** | SDSS magnitude. |
| g | 0.2006 | — | **USE** | SDSS magnitude. |
| i | 0.1795 | — | **USE** | SDSS magnitude. |
| alpha | 0.1432 | — | **USE** | Right ascension (astrometry), weak signal. |
| delta | 0.1335 | — | **USE** | Declination (astrometry), weak signal. |
| r | 0.1107 | — | **USE** | SDSS magnitude, weakest photometric. |
| id | 0.0005 | — | **DROP** | Trivial index, zero signal. Not a feature; drop before C3. |

No feature is `USE-WITH-ADR` and none is a leak: the two large-V categoricals are
explicitly cleared because their association is strong-but-non-deterministic and
physically grounded, not constructed from the target.

---

## BL-005 — Frozen folds

- **Splitter:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` (the
  `cv_splitter` instantiated in `data_loader.py`, executed here for the first time).
- **Artifact:** `data/folds/cv_indices.npz` (10 arrays: `fold{k}_train`, `fold{k}_val`)
  + `data/folds/cv_indices_meta.json` (provenance).
- **Indices are positional** (into the `load_train()` RangeIndex) — apply with `.iloc`.
- **Versioning:** `data/` is gitignored (`**/data/`). Folds are **reproducible from
  seed 42**, so the `.npz` is a build artifact and is NOT committed. The generating
  script `src/make_folds.py` IS committed.

### Fold integrity (all asserted, re-verified from the persisted `.npz`)

| check | result |
|---|---|
| every row validated exactly once (∪ val = 0..577346, no repeats) | PASS |
| no cross-fold validation overlap | PASS |
| train/val disjoint within each fold | PASS (overlap = 0 for all 5) |
| class proportions preserved per fold | PASS (65.38/20.29/14.33 on all 5) |

| fold | n_train | n_val | val GALAXY | val QSO | val STAR |
|---|---:|---:|---:|---:|---:|
| 0 | 461,877 | 115,470 | 0.6538 | 0.2029 | 0.1433 |
| 1 | 461,877 | 115,470 | 0.6538 | 0.2029 | 0.1433 |
| 2 | 461,878 | 115,469 | 0.6538 | 0.2029 | 0.1433 |
| 3 | 461,878 | 115,469 | 0.6538 | 0.2029 | 0.1433 |
| 4 | 461,878 | 115,469 | 0.6538 | 0.2029 | 0.1433 |

---

## Estimator limitations (W2 / W3 — disclosed per math-critic)

These do not change any verdict; they bound how the numbers should be read.

### W2 — the MI/H column is a coarse ORDINAL ranking, not a uniform estimator

The MI/H column mixes **two different estimators**:

- **Numeric features** (`alpha … redshift`): MI estimated with sklearn's
  **k-NN estimator** (Kraskov/Ross) on a **stratified 80k-row sample**. This
  estimator has its own bias (k-dependent) and sampling variance.
- **Categorical features** (`spectral_type`, `galaxy_population`, `id`): MI
  computed **EXACTLY** via the closed-form plug-in on the **full 577,347-row**
  contingency table — no sampling, a different (and smaller) bias profile.

Because the two halves of the column carry **heterogeneous biases**, the MI/H
numbers are **not commensurable to the third decimal** across the numeric/
categorical boundary. Treat the column as a **coarse ordinal ranking** ("which
features carry more signal"), not as a calibrated cross-estimator metric.

**Why this is safe for the gate.** The verdict depends only on the *ordering* and
on the *gap to the proxy line* (`MI/H = 1.0`), both of which are robust to this
bias (it moves the 3rd decimal, not the rank): `redshift` dominates, `id ≈ 0`,
and the largest value (0.584) is nowhere near 1.0. **No USE/DROP verdict turns on
a difference this bias could flip.** The categorical headline numbers
(`spectral_type`, `galaxy_population`) are the EXACT ones, so the two features
most scrutinised for leakage are measured without sampling at all.

### W3 — adversarial validation AUC is conservative by construction

The adversarial-validation design matrix encodes the nominal categoricals with an
**`OrdinalEncoder`**, which imposes an **artificial order** on unordered
categories (`A/F < G/K < M < O/B` is meaningless). For the **linear**
`LogisticRegression` used here, an arbitrary ordinal axis can only **underfit**
the true categorical separability — a linear boundary cannot exploit a category
structure that the encoding scrambled.

Therefore the reported **AUC = 0.4997 is, if anything, a lower bound** on any
separability those categoricals could contribute: the measurement is biased in
the **safe direction** ("do not over-report shift"). Since even this conservative
read lands at chance (≈0.50), the **no-covariate-shift conclusion is robust** — a
more expressive encoding could only push the AUC up from an already-null result,
and the numeric features (one-hot-free, faithfully standardised) already had full
opportunity to separate the splits and did not. The categoricals are a minor part
of the design; the conclusion does not hinge on them.

---

## GATE VERDICT

```
BLOCKANTES: 0
ADVERTENCIAS: 0

VEREDICTO: APROBADO — C2 cierra, se autoriza avance a C3.
```

- **Leakage (temporal / target / cross-split):** none. No temporal column exists;
  no feature is a target proxy (max MI/H = 0.584 ≪ 1.0); 0 cross-split duplicates.
- **Drift train vs test:** none (adversarial AUC 0.4997).
- **Folds:** frozen, disjoint, complete, stratified — reproducible from seed 42.
- **Feature decisions for C3:** keep all 10 features (USE). Drop `id` (identity).
  `spectral_type` and `galaxy_population` are explicitly cleared despite large
  Cramér's V — strong-but-non-deterministic physical signal, not leakage.

Handing C2 → C3. EDA on the frozen partition may proceed.

---

## Statistical corrections applied (post-audit)

The statistical review approved the audit (0 blockers, 0-leakage confirmed) and requested
three cheap corrections, all applied before closing C2:

- **W1 (fix):** `make_folds.verify_folds` now ASSERTS per fold that
  `train ∪ val == arange(n)` (check 3b), matching the docstring's promise to
  verify completeness rather than assume it. A fold that silently drops a row now
  fails the gate. Re-run on real data still passes (folds are a true partition).
- **Tests:** `src/tests/test_leakage_stats.py` formalises the identities
  verified by hand — `H(class) ≈ 0.8798`, `MI(y;y) == H(y)`,
  `MI(x;y) ≤ H(y)`, `MI(indep) ≈ 0`, `V(y,y) == 1`, `V(indep) ≈ 0`, and the W1
  completeness failure. **15/15 pass**; coverage of the four stats functions is
  **98.2%** aggregate (each ≥ 80%).
- **W2 / W3 (doc):** the "Estimator limitations" section above discloses the
  heterogeneous-estimator caveat (W2) and the conservative-AUC caveat (W3).

## Tech-debt and code-review corrections applied (post-audit)

The tech-debt scan (3 blockers) and code review (1 blocker + warnings) reviewed
the scripts; all fixes applied, gate verdict unchanged:

- **debt B1/B2/B3:** `N_SPLITS` and `SEED` imported from `data_loader` (no
  literal `5`, no local seed); `_STRAT_TOL` named constant for the BL-005
  stratification threshold; `roc_auc_score` hoisted to top-level imports.
- **critic B1 (BLOCKER, fixed):** `duplicate_report` cross-split count was a raw
  inner merge, which multiplies multiplicities (k train copies × m test copies =
  k·m rows) and would OVER-REPORT leakage on a split with internal duplicates. It
  is now a set-membership test — both sides deduplicated before the merge — so a
  shared feature-row counts once. Real-data result unchanged (cross-split = 0,
  train has no internal dups), but the count is now correct for any input. Guarded
  by `test_cross_split_counts_distinct_membership_not_multiplicity_b1` (3 train
  copies + 1 test copy ⇒ asserts == 1, not 3).
- **critic W-A1/W-A2/W-A3/W-A4:** `duplicate_report` internal-dup branch now
  tested (rows + groups); `SEED` unified with `data_loader`; `zip(..., strict=True)`
  in `numeric_mi_nats`; rare-class sampling guarantee scoped to the current
  dataset in the docstring.

Tests: **43/43 pass** (4 new duplicate_report tests). Coverage of the five
audited functions (entropy, MI, Cramér's V, duplicate_report, verify_folds) is
**98.5%** aggregate, each ≥ 80%. `ruff F401/F841` clean on all three files.

## Code review gate note

Artifacts `src/leakage_audit.py`, `src/make_folds.py` and
`src/tests/test_leakage_stats.py` are pending code review re-approval (cycle
1/2) after the B1 fix. This report records the executed numbers; everything is
reproducible from seed 42.
