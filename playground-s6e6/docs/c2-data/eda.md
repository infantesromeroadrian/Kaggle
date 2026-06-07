# EDA Report — BL-004 (Cycle C2 Data, closing step)

**Scope:** train-only exploratory analysis + SHAP baseline (ceiling signal).
**Dataset:** Kaggle Playground S6E6 — stellar classification (GALAXY / QSO / STAR).
**Upstream:** leakage audit (0-leakage, adversarial AUC 0.4997 → CV reliable).
**Determinism:** every stochastic step pinned to `seed = 42`.
**Reproduce:** `PYTHONPATH=<repo> .venv/bin/python -m src.eda` (writes figures, prints JSON summary to stdout).

> Hard boundaries honoured: test.csv is NEVER read; folds come ONLY from the
> frozen `data/folds/cv_indices.npz` (BL-005); feature engineering is MOTIVATED
> for C3, not implemented here. The model below is a RAW-feature ceiling signal,
> NOT the official baseline (that is BL-008 in C5).

---

## 1. Data Quality Report

| Metric | Value |
|---|---|
| Rows | 577,347 |
| Columns | 12 (10 features + `id` + `class`) |
| Total NaN | 0 |
| Duplicate rows | 0 |
| Class imbalance (majority/minority) | 4.5631 |

Target distribution (`figures/target_distribution.png`):

| Class | Count | Proportion |
|---|---|---|
| GALAXY | 377,480 | 0.6538 |
| QSO | 117,143 | 0.2029 |
| STAR (minority) | 82,724 | 0.1433 |

The class imbalance is moderate (4.56:1), not extreme. The official metric is
**balanced_accuracy** (unweighted mean of per-class recall), so all modelling
below balances the classes via per-sample weights (`class_weight='balanced'`
equivalent) to respect that metric rather than optimising GALAXY-dominated raw
accuracy.

---

## 2. Univariate Statistics (numeric features)

Figures: `figures/numeric_univariate.png` (histogram + KDE),
`figures/numeric_boxplots.png` (spread / outliers).

| Feature | Mean | Median | Std | Skewness | Excess Kurtosis | NaN % |
|---|---|---|---|---|---|---|
| alpha | 181.6167 | 188.6815 | 96.2429 | -0.055 | -0.403 | 0 |
| delta | 21.8347 | 21.4844 | 18.9336 | 0.175 | -1.120 | 0 |
| u | 22.4419 | 22.5702 | 2.0181 | -0.117 | -0.444 | 0 |
| g | 21.0073 | 21.4678 | 1.7954 | -0.629 | -0.292 | 0 |
| r | 19.9628 | 20.4312 | 1.6490 | -0.662 | -0.451 | 0 |
| i | 19.3789 | 19.6316 | 1.5801 | -0.480 | -0.477 | 0 |
| z | 19.0411 | 19.1886 | 1.5844 | -0.314 | -0.438 | 0 |
| **redshift** | 0.7231 | 0.4975 | 0.8101 | **2.305** | **7.055** | 0 |

Interpretation:

- **`redshift` is heavily right-skewed (skew 2.305, excess kurtosis 7.055).** This
  is the single feature with a non-trivial distributional shape and a long tail
  (QSO tail to ~7). A monotonic transform (`log1p`) is strongly motivated for any
  scale-sensitive downstream model (linear, NN); LightGBM is invariant, so the
  transform is for the *eventual* model family, not this ceiling signal.
- **Photometric bands `u,g,r,i,z` are mildly left-skewed, near-symmetric, low
  kurtosis** — well-behaved magnitudes. No transform needed for them.
- **`alpha` (right ascension) is essentially uniform** (skew ≈ 0, mean ≈ median ≈
  180, the midpoint of [0, 360]). **`delta` (declination) is platykurtic** (excess
  kurtosis -1.12, a broad flat distribution). Both are sky coordinates: any signal
  they carry is survey-footprint geometry, NOT physics — treat with suspicion
  (see §6, SHAP shows them weak).

A D'Agostino-Pearson normality signal was computed on a seeded 5000-row
subsample per feature; every feature rejects normality at the large-n scale,
which is expected and uninformative — reported only as a shape descriptor, not a
gate.

---

## 3. Bivariate — Separability by `redshift`

Figure: `figures/redshift_by_class.png` (clipped histogram + log1p violin).

| Class | n | Mean | Median | Std | q05 | q95 |
|---|---|---|---|---|---|---|
| GALAXY | 377,480 | 0.5090 | 0.4820 | 0.3091 | 0.0594 | 1.0570 |
| QSO | 117,143 | 1.8756 | 1.7989 | 1.0697 | 0.2304 | 3.6706 |
| STAR | 82,724 | 0.0681 | 0.0565 | 0.0645 | -0.0036 | 0.1785 |

This is the headline finding. The three classes occupy **almost disjoint
redshift regimes**:

- **STAR ≈ 0** (median 0.057, q95 0.178). Physically correct: stars are local;
  their redshift is essentially zero (tiny negative values are measurement noise,
  hence the contract floor of -0.01).
- **GALAXY intermediate** (median 0.482, bulk 0.06–1.06).
- **QSO high with a long tail** (median 1.80, q95 3.67).

The GALAXY/STAR boundary near redshift ≈ 0.1 and the GALAXY/QSO boundary near
≈ 1.0 are clean. Residual overlap (low-z QSO, high-z GALAXY) is the part of the
problem that needs the photometric features to resolve.

---

## 4. Multivariate — Photometric Band Collinearity

Figures: `figures/band_correlation_heatmap.png`,
`figures/feature_correlation_heatmap.png`.

Pearson correlation among the SDSS bands (wavelength order u→z):

| | u | g | r | i | z |
|---|---|---|---|---|---|
| u | 1.000 | 0.827 | 0.659 | 0.522 | 0.444 |
| g | 0.827 | 1.000 | 0.912 | 0.805 | 0.732 |
| r | 0.659 | 0.912 | 1.000 | 0.954 | 0.913 |
| i | 0.522 | 0.805 | 0.954 | 1.000 | 0.969 |
| z | 0.444 | 0.732 | 0.913 | 0.969 | 1.000 |

Interpretation:

- **Adjacent bands are severely collinear** (r–i = 0.954, i–z = 0.969, g–r =
  0.912). A bright object is bright in every band, so the *absolute* magnitudes
  largely encode one thing — overall brightness — plus correlated noise.
- **Correlation decays with wavelength separation** (u–z = 0.444 is the weakest
  pair). `u` carries the most independent information of the five bands; it is the
  band where the blue/UV excess of QSOs shows up.
- The physically meaningful, lower-collinearity quantities are the **colour
  indices** (band *differences*): u−g, g−r, r−i, i−z. These isolate the *shape* of
  the spectral energy distribution from the absolute brightness, and are standard
  in SDSS classification. **This is the primary FE recommendation for C3** (§7).

`correlation ≠ causation` disclaimer: these correlations describe association in
the observed train sample; they motivate feature construction, not causal claims.

---

## 5. Categorical Features — Association with Class

Figure: `figures/categorical_crosstabs.png`. Association strength via
bias-corrected **Cramér's V** (vs class):

| Feature | Cramér's V |
|---|---|
| `galaxy_population` | 0.5935 |
| `spectral_type` | 0.5248 |

Row-normalised cross-tabs (given the category, how do classes split):

**`spectral_type`:**

| Category | GALAXY | QSO | STAR |
|---|---|---|---|
| A/F | 0.1985 | 0.5037 | 0.2978 |
| G/K | 0.5678 | 0.1927 | 0.2395 |
| **M** | **0.9496** | 0.0128 | 0.0376 |
| O/B | 0.0828 | **0.7108** | 0.2064 |

**`galaxy_population`:**

| Category | GALAXY | QSO | STAR |
|---|---|---|---|
| Blue_Cloud | 0.3451 | 0.4200 | 0.2349 |
| **Red_Sequence** | **0.9028** | 0.0278 | 0.0694 |

Interpretation:

- Both categoricals are **strongly predictive but not deterministic**:
  `spectral_type == M` → 94.96% GALAXY; `galaxy_population == Red_Sequence` →
  90.28% GALAXY; `spectral_type == O/B` → 71.08% QSO.
- These are nominal (no inherent order), so they are fed to LightGBM via its
  **native categorical handling** (optimal subset splits) — NOT ordinal-encoded,
  which would impose a false order. For C3, target/one-hot encoding is the
  decision to make per the eventual model family (§7).

---

## 6. SHAP Baseline — Ceiling Signal

### 6.1 Model & protocol

- **Model:** LightGBM multiclass (`num_class=3`), 300 trees, lr 0.05,
  `num_leaves=63`, subsample/colsample 0.8, `reg_lambda=1.0`, `random_state=42`.
  Deliberately untuned — this measures the *ceiling* of the RAW features, not a
  baseline.
- **CV:** the 5 **frozen** folds from `cv_indices.npz` (BL-005), applied
  positionally with `.iloc`. Each fold trains on ~461,877 rows, validates on
  ~115,470. Per-sample balancing weights applied to TRAIN rows only.
- **Categoricals:** native LightGBM categorical (cast to `category` dtype).
- **SHAP:** `TreeExplainer` (exact for GBMs) on a seeded 4,000-row subsample of
  the last fold's training slice (model trained on the FULL fold; only the
  explanation set is subsampled, for speed).

### 6.2 Cross-validation result (balanced_accuracy)

| Fold | balanced_accuracy |
|---|---|
| 0 | 0.96505 |
| 1 | 0.96436 |
| 2 | 0.96438 |
| 3 | 0.96425 |
| 4 | 0.96435 |
| **Mean ± std** | **0.96448 ± 0.00029** |

The cross-fold standard deviation is **0.00029** — the frozen, stratified folds
give an extremely stable estimate. Combined with the 0-leakage audit (adversarial
AUC ≈ 0.50), this CV is a trustworthy ceiling.

**Per-class out-of-fold recall** (each row predicted exactly once; folds
partition the train set):

| Class | Recall |
|---|---|
| GALAXY | 0.9528 |
| QSO | 0.9748 |
| STAR (minority) | 0.9658 |

balanced_accuracy = mean of these three = 0.9645. Crucially, the **minority class
STAR is NOT under-served** (recall 0.966) — the balancing weights work as
intended.

**Aggregate out-of-fold confusion** (`figures/confusion_matrix.png`, rows = true,
cols = predicted, order GALAXY / QSO / STAR):

```
              pred GALAXY   pred QSO   pred STAR
true GALAXY      359,675      5,540      12,265
true QSO           1,754    114,190       1,199
true STAR          2,420        408      79,896
```

The dominant errors are **GALAXY misclassified as STAR (12,265)** and **GALAXY as
QSO (5,540)** — exactly the redshift-overlap boundary regions (low-z galaxies look
star-like; high-z galaxies look QSO-like). These are the cases the colour features
should help disambiguate (§7).

### 6.3 SHAP feature ranking

Figure: `figures/shap_importance.png`. Global = mean over classes of per-class
mean(|SHAP|).

| Rank | Feature | Global mean(\|SHAP\|) | GALAXY | QSO | STAR |
|---|---|---|---|---|---|
| 1 | **redshift** | **1.8640** | 1.0768 | 1.5200 | **2.9952** |
| 2 | z | 0.4724 | 0.5249 | 0.6968 | 0.1954 |
| 3 | g | 0.4375 | 0.4666 | 0.5115 | 0.3345 |
| 4 | u | 0.3255 | 0.1371 | **0.6640** | 0.1753 |
| 5 | i | 0.2468 | 0.1741 | 0.4999 | 0.0663 |
| 6 | spectral_type | 0.2244 | **0.5323** | 0.0520 | 0.0888 |
| 7 | alpha | 0.1693 | 0.2180 | 0.0979 | 0.1919 |
| 8 | delta | 0.1377 | 0.2003 | 0.1077 | 0.1052 |
| 9 | r | 0.1346 | 0.1366 | 0.1682 | 0.0989 |
| 10 | galaxy_population | 0.1263 | 0.2204 | 0.1333 | 0.0252 |

Interpretation:

- **`redshift` dominates by ~4x** the next feature (1.864 vs 0.472). It is by far
  the most important driver and is *especially* decisive for STAR (per-class
  |SHAP| 2.995) — consistent with STAR sitting at redshift ≈ 0.
- **Photometric bands `z`, `g`, `u`, `i`** form the second tier. Note the
  per-class structure for QSO: `redshift` dominates (1.520), then `z` (0.697),
  then **`u` (0.664) — QSO-specific: its QSO |SHAP| is ~3.8x its GALAXY (0.137) /
  STAR (0.175) values, the UV-excess signature of quasars** — but `u` is weak for
  GALAXY/STAR. This is signal the colour indices will sharpen.
- **`spectral_type` is GALAXY-specific** (per-class |SHAP| 0.532 for GALAXY, ~0.05
  for QSO) — it mainly helps confirm galaxies (the `M` → GALAXY rule).
- **`r` (0.135) and `galaxy_population` (0.126) are the weakest.** `r` is the most
  redundant band (highest collinearity, r–i = 0.954, r–z = 0.913), so the model
  reads its information through `i`/`z` instead. `galaxy_population` is largely
  subsumed by redshift + spectral_type.
- **`alpha`/`delta` (sky coordinates) are mid-low.** They carry some footprint
  signal but no physics; they are **candidates to scrutinise/drop** in C3 — if they
  help only via survey geometry, they risk not generalising. None of the features
  is SHAP ≈ 0, so nothing is an outright drop on this evidence alone, but `r`,
  `alpha`, `delta`, `galaxy_population` are the ablation candidates.

`correlation ≠ causation`: SHAP attributions are the model's *predictive* reliance
on each feature, not a causal effect of the feature on the class.

---

## 7. Handoff to C3 — Feature Engineering Recommendations (motivated, NOT implemented)

Priority order, each tied to the evidence above:

1. **Colour indices `u−g`, `g−r`, `r−i`, `i−z`** (HIGH priority). §4 shows bands
   are 0.83–0.97 collinear; their differences isolate SED shape from brightness
   and are the SDSS-standard discriminators. SHAP shows `u` is strongly
   QSO-specific (QSO |SHAP| 0.664, ~3.8x its GALAXY/STAR values) — `u−g` should
   directly capture the QSO UV excess. Expected to sharpen the
   GALAXY↔QSO and GALAXY↔STAR boundaries that produce the bulk of current errors
   (§6.2 confusion).
2. **`redshift` transform** (HIGH for scale-sensitive models). skew 2.305 / excess
   kurtosis 7.055 (§2). `log1p(redshift - min)` (the `min` shift covers the -0.01
   noise floor) compresses the QSO tail; the §3 violin already uses it and shows
   clean per-class separation. LightGBM does not need it; a linear/NN baseline
   will.
3. **`redshift` binning into physical regimes** (MEDIUM). The near-disjoint class
   regimes (STAR ~0, GALAXY ~0.5, QSO ~1.8) suggest coarse bins (e.g. <0.1,
   0.1–1.0, >1.0) as an interpretable interaction handle for linear models.
4. **Categorical encoding decision** (MEDIUM). `spectral_type` (V=0.525) and
   `galaxy_population` (V=0.594) are strong nominal predictors. For tree models:
   keep native categorical. For linear/NN: target encoding (with in-fold fit to
   avoid leakage) given the clean category→class skews (M→GALAXY 95%,
   Red_Sequence→GALAXY 90%).
5. **Scrutinise / ablate `r`, `alpha`, `delta`, `galaxy_population`** (MEDIUM).
   Lowest SHAP and (for `r`) highest redundancy. `alpha`/`delta` carry only sky
   geometry — run an ablation in C3 to confirm they generalise rather than fit the
   train footprint.
6. **Brightness aggregate** (LOW / exploratory). A mean or single-band proxy for
   overall magnitude could complement the colours, since the colours deliberately
   remove the brightness dimension.

**Ceiling note for C3/C5:** raw features already reach balanced_accuracy
**0.9645 ± 0.0003**. The formal baseline (BL-008, C5) must clear this bar with
the engineered features; the headroom above 0.9645 is mostly in the redshift
boundary-overlap cases visible in the confusion matrix.

---

## 8. Artifacts

| Artifact | Path |
|---|---|
| EDA + SHAP script | `src/eda.py` |
| This report | `docs/c2-data/eda.md` |
| Target distribution | `docs/c2-data/figures/target_distribution.png` |
| Numeric univariate (hist+KDE) | `docs/c2-data/figures/numeric_univariate.png` |
| Numeric boxplots | `docs/c2-data/figures/numeric_boxplots.png` |
| Redshift by class | `docs/c2-data/figures/redshift_by_class.png` |
| Band correlation heatmap | `docs/c2-data/figures/band_correlation_heatmap.png` |
| Feature correlation heatmap (Spearman) | `docs/c2-data/figures/feature_correlation_heatmap.png` |
| Categorical crosstabs | `docs/c2-data/figures/categorical_crosstabs.png` |
| SHAP importance (global + per-class) | `docs/c2-data/figures/shap_importance.png` |
| Confusion matrix (out-of-fold) | `docs/c2-data/figures/confusion_matrix.png` |

**Gate status:** SHAP baseline produced → C2 handoff complete; C3 is unblocked.
Review pipeline: statistical validation (SHAP + CV correctness) → tech-debt scan → code review.
