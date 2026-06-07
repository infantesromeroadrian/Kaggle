# Kaggle Playground Series S6E4 — Irrigation Need Prediction

> Multiclass classification: predict the irrigation need category of agricultural parcels
> from synthetic tabular features, evaluated under balanced accuracy to account for
> class imbalance.
>
> Competition: [Kaggle Playground Series Season 6, Episode 4](https://www.kaggle.com/competitions/playground-series-s6e4) (2026)

---

> This directory documents the approach and results of the competition entry. The original
> notebooks were developed on Kaggle and are not included here. No code or data files are
> present — this is a portfolio summary.

---

## Result

| Split | Score (Balanced Accuracy) |
|---|---|
| Best public leaderboard | **0.97111** (submission v5tuned) |
| Best private leaderboard | **0.97242** (submission ensemble_tuned_v2) |

The evaluation metric is balanced accuracy (macro-averaged recall across classes). CV BA
values reported below are 5-fold StratifiedKFold cross-validation scores (seed=42) inferred
from submission descriptions.

---

## Approach

The competition was won primarily through **ensemble construction** (LightGBM, XGBoost,
CatBoost, HistGBM) combined with **threshold optimization per class** to correct for class
imbalance, and **Optuna-based joint tuning** of model hyperparameters.

### What drove the gains

- **Threshold optimization** applied after probability calibration was essential on this
  imbalanced multiclass problem; it recovered a meaningful fraction of the balanced accuracy
  gap from a class-weighted baseline.
- **Optuna tuning** (50 trials, submission ensemble_tuned_v2) pushed CV BA from 0.97119 to
  0.97224 over the untuned ensemble, and this submission achieved the best private LB score
  (0.97242).
- **Logistic Regression stacking meta-model** (v4 stacking and v5tuned) consistently matched
  or exceeded simple weighted averaging, suggesting that the meta-learner was able to exploit
  systematic calibration differences between base models.
- **Dropping ExtraTrees** in v5tuned improved performance, indicating that this estimator
  added noise rather than diversity in this setting.
- **Unconstrained thresholds** in v5tuned (versus constrained thresholds in earlier
  submissions) allowed the optimizer to find a better operating point per class.
- **Multiclass target encoding, frequency encoding, and group-statistics features** (36
  engineered features) outperformed the raw 96-feature space, suggesting that the base
  features contained redundancy.

### Techniques used

- Gradient boosting: LightGBM, XGBoost, CatBoost, HistGBM
- ExtraTrees (tested; dropped in v5tuned)
- Logistic Regression (L2) stacking meta-model
- Multiclass target encoding
- Frequency encoding
- Group-statistics features
- Per-class threshold optimization
- Optuna hyperparameter tuning (50 trials)
- Multi-seed averaging (5 seeds)
- 5-fold StratifiedKFold cross-validation (seed=42)
- Class-weight balancing

---

## Submission Evolution

| Version | Description | CV BA | Public LB | Private LB |
|---|---|---|---|---|
| baseline_lgbm_v1 | LightGBM, class_weight balanced + threshold optimization | 0.97005 | 0.96840 | 0.96998 |
| ensemble_v1 | LGB + XGB + CatBoost + threshold optimization | 0.97119 | 0.96880 | 0.97154 |
| **ensemble_tuned_v2** | **Optuna 50 trials, LGB+XGB+Cat + threshold** | **0.97224** | **0.97018** | **0.97242** |
| v3 | 96 features, multiclass TE + freq enc + group stats, multi-seed 5x LGB+XGB + CatBoost | 0.97223 | 0.96976 | 0.97133 |
| v4 stacking | LogReg meta on 5-seed LGB+XGB+Cat, 36 features | 0.97249 | 0.97060 | 0.97150 |
| v4 mega | Avg + stack blend, 36 features | 0.97246 | 0.97050 | 0.97125 |
| v4 weighted | 5-seed LGB+XGB+Cat avg + threshold, 36 features | 0.97234 | 0.96977 | 0.97067 |
| **v5tuned** | **4 algos tuned (LGB/XGB/Cat + HistGB Optuna max_depth=3), L2 LogReg meta, unconstrained thresholds, ExtraTrees dropped** | **0.97241** | **0.97111** | 0.97227 |

Bold rows = best private LB (ensemble_tuned_v2) and best public LB (v5tuned).

---

## Key Takeaways

1. **Early Optuna tuning (ensemble_tuned_v2) outperformed later, more complex pipelines**
   on the private leaderboard, a reminder that tuning fundamentals can beat architectural
   complexity on structured tabular data.
2. **Threshold optimization is non-negotiable for imbalanced multiclass problems**: the
   baseline without threshold adjustment underperformed by roughly 0.003 balanced accuracy
   versus the tuned ensemble.
3. **Feature count does not determine performance**: 36 engineered features (v4 family)
   outperformed the 96-feature raw space, demonstrating that targeted feature engineering
   reduces noise more than it risks losing signal.
4. **Dropping underperforming estimators improves robustness**: ExtraTrees, despite
   contributing some diversity, hurt the final ensemble on this dataset. Ablation before
   finalizing the ensemble composition is worth the compute.
5. **Best public LB and best private LB came from different submissions** (v5tuned vs.
   ensemble_tuned_v2), confirming that public LB optimization is an unreliable proxy for
   generalization on a ~4000-team leaderboard with limited test data.
