# Kaggle Playground Series S6E3 — Customer Churn Prediction

> Binary classification: predict which telecom customers will churn, using synthetic tabular data
> derived from a real customer churn dataset.
>
> Competition: [Kaggle Playground Series Season 6, Episode 3](https://www.kaggle.com/competitions/playground-series-s6e3) (2026)

---

> This directory documents the approach and results of the competition entry. The original
> notebooks were developed on Kaggle and are not included here. No code or data files are
> present — this is a portfolio summary.

---

## Result

| Split | Score (ROC AUC) |
|---|---|
| Best public leaderboard | **0.91461** (submission v6) |
| Best private leaderboard | **0.91580** (submission v6) |

The evaluation metric is ROC AUC. OOF (out-of-fold) AUC values reported below are inferred
from submission descriptions; they reflect in-fold cross-validation performance prior to
generating the final test predictions.

---

## Approach

The core strategy across all submissions was a **gradient boosting ensemble** (LightGBM,
XGBoost, CatBoost, and in later iterations HistGBM) combined with iterative feature
engineering and Optuna-based hyperparameter search.

### What drove the gains

- **Pairwise target encoding** introduced in v6 yielded the most consistent lift across
  public and private leaderboard splits.
- **Polynomial logistic regression blend** as the meta-learner outperformed simple rank
  averaging when combined with GBDT base models.
- **Multi-seed averaging** (4-5 seeds per model family) reduced variance without adding
  new modeling complexity.
- **Feature engineering** — frequency encoding, n-gram target encoding, distribution
  features, and the `remaining_contract` derived feature — consistently improved
  cross-validation OOF AUC.
- **Rank averaging** across augmented and non-augmented pipelines improved diversity and
  smoothed overfitting to the augmented distribution.
- **Pseudo-labeling** (v9, ~27k pseudo-labeled rows, weight 0.5) hurt both public and
  private scores versus v7, demonstrating that the pseudo-label quality did not exceed
  the original signal.
- **Feature pruning** in v8 (86 of 108 features retained) did not outperform the
  full feature set on the private leaderboard despite a higher OOF AUC, suggesting mild
  OOF overfitting in the pruning selection.

### Techniques used

- Gradient boosting: LightGBM, XGBoost, CatBoost, HistGBM
- Pairwise target encoding
- Frequency encoding
- N-gram target encoding
- Distribution features
- Derived feature: `remaining_contract`
- Polynomial logistic regression (blend meta-model)
- Optuna hyperparameter tuning (up to 140 trials)
- Multi-seed averaging (up to 10 seeds)
- Rank averaging
- 5-fold and 20-fold cross-validation
- Data augmentation
- Pseudo-labeling

---

## Submission Evolution

| Version | Description | OOF AUC | Public LB | Private LB |
|---|---|---|---|---|
| tuned | Ensemble (LGBM + XGB + CatBoost), Optuna 140 trials | 0.91663 | 0.91386 | 0.91519 |
| v2 | Augmentation + v2 features + stacking blend | 0.91593 | 0.91399 | 0.91531 |
| v3a | Target encoding + frequency encoding + rank avg (no augmentation) | 0.91637 | 0.91391 | 0.91511 |
| v3 cross | Blend A+B (no-aug rank + aug rank), diversity ensemble | — | 0.91392 | 0.91512 |
| v4 multiseed | 4 seeds x 3 models x 5 folds = 60 models, mean OOF | 0.91585 | 0.91395 | 0.91522 |
| v5 | N-gram TE + distribution features + Ridge + digit features + 20-fold | 0.91616 | 0.91420 | 0.91546 |
| **v6** | **GBDT + PairwiseTE + PolyLR blend, remaining_contract feature** | **0.91646** | **0.91461** | **0.91580** |
| v6 gbdt-only | 91 features + remaining_contract + promote, GBDT without meta blend | 0.91611 | 0.91426 | 0.91548 |
| v7 | 5-seed GBDT (LGBM+XGB+CB) + PairTE + PolyLR + Optuna blend + augmentation | 0.91646 | 0.91445 | 0.91562 |
| v8 | Feature pruning (86/108), 10 seeds x 4 GBDT (LGBM+XGB+CB+HistGBM), GBDT-only | 0.91689 | 0.91418 | 0.91536 |
| v9 | v7 base + softmax T=500 + pseudo-labeling (27k, w=0.5) + Optuna blend | 0.91639 | 0.91257 | 0.91348 |

Bold row = best submission by private leaderboard score.

---

## Key Takeaways

1. **Pairwise target encoding was the single most impactful feature engineering step**,
   consistent with its private LB improvement from v5 to v6.
2. **Pseudo-labeling degraded performance** on this dataset; the synthetic data distribution
   made it difficult to generate reliable soft labels worth the regularization cost.
3. **Higher OOF AUC does not guarantee a higher private LB score**: v8 achieved the highest
   recorded OOF (0.91689) but ranked below v6 on the private leaderboard, indicating some
   overfitting in the feature-pruning selection process.
4. **Multi-seed averaging past 5 seeds yields diminishing returns** without a corresponding
   increase in model diversity (different architectures or feature sets).
5. **The augmented vs. non-augmented diversity blend** (v3 cross) outperformed either branch
   alone by a small margin, confirming that distribution diversity adds more signal than
   augmentation alone.
