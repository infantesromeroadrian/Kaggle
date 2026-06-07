# Kaggle

My Kaggle competition solutions — one subdirectory per challenge.

## Solutions

| Challenge | Task | Metric | Result |
|---|---|---|---|
| [playground-s6e6](./playground-s6e6) | Stellar classification (GALAXY / STAR / QSO) from synthetic SDSS photometry | `balanced_accuracy` | Public LB **0.96625**, rank **361 / 965 teams (top ~37%)**, 2 submissions, snapshot 2026-06-07 |

## Methodology

Each solution follows a structured ML pipeline: frozen StratifiedKFold cross-validation,
leakage audits with adversarial validation, feature ablation with paired statistical tests,
Optuna hyperparameter tuning, nested-CV weighted ensembling, and statistical-significance
review gates at each phase boundary.

Key practices applied across all solutions:

- **Leakage-first discipline**: adversarial validation (train vs test AUC) and mutual
  information audits before any modeling begins.
- **Frozen folds**: cross-validation folds are frozen once and reused positionally by every
  subsequent script, so all CV numbers are directly comparable.
- **Paired statistics**: improvements are only reported if the paired fold delta exceeds 2 SEM
  (a conservative ~2-sigma bar on 5 folds). Negative results are reported honestly.
- **Nested CV for ensemble weighting**: blend weights are selected on inner folds that never
  include the held-out scoring fold, preventing CV overfitting of the weight selection.
- **Held-out validation**: every CV result is cross-checked against the Kaggle public
  leaderboard. CV is considered trustworthy only when LB and CV agree within 0.005.

## Author

by [Adrián Infantes](https://www.kaggle.com/adrininfantesromero)
