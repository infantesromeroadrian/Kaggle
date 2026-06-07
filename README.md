# Kaggle

My Kaggle competition solutions — one subdirectory per challenge.

## Solutions

| Challenge (dir) | Task | Metric | Best Result | Notes |
|---|---|---|---|---|
| [playground-s6e6](./playground-s6e6) | Stellar classification (GALAXY / STAR / QSO) from synthetic SDSS photometry | `balanced_accuracy` | Public LB **0.96625**, rank **361 / 965 (top 37%)** | Includes code |
| [playground-s6e4](./playground-s6e4) | Irrigation-need prediction (multiclass) | `balanced_accuracy` | Public LB **0.97111** / Private LB **0.97242** | Documentation of results only |
| [playground-s6e3](./playground-s6e3) | Customer churn prediction (binary) | `ROC AUC` | Public LB **0.91461** / Private LB **0.91580** | Documentation of results only |
| [triagegeist](./triagegeist) | Emergency-triage acuity prediction (hackathon) | — | See subdir | Includes code |
| [nvidia-nemotron-reasoning](./nvidia-nemotron-reasoning) | LLM reasoning — SFT of Nemotron 3 Nano 30B with Chain-of-Thought data (LoRA) | — | Public LB **0.61** (Featured, ~4,041 teams) | Includes code (notebooks) |

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
