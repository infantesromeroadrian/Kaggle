# ADR-001 — Model choice and hyperparameter-tuning strategy (C5 POC)

- **Status:** Accepted
- **Date:** 2026-06-06
- **Deciders:** Adrián Infantes
- **Cycle:** C4 Design (minimal) → input to C5 POC
- **Supersedes:** —

## Context

C2 (Data) and C3 (Feature & Hypothesis) are closed. The frozen 5-fold CV
baseline is **balanced_accuracy = 0.96448 ± 0.00029** (LightGBM, `sample_weight`
balanced, features RAW). The ML Problem Statement target is **≥0.97**.

C3/BL-006 produced a *certified negative*: no engineered feature (SDSS colors
`u-g`/`g-r`/`r-i`/`i-z`, `redshift_log`) beats the baseline defensibly (best
paired Δ = +8.6e-05 < 2·SEM), and LightGBM's monotone-invariance to
single-variable transforms was confirmed empirically. The ~0.0055 gap to 0.97
is therefore **structural with first-order features**. Ablation also showed the
sky coordinates `alpha`/`delta` are strongly predictive (drop −0.009/−0.006) —
a synthetic-dataset footprint artifact, not leakage (C2 adversarial AUC 0.4997).

With feature engineering exhausted, the remaining lever to close the gap is
**hyperparameter tuning**. This ADR records the model and tuning decisions that
gate C5.

## Decision

1. **Model family: LightGBM (gradient-boosted trees).** Keep it. A 577k-row ×
   ~10-feature tabular problem with native categorical handling is squarely in
   GBT territory; LightGBM already reaches 0.9645. Deep learning is not
   justified at this scale and would add cost without expected lift.

2. **Tuning engine: Optuna.** TPESampler (seed=42) + MedianPruner, budget
   ~150 trials with a wall-clock timeout. Objective: **maximize
   balanced_accuracy CV** (the competition metric) over the frozen folds.

3. **Features: RAW.** BL-006 proved engineered features do not help; per
   simplicity-first, tune on the raw feature set.

4. **Evaluation protocol: the frozen folds.** Same `data/folds/cv_indices.npz`
   applied positionally (`.iloc`), reusing the `feature_eval.py` CV harness.
   This makes the BL-019 refactor (extract `src/cv_utils.py`) the natural first
   step, since the tuning harness becomes the third consumer of
   `compute_sample_weights` / `load_folds`.

5. **Anti-overfitting guardrail.** The headline risk is overfitting
   hyperparameters to a 5-fold CV. Any reported improvement must exceed the
   per-fold noise (Δ > 2·SEM of the paired difference), audited by
   the statistical-validation review. We do not chase Δ smaller than the fold std.

6. **GPU vs CPU: deferred to the ML engineering step + statistical-validation review.** Trade-off is
   GPU speed (helps 150 trials) vs CPU bit-for-bit determinism
   (`num_threads=1`, `deterministic=True`, `force_row_wise=True`, as in the
   C2/C3 baseline). The final reported model must be reproducible.

## Rationale

- Switching model family is premature: the baseline is strong and the data is
  small/tabular. Tuning is the cheapest, highest-EV lever.
- Optuna's TPE + pruning is the standard, reproducible way to search the
  LightGBM space within a fixed budget.
- Pinning the evaluation to the frozen folds keeps every C5 number comparable
  to the C2/C3 baseline — no moving denominator.

## Consequences

**Positive**
- Systematic, reproducible, MLflow-tracked search over the LightGBM space.
- Shared `cv_utils.py` removes the duplication debt (BL-019/DEBT-001).
- First Kaggle submission at the end of C5 validates the pipeline end-to-end and
  checks CV↔leaderboard fidelity.

**Negative / risks**
- Tuning may not close the gap. The 0.97 target may be unreachable (open blocker
  **B-004**); if Optuna does not beat 0.9645 defensibly, we renegotiate the
  target rather than overfit the leaderboard.
- Overfitting the 5-fold CV — mitigated by the statistical-validation guardrail above.

**Deferred (debt)**
- The C4 Excalidraw modeling-flow diagram (and the still-pending C1 context
  diagram it depends on) are deferred, consistent with the C1 decision to defer
  Excalidraw on a Playground competition. Tracked in the C3 Blockers note.
