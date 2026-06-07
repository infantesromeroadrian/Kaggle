# Strategy — NVIDIA Nemotron Model Reasoning Challenge

A working note on where we are, what the competition actually rewards, what the new
training script (`train/nemotron_sft_v6.py`) fixes, and the two work phases that follow.

## 1. The competition reality

- **Metric is answer accuracy, not loss.** Each problem is graded by extracting the last
  non-empty `\boxed{...}` from the model's generation (at temperature 0) and comparing it
  to the reference answer (exact match, with a small relative tolerance for floats). A
  lower training/validation loss does **not** reliably mean a higher score — the two
  diverge on this task. Optimising loss is a proxy; optimising boxed-answer accuracy is
  the goal.
- **Our current standing vs the public baseline.** Our best LoRA adapter scores **0.61**
  on the public leaderboard. A strong public baseline sits around **0.85**. That ~0.24 gap
  is not a hyperparameter-tuning gap — it is a *training-signal* gap.
- **The lever is the deterministic chain-of-thought.** The tasks (bit manipulation,
  ciphers, Roman numerals, numeric equations, cryptarithms, gravity/unit-conversion
  physics fits, etc.) are all *algorithmically solvable*. The highest-scoring approach is
  to teach the model to reproduce a **deterministic step-by-step solution trace** that
  ends in the correct `\boxed{}` answer. When the chain-of-thought is a faithful replay of
  an exact algorithm, the model lands the exact answer string the grader checks at
  temperature 0. Better answers come from a better CoT corpus, not a fancier optimiser.
- **Prompt/answer contract must match the grader byte-for-byte.** The training prompt
  appends the suffix:
  `\nPlease put your final answer inside `\boxed{}`. For example: `\boxed{your answer}``
  and the answer is extracted as the *last* non-empty `\boxed{...}`. Training and
  evaluation must use the identical prompt template and the identical extraction regex, or
  the adapter underperforms regardless of how well it was trained.

## 2. What v6 fixes (correctness, not cleverness)

The previous training scripts trained — but several silent bugs were capping accuracy.
`train/nemotron_sft_v6.py` keeps the working setup (the same base model load, the same
LoRA-rank-32 adapter, the same adapter packaging that already loads correctly) and
corrects the following:

1. **Completion-only loss masking, with the stop-token fixed.** Earlier versions
   computed the loss over the *entire* sequence (prompt included) and masked padding by
   token value. Because the padding token and the end-of-sequence token are the *same id*
   on this tokenizer, masking by value also masked the one real end-of-sequence token —
   so the model was never taught where to stop, causing runaway generations and broken
   answer parsing. v6 builds labels **positionally**: the system+user prompt is masked,
   the assistant reasoning (`<think>` body) and the `\boxed{}` answer are supervised, and
   exactly one terminal stop token is kept supervised. Two correctness guards back this:
   (a) the boundary between the masked prompt and the supervised completion is **verified
   token-for-token on every row** — the prompt rendering must be an exact token prefix of
   the full conversation; rows where the chat template diverges are repaired to the true
   shared boundary or dropped, never silently mis-masked; and (b) because long sequences
   are right-truncated, every row is checked **after truncation** to confirm the
   supervised completion still ends in exactly one stop token and still contains the boxed
   answer — any row whose ending was cut off is dropped (and the count printed) rather than
   teaching the model to never stop. These invariants are asserted across the **whole
   corpus**, not a three-sample peek.

2. **A learning rate matched to the strongest known recipe.** Set to **1e-4**, which is
   the *winner-equivalent* step at our adapter scaling: the strongest known recipe used
   2e-4 with a 1:1 adapter scaling, and our adapter uses a 2:1 scaling that already doubles
   the effective update, so the matching literal rate is half of theirs — 1e-4. The
   schedule also follows that recipe: a pure linear decay to ~0 with no warmup. The next
   learning-rate sweep (1e-4 vs a 2×-conservative 5e-5) will be decided by the Phase C
   accuracy cross-validation below — **not** by chasing public-leaderboard swings, which
   are noisy and easy to overfit to.

3. **Gradient checkpointing that actually trains the adapter.** With memory-saving
   gradient checkpointing on a frozen base model, the adapter's gradients can silently
   vanish. v6 re-enables the gradient path and uses the non-reentrant checkpointing mode,
   then asserts after one step that the adapter parameters received non-zero gradients —
   so a misconfigured run fails immediately instead of wasting a full GPU session.

4. **Adapting the *right* layers.** This base model is a state-space (Mamba) + mixture-of-
   experts hybrid; it does **not** have the usual attention projection names. The old
   target pattern only ever touched a couple of projection types and missed both the real
   attention layers and the per-expert feed-forward layers — most of the model's capacity
   was never adapted. v6 enumerates the model's actual linear layers at runtime, prints
   them, and adapts attention + expert feed-forward + state-space projections while keeping
   the expert *router* frozen (adapting the router destabilises routing for no benefit).
   It asserts that coverage strictly increased versus the old pattern, with a safe
   fallback to the old pattern if discovery ever finds nothing.

5. **Dynamic padding and a longer sequence budget.** Dropped fixed-length padding to 2048;
   v6 tokenizes without padding and pads each batch to its own longest sequence (rounded
   for tensor-core efficiency), with padded positions excluded from the loss. The
   truncation budget is raised to **8192 tokens** to match the strongest known recipe and
   avoid cutting off long deterministic reasoning traces before their final answer; because
   padding is per-batch, short rows never pay for the higher ceiling. (It can drop back to
   4096 if a future configuration runs out of memory.) This is faster than fixed padding
   and removes the value-based padding hazard. (An earlier de-duplication / re-balancing
   idea was dropped — it was based on a mis-reading of the task categories.)

6. **Reproducible, fail-loud data loading.** Instead of "use the first file that matches a
   recursive search", v6 resolves an explicit dataset path, refuses an ambiguous mount,
   and asserts a minimum row count and the expected message schema before training. All
   random seeds (Python, NumPy, PyTorch, the training framework) are fixed.

7. **A marked seam for accuracy-based model selection.** v6 leaves a clearly labelled hook
   where the Phase C accuracy cross-validation will replace loss-based checkpoint
   selection. For now it does a single one-pass run and saves the final adapter (mirroring
   the strongest known recipe), with an explicit note **not** to wire loss-based selection
   back in as a proxy.

v6 is the corrected *training harness*. On its own it will likely move the score, but the
large gains come from the next two phases, which v6 is deliberately built to slot into.

## 3. Phase B — regenerate the chain-of-thought corpus from deterministic solvers

The single biggest lever is the **quality and faithfulness of the CoT corpus**. Rather
than hand-writing or sampling reasoning traces, the plan is to **generate them from exact,
deterministic solvers**, one per task family, and train the model to imitate those traces
step by step. The blueprint:

1. **Vendor the deterministic solvers as-is.** One solver per task family
   (bit manipulation, cipher, Roman numerals, numeric equations, cryptarithms, gravity,
   unit conversion). Each emits a multi-line natural-language reasoning trace that ends in
   a `\boxed{}` answer. These are battle-tested algorithms — copy them in, do not rewrite.
2. **Provide the inputs.** Build the per-problem detail files (question, answer, worked
   examples) and the train index from the competition training set.
3. **Run the solvers and self-verify every trace.** For each problem, generate the trace,
   re-extract its own `\boxed{}` answer, and compare it to the stored answer (strict for
   exact-match categories, relative tolerance for floats). Only traces that *verify* flow
   into training. The per-category solve rate it prints is a coverage diagnostic: any
   family far below full coverage is a solver gap to investigate, not a training defect.
4. **(Optional) Add formatting sub-skill examples.** A handful of small "apply this exact
   transform to every row" tasks teach the formatting primitives the solvers rely on
   (character spelling, bracket merge/split, leading-space strip, bit-column matching).
   Worth an ablation: measure their marginal lift before paying their token cost.
5. **Assemble the training file.** For each verified trace, build the completion as
   `reasoning + closing </think> + \boxed{answer} + stop token`, and build the prompt
   through the base model's chat template with the exact prompt suffix. Emit one row per
   example, either as a chat pair or pre-tokenized with completion-only labels (prompt
   masked, completion supervised) — the exact format v6 consumes.
6. **Pin the prompt/answer contract.** The prompt suffix and the `\boxed{}` extraction
   regex must byte-match between corpus generation, training, and evaluation. This
   alignment is what makes a temperature-0 generation reproduce the trace and land the
   exact answer string.
7. **Re-derive the special tokens from the tokenizer.** Do not assume any specific
   reasoning-open / reasoning-close / stop-token strings; read them from this model's chat
   template and round-trip one assembled example to confirm it renders correctly before
   generating the whole corpus.
8. **Train with the v6 harness.** Adapter rank 32 over attention + feed-forward (and the
   output head if memory allows), conservative learning rate with decay, one pass over the
   corpus, completion-only loss, and category-balanced batching so every batch mixes all
   task families.
9. **Gate the corpus before trusting it.** Independently re-check the long-arithmetic and
   bit-gate math in the traces, review the corpus-assembly code, and add a regression
   test: re-run the self-verification over a sample of rows and require 100% trace/answer
   consistency *before* any training run.

The acceptance test for Phase B is simple and binary: **a sampled set of generated traces
must be 100% self-consistent** (each trace's own boxed answer equals the stored answer)
before we spend GPU time training on them.

## 4. Phase C — accuracy cross-validation harness

Loss-based model selection is the wrong objective for this competition. Phase C builds the
evaluation that the v6 hook is waiting for:

- **A frozen, category-stratified held-out split** of problems with known answers, so the
  signal is balanced across task families and stable across runs.
- **Temperature-0 generation** using the identical prompt contract, then **the grader's
  exact `\boxed{}` extraction and comparison** to produce a true accuracy number.
- **Accuracy as the selection metric.** Wire this in as the checkpoint-selection signal
  (higher is better) so that every recipe decision — the 1e-4-vs-5e-5 learning-rate sweep,
  whether to train the output head, whether the formatting sub-skills help — is decided on
  *local accuracy cross-validation*, not on public-leaderboard noise. The public
  leaderboard is for final confirmation only.

With Phase B supplying a verified deterministic-CoT corpus and Phase C giving us an honest
local accuracy signal, the v6 harness is the stable, correct training core that turns both
into leaderboard movement. The order of work is: ship v6 (done) → build Phase B corpus →
stand up Phase C cross-validation → sweep recipe choices against Phase C accuracy.
