"""Nemotron-3-Nano-30B-A3B SFT (v6) -- corrected native HuggingFace-PEFT path.

Target runtime: Kaggle GPU kernel (NVIDIA RTX Pro 6000), offline (enable_internet=false),
Kaggle BYOD image. This file is COPY-PASTEABLE into a Kaggle notebook -- each top-level
`# === CELL N ===` block maps to one notebook cell. It must NOT be run on Adrian's local
8 GB GPU; a 30B model only fits on the Kaggle host.

What this file is
-----------------
v6 keeps the *working* native path that scored 0.61 public LB (v2):
  - kagglehub.model_download of metric/nemotron-3-nano-30b-a3b-bf16
  - AutoModelForCausalLM(trust_remote_code=True, bf16, device_map='auto')
  - the cutlass / mamba_ssm utility-path + ptxas-blackwell bootstrap (cell 0)
  - PEFT get_peft_model + save_pretrained + zip(adapter_config.json, adapter_model.safetensors)
  - LoRA rank=32 (competition max), alpha=64

...and fixes the *correctness* bugs that were silently capping accuracy. Each fix is
labelled `FIX N` inline and explained at the top of the relevant cell.

The competition metric is ANSWER ACCURACY (exact `\\boxed{}` match at temp=0), NOT loss.
Every fix below is in service of "the temp=0 deterministic CoT lands the exact boxed
string the grader checks". See STRATEGY.md for the full rationale and the Phase B
(regenerate the CoT corpus from deterministic solvers) / Phase C (accuracy CV harness)
roadmap that this script is built to slot into.

Summary of the seven corrections vs v2:
  FIX 1  Completion-only loss masking, built POSITIONALLY (pad==eos so we cannot mask by
         value). Prompt is -100; the assistant <think> CoT, the \\boxed{} answer, and
         exactly ONE terminal EOS are supervised. The prompt-prefix invariant is VERIFIED
         per row by token-id equality (with a longest-common-prefix repair + drop), and
         truncation that decapitates the completion drops the row. The masking invariants
         are asserted over the WHOLE corpus, not a 3-sample peek.
  FIX 2  LR -> 1e-4, the winner-equivalent: winner used 2e-4 at alpha/r=1; our alpha/r=2
         doubles the effective update, so the literal LR halves to 1e-4. Schedule adopts
         the winner's pure linear decay with no warmup. (5e-5 = 2x-conservative fallback.)
  FIX 3  Gradient-checkpointing-with-PEFT fix: enable_input_require_grads() +
         use_reentrant=False, asserted by a non-zero LoRA grad after one step.
  FIX 4  target_modules discovered at RUNTIME from named_modules() (this is a Mamba2/MoE
         hybrid -- do not assume q/k/v/o/gate exist), MoE router kept FROZEN, coverage
         asserted strictly greater than the old regex.
  FIX 5  Dynamic padding via a causal-LM collator (pad-to-batch-max, multiple of 8, pad
         labels -> -100). No more padding='max_length'. Dedup/rebalance idea dropped.
  FIX 6  Robust data loading: explicit filename + row-count assertion, full seeding.
  FIX 7  Marked hook for the Phase C accuracy-CV callback that will replace
         metric_for_best_model='eval_loss' (kept as a documented placeholder for now).
"""

# === CELL 0 ===========================================================================
# Utility-path bootstrap. PRESERVED VERBATIM from the v2 working notebook.
# The Kaggle BYOD Triton backend ships a ptxas-blackwell binary in a read-only location
# and Triton tries to exec it in place -> permission error. We copy it somewhere writable
# and monkey-patch subprocess.Popen so any ptxas-blackwell invocation is redirected to the
# writable copy. We also add the nvidia-utility-script site-dirs so `cutlass` and
# `mamba_ssm` (required by Nemotron's custom Mamba2 modelling code) import cleanly.
# WHY this stays untouched: it is environment plumbing, not training logic; it already
# works, and reproducing the 0.61 baseline requires the exact same import surface.
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import site
import subprocess
import shutil
import glob

PTXAS_SRC = (
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/"
    "triton/backends/nvidia/bin/ptxas-blackwell"
)
PTXAS_DST = "/kaggle/working/ptxas-blackwell"
if os.path.exists(PTXAS_SRC):
    shutil.copy2(PTXAS_SRC, PTXAS_DST)
    os.chmod(PTXAS_DST, 0o755)
    _OrigPopen = subprocess.Popen

    class PatchedPopen(_OrigPopen):
        """Redirect any ptxas-blackwell exec to the writable copy."""

        def __init__(self, args, *a, **kw):
            if isinstance(args, (list, tuple)):
                args = list(args)
                if args and "ptxas-blackwell" in str(args[0]):
                    args[0] = PTXAS_DST
            super().__init__(args, *a, **kw)

    subprocess.Popen = PatchedPopen
    print(f"ptxas fix applied: {PTXAS_DST}")

# Make cutlass / mamba_ssm importable from the utility script (handles both dir spellings).
for _base in [
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script",
]:
    if os.path.exists(_base):
        site.addsitedir(_base)
        for _pp in glob.glob(os.path.join(_base, "**/python_packages"), recursive=True):
            if _pp not in sys.path:
                site.addsitedir(_pp)
print("Setup complete")


# === CELL 1 ===========================================================================
# Imports, global seeding, and the prompt contract.
#
# FIX 6 (part 1): seed EVERYTHING before any RNG is touched (model init, data shuffle,
# dropout, collator). Reproducibility is non-negotiable for the Phase C accuracy CV --
# a noisy adapter makes the CV signal unreadable.
import json
import random
import statistics
from pathlib import Path
from collections import Counter

import numpy as np
import torch

# mamba_ssm must import successfully or the Nemotron Mamba2 layers will fall back to a
# slow/absent path. Import it explicitly so failures surface here, not mid-forward.
import mamba_ssm  # noqa: F401

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
import kagglehub

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)  # transformers: seeds python/numpy/torch + dataloader workers

OUTPUT_DIR = Path("/kaggle/working")

# --- Prompt contract -------------------------------------------------------------------
# This MUST byte-match the eval harness and the corpus builder (Phase B). The grader
# extracts the LAST non-empty \boxed{} from the temp=0 generation, so the training prompt
# has to instruct exactly the same boxed format. Do not "improve" this string.
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# Truncation budget. v2 used 2048 (and v4 tried 1024); the winner used 8192 with right
# truncation, and we MATCH that. Long deterministic CoTs (gravity/bit_manipulation
# long-arithmetic traces) can exceed 4096, and clipping the trace before its closing
# </think>+\boxed{}+EOS would either teach the model to never stop or silently drop the
# row (see the post-truncation completion-integrity guard in CELL 6). The RTX Pro 6000 has
# ~96 GB and dynamic padding (FIX 5) means short rows never pay for the long ceiling, so
# 8192 is affordable. Right-truncation is the HF tokenizer default (truncation_side=
# 'right'); we keep it. Drop back to 4096 if a future config OOMs at 8192.
MAX_SEQ_LENGTH = 8192

print("Seeded everything with", SEED)


# === CELL 2 ===========================================================================
# Load base model + tokenizer. PRESERVED from v2 (the path that loads correctly on Kaggle
# and whose PEFT adapter keys already pass the competition loader -- the 0.61 proves it).
MODEL_PATH = kagglehub.model_download(
    "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# pad_token == eos_token on this tokenizer. This is the exact reason FIX 1 must build
# labels POSITIONALLY: we cannot mask pads by value without also masking the real
# terminal EOS we want to supervise.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Vocab: {len(tokenizer)}")
print(f"pad_token_id={tokenizer.pad_token_id}  eos_token_id={tokenizer.eos_token_id}")
assert tokenizer.padding_side == "right", (
    "Right padding is required: the completion-only collator masks trailing pads by "
    "index and keeps the single terminal EOS. Left padding would invalidate that logic."
)


# === CELL 3 ===========================================================================
# FIX 4: discover target_modules at RUNTIME, do not hard-code q/k/v/o/gate.
#
# Nemotron-3-Nano-30B-A3B is a Mamba2 + MoE hybrid. The old v2 regex
#   r'.*\.(in_proj|out_proj|up_proj|down_proj)$'
# only ever matches Mamba in/out projections and a generic MLP up/down -- it misses the
# real attention linears and, critically, the per-expert MoE projections, so most of the
# A3B capacity was never adapted. Worse, a naive "adapt everything linear" would also wrap
# the MoE *router/gate*, which must stay frozen (LoRA on the router destabilises expert
# routing and there is no benefit for SFT imitation).
#
# Strategy: enumerate every leaf nn.Linear, print the unique leaf names so we can eyeball
# the real architecture once, then build target_modules to cover attention + expert MLP +
# Mamba projections while EXCLUDING router/gate-style modules. We assert the matched count
# and trainable% are strictly higher than the old regex, and fall back to the old regex if
# discovery somehow finds nothing (so the run can never be *worse* than v2).
import re
import torch.nn as nn

# Names that indicate the MoE router / load-balancing gate. These must NOT get LoRA.
# Nemotron MoE typically names the router 'gate' or 'router'; we exclude both, plus any
# explicit 'router'/'expert_gate'. NOTE: a per-expert MLP "gate_proj" is a real FFN
# projection and SHOULD be trained -- we only exclude the bare routing gate, hence the
# word-boundary patterns below rather than a substring match on "gate".
ROUTER_EXCLUDE = re.compile(r"(^|\.)(router|gate|expert_gate|routing)$")

# Collect the unique *leaf* names of every nn.Linear in the model.
leaf_linear_names: Counter = Counter()
leaf_linear_examples: dict[str, str] = {}
for full_name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        leaf = full_name.split(".")[-1]
        leaf_linear_names[leaf] += 1
        leaf_linear_examples.setdefault(leaf, full_name)

print("=== Leaf nn.Linear names discovered (name: count -> example full path) ===")
for leaf, count in leaf_linear_names.most_common():
    print(f"  {leaf:>20s}: {count:5d}  e.g. {leaf_linear_examples[leaf]}")

# Build the LoRA target set: every leaf-linear name that is NOT the router/gate and is NOT
# the lm_head (the head is handled separately via modules_to_save to replicate the
# winner's train_unembed=True, but only if VRAM allows -- see below).
candidate_leaves = {
    leaf
    for leaf in leaf_linear_names
    if not ROUTER_EXCLUDE.search(leaf) and leaf != "lm_head"
}

# Defensive fallback: if enumeration found nothing usable (e.g. modelling code changed),
# fall back to the exact v2 regex so v6 is never strictly worse than the 0.61 baseline.
OLD_REGEX = r".*\.(in_proj|out_proj|up_proj|down_proj)$"
if not candidate_leaves:
    print("WARNING: no leaf linears matched discovery; falling back to v2 regex.")
    target_modules: object = OLD_REGEX
else:
    # peft accepts a list of leaf names; it matches any module whose name endswith one.
    target_modules = sorted(candidate_leaves)
    print(f"\nSelected {len(target_modules)} target leaf names: {target_modules}")

# --- Coverage assertion vs the old regex (FIX 4 acceptance test) -----------------------
# Count how many modules the OLD regex would have matched vs how many our new selection
# matches, so we can PROVE coverage strictly increased.
old_regex_re = re.compile(OLD_REGEX)
old_matches = sum(
    1
    for n, m in model.named_modules()
    if isinstance(m, nn.Linear) and old_regex_re.match(n)
)
if isinstance(target_modules, list):
    new_matches = sum(
        1
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.split(".")[-1] in candidate_leaves
    )
    print(f"Module coverage -- old regex: {old_matches}, new selection: {new_matches}")
    assert new_matches > old_matches, (
        f"New target_modules ({new_matches}) did not strictly exceed the old regex "
        f"({old_matches}); investigate the discovered names before training."
    )


# === CELL 4 ===========================================================================
# Apply LoRA. Rank 32 (competition max), alpha 64 (2x scaling, matching v2 and the
# winner's "alpha=64 for 2x" guidance). dropout 0.0 to match the winner (Tinker default
# was no dropout); v2 used 0.05 -- for a 1-epoch imitation run we want the trace copied
# faithfully, so we drop dropout.
#
# modules_to_save / lm_head note: the winner trained the unembedding (train_unembed=True).
# Full-training lm_head on a 30B vocab head is a large memory hit. We expose it as a flag,
# default OFF for the first v6 run to stay within the working VRAM envelope that produced
# 0.61; flip it ON in a later sweep and gate on the Phase C accuracy CV. This is recorded
# as a known recipe deviation in STRATEGY.md.
TRAIN_LM_HEAD = False  # set True in a later sweep; gate the decision on Phase C accuracy.

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=target_modules,
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    modules_to_save=["lm_head"] if TRAIN_LM_HEAD else None,
)
model = get_peft_model(model, lora_config)

# FIX 3 (part 1): with gradient checkpointing + PEFT on a frozen base, the input
# embeddings produce no grad by default, so the checkpointed blocks have nothing to
# backprop into and LoRA grads silently stay None. enable_input_require_grads() registers
# a hook that makes the embedding output require grad, restoring the gradient path.
model.enable_input_require_grads()

model.print_trainable_parameters()


# === CELL 5 ===========================================================================
# FIX 6 (part 2): robust data loading -- explicit filename, no "first glob match wins".
#
# The training corpus is the CoT JSONL. Each row is expected to be
#   {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}],
#    "type": "<category>"}        # 'type' optional, used only for stratification/logging
# where the assistant content already ends the completion with the closing </think>, the
# \boxed{answer}, and the model's stop token. Phase B will regenerate this corpus from the
# deterministic solvers; until then v6 consumes whatever CoT JSONL is mounted, but it
# REFUSES to silently train on the wrong file.
#
# We resolve a single explicit path and assert a sane row count instead of grabbing the
# first recursive glob hit (which previously could pick up a stale/partial file).
COT_DATASET_DIR = "/kaggle/input/nemotron-cot-training-data"
COT_FILENAME = "cot_train.jsonl"
EXPECTED_MIN_ROWS = 100  # tripwire: a truncated/empty mount must fail loudly, not train.

explicit_path = Path(COT_DATASET_DIR) / COT_FILENAME
if explicit_path.exists():
    COT_PATH = explicit_path
else:
    # Controlled fallback: search ONLY for the exact filename, then sort for determinism
    # and require a unique match. Print every candidate so an ambiguous mount is visible.
    matches = sorted(
        Path(p) for p in glob.glob(f"/kaggle/input/**/{COT_FILENAME}", recursive=True)
    )
    if not matches:
        print("Available /kaggle/input contents:")
        for d in sorted(Path("/kaggle/input").iterdir()):
            print(f"  {d.name}/", [f.name for f in list(d.iterdir())[:5]])
        raise FileNotFoundError(
            f"{COT_FILENAME} not found under {COT_DATASET_DIR} or /kaggle/input/**"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous data mount -- multiple {COT_FILENAME} found: {matches}. "
            "Pin COT_DATASET_DIR to the intended dataset to avoid training on the wrong file."
        )
    COT_PATH = matches[0]

examples = []
with open(COT_PATH) as f:
    for line in f:
        if line.strip():
            examples.append(json.loads(line))

# Row-count assertion (FIX 6): fail loudly on a truncated/empty file.
assert len(examples) >= EXPECTED_MIN_ROWS, (
    f"Loaded only {len(examples)} rows from {COT_PATH}; expected >= {EXPECTED_MIN_ROWS}. "
    "Refusing to train on a truncated corpus."
)
# Schema assertion: EVERY row (not just a 50-row sample) must carry a well-formed SFT pair
# -- 'messages' present, a user turn in the context, the last turn an assistant turn, and
# non-empty content on every message. A single malformed row deep in the file would
# otherwise crash mid-tokenisation (or, worse, train on garbage); we want it to fail here.
for i, ex in enumerate(examples):
    assert "messages" in ex and isinstance(ex["messages"], list) and ex["messages"], (
        f"Row {i} has no non-empty 'messages' list."
    )
    msgs = ex["messages"]
    assert msgs[-1]["role"] == "assistant", (
        f"Row {i} last turn is '{msgs[-1]['role']}', expected 'assistant': "
        f"roles={[m['role'] for m in msgs]}"
    )
    assert any(m["role"] == "user" for m in msgs[:-1]), (
        f"Row {i} has no user turn before the assistant: "
        f"roles={[m['role'] for m in msgs]}"
    )
    assert all(str(m.get("content", "")).strip() for m in msgs), (
        f"Row {i} has an empty-content message: "
        f"roles={[m['role'] for m in msgs]}"
    )

print(f"Loaded {len(examples)} rows from {COT_PATH}")
print("Category mix:", Counter(ex.get("type", "?") for ex in examples).most_common())


# === CELL 6 ===========================================================================
# FIX 1 (v6.1): COMPLETION-ONLY LOSS MASKING via PROMPT + COMPLETION CONCATENATION.
#
# The earlier v6 derived the masked prompt length L by asserting that
# apply_chat_template(messages[:-1], add_generation_prompt=True) is a TOKEN-EXACT PREFIX of
# apply_chat_template(messages, add_generation_prompt=False). That invariant is FALSE for
# this Nemotron template and silently dropped ALL 6046 rows (verified on the Kaggle
# tokenizer). Nemotron uses a REASONING template: at generation time it opens the assistant
# turn with "<|im_start|>assistant\n<think>\n", but when it renders a COMPLETED assistant
# message whose content does not itself start with "<think>", it splices the content
# directly after "assistant\n" WITHOUT the "<think>\n" opener. So the prompt rendering and
# the full-conversation rendering DIVERGE right after the assistant header, the
# longest-common-prefix collapses, and every row falls out as "no-supervision".
# (Separately, apply_chat_template(tokenize=True) on this stack returned a 2-element
# Encoding list, not a token-id list -- another reason to avoid that path entirely.)
#
# The robust construction -- which matches the winner's 2-segment masking and, crucially,
# matches INFERENCE exactly -- is to build the training sequence by CONCATENATION:
#   prompt_ids     = encode( apply_chat_template(messages[:-1], add_generation_prompt=True,
#                            tokenize=False) )             # ends "...assistant\n<think>\n"
#   completion_ids = encode( assistant_content ) + [EOS]   # CoT</think>\boxed{}<|im_end|>
#   input_ids = prompt_ids + completion_ids
#   labels    = [-100]*len(prompt_ids) + completion_ids
# The prompt-prefix invariant is now TRUE BY CONSTRUCTION (input_ids literally starts with
# prompt_ids), so there is no LCP guesswork. At eval the harness feeds
# apply_chat_template(..., add_generation_prompt=True) to vLLM, which renders the SAME
# "<think>\n" opener and then generates the completion -- so what we supervise is exactly
# what the model must produce. Verified on Kaggle (nemotron-mask-verify): kept 6046/6046,
# 0 dropped, prompt tail "...assistant\n<think>\n", completion "The cipher is ...
# \boxed{...}<|im_end|>", last id == EOS.
#
# pad_token_id == eos_token_id (CELL 2 set it). Masking is positional: the ONE terminal EOS
# inside the completion is supervised; trailing PADs appended later by the collator are
# masked there, by index. MAX_SEQ_LENGTH truncation is right-side; if it ever cut the
# completion tail we DROP the row (BLOCKER 2) rather than teach a "never stop" target.
# Empirically no current row exceeds ~766 tokens, so truncation never fires -- the guard
# stays for future long-CoT categories (e.g. bit_manipulation).

from torch.utils.data import Dataset

EOS_ID = tokenizer.eos_token_id

# Drop counters, populated during dataset construction and printed once.
_DROP_TRUNCATED = 0       # rows dropped: right-truncation decapitated the completion.
_DROP_NO_SUPERVISION = 0  # rows dropped: prompt alone filled the budget (no target left).


def encode_example(ex) -> dict | None:
    """Tokenize one SFT row into {input_ids, attention_mask, labels, prompt_len} with
    completion-only labels, or None if the row must be dropped. Built by concatenation, so
    labels[:prompt_len] == -100 is correct BY CONSTRUCTION (no prefix guesswork). No padding
    here -- the collator pads dynamically (FIX 5)."""
    global _DROP_TRUNCATED, _DROP_NO_SUPERVISION
    messages = ex["messages"]
    assistant_content = messages[-1]["content"]

    # Prompt context, rendered to TEXT then tokenized. add_generation_prompt=True appends the
    # assistant header + the "<think>\n" opener (the reasoning-template generation cue). We
    # render to text and encode it -- NOT apply_chat_template(tokenize=True), which is broken
    # on this stack (returns Encoding objects, not ids).
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Completion = the assistant CoT (which already carries its own closing </think> and the
    # \boxed{answer}) followed by the single terminal EOS the model must emit to stop.
    completion_ids = tokenizer.encode(assistant_content, add_special_tokens=False) + [EOS_ID]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + list(completion_ids)

    # BLOCKER 2: right-truncate, THEN verify the completion survived. A cut that drops the
    # closing </think>+\boxed{}+EOS while keeping earlier CoT tokens would train a
    # "never-stop" target -> drop the row instead.
    if len(input_ids) > MAX_SEQ_LENGTH:
        input_ids = input_ids[:MAX_SEQ_LENGTH]
        labels = labels[:MAX_SEQ_LENGTH]
        supervised_ids = [t for t in labels if t != -100]
        ends_in_single_eos = bool(supervised_ids) and (
            supervised_ids[-1] == EOS_ID
            and (len(supervised_ids) == 1 or supervised_ids[-2] != EOS_ID)
        )
        has_boxed = bool(supervised_ids) and "\\boxed{" in tokenizer.decode(
            supervised_ids, skip_special_tokens=False
        )
        if not (ends_in_single_eos and has_boxed):
            _DROP_TRUNCATED += 1
            return None

    if len(prompt_ids) >= len(input_ids):
        # Prompt alone fills the budget: nothing supervised survives. Drop.
        _DROP_NO_SUPERVISION += 1
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_len": len(prompt_ids),  # consumed by CELL 7 assertions; dropped by collator.
    }


class SFTDataset(Dataset):
    """Pre-tokenized, completion-only-masked, UNPADDED examples. Rows whose completion was
    decapitated by truncation (BLOCKER 2) or that left no supervised target are dropped
    during construction and reported in the printed counts."""

    def __init__(self, rows):
        self.items = []
        for ex in rows:
            enc = encode_example(ex)
            if enc is None:  # counters already updated inside encode_example.
                continue
            self.items.append(enc)
        print(
            f"SFTDataset: kept {len(self.items)} / {len(rows)}. "
            f"Dropped: {_DROP_TRUNCATED} truncated-completion, "
            f"{_DROP_NO_SUPERVISION} no-supervision."
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# Shuffle once for good category mixing. NOTE: the winner used per-batch category
# stratification (every batch an even mix of the categories). A seeded global shuffle is a
# cheap approximation; a true stratified sampler is a Phase-C-adjacent improvement (see
# STRATEGY.md). We keep the shuffle deterministic via SEED.
random.Random(SEED).shuffle(examples)
train_ds = SFTDataset(examples)

# FAIL LOUD on a fully-empty dataset. The prior v6 silently produced 0 rows (template/mask
# mismatch) and only crashed much later in a stats call -- an assertion here names the real
# cause instead of leaving a cryptic StatisticsError.
assert len(train_ds) > 0, (
    "SFTDataset kept 0 rows -- every row was dropped. This indicates a prompt/template or "
    "EOS mismatch in encode_example (CELL 6), NOT a hardware problem. Inspect the drop "
    "counts printed above before re-running."
)
print(f"Train examples: {len(train_ds)}")


# === CELL 7 ===========================================================================
# FIX 1 ACCEPTANCE TEST -- run over the WHOLE corpus, not a 3-sample peek. The ids are
# already in memory (train_ds is fully materialised), so a full pass is cheap and catches a
# bad row anywhere in the file, not just at indices 0-2. For every kept row we assert that
#   (a) every prompt token is -100, and supervision starts at or after L (no prompt leak),
#   (b) the supervised span contains the closing </think> AND a \boxed{...},
#   (c) exactly ONE EOS is supervised and it is the LAST supervised token (clean stop).
#
# These are REAL invariants of the masking logic (CELL 6 already drops rows that cannot
# satisfy (b)/(c) under truncation), so this pass is a guard against regressions in the
# encode logic, not a tautology over a value we just set.

EOS_ID = tokenizer.eos_token_id


def _decode(ids):
    return tokenizer.decode(ids, skip_special_tokens=False)


def assert_masking_correct(item, idx):
    input_ids = item["input_ids"]
    labels = item["labels"]
    L = item["prompt_len"]

    # (a) prompt is fully masked.
    assert all(t == -100 for t in labels[:L]), f"[row {idx}] prompt not fully masked."
    # The completion must contain at least one supervised token.
    supervised_positions = [i for i, t in enumerate(labels) if t != -100]
    assert supervised_positions, f"[row {idx}] no supervised tokens."
    assert supervised_positions[0] >= L, (
        f"[row {idx}] supervision leaks into the prompt span "
        f"(first supervised pos {supervised_positions[0]} < L={L})."
    )

    # (b) the supervised span decodes to text that contains the closing </think> and a
    # \boxed{...}. (The opening <think> is in the masked prompt; the CoT body + closing
    # tag + boxed answer are all in the supervised completion.)
    supervised_ids = [input_ids[i] for i in supervised_positions]
    supervised_text = _decode(supervised_ids)
    assert "</think>" in supervised_text, (
        f"[row {idx}] closing </think> not in supervised span -- CoT body not learned."
    )
    assert "\\boxed{" in supervised_text, (
        f"[row {idx}] \\boxed{{}} answer not in supervised span -- answer not learned."
    )

    # (c) exactly ONE terminal EOS is supervised, and it is the LAST supervised token.
    supervised_eos_positions = [
        i for i in supervised_positions if input_ids[i] == EOS_ID
    ]
    assert len(supervised_eos_positions) == 1, (
        f"[row {idx}] expected exactly 1 supervised EOS, got "
        f"{len(supervised_eos_positions)} -- model would not learn a clean stop."
    )
    assert supervised_eos_positions[0] == supervised_positions[-1], (
        f"[row {idx}] the supervised EOS is not the terminal token."
    )


for _idx in range(len(train_ds)):
    assert_masking_correct(train_ds[_idx], _idx)
print(f"FIX 1 masking assertions passed on ALL {len(train_ds)} kept rows "
      "(prompt -100, CoT+boxed supervised, exactly one terminal EOS supervised).")

# Token-length stats on the supervised corpus (sanity check vs MAX_SEQ_LENGTH).
_lens = [len(train_ds[i]["input_ids"]) for i in range(min(500, len(train_ds)))]
print(
    f"Token lengths (first {len(_lens)}): mean={statistics.mean(_lens):.0f}, "
    f"max={max(_lens)}, p95={sorted(_lens)[int(len(_lens) * 0.95)]}"
)


# === CELL 8 ===========================================================================
# FIX 5: dynamic-padding collator for causal LM.
#
# v2 used padding='max_length' (every sequence padded to 2048) which wastes compute and,
# combined with the value-based mask, mishandled EOS. We replace it with a tiny custom
# collator that:
#   - pads input_ids/attention_mask to the batch max, rounded up to a multiple of 8
#     (tensor-core friendly),
#   - sets label = -100 on every pad position (by index, since these pads are appended
#     AFTER the real terminal EOS, the single supervised EOS is untouched),
#   - drops the bookkeeping 'prompt_len' key.
# This is the explicit equivalent of DataCollatorForLanguageModeling(mlm=False) but with
# our own labels preserved (that collator would overwrite labels from input_ids).

PAD_TO_MULTIPLE_OF = 8


def collate_causal_lm(features):
    pad_id = tokenizer.pad_token_id
    max_len = max(len(f["input_ids"]) for f in features)
    if PAD_TO_MULTIPLE_OF:
        max_len = (
            (max_len + PAD_TO_MULTIPLE_OF - 1) // PAD_TO_MULTIPLE_OF
        ) * PAD_TO_MULTIPLE_OF

    input_ids, attention_mask, labels = [], [], []
    for f in features:
        n_pad = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_id] * n_pad)
        attention_mask.append(f["attention_mask"] + [0] * n_pad)
        # Pad labels with -100 so loss ignores appended pads (FIX 1 preserved through padding).
        labels.append(f["labels"] + [-100] * n_pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# === CELL 9 ===========================================================================
# Training configuration.
#
# FIX 2: LR -> 1e-4, the WINNER-EQUIVALENT at our scaling. The winner used 2e-4 with
# alpha:rank = 1:1 (effective LoRA scaling alpha/r = 1). We run alpha:rank = 2:1
# (alpha/r = 2), which DOUBLES the effective LoRA update at the same literal LR; to land
# the SAME effective step as the winner we therefore HALVE the literal LR: 2e-4 / 2 = 1e-4.
# So:
#   * 1e-4  = winner-equivalent (the default we ship).
#   * 5e-5  = 2x-conservative (a deliberate half-step fallback).
# NEXT SWEEP NOTE: the {1e-4, 5e-5} comparison is gated on the Phase C boxed-answer
# accuracy CV (temp=0), NOT on public-leaderboard swings (which are noisy).
#
# Schedule: adopt the winner's setup -- pure LINEAR decay to ~0 with NO warmup
# (lr_scheduler_type='linear', warmup_ratio=0.0). Earlier versions used cosine +
# warmup_ratio=0.03; the winner is ground truth here, and on a single-epoch LoRA imitation
# pass a warmup buys nothing.
#
# num_train_epochs=1 (matches v5 and the winner): a single imitation pass. More epochs
# overfit the CoT template and erode generalisation at this LR.
#
# Effective batch: per_device 1 x grad_accum 16 = 16. The winner used 64; on a 30B-A3B
# model even on the Kaggle RTX Pro 6000 we keep per-device at 1 and use accumulation. Bump
# grad_accum toward 64 if memory headroom allows -- but accuracy, not throughput, is the
# objective, so this is a secondary knob.
LEARNING_RATE = 1e-4  # FIX 2: winner-equivalent at alpha/r=2 + rsLoRA (2e-4 / 2). 5e-5 = 2x-conservative.
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 16

args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "ckpt"),
    num_train_epochs=1,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,  # FIX 2: 1e-4 (winner-equivalent)
    lr_scheduler_type="linear",  # winner: pure linear decay to ~0
    warmup_ratio=0.0,  # winner: no warmup
    weight_decay=0.0,  # match the winner (Adam weight_decay=0.0); v2's 0.01 was arbitrary.
    bf16=True,
    logging_steps=25,
    # ---- Checkpoint selection (FIX 7 placeholder) -------------------------------------
    # The competition metric is boxed-answer ACCURACY at temp=0, which eval_loss only
    # loosely tracks. The real selector is the Phase C accuracy-CV callback (see the
    # TODO hook in CELL 11). For now we run a single 1-epoch pass and save the final
    # adapter (mirrors the winner: "last step of a 1-epoch run"), so no eval/selection is
    # active. eval_strategy='no' is the honest placeholder until the accuracy CV lands.
    eval_strategy="no",
    save_strategy="no",
    gradient_checkpointing=True,
    # FIX 3 (part 2): non-reentrant checkpointing is required for LoRA grads to flow
    # correctly under PEFT; the reentrant path drops grads for the injected adapters.
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    # Adam betas/eps match the winner (0.9, 0.95, 1e-8). max_grad_norm=1.0 is a mild safety
    # net; the winner effectively disabled clipping, but 1.0 never hurts a LoRA SFT run.
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    report_to="none",
    seed=SEED,
    data_seed=SEED,
    dataloader_pin_memory=True,
)


# === CELL 10 ==========================================================================
# Build the Trainer and run the FIX 3 gradient assertion BEFORE the full train.
#
# We do one manual forward/backward on a single batch and assert that at least one LoRA
# parameter received a non-zero gradient. If enable_input_require_grads() or
# use_reentrant=False were missing, this catches it immediately instead of burning a full
# Kaggle GPU session on a run that silently trained nothing.

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collate_causal_lm,
)

# --- FIX 3 acceptance test: non-zero LoRA grad after one step -------------------------
model.train()
_probe_batch = collate_causal_lm([train_ds[i] for i in range(min(PER_DEVICE_BATCH, len(train_ds)))])
_probe_batch = {k: v.to(model.device) for k, v in _probe_batch.items()}
model.zero_grad(set_to_none=True)
_out = model(**_probe_batch)
_out.loss.backward()

_lora_grad_norms = [
    p.grad.detach().float().norm().item()
    for n, p in model.named_parameters()
    if p.requires_grad and "lora_" in n and p.grad is not None
]
assert _lora_grad_norms, "No LoRA parameter received a gradient -- check FIX 3 wiring."
assert max(_lora_grad_norms) > 0.0, (
    "All LoRA gradients are exactly zero -- gradient checkpointing is dropping grads. "
    "Verify enable_input_require_grads() and gradient_checkpointing_kwargs use_reentrant=False."
)
print(
    f"FIX 3 grad check passed: {len(_lora_grad_norms)} LoRA params have grads, "
    f"max grad norm={max(_lora_grad_norms):.3e}"
)
model.zero_grad(set_to_none=True)  # clear the probe grads before real training.


# === CELL 11 ==========================================================================
# TODO / HOOK -- PHASE C ACCURACY-CV CALLBACK (replaces eval_loss selection).
# -------------------------------------------------------------------------------------
# This is the marked seam where the Phase C harness plugs in. When ready, this callback
# will, at the end of each epoch (or every N steps):
#   1. Take a frozen, stratified held-out split of problems (with known answers).
#   2. Generate at temperature=0 using the SAME prompt contract (PROMPT_SUFFIX) so train
#      and eval agree byte-for-byte.
#   3. Extract the LAST non-empty \boxed{...} with the grader's exact regex
#      r"\\boxed\{([^}]*)(?:\}|$)" and score with compare_answer (binary strict / float
#      rel_tol=1e-2 / else case-insensitive) -- the competition's accuracy metric.
#   4. Set metric_for_best_model='eval_accuracy', greater_is_better=True, and drive
#      load_best_model_at_end so adapter selection optimises ACCURACY, not loss.
#
# Until that lands, selection is "final adapter of a 1-epoch run" (eval_loss is NOT used;
# eval_strategy='no'). Do NOT wire metric_for_best_model='eval_loss' back in as a proxy --
# loss and boxed-accuracy diverge on this task, which is the whole reason for Phase C.
#
# from transformers import TrainerCallback
# class BoxedAccuracyCallback(TrainerCallback):
#     def on_epoch_end(self, args, state, control, model=None, **kw):
#         acc = evaluate_boxed_accuracy(model, tokenizer, held_out_problems)  # temp=0
#         state.log_history.append({"eval_accuracy": acc, "step": state.global_step})
#         # set control.should_save based on best-so-far accuracy
# trainer.add_callback(BoxedAccuracyCallback())
# -------------------------------------------------------------------------------------

print(f"Starting training: {len(train_ds)} examples, 1 epoch, "
      f"effective batch {PER_DEVICE_BATCH * GRAD_ACCUM}, LR {LEARNING_RATE}")
trainer.train()
print("Training complete.")


# === CELL 12 ==========================================================================
# Save the adapter and build submission.zip. PRESERVED from v2: native PEFT save_pretrained
# emits correctly-named adapter keys that the competition loader accepts (the 0.61 proves
# it), then we zip exactly the two files the submission expects. Do NOT add any SVD /
# key-rename step here -- that is the precise risk the winner analysis warns against.
model.save_pretrained(str(OUTPUT_DIR))

adapter_config = OUTPUT_DIR / "adapter_config.json"
adapter_model = OUTPUT_DIR / "adapter_model.safetensors"
assert adapter_config.exists(), f"Missing {adapter_config}"
assert adapter_model.exists(), f"Missing {adapter_model}"
print(f"adapter_config.json: {adapter_config.stat().st_size / 1024:.1f} KB")
print(f"adapter_model.safetensors: {adapter_model.stat().st_size / 1024 / 1024:.1f} MB")

os.chdir(str(OUTPUT_DIR))
# Idempotent zip: `zip` APPENDS to an existing archive, so re-running this cell (routine in
# a notebook) would otherwise leave stale members or a mismatched entry. Delete any prior
# submission.zip first so the rebuild is from scratch every time.
SUBMISSION_ZIP = OUTPUT_DIR / "submission.zip"
if SUBMISSION_ZIP.exists():
    SUBMISSION_ZIP.unlink()
subprocess.run(
    ["zip", "submission.zip", "adapter_config.json", "adapter_model.safetensors"],
    check=True,
)
print(f"submission.zip: {os.path.getsize('submission.zip') / 1024 / 1024:.1f} MB")
print("Done.")
