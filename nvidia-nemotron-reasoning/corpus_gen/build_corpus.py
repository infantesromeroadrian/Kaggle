"""Assemble the SFT training corpus for the NVIDIA Nemotron reasoning challenge.

Clean-room provenance
---------------------
Every rule used here was derived from the competition train.csv by the four
deterministic reasoners under ``reasoners/`` (numeral, unit_conversion, gravity,
cipher). The overall APPROACH -- deterministic per-category chain-of-thought,
supervised fine-tuning, temperature=0 decoding -- is inspired by the PUBLIC
competition writeup (Kaggle discussion 689915). ALL code and data in this
repository are original / clean-room: nothing was read or copied from any
third-party (unlicensed) solution. See README.md.

What this script does
---------------------
1. Loads /tmp/nemo-comp/train.csv (id, prompt, answer).
2. Classifies every row by a prompt-substring discriminator (one per category).
   The discriminators were chosen from the train.csv prompt wording itself and
   match the per-reasoner totals exactly (numeral 1576, unit_conversion 1594,
   gravity 1597, cipher 1576). Rows of not-yet-implemented categories (no
   reasoner) are skipped.
3. For each of the 4 implemented categories, calls the reasoner's
   ``generate_cot(prompt, answer)`` to produce a chain-of-thought. The reasoner
   already self-checks against ground truth and returns None on a wrong/unparseable
   problem -- those are skipped at the source.
4. Assembles each kept row into the EXACT schema ``nemotron_sft_v6.py`` consumes
   (see "Schema contract" below), then runs a FINAL self-verification gate: the
   LAST ``\\boxed{...}`` is re-extracted from the assembled assistant completion
   with the grader's regex and must equal ground truth. Any row that fails (wrong
   box, or no box) is dropped. This gate validates the ASSEMBLED line, not merely
   the reasoner's internal check.
5. Writes cot_corpus_v1.jsonl, one JSON object per line.

Schema contract (must byte-match nemotron_sft_v6.py)
----------------------------------------------------
Each line::

    {
      "type": "<category>",
      "messages": [
        {"role": "user",      "content": <competition prompt> + PROMPT_SUFFIX},
        {"role": "assistant", "content": <reasoning> + "\\n</think>\\n\\boxed{" + answer + "}"}
      ]
    }

* PROMPT_SUFFIX is copied verbatim from nemotron_sft_v6.py (CELL 1) so the
  training prompt byte-matches the eval harness.
* v6 supervises the assistant completion INCLUDING the closing </think>, the
  boxed answer, and the stop token. The chat template's add_generation_prompt
  emits the OPENING "<think>\\n" into the (masked) prompt span, so the assistant
  content must NOT re-open <think>; it starts directly with the reasoning body
  and ends with the closing "</think>" + boxed answer.
* The tokenizer appends the stop token, so the assistant content does NOT
  hardcode <|im_end|> / EOS (v6 CELL 1/6 note).

Determinism: every reasoner is a pure function of the prompt; this builder adds
no randomness. Re-running on the same train.csv yields a byte-identical JSONL.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

TRAIN_CSV = Path("/tmp/nemo-comp/train.csv")
HERE = Path(__file__).resolve().parent
REASONERS_DIR = HERE / "reasoners"
OUTPUT_JSONL = HERE / "cot_corpus_v1.jsonl"

# --------------------------------------------------------------------------- #
# Prompt contract -- copied VERBATIM from nemotron_sft_v6.py CELL 1.
# Do not "improve" this string: train and eval must byte-match, and the boxed
# format the grader extracts is defined here.
# --------------------------------------------------------------------------- #

PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# --------------------------------------------------------------------------- #
# Category discriminators -- substrings taken from the train.csv prompt wording.
# Order does not matter; each row matches at most one (verified: no overlaps).
# Maps category name -> (discriminator substring, reasoner module filename).
# --------------------------------------------------------------------------- #

CATEGORIES: dict[str, tuple[str, str]] = {
    "numeral": (
        "numbers are secretly converted into a different numeral system",
        "numeral.py",
    ),
    "unit_conversion": (
        "secret unit conversion is applied to measurements",
        "unit_conversion.py",
    ),
    "gravity": (
        "gravitational constant",
        "gravity.py",
    ),
    "cipher": (
        "encryption rules are used on text",
        "cipher.py",
    ),
}

# The grader extracts the LAST non-empty \boxed{...}; this mirrors that regex.
# DOTALL so a multi-line / multi-word boxed answer (cipher decryptions span
# spaces) is captured whole. Non-greedy up to the matching closing brace.
_BOXED_RE = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)


def _load_reasoner(filename: str):
    """Import a reasoner module by filename from the reasoners/ directory.

    Returns the module object (exposing generate_cot). Raises on a missing file
    so a typo fails loudly rather than silently dropping a whole category.
    """
    path = REASONERS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Reasoner not found: {path}")
    # Use the stem as a unique module name to avoid sys.modules collisions.
    spec = importlib.util.spec_from_file_location(f"reasoner_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "generate_cot"):
        raise AttributeError(f"{path} does not expose generate_cot(prompt, answer)")
    return module


def classify(prompt: str) -> str | None:
    """Return the category for a prompt, or None if no implemented category matches.

    A None result means the row belongs to a not-yet-implemented category (no
    reasoner) and is skipped -- it is NOT an error.
    """
    matched = [
        name for name, (disc, _) in CATEGORIES.items() if disc in prompt
    ]
    if len(matched) == 1:
        return matched[0]
    # Zero matches -> unimplemented category (skip). More than one would mean the
    # discriminators are ambiguous; we treat that as "do not trust" and skip too,
    # but it does not happen on this train.csv (verified: zero multi-matches).
    return None


def extract_last_boxed(text: str) -> str | None:
    """Re-extract the LAST \\boxed{...} payload from a completion, grader-style.

    Returns the inner string (stripped), or None if no \\boxed{} is present.
    Mirrors the competition grader, which scores the final boxed span.
    """
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def build_assistant_content(cot: str, answer: str) -> str | None:
    """Assemble the v6 assistant completion from a reasoner CoT and ground truth.

    The reasoners emit a free-form reasoning trace whose final line is a
    ``\\boxed{<answer>}``. The v6 contract supervises::

        <reasoning> + "\\n</think>\\n\\boxed{" + answer + "}"

    so we:
      1. Strip the reasoner's own trailing \\boxed{...} line (everything from the
         LAST \\boxed{ to the end) -- that is the part we re-emit canonically.
      2. Use the remaining reasoning body as <reasoning>.
      3. Append the canonical close: "\\n</think>\\n\\boxed{" + answer + "}".

    This guarantees the completion contains EXACTLY one closing </think> and one
    \\boxed{}, holding the ground-truth answer string verbatim (so the grader's
    final-box extraction lands the exact expected string). The opening <think> is
    intentionally absent (the chat template's add_generation_prompt emits it into
    the masked prompt span). No <|im_end|>/EOS is added (the tokenizer appends it).

    Returns the assembled string, or None if the CoT has no \\boxed{} to strip
    (an off-spec reasoner output we refuse to ship).
    """
    last = _BOXED_RE.search(cot)
    if last is None:
        # Reasoner produced no boxed marker at all -> off-spec, drop.
        return None
    # Find the START of the LAST \boxed{ occurrence so we strip the whole final
    # boxed line, not just the first one (cipher/gravity bodies never contain a
    # stray \boxed, but we cut at the last match to be safe).
    cut = cot.rfind("\\boxed{")
    reasoning = cot[:cut].rstrip()
    if not reasoning:
        # Degenerate CoT that is only a boxed line with no reasoning -> drop:
        # an empty <think> body teaches nothing and would fail v6's "CoT body
        # not learned" intent.
        return None
    return f"{reasoning}\n</think>\n\\boxed{{{answer}}}"


def main() -> int:
    if not TRAIN_CSV.exists():
        print(f"ERROR: train.csv not found at {TRAIN_CSV}", file=sys.stderr)
        return 1

    # Load reasoners once.
    reasoners = {
        name: _load_reasoner(filename)
        for name, (_, filename) in CATEGORIES.items()
    }

    # Large prompts (cipher example blocks) can exceed the default csv field cap.
    csv.field_size_limit(10 ** 7)

    # Per-category bookkeeping.
    total_rows = 0
    classified = {name: 0 for name in CATEGORIES}
    unclassified = 0
    written = {name: 0 for name in CATEGORIES}
    dropped_reasoner_none = {name: 0 for name in CATEGORIES}
    dropped_assembly = {name: 0 for name in CATEGORIES}
    dropped_verify = {name: 0 for name in CATEGORIES}

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_CSV, newline="") as fh, open(OUTPUT_JSONL, "w") as out:
        reader = csv.DictReader(fh)
        for row in reader:
            total_rows += 1
            prompt = row["prompt"]
            answer = row["answer"]

            category = classify(prompt)
            if category is None:
                unclassified += 1
                continue
            classified[category] += 1

            # (3) Ask the reasoner for a chain-of-thought. It self-checks against
            # ground truth and returns None on wrong/unparseable -> skip here.
            cot = reasoners[category].generate_cot(prompt, answer)
            if cot is None:
                dropped_reasoner_none[category] += 1
                continue

            # Assemble the v6 assistant completion (strip the reasoner's own box,
            # re-emit the canonical </think> + \boxed{answer}).
            assistant_content = build_assistant_content(cot, answer)
            if assistant_content is None:
                dropped_assembly[category] += 1
                continue

            # (4) FINAL self-verification gate: re-extract the LAST \boxed{} from
            # the ASSEMBLED completion and require an exact match to ground truth.
            produced = extract_last_boxed(assistant_content)
            if produced is None or produced != answer.strip():
                dropped_verify[category] += 1
                continue

            # (1) schema-exact line for nemotron_sft_v6.py.
            record = {
                "type": category,
                "messages": [
                    {"role": "user", "content": prompt + PROMPT_SUFFIX},
                    {"role": "assistant", "content": assistant_content},
                ],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written[category] += 1

    # --- Report ------------------------------------------------------------- #
    total_written = sum(written.values())
    print("=" * 68)
    print("SFT corpus assembly complete")
    print("=" * 68)
    print(f"train.csv rows read         : {total_rows}")
    print(f"unclassified (no reasoner)  : {unclassified}")
    print(f"output                      : {OUTPUT_JSONL}")
    print("-" * 68)
    print(f"{'category':<18}{'classified':>11}{'written':>9}{'coverage':>11}")
    print("-" * 68)
    for name in CATEGORIES:
        c = classified[name]
        w = written[name]
        cov = (100.0 * w / c) if c else 0.0
        print(f"{name:<18}{c:>11}{w:>9}{cov:>10.2f}%")
    print("-" * 68)
    grand_classified = sum(classified.values())
    grand_cov = (100.0 * total_written / grand_classified) if grand_classified else 0.0
    print(f"{'TOTAL':<18}{grand_classified:>11}{total_written:>9}{grand_cov:>10.2f}%")
    print("-" * 68)
    # Drop breakdown (should be: reasoner-none accounts for the whole residual;
    # assembly/verify drops should be zero since the reasoner already self-checks).
    print("drop breakdown (per category): reasoner-None / assembly / verify-fail")
    for name in CATEGORIES:
        print(
            f"  {name:<16}: "
            f"{dropped_reasoner_none[name]:>4} / "
            f"{dropped_assembly[name]:>3} / "
            f"{dropped_verify[name]:>3}"
        )
    print("=" * 68)

    # Hard guard: the final verification gate must never silently let a wrong box
    # through. If any verify-fail fired, the assembled line diverged from the
    # reasoner's self-check and that is a bug worth failing on.
    total_verify_fail = sum(dropped_verify.values())
    if total_verify_fail:
        print(
            f"WARNING: {total_verify_fail} rows failed the assembled-line verify "
            "gate (assembled box != ground truth). Investigate before training.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
