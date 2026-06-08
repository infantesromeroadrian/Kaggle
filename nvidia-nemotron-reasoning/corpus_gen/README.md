# corpus_gen — clean-room SFT corpus for the Nemotron reasoning challenge

This directory builds the supervised fine-tuning (SFT) corpus consumed by
`../train/nemotron_sft_v6.py`. Each kept example is a deterministic
chain-of-thought (CoT) that lands the exact `\boxed{}` answer the competition
grader checks at temperature 0.

## Approach credit (public writeup) vs. originality (clean-room)

The high-level **approach** — *deterministic, per-category chain-of-thought
solvers; supervise the CoT via SFT; decode at temperature 0 so the boxed answer
is reproducible* — is **inspired by the public competition writeup**:

> Kaggle discussion **689915**
> https://www.kaggle.com/competitions/nvidia-nemotron-nano-model-reasoning/discussion/689915

**All code and data in this repository are original / clean-room.** Every
per-category rule was re-derived from the competition `train.csv` examples by the
four reasoners under `reasoners/`. Nothing was read, opened, or copied from any
third-party (unlicensed) solution. The only inputs consulted were the public
`train.csv` and the public discussion above.

## Pipeline

```
/tmp/nemo-comp/train.csv
        │  classify by prompt substring (one discriminator per category)
        ▼
reasoners/{numeral,unit_conversion,gravity,cipher}.py  →  generate_cot(prompt, answer)
        │  reasoner self-checks vs ground truth; returns None on wrong/unparseable
        ▼
build_corpus.py
        │  assemble v6 line; FINAL self-verification gate (re-extract \boxed → must == answer)
        ▼
cot_corpus_v1.jsonl   (consumed by ../train/nemotron_sft_v6.py)
```

`build_corpus.py` is a pure function of `train.csv` — no randomness — so a
re-run produces a byte-identical JSONL.

## Categories (4 implemented)

| category          | discriminator (prompt substring)                                  | rule (clean-room, from train.csv)                          |
|-------------------|-------------------------------------------------------------------|------------------------------------------------------------|
| `numeral`         | `numbers are secretly converted into a different numeral system`   | canonical integer → Roman numeral (subtractive)            |
| `unit_conversion` | `secret unit conversion is applied to measurements`               | through-origin linear scale; factor from rounding intervals|
| `gravity`         | `gravitational constant`                                          | `d = 0.5·g·t²`; recover `g` from rounded observations      |
| `cipher`          | `encryption rules are used on text`                              | monoalphabetic substitution + closed Wonderland vocabulary |

Rows of not-yet-implemented categories (no reasoner) are skipped, not errored.

## Output schema (must byte-match `nemotron_sft_v6.py`)

One JSON object per line:

```json
{
  "type": "<category>",
  "messages": [
    {"role": "user",      "content": "<competition prompt>" + PROMPT_SUFFIX},
    {"role": "assistant", "content": "<reasoning>\n</think>\n\\boxed{<answer>}"}
  ]
}
```

* **`PROMPT_SUFFIX`** is copied verbatim from `nemotron_sft_v6.py` (CELL 1) and is
  byte-verified equal to it:
  `"\nPlease put your final answer inside \`\\boxed{}\`. For example: \`\\boxed{your answer}\`"`.
  Train and eval prompts therefore byte-match.
* The chat template's `add_generation_prompt=True` emits the **opening
  `<think>\n`** into the (masked) prompt span, so the assistant content does
  **not** re-open `<think>`. It starts with the reasoning body and ends with the
  closing `</think>` followed by `\boxed{<answer>}`.
* v6 supervises the closing `</think>`, the boxed answer, and the stop token. The
  tokenizer appends the stop token, so the assistant content does **not** hardcode
  `<|im_end|>` / EOS.

## Self-verification gate

After assembling each line, the builder re-extracts the **last** `\boxed{...}`
with the grader-style regex and drops any row whose box ≠ ground truth (or whose
box is missing). This validates the assembled completion, not just the reasoner's
internal check. On the current corpus the assembly/verify drops are **zero**: the
only residual losses are rows the reasoner itself declined (rounding-boundary ties
in `unit_conversion` and `gravity`).

### Body/box self-consistency (no snapping)

For the two numeric categories (`gravity`, `unit_conversion`) the reasoner now
emits a row ONLY when its genuinely-derived value reproduces the ground-truth 2dp
string **exactly**. Earlier the reasoners would, on a rounding-boundary tie, accept
a value numerically within tolerance and then SNAP the `\boxed{}` to ground truth —
leaving the CoT body rounding to one cent while the box showed the neighbouring
cent. For a reasoning corpus that self-inconsistency teaches broken arithmetic, so
those ties (149 `gravity` rows) are now **dropped, not snapped**. The
`unit_conversion` Step-4 line also prints the product to 6 decimals rather than 4 so
its stated half-up rounding never sits on a false display boundary. Result: every
emitted row has *body-computed value == boxed value == ground truth*.

## Run

```bash
python3 build_corpus.py
```

### Verified coverage (this corpus: `cot_corpus_v1.jsonl`)

| category          | classified | written | coverage |
|-------------------|-----------:|--------:|---------:|
| numeral           |       1576 |    1576 |  100.00% |
| unit_conversion   |       1594 |    1446 |   90.72% |
| gravity           |       1597 |    1448 |   90.67% |
| cipher            |       1576 |    1576 |  100.00% |
| **TOTAL**         |   **6343** | **6046**|  **95.32%** |

`train.csv` rows read: 9500 · unclassified (categories without a reasoner): 3157.

Every written line passes the final `\boxed{}` self-verification against ground
truth, so corpus precision is 100% by construction.
