"""Clean-room deterministic reasoner for the 'unit_conversion' category of the
NVIDIA Nemotron model-reasoning challenge.

Category fingerprint (substring on the prompt's opening sentence):
    "secret unit conversion is applied to measurements"

Problem shape
-------------
Each prompt states that a "secret unit conversion is applied to measurements",
then lists 3-5 worked example pairs of the form::

    10.08 m becomes 6.69
    17.83 m becomes 11.83
    ...

and finally asks::

    Now, convert the following measurement: 25.09 m

The Rule (derived from train.csv, clean-room)
---------------------------------------------
The hidden conversion is a pure through-origin linear scaling::

    output = factor * input          (no additive offset)

This was established empirically:
  * an affine (offset != 0) least-squares fit does *worse* than the
    through-origin fit, so the offset is genuinely zero;
  * every example output equals round(factor * input, 2) (half-up, 2 dp),
    i.e. the examples are the true product *after* 2-dp rounding.

Because the examples are pre-rounded, the factor cannot be read off exactly
from any single pair. Each pair (a, b) only constrains the factor to the
interval that rounds b back, namely::

    (b - 0.005) / a  <=  factor  <  (b + 0.005) / a

Intersecting these per-pair intervals across all examples gives the tightest
recoverable band for the factor; its midpoint is the best point estimate.
The query answer is then round(factor * Q, 2) rendered with EXACTLY two
decimals (trailing zeros kept, e.g. 19.00, 44.00).

Answer format
-------------
Fixed 2-decimal string via f"{x:.2f}" with half-up rounding. Confirmed against
all 1594 train rows: every answer has exactly two decimal places, integers
render with trailing zeros ("19.00", "44.00").

Coverage
--------
The interval-midpoint estimator reaches ~90.7% exact match. The residual ~9%
are rounding-boundary cases: the rounded examples simply do not pin the factor
tightly enough to decide which side of an x.xx5 boundary the true product
lands on. This is inherent precision loss, not a rule error.

Poison-avoidance
----------------
During corpus generation the ground-truth `answer` is available. We use it ONLY
as a self-check gate: if our deterministic derivation does not reproduce the
ground truth, generate_cot returns None (skip) rather than emit a CoT that
reaches a wrong boxed value. The reasoning itself never peeks at the answer to
*derive* the result; it derives independently and then verifies.
"""

from __future__ import annotations

import csv
import decimal
import re
import sys

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Example pairs: "<a> m becomes <b>" with flexible surrounding whitespace.
_PAIR_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*m\s+becomes\s+([0-9]+(?:\.[0-9]+)?)")
# The query measurement after the final instruction.
_QUERY_RE = re.compile(
    r"convert the following measurement:\s*([0-9]+(?:\.[0-9]+)?)\s*m"
)

# Half of one unit in the last (2nd) decimal place — the rounding half-window.
_HALF = 0.005


def _parse(prompt: str):
    """Extract (example_pairs, query) from a unit_conversion prompt.

    Returns ([(a, b), ...], Q) with floats, or None if the prompt does not
    match the expected unit_conversion shape (defensive — caller skips).
    """
    pairs = [(float(a), float(b)) for a, b in _PAIR_RE.findall(prompt)]
    qm = _QUERY_RE.search(prompt)
    if not pairs or qm is None:
        return None
    return pairs, float(qm.group(1))


# ---------------------------------------------------------------------------
# Factor recovery
# ---------------------------------------------------------------------------

def _interval(pairs):
    """Intersect the per-pair rounding intervals for the factor.

    Each example (a, b) with b = round(factor*a, 2) forces
        (b - 0.005)/a <= factor < (b + 0.005)/a.
    Returns (lo, hi). lo > hi signals an empty (inconsistent-under-rounding)
    intersection, in which case the caller falls back to least squares.
    """
    lo = float("-inf")
    hi = float("inf")
    for a, b in pairs:
        lo = max(lo, (b - _HALF) / a)
        hi = min(hi, (b + _HALF) / a)
    return lo, hi


def _least_squares(pairs):
    """Through-origin least-squares factor: sum(a*b) / sum(a*a)."""
    num = sum(a * b for a, b in pairs)
    den = sum(a * a for a, b in pairs)
    return num / den


def _recover_factor(pairs):
    """Best point estimate of the hidden factor.

    Primary: midpoint of the intersected rounding intervals (the tightest
    band the rounded examples allow). Fallback: through-origin least squares
    when the intervals do not intersect (rare numeric edge cases).
    """
    lo, hi = _interval(pairs)
    if lo > hi:
        return _least_squares(pairs)
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _round_half_up_2dp(x: float) -> str:
    """Round x to 2 decimals half-up and render with exactly 2 decimals.

    Keeps trailing zeros ("19.00", "44.00") to match the competition's fixed
    2-dp answer format. Uses Decimal to get deterministic half-up behaviour
    independent of binary float ties.
    """
    d = decimal.Decimal(str(x)).quantize(
        decimal.Decimal("0.01"), rounding=decimal.ROUND_HALF_UP
    )
    return f"{d:.2f}"


# ---------------------------------------------------------------------------
# Deterministic solve + CoT emission
# ---------------------------------------------------------------------------

def _solve(prompt: str):
    """Deterministically derive (boxed_answer_str, pairs, Q, factor).

    Returns None if the prompt is not a parseable unit_conversion problem.
    Does NOT consult any ground-truth answer.
    """
    parsed = _parse(prompt)
    if parsed is None:
        return None
    pairs, Q = parsed
    factor = _recover_factor(pairs)
    boxed = _round_half_up_2dp(factor * Q)
    return boxed, pairs, Q, factor


def generate_cot(prompt: str, answer: str) -> str | None:
    """Produce a step-by-step CoT ending in \\boxed{...} for a unit_conversion
    problem, or None if it cannot be solved confidently.

    The CoT is written so a temperature=0 model can imitate it: parse the
    example pairs, recover the hidden multiplicative factor by intersecting the
    rounding intervals, apply it to the query, and round half-up to 2 decimals.

    `answer` (ground truth) is used ONLY as a final self-check gate: if our
    independent derivation does not reproduce it, we return None instead of
    emitting a CoT that would poison training with a wrong boxed value.
    """
    solved = _solve(prompt)
    if solved is None:
        return None
    boxed, pairs, Q, factor = solved

    # Self-check gate: never emit a CoT whose boxed value disagrees with the
    # provided ground truth. We require an EXACT fixed-2dp STRING match, not a
    # numeric-tolerance match. A numeric gate (e.g. abs diff <= 1e-9) would let
    # through a genuinely-derived "19.0"/"19.00" vs ground-truth string mismatch,
    # or a rounding-boundary tie where our interval-midpoint factor rounds the
    # product to one cent while ground truth is the neighbouring cent. Emitting
    # such a row would leave the reasoning body (which rounds factor*Q to `boxed`)
    # disagreeing with a snapped \boxed{ground_truth}. For a reasoning corpus that
    # self-inconsistency teaches broken arithmetic, so we drop the row instead.
    if answer is not None:
        gt = answer.strip()
        # Reject off-spec (non-numeric) ground truth for this always-numeric
        # category; a parse failure is a drop, never a silent emit.
        try:
            float(gt)
        except ValueError:
            return None
        if boxed != gt:
            return None

    # --- Build the verbalizable reasoning ----------------------------------
    lines = []
    lines.append(
        "The Wonderland unit conversion multiplies each measurement by a "
        "single hidden factor (a pure scaling with no added constant), and the "
        "example outputs are that product rounded to 2 decimals."
    )
    lines.append("")
    lines.append("Step 1: Read off the example pairs (input m -> output):")
    for a, b in pairs:
        lines.append(f"  {a:g} m -> {b:g}")
    lines.append("")

    lines.append(
        "Step 2: Each pair gives a rough factor = output / input. Because the "
        "outputs are already rounded to 2 decimals, each pair only pins the "
        "factor to the band [(output - 0.005)/input, (output + 0.005)/input]:"
    )
    lo = float("-inf")
    hi = float("inf")
    for a, b in pairs:
        plo = (b - _HALF) / a
        phi = (b + _HALF) / a
        lo = max(lo, plo)
        hi = min(hi, phi)
        lines.append(
            f"  {b:g}/{a:g} = {b / a:.4f}  ->  band [{plo:.4f}, {phi:.4f}]"
        )
    lines.append("")

    interval_empty = lo > hi
    if not interval_empty:
        lines.append(
            "Step 3: Intersect all the bands to get the tightest range the "
            f"rounded examples allow: [{lo:.4f}, {hi:.4f}]. Take its midpoint "
            f"as the factor: factor = ({lo:.4f} + {hi:.4f}) / 2 = {factor:.4f}."
        )
    else:
        # Fallback path: intervals did not intersect, use least squares.
        ls = _least_squares(pairs)
        lines.append(
            "Step 3: The rounded bands do not overlap exactly, so estimate the "
            "factor by the best through-origin fit "
            "sum(input*output)/sum(input*input) = "
            f"{ls:.4f}."
        )

    lines.append("")
    # Print the product to 6 decimals, not 4. A 4-decimal display can sit on a
    # false rounding boundary: e.g. a true product of 10.96499... prints as
    # "10.9650" at 4dp, whose half-up round to 2dp LOOKS like 10.97 even though
    # the genuine value rounds to 10.96 (the box). Showing 6dp exposes the
    # sub-cent digit so the body's stated half-up round is unambiguous and the
    # reasoning stays consistent with \boxed{} (boxed is computed from the full
    # product, not from this display).
    lines.append(
        f"Step 4: Apply the factor to the query {Q:g} m: "
        f"{factor:.4f} * {Q:g} = {factor * Q:.6f}."
    )
    lines.append(
        f"Step 5: Round half-up to 2 decimals (keeping trailing zeros): {boxed}."
    )
    lines.append("")
    lines.append(f"\\boxed{{{boxed}}}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test over train.csv unit_conversion rows
# ---------------------------------------------------------------------------

_FINGERPRINT = "secret unit conversion is applied to measurements"
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_TRAIN_CSV = "/tmp/nemo-comp/train.csv"


def _iter_unit_conversion_rows(path: str = _TRAIN_CSV):
    csv.field_size_limit(10 ** 7)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if _FINGERPRINT in row["prompt"]:
                yield row


def _extract_boxed(cot: str):
    m = _BOXED_RE.search(cot)
    return m.group(1) if m else None


def _self_test(path: str = _TRAIN_CSV) -> None:
    # The competition answers are fixed 2-decimal strings, so the meaningful
    # metric is EXACT string match (a 0.01 miss is wrong). We report two things:
    #   * deterministic solve = how often the ungated solver hits the answer
    #     exactly (the honest skill ceiling / "coverage");
    #   * gated emission = the CoTs that actually enter the corpus. The
    #     self-check gate only ever *removes* wrong emissions, so this subset is
    #     100% exact by construction (verified here, not assumed).
    total = 0
    solved = 0
    emitted = 0
    emitted_correct = 0
    fails = []
    for row in _iter_unit_conversion_rows(path):
        total += 1
        gt = row["answer"].strip()

        # Ungated derivation (skill ceiling), compared by EXACT string match.
        s = _solve(row["prompt"])
        if s is not None:
            boxed = s[0]
            if boxed == gt:
                solved += 1
            elif len(fails) < 8:
                fails.append((row["id"], gt, boxed))

        # Gated emission (what actually enters the corpus).
        cot = generate_cot(row["prompt"], row["answer"])
        if cot is not None:
            emitted += 1
            if _extract_boxed(cot) == gt:
                emitted_correct += 1

    pct = 100.0 * solved / total if total else 0.0
    print(f"[unit_conversion] deterministic solve (exact): {solved}/{total} = {pct:.2f}%")
    if emitted:
        print(
            f"[unit_conversion] gated emission: {emitted} CoTs emitted, "
            f"{emitted_correct} exact "
            f"({100.0 * emitted_correct / emitted:.2f}% precision)"
        )
    else:
        print("[unit_conversion] gated emission: none")
    if fails:
        print("  sample ungated misses (id, ground_truth, derived):")
        for fid, gt, got in fails:
            print(f"    {fid}  gt={gt}  got={got}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else _TRAIN_CSV
    _self_test(path)
