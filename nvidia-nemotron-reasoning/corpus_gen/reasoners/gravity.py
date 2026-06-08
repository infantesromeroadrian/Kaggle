"""Clean-room deterministic reasoner for the 'gravity' category.

NVIDIA Nemotron Model Reasoning Challenge.

Problem shape (derived solely from competition train.csv):

    In Alice's Wonderland, the gravitational constant has been secretly changed.
    Here are some example observations:
    For t = T1 s, distance = D1 m
    For t = T2 s, distance = D2 m
    ...
    Now, determine the falling distance for t = Tq s given d = 0.5*g*t^2.

The physics formula d = 0.5 * g * t^2 is stated explicitly in the prompt; only
the constant g is secret and changes per prompt. Each example observation lets
us recover g via g = 2*D / T^2. The catch: the example distances D are rounded
to 2 decimals, so a single observation only pins g to an interval. We intersect
the per-observation intervals and take the midpoint, then evaluate the formula
at the query time Tq.

Empirically (measured on all 1597 train gravity rows): interval-midpoint reaches
~90.67% EXACT 2dp-string match (1448/1597). The ~9% residual are rounding-boundary
ties where the rounded examples cannot pin g tightly enough to decide a final-digit
rounding (almost all off by exactly 0.01). This is inherent precision loss from the
rounded inputs, not a rule error. Snapping g to "clean" decimals was tested and
makes it worse, so g is treated as an arbitrary float.

Self-consistency policy (corpus): generate_cot emits a row ONLY when the genuinely
derived value reproduces the ground-truth 2dp string EXACTLY. The ~9% boundary-tie
rows are DROPPED, never snapped: snapping the \boxed{} answer to ground truth while
the reasoning body rounds d_pred to the neighbouring cent would ship a CoT whose
body and box disagree -- self-inconsistent arithmetic that poisons a reasoning
corpus. A dropped row is strictly better than an inconsistent one.

Answer format (derived from train.csv): str(round(x, 2)) — Python default that
STRIPS trailing zeros (45.0, not 45.00; 38.6; 154.62). Every observed answer has
a decimal point. This is DISTINCT from the unit_conversion category which keeps a
fixed 2dp; using fixed 2dp here drops accuracy to ~81.8%, so the stripped format
is load-bearing.

Public API:
    generate_cot(prompt, answer) -> str | None
        Returns a step-by-step chain-of-thought ending in \\boxed{...}, or None
        if the prompt cannot be parsed (skip rather than emit a wrong CoT, since
        wrong CoTs poison training data).
"""

import re

# Half-width of the rounding interval for a value rounded to 2 decimals.
# A reported D could be the round-to-2dp of anything in [D - 0.005, D + 0.005).
_HALF = 0.005

# Regexes compiled once. The example lines and the query share the same number
# grammar; t and d are non-negative decimals.
_OBS_RE = re.compile(r"For t = ([\d.]+)\s*s,\s*distance = ([\d.]+)\s*m")
_QUERY_RE = re.compile(r"determine the falling distance for t = ([\d.]+)\s*s")


def _parse(prompt):
    """Extract (observations, query_time) from a gravity prompt.

    Returns (list[(t, d)], tq) or None if the prompt is not a parseable gravity
    problem. We require at least one observation and a query time; otherwise the
    constant g is unrecoverable and we must skip.
    """
    obs = [(float(t), float(d)) for t, d in _OBS_RE.findall(prompt)]
    qm = _QUERY_RE.search(prompt)
    if not obs or qm is None:
        return None
    return obs, float(qm.group(1))


def _recover_g(obs):
    """Recover the secret constant g from rounded observations.

    Each observation d = round(0.5 * g * t^2, 2) constrains g to the interval
    [2*(d - 0.005)/t^2, 2*(d + 0.005)/t^2]. Intersecting all such intervals
    narrows g; we return the midpoint of the intersection.

    Fallback: if the intervals do not overlap (possible when an example sits
    exactly on a rounding boundary), fall back to the through-origin
    least-squares estimate g = 2*sum(d*t^2)/sum(t^4), which uses all data
    jointly. Affine/with-offset fits do worse, confirming the model is
    through-origin.
    """
    lo, hi = -float("inf"), float("inf")
    for t, d in obs:
        t2 = t * t
        lo = max(lo, 2.0 * (d - _HALF) / t2)
        hi = min(hi, 2.0 * (d + _HALF) / t2)
    if lo <= hi:
        return (lo + hi) / 2.0
    num = sum(d * t * t for t, d in obs)
    den = sum(t ** 4 for t, _ in obs)
    return 2.0 * num / den


def _format_answer(x):
    """Render the distance the way train.csv answers are rendered.

    str(round(x, 2)) strips trailing zeros (45.0 not 45.00). This matches every
    observed gravity answer and is intentionally different from unit_conversion's
    fixed-2dp format.
    """
    return str(round(x, 2))


def solve(prompt):
    """Deterministically solve a gravity prompt to its answer string.

    Returns the formatted answer, or None if the prompt cannot be parsed.
    """
    parsed = _parse(prompt)
    if parsed is None:
        return None
    obs, tq = parsed
    g = _recover_g(obs)
    d = 0.5 * g * tq * tq
    return _format_answer(d)


def generate_cot(prompt, answer):
    """Build a step-by-step CoT for a gravity problem.

    The CoT derives the answer by genuine reasoning (recover g from the
    observations, then apply the formula at the query time). The ground-truth
    `answer` is used ONLY as a final self-check: if our deterministic derivation
    disagrees with the ground truth, we return None so we never emit a CoT whose
    \\boxed{} value is wrong (wrong CoTs poison training).

    Returns the CoT string ending in \\boxed{...}, or None to skip.
    """
    parsed = _parse(prompt)
    if parsed is None:
        return None
    obs, tq = parsed

    # Recover g and the per-observation estimates so the CoT can show its work.
    g = _recover_g(obs)
    per_obs = [(t, d, 2.0 * d / (t * t)) for t, d in obs]
    d_pred = 0.5 * g * tq * tq
    pred_str = _format_answer(d_pred)

    # Self-check against ground truth. We require an EXACT 2dp-string match
    # between our genuinely-derived value and the ground truth. We deliberately
    # do NOT accept a numeric-tolerance match and then snap the box to the
    # ground-truth string: on rounding-boundary ties (the rounded example
    # distances cannot pin g tightly enough to decide the final digit) the body
    # of the CoT genuinely rounds d_pred to one 2dp value while ground truth is
    # the neighbouring cent. Snapping the box would leave the reasoning body and
    # the boxed answer disagreeing -- a self-inconsistent CoT that teaches broken
    # arithmetic. For a reasoning corpus that is worse than dropping the row, so
    # we drop: if the honest derivation does not reproduce the ground-truth 2dp
    # string exactly, return None.
    if pred_str != answer:
        return None

    # ---- Build the chain-of-thought ----
    lines = []
    lines.append(
        "The problem states the falling distance follows d = 0.5*g*t^2, where g "
        "is a secret constant that is the same for every observation. I will "
        "recover g from the example observations, then apply the formula at the "
        "requested time."
    )
    lines.append("")
    lines.append(
        "Step 1: Solve the formula for g. From d = 0.5*g*t^2 we get g = 2*d / t^2."
    )
    lines.append("")
    lines.append("Step 2: Estimate g from each observation:")
    for t, d, gi in per_obs:
        lines.append(
            f"  - t = {t:g} s, d = {d:g} m  ->  g = 2*{d:g} / {t:g}^2 = {gi:.4f}"
        )
    lines.append("")
    lines.append(
        "Step 3: The example distances are rounded to 2 decimals, so each one "
        "only constrains g to a small interval [2*(d-0.005)/t^2, 2*(d+0.005)/t^2]. "
        "I intersect those intervals across all observations and take the "
        "midpoint as the best estimate of the true g."
    )
    lines.append(f"  Combined estimate: g = {g:.4f}")
    lines.append("")
    lines.append(
        f"Step 4: Apply d = 0.5*g*t^2 at the query time t = {tq:g} s:"
    )
    lines.append(
        f"  d = 0.5 * {g:.4f} * {tq:g}^2 = 0.5 * {g:.4f} * {tq * tq:.4f} = {d_pred:.4f}"
    )
    lines.append("")
    lines.append(
        f"Step 5: Round to 2 decimals (dropping trailing zeros): {pred_str}."
    )
    lines.append("")
    lines.append(f"\\boxed{{{pred_str}}}")

    return "\n".join(lines)


def _self_test():
    """Run the reasoner over all gravity rows in train.csv and report coverage."""
    import csv
    import os

    csv.field_size_limit(10 ** 7)
    candidates = [
        "/tmp/nemo-comp/train.csv",
        os.path.join(os.path.dirname(__file__), "train.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        print("train.csv not found in any known location; skipping self-test.")
        return

    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if "gravitational constant" in row["prompt"]:
                rows.append(row)

    solved = 0
    skipped = 0
    boxed_re = re.compile(r"\\boxed\{([^}]*)\}")
    for row in rows:
        cot = generate_cot(row["prompt"], row["answer"])
        if cot is None:
            skipped += 1
            continue
        m = boxed_re.search(cot)
        if not m:
            continue
        boxed = m.group(1)
        if boxed == row["answer"]:
            solved += 1
        else:
            # Numeric tolerance fallback (relative 1e-2).
            try:
                if abs(float(boxed) - float(row["answer"])) <= 1e-2 * max(
                    abs(float(row["answer"])), 1e-9
                ):
                    solved += 1
            except (ValueError, TypeError):
                pass

    total = len(rows)
    cov = 100.0 * solved / total if total else 0.0
    print(f"gravity self-test: solved {solved}/{total} = {cov:.2f}%  (skipped {skipped})")


if __name__ == "__main__":
    _self_test()
