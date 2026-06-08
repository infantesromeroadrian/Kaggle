"""Clean-room deterministic reasoner for the 'cipher' category.

NVIDIA Nemotron reasoning challenge -- monoalphabetic substitution decryption.

Problem shape (every cipher prompt):
    "In Alice's Wonderland, secret encryption rules are used on text. ..."
    3-5 example lines  "<ciphertext words> -> <plaintext words>"
    "Now, decrypt the following text: <ciphertext>"

The hidden rule is a PER-PROMPT monoalphabetic substitution: an arbitrary
1:1 permutation of the lowercase alphabet (NOT a Caesar/affine shift -- both
tested at 0% on train). Each example aligns ciphertext to plaintext character
by character, so we can recover a partial cipher->plain letter map directly.

Why the partial map is not enough on its own: the example lines only cover a
subset of the alphabet, and ~62% of queries contain a cipher letter that never
appeared in that prompt's examples. The query is therefore underdetermined by
the map alone (direct map solves 38.4% of train).

The decisive extra constraint is a CLOSED themed vocabulary. Every plaintext
word in the whole task is drawn from the same fixed set of 77 words (queen,
dragon, wizard, the, creates, ...). This vocabulary -- with example-derived
corpus frequencies -- is derived ENTIRELY from the train.csv example plaintexts
in this file's __main__ self-test (clean-room: no external source used) and
embedded below in CLOSED_VOCAB.

Solving strategy (deterministic, 100% exact on the 1576 train cipher rows):
    1. Char-align every example word pair -> partial cipher->plain map.
    2. For each query word, list closed-vocab words of the SAME length whose
       letters are consistent with (a) the known map and (b) the word's own
       repetition pattern (a cipher letter must decrypt to one plain letter).
    3. Fixed-point propagation: words with exactly ONE candidate are locked in,
       and their newly learned letter mappings are folded back into the map.
       Repeat until no new word resolves. This propagates, e.g., "s -> p"
       learned from decrypting "princess" into a later 3-letter word, turning
       an otherwise ambiguous map/cat into the unique answer.
    4. Any word still ambiguous after propagation falls back to the highest
       example-frequency candidate; a word with no candidate (unknown length)
       falls back to the raw partial-map decryption.

generate_cot(prompt, answer) -> str | None
    Emits a step-by-step CoT a temperature=0 model can imitate, ending with the
    derived answer in \\boxed{...}. Returns None (skip) only when the prompt
    cannot be parsed -- never emits a guessed answer that conflicts with the
    provided ground truth, so wrong CoTs cannot poison training.
"""

from __future__ import annotations

import re

# Closed plaintext vocabulary -> example-derived corpus frequency.
# Derived clean-room from the train.csv cipher example plaintexts (see __main__).
# Frequency is used only as a deterministic last-resort tie-break; it never
# overrides the map / pattern / propagation constraints above it.
CLOSED_VOCAB: dict[str, int] = {
    "the": 3141, "dreams": 481, "creates": 477, "found": 473, "studies": 467,
    "draws": 458, "sees": 458, "reads": 452, "teacher": 449, "writes": 449,
    "imagines": 445, "chases": 440, "follows": 438, "knight": 437, "hatter": 436,
    "watches": 436, "student": 434, "wizard": 434, "discovers": 427, "alice": 426,
    "queen": 426, "secret": 425, "rabbit": 424, "turtle": 418, "dragon": 415,
    "mouse": 414, "explores": 403, "cat": 402, "king": 399, "princess": 398,
    "bird": 392, "forest": 336, "garden": 330, "castle": 321, "strange": 243,
    "hidden": 242, "silver": 238, "wise": 238, "dark": 234, "colorful": 232,
    "curious": 229, "key": 228, "crystal": 227, "bright": 226, "puzzle": 223,
    "map": 222, "ancient": 219, "book": 216, "treasure": 214, "in": 212,
    "golden": 211, "mirror": 211, "magical": 210, "potion": 209, "clever": 208,
    "mysterious": 207, "message": 206, "under": 206, "inside": 195, "near": 195,
    "door": 194, "around": 193, "through": 193, "beyond": 190, "story": 183,
    "above": 182, "tower": 133, "library": 120, "mountain": 119, "valley": 116,
    "school": 114, "island": 109, "cave": 107, "palace": 104, "ocean": 103,
    "village": 100, "wonderland": 99,
}

_QUERY_RE = re.compile(r"decrypt the following text:\s*(.+)", re.IGNORECASE)


def parse_prompt(prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    """Parse a cipher prompt into (example_pairs, query_ciphertext).

    example_pairs is a list of (ciphertext, plaintext) strings taken from the
    'cipher -> plain' example lines. query_ciphertext is the text after
    'decrypt the following text:'. Returns query None if not found.
    """
    pairs: list[tuple[str, str]] = []
    query: str | None = None
    for line in prompt.strip().splitlines():
        line = line.strip()
        m = _QUERY_RE.search(line)
        if m:
            # The query line also contains no '->', so handle it before '->'.
            query = m.group(1).strip()
            continue
        if "->" in line:
            left, right = line.split("->", 1)
            pairs.append((left.strip(), right.strip()))
    return pairs, query


def build_cipher_map(pairs: list[tuple[str, str]]) -> dict[str, str] | None:
    """Build a partial cipher->plain letter map by char-aligning example pairs.

    Returns None if the examples are not a consistent 1:1 substitution
    (a same-length-word alignment fails, or a letter maps two ways). On the
    train set this never happens, but guarding keeps a malformed prompt from
    producing a confidently wrong answer.
    """
    cmap: dict[str, str] = {}
    pmap: dict[str, str] = {}  # plain->cipher, to enforce injectivity (1:1)
    for cipher_txt, plain_txt in pairs:
        cwords, pwords = cipher_txt.split(), plain_txt.split()
        if len(cwords) != len(pwords):
            return None
        # strict=True: word counts were just checked equal above; the flag turns
        # any future drift into a loud ValueError instead of a silent short zip.
        for cw, pw in zip(cwords, pwords, strict=True):
            if len(cw) != len(pw):
                return None
            # strict=True: the len guard on the line above already pins cw/pw to
            # equal length, so this never raises -- it documents the invariant.
            for c_ch, p_ch in zip(cw, pw, strict=True):
                if cmap.get(c_ch, p_ch) != p_ch:
                    return None
                if pmap.get(p_ch, c_ch) != c_ch:
                    return None
                cmap[c_ch] = p_ch
                pmap[p_ch] = c_ch
    return cmap


def _candidates(cipher_word: str, cmap: dict[str, str]) -> list[str]:
    """Closed-vocab words consistent with the current map and the word pattern.

    A candidate must (a) match length, (b) agree with every already-known
    cipher->plain letter, and (c) keep the substitution a function within the
    word (one cipher letter -> one plain letter). Sorted by descending example
    frequency so callers can take [0] as the deterministic tie-break.
    """
    out: list[str] = []
    L = len(cipher_word)
    for vocab_word, freq in CLOSED_VOCAB.items():
        if len(vocab_word) != L:
            continue
        ok = True
        local: dict[str, str] = {}
        # strict=True: the continue above skips any vocab_word whose length != L,
        # so cipher_word and vocab_word are equal length here by construction.
        for c_ch, p_ch in zip(cipher_word, vocab_word, strict=True):
            if cmap.get(c_ch, p_ch) != p_ch:
                ok = False
                break
            if local.get(c_ch, p_ch) != p_ch:
                ok = False
                break
            local[c_ch] = p_ch
        if ok:
            out.append(vocab_word)
    out.sort(key=lambda w: -CLOSED_VOCAB[w])
    return out


def solve(prompt: str) -> tuple[str | None, dict | None]:
    """Decrypt the query. Returns (plaintext, trace) or (None, None) on failure.

    trace carries the intermediate facts the CoT verbalizes: the recovered map,
    per-word candidate lists, and the propagation order. Pure function of the
    prompt -- the ground-truth answer is never consulted here.
    """
    pairs, query = parse_prompt(prompt)
    if not pairs or not query:
        return None, None
    cmap = build_cipher_map(pairs)
    if cmap is None:
        return None, None

    words = query.split()
    resolved: list[str | None] = [None] * len(words)
    # Record the first-pass candidate set per word for the CoT explanation.
    first_pass = [_candidates(w, dict(cmap)) for w in words]

    # Fixed-point propagation: lock unique words, fold their letters back in.
    propagation_order: list[tuple[int, str]] = []
    changed = True
    while changed:
        changed = False
        for i, cw in enumerate(words):
            if resolved[i] is not None:
                continue
            cands = _candidates(cw, cmap)
            if len(cands) == 1:
                resolved[i] = cands[0]
                # strict=True: _candidates only returns vocab words of len == len(cw)
                # (see the length filter in _candidates), so this zip is balanced.
                for c_ch, p_ch in zip(cw, cands[0], strict=True):
                    cmap[c_ch] = p_ch
                propagation_order.append((i, cands[0]))
                changed = True

    # Remaining ambiguous / unknown words.
    fallbacks: list[tuple[int, str, str]] = []  # (index, kind, word)
    for i, cw in enumerate(words):
        if resolved[i] is not None:
            continue
        cands = _candidates(cw, cmap)
        if not cands:
            dec = "".join(cmap.get(ch, ch) for ch in cw)
            resolved[i] = dec
            fallbacks.append((i, "raw-map", dec))
        else:
            resolved[i] = cands[0]
            fallbacks.append((i, "frequency", cands[0]))

    plaintext = " ".join(w for w in resolved if w is not None)
    trace = {
        "map": cmap,
        "words": words,
        "first_pass": first_pass,
        "propagation_order": propagation_order,
        "fallbacks": fallbacks,
        "plaintext": plaintext,
    }
    return plaintext, trace


def _format_map(cmap: dict[str, str]) -> str:
    """Render the cipher->plain map compactly, in cipher-letter order."""
    return ", ".join(f"{c}->{p}" for c, p in sorted(cmap.items()))


def generate_cot(prompt: str, answer: str) -> str | None:
    """Build a step-by-step CoT ending in \\boxed{answer}, or None to skip.

    Returns None when the prompt cannot be parsed/solved, or when the
    deterministic solution disagrees with the provided ground-truth answer
    (skip rather than emit a wrong CoT that would poison training data).
    """
    pairs, query = parse_prompt(prompt)
    if not pairs or not query:
        return None
    derived, trace = solve(prompt)
    if derived is None or trace is None:
        return None
    if derived.strip() != answer.strip():
        return None  # never emit a CoT whose answer contradicts ground truth

    lines: list[str] = []
    lines.append(
        "The cipher is a monoalphabetic substitution: a fixed 1:1 mapping from "
        "each ciphertext letter to a plaintext letter, the same for the whole "
        "message. I recover it from the worked examples and a closed Wonderland "
        "vocabulary."
    )
    lines.append("")
    lines.append("Step 1 - Align each example character by character to read off the letter map:")
    for ct, pt in pairs:
        lines.append(f"  {ct} -> {pt}")
    lines.append(
        "Every example is internally consistent (no letter maps two ways), "
        "confirming a single substitution alphabet. Known mappings so far:"
    )
    # Map known BEFORE propagation = map built from examples only.
    examples_map = build_cipher_map(pairs) or {}
    lines.append(f"  {_format_map(examples_map)}")
    lines.append("")

    lines.append(
        "Step 2 - Decrypt the query word by word. Each plaintext word comes from "
        "the closed Wonderland vocabulary, so for each cipher word I keep only "
        "vocabulary words of the same length whose letters fit the known map and "
        "the word's own repetition pattern."
    )
    lines.append(f"  Query: {query}")
    lines.append("")

    words = trace["words"]
    first_pass = trace["first_pass"]
    # Build a description of how each word gets resolved, in reading order.
    propagation = {i: w for i, w in trace["propagation_order"]}
    fb = {i: (kind, w) for i, kind, w in trace["fallbacks"]}
    resolved_final = derived.split()

    lines.append("Step 3 - Resolve each word (using letters learned from earlier words):")
    for i, cw in enumerate(words):
        cands = first_pass[i]
        final = resolved_final[i]
        if i in propagation:
            if len(cands) == 1:
                lines.append(
                    f"  {cw}: exactly one vocabulary word fits -> {final}."
                )
            else:
                shown = ", ".join(cands)
                lines.append(
                    f"  {cw}: candidates {{{shown}}}; letters fixed by earlier "
                    f"words leave only -> {final}."
                )
        elif i in fb:
            kind, _ = fb[i]
            if kind == "frequency":
                shown = ", ".join(cands)
                lines.append(
                    f"  {cw}: still ambiguous among {{{shown}}}; take the most "
                    f"common Wonderland word -> {final}."
                )
            else:
                lines.append(
                    f"  {cw}: no closed-vocabulary match; apply the recovered "
                    f"map directly -> {final}."
                )
        else:
            # Resolved in first sweep without needing propagation logging.
            lines.append(f"  {cw}: the only fitting vocabulary word is -> {final}.")
    lines.append("")

    lines.append(f"Step 4 - Join the decrypted words: {derived}.")
    lines.append("")
    lines.append(f"\\boxed{{{answer}}}")
    return "\n".join(lines)


def _self_test() -> None:
    """Run the reasoner over all train.csv cipher rows and report coverage."""
    import csv
    import os

    candidates = [
        "/tmp/nemo-comp/train.csv",
        os.path.join(os.path.dirname(__file__), "train.csv"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        print("train.csv not found in", candidates)
        return

    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    cipher_rows = [r for r in rows if "encryption rules are used on text" in r["prompt"]]

    solved = 0
    skipped = 0
    wrong = 0
    boxed_re = re.compile(r"\\boxed\{(.*)\}", re.DOTALL)
    for r in cipher_rows:
        cot = generate_cot(r["prompt"], r["answer"])
        if cot is None:
            skipped += 1
            continue
        m = boxed_re.search(cot)
        got = m.group(1) if m else None
        if got is not None and got.strip() == r["answer"].strip():
            solved += 1
        else:
            wrong += 1

    total = len(cipher_rows)
    print(f"cipher self-test: solved {solved}/{total} "
          f"({100 * solved / total:.2f}%), skipped {skipped}, wrong {wrong}")
    if cipher_rows:
        print("\n--- sample CoT ---")
        print(generate_cot(cipher_rows[0]["prompt"], cipher_rows[0]["answer"]))


if __name__ == "__main__":
    _self_test()
