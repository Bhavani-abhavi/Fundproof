"""Seeded mutations of reference sentences, for measuring the numeric checker (docs/innovations.md, SIG "Data").

Each mutation changes ONE number (or its comparator) in a cited reference sentence, keeping the citation. All of them
are errors the checker should flag:
- value_perturb: the value moved by 3% to 30% (same format), never onto a value the cited section contains;
- magnitude_x10: multiplied or divided by 10 ($500,000 -> $50,000);
- unit_swap: % <-> bps, months <-> years, without converting the value;
- comparator_flip: the comparator next to the number changed class (at least -> more than, or less -> or more);
- cross_section: the value replaced by a same-unit value from an uncited section, absent from the cited one;
- wrong_row: (table rows only) another row's value from the same column of the same table. This is the expected blind
  spot: the value IS in the cited section, just bound to another row.

Controls that must NOT be flagged:
- unmutated: the reference sentence as it is;
- equivalent_rewrite: the same quantity written differently ($1,000,000 -> $1 million; 24 months -> 2 years;
  1.25 -> 1.25x; 10.0% -> 10 percent).

A mutation that would accidentally produce a value the cited section contains (other than wrong_row) is redrawn up to
20 times and dropped if no valid draw exists, so every counted mutation is a real error.
"""

import random
import re
from dataclasses import asdict, dataclass
from fractions import Fraction

from uwcopilot.provenance.numbers import DIGITS, NumberMention, extract_numbers, same_quantity
from uwcopilot.provenance.reference import ReferenceSentence

MUTATION_TYPES: tuple[str, ...] = (
    "value_perturb",
    "magnitude_x10",
    "unit_swap",
    "comparator_flip",
    "cross_section",
    "wrong_row",
)
CONTROL_TYPES: tuple[str, ...] = ("unmutated", "equivalent_rewrite")
MAX_DRAWS = 20
DIGIT_TOKEN = re.compile(DIGITS)
# Comparator flips: phrase -> replacement of a different class.
COMPARATOR_FLIPS: dict[str, str] = {
    "no more than": "no less than",
    "no less than": "no more than",
    "not more than": "not less than",
    "not less than": "not more than",
    "at least": "more than",
    "more than": "at least",
    "less than": "more than",
    "at or above": "below",
    "below": "at or above",
    "above": "below",
    "or less": "or more",
    "or more": "or less",
    "up to": "more than",
    "at most": "at least",
    "minimum": "maximum",
    "maximum": "minimum",
    "exceeds": "is below",
    "fewer than": "more than",
}
# Longest phrases first, so "no more than" is flipped as a whole and never as "no" + "more than".
FLIP_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(key) for key in sorted(COMPARATOR_FLIPS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Mutation:
    """One mutated (or control) sentence."""

    mutation_id: str
    mutation_type: str
    ref_id: str
    section_id: str
    original: str
    mutated: str
    target: str
    replacement: str
    expect_flag: bool

    def to_json(self) -> dict:
        """Plain dict for JSONL."""
        return asdict(self)


def digit_span(mention: NumberMention) -> tuple[int, int, str] | None:
    """Absolute (start, end, digits) of the digit part of a mention ("$2 million" -> the "2")."""
    match = DIGIT_TOKEN.search(mention.text)
    if match is None:
        return None
    return mention.start + match.start(), mention.start + match.end(), match.group(0)


def format_like(template: str, value: Fraction) -> str:
    """Write value with the same number of decimals and thousands separators as template."""
    decimals = len(template.split(".")[1]) if "." in template else 0
    use_commas = "," in template or (abs(value) >= 10_000 and "," not in template and len(template) > 4)
    number = float(value)
    text = f"{number:,.{decimals}f}" if use_commas else f"{number:.{decimals}f}"
    return text


def replace_span(text: str, start: int, end: int, replacement: str) -> str:
    """text with text[start:end] replaced."""
    return text[:start] + replacement + text[end:]


def occurs_in(section_text: str, sentence: str, start: int, end: int, replacement: str) -> bool:
    """True when the mutated number (with its unit, as re-extracted) states a quantity the section contains."""
    mutated = replace_span(sentence, start, end, replacement)
    new_mentions = [m for m in extract_numbers(mutated) if m.start <= start < m.end + 1 or m.start == start]
    if not new_mentions:
        return False
    section_mentions = extract_numbers(section_text)
    return any(same_quantity(new, old) for new in new_mentions for old in section_mentions)


def numeric_mentions(sentence: str) -> list[NumberMention]:
    """Mentions with a digit part that a mutation can rewrite."""
    return [mention for mention in extract_numbers(sentence) if digit_span(mention) is not None]


def mutate_value(
    ref: ReferenceSentence, section_text: str, rng: random.Random, magnitude: bool
) -> tuple[str, str, str] | None:
    """value_perturb (magnitude False) or magnitude_x10 (True): (mutated sentence, target, replacement)."""
    mentions = numeric_mentions(ref.text)
    if not mentions:
        return None
    for _ in range(MAX_DRAWS):
        mention = rng.choice(mentions)
        start, end, digits = digit_span(mention)  # type: ignore[misc]
        value = Fraction(digits.replace(",", ""))
        if magnitude:
            factor = Fraction(10) if rng.random() < 0.5 or value < 1 else Fraction(1, 10)
        else:
            change = Fraction(rng.randint(3, 30), 100)
            factor = 1 + change if rng.random() < 0.5 else 1 - change
        replacement = format_like(digits, value * factor)
        if Fraction(replacement.replace(",", "")) == value:
            continue
        if not occurs_in(section_text, ref.text, start, end, replacement):
            return replace_span(ref.text, start, end, replacement), digits, replacement
    return None


UNIT_SWAPS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\s?%"), " bps"),
    (re.compile(r"^\s?(?:percent|per cent)\b", re.IGNORECASE), " bps"),
    (re.compile(r"^\s?(?:basis points|bps)\b", re.IGNORECASE), "%"),
    (re.compile(r"^([\s-]?)months\b", re.IGNORECASE), r"\1years"),
    (re.compile(r"^([\s-]?)month\b", re.IGNORECASE), r"\1year"),
    (re.compile(r"^([\s-]?)years\b", re.IGNORECASE), r"\1months"),
    (re.compile(r"^([\s-]?)year\b", re.IGNORECASE), r"\1month"),
)


def mutate_unit(ref: ReferenceSentence, section_text: str, rng: random.Random) -> tuple[str, str, str] | None:
    """unit_swap: change the unit word or symbol right after a number, keeping the value."""
    candidates = []
    for mention in numeric_mentions(ref.text):
        _, digit_end, _ = digit_span(mention)  # type: ignore[misc]
        tail = ref.text[digit_end:]
        for pattern, replacement in UNIT_SWAPS:
            match = pattern.match(tail)
            if match:
                candidates.append((digit_end, digit_end + match.end(), match.group(0), match.expand(replacement)))
                break
    rng.shuffle(candidates)
    for start, end, target, replacement in candidates:
        mutated = replace_span(ref.text, start, end, replacement)
        changed = [m for m in extract_numbers(mutated) if m.start < end + len(replacement) and m.end > start - 12]
        section_mentions = extract_numbers(section_text)
        if changed and not any(same_quantity(new, old) for new in changed for old in section_mentions):
            return mutated, target, replacement
    return None


def mutate_comparator(ref: ReferenceSentence, rng: random.Random) -> tuple[str, str, str] | None:
    """comparator_flip: replace one comparator phrase that sits within 5 tokens of a number."""
    mentions = numeric_mentions(ref.text)
    candidates = []
    for match in FLIP_PATTERN.finditer(ref.text):
        near = any(len(ref.text[min(match.end(), m.start) : max(match.start(), m.end)].split()) <= 6 for m in mentions)
        if near:
            candidates.append(match)
    if not candidates:
        return None
    match = rng.choice(candidates)
    target = match.group(0)
    replacement = COMPARATOR_FLIPS[" ".join(target.lower().split())]
    if target[0].isupper():
        replacement = replacement[0].upper() + replacement[1:]
    return replace_span(ref.text, match.start(), match.end(), replacement), target, replacement


def mutate_cross_section(
    ref: ReferenceSentence, sections: dict[str, str], rng: random.Random
) -> tuple[str, str, str] | None:
    """cross_section: a same-unit value from another section that the cited section does not contain."""
    mentions = numeric_mentions(ref.text)
    if not mentions:
        return None
    cited_text = sections[ref.section_id]
    others = [(sid, text) for sid, text in sections.items() if sid != ref.section_id]
    for _ in range(MAX_DRAWS):
        mention = rng.choice(mentions)
        _, other_text = rng.choice(others)
        pool = [m for m in extract_numbers(other_text) if m.unit == mention.unit and digit_span(m) is not None]
        if not pool:
            continue
        donor = rng.choice(pool)
        start, end, digits = digit_span(mention)  # type: ignore[misc]
        replacement = digit_span(donor)[2]  # type: ignore[index]
        if replacement == digits:
            continue
        if not occurs_in(cited_text, ref.text, start, end, replacement):
            return replace_span(ref.text, start, end, replacement), digits, replacement
    return None


def mutate_wrong_row(
    ref: ReferenceSentence, table_rows: list[ReferenceSentence], rng: random.Random
) -> tuple[str, str, str] | None:
    """wrong_row: the value of another row in the same table column (table sentences only)."""
    if ref.source != "table":
        return None
    siblings = [r for r in table_rows if r.table == ref.table and r.column == ref.column and r.row != ref.row]
    own = numeric_mentions(ref.text)
    if not siblings or not own:
        return None
    rng.shuffle(siblings)
    target_mention = own[-1]
    start, end, digits = digit_span(target_mention)  # type: ignore[misc]
    for sibling in siblings:
        sibling_mentions = [m for m in numeric_mentions(sibling.text) if m.unit == target_mention.unit]
        if not sibling_mentions:
            continue
        replacement = digit_span(sibling_mentions[-1])[2]  # type: ignore[index]
        if replacement != digits:
            return replace_span(ref.text, start, end, replacement), digits, replacement
    return None


def equivalent_rewrite(ref: ReferenceSentence) -> tuple[str, str, str] | None:
    """The same quantity written another way, or None when no rule applies."""
    for mention in extract_numbers(ref.text):
        text = mention.text
        if mention.unit == "USD" and text.startswith("$") and mention.value % 1_000_000 == 0 and mention.value > 0:
            millions = int(mention.value / 1_000_000)
            return (
                replace_span(ref.text, mention.start, mention.end, f"${millions} million"),
                text,
                f"${millions} million",
            )
        range_end = re.search(r"(?:to|-|–)\s*$", ref.text[: mention.start]) is not None
        if mention.unit == "MONTHS" and mention.value % 12 == 0 and re.fullmatch(r"\d+ months", text) and not range_end:
            years = int(mention.value / 12)
            new = f"{years} years" if years != 1 else "1 year"
            return replace_span(ref.text, mention.start, mention.end, new), text, new
        if mention.unit == "PERCENT" and text.endswith("%"):
            new = text[:-1] + " percent"
            return replace_span(ref.text, mention.start, mention.end, new), text, new
        if mention.unit == "NONE" and re.fullmatch(r"\d+\.\d{2}", text):
            return replace_span(ref.text, mention.start, mention.end, text + "x"), text, text + "x"
    return None


def build_mutations(refs: list[ReferenceSentence], sections: dict[str, str], seed: int = 42) -> list[Mutation]:
    """Every mutation type and control that applies to each reference sentence, seeded."""
    rng = random.Random(seed)
    table_rows = [ref for ref in refs if ref.source == "table"]
    mutations: list[Mutation] = []

    def add(kind: str, ref: ReferenceSentence, result: tuple[str, str, str] | None, expect_flag: bool) -> None:
        if result is None:
            return
        mutated, target, replacement = result
        mutation_id = f"MUT-{len(mutations) + 1:05d}"
        mutations.append(
            Mutation(mutation_id, kind, ref.ref_id, ref.section_id, ref.text, mutated, target, replacement, expect_flag)
        )

    for ref in refs:
        section_text = sections[ref.section_id]
        add("unmutated", ref, (ref.text, "", ""), False)
        add("equivalent_rewrite", ref, equivalent_rewrite(ref), False)
        add("value_perturb", ref, mutate_value(ref, section_text, rng, magnitude=False), True)
        add("magnitude_x10", ref, mutate_value(ref, section_text, rng, magnitude=True), True)
        add("unit_swap", ref, mutate_unit(ref, section_text, rng), True)
        add("comparator_flip", ref, mutate_comparator(ref, rng), True)
        add("cross_section", ref, mutate_cross_section(ref, sections, rng), True)
        add("wrong_row", ref, mutate_wrong_row(ref, table_rows, rng), True)
    return mutations
