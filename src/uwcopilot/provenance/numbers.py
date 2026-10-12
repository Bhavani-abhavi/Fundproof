"""Find every number in a piece of text and normalize it to (value, unit).

This is step 1 of provenance-locked generation (docs/innovations.md, SIG): before a checker can ask "where did this
number come from?", it has to find the number, whatever way it was written:

- currency: "$2,000,000", "$2 million", "$2M", "$1.2k", "USD 500,000", "2,000 dollars";
- percentages: "12%", "12 percent", "2.00 percentage points";
- basis points: "200 bps", "200 basis points";
- ratios: "1.25", "1.25x", "1.25 times";
- durations: "24 months", "12-month", "2 years", "90 days";
- counts: "3 claims", "two principals";
- number words: zero to ninety-nine, hundred, thousand, million, billion, and the fractions half, quarter, third.

Each mention becomes a NumberMention with an exact value (a Fraction, so "one third" stays exact) and a unit:
USD, PERCENT, BPS, RATIO, MONTHS, YEARS, DAYS, COUNT or NONE (a bare number such as "1.25").

Some digits are not quantities and are skipped (exempt):
- section ids and rule ids: "NBC-4.2", "HM-11.2", "R-NBC-04";
- anything inside square brackets (citations such as "[NBC-4.2]") or slot braces ("{{claim:dscr}}");
- dates: "2026-07-01", "07/01/2026", "2026-03", "July 1, 2026", and four-digit years 1900-2099 used as years;
- list markers at the start of a line or sentence: "1.", "2)", "(3)".

same_quantity() decides whether two mentions state the same quantity, using a documented unit-compatibility table:
- equal units compare values directly;
- YEARS and MONTHS convert (1 year = 12 months);
- PERCENT and BPS convert (1 percent = 100 bps), so "2 bps" never equals "2%";
- NONE (a bare number) matches RATIO and COUNT with the same value;
- DAYS do not convert to months or years (months have different lengths), and USD only matches USD.
"""

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

Unit = Literal["USD", "PERCENT", "BPS", "RATIO", "MONTHS", "YEARS", "DAYS", "COUNT", "NONE"]
UNITS: tuple[str, ...] = ("USD", "PERCENT", "BPS", "RATIO", "MONTHS", "YEARS", "DAYS", "COUNT", "NONE")


@dataclass(frozen=True)
class NumberMention:
    """One number found in text: the exact span, its normalized value and its unit."""

    text: str
    start: int
    end: int
    value: Fraction
    unit: Unit

    def as_decimal_string(self, places: int = 4) -> str:
        """The value as a plain decimal string (for reports and JSON)."""
        return f"{float(self.value):.{places}f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------------------------------------------
# Exemptions: spans whose digits are not quantities. They are blanked out (same length) before extraction, so the
# offsets of the remaining mentions still point into the original text.
# --------------------------------------------------------------------------------------------------------------

MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
EXEMPT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[[^\[\]]*\]"),  # citation brackets
    re.compile(r"\{\{[^{}]*\}\}"),  # slot tokens
    re.compile(r"\[\^[^\]]*\]"),  # footnote markers
    re.compile(r"\b(?:R-)?[A-Z]{1,5}-\d+(?:\.\d+)?\b"),  # section ids and rule ids
    re.compile(r"\b(?:19|20)\d{2}-\d{2}(?:-\d{2})?\b"),  # ISO dates and year-months
    re.compile(r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b"),  # US dates
    re.compile(rf"\b(?:{MONTH_NAMES})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}\b"),  # July 1, 2026
    re.compile(rf"\b\d{{1,2}}\s+(?:{MONTH_NAMES})\.?\s+(?:19|20)\d{{2}}\b"),  # 1 July 2026
    re.compile(rf"\b(?:{MONTH_NAMES})\.?\s+(?:19|20)\d{{2}}\b"),  # July 2026
    re.compile(r"(?:^|(?<=[.!?:]\s))\s*(?:\(\d{1,2}\)|\d{1,2}[.)])(?=\s)", re.MULTILINE),  # list markers
)
# A four-digit year 1900-2099 that is not money, not followed by a unit word and not part of a bigger number.
BARE_YEAR = re.compile(
    r"(?<![\w$.,])(?:19|20)\d{2}(?![\w,%])(?!\.\d)"
    r"(?!\s*(?:dollars|percent|bps|basis|months?|years?|days?|x\b|times\b))"
)


def mask_exemptions(text: str) -> str:
    """Replace every exempt span with spaces of the same length."""
    masked = text
    for pattern in (*EXEMPT_PATTERNS, BARE_YEAR):
        masked = pattern.sub(lambda match: " " * len(match.group(0)), masked)
    return masked


# --------------------------------------------------------------------------------------------------------------
# Numeric tokens
# --------------------------------------------------------------------------------------------------------------

DIGITS = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"
MAGNITUDE_WORDS: dict[str, int] = {
    "k": 1_000,
    "thousand": 1_000,
    "m": 1_000_000,
    "mm": 1_000_000,
    "mn": 1_000_000,
    "million": 1_000_000,
    "b": 1_000_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
}
NUMERIC = re.compile(
    rf"(?P<currency>\$|USD\s?|US\$)?(?P<number>{DIGITS})"
    r"(?:\s?(?P<magnitude>k|K|thousand|M|MM|mn|million|B|bn|billion)\b)?"
    r"(?P<suffix>\s?%|x\b|×)?"
)

# Unit words that may follow a number (a hyphen is allowed: "12-month"). Order matters: longer phrases first.
UNIT_WORDS: tuple[tuple[re.Pattern[str], Unit], ...] = (
    (re.compile(r"^[\s-]*(?:percentage\s+points?|percent|per\s+cent|pct)\b", re.IGNORECASE), "PERCENT"),
    (re.compile(r"^[\s-]*(?:basis\s+points?|bps|bp)\b", re.IGNORECASE), "BPS"),
    (re.compile(r"^[\s-]*(?:business\s+days?|calendar\s+days?|days?)\b", re.IGNORECASE), "DAYS"),
    (re.compile(r"^[\s-]*(?:calendar\s+months?|months?|mos?)\b", re.IGNORECASE), "MONTHS"),
    (re.compile(r"^[\s-]*(?:years?|yrs?)\b", re.IGNORECASE), "YEARS"),
    (re.compile(r"^[\s-]*(?:dollars?|usd)\b", re.IGNORECASE), "USD"),
    (re.compile(r"^\s*times\b", re.IGNORECASE), "RATIO"),
)
# A following word that makes an integer a count ("3 claims"). Anything that is a unit or a connector is excluded.
COUNT_NOUN = re.compile(r"^\s+(?!(?:or|and|to|of|in|on|per|the|a|an|is|are|was|were)\b)[A-Za-z][A-Za-z-]*")
RANGE_CONNECTOR = re.compile(r"^\s*(?:to|-|–|—|through)\s*$", re.IGNORECASE)
# "and" joins a range only after "between": "between 12 and 84 months". "1.00 and $2" is two separate quantities.
BETWEEN_AND = re.compile(r"^\s*and\s*$", re.IGNORECASE)
BETWEEN_BEFORE = re.compile(r"\bbetween\s*$", re.IGNORECASE)

# --------------------------------------------------------------------------------------------------------------
# Number words
# --------------------------------------------------------------------------------------------------------------

UNITS_WORDS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS_WORDS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
SCALE_WORDS: dict[str, int] = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
FRACTION_WORDS: dict[str, Fraction] = {
    "half": Fraction(1, 2),
    "halves": Fraction(1, 2),
    "quarter": Fraction(1, 4),
    "quarters": Fraction(1, 4),
    "third": Fraction(1, 3),
    "thirds": Fraction(1, 3),
}
NUMBER_WORD = "|".join([*UNITS_WORDS, *TENS_WORDS, *SCALE_WORDS, *FRACTION_WORDS, "a", "an", "and"])
WORD_RUN = re.compile(rf"\b(?:{NUMBER_WORD})(?:[\s-]+(?:{NUMBER_WORD}))*\b", re.IGNORECASE)
# Phrases where "one" is not a quantity.
NON_QUANTITY_ONE = re.compile(r"\b(?:no one|someone|anyone|everyone|one another|one of|each one|any one)\b", re.I)


def words_to_value(words: list[str]) -> Fraction | None:
    """Parse a run of number words ("two hundred fifty thousand", "one third", "a quarter") into a value.

    Returns None when the run holds no quantity word (for example only "a" or "and").
    """
    tokens = [word.lower() for word in words if word.lower() not in ("and",)]
    if not tokens or all(token in ("a", "an") for token in tokens):
        return None
    total = 0
    current = 0
    seen_number = False
    fraction: Fraction | None = None
    for token in tokens:
        if token in ("a", "an"):
            current = max(current, 1) if not seen_number else current
            continue
        if token in UNITS_WORDS:
            current += UNITS_WORDS[token]
            seen_number = True
        elif token in TENS_WORDS:
            current += TENS_WORDS[token]
            seen_number = True
        elif token == "hundred":
            current = max(current, 1) * 100
            seen_number = True
        elif token in ("thousand", "million", "billion"):
            total += max(current, 1) * SCALE_WORDS[token]
            current = 0
            seen_number = True
        elif token in FRACTION_WORDS:
            numerator = max(current, 1)
            fraction = numerator * FRACTION_WORDS[token]
            current = 0
            seen_number = True
    if not seen_number:
        return None
    if fraction is not None:
        return Fraction(total) + fraction
    return Fraction(total + current)


# --------------------------------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------------------------------


def parse_digits(raw: str) -> Fraction:
    """'1,234.50' -> Fraction(246901, 200). Commas are thousands separators."""
    return Fraction(raw.replace(",", ""))


# "3 or more months", "6 or fewer days": a comparator phrase may sit between a number and its unit.
INNER_COMPARATOR = re.compile(r"^\s+or\s+(?:more|less|fewer|greater|longer|shorter|higher|lower)\b", re.IGNORECASE)


def unit_after(text: str, end: int) -> tuple[Unit | None, int]:
    """The unit word right after position end (an "or more"-style phrase may come first), and where it ends."""
    skip = INNER_COMPARATOR.match(text[end : end + 40])
    offset = skip.end() if skip else 0
    tail = text[end + offset : end + offset + 40]
    for pattern, unit in UNIT_WORDS:
        match = pattern.match(tail)
        if match:
            return unit, end + offset + match.end()
    return None, end


def unit_for(match: re.Match[str], text: str) -> tuple[Unit, int]:
    """Decide the unit of a numeric match from its currency sign, suffix and the word that follows it."""
    end = match.end()
    if match.group("currency"):
        unit_word, word_end = unit_after(text, end)
        return "USD", word_end if unit_word == "USD" else end
    suffix = (match.group("suffix") or "").strip()
    if suffix == "%":
        return "PERCENT", end
    if suffix in ("x", "×"):
        return "RATIO", end
    unit_word, word_end = unit_after(text, end)
    if unit_word is not None:
        return unit_word, word_end
    if "." not in match.group("number") and COUNT_NOUN.match(text[end : end + 40]):
        return "COUNT", end
    return "NONE", end


def digit_mentions(original: str, masked: str) -> list[NumberMention]:
    """Numbers written with digits (in the masked text, reported with original spans)."""
    mentions = []
    for match in NUMERIC.finditer(masked):
        raw_number = match.group("number")
        if raw_number in (".",) or not any(character.isdigit() for character in raw_number):
            continue
        value = parse_digits(raw_number)
        magnitude = match.group("magnitude")
        if magnitude:
            value *= MAGNITUDE_WORDS[magnitude.lower()]
        unit, end = unit_for(match, masked)
        mentions.append(NumberMention(original[match.start() : end], match.start(), end, value, unit))
    return mentions


def word_mentions(original: str, masked: str, taken: list[tuple[int, int]]) -> list[NumberMention]:
    """Numbers written in words, skipping spans already covered by digit mentions and non-quantity 'one'."""
    blocked = [(match.start(), match.end()) for match in NON_QUANTITY_ONE.finditer(masked)]
    mentions = []
    for match in WORD_RUN.finditer(masked):
        start, end = match.start(), match.end()
        if any(start < b_end and end > b_start for b_start, b_end in [*taken, *blocked]):
            continue
        words = re.split(r"[\s-]+", match.group(0))
        value = words_to_value(words)
        if value is None:
            continue
        unit_word, unit_end = unit_after(masked, end)
        unit: Unit = unit_word if unit_word is not None else "NONE"
        if unit == "NONE" and COUNT_NOUN.match(masked[end : end + 40]) and value.denominator == 1:
            unit = "COUNT"
        mentions.append(NumberMention(original[start:unit_end], start, unit_end, value, unit))
    return mentions


def propagate_range_units(text: str, mentions: list[NumberMention]) -> list[NumberMention]:
    """'12 to 84 months', 'between 12 and 84 months': the first number of a range takes the unit of the second."""
    result = list(mentions)
    for index in range(len(result) - 1):
        first, second = result[index], result[index + 1]
        if first.unit in ("NONE", "COUNT") and second.unit not in ("NONE", "COUNT"):
            connector = text[first.end : second.start]
            is_between = BETWEEN_AND.match(connector) and BETWEEN_BEFORE.search(
                text[max(0, first.start - 12) : first.start]
            )
            if RANGE_CONNECTOR.match(connector) or is_between:
                result[index] = NumberMention(first.text, first.start, first.end, first.value, second.unit)
    return result


def extract_numbers(text: str) -> list[NumberMention]:
    """Every quantity mentioned in text, in order of appearance (exempt spans skipped)."""
    masked = mask_exemptions(text)
    digits = digit_mentions(text, masked)
    words = word_mentions(text, masked, [(mention.start, mention.end) for mention in digits])
    mentions = sorted([*digits, *words], key=lambda mention: mention.start)
    return propagate_range_units(text, mentions)


# --------------------------------------------------------------------------------------------------------------
# Comparing quantities
# --------------------------------------------------------------------------------------------------------------

# Conversions to a base unit per family: (family, factor to the base unit).
UNIT_FAMILY: dict[str, tuple[str, Fraction]] = {
    "USD": ("money", Fraction(1)),
    "PERCENT": ("share", Fraction(1)),
    "BPS": ("share", Fraction(1, 100)),
    "MONTHS": ("time", Fraction(1)),
    "YEARS": ("time", Fraction(12)),
    "DAYS": ("days", Fraction(1)),
    "RATIO": ("plain", Fraction(1)),
    "COUNT": ("plain", Fraction(1)),
}


def compatible(unit_a: Unit, unit_b: Unit) -> bool:
    """True when two units can describe the same quantity (see the table in the module docstring)."""
    if unit_a == unit_b:
        return True
    if "NONE" in (unit_a, unit_b):
        other = unit_b if unit_a == "NONE" else unit_a
        return other in ("RATIO", "COUNT")
    return UNIT_FAMILY[unit_a][0] == UNIT_FAMILY[unit_b][0]


def base_value(mention: NumberMention) -> Fraction:
    """The value in its family's base unit (months for time, percent for shares)."""
    if mention.unit == "NONE":
        return mention.value
    return mention.value * UNIT_FAMILY[mention.unit][1]


def same_quantity(a: NumberMention, b: NumberMention) -> bool:
    """True when a and b state the same quantity: compatible units and equal values after conversion."""
    return compatible(a.unit, b.unit) and base_value(a) == base_value(b)
