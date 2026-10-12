"""Table tests for number extraction, normalization, exemptions and unit compatibility (provenance/numbers.py)."""

from fractions import Fraction

import pytest

from uwcopilot.provenance.numbers import compatible, extract_numbers, same_quantity, words_to_value

# (text, expected [(value, unit), ...]) — values as strings parsed with Fraction.
EXTRACTION_CASES: list[tuple[str, list[tuple[str, str]]]] = [
    ("$2,000,000", [("2000000", "USD")]),
    ("$2 million", [("2000000", "USD")]),
    ("$2M", [("2000000", "USD")]),
    ("$1.2k", [("1200", "USD")]),
    ("USD 500,000", [("500000", "USD")]),
    ("500 dollars", [("500", "USD")]),
    ("$25,000 to $2,000,000", [("25000", "USD"), ("2000000", "USD")]),
    ("12%", [("12", "PERCENT")]),
    ("12 percent", [("12", "PERCENT")]),
    ("10.0% or less", [("10", "PERCENT")]),
    ("2.00 percentage points", [("2", "PERCENT")]),
    ("200 bps", [("200", "BPS")]),
    ("200 basis points", [("200", "BPS")]),
    ("a DSCR of 1.25", [("1.25", "NONE")]),
    ("1.25x coverage", [("1.25", "RATIO")]),
    ("1.35 times debt service", [("1.35", "RATIO")]),
    ("24 months", [("24", "MONTHS")]),
    ("12-month commitment", [("12", "MONTHS")]),
    ("2 years", [("2", "YEARS")]),
    ("5-year loss ratio", [("5", "YEARS")]),
    ("90 days", [("90", "DAYS")]),
    ("2 business days", [("2", "DAYS")]),
    ("90-day window", [("90", "DAYS")]),
    ("12 to 84 months", [("12", "MONTHS"), ("84", "MONTHS")]),
    ("3 or more months", [("3", "MONTHS")]),
    ("3 claims", [("3", "COUNT")]),
    ("6 non-sufficient-funds events", [("6", "COUNT")]),
    ("two principals", [("2", "COUNT")]),
    ("twelve months", [("12", "MONTHS")]),
    ("two hundred fifty thousand dollars", [("250000", "USD")]),
    ("one third", [("1/3", "NONE")]),
    ("a quarter", [("1/4", "NONE")]),
    ("half of the claims", [("1/2", "NONE")]),
    ("one million", [("1000000", "NONE")]),
    ("ninety-nine", [("99", "NONE")]),
]

EXEMPT_CASES: list[str] = [
    "See NBC-4.2 and HM-11.2.",
    "Trigger R-NBC-04 fires.",
    "Effective 2026-07-01.",
    "Statement month 2026-03.",
    "Dated 07/01/2026.",
    "On July 1, 2026 the guideline changed.",
    "Since 2019.",
    "[NBC-4.2]",
    "[S1][S3]",
    "{{claim:dscr_2}}",
    "1. The applicant is organized.",
    "no one, someone, anyone, one of the owners",
]


@pytest.mark.parametrize(("text", "expected"), EXTRACTION_CASES)
def test_extraction_table(text: str, expected: list[tuple[str, str]]) -> None:
    """Every documented format is found with the right value and unit."""
    found = [(mention.value, mention.unit) for mention in extract_numbers(text)]
    assert found == [(Fraction(value), unit) for value, unit in expected]


@pytest.mark.parametrize("text", EXEMPT_CASES)
def test_exemptions(text: str) -> None:
    """Section ids, rule ids, dates, years, citations, slots, list markers and non-quantity 'one' are skipped."""
    assert extract_numbers(text) == []


def test_offsets_point_into_the_original_text() -> None:
    """Mentions carry spans of the original text, so a mutation can replace exactly that span."""
    text = "Revenue of $1,180,000 [NBC-3.2] and a gap of 27.3%."
    for mention in extract_numbers(text):
        assert text[mention.start : mention.end] == mention.text


def test_words_to_value_handles_scales_and_fractions() -> None:
    """The number-word parser handles hundreds, thousands, millions and simple fractions."""
    assert words_to_value(["one", "hundred", "and", "five"]) == 105
    assert words_to_value(["three", "million", "two", "hundred", "thousand"]) == 3_200_000
    assert words_to_value(["two", "thirds"]) == Fraction(2, 3)
    assert words_to_value(["a"]) is None


def test_unit_compatibility_table() -> None:
    """The documented compatibility rules."""
    assert compatible("NONE", "RATIO") and compatible("NONE", "COUNT")
    assert not compatible("NONE", "PERCENT") and not compatible("NONE", "USD")
    assert compatible("YEARS", "MONTHS") and compatible("PERCENT", "BPS")
    assert not compatible("DAYS", "MONTHS") and not compatible("USD", "PERCENT")


def test_same_quantity_converts_units() -> None:
    """2 years = 24 months; 2% = 200 bps; 2% != 2 bps; $1 million = $1,000,000."""

    def first(text: str):
        return extract_numbers(text)[0]

    assert same_quantity(first("2 years"), first("24 months"))
    assert same_quantity(first("2%"), first("200 bps"))
    assert not same_quantity(first("2%"), first("2 bps"))
    assert same_quantity(first("$1 million"), first("$1,000,000"))
    assert same_quantity(first("1.25"), first("1.25x"))
    assert not same_quantity(first("1.25"), first("1.25%"))
