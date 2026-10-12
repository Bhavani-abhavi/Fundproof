"""One test per validator code of the numberless memo grammar, plus the documented exceptions."""

import pytest

from uwcopilot.provenance.memo_grammar import VIOLATION_CODES, validate

CLAIMS = {"verified_annual_revenue", "dscr", "revenue_gap_pct"}
ASSERTIONS = {"R-NBC-04", "R-NBC-02"}
SECTIONS = {"NBC-3.2", "NBC-3.4", "NBC-4.2"}


def codes(text: str, required: list[str] | None = None) -> set[str]:
    """The violation codes the validator reports for text."""
    return {violation.code for violation in validate(text, CLAIMS, ASSERTIONS, required or [], SECTIONS)}


@pytest.mark.parametrize(
    ("text", "required", "expected"),
    [
        ("Revenue is 1.2 million [NBC-3.2].", [], "V1_DIGIT"),
        ("Revenue is about twelve months old [NBC-3.2].", [], "V2_NUMBER_WORD"),
        ("Revenue in $ terms is {{claim:verified_annual_revenue}} [NBC-3.2].", [], "V3_SYMBOL"),
        ("Coverage is {{claim:global_dscr}} [NBC-4.2].", [], "V4_UNKNOWN_SLOT"),
        ("Coverage of {{claim:dscr}} is below the minimum [NBC-4.2].", [], "V5_UNSLOTTED_COMPARATIVE"),
        ("Coverage is {{claim:dscr}} [NBC-4.2].", ["assert:R-NBC-04"], "V6_OMISSION"),
        ("The loan is approved [NBC-4.2].", [], "V7_DISPOSITION_LANGUAGE"),
        ("Coverage is {claim:dscr} [NBC-4.2].", [], "V8_MALFORMED_SLOT"),
        ("Coverage is {{claim:dscr}} [NBC-7.1].", [], "V9_UNKNOWN_CITATION"),
    ],
)
def test_each_code(text: str, required: list[str], expected: str) -> None:
    """Each violation code fires on a sentence built to break exactly that rule."""
    assert expected in codes(text, required)


def test_every_code_is_covered_by_a_test() -> None:
    """The parametrized cases above cover all nine codes."""
    assert len(VIOLATION_CODES) == 9


def test_valid_memo_passes() -> None:
    """Prose with slots and allowed citations is valid, including the required references."""
    text = (
        "Verified revenue is {{claim:verified_annual_revenue}} [NBC-3.2]. The decision DSCR of {{claim:dscr}} "
        "{{assert:R-NBC-04}} [NBC-4.2]. The gap is {{claim:revenue_gap_pct}} and it {{assert:R-NBC-02}} [NBC-3.4]."
    )
    assert codes(text, ["assert:R-NBC-04", "claim:revenue_gap_pct"]) == set()


def test_number_word_exceptions() -> None:
    """'no one', 'one of', 'someone' are not quantities."""
    assert codes("No one has reviewed it; one of the owners signed; someone will call [NBC-3.2].") == set()


def test_comparative_is_allowed_with_an_assert_slot() -> None:
    """A comparative word is fine in a sentence that carries the rule's assert slot."""
    assert codes("The DSCR {{assert:R-NBC-04}}, which is below what a lender wants [NBC-4.2].") == set()


def test_digits_inside_slots_and_citations_are_not_violations() -> None:
    """Digits in a slot id or a section id are part of the token, not the prose."""
    assert codes("The gap {{assert:R-NBC-02}} [NBC-3.4].") == set()
