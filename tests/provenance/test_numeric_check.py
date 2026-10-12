"""Tests for the free-form numeric-provenance checker (provenance/numeric_check.py)."""

from uwcopilot.provenance.numbers import extract_numbers
from uwcopilot.provenance.numeric_check import (
    check_numbers,
    cited_ids,
    comparator_near,
    expand_tables,
    split_sentences,
)

SECTION = (
    "Every term loan above $100,000 is stress tested.\n\n"
    "- Revenue stress: reduce verified annual revenue by 20%, reduce NOI by the same dollar amount. A stressed DSCR\n"
    "  below 1.00 is a referral to a Senior Underwriter.\n"
    "- Rate stress: add 2.00 percentage points to the note rate.\n"
)
TABLE_SECTION = "| Product | Minimum decision DSCR |\n|---|---|\n| Term loan | 1.25 |\n| Revolving line | 1.35 |\n"
BANDS_SECTION = "| Revenue gap | Classification |\n|---|---|\n| 10.0% or less | Benign |\n| More than 10.0% | Refer |\n"
SOURCES = {"NBC-4.5": SECTION, "NBC-4.2": TABLE_SECTION, "NBC-3.4": BANDS_SECTION}


def statuses(text: str) -> list[str]:
    """The status of every number in text, against SOURCES."""
    return [item.status for item in check_numbers(text, SOURCES).mentions]


def test_traceable_numbers_pass() -> None:
    """A restated value with its citation passes, also with an equivalent unit."""
    assert statuses("Revenue stress cuts revenue by 20% [NBC-4.5].") == ["traceable"]
    assert statuses("Rate stress adds 200 basis points [NBC-4.5].") == ["traceable"]


def test_untraceable_number_is_flagged() -> None:
    """A value the cited section does not contain is UNTRACEABLE_NUMBER."""
    result = check_numbers("Revenue stress cuts revenue by 25% [NBC-4.5].", SOURCES)
    assert result.flagged and [item.status for item in result.mentions] == ["untraceable"]


def test_uncited_or_out_of_context_citation_is_untraceable() -> None:
    """No citation, or a citation whose text was not in context, cannot make a number traceable."""
    assert statuses("Revenue stress cuts revenue by 20%.") == ["untraceable"]
    assert statuses("Revenue stress cuts revenue by 20% [NBC-9.9].") == ["untraceable"]


def test_bps_is_not_percent_without_conversion() -> None:
    """'2 bps' is not '2.00 percentage points' (a unit swap)."""
    assert statuses("Rate stress adds 2 bps [NBC-4.5].") == ["untraceable"]


def test_comparator_mismatch() -> None:
    """The source pairs 1.00 with 'below' (LT); 'at least 1.00' is a different class."""
    assert statuses("A stressed DSCR of at least 1.00 is a referral [NBC-4.5].") == ["comparator_mismatch"]
    assert statuses("A stressed DSCR below 1.00 is a referral [NBC-4.5].") == ["traceable"]


def test_added_comparator_on_a_plain_value_is_a_mismatch() -> None:
    """The source states 20% with no comparator; 'at least 20%' asserts more than the source says."""
    result = check_numbers("Revenue stress cuts revenue by at least 20% [NBC-4.5].", SOURCES)
    assert [item.status for item in result.mentions] == ["comparator_mismatch"]
    assert "without a comparator" in result.mentions[0].detail


def test_table_header_binds_to_its_cells() -> None:
    """'Minimum decision DSCR' in the header pairs with 1.25, so 'minimum ... 1.25' is traceable."""
    assert statuses("The minimum DSCR for a term loan is 1.25 [NBC-4.2].") == ["traceable"]
    assert "The minimum decision dscr for Term loan is 1.25." in expand_tables(TABLE_SECTION)


def test_value_used_with_several_comparators_is_unverifiable() -> None:
    """NBC-3.4 pairs 10.0% with both 'or less' and 'more than': the comparator cannot be checked (counted)."""
    result = check_numbers("A gap of at least 10.0% is benign [NBC-3.4].", SOURCES)
    assert [item.status for item in result.mentions] == ["comparator_unverifiable"]
    assert not result.flagged


def test_comparator_window_is_five_tokens_and_stays_in_the_sentence() -> None:
    """Comparators farther than 5 tokens, or in another sentence, do not count."""
    near = "a DSCR of at least 1.25"
    assert comparator_near(near, extract_numbers(near)[0]) == "GE"
    far = "minimum required by the bank for the loan is 1.25"
    assert comparator_near(far, extract_numbers(far)[0]) is None


def test_citation_stays_with_its_sentence_and_lists_are_split() -> None:
    """'[NBC-4.2]' after a full stop belongs to that sentence; '[A, B]' cites both."""
    sentences = [text for _, text in split_sentences("DSCR is 1.25. [NBC-4.2] Next sentence here.")]
    assert "[NBC-4.2]" in sentences[0] and "Next" not in sentences[0]
    assert cited_ids("Text [NBC-4.2, NBC-4.5] and [S1].") == ("NBC-4.2", "NBC-4.5", "S1")


def test_soft_line_wraps_do_not_split_comparators() -> None:
    """A comparator broken across a markdown line wrap still binds to its value."""
    source = {"X-1.1": "businesses that derive more\n  than 10% of revenue are prohibited."}
    assert [m.status for m in check_numbers("More than 10% of revenue [X-1.1].", source).mentions] == ["traceable"]
