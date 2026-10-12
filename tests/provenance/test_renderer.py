"""Renderer formats, footnotes, and the round-trip property: every rendered number is a ledger value."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from uwcopilot.provenance.renderer import (
    SimpleAssertion,
    SimpleClaim,
    SimpleLedger,
    UnknownSlotError,
    format_value,
    render,
    round_trip_audit,
)


@pytest.mark.parametrize(
    ("value", "display", "expected"),
    [
        (1180000.4, "usd0", "$1,180,000"),
        (1234.5, "usd2", "$1,234.50"),
        (1.0549, "ratio2", "1.05"),
        (1.3, "ratio2x", "1.30x"),
        (27.31, "pct1", "27.3%"),
        (11.6, "int", "12"),
        (24, "months", "24 months"),
        (None, "usd0", "not available"),
        ("Term loan", "text", "Term loan"),
    ],
)
def test_format_value(value: object, display: str, expected: str) -> None:
    """Each display format prints the value the documented way."""
    assert format_value(value, display) == expected


def ledger() -> SimpleLedger:
    """A two-claim, one-assertion ledger."""
    return SimpleLedger(
        claims={
            "dscr": SimpleClaim("dscr", "Decision DSCR", 1.0549, "ratio2", provenance="calculation CALC-dscr"),
            "gap": SimpleClaim("gap", "Revenue gap", 27.31, "pct1", status="CONFLICTING"),
        },
        assertions={"R-NBC-04": SimpleAssertion("R-NBC-04", False, "meets the minimum", "is below the minimum")},
    )


def test_render_fills_slots_and_footnotes_each_claim_once() -> None:
    """Claims render with a footnote marker; a repeated claim reuses its footnote; status suffixes appear."""
    memo = render("DSCR {{claim:dscr}} {{assert:R-NBC-04}}; again {{claim:dscr}}; gap {{claim:gap}}.", ledger())
    assert memo.body == ("DSCR 1.05[^c1] is below the minimum; again 1.05[^c1]; gap 27.3% (conflicting sources)[^c2].")
    assert memo.footnotes == ["[^c1]: Decision DSCR: calculation CALC-dscr", "[^c2]: Revenue gap: calculation"]
    assert round_trip_audit(memo, ledger()) == []


def test_unknown_slot_raises() -> None:
    """Rendering never invents a value for an id the ledger lacks."""
    with pytest.raises(UnknownSlotError):
        render("{{claim:missing}}", ledger())


def test_round_trip_catches_a_typed_number() -> None:
    """A number typed into the prose (not from a slot) is reported by the audit."""
    memo = render("DSCR {{claim:dscr}} but the text also says 1.30.", ledger())
    assert round_trip_audit(memo, ledger()) == ["1.30"]


CLAIM_VALUES = st.floats(min_value=0.01, max_value=50_000_000, allow_nan=False, allow_infinity=False)
DISPLAYS = st.sampled_from(["usd0", "usd2", "ratio2", "ratio2x", "pct1", "int"])
WORDS = st.sampled_from(["The", "file", "shows", "revenue", "coverage", "and", "the", "reviewer", "notes"])


@settings(max_examples=200, deadline=None)
@given(values=st.lists(st.tuples(CLAIM_VALUES, DISPLAYS), min_size=1, max_size=5), words=st.lists(WORDS))
def test_property_every_rendered_number_is_traceable(values: list[tuple[float, str]], words: list[str]) -> None:
    """Random valid slot memos render with every number equal to a ledger value in its display format."""
    claims = {
        f"c{index}": SimpleClaim(f"c{index}", f"Claim {index}", value, display)
        for index, (value, display) in enumerate(values)
    }
    text = " ".join(words) + " " + " and ".join(f"{{{{claim:c{index}}}}}" for index in range(len(values))) + "."
    memo = render(text, SimpleLedger(claims=claims))
    assert round_trip_audit(memo, SimpleLedger(claims=claims)) == []
