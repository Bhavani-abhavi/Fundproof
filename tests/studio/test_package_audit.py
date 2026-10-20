"""The package audit: founder text is scoped to where it is printed, layout labels are bounded (SYNTHETIC)."""

import pytest
from package_helpers import package

from uwcopilot.evidence.ledger_models import Ledger
from uwcopilot.studio.package.audit import mask_layout, untraced_numbers

METRO, TYPE = "synthetic_metro", "car_wash"


def ledger_with_business_name(name: str) -> Ledger:
    """The golden package ledger with its first text claim renamed to a founder-typed name holding a digit."""
    ledger = package(METRO, TYPE).ledger
    claims = list(ledger.claims)
    index = next(position for position, item in enumerate(claims) if item.display == "text")
    claims[index] = claims[index].model_copy(update={"value": name})
    return ledger.model_copy(update={"claims": claims})


def test_digits_inside_a_printed_founder_name_are_data() -> None:
    ledger = ledger_with_business_name("Route 9 Wash")
    assert untraced_numbers("Route 9 Wash opens next spring.", ledger) == []
    assert untraced_numbers("Route 9\nWash opens next spring.", ledger) == []


def test_the_same_digits_elsewhere_must_still_trace() -> None:
    ledger = ledger_with_business_name("Unit 777777 Wash")
    assert untraced_numbers("Unit 777777 Wash opens next spring.", ledger) == []
    assert untraced_numbers("The owner adds $777,777 of equity.", ledger) != []


@pytest.mark.parametrize("label", ["Page 3/14", "P90", "Year 2", "year-2", "Month 60", "Loan year 25", "Rank 10"])
def test_layout_labels_are_blanked(label: str) -> None:
    assert not any(character.isdigit() for character in mask_layout(label))


@pytest.mark.parametrize(
    "figure", ["change 37%", "Year 47", "Month 61", "Rank 93", "Loan year 31", "Year 2.5", "rank 3 percent"]
)
def test_figures_that_look_like_labels_are_not_blanked(figure: str) -> None:
    assert any(character.isdigit() for character in mask_layout(figure))
