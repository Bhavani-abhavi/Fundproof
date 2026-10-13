"""targets.py: the three default targets, imported from the lender's constants, all editable (Studio S5)."""

import pytest
from pydantic import ValidationError

from uwcopilot.agent.policy_credit import DSCR_MINIMUM, REVENUE_STRESS_MIN_DSCR, REVENUE_STRESS_SHARE
from uwcopilot.studio.path_to_approval.targets import TARGET_NAMES, custom_target, default_target, describe


def test_default_targets_and_their_sources() -> None:
    target = default_target()
    assert target.dscr_min == DSCR_MINIMUM["term_loan"] == 1.25
    assert (target.p_dscr_min, target.p_stressed_min, target.p_cash_negative_max) == (0.80, 0.80, 0.10)
    assert target.stress_revenue_share == REVENUE_STRESS_SHARE
    assert target.stressed_dscr_min == REVENUE_STRESS_MIN_DSCR
    assert target.cash_months == 24 and target.year == 2
    assert "NSP-4.1" in target.sources["dscr_min"] and "NBC-4.2" in target.sources["dscr_min"]
    assert "SYNTHETIC" in target.sources["dscr_min"]
    assert "NSP-4.2" in target.sources["stressed_dscr_min"]
    assert "not a lender rule" in target.sources["p_dscr_min"]
    assert TARGET_NAMES == ("dscr_year2", "stressed_dscr_year2", "cash_negative")


def test_targets_are_editable_and_marked_as_edited() -> None:
    target = custom_target(p_stressed_min=0.6, dscr_min=1.35)
    assert target.p_stressed_min == 0.6 and target.dscr_min == 1.35
    assert target.sources["p_stressed_min"] == "edited by the user"
    assert target.sources["p_dscr_min"] == default_target().sources["p_dscr_min"]
    with pytest.raises(ValueError, match="cannot edit"):
        custom_target(sources=1)
    with pytest.raises(ValidationError):
        custom_target(p_dscr_min=1.5)


def test_describe_names_all_three_targets() -> None:
    lines = describe(default_target())
    assert len(lines) == 3
    assert "1.25" in lines[0] and "20% revenue stress" in lines[1] and "24 months" in lines[2]
