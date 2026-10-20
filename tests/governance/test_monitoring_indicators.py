"""Indicators read from results files: values, 'not measured', 'not available', and no guessing (SYNTHETIC)."""

import json
from pathlib import Path
from typing import Any

import pytest

from uwcopilot.governance.monitoring_config import LevelRule, load_monitoring_config
from uwcopilot.governance.monitoring_indicators import (
    agent_indicators,
    as_number,
    extraction_indicators,
    find_key,
    override_indicators,
    read_indicators,
)

NOT_MEASURED_TEXT = "not measured: no LLM provider was used"


@pytest.fixture(scope="module")
def rules() -> dict[str, LevelRule]:
    """The committed indicator rules."""
    return load_monitoring_config().thresholds.indicators


def docintel(supported: float = 0.9984, ocr_pages: int = 159, pages: int = 949) -> dict[str, Any]:
    """The parts of a docintel_metrics.json the monitor reads (SYNTHETIC values)."""
    return {
        "rules": {
            "citations": {
                "supported_all_values": {
                    "value": supported,
                    "ci_low": supported - 0.001,
                    "ci_high": min(1.0, supported + 0.001),
                    "numerator": 36230,
                    "denominator": 36288,
                }
            },
            "latency": {"ocr_pages_total": ocr_pages},
            "page_classification": {"all": {"denominator": pages}},
        }
    }


def agent(escalated: float = 1.0) -> dict[str, Any]:
    """The parts of an agent_metrics.json the monitor reads (SYNTHETIC values)."""
    return {
        "deterministic_path": {
            "cases": 65,
            "escalation": {"all_required_escalations": {"value": escalated, "ci_low": 1.0, "ci_high": 1.0, "n": 60}},
        },
        "llm_dependent": {"unsupported_claim_rate": NOT_MEASURED_TEXT},
    }


def by_name(indicators: list[Any]) -> dict[str, Any]:
    """Indicators keyed by name."""
    return {indicator.name: indicator for indicator in indicators}


def test_extraction_indicators_read_citation_support_and_ocr_share(rules: dict[str, LevelRule]) -> None:
    found = by_name(extraction_indicators(docintel(), rules))
    citation = found["citation_support_rate"]
    assert (citation.status, citation.value, citation.level) == ("measured", 0.9984, "ok")
    assert citation.interval == pytest.approx((0.9974, 0.9994))
    assert citation.detail["citation_failures"] == 58
    ocr = found["ocr_page_share"]
    assert ocr.value == pytest.approx(159 / 949) and ocr.detail == {"ocr_pages": 159, "pages": 949}
    assert found["low_confidence_field_share"].status == "not available"


def test_a_low_citation_support_and_a_high_ocr_share_alert(rules: dict[str, LevelRule]) -> None:
    found = by_name(extraction_indicators(docintel(supported=0.95, ocr_pages=500, pages=949), rules))
    assert found["citation_support_rate"].level == "alert"
    assert found["ocr_page_share"].level == "alert"
    check = found["citation_support_rate"].as_check()
    assert (check.check, check.subject, check.level) == ("indicator", "citation_support_rate", "alert")


def test_missing_files_make_every_indicator_not_available(rules: dict[str, LevelRule]) -> None:
    indicators = extraction_indicators(None, rules) + agent_indicators(None, rules)
    assert {indicator.status for indicator in indicators} == {"not available"}
    assert all("not present" in indicator.note for indicator in indicators)
    with pytest.raises(ValueError):
        indicators[0].as_check()


def test_agent_indicators_keep_not_measured_as_not_measured(rules: dict[str, LevelRule]) -> None:
    found = by_name(agent_indicators(agent(), rules))
    assert found["required_escalations_escalated"].level == "ok"
    share = found["escalation_share"]
    assert share.value == pytest.approx(60 / 65) and share.level is None and share.rule is None
    claims = found["unsupported_claim_rate"]
    assert claims.status == "not measured" and claims.value is None and claims.note == NOT_MEASURED_TEXT
    assert by_name(agent_indicators(agent(escalated=0.98), rules))["required_escalations_escalated"].level == "alert"


def test_override_rate_is_not_available_without_a_feedback_section(rules: dict[str, LevelRule]) -> None:
    for governance in (None, {}, {"monitoring": {"results": {}}}):
        (indicator,) = override_indicators(governance, rules)
        assert indicator.status == "not available" and indicator.value is None


def test_override_rate_and_reasons_are_read_wherever_the_feedback_section_puts_them(
    rules: dict[str, LevelRule],
) -> None:
    governance = {
        "feedback": {
            "results": {
                "data_label": "SIMULATED",
                "overrides": {
                    "override_rate": {"estimate": 0.24, "ci_low": 0.18, "ci_high": 0.30},
                    "override_reasons": {"missing_document": 7, "policy_exception": 3},
                },
            },
            "metadata": {"git_commit": "abc123"},
        }
    }
    (indicator,) = override_indicators(governance, rules)
    assert (indicator.value, indicator.level, indicator.interval) == (0.24, "alert", (0.18, 0.30))
    assert indicator.detail["override_reasons"] == {"missing_document": 7, "policy_exception": 3}
    assert indicator.detail["feedback_section_git_commit"] == "abc123"


def test_override_rate_without_a_number_is_never_guessed(rules: dict[str, LevelRule]) -> None:
    text_only = {"feedback": {"results": {"override_rate": "high"}}}
    assert override_indicators(text_only, rules)[0].status == "not available"
    placeholder = {"feedback": {"results": {"override_rate": {"status": "not measured", "reason": "no reviews"}}}}
    (indicator,) = override_indicators(placeholder, rules)
    assert (indicator.status, indicator.note) == ("not measured", "no reviews")


def test_helpers_reject_non_numbers() -> None:
    assert as_number(True) is None and as_number(float("nan")) is None and as_number("0.5") is None
    assert as_number({"value": 0.5}) == 0.5 and as_number(3) == 3.0
    assert find_key({"a": [{"b": {"override_rate": 0.1}}]}, "override_rate") == 0.1
    assert find_key({"a": 1}, "override_rate") is None


def test_read_indicators_hashes_each_source_file(tmp_path: Path, rules: dict[str, LevelRule]) -> None:
    (tmp_path / "docintel_metrics.json").write_text(json.dumps(docintel()), encoding="utf-8")
    indicators, sources = read_indicators(tmp_path, rules)
    assert len(sources["results/docintel_metrics.json"]["sha256"]) == 64
    assert sources["results/agent_metrics.json"]["sha256"] is None
    statuses = {indicator.name: indicator.status for indicator in indicators}
    assert statuses["citation_support_rate"] == "measured" and statuses["override_rate"] == "not available"
