"""Tests for uwcopilot.kyc.screening: the screen() entry point and the allow-listed screen_entity tool."""

import json
from pathlib import Path

import pytest
from kyc_fixtures import tiny_matcher
from pydantic import ValidationError

from uwcopilot.kyc import screening
from uwcopilot.kyc.screening import (
    ScreeningNotConfigured,
    config_from_operating_point,
    load_selected_operating_point,
    screen,
    set_default_matcher,
)
from uwcopilot.tools.registry import call_tool
from uwcopilot.tools.schemas import ScreeningResult


@pytest.fixture
def installed_matcher():
    """Install the tiny matcher as the default for one test, then reset it."""
    set_default_matcher(tiny_matcher())
    yield
    set_default_matcher(None)


def test_screen_accepts_a_dict_and_rejects_unknown_fields() -> None:
    matcher = tiny_matcher()
    result = screen({"name": "Mohammed Ali Hassan", "entity_type": "person"}, matcher)
    assert result.status == "potential_match"
    with pytest.raises(ValidationError):
        screen({"name": "X", "entity_type": "person", "auto_clear": True}, matcher)


def test_registry_tool_uses_the_installed_matcher(installed_matcher) -> None:
    result = call_tool("screen_entity", {"name": "Ivan Sidorenko", "entity_type": "person", "country": "RU"})
    assert isinstance(result, ScreeningResult)
    assert result.status == "potential_match"
    assert result.candidates[0].list_entry_id == "OS-NK-test-1"
    assert result.requires_human_review is True
    clear = call_tool("screen_entity", {"name": "Penelope Quarterfield", "entity_type": "person"})
    assert clear.status == "clear" and clear.list_snapshots


def test_missing_results_file_refuses_instead_of_guessing(tmp_path: Path) -> None:
    with pytest.raises(ScreeningNotConfigured):
        load_selected_operating_point(tmp_path / "missing.json")
    empty = tmp_path / "kyc_metrics.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(ScreeningNotConfigured):
        load_selected_operating_point(empty)


def test_operating_point_becomes_matcher_config(tmp_path: Path) -> None:
    path = tmp_path / "kyc_metrics.json"
    path.write_text(json.dumps({"selected_operating_point": {"threshold": 0.91, "reranker_weight": 0.2}}))
    config = config_from_operating_point(load_selected_operating_point(path))
    assert config.threshold == 0.91 and config.reranker_weight == 0.2


def test_default_matcher_is_built_once(monkeypatch: pytest.MonkeyPatch) -> None:
    built = []

    def fake_build() -> object:
        """Count builds instead of loading data/raw."""
        built.append(1)
        return tiny_matcher()

    monkeypatch.setattr(screening, "build_default_matcher", fake_build)
    set_default_matcher(None)
    try:
        first = screening.default_matcher()
        second = screening.default_matcher()
    finally:
        set_default_matcher(None)
    assert first is second and len(built) == 1
