"""Tests for uwcopilot.tools.registry: only allow-listed tools run, with validated inputs and outputs."""

import pytest
from pydantic import ValidationError

from uwcopilot.tools import registry
from uwcopilot.tools.registry import (
    ALLOWED_TOOLS,
    ToolNotAllowed,
    ToolSpec,
    ToolUnavailable,
    call_tool,
    get_tool_spec,
    resolve_handler,
    tool_definitions,
)
from uwcopilot.tools.schemas import GuidelineQuery, GuidelineSearchResult, ScreeningRequest

SNAPSHOT = {"source": "OFAC", "file_name": "sdn.csv", "sha256": "b" * 64}


def fake_search(query: GuidelineQuery) -> dict:
    """A stand-in retriever that returns one fixed hit."""
    return {
        "query": query.model_dump(),
        "hits": [
            {
                "section_id": "NBC-4.2",
                "title": "DSCR minimums",
                "source_doc": "northwind_business_credit",
                "text": "Term loans need a DSCR of at least 1.25.",
                "score": 0.9,
                "rank": 1,
            }
        ],
    }


def fake_screen(request: ScreeningRequest) -> dict:
    """A stand-in screener that never finds anything."""
    return {"request": request.model_dump(), "candidates": [], "status": "clear", "list_snapshots": [SNAPSHOT]}


def test_allow_list_contents_are_fixed() -> None:
    assert set(ALLOWED_TOOLS) == {"search_guidelines", "screen_entity", "score_risk"}
    with pytest.raises(TypeError):
        ALLOWED_TOOLS["delete_records"] = ALLOWED_TOOLS["score_risk"]  # type: ignore[index]


@pytest.mark.parametrize(
    "name", ["approve_loan", "decline_application", "clear_sanctions_alert", "", "SEARCH_GUIDELINES"]
)
def test_unknown_tools_are_rejected(name: str) -> None:
    with pytest.raises(ToolNotAllowed):
        get_tool_spec(name)
    with pytest.raises(ToolNotAllowed):
        call_tool(name, {})


def test_call_tool_validates_input_and_output() -> None:
    result = call_tool("search_guidelines", {"query": "DSCR"}, handler_overrides={"search_guidelines": fake_search})
    assert isinstance(result, GuidelineSearchResult)
    assert result.hits[0].citation == "[NBC-4.2]"
    with pytest.raises(ValidationError):
        call_tool("search_guidelines", {"query": "DSCR", "extra": 1}, {"search_guidelines": fake_search})


def test_bad_handler_output_is_rejected() -> None:
    def broken(request: ScreeningRequest) -> dict:
        return {"request": request.model_dump(), "candidates": [], "status": "cleared", "list_snapshots": [SNAPSHOT]}

    with pytest.raises(ValidationError):
        call_tool("screen_entity", {"name": "X Corp", "entity_type": "organization"}, {"screen_entity": broken})
    ok = call_tool("screen_entity", {"name": "X Corp", "entity_type": "organization"}, {"screen_entity": fake_screen})
    assert ok.status == "clear"


def test_overrides_cannot_add_new_tools() -> None:
    with pytest.raises(ToolNotAllowed):
        call_tool("search_guidelines", {"query": "x"}, handler_overrides={"run_shell": fake_search})


def test_missing_handler_module_is_reported_clearly() -> None:
    spec = ToolSpec(
        name="search_guidelines",
        description="test",
        input_model=GuidelineQuery,
        output_model=GuidelineSearchResult,
        handler_path="uwcopilot.does_not_exist:search",
    )
    with pytest.raises(ToolUnavailable):
        resolve_handler(spec)


def test_handler_paths_point_inside_the_package() -> None:
    for spec in ALLOWED_TOOLS.values():
        module_name, separator, function_name = spec.handler_path.partition(":")
        assert separator == ":" and function_name
        assert module_name.startswith("uwcopilot.")


# Tools whose handler module is knowingly not built yet. Empty: every allowed tool must resolve, so a dangling
# handler path fails here instead of only at call time.
PENDING_TOOLS: frozenset[str] = frozenset()


@pytest.mark.parametrize("name", sorted(ALLOWED_TOOLS))
def test_every_allowed_handler_resolves(name: str) -> None:
    if name in PENDING_TOOLS:
        pytest.xfail(f"{name} handler is not built yet")
    assert callable(resolve_handler(ALLOWED_TOOLS[name]))


def test_tool_definitions_expose_json_schemas() -> None:
    definitions = tool_definitions()
    assert [item["name"] for item in definitions] == sorted(ALLOWED_TOOLS)
    search = next(item for item in definitions if item["name"] == "search_guidelines")
    assert search["input_schema"]["additionalProperties"] is False
    assert "query" in search["input_schema"]["required"]


def test_registry_module_does_not_import_handlers_eagerly() -> None:
    assert not hasattr(registry, "search_guidelines")
