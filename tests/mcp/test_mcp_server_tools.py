"""The MCP tools, called through the official MCP SDK client connected to the server in-process.

Fakes only: fake-embedder index in an in-memory Qdrant, a tiny invented watchlist, httpx.MockTransport for the
risk service (plus one real connection attempt to a closed local port).
"""

import socket
from typing import Any

import httpx
import pytest
from mcp_test_helpers import (
    build_guideline_retriever,
    json_handler,
    make_server,
    raising_handler,
    risk_api,
    risk_response,
    run_with_client,
    tiny_matcher,
)

from mcp import Client
from uwcopilot.kyc import screening
from uwcopilot.mcp_server import server as server_module
from uwcopilot.mcp_server.server import (
    MCP_TOOL_BINDINGS,
    ServerDependencies,
    ToolBindingError,
    build_server,
    check_tool_bindings,
    make_tool,
)
from uwcopilot.rag.index import IndexNotBuiltError
from uwcopilot.risk.client import RiskApiClient
from uwcopilot.tools.registry import ALLOWED_TOOLS, ToolNotAllowed
from uwcopilot.tools.schemas import GuidelineSearchResult, ScreeningResult

EXPECTED_TOOLS = ["get_guideline_section", "score_application", "screen_sanctions", "search_underwriting_guidelines"]


async def call(client: Client, name: str, arguments: dict[str, Any]) -> Any:
    """Call one tool and return the CallToolResult."""
    return await client.call_tool(name, arguments)


def call_once(server: Any, name: str, arguments: dict[str, Any], mode: str = "legacy") -> Any:
    """Connect, call one tool, disconnect."""

    async def scenario(client: Client) -> Any:
        """The single call."""
        return await call(client, name, arguments)

    return run_with_client(server, scenario, mode=mode)


def closed_local_port() -> int:
    """A local port with nothing listening (bound and released)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------------------------------------------
# Listing and the allow-list
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_lists_exactly_the_four_tools(guideline_retriever: Any, mode: str) -> None:
    """Both client modes (JSON-RPC handshake and 2026-07-28 direct dispatch) see the same four tools."""
    server, _ = make_server(guideline_retriever)

    async def scenario(client: Client) -> Any:
        """List the tools."""
        return (await client.list_tools()).tools

    tools = run_with_client(server, scenario, mode=mode)
    assert sorted(tool.name for tool in tools) == EXPECTED_TOOLS


def test_every_tool_is_bound_to_an_allow_listed_registry_tool() -> None:
    """MCP tools only exist on top of tools in the registry allow-list."""
    assert sorted(MCP_TOOL_BINDINGS) == EXPECTED_TOOLS
    assert set(MCP_TOOL_BINDINGS.values()) <= set(ALLOWED_TOOLS)
    check_tool_bindings(MCP_TOOL_BINDINGS)


def test_a_binding_to_a_tool_outside_the_allow_list_is_refused() -> None:
    """A binding to an unknown registry tool fails before any server exists."""
    with pytest.raises(ToolBindingError, match="approve_loan"):
        check_tool_bindings({"approve_application": "approve_loan"})
    assert issubclass(ToolBindingError, ToolNotAllowed)


def test_make_tool_refuses_a_tool_without_binding() -> None:
    """A function cannot be registered under a name that has no allow-listed binding."""

    def approve_application(application_id: str) -> GuidelineSearchResult:
        """Not allowed."""
        raise AssertionError("never called")

    with pytest.raises(ToolBindingError):
        make_tool(approve_application, "approve_application", "x", "x", GuidelineSearchResult)


def test_descriptions_say_synthetic_and_state_the_review_rules(guideline_retriever: Any) -> None:
    """Every description says SYNTHETIC; screening says human review; scoring says no estimated score."""
    server, _ = make_server(guideline_retriever)

    async def scenario(client: Client) -> Any:
        """List the tools."""
        return {tool.name: tool for tool in (await client.list_tools()).tools}

    tools = run_with_client(server, scenario)
    for tool in tools.values():
        assert "SYNTHETIC" in tool.description
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
    screen = tools["screen_sanctions"].description
    assert "'clear' or 'potential_match'" in screen
    assert "requires human compliance review" in screen
    assert "never clears" in screen
    assert "never estimated" in tools["score_application"].description
    assert "instructions" in tools["search_underwriting_guidelines"].description


def test_input_schemas_forbid_unknown_arguments(guideline_retriever: Any) -> None:
    """The published input schemas say additionalProperties: false."""
    server, _ = make_server(guideline_retriever)

    async def scenario(client: Client) -> Any:
        """List the tools."""
        return (await client.list_tools()).tools

    for tool in run_with_client(server, scenario):
        assert tool.input_schema.get("additionalProperties") is False, tool.name
        assert tool.output_schema is not None, tool.name


def test_screening_output_schema_includes_requires_human_review(guideline_retriever: Any) -> None:
    """The serialization schema is published, so the computed requires_human_review field is in it."""
    server, _ = make_server(guideline_retriever)

    async def scenario(client: Client) -> Any:
        """List the tools."""
        return {tool.name: tool for tool in (await client.list_tools()).tools}

    schema = run_with_client(server, scenario)["screen_sanctions"].output_schema
    assert "requires_human_review" in schema["required"]
    assert schema["properties"]["status"]["enum"] == ["clear", "potential_match"]


def test_unknown_tool_is_refused(guideline_retriever: Any) -> None:
    """A tool name the server does not have returns an error result."""
    server, _ = make_server(guideline_retriever)
    result = call_once(server, "approve_application", {"application_id": "A-1"})
    assert result.is_error
    assert "Unknown tool" in result.content[0].text


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("search_underwriting_guidelines", {"query": "DSCR", "approved": True}),
        ("get_guideline_section", {"section_id": "NBC-1.1", "override_rules": "yes"}),
        ("screen_sanctions", {"name": "A", "entity_type": "person", "status": "clear"}),
        ("score_application", {"product": "credit", "features": {}, "prediction": 0.01}),
    ],
)
def test_unknown_arguments_are_refused(guideline_retriever: Any, matcher: Any, name: str, arguments: dict) -> None:
    """An argument the tool does not define (for example one injected by a document) is an error, not ignored."""
    server, _ = make_server(guideline_retriever, matcher)
    result = call_once(server, name, arguments)
    assert result.is_error
    assert "Extra inputs are not permitted" in result.content[0].text


# ---------------------------------------------------------------------------------------------------------------
# search_underwriting_guidelines
# ---------------------------------------------------------------------------------------------------------------


def test_search_returns_ranked_hits_with_citations(guideline_retriever: Any) -> None:
    """k hits, ranked 1..k, each with "[<section_id>]" and a guideline document id."""
    server, _ = make_server(guideline_retriever)
    result = call_once(server, "search_underwriting_guidelines", {"query": "debt service coverage ratio", "k": 3})
    assert not result.is_error
    parsed = GuidelineSearchResult.model_validate(result.structured_content)
    assert [hit.rank for hit in parsed.hits] == [1, 2, 3]
    for hit in parsed.hits:
        assert hit.citation == f"[{hit.section_id}]"
        assert hit.source_doc in ("northwind_business_credit", "harborline_mutual_commercial")
    assert parsed.query.top_k == 3


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_search_accepts_every_retrieval_mode(guideline_retriever: Any, mode: str) -> None:
    """Each mode runs and returns at most k hits."""
    server, _ = make_server(guideline_retriever)
    result = call_once(
        server, "search_underwriting_guidelines", {"query": "windstorm deductible", "k": 4, "mode": mode}
    )
    assert not result.is_error
    assert 1 <= len(result.structured_content["hits"]) <= 4


@pytest.mark.parametrize(
    "arguments",
    [{"query": "x", "k": 0}, {"query": "x", "k": 21}, {"query": "", "k": 3}, {"query": "x", "mode": "magic"}],
)
def test_search_rejects_bad_arguments(guideline_retriever: Any, arguments: dict) -> None:
    """k outside 1..20, an empty query and an unknown mode are refused."""
    server, _ = make_server(guideline_retriever)
    assert call_once(server, "search_underwriting_guidelines", arguments).is_error


def test_search_goes_through_the_registry(guideline_retriever: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The search runs as the allow-listed search_guidelines registry tool."""
    seen: list[str] = []
    real_call_tool = server_module.call_tool

    def spy(name: str, arguments: Any, handler_overrides: Any = None) -> Any:
        """Record the registry tool name, then call the real registry."""
        seen.append(name)
        return real_call_tool(name, arguments, handler_overrides=handler_overrides)

    monkeypatch.setattr(server_module, "call_tool", spy)
    server, _ = make_server(guideline_retriever)
    assert not call_once(server, "search_underwriting_guidelines", {"query": "loss ratio"}).is_error
    assert seen == ["search_guidelines"]


def test_search_without_an_index_is_a_clear_error() -> None:
    """When the collection was never built, the tool says to run `make index` (no crash, no empty success)."""

    def no_index() -> Any:
        """Behave like a missing collection."""
        raise IndexNotBuiltError("Qdrant collection 'uwc_guidelines' does not exist. Build it with `make index`.")

    server, _ = make_server(retriever_factory=no_index)
    result = call_once(server, "search_underwriting_guidelines", {"query": "DSCR"})
    assert result.is_error
    assert "make index" in result.content[0].text


def test_retriever_is_built_once_and_rebuilt_after_close() -> None:
    """The factory runs on first use only; close() closes that retriever, and the next use builds a new one."""
    built: list[Any] = []

    def factory() -> Any:
        """Build a fresh fake retriever and remember it."""
        built.append(build_guideline_retriever())
        return built[-1]

    dependencies = ServerDependencies(retriever_factory=factory)
    first = dependencies.retriever()
    assert dependencies.retriever() is first
    assert len(built) == 1
    dependencies.close()
    dependencies.close()  # closing twice is harmless
    assert dependencies.retriever() is not first
    assert len(built) == 2
    dependencies.close()


# ---------------------------------------------------------------------------------------------------------------
# get_guideline_section
# ---------------------------------------------------------------------------------------------------------------


def test_get_section_returns_the_whole_section(guideline_retriever: Any, store: Any) -> None:
    """HM-9.3 is split in two parts in the index, but the tool returns the whole section."""
    server, _ = make_server(guideline_retriever)
    result = call_once(server, "get_guideline_section", {"section_id": "hm-9.3"})
    assert not result.is_error
    payload = result.structured_content
    assert payload["section_id"] == "HM-9.3"
    assert payload["citation"] == "[HM-9.3]"
    assert payload["resource_uri"] == "guidelines://HM/HM-9.3"
    assert payload["synthetic"] is True
    assert payload["text"] == store.get_section("HM-9.3").text
    assert "R-HM-17" in payload["text"]


@pytest.mark.parametrize("section_id", ["NBC-99.9", "SR26-2-p5-1", "../etc/passwd", "NBC-3.2; drop"])
def test_get_section_rejects_unknown_or_malformed_ids(guideline_retriever: Any, section_id: str) -> None:
    """Unknown ids, regulatory chunk ids and anything that is not a section id are errors."""
    server, _ = make_server(guideline_retriever)
    assert call_once(server, "get_guideline_section", {"section_id": section_id}).is_error


def test_get_section_is_gated_on_the_allow_list(guideline_retriever: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """If search_guidelines left the allow-list, the section lookup would stop working too."""

    def not_allowed(name: str) -> Any:
        """Act as if the registry no longer allowed the tool."""
        raise ToolNotAllowed(f"Tool {name!r} is not on the allow-list.")

    server, _ = make_server(guideline_retriever)  # built while the tool is still allowed
    monkeypatch.setattr(server_module, "get_tool_spec", not_allowed)
    assert call_once(server, "get_guideline_section", {"section_id": "NBC-1.1"}).is_error
    with pytest.raises(ToolBindingError):  # and no new server can be built
        make_server(guideline_retriever)


# ---------------------------------------------------------------------------------------------------------------
# screen_sanctions
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_screen_potential_match_requires_human_review(guideline_retriever: Any, matcher: Any, mode: str) -> None:
    """A listed name is a potential match with candidates, explanations and requires_human_review true."""
    server, _ = make_server(guideline_retriever, matcher)
    arguments = {"name": "Zarkovian Holdings Testco Limited", "entity_type": "organization", "country": "Cyprus"}
    result = call_once(server, "screen_sanctions", arguments, mode=mode)
    assert not result.is_error
    parsed = ScreeningResult.model_validate(result.structured_content)
    assert parsed.status == "potential_match"
    assert result.structured_content["requires_human_review"] is True
    assert parsed.candidates[0].list_entry_id == "OFAC-T-9001"
    assert parsed.candidates[0].explanation
    assert parsed.list_snapshots


def test_screen_clear_name(guideline_retriever: Any, matcher: Any) -> None:
    """An unrelated name is clear, with no candidates and no review flag."""
    server, _ = make_server(guideline_retriever, matcher)
    result = call_once(server, "screen_sanctions", {"name": "Maple Street Bakery", "entity_type": "organization"})
    assert not result.is_error
    assert result.structured_content["status"] == "clear"
    assert result.structured_content["candidates"] == []
    assert result.structured_content["requires_human_review"] is False


def test_screen_rejects_an_unknown_entity_type(guideline_retriever: Any, matcher: Any) -> None:
    """entity_type is person or organization only."""
    server, _ = make_server(guideline_retriever, matcher)
    assert call_once(server, "screen_sanctions", {"name": "X", "entity_type": "vessel"}).is_error


def test_screen_without_injected_matcher_uses_the_registry_handler(
    guideline_retriever: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no matcher passed in, the registry's own screen_entity handler runs (with no override)."""
    overrides_seen: list[Any] = []
    real_call_tool = server_module.call_tool

    def spy(name: str, arguments: Any, handler_overrides: Any = None) -> Any:
        """Record the overrides, then call the real registry."""
        overrides_seen.append(handler_overrides)
        return real_call_tool(name, arguments, handler_overrides=handler_overrides)

    monkeypatch.setattr(server_module, "call_tool", spy)
    screening.set_default_matcher(tiny_matcher())  # the registry handler's matcher, instead of the full lists
    try:
        server, _ = make_server(guideline_retriever, matcher=None)
        result = call_once(server, "screen_sanctions", {"name": "Ivo Testarov Belinsky", "entity_type": "person"})
    finally:
        screening.set_default_matcher(None)
    assert overrides_seen == [None]
    assert result.structured_content["status"] == "potential_match"


def test_screen_error_message_does_not_repeat_the_name(guideline_retriever: Any) -> None:
    """When screening cannot run, the error says why without echoing the screened name."""

    class BrokenMatcher:
        """A matcher whose lists are not configured."""

        def screen(self, request: Any) -> Any:
            """Fail like an unconfigured screener."""
            raise screening.ScreeningNotConfigured("results/kyc_metrics.json not found. Run `make eval-kyc`.")

    server, _ = make_server(guideline_retriever, matcher=BrokenMatcher())  # type: ignore[arg-type]
    result = call_once(server, "screen_sanctions", {"name": "Secret Person Name", "entity_type": "person"})
    assert result.is_error
    assert "make eval-kyc" in result.content[0].text
    assert "Secret Person Name" not in result.content[0].text


# ---------------------------------------------------------------------------------------------------------------
# score_application
# ---------------------------------------------------------------------------------------------------------------


def test_score_success_passes_the_service_answer_through(guideline_retriever: Any) -> None:
    """A valid service answer becomes status "scored" with exactly that score."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and answer with a valid score."""
        sent.append(request)
        return httpx.Response(200, json=risk_response(scenario="revenue_down_20"))

    server, _ = make_server(guideline_retriever, handler=handler)
    arguments = {
        "product": "credit",
        "features": {"dscr": 1.4, "years_in_business": None},
        "overrides": {"dscr": 1.1},
        "scenario": "revenue_down_20",
    }
    result = call_once(server, "score_application", arguments)
    assert not result.is_error
    payload = result.structured_content
    assert payload["status"] == "scored"
    assert payload["error"] is None
    assert payload["score"]["prediction"] == 0.083
    assert payload["score"]["model_version"] == "fake-risk-model-0.0.1"
    body = httpx.Response(200, content=sent[0].content).json()
    assert sent[0].method == "POST"
    assert body == arguments


@pytest.mark.parametrize(
    ("handler", "code"),
    [
        (raising_handler(httpx.ConnectError), "risk_api_unreachable"),
        (raising_handler(httpx.ReadTimeout), "risk_api_timeout"),
        (json_handler(422, {"detail": "dscr override 9.0 is outside the allowed range"}), "risk_api_rejected_request"),
        (json_handler(503, {"detail": "model not loaded"}), "risk_api_server_error"),
        (json_handler(200, {"prediction": 0.01}), "risk_api_invalid_response"),
    ],
)
def test_score_failures_are_structured_errors_without_a_score(
    guideline_retriever: Any, handler: Any, code: str
) -> None:
    """Every failure is is_error with status "error", a code and no score; nothing is made up."""
    server, _ = make_server(guideline_retriever, handler=handler)
    result = call_once(server, "score_application", {"product": "credit", "features": {"dscr": 1.4}})
    assert result.is_error
    payload = result.structured_content
    assert payload["status"] == "error"
    assert payload["score"] is None
    assert payload["error"]["code"] == code
    assert "prediction" not in result.content[0].text


def test_score_against_a_closed_local_port_is_unreachable(guideline_retriever: Any) -> None:
    """A real connection attempt (no mock) to a port with nothing listening gives risk_api_unreachable."""
    port = closed_local_port()
    dependencies = ServerDependencies(
        retriever_factory=lambda: guideline_retriever,
        risk_api=RiskApiClient(f"http://127.0.0.1:{port}", timeout_s=5.0),
    )
    result = call_once(build_server(dependencies), "score_application", {"product": "credit", "features": {}})
    assert result.is_error
    assert result.structured_content["error"]["code"] == "risk_api_unreachable"
    assert result.structured_content["error"]["endpoint"] == f"http://127.0.0.1:{port}/score"


def test_score_without_a_configured_service_sends_nothing(guideline_retriever: Any) -> None:
    """UWC_RISK_API_URL unset: a structured risk_api_not_configured error, and no request to any default host."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record any request (there must be none)."""
        sent.append(request)
        return httpx.Response(200, json=risk_response())

    dependencies = ServerDependencies(
        retriever_factory=lambda: guideline_retriever,
        risk_api=RiskApiClient(None, transport=httpx.MockTransport(handler)),
    )
    result = call_once(build_server(dependencies), "score_application", {"product": "credit", "features": {}})
    assert result.is_error
    assert result.structured_content["status"] == "error"
    assert result.structured_content["score"] is None
    assert result.structured_content["error"]["code"] == "risk_api_not_configured"
    assert result.structured_content["error"]["endpoint"] is None
    assert sent == []


def test_score_error_never_shows_credentials_from_the_url(guideline_retriever: Any) -> None:
    """A URL with user:password is shown without them."""
    dependencies = ServerDependencies(
        retriever_factory=lambda: guideline_retriever,
        risk_api=risk_api(raising_handler(httpx.ConnectError), url="http://svc:hunter2@risk.test"),
    )
    result = call_once(build_server(dependencies), "score_application", {"product": "credit", "features": {}})
    text = result.content[0].text + str(result.structured_content)
    assert "hunter2" not in text
    assert result.structured_content["error"]["endpoint"] == "http://risk.test/score"


@pytest.mark.parametrize(
    "arguments",
    [
        {"product": "mortgage", "features": {}},
        {"product": "credit", "features": {"bad-name": 1}},
        {"product": "credit", "features": {"x": [1, 2]}},
        {"product": "credit", "features": {}, "scenario": ""},
    ],
)
def test_score_rejects_bad_arguments_before_calling_the_service(guideline_retriever: Any, arguments: dict) -> None:
    """Unknown products, odd feature names, non-scalar values and empty scenario names never reach the service."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record any request (there must be none)."""
        sent.append(request)
        return httpx.Response(200, json=risk_response())

    server, _ = make_server(guideline_retriever, handler=handler)
    result = call_once(server, "score_application", arguments)
    assert result.is_error
    assert sent == []
