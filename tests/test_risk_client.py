"""Tests for uwcopilot.risk.client: one meaning of UWC_RISK_API_URL for the agent, the MCP server and the registry."""

from pathlib import Path

import httpx
import pytest
import respx

from uwcopilot.agent.settings import AgentSettings, load_agent_settings
from uwcopilot.agent.tools import default_tool_handlers
from uwcopilot.mcp_server.settings import load_mcp_settings
from uwcopilot.risk import client as risk_client
from uwcopilot.risk.client import (
    RiskApiClient,
    RiskApiSettings,
    RiskApiUnavailable,
    RiskSettingsError,
    describe_risk_service,
    load_risk_api_settings,
    score_endpoint,
    validate_base_url,
)
from uwcopilot.tools.registry import ToolUnavailable, call_tool
from uwcopilot.tools.schemas import RiskScoreRequest

REQUEST = RiskScoreRequest(product="credit", features={"dscr": 1.3})
# A successful POST /score body in the SERVICE contract (uwcopilot.risk.service.ScoreResponse); FAKE numbers.
BODY = {
    "product": "credit",
    "prediction_name": "probability_of_default",
    "prediction_unit": "probability",
    "prediction": 0.05,
    "risk_band": "C",
    "reason_codes": [],
    "model": "lightgbm",
    "model_version": "test-1",
    "risk_band_definition": {"method": "quantiles"},
    "scenario": None,
    "note": "FAKE test response",
}


def test_base_url_gets_score_appended_once() -> None:
    assert score_endpoint("http://risk:8000") == "http://risk:8000/score"
    assert score_endpoint("http://risk:8000/") == "http://risk:8000/score"
    assert RiskApiSettings(base_url="http://risk:8000/v1").score_url == "http://risk:8000/v1/score"
    assert RiskApiSettings().score_url is None


@pytest.mark.parametrize(
    "url",
    ["http://risk:8000/score", "http://risk:8000/score/", "http://risk?x=1", "http://risk#a", "risk:8000", "ftp://r"],
)
def test_full_endpoints_and_bad_urls_are_refused(url: str) -> None:
    with pytest.raises(RiskSettingsError):
        validate_base_url(url)


def test_unset_means_not_configured() -> None:
    settings = load_risk_api_settings(env_file=None, environ={})
    assert settings == RiskApiSettings()
    assert describe_risk_service(settings).startswith("unavailable: UWC_RISK_API_URL not set")


def test_env_file_wins(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("UWC_RISK_API_URL=http://file:1\nUWC_RISK_API_TIMEOUT_S=3\n", encoding="utf-8")
    settings = load_risk_api_settings(env_file=env_file, environ={"UWC_RISK_API_URL": "http://env:2"})
    assert settings == RiskApiSettings(base_url="http://file:1", timeout_s=3.0)


@pytest.mark.parametrize("value", ["http://localhost:8000", "http://localhost:8000/", "https://risk.example.test/v1"])
def test_agent_and_mcp_server_post_to_the_same_endpoint(value: str) -> None:
    """The same UWC_RISK_API_URL value makes the agent and the MCP server send requests to the same URL."""
    environ = {"UWC_RISK_API_URL": value}
    agent_settings = load_agent_settings(env_file=None, environ=environ)
    mcp_settings = load_mcp_settings(env_file=None, environ=environ)
    assert agent_settings.risk_api == mcp_settings.risk_api
    expected = value.rstrip("/") + "/score"
    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(expected).mock(return_value=httpx.Response(200, json=BODY))
        default_tool_handlers(agent_settings)["score_risk"](REQUEST)
        RiskApiClient.from_settings(mcp_settings.risk_api).score(REQUEST)
    assert [str(call.request.url) for call in route.calls] == [expected, expected]


def test_unset_is_unavailable_for_agent_and_mcp_server_alike() -> None:
    """Neither component sends anything when the variable is unset; both report the tool as unavailable."""
    agent_handler = default_tool_handlers(AgentSettings())["score_risk"]
    mcp_client = RiskApiClient.from_settings(load_mcp_settings(env_file=None, environ={}).risk_api)
    with respx.mock(assert_all_mocked=True) as mock:
        with pytest.raises(ToolUnavailable):
            agent_handler(REQUEST)
        with pytest.raises(RiskApiUnavailable) as raised:
            mcp_client.score(REQUEST)
        assert not mock.calls
    assert raised.value.info.code == "risk_api_not_configured"
    assert raised.value.info.endpoint is None


def test_registry_handler_uses_the_shared_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry's score_risk handler (uwcopilot.risk.client:score_risk) resolves and follows the settings."""
    monkeypatch.setattr(risk_client, "load_risk_api_settings", lambda: RiskApiSettings())
    with pytest.raises(ToolUnavailable):
        call_tool("score_risk", REQUEST)
    monkeypatch.setattr(risk_client, "load_risk_api_settings", lambda: RiskApiSettings(base_url="http://risk.test"))
    with respx.mock() as mock:
        mock.post("http://risk.test/score").mock(return_value=httpx.Response(200, json=BODY))
        assert call_tool("score_risk", REQUEST).risk_band == "C"
