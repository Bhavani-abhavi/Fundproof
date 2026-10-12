"""The shared RiskApiClient as the MCP server uses it: request shape, status handling, no failure yields a score."""

import json

import httpx
import pytest
from mcp_test_helpers import json_handler, raising_handler, risk_api, risk_response

from uwcopilot.mcp_server.models import ScoreApplicationResult
from uwcopilot.risk.client import MAX_DETAIL_CHARS, RiskApiError, RiskApiErrorInfo, short_detail
from uwcopilot.tools.schemas import RiskScoreRequest, RiskScoreResponse

REQUEST = RiskScoreRequest(product="credit", features={"dscr": 1.3, "fico": None})


def test_success_returns_the_validated_response() -> None:
    """A 200 with a valid body is returned as RiskScoreResponse."""
    response = risk_api(json_handler(200, risk_response())).score(REQUEST)
    assert isinstance(response, RiskScoreResponse)
    assert response.risk_band == "B"


def test_request_is_a_json_post_of_the_request_model() -> None:
    """The body is the SERVICE contract (product, features; overrides only when given), sent to <base URL>/score."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record and answer."""
        sent.append(request)
        return httpx.Response(200, json=risk_response())

    risk_api(handler, url="http://risk.test/v1").score(REQUEST)
    assert sent[0].method == "POST"
    assert str(sent[0].url) == "http://risk.test/v1/score"
    assert json.loads(sent[0].content) == {
        "product": "credit",
        "features": {"dscr": 1.3, "fico": None},
    }


@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        (httpx.ConnectError, "risk_api_unreachable"),
        (httpx.RemoteProtocolError, "risk_api_unreachable"),
        (httpx.ConnectTimeout, "risk_api_timeout"),
        (httpx.ReadTimeout, "risk_api_timeout"),
    ],
)
def test_network_failures_have_their_own_codes(error_type: type[httpx.TransportError], code: str) -> None:
    """Timeouts are told apart from other network failures."""
    with pytest.raises(RiskApiError) as raised:
        risk_api(raising_handler(error_type)).score(REQUEST)
    assert raised.value.info.code == code
    assert raised.value.info.http_status is None


@pytest.mark.parametrize(
    ("status", "body", "code"),
    [
        (400, {"detail": "bad"}, "risk_api_rejected_request"),
        (422, {"detail": [{"loc": ["body", "overrides", "dscr"], "msg": "out of range"}]}, "risk_api_rejected_request"),
        (500, "Internal Server Error", "risk_api_server_error"),
        (503, {"detail": "warming up"}, "risk_api_server_error"),
        (204, None, "risk_api_invalid_response"),
        (302, None, "risk_api_invalid_response"),
        (200, {"product": "credit"}, "risk_api_invalid_response"),
        (200, {**risk_response(), "prediction": "high"}, "risk_api_invalid_response"),
        (200, {**risk_response(), "approved": True}, "risk_api_invalid_response"),
    ],
)
def test_http_failures_and_bad_bodies_are_errors(status: int, body: object, code: str) -> None:
    """4xx, 5xx, unexpected statuses and anything that is not a valid RiskScoreResponse raise RiskApiError."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Answer with the given status and body."""
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        if body is None:
            return httpx.Response(status)
        return httpx.Response(status, json=body)

    with pytest.raises(RiskApiError) as raised:
        risk_api(handler).score(REQUEST)
    assert raised.value.info.code == code
    assert raised.value.info.http_status == status


def test_fastapi_validation_detail_is_summarised() -> None:
    """A FastAPI-style list of errors becomes "loc: msg" pairs."""
    response = httpx.Response(422, json={"detail": [{"loc": ["body", "overrides", "dscr"], "msg": "out of range"}]})
    assert short_detail(response) == "body.overrides.dscr: out of range"


def test_long_detail_is_truncated() -> None:
    """The service's own message is cut so a huge error body cannot flood the tool output."""
    response = httpx.Response(500, json={"detail": "x" * 5000})
    assert len(short_detail(response)) == MAX_DETAIL_CHARS + 3


def test_client_ignores_proxy_settings_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """trust_env=False: a proxy variable in the environment does not change where the request goes."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9")
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:9")
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record and answer."""
        sent.append(request)
        return httpx.Response(200, json=risk_response())

    risk_api(handler).score(REQUEST)
    assert sent[0].url.host == "risk.test"


def test_result_envelope_cannot_mix_score_and_error() -> None:
    """status "scored" needs a score; status "error" needs an error and forbids a score."""
    score = RiskScoreResponse(
        product="credit", prediction=0.083, risk_band="B", reason_codes=[], model_version="fake-1", scenario="baseline"
    )
    error = RiskApiErrorInfo(code="risk_api_unreachable", message="down", endpoint="http://x/score")
    assert ScoreApplicationResult(status="scored", score=score).error is None
    assert ScoreApplicationResult(status="error", error=error).score is None
    with pytest.raises(ValueError):
        ScoreApplicationResult(status="error", error=error, score=score)
    with pytest.raises(ValueError):
        ScoreApplicationResult(status="scored")
    with pytest.raises(ValueError):
        ScoreApplicationResult(status="error")
