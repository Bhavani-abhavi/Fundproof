"""The one client of the risk scoring REST service, shared by the agent, the MCP server and the tool registry.

One setting, one meaning, for every caller:

- UWC_RISK_API_URL is the BASE URL of the service, for example http://localhost:8000. Every client sends
  POST <base URL>/score (see score_endpoint()). A value that already ends in /score is refused with a clear
  message, so nobody ends up posting to .../score/score.
- Unset means "not configured": no request is sent anywhere, and the score_risk tool reports that it is
  unavailable (error code risk_api_not_configured). There is no localhost default and no score is ever made up.
- UWC_RISK_API_TIMEOUT_S: seconds to wait for an answer (default 10, at most 120).

The request body is a RiskScoreRequest and a successful answer must be a valid RiskScoreResponse. Every failure
becomes a RiskApiError carrying a RiskApiErrorInfo with a fixed code, so the MCP server can return it as a
structured error. Failures that mean "the service is not there" (not configured, unreachable, timeout) are
RiskApiUnavailable, which is also a registry ToolUnavailable: the agent records the tool as unavailable and the
policy reports the model PD as "not evaluated".

Two deliberate choices:
- trust_env=False: proxy and .netrc settings in the environment are never applied. Only UWC_RISK_API_URL decides
  where application data goes.
- Error messages and logs show the endpoint without user name, password, query string or fragment
  (public_endpoint()), and quote at most MAX_DETAIL_CHARS of the service's own error detail.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import Field, ValidationError

from uwcopilot.common.paths import ENV_FILE
from uwcopilot.common.settings import env_reader, is_http_url
from uwcopilot.tools.registry import ToolUnavailable
from uwcopilot.tools.schemas import RiskScoreRequest, RiskScoreResponse, StrictOutput

SCORE_PATH = "/score"
DEFAULT_TIMEOUT_S = 10.0
MAX_TIMEOUT_S = 120.0
MAX_DETAIL_CHARS = 300
NOT_CONFIGURED_MESSAGE = (
    "UWC_RISK_API_URL is not set, so the risk scoring service was not called. No score was produced."
)

RiskApiErrorCode = Literal[
    "risk_api_not_configured",
    "risk_api_unreachable",
    "risk_api_timeout",
    "risk_api_rejected_request",
    "risk_api_server_error",
    "risk_api_invalid_response",
]
# Codes that mean the service is not there at all (as opposed to answering badly).
UNAVAILABLE_CODES: frozenset[str] = frozenset({"risk_api_not_configured", "risk_api_unreachable", "risk_api_timeout"})


class RiskSettingsError(ValueError):
    """Raised when UWC_RISK_API_URL or UWC_RISK_API_TIMEOUT_S has an invalid value."""


class RiskApiErrorInfo(StrictOutput):
    """Why no score was returned. endpoint never contains credentials or a query string (None: not configured)."""

    code: RiskApiErrorCode
    message: str = Field(min_length=1)
    endpoint: str | None = None
    http_status: int | None = None


class RiskApiError(RuntimeError):
    """The risk service did not return a valid score. info says why, in a form that is safe to show."""

    def __init__(self, info: RiskApiErrorInfo) -> None:
        """Keep the structured error; the exception text is its message."""
        super().__init__(info.message)
        self.info = info


class RiskApiUnavailable(RiskApiError, ToolUnavailable):
    """The service is not configured, cannot be reached, or did not answer in time."""


@dataclass(frozen=True)
class RiskApiSettings:
    """Where the risk service is (None: not configured) and how long to wait for it."""

    base_url: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S

    @property
    def score_url(self) -> str | None:
        """The full score endpoint, or None when the service is not configured."""
        return score_endpoint(self.base_url) if self.base_url else None


def validate_base_url(url: str) -> str:
    """An absolute http(s) base URL with a host, no query or fragment, not ending in /score; trailing / removed."""
    cleaned = url.strip().rstrip("/")
    if not is_http_url(cleaned):
        raise RiskSettingsError("UWC_RISK_API_URL must be an absolute http:// or https:// URL with a host.")
    parts = urlsplit(cleaned)
    if parts.query or parts.fragment:
        raise RiskSettingsError("UWC_RISK_API_URL must not contain a query string or fragment.")
    if parts.path.endswith(SCORE_PATH):
        raise RiskSettingsError(
            "UWC_RISK_API_URL is the service's base URL (for example http://localhost:8000); the client adds "
            f"{SCORE_PATH} itself, so leave it off."
        )
    return cleaned


def validate_timeout(value: str) -> float:
    """A positive number of seconds, at most MAX_TIMEOUT_S."""
    try:
        seconds = float(value)
    except ValueError as error:
        raise RiskSettingsError(f"UWC_RISK_API_TIMEOUT_S must be a number of seconds, got {value!r}.") from error
    if not 0 < seconds <= MAX_TIMEOUT_S:
        raise RiskSettingsError(f"UWC_RISK_API_TIMEOUT_S must be above 0 and at most {MAX_TIMEOUT_S:g}.")
    return seconds


def score_endpoint(base_url: str) -> str:
    """The score endpoint of a service at base_url: base URL + /score."""
    return base_url.rstrip("/") + SCORE_PATH


def public_endpoint(url: str) -> str:
    """The URL without user name, password, query string or fragment, safe to show in tool output and logs."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:  # an IPv6 address needs its brackets back
        host = f"[{host}]"
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def load_risk_api_settings(
    env_file: Path | None = ENV_FILE, environ: Mapping[str, str] | None = None
) -> RiskApiSettings:
    """Read UWC_RISK_API_URL and UWC_RISK_API_TIMEOUT_S (the repo .env file wins; env_file=None skips it)."""
    read = env_reader(env_file, environ)
    url = read("UWC_RISK_API_URL")
    timeout = read("UWC_RISK_API_TIMEOUT_S")
    return RiskApiSettings(
        base_url=validate_base_url(url) if url else None,
        timeout_s=validate_timeout(timeout) if timeout else DEFAULT_TIMEOUT_S,
    )


def short_detail(response: httpx.Response) -> str:
    """The service's error detail (FastAPI style {"detail": ...} when present), cut to MAX_DETAIL_CHARS."""
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    if isinstance(body, dict) and "detail" in body:
        body = body["detail"]
    if isinstance(body, list):
        parts = []
        for item in body:
            if isinstance(item, dict) and "msg" in item:
                location = ".".join(str(part) for part in item.get("loc", []))
                parts.append(f"{location}: {item['msg']}" if location else str(item["msg"]))
            else:
                parts.append(str(item))
        body = "; ".join(parts)
    text = body if isinstance(body, str) else json.dumps(body)
    text = " ".join(text.split())
    if len(text) > MAX_DETAIL_CHARS:
        text = text[:MAX_DETAIL_CHARS] + "..."
    return text


class RiskApiClient:
    """Calls POST <base_url>/score. transport lets tests replace the network (httpx.MockTransport)."""

    def __init__(
        self,
        base_url: str | None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """base_url None means not configured: score() then refuses without sending anything."""
        self.score_url = score_endpoint(validate_base_url(base_url)) if base_url else None
        self.endpoint = public_endpoint(self.score_url) if self.score_url else None
        self.timeout_s = timeout_s
        self.transport = transport

    @classmethod
    def from_settings(cls, settings: RiskApiSettings, transport: httpx.BaseTransport | None = None) -> "RiskApiClient":
        """A client for the configured service (or an unconfigured one that always refuses)."""
        return cls(settings.base_url, settings.timeout_s, transport)

    def error(self, code: RiskApiErrorCode, message: str, http_status: int | None = None) -> RiskApiError:
        """A RiskApiError (RiskApiUnavailable for the "service is not there" codes) for this endpoint."""
        info = RiskApiErrorInfo(code=code, message=message, endpoint=self.endpoint, http_status=http_status)
        if code in UNAVAILABLE_CODES:
            return RiskApiUnavailable(info)
        return RiskApiError(info)

    def post(self, score_url: str, request: RiskScoreRequest) -> httpx.Response:
        """Send the request; network failures become RiskApiUnavailable (unreachable or timeout)."""
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout_s, trust_env=False) as client:
                return client.post(score_url, json=request.model_dump(mode="json"))
        except httpx.TimeoutException as error:
            message = f"The risk service did not answer within {self.timeout_s:g} s ({type(error).__name__})."
            raise self.error("risk_api_timeout", message) from error
        except httpx.TransportError as error:
            message = (
                f"Could not connect to the risk service ({type(error).__name__}). Start it, or set "
                "UWC_RISK_API_URL to its base URL. No score was produced."
            )
            raise self.error("risk_api_unreachable", message) from error

    def score(self, request: RiskScoreRequest) -> RiskScoreResponse:
        """Score one applicant. Raises RiskApiError for any failure; never returns a made-up score."""
        if self.score_url is None:
            raise self.error("risk_api_not_configured", NOT_CONFIGURED_MESSAGE)
        response = self.post(self.score_url, request)
        status = response.status_code
        if 400 <= status < 500:
            message = f"The risk service rejected the request (HTTP {status}): {short_detail(response)}"
            raise self.error("risk_api_rejected_request", message, status)
        if status >= 500:
            message = f"The risk service failed (HTTP {status}): {short_detail(response)}"
            raise self.error("risk_api_server_error", message, status)
        if status != 200:
            message = f"The risk service returned HTTP {status} instead of 200 (redirects are not followed)."
            raise self.error("risk_api_invalid_response", message, status)
        try:
            return RiskScoreResponse.model_validate(response.json())
        except (ValueError, ValidationError) as error:
            message = "The risk service answered, but not with a valid RiskScoreResponse. No score was produced."
            raise self.error("risk_api_invalid_response", message, status) from error


def describe_risk_service(settings: RiskApiSettings) -> str:
    """A one-line description of the risk tool's configuration for reports (no credentials)."""
    if settings.score_url is None:
        return "unavailable: UWC_RISK_API_URL not set, so model PD is 'not evaluated' by the policy"
    return f"HTTP service at {public_endpoint(settings.score_url)}"


def score_risk(request: RiskScoreRequest) -> RiskScoreResponse:
    """The registry's score_risk handler: score with the service configured in .env / the environment."""
    return RiskApiClient.from_settings(load_risk_api_settings()).score(request)
