"""PoliteClient: the one HTTP client every Studio connector uses.

What "polite" means here (Studio addendum rule F):
- a minimum interval between two requests to the same host (HostPolicy.min_interval_s);
- retries with exponential backoff on network errors and on HTTP 429/500/502/503/504, honoring Retry-After;
- explicit timeouts on every request;
- identical requests are answered from the on-disk cache (cache.ResponseCache) with the original request date.

What "safe" means here:
- trust_env=True, so the machine's proxy and CA-bundle settings apply (the build environment's agent proxy needs it);
- secret query parameters, JSON fields and headers are named by the caller; they are removed from the cache key,
  from the stored URL and parameters, from log lines and from error messages. Their values never leave the request.

Tests pass an httpx.MockTransport and a fake sleep, so they never touch the network or wait.
"""

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from uwcopilot.studio.data.cache import CachedResponse, ResponseCache, request_key, sha256_bytes
from uwcopilot.studio.data.errors import HostUnreachable, UpstreamError

LOGGER = logging.getLogger(__name__)
RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
USER_AGENT = "uwcopilot-studio/0.1 (portfolio project; public data; https://github.com/vajja1405/Underwriting-Copilot-)"
MAX_BACKOFF_S = 60.0
REDACTED = "***"


@dataclass(frozen=True)
class HostPolicy:
    """Politeness settings for one host. The defaults are the project's design choices, not documented limits;
    replace them with a source's documented limit when it is recorded in docs/studio/data_sources.md."""

    min_interval_s: float = 1.0
    max_retries: int = 3
    timeout_s: float = 30.0
    backoff_base_s: float = 2.0


# Per-host settings. Overpass asks heavy users to slow down, so it gets a longer interval and timeout.
DEFAULT_POLICIES: dict[str, HostPolicy] = {
    "overpass-api.de": HostPolicy(min_interval_s=5.0, timeout_s=60.0, max_retries=1),
    "www2.census.gov": HostPolicy(min_interval_s=1.0, timeout_s=120.0),
    "geo.dot.gov": HostPolicy(min_interval_s=1.0, timeout_s=60.0),
}
DEFAULT_POLICY = HostPolicy()


@dataclass(frozen=True)
class FetchResult:
    """One response body with the facts a SourceRecord needs."""

    body: bytes
    sha256: str
    retrieved_at: datetime
    url: str
    params: dict[str, str]
    status_code: int
    content_type: str
    from_cache: bool

    def text(self) -> str:
        """The body decoded as UTF-8 (a byte-order mark is dropped)."""
        return self.body.decode("utf-8-sig")

    def json(self) -> Any:
        """The body parsed as JSON."""
        return json.loads(self.text())


@dataclass
class PoliteClient:
    """Rate-limited, retrying, caching HTTP client. Build one per run and reuse it for every request."""

    cache: ResponseCache | None = field(default_factory=ResponseCache)
    transport: httpx.BaseTransport | None = None
    policies: Mapping[str, HostPolicy] = field(default_factory=lambda: dict(DEFAULT_POLICIES))
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    offline: bool = False
    last_request_at: dict[str, float] = field(default_factory=dict)

    def policy_for(self, host: str) -> HostPolicy:
        """The host's policy, or the default."""
        return self.policies.get(host, DEFAULT_POLICY)

    def get(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        secret_params: Sequence[str] = (),
        headers: Mapping[str, str] | None = None,
        secret_headers: Sequence[str] = (),
        use_cache: bool = True,
    ) -> FetchResult:
        """GET url?params. secret_params / secret_headers name the values that must never be stored or logged."""
        all_params = {str(name): str(value) for name, value in (params or {}).items()}
        public = {name: value for name, value in all_params.items() if name not in set(secret_params)}
        secrets = [all_params[name] for name in secret_params if name in all_params]
        for name in secret_headers:
            value = str((headers or {}).get(name, ""))
            # Both the whole value ("Bearer abc") and the token alone ("abc") are redacted.
            secrets += [value, value.split()[-1]] if value.strip() else []
        key = request_key("GET", url, public)
        return self.fetch("GET", url, all_params, public, None, None, headers, secrets, key, use_cache)

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        secret_fields: Sequence[str] = (),
        use_cache: bool = True,
    ) -> FetchResult:
        """POST a JSON payload. secret_fields are top-level payload keys (e.g. a registration key) never stored."""
        public_payload = {name: value for name, value in payload.items() if name not in set(secret_fields)}
        secrets = [str(payload[name]) for name in secret_fields if name in payload]
        public_body = json.dumps(public_payload, sort_keys=True).encode("utf-8")
        key = request_key("POST", url, {}, public_body)
        body = json.dumps(dict(payload)).encode("utf-8")
        return self.fetch("POST", url, {}, {}, body, public_payload, None, secrets, key, use_cache)

    def fetch(
        self,
        method: str,
        url: str,
        all_params: dict[str, str],
        public_params: dict[str, str],
        body: bytes | None,
        public_payload: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        secrets: list[str],
        key: str,
        use_cache: bool,
    ) -> FetchResult:
        """Answer from the cache when possible; otherwise send the request with rate limiting and retries."""
        host = urlsplit(url).hostname or url
        if use_cache and self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                LOGGER.info("cache hit %s %s", method, url)
                return to_result(cached, from_cache=True)
        if self.offline:
            raise HostUnreachable(host, "offline mode is on and the request is not in the cache")
        response = self.send_with_retries(method, url, all_params, body, headers, secrets, host)
        stored_params = dict(public_params)
        if public_payload is not None:
            stored_params = {"json": json.dumps(public_payload, sort_keys=True)}
        cached = CachedResponse(
            body=response.content,
            sha256=sha256_bytes(response.content),
            retrieved_at=datetime.now(UTC),
            url=url,
            params=stored_params,
            status_code=response.status_code,
            content_type=response.headers.get("content-type", ""),
        )
        if use_cache and self.cache is not None:
            self.cache.put(key, cached)
        return to_result(cached, from_cache=False)

    def send_with_retries(
        self,
        method: str,
        url: str,
        params: dict[str, str],
        body: bytes | None,
        headers: Mapping[str, str] | None,
        secrets: list[str],
        host: str,
    ) -> httpx.Response:
        """Send one request, retrying transient failures. Raises HostUnreachable or UpstreamError at the end."""
        policy = self.policy_for(host)
        request_headers = {"User-Agent": USER_AGENT, **dict(headers or {})}
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        last_problem = "no attempt made"
        for attempt in range(policy.max_retries + 1):
            self.wait_for_turn(host, policy)
            try:
                with httpx.Client(
                    transport=self.transport, timeout=policy.timeout_s, trust_env=True, follow_redirects=True
                ) as client:
                    response = client.request(method, url, params=params or None, content=body, headers=request_headers)
            except httpx.TransportError as exc:
                last_problem = redact(f"{type(exc).__name__}: {exc}", secrets)
                LOGGER.warning("%s %s failed (attempt %d): %s", method, url, attempt + 1, last_problem)
                if attempt < policy.max_retries:
                    self.sleep(backoff_seconds(policy, attempt, None))
                continue
            if response.status_code in RETRY_STATUSES and attempt < policy.max_retries:
                LOGGER.warning("%s %s answered %d (attempt %d)", method, url, response.status_code, attempt + 1)
                self.sleep(backoff_seconds(policy, attempt, response.headers.get("retry-after")))
                continue
            if response.status_code >= 400:
                snippet = redact(response.text[:200], secrets)
                raise UpstreamError(host, snippet or "(empty body)", response.status_code)
            return response
        raise HostUnreachable(host, f"{last_problem} (after {policy.max_retries + 1} attempts)")

    def wait_for_turn(self, host: str, policy: HostPolicy) -> None:
        """Sleep until at least min_interval_s has passed since the last request to this host."""
        last = self.last_request_at.get(host)
        now = self.clock()
        if last is not None and now - last < policy.min_interval_s:
            self.sleep(policy.min_interval_s - (now - last))
        self.last_request_at[host] = self.clock()


def backoff_seconds(policy: HostPolicy, attempt: int, retry_after: str | None) -> float:
    """Seconds to wait before the next attempt: Retry-After when it is a number, else base x 2^attempt (capped)."""
    if retry_after is not None and retry_after.strip().isdigit():
        return min(float(retry_after.strip()), MAX_BACKOFF_S)
    return min(policy.backoff_base_s * (2**attempt), MAX_BACKOFF_S)


def redact(text: str, secrets: Sequence[str]) -> str:
    """Replace every secret value in text with ***."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    return text


def to_result(cached: CachedResponse, from_cache: bool) -> FetchResult:
    """Convert a cache entry to a FetchResult."""
    return FetchResult(
        body=cached.body,
        sha256=cached.sha256,
        retrieved_at=cached.retrieved_at,
        url=cached.url,
        params=dict(cached.params),
        status_code=cached.status_code,
        content_type=cached.content_type,
        from_cache=from_cache,
    )
