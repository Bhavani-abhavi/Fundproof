"""Tests for the request protections: headers, CSRF, rate limits, body size, CORS and the redacted access log."""

import base64
import hashlib
import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from web_helpers import csrf_headers, dev_login

from uwcopilot.web.app import create_app
from uwcopilot.web.security import RateLimiter, build_csp, client_ip, inline_script_hashes


def test_security_headers_on_api_and_error_responses(client: TestClient) -> None:
    for response in (client.get("/api/health"), client.get("/api/nope"), client.post("/api/auth/logout")):
        headers = response.headers
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "strict-origin-when-cross-origin"
        assert "camera=()" in headers["permissions-policy"]
        csp = headers["content-security-policy"]
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'self' https://huggingface.co" in csp
        assert "x-frame-options" not in headers  # frame-ancestors replaces it (the Space embeds the app)


def test_default_csp_never_allows_inline_scripts() -> None:
    script_src = build_csp().split("script-src ")[1].split(";")[0]
    assert script_src == "'self'"
    assert "worker-src 'self' blob:" in build_csp()


def test_inline_script_hashes_match_what_browsers_compute() -> None:
    script = 'self.__next_f.push([1,"hello </div>"])'
    html = f'<html><script src="/_next/a.js"></script><script>{script}</script><script>{script}</script></html>'
    expected = "sha256-" + base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert inline_script_hashes(html) == [expected]  # external scripts skipped, duplicates listed once


# ---- CSRF ----


def test_csrf_is_required_for_state_changing_api_requests(dev_client: TestClient) -> None:
    body = {"username": "demo", "role": "founder"}
    assert dev_client.post("/api/auth/dev-login", json=body).status_code == 403
    csrf_headers(dev_client)  # the cookie is now set, but the header is still missing
    assert dev_client.post("/api/auth/dev-login", json=body).status_code == 403
    wrong = {"X-CSRF-Token": "A" * 43}
    assert dev_client.post("/api/auth/dev-login", json=body, headers=wrong).status_code == 403
    assert dev_client.post("/api/auth/dev-login", json=body, headers=csrf_headers(dev_client)).status_code == 200


# ---- rate limits ----


def test_ip_rate_limit_returns_429_with_retry_after(make_settings) -> None:
    settings = make_settings(UWC_RATE_LIMIT_IP_BURST="3", UWC_RATE_LIMIT_IP_PER_MIN="1")
    with TestClient(create_app(settings)) as test_client:
        statuses = [test_client.get("/api/me").status_code for _ in range(4)]
        assert statuses == [200, 200, 200, 429]
        limited = test_client.get("/api/me")
        assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1
        assert limited.headers["x-content-type-options"] == "nosniff"
        assert test_client.get("/api/health").status_code == 200  # the health probe is never limited


def test_user_rate_limit_is_separate_from_the_ip_limit(make_settings) -> None:
    settings = make_settings(UWC_DEV_AUTH="1", UWC_RATE_LIMIT_USER_BURST="2", UWC_RATE_LIMIT_USER_PER_MIN="1")
    with TestClient(create_app(settings)) as test_client:
        dev_login(test_client)  # signed-out requests count only against the IP bucket
        statuses = [test_client.get("/api/me").status_code for _ in range(3)]
        assert statuses == [200, 200, 429]


def test_rate_limiter_refills_over_time() -> None:
    now = [0.0]
    limiter = RateLimiter(per_minute=60, burst=2, clock=lambda: now[0])
    assert limiter.check("ip:a") == 0.0 and limiter.check("ip:a") == 0.0
    assert limiter.check("ip:a") == pytest.approx(1.0)  # one token per second
    assert limiter.check("ip:b") == 0.0  # other keys have their own bucket
    now[0] += 1.0
    assert limiter.check("ip:a") == 0.0


def test_rate_limiter_forgets_idle_buckets() -> None:
    now = [0.0]
    limiter = RateLimiter(per_minute=60, burst=1, clock=lambda: now[0], max_keys=3)
    for key in ("a", "b", "c"):
        limiter.check(key)
    now[0] += 10.0  # every bucket has refilled
    limiter.check("d")
    assert len(limiter) == 1


@pytest.mark.parametrize(
    ("forwarded", "hops", "expected"),
    [
        ("203.0.113.9", 1, "203.0.113.9"),
        ("1.1.1.1, 203.0.113.9", 1, "203.0.113.9"),  # the left entry was written by the client
        ("1.1.1.1, 203.0.113.9, 10.0.0.2", 2, "203.0.113.9"),
        ("203.0.113.9", 0, "10.9.9.9"),  # no trusted proxy: the socket address
        ("", 1, "10.9.9.9"),
    ],
)
def test_client_ip_uses_only_trusted_forwarded_entries(forwarded: str, hops: int, expected: str) -> None:
    scope = {"type": "http", "headers": [(b"x-forwarded-for", forwarded.encode())], "client": ("10.9.9.9", 5000)}
    assert client_ip(scope, hops) == expected


# ---- body size ----


@pytest.fixture
def small_body_client(make_settings) -> Iterator[TestClient]:
    """Dev login on, 100-byte body limit."""
    with TestClient(create_app(make_settings(UWC_DEV_AUTH="1", UWC_MAX_BODY_BYTES="100"))) as test_client:
        yield test_client


def test_declared_body_over_the_limit_is_413(small_body_client: TestClient) -> None:
    headers = {**csrf_headers(small_body_client), "Content-Type": "application/json"}
    response = small_body_client.post("/api/auth/dev-login", content=b"x" * 101, headers=headers)
    assert response.status_code == 413


def test_streamed_body_over_the_limit_is_413(small_body_client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(5):
            yield b"y" * 40

    headers = {**csrf_headers(small_body_client), "Content-Type": "application/json"}
    response = small_body_client.post("/api/auth/dev-login", content=chunks(), headers=headers)
    assert response.status_code == 413


def test_body_under_the_limit_reaches_the_route(small_body_client: TestClient) -> None:
    body = {"username": "demo", "role": "founder"}
    response = small_body_client.post("/api/auth/dev-login", json=body, headers=csrf_headers(small_body_client))
    assert response.status_code == 200


# ---- CORS ----


def test_cors_allows_the_next_dev_server_only_with_dev_login(make_settings) -> None:
    preflight = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-CSRF-Token",
    }
    with TestClient(create_app(make_settings(UWC_DEV_AUTH="1"))) as dev:
        response = dev.options("/api/auth/logout", headers=preflight)
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
        other = dev.options("/api/auth/logout", headers={**preflight, "Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in other.headers
    with TestClient(create_app(make_settings())) as production_like:
        response = production_like.get("/api/me", headers={"Origin": "http://localhost:3000"})
        assert "access-control-allow-origin" not in response.headers


# ---- access log ----


def test_access_log_masks_personal_data(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="uwcopilot.web.access"):
        client.get("/api/me", params={"email": "jane.doe@example.com", "ssn": "123-45-6789"})
        client.get("/lookup/jane.doe%40example.com/123-45-6789")
    lines = [record.getMessage() for record in caplog.records if record.name == "uwcopilot.web.access"]
    assert lines[-2].startswith("GET /api/me?email=*&ssn=* 200 ")  # query values are never logged
    assert lines[-1].startswith("GET /lookup/[EMAIL]/[SSN] 404 ")  # identifiers in paths are masked
    assert not any("jane.doe" in line or "123-45-6789" in line for line in lines)
