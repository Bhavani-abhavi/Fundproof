"""Tests for /api/health, /api/config, the OpenAPI docs and unknown API paths."""

import sys

import pytest
from fastapi.testclient import TestClient

from uwcopilot.web.app import SWAGGER_CSS_SRI, SWAGGER_CSS_URL, SWAGGER_JS_SRI, SWAGGER_JS_URL, create_app
from uwcopilot.web.routers.system import BUSINESS_TYPES_NOT_AVAILABLE, METROS_NOT_AVAILABLE
from uwcopilot.web.settings import DISCLAIMER


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["version"] and "git_sha" in body


def test_health_reports_the_git_sha_when_known(make_settings) -> None:
    with TestClient(create_app(make_settings(UWC_GIT_SHA="abc1234"))) as test_client:
        assert test_client.get("/api/health").json()["git_sha"] == "abc1234"


def test_config_shape(client: TestClient) -> None:
    body = client.get("/api/config").json()
    assert body["product_name"] == "Fundproof"
    assert body["disclaimer"] == DISCLAIMER
    assert body["sides"] == ["founder", "lender"]
    assert body["data_labels"]["synthetic"] == "SYNTHETIC"
    assert body["roles"] == ["founder", "lender_reviewer", "lender_senior", "admin"]
    assert body["hf_oauth"] is False and body["dev_auth"] is False
    assert body["payments_enabled"] is False
    assert body["is_space"] is False and body["space_url"] is None
    for key in ("metros", "metros_note", "business_types", "business_types_note"):
        assert key in body


def test_config_says_when_dev_login_is_on(dev_client: TestClient) -> None:
    assert dev_client.get("/api/config").json()["dev_auth"] is True


def test_config_lists_studio_metros_when_they_load(client: TestClient) -> None:
    body = client.get("/api/config").json()
    if body["metros_note"] is not None:  # the Studio package is still being built; covered by the next test
        pytest.skip("Studio metro configs do not load in this checkout")
    modes = {metro["mode"] for metro in body["metros"]}
    assert modes <= {"full", "limited", "synthetic"}
    synthetic = [metro for metro in body["metros"] if metro["data_label"] == "SYNTHETIC"]
    assert all(metro["mode"] == "synthetic" for metro in synthetic)


def test_config_falls_back_when_studio_is_not_importable(make_settings, monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes `import` raise ImportError, like a Studio module that does not exist yet.
    monkeypatch.setitem(sys.modules, "uwcopilot.studio.metros", None)
    monkeypatch.setitem(sys.modules, "uwcopilot.studio.profiles", None)
    with TestClient(create_app(make_settings())) as test_client:
        body = test_client.get("/api/config").json()
    assert body["metros"] == [] and body["metros_note"] == METROS_NOT_AVAILABLE
    assert body["business_types"] == [] and body["business_types_note"] == BUSINESS_TYPES_NOT_AVAILABLE


def test_unknown_api_path_is_a_json_404(client: TestClient) -> None:
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_openapi_lists_the_shell_endpoints(client: TestClient) -> None:
    paths = client.get("/api/openapi.json").json()["paths"]
    for path in ("/api/health", "/api/config", "/api/me", "/api/auth/csrf", "/api/auth/login", "/api/auth/logout"):
        assert path in paths
    assert "/api/auth/dev-login" in paths


def test_docs_page_has_its_own_csp(client: TestClient) -> None:
    response = client.get("/api/docs")
    assert response.status_code == 200 and "swagger" in response.text.lower()
    csp = response.headers["content-security-policy"]
    script_src = csp.split("script-src")[1].split(";")[0]
    assert "'sha256-" in script_src and "'unsafe-inline'" not in script_src
    # Only the two pinned files are allowed, never the whole CDN host.
    assert SWAGGER_JS_URL in script_src and "https://cdn.jsdelivr.net " not in csp + " "
    assert SWAGGER_CSS_URL in csp.split("style-src")[1].split(";")[0]


def test_docs_page_pins_swagger_ui_with_subresource_integrity(client: TestClient) -> None:
    html = client.get("/api/docs").text
    assert f'src="{SWAGGER_JS_URL}" integrity="{SWAGGER_JS_SRI}" crossorigin="anonymous"' in html
    assert f'href="{SWAGGER_CSS_URL}" integrity="{SWAGGER_CSS_SRI}" crossorigin="anonymous"' in html
    assert "@5/" not in html  # no floating major version
    assert "fastapi.tiangolo.com" not in html  # no third-party favicon
