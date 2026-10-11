"""Tests for uwcopilot.llm.factory: clear errors without keys, cache and budget wired around real clients."""

from pathlib import Path

import pytest

from llm_test_helpers import RecordingTransport, anthropic_message_json, openai_response_json
from uwcopilot.common.paths import CONFIG_DIR
from uwcopilot.llm.base import LLMMessage
from uwcopilot.llm.budget import BudgetExceeded, CostTracker, load_prices
from uwcopilot.llm.factory import MissingAPIKeyError, get_llm_client
from uwcopilot.llm.settings import LLMSettings, load_llm_settings

USER = [LLMMessage(role="user", content="hello")]


def make_settings(tmp_path: Path, **overrides: object) -> LLMSettings:
    """Settings that use the repo price list and a temporary cache folder."""
    values = {"cache_dir": tmp_path / "llm_cache", "pricing_file": CONFIG_DIR / "llm_pricing.yaml"}
    values.update(overrides)
    return LLMSettings(**values)


@pytest.mark.parametrize(("provider", "key_name"), [("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")])
def test_missing_key_raises_a_clear_error(tmp_path: Path, provider: str, key_name: str) -> None:
    with pytest.raises(MissingAPIKeyError, match=key_name):
        get_llm_client(provider, settings=make_settings(tmp_path))


def test_unknown_provider_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        get_llm_client("mystery", settings=make_settings(tmp_path))


def test_fake_provider_needs_no_key(tmp_path: Path) -> None:
    client = get_llm_client("fake", settings=make_settings(tmp_path), fake_responses=["scripted"])
    response = client.complete("sys", USER, max_tokens=10)
    assert response.text == "scripted"
    assert response.is_fake
    assert response.cost_usd == 0.0


def test_anthropic_client_is_cached_and_costed(tmp_path: Path) -> None:
    transport = RecordingTransport(anthropic_message_json(input_tokens=12, output_tokens=3))
    settings = make_settings(tmp_path, anthropic_api_key="test-key")
    client = get_llm_client("anthropic", settings=settings, http_client=transport.client())
    first = client.complete("sys", USER, max_tokens=100)
    second = client.complete("sys", USER, max_tokens=100)
    assert len(transport.requests) == 1
    assert first.cost_usd == pytest.approx((12 * 0.10 + 3 * 0.50) / 1e6)
    assert second.cached is True
    assert second.cost_usd == 0.0


def test_factory_ignores_platform_base_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://platform-proxy.invalid")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    loaded = load_llm_settings(env_file=None)
    settings = make_settings(tmp_path, anthropic_api_key=loaded.anthropic_api_key)
    assert loaded.anthropic_base_url == "https://api.anthropic.com"
    transport = RecordingTransport(anthropic_message_json())
    client = get_llm_client("anthropic", settings=settings, http_client=transport.client(), use_cache=False)
    client.complete("sys", USER, max_tokens=10)
    assert transport.requests[-1].url.host == "api.anthropic.com"


def test_shared_tracker_stops_the_run_before_the_request(tmp_path: Path) -> None:
    transport = RecordingTransport(openai_response_json())
    settings = make_settings(tmp_path, openai_api_key="test-key")
    tracker = CostTracker(0.00001, load_prices(settings.pricing_file))
    client = get_llm_client("openai", settings=settings, http_client=transport.client(), cost_tracker=tracker)
    with pytest.raises(BudgetExceeded):
        client.complete("sys", USER, max_tokens=10_000)
    assert transport.requests == []
