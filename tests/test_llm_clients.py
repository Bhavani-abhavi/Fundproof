"""Tests for the Anthropic and OpenAI clients against mock HTTP transports (no network, no real keys)."""

import pytest

from llm_test_helpers import RecordingTransport, anthropic_message_json, openai_response_json
from uwcopilot.llm.anthropic_client import AnthropicClient
from uwcopilot.llm.base import IncompleteLLMResponse, LLMClient, LLMMessage
from uwcopilot.llm.openai_client import OpenAIClient
from uwcopilot.llm.settings import ImplicitSDKConfigError, SettingsError

USER = [LLMMessage(role="user", content="hello")]


def test_anthropic_client_returns_only_text_blocks_and_usage() -> None:
    transport = RecordingTransport(anthropic_message_json("hello", input_tokens=12, output_tokens=3))
    client = AnthropicClient(
        "test-key", "https://api.anthropic.com", "claude-haiku-5-5", http_client=transport.client()
    )
    response = client.complete("system prompt", USER, max_tokens=100)
    assert response.text == "hello"
    assert (response.input_tokens, response.output_tokens) == (12, 3)
    assert response.provider == "anthropic"
    assert response.cost_usd is None
    assert not response.is_fake
    assert isinstance(client, LLMClient)


def test_anthropic_request_shape_omits_sampling_parameters() -> None:
    transport = RecordingTransport(anthropic_message_json())
    client = AnthropicClient(
        "test-key", "https://api.anthropic.com", "claude-haiku-5-5", http_client=transport.client()
    )
    client.complete("system prompt", USER, max_tokens=100, temperature=0.9)
    body = transport.last_json()
    assert body == {
        "model": "claude-haiku-5-5",
        "max_tokens": 100,
        "system": "system prompt",
        "messages": [{"role": "user", "content": "hello"}],
        "output_config": {"effort": "low"},
    }
    assert transport.requests[-1].headers["x-api-key"] == "test-key"


def test_anthropic_effort_is_explicit_and_validated() -> None:
    transport = RecordingTransport(anthropic_message_json())
    client = AnthropicClient(
        "test-key", "https://api.anthropic.com", "claude-haiku-5-5", http_client=transport.client(), effort="medium"
    )
    client.complete("sys", USER, max_tokens=4000)
    assert transport.last_json()["output_config"] == {"effort": "medium"}
    assert client.request_fingerprint == {"base_url": "https://api.anthropic.com", "effort": "medium"}
    with pytest.raises(SettingsError):
        AnthropicClient("test-key", "https://api.anthropic.com", "claude-haiku-5-5", effort="extreme")


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_anthropic_cut_off_or_refused_answers_are_not_returned_as_text(stop_reason: str) -> None:
    transport = RecordingTransport(
        anthropic_message_json("partial memo {", output_tokens=4000, stop_reason=stop_reason)
    )
    client = AnthropicClient(
        "test-key", "https://api.anthropic.com", "claude-haiku-5-5", http_client=transport.client()
    )
    with pytest.raises(IncompleteLLMResponse) as raised:
        client.complete("sys", USER, max_tokens=4000)
    assert raised.value.response.stop_reason == stop_reason
    assert raised.value.response.output_tokens == 4000
    assert "partial memo" not in str(raised.value)


def test_anthropic_base_url_env_does_not_change_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://platform-proxy.invalid")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "platform-token")
    transport = RecordingTransport(anthropic_message_json())
    client = AnthropicClient(
        "test-key", "https://api.anthropic.com", "claude-haiku-5-5", http_client=transport.client()
    )
    assert client.base_url == "https://api.anthropic.com"
    client.complete("sys", USER, max_tokens=10)
    request = transport.requests[-1]
    assert request.url.host == "api.anthropic.com"
    assert "authorization" not in request.headers


def test_anthropic_custom_headers_env_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Injected: 1")
    with pytest.raises(ImplicitSDKConfigError):
        AnthropicClient("test-key", "https://api.anthropic.com", "claude-haiku-5-5")


def test_anthropic_client_requires_key_and_base_url() -> None:
    with pytest.raises(ValueError):
        AnthropicClient("", "https://api.anthropic.com", "claude-haiku-5-5")
    with pytest.raises(ValueError):
        AnthropicClient("key", "", "claude-haiku-5-5")


def test_openai_client_uses_responses_api_and_ignores_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://platform-proxy.invalid/v1")
    transport = RecordingTransport(openai_response_json("hi there", input_tokens=9, output_tokens=2))
    client = OpenAIClient("test-key", "https://api.openai.com/v1", "gpt-6-luna", http_client=transport.client())
    response = client.complete("system prompt", USER, max_tokens=50, temperature=0.3)
    request = transport.requests[-1]
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == "Bearer test-key"
    body = transport.last_json()
    assert body["instructions"] == "system prompt"
    assert body["max_output_tokens"] == 50
    assert body["store"] is False
    assert "temperature" not in body
    assert response.text == "hi there"
    assert (response.input_tokens, response.output_tokens) == (9, 2)
    assert response.provider == "openai"


def test_openai_temperature_is_sent_only_when_supported() -> None:
    transport = RecordingTransport(openai_response_json())
    client = OpenAIClient(
        "test-key", "https://api.openai.com/v1", "gpt-6-luna", http_client=transport.client(), supports_temperature=True
    )
    client.complete("sys", USER, max_tokens=10, temperature=0.3)
    assert transport.last_json()["temperature"] == 0.3


def test_openai_incomplete_response_raises() -> None:
    transport = RecordingTransport(openai_response_json("cut", incomplete_reason="max_output_tokens"))
    client = OpenAIClient("test-key", "https://api.openai.com/v1", "gpt-6-luna", http_client=transport.client())
    with pytest.raises(IncompleteLLMResponse) as raised:
        client.complete("sys", USER, max_tokens=10)
    assert raised.value.response.stop_reason == "max_output_tokens"


def test_openai_missing_usage_is_an_error() -> None:
    body = openai_response_json()
    body["usage"] = None
    client = OpenAIClient(
        "test-key", "https://api.openai.com/v1", "gpt-6-luna", http_client=RecordingTransport(body).client()
    )
    with pytest.raises(RuntimeError, match="usage"):
        client.complete("sys", USER, max_tokens=10)


@pytest.mark.parametrize("name", ["OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_ADMIN_KEY", "OPENAI_CUSTOM_HEADERS"])
def test_openai_implicit_env_is_refused(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "something")
    with pytest.raises(ImplicitSDKConfigError):
        OpenAIClient("test-key", "https://api.openai.com/v1", "gpt-6-luna")
