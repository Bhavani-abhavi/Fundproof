"""Tests for uwcopilot.llm.settings: explicit configuration, never platform environment variables."""

from pathlib import Path

import pytest

from uwcopilot.llm import settings as llm_settings
from uwcopilot.llm.settings import (
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_OPENAI_BASE_URL,
    ImplicitSDKConfigError,
    SettingsError,
    check_no_implicit_sdk_env,
    load_llm_settings,
)


def test_defaults_when_nothing_is_set() -> None:
    settings = load_llm_settings(env_file=None, environ={})
    assert settings.provider == "anthropic"
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None
    assert settings.anthropic_base_url == DEFAULT_ANTHROPIC_BASE_URL
    assert settings.openai_base_url == DEFAULT_OPENAI_BASE_URL
    assert settings.max_run_cost_usd == 10.0
    assert settings.model_for("anthropic") == llm_settings.DEFAULT_MODELS["anthropic"]


def test_sdk_base_url_variables_are_ignored() -> None:
    environ = {
        "ANTHROPIC_BASE_URL": "https://platform-proxy.invalid",
        "OPENAI_BASE_URL": "https://other-proxy.invalid/v1",
        "ANTHROPIC_AUTH_TOKEN": "platform-token",
    }
    settings = load_llm_settings(env_file=None, environ=environ)
    assert settings.anthropic_base_url == DEFAULT_ANTHROPIC_BASE_URL
    assert settings.openai_base_url == DEFAULT_OPENAI_BASE_URL
    assert settings.anthropic_api_key is None


def test_project_base_url_variables_are_used() -> None:
    environ = {"UWC_ANTHROPIC_BASE_URL": "https://gateway.example.com/", "UWC_OPENAI_BASE_URL": "http://localhost:8080"}
    settings = load_llm_settings(env_file=None, environ=environ)
    assert settings.anthropic_base_url == "https://gateway.example.com"
    assert settings.openai_base_url == "http://localhost:8080"


def test_env_file_wins_over_process_environment(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ANTHROPIC_API_KEY=from-file\nUWC_LLM_PROVIDER=openai\nMAX_RUN_COST_USD=2.5\n", encoding="utf-8"
    )
    environ = {"ANTHROPIC_API_KEY": "from-process", "OPENAI_API_KEY": "process-only"}
    settings = load_llm_settings(env_file=env_file, environ=environ)
    assert settings.anthropic_api_key == "from-file"
    assert settings.openai_api_key == "process-only"
    assert settings.provider == "openai"
    assert settings.max_run_cost_usd == 2.5


def test_empty_values_count_as_unset(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=\nOPENAI_API_KEY=   \n", encoding="utf-8")
    settings = load_llm_settings(env_file=env_file, environ={})
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None


def test_api_keys_are_hidden_from_repr() -> None:
    settings = load_llm_settings(env_file=None, environ={"ANTHROPIC_API_KEY": "sk-secret-value"})
    assert "sk-secret-value" not in repr(settings)


def test_model_override_applies_only_to_the_configured_provider() -> None:
    settings = load_llm_settings(env_file=None, environ={"UWC_LLM_PROVIDER": "anthropic", "UWC_LLM_MODEL": "x-model"})
    assert settings.model_for("anthropic") == "x-model"
    assert settings.model_for("openai") == llm_settings.DEFAULT_MODELS["openai"]


@pytest.mark.parametrize(
    "environ",
    [
        {"UWC_LLM_PROVIDER": "mystery"},
        {"UWC_ANTHROPIC_BASE_URL": "http://insecure.example.com"},
        {"UWC_ANTHROPIC_BASE_URL": "http://localhost.evil.example/v1"},
        {"UWC_ANTHROPIC_BASE_URL": "http://127.0.0.1.nip.io"},
        {"UWC_ANTHROPIC_BASE_URL": "http://localhost@evil.example"},
        {"UWC_OPENAI_BASE_URL": "https://user:secret@gateway.example.com/v1"},
        {"UWC_OPENAI_BASE_URL": "api.openai.com/v1"},
        {"MAX_RUN_COST_USD": "ten"},
        {"MAX_RUN_COST_USD": "0"},
        {"UWC_ANTHROPIC_EFFORT": "extreme"},
    ],
)
def test_invalid_settings_raise(environ: dict[str, str]) -> None:
    with pytest.raises(SettingsError):
        load_llm_settings(env_file=None, environ=environ)


@pytest.mark.parametrize("url", ["http://localhost:8080/v1", "http://127.0.0.1:9", "http://[::1]:8080"])
def test_plain_http_is_allowed_only_for_loopback_hosts(url: str) -> None:
    assert load_llm_settings(env_file=None, environ={"UWC_OPENAI_BASE_URL": url}).openai_base_url == url


def test_anthropic_effort_defaults_to_low_and_can_be_set() -> None:
    assert load_llm_settings(env_file=None, environ={}).anthropic_effort == "low"
    assert load_llm_settings(env_file=None, environ={"UWC_ANTHROPIC_EFFORT": "high"}).anthropic_effort == "high"


def test_implicit_sdk_variables_are_refused() -> None:
    check_no_implicit_sdk_env(("OPENAI_ORG_ID",), environ={})
    with pytest.raises(ImplicitSDKConfigError, match="OPENAI_ORG_ID"):
        check_no_implicit_sdk_env(("OPENAI_ORG_ID",), environ={"OPENAI_ORG_ID": "org-123"})
