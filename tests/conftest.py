"""Shared pytest setup.

Unit tests must not depend on the machine they run on. Two kinds of variables are removed for every test:

- SDK variables that the anthropic and openai SDKs read even when every setting is passed explicitly (the LLM
  clients refuse to start while those are set);
- the project's own settings (every UWC_* name, MAX_RUN_COST_USD and the two API keys), so a developer who exports
  for example UWC_ANTHROPIC_BASE_URL for a corporate gateway still gets the documented defaults in tests.

Tests that check how a variable is handled set it again with monkeypatch or pass an explicit environ mapping.
"""

import os

import pytest

from uwcopilot.llm.anthropic_client import IMPLICIT_SDK_ENV_VARS as ANTHROPIC_IMPLICIT_ENV
from uwcopilot.llm.openai_client import IMPLICIT_SDK_ENV_VARS as OPENAI_IMPLICIT_ENV

PROJECT_SETTING_NAMES: tuple[str, ...] = ("MAX_RUN_COST_USD", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")


@pytest.fixture(autouse=True)
def clear_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset SDK variables that would be applied implicitly, and the project's own settings variables."""
    project_names = [name for name in os.environ if name.startswith("UWC_")] + list(PROJECT_SETTING_NAMES)
    for name in ANTHROPIC_IMPLICIT_ENV + OPENAI_IMPLICIT_ENV + tuple(project_names):
        monkeypatch.delenv(name, raising=False)
