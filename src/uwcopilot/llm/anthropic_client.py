"""Anthropic (Claude) client using the official anthropic SDK and the Messages API.

The API key and base URL are always passed explicitly. Passing them stops the SDK from reading
ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, profiles, or ANTHROPIC_BASE_URL from the environment.

Sampling parameters: current Claude models reject temperature, top_p and top_k (the Claude Haiku 5.5 model page
says "Omit temperature, top_p, and top_k"), and anthropic SDK 1.x no longer accepts them on messages.create.
So `temperature` is part of the shared interface (and of the cache key) but is not sent to Anthropic.

Thinking and effort: current models think adaptively by default (Claude Haiku 5.5: adaptive thinking on, effort
default "medium"), and thinking tokens count toward max_tokens. So every request sends an explicit effort
(output_config={"effort": ...}, default "low", set with UWC_ANTHROPIC_EFFORT) and callers use max_tokens budgets
of several thousand tokens. Only text blocks are returned; thinking blocks are dropped.

Stop reasons: a response that stopped on "max_tokens" (cut off) or "refusal" (declined) raises
IncompleteLLMResponse instead of returning partial or empty text. The budget layer still records its cost.
None of this has run against the real API in this repo (no key); see the "not measured" sections in the docs.
"""

import time
from typing import Any

import anthropic

from uwcopilot.llm.base import LLMMessage, LLMResponse, raise_if_incomplete
from uwcopilot.llm.settings import DEFAULT_ANTHROPIC_EFFORT, check_no_implicit_sdk_env, validate_effort

PROVIDER = "anthropic"
# Read by the anthropic SDK even when api_key and base_url are passed explicitly.
IMPLICIT_SDK_ENV_VARS: tuple[str, ...] = ("ANTHROPIC_CUSTOM_HEADERS",)
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 2


class AnthropicClient:
    """LLMClient for Claude models."""

    provider = PROVIDER

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        http_client: Any | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        effort: str = DEFAULT_ANTHROPIC_EFFORT,
    ) -> None:
        """Create the SDK client. http_client is for tests (an httpx2.Client with a mock transport)."""
        if not api_key:
            raise ValueError("AnthropicClient needs a non-empty api_key.")
        if not base_url:
            raise ValueError("AnthropicClient needs an explicit base_url.")
        check_no_implicit_sdk_env(IMPLICIT_SDK_ENV_VARS)
        self.model = model
        self.effort = validate_effort(effort)
        self._sdk = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            timeout=timeout_s,
            max_retries=max_retries,
        )

    @property
    def base_url(self) -> str:
        """The base URL the SDK will send requests to, without a trailing slash."""
        return str(self._sdk.base_url).rstrip("/")

    @property
    def request_fingerprint(self) -> dict[str, str]:
        """Settings that change the answer without being part of the prompt (part of the LLM cache key)."""
        return {"base_url": self.base_url, "effort": self.effort}

    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float = 0,
    ) -> LLMResponse:
        """Send one Messages API request and return the concatenated text blocks.

        Raises IncompleteLLMResponse when the model stopped on max_tokens or refused.
        """
        started = time.perf_counter()
        message = self._sdk.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": item.role, "content": item.content} for item in messages],
            output_config={"effort": self.effort},
        )
        latency = time.perf_counter() - started
        text_parts = [block.text for block in message.content if block.type == "text"]
        response = LLMResponse(
            text="".join(text_parts),
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            latency_s=latency,
            cost_usd=None,
            cached=False,
            provider=PROVIDER,
            stop_reason=message.stop_reason,
        )
        return raise_if_incomplete(response)
