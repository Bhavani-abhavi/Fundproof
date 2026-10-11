"""OpenAI client using the official openai SDK and the Responses API.

The API key and base URL are always passed explicitly, so OPENAI_BASE_URL in the environment is ignored.
The SDK also reads OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_ADMIN_KEY and OPENAI_CUSTOM_HEADERS on its own;
the client refuses to start while any of those is set (see settings.check_no_implicit_sdk_env).

temperature is only sent when supports_temperature=True, because some OpenAI models reject a temperature
parameter. Responses are not stored on OpenAI's side (store=False).

A response that is incomplete because of the output-token limit or a content filter raises IncompleteLLMResponse
instead of returning partial text (the budget layer still records its cost).
"""

import time
from typing import Any

import openai

from uwcopilot.llm.base import LLMMessage, LLMResponse, raise_if_incomplete
from uwcopilot.llm.settings import check_no_implicit_sdk_env

PROVIDER = "openai"
IMPLICIT_SDK_ENV_VARS: tuple[str, ...] = (
    "OPENAI_ORG_ID",
    "OPENAI_PROJECT_ID",
    "OPENAI_ADMIN_KEY",
    "OPENAI_CUSTOM_HEADERS",
)
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 2


class OpenAIClient:
    """LLMClient for OpenAI models."""

    provider = PROVIDER

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        http_client: Any | None = None,
        supports_temperature: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """Create the SDK client. http_client is for tests (an httpx2.Client with a mock transport)."""
        if not api_key:
            raise ValueError("OpenAIClient needs a non-empty api_key.")
        if not base_url:
            raise ValueError("OpenAIClient needs an explicit base_url.")
        check_no_implicit_sdk_env(IMPLICIT_SDK_ENV_VARS)
        self.model = model
        self.supports_temperature = supports_temperature
        self._sdk = openai.OpenAI(
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
        return {"base_url": self.base_url, "supports_temperature": str(self.supports_temperature)}

    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float = 0,
    ) -> LLMResponse:
        """Send one Responses API request and return its output text (IncompleteLLMResponse if cut off)."""
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": [{"role": item.role, "content": item.content} for item in messages],
            "max_output_tokens": max_tokens,
            "store": False,
        }
        if self.supports_temperature:
            request["temperature"] = temperature
        started = time.perf_counter()
        response = self._sdk.responses.create(**request)
        latency = time.perf_counter() - started
        usage = response.usage
        if usage is None:
            # Without token counts the cost of the call is unknown and the run's cost cap cannot be enforced.
            raise RuntimeError(f"OpenAI response {response.id} has no usage block; cannot account for its cost.")
        stop_reason = response.status
        if response.incomplete_details is not None and response.incomplete_details.reason:
            stop_reason = response.incomplete_details.reason
        result = LLMResponse(
            text=response.output_text,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_s=latency,
            cost_usd=None,
            cached=False,
            provider=PROVIDER,
            stop_reason=stop_reason,
        )
        return raise_if_incomplete(result)
