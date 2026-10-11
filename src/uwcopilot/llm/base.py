"""The provider-neutral types every LLM client uses.

Callers only ever see LLMMessage, LLMResponse and the LLMClient protocol, so swapping Anthropic for OpenAI
(or for the fake used in tests) does not change any calling code.
"""

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

FAKE_PROVIDER = "fake"
# Stop reasons that mean the text is not a finished answer: the token limit cut it off (Anthropic "max_tokens",
# OpenAI "max_output_tokens") or the model declined (Anthropic "refusal", OpenAI "content_filter").
INCOMPLETE_STOP_REASONS: frozenset[str] = frozenset({"max_tokens", "max_output_tokens", "refusal", "content_filter"})


class LLMMessage(BaseModel):
    """One turn of a conversation. The system prompt is passed separately, never as a message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    """What a client returns for one completion.

    cost_usd is None when the model has no verified price in config/llm_pricing.yaml ("not measured").
    cached is True when the response came from the on-disk cache; a cached response cost nothing in this run,
    so its cost_usd is 0.0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_s: float = Field(ge=0)
    cost_usd: float | None = None
    cached: bool = False
    provider: str
    stop_reason: str | None = None

    @property
    def is_fake(self) -> bool:
        """True when the text was produced by the deterministic fake client, not by a real model."""
        return self.provider == FAKE_PROVIDER


class IncompleteLLMResponse(RuntimeError):
    """The model stopped before finishing (token limit or refusal), so its text must not be used as an answer.

    The response is kept on the exception so the budget layer can still record what the call cost.
    """

    def __init__(self, response: LLMResponse) -> None:
        """Keep the incomplete response; the message names the stop reason, never the text."""
        super().__init__(
            f"{response.provider}:{response.model} stopped with {response.stop_reason!r} after "
            f"{response.output_tokens} output tokens; the partial text was not used."
        )
        self.response = response


def raise_if_incomplete(response: LLMResponse) -> LLMResponse:
    """Return the response, or raise IncompleteLLMResponse when its stop reason means it is not finished."""
    if response.stop_reason in INCOMPLETE_STOP_REASONS:
        raise IncompleteLLMResponse(response)
    return response


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can turn a system prompt plus messages into an LLMResponse."""

    provider: str
    model: str

    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float = 0,
    ) -> LLMResponse:
        """Return one completion. Implementations must not retry silently beyond their SDK's own retries."""
        ...
