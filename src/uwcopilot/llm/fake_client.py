"""A deterministic fake LLM for tests and offline demos. It never calls a network.

Its output is always labeled as fake: provider is "fake", the model name must start with "fake-", and
LLMResponse.is_fake is True. Code that reports results can check is_fake and refuse to present fake text as a
model's work.

Two ways to drive it:
- a list of scripted strings, returned in order (an error is raised when the list runs out);
- a callable (system, messages) -> str, for responses that depend on the input.
"""

from collections.abc import Callable, Sequence

from uwcopilot.llm.base import FAKE_PROVIDER, LLMMessage, LLMResponse

FakeResponder = Callable[[str, list[LLMMessage]], str]
FAKE_MODEL_PREFIX = "fake-"


class FakeLLMExhausted(RuntimeError):
    """Raised when a scripted FakeLLM is asked for more responses than it was given."""


def count_fake_tokens(text: str) -> int:
    """A deterministic stand-in for a tokenizer: the number of whitespace-separated words."""
    return len(text.split())


class FakeLLM:
    """LLMClient that returns scripted or computed text. Every call is recorded in self.calls."""

    provider = FAKE_PROVIDER

    def __init__(self, responses: Sequence[str] | FakeResponder, model: str = "fake-llm") -> None:
        """responses is either a list of strings (used in order) or a function of (system, messages)."""
        if not model.startswith(FAKE_MODEL_PREFIX):
            raise ValueError(f"Fake model names must start with {FAKE_MODEL_PREFIX!r} so output stays labeled.")
        self.model = model
        self._responder: FakeResponder | None = responses if callable(responses) else None
        self._scripted: list[str] = [] if callable(responses) else list(responses)
        self._next_index = 0
        self.calls: list[dict] = []

    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float = 0,
    ) -> LLMResponse:
        """Return the next scripted response (or the responder's output) and record the call."""
        self.calls.append(
            {"system": system, "messages": list(messages), "max_tokens": max_tokens, "temperature": temperature}
        )
        if self._responder is not None:
            text = self._responder(system, list(messages))
        else:
            if self._next_index >= len(self._scripted):
                raise FakeLLMExhausted(f"FakeLLM ran out of scripted responses after {len(self._scripted)} call(s).")
            text = self._scripted[self._next_index]
            self._next_index += 1
        prompt_text = system + "".join(message.content for message in messages)
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=count_fake_tokens(prompt_text),
            output_tokens=count_fake_tokens(text),
            latency_s=0.0,
            cost_usd=0.0,
            cached=False,
            provider=FAKE_PROVIDER,
            stop_reason="end_turn",
        )
