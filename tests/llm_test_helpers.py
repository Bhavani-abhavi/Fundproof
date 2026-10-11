"""Shared test doubles for the LLM tests: mock HTTP transports and a pretend paid client. No network access."""

import json
from collections.abc import Callable
from typing import Any

import httpx2

from uwcopilot.llm.base import LLMMessage, LLMResponse, raise_if_incomplete


def anthropic_message_json(
    text: str = "hello", input_tokens: int = 12, output_tokens: int = 3, stop_reason: str = "end_turn"
) -> dict[str, Any]:
    """A minimal Messages API response body, including an (empty) thinking block that must be ignored."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": text},
        ],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def openai_response_json(
    text: str = "hi there", input_tokens: int = 9, output_tokens: int = 2, incomplete_reason: str | None = None
) -> dict[str, Any]:
    """A minimal Responses API response body (status "incomplete" when incomplete_reason is given)."""
    body: dict[str, Any] = {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6-luna",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_test",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    if incomplete_reason is not None:
        body["status"] = "incomplete"
        body["incomplete_details"] = {"reason": incomplete_reason}
    return body


class RecordingTransport:
    """Builds an httpx2.Client whose requests are answered by a function and recorded, never sent anywhere."""

    def __init__(self, body: dict[str, Any] | Callable[[httpx2.Request], dict[str, Any]]) -> None:
        """body is a fixed JSON response body or a function of the request."""
        self.body = body
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        """Record the request and answer it with JSON."""
        self.requests.append(request)
        payload = self.body(request) if callable(self.body) else self.body
        return httpx2.Response(200, json=payload)

    def client(self) -> httpx2.Client:
        """An httpx2.Client that uses this transport."""
        return httpx2.Client(transport=httpx2.MockTransport(self.handler))

    def last_json(self) -> dict[str, Any]:
        """The JSON body of the most recent request."""
        return json.loads(self.requests[-1].content)


class PretendPaidClient:
    """Looks like a paid provider to the budget layer but returns canned token counts without any network."""

    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        input_tokens: int = 1000,
        output_tokens: int = 500,
        stop_reason: str = "end_turn",
    ) -> None:
        """Token counts and the stop reason are what the pretend API reports for every call."""
        self.provider = "anthropic"
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.stop_reason = stop_reason
        self.calls = 0

    def complete(self, system: str, messages: list[LLMMessage], max_tokens: int, temperature: float = 0) -> LLMResponse:
        """Count the call and return a fixed response (raising like a real client when it is incomplete)."""
        self.calls += 1
        response = LLMResponse(
            text="ok",
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_s=0.0,
            provider=self.provider,
            stop_reason=self.stop_reason,
        )
        return raise_if_incomplete(response)
