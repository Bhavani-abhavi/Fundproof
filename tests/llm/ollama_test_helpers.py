"""A pretend Ollama server for tests: answers /api/version, /api/tags and /api/chat through httpx.MockTransport."""

import json
from typing import Any

import httpx

TEST_MODEL = "qwen-test"
TEST_DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
TEST_VERSION = "0.40.3"


def tags_json(model: str = TEST_MODEL, digest: str = TEST_DIGEST) -> dict[str, Any]:
    """A /api/tags body listing one model under its 'name:latest' form, as Ollama does."""
    return {
        "models": [
            {
                "name": f"{model}:latest",
                "model": f"{model}:latest",
                "digest": digest,
                "details": {
                    "format": "gguf",
                    "family": "qwen2",
                    "parameter_size": "3.1B",
                    "quantization_level": "Q4_K_M",
                },
            }
        ]
    }


def chat_json(
    content: str = "The minimum is in [NBC-1.1].",
    prompt_eval_count: int | None = 20,
    eval_count: int | None = 5,
    done_reason: str = "stop",
) -> dict[str, Any]:
    """A non-streaming /api/chat body with the fields Ollama 0.40.3 returns."""
    body: dict[str, Any] = {
        "model": TEST_MODEL,
        "created_at": "2026-10-12T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "total_duration": 7_000_000_000,
        "load_duration": 6_000_000_000,
        "prompt_eval_cached_count": 0,
        "prompt_eval_duration": 500_000_000,
        "eval_duration": 100_000_000,
    }
    if prompt_eval_count is not None:
        body["prompt_eval_count"] = prompt_eval_count
    if eval_count is not None:
        body["eval_count"] = eval_count
    return body


class FakeOllamaServer:
    """Routes requests by path, records them, and answers with configurable JSON."""

    def __init__(
        self,
        chat: dict[str, Any] | None = None,
        chat_status: int = 200,
        tags: dict[str, Any] | None = None,
        version: str = TEST_VERSION,
    ) -> None:
        """chat/chat_status answer POST /api/chat; tags answers GET /api/tags."""
        self.chat = chat if chat is not None else chat_json()
        self.chat_status = chat_status
        self.tags = tags if tags is not None else tags_json()
        self.version = version
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Record the request and answer by path."""
        self.requests.append(request)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": self.version})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=self.tags)
        if request.url.path == "/api/chat":
            return httpx.Response(self.chat_status, json=self.chat)
        return httpx.Response(404, json={"error": "not found"})

    def client(self) -> httpx.Client:
        """An httpx.Client whose requests go to this pretend server only."""
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def chat_bodies(self) -> list[dict[str, Any]]:
        """The JSON bodies of every /api/chat request so far."""
        return [json.loads(request.content) for request in self.requests if request.url.path == "/api/chat"]
