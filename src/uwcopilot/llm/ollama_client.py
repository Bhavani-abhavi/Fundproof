"""Ollama client: open-weight models served on this machine through Ollama's native REST API.

Why the native API (POST /api/chat) and not Ollama's OpenAI-compatible endpoint: the native API takes the model
options per request (num_ctx, seed, num_predict, temperature, plus an optional JSON-schema `format`) and returns
the raw token counts and durations. The context size matters most: num_ctx decides how much prompt the model
can see, and the OpenAI-compatible endpoint has no per-request way to set it.

What the client does:
- At start-up it reads the server version (GET /api/version) and the model digest (GET /api/tags). The digest
  identifies the exact model weights, so it goes into the LLM cache key: re-importing a different model under
  the same name never serves an old cached answer. If an expected digest is given, a mismatch stops the run.
- Before each call, a context guard estimates the prompt size as est_upper = ceil(characters / 2) tokens. If
  est_upper + num_predict > num_ctx, the call is refused with ContextBudgetExceeded (Ollama would otherwise
  silently cut the start of the prompt). The estimate is a heuristic, not a guarantee, so after the call the
  client also sets metadata["possible_truncation"] when prompt_eval_count + eval_count >= num_ctx. Ollama may
  report a smaller prompt_eval_count when it reuses a cached prompt prefix; the count is recorded as reported.
- The response maps message.content to text, prompt_eval_count to input_tokens, eval_count to output_tokens and
  done_reason to stop_reason. A "length" stop (num_predict reached) raises IncompleteLLMResponse, like a
  max_tokens stop from the paid APIs. Raw durations, the digest and the Ollama version go into metadata.
- Local inference has no API charge, so cost_usd is 0.0. CPU time is reported separately (latency and durations).

Network: the client builds its own httpx.Client with trust_env=False, so proxy variables in the environment
(HTTPS_PROXY, ALL_PROXY) never receive local model traffic. The base URL must be https, or plain http to a
loopback host (settings.validate_base_url). There are no automatic retries: a failed call raises OllamaError.
"""

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import httpx

from uwcopilot.llm.base import LLMMessage, LLMResponse, raise_if_incomplete
from uwcopilot.llm.settings import validate_base_url

PROVIDER = "ollama"
DEFAULT_NUM_CTX = 4096
DEFAULT_SEED = 42
# One CPU-only server answers one request at a time; a long grounded answer under load can take minutes.
DEFAULT_TIMEOUT_S = 600.0
# The context guard's estimate: at most one token per CHARS_PER_TOKEN_ESTIMATE characters of prompt text.
CHARS_PER_TOKEN_ESTIMATE = 2
MAX_ERROR_TEXT_CHARS = 300


class OllamaError(RuntimeError):
    """The Ollama server could not be reached, answered with an error, or returned an unexpected body."""


class OllamaModelNotFound(OllamaError):
    """The requested model name is not installed on the Ollama server."""


class OllamaDigestMismatch(OllamaError):
    """The installed model's digest is not the one the configuration pinned."""


class ContextBudgetExceeded(RuntimeError):
    """The prompt estimate plus the output budget does not fit in num_ctx, so the call was not made."""

    def __init__(self, est_upper_tokens: int, num_predict: int, num_ctx: int) -> None:
        """Keep the three numbers so callers can record why the call was refused."""
        super().__init__(
            f"Prompt estimate {est_upper_tokens} tokens + num_predict {num_predict} > num_ctx {num_ctx}; "
            "the call was not made. Shorten the context (fewer or shorter passages) or raise num_ctx."
        )
        self.est_upper_tokens = est_upper_tokens
        self.num_predict = num_predict
        self.num_ctx = num_ctx


@dataclass(frozen=True)
class OllamaOptions:
    """Settings that change the answer but are not arguments of complete().

    temperature and num_predict come from complete(temperature=..., max_tokens=...). format_schema is an
    optional JSON schema that Ollama uses to constrain the output. expected_digest pins the model weights.
    """

    num_ctx: int = DEFAULT_NUM_CTX
    seed: int = DEFAULT_SEED
    format_schema: dict[str, Any] | None = None
    expected_digest: str | None = None

    def __post_init__(self) -> None:
        """Reject a context size that cannot hold any prompt."""
        if self.num_ctx <= 0:
            raise ValueError("num_ctx must be greater than zero.")


def canonical_json_sha256(value: Any) -> str:
    """SHA-256 of the canonical JSON form of a value (sorted keys, no extra whitespace)."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def strip_digest_prefix(digest: str) -> str:
    """'sha256:abc...' and 'abc...' name the same digest; keep only the hex part."""
    if digest.startswith("sha256:"):
        return digest[len("sha256:") :]
    return digest


def estimate_prompt_tokens(system: str, messages: list[LLMMessage]) -> int:
    """The context guard's prompt estimate: ceil(characters / 2) over the system prompt and all messages."""
    characters = len(system)
    for message in messages:
        characters += len(message.content)
    return math.ceil(characters / CHARS_PER_TOKEN_ESTIMATE)


def build_chat_body(
    model: str,
    system: str,
    messages: list[LLMMessage],
    max_tokens: int,
    temperature: float,
    options: OllamaOptions,
) -> dict[str, Any]:
    """The JSON body for POST /api/chat. An empty system prompt is not sent (the model's template decides)."""
    chat_messages: list[dict[str, str]] = []
    if system:
        chat_messages.append({"role": "system", "content": system})
    for message in messages:
        chat_messages.append({"role": message.role, "content": message.content})
    body: dict[str, Any] = {
        "model": model,
        "messages": chat_messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": options.seed,
            "num_ctx": options.num_ctx,
            "num_predict": max_tokens,
        },
    }
    if options.format_schema is not None:
        body["format"] = options.format_schema
    return body


def find_model_entry(tags: dict[str, Any], model: str) -> dict[str, Any] | None:
    """The /api/tags entry for a model name; 'name' and 'name:latest' are the same model in Ollama."""
    wanted = {model, f"{model}:latest"}
    for entry in tags.get("models") or []:
        if entry.get("name") in wanted or entry.get("model") in wanted:
            return entry
    return None


def error_text(reply: httpx.Response) -> str:
    """A short description of an error reply: Ollama's {"error": ...} message, or the start of the body."""
    try:
        payload = reply.json()
    except ValueError:
        return reply.text[:MAX_ERROR_TEXT_CHARS]
    if isinstance(payload, dict) and "error" in payload:
        return str(payload["error"])[:MAX_ERROR_TEXT_CHARS]
    return reply.text[:MAX_ERROR_TEXT_CHARS]


class OllamaClient:
    """LLMClient for models served by a local Ollama server (native API)."""

    provider = PROVIDER

    def __init__(
        self,
        base_url: str,
        model: str,
        options: OllamaOptions | None = None,
        http_client: httpx.Client | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        """Connect to the server, read its version and the model digest.

        base_url is explicit (settings.ollama_base_url, default http://127.0.0.1:11434). http_client is for tests
        (an httpx.Client with a MockTransport); timeout_s applies only to the client built here. Raises
        OllamaError when the server cannot be reached, OllamaModelNotFound when the model is not installed and
        OllamaDigestMismatch when options.expected_digest does not match.
        """
        if not model:
            raise ValueError("OllamaClient needs a model name.")
        self.base_url = validate_base_url(base_url)
        self.model = model
        self.options = options or OllamaOptions()
        self._http = http_client or httpx.Client(timeout=timeout_s, trust_env=False)
        self.runtime_version = self._read_version()
        entry = self._read_model_entry()
        self.digest = strip_digest_prefix(str(entry.get("digest", "")))
        self.model_details = self._pick_details(entry)
        expected = self.options.expected_digest
        if expected and strip_digest_prefix(expected) != self.digest:
            raise OllamaDigestMismatch(
                f"Model {model!r} on {self.base_url} has digest {self.digest}, but the configuration expects "
                f"{strip_digest_prefix(expected)}. Re-import the pinned model (scripts/fetch_local_model.py) or "
                "update the configuration through a migration review."
            )

    def _get_json(self, path: str) -> dict[str, Any]:
        """GET a JSON document from the server, turning transport and HTTP errors into OllamaError."""
        try:
            reply = self._http.get(f"{self.base_url}{path}")
        except httpx.TransportError as error:
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url} ({type(error).__name__}). Is `ollama serve` running?"
            ) from error
        if reply.status_code != 200:
            raise OllamaError(f"Ollama GET {path} returned HTTP {reply.status_code}: {error_text(reply)}")
        return reply.json()

    def _read_version(self) -> str:
        """The server's version string from GET /api/version."""
        return str(self._get_json("/api/version").get("version", "unknown"))

    def _read_model_entry(self) -> dict[str, Any]:
        """The /api/tags entry for this client's model, or OllamaModelNotFound."""
        tags = self._get_json("/api/tags")
        entry = find_model_entry(tags, self.model)
        if entry is None:
            installed = sorted(str(item.get("name")) for item in tags.get("models") or [])
            raise OllamaModelNotFound(
                f"Model {self.model!r} is not installed on {self.base_url} (installed: {installed}). "
                "Import it with scripts/fetch_local_model.py."
            )
        return entry

    @staticmethod
    def _pick_details(entry: dict[str, Any]) -> dict[str, str]:
        """The descriptive fields of a /api/tags entry that are worth recording (quantization, size, family)."""
        details = entry.get("details") or {}
        picked: dict[str, str] = {}
        for key in ("format", "family", "parameter_size", "quantization_level"):
            if key in details:
                picked[key] = str(details[key])
        return picked

    @property
    def format_sha256(self) -> str:
        """SHA-256 of the JSON schema sent as `format`, or '' when no schema is used."""
        if self.options.format_schema is None:
            return ""
        return canonical_json_sha256(self.options.format_schema)

    @property
    def request_fingerprint(self) -> dict[str, str]:
        """Settings that change the answer without being part of the prompt (part of the LLM cache key).

        temperature and num_predict (max_tokens) are already in the cache key as call parameters.
        """
        return {
            "base_url": self.base_url,
            "digest": self.digest,
            "num_ctx": str(self.options.num_ctx),
            "seed": str(self.options.seed),
            "format_sha256": self.format_sha256,
        }

    def model_record(self, license_text: str | None) -> dict[str, str | None]:
        """The model entry for a results provenance block (governance.provenance)."""
        return {
            "provider": PROVIDER,
            "model": self.model,
            "digest": self.digest,
            "license": license_text,
            "runtime_version": f"ollama {self.runtime_version}",
        }

    def check_context(self, system: str, messages: list[LLMMessage], max_tokens: int) -> int:
        """Return the prompt estimate, or raise ContextBudgetExceeded if estimate + max_tokens > num_ctx."""
        est_upper = estimate_prompt_tokens(system, messages)
        if est_upper + max_tokens > self.options.num_ctx:
            raise ContextBudgetExceeded(est_upper, max_tokens, self.options.num_ctx)
        return est_upper

    def complete(
        self,
        system: str,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float = 0,
    ) -> LLMResponse:
        """Send one POST /api/chat request (stream off) and return the answer.

        Raises ContextBudgetExceeded before the request when the prompt may not fit, OllamaError on a transport
        or HTTP error, and IncompleteLLMResponse when the model stopped because num_predict was reached.
        """
        est_upper = self.check_context(system, messages, max_tokens)
        body = build_chat_body(self.model, system, messages, max_tokens, temperature, self.options)
        started = time.perf_counter()
        try:
            reply = self._http.post(f"{self.base_url}/api/chat", json=body)
        except httpx.TransportError as error:
            raise OllamaError(
                f"Ollama chat request to {self.base_url} failed ({type(error).__name__}). Is `ollama serve` running?"
            ) from error
        latency = time.perf_counter() - started
        if reply.status_code != 200:
            raise OllamaError(f"Ollama POST /api/chat returned HTTP {reply.status_code}: {error_text(reply)}")
        try:
            data = reply.json()
        except ValueError as error:
            raise OllamaError("Ollama /api/chat reply is not JSON; the answer cannot be used.") from error
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaError("Ollama /api/chat reply has no message.content; the answer cannot be used.")

        prompt_eval_count = data.get("prompt_eval_count")
        eval_count = data.get("eval_count")
        used_tokens = (prompt_eval_count or 0) + (eval_count or 0)
        metadata: dict[str, Any] = {
            "ollama_version": self.runtime_version,
            "digest": self.digest,
            "model_details": dict(self.model_details),
            "options": dict(body["options"]),
            "format_sha256": self.format_sha256,
            "est_upper_prompt_tokens": est_upper,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_cached_count": data.get("prompt_eval_cached_count"),
            "eval_count": eval_count,
            "total_duration_ns": data.get("total_duration"),
            "load_duration_ns": data.get("load_duration"),
            "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
            "eval_duration_ns": data.get("eval_duration"),
            "possible_truncation": used_tokens >= self.options.num_ctx,
        }
        response = LLMResponse(
            text=message["content"],
            model=str(data.get("model") or self.model),
            input_tokens=prompt_eval_count or 0,
            output_tokens=eval_count or 0,
            latency_s=latency,
            cost_usd=0.0,
            cached=False,
            provider=PROVIDER,
            stop_reason=data.get("done_reason"),
            metadata=metadata,
        )
        return raise_if_incomplete(response)

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._http.close()
