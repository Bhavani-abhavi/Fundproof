"""OpenTelemetry tracing for the agent, with OpenInference semantic-convention attributes.

One span per graph node (span kind CHAIN), per tool call (TOOL) and per LLM call (LLM). Attributes follow the
OpenInference conventions that Arize Phoenix and other OpenInference-aware backends read:
    openinference.span.kind, session.id, graph.node.id, graph.node.name, input.value, output.value,
    tool.name, tool.parameters, llm.provider, llm.model_name, llm.token_count.prompt / .completion / .total,
    llm.cost.total, llm.prompt_template.version
plus a few project attributes under "uwc." (latency, fake-LLM flag, tool status, interrupt flag).
Every input and output value is passed through a PiiRedactor first (see redaction.py).

Exporters:
- build_tracer_provider(otlp_endpoint=...) exports over OTLP/HTTP to that endpoint (set UWC_OTLP_ENDPOINT, for
  example a local Arize Phoenix at http://localhost:6006/v1/traces);
- with no endpoint, spans are created and dropped (nothing leaves the process);
- tests pass an InMemorySpanExporter.

Like the LLM clients, the exporter is configured only explicitly: the endpoint, headers (none), timeout,
compression and HTTP session are all passed in, and the session ignores proxy variables. The OTLP exporter would
still read a few OTEL_EXPORTER_OTLP_* variables (headers, certificates, credential provider) on its own, so
building one refuses to start while any of those is set. The global tracer provider is never changed.
"""

import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import requests
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Span, Status, StatusCode

from uwcopilot.llm.base import LLMResponse
from uwcopilot.observability.redaction import PiiRedactor

SERVICE_NAME = "uwcopilot-agent"
PROJECT_NAME = "uwcopilot"
TRACER_NAME = "uwcopilot.agent"
EXPORT_TIMEOUT_S = 10.0
IMPLICIT_OTLP_ENV_VARS: tuple[str, ...] = (
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_EXPORTER_OTLP_CERTIFICATE",
    "OTEL_EXPORTER_OTLP_TRACES_CERTIFICATE",
    "OTEL_EXPORTER_OTLP_CLIENT_KEY",
    "OTEL_EXPORTER_OTLP_TRACES_CLIENT_KEY",
    "OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE",
    "OTEL_EXPORTER_OTLP_TRACES_CLIENT_CERTIFICATE",
    "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER",
)
SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
LLM = OpenInferenceSpanKindValues.LLM.value
AGENT = OpenInferenceSpanKindValues.AGENT.value
JSON_MIME = "application/json"


class TracingConfigError(RuntimeError):
    """Raised when the OTLP exporter would pick up configuration implicitly from the environment."""


def check_no_implicit_otlp_env(environ: Mapping[str, str] | None = None) -> None:
    """Refuse to build an OTLP exporter while variables it would apply implicitly are set."""
    process_env = os.environ if environ is None else environ
    present = [name for name in IMPLICIT_OTLP_ENV_VARS if process_env.get(name)]
    if present:
        raise TracingConfigError(
            f"{', '.join(present)} would be applied implicitly by the OTLP exporter. This project configures "
            "tracing only explicitly (UWC_OTLP_ENDPOINT); unset them and try again."
        )


def make_otlp_exporter(endpoint: str, environ: Mapping[str, str] | None = None) -> OTLPSpanExporter:
    """An OTLP/HTTP span exporter configured only from its arguments."""
    check_no_implicit_otlp_env(environ)
    session = requests.Session()
    session.trust_env = False  # no proxy or netrc settings from the environment
    return OTLPSpanExporter(
        endpoint=endpoint,
        headers={},
        timeout=EXPORT_TIMEOUT_S,
        compression=Compression.NoCompression,
        session=session,
    )


def build_tracer_provider(
    otlp_endpoint: str | None = None,
    exporter: SpanExporter | None = None,
    environ: Mapping[str, str] | None = None,
) -> TracerProvider:
    """A TracerProvider for the agent. exporter (tests) wins over otlp_endpoint; neither means no export."""
    resource = Resource({"service.name": SERVICE_NAME, "openinference.project.name": PROJECT_NAME})
    provider = TracerProvider(resource=resource)
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    elif otlp_endpoint:
        provider.add_span_processor(BatchSpanProcessor(make_otlp_exporter(otlp_endpoint, environ)))
    return provider


class AgentTracer:
    """Creates the agent's spans with redacted inputs and outputs."""

    def __init__(self, provider: TracerProvider | None = None) -> None:
        """provider None gives a tracer whose spans are not recorded anywhere."""
        self.provider = provider
        self.tracer = provider.get_tracer(TRACER_NAME) if provider is not None else trace.NoOpTracer()

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any]) -> Iterator[Span]:
        """A span that records an exception as an error. Interrupts are handled by the caller."""
        with self.tracer.start_as_current_span(
            name, attributes=attributes, record_exception=False, set_status_on_exception=False
        ) as span:
            yield span

    @contextmanager
    def run_span(self, thread_id: str, application_id: str, phase: str) -> Iterator[Span]:
        """The root span of one graph invocation (start or resume)."""
        attributes = {
            SPAN_KIND: AGENT,
            SpanAttributes.SESSION_ID: thread_id,
            SpanAttributes.AGENT_NAME: "underwriting_agent",
            "uwc.application_id": application_id,
            "uwc.run_phase": phase,
        }
        with self.span(f"agent.{phase}", attributes) as span:
            yield span

    @contextmanager
    def node_span(self, node: str, thread_id: str, redactor: PiiRedactor, node_input: Any) -> Iterator[Span]:
        """A span for one graph node, with its (redacted) input."""
        attributes = {
            SPAN_KIND: CHAIN,
            SpanAttributes.SESSION_ID: thread_id,
            SpanAttributes.GRAPH_NODE_ID: node,
            SpanAttributes.GRAPH_NODE_NAME: node,
            SpanAttributes.INPUT_VALUE: redactor.to_json(node_input),
            SpanAttributes.INPUT_MIME_TYPE: JSON_MIME,
        }
        with self.span(f"node.{node}", attributes) as span:
            yield span

    @contextmanager
    def tool_span(
        self, tool: str, result_id: str, thread_id: str, redactor: PiiRedactor, arguments: Any
    ) -> Iterator[Span]:
        """A span for one tool call, with its (redacted) parameters."""
        attributes = {
            SPAN_KIND: TOOL,
            SpanAttributes.SESSION_ID: thread_id,
            SpanAttributes.TOOL_NAME: tool,
            SpanAttributes.TOOL_ID: result_id,
            SpanAttributes.TOOL_PARAMETERS: redactor.to_json(arguments),
            SpanAttributes.INPUT_VALUE: redactor.to_json(arguments),
            SpanAttributes.INPUT_MIME_TYPE: JSON_MIME,
        }
        with self.span(f"tool.{tool}", attributes) as span:
            yield span

    @contextmanager
    def llm_span(
        self,
        provider: str,
        model: str,
        prompt_version: str,
        thread_id: str,
        redactor: PiiRedactor,
        prompt: Any,
    ) -> Iterator[Span]:
        """A span for one LLM call, with the (redacted) prompt. Call record_llm_response() when it returns."""
        attributes = {
            SPAN_KIND: LLM,
            SpanAttributes.SESSION_ID: thread_id,
            SpanAttributes.LLM_PROVIDER: provider,
            SpanAttributes.LLM_MODEL_NAME: model,
            SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION: prompt_version,
            SpanAttributes.INPUT_VALUE: redactor.to_json(prompt),
            SpanAttributes.INPUT_MIME_TYPE: JSON_MIME,
        }
        with self.span(f"llm.{provider}", attributes) as span:
            yield span


def set_output(span: Span, redactor: PiiRedactor, value: Any) -> None:
    """Attach a redacted output value to a span."""
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, redactor.to_json(value))
    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, JSON_MIME)


def record_error(span: Span, error: BaseException) -> None:
    """Mark a span as failed with the error type (the message may hold data, so it is not recorded)."""
    span.set_status(Status(StatusCode.ERROR, type(error).__name__))
    span.set_attribute("uwc.error_type", type(error).__name__)


def record_llm_response(span: Span, redactor: PiiRedactor, response: LLMResponse) -> None:
    """Token counts, cost, latency and the (redacted) output of one LLM call."""
    span.set_attribute(SpanAttributes.LLM_MODEL_NAME, response.model)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, response.input_tokens)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, response.output_tokens)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, response.input_tokens + response.output_tokens)
    if response.cost_usd is not None:
        span.set_attribute(SpanAttributes.LLM_COST_TOTAL, response.cost_usd)
    if response.stop_reason:
        span.set_attribute(SpanAttributes.LLM_FINISH_REASON, response.stop_reason)
    span.set_attribute("uwc.latency_s", response.latency_s)
    span.set_attribute("uwc.llm_cached", response.cached)
    span.set_attribute("uwc.fake_llm", response.is_fake)
    set_output(span, redactor, response.text)


def elapsed_ms(started: float) -> float:
    """Milliseconds since a time.perf_counter() reading."""
    return round((time.perf_counter() - started) * 1000.0, 3)
