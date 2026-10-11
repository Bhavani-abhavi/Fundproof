"""Tests for agent tracing: OpenInference attributes on node, tool and LLM spans, with PII redacted."""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agent.agent_helpers import committed_by_scenario, fake_handlers, make_deps, memory_agent
from uwcopilot.agent.graph import NODES
from uwcopilot.agent.runner import resume_run, start_run
from uwcopilot.observability.tracing import (
    AgentTracer,
    TracingConfigError,
    build_tracer_provider,
    check_no_implicit_otlp_env,
    make_otlp_exporter,
)

KIND = "openinference.span.kind"


def traced_run(tmp_path: Path) -> tuple[list, object]:
    """Run one application to completion with an in-memory exporter; return (spans, application)."""
    application = committed_by_scenario("watchlist_owner_1", line="credit")
    planted = application.ground_truth.watchlist_derived_owners[0].owner_id
    name = next(owner.name for owner in application.owners if owner.owner_id == planted)
    exporter = InMemorySpanExporter()
    tracer = AgentTracer(build_tracer_provider(exporter=exporter))
    agent = memory_agent(make_deps(tmp_path, handlers=fake_handlers({name}), tracer=tracer))
    start_run(agent, application.model_dump(mode="json"), "trace-thread", tracer)
    resume_run(
        agent,
        "trace-thread",
        {
            "action": "refer",
            "reviewer_id": "senior.one",
            "role": "senior_underwriter",
            "rationale": "Compliance must disposition the match first.",
        },
        tracer,
    )
    return list(exporter.get_finished_spans()), application


def test_every_node_tool_and_llm_call_has_a_span(tmp_path: Path) -> None:
    spans, _ = traced_run(tmp_path)
    kinds = {span.name: span.attributes.get(KIND) for span in spans}
    node_spans = {span.attributes["graph.node.id"] for span in spans if span.attributes.get(KIND) == "CHAIN"}
    assert node_spans == set(NODES)
    assert kinds["agent.start"] == kinds["agent.resume"] == "AGENT"
    tool_spans = [span for span in spans if span.attributes.get(KIND) == "TOOL"]
    assert {span.attributes["tool.name"] for span in tool_spans} == {"search_guidelines", "screen_entity", "score_risk"}
    assert all(span.attributes["session.id"] == "trace-thread" for span in spans if "session.id" in span.attributes)
    statuses = {span.attributes["tool.name"]: span.attributes["uwc.tool.status"] for span in tool_spans}
    assert statuses["score_risk"] == "unavailable"


def test_llm_span_has_model_tokens_cost_and_prompt_version(tmp_path: Path) -> None:
    spans, _ = traced_run(tmp_path)
    [llm] = [span for span in spans if span.attributes.get(KIND) == "LLM"]
    attributes = llm.attributes
    assert attributes["llm.provider"] == "fake" and attributes["llm.model_name"] == "fake-llm"
    assert attributes["llm.token_count.prompt"] > 0 and attributes["llm.token_count.completion"] > 0
    assert attributes["llm.token_count.total"] == (
        attributes["llm.token_count.prompt"] + attributes["llm.token_count.completion"]
    )
    assert attributes["llm.cost.total"] == 0.0
    assert attributes["uwc.fake_llm"] is True
    assert attributes["llm.prompt_template.version"].startswith("memo-prompt-v1-")


def test_spans_nest_under_the_run_and_interrupts_are_not_errors(tmp_path: Path) -> None:
    spans, _ = traced_run(tmp_path)
    root = next(span for span in spans if span.name == "agent.start")
    started = [span for span in spans if span.context.trace_id == root.context.trace_id]
    assert {span.name for span in started} >= {"node.intake", "node.run_tool", "tool.screen_entity", "llm.fake"}
    review = [span for span in spans if span.name == "node.human_review"]
    assert any(span.attributes.get("uwc.interrupted") for span in review)
    assert all(span.status.status_code != StatusCode.ERROR for span in spans)


def test_no_pii_reaches_any_span(tmp_path: Path) -> None:
    spans, application = traced_run(tmp_path)
    secrets = [application.applicant.legal_name, application.applicant.fein, application.bank_statements.account_number]
    secrets += [owner.name for owner in application.owners]
    secrets += [part for owner in application.owners if owner.entity_type == "person" for part in owner.name.split()]
    for span in spans:
        for value in span.attributes.values():
            text = str(value)
            for secret in secrets:
                assert secret.lower() not in text.lower(), (span.name, secret)


def test_otlp_exporter_is_configured_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = make_otlp_exporter("http://localhost:6006/v1/traces", environ={})
    assert isinstance(exporter, OTLPSpanExporter)
    with pytest.raises(TracingConfigError):
        check_no_implicit_otlp_env({"OTEL_EXPORTER_OTLP_HEADERS": "authorization=secret"})
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "x=y")
    with pytest.raises(TracingConfigError):
        make_otlp_exporter("http://localhost:6006/v1/traces")
    provider = build_tracer_provider("http://localhost:6006/v1/traces", environ={})
    processors = provider._active_span_processor._span_processors  # checking the wiring only
    assert any(isinstance(processor, BatchSpanProcessor) for processor in processors)
    provider.shutdown()


def test_no_endpoint_means_no_export() -> None:
    provider = build_tracer_provider(None)
    assert provider._active_span_processor._span_processors == ()
    with AgentTracer(None).span("x", {}) as span:
        assert not span.is_recording()


class CaptureOtlp(BaseHTTPRequestHandler):
    """A stand-in OTLP/HTTP collector that keeps every POST body."""

    received: list[tuple[str, str, bytes]] = []

    def do_POST(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
        """Store the request and answer 200."""
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        CaptureOtlp.received.append((self.path, self.headers.get("Content-Type", ""), body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        """Keep test output quiet."""


def test_spans_are_exported_over_otlp_http_without_pii(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureOtlp)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    CaptureOtlp.received = []
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1/traces"
        provider = build_tracer_provider(endpoint, environ={})
        tracer = AgentTracer(provider)
        application = committed_by_scenario("clean_term_dental")
        agent = memory_agent(make_deps(tmp_path, handlers=fake_handlers(), tracer=tracer))
        start_run(agent, application.model_dump(mode="json"), "otlp-thread", tracer)
        assert provider.force_flush(10_000)
        provider.shutdown()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert CaptureOtlp.received, "no OTLP export reached the collector"
    paths = {path for path, _, _ in CaptureOtlp.received}
    assert paths == {"/v1/traces"}
    assert all(content_type == "application/x-protobuf" for _, content_type, _ in CaptureOtlp.received)
    payload = b"".join(body for _, _, body in CaptureOtlp.received)
    assert b"node.intake" in payload and b"openinference.span.kind" in payload
    for owner in application.owners:
        assert owner.name.encode() not in payload
    assert application.applicant.legal_name.encode() not in payload
    assert application.applicant.fein.encode() not in payload
