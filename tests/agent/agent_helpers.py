"""Shared helpers for the agent tests. Everything runs offline: no data/raw, no models, no network.

- applications: the committed eval/applications files, or fresh ones built with the generator over a small
  fictional watchlist (FIXTURE_POOL);
- tool handlers: a term-overlap guideline search over the committed corpus, a screening handler that flags a
  given set of names, and a fixed risk score;
- deps/agents: AgentDeps with a fixed clock, a temporary decision log, and an optional in-memory span exporter.
"""

import json
import random
import re
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from faker import Faker
from langgraph.checkpoint.memory import InMemorySaver

from uwcopilot.agent.applications import CreditApplication, InsuranceApplication, parse_application
from uwcopilot.agent.decision_record import DecisionLog
from uwcopilot.agent.fake_memo import template_memo_responder
from uwcopilot.agent.graph import AgentDeps
from uwcopilot.agent.runner import compile_agent
from uwcopilot.agent.synthetic_apps import GenerationContext, WatchlistOrg, shuffled
from uwcopilot.agent.synthetic_credit import CLEAN as CLEAN_CREDIT
from uwcopilot.agent.synthetic_credit import build_credit_application
from uwcopilot.agent.synthetic_insurance import CLEAN as CLEAN_INSURANCE
from uwcopilot.agent.synthetic_insurance import build_insurance_application
from uwcopilot.common.paths import REPO_ROOT
from uwcopilot.llm.base import LLMMessage
from uwcopilot.llm.fake_client import FakeLLM
from uwcopilot.observability.tracing import AgentTracer
from uwcopilot.rag.guidelines import load_guideline_chunks
from uwcopilot.tools.schemas import (
    GuidelineHit,
    GuidelineQuery,
    GuidelineSearchResult,
    ListSnapshot,
    RiskScoreRequest,
    RiskScoreResponse,
    ScreeningCandidate,
    ScreeningRequest,
    ScreeningResult,
)

APPLICATIONS_DIR = REPO_ROOT / "eval" / "applications"
FIXED_TIME = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
FAKE_SHA = "c" * 64
FIXTURE_POOL: list[WatchlistOrg] = [
    WatchlistOrg("OFAC-T1", "AURORA FICTIONAL SHIPPING COMPANY", "IR"),
    WatchlistOrg("OFAC-T2", "BLUEFIN TEST TRADING HOUSE", None),
    WatchlistOrg("OFAC-T3", "CORVUS PRETEND METALS GROUP", "SY"),
    WatchlistOrg("OFAC-T4", "DELTA IMAGINARY PETROCHEMICAL WORKS", "VE"),
    WatchlistOrg("OFAC-T5", "EMBER MADEUP LOGISTICS", "KP"),
    WatchlistOrg("OFAC-T6", "FALCON SYNTHETIC MARINE SERVICES", None),
]
SENIOR_REFER = {
    "action": "refer",
    "reviewer_id": "senior.one",
    "role": "senior_underwriter",
    "rationale": "Refer to the named function for its review.",
}


def fixed_clock() -> datetime:
    """A clock that always returns the same time, so runs can be compared exactly."""
    return FIXED_TIME


def committed_application(application_id: str) -> CreditApplication | InsuranceApplication:
    """One application from eval/applications (committed to the repo)."""
    path = APPLICATIONS_DIR / f"{application_id}.json"
    return parse_application(json.loads(path.read_text(encoding="utf-8")))


def committed_by_scenario(scenario: str, line: str | None = None) -> CreditApplication | InsuranceApplication:
    """The committed application built for a scenario name."""
    for path in sorted(APPLICATIONS_DIR.glob("APP-*.json")):
        application = parse_application(json.loads(path.read_text(encoding="utf-8")))
        if application.ground_truth.scenario == scenario and (line is None or application.line == line):
            return application
    raise KeyError(scenario)


def fixture_context(seed: int = 7) -> GenerationContext:
    """A seeded generation context over the fictional watchlist."""
    fake = Faker("en_US")
    fake.seed_instance(seed)
    return GenerationContext(rng=random.Random(seed), fake=fake, watchlist=shuffled(FIXTURE_POOL, random.Random(seed)))


def credit_app(**changes: Any) -> CreditApplication:
    """A credit application built from the clean scenario with some targets changed."""
    from dataclasses import replace

    spec = replace(CLEAN_CREDIT, **changes)
    application = parse_application(build_credit_application(spec, "APP-C-900", fixture_context()))
    assert isinstance(application, CreditApplication)
    return application


def insurance_app(**changes: Any) -> InsuranceApplication:
    """An insurance submission built from the clean scenario with some inputs changed."""
    from dataclasses import replace

    spec = replace(CLEAN_INSURANCE, **changes)
    application = parse_application(build_insurance_application(spec, "APP-I-900", fixture_context()))
    assert isinstance(application, InsuranceApplication)
    return application


def edited(application: CreditApplication | InsuranceApplication, **fields: Any) -> dict[str, Any]:
    """The application as a dict with some top-level fields replaced."""
    data = application.model_dump(mode="json")
    data.update(fields)
    return data


# --------------------------------------------------------------------------------------------------------------
# Tool handlers
# --------------------------------------------------------------------------------------------------------------


def words(text: str) -> set[str]:
    """Lowercase words of three or more letters."""
    return {word for word in re.findall(r"[a-z]{3,}", text.lower())}


def corpus_search_handler() -> Callable[[GuidelineQuery], GuidelineSearchResult]:
    """Offline guideline search: rank the committed corpus sections by word overlap with the query."""
    chunks = load_guideline_chunks()

    def search(query: GuidelineQuery) -> GuidelineSearchResult:
        """Top-k sections by overlap, restricted to the requested documents."""
        query_words = words(query.query)
        allowed = [chunk for chunk in chunks if query.source_docs is None or chunk.doc_id in query.source_docs]
        scored = sorted(allowed, key=lambda chunk: (-len(query_words & words(chunk.text)), chunk.chunk_id))
        hits = [
            GuidelineHit(
                section_id=chunk.chunk_id,
                title=chunk.title,
                source_doc=chunk.doc_id,
                text=chunk.text,
                score=float(len(query_words & words(chunk.text))),
                rank=rank,
            )
            for rank, chunk in enumerate(scored[: query.top_k], start=1)
        ]
        return GuidelineSearchResult(query=query, hits=hits)

    return search


def screen_handler(flagged_names: Iterable[str] = ()) -> Callable[[ScreeningRequest], ScreeningResult]:
    """Screening that returns a potential match for exactly the given names."""
    flagged = set(flagged_names)
    snapshot = ListSnapshot(source="OFAC", file_name="sdn.csv", sha256=FAKE_SHA, snapshot_date="2026-10-01")

    def screen(request: ScreeningRequest) -> ScreeningResult:
        """Flag the configured names, clear everything else."""
        candidates = []
        if request.name in flagged:
            candidates.append(
                ScreeningCandidate(
                    list_entry_id="OFAC-T1",
                    list_name="OFAC SDN (TEST)",
                    source="OFAC",
                    matched_name="AURORA FICTIONAL SHIPPING COMPANY",
                    score=0.9712,
                    explanation=["test fixture match"],
                )
            )
        return ScreeningResult(
            request=request,
            candidates=candidates,
            status="potential_match" if candidates else "clear",
            list_snapshots=[snapshot],
        )

    return screen


def risk_handler(prediction: float = 0.03) -> Callable[[RiskScoreRequest], RiskScoreResponse]:
    """A risk service stand-in that always returns the given prediction."""

    def score(request: RiskScoreRequest) -> RiskScoreResponse:
        """A fixed score with one reason code."""
        return RiskScoreResponse(
            product=request.product,
            prediction=prediction,
            risk_band="E" if prediction > 0.12 else "B",
            reason_codes=[{"feature": "requested_amount", "contribution": 0.0123, "description": "loan size"}],
            model_version="test-model-1",
            scenario=request.scenario,
            unknown_features=[],
        )

    return score


def fake_handlers(flagged_names: Iterable[str] = (), prediction: float | None = None) -> dict[str, Callable]:
    """Offline handlers for every allowed tool (risk unavailable unless a prediction is given)."""
    handlers: dict[str, Callable] = {
        "search_guidelines": corpus_search_handler(),
        "screen_entity": screen_handler(flagged_names),
    }
    if prediction is not None:
        handlers["score_risk"] = risk_handler(prediction)
    return handlers


# --------------------------------------------------------------------------------------------------------------
# LLMs and agents
# --------------------------------------------------------------------------------------------------------------


def template_llm() -> FakeLLM:
    """The deterministic template memo writer."""
    return FakeLLM(template_memo_responder)


def memo_json(recommendation: str, reasons: list[tuple[str, list[str]]], **extra: Any) -> str:
    """A memo as the LLM would return it."""
    body = {
        "recommendation": recommendation,
        "summary": "Test memo.",
        "reasons": [{"text": text, "citations": citations} for text, citations in reasons],
    }
    body.update(extra)
    return json.dumps(body)


def make_deps(
    tmp_path: Path,
    llm: Any | None = None,
    handlers: dict[str, Callable] | None = None,
    tracer: AgentTracer | None = None,
) -> AgentDeps:
    """AgentDeps for tests: fake LLM and tools, fixed clock, decision log in tmp_path."""
    return AgentDeps(
        llm=llm or template_llm(),
        tool_handlers=handlers if handlers is not None else fake_handlers(),
        decision_log=DecisionLog(tmp_path / "decision_records.jsonl"),
        tracer=tracer or AgentTracer(),
        clock=fixed_clock,
    )


def memory_agent(deps: AgentDeps) -> Any:
    """The compiled graph with an in-memory checkpointer."""
    return compile_agent(deps, InMemorySaver())


def last_user_message(messages: list[LLMMessage]) -> str:
    """The text of the first user message (the memo prompt)."""
    return messages[0].content
