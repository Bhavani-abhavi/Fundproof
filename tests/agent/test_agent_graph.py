"""End-to-end tests of the LangGraph agent with the fake LLM and offline tools."""

import json
from pathlib import Path

from agent_helpers import (
    SENIOR_REFER,
    committed_application,
    committed_by_scenario,
    fake_handlers,
    make_deps,
    memory_agent,
    template_llm,
)

from uwcopilot.agent.decision_record import DecisionLog
from uwcopilot.agent.fake_memo import template_memo_responder
from uwcopilot.agent.runner import resume_run, start_run
from uwcopilot.agent.vocab import FAKE_LLM_STAMP
from uwcopilot.llm.budget import BudgetExceeded
from uwcopilot.llm.fake_client import FakeLLM

UNDERWRITER_APPROVE = {
    "action": "approve",
    "reviewer_id": "uw.one",
    "role": "underwriter",
    "rationale": "Policy checks pass and the memo is supported.",
}


def payload(application) -> dict:
    """The application as the JSON a caller would submit (ground truth included, as on disk)."""
    return application.model_dump(mode="json")


def test_clean_application_waits_for_a_human_and_records_the_decision(tmp_path: Path) -> None:
    application = committed_by_scenario("clean_term_dental")
    deps = make_deps(tmp_path)
    agent = memory_agent(deps)
    outcome = start_run(agent, payload(application), "thread-1")
    assert outcome.status == "awaiting_review"
    review = outcome.review_request
    assert review["reference_recommendation"] == "approve"
    assert review["memo_status"] == "passed"
    assert review["stamp"] == FAKE_LLM_STAMP and FAKE_LLM_STAMP in review["memo"]["summary"]
    assert outcome.state["decision_record"] is None
    assert DecisionLog(tmp_path / "decision_records.jsonl").records() == []  # nothing is decided without a human

    done = resume_run(agent, "thread-1", UNDERWRITER_APPROVE)
    assert done.status == "completed"
    assert done.state["status"] == "approved_by_human"
    log = DecisionLog(tmp_path / "decision_records.jsonl")
    [record] = log.records()
    assert log.verify_chain() == []
    assert record["application_id"] == application.application_id
    assert record["reviewer"]["role"] == "underwriter" and record["reviewer"]["is_override"] is False
    assert record["fake_llm"] is True and record["stamp"] == FAKE_LLM_STAMP
    assert record["versions"]["prompt_version"].startswith("memo-prompt-v1-")
    assert len(record["tool_calls"]) == 7 and record["inputs_sha256"] == done.state["inputs_sha256"]


def test_tools_run_with_sequential_result_ids_and_unavailable_risk_is_reported(tmp_path: Path) -> None:
    application = committed_by_scenario("watchlist_owner_1", line="credit")
    agent = memory_agent(make_deps(tmp_path))
    state = start_run(agent, payload(application), "t").state
    ids = sorted(int(result["result_id"].split("-")[1]) for result in state["tool_results"])
    assert ids == list(range(1, len(ids) + 1))
    statuses = {(result["tool"], result["status"]) for result in state["tool_results"]}
    assert ("score_risk", "unavailable") in statuses and ("screen_entity", "ok") in statuses
    rules = {item["rule"] for item in state["policy_result"]["not_evaluated"]}
    assert "risk_model_pd" in rules


def test_ground_truth_never_reaches_the_agent(tmp_path: Path) -> None:
    llm = template_llm()
    application = committed_by_scenario("clean_equipment_machine_shop")
    state = start_run(memory_agent(make_deps(tmp_path, llm=llm)), payload(application), "t").state
    assert "ground_truth" not in state["application"]
    prompt = llm.calls[0]["messages"][0].content
    assert application.ground_truth.scenario not in prompt
    assert application.ground_truth.rationale not in prompt


def test_refused_decision_keeps_the_run_waiting(tmp_path: Path) -> None:
    application = committed_by_scenario("prohibited_industry")
    agent = memory_agent(make_deps(tmp_path))
    start_run(agent, payload(application), "t")
    underwriter_decline = {**UNDERWRITER_APPROVE, "action": "decline", "rationale": "Prohibited industry per NBC-2.2."}
    waiting = resume_run(agent, "t", underwriter_decline)
    assert waiting.status == "awaiting_review"
    assert waiting.review_request["previous_rejections"][0]["errors"]
    done = resume_run(agent, "t", {**underwriter_decline, "role": "senior_underwriter"})
    assert done.status == "completed" and done.state["status"] == "declined_by_human"
    [record] = DecisionLog(tmp_path / "decision_records.jsonl").records()
    assert len(record["rejected_reviewer_attempts"]) == 1


def failing_then(responder_after: int) -> FakeLLM:
    """A fake LLM that answers with an unusable memo `responder_after` times, then with the template memo."""
    calls = {"n": 0}

    def respond(system: str, messages: list) -> str:
        calls["n"] += 1
        if calls["n"] <= responder_after:
            return json.dumps({"recommendation": "approve", "summary": "Looks fine.", "reasons": []})
        return template_memo_responder(system, messages)

    return FakeLLM(respond)


def test_failed_memo_is_regenerated_once_with_feedback(tmp_path: Path) -> None:
    llm = failing_then(1)
    application = committed_by_scenario("nsf_events")
    state = start_run(memory_agent(make_deps(tmp_path, llm=llm)), payload(application), "t").state
    assert state["memo_status"] == "passed" and state["memo_attempts"] == 2
    assert len(llm.calls) == 2
    feedback = llm.calls[1]["messages"][-1].content
    assert "recommendation_mismatch" in feedback and "no_reasons" in feedback


def test_memo_failing_twice_goes_to_the_human_with_its_problems(tmp_path: Path) -> None:
    llm = failing_then(10)
    application = committed_by_scenario("nsf_events")
    outcome = start_run(memory_agent(make_deps(tmp_path, llm=llm)), payload(application), "t")
    assert outcome.status == "awaiting_review"
    assert outcome.review_request["memo_status"] == "failed_check"
    assert len(llm.calls) == 2
    kinds = {issue["kind"] for issue in outcome.review_request["memo_check"]["issues"]}
    assert "recommendation_mismatch" in kinds
    assert outcome.review_request["reference_recommendation"] == "refer"


def test_invalid_json_is_retried(tmp_path: Path) -> None:
    answers = iter(["this is not json"])

    def respond(system: str, messages: list) -> str:
        return next(answers, None) or template_memo_responder(system, messages)

    state = start_run(
        memory_agent(make_deps(tmp_path, llm=FakeLLM(respond))), payload(committed_application("APP-C-001")), "t"
    ).state
    assert state["memo_status"] == "passed" and state["memo_attempts"] == 2


def test_llm_failure_or_budget_refusal_still_reaches_the_human(tmp_path: Path) -> None:
    def over_budget(system: str, messages: list) -> str:
        raise BudgetExceeded("worst-case cost would exceed MAX_RUN_COST_USD")

    for llm in (FakeLLM([]), FakeLLM(over_budget)):
        agent = memory_agent(make_deps(tmp_path, llm=llm))
        outcome = start_run(agent, payload(committed_application("APP-C-001")), "t")
        assert outcome.status == "awaiting_review"
        assert outcome.review_request["memo_status"] == "llm_unavailable"
        assert outcome.review_request["memo"] is None
        assert resume_run(agent, "t", SENIOR_REFER).status == "completed"


def test_failed_screening_blocks_approval(tmp_path: Path) -> None:
    def broken_screen(request: object) -> object:
        raise ConnectionError("list service down")

    handlers = {**fake_handlers(), "screen_entity": broken_screen}
    agent = memory_agent(make_deps(tmp_path, handlers=handlers))
    outcome = start_run(agent, payload(committed_by_scenario("clean_term_dental")), "t")
    assert outcome.review_request["reference_recommendation"] == "refer"
    assert outcome.review_request["review_queue"] == "Compliance"
    still = resume_run(
        agent,
        "t",
        {**UNDERWRITER_APPROVE, "role": "senior_underwriter", "override_reason": "referral_resolved_in_writing"},
    )
    assert still.status == "awaiting_review"


def test_risk_score_feeds_the_policy_when_the_service_answers(tmp_path: Path) -> None:
    handlers = fake_handlers(prediction=0.2)
    outcome = start_run(
        memory_agent(make_deps(tmp_path, handlers=handlers)), payload(committed_by_scenario("clean_term_dental")), "t"
    )
    rules = {item["rule"] for item in outcome.review_request["escalations"]}
    assert "model_pd_above_threshold" in rules
    assert outcome.review_request["memo_status"] == "passed"


def test_insurance_flagged_owner_goes_to_compliance(tmp_path: Path) -> None:
    application = committed_by_scenario("watchlist_owner_1", line="insurance")
    planted = application.ground_truth.watchlist_derived_owners[0].owner_id
    name = next(owner.name for owner in application.owners if owner.owner_id == planted)
    deps = make_deps(tmp_path, handlers=fake_handlers(flagged_names={name}))
    outcome = start_run(memory_agent(deps), payload(application), "t")
    assert outcome.review_request["review_queue"] == "Compliance"
    assert outcome.review_request["blocking_open"] is True
    assert outcome.review_request["allowed_actions"] == ["refer", "request_more_information"]
