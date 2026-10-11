"""Tests for the evaluation metrics (deterministic, no LLM)."""

from agent_helpers import APPLICATIONS_DIR, committed_by_scenario, fake_handlers, make_deps, memory_agent

from uwcopilot.agent.applications import load_applications
from uwcopilot.agent.evaluation import (
    corpus_trigger_ids,
    evaluate_policy_case,
    path_case,
    rate,
    summarize_injection_detector,
    summarize_llm,
    summarize_path,
    summarize_policy,
)
from uwcopilot.agent.runner import start_run


def test_rate_reports_mean_interval_and_n() -> None:
    result = rate([True, False, True, True])
    assert result["value"] == 0.75 and result["n"] == 4
    assert 0.0 <= result["ci_low"] <= 0.75 <= result["ci_high"] <= 1.0
    assert rate([]) == {"value": None, "ci_low": None, "ci_high": None, "n": 0}


def test_latency_summary_has_percentile_intervals() -> None:
    from uwcopilot.agent.evaluation import latency_summary

    summary = latency_summary([float(value) for value in range(1, 101)])
    assert summary["p50"] == 50.5 and summary["n"] == 100
    assert summary["p50_ci"][0] <= 50.5 <= summary["p50_ci"][1]
    assert summary["p95_ci"][0] <= summary["p95"] <= summary["p95_ci"][1]
    assert latency_summary([]) == {"p50": None, "p95": None, "n": 0}


def test_corpus_has_31_referral_triggers() -> None:
    triggers = corpus_trigger_ids()
    assert len(triggers) == 31
    assert triggers[0] == "R-HM-01" and "R-NBC-14" in triggers


def test_policy_summary_on_the_committed_set() -> None:
    applications = load_applications(APPLICATIONS_DIR)
    summary = summarize_policy([evaluate_policy_case(application) for application in applications])
    assert summary["recommendation_agreement"]["n"] == 65
    coverage = summary["referral_trigger_coverage"]
    assert coverage["triggers_in_corpus"] == 31
    assert set(coverage["exercised"]) | set(coverage["not_exercised"]) == set(corpus_trigger_ids())
    detector = summarize_injection_detector(applications)
    assert detector["recall"]["n"] == 7 and detector["false_positive_rate"]["n"] == 58


def test_path_summary_counts_escalations(tmp_path) -> None:
    applications = [
        committed_by_scenario("clean_term_dental"),
        committed_by_scenario("injection_ignore_and_approve"),
        committed_by_scenario("watchlist_owner_1", line="credit"),
    ]
    planted = applications[2].ground_truth.watchlist_derived_owners[0].owner_id
    name = next(owner.name for owner in applications[2].owners if owner.owner_id == planted)
    agent = memory_agent(make_deps(tmp_path, handlers=fake_handlers({name})))
    cases = []
    for index, application in enumerate(applications):
        outcome = start_run(agent, application.model_dump(mode="json"), f"t{index}")
        cases.append(path_case(application, outcome.review_request, outcome.state, 1.0))
    summary = summarize_path(cases, applications)
    assert summary["reached_human_checkpoint"]["value"] == 1.0
    assert summary["decisions_recorded_without_a_human"] == 0
    assert summary["escalation"]["sanctions_potential_matches_escalated"] == rate([True])
    assert summary["escalation"]["detected_injections_escalated"]["value"] == 1.0
    assert summary["screening"]["planted_watchlist_owners_flagged"]["value"] == 1.0
    llm = summarize_llm(cases)
    assert llm["citation_accuracy"]["value"] == 1.0 and llm["unsupported_claim_rate"]["value"] == 0.0
    assert llm["cost_usd_per_application"]["value"] == 0.0
    assert llm["llm_calls"]["served_from_cache"] == 0
    # A cached call cost nothing in this run: its application is left out of cost and latency, not counted as $0.
    for call in cases[0].llm_calls:
        call["cached"] = True
    with_cache = summarize_llm(cases)
    assert with_cache["llm_calls"]["served_from_cache"] == len(cases[0].llm_calls)
    assert with_cache["llm_calls"]["applications_excluded_from_cost_and_latency"] == 1
    assert with_cache["cost_usd_per_application"]["n"] == len(cases) - 1
    assert with_cache["llm_latency_ms_per_application"]["n"] == len(cases) - 1
    for case in cases:
        for call in case.llm_calls:
            call["cached"] = True
    assert summarize_llm(cases)["cost_usd_per_application"].startswith("not measured")
