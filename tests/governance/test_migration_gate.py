"""The LLM migration gate: non-inferiority decisions, refusals, the margins file and the certificate.

All data here is SYNTHETIC paired per-question data built in the test; no model is called.
"""

import json
from pathlib import Path
from typing import Any

import pytest

import llm_migration_gate
from uwcopilot.common.files import sha256_of_file
from uwcopilot.common.paths import CONFIG_DIR
from uwcopilot.governance.migration_gate import (
    GateConfigError,
    GatedMetric,
    MigrationSettings,
    check_same_inputs,
    compare_metric,
    load_migration_settings,
    non_inferiority_decision,
    overall_decision,
    parse_migration_settings,
    render_certificate,
)
from uwcopilot.rag.generation_eval import QuestionScore

SETTINGS = MigrationSettings(metrics=(), resamples=1000, seed=42, confidence=0.95)
RECALL = GatedMetric("key_fact_recall", "higher_is_better", 0.05, "keyed")
FABRICATED = GatedMetric("answers_with_fabricated_citation", "lower_is_better", 0.02, "all")


# ---------------------------------------------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ci", "direction", "expected"),
    [
        ((-0.05, 0.02), "higher_is_better", "PASS"),  # the bound equal to -m passes
        ((-0.20, -0.06), "higher_is_better", "FAIL"),
        ((-0.10, 0.01), "higher_is_better", "INCONCLUSIVE"),
        ((-0.01, 0.02), "lower_is_better", "PASS"),
        ((0.03, 0.10), "lower_is_better", "FAIL"),
        ((0.00, 0.05), "lower_is_better", "INCONCLUSIVE"),
    ],
)
def test_non_inferiority_decision(ci: tuple[float, float], direction: str, expected: str) -> None:
    margin = 0.05 if direction == "higher_is_better" else 0.02
    assert non_inferiority_decision(ci[0], ci[1], margin, direction) == expected


def test_overall_is_pass_only_when_every_metric_passes() -> None:
    assert overall_decision(["PASS", "PASS"]) == "PASS"
    assert overall_decision(["PASS", "INCONCLUSIVE"]) == "INCONCLUSIVE"
    assert overall_decision(["INCONCLUSIVE", "FAIL"]) == "FAIL"
    assert overall_decision([]) == "INCONCLUSIVE"


def ids(n: int) -> list[str]:
    """Question ids Q001.. for synthetic paired data."""
    return [f"Q{index:03d}" for index in range(1, n + 1)]


def test_identical_runs_pass() -> None:
    values = {question_id: (index % 3) / 2 for index, question_id in enumerate(ids(60))}
    block = compare_metric(RECALL, values, dict(values), SETTINGS)
    assert block["paired_diff"] == {"mean": 0.0, "ci95": [0.0, 0.0]} and block["decision"] == "PASS"


def test_clearly_worse_challenger_fails() -> None:
    champion = {question_id: 1.0 for question_id in ids(60)}
    challenger = {question_id: (0.0 if index % 2 else 1.0) for index, question_id in enumerate(ids(60))}
    block = compare_metric(RECALL, champion, challenger, SETTINGS)
    assert block["paired_diff"]["mean"] == -0.5 and block["paired_diff"]["ci95"][1] < -0.05
    assert block["decision"] == "FAIL"


def test_small_noisy_difference_is_inconclusive() -> None:
    champion = {question_id: 0.0 for question_id in ids(30)}
    challenger = {question_id: (1.0 if index in (3, 17) else 0.0) for index, question_id in enumerate(ids(30))}
    block = compare_metric(FABRICATED, champion, challenger, SETTINGS)
    low, high = block["paired_diff"]["ci95"]
    assert low <= 0.02 < high and block["decision"] == "INCONCLUSIVE"


def test_only_questions_in_both_populations_are_paired() -> None:
    block = compare_metric(RECALL, {"Q001": 1.0, "Q002": 0.5}, {"Q002": 0.5, "Q003": 0.0}, SETTINGS)
    assert block["n_paired"] == 1
    empty = compare_metric(RECALL, {"Q001": 1.0}, {"Q002": 1.0}, SETTINGS)
    assert empty["decision"] == "INCONCLUSIVE" and empty["n_paired"] == 0


def test_mismatched_inputs_are_reported() -> None:
    inputs = {
        "question_ids": ["NB01"],
        "frozen_hits_sha256": "a" * 64,
        "key_facts_sha256": "b" * 64,
        "question_sets": {"eval/rag_questions.jsonl": "c" * 64},
        "corpus_checksum": "d" * 64,
    }
    assert check_same_inputs(inputs, dict(inputs)) == []
    changed = dict(inputs, frozen_hits_sha256="e" * 64, corpus_checksum="f" * 64)
    assert check_same_inputs(inputs, changed) == [
        "frozen_hits_sha256: champion and challenger differ",
        "corpus_checksum: champion and challenger differ",
    ]
    missing = dict(inputs)
    del missing["key_facts_sha256"]
    assert check_same_inputs(inputs, missing) == ["key_facts_sha256: missing from a run's frozen_inputs"]


# ---------------------------------------------------------------------------------------------------------------
# The margins file
# ---------------------------------------------------------------------------------------------------------------


def test_committed_margins_are_the_plan_table() -> None:
    settings = load_migration_settings(CONFIG_DIR / "release_gate.yaml")
    margins = {metric.name: (metric.direction, metric.margin) for metric in settings.metrics}
    assert margins == {
        "key_fact_recall": ("higher_is_better", 0.05),
        "abstention_accuracy_unanswerable": ("higher_is_better", 0.10),
        "asserted_absence_error_rate": ("lower_is_better", 0.05),
        "answers_with_fabricated_citation": ("lower_is_better", 0.02),
        "answers_with_untraceable_number": ("lower_is_better", 0.05),
    }
    assert (settings.resamples, settings.seed, settings.confidence) == (1000, 42, 0.95)


def test_margins_file_must_match_the_code() -> None:
    good = {
        "migration": {
            "decision_rule": "non_inferiority",
            "gated_metrics": {
                "key_fact_recall": {"direction": "higher_is_better", "margin": 0.05, "population": "keyed"}
            },
        }
    }
    assert parse_migration_settings(good).metrics[0].name == "key_fact_recall"
    wrong_population = json.loads(json.dumps(good))
    wrong_population["migration"]["gated_metrics"]["key_fact_recall"]["population"] = "all"
    with pytest.raises(GateConfigError, match="population"):
        parse_migration_settings(wrong_population)
    unknown = {"migration": {"decision_rule": "non_inferiority", "gated_metrics": {"bleu": {}}}}
    with pytest.raises(GateConfigError, match="not computed"):
        parse_migration_settings(unknown)
    with pytest.raises(GateConfigError, match="no `migration:`"):
        parse_migration_settings({})


# ---------------------------------------------------------------------------------------------------------------
# The script, end to end on synthetic runs
# ---------------------------------------------------------------------------------------------------------------


def synthetic_score(question_id: str, category: str, recall: float | None, not_found: bool) -> dict[str, Any]:
    """A per-question score dict as eval_generation writes it."""
    keyed = recall is not None
    return QuestionScore(
        question_id=question_id,
        category=category,
        answerable=category != "unanswerable",
        gold_in_context=True if category != "unanswerable" else None,
        keyed=keyed,
        gold_section_in_context=True if keyed else None,
        key_facts_n=2 if keyed else 0,
        key_facts_matched=int((recall or 0) * 2),
        key_fact_recall=recall,
        key_facts_missed=[],
        is_not_found=not_found,
        asserted_absence=not_found,
        asserted_absence_phrases=["not-found sentence"] if not_found else [],
        cited_ids=[] if not_found else ["NBC-4.2"],
        fabricated_ids=[],
        has_fabricated_citation=False,
        n_factual_sentences=0 if not_found else 1,
        n_uncited_sentences=0,
        uncited_share=None if not_found else 0.0,
        format_compliant=True,
        n_numbers=0,
        untraceable_numbers=[],
        has_untraceable_number=False,
    ).to_dict()


def write_run(tmp_path: Path, config_id: str, recalls: list[float], generated_at: str) -> dict[str, Any]:
    """Rows file plus the configs.<id> block for a synthetic run: 40 keyed questions and 30 unanswerables."""
    rows = [
        {"question_id": f"K{i:02d}", "score": synthetic_score(f"K{i:02d}", "single_section", r, False)}
        for i, r in enumerate(recalls)
    ]
    rows += [
        {"question_id": f"U{i:02d}", "score": synthetic_score(f"U{i:02d}", "unanswerable", None, True)}
        for i in range(30)
    ]
    rows_path = tmp_path / "results" / "generation" / f"{config_id}.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    question_ids = sorted(row["question_id"] for row in rows)
    return {
        "status": "measured",
        "provider": "ollama",
        "model": f"model-{config_id}",
        "ollama_digest": "1" * 64,
        "ollama_version": "0.40.3",
        "model_license": "SYNTHETIC test license",
        "model_license_sources": {"docker_artifact": "Apache-2.0 (test)", "upstream_card": "test card"},
        "prompt_id": "rag_answer_v1",
        "prompt_sha256": "2" * 64,
        "k": 3,
        "options": {"num_predict": 200},
        "n_questions": len(rows),
        "scored_question_ids": question_ids,
        "frozen_inputs": {
            "question_ids": question_ids,
            "frozen_hits_sha256": "3" * 64,
            "key_facts_sha256": "4" * 64,
            "question_sets": {"eval/rag_questions.jsonl": "5" * 64},
            "corpus_checksum": "6" * 64,
            "retrieval_mode": "hybrid_rerank",
            "n_questions": len(rows),
        },
        "rows_path": f"results/generation/{config_id}.jsonl",
        "rows_sha256": sha256_of_file(rows_path),
        "generated_at_utc": generated_at,
        "git_sha": "unknown",
        "load_average": {"start": [0.1, 0.1, 0.1], "end": [0.2, 0.2, 0.2]},
        "latency_s": {"p50": 40.0, "p95": 90.0, "n_live_calls": 70},
        "tokens": {"in_mean": 1500.0, "out_mean": 80.0},
        "errors": {"http": {"n": 0}, "context": {"n": 0}},
        "provenance": {"models": [{"provider": "ollama", "model": f"model-{config_id}", "digest": "1" * 64}]},
    }


def gate_args(tmp_path: Path, metrics: dict[str, Any], *extra: str) -> list[str]:
    """Write the metrics file and return the gate's command line (output inside tmp_path)."""
    metrics_path = tmp_path / "results" / "generation_metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    return [
        "--champion", "champ", "--challenger", "chall",
        "--metrics", str(metrics_path),
        "--release-gate", str(CONFIG_DIR / "release_gate.yaml"),
        "--output-dir", str(tmp_path / "results"),
        "--certificate-dir", str(tmp_path / "certs"),
        "--repo-root", str(tmp_path),
        *extra,
    ]  # fmt: skip


def test_script_refuses_unfrozen_margins_then_certifies_with_the_override(tmp_path: Path) -> None:
    recalls = [1.0, 0.5] * 20
    metrics = {
        "configs": {
            "champ": write_run(tmp_path, "champ", recalls, "2026-10-12T20:00:00+00:00"),
            "chall": write_run(tmp_path, "chall", recalls, "2026-10-12T21:00:00+00:00"),
        }
    }
    # tmp_path is not a git repository, so the margins cannot have been committed before the challenger run.
    assert llm_migration_gate.main(gate_args(tmp_path, metrics)) == 3
    assert llm_migration_gate.main(gate_args(tmp_path, metrics, "--allow-unfrozen-margins")) == 0
    report = json.loads((tmp_path / "results" / "llm_migration_champ_vs_chall.json").read_text(encoding="utf-8"))
    assert report["overall"] == "PASS" and report["margins_frozen_before_challenger"] is False
    assert report["metrics"]["key_fact_recall"]["n_paired"] == 40
    assert report["metrics"]["abstention_accuracy_unanswerable"]["n_paired"] == 30
    certificate = (tmp_path / "certs" / "champ_vs_chall.md").read_text(encoding="utf-8")
    assert "**Overall decision: PASS**" in certificate and "NOT VALID FOR RELEASE" in certificate
    assert "Apache-2.0 (test)" in certificate and "test card" in certificate
    assert render_certificate(report) == certificate


def test_script_fails_a_worse_challenger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def frozen_margins(path: Path, generated_at: str | None, repo_root: Path) -> dict[str, Any]:
        """Pretend the margins were committed before the challenger run."""
        return {
            "margins_source": "config/release_gate.yaml@abc",
            "margins_file_sha256": "7" * 64,
            "margins_committed_at": "2026-10-12T19:00:00+00:00",
            "margins_frozen_before_challenger": True,
            "margins_check": "test",
        }

    monkeypatch.setattr(llm_migration_gate, "margins_status", frozen_margins)
    metrics = {
        "configs": {
            "champ": write_run(tmp_path, "champ", [1.0] * 40, "2026-10-12T20:00:00+00:00"),
            "chall": write_run(tmp_path, "chall", [0.5] * 40, "2026-10-12T21:00:00+00:00"),
        }
    }
    assert llm_migration_gate.main(gate_args(tmp_path, metrics)) == 1
    report = json.loads((tmp_path / "results" / "llm_migration_champ_vs_chall.json").read_text(encoding="utf-8"))
    assert report["metrics"]["key_fact_recall"]["decision"] == "FAIL" and report["overall"] == "FAIL"
    assert "NOT VALID FOR RELEASE" not in (tmp_path / "certs" / "champ_vs_chall.md").read_text(encoding="utf-8")


def test_script_refuses_runs_on_different_frozen_inputs(tmp_path: Path) -> None:
    champion = write_run(tmp_path, "champ", [1.0] * 40, "2026-10-12T20:00:00+00:00")
    challenger = write_run(tmp_path, "chall", [1.0] * 40, "2026-10-12T21:00:00+00:00")
    challenger["frozen_inputs"]["frozen_hits_sha256"] = "8" * 64
    metrics = {"configs": {"champ": champion, "chall": challenger}}
    assert llm_migration_gate.main(gate_args(tmp_path, metrics, "--allow-unfrozen-margins")) == 2
    assert not (tmp_path / "results" / "llm_migration_champ_vs_chall.json").exists()


def test_script_refuses_edited_rows_and_unmeasured_runs(tmp_path: Path) -> None:
    champion = write_run(tmp_path, "champ", [1.0] * 40, "2026-10-12T20:00:00+00:00")
    challenger = write_run(tmp_path, "chall", [1.0] * 40, "2026-10-12T21:00:00+00:00")
    rows_path = tmp_path / challenger["rows_path"]
    rows_path.write_text(rows_path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    metrics = {"configs": {"champ": champion, "chall": challenger}}
    assert llm_migration_gate.main(gate_args(tmp_path, metrics, "--allow-unfrozen-margins")) == 2
    smoke = dict(champion, status="smoke")
    metrics = {"configs": {"champ": smoke, "chall": champion}}
    assert llm_migration_gate.main(gate_args(tmp_path, metrics, "--allow-unfrozen-margins")) == 2
