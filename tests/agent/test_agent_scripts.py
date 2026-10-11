"""Tests for scripts/run_agent.py, scripts/eval_agent.py and scripts/make_applications.py (offline)."""

import csv
import json
from pathlib import Path

import pytest
from agent_helpers import APPLICATIONS_DIR, fake_handlers

import eval_agent
import make_applications
import run_agent
from uwcopilot.agent.settings import AgentSettings, load_agent_settings
from uwcopilot.agent.vocab import FAKE_LLM_STAMP
from uwcopilot.llm import factory
from uwcopilot.llm.settings import load_llm_settings

APPLICATION = APPLICATIONS_DIR / "APP-C-008.json"


@pytest.fixture
def no_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore any developer .env: settings come only from the (cleaned) process environment."""
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "UWC_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(factory, "load_llm_settings", lambda: load_llm_settings(env_file=None))
    monkeypatch.setattr(run_agent, "load_llm_settings", lambda: load_llm_settings(env_file=None))


@pytest.fixture
def offline_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_env_file: None) -> Path:
    """run_agent with offline tools and state files in tmp_path."""
    settings = AgentSettings(checkpoint_db=tmp_path / "cp.sqlite", decision_log=tmp_path / "records.jsonl")
    monkeypatch.setattr(run_agent, "load_agent_settings", lambda: settings)
    monkeypatch.setattr(run_agent, "default_tool_handlers", lambda settings, mode: fake_handlers())
    return tmp_path


def test_run_agent_refuses_a_real_provider_without_a_key(no_env_file: None, capsys: pytest.CaptureFixture) -> None:
    for provider in ("anthropic", "openai"):
        assert run_agent.main(["--application", str(APPLICATION), "--provider", provider]) == run_agent.EXIT_REFUSED
        assert "Refusing to run" in capsys.readouterr().err


def test_run_agent_fake_provider_stamps_output_and_resumes_from_sqlite(
    offline_agent: Path, capsys: pytest.CaptureFixture
) -> None:
    base = ["--application", str(APPLICATION), "--provider", "fake", "--thread-id", "cli-1"]
    assert run_agent.main(base) == 0
    output = capsys.readouterr().out
    review_lines = [line for line in output.splitlines() if line.strip() and not line.startswith("  uv run")]
    assert all(line.startswith(f"[{FAKE_LLM_STAMP}]") for line in review_lines)
    assert "Reference policy recommendation: refer" in output
    assert "Citations used:" in output and "NBC-4.2" in output

    refused = run_agent.main(
        [
            *base,
            "--decision",
            "decline",
            "--reviewer-id",
            "u1",
            "--role",
            "underwriter",
            "--rationale",
            "Trying a decline as an underwriter.",
        ]
    )
    assert refused == run_agent.EXIT_DECISION_REFUSED
    capsys.readouterr()
    accepted = run_agent.main(
        [
            *base,
            "--decision",
            "refer",
            "--reviewer-id",
            "s1",
            "--role",
            "senior_underwriter",
            "--rationale",
            "Send to the senior queue for the DSCR exception.",
        ]
    )
    assert accepted == 0
    assert "Decision recorded: referred_by_human" in capsys.readouterr().out
    records = (offline_agent / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1 and json.loads(records[0])["stamp"] == FAKE_LLM_STAMP


def test_agent_settings_read_project_variables_only() -> None:
    settings = load_agent_settings(env_file=None, environ={"UWC_RISK_API_URL": "http://risk:8000/", "RISK_URL": "x"})
    assert settings.risk_api.base_url == "http://risk:8000"
    assert settings.risk_api.score_url == "http://risk:8000/score"
    assert load_agent_settings(env_file=None, environ={}).otlp_endpoint is None


def test_eval_agent_writes_not_measured_llm_metrics(
    tmp_path: Path, no_env_file: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(eval_agent, "load_agent_settings", lambda: AgentSettings())
    output = tmp_path / "agent_metrics.json"
    code = eval_agent.main(
        [
            "--screening",
            "ground_truth",
            "--retrieval",
            "off",
            "--limit",
            "8",
            "--output",
            str(output),
            "--per-application",
            str(tmp_path / "per_app.jsonl"),
        ]
    )
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["data_label"] == "SYNTHETIC"
    assert report["llm_dependent"]["citation_accuracy"].startswith("not measured")
    assert report["deterministic_path"]["reached_human_checkpoint"]["value"] == 1.0
    assert report["deterministic_path"]["decisions_recorded_without_a_human"] == 0
    assert report["reference_policy"]["recommendation_agreement"]["n"] == 8
    assert len((tmp_path / "per_app.jsonl").read_text(encoding="utf-8").splitlines()) == 8


def test_eval_agent_refuses_a_real_provider_without_a_key(tmp_path: Path, no_env_file: None) -> None:
    assert eval_agent.main(["--provider", "anthropic", "--output", str(tmp_path / "x.json")]) == 2


def write_ofac_fixture(directory: Path) -> tuple[Path, Path]:
    """Tiny header-less sdn.csv / alt.csv files in the OFAC layout, with fictional entries."""
    sdn = directory / "sdn.csv"
    alt = directory / "alt.csv"
    rows = [
        [str(100 + index), name, "-0-", "SDGT", "-0-", "-0-", "-0-", "-0-", "-0-", "-0-", "-0-", "-0-"]
        for index, name in enumerate(
            [
                "AURORA FICTIONAL SHIPPING COMPANY",
                "BLUEFIN TEST TRADING HOUSE",
                "CORVUS PRETEND METALS GROUP",
                "DELTA IMAGINARY PETROCHEMICAL WORKS",
                "EMBER MADEUP LOGISTICS",
                "FALCON SYNTHETIC MARINE SERVICES",
            ]
        )
    ]
    with sdn.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    alt.write_text("", encoding="utf-8")
    return sdn, alt


def write_opensanctions_fixture(directory: Path, names: list[str]) -> Path:
    """A tiny OpenSanctions-style CSV with fictional person entries."""
    path = directory / "targets.simple.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "schema", "name", "aliases", "countries", "program_ids", "dataset"])
        for index, name in enumerate(names):
            writer.writerow([f"t{index}", "Person", name, "", "us", "", "Test list"])
    return path


def test_make_applications_writes_files_and_manifest(tmp_path: Path) -> None:
    sdn, alt = write_ofac_fixture(tmp_path)
    plain = tmp_path / "plain"
    assert (
        make_applications.main(
            [
                "--out-dir",
                str(plain),
                "--sdn",
                str(sdn),
                "--alt",
                str(alt),
                "--opensanctions",
                str(write_opensanctions_fixture(tmp_path, ["Nobody Listed"])),
            ]
        )
        == 0
    )
    first_owner = json.loads((plain / "APP-C-001.json").read_text(encoding="utf-8"))["owners"][0]["name"]
    listed = write_opensanctions_fixture(tmp_path, [first_owner])
    out = tmp_path / "apps"
    out.mkdir()
    (out / "APP-C-999.json").write_text("{}", encoding="utf-8")  # stale file is removed
    args = ["--out-dir", str(out), "--sdn", str(sdn), "--alt", str(alt), "--opensanctions", str(listed)]
    assert make_applications.main(args) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["clean_name_check"]["names_redrawn"] >= 1
    regenerated = json.loads((out / "APP-C-001.json").read_text(encoding="utf-8"))["owners"][0]["name"]
    assert regenerated != first_owner  # a listed name is never used for a clean owner
    assert manifest["count"] == 65 and manifest["data_label"] == "SYNTHETIC"
    assert not (out / "APP-C-999.json").exists()
    assert len(list(out.glob("APP-*.json"))) == 65
    assert manifest["watchlist_source"]["entries_used"].startswith("OFAC SDN organization entries only")
