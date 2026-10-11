"""Tests for scripts/update_readme_results.py: the README results block comes from results/*.json only."""

import json
from pathlib import Path

import pytest

import update_readme_results as readme_results
from uwcopilot.common.paths import RESULTS_DIR


def test_missing_results_files_are_not_measured(tmp_path: Path) -> None:
    block = readme_results.render_block(tmp_path)
    assert block.count(readme_results.NO_FILE) >= 4
    assert "0." not in block.split("Written by")[0]  # no number appears without a results file


def test_values_come_from_the_results_files() -> None:
    block = readme_results.render_block(RESULTS_DIR)
    kyc = json.loads((RESULTS_DIR / "kyc_metrics.json").read_text(encoding="utf-8"))
    selected = kyc["selected_operating_point"]
    recall = kyc["operating_points"][str(selected["alert_budget_per_1000"])][selected["method"]]["test"]["recall"]
    assert f"{recall['value']:.4f} [{recall['ci95_low']:.4f}, {recall['ci95_high']:.4f}]" in block
    agent = json.loads((RESULTS_DIR / "agent_metrics.json").read_text(encoding="utf-8"))
    escalation = agent["deterministic_path"]["escalation"]["all_required_escalations"]
    assert f"(n={escalation['n']})" in block
    assert "not measured" in block  # the LLM-dependent rows


def test_replace_block_needs_exactly_one_pair_of_markers() -> None:
    text = f"intro\n{readme_results.START_MARKER}\nold\n{readme_results.END_MARKER}\noutro\n"
    assert readme_results.replace_block(text, "new") == (
        f"intro\n{readme_results.START_MARKER}\nnew\n{readme_results.END_MARKER}\noutro\n"
    )
    with pytest.raises(ValueError):
        readme_results.replace_block("no markers here", "new")


def test_committed_readme_matches_the_committed_results() -> None:
    """Docs cannot drift from results/*.json: re-render and compare with the README in the repo."""
    assert readme_results.main(["--check", "--readme", str(readme_results.README)]) == 0
