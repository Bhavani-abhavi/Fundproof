"""The HTML monitoring report: static, strict CSP, every value escaped, small inline charts (SYNTHETIC run)."""

import base64
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from monitoring_test_helpers import run_paths, small_config, write_inputs

import monitor
from uwcopilot.governance.monitoring_config import MonitoringConfig
from uwcopilot.governance.monitoring_report import CSP, escape, fmt, fmt_interval, render_monitoring_report, table

HOSTILE = '<script>alert("x")</script>'


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, dict[str, Any]]:
    """A report from a SYNTHETIC run whose scenario description and model id carry HTML."""
    tmp_path: Path = tmp_path_factory.mktemp("report")
    policies, files, results_dir = write_inputs(tmp_path)
    document = small_config().model_dump(mode="json")
    document["stream"]["scenarios"]["no_drift"]["description"] = f"SYNTHETIC {HOSTILE}"
    document["monitored_model"]["model_id"] = f"model-{HOSTILE}"
    paths = run_paths(tmp_path, results_dir)
    payload = monitor.run(MonitoringConfig.model_validate(document), policies, files, paths)
    return paths.report.read_text(encoding="utf-8"), payload


def test_the_report_is_static_with_a_strict_csp(report: tuple[str, dict[str, Any]]) -> None:
    page, _ = report
    assert f'<meta http-equiv="Content-Security-Policy" content="{CSP}">' in page
    assert "default-src 'none'" in CSP and "script-src" not in CSP
    assert "<script" not in page.lower() and "javascript:" not in page.lower()
    assert "SYNTHETIC" in page and "SIMULATED" in page


def test_every_value_is_escaped(report: tuple[str, dict[str, Any]]) -> None:
    page, _ = report
    assert HOSTILE not in page
    assert escape(HOSTILE) in page


def test_charts_are_small_inline_pngs(report: tuple[str, dict[str, Any]]) -> None:
    page, _ = report
    images = re.findall(r'<img alt="[^"]*" src="data:image/png;base64,([^"]+)">', page)
    assert len(images) == 3
    for encoded in images:
        png = base64.b64decode(encoded)
        assert png.startswith(b"\x89PNG") and len(png) < 60_000
    assert re.findall(r"<img[^>]+src=\"(?!data:image/png;base64,)", page) == []


def test_the_report_shows_the_payload_numbers(report: tuple[str, dict[str, Any]]) -> None:
    page, payload = report
    calibration = payload["monitored_model"]["held_out_calibration"]
    assert fmt(calibration["observed_to_predicted"]) in page
    assert payload["alerts"]["run_id"] in page
    assert "Lending Club credit model" in page and "not measured" in page


def test_formatting_helpers() -> None:
    assert fmt(None) == "not available" and fmt(0.123456) == "0.1235" and fmt(True) == "yes"
    assert fmt_interval(None) == "not available" and fmt_interval([0.1, 0.2], 2) == "[0.10, 0.20]"
    assert "&lt;b&gt;" in table(["<b>"], [["x"]])


def test_rendering_twice_gives_the_same_page(report: tuple[str, dict[str, Any]]) -> None:
    _, payload = report
    at = datetime(2026, 10, 20, tzinfo=UTC)
    assert render_monitoring_report(payload, at) == render_monitoring_report(payload, at)
