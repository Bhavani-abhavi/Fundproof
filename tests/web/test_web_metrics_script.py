"""Tests for scripts/web_metrics.py: the parts that need no browser, no Node and no network.

The browser session itself (Playwright, axe-core, Lighthouse) is exercised by running the script against a local
server; here the result summaries, the decision choice, the step log, the argument checks and the results file are
tested with small hand-written inputs.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import web_metrics
from web_metrics import (
    CheckFailed,
    FlowLog,
    FlowStopped,
    MetricsConfigError,
    NodeTools,
    Observer,
    axe_totals,
    check_arguments,
    choose_action,
    find_node_tools,
    lighthouse_command,
    median_of_runs,
    merge_run,
    mirror_math_round,
    needs_override,
    read_server_config,
    require,
    run_flow,
    since_ms,
    summarize_axe,
    summarize_lighthouse,
)


def arguments(**overrides: Any) -> argparse.Namespace:
    """Parsed arguments with defaults, then the overrides."""
    args = web_metrics.parse_args(["--url", "http://127.0.0.1:7862", "--target", "local-container"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def axe_rule(rule_id: str, impact: str | None, nodes: int) -> dict[str, Any]:
    """One axe violation or incomplete item with `nodes` flagged elements."""
    return {
        "id": rule_id,
        "impact": impact,
        "help": f"help for {rule_id}",
        "tags": ["wcag2aa"],
        "nodes": [{"target": [f"#node-{index}"]} for index in range(nodes)],
    }


def lighthouse_report(performance: float | None = 0.9, accessibility: float = 1.0) -> dict[str, Any]:
    """A minimal Lighthouse result (lhr) with the fields the summary reads."""
    return {
        "lighthouseVersion": "13.5.0",
        "configSettings": {"formFactor": "mobile", "throttlingMethod": "simulate"},
        "categories": {
            "performance": {"score": performance, "auditRefs": []},
            "accessibility": {
                "score": accessibility,
                "auditRefs": [{"id": "color-contrast", "weight": 7}, {"id": "landmark-one-main", "weight": 0}],
            },
            "best-practices": {"score": 1.0, "auditRefs": [{"id": "errors-in-console", "weight": 1}]},
        },
        "audits": {
            "first-contentful-paint": {"numericValue": 1200.5},
            "largest-contentful-paint": {"numericValue": 2400.0},
            "total-blocking-time": {"numericValue": 30.0},
            "cumulative-layout-shift": {"numericValue": 0.01},
            "speed-index": {"numericValue": 1500.0},
            "color-contrast": {"score": 0 if accessibility < 1 else 1},
            "landmark-one-main": {"score": 0},
            "errors-in-console": {"score": 1},
        },
        "runWarnings": [],
    }


# ---- arguments and tools ------------------------------------------------------------------------------------------


def test_check_arguments_accepts_a_loopback_url() -> None:
    assert check_arguments(arguments()) == "http://127.0.0.1:7862"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"url": "https://example.hf.space"}, "loopback"),
        ({"lighthouse_runs": 0}, "--lighthouse-runs"),
        ({"step_timeout_s": 0}, "--step-timeout-s"),
        ({"container": "fundproof-metrics"}, "--docker-image"),
        ({"url": "http://127.0.0.1:7862/studio"}, "base URL"),
    ],
)
def test_check_arguments_refuses_what_does_not_fit(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(MetricsConfigError, match=message):
        check_arguments(arguments(**overrides))


def test_the_space_is_not_a_target() -> None:
    with pytest.raises(SystemExit):
        web_metrics.parse_args(["--url", "https://x.hf.space", "--target", "hf-space"])


def test_find_node_tools_names_the_install_command(tmp_path: Path) -> None:
    with pytest.raises(MetricsConfigError, match="npm ci"):
        find_node_tools(tmp_path, need_lighthouse=True)


def test_find_node_tools_reads_the_installed_versions(tmp_path: Path) -> None:
    modules = tmp_path / "node_modules"
    (modules / "axe-core").mkdir(parents=True)
    (modules / "axe-core" / "axe.min.js").write_text("/* axe */", encoding="utf-8")
    (modules / "axe-core" / "package.json").write_text('{"version": "4.13.0"}', encoding="utf-8")
    (modules / "lighthouse" / "cli").mkdir(parents=True)
    (modules / "lighthouse" / "cli" / "index.js").write_text("// cli", encoding="utf-8")
    (modules / "lighthouse" / "package.json").write_text('{"version": "13.5.0"}', encoding="utf-8")
    if web_metrics.shutil.which("node") is None:
        pytest.skip("node is not installed on this machine")
    tools = find_node_tools(tmp_path, need_lighthouse=True)
    assert (tools.axe_version, tools.lighthouse_version) == ("4.13.0", "13.5.0")


def test_axe_alone_is_enough_when_lighthouse_is_skipped(tmp_path: Path) -> None:
    axe_dir = tmp_path / "node_modules" / "axe-core"
    axe_dir.mkdir(parents=True)
    (axe_dir / "axe.min.js").write_text("/* axe */", encoding="utf-8")
    (axe_dir / "package.json").write_text('{"version": "4.13.0"}', encoding="utf-8")
    if web_metrics.shutil.which("node") is None:
        pytest.skip("node is not installed on this machine")
    tools = find_node_tools(tmp_path, need_lighthouse=False)
    assert tools.lighthouse_version == "not installed"


def test_lighthouse_command_keeps_the_session_and_picks_the_preset() -> None:
    tools = NodeTools("node", Path("cli/index.js"), Path("axe.min.js"), "13.5.0", "4.13.0")
    mobile = lighthouse_command(tools, "http://127.0.0.1:7862/", 9222, "mobile")
    desktop = lighthouse_command(tools, "http://127.0.0.1:7862/", 9222, "desktop")
    assert "--disable-storage-reset" in mobile and "--port=9222" in mobile
    assert "--only-categories=performance,accessibility,best-practices" in mobile
    assert "--preset=desktop" not in mobile and "--preset=desktop" in desktop
    with pytest.raises(MetricsConfigError):
        lighthouse_command(tools, "http://127.0.0.1:7862/", 9222, "tablet")


# ---- the server check ---------------------------------------------------------------------------------------------


def config_server(dev_auth: bool) -> httpx.MockTransport:
    """A fake app answering /api/health and /api/config."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/health":
            return httpx.Response(200, json={"status": "ok", "version": "0.1.0", "git_sha": None})
        body = {"product_name": "Fundproof", "disclaimer": "Not advice.", "dev_auth": dev_auth, "is_space": False}
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def test_read_server_config_needs_the_dev_login() -> None:
    with pytest.raises(MetricsConfigError, match="UWC_DEV_AUTH=1"):
        read_server_config("http://127.0.0.1:7862", config_server(dev_auth=False))
    facts = read_server_config("http://127.0.0.1:7862", config_server(dev_auth=True))
    assert facts["dev_auth"] is True and facts["disclaimer"] == "Not advice."


def test_read_server_config_reports_a_server_that_stays_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    waits: list[float] = []
    with pytest.raises(MetricsConfigError, match="did not become healthy within 0 s"):
        read_server_config("http://127.0.0.1:7862", httpx.MockTransport(handler), wait_s=0, sleep=waits.append)
    assert waits == []


def test_read_server_config_waits_for_a_starting_container() -> None:
    attempts = {"health": 0}
    ready = config_server(dev_auth=True)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/health":
            attempts["health"] += 1
            if attempts["health"] < 3:
                raise httpx.ConnectError("not listening yet", request=request)
        return ready.handle_request(request)

    waits: list[float] = []
    facts = read_server_config("http://127.0.0.1:7862", httpx.MockTransport(handler), wait_s=60, sleep=waits.append)
    assert facts["version"] == "0.1.0"
    assert waits == [web_metrics.HEALTH_POLL_S, web_metrics.HEALTH_POLL_S]


# ---- axe ----------------------------------------------------------------------------------------------------------


def test_summarize_axe_counts_rules_and_nodes_per_impact() -> None:
    result = {
        "violations": [axe_rule("color-contrast", "serious", 3), axe_rule("region", "moderate", 2)],
        "incomplete": [axe_rule("color-contrast", "serious", 1)],
    }
    summary = summarize_axe(result)
    assert summary["violations_by_impact"] == {"critical": 0, "serious": 1, "moderate": 1, "minor": 0}
    assert summary["nodes_by_impact"] == {"critical": 0, "serious": 3, "moderate": 2, "minor": 0}
    assert summary["incomplete_by_impact"]["serious"] == 1
    assert [rule["id"] for rule in summary["serious_or_critical_rules"]] == ["color-contrast"]
    assert summary["serious_or_critical_rules"][0]["first_targets"] == [["#node-0"], ["#node-1"], ["#node-2"]]
    assert [rule["id"] for rule in summary["other_violation_rules"]] == ["region"]
    assert summary["other_violation_rules"][0]["nodes"] == 2


def test_summarize_axe_of_a_clean_page_is_all_zero() -> None:
    summary = summarize_axe({"violations": [], "incomplete": []})
    assert set(summary["violations_by_impact"].values()) == {0}
    assert summary["serious_or_critical_rules"] == []


def test_axe_totals_add_up_serious_and_critical() -> None:
    scans = [
        summarize_axe({"violations": [axe_rule("a", "critical", 1), axe_rule("b", "serious", 2)]}),
        summarize_axe({"violations": [axe_rule("c", "minor", 4)]}),
        summarize_axe({"violations": [axe_rule("d", "serious", 1)]}),
    ]
    totals = axe_totals(scans)
    assert totals == {
        "scans": 3,
        "critical_violation_rules": 1,
        "critical_nodes": 1,
        "serious_violation_rules": 2,
        "serious_nodes": 3,
        "scans_with_serious_or_critical": 2,
    }


# ---- Lighthouse ---------------------------------------------------------------------------------------------------


def test_mirror_math_round_rounds_halves_up_like_javascript() -> None:
    assert [mirror_math_round(value) for value in (96.5, 97.4999, 0.5, 89.0)] == [97, 97, 1, 89]


def test_summarize_lighthouse_reads_scores_metrics_and_failed_audits() -> None:
    summary = summarize_lighthouse(lighthouse_report(performance=0.875, accessibility=0.96))
    assert summary["scores"] == {"performance": 0.875, "accessibility": 0.96, "best-practices": 1.0}
    assert summary["scores_0_100"] == {"performance": 88, "accessibility": 96, "best-practices": 100}
    assert summary["metrics"]["first-contentful-paint"] == 1200.5
    # landmark-one-main failed but has weight 0, so only the weighted failure is listed
    assert summary["failed_weighted_audits"] == ["color-contrast"]
    assert (summary["form_factor"], summary["throttling_method"]) == ("mobile", "simulate")


def test_summarize_lighthouse_keeps_a_missing_score_as_none() -> None:
    report = lighthouse_report(performance=None)
    report["runtimeError"] = {"code": "NO_FCP", "message": "The page did not paint any content."}
    summary = summarize_lighthouse(report)
    assert summary["scores"]["performance"] is None and summary["scores_0_100"]["performance"] is None
    assert summary["runtime_error"] == "NO_FCP"


def test_median_of_runs_ignores_runs_without_scores() -> None:
    runs = [
        summarize_lighthouse(lighthouse_report(performance=0.8)),
        summarize_lighthouse(lighthouse_report(performance=0.9)),
        summarize_lighthouse(lighthouse_report(performance=0.7)),
        {"error": "Lighthouse did not finish"},
    ]
    median = median_of_runs(runs)
    assert median["scores"]["performance"] == 0.8
    assert median["scores_0_100"]["performance"] == 80
    assert median["runs_with_scores"] == 3
    assert median_of_runs([{"error": "x"}])["scores"]["performance"] is None


# ---- the decision choice ------------------------------------------------------------------------------------------


def checks(*allowed: str, override: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """action_checks as the lender file answer has them."""
    return [
        {"action": action, "allowed": action in allowed, "reasons": [], "needs_override_reason": action in override}
        for action in ("approve", "refer", "request_more_information", "decline")
    ]


def test_choose_action_follows_the_policy_when_allowed() -> None:
    assert choose_action("approve", checks("approve", "refer")) == "approve"
    assert choose_action("decline_for_review", checks("decline", "refer")) == "decline"


def test_choose_action_falls_back_to_the_first_allowed_action() -> None:
    assert choose_action("decline_for_review", checks("refer", "request_more_information")) == "refer"
    assert choose_action("approve", checks()) is None


def test_needs_override_reads_the_server_answer() -> None:
    assert needs_override("approve", checks("approve", override=("approve",))) is True
    assert needs_override("refer", checks("approve", "refer", override=("approve",))) is False


# ---- the step log -------------------------------------------------------------------------------------------------


class FakeResponse:
    """The parts of a Playwright Response the observer reads."""

    def __init__(self, method: str, url: str, status: int) -> None:
        self.request = type("Request", (), {"method": method})()
        self.url = url
        self.status = status


def test_a_failed_step_stops_its_flow_and_is_recorded() -> None:
    observer = Observer()
    log = FlowLog("demo", observer, (CheckFailed,))

    def body(flow: FlowLog) -> None:
        with flow.step("first"):
            pass
        with flow.step("second"):
            require(False, "the page did not say Not found")
        with flow.step("never_runs"):
            pass

    summary = run_flow(log, body)
    assert [step["name"] for step in summary["steps"]] == ["first", "second"]
    assert summary["steps"][0]["passed"] is True
    assert summary["steps"][1]["passed"] is False
    assert "Not found" in summary["steps"][1]["detail"]
    assert summary["passed"] is False


def test_other_errors_are_not_swallowed() -> None:
    log = FlowLog("demo", Observer(), (CheckFailed,))
    with pytest.raises(ZeroDivisionError), log.step("broken"):
        _ = 1 / 0


def test_expected_404s_do_not_fail_the_flow_but_unexpected_ones_do() -> None:
    observer = Observer()
    log = FlowLog("errors", observer, (CheckFailed,))

    def body(flow: FlowLog) -> None:
        with flow.step("unknown_plan", ("status of 404",), (404,)):
            observer.on_response(FakeResponse("GET", "http://127.0.0.1:7862/api/studio/projects/0000", 404))
            observer.add_console("Failed to load resource: the server responded with a status of 404 (Not Found)")
        with flow.step("next"):
            observer.on_response(FakeResponse("GET", "http://127.0.0.1:7862/studio/", 200))

    summary = run_flow(log, body)
    assert summary["passed"] is True
    assert summary["api_errors"][0]["expected"] is True
    assert summary["console_errors"][0]["expected"] is True

    later = FlowLog("later", observer, (CheckFailed,))

    def failing_api(flow: FlowLog) -> None:
        with flow.step("feasibility"):
            observer.on_response(FakeResponse("POST", "http://127.0.0.1:7862/api/studio/projects/x/feasibility", 409))

    assert run_flow(later, failing_api)["unexpected_api_errors"] == 1
    assert run_flow(later, failing_api)["passed"] is False


def test_first_answer_and_since_ms_time_api_answers() -> None:
    observer = Observer()
    observer.api_answers = [
        {"method": "POST", "path": "/api/studio/projects", "status": 201, "at": 10.0},
        {"method": "POST", "path": "/api/studio/projects/abc/feasibility", "status": 200, "at": 10.4},
    ]
    answer = observer.first_answer("POST", r"/feasibility$", since=9.9)
    assert answer is not None and since_ms(9.9, answer) == 500.0
    assert observer.first_answer("POST", r"/feasibility$", since=10.5) is None
    assert since_ms(9.9, None) is None


def test_flow_stopped_is_an_exception_not_a_check() -> None:
    assert not issubclass(FlowStopped, AssertionError)


# ---- the results file ---------------------------------------------------------------------------------------------


def test_merge_run_replaces_only_its_own_target(tmp_path: Path) -> None:
    output = tmp_path / "web_metrics.json"
    output.write_text(json.dumps({"runs": {"local-server": {"old": True}, "local-container": {"old": True}}}))
    document = merge_run(output, "local-container", {"new": True})
    assert document["runs"] == {"local-container": {"new": True}, "local-server": {"old": True}}
    assert "LOCAL" in document["description"]


def test_merge_run_refuses_a_broken_file(tmp_path: Path) -> None:
    output = tmp_path / "web_metrics.json"
    output.write_text("{not json")
    with pytest.raises(MetricsConfigError, match="not valid JSON"):
        merge_run(output, "local-container", {})


def test_main_stops_before_the_browser_on_bad_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert web_metrics.main(["--url", "http://10.0.0.5:7860", "--target", "local-container"]) == 2
    assert "loopback" in capsys.readouterr().err


def test_every_target_label_says_local_and_not_the_space() -> None:
    for label in web_metrics.TARGET_LABELS.values():
        assert "LOCAL" in label and "NOT the Hugging Face Space" in label and "UWC_DEV_AUTH=1" in label


def test_optimize_png_reports_what_it_did(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    path = tmp_path / "shot.png"
    image_module.new("RGB", (64, 64), (240, 236, 228)).save(path, format="PNG", compress_level=0)
    assert "re-encoded losslessly" in web_metrics.optimize_png(path)
    assert "was not smaller" in web_metrics.optimize_png(path)


def fake_git(head: str, status: str) -> Any:
    """A stand-in for web_metrics.git_output with a fixed HEAD and `git status --porcelain` text."""

    def run(*arguments: str) -> str | None:
        if arguments[:1] == ("rev-parse",):
            return head + "\n"
        if arguments[:1] == ("status",):
            return status
        return "diff --git a/x b/x\n"

    return run


def test_a_clean_stamped_build_is_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_metrics, "git_output", fake_git("a" * 40, ""))
    checked = web_metrics.reproducibility("a" * 40)
    assert checked["reproducible"] is True and checked["reasons"] == []
    assert checked["working_tree_digest_sha256"] is None


@pytest.mark.parametrize(
    ("server_sha", "status", "expected"),
    [
        ("a" * 40, " M src/app.py\n", "uncommitted changes"),
        (None, "", "does not name its commit"),
        ("b" * 40, "", "not HEAD"),
        ("a" * 40 + "-dirty", "", "uncommitted tree"),
    ],
)
def test_an_unreproducible_build_says_why(
    monkeypatch: pytest.MonkeyPatch, server_sha: str | None, status: str, expected: str
) -> None:
    monkeypatch.setattr(web_metrics, "git_output", fake_git("a" * 40, status))
    checked = web_metrics.reproducibility(server_sha)
    assert checked["reproducible"] is False and any(expected in reason for reason in checked["reasons"])
    assert checked["note"].startswith("UNREPRODUCIBLE")
    if status:
        assert len(checked["working_tree_digest_sha256"]) == 64


def test_main_refuses_an_unreproducible_build_without_the_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(web_metrics, "git_output", fake_git("a" * 40, " M src/app.py\n"))
    monkeypatch.setattr(web_metrics, "find_node_tools", lambda tools_dir, need_lighthouse: None)
    monkeypatch.setattr(web_metrics, "read_server_config", lambda base_url, wait_s: {"git_sha": "a" * 40})
    code = web_metrics.main(
        ["--url", "http://127.0.0.1:7862", "--target", "local-container", "--output", str(tmp_path / "out.json")]
    )
    assert code == 2 and "unreproducible" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()
