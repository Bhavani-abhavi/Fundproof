"""End-to-end flows, accessibility scans and Lighthouse scores of a LOCAL Fundproof server -> results/web_metrics.json.

    # 1. Install the pinned Node tools once (Lighthouse CLI + axe-core; not part of the app or the image)
    cd tools/web-metrics && npm ci
    # 2. Start the app WITH the dev login, because a script cannot sign in with Hugging Face. The container:
    docker run -d --rm --name fundproof-metrics -e UWC_DEV_AUTH=1 -p 127.0.0.1:7862:7860 fundproof:dev
    # 3. Run this script (Playwright and Chromium come from the Makefile's pinned versions)
    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers uv run --with playwright==1.56.0 --with pillow \
        python scripts/web_metrics.py --url http://127.0.0.1:7862 --target local-container \
        --docker-image fundproof:dev --container fundproof-metrics

What it does, in one headless Chromium profile (Playwright, `channel="chromium"`):
1. Signed out: axe scans of the public pages, and the error path "a signed-in page asks you to sign in".
2. Signs in through the DEV LOGIN form as a new SYNTHETIC demo founder (a fresh username per run).
3. Founder flow on the SYNTHETIC metro: the six-step wizard (city, business type, idea, money, two sites: one
   suggested area and one tract from the list), "Create plan", then the results page: site scores, the simulation
   and the path to approval. It then edits the three planning targets and, when the search finds a change-set,
   clicks "Apply this change" and waits for the re-run on the new plan version.
4. Lender demo flow: the demo queue, the first file in it, and one DEMO decision through the decision panel (the
   action the policy suggests when the checkpoint allows it for this user, otherwise the first allowed action).
5. Error paths: an unknown plan id (the API answers 404 and the page says "Not found"), the wizard's own input
   check (Continue with nothing chosen), and an unknown page (the 404 page).
6. axe-core (injected with Playwright's evaluate, so the app's CSP stays on) at every page and wizard state above,
   in the light and the dark color scheme. Its default rule set; counts per impact level are recorded.
7. Lighthouse (CLI, attached to the same Chromium over the DevTools port, `--disable-storage-reset` so the session
   cookie is kept; the HTTP cache is cleared before each run) on the landing page, the founder's results page and
   the lender file, with the mobile (Lighthouse default) and desktop presets, --lighthouse-runs times each.
8. Deletes the plan through "Delete plan" and checks that its page then says "Not found".

Reproducibility: the results must name the code they measured. The server's /api/health git_sha (the image's
UWC_GIT_SHA stamp, set by `make docker-build-stamped`, which marks an uncommitted tree "-dirty") must name a commit of
this repository. Otherwise the script REFUSES to write, unless --allow-unreproducible is given; the entry then says
"reproducible": false with the reason and a SHA-256 digest of the checkout's uncommitted changes (so the tree can at
least be identified), and its numbers must not be quoted as the release's.

The results page is also scanned with every "Show the numbers as a table" opened, and the lender file with the
decision record opened, because those scrolling boxes are where keyboard access can break.

Every step is timed with time.perf_counter() and recorded as passed or failed; a failed step stops its flow. Browser
console errors and API answers of HTTP 400 or more are recorded per step; the ones a step expects (the 404s of the
error paths) are marked expected. A flow passes when all its steps passed and nothing unexpected was logged.

Honesty rules: every number in the results file is measured by this script on the target it names. The target must
be a loopback URL (this is never the Hugging Face Space, which refuses the dev login). Lighthouse and the timings are
single-machine measurements with whatever else the machine was doing (the load average is in the provenance).
The results file keeps one entry per target (runs.local-container, runs.local-server); a new run replaces only its
own target's entry. Exit code: 0 when every flow passed and every Lighthouse run produced scores, 1 otherwise, 2 for
bad arguments or missing tools.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from optimize_pngs import optimize_file
from smoke_deploy import SmokeConfigError, container_facts, docker_image_facts, validate_target, write_json_atomically
from uwcopilot.common.paths import REPO_ROOT, RESULTS_DIR
from uwcopilot.governance.provenance import RunProvenance, display_path

GENERATED_BY = "scripts/web_metrics.py"
DEFAULT_OUTPUT = RESULTS_DIR / "web_metrics.json"
DEFAULT_TOOLS_DIR = REPO_ROOT / "tools" / "web-metrics"
DEFAULT_SCREENSHOT_DIR = REPO_ROOT / "docs" / "web" / "screenshots" / "e2e"
DEFAULT_BROWSERS_PATH = Path("/opt/pw-browsers")
DEFAULT_STEP_TIMEOUT_S = 120.0
DEFAULT_WAIT_S = 120.0
HEALTH_POLL_S = 0.5
DEFAULT_LIGHTHOUSE_RUNS = 3
LIGHTHOUSE_TIMEOUT_S = 240
LIGHTHOUSE_CATEGORIES: tuple[str, ...] = ("performance", "accessibility", "best-practices")
LIGHTHOUSE_PRESETS: tuple[str, ...] = ("mobile", "desktop")
# Lighthouse audits recorded next to the category scores (milliseconds, except CLS which has no unit).
LIGHTHOUSE_METRICS: tuple[str, ...] = (
    "first-contentful-paint",
    "largest-contentful-paint",
    "total-blocking-time",
    "cumulative-layout-shift",
    "speed-index",
)
AXE_IMPACTS: tuple[str, ...] = ("critical", "serious", "moderate", "minor")
SERIOUS_IMPACTS: tuple[str, ...] = ("critical", "serious")
COLOR_SCHEMES: tuple[str, ...] = ("light", "dark")
VIEWPORT = {"width": 1280, "height": 800}

TARGET_LABELS: dict[str, str] = {
    "local-container": (
        "LOCAL Docker container on the machine that ran this script, started WITH the dev login (UWC_DEV_AUTH=1) "
        "so the script can sign in without Hugging Face OAuth. NOT the Hugging Face Space, which refuses the dev "
        "login: no internet hop, no Space proxy, not Space hardware."
    ),
    "local-server": (
        "LOCAL uvicorn process (no container) on the machine that ran this script, started WITH the dev login "
        "(UWC_DEV_AUTH=1). NOT the Hugging Face Space."
    ),
}
DEV_LOGIN_NOTE = (
    "The browser signed in through the DEV LOGIN form (POST /api/auth/dev-login), which exists only when the server "
    "runs with UWC_DEV_AUTH=1 and is refused inside a Hugging Face Space. Hugging Face sign-in was not exercised."
)

# The SYNTHETIC demo inputs of the founder flow. They are design choices for a test run, not benchmarks or data.
DEMO_PLAN: dict[str, str] = {
    "metro_id": "synthetic_metro",
    "business_type_id": "laundromat",
    "idea": (
        "SYNTHETIC demo plan typed by scripts/web_metrics.py: a self-service laundromat with an attendant a few "
        "hours a day."
    ),
    "budget": "400000",
    "equity": "150000",
    "loan": "350000",
    "term": "120",
    "rate": "9",
}
# Edited planning targets (percent of simulated futures) for the "Apply this change" step: also test inputs.
DEMO_TARGET_EDITS: dict[str, str] = {"p_dscr_min": "50", "p_stressed_min": "30", "p_cash_negative_max": "15"}
DEMO_RATIONALE = "SYNTHETIC demo decision recorded by the web metrics script to test the review screen."
UNKNOWN_PLAN_ID = "0000000000000000"
UNKNOWN_PAGE = "/this-page-does-not-exist/"
# Reviewer actions (agent/vocab.py) as the decision panel labels them (frontend/components/lender/labels.ts).
ACTION_LABELS: dict[str, str] = {
    "approve": "Approve",
    "refer": "Refer",
    "request_more_information": "Request more information",
    "decline": "Decline with a reason",
}
ACTION_ORDER: tuple[str, ...] = ("approve", "refer", "request_more_information", "decline")
# What each reference recommendation (agent/vocab.py) maps to as a reviewer action.
RECOMMENDATION_TO_ACTION: dict[str, str] = {
    "approve": "approve",
    "refer": "refer",
    "request_more_information": "request_more_information",
    "decline_for_review": "decline",
}
# The three possible path-to-approval headlines (frontend/components/studio/project/path-panel.tsx).
PATH_STATUS_TEXT = re.compile(r"simulated targets|reaches all three targets|found no path")
EXPECTED_404_CONSOLE = "status of 404"

RESULTS_DESCRIPTION = (
    "End-to-end flows (Playwright), axe-core accessibility scans and Lighthouse scores of a LOCAL Fundproof server, "
    "written by scripts/web_metrics.py. One entry per target; each says what was measured (target.label) and has "
    "its own provenance block. A target that is not listed here was not measured. The Hugging Face Space is not "
    "measured by this script."
)
CAVEATS: tuple[str, ...] = (
    "A local run: the server and the browser share one machine (4 CPUs here, shared with other jobs; see "
    "provenance.environment). Timings and Lighthouse performance scores say nothing about the Space.",
    "Sign-in used the DEV LOGIN (UWC_DEV_AUTH=1); Hugging Face OAuth was not exercised.",
    "Lighthouse scores are listed per run with their median; no confidence interval is given because a few runs "
    "on one machine are not independent samples of anything wider.",
    "axe-core finds only the problems that can be detected automatically; zero violations is not a statement of "
    "WCAG conformance. Its 'incomplete' items need a person to review them.",
    "The founder plan, the lender decision and the edited targets are SYNTHETIC test inputs, not data.",
)


class MetricsConfigError(ValueError):
    """Bad arguments, a target without the dev login, or missing Node tools."""


class CheckFailed(AssertionError):
    """A step ran, but what it checks was not true."""


class FlowStopped(Exception):
    """A step failed, so the rest of its flow cannot run."""


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers (no browser): tested in tests/web/test_web_metrics_script.py
# ---------------------------------------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line arguments (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="base URL of the running app, e.g. http://127.0.0.1:7862")
    parser.add_argument("--target", required=True, choices=sorted(TARGET_LABELS), help="what is being measured")
    parser.add_argument("--docker-image", help="local image name to record (id and size), e.g. fundproof:dev")
    parser.add_argument("--container", help="running container to check against --docker-image")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="results file to update")
    parser.add_argument("--screenshots-dir", type=Path, default=DEFAULT_SCREENSHOT_DIR, help="where PNGs go")
    parser.add_argument("--tools-dir", type=Path, default=DEFAULT_TOOLS_DIR, help="tools/web-metrics (npm ci done)")
    parser.add_argument("--lighthouse-runs", type=int, default=DEFAULT_LIGHTHOUSE_RUNS, help="runs per page+preset")
    parser.add_argument("--skip-lighthouse", action="store_true", help="only the flows and the axe scans")
    parser.add_argument("--step-timeout-s", type=float, default=DEFAULT_STEP_TIMEOUT_S, help="timeout of each wait")
    parser.add_argument("--wait-s", type=float, default=DEFAULT_WAIT_S, help="how long to wait for /api/health")
    parser.add_argument(
        "--allow-unreproducible",
        action="store_true",
        help="write the results even when the tree is dirty or the server's commit stamp differs from HEAD "
        "(the entry is then marked reproducible: false)",
    )
    return parser.parse_args(argv)


def git_output(*arguments: str) -> str | None:
    """stdout of a git command in the repository, or None when git is missing or fails."""
    try:
        completed = subprocess.run(["git", *arguments], cwd=REPO_ROOT, capture_output=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.decode("utf-8", errors="replace")


def worktree_digest(status: str) -> str:
    """SHA-256 over `git diff HEAD --binary` and the untracked files (path + bytes): identifies a dirty tree."""
    digest = hashlib.sha256((git_output("diff", "HEAD", "--binary") or "").encode("utf-8"))
    for line in sorted(status.splitlines()):
        if line.startswith("?? "):
            path = REPO_ROOT / line[3:].strip().strip('"')
            digest.update(line.encode("utf-8"))
            if path.is_file():
                digest.update(path.read_bytes())
    return digest.hexdigest()


def reproducibility(server_git_sha: str | None) -> dict[str, Any]:
    """Whether the measured server can be rebuilt from a commit: its /api/health stamp must name a commit of this
    repository, without the "-dirty" mark that `make docker-build-stamped` adds for an uncommitted tree.

    The checkout's own HEAD and working-tree state are recorded too (when the image was built from a commit, for
    example `git archive <commit> | docker build ... -`, later edits to the checkout do not change the image).
    """
    head = (git_output("rev-parse", "HEAD") or "").strip() or None
    status = git_output("status", "--porcelain")
    dirty = status is None or bool(status.strip())
    reasons = []
    if not server_git_sha:
        reasons.append("the server does not name its commit (build with `make docker-build-stamped`)")
    elif server_git_sha.endswith("-dirty"):
        reasons.append(f"the server was built from an uncommitted tree ({server_git_sha})")
    elif git_output("cat-file", "-e", f"{server_git_sha}^{{commit}}") is None:
        reasons.append(f"the server's commit {server_git_sha} is not a commit of this repository")
    return {
        "reproducible": not reasons,
        "reasons": reasons,
        "server_git_sha": server_git_sha,
        "head": head,
        "server_is_head": bool(head and server_git_sha and head == server_git_sha),
        "working_tree_dirty": dirty,
        "working_tree_digest_sha256": worktree_digest(status or "") if dirty and status is not None else None,
        "note": (
            f"Reproducible: rebuilding commit {server_git_sha} with the Dockerfile gives the measured app."
            if not reasons
            else "UNREPRODUCIBLE: these numbers describe an uncommitted or unstamped build; rebuild from a commit "
            "(make docker-build-stamped on a clean tree, or git archive <commit> | docker build ... -) and run again "
            "before quoting them."
        ),
    }


def check_arguments(args: argparse.Namespace) -> str:
    """The base URL after checking the arguments; MetricsConfigError when something does not fit."""
    if args.target not in TARGET_LABELS:
        raise MetricsConfigError(f"Unknown target {args.target!r}; choose one of {sorted(TARGET_LABELS)}.")
    try:
        base_url = validate_target(args.url, args.target)
    except SmokeConfigError as error:
        raise MetricsConfigError(str(error)) from error
    if args.lighthouse_runs < 1:
        raise MetricsConfigError("--lighthouse-runs must be at least 1.")
    if args.step_timeout_s <= 0:
        raise MetricsConfigError("--step-timeout-s must be positive.")
    if args.container and not args.docker_image:
        raise MetricsConfigError("--container needs --docker-image (the image it must have been started from).")
    return base_url


@dataclass(frozen=True)
class NodeTools:
    """Paths of the pinned Node tools in tools/web-metrics."""

    node: str
    lighthouse_cli: Path
    axe_script: Path
    lighthouse_version: str
    axe_version: str


def package_version(package_dir: Path) -> str:
    """The version field of an installed npm package's package.json."""
    manifest = json.loads((package_dir / "package.json").read_text(encoding="utf-8"))
    return str(manifest["version"])


def find_node_tools(tools_dir: Path, need_lighthouse: bool) -> NodeTools:
    """The installed Lighthouse CLI and axe-core script; MetricsConfigError with the fix when they are missing."""
    node = shutil.which("node")
    if node is None:
        raise MetricsConfigError("Node.js is not on PATH (Node 22 is needed for Lighthouse).")
    modules = tools_dir / "node_modules"
    axe_script = modules / "axe-core" / "axe.min.js"
    lighthouse_cli = modules / "lighthouse" / "cli" / "index.js"
    missing = [path for path in (axe_script, lighthouse_cli) if not path.is_file()]
    if not need_lighthouse and axe_script.is_file():
        missing = []
    if missing:
        names = ", ".join(display_path(path) for path in missing)
        raise MetricsConfigError(f"Missing {names}. Run `cd {display_path(tools_dir)} && npm ci` first.")
    lighthouse_version = package_version(modules / "lighthouse") if lighthouse_cli.is_file() else "not installed"
    return NodeTools(node, lighthouse_cli, axe_script, lighthouse_version, package_version(modules / "axe-core"))


def mirror_math_round(value: float) -> int:
    """JavaScript's Math.round (halves go up), the rounding Lighthouse's own report uses for 0-100 scores."""
    return math.floor(value + 0.5)


def summarize_axe(result: Mapping[str, Any]) -> dict[str, Any]:
    """Counts per impact level from an axe.run() result, plus the serious and critical rules in detail.

    violations_by_impact counts rules; nodes_by_impact counts the page elements those rules flagged. incomplete
    items are ones axe could not decide (a person has to look), counted the same way.
    """
    violations = list(result.get("violations") or [])
    incomplete = list(result.get("incomplete") or [])
    rules_by_impact = Counter(str(rule.get("impact") or "unknown") for rule in violations)
    nodes_by_impact: Counter[str] = Counter()
    for rule in violations:
        nodes_by_impact[str(rule.get("impact") or "unknown")] += len(rule.get("nodes") or [])
    incomplete_by_impact = Counter(str(rule.get("impact") or "unknown") for rule in incomplete)
    serious_or_critical = [describe_axe_rule(rule) for rule in violations if rule.get("impact") in SERIOUS_IMPACTS]
    others = [describe_axe_rule(rule) for rule in violations if rule.get("impact") not in SERIOUS_IMPACTS]
    return {
        "violations_by_impact": {impact: rules_by_impact.get(impact, 0) for impact in AXE_IMPACTS},
        "nodes_by_impact": {impact: nodes_by_impact.get(impact, 0) for impact in AXE_IMPACTS},
        "incomplete_by_impact": {impact: incomplete_by_impact.get(impact, 0) for impact in AXE_IMPACTS},
        "serious_or_critical_rules": serious_or_critical,
        "other_violation_rules": others,
    }


def describe_axe_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    """One violated axe rule: id, impact, help text, tags, how many elements, and the first three of them."""
    nodes = list(rule.get("nodes") or [])
    return {
        "id": rule.get("id"),
        "impact": rule.get("impact"),
        "help": rule.get("help"),
        "tags": list(rule.get("tags") or []),
        "nodes": len(nodes),
        "first_targets": [node.get("target") for node in nodes[:3]],
    }


def axe_totals(scans: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Totals over every scan: serious and critical violations (rules and nodes) and how many scans had any."""
    totals: dict[str, Any] = {"scans": len(scans)}
    for impact in ("critical", "serious"):
        totals[f"{impact}_violation_rules"] = sum(scan["violations_by_impact"][impact] for scan in scans)
        totals[f"{impact}_nodes"] = sum(scan["nodes_by_impact"][impact] for scan in scans)
    totals["scans_with_serious_or_critical"] = sum(1 for scan in scans if scan["serious_or_critical_rules"])
    return totals


def summarize_lighthouse(report: Mapping[str, Any]) -> dict[str, Any]:
    """The category scores (0-1 as Lighthouse gives them, and 0-100 as its report shows them) and key metrics."""
    settings = report.get("configSettings") or {}
    categories = report.get("categories") or {}
    audits = report.get("audits") or {}
    scores: dict[str, float | None] = {}
    displayed: dict[str, int | None] = {}
    for name in LIGHTHOUSE_CATEGORIES:
        score = (categories.get(name) or {}).get("score")
        scores[name] = score
        displayed[name] = None if score is None else mirror_math_round(score * 100)
    metrics = {name: (audits.get(name) or {}).get("numericValue") for name in LIGHTHOUSE_METRICS}
    failed_audits = sorted(
        audit_id
        for category in ("accessibility", "best-practices")
        for ref in (categories.get(category) or {}).get("auditRefs") or []
        if (audit_id := ref.get("id")) in audits
        and (ref.get("weight") or 0) > 0
        and audits[audit_id].get("score") is not None
        and audits[audit_id]["score"] < 1
    )
    runtime_error = report.get("runtimeError")
    return {
        "scores": scores,
        "scores_0_100": displayed,
        "metrics": metrics,
        "failed_weighted_audits": failed_audits,
        "form_factor": settings.get("formFactor"),
        "throttling_method": settings.get("throttlingMethod"),
        "lighthouse_version": report.get("lighthouseVersion"),
        "runtime_error": runtime_error.get("code") if isinstance(runtime_error, Mapping) else runtime_error,
        "run_warnings": list(report.get("runWarnings") or []),
    }


def median_of_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Median of each category score and metric over the runs that produced one (None when none did)."""
    median_scores: dict[str, float | None] = {}
    for name in LIGHTHOUSE_CATEGORIES:
        values = [run["scores"][name] for run in runs if run.get("scores", {}).get(name) is not None]
        median_scores[name] = statistics.median(values) if values else None
    median_metrics: dict[str, float | None] = {}
    for name in LIGHTHOUSE_METRICS:
        values = [run["metrics"][name] for run in runs if run.get("metrics", {}).get(name) is not None]
        median_metrics[name] = statistics.median(values) if values else None
    return {
        "scores": median_scores,
        "scores_0_100": {
            name: None if value is None else mirror_math_round(value * 100) for name, value in median_scores.items()
        },
        "metrics": median_metrics,
        "runs_with_scores": sum(1 for run in runs if run.get("scores", {}).get("performance") is not None),
    }


def choose_action(recommendation: str, action_checks: Sequence[Mapping[str, Any]]) -> str | None:
    """The action the policy suggests if the checkpoint allows it for this user, else the first allowed action."""
    allowed = {str(check["action"]) for check in action_checks if check.get("allowed")}
    suggested = RECOMMENDATION_TO_ACTION.get(recommendation)
    if suggested in allowed:
        return suggested
    for action in ACTION_ORDER:
        if action in allowed:
            return action
    return None


def needs_override(action: str, action_checks: Sequence[Mapping[str, Any]]) -> bool:
    """True when the decision panel will ask for an override reason for this action."""
    return any(check["action"] == action and check.get("needs_override_reason") for check in action_checks)


def lighthouse_command(tools: NodeTools, url: str, port: int, preset: str) -> list[str]:
    """The Lighthouse CLI command for one run against the Chromium already listening on port."""
    if preset not in LIGHTHOUSE_PRESETS:
        raise MetricsConfigError(f"Unknown Lighthouse preset {preset!r}; choose one of {list(LIGHTHOUSE_PRESETS)}.")
    command = [
        tools.node,
        str(tools.lighthouse_cli),
        url,
        f"--port={port}",
        "--output=json",
        "--output-path=stdout",
        "--quiet",
        "--disable-storage-reset",
        f"--only-categories={','.join(LIGHTHOUSE_CATEGORIES)}",
    ]
    if preset == "desktop":
        command.append("--preset=desktop")
    return command


def merge_run(output: Path, target: str, run_entry: dict[str, Any]) -> dict[str, Any]:
    """The results document with this target's entry replaced and other targets kept."""
    document: dict[str, Any] = {"description": RESULTS_DESCRIPTION, "runs": {}}
    if output.is_file():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except ValueError as error:
            raise MetricsConfigError(f"{display_path(output)} is not valid JSON; move it away first.") from error
        if isinstance(existing, dict) and isinstance(existing.get("runs"), dict):
            document["runs"] = dict(existing["runs"])
    document["runs"][target] = run_entry
    document["runs"] = dict(sorted(document["runs"].items()))
    return document


def wait_for_health(
    client: httpx.Client,
    wait_s: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> httpx.Response:
    """GET /api/health until it answers 200 or wait_s has passed (a container needs a few seconds to start)."""
    deadline = clock() + wait_s
    while True:
        try:
            response = client.get("/api/health")
            if response.status_code == 200:
                return response
            last = f"HTTP {response.status_code}"
        except httpx.HTTPError as error:
            last = type(error).__name__
        if clock() >= deadline:
            raise MetricsConfigError(f"The server did not become healthy within {wait_s} s (last answer: {last}).")
        sleep(HEALTH_POLL_S)


def read_server_config(
    base_url: str,
    transport: httpx.BaseTransport | None = None,
    wait_s: float = DEFAULT_WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """/api/health and /api/config of the target; MetricsConfigError when it stays down or has no dev login."""
    with httpx.Client(base_url=base_url, timeout=30, transport=transport) as client:
        health = wait_for_health(client, wait_s, sleep)
        try:
            config = client.get("/api/config")
        except httpx.HTTPError as error:
            raise MetricsConfigError(f"The server at {base_url} did not answer /api/config: {error}") from error
    if config.status_code != 200:
        raise MetricsConfigError(f"/api/config answered HTTP {config.status_code}.")
    config_body = config.json()
    if config_body.get("dev_auth") is not True:
        raise MetricsConfigError(
            "The server runs without the dev login, so the script cannot sign in. Start it with UWC_DEV_AUTH=1 "
            "(only ever on this machine; a Space refuses it)."
        )
    health_body = health.json()
    return {
        "version": health_body.get("version"),
        "git_sha": health_body.get("git_sha"),
        "product_name": config_body.get("product_name"),
        "disclaimer": config_body.get("disclaimer"),
        "dev_auth": config_body.get("dev_auth"),
        "is_space": config_body.get("is_space"),
        "payments_enabled": config_body.get("payments_enabled"),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Step log, console and API-answer recording
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Observer:
    """Browser console errors and API answers with HTTP >= 400, each tagged with the flow and step it happened in."""

    flow: str = ""
    step: str = ""
    expected_console: tuple[str, ...] = ()
    expected_status: tuple[int, ...] = ()
    console: list[dict[str, Any]] = field(default_factory=list)
    http_errors: list[dict[str, Any]] = field(default_factory=list)
    api_answers: list[dict[str, Any]] = field(default_factory=list)

    def on_console(self, message: Any) -> None:
        """Playwright 'console' event: keep errors."""
        if message.type == "error":
            self.add_console(str(message.text))

    def on_page_error(self, error: Any) -> None:
        """Playwright 'pageerror' event: an uncaught exception in the page."""
        self.add_console(f"pageerror: {error}")

    def add_console(self, text: str) -> None:
        """Record one console error and whether the current step expected it."""
        expected = any(pattern in text for pattern in self.expected_console)
        self.console.append({"flow": self.flow, "step": self.step, "text": text[:300], "expected": expected})

    def on_response(self, response: Any) -> None:
        """Playwright 'response' event: remember every /api/ answer (time, status); keep the failed ones."""
        parts = urlsplit(response.url)
        if not parts.path.startswith("/api/"):
            return
        answer = {
            "flow": self.flow,
            "step": self.step,
            "method": response.request.method,
            "path": parts.path,
            "status": response.status,
            "at": time.perf_counter(),
        }
        self.api_answers.append(answer)
        if response.status >= 400:
            failed = {key: answer[key] for key in ("flow", "step", "method", "path", "status")}
            failed["expected"] = response.status in self.expected_status
            self.http_errors.append(failed)

    def first_answer(self, method: str, path_pattern: str, since: float) -> dict[str, Any] | None:
        """The first API answer after `since` whose method matches and whose path matches the regex."""
        pattern = re.compile(path_pattern)
        for answer in self.api_answers:
            if answer["at"] >= since and answer["method"] == method and pattern.search(answer["path"]):
                return answer
        return None


@dataclass
class FlowLog:
    """The steps of one flow: name, passed, seconds, detail, plus any facts a step records."""

    name: str
    observer: Observer
    catch: tuple[type[BaseException], ...]
    # Called with (flow, step) when a step fails; returns where it saved a screenshot of the page, or None.
    on_failure: Callable[[str, str], str | None] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @contextlib.contextmanager
    def step(
        self, name: str, expected_console: tuple[str, ...] = (), expected_status: tuple[int, ...] = ()
    ) -> Iterator[dict[str, Any]]:
        """Time one step. A failure is recorded and stops the flow (FlowStopped)."""
        record: dict[str, Any] = {"name": name, "passed": False, "seconds": None, "detail": ""}
        self.steps.append(record)
        self.observer.flow, self.observer.step = self.name, name
        self.observer.expected_console, self.observer.expected_status = expected_console, expected_status
        started = time.perf_counter()
        try:
            yield record
        except self.catch as error:
            record["seconds"] = round(time.perf_counter() - started, 3)
            record["detail"] = f"{type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}"[:400]
            if self.on_failure is not None:
                record["failure_screenshot"] = self.on_failure(self.name, name)
            raise FlowStopped(name) from error
        finally:
            self.observer.expected_console, self.observer.expected_status = (), ()
        record["seconds"] = round(time.perf_counter() - started, 3)
        record["passed"] = True

    def summary(self) -> dict[str, Any]:
        """The flow's entry for the results file."""
        console = [item for item in self.observer.console if item["flow"] == self.name]
        http_errors = [item for item in self.observer.http_errors if item["flow"] == self.name]
        unexpected_console = [item for item in console if not item["expected"]]
        unexpected_http = [item for item in http_errors if not item["expected"]]
        steps_passed = bool(self.steps) and all(step["passed"] for step in self.steps)
        return {
            "name": self.name,
            "passed": steps_passed and not unexpected_console and not unexpected_http,
            "steps": self.steps,
            "facts": self.facts,
            "console_errors": console,
            "api_errors": http_errors,
            "unexpected_console_errors": len(unexpected_console),
            "unexpected_api_errors": len(unexpected_http),
        }


def run_flow(log: FlowLog, body: Callable[[FlowLog], None]) -> dict[str, Any]:
    """Run one flow's body; a stopped flow keeps the steps it recorded."""
    with contextlib.suppress(FlowStopped):
        body(log)
    return log.summary()


def require(condition: bool, message: str) -> None:
    """Raise CheckFailed(message) unless condition holds."""
    if not condition:
        raise CheckFailed(message)


def since_ms(started: float, answer: Mapping[str, Any] | None) -> float | None:
    """Milliseconds from started to an API answer (None when there was no answer)."""
    return None if answer is None else round((answer["at"] - started) * 1000, 1)


def free_port() -> int:
    """An unused local TCP port (for Chromium's DevTools endpoint, which Lighthouse attaches to)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def optimize_png(path: Path) -> str:
    """Make a screenshot smaller without changing its pixels (scripts/optimize_pngs.py). Says what happened."""
    outcome = optimize_file(path)
    if not outcome.replaced:
        return f"kept as Chromium wrote it ({outcome.before} bytes; the {outcome.tool} re-encode was not smaller)"
    return f"re-encoded losslessly with {outcome.tool}: {outcome.before} -> {outcome.after} bytes"


# ---------------------------------------------------------------------------------------------------------------------
# The browser session
# ---------------------------------------------------------------------------------------------------------------------


class BrowserSession:
    """One headless Chromium profile: the flows, the axe scans, the screenshots and the Lighthouse runs."""

    def __init__(self, base_url: str, config: Mapping[str, Any], args: argparse.Namespace, tools: NodeTools) -> None:
        """Remember the target and the options; the browser starts in run()."""
        self.base_url = base_url
        self.config = config
        self.args = args
        self.tools = tools
        self.timeout_ms = args.step_timeout_s * 1000
        self.observer = Observer()
        self.axe_source = tools.axe_script.read_text(encoding="utf-8")
        self.scans: list[dict[str, Any]] = []
        self.flows: list[dict[str, Any]] = []
        self.screenshots: list[dict[str, Any]] = []
        self.lighthouse_pages: list[dict[str, Any]] = []
        self.username = f"metrics-founder-{secrets.token_hex(3)}"
        self.plan_id: str | None = None
        self.lender_file_id: str | None = None
        self.browser_facts: dict[str, Any] = {}

    # -- plumbing ---------------------------------------------------------------------------------------------------

    def url(self, path: str) -> str:
        """Absolute URL of a path on the target."""
        return f"{self.base_url}{path}"

    def new_log(self, name: str) -> FlowLog:
        """A step log for one flow (Playwright errors and failed checks stop the flow)."""
        from playwright.sync_api import Error as PlaywrightError

        return FlowLog(name, self.observer, (PlaywrightError, CheckFailed), on_failure=self.failure_screenshot)

    def failure_screenshot(self, flow: str, step: str) -> str | None:
        """A screenshot of the page as it was when a step failed (None if even that fails)."""
        from playwright.sync_api import Error as PlaywrightError

        try:
            self.screenshot(f"failed-{flow}-{step}")
        except PlaywrightError:
            return None
        return self.screenshots[-1]["file"]

    def goto(self, path: str) -> Any:
        """Open a path and wait until the network is quiet (the pages load their data after the first paint)."""
        response = self.page.goto(self.url(path), wait_until="networkidle", timeout=self.timeout_ms)
        self.page.evaluate("document.fonts.ready")
        return response

    def scan(self, name: str, path_label: str) -> None:
        """axe-core on the current page state, in the light and the dark color scheme (then back to light)."""
        if self.page.evaluate("() => typeof window.axe") != "object":
            self.page.evaluate(self.axe_source)
        for scheme in COLOR_SCHEMES:
            self.page.emulate_media(color_scheme=scheme)
            self.page.wait_for_timeout(300)
            result = self.page.evaluate(
                "async () => await window.axe.run(document, {resultTypes: ['violations', 'incomplete']})"
            )
            summary = summarize_axe(result)
            engine = result.get("testEngine") or {}
            self.scans.append({"page": name, "path": path_label, "color_scheme": scheme, **summary})
            self.browser_facts.setdefault("axe_engine", f"{engine.get('name')} {engine.get('version')}")
        self.page.emulate_media(color_scheme="light")

    def screenshot(self, name: str) -> None:
        """A full-page PNG of the current page into --screenshots-dir, optimized losslessly when Pillow is there."""
        directory: Path = self.args.screenshots_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.png"
        self.page.screenshot(path=str(path), full_page=True)
        self.screenshots.append({"file": display_path(path), "note": optimize_png(path)})

    def wait_text(self, pattern: str | re.Pattern[str]) -> None:
        """Wait until visible text matching pattern appears."""
        self.page.get_by_text(pattern).first.wait_for(timeout=self.timeout_ms)

    def wait_heading(self, name: str | re.Pattern[str]) -> None:
        """Wait until a heading with this accessible name is visible."""
        self.page.get_by_role("heading", name=name).first.wait_for(timeout=self.timeout_ms)

    # -- flows ------------------------------------------------------------------------------------------------------

    def signed_out(self, log: FlowLog) -> None:
        """Public pages (axe) and the signed-out prompt with a ?next= link."""
        with log.step("public_pages_load"):
            for name, path in (("landing", "/"), ("signin", "/signin/"), ("about", "/about/")):
                response = self.goto(path)
                require(
                    response is not None and response.status == 200, f"{path} answered {response and response.status}"
                )
                self.scan(f"{name} (signed out)", path)
        with log.step("signed_out_studio_asks_to_sign_in"):
            self.goto("/studio/")
            self.wait_heading("Sign in to open Founder Studio")
            require(self.page.locator("a[href*='next=']").count() > 0, "the sign-in link does not remember the page")
            self.scan("studio (signed out)", "/studio/")
        with log.step("unknown_page_is_404", expected_console=(EXPECTED_404_CONSOLE,)):
            response = self.goto(UNKNOWN_PAGE)
            require(response is not None and response.status == 404, f"answered {response and response.status}")
            self.wait_heading("Page not found")
            self.scan("404 page", UNKNOWN_PAGE)

    def founder(self, log: FlowLog) -> None:
        """Dev login, the wizard on the SYNTHETIC metro, the results page, edited targets and 'Apply this change'."""
        page = self.page
        with log.step("dev_login_form"):
            self.goto("/signin/")
            page.get_by_label("Username").fill(self.username)
            page.get_by_role("combobox", name="Role").click()
            page.get_by_role("option", name="Founder", exact=True).click()
            page.get_by_role("button", name="Sign in (dev)").click()
            page.wait_for_url("**/studio/", timeout=self.timeout_ms)
            self.wait_text("Your plans")
            self.scan("studio plans list", "/studio/")
        with log.step("wizard_city"):
            page.get_by_role("link", name=re.compile("Start a plan")).first.click()
            self.wait_heading("Which city?")
            self.scan("wizard: city step", "/studio/new/")
            page.locator(f"label[for='metro-{DEMO_PLAN['metro_id']}']").click()
            page.get_by_role("button", name="Continue").click()
            self.wait_heading("What kind of business?")
        with log.step("wizard_business_type"):
            page.locator(f"label[for='business-{DEMO_PLAN['business_type_id']}']").click()
            page.get_by_role("button", name="Continue").click()
            self.wait_heading("Describe your idea")
        with log.step("wizard_idea"):
            page.locator("#idea").fill(DEMO_PLAN["idea"])
            page.get_by_role("button", name="Continue").click()
            self.wait_heading("The money plan")
        with log.step("wizard_money"):
            for field_id in ("budget", "equity", "loan", "term", "rate"):
                page.locator(f"#{field_id}").fill(DEMO_PLAN[field_id])
            page.get_by_role("button", name="Continue").click()
            self.wait_heading("Candidate sites")
        with log.step("wizard_sites") as record:
            started = time.perf_counter()
            page.get_by_role("button", name=re.compile("^Suggest areas")).click()
            # The button's name is its visible text (WCAG 2.5.3); which area it is comes from its description.
            use_first = page.get_by_role("button", name="Use this area").first
            use_first.wait_for(timeout=self.timeout_ms)
            record["suggest_areas_ms"] = since_ms(
                started, self.observer.first_answer("POST", r"/areas/suggest$", started)
            )
            suggested = re.search(r"suggested-area-(\w+)", use_first.get_attribute("aria-describedby") or "")
            use_first.click()
            tract_select = page.get_by_label("Or pick a census tract from the list")
            values = tract_select.locator("option").evaluate_all("options => options.map(option => option.value)")
            used = suggested.group(1) if suggested else None
            # The tract with the lowest id that is not the suggested one: the same second site on every run.
            other = min((value for value in values if value and value != used), default=None)
            require(other is not None, "the tract list offers no other tract")
            tract_select.select_option(str(other))
            page.get_by_role("button", name="Add as a site").click()
            self.wait_text(re.compile(r"\(2 of 3\)"))
            self.scan("wizard: sites step (map, list, suggestions)", "/studio/new/")
            page.get_by_role("button", name="Continue").click()
            self.wait_heading("Check and create the plan")
            self.scan("wizard: review step", "/studio/new/")
        with log.step("create_plan_and_results") as record:
            started = time.perf_counter()
            with page.expect_response(self.is_path_answer, timeout=self.timeout_ms) as path_info:
                page.get_by_role("button", name="Create plan").click()
                page.wait_for_url(re.compile(r"/studio/project/\?id=[0-9a-f]{16}"), timeout=self.timeout_ms)
            plan_match = re.search(r"id=([0-9a-f]{16})", page.url)
            require(plan_match is not None, f"no plan id in {page.url}")
            self.plan_id = plan_match.group(1) if plan_match else None
            path_body = path_info.value.json()
            self.wait_text(PATH_STATUS_TEXT)
            record["rendered_ms"] = round((time.perf_counter() - started) * 1000, 1)
            for key, method, pattern in (
                ("create_answer_ms", "POST", r"/api/studio/projects$"),
                ("site_scores_answer_ms", "POST", r"/sites/score$"),
                ("feasibility_answer_ms", "POST", r"/feasibility$"),
                ("path_answer_ms", "POST", r"/path-to-approval$"),
            ):
                record[key] = since_ms(started, self.observer.first_answer(method, pattern, started))
            log.facts["path_status_default_targets"] = path_body.get("status")
            log.facts["path_mode"] = (path_body.get("settings") or {}).get("mode")
        with log.step("results_page_labels"):
            disclaimer = str(self.config.get("disclaimer") or "")
            require(bool(disclaimer) and page.get_by_text(disclaimer).count() > 0, "the disclaimer is not on the page")
            require(page.get_by_text("simulated, based on stated assumptions").count() > 0, "no simulation label")
            require(page.get_by_text(re.compile("simulated futures")).count() > 0, "no 'simulated futures' gauge text")
            require(page.get_by_text("SYNTHETIC").count() > 0, "no SYNTHETIC label on a SYNTHETIC-metro plan")
            self.scan("studio results", "/studio/project/?id=<plan>")
            self.screenshot("studio-results-1280-light")
        with log.step("results_tables_open"):
            # The chart tables scroll inside their own box: scan them open too (keyboard access to scrolling boxes).
            toggles = page.locator("figure details > summary")
            require(toggles.count() > 0, "no chart table toggle on the results page")
            for index in range(toggles.count()):
                toggles.nth(index).click()  # open
            self.scan("studio results: chart tables open", "/studio/project/?id=<plan>")
            for index in range(toggles.count()):
                toggles.nth(index).click()  # close again
        with log.step("edit_targets_and_apply_change") as record:
            page.get_by_role("button", name="Edit the targets").click()
            for name, value in DEMO_TARGET_EDITS.items():
                page.locator(f"#target-{name}").fill(value)
            with page.expect_response(self.is_path_answer, timeout=self.timeout_ms) as edited_info:
                page.get_by_role("button", name="Search again with these targets").click()
            edited = edited_info.value.json()
            log.facts["path_status_edited_targets"] = edited.get("status")
            self.wait_text(PATH_STATUS_TEXT)
            apply_buttons = page.locator("button:has-text('Apply this change'):enabled")
            if edited.get("status") != "paths_found" or apply_buttons.count() == 0:
                record["outcome"] = "no applicable change-set was found with the edited targets; nothing applied"
                return
            started = time.perf_counter()
            with page.expect_response(self.is_path_answer, timeout=self.timeout_ms) as rerun_info:
                apply_buttons.first.click()
                self.wait_text(re.compile(r"Your plan is now version 2"))
            rerun = rerun_info.value.json()
            require(rerun.get("project_version") == 2, f"the re-run used version {rerun.get('project_version')}")
            record["apply_put_ms"] = since_ms(
                started, self.observer.first_answer("PUT", r"/api/studio/projects/", started)
            )
            record["rerun_path_ms"] = round((time.perf_counter() - started) * 1000, 1)
            record["outcome"] = "applied the first change-set; the plan is version 2 and was re-run"

    def is_path_answer(self, response: Any) -> bool:
        """True for an answer of POST .../path-to-approval."""
        return response.request.method == "POST" and response.url.endswith("/path-to-approval")

    def lender(self, log: FlowLog) -> None:
        """The demo queue, the first file, and one DEMO decision through the decision panel."""
        page = self.page
        with log.step("demo_queue"):
            self.goto("/lender/")
            first_link = page.locator("a[href^='/lender/file/?id=']").first
            first_link.wait_for(timeout=self.timeout_ms)
            self.wait_text(re.compile("The copilot never decides"))
            self.scan("lender demo queue", "/lender/")
        with log.step("open_first_file") as record:
            started = time.perf_counter()
            with page.expect_response(
                lambda response: response.request.method == "GET" and "/api/lender/files/" in response.url,
                timeout=self.timeout_ms,
            ) as file_info:
                first_link.click()
            data = file_info.value.json()
            record["file_answer_ms"] = round((time.perf_counter() - started) * 1000, 1)
            self.wait_heading("Your decision")
            record["rendered_ms"] = round((time.perf_counter() - started) * 1000, 1)
            self.lender_file_id = str(data.get("id"))
            screening = data.get("screening") or {}
            log.facts.update(
                {
                    "file_id": self.lender_file_id,
                    "line": data.get("line"),
                    "reference_recommendation": data.get("reference_recommendation"),
                    "your_reviewer_role": data.get("your_reviewer_role"),
                    "memo_writer": data.get("memo_writer"),
                    "memo_check_passed": (data.get("memo_check") or {}).get("passed"),
                    "guideline_search": data.get("guideline_search"),
                    "screening_configured_lists": screening.get("configured_lists"),
                    "screening_lists_used": [
                        {"source": item.get("source"), "file": item.get("file_name"), "date": item.get("snapshot_date")}
                        for item in screening.get("lists_used") or []
                    ],
                    "screening_statuses": sorted({str(item.get("status")) for item in screening.get("results") or []}),
                }
            )
            require(data.get("label") == "SYNTHETIC DEMO", f"label is {data.get('label')!r}")
            self.wait_text("SYNTHETIC DEMO")
            self.scan("lender file (before the decision)", "/lender/file/?id=<file>")
        with log.step("record_demo_decision") as record:
            checks = data.get("action_checks") or []
            action = choose_action(str(data.get("reference_recommendation")), checks)
            require(action is not None, "the checkpoint allows no action for this user")
            page.get_by_role("button", name=ACTION_LABELS[str(action)], exact=True).click()
            page.locator("#rationale").fill(DEMO_RATIONALE)
            if needs_override(str(action), checks):
                reasons = page.locator("#override-reason option").evaluate_all("items => items.map(item => item.value)")
                reason = next((value for value in reasons if value), None)
                require(reason is not None, "the panel offers no override reason")
                page.locator("#override-reason").select_option(str(reason))
            started = time.perf_counter()
            with page.expect_response(
                lambda response: response.request.method == "POST" and response.url.endswith("/decision"),
                timeout=self.timeout_ms,
            ) as decision_info:
                page.get_by_role("button", name=re.compile("^Record:")).click()
            decision = decision_info.value
            require(decision.status == 200, f"the decision answered HTTP {decision.status}")
            self.wait_heading(re.compile(r"Your decision record \(DEMO\)"))
            record["decision_answer_ms"] = round((time.perf_counter() - started) * 1000, 1)
            body = decision.json()
            log.facts["action"] = action
            log.facts["final_status"] = (body.get("decision") or {}).get("final_status")
            log.facts["record_label"] = body.get("label")
            self.scan("lender file (after the decision)", "/lender/file/?id=<file>")
            self.screenshot("lender-decision-1280-light")
            page.locator("summary:has-text('Read the record')").first.click()
            self.scan("lender file: decision record open", "/lender/file/?id=<file>")

    def errors(self, log: FlowLog) -> None:
        """Signed-in error paths: an unknown plan id and the wizard's own input check."""
        with log.step("unknown_plan_says_not_found", (EXPECTED_404_CONSOLE,), (404,)):
            started = time.perf_counter()
            self.goto(f"/studio/project/?id={UNKNOWN_PLAN_ID}")
            self.wait_text("Not found")
            answer = self.observer.first_answer("GET", rf"/api/studio/projects/{UNKNOWN_PLAN_ID}$", started)
            require(answer is not None and answer["status"] == 404, f"the API answered {answer and answer['status']}")
            self.scan("studio results: unknown plan (error state)", "/studio/project/?id=<unknown>")
            self.screenshot("error-unknown-plan-1280-light")
        with log.step("wizard_requires_a_city"):
            self.goto("/studio/new/")
            self.wait_heading("Which city?")
            self.page.get_by_role("button", name="Continue").click()
            self.page.get_by_role("alert").first.wait_for(timeout=self.timeout_ms)
            self.wait_heading("Which city?")

    def delete_plan(self, log: FlowLog) -> None:
        """Delete the plan from its page, then check that the page says it is gone."""
        require(self.plan_id is not None, "no plan was created")
        with log.step("delete_plan", (EXPECTED_404_CONSOLE,), (404,)):
            self.goto(f"/studio/project/?id={self.plan_id}")
            self.page.get_by_role("button", name="Delete plan").click()
            started = time.perf_counter()
            self.page.get_by_role("button", name="Delete for good").click()
            self.page.wait_for_url("**/studio/", timeout=self.timeout_ms)
            answer = self.observer.first_answer("DELETE", rf"/api/studio/projects/{self.plan_id}$", started)
            require(answer is not None and answer["status"] == 200, f"DELETE answered {answer and answer['status']}")
            self.goto(f"/studio/project/?id={self.plan_id}")
            self.wait_text("Not found")

    # -- Lighthouse -------------------------------------------------------------------------------------------------

    def lighthouse(self) -> None:
        """Every Lighthouse run: three pages x the two presets x --lighthouse-runs."""
        pages = [("landing", "/", "/")]
        if self.plan_id:
            pages.append(("studio results", f"/studio/project/?id={self.plan_id}", "/studio/project/?id=<plan>"))
        if self.lender_file_id:
            pages.append(("lender file", f"/lender/file/?id={self.lender_file_id}", "/lender/file/?id=<file>"))
        for name, path, path_label in pages:
            for preset in LIGHTHOUSE_PRESETS:
                runs = [self.lighthouse_once(path, preset) for _ in range(self.args.lighthouse_runs)]
                self.lighthouse_pages.append(
                    {"page": name, "path": path_label, "preset": preset, "runs": runs, "median": median_of_runs(runs)}
                )

    def lighthouse_once(self, path: str, preset: str) -> dict[str, Any]:
        """One Lighthouse run after clearing the browser's HTTP cache (cookies are kept: the session stays)."""
        session = self.context.new_cdp_session(self.page)
        session.send("Network.clearBrowserCache")
        session.detach()
        command = lighthouse_command(self.tools, self.url(path), self.cdp_port, preset)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=LIGHTHOUSE_TIMEOUT_S, check=False
            )
        except subprocess.TimeoutExpired:
            return {"error": f"Lighthouse did not finish within {LIGHTHOUSE_TIMEOUT_S} s"}
        seconds = round(time.perf_counter() - started, 1)
        if completed.returncode != 0:
            return {"error": (completed.stderr.strip().splitlines() or ["no output"])[-1][:400], "seconds": seconds}
        summary = summarize_lighthouse(json.loads(completed.stdout))
        summary["seconds"] = seconds
        return summary

    # -- the whole session --------------------------------------------------------------------------------------------

    def run(self) -> None:
        """Start Chromium, run every flow, the scans and Lighthouse, then close it."""
        from playwright.sync_api import sync_playwright

        self.cdp_port = free_port()
        with tempfile.TemporaryDirectory(prefix="uwc-web-metrics-") as profile_dir, sync_playwright() as playwright:
            self.context = playwright.chromium.launch_persistent_context(
                profile_dir,
                channel="chromium",
                headless=True,
                viewport=VIEWPORT,
                color_scheme="light",
                args=[f"--remote-debugging-port={self.cdp_port}"],
            )
            try:
                self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
                self.page.set_default_timeout(self.timeout_ms)
                self.page.on("console", self.observer.on_console)
                self.page.on("pageerror", self.observer.on_page_error)
                self.page.on("response", self.observer.on_response)
                self.browser_facts.update(
                    {
                        "chromium_version": self.context.browser.version if self.context.browser else None,
                        "playwright_version": playwright_version(),
                        "headless": "new headless (Playwright channel 'chromium')",
                        "viewport": VIEWPORT,
                    }
                )
                self.flows.append(run_flow(self.new_log("signed_out"), self.signed_out))
                self.flows.append(run_flow(self.new_log("founder_synthetic_plan"), self.founder))
                self.flows.append(run_flow(self.new_log("lender_demo_decision"), self.lender))
                self.flows.append(run_flow(self.new_log("error_paths"), self.errors))
                if not self.args.skip_lighthouse:
                    self.lighthouse()
                if self.plan_id:
                    self.flows.append(run_flow(self.new_log("founder_delete_plan"), self.delete_plan))
            finally:
                self.context.close()


def playwright_version() -> str:
    """The installed Playwright for Python version."""
    from importlib.metadata import version

    return version("playwright")


# ---------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------


def build_entry(
    session: BrowserSession,
    args: argparse.Namespace,
    base_url: str,
    server: Mapping[str, Any],
    docker: Mapping[str, Any] | None,
    container: Mapping[str, Any] | None,
    provenance_run: RunProvenance,
) -> dict[str, Any]:
    """The results entry for this target."""
    lighthouse_runs = [run for page in session.lighthouse_pages for run in page["runs"]]
    return {
        "target": {"kind": args.target, "label": TARGET_LABELS[args.target], "url": base_url},
        "reproducibility": reproducibility(server.get("git_sha")),
        "measured_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "auth": {"mode": "dev", "note": DEV_LOGIN_NOTE, "username": session.username},
        "server": dict(server),
        "browser": session.browser_facts,
        "demo_inputs": {
            "label": "SYNTHETIC test inputs typed by this script (design choices, not data or benchmarks)",
            "plan": DEMO_PLAN,
            "target_edits_percent": DEMO_TARGET_EDITS,
            "decision_rationale": DEMO_RATIONALE,
        },
        "e2e": {"all_flows_passed": all(flow["passed"] for flow in session.flows), "flows": session.flows},
        "axe": {
            "engine": session.browser_facts.get("axe_engine"),
            "rule_set": "axe-core's default rules (no runOnly); resultTypes violations + incomplete",
            "color_schemes": list(COLOR_SCHEMES),
            "viewport": VIEWPORT,
            "totals": axe_totals(session.scans),
            "scans": session.scans,
        },
        "lighthouse": {
            "version": session.tools.lighthouse_version,
            "skipped": bool(args.skip_lighthouse),
            "categories": list(LIGHTHOUSE_CATEGORIES),
            "runs_per_page_and_preset": args.lighthouse_runs,
            "method": (
                "Lighthouse CLI attached over the DevTools port to the signed-in Chromium profile, "
                "--disable-storage-reset (keeps the session cookie), HTTP cache cleared through CDP before each run; "
                "'mobile' is Lighthouse's default config (simulated throttling), 'desktop' is --preset=desktop"
            ),
            "all_runs_scored": bool(lighthouse_runs) and all("scores" in run for run in lighthouse_runs),
            "pages": session.lighthouse_pages,
        },
        "screenshots": session.screenshots,
        "docker_image": docker,
        "container": container,
        "caveats": list(CAVEATS),
        "provenance": provenance_run.finish(
            corpus_checksum="not recorded (the guideline index inside the target is used; see the lender flow facts)",
            data_snapshots={
                "deployment_target": {"kind": args.target, "url": base_url},
                "tools": {
                    "lighthouse": session.tools.lighthouse_version,
                    "axe-core": session.tools.axe_version,
                    "playwright": session.browser_facts.get("playwright_version"),
                    "chromium": session.browser_facts.get("chromium_version"),
                    "node": node_version(session.tools.node),
                },
            },
        ),
    }


def node_version(node: str) -> str:
    """`node --version`, or 'unknown'."""
    try:
        completed = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip()


def print_summary(entry: Mapping[str, Any]) -> None:
    """A short human-readable summary."""
    print(f"Target: {entry['target']['kind']} ({entry['target']['url']})")
    checked = entry["reproducibility"]
    print(
        f"  reproducible: {checked['reproducible']}"
        + (f" ({'; '.join(checked['reasons'])})" if checked["reasons"] else "")
    )
    for flow in entry["e2e"]["flows"]:
        print(f"  [{'PASS' if flow['passed'] else 'FAIL'}] flow {flow['name']}")
        for step in flow["steps"]:
            mark = "ok " if step["passed"] else "BAD"
            print(f"      {mark} {step['name']} {step['seconds']} s {step['detail']}")
    totals = entry["axe"]["totals"]
    print(
        f"  axe: {totals['scans']} scans, critical rules {totals['critical_violation_rules']}, serious rules "
        f"{totals['serious_violation_rules']}, scans with serious/critical {totals['scans_with_serious_or_critical']}"
    )
    for page in entry["lighthouse"]["pages"]:
        print(f"  lighthouse {page['page']} ({page['preset']}): median {page['median']['scores_0_100']}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run everything, update the results file, and return the exit code."""
    args = parse_args(argv)
    try:
        base_url = check_arguments(args)
        tools = find_node_tools(args.tools_dir, need_lighthouse=not args.skip_lighthouse)
        server = read_server_config(base_url, wait_s=args.wait_s)
        docker = docker_image_facts(args.docker_image) if args.docker_image else None
        container = container_facts(args.container) if args.container else None
    except (MetricsConfigError, SmokeConfigError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    checked = reproducibility(server.get("git_sha"))
    if not checked["reproducible"] and not args.allow_unreproducible:
        print(
            "Error: refusing to measure an unreproducible build: " + "; ".join(checked["reasons"]) + ". Commit, "
            "rebuild with `make docker-build-stamped`, or pass --allow-unreproducible to record it marked as such.",
            file=sys.stderr,
        )
        return 2
    if "PLAYWRIGHT_BROWSERS_PATH" not in os.environ and DEFAULT_BROWSERS_PATH.is_dir():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(DEFAULT_BROWSERS_PATH)
    provenance_run = RunProvenance.start(GENERATED_BY)
    session = BrowserSession(base_url, server, args, tools)
    session.run()
    entry = build_entry(session, args, base_url, server, docker, container, provenance_run)
    try:
        document = merge_run(args.output, args.target, entry)
    except MetricsConfigError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    write_json_atomically(args.output, document)
    print_summary(entry)
    print(f"Wrote {display_path(args.output)} (runs.{args.target})")
    lighthouse_ok = args.skip_lighthouse or entry["lighthouse"]["all_runs_scored"]
    return 0 if entry["e2e"]["all_flows_passed"] and lighthouse_ok else 1


if __name__ == "__main__":
    sys.exit(main())
