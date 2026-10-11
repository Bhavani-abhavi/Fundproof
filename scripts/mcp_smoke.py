"""Smoke test of the MCP server as a real subprocess, driven by the official MCP Python SDK client.

    uv run python scripts/mcp_smoke.py                              # stdio (default)
    uv run python scripts/mcp_smoke.py --transport streamable-http  # HTTP on a free local port

stdio: the SDK client starts `python -m uwcopilot.mcp_server` itself and talks JSON-RPC over the child's
stdin/stdout, exactly as Claude Desktop or Claude Code would. streamable-http: this script starts the server with
`--transport streamable-http` on a free 127.0.0.1 port, waits for it, and connects to http://127.0.0.1:<port>/mcp.

What it checks (each check is recorded with pass/fail in the output file):
- the server lists exactly the four expected tools, each described as SYNTHETIC demo data;
- the guideline resources and the underwriting_review prompt are listed and readable;
- search_underwriting_guidelines returns k ranked hits with citations, twice in a row in the same order;
- get_guideline_section returns the section that the resource guidelines://<code>/<id> shows;
- score_application against a risk API URL with nothing listening returns status "error" (risk_api_unreachable)
  and no score. No risk service exists on this branch, so this is the only score path that can run for real;
- an unknown tool name and an unknown argument are refused.
Sanctions screening is not called: the default matcher loads the full watchlists (see uwcopilot.kyc.screening).

Needs the Qdrant index (`make index`) and the fastembed models it downloaded into .cache/models. No LLM is called.
Writes results/mcp_smoke_<transport>.json; the exit code is 0 only when every check passed. The timings are from
one run on whatever machine ran it and are not a benchmark.
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import anyio
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, get_default_environment

from uwcopilot.common.paths import REPO_ROOT, RESULTS_DIR
from uwcopilot.mcp_server.guideline_store import GuidelineStore, section_resource_uri
from uwcopilot.mcp_server.server import MCP_TOOL_BINDINGS
from uwcopilot.rag.index import make_qdrant_client
from uwcopilot.rag.settings import load_rag_settings

DEFAULT_QUERY = "What is the minimum debt service coverage ratio for a term loan?"
DEFAULT_K = 5
DEFAULT_MODE = "hybrid_rerank"
SERVER_MODULE = "uwcopilot.mcp_server"
READ_TIMEOUT_S = 300.0
HTTP_START_TIMEOUT_S = 120.0
INDEX_MANIFEST = RESULTS_DIR / "rag_index_manifest.json"


class SmokeSetupError(RuntimeError):
    """Raised when the smoke test cannot start (no index, server did not come up)."""


@dataclass
class CheckLog:
    """Named pass/fail checks, in the order they ran."""

    checks: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, passed: bool, detail: str = "") -> bool:
        """Add one check and return whether it passed."""
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})
        return bool(passed)

    @property
    def all_passed(self) -> bool:
        """True when at least one check ran and none failed."""
        return bool(self.checks) and all(check["passed"] for check in self.checks)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command line options."""
    parser = argparse.ArgumentParser(description="Smoke-test the MCP server over stdio or streamable HTTP.")
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=("dense", "bm25", "hybrid", "hybrid_rerank"))
    parser.add_argument("--output", type=Path, default=None, help="default: results/mcp_smoke_<transport>.json")
    return parser.parse_args(argv)


def free_local_port() -> int:
    """A TCP port on 127.0.0.1 that nothing is listening on right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def port_is_open(port: int) -> bool:
    """True when something accepts TCP connections on 127.0.0.1:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def check_index() -> dict[str, Any]:
    """Confirm the Qdrant collection exists, then release it (a local Qdrant folder allows one process at a time)."""
    settings = load_rag_settings()
    client = make_qdrant_client(settings)
    try:
        exists = client.collection_exists(settings.collection)
    finally:
        client.close()
    if not exists:
        raise SmokeSetupError(f"Qdrant collection {settings.collection!r} not found. Run `make index` first.")
    manifest = json.loads(INDEX_MANIFEST.read_text(encoding="utf-8")) if INDEX_MANIFEST.exists() else {}
    return {
        "qdrant_mode": settings.mode,
        "collection": settings.collection,
        "corpus_checksum_sha256": manifest.get("corpus_checksum_sha256"),
    }


def server_environment(risk_api_url: str) -> dict[str, str]:
    """The child's environment: the SDK's safe defaults (PATH, HOME, ...) plus the risk API base URL, nothing else."""
    return get_default_environment() | {"UWC_RISK_API_URL": risk_api_url}


def tool_payload(result: Any) -> dict[str, Any] | None:
    """The structured content of a tool result (None when the tool returned only text)."""
    return result.structured_content


def hit_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The fields of each search hit worth keeping in the results file (not the full text)."""
    return [
        {
            "rank": hit["rank"],
            "section_id": hit["section_id"],
            "citation": hit["citation"],
            "source_doc": hit["source_doc"],
            "score": hit["score"],
        }
        for hit in payload["hits"]
    ]


async def check_listing(client: Client, log: CheckLog, store: GuidelineStore) -> None:
    """Tools, resources and prompts are listed as expected."""
    tools = (await client.list_tools()).tools
    names = sorted(tool.name for tool in tools)
    log.record("list_tools: exactly the four expected tools", names == sorted(MCP_TOOL_BINDINGS), ", ".join(names))
    synthetic = all("SYNTHETIC" in (tool.description or "") for tool in tools)
    log.record("list_tools: every description says SYNTHETIC", synthetic)
    resources = (await client.list_resources()).resources
    expected = len(store.documents) + len(store.section_ids())
    log.record("list_resources: documents + sections", len(resources) == expected, f"{len(resources)} resources")
    prompts = [prompt.name for prompt in (await client.list_prompts()).prompts]
    log.record("list_prompts: underwriting_review", prompts == ["underwriting_review"], ", ".join(prompts))


async def timed_search(client: Client, query: str, k: int, mode: str) -> tuple[Any, float]:
    """One search call and its wall-clock seconds."""
    start = time.perf_counter()
    result = await client.call_tool("search_underwriting_guidelines", {"query": query, "k": k, "mode": mode})
    return result, time.perf_counter() - start


async def check_search(client: Client, log: CheckLog, query: str, k: int, mode: str) -> dict[str, Any]:
    """Two identical searches: k hits ranked 1..k with "[id]" citations, the same order both times."""
    first, first_seconds = await timed_search(client, query, k, mode)
    payload = tool_payload(first)
    ok = log.record("search: no error", not first.is_error and payload is not None, first.content[0].text[:300])
    report: dict[str, Any] = {"query": query, "k": k, "mode": mode, "first_call_seconds": round(first_seconds, 3)}
    if not ok:
        return report
    hits = payload["hits"]
    log.record("search: k hits", len(hits) == k, f"{len(hits)} hits")
    log.record("search: ranks 1..k", [hit["rank"] for hit in hits] == list(range(1, len(hits) + 1)))
    citations_ok = all(hit["citation"] == f"[{hit['section_id']}]" for hit in hits)
    log.record("search: every hit has its [section_id] citation", citations_ok)
    second, second_seconds = await timed_search(client, query, k, mode)
    second_payload = tool_payload(second) or {"hits": []}
    same = [hit["section_id"] for hit in second_payload["hits"]] == [hit["section_id"] for hit in hits]
    log.record("search: repeated call returns the same order", same)
    report.update({"second_call_seconds": round(second_seconds, 3), "hits": hit_summary(payload)})
    return report


async def check_section(client: Client, log: CheckLog, section_id: str) -> None:
    """get_guideline_section and the section resource show the same text."""
    result = await client.call_tool("get_guideline_section", {"section_id": section_id})
    payload = tool_payload(result)
    ok = log.record(f"get_guideline_section({section_id})", not result.is_error and payload is not None)
    if not ok:
        return
    resource = await client.read_resource(section_resource_uri(section_id))
    resource_text = resource.contents[0].text
    log.record(
        f"resource {section_resource_uri(section_id)} contains the section text", payload["text"] in resource_text
    )


async def check_prompt(client: Client, log: CheckLog) -> None:
    """The prompt renders and keeps its never-approve rule."""
    arguments = {
        "applicant_name": "Example Bakery LLC (SYNTHETIC)",
        "line_of_business": "small_business_credit",
        "application_summary": "SYNTHETIC: bakery, 6 years in business, requests a $150,000 term loan.",
    }
    prompt = await client.get_prompt("underwriting_review", arguments)
    text = prompt.messages[0].content.text if prompt.messages else ""
    log.record(
        "get_prompt underwriting_review: never-approve rule present", "never approve" in text, f"{len(text)} chars"
    )


async def check_score_error(client: Client, log: CheckLog, risk_api_url: str) -> dict[str, Any]:
    """score_application with no risk service listening: a structured error and no score."""
    arguments = {"product": "credit", "features": {"loan_amount": 150000, "years_in_business": 6}}
    result = await client.call_tool("score_application", arguments)
    payload = tool_payload(result) or {}
    error = payload.get("error") or {}
    log.record("score_application: is_error with status 'error'", result.is_error and payload.get("status") == "error")
    log.record("score_application: no score in an error result", payload.get("score") is None)
    log.record("score_application: code risk_api_unreachable", error.get("code") == "risk_api_unreachable")
    return {"risk_api_url": risk_api_url, "is_error": result.is_error, "structured_content": payload}


async def check_refusals(client: Client, log: CheckLog) -> None:
    """A tool that is not on the server and an argument a tool does not define are both refused."""
    unknown_tool = await client.call_tool("approve_application", {"application_id": "X"})
    log.record("unknown tool refused", unknown_tool.is_error, unknown_tool.content[0].text[:200])
    extra = await client.call_tool("get_guideline_section", {"section_id": "NBC-1.1", "approve": True})
    log.record("unknown argument refused", extra.is_error, extra.content[0].text[:200])


async def run_checks(client: Client, args: argparse.Namespace, risk_api_url: str) -> dict[str, Any]:
    """Every check against a connected client; returns the report (without run metadata)."""
    log = CheckLog()
    store = GuidelineStore()
    report: dict[str, Any] = {"protocol_version": client.protocol_version}
    report["server_name"] = client.server_info.name if client.server_info else None
    await check_listing(client, log, store)
    report["search"] = await check_search(client, log, args.query, args.k, args.mode)
    guideline_hits = [hit for hit in report["search"].get("hits", []) if hit["section_id"] in store.section_ids()]
    section_id = guideline_hits[0]["section_id"] if guideline_hits else store.section_ids()[0]
    await check_section(client, log, section_id)
    await check_prompt(client, log)
    report["score_application"] = await check_score_error(client, log, risk_api_url)
    await check_refusals(client, log)
    report["checks"] = log.checks
    report["all_passed"] = log.all_passed
    return report


async def smoke_stdio(args: argparse.Namespace, risk_api_url: str) -> dict[str, Any]:
    """Let the SDK client start the server over stdio and run the checks."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", SERVER_MODULE],
        env=server_environment(risk_api_url),
        cwd=str(REPO_ROOT),
    )
    async with Client(params, read_timeout_seconds=READ_TIMEOUT_S) as client:
        report = await run_checks(client, args, risk_api_url)
    report["server_command"] = f"python -m {SERVER_MODULE}"
    return report


def wait_for_port(process: subprocess.Popen, port: int, timeout_s: float) -> None:
    """Wait until the HTTP server accepts connections (or exits, or the timeout passes)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SmokeSetupError(f"The HTTP server exited early with code {process.returncode}.")
        if port_is_open(port):
            return
        time.sleep(0.25)
    raise SmokeSetupError(f"The HTTP server did not open port {port} within {timeout_s:g} s.")


async def smoke_http(args: argparse.Namespace, risk_api_url: str) -> dict[str, Any]:
    """Start the server with streamable HTTP on a free port, connect, run the checks, stop the server."""
    port = free_local_port()
    command = [sys.executable, "-m", SERVER_MODULE, "--transport", "streamable-http", "--port", str(port)]
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, env=server_environment(risk_api_url), stdout=subprocess.DEVNULL, stderr=None
    )
    try:
        await anyio.to_thread.run_sync(wait_for_port, process, port, HTTP_START_TIMEOUT_S)
        async with Client(f"http://127.0.0.1:{port}/mcp", read_timeout_seconds=READ_TIMEOUT_S) as client:
            report = await run_checks(client, args, risk_api_url)
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
    report["server_command"] = f"python -m {SERVER_MODULE} --transport streamable-http --port <free port>"
    return report


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run the smoke test for args.transport and return the full report."""
    index = check_index()
    closed_port = free_local_port()  # nothing listens here, so the risk API call must fail to connect
    risk_api_url = f"http://127.0.0.1:{closed_port}"  # a base URL: the client posts to <base>/score
    started = datetime.now(UTC)
    load_average = [round(value, 2) for value in os.getloadavg()]
    runner = smoke_stdio if args.transport == "stdio" else smoke_http
    report = anyio.run(runner, args, risk_api_url)
    return {
        "generated_at": started.isoformat(timespec="seconds"),
        "generated_by": "scripts/mcp_smoke.py",
        "transport": args.transport,
        "client": f"MCP Python SDK Client (mcp {version('mcp')})",
        "python": platform.python_version(),
        "machine": {"cpu_count": os.cpu_count(), "load_average_1_5_15_min_at_start": load_average},
        "index": index,
        "llm_calls": 0,
        "note": "One run on a shared machine; timings are not a benchmark. The risk API URL had no listener.",
        **report,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke test, write the JSON report, print a one-line summary per check."""
    args = parse_args(argv)
    output = args.output or RESULTS_DIR / f"mcp_smoke_{args.transport.replace('-', '_')}.json"
    try:
        report = run_smoke(args)
    except SmokeSetupError as error:
        print(f"Smoke test could not start: {error}", file=sys.stderr)
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for check in report["checks"]:
        print(f"{'PASS' if check['passed'] else 'FAIL'}  {check['name']}")
    print(f"Wrote {output}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
