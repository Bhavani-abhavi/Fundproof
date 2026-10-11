"""Write the GenAI-track results table in README.md from results/*.json (`make readme-results`).

Nothing in the table is typed by hand. The script reads:
- results/rag_metrics.json      retrieval quality per mode (and the "not measured" generation metrics)
- results/kyc_metrics.json      sanctions screening at the selected operating point, and the homonym stratum
- results/agent_metrics.json    routing, escalation and injection-detector metrics; LLM metrics
- results/agent_metrics_homonyms_kept.json   the same evaluation on the set without the namesake redraw
- results/mcp_smoke_*.json      MCP smoke-test checks per transport

and replaces everything between the README markers <!-- GENAI-RESULTS:START --> and <!-- GENAI-RESULTS:END -->.
A metric that a results file reports as "not measured ..." is shown as "not measured" with its reason; a results
file that does not exist gives "not measured (no results file)".

Usage:
    uv run python scripts/update_readme_results.py           # rewrite the block
    uv run python scripts/update_readme_results.py --check   # exit 1 if the README block is out of date
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from uwcopilot.common.paths import REPO_ROOT, RESULTS_DIR

README = REPO_ROOT / "README.md"
START_MARKER = "<!-- GENAI-RESULTS:START -->"
END_MARKER = "<!-- GENAI-RESULTS:END -->"
NO_FILE = "not measured (no results file)"
RAG_MODES: tuple[str, ...] = ("dense", "bm25", "hybrid", "hybrid_rerank")
MCP_TRANSPORTS: tuple[str, ...] = ("stdio", "streamable_http")
Row = tuple[str, str, str]


def load(name: str, results_dir: Path) -> dict[str, Any] | None:
    """The parsed results file, or None when it does not exist."""
    path = results_dir / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: float, decimals: int = 4) -> str:
    """A metric value with a fixed number of decimals."""
    return f"{value:.{decimals}f}"


def with_interval(value: float, low: float, high: float, decimals: int = 4) -> str:
    """'value [low, high]'."""
    return f"{number(value, decimals)} [{number(low, decimals)}, {number(high, decimals)}]"


def agent_rate(rate: dict[str, Any] | str) -> str:
    """An agent rate {value, ci_low, ci_high, n} as text, or its "not measured" reason."""
    if isinstance(rate, str):
        return not_measured(rate)
    if rate.get("value") is None:
        return "not measured (no cases)"
    return f"{with_interval(rate['value'], rate['ci_low'], rate['ci_high'])} (n={rate['n']})"


def kyc_rate(metric: dict[str, Any], decimals: int = 4) -> str:
    """A KYC metric {value, ci95_low, ci95_high} as text."""
    return with_interval(metric["value"], metric["ci95_low"], metric["ci95_high"], decimals)


def not_measured(reason: str) -> str:
    """Shorten a long "not measured ..." text to its first clause, without the parenthesised detail."""
    text = reason.split(";")[0].strip()
    if not text.startswith("not measured"):
        return f"not measured ({text})"
    if text.startswith("not measured ("):
        return text
    return text.split(" (")[0]


def rag_rows(rag: dict[str, Any] | None) -> list[Row]:
    """Retrieval recall@5 and MRR@10 per mode, then the generation metrics."""
    if rag is None:
        return [("RAG retrieval", "Recall@5", NO_FILE)]
    count = rag["question_set"]["n_answerable"]
    rows: list[Row] = []
    for mode in RAG_MODES:
        metrics = rag["modes"][mode]["metrics"]
        recall, mrr = metrics["recall@5"], metrics["mrr@10"]
        rows.append(
            (
                f"RAG retrieval, {mode} ({count} answerable SYNTHETIC questions)",
                "Recall@5 / MRR@10 [95% CI]",
                f"{with_interval(recall['mean'], *recall['ci95'])} / {with_interval(mrr['mean'], *mrr['ci95'])}",
            )
        )
    paired = rag["paired_bootstrap"]
    rows.append(
        (
            f"RAG retrieval, {paired['mode_a']} minus {paired['mode_b']}",
            f"Paired difference in {paired['metric']} [95% CI]",
            with_interval(paired["mean_difference"], *paired["ci95"]),
        )
    )
    generation = rag["generation"]
    rows.append(("RAG answers", "Faithfulness / citation accuracy", not_measured(generation["faithfulness"])))
    return rows


def kyc_rows(kyc: dict[str, Any] | None) -> list[Row]:
    """The selected operating point on the test split, and the exact-name homonym stratum."""
    if kyc is None:
        return [("Sanctions screening", "Recall at alerts per 1,000", NO_FILE)]
    selected = kyc["selected_operating_point"]
    point = kyc["operating_points"][str(selected["alert_budget_per_1000"])][selected["method"]]["test"]
    label = f"Sanctions screening, {selected['method'].replace('_', ' ')} (SYNTHETIC test customers)"
    rows: list[Row] = [
        (label, "Recall of true matches [95% CI]", kyc_rate(point["recall"])),
        (label, "Alerts per 1,000 screened [95% CI]", kyc_rate(point["alerts_per_1000"], 1)),
        (label, "False-positive rate on clean customers [95% CI]", kyc_rate(point["clean_false_positive_rate"])),
    ]
    stratum = kyc.get("exact_name_homonym_stratum")
    if isinstance(stratum, dict):
        rows.append(
            (
                f"Sanctions screening, exact-name namesakes (separate stratum, n={stratum['customers']})",
                "Alert rate at the same threshold [95% CI]",
                kyc_rate(stratum["alert_rate_at_selected_point"]),
            )
        )
    return rows


def agent_rows(agent: dict[str, Any] | None) -> list[Row]:
    """Deterministic routing/escalation metrics, the injection detector, and the LLM metrics."""
    if agent is None:
        return [("Agent", "Escalation correctness", NO_FILE)]
    path = agent["deterministic_path"]
    count = agent["applications"]["count"]
    llm = agent["llm_dependent"]
    latency = path["latency_ms_per_application"]
    return [
        (
            f"Agent ({count} SYNTHETIC applications, template memo)",
            "Required escalations that reached the right human queue [95% CI]",
            agent_rate(path["escalation"]["all_required_escalations"]),
        ),
        (
            "Agent",
            "End-to-end routing agreement with designed labels [95% CI]",
            agent_rate(path["end_to_end_routing_agreement_with_designed_labels"]),
        ),
        ("Agent", "Decisions recorded without a human", str(path["decisions_recorded_without_a_human"])),
        (
            "Agent injection detector",
            "Recall on the generator's own injection phrasings [95% CI]",
            agent_rate(agent["injection_detector"]["recall"]),
        ),
        (
            "Agent",
            "Latency p95 per application, no LLM call (ms) [95% CI]",
            with_interval(latency["p95"], *latency["p95_ci"], decimals=1),
        ),
        ("Agent", "Memo agreement / citation accuracy (needs an LLM)", not_measured(llm["citation_accuracy"])),
        ("Agent", "LLM cost and p95 latency per application", not_measured(llm["cost_usd_per_application"])),
    ]


def agent_sensitivity_rows(sensitivity: dict[str, Any] | None, final: dict[str, Any] | None) -> list[Row]:
    """Routing agreement and clean-party flag rate on the set that keeps exact-name namesakes, next to the final."""
    if sensitivity is None or final is None:
        return []
    rows: list[Row] = []
    for label, data in (("final set", final), ("namesakes kept, sensitivity run", sensitivity)):
        path = data["deterministic_path"]
        rows.append(
            (
                f"Agent screening, {label}",
                "Routing agreement / clean parties flagged [95% CI]",
                f"{agent_rate(path['end_to_end_routing_agreement_with_designed_labels'])} / "
                f"{agent_rate(path['screening']['clean_parties_flagged'])}",
            )
        )
    return rows


def mcp_rows(results_dir: Path) -> list[Row]:
    """MCP smoke-test checks passed per transport."""
    rows: list[Row] = []
    for transport in MCP_TRANSPORTS:
        smoke = load(f"mcp_smoke_{transport}.json", results_dir)
        if smoke is None:
            rows.append((f"MCP server smoke test ({transport})", "Checks passed", NO_FILE))
            continue
        passed = sum(1 for check in smoke["checks"] if check["passed"])
        rows.append((f"MCP server smoke test ({transport})", "Checks passed", f"{passed} / {len(smoke['checks'])}"))
    return rows


def render_block(results_dir: Path = RESULTS_DIR) -> str:
    """The markdown that goes between the markers."""
    rows = (
        rag_rows(load("rag_metrics.json", results_dir))
        + kyc_rows(load("kyc_metrics.json", results_dir))
        + agent_rows(load("agent_metrics.json", results_dir))
        + agent_sensitivity_rows(
            load("agent_metrics_homonyms_kept.json", results_dir), load("agent_metrics.json", results_dir)
        )
        + mcp_rows(results_dir)
    )
    lines = ["| Component | Metric | Value |", "|---|---|---|"]
    lines += [f"| {component} | {metric} | {value} |" for component, metric, value in rows]
    lines.append("")
    lines.append(
        "Written by `scripts/update_readme_results.py` from `results/rag_metrics.json`, `results/kyc_metrics.json`, "
        "`results/agent_metrics*.json` and `results/mcp_smoke_*.json`. Intervals: percentile bootstrap, 1,000 "
        "resamples, seed 42. All data is SYNTHETIC or public; no LLM call was made, so LLM-dependent metrics are "
        "not measured."
    )
    return "\n".join(lines)


def replace_block(readme_text: str, block: str) -> str:
    """README text with the block between the markers replaced (markers must exist exactly once)."""
    if readme_text.count(START_MARKER) != 1 or readme_text.count(END_MARKER) != 1:
        raise ValueError(f"README must contain {START_MARKER} and {END_MARKER} exactly once.")
    before, rest = readme_text.split(START_MARKER)
    _, after = rest.split(END_MARKER)
    return f"{before}{START_MARKER}\n{block}\n{END_MARKER}{after}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if README.md is not up to date")
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Rewrite (or check) the README block."""
    args = parse_args(argv)
    current = args.readme.read_text(encoding="utf-8")
    updated = replace_block(current, render_block(args.results_dir))
    if args.check:
        if updated != current:
            print("README.md results block is out of date: run `make readme-results`.", file=sys.stderr)
            return 1
        print("README.md results block matches results/*.json.")
        return 0
    if updated != current:
        args.readme.write_text(updated, encoding="utf-8")
        print(f"Updated the results block in {args.readme}.")
    else:
        print("README.md results block already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
