"""Evidence-request simulator (innovation S5): VOI vs checklist vs random vs oracle on SYNTHETIC JSON-only cases.

Usage (the `make simulate-requests` target):
    uv run python scripts/simulate_requests.py
    uv run python scripts/simulate_requests.py --split dev --no-write    # development runs (dev split only)

What it does:
1. builds 300 JSON-only cases per line (credit, insurance) with the Phase 2 generator: build_packet(draw=False),
   the code path of `scripts/make_packets.py --json-only`, master seed 4242, packet ids PKT-C-001..300 and
   PKT-I-001..300. The test split (200 per line) is the generator's own seeded split (generate.test_split); the
   other 100 per line are the development split. Cases are built in memory; nothing is written under data/packets.
   The seed differs from the Phase 2 evaluation packets (seed 42) and the engine's test packets (seed 4040), so the
   same packet id names a different case here.
2. runs every policy of evidence/simulator.py (voi, voi_lookahead, checklist, random, oracle) on every case, under
   every priors variant (documented, latent_x0.5, latent_x1.5, latent_removed; the oracle under documented only);
3. writes the "simulator" section (see "Results files" below) and one JSON line per run to
   data/processed/evidence/simulator_runs.jsonl (gitignored).

HELD-OUT PROBLEM FAMILY: claims_history_mismatch (insurance). It was chosen and written down on 2026-10-14, BEFORE
any simulation was run, following the example in the Phase 4 spec. Every case whose truth contains it is held out
of all development: `--split dev` never simulates it, and the final run reports the test-split cases that contain it
separately ("held_out_family"), never mixed into the main test metrics. Development-split cases that contain it are
never simulated at all. Disclosure: the engine's detection of this problem (evidence/open_items_insurance.py
claims_history_items) was written and unit-tested by the engine build, on other generated packets (seed 4040),
before the simulator existed; the hold-out covers the simulator's development and every simulation result.

Development used the development split only (`--split dev --no-write`). The main results are the test split without
the held-out family. Intervals: percentile bootstrap over cases, 1,000 resamples, seed 42; the headline is the
paired VOI-minus-checklist difference on the same cases.

Results files (two copies of one payload, so the packet evaluation can never overwrite this section):
- results/evidence_metrics_simulator.json: written only by this script, with section "simulator";
- results/evidence_metrics.json: the same section merged in with uwcopilot.risk.results.update_results, which holds
  an exclusive lock on results/evidence_metrics.json.lock while it reads the file, replaces ONLY section
  "simulator" and writes it back (atomic rename). scripts/eval_evidence.py writes section "packets"; if it rewrites
  the whole file instead of using update_results, it must copy the "simulator" section of
  evidence_metrics_simulator.json. The two copies hold the same results; each metadata block is stamped when its
  file is written.
Writing to the default results files needs the default design (seed, case counts, both lines, all cases); other
designs must pass --no-write or other --results / --standalone paths.

Everything here is SYNTHETIC. Extraction is an oracle (the generator's perfect extraction), so no number here says
anything about extraction accuracy. Priors and cost weights are planning assumptions written by the project author.
Not claimable: real time or cost savings, or that the VOI score is optimal or an economic model.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from uwcopilot.common.paths import DATA_DIR, RESULTS_DIR
from uwcopilot.docintel.synthetic import build_packet
from uwcopilot.docintel.synthetic.common import packet_seed
from uwcopilot.docintel.synthetic.generate import packet_ids, test_split
from uwcopilot.docintel.truth import GENERATOR_VERSION, PacketTruth, truth_sha256
from uwcopilot.evidence.requests import catalog_version, templates_version
from uwcopilot.evidence.simulator import run_context, simulate
from uwcopilot.evidence.simulator_cases import prepare_case
from uwcopilot.evidence.simulator_models import (
    MAX_REQUESTS,
    ORACLE_MAX_OPEN_REQUESTS,
    ORACLE_MAX_POSITIONS,
    POLICIES,
    PRIORS_VARIANTS,
    SimulationResult,
)
from uwcopilot.evidence.simulator_summary import summarize
from uwcopilot.evidence.voi import SCORE_NAME
from uwcopilot.governance.provenance import RunProvenance
from uwcopilot.risk.results import update_results

SEED = 4242
CASES_PER_LINE = 300
TEST_PER_LINE = 200
LINES: tuple[str, ...] = ("credit", "insurance")
HELD_OUT_FAMILY = "claims_history_mismatch"
HELD_OUT_CHOSEN = "2026-10-14, before any simulation was run (the Phase 4 spec's example)"
RESULTS_FILE = RESULTS_DIR / "evidence_metrics.json"
STANDALONE_FILE = RESULTS_DIR / "evidence_metrics_simulator.json"
SECTION = "simulator"
RUNS_FILE = DATA_DIR / "processed" / "evidence" / "simulator_runs.jsonl"
HEADLINE_METRICS: tuple[str, ...] = (
    "cost_weighted_requests",
    "n_requests",
    "n_zero_value_requests",
    "decision_error",
    "requests_to_stable",
)
POLICY_DESCRIPTIONS: dict[str, str] = {
    "voi": "highest nonzero VOI first; stop when no open request has a VOI above zero (innovation S5 rule)",
    "voi_lookahead": "voi, then the zero-VOI request with the highest nonzero lookahead value (an addition)",
    "checklist": "every open request in the fixed severity order of evidence.open_items.PROBLEM_ORDER (the catalog "
    "has no severity field)",
    "random": "every open request in a seeded shuffled order",
    "oracle": "knows the truth: cheapest cost-weighted request sequence that reaches the full-information "
    f"recommendation (exhaustive uniform-cost search over request orders; only cases with at most "
    f"{ORACLE_MAX_OPEN_REQUESTS} open requests at the start)",
}
VARIANT_DESCRIPTIONS: dict[str, str] = {
    "documented": "template priors of config/resolution_templates.yaml",
    "latent_x0.5": "prior of the latent (true) option x0.5, renormalized",
    "latent_x1.5": "prior of the latent (true) option x1.5, renormalized",
    "latent_removed": "latent (true) option removed from the option set, renormalized",
}
READING_NOTES: tuple[str, ...] = (
    "The voi policy only makes requests whose VOI is above zero, so its zero-value request count is 0 by "
    "construction; the comparison is how many zero-value requests the other policies make.",
    "Scaling the latent option's prior by 0.5 or 1.5 keeps every prior above zero, so it cannot change WHICH "
    "requests have a VOI above zero, only their values and order; for the voi policy, differences between the "
    "documented, x0.5 and x1.5 rows come only from the order of several nonzero requests. Removing the latent "
    "option can change which requests have value. voi_lookahead also depends on which option is most likely.",
    "Checklist, random and oracle make the same requests under every priors variant; only the VOI they are valued "
    "at, and so the zero-value count, can change.",
    "The full-information recommendation reveals every planted problem. A planted problem that no open item "
    "stands for cannot be requested by any policy; decision_error_rate_when_every_planted_problem_detected leaves "
    "those cases out.",
    "requests_to_stable_when_reached averages only the cases where the run ends on the full-information "
    "recommendation, so its case set differs between policies; the paired comparison uses the cases both reached.",
    "The oracle stops at the first state whose recommendation equals the full-information recommendation. That "
    "state need not be the full-information state: when a planted problem is undetected, another open problem can "
    "give the same recommendation, so 'oracle reached' does not mean every problem was revealed.",
    "The full-information recommendation is what the SYNTHETIC reference policy (agent/policy.py) gives once every "
    "planted problem is revealed; it is the benchmark here, not a claim about the right lending or underwriting "
    "decision.",
)
NOT_CLAIMABLE: tuple[str, ...] = (
    "real time or cost savings (cost weights and priors are planning assumptions)",
    "that the VOI score is optimal or an economic model",
    "extraction accuracy (the simulator uses the generator's oracle extraction)",
    "anything about real applications: every case is SYNTHETIC",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description="Simulate evidence-request policies on SYNTHETIC cases (S5).")
    parser.add_argument("--seed", type=int, default=SEED, help="master seed of the case generator")
    parser.add_argument("--cases-per-line", type=int, default=CASES_PER_LINE, help="cases per line")
    parser.add_argument("--test-per-line", type=int, default=TEST_PER_LINE, help="test-split cases per line")
    parser.add_argument("--lines", nargs="+", choices=LINES, default=list(LINES), help="lines to simulate")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all", help="which split to simulate")
    parser.add_argument("--limit", type=int, default=None, help="at most this many cases per line and split")
    parser.add_argument("--results", type=Path, default=RESULTS_FILE, help="shared results JSON (section merged)")
    parser.add_argument("--standalone", type=Path, default=STANDALONE_FILE, help="this script's own results JSON")
    parser.add_argument("--runs-out", type=Path, default=RUNS_FILE, help="per-run JSONL (gitignored)")
    parser.add_argument("--no-write", action="store_true", help="print the summary; write no file")
    return parser.parse_args(argv)


def write_problem(args: argparse.Namespace) -> str | None:
    """Why this run must not write its results (None when it may)."""
    if args.split != "all" or args.limit is not None:
        return "Partial runs (--split dev/test or --limit) are not written; pass --no-write."
    default_paths = args.results == RESULTS_FILE or args.standalone == STANDALONE_FILE
    default_design = (
        args.seed == SEED
        and args.cases_per_line == CASES_PER_LINE
        and args.test_per_line == TEST_PER_LINE
        and tuple(args.lines) == LINES
    )
    if default_paths and not default_design:
        return "Only the default design writes the default results files; pass --no-write or other paths."
    return None


# --------------------------------------------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------------------------------------------


def has_held_out_family(truth: PacketTruth) -> bool:
    """True when a planted problem belongs to the held-out family."""
    return any(problem.problem_type == HELD_OUT_FAMILY for problem in truth.problems)


def build_cases(line: str, args: argparse.Namespace) -> list[PacketTruth]:
    """The line's JSON-only cases of the requested split (development cases with the held-out family skipped)."""
    ids = packet_ids(line, args.cases_per_line)
    test_ids = test_split(ids, args.test_per_line, args.seed, line)
    counts = {"dev": 0, "test": 0}
    truths = []
    for packet_id in ids:
        split = "test" if packet_id in test_ids else "dev"
        if args.split != "all" and split != args.split:
            continue
        if args.limit is not None and counts[split] >= args.limit:
            continue
        _, truth = build_packet(packet_id, line, packet_seed(args.seed, packet_id), split, draw=False)
        if split == "dev" and has_held_out_family(truth):
            continue
        counts[split] += 1
        truths.append(truth)
    return truths


def case_rng(seed: int, case_id: str) -> random.Random:
    """The random policy's generator for one case (the same order under every priors variant)."""
    digest = hashlib.sha256(f"{seed}|{case_id}|random-policy".encode()).hexdigest()
    return random.Random(int(digest[:8], 16))


def run_case(truth: PacketTruth, seed: int) -> list[SimulationResult]:
    """Every policy under every priors variant on one case (the oracle under the documented priors only)."""
    case = prepare_case(truth)
    try:
        context = run_context(case)
    except ValueError as error:
        reason = f"{type(error).__name__}: the case could not be evaluated"
        print(f"{truth.packet_id}: not computed ({reason})", file=sys.stderr)
        context = None
    results = []
    for variant in PRIORS_VARIANTS:
        for policy in POLICIES:
            if policy == "oracle" and variant != "documented":
                continue
            rng = case_rng(seed, truth.packet_id) if policy == "random" else None
            results.append(simulate(truth, policy, variant, rng, prepared=case, context=context))
    return results


# --------------------------------------------------------------------------------------------------------------
# Payload
# --------------------------------------------------------------------------------------------------------------


def case_set(truths: list[PacketTruth]) -> dict[str, Any]:
    """Counts and SHA-256 (over the sorted truth hashes) of the simulated cases per line and split."""
    entries: dict[str, Any] = {}
    for line in LINES:
        for split in ("dev", "test"):
            chosen = [truth for truth in truths if truth.line == line and truth.split == split]
            if not chosen:
                continue
            hashes = sorted(truth_sha256(truth) for truth in chosen)
            entries[f"{line}_{split}"] = {
                "n_cases": len(chosen),
                "n_with_held_out_family": sum(1 for truth in chosen if has_held_out_family(truth)),
                "truth_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
            }
    return entries


def group_summaries(results: list[SimulationResult]) -> dict[str, Any]:
    """Summaries per line and for all lines together."""
    summaries: dict[str, Any] = {}
    for line in (*LINES, "all"):
        chosen = [result for result in results if line == "all" or result.line == line]
        if chosen:
            summaries[line] = summarize(chosen, POLICIES, PRIORS_VARIANTS)
    return summaries


def design(args: argparse.Namespace) -> dict[str, Any]:
    """The fixed design of the run (written before the results)."""
    return {
        "seed": args.seed,
        "cases_per_line": args.cases_per_line,
        "test_cases_per_line": args.test_per_line,
        "development_cases_per_line": args.cases_per_line - args.test_per_line,
        "lines": list(args.lines),
        "generator": f"docintel.synthetic.build_packet(draw=False) = make_packets.py --json-only; {GENERATOR_VERSION}",
        "extraction": "oracle: truth.oracle_extraction through the real checks and contradiction builder, with the "
        "oracle page text scanned for embedded instructions (no OCR, no PDF)",
        "held_out_family": {
            "problem_type": HELD_OUT_FAMILY,
            "chosen": HELD_OUT_CHOSEN,
            "rule": "cases containing it were never simulated during development (development-split cases with it "
            "are never simulated); test-split cases with it are reported only under held_out_family",
            "disclosure": "the engine's detection of this problem (open_items_insurance.claims_history_items) was "
            "written and unit-tested by the engine build on other generated packets (seed 4040) before the "
            "simulator existed; the hold-out covers simulator development and every simulation result",
        },
        "score_name": SCORE_NAME,
        "policies": POLICY_DESCRIPTIONS,
        "headline_policy": "voi (the specified S5 stopping rule); voi_lookahead is reported as an addition",
        "priors_variants": VARIANT_DESCRIPTIONS,
        "max_requests_per_run": MAX_REQUESTS,
        "oracle_search_guard_positions": ORACLE_MAX_POSITIONS,
        "metrics": {
            "requests_per_case": "number of requests the policy made",
            "cost_weighted_requests_per_case": "sum of the catalog cost weights of those requests",
            "zero_value_requests_per_case": "requests whose VOI was 0 just before (under the variant's priors) and "
            "whose reveal did not change the recommendation",
            "decision_error_rate": "share of cases whose final recommendation differs from the full-information "
            "recommendation (every planted problem revealed)",
            "requests_to_stable_when_reached": "requests after which the recommendation equals the "
            "full-information recommendation and stays; cases ending on another recommendation are left out",
        },
        "reading_notes": list(READING_NOTES),
        "bootstrap": {"n_resamples": 1000, "seed": 42, "method": "percentile over cases"},
        "evidence_request_catalog": catalog_version(),
        "resolution_templates": templates_version(),
    }


def headline(test_summaries: dict[str, Any]) -> dict[str, Any]:
    """The paired VOI-minus-checklist differences on the test split, all lines, documented priors (copied)."""
    summary = test_summaries.get("all")
    if summary is None:
        return {"status": "not measured", "reason": "no test-split cases were simulated"}
    paired = summary["paired_voi_minus_checklist"]["documented"]
    return {
        "group": "test split without the held-out family, all lines, documented priors",
        "comparison": "voi minus checklist, per case, paired bootstrap (1,000 resamples, seed 42)",
        "n_cases": paired["n_cases"],
        "voi_minus_checklist": {metric: paired[metric] for metric in HEADLINE_METRICS},
    }


def build_payload(
    results: list[SimulationResult], truths: list[PacketTruth], args: argparse.Namespace, run: RunProvenance
) -> dict[str, Any]:
    """The "simulator" section."""
    started = time.perf_counter()
    held_out = {truth.packet_id for truth in truths if has_held_out_family(truth)}
    development = [result for result in results if result.split == "dev"]
    test_main = [result for result in results if result.split == "test" and result.case_id not in held_out]
    test_held_out = [result for result in results if result.split == "test" and result.case_id in held_out]
    cases = case_set(truths)
    test = group_summaries(test_main)
    payload = {
        "data_label": "SYNTHETIC",
        "description": "Innovation S5: evidence-request policies on SYNTHETIC JSON-only cases with a hidden latent "
        "truth. Replaces the circular 'next-best-evidence top-1 vs resolving document' headline.",
        "design": design(args),
        "case_set": cases,
        "headline": headline(test),
        "test": test,
        "held_out_family": group_summaries(test_held_out),
        "development": group_summaries(development),
        "not_claimable": list(NOT_CLAIMABLE),
        "provenance": run.finish(data_snapshots={"simulator_case_set": cases}),
    }
    payload["summary_seconds"] = round(time.perf_counter() - started, 1)
    return payload


def headline_lines(payload: dict[str, Any]) -> list[str]:
    """A few lines copied from the payload for the console."""
    lines = []
    for group in ("test", "held_out_family", "development"):
        summary = payload[group].get("all")
        if summary is None:
            continue
        cases = summary["cases"]["n_computed"]
        lines.append(f"{group}: {cases} cases (all lines), documented priors")
        metrics = summary["by_priors_variant"]["documented"]
        for policy in ("voi", "voi_lookahead", "checklist", "random"):
            row = metrics[policy]
            cost = row["cost_weighted_requests_per_case"]
            error = row["decision_error_rate"]
            lines.append(
                f"  {policy:14s} requests {row['requests_per_case']['value']}  cost {cost['value']} "
                f"[{cost['ci_low']}, {cost['ci_high']}]  decision error {error['value']} "
                f"[{error['ci_low']}, {error['ci_high']}]"
            )
        paired = summary["paired_voi_minus_checklist"]["documented"]
        for metric in ("cost_weighted_requests", "decision_error", "n_zero_value_requests"):
            row = paired[metric]
            lines.append(
                f"  voi - checklist {metric}: {row['mean_difference']} [{row['ci_low']}, {row['ci_high']}] "
                f"(n={row['n']})"
            )
    return lines


def write_runs(path: Path, results: list[SimulationResult]) -> None:
    """One JSON line per run (gitignored audit trail of every request made)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.model_dump(mode="json"), sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Build the cases, run every policy, summarize and write the results."""
    args = parse_args(argv)
    if args.test_per_line > args.cases_per_line:
        print("The test split cannot be larger than the number of cases.", file=sys.stderr)
        return 2
    problem = None if args.no_write else write_problem(args)
    if problem is not None:
        print(problem, file=sys.stderr)
        return 2
    run = RunProvenance.start()
    started = time.perf_counter()
    truths: list[PacketTruth] = []
    results: list[SimulationResult] = []
    for line in args.lines:
        for truth in build_cases(line, args):
            truths.append(truth)
            results.extend(run_case(truth, args.seed))
    simulation_seconds = round(time.perf_counter() - started, 1)
    payload = build_payload(results, truths, args, run)
    payload["simulation_seconds"] = simulation_seconds
    for text in headline_lines(payload):
        print(text)
    if args.no_write:
        return 0
    write_runs(args.runs_out, results)
    update_results(args.standalone, SECTION, payload)
    update_results(args.results, SECTION, payload)
    print(
        f"Wrote section '{SECTION}' of {args.standalone.name} and {args.results.name}, "
        f"and {len(results)} runs to {args.runs_out.name}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
