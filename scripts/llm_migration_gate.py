"""LLM migration gate: champion vs challenger non-inferiority on the frozen generation eval (docs/innovations.md, S1).

Usage:
    uv run python scripts/llm_migration_gate.py --champion qwen3b_v1_k3 --challenger qwen15b_v1_k3

Needs both configurations evaluated in full by scripts/eval_generation.py (status "measured" in
results/generation_metrics.json, per-question rows in results/generation/<config>.jsonl).

Steps:
1. Refuse unless both runs used identical question ids, frozen-hits hash, key-facts hash, question-file hashes and
   corpus checksum, scored the same questions, and their per-question files still match the sha256 recorded when
   they were written.
2. Read the margins from the `migration:` section of config/release_gate.yaml. The margins must have been committed
   BEFORE the challenger run: the gate compares the last commit that touched the file with the challenger's
   generated_at_utc and refuses when the file has uncommitted changes or a newer commit (--allow-unfrozen-margins
   overrides this for development; the certificate then says "NOT VALID FOR RELEASE").
3. Paired bootstrap of (challenger - champion) per gated metric and the non-inferiority decision
   (uwcopilot.governance.migration_gate). Overall PASS only if every gated metric passes.
4. Write results/llm_migration_<A>_vs_<B>.json and render docs/governance/migration_certificates/<A>_vs_<B>.md
   from that JSON only.

No model is called. Exit codes: 0 overall PASS, 1 FAIL or INCONCLUSIVE, 2 refused (inputs differ or missing),
3 refused (margins not frozen before the challenger run).
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from uwcopilot.common.files import sha256_of_file
from uwcopilot.common.paths import CONFIG_DIR, DOCS_DIR, REPO_ROOT, RESULTS_DIR
from uwcopilot.governance.migration_gate import (
    InputMismatch,
    MigrationSettings,
    check_same_inputs,
    compare_runs,
    load_migration_settings,
    render_certificate,
)
from uwcopilot.governance.provenance import display_path, provenance_block
from uwcopilot.rag.generation_eval import QuestionScore, metric_values

METRICS_FILE = RESULTS_DIR / "generation_metrics.json"
RELEASE_GATE_FILE = CONFIG_DIR / "release_gate.yaml"
CERTIFICATE_DIR = DOCS_DIR / "governance" / "migration_certificates"


class MarginsNotFrozen(RuntimeError):
    """The margins file was not committed before the challenger run."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--champion", required=True, help="champion config id")
    parser.add_argument("--challenger", required=True, help="challenger config id")
    parser.add_argument("--metrics", type=Path, default=METRICS_FILE, help="results/generation_metrics.json")
    parser.add_argument("--release-gate", type=Path, default=RELEASE_GATE_FILE, help="margins file")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="where the gate JSON goes")
    parser.add_argument("--certificate-dir", type=Path, default=CERTIFICATE_DIR, help="where the certificate goes")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="repository (for paths and git)")
    parser.add_argument(
        "--allow-unfrozen-margins",
        action="store_true",
        help="development only: run even if the margins were not committed before the challenger run",
    )
    return parser.parse_args(argv)


def git_output(arguments: list[str], repo_root: Path) -> str | None:
    """stdout of a git command, or None when git fails."""
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=repo_root, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def margins_status(path: Path, challenger_generated_at: str | None, repo_root: Path) -> dict[str, Any]:
    """Which commit the margins come from, and whether that commit predates the challenger run."""
    shown = display_path(path, repo_root)
    last_commit = git_output(["log", "-1", "--format=%H%x09%cI", "--", shown], repo_root) or ""
    pending = git_output(["status", "--porcelain", "--", shown], repo_root)
    committed = bool(last_commit)
    dirty = pending is None or bool(pending)
    commit_sha: str | None = None
    committed_at: str | None = None
    if committed:
        parts = last_commit.split("\t")
        commit_sha = parts[0]
        committed_at = parts[1] if len(parts) > 1 else None
    frozen = False
    check = "the margins file has never been committed"
    if committed and dirty:
        check = "the margins file has uncommitted changes"
    elif committed and committed_at and challenger_generated_at:
        frozen = datetime.fromisoformat(committed_at) <= datetime.fromisoformat(challenger_generated_at)
        check = (
            f"last commit {committed_at} is before the challenger run {challenger_generated_at}"
            if frozen
            else f"last commit {committed_at} is AFTER the challenger run {challenger_generated_at}"
        )
    elif committed:
        check = "the challenger run has no generated_at_utc"
    source = f"{shown}@{commit_sha}" if committed and not dirty else f"{shown}@uncommitted"
    return {
        "margins_source": source,
        "margins_file_sha256": sha256_of_file(path),
        "margins_committed_at": committed_at,
        "margins_frozen_before_challenger": frozen,
        "margins_check": check,
    }


def gate_report_path(args: argparse.Namespace) -> Path:
    """results/llm_migration_<champion>_vs_<challenger>.json (in --output-dir)."""
    return args.output_dir / f"llm_migration_{args.champion}_vs_{args.challenger}.json"


def load_config_block(document: dict[str, Any], config_id: str) -> dict[str, Any]:
    """The measured configs.<id> block, or InputMismatch."""
    block = (document.get("configs") or {}).get(config_id)
    if block is None:
        raise InputMismatch(
            f"{config_id}: not in the metrics file; run scripts/eval_generation.py --config {config_id}"
        )
    if block.get("status") != "measured":
        raise InputMismatch(f"{config_id}: status is {block.get('status')!r}, not 'measured'")
    return block


def load_scores(block: dict[str, Any], repo_root: Path) -> list[QuestionScore]:
    """The per-question scores of a run, after checking the rows file against its recorded sha256."""
    path = repo_root / block["rows_path"]
    if not path.exists():
        raise InputMismatch(f"{block['rows_path']} is missing")
    if sha256_of_file(path) != block.get("rows_sha256"):
        raise InputMismatch(f"{block['rows_path']} changed since the evaluation wrote it (sha256 mismatch)")
    scores: list[QuestionScore] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("score"):
                scores.append(QuestionScore.from_dict(row["score"]))
    return scores


def check_runs_comparable(champion: dict[str, Any], challenger: dict[str, Any]) -> list[str]:
    """Every reason the two runs cannot be compared ([] when they can)."""
    problems = check_same_inputs(champion.get("frozen_inputs", {}), challenger.get("frozen_inputs", {}))
    if champion.get("scored_question_ids") != challenger.get("scored_question_ids"):
        problems.append("scored_question_ids: the runs scored different questions (errors in one run?)")
    return problems


def values_by_metric(scores: list[QuestionScore], settings: MigrationSettings) -> dict[str, dict[str, float]]:
    """{gated metric: {question id: value}} for one run."""
    return {metric.name: metric_values(scores, metric.name) for metric in settings.metrics}


def config_summary(config_id: str, block: dict[str, Any]) -> dict[str, Any]:
    """What the certificate says about one configuration."""
    return {
        "config": config_id,
        "provider": block.get("provider"),
        "model": block.get("model"),
        "ollama_digest": block.get("ollama_digest"),
        "ollama_version": block.get("ollama_version"),
        "model_license": block.get("model_license"),
        "model_license_sources": block.get("model_license_sources"),
        "prompt_id": block.get("prompt_id"),
        "prompt_sha256": block.get("prompt_sha256"),
        "k": block.get("k"),
        "options": block.get("options"),
        "n_questions": block.get("n_questions"),
        "generated_at_utc": block.get("generated_at_utc"),
        "git_sha": block.get("git_sha"),
        "load_average": block.get("load_average"),
        "rows_path": block.get("rows_path"),
        "rows_sha256": block.get("rows_sha256"),
    }


def not_gated_summary(block: dict[str, Any]) -> dict[str, Any]:
    """Latency, tokens and error counts (reported, never gated)."""
    latency = block.get("latency_s")
    tokens = block.get("tokens")
    errors = block.get("errors") or {}
    return {
        "latency_p50_s": latency.get("p50") if isinstance(latency, dict) else latency,
        "latency_p95_s": latency.get("p95") if isinstance(latency, dict) else None,
        "n_live_calls": latency.get("n_live_calls") if isinstance(latency, dict) else 0,
        "tokens_in_mean": tokens.get("in_mean") if isinstance(tokens, dict) else None,
        "tokens_out_mean": tokens.get("out_mean") if isinstance(tokens, dict) else None,
        "errors": {name: (value.get("n") if isinstance(value, dict) else value) for name, value in errors.items()},
    }


def build_report(
    args: argparse.Namespace, document: dict[str, Any], settings: MigrationSettings, margins: dict[str, Any]
) -> dict[str, Any]:
    """Compare the two runs and assemble the gate's JSON report."""
    champion = load_config_block(document, args.champion)
    challenger = load_config_block(document, args.challenger)
    problems = check_runs_comparable(champion, challenger)
    if problems:
        raise InputMismatch("; ".join(problems))
    champion_scores = load_scores(champion, args.repo_root)
    challenger_scores = load_scores(challenger, args.repo_root)
    metrics, overall = compare_runs(
        values_by_metric(champion_scores, settings), values_by_metric(challenger_scores, settings), settings
    )
    frozen = dict(champion["frozen_inputs"])
    frozen.pop("question_ids", None)
    reproducibility = document.get("reproducibility")
    if not (isinstance(reproducibility, dict) and reproducibility.get("config") == args.champion):
        reproducibility = "not measured for the champion (scripts/eval_generation.py --reproducibility 20)"
    report_path = gate_report_path(args)
    models = list(champion.get("provenance", {}).get("models", [])) + list(
        challenger.get("provenance", {}).get("models", [])
    )
    return {
        "champion": config_summary(args.champion, champion),
        "challenger": config_summary(args.challenger, challenger),
        "n_paired": len(champion["scored_question_ids"]),
        **margins,
        "bootstrap": {
            "resamples": settings.resamples,
            "seed": settings.seed,
            "confidence": settings.confidence,
            "unit": "question",
            "method": "paired percentile bootstrap of challenger minus champion",
        },
        "metrics": metrics,
        "overall": overall,
        "not_gated": {"champion": not_gated_summary(champion), "challenger": not_gated_summary(challenger)},
        "reproducibility": reproducibility,
        "frozen_inputs": frozen,
        "report_path": display_path(report_path, args.repo_root),
        "provenance": provenance_block(
            prompts={
                champion["prompt_id"]: champion["prompt_sha256"],
                challenger["prompt_id"]: challenger["prompt_sha256"],
            },
            eval_sets=frozen["question_sets"],
            models=models,
            corpus_checksum=frozen["corpus_checksum"],
            data_snapshots={
                "frozen_hits_sha256": frozen["frozen_hits_sha256"],
                "key_facts_sha256": frozen["key_facts_sha256"],
                "margins_source": margins["margins_source"],
            },
            generated_by="scripts/llm_migration_gate.py",
            repo_root=args.repo_root,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the gate. Returns a process exit code (see the module docstring)."""
    args = parse_args(argv)
    document = json.loads(args.metrics.read_text(encoding="utf-8")) if args.metrics.exists() else {}
    settings = load_migration_settings(args.release_gate)
    challenger_block = (document.get("configs") or {}).get(args.challenger) or {}
    margins = margins_status(args.release_gate, challenger_block.get("generated_at_utc"), args.repo_root)
    if not margins["margins_frozen_before_challenger"] and not args.allow_unfrozen_margins:
        print(f"REFUSED: margins not frozen before the challenger run ({margins['margins_check']}).", file=sys.stderr)
        return 3
    try:
        report = build_report(args, document, settings, margins)
    except InputMismatch as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    report_path = gate_report_path(args)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    certificate_path = args.certificate_dir / f"{args.champion}_vs_{args.challenger}.md"
    certificate_path.parent.mkdir(parents=True, exist_ok=True)
    certificate_path.write_text(render_certificate(json.loads(report_path.read_text(encoding="utf-8"))), "utf-8")
    for name, block in report["metrics"].items():
        diff = block.get("paired_diff") or {}
        print(
            f"{name:36s} n={block['n_paired']:3d} diff={diff.get('mean')} ci95={diff.get('ci95')} "
            f"margin={block['margin']} -> {block['decision']}"
        )
    print(f"Overall: {report['overall']}")
    print(f"Wrote {report['report_path']} and {display_path(certificate_path, args.repo_root)}")
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
