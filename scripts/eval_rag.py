"""Evaluate retrieval on eval/rag_questions.jsonl and write results/rag_metrics.json.

What it does:
1. Opens the index built by scripts/build_index.py and checks that it matches the current source files exactly
   (every chunk's text checksum). A stale index is an error: rebuild it with `make index`.
2. Loads the questions and checks that every gold reference matches at least one indexed chunk.
3. For each retrieval mode (dense, bm25, hybrid, hybrid_rerank) searches every question (top 10) and records the
   rank of each gold item and the query latency (after one untimed warm-up query per mode).
4. Reports recall@1/3/5, MRR@10 and nDCG@10 on the answerable questions with bootstrap 95% CIs (1,000 resamples
   of questions, seed 42), latency p50/p95, and a paired bootstrap of hybrid_rerank minus dense on recall@5.
5. Generation metrics (faithfulness, citation accuracy) need an LLM and are reported as "not measured (no LLM
   key)". This script never calls an LLM.

Usage:
    python scripts/eval_rag.py
    python scripts/eval_rag.py --modes dense hybrid    # a subset of modes (no paired comparison unless both)
"""

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from uwcopilot.common.files import sha256_of_file
from uwcopilot.common.paths import RESULTS_DIR
from uwcopilot.rag.chunking import ChunkingParams
from uwcopilot.rag.corpus import load_corpus
from uwcopilot.rag.embeddings import FastEmbedBM25, FastEmbedDense
from uwcopilot.rag.eval_set import QUESTIONS_FILE, lexical_overlap_stats, load_questions, validate_gold
from uwcopilot.rag.evaluation import EVAL_DEPTH, paired_comparison, run_mode, summarize_mode
from uwcopilot.rag.index import describe_qdrant, index_matches_corpus, make_qdrant_client, read_all_payloads
from uwcopilot.rag.rerank import FastEmbedReranker
from uwcopilot.rag.retriever import RETRIEVAL_MODES, RetrievalParams, Retriever
from uwcopilot.rag.settings import load_rag_settings

METRICS_FILE = RESULTS_DIR / "rag_metrics.json"
PER_QUESTION_FILE = RESULTS_DIR / "rag_per_question.jsonl"
NOT_MEASURED_NO_KEY = "not measured (no LLM key)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--output", type=Path, default=METRICS_FILE)
    parser.add_argument("--per-question", type=Path, default=PER_QUESTION_FILE)
    parser.add_argument("--modes", nargs="+", choices=RETRIEVAL_MODES, default=list(RETRIEVAL_MODES))
    parser.add_argument("--threads", type=int, default=2, help="ONNX threads for the models (default 2)")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def load_average() -> list[float] | None:
    """The machine's 1/5/15-minute load average (context for latency numbers), or None where unavailable."""
    try:
        return [round(value, 2) for value in os.getloadavg()]
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and write the results files. Returns a process exit code."""
    args = parse_args(argv)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    load_start = load_average()
    settings = load_rag_settings()
    cache = settings.model_cache_dir
    dense = FastEmbedDense(cache_dir=cache, threads=args.threads)
    sparse = FastEmbedBM25(cache_dir=cache, threads=args.threads)
    reranker = FastEmbedReranker(cache_dir=cache, threads=args.threads) if "hybrid_rerank" in args.modes else None

    client = make_qdrant_client(settings)
    try:
        payloads = read_all_payloads(client, settings.collection)
        corpus = load_corpus(count_tokens=dense.count_tokens, params=ChunkingParams())
        differences = index_matches_corpus(payloads, corpus.chunks)
        if differences:
            print("ERROR: the index does not match the current source files; run `make index`.")
            print("\n".join(differences[:20]))
            return 1

        questions = load_questions(args.questions)
        problems = validate_gold(questions, payloads)
        if problems:
            print("ERROR: gold references that match no indexed chunk:\n" + "\n".join(problems))
            return 1

        params = RetrievalParams()
        retriever = Retriever(client, settings.collection, dense, sparse, reranker, params)
        all_results = {}
        summaries = {}
        for mode in args.modes:
            mode_started = time.perf_counter()
            all_results[mode] = run_mode(retriever, questions, mode)
            summaries[mode] = summarize_mode(all_results[mode], args.resamples, args.seed)
            print(
                f"{mode:14s} recall@5={summaries[mode]['metrics']['recall@5']['mean']:.4f} "
                f"mrr@10={summaries[mode]['metrics']['mrr@10']['mean']:.4f} "
                f"p50={summaries[mode]['latency_ms']['p50']} ms ({time.perf_counter() - mode_started:.1f} s)"
            )
        qdrant = describe_qdrant(client, settings)
    finally:
        client.close()

    comparison = None
    if "hybrid_rerank" in all_results and "dense" in all_results:
        comparison = paired_comparison(
            all_results["hybrid_rerank"], all_results["dense"], "recall@5", args.resamples, args.seed
        )

    answerable = [question for question in questions if question.answerable]
    categories: dict[str, int] = {}
    for question in questions:
        categories[question.category] = categories.get(question.category, 0) + 1
    report = {
        "generated_at": started_at,
        "generated_by": "scripts/eval_rag.py",
        "status": "measured",
        "corpus": {
            "checksum_sha256": corpus.checksum(),
            "source_files": [asdict(source) for source in corpus.files],
            "skipped_regulatory_files": corpus.skipped_regulatory,
            "chunks_by_doc": corpus.chunk_counts(),
            "points_in_index": len(payloads),
            "synthetic_documents": sorted({chunk.doc_id for chunk in corpus.chunks if chunk.synthetic}),
        },
        "index": {"qdrant": qdrant, "chunking": asdict(ChunkingParams())},
        "models": {
            "dense": {"id": dense.model_id, "dim": dense.dim, "query_prefix": dense.query_prefix},
            "sparse": {"id": sparse.model_id, "qdrant_modifier": "idf"},
            "reranker": {"id": reranker.model_id} if reranker else None,
            "fastembed_version": version("fastembed"),
        },
        "retrieval_params": {**asdict(params), "eval_depth": EVAL_DEPTH},
        "question_set": {
            "path": "eval/rag_questions.jsonl",
            "sha256": sha256_of_file(args.questions),
            "n_total": len(questions),
            "n_answerable": len(answerable),
            "by_category": categories,
            "lexical_overlap": lexical_overlap_stats(questions, payloads),
        },
        "bootstrap": {
            "resamples": args.resamples,
            "seed": args.seed,
            "confidence": 0.95,
            "unit": "question",
            "method": "percentile",
        },
        "modes": summaries,
        "paired_bootstrap": comparison,
        "generation": {
            "faithfulness": NOT_MEASURED_NO_KEY,
            "citation_accuracy": NOT_MEASURED_NO_KEY,
            "answer_abstention_on_unanswerable": NOT_MEASURED_NO_KEY,
        },
        "environment": {
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "onnx_threads": args.threads,
            "load_average_start": load_start,
            "load_average_end": load_average(),
            "latency_note": "wall-clock per query on a shared machine; other jobs were running",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.per_question.open("w", encoding="utf-8") as handle:
        for mode_results in all_results.values():
            for result in mode_results:
                handle.write(json.dumps(asdict(result)) + "\n")
    print(f"Wrote {args.output} and {args.per_question}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
