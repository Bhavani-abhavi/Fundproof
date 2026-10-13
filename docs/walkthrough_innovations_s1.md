# Walkthrough: innovation S1b (frozen generation eval and LLM migration gate)

Written so that both owners (Rahul Vajja and Bhavani Adula) can explain this part of the project. It covers the
files added for S1b, the design choices, and likely interview questions. It contains no measured generation numbers:
those appear only in `results/generation_metrics.json` and `results/llm_migration_*.json` after the full runs. All
guidelines and questions are SYNTHETIC; the regulatory documents are public (SR 26-2, NY DFS Circular Letter 2024-7).

## What problem this solves

A RAG system has two parts that can change: retrieval and generation. If you swap the model and re-run everything,
any change in quality could come from either part. S1b freezes retrieval, so a model or prompt change is measured
on exactly the same documents. Then a "migration gate" decides, with a pre-registered statistical rule, whether a
challenger configuration may replace the champion.

## Files

| File | What it does |
|---|---|
| `eval/rag_key_facts.jsonl` | Frozen test set: for 57 answerable guideline questions, the short facts a correct answer must state (a number, a role such as "Senior Underwriter", or a named term). Written before any model ran. |
| `eval/rag_unanswerable_extra.jsonl` | Frozen test set: 20 new unanswerable questions (10 near-misses about products or lines the guidelines do not cover, 10 plausible but out of scope). With the original 10, 30 unanswerables. |
| `tests/rag/test_key_facts.py` | Proves every key fact appears in its gold section's text (numbers through the SIG number normalizer). |
| `config/release_gate.yaml` | The `migration:` margins table, committed before any challenger run. |
| `src/uwcopilot/rag/prompts.py` | Prompt registry: `rag_answer_v1` points at the existing prompt constants; `prompt_sha256(id)` hashes them. |
| `src/uwcopilot/rag/bm25.py` | Pure-Python BM25 over guideline sections; only the offline fallback retriever. |
| `src/uwcopilot/rag/frozen_hits.py` | The frozen-hits file format (one JSON line per question) and its reader. |
| `scripts/freeze_retrieval.py` | Runs hybrid_rerank retrieval once per question and writes `eval/generation/frozen_hits.jsonl` plus a manifest. |
| `src/uwcopilot/rag/generation_eval.py` | Pure scoring functions: key-fact recall, abstention, asserted absence, citations, format, untraceable numbers. |
| `scripts/eval_generation.py` | Runs one named config on the frozen hits through Ollama, writes per-question rows and aggregate metrics; `--reproducibility 20` is run R5. |
| `src/uwcopilot/governance/migration_gate.py` | The non-inferiority rule, input checks and the certificate renderer. |
| `scripts/llm_migration_gate.py` | Compares two measured configs and writes the gate JSON and the Markdown certificate. |
| `tests/rag/test_generation_eval.py`, `tests/governance/test_migration_gate.py` | Offline tests (FakeLLM and synthetic paired data; no server). |

## How a run flows

1. `make freeze-retrieval` (once): hybrid_rerank top-5 hits for all 133 questions. Before searching, the script checks
   that the Qdrant index holds exactly the chunks of the current source files, so the recorded corpus checksum is
   true. The manifest records the mode, the corpus checksum, the question-file hashes and the hits-file hash.
2. `make eval-generation CONFIG=qwen3b_v1_k3` (R1) and `CONFIG=qwen15b_v1_k3` (R2): each question is answered from its
   first k=3 frozen hits with `rag.answer.answer_from_hits`, under the Ollama queue lock, through the LLM cache.
3. `make eval-generation-repro CONFIG=qwen3b_v1_k3` (R5): 20 seeded questions re-run with the cache off.
4. `make llm-migration-gate`: refuses unless both runs used identical frozen inputs; then a paired bootstrap per
   gated metric and the non-inferiority decision; writes `results/llm_migration_<A>_vs_<B>.json` and renders
   `docs/governance/migration_certificates/<A>_vs_<B>.md` from that JSON only.

## Design choices worth defending

- **Frozen hits are committed** (`eval/generation/`), not regenerated: anyone can re-run the generation eval
  without the index, and the file hash proves which documents were in the prompt.
- **The freeze default is hybrid_rerank, not "auto"**: replacing the committed hits with fallback hits must be an
  explicit choice; the script refuses to overwrite hits made with a different mode without `--force`.
- **Errors are not scored**: a question refused by the context guard or failed over HTTP gets an error row, and the
  gate refuses to compare runs whose scored questions differ. An answer cut off at `num_predict` is scored on the
  text produced so far and counted separately (`errors.length_truncated`).
- **Cached answers never count for latency**: latency and tokens/second use live calls only.
- **The margins are checked against git**: the gate refuses (exit 3) if `config/release_gate.yaml` has uncommitted
  changes or its last commit is newer than the challenger run.

## Interview questions and answers

1. **Why freeze retrieval?** So a model or prompt change is measured on the same documents. Otherwise a
   difference could come from retrieval randomness or an index rebuild, not from the model. The hits file hash ties
   every result to exactly what was in the prompt.
2. **Why key facts instead of an LLM judge?** A judge is another model whose errors we would have to validate. Key
   facts are written once, before any run, and checked deterministically: numbers through the same normalizer the
   numeric-provenance checker uses ("$500k" equals "$500,000"), terms as normalized substrings. A test proves every
   fact is in its gold section. The cost: it is a proxy for faithfulness, not a measure of it, and the same author
   wrote the questions, the guidelines and the facts.
3. **Non-inferiority vs superiority?** A migration (for example to a smaller, cheaper model) does not need to be
   better; it must not be worse by more than an agreed margin. So we ask whether the CI of (challenger - champion)
   stays above -m (or below +m for error rates), not whether it excludes zero.
4. **Why does INCONCLUSIVE block?** If the CI straddles the margin, we cannot rule out a harmful drop. Treating
   "not proven harmful" as "safe" would let regressions through whenever the sample is small.
5. **Why are the CIs wide?** About 30-133 questions per metric. A paired bootstrap helps (both models answer the
   same questions), but small n still gives wide intervals. That is why INCONCLUSIVE is an honest, common outcome.
6. **What is a migration certificate for?** It is the audit record of a model change: which models (digests),
   runtime version, licenses, prompt hashes, frozen-input hashes, git SHA, load averages, n, and the per-metric
   decisions. It is rendered only from the gate's JSON, so it cannot say anything the computation did not produce.
7. **What is the Qwen license point?** The Docker Hub artifacts we imported carry an Apache-2.0 license file, but the
   upstream Hugging Face card for Qwen2.5-3B-Instruct says "Qwen Research License", while the 1.5B card says
   Apache-2.0. We record both sources and do not pick one; a real institution would choose an Apache-2.0 or
   enterprise-hosted model after legal review.
8. **Why Ollama's native API and not its OpenAI-compatible endpoint?** The native `/api/chat` takes `num_ctx`,
   `seed` and `num_predict` per request and returns token counts and durations. The context size matters most: it
   decides how much prompt the model sees, and the OpenAI-compatible endpoint has no per-request way to set it.
9. **Why were the margins committed before the challenger run?** So nobody can pick margins after seeing the
   challenger's numbers. The gate checks the commit time of `config/release_gate.yaml` against the challenger's
   `generated_at_utc`.
10. **Why is the challenger not a clean "size only" comparison?** The only 1.5B artifact available was F16 while the
    3B is Q4_K_M, so size and quantization differ together; the config file says so, and a result cannot be
    attributed to size alone.

## What cannot be claimed

Anything about GPT, Claude, Gemini or Azure quality; production latency; faithfulness in general; regulatory
compliance; or that small-model results transfer to other models. This is a demonstration on SYNTHETIC data, not
a compliance program.
