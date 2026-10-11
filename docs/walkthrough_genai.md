# GenAI track walkthrough: LLM adapter, RAG on Qdrant, tools over MCP

This walkthrough covers the GenAI engineering layer that the agent (Phase 5) and the KYC phase build on: a
provider-neutral LLM adapter, retrieval over guidelines and regulatory documents in a vector database, one
allow-list of tools with strict schemas, and an MCP server that exposes those tools to any MCP client. It explains
every file, the design choices, what was actually run, and likely interview questions.

Detailed design and results live in [rag_design.md](rag_design.md) (retrieval) and [mcp.md](mcp.md) (MCP server).
The agent is in [agent_design.md](agent_design.md) and [walkthrough_phase5.md](walkthrough_phase5.md).

**What ran and what did not.**
- Ran for real: index building with fastembed models (ONNX, no PyTorch), the retrieval evaluation on 113 SYNTHETIC
  questions, and the MCP smoke test as a real subprocess over stdio and streamable HTTP with the MCP Python SDK
  client.
- Not run: any LLM call. There is no API key in this environment, so answer faithfulness, citation accuracy, LLM
  cost and LLM latency are **not measured**. The Anthropic and OpenAI clients are tested only against mocked HTTP
  transports. A Qdrant *server*, Claude Desktop and Claude Code were not run either.

## 1. The LLM adapter (`src/uwcopilot/llm/`)

| File | What it does | Why it is built this way |
|---|---|---|
| `base.py` | `LLMMessage`, `LLMResponse` (text, model, token counts, latency, cost, cached flag, stop reason) and the `LLMClient` protocol. `IncompleteLLMResponse` and `raise_if_incomplete()` turn a cut-off (`max_tokens`, `max_output_tokens`) or refused (`refusal`, `content_filter`) answer into an error that carries the response. | Callers depend on one small interface, so Anthropic, OpenAI and the fake are interchangeable. A truncated memo must never be parsed as if it were complete, but its cost must still be counted. |
| `settings.py` | Reads only named variables: `UWC_LLM_PROVIDER`, `UWC_LLM_MODEL`, the two API keys, `UWC_ANTHROPIC_BASE_URL`, `UWC_OPENAI_BASE_URL`, `UWC_ANTHROPIC_EFFORT`, `MAX_RUN_COST_USD`, the cache and pricing paths. The repo `.env` wins over the environment. Base URLs must be https, or plain http to a loopback host (parsed with `urlsplit`; user info is refused). | A hosting platform can set `ANTHROPIC_BASE_URL` or `ANTHROPIC_AUTH_TOKEN` for its own use. If the SDK picked those up, documents would go to a server nobody chose. Prefix checks are bypassable (`http://localhost.evil.example`), so the host is parsed. |
| `anthropic_client.py` | Messages API through the official SDK, with the key and base URL passed explicitly. Sends `output_config={"effort": ...}` (default `low`). Refuses to start while `ANTHROPIC_CUSTOM_HEADERS` is set. Drops thinking blocks, returns text. | Claude Haiku 5.5 thinks adaptively by default (effort default `medium`) and thinking tokens count toward `max_tokens`; an explicit low effort and budgets of several thousand tokens keep short structured tasks from being cut off. Sampling parameters are not sent because current models reject them. |
| `openai_client.py` | Responses API through the official SDK, `store=False`, explicit key and base URL; refuses to start while `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`, `OPENAI_ADMIN_KEY` or `OPENAI_CUSTOM_HEADERS` is set. | Same reasoning: no implicit configuration. A response without a usage block is an error, because its cost could not be counted. |
| `fake_client.py` | A deterministic fake: scripted strings or a responder function; model names must start with `fake`. | Every test and every offline demo runs without a network. Output from it is labelled, so fake text is never presented as a model's work. |
| `budget.py` | `CostTracker` and `BudgetedLLMClient`: before each call, a pessimistic worst case (every prompt byte one token, plus `max_tokens`) must fit under `MAX_RUN_COST_USD`; afterwards the real cost is recorded, including for incomplete answers. Prices come from `config/llm_pricing.yaml` with a source URL and a date; a model without a verified price is refused. | Ground rule 4 (stop before a run exceeds $10). Refusing an unpriced model is safer than guessing its price. |
| `cache.py` | On-disk JSON cache keyed by the SHA-256 of the provider, model, prompt, parameters and the client's request fingerprint (base URL, effort). A hit returns `cached=True`, `cost_usd=0.0`. | Re-running a script costs nothing. The fingerprint makes sure a response from one gateway or effort level is never served for another. |
| `factory.py` | `get_llm_client()`: real providers become `Cached(Budgeted(provider))`; the fake is only budgeted. Raises `MissingAPIKeyError` (no request attempted) when a key is missing. | One place builds clients, so the cap and the cache cannot be forgotten. |

Evaluation scripts turn the cache **off** by default (`eval_agent.py --llm-cache` turns it on), and
`summarize_llm()` leaves any cached call out of cost and latency. Otherwise a re-run served from the cache would
report near-zero cost and millisecond latency as if measured.

## 2. Shared settings and the risk client

| File | What it does | Why |
|---|---|---|
| `src/uwcopilot/common/settings.py` | `env_reader()` (read a named setting: repo `.env` first, then the process environment, blank means unset) and `is_http_url()`. Used by the LLM, RAG, agent and risk settings. | Four modules had copied the same helper; one copy cannot drift. |
| `src/uwcopilot/risk/client.py` | The only client of the risk scoring REST service. `UWC_RISK_API_URL` is the service's **base URL**; requests go to `<base>/score`. Unset means not configured (`risk_api_not_configured`, nothing sent). Every failure is a `RiskApiError` with a fixed code; "service not there" failures are also `ToolUnavailable`. `score_risk()` is the registry handler. | The agent and the MCP server used to read the same variable with two meanings (base URL vs full endpoint), so one value broke one of them, and unset meant "unavailable" in one and "call localhost" in the other. A test now checks that both send to the same URL. |

## 3. Retrieval (`src/uwcopilot/rag/`)

| File | What it does |
|---|---|
| `chunks.py` | The `Chunk` model: id, citation, document, title, text, metadata. |
| `guidelines.py` | Loads the two SYNTHETIC guideline documents in `corpus/guidelines/`: one chunk per numbered section (a long section is split into parts). |
| `regulatory.py` | Loads the public regulatory documents downloaded by `make data-regulatory` (Federal Reserve SR 26-2 PDF, NY DFS Circular Letter 7 (2024) HTML) into sentence units with page or paragraph numbers, then windows. |
| `chunking.py` | The generic 400-token window with 60-token overlap used for regulatory text. |
| `corpus.py` | Loads everything and fingerprints every source file (SHA-256), so the index can be checked for staleness. |
| `embeddings.py` | fastembed models (dense `BAAI/bge-small-en-v1.5` with the BGE query instruction; sparse `Qdrant/bm25`) and deterministic hashing fakes for tests. |
| `index.py` | Writes chunks into one Qdrant collection with two named vectors (`dense` cosine, `bm25` sparse with Qdrant's IDF modifier). Point ids are `uuid5` of the chunk key, so a rebuild gives the same ids. |
| `retriever.py` | Four modes: `dense`, `bm25`, `hybrid` (Qdrant prefetch of both, reciprocal rank fusion k = 60) and `hybrid_rerank` (cross-encoder on the top 20). |
| `fusion.py` | A plain-Python RRF reference; a test checks Qdrant's fused result equals it, scores included. |
| `rerank.py` | The fastembed cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`) and a term-overlap fake. |
| `search.py` | The `search_guidelines` tool handler (allow-listed), default mode `hybrid_rerank`. |
| `prompt.py` | The grounded-answer prompt: each chunk in its own escaped `<document>` block, an instruction-like-text tripwire that adds a warning attribute. |
| `answer.py` | Retrieve, prompt, call the LLM (4,000-token budget), check citations. No retrieval hits means no LLM call. |
| `citations.py` | Deterministic citation checker: fabricated ids and uncited factual sentences. |
| `eval_set.py`, `evaluation.py`, `metrics.py` | The question schema and gold matching, the evaluation loop with timing, recall@k, MRR, nDCG and bootstrap intervals. |
| `settings.py` | `UWC_QDRANT_URL`, `UWC_QDRANT_API_KEY`, `UWC_QDRANT_PATH`, `UWC_QDRANT_COLLECTION`. Model ids are fixed in code because the results are only valid for those exact models. |

Scripts: `build_index.py` (`make index`), `eval_rag.py` (`make eval-rag`). Results: `results/rag_metrics.json`,
`results/rag_per_question.jsonl`, `results/rag_index_manifest.json`.

Headline numbers, from `results/rag_metrics.json` (103 answerable SYNTHETIC questions, 95% bootstrap intervals):
recall@5 is 0.9078 [0.8544, 0.9563] for `hybrid_rerank` and 0.8786 [0.8155, 0.9369] for `dense`; the paired
difference is 0.0291 [-0.0291, 0.0874], so on this set the reranker did not show a recall@5 gain. The reranker's
p50 latency was 3215.49 ms per query against 37.97 ms for `hybrid`, on a loaded 4-CPU machine. The same team wrote
the corpus and the questions; see the bias section of [rag_design.md](rag_design.md).

## 4. Tools and the MCP server

| File | What it does |
|---|---|
| `src/uwcopilot/tools/schemas.py` | Pydantic models for every tool input and output. Inputs forbid unknown fields. `ScreeningResult` can only be `clear` or `potential_match`, and `requires_human_review` is computed, so it cannot be set to false for a match. `ListSnapshot` carries the OpenSanctions attribution. |
| `src/uwcopilot/tools/registry.py` | `ALLOWED_TOOLS`: `search_guidelines`, `screen_entity`, `score_risk`. `call_tool()` rejects any other name, validates arguments and output. Handlers are imported by path on first use; a test checks that every path resolves. |
| `src/uwcopilot/mcp_server/server.py` | The MCP server (official MCP Python SDK): four read-only tools bound to registry tools (checked at build time), guideline documents and sections as resources, and the `underwriting_review` prompt. Unknown arguments are rejected; the published output schemas are the serialization schemas. |
| `src/uwcopilot/mcp_server/guideline_store.py` | Guideline documents and exact section lookup for resources and `get_guideline_section`. |
| `src/uwcopilot/mcp_server/models.py` | `GuidelineSection` and `ScoreApplicationResult` (either a score or an error, never both). |
| `src/uwcopilot/mcp_server/prompts.py` | The prompt text, which tells the client's model never to approve and to treat returned text as data. |
| `src/uwcopilot/mcp_server/settings.py` | The risk service settings through the shared loader. |
| `src/uwcopilot/mcp_server/cli.py`, `__main__.py` | `python -m uwcopilot.mcp_server [--transport stdio|streamable-http]`; HTTP listens on loopback by default. |
| `scripts/mcp_smoke.py` | Starts the server as a real subprocess and drives it with the SDK client; writes `results/mcp_smoke_<transport>.json`. |

`results/mcp_smoke_stdio.json` and `results/mcp_smoke_streamable_http.json` each record 17 of 17 checks passed and
0 LLM calls.

## 5. Design choices

- **Explicit configuration only.** Every client gets its base URL and key from named project settings. The
  platform's `ANTHROPIC_BASE_URL` and similar are never read; SDK variables that would apply implicitly make the
  client refuse to start. Tests clear every `UWC_*` variable, so a developer's shell cannot change them.
- **One allow-list, enforced twice.** The registry validates names, arguments and outputs for the agent; the MCP
  server binds each of its tools to a registry tool and adds strict argument checking on top of the SDK.
- **Hybrid retrieval with rank fusion.** Underwriting questions mix paraphrase with exact tokens (ids, "1.25x",
  "SDN"). Dense handles the first, BM25 the second; RRF fuses by rank, so no score calibration is needed.
- **Retrieved text is data.** Chunks are escaped and delimited; instruction-like text is flagged but still shown,
  because the guidelines themselves quote such phrases. Nothing a model writes can approve or decline.
- **Fail closed, never invent.** No key: no call, "not measured". No price: no call. No risk service: a structured
  error with `score: null`. A cut-off answer: an error, not partial text.

## 6. Likely interview questions

1. **How do you make an LLM integration provider-agnostic?**
   A small protocol (`LLMClient.complete(system, messages, max_tokens)`) and a provider-neutral response type. Each
   provider client maps its SDK to that type; caching and the cost cap are wrappers, so they work for every
   provider. Swapping Anthropic for OpenAI is a settings change, and tests use a deterministic fake.

2. **Why not just let the SDK read `ANTHROPIC_BASE_URL` and the API key from the environment?**
   Because the environment is not under the application's control. A hosting platform here sets
   `ANTHROPIC_BASE_URL` for its own purposes; an SDK that picked it up would send application documents to that
   endpoint. So the base URL and key are passed explicitly from project settings, and variables the SDK would
   apply anyway make the client refuse to start.

3. **How do you enforce a cost cap before spending money?**
   Before each call, a pessimistic worst case (every prompt byte as one token plus `max_tokens` output) is priced
   from a pricing file with sources and dates. If spent-so-far plus that worst case exceeds the cap, the call is
   refused. After the call the real cost from the returned token counts is recorded, also for a truncated
   answer. A model without a verified price is refused, because the cap could not be enforced.

4. **What changes with reasoning models that think before answering?**
   Thinking tokens count toward `max_tokens`. With a small budget a model can spend it thinking and return a
   truncated or empty answer. So the client sets effort explicitly (low for short structured tasks), budgets are
   several thousand tokens, and a `max_tokens` or `refusal` stop raises an error instead of returning partial
   text. None of this has been measured against a real model here.

5. **Why hybrid search, and why reciprocal rank fusion instead of adding scores?**
   Dense embeddings capture meaning, BM25 captures exact terms and ids; underwriting questions need both. Cosine
   and BM25 scores are on different scales, so adding them needs calibration. RRF uses only ranks
   (`1 / (k + rank)`), so it is robust, and a test checks Qdrant's fusion against a plain-Python implementation.

6. **Did the cross-encoder reranker help?**
   Point estimates were higher at the top of the list, but the paired bootstrap of recall@5 against dense
   retrieval was 0.0291 with a 95% interval of [-0.0291, 0.0874], which includes zero, and it cost about 3.2 s per
   query on this CPU against about 38 ms for hybrid. So on this set it did not earn its cost for recall@5. The
   default stayed as designed before the evaluation; changing it after seeing results would be tuning on test.

7. **How do you evaluate retrieval honestly when you wrote both the corpus and the questions?**
   Questions were paraphrased away from the section wording and checked for lexical overlap; unanswerable
   questions are included; all parameters were fixed before the first run; results carry bootstrap intervals and
   per-category breakdowns; and the bias is stated in the design doc rather than hidden.

8. **What is MCP and why expose the tools through it?**
   The Model Context Protocol is a standard way for an AI client to discover and call tools, read resources and
   use prompt templates. Exposing the underwriting tools over MCP lets any MCP client use them without new glue
   code, while the guarantees (allow-list, schemas, human review, no decision tools) stay enforced in the server.

9. **How do you stop prompt injection from retrieved documents?**
   Structure first: retrieved text is escaped and delimited as data, tools come from an allow-list chosen in code,
   and no tool can approve, decline or clear anything. A pattern tripwire flags instruction-like text for the
   audit record, but it is not relied on, because unknown phrasings pass.

10. **How do you keep the README from drifting from the results?**
    `scripts/update_readme_results.py` writes the results table from `results/*.json`, and a unit test re-renders
    it and fails if the committed README differs. Every metric that needs an LLM says "not measured".
