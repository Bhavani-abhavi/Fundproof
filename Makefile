# Every step of the project has a make target (ground rule 3).
# Run `make help` to list them.

.PHONY: help setup lint format test test-integration data data-regulatory index eval-rag all clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

setup:  ## Create the virtualenv from the lockfile (fails if uv.lock is out of date)
	uv sync --locked

lint:  ## Run ruff lint and format checks
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-format and auto-fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

test:  ## Run the offline test suite (integration tests excluded)
	uv run pytest

test-integration:  ## Run integration tests; needs `make data index kyc-data eval-kyc` first (missing inputs are skipped)
	uv run pytest -m integration

data:  ## Download all public datasets into data/raw/ (needs Kaggle key for Lending Club)
	uv run python scripts/download_data.py

data-regulatory:  ## Download only the public regulatory documents used for RAG
	uv run python scripts/download_data.py --datasets regulatory_docs

index:  ## Build the Qdrant index (guidelines + regulatory docs; needs `make data-regulatory`); UWC_QDRANT_URL = server
	uv run python scripts/build_index.py

eval-rag:  ## Evaluate dense/BM25/hybrid/hybrid+rerank retrieval (needs `make index`); writes results/rag_metrics.json
	uv run python scripts/eval_rag.py

.PHONY: kyc-data eval-kyc

kyc-data:  ## Make the SYNTHETIC KYC screening customers + ground truth (needs `make data`); data/processed/kyc/
	uv run python scripts/make_kyc_customers.py

eval-kyc:  ## Evaluate sanctions screening (needs `make kyc-data`); writes results/kyc_metrics.json and a figure
	uv run python scripts/eval_kyc.py

.PHONY: applications agent-demo eval-agent eval-agent-homonyms readme-results ci

DEMO_APPLICATION ?= eval/applications/APP-C-027.json

applications:  ## Make the SYNTHETIC agent evaluation applications (needs `make data`); eval/applications/
	uv run python scripts/make_applications.py

agent-demo:  ## Fake-LLM agent run (needs `make index` and `make eval-kyc`): stops at the human checkpoint, then a scripted demo decision
	@THREAD=demo-$$(date +%s); \
	uv run python scripts/run_agent.py --application $(DEMO_APPLICATION) --provider fake --thread-id $$THREAD && \
	uv run python scripts/run_agent.py --application $(DEMO_APPLICATION) --provider fake --thread-id $$THREAD \
		--decision refer --reviewer-id demo.senior --role senior_underwriter \
		--rationale "Demo: the notes claim a prior approval; refer the file to Fraud Operations."

eval-agent:  ## Evaluate the agent (needs `make data index`); writes results/agent_metrics.json
	uv run python scripts/eval_agent.py

eval-agent-homonyms:  ## Sensitivity run: the application set WITHOUT the exact-name redraw; writes results/agent_metrics_homonyms_kept.json
	uv run python scripts/make_applications.py --keep-exact-name-homonyms
	uv run python scripts/eval_agent.py --applications-dir data/processed/agent/applications_homonyms_kept \
		--output results/agent_metrics_homonyms_kept.json \
		--per-application results/agent_per_application_homonyms_kept.jsonl

readme-results:  ## Write the GenAI-track results table in README.md from results/*.json
	uv run python scripts/update_readme_results.py

ci:  ## What CI runs: lint and the offline test suite
	$(MAKE) lint
	$(MAKE) test

# The full pipeline, in dependency order. `data` downloads several GB and needs a Kaggle key for Lending Club.
all: setup lint test data index eval-rag kyc-data eval-kyc applications eval-agent eval-agent-homonyms mcp-smoke readme-results  ## Run every step end to end, in order

clean:  ## Remove caches (keeps downloaded data)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: mcp mcp-smoke

# The leading @ matters: make echoes recipe lines to stdout, which is the MCP stdio channel.
mcp:  ## Run the MCP server over stdio for an MCP client; MCP_ARGS="--transport streamable-http" serves HTTP
	@uv run --quiet python -m uwcopilot.mcp_server $(MCP_ARGS)

mcp-smoke:  ## MCP server as a subprocess + SDK client (needs `make index`, `make eval-kyc`); writes results/mcp_smoke_*.json
	uv run python scripts/mcp_smoke.py
	uv run python scripts/mcp_smoke.py --transport streamable-http
