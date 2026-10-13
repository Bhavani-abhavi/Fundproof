# Every step of the project has a make target (ground rule 3).
# Run `make help` to list them.

.PHONY: help setup lint format test test-integration data data-regulatory index eval-rag all clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

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
all: setup lint test data phase1 index eval-rag kyc-data eval-kyc applications eval-agent eval-agent-homonyms mcp-smoke readme-results  ## Run every step end to end, in order

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

# Phase 1

.PHONY: fairness
fairness:  ## Fairness demo on UCI German Credit (AIR, equal opportunity, proxy check) -> results/risk_metrics.json
	uv run python scripts/fairness_german.py

.PHONY: train-insurance
train-insurance:  ## freMTPL2 frequency + severity models (GLM vs LightGBM) -> results/risk_metrics.json, models/insurance/
	uv run python scripts/train_insurance.py

.PHONY: train-credit
train-credit:  ## Lending Club credit PD (scorecard vs LightGBM, out-of-time) -> results/risk_metrics.json, models/credit/
	uv run python scripts/train_credit.py

.PHONY: reports
reports:  ## Render model cards, validation report, walkthrough and README results from results/risk_metrics.json
	uv run python scripts/render_reports.py

.PHONY: phase1
phase1:  ## Phase 1 end to end, in order: train-insurance, fairness, train-credit, reports
	$(MAKE) train-insurance
	$(MAKE) fairness
	$(MAKE) train-credit
	$(MAKE) reports

.PHONY: serve
serve:  ## Start the scoring API (POST /score, GET /health) on http://127.0.0.1:8000 (docs at /docs)
	uv run uvicorn uwcopilot.risk.service:app --host 127.0.0.1 --port 8000

# Local open-weight model (innovations S1). Never part of CI: downloads GBs and needs a running `ollama serve`.

.PHONY: fetch-local-model test-ollama-live

LOCAL_MODEL ?= qwen2.5-3b-instruct-q4km
OLLAMA_BIN ?= ollama

fetch-local-model:  ## Import a pinned local model into Ollama from Docker Hub (LOCAL_MODEL=<name>|all, OLLAMA_BIN=path)
	uv run python scripts/fetch_local_model.py --model $(LOCAL_MODEL) --ollama-bin $(OLLAMA_BIN)

test-ollama-live:  ## One live smoke call to the local Ollama server (needs `make fetch-local-model` and `ollama serve`)
	UWC_OLLAMA_LIVE=1 uv run pytest -m integration -s tests/llm/test_ollama_client.py

# SYNTHETIC startup lending program (corpus/programs/) and its example files. Offline, no downloads.

.PHONY: startup-applications

startup-applications:  ## Make the SYNTHETIC startup credit examples (a case per program rule); eval/startup_applications/
	uv run python scripts/make_startup_applications.py

# Web app backend (src/uwcopilot/web). Serves /api and the static export in frontend/out, like the container does.

.PHONY: web-serve

web-serve:  ## Serve the web app on http://127.0.0.1:7860 (API + frontend/out; dev login OFF; build the frontend first)
	uv run uvicorn uwcopilot.web.app:app --host 127.0.0.1 --port 7860 --no-access-log

# Web app frontend (frontend/, Next.js static export). Needs Node 22 + npm. See docs/web/frontend.md.

.PHONY: web-install web-build web-lint web-dev web-test web-screenshots

web-install:  ## Install the exact frontend dependencies from frontend/package-lock.json (npm ci)
	cd frontend && npm ci

web-build:  ## Build the static frontend into frontend/out (served by the FastAPI app and the container)
	cd frontend && npm run build

web-lint:  ## Frontend checks: ESLint, TypeScript, WCAG contrast of the design tokens
	cd frontend && npm run lint && npm run typecheck && npm run check:contrast

web-dev:  ## Local dev: API with DEV LOGIN on :8000 + Next dev server on http://localhost:3000 (proxies /api)
	@trap 'kill 0' EXIT INT TERM; \
	UWC_DEV_AUTH=1 uv run uvicorn uwcopilot.web.app:app --host 127.0.0.1 --port 8000 --reload & \
	cd frontend && UWC_API_DEV_ORIGIN=http://127.0.0.1:8000 npm run dev

web-test:  ## Web checks: backend tests (tests/web) + frontend lint, typecheck and contrast
	uv run pytest tests/web -q
	$(MAKE) web-lint

web-screenshots:  ## Screenshots of frontend/out at 1280/375 px, light + dark -> docs/web/screenshots/ (Playwright)
	PLAYWRIGHT_BROWSERS_PATH=$${PLAYWRIGHT_BROWSERS_PATH:-/opt/pw-browsers} uv run --no-project --with playwright==1.56.0 python frontend/scripts/screenshots.py $(if $(LIVE_URL),--live-url $(LIVE_URL),)

# Founder Studio S1: metros, data snapshots and the coverage report. Downloads only public data; needs network.

.PHONY: studio-metros studio-data studio-coverage studio-fixtures studio-test

METRO ?= all

studio-metros:  ## Fill the real metros' CBSA codes and county FIPS from the Census delineation file (+ population rank)
	uv run python scripts/studio_fetch_delineation.py

studio-data:  ## Download a metro's data snapshot (METRO=<id>|all); exits 1 naming each unreachable host or missing key
	uv run python -m uwcopilot.studio.data.snapshot --metro $(METRO)

studio-coverage:  ## Data coverage per metro x business type x factor -> results/studio_metro_coverage.json (offline)
	uv run python scripts/studio_metro_coverage.py

studio-fixtures:  ## Re-record the Studio test fixtures from keyless live hosts (never in CI)
	uv run python tests/studio/fixtures/record_fixtures.py

studio-test:  ## Studio tests only (offline, recorded fixtures)
	uv run pytest tests/studio -q

# Phase 2: document intelligence

.PHONY: packets eval-docintel

packets:  ## Make the SYNTHETIC document packets (150 credit + 150 insurance, seed 42) -> data/processed/packets/, eval/packets/manifest.json
	uv run python scripts/make_packets.py --out data/processed/packets --n-credit 150 --n-insurance 150 --seed 42 --test-credit 30 --test-insurance 30

eval-docintel:  ## Evaluate the document pipeline on the 60 test packets, incl. forced OCR (needs `make packets`; ~40 min) -> results/docintel_metrics.json
	uv run python scripts/eval_docintel.py

# Innovation SIG: numeric provenance (docs/innovations.md). Offline, no LLM. The reference sentences and the mutations
# are FROZEN test sets: rebuild them only when a change to provenance/numbers.py changes their generation, and say so.
# eval-provenance records whether regenerating the mutations with the seed gives exactly the committed file.

.PHONY: reference-sentences mutations eval-provenance

reference-sentences:  ## Rebuild the frozen eval/provenance/reference_sentences.jsonl from the SYNTHETIC guidelines (see note above)
	uv run python scripts/make_reference_sentences.py

mutations:  ## Rebuild the frozen eval/provenance/mutations.jsonl from the reference sentences, seed 42 (see note above)
	uv run python scripts/make_mutations.py --seed 42

eval-provenance:  ## SIG checker mutation test, incumbent baseline and validator adversarial set -> results/provenance_metrics.json
	uv run python scripts/eval_provenance.py

# Innovation S1b: frozen retrieval, generation eval on a local model, LLM migration gate. The eval targets call the
# local Ollama server (one job at a time, data/cache/ollama.lock) and are never part of CI.

.PHONY: freeze-retrieval eval-generation eval-generation-repro llm-migration-gate

CONFIG ?= qwen3b_v1_k3
CHAMPION ?= qwen3b_v1_k3
CHALLENGER ?= qwen15b_v1_k3

freeze-retrieval:  ## Freeze hybrid_rerank top-5 hits per eval question (needs `make index`) -> eval/generation/frozen_hits.jsonl + manifest
	uv run python scripts/freeze_retrieval.py

eval-generation:  ## Frozen RAG generation eval for CONFIG=<id> on Ollama (hours) -> results/generation/<id>.jsonl, results/generation_metrics.json
	uv run python scripts/eval_generation.py --config $(CONFIG)

eval-generation-repro:  ## R5: rerun 20 seeded questions of CONFIG with the cache off; count bit-identical answers
	uv run python scripts/eval_generation.py --config $(CONFIG) --reproducibility 20

llm-migration-gate:  ## Champion vs challenger non-inferiority (margins: config/release_gate.yaml) -> results/llm_migration_*.json + certificate
	uv run python scripts/llm_migration_gate.py --champion $(CHAMPION) --challenger $(CHALLENGER)

# Deployment (S11): the container image, its local smoke test, and the Hugging Face Space bundle. See docs/deployment.md.
# Behind a TLS-inspecting proxy, pass its CA bundle: make docker-build EXTRA_CA_CERTS=/path/to/ca-bundle.crt

.PHONY: docker-build docker-run docker-smoke space-bundle

DOCKER_IMAGE ?= fundproof:dev
DOCKER_PORT ?= 7860
DOCKER_SMOKE_NAME ?= fundproof-smoke
EXTRA_CA_CERTS ?=
# A literal comma, because a comma inside $(if ...) separates its arguments.
comma := ,

docker-build:  ## Build the container image DOCKER_IMAGE (frontend export + locked Python deps) from the Dockerfile
	docker build $(if $(EXTRA_CA_CERTS),--secret id=extra_ca_certs$(comma)src=$(EXTRA_CA_CERTS),) -t $(DOCKER_IMAGE) .

docker-run:  ## Run DOCKER_IMAGE on http://127.0.0.1:7860 (dev login off, data lost when it stops; Ctrl+C stops it)
	docker run --rm -p 127.0.0.1:$(DOCKER_PORT):7860 $(DOCKER_IMAGE)

docker-smoke:  ## Start DOCKER_IMAGE, smoke-test it, stop it -> results/deployment_metrics.json (runs.local-container)
	@docker rm -f $(DOCKER_SMOKE_NAME) >/dev/null 2>&1 || true
	docker run -d --rm --name $(DOCKER_SMOKE_NAME) -p 127.0.0.1:$(DOCKER_PORT):7860 $(DOCKER_IMAGE) && \
	uv run python scripts/smoke_deploy.py --url http://127.0.0.1:$(DOCKER_PORT) --target local-container \
		--docker-image $(DOCKER_IMAGE) --container $(DOCKER_SMOKE_NAME); \
	status=$$?; docker stop $(DOCKER_SMOKE_NAME) >/dev/null 2>&1; exit $$status

space-bundle:  ## Write the Space bundle (what the deploy workflow uploads) to build/space, for inspection
	rm -rf build/space
	uv run python scripts/deploy_space.py bundle --out build/space --git-sha $$(git rev-parse HEAD)

# Founder Studio S3: Monte Carlo feasibility (SYNTHETIC worlds, offline)

.PHONY: studio-feasibility

studio-feasibility:  ## Validate the feasibility simulator on SYNTHETIC worlds -> results/studio_feasibility_metrics.json
	uv run python scripts/eval_studio_feasibility.py

# Founder Studio S2: SiteIQ location scoring (SYNTHETIC cities with known truth, offline)

.PHONY: studio-siteiq studio-siteiq-test

studio-siteiq:  ## Validate SiteIQ on 10 SYNTHETIC cities (+ real-metro sanity table) -> results/studio_siteiq_metrics.json
	uv run python scripts/eval_studio_siteiq.py

studio-siteiq-test:  ## SiteIQ tests only (offline, fixtures)
	uv run pytest tests/studio -q -k siteiq

.PHONY: studio-siteiq-fixtures

studio-siteiq-fixtures:  ## Re-record the OpenRouteService drive-time fixture (needs ORS_API_KEY; one request; never in CI)
	uv run python tests/studio/fixtures/ors/record_isochrones.py

.PHONY: studio-data-refresh

studio-data-refresh:  ## Like studio-data, but ignore cached answers and fetch every source again (METRO=<id>|all)
	uv run python -m uwcopilot.studio.data.snapshot --metro $(METRO) --refresh

# Innovation S2: injection containment by layer (docs/innovations.md). eval/injection_payloads.jsonl is frozen
# (its sha256 is in eval/injection_payloads_manifest.json) and must be committed before the full run.

.PHONY: injection-payloads-check eval-injection eval-injection-smoke injection-detector test-security

injection-payloads-check:  ## Rebuild the payload set from the pinned dataset revision (network) and compare with the frozen file
	uv run python scripts/fetch_injection_payloads.py --check

eval-injection:  ## R4 on Ollama: 60 payloads x L0/L1/L2 + 30 benign L1/L2 calls (hours) -> results/injection_metrics.json
	uv run python scripts/eval_injection.py

eval-injection-smoke:  ## 3-call live smoke of the injection eval (Ollama) -> results/injection/smoke/
	uv run python scripts/eval_injection.py --limit 3

injection-detector:  ## Tripwire recall and false positives only, no model -> results/injection/injection_metrics_detector_only.json
	uv run python scripts/eval_injection.py --detector-only

test-security:  ## Containment invariants, normalization, applicant block, scorer and payload freeze tests (offline)
	uv run pytest tests/security -q

# Founder Studio S5: path to approval (seeded SYNTHETIC plans, offline; real metros in their current mode)

.PHONY: studio-paths studio-paths-test

studio-paths:  ## Path-to-approval search on SYNTHETIC plans (6 metros x 6 types x 15) -> results/studio_path_metrics.json
	uv run python scripts/eval_studio_paths.py

studio-paths-test:  ## Path-to-approval tests only (offline)
	uv run pytest tests/studio -q -k path

# Founder Studio data layers (docs/studio/data_sources.md): Overture places, the HPMS road layer and LODES daytime
# workers are built by `make studio-data`; these targets check NAICS codes and re-record the layers' test fixtures.

.PHONY: studio-naics studio-layer-fixtures

studio-naics:  ## Check the profiles' NAICS codes against the BLS QCEW industry titles file -> results/studio_naics_check.json
	uv run python scripts/studio_verify_naics.py

studio-layer-fixtures:  ## Re-record the Overture, LODES, HPMS-page and BLS-title test fixtures (network; never in CI)
	uv run python tests/studio/fixtures/record_fixtures.py --only overture,lodes,hpms_pages,industry_titles

# Web app feature pages without a backend (docs/web/frontend.md, "Mock API"): the mock API replays answers recorded
# from the real app on SYNTHETIC inputs.

.PHONY: web-mock-fixtures web-mock web-screenshots-mock

web-mock-fixtures:  ## Re-record frontend/mock-api/fixtures from the real app in-process (dev login, SYNTHETIC demo plan and files)
	UWC_SCREENING_LISTS=ofac uv run python frontend/scripts/record_mock_fixtures.py

web-mock:  ## Serve frontend/out + the mock API on http://127.0.0.1:4174 (build the frontend first; Node only)
	cd frontend && npm run mock

web-screenshots-mock:  ## Feature-page screenshots on the mock API, 1280/375 px, light + dark -> docs/web/screenshots/mock/
	PLAYWRIGHT_BROWSERS_PATH=$${PLAYWRIGHT_BROWSERS_PATH:-/opt/pw-browsers} uv run --with playwright==1.56.0 --with pillow python frontend/scripts/screenshots_mock.py --small

# Web quality metrics on the LOCAL container (scripts/web_metrics.py, docs/deployment.md): Playwright end-to-end flows,
# axe-core scans and Lighthouse. The container runs WITH the dev login, because a script cannot sign in with Hugging
# Face; a Space refuses the dev login. Build the image first (make docker-build).

.PHONY: web-metrics-tools web-metrics

WEB_METRICS_PORT ?= 7862
WEB_METRICS_NAME ?= fundproof-metrics

web-metrics-tools:  ## Install the pinned Lighthouse CLI and axe-core into tools/web-metrics (npm ci; Node 22)
	cd tools/web-metrics && npm ci

web-metrics:  ## Start DOCKER_IMAGE with dev login, run flows + axe + Lighthouse, stop it -> results/web_metrics.json
	@docker rm -f $(WEB_METRICS_NAME) >/dev/null 2>&1 || true
	docker run -d --rm --name $(WEB_METRICS_NAME) -e UWC_DEV_AUTH=1 -p 127.0.0.1:$(WEB_METRICS_PORT):7860 $(DOCKER_IMAGE) && \
	PLAYWRIGHT_BROWSERS_PATH=$${PLAYWRIGHT_BROWSERS_PATH:-/opt/pw-browsers} uv run --with playwright==1.56.0 --with pillow \
		python scripts/web_metrics.py --url http://127.0.0.1:$(WEB_METRICS_PORT) --target local-container \
		--docker-image $(DOCKER_IMAGE) --container $(WEB_METRICS_NAME); \
	status=$$?; docker stop $(WEB_METRICS_NAME) >/dev/null 2>&1; exit $$status
