# Every step of the project has a make target (ground rule 3).
# Run `make help` to list them.

.PHONY: help setup lint format test data all clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

setup:  ## Create the virtualenv from the lockfile (fails if uv.lock is out of date)
	uv sync --locked

lint:  ## Run ruff lint and format checks
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-format and auto-fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

test:  ## Run the test suite
	uv run pytest

data:  ## Download all public datasets into data/raw/ (needs Kaggle key for Lending Club)
	uv run python scripts/download_data.py

all: setup lint test data  ## Set up, lint, test and download data (then `make phase1` trains and renders reports)

clean:  ## Remove caches (keeps downloaded data)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

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
