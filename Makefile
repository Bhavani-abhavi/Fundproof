# Every step of the project has a make target (ground rule 3).
# Run `make help` to list them.

.PHONY: help setup lint format test data all clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

setup:  ## Create the virtualenv from the lockfile (uv.lock)
	uv sync --frozen

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

all: setup lint test data  ## Run every step end to end

clean:  ## Remove caches (keeps downloaded data)
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
