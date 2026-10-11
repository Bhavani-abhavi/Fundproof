"""Integration test: the graph with the real tools (full watchlists in data/raw, the Qdrant index, fastembed models).

Excluded by default; run `make test-integration`. Skipped when the lists, the validated screening threshold or the
index are missing. The LLM is still the deterministic template writer (no API key is used).
"""

from pathlib import Path

import pytest
from agent_helpers import committed_by_scenario, make_deps, memory_agent

from uwcopilot.agent.runner import start_run
from uwcopilot.agent.settings import AgentSettings
from uwcopilot.agent.tools import close_tool_handlers, default_tool_handlers
from uwcopilot.common.paths import RAW_DATA_DIR, REPO_ROOT, RESULTS_DIR

pytestmark = pytest.mark.integration

NEEDED = [
    RAW_DATA_DIR / "ofac" / "sdn.csv",
    RAW_DATA_DIR / "opensanctions" / "targets.simple.csv",
    RESULTS_DIR / "kyc_metrics.json",
    REPO_ROOT / "qdrant_storage",
]


@pytest.mark.skipif(not all(path.exists() for path in NEEDED), reason="needs data/raw, kyc_metrics.json and the index")
def test_real_tools_feed_the_graph_and_escalation_follows_screening(tmp_path: Path) -> None:
    application = committed_by_scenario("watchlist_owner_1", line="insurance")
    handlers = default_tool_handlers(AgentSettings(), retrieval_mode="hybrid")
    try:
        outcome = start_run(
            memory_agent(make_deps(tmp_path, handlers=handlers)), application.model_dump(mode="json"), "integration"
        )
    finally:
        close_tool_handlers(handlers)
    assert outcome.status == "awaiting_review"
    results = outcome.state["tool_results"]
    searches = [result for result in results if result["tool"] == "search_guidelines"]
    assert len(searches) == 3 and all(result["status"] == "ok" for result in searches)
    hits = [hit for result in searches for hit in result["output"]["hits"]]
    assert hits and all(hit["source_doc"] == "harborline_mutual_commercial" for hit in hits)
    screenings = [result for result in results if result["tool"] == "screen_entity"]
    assert len(screenings) == len(application.owners) + 1
    assert all(result["status"] == "ok" for result in screenings)
    snapshots = screenings[0]["output"]["list_snapshots"]
    assert {snapshot["source"] for snapshot in snapshots} == {"OFAC", "OpenSanctions"}
    flagged = [result for result in screenings if result["output"]["status"] == "potential_match"]
    review = outcome.review_request
    if flagged:
        assert review["review_queue"] == "Compliance" and review["blocking_open"]
    assert review["memo_status"] == "passed"
