"""Integration: scripts/mcp_smoke.py against the real Qdrant index and fastembed models (`make index` first).

Excluded by default; run with `uv run pytest -m integration tests/mcp`. Skipped, with the reason, when the index,
the watchlists or the validated screening threshold are missing (`make data index kyc-data eval-kyc`).
"""

import json
from pathlib import Path

import pytest

import mcp_smoke
from uwcopilot.common.paths import RAW_DATA_DIR, RESULTS_DIR
from uwcopilot.rag.settings import load_rag_settings

NEEDED_FILES = [
    RAW_DATA_DIR / "ofac" / "sdn.csv",
    RAW_DATA_DIR / "opensanctions" / "targets.simple.csv",
    RESULTS_DIR / "kyc_metrics.json",
]


@pytest.fixture
def smoke_inputs() -> None:
    """Skip unless the watchlists, the screening threshold and the Qdrant collection all exist."""
    missing = [str(path) for path in NEEDED_FILES if not path.exists()]
    if missing:
        pytest.skip(f"missing {missing}: run `make data kyc-data eval-kyc`")
    settings = load_rag_settings()
    if settings.mode == "local-embedded" and not Path(settings.qdrant_path).exists():
        pytest.skip(f"no local index at {settings.qdrant_path}: run `make index`")  # do not create an empty folder
    try:
        mcp_smoke.check_index()
    except mcp_smoke.SmokeSetupError as error:
        pytest.skip(str(error))


@pytest.mark.integration
@pytest.mark.parametrize("transport", ["stdio", "streamable-http"])
def test_smoke_script_passes_every_check(tmp_path: Path, transport: str, smoke_inputs: None) -> None:
    """Every smoke check passes and the report says which transport and protocol version ran."""
    output = tmp_path / f"smoke_{transport}.json"
    assert mcp_smoke.main(["--transport", transport, "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["all_passed"] is True
    assert report["transport"] == transport
    assert report["llm_calls"] == 0
    assert len(report["search"]["hits"]) == report["search"]["k"]
    assert report["score_application"]["structured_content"]["error"]["code"] == "risk_api_unreachable"
