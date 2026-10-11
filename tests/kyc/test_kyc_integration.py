"""Integration tests on the real downloaded lists (data/raw). Excluded by default; run `make test-integration`.

They need data/raw/ofac and data/raw/opensanctions (`make data`), take about a minute and a few GB of memory,
and download nothing.
"""

import pytest

from uwcopilot.kyc.matcher import MatcherConfig, SanctionsMatcher
from uwcopilot.kyc.watchlist import DEFAULT_OFAC_ALT, DEFAULT_OFAC_SDN, DEFAULT_OPENSANCTIONS, load_watchlist
from uwcopilot.tools.schemas import ScreeningRequest

pytestmark = pytest.mark.integration

needs_lists = pytest.mark.skipif(
    not (DEFAULT_OFAC_SDN.exists() and DEFAULT_OFAC_ALT.exists() and DEFAULT_OPENSANCTIONS.exists()),
    reason="needs data/raw OFAC and OpenSanctions files (make data)",
)


@needs_lists
def test_full_watchlist_loads_and_a_listed_ofac_name_is_flagged() -> None:
    watchlist = load_watchlist()
    assert len(watchlist.entries) > 1_000_000
    assert {snapshot.source for snapshot in watchlist.snapshots} == {"OFAC", "OpenSanctions"}
    # Rules only, with a fixed threshold for this check (the validated one lives in results/kyc_metrics.json).
    matcher = SanctionsMatcher(watchlist, MatcherConfig(threshold=0.9))
    result = matcher.screen(ScreeningRequest(name="Aero Caribbean Airlines", entity_type="organization"))
    assert result.status == "potential_match"
    assert "OFAC-36" in {candidate.list_entry_id for candidate in result.candidates}
