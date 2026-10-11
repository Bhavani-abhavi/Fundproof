"""Tiny, hand-written watchlists and matchers for the KYC unit tests. No data/raw, no model downloads.

The names below are invented for tests; they only imitate the shape of OFAC and OpenSanctions entries.
"""

from uwcopilot.kyc.blocking import BlockingConfig
from uwcopilot.kyc.matcher import MatcherConfig, SanctionsMatcher
from uwcopilot.kyc.reranker import HashingEmbedder
from uwcopilot.kyc.watchlist import Watchlist, watchlist_from_records
from uwcopilot.tools.schemas import ListSnapshot

FAKE_SHA256 = "a" * 64

TEST_RECORDS: list[dict] = [
    {
        "list_entry_id": "OFAC-1001",
        "name": "HASSAN, Mohammed Ali",
        "aliases": ["Muhammad Ali HASAN"],
        "entity_type": "person",
        "countries": ["Iraq"],
        "programs": ["SDGT"],
        "source": "OFAC",
        "list_name": "OFAC SDN (SDGT)",
    },
    {
        "list_entry_id": "OFAC-1002",
        "name": "AEROCARIBBEAN TEST AIRLINES",
        "aliases": ["AERO-CARIBBEAN TEST"],
        "entity_type": "organization",
        "countries": ["cu"],
        "programs": ["CUBA"],
        "source": "OFAC",
        "list_name": "OFAC SDN (CUBA)",
    },
    {
        "list_entry_id": "OS-NK-test-1",
        "name": "Ivan Petrovich Sidorenko",
        "aliases": ["Иван Сидоренко"],
        "entity_type": "person",
        "countries": ["ru"],
        "programs": ["EU-RUS"],
        "source": "OpenSanctions",
        "list_name": "OpenSanctions (EU Sanctions Map)",
    },
    {
        "list_entry_id": "OS-NK-test-2",
        "name": "Northwind Shadow Trading LLC",
        "aliases": [],
        "entity_type": "organization",
        "countries": ["ae"],
        "programs": ["US-OFAC"],
        "source": "OpenSanctions",
        "list_name": "OpenSanctions (US OFAC SDN)",
    },
    {
        "list_entry_id": "OS-NK-test-3",
        "name": "OCEAN TESTER",
        "aliases": [],
        "entity_type": "vessel",
        "countries": ["pa"],
        "programs": [],
        "source": "OpenSanctions",
        "list_name": "OpenSanctions (Vessels)",
    },
    {
        "list_entry_id": "OS-NK-test-4",
        "name": "Johanna Maria Kowalczyk",
        "aliases": [],
        "entity_type": "person",
        "countries": ["pl"],
        "programs": [],
        "source": "OpenSanctions",
        "list_name": "OpenSanctions (Poland Wanted Persons)",
    },
]


def fake_snapshots() -> list[ListSnapshot]:
    """One fake snapshot per source (valid format, obviously not a real file hash)."""
    return [
        ListSnapshot(source="OFAC", file_name="sdn.csv", sha256=FAKE_SHA256, snapshot_date="2026-10-10"),
        ListSnapshot(source="OpenSanctions", file_name="targets.simple.csv", sha256="b" * 64, snapshot_date=None),
    ]


def tiny_watchlist() -> Watchlist:
    """The six test entries above as a Watchlist."""
    return watchlist_from_records(TEST_RECORDS, fake_snapshots())


def tiny_matcher(threshold: float = 0.8, reranker_weight: float = 0.0, with_embedder: bool = False) -> SanctionsMatcher:
    """A matcher over the tiny watchlist; the embedder (if any) is the offline HashingEmbedder."""
    config = MatcherConfig(
        blocking=BlockingConfig(max_candidates=50),
        threshold=threshold,
        reranker_weight=reranker_weight,
    )
    return SanctionsMatcher(tiny_watchlist(), config, HashingEmbedder() if with_embedder else None)
