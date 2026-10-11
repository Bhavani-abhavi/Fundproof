"""Offline fakes for the MCP server tests. No data/raw, no model downloads, no network, no LLM.

- build_guideline_retriever(): the real SYNTHETIC guideline sections from corpus/guidelines/, indexed with the
  deterministic FAKE hashing embedders into an in-memory Qdrant, with the FAKE term-overlap reranker.
- tiny_matcher(): a sanctions matcher over three invented watchlist entries (fake checksums).
- risk_api(handler): a RiskApiClient whose network is an httpx.MockTransport.
- run_with_client(server, scenario): connect the MCP SDK client to the server in-process and run scenario(client).
"""

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import anyio
import httpx
from qdrant_client import QdrantClient

from mcp import Client
from uwcopilot.kyc.blocking import BlockingConfig
from uwcopilot.kyc.matcher import MatcherConfig, SanctionsMatcher
from uwcopilot.kyc.watchlist import watchlist_from_records
from uwcopilot.mcp_server.guideline_store import GuidelineStore
from uwcopilot.mcp_server.server import ServerDependencies, build_server
from uwcopilot.rag.embeddings import HashingDenseEmbedder, HashingSparseEmbedder
from uwcopilot.rag.guidelines import load_guideline_chunks
from uwcopilot.rag.index import create_collection, index_chunks
from uwcopilot.rag.rerank import TermOverlapReranker
from uwcopilot.rag.retriever import Retriever
from uwcopilot.risk.client import RiskApiClient
from uwcopilot.tools.schemas import ListSnapshot

T = TypeVar("T")
TEST_COLLECTION = "mcp_test_guidelines"
RISK_URL = "http://risk.test"  # base URL; requests go to http://risk.test/score

# Invented names that only imitate the shape of OFAC and OpenSanctions entries.
WATCHLIST_RECORDS: list[dict[str, Any]] = [
    {
        "list_entry_id": "OFAC-T-9001",
        "name": "ZARKOVIAN HOLDINGS TESTCO LTD",
        "aliases": ["ZARKOVIAN TESTCO"],
        "entity_type": "organization",
        "countries": ["cy"],
        "programs": ["TEST-PROGRAM"],
        "source": "OFAC",
        "list_name": "OFAC SDN (TEST)",
    },
    {
        "list_entry_id": "OS-T-9002",
        "name": "Ivo Testarov Belinsky",
        "aliases": [],
        "entity_type": "person",
        "countries": ["ru"],
        "programs": [],
        "source": "OpenSanctions",
        "list_name": "OpenSanctions (TEST)",
    },
    {
        "list_entry_id": "OS-T-9003",
        "name": "Quillmarsh Example Shipping",
        "aliases": [],
        "entity_type": "organization",
        "countries": ["pa"],
        "programs": [],
        "source": "OpenSanctions",
        "list_name": "OpenSanctions (TEST)",
    },
]


def fake_snapshots() -> list[ListSnapshot]:
    """One obviously fake snapshot per source."""
    return [
        ListSnapshot(source="OFAC", file_name="sdn.csv", sha256="a" * 64, snapshot_date="2026-10-10"),
        ListSnapshot(source="OpenSanctions", file_name="targets.simple.csv", sha256="b" * 64),
    ]


def tiny_matcher(threshold: float = 0.85) -> SanctionsMatcher:
    """A rules-only matcher over WATCHLIST_RECORDS."""
    config = MatcherConfig(blocking=BlockingConfig(max_candidates=50), threshold=threshold)
    return SanctionsMatcher(watchlist_from_records(WATCHLIST_RECORDS, fake_snapshots()), config)


def build_guideline_retriever() -> Retriever:
    """The real guideline sections in an in-memory Qdrant with FAKE embedders and reranker."""
    chunks = load_guideline_chunks()
    client = QdrantClient(location=":memory:")
    dense = HashingDenseEmbedder(dim=128)
    sparse = HashingSparseEmbedder()
    create_collection(client, TEST_COLLECTION, dense.dim)
    index_chunks(client, TEST_COLLECTION, chunks, dense, sparse)
    return Retriever(client, TEST_COLLECTION, dense, sparse, TermOverlapReranker())


def risk_response(product: str = "credit", scenario: str = "baseline") -> dict[str, Any]:
    """A valid RiskScoreResponse body, as the risk service would send it (FAKE numbers)."""
    return {
        "product": product,
        "prediction": 0.083,
        "risk_band": "B",
        "reason_codes": [{"feature": "dscr", "contribution": -0.4, "description": "DSCR above the minimum"}],
        "model_version": "fake-risk-model-0.0.1",
        "scenario": scenario,
        "unknown_features": [],
    }


def json_handler(status: int, body: Any) -> Callable[[httpx.Request], httpx.Response]:
    """A MockTransport handler that always answers with this status and JSON body."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the fixed response."""
        return httpx.Response(status, json=body)

    return handler


def raising_handler(error_type: type[httpx.TransportError]) -> Callable[[httpx.Request], httpx.Response]:
    """A MockTransport handler that fails like the network would (connection refused, timeout, ...)."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Raise the transport error."""
        raise error_type("simulated network failure", request=request)

    return handler


def risk_api(handler: Callable[[httpx.Request], httpx.Response], url: str = RISK_URL) -> RiskApiClient:
    """A RiskApiClient for the service at base URL url, whose requests go to handler instead of the network."""
    return RiskApiClient(url, timeout_s=2.0, transport=httpx.MockTransport(handler))


def make_server(
    retriever: Retriever | None = None,
    matcher: SanctionsMatcher | None = None,
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
    store: GuidelineStore | None = None,
    retriever_factory: Callable[[], Retriever] | None = None,
) -> tuple[Any, ServerDependencies]:
    """A server wired to fakes (the risk service answers with a valid score unless handler says otherwise)."""
    factory = retriever_factory or (lambda: retriever)
    dependencies = ServerDependencies(
        retriever_factory=factory,
        matcher=matcher,
        risk_api=risk_api(handler or json_handler(200, risk_response())),
        guideline_store=store,
    )
    return build_server(dependencies), dependencies


def run_with_client(server: Any, scenario: Callable[[Client], Awaitable[T]], mode: str = "legacy") -> T:
    """Connect the SDK client in-process and return scenario(client).

    mode "legacy" sends real JSON-RPC messages over in-memory streams after the initialize handshake; "auto"
    uses the SDK's direct in-process dispatch for the 2026-07-28 protocol.
    """

    async def main() -> T:
        """Open the client, run the scenario, close the client."""
        async with Client(server, mode=mode) as client:
            return await scenario(client)

    return anyio.run(main)
