"""Fixtures shared by the MCP server tests (all offline; see mcp_test_helpers.py).

Do not add an __init__.py to this folder: pytest puts tests/ on sys.path, and a regular package named "mcp" here
would shadow the MCP SDK. (Because of the folder name, ruff sorts `mcp` imports with the first-party ones in these
files only.)
"""

import pytest
from mcp_test_helpers import build_guideline_retriever, tiny_matcher

from uwcopilot.kyc.matcher import SanctionsMatcher
from uwcopilot.mcp_server.guideline_store import GuidelineStore
from uwcopilot.rag.retriever import Retriever


@pytest.fixture(scope="session")
def guideline_retriever() -> Retriever:
    """The guideline sections in an in-memory Qdrant with fake embedders (built once per test session)."""
    return build_guideline_retriever()


@pytest.fixture(scope="session")
def matcher() -> SanctionsMatcher:
    """A rules-only matcher over three invented watchlist entries."""
    return tiny_matcher()


@pytest.fixture(scope="session")
def store() -> GuidelineStore:
    """The SYNTHETIC guideline documents from corpus/guidelines/."""
    return GuidelineStore()
