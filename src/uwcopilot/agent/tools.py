"""The real tool handlers the agent uses outside tests.

- search_guidelines: the RAG retriever over Qdrant (uwcopilot.rag.search), default mode hybrid_rerank, or another
  mode for faster batch evaluation;
- screen_entity: the sanctions matcher over the downloaded OFAC and OpenSanctions lists (uwcopilot.kyc.screening);
- score_risk: the shared risk service client (uwcopilot.risk.client). When UWC_RISK_API_URL is not set it sends
  nothing and reports the tool as unavailable.

The retriever (embedded Qdrant) and the matcher are shared objects that were not written for concurrent use, so
each of those handlers is guarded by its own lock: calls to different tools still run in parallel, calls to the
same one run one at a time. Heavy modules are imported on first call, not at import time.
"""

import threading
from collections.abc import Callable
from typing import Any

from uwcopilot.agent.settings import AgentSettings
from uwcopilot.risk.client import RiskApiClient
from uwcopilot.tools.registry import ToolHandler


def locked(handler: Callable[[Any], Any]) -> ToolHandler:
    """Wrap a handler so only one thread runs it at a time."""
    lock = threading.Lock()

    def run(arguments: Any) -> Any:
        """Call the handler while holding its lock."""
        with lock:
            return handler(arguments)

    return run


class GuidelineSearchTool:
    """The search_guidelines handler. It builds its own retriever on first use and can close it again.

    Closing matters for embedded Qdrant: its storage folder can be opened by one client at a time, and a client
    left open until interpreter shutdown prints an error while Python is tearing down.
    """

    def __init__(self, mode: str | None = None) -> None:
        """mode None keeps the RAG default (hybrid_rerank)."""
        self.mode = mode
        self._retriever: Any | None = None
        self._lock = threading.Lock()

    def __call__(self, query: Any) -> Any:
        """Run one guideline search (one at a time)."""
        from uwcopilot.rag.search import DEFAULT_MODE, build_retriever, search_guidelines

        with self._lock:
            if self._retriever is None:
                self._retriever = build_retriever()
            return search_guidelines(query, retriever=self._retriever, mode=self.mode or DEFAULT_MODE)  # type: ignore[arg-type]

    def close(self) -> None:
        """Close the retriever's Qdrant client if it was opened."""
        with self._lock:
            if self._retriever is not None:
                self._retriever.client.close()
                self._retriever = None


def guideline_search_handler(mode: str | None = None) -> GuidelineSearchTool:
    """The RAG search handler; mode None keeps the RAG default (hybrid_rerank)."""
    return GuidelineSearchTool(mode)


def close_tool_handlers(handlers: dict[str, ToolHandler]) -> None:
    """Release resources held by handlers that have a close() method."""
    for handler in handlers.values():
        close = getattr(handler, "close", None)
        if callable(close):
            close()


def screening_handler() -> ToolHandler:
    """The sanctions screening handler with the full default matcher (loaded on first call)."""

    def screen(request: Any) -> Any:
        """Screen one name with the shared matcher."""
        from uwcopilot.kyc.screening import screen_entity

        return screen_entity(request)

    return locked(screen)


def default_tool_handlers(settings: AgentSettings, retrieval_mode: str | None = None) -> dict[str, ToolHandler]:
    """Handlers for every allowed tool, configured from settings."""
    return {
        "search_guidelines": guideline_search_handler(retrieval_mode),
        "screen_entity": screening_handler(),
        "score_risk": RiskApiClient.from_settings(settings.risk_api).score,
    }
