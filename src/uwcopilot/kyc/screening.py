"""screen(request) -> ScreeningResult: the sanctions-screening tool used by the agent and the MCP server.

The tool registry (uwcopilot.tools.registry) calls screen_entity(), which screens with a default matcher:
- the full watchlist from data/raw (OFAC SDN + OpenSanctions), loaded and indexed on first use (about a
  minute and a couple of GB of memory at full size), then reused for every later call in the process;
- the operating point chosen on the validation split by scripts/eval_kyc.py and stored in
  results/kyc_metrics.json under "selected_operating_point" (threshold, and the reranker weight; a weight of
  0 means rules only, and then no embedding model is loaded).

If that file is missing, screening refuses to run (ScreeningNotConfigured) instead of guessing a threshold.
Tests and demos can install a ready-made matcher with set_default_matcher() or pass one to screen().

Every result carries the list snapshots (file checksums) it was screened against. A "potential_match" result
always requires human review; nothing here can clear or confirm a match.
"""

import json
import threading
from pathlib import Path

from uwcopilot.common.paths import RESULTS_DIR
from uwcopilot.kyc.matcher import MatcherConfig, SanctionsMatcher
from uwcopilot.kyc.watchlist import load_watchlist
from uwcopilot.tools.schemas import ScreeningRequest, ScreeningResult

SELECTED_CONFIG_FILE = RESULTS_DIR / "kyc_metrics.json"

_default_matcher: SanctionsMatcher | None = None
_default_matcher_lock = threading.Lock()


class ScreeningNotConfigured(RuntimeError):
    """Raised when no validated threshold exists yet (run `make eval-kyc` first)."""


def load_selected_operating_point(path: Path = SELECTED_CONFIG_FILE) -> dict:
    """Read the threshold and reranker weight chosen on the validation split."""
    if not path.exists():
        raise ScreeningNotConfigured(f"{path} not found. Run `make kyc-data` and `make eval-kyc` first.")
    point = json.loads(path.read_text(encoding="utf-8")).get("selected_operating_point")
    if not point or "threshold" not in point:
        raise ScreeningNotConfigured(f"{path} has no selected_operating_point. Run `make eval-kyc` again.")
    return point


def config_from_operating_point(point: dict) -> MatcherConfig:
    """MatcherConfig with the validated threshold and reranker weight (other settings stay at their defaults)."""
    return MatcherConfig(threshold=float(point["threshold"]), reranker_weight=float(point.get("reranker_weight", 0.0)))


def build_default_matcher(config_file: Path = SELECTED_CONFIG_FILE) -> SanctionsMatcher:
    """Load the full watchlist and the validated settings; load the embedding model only if it is used."""
    config = config_from_operating_point(load_selected_operating_point(config_file))
    embedder = None
    if config.reranker_weight > 0:
        from uwcopilot.kyc.reranker import FastEmbedEmbedder  # loads onnxruntime only when needed

        embedder = FastEmbedEmbedder()
    return SanctionsMatcher(load_watchlist(), config, embedder)


def set_default_matcher(matcher: SanctionsMatcher | None) -> None:
    """Install the matcher that screen_entity() uses (None resets it, so the next call rebuilds it)."""
    global _default_matcher
    with _default_matcher_lock:
        _default_matcher = matcher


def default_matcher() -> SanctionsMatcher:
    """Return the shared matcher, building it once on first use."""
    global _default_matcher
    with _default_matcher_lock:
        if _default_matcher is None:
            _default_matcher = build_default_matcher()
        return _default_matcher


def screen(request: ScreeningRequest | dict, matcher: SanctionsMatcher | None = None) -> ScreeningResult:
    """Screen one name. A dict is validated as a ScreeningRequest first (unknown fields are rejected)."""
    validated = request if isinstance(request, ScreeningRequest) else ScreeningRequest.model_validate(request)
    return (matcher or default_matcher()).screen(validated)


def screen_entity(request: ScreeningRequest) -> ScreeningResult:
    """Handler for the allow-listed "screen_entity" tool (see uwcopilot.tools.registry)."""
    return screen(request)
