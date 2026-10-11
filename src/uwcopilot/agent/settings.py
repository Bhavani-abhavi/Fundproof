"""Agent settings, read explicitly from the repo .env file and a few project variables (the .env file wins).

Only these names are read:
- UWC_RISK_API_URL / UWC_RISK_API_TIMEOUT_S: the risk scoring service, read by the shared loader in
  uwcopilot.risk.client so the agent and the MCP server give the variable the same meaning (the service's base
  URL; requests go to <base URL>/score). Unset means the score_risk tool is reported as unavailable; nothing is
  guessed.
- UWC_AGENT_DB: SQLite file for LangGraph checkpoints (default .cache/agent/checkpoints.sqlite).
- UWC_DECISION_LOG: append-only JSONL file of decision records (default data/processed/agent/decision_records.jsonl).
- UWC_OTLP_ENDPOINT: OTLP/HTTP traces endpoint, for example http://localhost:6006/v1/traces for a local Arize
  Phoenix. Unset means spans are created but not exported.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from uwcopilot.common.paths import CACHE_DIR, DATA_DIR, ENV_FILE
from uwcopilot.common.settings import env_reader, is_http_url
from uwcopilot.risk.client import RiskApiSettings, load_risk_api_settings

DEFAULT_CHECKPOINT_DB = CACHE_DIR / "agent" / "checkpoints.sqlite"
DEFAULT_DECISION_LOG = DATA_DIR / "processed" / "agent" / "decision_records.jsonl"


class AgentSettingsError(ValueError):
    """Raised when an agent setting has an invalid value."""


@dataclass(frozen=True)
class AgentSettings:
    """Where the agent keeps its state and which optional services it talks to."""

    risk_api: RiskApiSettings = field(default_factory=RiskApiSettings)
    checkpoint_db: Path = DEFAULT_CHECKPOINT_DB
    decision_log: Path = DEFAULT_DECISION_LOG
    otlp_endpoint: str | None = None


def validate_service_url(name: str, url: str) -> str:
    """Require an absolute http(s) URL with a host; trailing slashes are removed."""
    cleaned = url.strip().rstrip("/")
    if not is_http_url(cleaned):
        raise AgentSettingsError(f"{name} must be an http(s) URL (got {url!r}).")
    return cleaned


def load_agent_settings(env_file: Path | None = ENV_FILE, environ: Mapping[str, str] | None = None) -> AgentSettings:
    """Build AgentSettings from the .env file and the process environment. Pass env_file=None to skip .env."""
    read = env_reader(env_file, environ)
    otlp = read("UWC_OTLP_ENDPOINT")
    checkpoint = read("UWC_AGENT_DB")
    decision_log = read("UWC_DECISION_LOG")
    return AgentSettings(
        risk_api=load_risk_api_settings(env_file, environ),
        checkpoint_db=Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT_DB,
        decision_log=Path(decision_log) if decision_log else DEFAULT_DECISION_LOG,
        otlp_endpoint=validate_service_url("UWC_OTLP_ENDPOINT", otlp) if otlp else None,
    )
