"""Settings for the MCP server. Its only setting of its own is where the risk scoring service is.

That setting is shared with the agent, so it is read by one loader in uwcopilot.risk.client and has one meaning:

- UWC_RISK_API_URL: the risk service's BASE URL, for example http://localhost:8000. Requests go to
  <base URL>/score. Unset means not configured: score_application returns the structured error
  risk_api_not_configured and nothing is sent anywhere.
- UWC_RISK_API_TIMEOUT_S: seconds to wait for it (default 10, at most 120).

The repo .env file wins over the process environment. The index location comes from the RAG settings
(UWC_QDRANT_*), and the screening threshold from results/kyc_metrics.json, so neither is repeated here.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from uwcopilot.common.paths import ENV_FILE
from uwcopilot.risk.client import RiskApiSettings, load_risk_api_settings


@dataclass(frozen=True)
class McpServerSettings:
    """The risk service settings. The URL may hold credentials, so errors only show risk.client.public_endpoint()."""

    risk_api: RiskApiSettings = field(default_factory=RiskApiSettings)


def load_mcp_settings(env_file: Path | None = ENV_FILE, environ: Mapping[str, str] | None = None) -> McpServerSettings:
    """Build McpServerSettings from the .env file and the process environment (the .env file wins).

    Pass env_file=None to ignore the .env file (tests do this).
    """
    return McpServerSettings(risk_api=load_risk_api_settings(env_file, environ))
