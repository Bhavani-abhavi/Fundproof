"""MCP settings, the command line, the Makefile target, and the offline parts of scripts/mcp_smoke.py."""

from pathlib import Path

import pytest
from pydantic import ValidationError

import mcp_smoke
from uwcopilot.common.paths import REPO_ROOT
from uwcopilot.mcp_server import cli
from uwcopilot.mcp_server.server import validation_summary
from uwcopilot.mcp_server.settings import load_mcp_settings
from uwcopilot.risk.client import DEFAULT_TIMEOUT_S, RiskSettingsError, public_endpoint
from uwcopilot.tools.schemas import ScreeningRequest

# ---------------------------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------------------------


def test_defaults_without_env() -> None:
    """No variables: the risk service is not configured (no localhost default) and the default timeout."""
    settings = load_mcp_settings(env_file=None, environ={})
    assert settings.risk_api.base_url is None
    assert settings.risk_api.score_url is None
    assert settings.risk_api.timeout_s == DEFAULT_TIMEOUT_S


def test_values_from_the_environment() -> None:
    """UWC_RISK_API_URL (a base URL) and UWC_RISK_API_TIMEOUT_S are read."""
    environ = {"UWC_RISK_API_URL": " https://risk.example.test/ ", "UWC_RISK_API_TIMEOUT_S": "2.5"}
    settings = load_mcp_settings(env_file=None, environ=environ)
    assert settings.risk_api.base_url == "https://risk.example.test"
    assert settings.risk_api.score_url == "https://risk.example.test/score"
    assert settings.risk_api.timeout_s == 2.5


def test_env_file_wins_over_the_environment(tmp_path: Path) -> None:
    """The repo .env file has priority, like the other project settings."""
    env_file = tmp_path / ".env"
    env_file.write_text("UWC_RISK_API_URL=http://from-file:8000\n", encoding="utf-8")
    settings = load_mcp_settings(env_file=env_file, environ={"UWC_RISK_API_URL": "http://from-env:8000"})
    assert settings.risk_api.base_url == "http://from-file:8000"


def test_other_variables_are_ignored() -> None:
    """Only the UWC_ names count (for example ANTHROPIC_BASE_URL does not change anything)."""
    environ = {"ANTHROPIC_BASE_URL": "http://elsewhere", "RISK_API_URL": "http://elsewhere"}
    assert load_mcp_settings(env_file=None, environ=environ).risk_api.base_url is None


@pytest.mark.parametrize(
    "environ",
    [
        {"UWC_RISK_API_URL": "localhost:8000"},
        {"UWC_RISK_API_URL": "ftp://risk"},
        {"UWC_RISK_API_URL": "http:///score"},
        {"UWC_RISK_API_URL": "http://risk:8000/score"},
        {"UWC_RISK_API_URL": "http://risk:8000?token=1"},
        {"UWC_RISK_API_TIMEOUT_S": "soon"},
        {"UWC_RISK_API_TIMEOUT_S": "0"},
        {"UWC_RISK_API_TIMEOUT_S": "500"},
    ],
)
def test_invalid_values_are_refused(environ: dict[str, str]) -> None:
    """Bad URLs (including a full /score endpoint) and bad timeouts fail at startup with a clear error."""
    with pytest.raises(RiskSettingsError):
        load_mcp_settings(env_file=None, environ=environ)


@pytest.mark.parametrize(
    ("url", "shown"),
    [
        ("http://user:pass@risk:8000/score?key=1#x", "http://risk:8000/score"),
        ("https://risk.example.test/v1/score", "https://risk.example.test/v1/score"),
        ("http://[::1]:8000/score", "http://[::1]:8000/score"),
    ],
)
def test_public_endpoint_drops_credentials_and_query(url: str, shown: str) -> None:
    """The endpoint shown in errors never carries user info, query strings or fragments."""
    assert public_endpoint(url) == shown


def test_validation_summary_leaves_out_rejected_values() -> None:
    """Error text for a bad screening request names the field, not the (possibly personal) value."""
    with pytest.raises(ValidationError) as raised:
        ScreeningRequest(name="Jane Q Example" * 30, entity_type="person")
    summary = validation_summary(raised.value)
    assert summary.startswith("name:")
    assert "Jane" not in summary


# ---------------------------------------------------------------------------------------------------------------
# Command line and Makefile
# ---------------------------------------------------------------------------------------------------------------


def test_cli_defaults_to_stdio() -> None:
    """No options: stdio, and HTTP defaults that only listen on loopback."""
    args = cli.parse_args([])
    assert args.transport == "stdio"
    assert args.host == "127.0.0.1"
    assert args.port == cli.DEFAULT_HTTP_PORT
    assert args.path == "/mcp"


@pytest.mark.parametrize(("host", "loopback"), [("127.0.0.1", True), ("localhost", True), ("::1", True)])
def test_loopback_hosts(host: str, loopback: bool) -> None:
    """Loopback addresses are recognised."""
    assert cli.is_loopback(host) is loopback


@pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "example.com", "::"])
def test_non_loopback_http_host_is_refused_without_the_flag(host: str) -> None:
    """The HTTP transport has no authentication, so remote binding needs --unsafe-allow-remote."""
    with pytest.raises(cli.UnsafeHostError):
        cli.check_http_host(host, allow_remote=False)
    cli.check_http_host(host, allow_remote=True)


def test_main_refuses_remote_host_before_starting(capsys: pytest.CaptureFixture[str]) -> None:
    """main() returns 2 and starts nothing for a remote host without the flag."""
    assert cli.main(["--transport", "streamable-http", "--host", "0.0.0.0"]) == 2
    assert "no authentication" in capsys.readouterr().err


def test_makefile_mcp_target_runs_the_server_without_echo() -> None:
    """`make mcp` must not echo its command: stdout is the MCP stdio channel."""
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    lines = makefile.splitlines()
    target = lines.index(next(line for line in lines if line.startswith("mcp:")))
    recipe = lines[target + 1]
    assert recipe.startswith("\t@")
    assert "python -m uwcopilot.mcp_server" in recipe
    assert any(line.startswith("mcp-smoke:") for line in lines)


# ---------------------------------------------------------------------------------------------------------------
# scripts/mcp_smoke.py (offline helpers; the real run is the integration test)
# ---------------------------------------------------------------------------------------------------------------


def test_smoke_check_log() -> None:
    """all_passed needs at least one check and no failures."""
    log = mcp_smoke.CheckLog()
    assert not log.all_passed
    assert log.record("one", True)
    assert log.all_passed
    assert not log.record("two", False, "why")
    assert not log.all_passed
    assert log.checks[1] == {"name": "two", "passed": False, "detail": "why"}


def test_smoke_free_port_has_no_listener() -> None:
    """The port used for the risk API URL has nothing listening on it."""
    port = mcp_smoke.free_local_port()
    assert 0 < port < 65536
    assert not mcp_smoke.port_is_open(port)


def test_smoke_server_environment_is_minimal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The child process gets the SDK's safe defaults plus the risk URL, not the parent's other variables."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://platform-only")
    monkeypatch.setenv("UWC_SOMETHING_ELSE", "x")
    env = mcp_smoke.server_environment("http://127.0.0.1:9")
    assert env["UWC_RISK_API_URL"] == "http://127.0.0.1:9"
    assert "ANTHROPIC_BASE_URL" not in env
    assert "UWC_SOMETHING_ELSE" not in env


def test_smoke_parse_args_defaults() -> None:
    """stdio, k=5, hybrid_rerank by default."""
    args = mcp_smoke.parse_args([])
    assert (args.transport, args.k, args.mode) == ("stdio", 5, "hybrid_rerank")


def test_smoke_hit_summary_keeps_ids_not_text() -> None:
    """The results file keeps rank, id, citation, document and score of each hit, not the text."""
    payload = {
        "hits": [
            {
                "rank": 1,
                "section_id": "NBC-4.2",
                "citation": "[NBC-4.2]",
                "source_doc": "northwind_business_credit",
                "score": 1.5,
                "title": "t",
                "text": "long text",
            }
        ]
    }
    assert mcp_smoke.hit_summary(payload) == [
        {
            "rank": 1,
            "section_id": "NBC-4.2",
            "citation": "[NBC-4.2]",
            "source_doc": "northwind_business_credit",
            "score": 1.5,
        }
    ]
