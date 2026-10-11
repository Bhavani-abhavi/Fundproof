"""The real stdio transport: the SDK client starts `python -m uwcopilot.mcp_server` as a subprocess.

Offline: only calls that need neither the Qdrant index nor any model (listing, a guideline section, a resource,
refusals). The search over the real index is covered by the integration test (scripts/mcp_smoke.py).
"""

import sys
from typing import Any

import anyio

from mcp import Client, StdioServerParameters
from uwcopilot.common.paths import REPO_ROOT

STARTUP_AND_CALLS_TIMEOUT_S = 120


def test_stdio_server_lists_tools_and_serves_sections() -> None:
    """Start the server over stdio, list it, read a section two ways, and check two refusals."""
    params = StdioServerParameters(command=sys.executable, args=["-m", "uwcopilot.mcp_server"], cwd=str(REPO_ROOT))

    async def main() -> dict[str, Any]:
        """One stdio session."""
        with anyio.fail_after(STARTUP_AND_CALLS_TIMEOUT_S):
            async with Client(params) as client:
                tools = sorted(tool.name for tool in (await client.list_tools()).tools)
                section = await client.call_tool("get_guideline_section", {"section_id": "NBC-2.2"})
                resource = await client.read_resource("guidelines://NBC/NBC-2.2")
                unknown = await client.call_tool("approve_application", {})
                extra = await client.call_tool("get_guideline_section", {"section_id": "NBC-2.2", "x": 1})
                return {
                    "tools": tools,
                    "section": section,
                    "resource_text": resource.contents[0].text,
                    "unknown": unknown,
                    "extra": extra,
                    "server_name": client.server_info.name if client.server_info else None,
                }

    outcome = anyio.run(main)
    assert outcome["server_name"] == "uwcopilot-underwriting"
    assert outcome["tools"] == [
        "get_guideline_section",
        "score_application",
        "screen_sanctions",
        "search_underwriting_guidelines",
    ]
    assert not outcome["section"].is_error
    assert outcome["section"].structured_content["citation"] == "[NBC-2.2]"
    assert outcome["section"].structured_content["text"] in outcome["resource_text"]
    assert outcome["unknown"].is_error
    assert outcome["extra"].is_error
