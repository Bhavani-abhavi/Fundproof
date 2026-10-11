"""MCP server that exposes the underwriting tools to any MCP client (docs/mcp.md).

Modules:
    server.py           build_server(): four read-only tools, guideline resources, the underwriting_review prompt
    guideline_store.py  the SYNTHETIC guideline documents: whole documents, sections, resource URIs
    models.py           MCP-only output models (GuidelineSection, ScoreApplicationResult)
    prompts.py          text of the underwriting_review prompt template
    settings.py         the risk service settings (UWC_RISK_API_URL is a base URL; loader shared with the agent)

The risk scoring client itself (uwcopilot.risk.client) is shared with the agent and the tool registry, so
UWC_RISK_API_URL means the same thing to every caller.
    cli.py              `python -m uwcopilot.mcp_server [--transport stdio|streamable-http]`

Every MCP tool runs under a tool on the registry allow-list (uwcopilot.tools.registry). No tool approves, declines
or clears anything.
"""
