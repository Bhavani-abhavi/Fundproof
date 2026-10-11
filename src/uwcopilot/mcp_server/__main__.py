"""`python -m uwcopilot.mcp_server`: run the MCP server (stdio by default; see cli.py for options)."""

import sys

from uwcopilot.mcp_server.cli import main

if __name__ == "__main__":
    sys.exit(main())
