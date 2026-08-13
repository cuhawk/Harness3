#!/usr/bin/env python3
"""MCP server entry point - exposes harness_agent's tools over stdio to any MCP client."""
from __future__ import annotations

from harness_agent.mcp_server import serve_stdio

if __name__ == "__main__":
    try:
        serve_stdio()
    except KeyboardInterrupt:
        pass
