"""Discover MCP servers and create MCPFunction instances for their tools."""

from __future__ import annotations

from agentica.unmcp import MCPFunction
from fastmcp.mcp_config import MCPConfig


async def load_tools(configs: dict[str, MCPConfig]) -> dict[str, MCPFunction]:
    """Connect to each MCP server and create MCPFunction wrappers for all tools.

    Args:
        configs: Dict of {server_name: MCPConfig} from config reader.

    Returns:
        Dict of {tool_name: MCPFunction} ready for REPL injection.
    """
    tools: dict[str, MCPFunction] = {}

    for server_name, config in configs.items():
        try:
            functions = await MCPFunction.from_mcp_config(config)
            for fn in functions:
                tools[fn.__name__] = fn
        except Exception as e:
            # Log but continue — one broken server shouldn't block others
            print(f"[agentica-mcp-runtime] Failed to load tools from '{server_name}': {e}")

    return tools
