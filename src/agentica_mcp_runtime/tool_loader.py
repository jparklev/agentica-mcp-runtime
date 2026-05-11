"""Discover MCP servers and create MCPFunction instances for their tools."""

from __future__ import annotations

import asyncio

from agentica.unmcp import MCPFunction
from fastmcp.mcp_config import MCPConfig


async def load_tools(configs: dict[str, MCPConfig]) -> dict[str, MCPFunction]:
    """Connect to each MCP server and create MCPFunction wrappers for all tools.

    Connections fan out in parallel via `asyncio.gather`. With ~10 cloud +
    stdio servers and ~1-3s per handshake (TLS + MCP protocol + tools/list),
    serial wins this 20-30s; the parallel version is gated by the single
    slowest server (~3-5s typical). Each Client gets its own HTTPS / stdio
    pipe so there's no shared state to fight over.

    Per-server failures are isolated: a broken server logs and returns []
    for its slot. We never raise out of the gather — that would cancel
    every other in-flight handshake and tank startup over one bad server.

    Args:
        configs: Dict of {server_name: MCPConfig} from config reader.

    Returns:
        Dict of {tool_name: MCPFunction} ready for REPL injection. Result
        order matches `configs.items()` iteration (gather preserves input
        order regardless of completion order), so tool-name collisions
        resolve the same way the serial version did.
    """

    async def _load_one(server_name: str, config: MCPConfig) -> list[MCPFunction]:
        try:
            return await MCPFunction.from_mcp_config(config)
        except Exception as e:
            # Match the serial version's log shape so existing log scrapers
            # / oncall runbooks keep working.
            print(f"[agentica-mcp-runtime] Failed to load tools from '{server_name}': {e}")
            return []

    results = await asyncio.gather(
        *(_load_one(name, cfg) for name, cfg in configs.items())
    )

    tools: dict[str, MCPFunction] = {}
    for fns in results:
        for fn in fns:
            tools[fn.__name__] = fn
    return tools
