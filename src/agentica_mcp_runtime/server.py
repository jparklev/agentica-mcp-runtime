"""MCP server — single `python` tool for programmatic MCP tool use."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.mcp_config import MCPConfig
from fastmcp.tools.tool import Tool

from agentica_mcp_runtime.tool_loader import load_tools
from agentica_mcp_runtime.sandbox import SandboxSession, generate_stubs


_session = SandboxSession()
_configs: dict[str, MCPConfig] = {}


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Discover MCP tools at startup, register `python` with stubs in its description."""
    tools = {}
    stubs = ""
    if _configs:
        tools = await load_tools(_configs)

    _session.start(tools)
    if tools:
        stubs = generate_stubs(tools)
        print(f"[agentica-mcp-runtime] Loaded {len(tools)} tool(s)", file=sys.stderr)
    else:
        print("[agentica-mcp-runtime] No MCP tools discovered — sandbox REPL is still available", file=sys.stderr)

    description = _build_execute_description(stubs)
    tool = Tool.from_function(fn=_execute_impl, name="python", description=description)
    server.add_tool(tool)

    yield {}

    _session.stop()


MAX_OUTPUT_CHARS = 4000


def _build_execute_description(stubs: str) -> str:
    parts = [
        "Stateful Python REPL with MCP tools available as async functions.",
        "All variables, imports, and definitions persist across calls.",
        "Use `await` to call tools and `print()` to surface results.",
        "",
        "CRITICAL: Minimize the number of python() calls. Each call costs a full API roundtrip.",
        "Do as much as possible in a SINGLE call: fetch data, process it, and print the answer.",
        "Use asyncio.gather() to run independent tool calls in parallel:",
        "  users, tables = await asyncio.gather(get_users(), list_tables())",
        "",
        "IMPORTANT: Keep output concise to save context.",
        "Store results in variables and process them in Python.",
        "Only print() final summaries or specific fields — never raw API responses.",
    ]
    if stubs:
        parts.append("")
        parts.append("Available tools in the REPL:")
        parts.append("")
        parts.append(stubs)
    return "\n".join(parts)


async def _execute_impl(code: str) -> str:
    result = await _session.execute(code)

    parts: list[str] = []

    if result.output:
        parts.append(result.output)

    if result.result_repr:
        parts.append(result.result_repr)

    if result.error:
        parts.append(f"ERROR ({result.exception_name}): {result.error}")

    if result.added_vars:
        parts.append(f"New variables: {', '.join(result.added_vars)}")

    if not parts:
        parts.append("(no output)")

    parts.append(f"[{result.duration:.3f}s]")

    output = "\n".join(parts)

    if len(output) > MAX_OUTPUT_CHARS:
        truncated = output[:MAX_OUTPUT_CHARS]
        remaining = len(output) - MAX_OUTPUT_CHARS
        truncated += f"\n\n... truncated ({remaining} chars). Store data in variables and print only what you need."
        return truncated

    return output


def create_server(configs: dict[str, MCPConfig], name: str = "agentica-mcp-runtime") -> FastMCP:
    """Create a FastMCP server pre-configured with the given MCP tool configs.

    Args:
        configs: Dict of {server_name: MCPConfig} to discover tools from.
        name: Name for the FastMCP server instance.

    Returns:
        A FastMCP instance ready to .run().
    """
    global _configs
    _configs = configs
    return FastMCP(name, lifespan=lifespan)


def run_server(configs: dict[str, MCPConfig], transport: str = "stdio") -> None:
    """Create and run the MCP server.

    Args:
        configs: Dict of {server_name: MCPConfig} to discover tools from.
        transport: Transport protocol (default: "stdio").
    """
    server = create_server(configs)
    server.run(transport=transport)
