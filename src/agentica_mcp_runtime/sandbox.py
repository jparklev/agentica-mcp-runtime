"""Sandbox session: BaseRepl with injected MCP tool functions."""

from __future__ import annotations

import asyncio
import functools
import json
from collections.abc import Callable
from dataclasses import dataclass

from agentica.unmcp import MCPFunction
from agentica_internal.repl.repl import BaseRepl
from fastmcp.client import Client


@dataclass
class ExecutionResult:
    """Simplified view of REPL execution results."""

    output: str
    result_repr: str | None
    error: str | None
    exception_name: str | None
    added_vars: tuple[str, ...]
    changed_vars: tuple[str, ...]
    duration: float


class SandboxSession:
    """A sandbox session wrapping a BaseRepl with MCP tool functions."""

    def __init__(self) -> None:
        self._repl: BaseRepl | None = None
        self._tools: dict[str, MCPFunction] = {}
        self._has_executed: bool = False

    @property
    def is_active(self) -> bool:
        return self._repl is not None

    @property
    def has_executed(self) -> bool:
        return self._has_executed

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def start(self, tools: dict[str, MCPFunction]) -> None:
        """Initialize the REPL with MCP tool functions as globals."""
        self._tools = tools
        self._repl = BaseRepl()

        # Set the event loop so async MCPFunction calls work
        loop = asyncio.get_running_loop()
        self._repl.set_loop(loop)

        # Wrap MCPFunctions to handle text content (not just structuredContent)
        wrapped = {name: _wrap_mcp_function(fn) for name, fn in tools.items()}

        # Pre-inject asyncio + json so agent doesn't need to import them
        globals_dict = dict(wrapped)
        globals_dict["asyncio"] = asyncio
        globals_dict["json"] = json

        self._repl.initialize(
            local_vars=None,
            global_vars=globals_dict,
            hidden_vars=(),
        )

    async def execute(self, code: str) -> ExecutionResult:
        """Execute Python code in the REPL.

        Code can call MCP tool functions as regular async functions using await.
        Variables persist across executions.
        """
        if self._repl is None:
            raise RuntimeError("Sandbox not initialized — no MCP tools were discovered at startup.")

        self._has_executed = True
        info = await self._repl.async_run_code_info(code)

        error_str = None
        if info.has_error:
            error_str = info.traceback_str or info.exception_name or "Unknown error"

        return ExecutionResult(
            output=info.output,
            result_repr=info.out_str,
            error=error_str,
            exception_name=info.exception_name,
            added_vars=info.added_locals,
            changed_vars=info.changed_locals,
            duration=info.duration,
        )

    def stop(self) -> None:
        """Tear down the REPL session."""
        if self._repl is not None:
            self._repl.reset()
            self._repl = None
        self._tools = {}


def _wrap_mcp_function(fn: MCPFunction) -> Callable:
    """Wrap an MCPFunction to properly extract text content from MCP responses.

    MCPFunction only reads structuredContent, but most MCP servers return
    plain text content blocks. This wrapper handles both.
    """

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        # Build argument dict the same way MCPFunction does.
        # NOTE: accessing name-mangled private attrs — fragile if MCPFunction internals change.
        args_dict = dict(kwargs)
        input_schema = fn._MCPFunction__tool_inputSchema
        if input_schema is not None and "properties" in input_schema:
            missing_keys = [
                k for k in input_schema["properties"].keys() if k not in args_dict
            ]
            for i, key in enumerate(missing_keys):
                if i >= len(args):
                    break
                args_dict[key] = args[i]

        async with Client(fn._MCPFunction__mcp_config) as client:
            result = await client.session.call_tool(fn.__name__, args_dict)
            if result.isError:
                raise RuntimeError(";".join(tc.text for tc in result.content))

            # Try structuredContent first
            structured = getattr(result, "structuredContent", None)
            if isinstance(structured, dict):
                if "result" in structured:
                    return structured["result"]
                if structured:
                    return structured

            # Fall back to text content
            if result.content:
                texts = [tc.text for tc in result.content if hasattr(tc, "text")]
                if len(texts) == 1:
                    return texts[0]
                if texts:
                    return "\n".join(texts)

            return None

    # Preserve signature for stub generation and REPL introspection
    wrapper.__signature__ = fn.__signature__
    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def generate_stubs(tools: dict[str, MCPFunction]) -> str:
    """Generate Python function stubs for all tools."""
    stubs: list[str] = []
    for fn in tools.values():
        stubs.append(_stub(fn))
    return "\n\n".join(stubs)


def _stub(fn: MCPFunction) -> str:
    """Generate a Python stub for one MCPFunction."""
    sig = fn.__signature__
    doc = fn.__doc__ or ""
    lines = [f"async def {fn.__name__}{sig}:"]
    if doc:
        # Indent docstring lines
        doc_lines = doc.strip().splitlines()
        if len(doc_lines) == 1:
            lines.append(f'    """{doc_lines[0]}"""')
        else:
            lines.append('    """')
            for dl in doc_lines:
                lines.append(f"    {dl}" if dl.strip() else "")
            lines.append('    """')
    lines.append("    ...")
    return "\n".join(lines)
