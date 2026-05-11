"""Subprocess entry point for `python_isolated`.

The persistent runtime exposes one stateful `python(code)` tool whose REPL is
a singleton serial executor (`SandboxSession` wraps one `BaseRepl`). When
multiple callers (typically Claude Code sub-agents that share the parent's
MCP servers) issue `python(...)` concurrently, they queue inside the
runtime's request handler. Under load this looks like a transport wedge —
all callers see `-32000 Connection closed` while the runtime PID stays alive
at 0% CPU.

`python_isolated` works around the constraint by spawning a fresh Python
subprocess per call. Each subprocess bootstraps the same REPL surface
(helpers.py + presets.md + MCP tool wrappers from servers.json) and execs
the caller's code in isolation. Concurrent calls fan out across multiple
subprocesses, so N sub-agents can do real MCP work in parallel.

Trade-offs vs the persistent `python`:
- Cold ~3-5s startup per call (MCP Client opens).
- No persistent variable state across calls — use `artifact_save` /
  `artifact_get` to bridge data across subprocesses.
- Slightly higher resource footprint (Client opens per call, not amortized).
"""

from __future__ import annotations

import asyncio
import ast
import inspect
import json
import os
import re
import sys
import traceback
from contextlib import suppress


def _load_helpers_into(globals_dict: dict) -> None:
    """Auto-load helpers.py + presets.md code blocks into the given globals.

    Mirrors SandboxSession.start's helpers/presets loading exactly so the
    isolated subprocess sees the same global namespace the persistent REPL
    sees. Failures go to stderr but the bootstrap continues — a broken
    helper shouldn't block all isolated work.
    """
    helpers_path = os.path.expanduser("~/.local/share/agentica-runtime/helpers.py")
    if os.path.exists(helpers_path):
        try:
            with open(helpers_path) as f:
                exec(compile(f.read(), helpers_path, "exec"), globals_dict)
        except Exception as e:
            print(f"[isolated_runner] helpers.py load failed: {e}", file=sys.stderr)

    presets_path = os.path.expanduser("~/.local/share/agentica-runtime/presets.md")
    if os.path.exists(presets_path):
        try:
            md = open(presets_path).read()
            for src in re.findall(r"```python[^\n]*\r?\n(.*?)```", md, re.DOTALL):
                with suppress(Exception):
                    exec(src, globals_dict)
        except Exception as e:
            print(f"[isolated_runner] presets.md load failed: {e}", file=sys.stderr)


async def _bootstrap_globals() -> dict:
    """Build the REPL globals dict the user code will exec into.

    Order matters: helpers.py first so the pre-open keychain hook is
    registered before `load_tools` opens any MCP Clients. Otherwise the
    initial tools/list calls go out with a stale bearer (still usually
    works for low-volume reads, but failure mode is silent).
    """
    from agentica_mcp_runtime.sandbox import _wrap_mcp_function
    from agentica_mcp_runtime.tool_loader import load_tools
    from fastmcp.mcp_config import MCPConfig

    servers_path = os.environ.get(
        "AGENTICA_SERVERS_CONFIG",
        os.path.expanduser("~/.local/share/agentica-runtime/servers.json"),
    )

    globals_dict: dict = {"asyncio": asyncio, "json": json}

    # Pre-register hooks (helpers.py installs the bearer refresher) BEFORE
    # opening any Clients.
    _load_helpers_into(globals_dict)

    if not os.path.exists(servers_path):
        print(
            f"[isolated_runner] no servers.json at {servers_path}; MCP tools unavailable",
            file=sys.stderr,
        )
        return globals_dict

    try:
        with open(servers_path) as f:
            servers_data = json.load(f)
        configs: dict[str, MCPConfig] = {
            name: MCPConfig.model_validate({"mcpServers": {name: spec}})
            for name, spec in servers_data.items()
        }
        tools = await load_tools(configs)
        for fn in tools.values():
            w = _wrap_mcp_function(fn)
            globals_dict[w.__name__] = w
    except Exception as e:
        print(f"[isolated_runner] load_tools failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    # Re-load helpers + presets now that wrapped tools are in globals — any
    # helper that closes over a tool name (e.g. linear_issues references
    # `list_issues`) picks up the binding on this second pass.
    _load_helpers_into(globals_dict)

    return globals_dict


async def _exec_user_code(code: str, globals_dict: dict) -> int:
    """Execute user code with top-level-await support.

    Uses `ast.PyCF_ALLOW_TOP_LEVEL_AWAIT` so `await foo()` at the top level
    of the user's code works without wrapping. If the compiled code object
    is a coroutine factory (has CO_COROUTINE flag), we await it; otherwise
    a regular exec runs the code.
    """
    try:
        tree = ast.parse(code, mode="exec")
        compiled = compile(
            tree, "<isolated>", "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
    except SyntaxError:
        traceback.print_exc(file=sys.stderr)
        return 1

    try:
        if compiled.co_flags & inspect.CO_COROUTINE:
            # Top-level await present — eval returns a coroutine.
            coro = eval(compiled, globals_dict)
            await coro
        else:
            exec(compiled, globals_dict)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


async def _main_async(code_file: str) -> int:
    if not os.path.exists(code_file):
        print(f"[isolated_runner] code_file not found: {code_file}", file=sys.stderr)
        return 2

    with open(code_file) as f:
        user_code = f.read()

    # Auto-delete /tmp ephemerals so the caller doesn't accumulate junk.
    TMP_PREFIXES = ("/tmp/", "/var/folders/", "/private/var/folders/")
    if code_file.endswith(".py") and code_file.startswith(TMP_PREFIXES):
        with suppress(OSError):
            os.unlink(code_file)

    globals_dict = await _bootstrap_globals()
    return await _exec_user_code(user_code, globals_dict)


def main():
    if len(sys.argv) != 2:
        print(
            "usage: python -m agentica_mcp_runtime.isolated_runner <code_file>",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(asyncio.run(_main_async(sys.argv[1])))


if __name__ == "__main__":
    main()
