"""MCP server — single `python` tool for programmatic MCP tool use."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.mcp_config import MCPConfig
from fastmcp.tools.tool import Tool

from agentica_mcp_runtime.tool_loader import load_tools
from agentica_mcp_runtime.sandbox import SandboxSession, close_persistent_clients, generate_stubs


_session = SandboxSession()
_configs: dict[str, MCPConfig] = {}


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Discover MCP tools at startup, register `python` with a server catalog
    in its description (not full stubs — see _build_execute_description)."""
    tools = {}
    if _configs:
        tools = await load_tools(_configs)

    _session.start(tools)
    if tools:
        print(f"[agentica-mcp-runtime] Loaded {len(tools)} tool(s)", file=sys.stderr)
    else:
        print("[agentica-mcp-runtime] No MCP tools discovered — sandbox REPL is still available", file=sys.stderr)

    description = _build_execute_description(tools)
    tool = Tool.from_function(fn=_execute_impl, name="python", description=description)
    server.add_tool(tool)

    # Concurrent-safe twin of `python`. Spawns a fresh subprocess per call so
    # multiple sub-agents fanning out don't queue on the persistent REPL.
    # See `isolated_runner.py` for the bootstrap details. The description here
    # is intentionally short — sub-agents brief themselves on it, so we want
    # to keep the tool surface minimal.
    isolated_tool = Tool.from_function(
        fn=_execute_isolated_impl,
        name="python_isolated",
        description=_ISOLATED_DESCRIPTION,
    )
    server.add_tool(isolated_tool)

    yield {}

    _session.stop()
    await close_persistent_clients()


MAX_OUTPUT_CHARS = 4000


def _build_execute_description(tools) -> str:
    """Render a lean server catalog instead of dumping full tool stubs.

    Empirically, dumping signatures + docstrings for ~200 wrapped MCP tools
    costs ~37k tokens of static description. Most agents touch 1-3 servers
    per session. Showing only the catalog (server name → count + sample
    tool names) saves ~95% of those tokens at startup; agents drill down
    on demand via `tools_for("slack")` and `tool_help("notion_search")`.
    """
    parts = [
        "Stateful Python REPL with MCP tools available as async functions.",
        "All variables, imports, and definitions persist across calls.",
        "Use `await` to call tools and `print()` to surface results.",
        "",
        "PRESENTATION: For non-trivial code (>5 lines), prefer the two-step",
        "pattern: first write the source to /tmp/agent-py-<short-name>.py using",
        "your host's file-write tool (Claude Code: Write; Codex: write/apply_patch),",
        "then call this tool with code_file=<that path>. Hosts render file-write",
        "diffs with syntax highlighting; the python call itself stays a clean",
        "one-liner referencing the file. For trivial snippets (<=5 lines),",
        "pass `code` inline.",
        "",
        "CRITICAL: Minimize the number of python() calls. Each call costs a full API roundtrip.",
        "Do as much as possible in a SINGLE call: fetch data, process it, and print the answer.",
        "Use asyncio.gather() to run independent tool calls in parallel:",
        "  users, tables = await asyncio.gather(get_users(), list_tables())",
        "",
        "IMPORTANT: Keep output concise to save context.",
        "Store results in variables and process them in Python.",
        "Only print() final summaries or specific fields — never raw API responses.",
        "",
        "HELPERS auto-loaded into the REPL. Discover via:",
        "  list_servers()                       — wrapped MCP server catalog",
        "  find_tool('intent')                  — substring search across all ~200 tools",
        "  tools_for('slack', fmt='dict')       — stubs for one server",
        "  tool_help('notion_search')           — full signature + docstring",
        "  find_preset()  / find_preset('rfq')  — preset catalog / by intent",
        "  Domain helpers (full list via the above): prom, loki, ch, slack_search,",
        "  notion_top, linear_my_issues, dune_search, pd_oncall, codex_each, ask_gemini,",
        "  judge (cross-model), slack_hit_thread, grafana_*_url.",
        "",
        "DISCLOSURE & STATE (the load-bearing primitives):",
        "  cap(obj, max_chars=...)              — wraps big returns w/ structural summary + result_id",
        "  peek(result_id, slice='chars:0:1k')  — pull full or windowed cached value",
        "  run_many({'a': call_a(), ...})       — asyncio.gather + cap on each return",
        "  save(name, obj) / load(name|sha)     — jj-backed durable state across restarts",
        "  save(name, obj, persist=False)       — ephemeral, same lookup API",
        "  runtime_status()                     — one-call triage (cache + warnings + oncall + Prom up)",
        "  runtime_warnings(kind=...)           — bounded event log (evictions, token refresh, ...)",
        "  slow_tool_log() / journal_*          — slow-call ring buffer / cross-session breadcrumbs",
        "",
        "PRESETS at ~/.local/share/agentica-runtime/presets.md (agent-editable institutional",
        "memory). Use find_preset() to browse, preset_reload() after editing the .md (purges",
        "deleted cards). `git log presets.md` shows how knowledge evolved. Grep `^## ` for",
        "catalog and `UNVERIFIED` for cards needing attention.",
        "",
        "GOTCHAS:",
        "  - Short helper names (ch, pp, cap) shadow easily — don't reuse as loop vars.",
        "  - MCP tool returns are auto-parsed when whole-block JSON (no json.loads needed).",
        "  - Slack search hits use 'message_ts' (not 'ts'); use slack_channel_id(hit) for the channel ID.",
        "  - On MCP errors, look for `HINT: ...` appended to the message — actionable fixes.",
    ]
    if tools:
        from collections import defaultdict
        from agentica.unmcp.sigs import sanitize_param_name as _sanitize
        by_server: dict[str, list[str]] = defaultdict(list)
        for fn in tools.values():
            cfg = getattr(fn, "_MCPFunction__mcp_config", None)
            srv_map = getattr(cfg, "mcpServers", None) if cfg is not None else None
            srv = next(iter(srv_map.keys())) if srv_map else "?"
            by_server[srv].append(_sanitize(fn.__name__))
        parts.extend([
            "",
            f"WRAPPED MCP SERVERS ({sum(len(v) for v in by_server.values())} tools across {len(by_server)} servers):",
        ])
        for srv in sorted(by_server, key=lambda s: -len(by_server[s])):
            names = sorted(by_server[srv])
            # Show all tools for small servers (≤8) — saves a tools_for()
            # drill-down hop. Cap at 5+more for big servers to keep the
            # catalog block compact.
            if len(names) <= 8:
                sample, more = ", ".join(names), ""
            else:
                sample = ", ".join(names[:5])
                more = f", … (+{len(names) - 5})"
            parts.append(f"  {srv:<14} ({len(names):>2})   {sample}{more}")
        parts.extend([
            "",
            "Drill down inside python(): tools_for('slack') for one server's stubs;",
            "tool_help('notion_search') for one tool's full signature + docstring.",
        ])
    return "\n".join(parts)


async def _execute_impl(code: str = "", code_file: str = "") -> str:
    """Execute Python in the persistent REPL.

    Pass `code` for inline source, OR `code_file` for an absolute path to a
    .py file we should read and exec. The code_file path lets the agent emit
    a Write(...) tool call first (which Claude Code renders with full syntax
    highlighting in the diff view) and then invoke this tool with just the
    path — keeping the gnarly inline-string view of `code` from cluttering
    the transcript when sources are non-trivial.
    """
    if code and code_file:
        return "ERROR: pass either `code` (inline source) or `code_file` (path), not both."
    if code_file and not code:
        try:
            with open(code_file) as _f:
                code = _f.read()
        except Exception as e:
            return f"ERROR reading code_file={code_file!r}: {type(e).__name__}: {e}"
        # Auto-delete the file if it looks like an ephemeral temp artifact —
        # /tmp, /var/folders, or /private/var/folders (macOS resolves the
        # latter from the former via a system symlink, so a path can appear
        # in either form depending on how it was created). Project files
        # outside those roots are left alone.
        TMP_PREFIXES = ("/tmp/", "/var/folders/", "/private/var/folders/")
        if code_file.endswith(".py") and code_file.startswith(TMP_PREFIXES):
            try:
                import os as _os
                _os.unlink(code_file)
            except OSError:
                pass
    if not code:
        return "ERROR: pass either `code` (inline source) or `code_file` (absolute path)."

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


# ---------------------------------------------------------------------------
# `python_isolated`: concurrent-safe twin of `python`.
#
# Spawns a fresh subprocess per call. No shared REPL state across calls — use
# artifact_save / artifact_get to bridge data. ~3-5s cold startup per call;
# trades that for true concurrency under multi-caller fanout.
# ---------------------------------------------------------------------------
_ISOLATED_TIMEOUT_SEC = 180

_ISOLATED_DESCRIPTION = (
    "Concurrent-safe Python execution in a fresh subprocess per call.\n\n"
    "Use this from Claude Code sub-agents that share the parent's MCP "
    "runtime — the stateful `python` tool serializes everything through "
    "one REPL, so N concurrent sub-agent calls queue and may time out as "
    "`-32000 Connection closed`. This tool side-steps that by spawning "
    "an isolated subprocess for each invocation; N concurrent calls fan "
    "out across N subprocesses.\n\n"
    "Trade-offs:\n"
    "  - ~3-5s startup per call (cold MCP Client opens).\n"
    "  - NO persistent variable state across calls. Use `artifact_save` "
    "and `artifact_get` (SQLite-backed, multi-writer safe) to share state.\n"
    "  - `subagent_namespace()` + `subagent_corpus()` are the recommended "
    "convention: write `artifact_save(f\"{ns}/<topic>\", obj)` from each "
    "sub-agent, and the parent reads them all back via `subagent_corpus(ns)`.\n\n"
    "Helpers (helpers.py + presets.md) and all wrapped MCP tools are loaded "
    "fresh on each call — same surface as `python`, just no persistence."
)


async def _execute_isolated_impl(code: str = "", code_file: str = "") -> str:
    """Spawn an isolated subprocess to execute `code` (or read it from
    `code_file`). Returns combined stdout/stderr, capped at MAX_OUTPUT_CHARS.

    See module-level `_ISOLATED_DESCRIPTION` for the design rationale.
    """
    if code and code_file:
        return "ERROR: pass either `code` (inline) or `code_file` (path), not both."
    if not code and not code_file:
        return "ERROR: pass either `code` (inline) or `code_file` (path)."

    cleanup_path: str | None = None
    if code and not code_file:
        # Materialize inline code to a temp file so the subprocess has a
        # consistent path-based contract with isolated_runner.
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="agentica-isolated-", delete=False,
        ) as tf:
            tf.write(code)
            code_file = tf.name
            cleanup_path = code_file

    import asyncio as _asyncio_local
    import time as _time_local

    t0 = _time_local.monotonic()
    try:
        proc = await _asyncio_local.create_subprocess_exec(
            sys.executable, "-m", "agentica_mcp_runtime.isolated_runner", code_file,
            stdout=_asyncio_local.subprocess.PIPE,
            stderr=_asyncio_local.subprocess.PIPE,
        )
        try:
            stdout, stderr = await _asyncio_local.wait_for(
                proc.communicate(), timeout=_ISOLATED_TIMEOUT_SEC,
            )
        except _asyncio_local.TimeoutError:
            proc.kill()
            await proc.wait()
            return (
                f"ERROR: python_isolated timed out after {_ISOLATED_TIMEOUT_SEC}s. "
                "Long-running work belongs in the persistent `python` tool."
            )
    finally:
        # Defensive cleanup. The subprocess's isolated_runner auto-deletes
        # /tmp ephemerals on read; this handles the case where the subprocess
        # crashed before reading the file.
        if cleanup_path:
            import os as _os_local
            try:
                _os_local.unlink(cleanup_path)
            except OSError:
                pass

    elapsed = _time_local.monotonic() - t0
    out = stdout.decode("utf-8", errors="replace")
    err = stderr.decode("utf-8", errors="replace")

    parts: list[str] = []
    if out.strip():
        parts.append(out.rstrip())
    if err.strip():
        parts.append(f"[STDERR]\n{err.rstrip()}")
    if proc.returncode != 0:
        parts.append(f"[exit code: {proc.returncode}]")
    parts.append(f"[isolated; {elapsed:.2f}s]")

    output = "\n".join(parts) if parts else "(no output)"
    if len(output) > MAX_OUTPUT_CHARS:
        truncated = output[:MAX_OUTPUT_CHARS]
        remaining = len(output) - MAX_OUTPUT_CHARS
        truncated += f"\n\n... truncated ({remaining} chars)."
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
