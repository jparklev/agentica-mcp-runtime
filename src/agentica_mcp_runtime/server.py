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
        "HELPERS auto-loaded into REPL (see ~/.local/share/agentica-runtime/CONTEXT.md):",
        "  time:        ago(**kw), now_rfc()",
        "  grafana:     prom(expr), loki(query), gf_uid(name)",
        "  clickhouse:  ch(sql), ch_describe(table), ch_tables(db)",
        "  slack:       slack_search(q), slack_thread(c, ts), slack_user(uid), slack_channel_id(hit)",
        "  notion:      notion_top(q), notion_text(id)",
        "  linear:      linear_team(), linear_my_issues()",
        "  dune:        dune_search(q)",
        "  pagerduty:   pd_oncall()",
        "  codex:       codex_each(prompts)        # parallel codex fan-out",
        "  disclosure:  cap, peek, run_many, results_log, results_cache_stats",
        "  warnings:    runtime_warnings(kind=...) for background events (eviction, token refresh, ...)",
        "  status:      runtime_status() — single triage summary (cache + warnings + oncall + Prom up)",
        "  slow tools:  slow_tool_log() — MCP calls that took >10s",
        "  journal:     journal_append(text, kind=...), journal_read(grep=...) — breadcrumbs across sessions",
        "  persistence: save(name, obj), load(name|sha), state_log()  # jj-backed, durable",
        "  serializer:  to_compact(obj) -> (text, fmt)  # picks shorter of JSON / TOON",
        "  tokens:      refresh_proxy_token()  # 401 recovery",
        "  discovery:   tools_for(server), tool_help(name), find_tool(intent)",
        "  judge:       await judge(answer, criteria, judge_model='gemini-3-flash-preview')",
        "  slack:       await slack_hit_thread(hit) goes search-hit -> full thread in one step",
        "  urls:        grafana_prom_url(expr), grafana_loki_url(query), grafana_dashboard_url(uid)",
        "",
        "PRESETS (greppable institutional memory at ~/.local/share/agentica-runtime/presets.md):",
        "  presets_index() lists everything; find_preset('incident') greps by intent;",
        "  preset_reload() hot-loads after you Edit presets.md (purges deleted ones).",
        "  Examples: find_topic('settler'), oncall_now(), ch_describe_meta_aggregator().",
        "  When a preset is stale (renamed metric, dropped column), Edit the relevant",
        "  ## section in presets.md, call preset_reload(). `git log presets.md` shows",
        "  how institutional knowledge has evolved. Grep `^## ` for catalog,",
        "  `UNVERIFIED` for cards needing attention.",
        "",
        "GOTCHAS:",
        "  - Short helper names (ch, pp, cap) shadow easily — don't reuse as loop vars.",
        "  - MCP tool returns are auto-parsed when whole-block JSON (no json.loads needed).",
        "  - Slack search hits use 'message_ts' (not 'ts'); use slack_channel_id(hit) for the channel ID.",
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
