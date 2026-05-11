"""Sandbox session: BaseRepl with injected MCP tool functions."""

from __future__ import annotations

import asyncio
import functools
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentica.unmcp import MCPFunction
from agentica.unmcp.sigs import sanitize_param_name
from agentica_internal.repl.repl import BaseRepl
from fastmcp.client import Client


# Module-level pool of persistent MCP clients, keyed by id(mcp_config).
# Without this, every wrapped tool call spawns a fresh stdio subprocess (or a
# fresh HTTP session), which breaks any server that holds in-memory state
# across calls — most notably codex's conversation threads.
#
# Each client's async context lifecycle is managed directly (not via a shared
# AsyncExitStack) so `_evict_persistent_client` can tear down ONE client
# mid-life without disturbing the others. The shutdown path
# (`close_persistent_clients`) iterates the cache and closes everything that
# survived.
_persistent_clients: dict[int, Client] = {}
_open_lock: asyncio.Lock | None = None

# Slow-tool capture: every MCP call whose wall time exceeds the threshold
# gets logged here. helpers.py's slow_tool_log() reads this ring buffer
# so agents can spot bogged-down tools without instrumenting their code.
_MCP_SLOW_THRESHOLD_SEC = 10.0
_MCP_SLOW_LOG_MAX = 100
_MCP_SLOW_LOG: list[dict] = []


# Hint dispatch: reads ~/.local/share/agentica-runtime/hints.md (mtime-cached
# so agent edits go live without a restart). When the file is missing, falls
# back to the baked-in list below — small and high-confidence to cover bare-
# bones installs. The .md format is greppable cards:
#
#   ## name
#   **triggers**:
#   - substring 1
#   - substring 2
#   **hint**: actionable one-liner.
#
# All triggers must appear (case-insensitively) in the lowercased error
# message for a card to match. First match wins.
import os as _hints_os
import re as _hints_re

_HINTS_FALLBACK: list[tuple[list[str], str]] = [
    (["missing a required argument: 'filters'"],
     "Pass `filters={}` if you don't need filtering — the schema requires the kwarg."),
    (["stepseconds must be provided", "range"],
     "Range queries need `stepSeconds` (typically 60). The prom() helper supplies this automatically."),
    (["code: 47", "db::exception"],
     "Column doesn't exist. Run `ch_describe(table)` to see the real schema."),
    (["unknown expression identifier"],
     "Column doesn't exist. Run `ch_describe(table)` to see the real schema."),
    (["session not found for thread_id"],
     "codex thread state doesn't survive runtime restarts. codex_reply only works within the same runtime spawn."),
    (["mcp_unauthorized_no_token"],
     "MCP server isn't authenticated in the local proxy. Confirm it appears in `claude mcp list` as connected."),
    (["rate limit"], "Rate-limited or quota exhausted. Back off, or check `await getUsage()` for Dune/Context7."),
]

_HINTS_PATH = _hints_os.path.expanduser("~/.local/share/agentica-runtime/hints.md")
_HINTS_CACHE: dict = {"mtime": 0.0, "rules": _HINTS_FALLBACK, "source": "fallback"}


def _parse_hints_md(text: str) -> list[tuple[list[str], str]]:
    """Parse hints.md cards into (triggers, hint) pairs."""
    # Strip fenced code blocks so format examples in docs don't get parsed as
    # real rules (the doc preamble has a ```…``` example whose `## name`
    # would otherwise show up alongside the real cards).
    text = _hints_re.sub(r"```.*?```", "", text, flags=_hints_re.DOTALL)
    rules: list[tuple[list[str], str]] = []
    for section in _hints_re.split(r"\n## ", text)[1:]:
        triggers: list[str] = []
        hint = ""
        # `[ \t]*` (not `\s*`) keeps the same-line inline group from spilling
        # onto the next line; `[-*][ \t]+` requires real list-bullet syntax so
        # the following `**hint**: …` line doesn't get scooped up as a trigger.
        m = _hints_re.search(r"\*\*triggers\*\*:?[ \t]*([^\n]*)((?:\n[ \t]*[-*][ \t]+[^\n]+)*)", section)
        if m:
            inline = m.group(1).strip()
            if inline:
                triggers.append(inline.strip("`"))
            for ln in (m.group(2) or "").splitlines():
                t = ln.strip().lstrip("-*").strip().strip("`")
                if t:
                    triggers.append(t)
        h = _hints_re.search(r"\*\*hint\*\*:?\s*(.+?)(?=\n\n|\n##|\Z)", section, _hints_re.DOTALL)
        if h:
            hint = h.group(1).strip()
        if triggers and hint:
            rules.append((triggers, hint))
    return rules


def _load_hints() -> list[tuple[list[str], str]]:
    """Return the active hint list. Reloads from disk on mtime change."""
    if not _hints_os.path.exists(_HINTS_PATH):
        if _HINTS_CACHE["source"] != "fallback":
            _HINTS_CACHE.update(mtime=0.0, rules=_HINTS_FALLBACK, source="fallback")
        return _HINTS_FALLBACK
    try:
        mtime = _hints_os.path.getmtime(_HINTS_PATH)
    except OSError:
        return _HINTS_CACHE["rules"]
    if mtime != _HINTS_CACHE["mtime"]:
        try:
            text = open(_HINTS_PATH).read()
            parsed = _parse_hints_md(text)
            _HINTS_CACHE.update(mtime=mtime, rules=(parsed or _HINTS_FALLBACK),
                                source=("hints.md" if parsed else "fallback"))
        except Exception:
            pass
    return _HINTS_CACHE["rules"]


def _hint_for_mcp_error(msg: str, tool_name: str) -> str | None:
    """Map known MCP error signatures to actionable hints. Data lives in
    ~/.local/share/agentica-runtime/hints.md (agent-editable, hot-reloaded
    via mtime). Returns None if no rule matches."""
    m = msg.lower()
    for triggers, hint in _load_hints():
        if all(t.lower() in m for t in triggers):
            return hint
    return None


def _maybe_parse_json(text):
    """If `text` is a whole-block JSON object/array, return parsed; else string.

    Most MCP tools emit their payload as a JSON-encoded string in a text content
    block. Auto-parsing here removes the `json.loads(x) if isinstance(x, str)
    else x` ritual every caller would otherwise need to do. Conservative — only
    triggers on leading {/[ to avoid mis-parsing prose that incidentally
    contains JSON-like fragments.
    """
    if not isinstance(text, str):
        return text
    s = text.strip()
    if not (s.startswith("{") or s.startswith("[")):
        return text
    try:
        parsed = json.loads(s)
        if isinstance(parsed, (dict, list)):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return text


# Pre-open hooks: app-level callbacks invoked once before each cold Client
# open. Each receives the mcp_config and may mutate it in place. The
# canonical use is re-reading a rotated OAuth bearer from a system keychain
# (kept out of this module so the fork stays generic). Failures in a hook
# are swallowed + recorded so a broken hook can't deadlock the pool — a
# stale-bearer cold open will surface as a 401 on the next call, which
# triggers eviction, which re-fires the hook with the fixed callback.
_PRE_OPEN_HOOKS: list[Callable[[Any], None]] = []


def register_pre_open_hook(fn: Callable[[Any], None]) -> None:
    """Register a callback to run before each cold MCP `Client` open.

    The callback receives the `mcp_config` and may mutate it in place
    (typically to refresh a rotated auth header). Use case in our
    deployment: helpers.py registers a hook that re-reads the Claude
    Code OAuth bearer from the macOS keychain and patches the
    Authorization header on cloud-MCP entries. With this in place the
    "any tool error → evict cached Client → next call cold-opens" flow
    transparently picks up rotated tokens — replacing what used to be
    an agent-visible `refresh_proxy_token()` verb.
    """
    _PRE_OPEN_HOOKS.append(fn)


async def _get_persistent_client(mcp_config) -> Client:
    """Return an open Client for this mcp_config, opening one if needed."""
    global _open_lock
    cid = id(mcp_config)
    cached = _persistent_clients.get(cid)
    if cached is not None:
        return cached
    if _open_lock is None:
        _open_lock = asyncio.Lock()
    async with _open_lock:
        cached = _persistent_clients.get(cid)
        if cached is not None:
            return cached
        # Fire pre-open hooks BEFORE constructing the Client — they may
        # mutate `mcp_config.mcpServers[*].headers` (rotated bearer etc.)
        # which fastmcp reads at Client construction time.
        for _hook in _PRE_OPEN_HOOKS:
            try:
                _hook(mcp_config)
            except Exception as e:
                import sys as _sys
                print(
                    f"[agentica-mcp-runtime] pre-open hook {_hook.__name__} failed: "
                    f"{type(e).__name__}: {e}",
                    file=_sys.stderr,
                )
        client = Client(mcp_config)
        await client.__aenter__()
        _persistent_clients[cid] = client
        return client


async def _evict_persistent_client(mcp_config) -> None:
    """Drop the cached Client for `mcp_config`.

    Called from `_wrap_mcp_function` whenever a tool call fails — either by
    raising at the transport layer or by returning `isError=True` from the
    server. The next call rebuilds the Client, which:

      * re-runs every registered pre-open hook (so a rotated keychain bearer
        gets patched into the headers — no agent-visible refresh verb), and
      * starts a fresh stdio subprocess / HTTP session if the previous one
        was wedged.

    Best-effort close: failures during teardown are swallowed because the
    eviction itself must always succeed — leaving a stale Client cached
    would defeat the point.
    """
    cid = id(mcp_config)
    client = _persistent_clients.pop(cid, None)
    if client is None:
        return
    try:
        await client.__aexit__(None, None, None)
    except Exception:
        pass


async def close_persistent_clients() -> None:
    """Close all open persistent clients. Called from the server lifespan teardown."""
    clients = list(_persistent_clients.values())
    _persistent_clients.clear()
    for client in clients:
        try:
            await client.__aexit__(None, None, None)
        except Exception:
            pass


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

        # Wrap MCPFunctions to handle text content (not just structuredContent).
        # Key by the wrapper's sanitized python name so hyphenated MCP tools
        # (e.g. "codex-reply") are reachable in the REPL as snake_case identifiers.
        # Also build a tool->server lookup so the helpers can do progressive
        # disclosure: tools_for("slack") and tool_help("notion_search").
        wrapped: dict[str, Callable] = {}
        tool_servers: dict[str, str] = {}
        for fn in tools.values():
            w = _wrap_mcp_function(fn)
            name = w.__name__
            # Detect name collisions after sanitization: e.g. an MCP server
            # exposing both `foo-bar` and `foo_bar` would otherwise silently
            # have one wrapper overwrite the other in REPL globals. Suffix
            # the second one and warn so the agent can see what happened.
            if name in wrapped:
                import sys as _sys2
                suffix = 2
                while f"{name}_{suffix}" in wrapped:
                    suffix += 1
                new_name = f"{name}_{suffix}"
                print(
                    f"[agentica-mcp-runtime] tool-name collision: {fn.__name__!r} -> {new_name!r}",
                    file=_sys2.stderr,
                )
                w.__name__ = new_name
                w.__qualname__ = new_name
                name = new_name
            wrapped[name] = w
            cfg = getattr(fn, "_MCPFunction__mcp_config", None)
            srv_map = getattr(cfg, "mcpServers", None) if cfg is not None else None
            if srv_map:
                # MCPConfig.mcpServers is a single-entry dict (one server per
                # config), so the only key is the server name.
                tool_servers[name] = next(iter(srv_map.keys()))

        # Pre-inject asyncio + json so agent doesn't need to import them
        globals_dict = dict(wrapped)
        globals_dict["asyncio"] = asyncio
        globals_dict["json"] = json
        globals_dict["_AGENTICA_TOOL_SERVERS"] = tool_servers

        # Auto-load user helpers + presets, so they're always in REPL globals.
        # Helpers runs as Python; presets live as markdown (greppable, agent-
        # editable cards) and load via extract-python-blocks-then-exec.
        # Missing files are tolerated; load errors go to stderr.
        import os as _os, sys as _sys, re as _re_local
        # 1) helpers.py — vanilla Python module
        _helpers_path = _os.path.expanduser("~/.local/share/agentica-runtime/helpers.py")
        if _os.path.exists(_helpers_path):
            try:
                with open(_helpers_path) as _f:
                    _src = _f.read()
                exec(compile(_src, _helpers_path, "exec"), globals_dict)
            except Exception as _e:
                print(f"[agentica-mcp-runtime] helpers.py load failed: {_e}", file=_sys.stderr)
        # 2) presets.md — extract ```python blocks and exec each. Falls back
        # silently to presets.py if only the legacy file exists.
        _presets_md = _os.path.expanduser("~/.local/share/agentica-runtime/presets.md")
        _presets_py = _os.path.expanduser("~/.local/share/agentica-runtime/presets.py")
        if _os.path.exists(_presets_md):
            try:
                _md = open(_presets_md).read()
                for _src in _re_local.findall(r"```python[^\n]*\r?\n(.*?)```", _md, _re_local.DOTALL):
                    try:
                        exec(_src, globals_dict)
                    except Exception as _e:
                        print(f"[agentica-mcp-runtime] presets.md block load: {_e}", file=_sys.stderr)
            except Exception as _e:
                print(f"[agentica-mcp-runtime] presets.md load failed: {_e}", file=_sys.stderr)
        elif _os.path.exists(_presets_py):
            try:
                with open(_presets_py) as _f:
                    exec(compile(_f.read(), _presets_py, "exec"), globals_dict)
            except Exception as _e:
                print(f"[agentica-mcp-runtime] presets.py load failed: {_e}", file=_sys.stderr)

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

    py_to_mcp = getattr(fn, "_MCPFunction__py_to_mcp_name", {})
    sig = fn.__signature__

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        # Use the Python signature to bind args. signature.bind() raises the
        # same TypeErrors a regular call would ("multiple values for x",
        # "missing required y", "unexpected keyword z"). Avoids the silent
        # misrouting our previous manual positional fill could produce.
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError as e:
            # Run the same hint dispatch we use for server-side errors —
            # signature.bind() catches things like "missing argument: filters"
            # locally before the call reaches the MCP server, but the agent
            # still benefits from the same actionable hint.
            base = f"{fn.__name__}: {e}"
            hint = _hint_for_mcp_error(base, fn.__name__)
            raise TypeError(f"{base}\n\nHINT: {hint}" if hint else base) from None
        args_dict: dict = {py_to_mcp.get(k, k): v for k, v in bound.arguments.items()}

        mcp_config = fn._MCPFunction__mcp_config
        client = await _get_persistent_client(mcp_config)
        _t0 = time.time()
        # Evict-on-failure: any error from this tool call (transport raise OR
        # server-reported isError) drops the cached Client so the next call
        # re-opens fresh. That's how we let a rotated keychain bearer take
        # effect without the agent ever calling a refresh verb — the auth
        # resolution path runs at Client open time, so a new Client picks up
        # whatever's currently in the keychain.
        try:
            result = await client.session.call_tool(fn.__name__, args_dict)
        except Exception:
            await _evict_persistent_client(mcp_config)
            raise
        _elapsed = time.time() - _t0
        # Slow-tool capture: ring buffer of calls over the threshold so
        # slow_tool_log() / runtime_status() can surface bogged-down tools.
        if _elapsed > _MCP_SLOW_THRESHOLD_SEC:
            _MCP_SLOW_LOG.append({
                "tool": fn.__name__,
                "elapsed_sec": round(_elapsed, 2),
                "args_preview": str(args_dict)[:140],
                "ts": _t0,
            })
            if len(_MCP_SLOW_LOG) > _MCP_SLOW_LOG_MAX:
                _MCP_SLOW_LOG.pop(0)
        if result.isError:
            msg = ";".join(getattr(tc, "text", repr(tc)) for tc in result.content)
            await _evict_persistent_client(mcp_config)
            hint = _hint_for_mcp_error(msg, fn.__name__)
            raise RuntimeError(f"{msg}\n\nHINT: {hint}" if hint else msg)

        # Try structuredContent first
        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            if "result" in structured:
                return structured["result"]
            if structured:
                return structured

        # Fall back to text content; auto-parse whole-block JSON.
        if result.content:
            texts = [tc.text for tc in result.content if hasattr(tc, "text")]
            if len(texts) == 1:
                return _maybe_parse_json(texts[0])
            if texts:
                return [_maybe_parse_json(t) for t in texts]

        return None

    # Preserve signature for stub generation and REPL introspection. Use a
    # python-safe identifier for the function name so the agent can call it
    # (e.g. "codex-reply" -> "codex_reply"). The original name is still used
    # internally for the MCP call_tool RPC via fn.__name__.
    py_name = sanitize_param_name(fn.__name__)
    wrapper.__signature__ = fn.__signature__
    wrapper.__name__ = py_name
    wrapper.__qualname__ = py_name
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
    py_name = sanitize_param_name(fn.__name__)
    lines = [f"async def {py_name}{sig}:"]
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
