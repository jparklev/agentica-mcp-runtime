<p align="center">
  <img src="agentica-sdk.svg" alt="Agentica SDK" height="50" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="mcp.png" alt="MCP" height="50" />
</p>

# Agentica MCP Runtime

Host-agnostic MCP sandbox runtime — execute MCP tools as Python functions in a sandboxed REPL.

Any agent framework can use this to give its agent a single `python` tool that has every MCP tool pre-loaded as an async function. The agent writes Python code to call them — with loops, conditionals, `asyncio.gather()`, and data pipelines — all in one tool call.

## Quickstart

### As an MCP server (`.mcp.json`)

To use this runtime as an MCP server in a host that supports `.mcp.json` (e.g. Claude Code),
create a `.mcp.json` at your project root:

```json
{
  "mcpServers": {
    "agentica-runtime": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "agentica_mcp_runtime",
      ]
    }
  }
}
```

### As a library

```python
from agentica_mcp_runtime import create_server

configs = {
    "slack": MCPConfig.from_dict({"slack": {"command": "npx", "args": ["-y", "@anthropic/slack-mcp"]}}),
    "postgres": MCPConfig.from_dict({"postgres": {"command": "npx", "args": ["-y", "@anthropic/postgres-mcp"]}}),
}

mcp = create_server(configs, name="my-agent")
mcp.run(transport="stdio")
```

### As a CLI

```bash
python -m agentica_mcp_runtime --config '{"slack": {"command": "npx", "args": ["-y", "@anthropic/slack-mcp"]}}'
```

Or point to a JSON file:

```bash
python -m agentica_mcp_runtime --config servers.json
```

## How it works

When the server starts, it:

1. Connects to each configured MCP server and discovers their tools
2. Creates callable async Python functions for each tool
3. Exposes a single `python(code)` tool backed by a stateful REPL with all tools pre-loaded

The agent sees one tool — `python` — with full function stubs (signatures + docstrings) for every discovered MCP tool in its description. It writes Python code to call them.

## Tool

### `python(code: str) -> str`

Stateful Python REPL with all discovered MCP tools available as async functions. Variables, imports, and definitions persist across calls.

```python
# Parallel fetch from multiple sources
users, tables = await asyncio.gather(
    slack_get_users(),
    list_tables()
)

# Process in Python, print only the summary
engineers = [u for u in users if u['dept'] == 'Engineering']
print(f"{len(engineers)} engineers across {len(tables)} tables")
```

`asyncio` and `json` are pre-imported.

## API

### `create_server(configs, name="agentica-mcp-runtime") -> FastMCP`

Create a FastMCP server with the given MCP tool configs. Returns a server instance you can `.run()`.

### `run_server(configs, transport="stdio")`

Create and immediately run the server. Convenience wrapper around `create_server`.

### `load_tools(configs) -> dict[str, MCPFunction]`

Connect to MCP servers and return tool functions — useful if you want the tools without the server.

### `SandboxSession`

Low-level REPL session. Call `.start(tools)` to initialize, `.execute(code)` to run code, `.stop()` to tear down.

## Architecture

```
Your Agent Framework
    ↓ (MCP protocol, stdio)
agentica-mcp-runtime (this repo)
    ├── Tool Loader → connects to MCP servers, discovers tools
    ├── MCPFunction factory → callable wrappers per tool
    ├── Sandbox REPL → stateful Python REPL with injected tool functions
    └── FastMCP server → exposes `python` tool
```

## Dependencies

- [agentica-internal](https://github.com/symbolica-ai/agentica-internal) — BaseRepl
- [symbolica-agentica](https://github.com/symbolica-ai/agentica-python-sdk) — MCPFunction
- [fastmcp](https://github.com/jlowin/fastmcp) — MCP server + client

## License

MIT — see [LICENSE](LICENSE).
