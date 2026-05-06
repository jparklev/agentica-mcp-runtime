"""CLI entry point: python -m agentica_mcp_runtime --config <json-or-path>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastmcp.mcp_config import MCPConfig

from agentica_mcp_runtime.server import run_server


def _parse_configs(value: str) -> dict[str, MCPConfig]:
    """Parse a config value as inline JSON or a file path.

    JSON format: {"server_name": {"command": "...", "args": [...]}}
    """
    # Try inline JSON first
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        # Fall back to file path
        path = Path(value)
        if not path.exists():
            print(f"[agentica-mcp-runtime] Config file not found: {value}", file=sys.stderr)
            sys.exit(1)
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"[agentica-mcp-runtime] Invalid JSON in {value}: {e}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(raw, dict):
        print("[agentica-mcp-runtime] Config must be a JSON object", file=sys.stderr)
        sys.exit(1)

    configs: dict[str, MCPConfig] = {}
    for name, server_cfg in raw.items():
        try:
            configs[name] = MCPConfig.from_dict({name: server_cfg})
        except Exception as e:
            print(f"[agentica-mcp-runtime] Skipping server '{name}': {e}", file=sys.stderr)

    return configs


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="agentica-mcp-runtime",
        description="Run the Agentica MCP Runtime server.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="MCP server configs as inline JSON or path to a JSON file (optional - empty if not provided)",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    args = parser.parse_args()

    configs = _parse_configs(args.config) if args.config else {}
    run_server(configs, transport=args.transport)


if __name__ == "__main__":
    main()
