"""Shared fixtures for agentica-mcp-runtime tests."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


@pytest.fixture
def mock_mcp_function():
    """Factory fixture to create fake MCPFunction-like objects."""

    def _make(
        name: str = "my_tool",
        doc: str | None = "A tool description.",
        params: list[inspect.Parameter] | None = None,
        input_schema: dict | None = None,
        mcp_config=None,
    ):
        if params is None:
            params = [
                inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int),
                inspect.Parameter("y", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str, default="hello"),
            ]
        sig = inspect.Signature(params)
        fn = SimpleNamespace(
            __name__=name,
            __qualname__=name,
            __doc__=doc,
            __signature__=sig,
            __module__="test",
            _MCPFunction__tool_inputSchema=input_schema
            or {"properties": {"x": {"type": "integer"}, "y": {"type": "string"}}},
            _MCPFunction__mcp_config=mcp_config or SimpleNamespace(),
        )
        return fn

    return _make
