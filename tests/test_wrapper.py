"""Tests for _wrap_mcp_function — text/structured content handling."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agentica_mcp_runtime.sandbox import _wrap_mcp_function


def _make_mock_fn(
    name: str = "my_tool",
    doc: str = "A tool.",
    input_schema: dict | None = None,
    mcp_config=None,
):
    params = [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int)]
    sig = inspect.Signature(params)
    return SimpleNamespace(
        __name__=name,
        __qualname__=name,
        __doc__=doc,
        __signature__=sig,
        __module__="test",
        _MCPFunction__tool_inputSchema=input_schema or {"properties": {"x": {"type": "integer"}}},
        _MCPFunction__mcp_config=mcp_config or SimpleNamespace(),
    )


def _make_client_mock(result):
    """Build an AsyncMock that works as ``async with Client(cfg) as client:``."""
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.session.call_tool.return_value = result
    return client


async def test_wrapper_returns_text_content():
    fn = _make_mock_fn()
    wrapped = _wrap_mcp_function(fn)

    mock_result = SimpleNamespace(
        isError=False,
        content=[SimpleNamespace(text="hello")],
    )
    mock_client = _make_client_mock(mock_result)

    with patch("agentica_mcp_runtime.sandbox.Client", return_value=mock_client):
        result = await wrapped(x=42)
    assert result == "hello"


async def test_wrapper_returns_structured_content():
    fn = _make_mock_fn()
    wrapped = _wrap_mcp_function(fn)

    mock_result = SimpleNamespace(
        isError=False,
        content=[],
        structuredContent={"result": {"key": "val"}},
    )
    mock_client = _make_client_mock(mock_result)

    with patch("agentica_mcp_runtime.sandbox.Client", return_value=mock_client):
        result = await wrapped(x=42)
    assert result == {"key": "val"}


async def test_wrapper_raises_on_error():
    fn = _make_mock_fn()
    wrapped = _wrap_mcp_function(fn)

    mock_result = SimpleNamespace(
        isError=True,
        content=[SimpleNamespace(text="fail")],
    )
    mock_client = _make_client_mock(mock_result)

    with patch("agentica_mcp_runtime.sandbox.Client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="fail"):
            await wrapped(x=42)


def test_wrapper_preserves_signature():
    fn = _make_mock_fn(name="my_tool", doc="My doc.")
    wrapped = _wrap_mcp_function(fn)
    assert wrapped.__name__ == "my_tool"
    assert wrapped.__doc__ == "My doc."
    assert wrapped.__signature__ == fn.__signature__
