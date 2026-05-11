"""Tests for server output formatting, description builder, and create_server."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agentica_mcp_runtime.sandbox import ExecutionResult
from agentica_mcp_runtime.server import MAX_OUTPUT_CHARS, _build_execute_description, _execute_impl, create_server


def test_description_with_tools():
    # `_build_execute_description` switched from taking pre-rendered
    # stubs to taking the tools dict and rendering a server catalog
    # (saves ~95% of tool-description tokens). Verify the catalog
    # block appears and the server name shows up.
    from types import SimpleNamespace

    fake_cfg = SimpleNamespace(mcpServers={"slack": object()})
    fake_fn = SimpleNamespace(
        __name__="search",
        _MCPFunction__mcp_config=fake_cfg,
    )
    desc = _build_execute_description({"search": fake_fn})
    assert "WRAPPED MCP SERVERS" in desc
    assert "slack" in desc


def test_description_without_tools():
    desc = _build_execute_description({})
    # Catalog block only renders when tools are present; the static
    # discovery / disclosure preamble is always there.
    assert "WRAPPED MCP SERVERS" not in desc
    assert "Stateful Python REPL" in desc


async def test_execute_output_formatting(monkeypatch):
    mock = AsyncMock()
    mock.is_active = True
    mock.execute = AsyncMock(
        return_value=ExecutionResult(
            output="hello world\n",
            result_repr=None,
            error=None,
            exception_name=None,
            added_vars=("x",),
            changed_vars=(),
            duration=0.123,
        )
    )
    monkeypatch.setattr("agentica_mcp_runtime.server._session", mock)

    result = await _execute_impl("print('hello world')")
    assert "hello world" in result
    assert "New variables: x" in result
    assert "0.123s" in result


async def test_execute_error_formatting(monkeypatch):
    mock = AsyncMock()
    mock.is_active = True
    mock.execute = AsyncMock(
        return_value=ExecutionResult(
            output="",
            result_repr=None,
            error="division by zero",
            exception_name="ZeroDivisionError",
            added_vars=(),
            changed_vars=(),
            duration=0.050,
        )
    )
    monkeypatch.setattr("agentica_mcp_runtime.server._session", mock)

    result = await _execute_impl("1/0")
    assert "ERROR (ZeroDivisionError):" in result


async def test_execute_truncation(monkeypatch):
    long_output = "x" * (MAX_OUTPUT_CHARS + 1000)
    mock = AsyncMock()
    mock.is_active = True
    mock.execute = AsyncMock(
        return_value=ExecutionResult(
            output=long_output,
            result_repr=None,
            error=None,
            exception_name=None,
            added_vars=(),
            changed_vars=(),
            duration=0.100,
        )
    )
    monkeypatch.setattr("agentica_mcp_runtime.server._session", mock)

    result = await _execute_impl("code")
    assert "truncated" in result
    assert len(result) < len(long_output)


def test_create_server_returns_fastmcp():
    server = create_server({})
    assert server is not None
    assert server.name == "agentica-mcp-runtime"


def test_create_server_custom_name():
    server = create_server({}, name="my-server")
    assert server.name == "my-server"
