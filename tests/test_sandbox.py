"""Tests for SandboxSession lifecycle with real BaseRepl."""

from __future__ import annotations

import pytest

from agentica_mcp_runtime.sandbox import SandboxSession


async def test_session_lifecycle():
    session = SandboxSession()
    assert not session.is_active
    session.start(tools={})
    assert session.is_active
    await session.execute("pass")  # ensure REPL history exists before reset
    session.stop()
    assert not session.is_active


async def test_execute_simple_code():
    session = SandboxSession()
    session.start(tools={})
    try:
        result = await session.execute("x = 1 + 1\nprint(x)")
        assert "2" in result.output
        assert "x" in result.added_vars
    finally:
        session.stop()


async def test_execute_persists_variables():
    session = SandboxSession()
    session.start(tools={})
    try:
        await session.execute("x = 42")
        result = await session.execute("print(x)")
        assert "42" in result.output
    finally:
        session.stop()


async def test_execute_error():
    session = SandboxSession()
    session.start(tools={})
    try:
        result = await session.execute("1/0")
        assert result.error is not None
        assert "ZeroDivisionError" in result.exception_name
    finally:
        session.stop()


async def test_pre_injected_modules():
    session = SandboxSession()
    session.start(tools={})
    try:
        result = await session.execute("print(type(asyncio))")
        assert "module" in result.output
    finally:
        session.stop()


async def test_execute_before_start_raises():
    session = SandboxSession()
    with pytest.raises(RuntimeError):
        await session.execute("x = 1")
