"""Tests for stub generation — _stub() and generate_stubs()."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from agentica_mcp_runtime.sandbox import _stub, generate_stubs


def _make_fn(name: str, doc: str | None, params: list[inspect.Parameter]):
    """Helper to create a fake MCPFunction-like object for stub tests."""
    return SimpleNamespace(
        __name__=name,
        __doc__=doc,
        __signature__=inspect.Signature(params),
    )


def test_stub_single_line_doc():
    fn = _make_fn(
        "my_tool",
        "A short description.",
        [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int)],
    )
    result = _stub(fn)
    expected = 'async def my_tool(x: int):\n    """A short description."""\n    ...'
    assert result == expected


def test_stub_multiline_doc():
    doc = "First line.\n\nSecond paragraph.\nThird line."
    fn = _make_fn(
        "my_tool",
        doc,
        [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)],
    )
    result = _stub(fn)
    assert result.startswith("async def my_tool(x):")
    lines = result.splitlines()
    assert lines[1] == '    """'
    assert "    First line." in result
    assert "    Second paragraph." in result
    assert "    Third line." in result
    assert lines[-2] == '    """'
    assert lines[-1] == "    ..."


def test_stub_no_doc():
    fn = _make_fn(
        "my_tool",
        None,
        [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD)],
    )
    result = _stub(fn)
    assert '"""' not in result
    assert result == "async def my_tool(x):\n    ..."


def test_generate_stubs_multiple():
    tools = {
        "tool_a": _make_fn("tool_a", "Doc A.", []),
        "tool_b": _make_fn("tool_b", "Doc B.", []),
        "tool_c": _make_fn("tool_c", "Doc C.", []),
    }
    result = generate_stubs(tools)
    assert "tool_a" in result
    assert "tool_b" in result
    assert "tool_c" in result
    # Stubs are separated by double newlines
    stubs = result.split("\n\n")
    assert len(stubs) == 3
