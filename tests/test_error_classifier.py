"""Classifier + eviction-policy contract.

Pins the load-bearing behavior our invisible-token-rotation UX rides on:

  1. `_classify_error` returns the right class for the error shapes we
     actually see in production (HTTP 401/403, 5xx, 408/429, Node-style
     symbolic codes, opaque "socket hang up", and the boring bad-args
     case).
  2. `_wrap_mcp_function` only evicts the cached Client on `transport`
     / `auth` failures. Bad-args / schema-rejection (`tool`) failures
     leave the cache alone so we don't double-RPC and don't burn a
     keychain re-read for nothing.

The eviction tests are what guard the "the agent never sees a refresh
verb" property — a future refactor that quietly changed eviction to
fire on every error (or never fire) wouldn't break the happy path,
but would silently regress either latency or self-healing. These
tests catch both directions.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agentica_mcp_runtime import sandbox
from agentica_mcp_runtime.sandbox import (
    _classify_error,
    _persistent_clients,
    _wrap_mcp_function,
)


# ----------------------------- classifier ---------------------------------


def test_classify_auth_via_exception_name():
    class UnauthorizedError(Exception):
        pass

    assert _classify_error(UnauthorizedError("nope"), "nope") == "auth"


def test_classify_auth_via_http_401():
    err = Exception("oops")
    err.status_code = 401  # type: ignore[attr-defined]
    assert _classify_error(err, "oops") == "auth"


def test_classify_auth_via_message_keyword():
    # The proxy frequently returns isError=True with a body like
    # "401 Unauthorized" — we have no exception object on the
    # isError path, just the string.
    assert _classify_error(None, "401 Unauthorized") == "auth"
    assert _classify_error(None, "request failed: unauthorized") == "auth"
    assert _classify_error(None, "Token expired, please reauthenticate") == "auth"


def test_classify_transport_via_5xx():
    err = Exception("server boom")
    err.status_code = 503  # type: ignore[attr-defined]
    assert _classify_error(err, "server boom") == "transport"


def test_classify_transport_via_408_429():
    err1 = Exception("slow")
    err1.status_code = 408  # type: ignore[attr-defined]
    err2 = Exception("rate-limited")
    err2.status_code = 429  # type: ignore[attr-defined]
    assert _classify_error(err1, "slow") == "transport"
    assert _classify_error(err2, "rate-limited") == "transport"


def test_classify_transport_via_symbolic_code():
    # Node-style symbolic err.code that shouldn't be confused with
    # numeric HTTP status. Tests `_extract_http_code` skips it AND
    # the keyword fallback catches it.
    err = Exception("read fail")
    err.code = "ECONNRESET"  # type: ignore[attr-defined]
    assert _classify_error(err, "read fail") == "transport"

    err2 = Exception("getaddrinfo")
    err2.code = "ENOTFOUND"  # type: ignore[attr-defined]
    assert _classify_error(err2, "getaddrinfo") == "transport"


def test_classify_transport_via_socket_hang_up():
    # `socket hang up` is the message Node surfaces with no symbolic
    # code at all on aborted keepalive connections. If the classifier
    # only looked at codes, this would slip to `unknown` and break
    # long-running daemons under proxy reaping.
    assert _classify_error(Exception("socket hang up"), "socket hang up") == "transport"


def test_classify_tool_via_4xx():
    err = Exception("bad args")
    err.status_code = 400  # type: ignore[attr-defined]
    assert _classify_error(err, "bad args") == "tool"


def test_classify_tool_via_bad_args_message():
    # The common bad-args case: server understood the call, rejected
    # it, returned isError=True with a schema-shaped message. Must
    # classify as `tool` so the wrapper doesn't evict — eviction
    # here would burn a keychain read on every user typo.
    assert _classify_error(None, "Invalid arguments: missing 'database'") == "tool"
    assert _classify_error(None, "Schema validation failed: expected string") == "tool"


def test_classify_unknown_falls_back():
    # Opaque message, no exception attributes — we default to
    # `unknown` and the wrapper treats that as no-evict so we
    # don't thrash on bugs in our own code.
    assert _classify_error(None, "something weird happened") == "unknown"
    assert _classify_error(Exception("just bad"), "just bad") == "unknown"


def test_classify_httpx_style_nested_response():
    # httpx-style exceptions expose .response.status_code rather
    # than a top-level .status_code. Make sure the classifier
    # walks the nested attribute.
    response = SimpleNamespace(status_code=502)
    err = Exception("upstream boom")
    err.response = response  # type: ignore[attr-defined]
    assert _classify_error(err, "upstream boom") == "transport"


# --------------------------- eviction policy ------------------------------


def _make_mock_fn(mcp_config=None):
    params = [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int)]
    sig = inspect.Signature(params)
    return SimpleNamespace(
        __name__="my_tool",
        __qualname__="my_tool",
        __doc__="A tool.",
        __signature__=sig,
        __module__="test",
        _MCPFunction__tool_inputSchema={"properties": {"x": {"type": "integer"}}},
        _MCPFunction__mcp_config=mcp_config or SimpleNamespace(),
    )


def _make_client_mock(result=None, raise_exc: Exception | None = None):
    """AsyncMock that drives __aenter__/__aexit__/session.call_tool."""
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    if raise_exc is not None:
        client.session.call_tool.side_effect = raise_exc
    else:
        client.session.call_tool.return_value = result
    return client


@pytest.fixture(autouse=True)
def _reset_pool():
    """Tests share module-level _persistent_clients; clear between cases."""
    _persistent_clients.clear()
    yield
    _persistent_clients.clear()


async def test_tool_error_does_not_evict():
    # Bad-args / schema-rejection case: server returns isError=True
    # with a clearly `tool`-class message. Cached Client must persist
    # so the next call (probably with corrected args) reuses the same
    # connection instead of paying a cold-open + keychain re-read.
    fn = _make_mock_fn()
    wrapped = _wrap_mcp_function(fn)
    mock_result = SimpleNamespace(
        isError=True,
        content=[SimpleNamespace(text="Invalid arguments: missing 'database'")],
    )
    mock_client = _make_client_mock(result=mock_result)
    with patch.object(sandbox, "Client", return_value=mock_client):
        with pytest.raises(RuntimeError):
            await wrapped(x=42)
    assert len(_persistent_clients) == 1, (
        "tool-class error should leave the cached Client intact "
        "so the next call doesn't pay a cold-reconnect for nothing"
    )


async def test_auth_error_evicts():
    # 401-equivalent envelope returned via isError=True. Eviction is
    # what triggers the pre-open hook on the next call → fresh
    # keychain bearer → invisible rotation.
    fn = _make_mock_fn()
    wrapped = _wrap_mcp_function(fn)
    mock_result = SimpleNamespace(
        isError=True,
        content=[SimpleNamespace(text="401 Unauthorized: invalid bearer")],
    )
    mock_client = _make_client_mock(result=mock_result)
    with patch.object(sandbox, "Client", return_value=mock_client):
        with pytest.raises(RuntimeError):
            await wrapped(x=42)
    assert _persistent_clients == {}, (
        "auth-class error MUST evict — otherwise the rotated bearer "
        "won't take effect on the next call and the agent sees a 401"
    )


async def test_transport_raise_evicts():
    # call_tool raises a transport-class exception (mocked here as a
    # bare Exception with a network-keyword message). The transport
    # path is exception-driven, not isError-driven, so we exercise
    # it via side_effect.
    fn = _make_mock_fn()
    wrapped = _wrap_mcp_function(fn)
    transport_err = Exception("socket hang up")
    mock_client = _make_client_mock(raise_exc=transport_err)
    with patch.object(sandbox, "Client", return_value=mock_client):
        with pytest.raises(Exception, match="socket hang up"):
            await wrapped(x=42)
    assert _persistent_clients == {}, (
        "transport raise must evict so the next call gets a fresh "
        "Client / fresh underlying connection"
    )


async def test_unknown_raise_does_not_evict():
    # An exception we can't classify falls to `unknown` and gets
    # the safe default — leave the cached Client alone. This guards
    # against bugs in our own code burning keychain reads every
    # retry: if a wrapper or helper raises some opaque AssertionError,
    # we shouldn't tear down the connection.
    fn = _make_mock_fn()
    wrapped = _wrap_mcp_function(fn)
    opaque_err = Exception("something weird happened")
    mock_client = _make_client_mock(raise_exc=opaque_err)
    with patch.object(sandbox, "Client", return_value=mock_client):
        with pytest.raises(Exception, match="something weird"):
            await wrapped(x=42)
    assert len(_persistent_clients) == 1, (
        "unknown-class error should NOT evict; the cached Client may "
        "still be perfectly healthy"
    )
