"""Per-server semaphore + crash-breadcrumb contract.

Three invariants worth pinning:

  1. **Concurrent calls to the same server cap at N.** A `gather` of 20 tool
     calls against one server must never see more than N (default 8) in
     flight at once. This is the structural fix for the "40-query fanout
     killed the transport" incident.
  2. **Different servers don't compete.** Slack calls don't take Notion's
     slots and vice versa — the per-server quota is independent.
  3. **Failures release the slot.** An exception inside the held semaphore
     must release it (otherwise one bad call starves the cap forever).
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agentica_mcp_runtime import sandbox
from agentica_mcp_runtime.sandbox import (
    _DEFAULT_SERVER_CONCURRENCY,
    _SERVER_SEMAPHORES,
    _persistent_clients,
    _wrap_mcp_function,
    set_server_concurrency,
)


@pytest.fixture(autouse=True)
def _reset_state():
    _SERVER_SEMAPHORES.clear()
    sandbox._SERVER_CAP_OVERRIDES.clear()
    _persistent_clients.clear()
    yield
    _SERVER_SEMAPHORES.clear()
    sandbox._SERVER_CAP_OVERRIDES.clear()
    _persistent_clients.clear()


def _make_fn_for_server(name: str, server: str):
    """Build a fake MCPFunction wired to a specific server name. The wrapper
    reads `mcp_config.mcpServers` to identify the server; SimpleNamespace
    with the right shape satisfies it."""
    params = [inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int)]
    sig = inspect.Signature(params)
    cfg = SimpleNamespace(mcpServers={server: object()})
    return SimpleNamespace(
        __name__=name,
        __qualname__=name,
        __doc__="",
        __signature__=sig,
        __module__="test",
        _MCPFunction__tool_inputSchema={"properties": {"x": {"type": "integer"}}},
        _MCPFunction__mcp_config=cfg,
    )


def _make_throttled_client(in_flight_counter: dict, peak_key: str = "peak"):
    """AsyncMock client whose call_tool tracks max-in-flight across calls.

    Sleeps briefly so the gather'd tasks actually overlap. Without the sleep,
    each task completes before the next starts and the peak is always 1.
    """
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False

    async def _call_tool(_name, _args):
        in_flight_counter["current"] = in_flight_counter.get("current", 0) + 1
        in_flight_counter[peak_key] = max(
            in_flight_counter.get(peak_key, 0), in_flight_counter["current"]
        )
        # Long enough to keep the gather'd tasks overlapping in real time.
        await asyncio.sleep(0.05)
        in_flight_counter["current"] -= 1
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text="ok")])

    client.session.call_tool.side_effect = _call_tool
    return client


async def test_same_server_calls_cap_at_default_concurrency():
    """20 parallel calls to one server peak at exactly _DEFAULT_SERVER_CONCURRENCY."""
    fn = _make_fn_for_server("my_tool", "slack")
    wrapped = _wrap_mcp_function(fn)
    counter = {"current": 0, "peak": 0}
    client = _make_throttled_client(counter)
    with patch.object(sandbox, "Client", return_value=client):
        # Each wrapped() call goes through _get_persistent_client. With
        # one shared client (return_value, not side_effect), they all
        # share the same `client` instance but each acquires its own
        # semaphore slot.
        await asyncio.gather(*(wrapped(x=i) for i in range(20)))

    assert counter["peak"] <= _DEFAULT_SERVER_CONCURRENCY, (
        f"in-flight peak {counter['peak']} exceeded cap {_DEFAULT_SERVER_CONCURRENCY}; "
        f"the semaphore did not throttle"
    )
    # The semaphore must actually be exercised — if peak == 1 we never
    # achieved real concurrency and the test is meaningless.
    assert counter["peak"] >= 2, (
        f"in-flight peak only {counter['peak']}; the gather didn't overlap, "
        f"test doesn't actually verify throttling"
    )


async def test_different_servers_dont_share_quota():
    """Slack and Notion get their own quotas — Slack saturated doesn't
    block Notion."""
    fn_slack = _make_fn_for_server("slack_search", "slack")
    fn_notion = _make_fn_for_server("notion_search", "notion")
    w_slack = _wrap_mcp_function(fn_slack)
    w_notion = _wrap_mcp_function(fn_notion)
    counter = {"current": 0, "peak_slack": 0, "peak_notion": 0}

    async def _call_tool_slack(_name, _args):
        counter["current"] = counter.get("current", 0) + 1
        counter["peak_slack"] = max(counter.get("peak_slack", 0), counter["current"])
        await asyncio.sleep(0.05)
        counter["current"] -= 1
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text="ok")])

    async def _call_tool_notion(_name, _args):
        counter["current"] = counter.get("current", 0) + 1
        counter["peak_notion"] = max(counter.get("peak_notion", 0), counter["current"])
        await asyncio.sleep(0.05)
        counter["current"] -= 1
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text="ok")])

    slack_client = AsyncMock()
    slack_client.__aenter__.return_value = slack_client
    slack_client.__aexit__.return_value = False
    slack_client.session.call_tool.side_effect = _call_tool_slack

    notion_client = AsyncMock()
    notion_client.__aenter__.return_value = notion_client
    notion_client.__aexit__.return_value = False
    notion_client.session.call_tool.side_effect = _call_tool_notion

    # `Client(cfg)` must return slack_client for slack's cfg and notion_client
    # for notion's cfg. We can't tell from cfg alone (both are SimpleNamespace),
    # so dispatch by call-count: first call (slack's _get_persistent_client)
    # gets slack_client, second gets notion_client. This relies on
    # _get_persistent_client being called once per cfg before any gather work
    # starts — true because the test launches both gathers in sequence.
    clients = [slack_client, notion_client]

    def _client_factory(_cfg):
        return clients.pop(0)

    with patch.object(sandbox, "Client", side_effect=_client_factory):
        # Saturate slack with 20 concurrent calls, send 4 notion at once.
        # If notion shared slack's quota, notion's peak would be capped
        # by what's left of slack's. Independence means notion sees its
        # full 4-way concurrency immediately.
        await asyncio.gather(
            *(w_slack(x=i) for i in range(20)),
            *(w_notion(x=i) for i in range(4)),
        )

    assert counter["peak_slack"] <= _DEFAULT_SERVER_CONCURRENCY
    assert counter["peak_notion"] >= 2, (
        f"notion peak only {counter['peak_notion']}; quotas appear shared"
    )


async def test_exception_inside_semaphore_releases_slot():
    """Failing call must release the slot. Otherwise N failing calls
    starve the cap and every subsequent call deadlocks."""
    fn = _make_fn_for_server("flaky_tool", "slack")
    wrapped = _wrap_mcp_function(fn)

    # Custom override: cap of 2 makes the bug easy to detect — if releases
    # are broken, 3 failing calls deadlock the 4th forever.
    set_server_concurrency("slack", 2)

    fail_client = AsyncMock()
    fail_client.__aenter__.return_value = fail_client
    fail_client.__aexit__.return_value = False
    fail_client.session.call_tool.side_effect = RuntimeError("simulated boom")

    with patch.object(sandbox, "Client", return_value=fail_client):
        # If the slot leaks, the 5th call hangs. Use wait_for to
        # turn the hang into a timeout-failure that pytest can surface.
        async def _one():
            try:
                await wrapped(x=1)
            except RuntimeError:
                pass

        await asyncio.wait_for(
            asyncio.gather(*(_one() for _ in range(5))),
            timeout=2.0,
        )


async def test_set_server_concurrency_takes_effect():
    """Lowering the cap mid-session caps subsequent gathers at the new value."""
    fn = _make_fn_for_server("notion_search", "notion")
    wrapped = _wrap_mcp_function(fn)
    counter = {"current": 0, "peak": 0}
    client = _make_throttled_client(counter)
    set_server_concurrency("notion", 3)
    with patch.object(sandbox, "Client", return_value=client):
        await asyncio.gather(*(wrapped(x=i) for i in range(10)))
    assert counter["peak"] <= 3, f"override cap of 3 ignored; peak was {counter['peak']}"


async def test_set_server_concurrency_rejects_invalid():
    with pytest.raises(ValueError):
        set_server_concurrency("slack", 0)


def test_record_eviction_appends_to_ring():
    """Direct unit test for _record_eviction. Ring-buffer cap enforced."""
    from agentica_mcp_runtime.sandbox import _RECENT_EVICTIONS, _record_eviction, _RECENT_EVICTIONS_MAX
    _RECENT_EVICTIONS.clear()
    for i in range(_RECENT_EVICTIONS_MAX + 5):
        _record_eviction(f"server_{i}", "transport", f"err {i}")
    assert len(_RECENT_EVICTIONS) == _RECENT_EVICTIONS_MAX
    # Oldest entries should have been popped — first surviving entry's
    # message should be "err 5".
    assert _RECENT_EVICTIONS[0]["msg"] == "err 5"
