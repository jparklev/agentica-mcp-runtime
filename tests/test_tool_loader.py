"""load_tools concurrency + failure-isolation contract.

Two load-bearing invariants:

  1. **Parallel fan-out.** Tool discovery runs through ~10 MCP servers
     at startup, each handshake is 1-3s. Serial = 20-30s wall clock.
     Parallel = gated by the single slowest server (~3-5s). We pin
     "the per-server `from_mcp_config` calls overlap" by giving each
     fake a measurable sleep and asserting total wall time < N * sleep.

  2. **One broken server doesn't block others.** A failure inside
     `from_mcp_config` must NOT propagate out of `load_tools` or
     cancel other in-flight handshakes. A future "let's surface
     errors cleanly" refactor that drops the per-server try/except
     would silently regress startup robustness (one misconfigured
     server takes the whole runtime down). This test catches that.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentica_mcp_runtime.tool_loader import load_tools


def _fake_fn(name: str):
    return SimpleNamespace(__name__=name)


async def test_load_tools_fans_out_in_parallel():
    """Wall time must be << N × per-handshake time. With 5 servers each
    sleeping 200ms, serial would be ≥1.0s. Parallel should be ~0.2s.
    We gate at 0.5s to leave headroom for CI noise but still catch a
    regression to serial execution.
    """
    configs = {f"server_{i}": SimpleNamespace() for i in range(5)}

    async def fake_from_mcp_config(_cfg):
        await asyncio.sleep(0.2)
        return [_fake_fn(f"tool_for_{id(_cfg)}")]

    with patch(
        "agentica_mcp_runtime.tool_loader.MCPFunction.from_mcp_config",
        side_effect=fake_from_mcp_config,
    ):
        t0 = time.monotonic()
        tools = await load_tools(configs)
        elapsed = time.monotonic() - t0

    assert len(tools) == 5
    assert elapsed < 0.5, (
        f"load_tools appears to be serial — elapsed {elapsed:.2f}s for 5 "
        f"servers × 0.2s sleep. Should be ~0.2s if parallel; ~1.0s if serial."
    )


async def test_load_tools_isolates_failures():
    """One server raising must not cancel others or propagate out. The
    surviving servers contribute their tools normally; the failing one
    contributes an empty slot.
    """
    # `_marker` distinguishes cfg objects in the fake — without it, two
    # empty SimpleNamespace() instances compare equal under `==` and the
    # `cfg is X` / `.index(cfg)` tricks misroute the calls.
    configs = {
        "good_a": SimpleNamespace(_marker="good_a"),
        "broken": SimpleNamespace(_marker="broken"),
        "good_b": SimpleNamespace(_marker="good_b"),
    }

    async def fake_from_mcp_config(cfg):
        if cfg._marker == "broken":
            raise RuntimeError("simulated transport boom")
        return [_fake_fn(f"tool_from_{cfg._marker}")]

    with patch(
        "agentica_mcp_runtime.tool_loader.MCPFunction.from_mcp_config",
        side_effect=fake_from_mcp_config,
    ):
        tools = await load_tools(configs)

    # Two surviving servers contributed one tool each.
    assert set(tools.keys()) == {"tool_from_good_a", "tool_from_good_b"}, (
        f"failure isolation broken — got {sorted(tools.keys())}"
    )


async def test_load_tools_empty_configs():
    """Edge case: no servers configured. Must return {} cleanly without
    spinning up an empty gather (asyncio.gather() with no args returns
    immediately, so this is mostly a regression guard against a future
    refactor that adds a guard clause raising on empty input).
    """
    tools = await load_tools({})
    assert tools == {}


async def test_load_tools_preserves_iteration_order_for_name_collisions():
    """If two servers expose the same tool name (rare but possible —
    e.g. two different Slack-shaped servers both exposing `search`),
    the later one wins, matching the serial version's last-write-wins
    behavior. The order is determined by `configs.items()` iteration,
    not by which server's handshake finishes first.
    """
    configs = {
        "first": SimpleNamespace(_marker="first"),
        "second": SimpleNamespace(_marker="second"),
    }

    async def fake_from_mcp_config(cfg):
        # Both return a tool named "shared" — exactly the collision case.
        # The "first" server's handshake finishes AFTER the "second"
        # server's, but the result merge should still favor "second"
        # because of configs.items() iteration order. Use `_marker`
        # to distinguish cfgs — bare SimpleNamespace() instances
        # compare equal under `==`.
        if cfg._marker == "first":
            await asyncio.sleep(0.05)
            return [SimpleNamespace(__name__="shared", source="first")]
        return [SimpleNamespace(__name__="shared", source="second")]

    with patch(
        "agentica_mcp_runtime.tool_loader.MCPFunction.from_mcp_config",
        side_effect=fake_from_mcp_config,
    ):
        tools = await load_tools(configs)

    assert len(tools) == 1
    # configs.items() yields "first" then "second"; merge iterates the
    # same order, so "second" overwrites "first". This matches the
    # pre-parallelization behavior exactly.
    assert tools["shared"].source == "second"
