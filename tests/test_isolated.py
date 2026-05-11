"""python_isolated subprocess contract.

The persistent `python` REPL is a singleton serial executor — N concurrent
callers queue inside the runtime and time out as `-32000` under load. The
`python_isolated` MCP tool side-steps that by spawning a fresh subprocess
per call. These tests pin the invariants the design rests on:

  1. Simple inline code returns stdout + an `[isolated; N.NNs]` footer.
  2. `code_file` parameter reads from a path; `/tmp/*` ephemerals are
     auto-deleted by `isolated_runner` (so the parent doesn't accumulate junk).
  3. Concurrent invocations TRULY run in parallel — that's the whole point.
     A semaphore-style serial implementation would fail this test.
  4. Top-level `await` works (CO_COROUTINE flag path via
     `PyCF_ALLOW_TOP_LEVEL_AWAIT`).
  5. Exceptions in user code don't crash the parent — stderr captured,
     non-zero exit code surfaced, parent stays responsive.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from agentica_mcp_runtime.server import _execute_isolated_impl


# Mark all tests in this module as integration: they spawn real subprocesses.
# We don't mock subprocess because the whole value prop is "real isolation."
pytestmark = pytest.mark.asyncio


async def test_isolated_inline_code_returns_stdout():
    out = await _execute_isolated_impl(code='print("hello from isolated")')
    assert "hello from isolated" in out
    assert "[isolated;" in out  # timing footer present
    # No exit-code line on success (return 0 means we don't emit it)
    assert "[exit code:" not in out


async def test_isolated_code_file_path(tmp_path: Path):
    f = tmp_path / "snippet.py"
    f.write_text('print("from-file")')
    out = await _execute_isolated_impl(code_file=str(f))
    assert "from-file" in out
    # pytest's tmp_path on macOS lives under /private/var/folders/... which
    # IS in TMP_PREFIXES, so the file gets auto-deleted. We don't assert
    # on the file's post-call existence — the auto-delete contract is
    # "ephemeral /tmp files don't accumulate," with no guarantee about
    # specific paths.


async def test_isolated_tmp_code_file_auto_deleted():
    # Build a /tmp file and confirm isolated_runner deletes it after read.
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".py", prefix="agentica-isolated-test-")
    os.close(fd)
    with open(path, "w") as f:
        f.write('print("ephemeral")')
    assert os.path.exists(path)
    out = await _execute_isolated_impl(code_file=path)
    assert "ephemeral" in out
    # Subprocess's isolated_runner should have unlinked it
    assert not os.path.exists(path)


async def test_isolated_rejects_both_args():
    out = await _execute_isolated_impl(code="x = 1", code_file="/tmp/foo.py")
    assert "ERROR" in out
    assert "not both" in out


async def test_isolated_rejects_no_args():
    out = await _execute_isolated_impl()
    assert "ERROR" in out
    # Either `code` or `code_file`
    assert "either" in out.lower()


async def test_isolated_runs_concurrently(monkeypatch):
    """The whole point: N concurrent calls fan out across N subprocesses.

    We point AGENTICA_SERVERS_CONFIG at a nonexistent path so the subprocess
    skips load_tools (which is the dominant startup cost in production —
    ~3-5s for 10 cloud MCP cold opens). Without that, all subprocesses
    still parallelize, but their startup floor masks the per-call sleep.

    Each call sleeps 1s. Serial → ~3s. Parallel → ~1-1.5s. Threshold 2.0s
    catches a regression to serial while leaving headroom for CI noise.
    """
    monkeypatch.setenv("AGENTICA_SERVERS_CONFIG", "/tmp/agentica-test-nonexistent.json")
    code = (
        "import time\n"
        "time.sleep(1.0)\n"
        "print('done')\n"
    )
    t0 = time.monotonic()
    outs = await asyncio.gather(
        _execute_isolated_impl(code=code),
        _execute_isolated_impl(code=code),
        _execute_isolated_impl(code=code),
    )
    elapsed = time.monotonic() - t0
    for o in outs:
        assert "done" in o
    assert elapsed < 2.0, (
        f"elapsed {elapsed:.2f}s — 3 concurrent calls × 1s sleep should be "
        f"~1-1.5s if truly parallel, ~3s if serialized. Regression?"
    )


async def test_isolated_supports_top_level_await():
    """`PyCF_ALLOW_TOP_LEVEL_AWAIT` path: user code can `await` at module
    scope without an outer `async def`."""
    code = (
        "import asyncio\n"
        "async def _inner():\n"
        "    await asyncio.sleep(0)\n"
        "    return 'awaited-ok'\n"
        "result = await _inner()\n"
        "print(result)\n"
    )
    out = await _execute_isolated_impl(code=code)
    assert "awaited-ok" in out


async def test_isolated_surfaces_user_exception():
    """A traceback from user code should land in stderr; the parent process
    must not crash. Exit code should be non-zero."""
    code = "raise ValueError('intentional test boom')\n"
    out = await _execute_isolated_impl(code=code)
    assert "intentional test boom" in out
    assert "[STDERR]" in out
    assert "[exit code:" in out  # subprocess returned non-zero


async def test_isolated_handles_syntax_error():
    out = await _execute_isolated_impl(code="this is ::: not valid python\n")
    assert "[STDERR]" in out
    # SyntaxError surfaces with traceback
    assert "SyntaxError" in out or "invalid" in out.lower()
