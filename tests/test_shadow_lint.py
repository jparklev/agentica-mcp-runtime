"""Pre-exec AST lint + post-error shadow-detector contract.

The REPL's globals dict is shared across every `python()` call AND
auto-loaded with ~150 helpers. User code like `cap = 1` rebinds the
helper to a non-callable; subsequent `cap(obj)` fails with `'int' object
is not callable`. We've hit this in practice; these tests pin two
structural defenses:

  1. `_scan_module_scope_assigns` finds the names a chunk of code assigns
     to at the user's module scope, while NOT descending into nested
     scopes (def / class / lambda / comprehensions). Lint warnings should
     fire for shadows at module scope and stay silent for assignments
     inside a local function.

  2. `SandboxSession.currently_shadowed_helpers()` reports the helper
     names whose current REPL-globals value is no longer callable. The
     server layer reads this on error and appends a `HINT: helper(s)
     shadowed ... call helpers_reload()` line — the post-error half of
     the defense.
"""

from __future__ import annotations

import pytest

from agentica_mcp_runtime.sandbox import (
    SandboxSession,
    _scan_module_scope_assigns,
    _lint_for_helper_shadow,
)


# ----------------------- AST scan unit tests ---------------------------------


def test_scan_plain_assign():
    assert _scan_module_scope_assigns("cap = 1") == {"cap"}


def test_scan_tuple_unpack():
    assert _scan_module_scope_assigns("a, b = 1, 2") == {"a", "b"}


def test_scan_starred_unpack():
    assert _scan_module_scope_assigns("a, *rest = (1, 2, 3)") == {"a", "rest"}


def test_scan_ann_assign():
    assert _scan_module_scope_assigns("cap: int = 5") == {"cap"}


def test_scan_aug_assign():
    # `x += 1` requires x to exist already, but for shadow detection
    # purposes we should still flag it.
    assert _scan_module_scope_assigns("cap = 0\ncap += 1") == {"cap"}


def test_scan_for_loop_target():
    assert _scan_module_scope_assigns("for cap in [1, 2]: pass") == {"cap"}


def test_scan_async_for_target():
    code = "async def f():\n    async for cap in iter: pass\n"
    # `cap` here is inside async def — local scope — should NOT be flagged.
    assert _scan_module_scope_assigns(code) == {"f"}


def test_scan_walrus():
    assert _scan_module_scope_assigns("x = (cap := 5)") == {"x", "cap"}


def test_scan_import_simple():
    assert _scan_module_scope_assigns("import cap") == {"cap"}


def test_scan_import_dotted_takes_top_module():
    # `import foo.bar` binds `foo`, not `bar`.
    assert _scan_module_scope_assigns("import foo.bar") == {"foo"}


def test_scan_import_as():
    assert _scan_module_scope_assigns("import foo.bar as cap") == {"cap"}


def test_scan_from_import():
    assert _scan_module_scope_assigns("from collections import cap") == {"cap"}


def test_scan_from_import_as():
    assert _scan_module_scope_assigns(
        "from collections import OrderedDict as cap"
    ) == {"cap"}


def test_scan_skips_function_body():
    """Assignments inside a function body are LOCAL scope — should not
    be flagged as module-scope shadows."""
    code = (
        "def foo():\n"
        "    cap = 1\n"
        "    return cap\n"
    )
    # We expect only `foo` (the def itself binds at module scope).
    assert _scan_module_scope_assigns(code) == {"foo"}


def test_scan_skips_class_body():
    code = (
        "class Foo:\n"
        "    cap = 1\n"
    )
    assert _scan_module_scope_assigns(code) == {"Foo"}


def test_scan_skips_lambda():
    # The lambda's parameter doesn't shadow at module scope.
    assert _scan_module_scope_assigns("f = lambda cap: cap + 1") == {"f"}


def test_scan_skips_listcomp_target():
    # List/set/dict/gen comprehensions have their own local scope in py3.
    assert _scan_module_scope_assigns("x = [cap for cap in [1, 2]]") == {"x"}


def test_scan_syntax_error_returns_empty():
    """A broken parse should not throw — the REPL will surface the
    SyntaxError itself with proper context."""
    assert _scan_module_scope_assigns("this !!! is not python") == set()


def test_scan_nested_for_loop_at_module_scope_still_flagged():
    """A for loop inside an if-block at module scope is still module scope."""
    code = (
        "if True:\n"
        "    for cap in []: pass\n"
    )
    assert _scan_module_scope_assigns(code) == {"cap"}


# ----------------------- lint helper ----------------------------------------


def test_lint_returns_only_intersection_with_protected():
    assigns = "cap = 1\nfoo = 2\nbar = 3"
    protected = {"cap", "bar", "missing"}
    assert _lint_for_helper_shadow(assigns, protected) == ["bar", "cap"]


def test_lint_empty_protected_returns_empty():
    assert _lint_for_helper_shadow("cap = 1", set()) == []


# ----------------------- SandboxSession integration -------------------------


@pytest.mark.asyncio
async def test_session_currently_shadowed_starts_empty():
    sess = SandboxSession()
    sess.start(tools={})
    try:
        # After start with no tools, no shadowing yet.
        assert sess.currently_shadowed_helpers() == []
    finally:
        # BaseRepl.reset() needs at least one prior execute or it crashes
        # on `del self.last_eval` — same workaround as test_session_lifecycle.
        await sess.execute("pass")
        sess.stop()


@pytest.mark.asyncio
async def test_session_lint_warns_on_shadow_assignment():
    """The pre-exec lint prepends a `[lint] WARNING` banner when user code
    will shadow a protected name. We seed a fake protected name to test."""
    sess = SandboxSession()
    sess.start(tools={})
    try:
        # Forge a protected name in the session — we want to test the
        # warning path, not the helpers.py load.
        object.__setattr__(sess, "_original_helper_names", frozenset({"cap"}))
        result = await sess.execute("cap = 1\nprint('after')")
        assert "[lint] WARNING" in result.output
        assert "cap" in result.output
        # Code still ran.
        assert "after" in result.output
    finally:
        sess.stop()


@pytest.mark.asyncio
async def test_session_currently_shadowed_after_user_assignment():
    sess = SandboxSession()
    sess.start(tools={})
    try:
        # Plant a callable helper in globals (a fake one — we own the dict).
        def _fake_cap(x):
            return x
        sess._globals_dict["cap"] = _fake_cap
        object.__setattr__(sess, "_original_helper_names", frozenset({"cap"}))

        # Initially callable — not shadowed.
        assert sess.currently_shadowed_helpers() == []

        # User code rebinds it to an int.
        await sess.execute("cap = 1")
        assert sess.currently_shadowed_helpers() == ["cap"]
    finally:
        sess.stop()


@pytest.mark.asyncio
async def test_session_currently_shadowed_ignores_deleted_names():
    """If a helper got `del`'d (not shadowed), don't report it as shadowed —
    that's a different kind of user intent."""
    sess = SandboxSession()
    sess.start(tools={})
    try:
        def _fake_cap(x): return x
        sess._globals_dict["cap"] = _fake_cap
        object.__setattr__(sess, "_original_helper_names", frozenset({"cap"}))
        await sess.execute("del cap")
        # del should not produce a "shadowed" report.
        assert sess.currently_shadowed_helpers() == []
    finally:
        sess.stop()
