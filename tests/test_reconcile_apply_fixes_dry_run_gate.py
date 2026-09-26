"""Issue #1475: the caller-side dry-run gate is a tested invariant, not just
a comment -- every ``apply_fixes`` call site in ``src/`` must be dominated by
a guard that proves ``dry_run`` is False.

This lives in its own module (rather than appended to
``tests/test_reconcile_apply_fixes.py``, the #1559 split host for the other
``apply_fixes`` lanes) because that file already sits at the 800-line
file-size ratchet cap -- the keystone gate in
``tests/test_file_size_ratchet.py`` fails any growth past a mark of 0 for an
unrecorded over-cap file. The behavioral half of the same invariant -- the
end-to-end ``reconcile(fix=True)`` dry-run pass that spies on
``push_branch`` -- lives in ``tests/test_reconcile_drift_salvage.py`` next to
the worktree-salvage fixtures it drives.

The scanner itself is unit-tested below over synthetic source strings: the
one real call site cannot exercise the visitor's guard-stack mechanics, so
each recognised gate shape gets a gated and an ungated fixture.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

_SRC_ROOT = Path(__file__).parents[1] / "src" / "charlie_work"

# ``apply_fixes`` is imported as ``apply_drift_fixes`` in
# orchestration/misc_reconcile.py; both spellings name the same function and
# are matched receiver-agnostically (bare ``Name`` or ``Attribute``).
_APPLY_FIXES_NAMES = {"apply_fixes", "apply_drift_fixes"}


def _apply_fixes_call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name) and func.id in _APPLY_FIXES_NAMES:
        return func.id
    if isinstance(func, ast.Attribute) and func.attr in _APPLY_FIXES_NAMES:
        return func.attr
    return None


def _is_dry_run_ref(expr: ast.expr) -> bool:
    return (isinstance(expr, ast.Name) and expr.id == "dry_run") or (
        isinstance(expr, ast.Attribute) and expr.attr == "dry_run"
    )


def _truthy_test_excludes_dry_run(test: ast.expr) -> bool:
    """True if ``test`` evaluating truthy implies ``dry_run`` is False:
    ``not dry_run`` appears as an ``and``-conjunct of the test (the shape
    ``if fix and not dry_run and drift:`` uses)."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return any(_truthy_test_excludes_dry_run(value) for value in test.values)
    return (
        isinstance(test, ast.UnaryOp)
        and isinstance(test.op, ast.Not)
        and _is_dry_run_ref(test.operand)
    )


def _falsy_test_excludes_dry_run(test: ast.expr) -> bool:
    """True if ``test`` evaluating falsy implies ``dry_run`` is False:
    ``dry_run`` appears as an ``or``-disjunct of the test, so the falsy
    branch only runs when every disjunct -- ``dry_run`` included -- is
    falsy (the ``if dry_run: ... else: <call>`` shape)."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        return any(_falsy_test_excludes_dry_run(value) for value in test.values)
    return _is_dry_run_ref(test)


def _apply_fixes_call_sites(
    tree: ast.AST,
) -> list[tuple[ast.Call, tuple[tuple[ast.expr, bool], ...]]]:
    """Every ``apply_fixes``/``apply_drift_fixes`` call with its guard stack.

    Each guard is ``(test, held)``: ``held=True`` means the call sits in the
    branch where ``test`` evaluated truthy (``if`` body / ``while`` body),
    ``held=False`` in the falsy branch (``else``/``elif`` descent). Guards
    reset at function/lambda boundaries -- an outer ``if`` does not dominate
    a nested ``def``'s body (decorators, default expressions, and the
    ``returns`` annotation DO evaluate at def-time in the outer scope, so
    they keep the outer guards; PEP 695 ``type_params`` are lazily evaluated
    like the body, so they reset with it).

    ``visit`` dispatches on ``node`` itself, then recurses through
    ``ast.iter_child_nodes`` for everything else. Dispatching on the
    *children* of ``node`` instead would leave every shape that arrives as
    the node -- an ``if``/``while`` directly in a function body, a nested
    ``def`` under a gated ``if``, an ``IfExp`` branch, a lambda body, a
    decorator, a call inside a test expression -- classified by the wrong
    frame or not recorded at all.
    """
    sites: list[tuple[ast.Call, tuple[tuple[ast.expr, bool], ...]]] = []

    def visit(node: ast.AST, guards: tuple[tuple[ast.expr, bool], ...]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                visit(decorator, guards)
            visit(node.args, guards)
            if node.returns is not None:
                visit(node.returns, guards)
            for type_param in getattr(node, "type_params", ()):
                visit(type_param, ())
            for stmt in node.body:
                visit(stmt, ())
        elif isinstance(node, ast.Lambda):
            visit(node.args, guards)
            visit(node.body, ())
        elif isinstance(node, ast.If):
            visit(node.test, guards)
            for stmt in node.body:
                visit(stmt, (*guards, (node.test, True)))
            # An ``elif`` is a nested ``If`` inside ``orelse`` and is
            # dispatched through this same branch, inheriting both guards.
            for stmt in node.orelse:
                visit(stmt, (*guards, (node.test, False)))
        elif isinstance(node, ast.IfExp):
            # Unlike ``ast.If``, body/orelse are single expressions.
            visit(node.test, guards)
            visit(node.body, (*guards, (node.test, True)))
            visit(node.orelse, (*guards, (node.test, False)))
        elif isinstance(node, ast.While):
            visit(node.test, guards)
            for stmt in node.body:
                visit(stmt, (*guards, (node.test, True)))
            # ``while ... else`` runs after loop completion -- the test
            # no longer dominates -- so orelse keeps the outer guards.
            for stmt in node.orelse:
                visit(stmt, guards)
        else:
            if isinstance(node, ast.Call) and _apply_fixes_call_name(node):
                sites.append((node, guards))
            for child in ast.iter_child_nodes(node):
                visit(child, guards)

    visit(tree, ())
    return sites


def _site_is_dry_run_gated(guards: tuple[tuple[ast.expr, bool], ...]) -> bool:
    """True if at least one dominating guard proves ``dry_run`` is False."""
    return any(
        _truthy_test_excludes_dry_run(test) if held else _falsy_test_excludes_dry_run(test)
        for test, held in guards
    )


def _scan_source(source: str) -> list[bool]:
    """Per-site gated verdicts for ``source``, in source order.

    Shared by the real-tree walk and the synthetic-source unit tests so the
    visitor's mechanics are covered independently of the one call site that
    exists in ``src/`` today.
    """
    return [
        _site_is_dry_run_gated(guards) for _, guards in _apply_fixes_call_sites(ast.parse(source))
    ]


def _scan(source: str) -> list[bool]:
    return _scan_source(textwrap.dedent(source))


def test_gate_and_conjunct() -> None:
    """``not dry_run`` as an ``and``-conjunct of the dominating ``if`` test."""
    assert _scan(
        """\
        if fix and not dry_run:
            apply_fixes()
        """
    ) == [True]
    assert _scan(
        """\
        if fix:
            apply_fixes()
        """
    ) == [False]
    # A ``dry_run`` conjunct without ``not`` does not exclude dry_run.
    assert _scan(
        """\
        if fix and dry_run:
            apply_fixes()
        """
    ) == [False]


def test_gate_and_conjunct_inside_function_body() -> None:
    """A gated ``if`` as a direct statement of a function body pushes its
    test: ``visit`` must dispatch on the node itself, not only on children
    of the enclosing frame."""
    assert _scan(
        """\
        def run():
            if fix and not dry_run:
                apply_fixes()
        """
    ) == [True]
    assert _scan(
        """\
        def run():
            if fix:
                apply_fixes()
        """
    ) == [False]


def test_gate_or_disjunct_else_branch() -> None:
    """The falsy branch of a test with ``dry_run`` as an ``or``-disjunct only
    runs when ``dry_run`` is falsy."""
    assert _scan(
        """\
        if dry_run or skip:
            report_only()
        else:
            apply_fixes()
        """
    ) == [True]
    assert _scan(
        """\
        def run():
            if dry_run:
                report_only()
            else:
                apply_fixes()
        """
    ) == [True]
    assert _scan(
        """\
        if skip:
            report_only()
        else:
            apply_fixes()
        """
    ) == [False]


def test_gate_elif() -> None:
    """An ``elif`` is a nested ``If`` in ``orelse`` and pushes its own test
    on top of the inherited falsy guard."""
    assert _scan(
        """\
        if busy:
            wait()
        elif fix and not dry_run:
            apply_fixes()
        """
    ) == [True]
    assert _scan(
        """\
        if busy:
            wait()
        elif fix:
            apply_fixes()
        """
    ) == [False]


def test_gate_ifexp() -> None:
    """Conditional-expression branches are guarded by the test's truth
    value; a ``Call`` nested inside one is still a site."""
    assert _scan("x = apply_fixes() if not dry_run else None\n") == [True]
    assert _scan("x = apply_fixes() if fix else None\n") == [False]
    # The falsy arm of an unnegated ``dry_run`` test also proves exclusion.
    assert _scan("x = None if dry_run else apply_fixes()\n") == [True]


def test_gate_while_and_while_else() -> None:
    """``while`` pushes its test for the body; ``while ... else`` runs after
    loop completion, so the test no longer dominates and the else clause
    keeps only the outer guards."""
    assert _scan(
        """\
        def run():
            while not dry_run and pending():
                apply_fixes()
        """
    ) == [True]
    assert _scan(
        """\
        while pending():
            apply_fixes()
        """
    ) == [False]
    # The else clause is NOT dominated by the loop's own test ...
    assert _scan(
        """\
        while not dry_run:
            pass
        else:
            apply_fixes()
        """
    ) == [False]
    # ... but keeps guards that dominate the whole loop.
    assert _scan(
        """\
        if not dry_run:
            while pending():
                pass
            else:
                apply_fixes()
        """
    ) == [True]


def test_nested_def_and_lambda_reset_guards() -> None:
    """A nested ``def``/``lambda`` body runs at call time, not def time --
    an outer gate does not dominate it, wherever the def appears. A gate
    inside the nested body itself does still count."""
    assert _scan(
        """\
        if not dry_run:
            def helper():
                apply_fixes()
        """
    ) == [False]
    assert _scan(
        """\
        while pending():
            def helper():
                apply_fixes()
        """
    ) == [False]
    assert _scan(
        """\
        if fix:
            def helper():
                if not dry_run:
                    apply_fixes()
        """
    ) == [True]
    assert _scan(
        """\
        if not dry_run:
            callback = lambda: apply_fixes()
        """
    ) == [False]
    assert _scan("callback = lambda: apply_fixes() if not dry_run else None\n") == [True]


def test_alias_and_attribute_names_match() -> None:
    """``apply_drift_fixes`` (the orchestration import alias) and attribute
    calls (``self.apply_fixes``) name the same gate target."""
    assert _scan(
        """\
        def run():
            if fix and not dry_run:
                apply_drift_fixes()
        """
    ) == [True]
    assert _scan(
        """\
        def run():
            if fix and not dry_run:
                self.apply_fixes()
        """
    ) == [True]
    assert _scan("apply_drift_fixes()\n") == [False]
    assert _scan("self.apply_fixes()\n") == [False]
    # Other names and other attribute spellings are not sites.
    assert _scan("self.apply_fixes_async()\n") == []


def test_call_in_test_or_decorator_is_a_site() -> None:
    """A call inside an ``if``/``while`` test or a decorator expression is
    still a site -- it arrives at ``visit`` as the node itself, so it must
    be dispatched, not skipped over."""
    assert _scan("if apply_fixes():\n    pass\n") == [False]
    assert _scan("while apply_fixes():\n    pass\n") == [False]
    assert _scan("@apply_fixes()\ndef f():\n    pass\n") == [False]
    # Decorators evaluate at def time in the enclosing scope, so they keep
    # outer guards.
    assert _scan(
        """\
        if not dry_run:
            @apply_fixes()
            def f():
                pass
        """
    ) == [True]


def test_scan_reports_every_site_in_order() -> None:
    """Sites are reported in source order with per-site verdicts; asserting
    the whole list also proves the scan is non-empty."""
    assert _scan(
        """\
        if not dry_run:
            apply_fixes()
        apply_drift_fixes()
        """
    ) == [True, False]


def test_every_apply_fixes_call_site_is_dry_run_gated() -> None:
    """Issue #1475: every ``apply_fixes`` call site under ``src/charlie_work``
    must be dominated by a guard that proves ``dry_run`` is False.

    Issue #1051 made the caller-level ``not dry_run`` gate the single point
    of enforcement: ``apply_fixes`` takes no ``dry_run`` parameter, so the
    ``push_branch`` call inside its ``session_unpublished_work_salvaged``
    lane has no internal short-circuit. The only thing keeping a real
    ``git push`` from firing under ``fleet supervise --dry-run`` /
    ``mop-up --fix --dry-run`` is the caller-side gate -- and until this
    test, nothing enforced it. If the gate is dropped from
    ``_reconcile_locked`` or a new ``apply_fixes`` call site is added
    without one, this test fails.

    Recognised gate shapes (matching the codebase's idiom):

    * ``if <...> and not dry_run and <...>:`` -- ``not dry_run`` as an
      ``and``-conjunct of a dominating ``if``/``elif``/``while`` test (the
      shape ``_reconcile_locked`` uses);
    * the falsy branch of a test that has ``dry_run`` as an ``or``-disjunct
      (``if dry_run: ... else: <call>``, or a conditional expression).

    Deliberately NOT recognised (fail-closed): early ``if dry_run: return``
    statement guards, ``assert not dry_run``, and intra-expression
    short-circuit (``not dry_run and apply_fixes()`` inside a test
    expression). A call site gated that way fails this test and must be
    restructured into a recognised shape or the scanner extended -- a
    visible, reviewable decision rather than a silent hole.
    """
    violations: list[str] = []
    site_count = 0
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call, guards in _apply_fixes_call_sites(tree):
            site_count += 1
            if not _site_is_dry_run_gated(guards):
                rel = path.relative_to(_SRC_ROOT.parent)
                violations.append(f"{rel}:{call.lineno}")
    assert site_count > 0, (
        "scanner found no apply_fixes call sites -- the function was renamed "
        "or removed and this test must be updated, not silently vacated"
    )
    assert not violations, (
        "apply_fixes call site(s) not dominated by a `not dry_run` guard "
        f"(issue #1475): {violations}"
    )
