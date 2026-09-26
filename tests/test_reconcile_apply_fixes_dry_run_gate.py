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
"""

from __future__ import annotations

import ast
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
    a nested ``def``'s body (decorators and default expressions DO evaluate
    at def-time in the outer scope, so they keep the outer guards).
    """
    sites: list[tuple[ast.Call, tuple[tuple[ast.expr, bool], ...]]] = []

    def visit(node: ast.AST, guards: tuple[tuple[ast.expr, bool], ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in child.decorator_list:
                    visit(decorator, guards)
                visit(child.args, guards)
                for stmt in child.body:
                    visit(stmt, ())
            elif isinstance(child, ast.Lambda):
                visit(child.args, guards)
                visit(child.body, ())
            elif isinstance(child, ast.If):
                visit(child.test, guards)
                for stmt in child.body:
                    visit(stmt, (*guards, (child.test, True)))
                # An ``elif`` is a nested ``If`` inside ``orelse`` and is
                # visited through this same path, inheriting both guards.
                for stmt in child.orelse:
                    visit(stmt, (*guards, (child.test, False)))
            elif isinstance(child, ast.IfExp):
                # Unlike ``ast.If``, body/orelse are single expressions.
                visit(child.test, guards)
                visit(child.body, (*guards, (child.test, True)))
                visit(child.orelse, (*guards, (child.test, False)))
            elif isinstance(child, ast.While):
                visit(child.test, guards)
                for stmt in child.body:
                    visit(stmt, (*guards, (child.test, True)))
                # ``while ... else`` runs after loop completion -- the test
                # no longer dominates -- so orelse keeps the outer guards.
                for stmt in child.orelse:
                    visit(stmt, guards)
            else:
                if isinstance(child, ast.Call) and _apply_fixes_call_name(child):
                    sites.append((child, guards))
                visit(child, guards)

    visit(tree, ())
    return sites


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
            gated = any(
                _truthy_test_excludes_dry_run(test) if held else _falsy_test_excludes_dry_run(test)
                for test, held in guards
            )
            if not gated:
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
