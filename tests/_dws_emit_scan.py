"""The dead-worker sweep's emit forms, recognised in one place.

The sweep's decide modules build events through ``emit("<kind>", ...)`` and the stalled
lane names its event with ``StateTxn(event_kind="<kind>")``. Both the sweep-local kind
check (``test_dws_event_kinds.py``) and the repo-wide consumer scanner
(``test_event_kind_consumers.py``) classify nodes through ``sweep_emit_kind`` so the two
cannot disagree about what counts as an emit site.
"""

from __future__ import annotations

import ast

SWEEP_PACKAGE_DIR = "dead_worker_sweep"


def is_decide_module(rel_path: str) -> bool:
    """True for ``dead_worker_sweep/decide*.py`` (POSIX path relative to ``charlie_work``)."""
    parent, _, name = rel_path.rpartition("/")
    return parent == SWEEP_PACKAGE_DIR and name.startswith("decide") and name.endswith(".py")


def literals(expr: ast.expr, tree: ast.Module) -> set[str] | None:
    """String literals ``expr`` can evaluate to, following one local assignment; else None."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return {expr.value}
    if isinstance(expr, ast.IfExp):
        body, orelse = literals(expr.body, tree), literals(expr.orelse, tree)
        return None if body is None or orelse is None else body | orelse
    if isinstance(expr, ast.Name):
        values = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == expr.id for t in node.targets)
        ]
        resolved = [literals(v, tree) for v in values]
        if values and all(r is not None for r in resolved):
            return set().union(*resolved)  # type: ignore[arg-type]
    return None


def _has_level(call: ast.Call) -> bool:
    return len(call.args) > 2 or any(kw.arg == "level" for kw in call.keywords)


def sweep_emit_kind(node: ast.AST, filename: str) -> tuple[ast.expr, bool] | None:
    """``(kind expression, carries an explicit level)`` if ``node`` is a sweep emit site.

    Sites are ``emit(<kind>, ...)`` calls and ``event_kind=<kind>`` keywords. In
    ``decide_common.py`` a bare-name kind is ``events()`` forwarding the kind its caller
    already chose, which is not a site.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "emit"
        and node.args
    ):
        kind_node, leveled = node.args[0], _has_level(node)
    elif isinstance(node, ast.keyword) and node.arg == "event_kind":
        kind_node, leveled = node.value, False
    else:
        return None
    if filename == "decide_common.py" and isinstance(kind_node, ast.Name):
        return None
    return kind_node, leveled
