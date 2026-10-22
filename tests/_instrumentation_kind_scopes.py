"""Scope-assignment collection for the emit-site kind scanner (#2751).

Extracted from ``tests/_instrumentation_kind_scanner.py`` so that file stays
under the repo's 800-line module cap while the scanner keeps its deferred
assignment walk. ``LazyScopeAssignments`` materialises a function's
``_collect_local_assignments`` map only on the first kind resolution that
needs it; a whole-tree scan otherwise pays that walk for every function, most
of which hold no emit call.

The ``tests/_*.py`` hoisted-helper convention is the sanctioned import target
for shared test helpers (see ``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

import ast


def _is_scope_boundary(node: ast.AST) -> bool:
    """True for nodes that start a new variable-assignment scope."""
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))


def _collect_local_assignments(scope: ast.AST) -> dict[str, list[ast.expr]]:
    """Every ``name = <expr>`` assigned directly within ``scope``.

    Descends through control flow (if/for/while/try/with) but stops at any
    nested function, lambda, or class body -- those are separate scopes with
    their own assignment map, not this one.
    """
    assigns: dict[str, list[ast.expr]] = {}

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if _is_scope_boundary(child):
                continue
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        assigns.setdefault(target.id, []).append(child.value)
            walk(child)

    walk(scope)
    return assigns


class LazyScopeAssignments:
    """Deferred ``_collect_local_assignments`` for one function scope (#2751).

    Collecting a function's assignments eagerly walks its whole subtree even
    when the function holds no emit call -- the common case across ``src/``.
    The map is materialised on the first ``kind`` resolution that actually
    needs it; every other reference only asks ``in`` / ``[]``, which trigger
    the same one-time walk. Behaves like the dict it wraps.
    """

    def __init__(self, scope: ast.AST) -> None:
        self._scope = scope
        self._assigns: dict[str, list[ast.expr]] | None = None

    def _materialize(self) -> dict[str, list[ast.expr]]:
        if self._assigns is None:
            self._assigns = _collect_local_assignments(self._scope)
        return self._assigns

    def __contains__(self, name: object) -> bool:
        return name in self._materialize()

    def __getitem__(self, name: str) -> list[ast.expr]:
        return self._materialize()[name]


LocalAssignments = dict[str, list[ast.expr]] | LazyScopeAssignments
