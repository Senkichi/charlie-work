"""Shared AST matcher for the gh ``--json`` field-list lint (issue #1609).

The repo-wide scan (``tests/test_doctor.py::
test_gh_field_lists_use_constants_no_inline_literals``) and the synthetic-module
regression tests (``tests/test_field_list_lint.py``) both call
``_find_gh_field_list_violations`` so they exercise the SAME code path. A
regression test that duplicated the matcher logic would pass even if the real
matcher's shape-3 branch were reverted -- exactly the vacuous-pass gap #1609
is closing.

Lives in a ``_``-prefixed helper module (the established shared-test-helper
pattern: ``_fakes_github.py``, ``_reconcile_fixtures.py``, ...) so it can be
imported by both test modules without one test module importing another.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _scan_field_list_for_inline_literals(
    list_items: list[ast.AST], py_file: Path
) -> list[tuple[str, int, str]]:
    """Scan an ``ast.List``'s ``elts`` for an inline literal after ``--json``.

    Shared by the ``args=[...]`` keyword branch and the single-positional-list
    ``gh.run([...])`` branch (issue #1609): both feed a list of ``--json``-bearing
    string elements here, and both need the same ``--json``-then-literal rule.
    Returns one ``(file, lineno, literal)`` tuple per violation.
    """
    violations: list[tuple[str, int, str]] = []
    for i, item in enumerate(list_items):
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            if item.value == "--json" and i + 1 < len(list_items):
                next_item = list_items[i + 1]
                if isinstance(next_item, ast.Constant) and isinstance(next_item.value, str):
                    violations.append((str(py_file), next_item.lineno, next_item.value))
                elif isinstance(next_item, ast.JoinedStr):
                    violations.append((str(py_file), next_item.lineno, "f-string field list"))
    return violations


def _find_gh_field_list_violations(tree: ast.AST, py_file: Path) -> list[tuple[str, int, str]]:
    """Run the field-list lint matcher against one parsed module.

    Recognises three ``gh.run(...)`` call shapes (issue #1609 added the third):

    1. separate positional string arguments:
       ``gh.run("pr", "list", "--json", "number")``
    2. an ``args=`` keyword whose value is a list:
       ``gh.run(args=["pr", "list", "--json", "number"])``
    3. a single positional list argument:
       ``gh.run(["pr", "list", "--json", "number"], json_output=True)``
    """
    violations: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "run":
            continue
        args = node.args
        # Shape 1: separate positional string arguments.
        for i, arg in enumerate(args):
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value == "--json" and i + 1 < len(args):
                    next_arg = args[i + 1]
                    if isinstance(next_arg, ast.Constant) and isinstance(next_arg.value, str):
                        violations.append((str(py_file), next_arg.lineno, next_arg.value))
                    elif isinstance(next_arg, ast.JoinedStr):
                        violations.append((str(py_file), next_arg.lineno, "f-string field list"))
        # Shape 2: an ``args=`` keyword whose value is a list.
        for keyword in node.keywords:
            if keyword.arg in ("args",) and isinstance(keyword.value, ast.List):
                violations.extend(
                    _scan_field_list_for_inline_literals(keyword.value.elts, py_file)
                )
        # Shape 3 (issue #1609): a single positional list argument --
        # ``gh.run(["pr", "list", "--json", "number"], json_output=True)``.
        # For this shape ``node.args`` holds one ``ast.List``; the positional
        # loop above sees no ``ast.Constant``, and ``node.keywords`` has no
        # ``args=`` entry, so without this branch the call was skipped entirely.
        if args and isinstance(args[0], ast.List):
            violations.extend(_scan_field_list_for_inline_literals(args[0].elts, py_file))
    return violations
