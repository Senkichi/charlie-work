"""Structural guard: ``_launch_workers`` is the only ``dispatch_sessions`` caller.

Issue #2041. Fleet-wide worker limits (provider throttle, fleet lock,
concurrency governor) are enforced by ``worker_launch_gate``'s permit, and
``_launch_workers`` is the only code that launches against it. A lane that
calls ``dispatch_sessions`` directly bypasses every limit -- the #2039 defect
shape, where the local rework lane launched Devin workers uncounted against
the fleet cap. This scan fails CI on any new direct reference.

Nothing is listed by hand: the allowed caller, the definition site, and the
re-export site are all derived from the live objects, and the scanned set is
every ``.py`` under the ``charlie_work`` package.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import charlie_work
import charlie_work.adapters as adapters
import charlie_work.workflow as workflow
from charlie_work.worker_launch_gate import _launch_workers
from _src_ast import parsed, source_files

TARGET = adapters.dispatch_sessions.__name__
PACKAGE_ROOT = Path(charlie_work.__file__).resolve().parent


def _module_name(path: Path) -> str:
    rel = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    return ".".join(rel.parts)


def _references(path: Path) -> list[tuple[str, str, int]]:
    """``(module, enclosing function qualname or "<module>", line)`` for every
    reference to TARGET: a bare name, an attribute, an imported alias, or the
    name as a string constant (``getattr(mod, "dispatch_sessions")``)."""
    tree = parsed(path)
    module = _module_name(path)
    found: list[tuple[str, str, int]] = []

    def visit(node: ast.AST, scope: list[str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = [*scope, node.name]
        hit = (
            (isinstance(node, ast.Name) and node.id == TARGET)
            or (isinstance(node, ast.Attribute) and node.attr == TARGET)
            or (isinstance(node, ast.Constant) and node.value == TARGET)
            or (
                isinstance(node, (ast.Import, ast.ImportFrom))
                and any(a.name.rsplit(".", 1)[-1] == TARGET for a in node.names)
            )
        )
        if hit:
            found.append((module, ".".join(scope) or "<module>", node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, [])
    return found


def _all_references() -> list[tuple[str, str, int]]:
    refs: list[tuple[str, str, int]] = []
    for path in source_files(PACKAGE_ROOT):
        refs.extend(_references(path))
    return refs


def _allowed() -> set[tuple[str, str]]:
    # The one launch point, derived from the function object itself.
    allowed = {(_launch_workers.__module__, _launch_workers.__qualname__)}
    # The workflow module's deliberate re-export (the name every test fake
    # patches) -- a module-level import, derived by identity, not by name.
    assert workflow.dispatch_sessions is adapters.dispatch_sessions
    allowed.add((workflow.__name__, "<module>"))
    return allowed


def test_scan_sees_the_known_launch_point() -> None:
    """Positive control: the scan must find ``_launch_workers``' own call --
    an empty scan would pass the guard below vacuously."""
    scopes = {(module, scope) for module, scope, _line in _all_references()}
    assert (_launch_workers.__module__, _launch_workers.__qualname__) in scopes
    # And the definition is where the object says it is (a def, not a reference).
    assert inspect.getmodule(adapters.dispatch_sessions) is adapters


def test_only_launch_workers_references_dispatch_sessions() -> None:
    allowed = _allowed()
    offenders = [
        f"{module}:{line} in {scope}"
        for module, scope, line in _all_references()
        if (module, scope) not in allowed
    ]
    assert not offenders, (
        f"direct {TARGET} reference(s) bypass the worker launch permit "
        "(issue #2041) -- launch through worker_launch_gate._launch_workers "
        "with a permit from issue_worker_launch_permit instead:\n  " + "\n  ".join(offenders)
    )
