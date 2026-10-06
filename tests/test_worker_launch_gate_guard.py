"""Structural guard: ``_launch_workers`` is the only worker-launch-port caller.

Issue #2041. Fleet-wide worker limits (provider throttle, fleet lock,
concurrency governor) are enforced by ``worker_launch_gate``'s permit, and
``_launch_workers`` is the only code that launches against it. A lane that
touches the ``worker_launch`` port -- or references ``dispatch_sessions``
directly, bypassing the port (issue #2229) -- skips every limit: the #2039
defect shape, where the local rework lane launched Devin workers uncounted
against the fleet cap. This scan fails CI on any new reference.

Nothing is listed by hand: the allowed caller, the port's Real binding, the
definition site, and the re-export site are all derived from the live
objects, and the scanned set is every ``.py`` under the ``charlie_work``
package.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import charlie_work
import charlie_work.adapters as adapters
import charlie_work.workflow as workflow
from charlie_work.host import RealWorkerLauncher
from charlie_work.worker_launch_gate import _launch_workers
from _src_ast import parsed, source_files

TARGET = adapters.dispatch_sessions.__name__
PACKAGE_ROOT = Path(charlie_work.__file__).resolve().parent


def _module_name(path: Path) -> str:
    rel = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    return ".".join(rel.parts)


def _references(path: Path, target: str) -> list[tuple[str, str, int]]:
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
            (isinstance(node, ast.Name) and node.id == target)
            or (isinstance(node, ast.Attribute) and node.attr == target)
            or (isinstance(node, ast.Constant) and node.value == target)
            or (
                isinstance(node, (ast.Import, ast.ImportFrom))
                and any(a.name.rsplit(".", 1)[-1] == target for a in node.names)
            )
        )
        if hit:
            found.append((module, ".".join(scope) or "<module>", node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, [])
    return found


def _attribute_reaches(path: Path, attr: str) -> list[tuple[str, str, int]]:
    """``(module, enclosing qualname, line)`` for every ``.<attr>`` attribute
    reach or ``getattr``-style string constant -- the shapes a port call takes.
    A bare ``name`` (a dataclass field like ``HostPorts.worker_launch``) is
    deliberately not a hit: only ``.worker_launch`` accesses bypass the gate.
    """
    tree = parsed(path)
    module = _module_name(path)
    found: list[tuple[str, str, int]] = []

    def visit(node: ast.AST, scope: list[str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = [*scope, node.name]
        if (isinstance(node, ast.Attribute) and node.attr == attr) or (
            isinstance(node, ast.Constant) and node.value == attr
        ):
            found.append((module, ".".join(scope) or "<module>", node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, [])
    return found


def _all_references(target: str) -> list[tuple[str, str, int]]:
    refs: list[tuple[str, str, int]] = []
    for path in source_files(PACKAGE_ROOT):
        refs.extend(_references(path, target))
    return refs


def _all_attribute_reaches(attr: str) -> list[tuple[str, str, int]]:
    refs: list[tuple[str, str, int]] = []
    for path in source_files(PACKAGE_ROOT):
        refs.extend(_attribute_reaches(path, attr))
    return refs


def _allowed() -> set[tuple[str, str]]:
    # The one launch point, derived from the function object itself: the gate
    # may still reach ``dispatch_sessions`` directly if it ever needs to.
    allowed = {(_launch_workers.__module__, _launch_workers.__qualname__)}
    # The port's Real binds ``workflow.dispatch_sessions`` at call time so
    # patches on that name keep intercepting (issue #2229).
    real_launch = RealWorkerLauncher.launch
    allowed.add((inspect.getmodule(real_launch).__name__, real_launch.__qualname__))
    # The workflow module's deliberate re-export (the name every test fake
    # patches) -- a module-level import, derived by identity, not by name.
    assert workflow.dispatch_sessions is adapters.dispatch_sessions
    allowed.add((workflow.__name__, "<module>"))
    return allowed


def test_scan_sees_the_known_launch_point() -> None:
    """Positive control: the scan must find the port Real's own call -- an
    empty scan would pass the guard below vacuously."""
    scopes = {(module, scope) for module, scope, _line in _all_references(TARGET)}
    real_launch = RealWorkerLauncher.launch
    assert (inspect.getmodule(real_launch).__name__, real_launch.__qualname__) in scopes
    # And the definition is where the object says it is (a def, not a reference).
    assert inspect.getmodule(adapters.dispatch_sessions) is adapters


def test_only_launch_workers_references_dispatch_sessions() -> None:
    """The leaf name predates #2229: ``launch_workers`` denotes the whole
    worker-launch path -- the gate's ``_launch_workers`` plus the port's
    ``RealWorkerLauncher.launch``, which is what ``_allowed()`` derives. The
    name is kept verbatim because the collect-only gate (#1538) fails a
    required check on any leaf-name removal, rename included, absent an
    operator exemption label."""
    allowed = _allowed()
    offenders = [
        f"{module}:{line} in {scope}"
        for module, scope, line in _all_references(TARGET)
        if (module, scope) not in allowed
    ]
    assert not offenders, (
        f"direct {TARGET} reference(s) bypass the worker launch permit "
        "(issue #2041) -- launch through worker_launch_gate._launch_workers "
        "with a permit from issue_worker_launch_permit instead:\n  " + "\n  ".join(offenders)
    )


def test_only_launch_workers_touches_the_worker_launch_port() -> None:
    """Any ``.worker_launch`` attribute reach outside ``_launch_workers`` --
    e.g. ``app.host.worker_launch.launch(...)`` in a new lane -- bypasses the
    permit entirely."""
    allowed = {(_launch_workers.__module__, _launch_workers.__qualname__)}
    reaches = _all_attribute_reaches("worker_launch")
    # Positive control: the scan must find the gate's own port call -- an
    # empty scan would pass the assert below vacuously.
    assert (_launch_workers.__module__, _launch_workers.__qualname__) in {
        (module, scope) for module, scope, _line in reaches
    }
    offenders = [
        f"{module}:{line} in {scope}"
        for module, scope, line in reaches
        if (module, scope) not in allowed
    ]
    assert not offenders, (
        "worker_launch port reference(s) bypass the worker launch permit "
        "(issues #2041, #2229) -- launch through "
        "worker_launch_gate._launch_workers instead:\n  " + "\n  ".join(offenders)
    )
