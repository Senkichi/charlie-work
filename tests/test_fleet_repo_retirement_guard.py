"""Structural guard: per-repo lanes go through retirement-aware selection.

Issue #2754. Retirement is only enforced if every function that builds a
per-repo ``OrchestratorApp`` from the fleet registry also goes through
``_select_repos`` (which derives ``retired``) or
``_iter_registered_repo_contexts`` (which skips retired repos). A new lane
that re-reads ``fleet.json`` and constructs an app directly would silently
reintroduce the defect this issue exists to remove.

The scanned set is every ``.py`` under the ``charlie_work`` package; nothing
is listed by hand except the two allowed selectors, which are imported from
their live modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

import charlie_work
from charlie_work.fleet_registry import _iter_registered_repo_contexts, _select_repos
from _src_ast import parsed, source_files, source_text

PACKAGE_ROOT = Path(charlie_work.__file__).resolve().parent
SELECTORS = {_select_repos.__name__, _iter_registered_repo_contexts.__name__}
# A lane's early-return/skip check must reference one of these -- the derived
# registry field, the shared derivation, or the wind-down disposition call.
RETIREMENT_REFERENCES = {
    "retired",
    "retired_pass_disposition",
    "entry_is_retired",
    "config_is_retired",
    "touched_retired_flag",
    "run_wind_down",
}


def _module_name(path: Path) -> str:
    rel = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
    return ".".join(rel.parts)


def _direct_calls(node: ast.AST) -> set[str]:
    """Names called directly inside *node*, not inside nested defs/classes."""
    calls: set[str] = set()

    def visit(current: ast.AST) -> None:
        if current is not node and isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            return
        if isinstance(current, ast.Call):
            func = current.func
            if isinstance(func, ast.Name):
                calls.add(func.id)
            elif isinstance(func, ast.Attribute):
                calls.add(func.attr)
        for child in ast.iter_child_nodes(current):
            visit(child)

    visit(node)
    return calls


def _direct_references(node: ast.AST) -> set[str]:
    """Identifiers and non-docstring string constants referenced by *node*."""
    refs: set[str] = set()
    docstring = (
        ast.get_docstring(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        else None
    )
    first_stmt = node.body[0] if docstring is not None and node.body else None

    def visit(current: ast.AST) -> None:
        if current is not node and isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            return
        if current is first_stmt:
            return
        if isinstance(current, ast.Name):
            refs.add(current.id)
        elif isinstance(current, ast.Attribute):
            refs.add(current.attr)
        elif isinstance(current, ast.keyword) and current.arg:
            refs.add(current.arg)
        elif isinstance(current, ast.Constant) and isinstance(current.value, str):
            refs.add(current.value)
        for child in ast.iter_child_nodes(current):
            visit(child)

    visit(node)
    return refs


def _lane_scopes() -> list[tuple[str, str, set[str], set[str], int]]:
    """``(module, qualname, calls, references, line)`` for every app-building scope."""
    found: list[tuple[str, str, set[str], set[str], int]] = []
    for path in source_files(PACKAGE_ROOT):
        text = source_text(path)
        if "_load_registry" not in text or "OrchestratorApp" not in text:
            continue
        tree = parsed(path)
        module = _module_name(path)

        def visit(node: ast.AST, scope: list[str]) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls = _direct_calls(node)
                if "_load_registry" in calls and "OrchestratorApp" in calls:
                    found.append(
                        (
                            module,
                            ".".join([*scope, node.name]),
                            calls,
                            _direct_references(node),
                            node.lineno,
                        )
                    )
                for child in ast.iter_child_nodes(node):
                    visit(child, [*scope, node.name])
            else:
                for child in ast.iter_child_nodes(node):
                    visit(child, scope)

        visit(tree, [])
    return found


def test_scan_sees_the_known_lane() -> None:
    """Positive control: the scan must find ``fleet_loop``'s app construction."""
    scopes = {(module, scope) for module, scope, _calls, _refs, _line in _lane_scopes()}
    assert ("charlie_work.fleet_dispatch", "fleet_loop") in scopes


def test_every_per_repo_lane_uses_retirement_aware_selection() -> None:
    offenders = [
        f"{module}:{line} {scope}"
        for module, scope, calls, _refs, line in _lane_scopes()
        if not (calls & SELECTORS)
    ]
    assert not offenders, (
        "per-repo lane(s) construct OrchestratorApp from the fleet registry "
        "without going through _select_repos/_iter_registered_repo_contexts, "
        "bypassing repo retirement (issue #2754):\n  " + "\n  ".join(offenders)
    )


def test_every_per_repo_lane_actually_checks_retirement() -> None:
    """Removing a lane's own skip/wind-down check must fail this guard.

    Selecting a retired repo through ``_select_repos`` only labels it; each
    lane still needs its own early return (the full-loop skip, the reap
    sweep's skip, the queue commands' skip). The selector-only guard passed
    even when every one of those checks was deleted.
    """
    offenders = [
        f"{module}:{line} {scope}"
        for module, scope, _calls, refs, line in _lane_scopes()
        if not (refs & RETIREMENT_REFERENCES)
    ]
    assert not offenders, (
        "per-repo lane(s) build OrchestratorApp without referencing the "
        "retired field/retired_pass_disposition, so removing retirement "
        "enforcement would be invisible (issue #2754):\n  " + "\n  ".join(offenders)
    )
