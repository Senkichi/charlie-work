"""Structural guard: per-repo lanes actually enforce repo retirement.

Issue #2754. Retirement is only enforced if every function that builds a
per-repo ``OrchestratorApp`` from the fleet registry also goes through
``_select_repos`` (which derives ``retired``) or
``_iter_registered_repo_contexts`` (which skips retired repos), *and* the
lane itself acts on the flag. A new lane that re-reads ``fleet.json`` and
constructs an app directly would silently reintroduce the defect this issue
exists to remove.

The scanned set is every ``.py`` under the ``charlie_work`` package; nothing
is listed by hand except the two allowed selectors, which are imported from
their live modules.

Evidence is AST-shaped, never textual. An earlier version counted string
constants, dict keys and keyword names as "references to retirement", so
``fleet_loop`` still passed after both its ``retired_pass_disposition``
block and its ``retired=`` kwarg were deleted -- the ``"retired"`` summary
key alone satisfied it. That is exactly the failure this module must catch,
so the checks below accept only:

* a direct call to a retirement helper (``retired_pass_disposition`` etc.),
* an ``if``/``while``/``if-exp`` that branches on a read of the ``retired``
  registry field (``entry.get("retired")`` or ``entry["retired"]``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import charlie_work
from charlie_work.fleet_registry import _iter_registered_repo_contexts, _select_repos
from _src_ast import parsed, source_files

PACKAGE_ROOT = Path(charlie_work.__file__).resolve().parent
SELECTORS = {_select_repos.__name__, _iter_registered_repo_contexts.__name__}

# Calling one of these *is* a retirement check. ``retired_pass_disposition``
# is the pass-level wind-down/skip decision; ``run_wind_down`` is the
# retirement-only lane body; the rest are the shared derivations.
RETIREMENT_CALLS = {
    "retired_pass_disposition",
    "run_wind_down",
    "entry_is_retired",
    "config_is_retired",
    "touched_retired_flag",
    "mark_selection_retired",
    "active_registry_repos",
}

# Every known per-repo consumer must prove its own check at its own call
# site. ``call`` requires a direct call to that helper (fleet_loop's
# wind-down dispatch must not be able to disappear behind a helper that is
# merely imported); ``field`` requires branching on the derived registry
# field. Keyed by ``module:qualname`` so a rename fails loudly instead of
# silently shrinking the checked set.
REQUIRED_EVIDENCE: dict[str, tuple[str, str]] = {
    "charlie_work.fleet_dispatch:fleet_loop": ("call", "retired_pass_disposition"),
    "charlie_work.fleet_lanes:_run_fleet_reap_sweep": ("field", "retired"),
    "charlie_work.cli:run_fleet_review_queue": ("field", "retired"),
    "charlie_work.cli:run_fleet_operator_queue": ("field", "retired"),
    "charlie_work.fleet_status:run_fleet_status": ("field", "retired"),
    # The config-retirement sweep consumes ``_select_repos`` output even
    # though it builds no app; a retired repo must not arm or block a
    # deprecated-key retirement.
    "charlie_work.config_retirement_sweep:_enumerate_layer_slots": ("field", "retired"),
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


def _same_scope_nodes(node: ast.AST):
    """Yield *node* and its descendants without descending into nested defs/classes."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield from _same_scope_nodes(child)


def _is_retired_key(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "retired"


def _reads_retired_field(node: ast.AST) -> bool:
    """Whether *node* reads the ``retired`` field through get/subscript/getattr."""
    for current in _same_scope_nodes(node):
        if not isinstance(current, ast.Call):
            continue
        func = current.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and current.args
            and _is_retired_key(current.args[0])
        ):
            return True
        if (
            isinstance(func, ast.Name)
            and func.id == "getattr"
            and len(current.args) >= 2
            and _is_retired_key(current.args[1])
        ):
            return True
        if isinstance(current, ast.Subscript) and _is_retired_key(current.slice):
            return True
    return False


def _branches_on_retired_field(node: ast.AST) -> bool:
    """Whether *node* has an if/while/ternary whose condition reads ``retired``."""
    return any(
        isinstance(current, (ast.If, ast.While, ast.IfExp)) and _reads_retired_field(current.test)
        for current in _same_scope_nodes(node)
    )


def _has_retirement_evidence(node: ast.AST) -> bool:
    return bool(_direct_calls(node) & RETIREMENT_CALLS) or _branches_on_retired_field(node)


def _lane_scopes() -> list[tuple[str, str, ast.AST, int]]:
    """``(module, qualname, node, line)`` for every app-building per-repo scope."""
    found: list[tuple[str, str, ast.AST, int]] = []
    for path in source_files(PACKAGE_ROOT):
        tree = parsed(path)
        module = _module_name(path)

        def visit(node: ast.AST, scope: list[str]) -> None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls = _direct_calls(node)
                if "_load_registry" in calls and "OrchestratorApp" in calls:
                    found.append((module, ".".join([*scope, node.name]), node, node.lineno))
                for child in ast.iter_child_nodes(node):
                    visit(child, [*scope, node.name])
            else:
                for child in ast.iter_child_nodes(node):
                    visit(child, scope)

        visit(tree, [])
    return found


def _scope_by_name(module: str, qualname: str) -> ast.AST:
    """Resolve one ``module:qualname`` to its function AST node.

    Used to check consumers that never build an app themselves (the config
    sweep's slot enumerator, ``fleet status``), which the discovery scan
    cannot classify as lanes.
    """
    for path in source_files(PACKAGE_ROOT):
        if _module_name(path) != module:
            continue
        tree = parsed(path)

        def visit(node: ast.AST, scope: list[str]) -> ast.AST | None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join([*scope, node.name])
                if name == qualname:
                    return node
                for child in ast.iter_child_nodes(node):
                    found = visit(child, [*scope, node.name])
                    if found is not None:
                        return found
            else:
                for child in ast.iter_child_nodes(node):
                    found = visit(child, scope)
                    if found is not None:
                        return found
            return None

        found = visit(tree, [])
        if found is not None:
            return found
    raise AssertionError(f"{module}:{qualname} not found -- update the guard's symbol")


def test_scan_sees_the_known_lane() -> None:
    """Positive control: the scan must find ``fleet_loop``'s app construction."""
    scopes = {(module, scope) for module, scope, _node, _line in _lane_scopes()}
    assert ("charlie_work.fleet_dispatch", "fleet_loop") in scopes


def test_every_per_repo_lane_uses_retirement_aware_selection() -> None:
    offenders = [
        f"{module}:{line} {scope}"
        for module, scope, node, line in _lane_scopes()
        if not (_direct_calls(node) & SELECTORS)
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
        for module, scope, node, line in _lane_scopes()
        if not _has_retirement_evidence(node)
    ]
    assert not offenders, (
        "per-repo lane(s) build OrchestratorApp without calling a retirement "
        "helper or branching on the retired field, so removing retirement "
        "enforcement would be invisible (issue #2754):\n  " + "\n  ".join(offenders)
    )


def test_required_lanes_prove_their_retirement_evidence() -> None:
    """Pin the exact AST evidence for every known consumer.

    ``fleet_loop`` must call ``retired_pass_disposition``; the reap sweep,
    queue commands, status and config sweep must branch on the retired
    field. A bare ``"retired"`` string constant or a ``retired=`` keyword
    does not count (see the module docstring).
    """
    failures: list[str] = []
    for symbol, (kind, expected) in REQUIRED_EVIDENCE.items():
        module, qualname = symbol.split(":", 1)
        node = _scope_by_name(module, qualname)
        if kind == "call":
            ok = expected in _direct_calls(node)
        else:
            ok = _branches_on_retired_field(node)
        if not ok:
            failures.append(f"{symbol} (expected {kind} evidence: {expected})")
    assert not failures, (
        "per-repo consumer(s) lost their retirement check (issue #2754):\n  "
        + "\n  ".join(failures)
    )
