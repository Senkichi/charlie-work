"""``scripts/heartbeat_check.py`` must import even when ``ci_fleet`` -- or
``charlie_work`` itself -- is absent.

Review finding on issue #1271: the initial fix declared
``EXPECTED_OPERATIONAL_KINDS`` in ``charlie_work.instrumentation``, which
imports ``ci_fleet.observability``/``ci_fleet.provenance`` at module load
(see that module's own comment near the bottom of the file). Having
``heartbeat_check.py`` import the constant from ``instrumentation`` meant a
broken ``ci_fleet`` install turned an intended ANOMALY line into an
unhandled ``ImportError`` -- on exactly the failure class this stdlib-only
script exists to report (``scripts/README.md``). The fix moved the
frozenset to ``charlie_work.event_kinds``, a genuine leaf module with no
``charlie_work``/``ci_fleet`` imports of its own, and re-exports it from
``instrumentation`` for in-package consumers.

That leaf-module fix closes the ``ci_fleet``-broken case, but the finding
also named "no test covering a charlie_work-less run" -- a distinct,
narrower failure mode where ``charlie_work`` itself (not just its
``ci_fleet`` dependency) is not importable. This is not theoretical here:
this script is routinely invoked via ``uv run --active``, which resolves
against whatever venv happens to be active rather than this project's own,
and the ``uv-worktree-virtualenv-shadowing`` project memory documents that
resolving to a wrong venv -- one without ``charlie_work`` installed at all
-- has actually happened in this fleet. ``heartbeat_check.py`` guards the
``charlie_work.event_kinds`` import with ``try/except ImportError``,
degrading to an empty ``EXPECTED_OPERATIONAL_KINDS`` (the exact pre-#1271
behavior: no bucketing, every warning kind in the flat detailed list)
instead of crashing.

Modeled on ``tests/test_cli_import_isolation.py``'s ci_fleet-confinement
pattern: static AST checks that always run regardless of environment, plus
subprocess runtime checks (meta-path blockers making a package genuinely
absent, not merely unimported) that prove both real failure modes are gone.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

_HEARTBEAT_CHECK = Path(__file__).parent.parent / "scripts" / "heartbeat_check.py"
_EVENT_KINDS = Path(__file__).parent.parent / "src" / "charlie_work" / "event_kinds.py"


def _module_scope_imports(path: Path) -> set[str]:
    """Every module name imported at module scope, ignoring function/class bodies.

    A module-scope ``try:``/``if:`` still executes at import, so this descends
    into those; it stops only at ``def``/``class``. Mirrors the helper in
    ``tests/test_cli_import_isolation.py``.
    """
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(path.read_text(encoding="utf-8")))
    return names


def test_heartbeat_check_never_imports_instrumentation_or_ci_fleet() -> None:
    """Static guard: no ci_fleet-reachable import at module scope, ever.

    Holds unconditionally -- no subprocess, no environment dependency -- so
    this is the guard that actually stops a regression; the runtime test
    below is stronger evidence (it proves real behaviour) but depends on
    ``ci_fleet`` being installed here to block in the first place.
    """
    offenders = sorted(
        name
        for name in _module_scope_imports(_HEARTBEAT_CHECK)
        if name == "ci_fleet"
        or name.startswith("ci_fleet.")
        or name == "charlie_work.instrumentation"
        or name.startswith("charlie_work.instrumentation.")
    )
    assert not offenders, (
        f"heartbeat_check.py imports a ci_fleet-reachable module at module scope: "
        f"{offenders}. This script must stay importable when ci_fleet is broken or "
        "absent -- import EXPECTED_OPERATIONAL_KINDS (or anything else shared with "
        "the package) from charlie_work.event_kinds, never from "
        "charlie_work.instrumentation."
    )


def test_event_kinds_is_a_genuine_leaf_module() -> None:
    """``charlie_work.event_kinds`` must import nothing beyond stdlib.

    This is the property the runtime test below relies on: if ``event_kinds``
    ever grows an import of its own, it could silently reintroduce the exact
    ci_fleet reachability this module exists to avoid.
    """
    offenders = sorted(
        name for name in _module_scope_imports(_EVENT_KINDS) if name != "__future__"
    )
    assert not offenders, (
        f"charlie_work/event_kinds.py imports {offenders} -- it must stay stdlib-only "
        "(no charlie_work or ci_fleet imports) so heartbeat_check.py can import "
        "EXPECTED_OPERATIONAL_KINDS from it unconditionally."
    )


_BLOCKER = '''
import sys


class _CiFleetBlocker:
    """Make ci_fleet look genuinely absent, not merely unimported."""

    def find_spec(self, name, path=None, target=None):
        if name == "ci_fleet" or name.startswith("ci_fleet."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, _CiFleetBlocker())
'''


def _run_blocked(body: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_the_ci_fleet_blocker_actually_blocks() -> None:
    """Positive control, written as a differential rather than a single result.

    A one-sided assertion is worthless here: the control has to show
    ``import ci_fleet`` succeeding without the blocker and failing with it,
    matching ``test_cli_import_isolation.py``'s
    ``test_the_blocker_actually_blocks``.
    """
    control = subprocess.run(
        [sys.executable, "-c", "import ci_fleet; print('PRESENT')"],
        capture_output=True,
        text=True,
    )
    assert control.returncode == 0 and "PRESENT" in control.stdout, (
        "fixture premise gone: ci_fleet is not importable even without the blocker, "
        f"so this file can no longer prove anything.\n{control.stderr}"
    )

    blocked = _run_blocked("import ci_fleet")
    assert blocked.returncode != 0, "blocker did not block; the runtime test below is vacuous"
    assert "ci_fleet" in blocked.stderr


def test_heartbeat_check_imports_with_ci_fleet_absent() -> None:
    """The regression test: heartbeat_check.py must load with ci_fleet gone.

    Reproduces the exact scenario the review finding named -- a broken
    charlie_work/ci_fleet install -- by making ``ci_fleet`` genuinely
    unimportable (a meta-path blocker) rather than deleting it from
    ``sys.modules``, which would just re-import the real thing on next use.
    """
    result = _run_blocked(
        f"""
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "heartbeat_check", r"{_HEARTBEAT_CHECK}"
        )
        module = importlib.util.module_from_spec(spec)
        # Registered before exec_module -- required for `from __future__ import
        # annotations` dataclasses to resolve their string annotations during
        # class creation (issue #1023; see tests/_script_loader.py).
        sys.modules["heartbeat_check"] = module
        spec.loader.exec_module(module)
        assert module.EXPECTED_OPERATIONAL_KINDS, "frozenset must be non-empty"
        print("IMPORT_OK", sorted(module.EXPECTED_OPERATIONAL_KINDS))
        """
    )
    assert result.returncode == 0, (
        "heartbeat_check.py failed to import with ci_fleet absent -- exactly the "
        f"failure class it exists to report:\n{result.stderr}"
    )
    assert "IMPORT_OK" in result.stdout


_CHARLIE_WORK_BLOCKER = '''
import sys


class _CharlieWorkBlocker:
    """Make charlie_work look genuinely absent, not merely unimported."""

    def find_spec(self, name, path=None, target=None):
        if name == "charlie_work" or name.startswith("charlie_work."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, _CharlieWorkBlocker())
'''


def _run_charlie_work_blocked(body: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-c", _CHARLIE_WORK_BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_the_charlie_work_blocker_actually_blocks() -> None:
    """Positive control -- same shape as ``test_the_ci_fleet_blocker_actually_blocks``."""
    control = subprocess.run(
        [sys.executable, "-c", "import charlie_work; print('PRESENT')"],
        capture_output=True,
        text=True,
    )
    assert control.returncode == 0 and "PRESENT" in control.stdout, (
        "fixture premise gone: charlie_work is not importable even without the "
        f"blocker, so this file can no longer prove anything.\n{control.stderr}"
    )

    blocked = _run_charlie_work_blocked("import charlie_work")
    assert blocked.returncode != 0, "blocker did not block; the runtime test below is vacuous"
    assert "charlie_work" in blocked.stderr


def test_heartbeat_check_imports_with_charlie_work_absent() -> None:
    """The narrower regression test the finding also named: heartbeat_check.py
    must load with ``charlie_work`` itself gone, not just ``ci_fleet``.

    This is the scenario ``uv run --active`` can produce in this fleet (see
    the module docstring): a venv with no ``charlie_work`` install at all.
    Unlike the ci_fleet-absent case, the frozenset must degrade to *empty*
    here (the pre-#1271 flat-listing behavior), not stay populated -- there
    is no source left to populate it from.
    """
    result = _run_charlie_work_blocked(
        f"""
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "heartbeat_check", r"{_HEARTBEAT_CHECK}"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["heartbeat_check"] = module
        spec.loader.exec_module(module)
        assert module.EXPECTED_OPERATIONAL_KINDS == frozenset(), (
            "must degrade to empty, not raise, not stay populated: "
            f"{{module.EXPECTED_OPERATIONAL_KINDS!r}}"
        )
        print("IMPORT_OK")
        """
    )
    assert result.returncode == 0, (
        "heartbeat_check.py failed to import with charlie_work absent -- exactly "
        f"the failure class the finding named:\n{result.stderr}"
    )
    assert "IMPORT_OK" in result.stdout
