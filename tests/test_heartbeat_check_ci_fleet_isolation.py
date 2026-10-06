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
import json
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from _src_ast import parsed

_HEARTBEAT_CHECK = Path(__file__).parent.parent / "scripts" / "heartbeat_check.py"
# Issue #1895: the events.db anomaly checks -- including the guarded
# ``charlie_work.event_kinds`` import that is this file's whole subject --
# now live in the sibling module heartbeat_check loads via importlib. The
# static scan below must cover BOTH files: heartbeat_check.py alone no
# longer contains the import surface a regression would land in.
_EVENT_ALARMS = Path(__file__).parent.parent / "scripts" / "heartbeat_event_alarms.py"
# Issue #2004 (PR #2023 rework): check_armable_backlog and its two guarded
# ``charlie_work`` leaf imports (``github_body_scan.parse_blockers``,
# ``config_deprecations.DEPRECATED_CONFIG_KEYS`` -- both module-scope leaves)
# live in this sibling; cover it for the same reason.
_ARMABLE_GATE = Path(__file__).parent.parent / "scripts" / "heartbeat_armable_gate.py"
# Issue #2048: same reasoning for the second extracted sibling --
# heartbeat_stale_mentions.py holds the stale-mention scanning/exclusion
# seam and is loaded via the same importlib pattern, so a forbidden import
# could equally land there.
_STALE_MENTIONS = Path(__file__).parent.parent / "scripts" / "heartbeat_stale_mentions.py"
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

    visit(parsed(path))
    return names


def test_heartbeat_check_never_imports_instrumentation_or_ci_fleet() -> None:
    """Static guard: no ci_fleet-reachable import at module scope, ever.

    Holds unconditionally -- no subprocess, no environment dependency -- so
    this is the guard that actually stops a regression; the runtime test
    below is stronger evidence (it proves real behaviour) but depends on
    ``ci_fleet`` being installed here to block in the first place.

    Since issue #1895 the guarded ``charlie_work.event_kinds`` import lives
    in ``heartbeat_event_alarms.py``, since #2004 the armable gate's guarded
    leaf imports live in ``heartbeat_armable_gate.py``, and since #2048 the
    stale-mention seam lives in ``heartbeat_stale_mentions.py`` -- all loaded
    via importlib, so the scan covers those siblings too: scanning
    heartbeat_check.py alone would no longer see the files where a forbidden
    import would land.
    """
    offenders = sorted(
        f"{path.name}: {name}"
        for path in (_HEARTBEAT_CHECK, _EVENT_ALARMS, _ARMABLE_GATE, _STALE_MENTIONS)
        for name in _module_scope_imports(path)
        if name == "ci_fleet"
        or name.startswith("ci_fleet.")
        or name == "charlie_work.instrumentation"
        or name.startswith("charlie_work.instrumentation.")
    )
    assert not offenders, (
        f"a heartbeat script imports a ci_fleet-reachable module at module scope: "
        f"{offenders}. These scripts must stay importable when ci_fleet is broken "
        "or absent -- import EXPECTED_OPERATIONAL_KINDS (or anything else shared "
        "with the package) from charlie_work.event_kinds, never from "
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


def _clean_env() -> dict[str, str]:
    """The environment minus ci-fleet's map-mode child hook.

    During a map build (the nightly) each test's process carries
    ``CI_FLEET_MAP_NODEID``; ``ci_fleet_probe.pth`` then imports ``ci_fleet`` at
    interpreter startup, before any meta-path blocker in ``-c`` code can run, so
    ``ci_fleet`` is never "absent" in the child. These tests prove import
    isolation, which map-mode attribution is irrelevant to.
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("CI_FLEET_MAP")}


def _run_blocked(body: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=_clean_env(),
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
        env=_clean_env(),
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
        env=_clean_env(),
    )


def test_the_charlie_work_blocker_actually_blocks() -> None:
    """Positive control -- same shape as ``test_the_ci_fleet_blocker_actually_blocks``."""
    control = subprocess.run(
        [sys.executable, "-c", "import charlie_work; print('PRESENT')"],
        capture_output=True,
        text=True,
        env=_clean_env(),
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


# --- file-path leaf fallback (charlie_work absent, repo src/ present) ---------

_SUPERVISOR_PROBE = """
import importlib.util, sys
from datetime import datetime, timezone

spec = importlib.util.spec_from_file_location("heartbeat_check", r"{script}")
module = importlib.util.module_from_spec(spec)
sys.modules["heartbeat_check"] = module
spec.loader.exec_module(module)
report = module.Report()
module.check_supervisor_heartbeat(report)
for line in report.lines:
    print(line)
"""


def _write_heartbeat(fleet: Path, last_beat: datetime, exited_at: str | None = None) -> None:
    fleet.mkdir(parents=True, exist_ok=True)
    (fleet / "supervisor-heartbeat.json").write_text(
        json.dumps(
            {
                "last_beat_at": last_beat.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "pid": 7,
                "exited_at": exited_at,
                "max_pass_runtime_seconds": 1800,
            }
        ),
        encoding="utf-8",
    )


def _run_probe(script: Path, fleet: Path) -> "subprocess.CompletedProcess[str]":
    env = {**os.environ, "CHARLIE_WORK_FLEET_DIR": str(fleet)}
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _CHARLIE_WORK_BLOCKER + textwrap.dedent(_SUPERVISOR_PROBE.format(script=script)),
        ],
        capture_output=True,
        text=True,
        env=env,
    )


def test_blocked_charlie_work_still_yields_real_supervisor_verdicts(tmp_path: Path) -> None:
    """Wrong-venv scenario: the leaves load by file path, so formerly-stdlib
    checks keep REAL verdicts instead of blanket "cannot evaluate"."""
    fresh = tmp_path / "fresh"
    _write_heartbeat(fresh, datetime.now(timezone.utc) - timedelta(minutes=5))
    ok = _run_probe(_HEARTBEAT_CHECK, fresh)
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.startswith("OK supervisor-heartbeat:"), ok.stdout
    assert "cannot evaluate" not in ok.stdout

    stale = tmp_path / "stale"
    _write_heartbeat(stale, datetime.now(timezone.utc) - timedelta(days=3))
    bad = _run_probe(_HEARTBEAT_CHECK, stale)
    assert bad.returncode == 0, bad.stderr
    assert "ANOMALY supervisor-heartbeat" in bad.stdout, bad.stdout
    assert "cannot evaluate" not in bad.stdout


def test_both_import_paths_unavailable_reports_loud_anomaly(tmp_path: Path) -> None:
    """With the package blocked AND no ``src/charlie_work`` beside the script,
    the check reports the cannot-evaluate ANOMALY (never silently green)."""
    scripts = tmp_path / "repo" / "scripts"
    scripts.mkdir(parents=True)
    for sibling in _HEARTBEAT_CHECK.parent.glob("heartbeat_*.py"):
        shutil.copy(sibling, scripts / sibling.name)
    fleet = tmp_path / "fleet"
    _write_heartbeat(fleet, datetime.now(timezone.utc) - timedelta(minutes=5))
    result = _run_probe(scripts / "heartbeat_check.py", fleet)
    assert result.returncode == 0, result.stderr
    assert "ANOMALY supervisor-heartbeat" in result.stdout, result.stdout
    assert "cannot evaluate" in result.stdout
