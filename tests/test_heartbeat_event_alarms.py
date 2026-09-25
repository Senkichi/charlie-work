"""Extraction-contract tests for ``scripts/heartbeat_event_alarms.py`` (issue #1895).

The five per-repo ``events.db`` kind/level anomaly checks
(``check_error_events``, ``check_warning_events``,
``check_infra_blocked_events``, ``check_draft_pr_blocked_events``,
``check_ci_headroom_unavailable``) were extracted verbatim out of
``scripts/heartbeat_check.py`` into the sibling module
``scripts/heartbeat_event_alarms.py`` for file-size-ratchet headroom.
``heartbeat_check`` loads the sibling via ``importlib`` from its own
directory and re-exports all five names plus ``EXPECTED_OPERATIONAL_KINDS``.

The behavioral coverage in ``tests/test_heartbeat_check_event_db_checks.py``
and ``tests/test_heartbeat_event_consumers.py`` keeps exercising the moved
functions through those re-exports unchanged. This file pins the extraction
wiring itself -- the part no pre-existing test could see:

* the sibling module loads standalone through the same importlib recipe
  ``heartbeat_check`` uses (stdlib-only posture; the guarded
  ``charlie_work.event_kinds`` leaf import moved here with
  ``check_warning_events``),
* each ``hb.check_*`` attribute IS the sibling's function object -- a local
  ``def`` re-added in ``heartbeat_check.py`` would silently shadow the
  re-export and split the tested object from the deployed one,
* the mirrored ``parse_iso`` stays byte-identical to ``heartbeat_check``'s
  (the ISO-vs-SQLite timestamp convention depends on the copies agreeing),
* the sibling never back-imports ``heartbeat_check`` (that would cycle
  through its loader block) and keeps its one sanctioned package import
  behind ``try``/``except ImportError``,
* the checks run end-to-end on duck-typed report/repo objects --
  ``Report``/``RepoInfo`` are ``TYPE_CHECKING``-only names in the sibling,
  so no ``heartbeat_check`` internals may be required at runtime,
* ``check_ci_headroom_unavailable`` (issue #1770) gets its first direct
  coverage: it had emitter-side tests only -- the payload ``reason``
  bucketing in the reader was previously exercised by nothing.
"""

from __future__ import annotations

import ast
import inspect
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from _heartbeat_check_fixtures import _iso, _load_heartbeat_check, _write_events_db
from _script_loader import load_script_module

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
_HEARTBEAT_CHECK = _SCRIPTS_DIR / "heartbeat_check.py"
_EVENT_ALARMS = _SCRIPTS_DIR / "heartbeat_event_alarms.py"

_CHECK_NAMES = (
    "check_error_events",
    "check_warning_events",
    "check_infra_blocked_events",
    "check_draft_pr_blocked_events",
    "check_ci_headroom_unavailable",
)


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


@pytest.fixture(scope="module")
def alarms() -> ModuleType:
    """Load ``heartbeat_event_alarms.py`` the same way ``heartbeat_check``
    does -- via ``importlib`` from the script path, never a bare ``import``
    (``scripts/`` is not a package and stays off ``sys.path``)."""
    return load_script_module(_EVENT_ALARMS, "heartbeat_event_alarms")


# ---------------------------------------------------------------------------
# Loader wiring: the sibling loads standalone and heartbeat_check re-exports
# its objects, not shadowing local definitions.
# ---------------------------------------------------------------------------


def test_sibling_module_loads_standalone_and_defines_all_five_checks(
    alarms: ModuleType,
) -> None:
    for name in _CHECK_NAMES:
        assert callable(getattr(alarms, name, None)), (
            f"heartbeat_event_alarms.py no longer defines {name} -- "
            "heartbeat_check.py's re-export of it would AttributeError at import"
        )
    assert callable(alarms.parse_iso)
    assert isinstance(alarms.EXPECTED_OPERATIONAL_KINDS, frozenset)


def test_heartbeat_check_reexports_are_the_sibling_objects(hb: ModuleType) -> None:
    """``hb.check_*`` must BE the sibling's function, not merely an
    equivalent-looking object. Identity is the only check that catches a
    local ``def`` written after the re-export block (it would shadow the
    re-export while the sibling's copy stayed tested-but-undeployed), so
    ``__code__.co_filename`` is asserted too -- the tested code must
    physically live in ``heartbeat_event_alarms.py``."""
    sibling = hb._event_alarms
    for name in _CHECK_NAMES:
        reexported = getattr(hb, name)
        assert reexported is getattr(sibling, name), (
            f"hb.{name} is not the heartbeat_event_alarms object -- a local "
            "definition is shadowing the re-export"
        )
        assert Path(reexported.__code__.co_filename).name == "heartbeat_event_alarms.py"
    assert hb.EXPECTED_OPERATIONAL_KINDS is sibling.EXPECTED_OPERATIONAL_KINDS


def test_heartbeat_check_has_no_local_check_defs_to_shadow_the_reexports() -> None:
    """Source-level belt for the identity test above: the five names must not
    reappear as module-scope ``def``s in ``heartbeat_check.py`` -- the
    extraction moved them, it did not fork them."""
    tree = ast.parse(_HEARTBEAT_CHECK.read_text(encoding="utf-8"))
    local_defs = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    shadowing = sorted(local_defs & set(_CHECK_NAMES))
    assert not shadowing, (
        f"heartbeat_check.py re-declares {shadowing} at module scope -- it "
        "shadows the sibling re-export (and would do so only where the "
        "assignment order lets it). The checks live in "
        "heartbeat_event_alarms.py now; fix them there."
    )


# ---------------------------------------------------------------------------
# parse_iso parity -- the sibling docstring requires a byte-identical mirror
# because the ISO-vs-SQLite comparison convention depends on the copies
# agreeing.
# ---------------------------------------------------------------------------


def test_parse_iso_is_byte_identical_to_heartbeat_checks(
    hb: ModuleType, alarms: ModuleType
) -> None:
    assert inspect.getsource(alarms.parse_iso) == inspect.getsource(hb.parse_iso), (
        "heartbeat_event_alarms.parse_iso drifted from heartbeat_check.parse_iso "
        "-- the two copies must stay byte-identical (see the mirror comment "
        "above the sibling's copy)"
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "not-a-timestamp",
        "2026-13-45T99:99:99Z",  # invalid fields -- ValueError path
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        datetime.now(timezone.utc).isoformat(),
        datetime.now().isoformat(),  # naive -- no tzinfo
    ],
)
def test_parse_iso_copies_agree_on_inputs(hb: ModuleType, alarms: ModuleType, value: Any) -> None:
    assert alarms.parse_iso(value) == hb.parse_iso(value)


# ---------------------------------------------------------------------------
# Sibling import posture: no back-import into heartbeat_check (would cycle
# through its loader block), and the one sanctioned package import
# (charlie_work.event_kinds -- never charlie_work.instrumentation, which
# reaches ci_fleet at module load) stays behind try/except ImportError so a
# charlie_work-less venv degrades to the empty frozenset instead of
# crashing the script that exists to report that failure.
# ---------------------------------------------------------------------------


def test_sibling_never_imports_heartbeat_check() -> None:
    """A back-import would cycle through heartbeat_check's loader block --
    this is also why ``parse_iso`` is mirrored rather than shared. Checked
    over the whole tree (``ast.walk``), not just module scope: even a
    function-local back-import fails at call time (``scripts/`` is never on
    ``sys.path``), so there is no legal place for one. The single permitted
    exception is an ``if TYPE_CHECKING:`` block -- never executed at runtime,
    so it cannot cycle; that is where the ``RepoInfo``/``Report`` annotation
    imports legitimately live."""
    tree = ast.parse(_EVENT_ALARMS.read_text(encoding="utf-8"))
    type_checking_only: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            type_checking_only.update(id(child) for child in ast.walk(node))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if id(node) in type_checking_only:
            continue
        if isinstance(node, ast.Import):
            offenders.extend(a.name for a in node.names if a.name == "heartbeat_check")
        elif isinstance(node, ast.ImportFrom) and node.module == "heartbeat_check":
            offenders.append(f"from {node.module} import ...")
    assert not offenders, (
        f"heartbeat_event_alarms.py imports heartbeat_check {offenders} -- a "
        "back-import cycles through heartbeat_check's own loader block. "
        "Mirror the primitive instead, the way parse_iso is mirrored."
    )


def test_sibling_event_kinds_import_stays_guarded() -> None:
    """``from charlie_work.event_kinds import EXPECTED_OPERATIONAL_KINDS`` must
    sit inside a ``try`` whose handlers cover ``ImportError`` -- the
    subprocess guard in ``tests/test_heartbeat_check_ci_fleet_isolation.py``
    proves the empty-frozenset degradation end-to-end through
    ``heartbeat_check``; this pins the guard inside the sibling it moved to,
    where a future edit could drop the ``try`` without that test's author
    noticing."""
    tree = ast.parse(_EVENT_ALARMS.read_text(encoding="utf-8"))

    def _imports_event_kinds(node: ast.AST) -> bool:
        return isinstance(node, ast.ImportFrom) and node.module == "charlie_work.event_kinds"

    assert any(_imports_event_kinds(n) for n in ast.walk(tree)), (
        "heartbeat_event_alarms.py no longer imports charlie_work.event_kinds -- "
        "EXPECTED_OPERATIONAL_KINDS must come from that leaf, never be "
        "re-declared locally"
    )
    for try_node in (n for n in ast.walk(tree) if isinstance(n, ast.Try)):
        if any(_imports_event_kinds(n) for child in try_node.body for n in ast.walk(child)):
            handles_import_error = any(
                (h.type is None)
                or (isinstance(h.type, ast.Name) and h.type.id in {"ImportError", "Exception"})
                or (
                    isinstance(h.type, ast.Tuple)
                    and any(
                        isinstance(e, ast.Name) and e.id in {"ImportError", "Exception"}
                        for e in h.type.elts
                    )
                )
                for h in try_node.handlers
            )
            assert handles_import_error, (
                "the charlie_work.event_kinds import's try does not catch "
                "ImportError -- a charlie_work-less venv would crash "
                "heartbeat_check at module load"
            )
            return
    raise AssertionError(
        "the charlie_work.event_kinds import in heartbeat_event_alarms.py is "
        "not inside a try block -- the ImportError degradation guard was lost"
    )


# ---------------------------------------------------------------------------
# Standalone functional coverage: the checks run on the (report, repo,
# baseline) duck-typed contract alone -- no heartbeat_check objects --
# including the first direct coverage of check_ci_headroom_unavailable's
# payload-reason bucketing (issue #1770 had emitter-side tests only).
# ---------------------------------------------------------------------------


class _RecordingReport:
    """Duck-typed stand-in for ``heartbeat_check.Report`` with the same
    line prefixes -- proves the sibling's checks need nothing from
    ``heartbeat_check`` at runtime (``Report``/``RepoInfo`` are
    ``TYPE_CHECKING``-only names there)."""

    def __init__(self) -> None:
        self.anomaly = False
        self.lines: list[str] = []

    def anom(self, check: str, detail: str) -> None:
        self.anomaly = True
        self.lines.append(f"ANOMALY {check}: {detail}")

    def warn(self, check: str, detail: str) -> None:
        self.lines.append(f"WARN {check}: {detail}")

    def ok(self, check: str, facts: str) -> None:
        self.lines.append(f"OK {check}: {facts}")


def _make_duck_repo(tmp_path: Path) -> Any:
    return SimpleNamespace(slug="owner/repo", state_dir=tmp_path / "state")


def _insert_event(state_dir: Path, ts: str, kind: str, level: str, payload: str) -> None:
    """Append one row to an existing events.db with a real payload -- the
    shared ``_write_events_db`` fixture always writes ``'{}'``, which cannot
    reach ``check_ci_headroom_unavailable``'s ``reason`` bucketing."""
    conn = sqlite3.connect(str(state_dir / "events.db"))
    try:
        conn.execute(
            "INSERT INTO events (ts, kind, payload, level) VALUES (?, ?, ?, ?)",
            (ts, kind, payload, level),
        )
        conn.commit()
    finally:
        conn.close()


def test_check_error_events_runs_on_duck_typed_report(alarms: ModuleType, tmp_path: Path) -> None:
    repo = _make_duck_repo(tmp_path)
    _write_events_db(repo.state_dir, [(_iso(2), "self_deploy_alarm", "error")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = _RecordingReport()
    alarms.check_error_events(report, repo, baseline)
    assert report.anomaly
    assert "self_deploy_alarm" in report.lines[-1]


def test_check_ci_headroom_unavailable_warns_and_buckets_payload_reasons(
    alarms: ModuleType, tmp_path: Path
) -> None:
    """New ``ci_headroom_unavailable`` rows surface as WARN (never ANOMALY --
    a fail-open headroom reading is a diagnosability gap, not a fleet
    emergency) with a per-``reason`` count bucketed out of the event
    payload. Rows with unparseable payloads count under ``unknown``; the
    unparseable-``ts`` arm fails toward visibility (counted as new)."""
    repo = _make_duck_repo(tmp_path)
    _write_events_db(repo.state_dir, [(_iso(60), "ci_headroom_unavailable", "warning")])
    _insert_event(
        repo.state_dir,
        _iso(1),
        "ci_headroom_unavailable",
        "warning",
        '{"reason": "stale_reading"}',
    )
    _insert_event(
        repo.state_dir,
        _iso(1),
        "ci_headroom_unavailable",
        "warning",
        '{"reason": "stale_reading"}',
    )
    _insert_event(repo.state_dir, _iso(1), "ci_headroom_unavailable", "warning", "not-json")
    _insert_event(repo.state_dir, "not-a-timestamp", "ci_headroom_unavailable", "warning", "{}")
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = _RecordingReport()
    alarms.check_ci_headroom_unavailable(report, repo, baseline)
    assert not report.anomaly, report.lines
    line = report.lines[-1]
    assert line.startswith("WARN ")
    assert "ci_headroom_unavailable since last beat: 4 event(s)" in line
    assert "'stale_reading': 2" in line
    assert "'unknown': 2" in line
    assert "unavailable_rows=5" in line


def test_check_ci_headroom_unavailable_ok_when_no_new_rows(
    alarms: ModuleType, tmp_path: Path
) -> None:
    """Only rows strictly newer than ``baseline`` count -- an old row and
    an unrelated kind both leave the check OK with the row-count facts."""
    repo = _make_duck_repo(tmp_path)
    _write_events_db(
        repo.state_dir,
        [
            (_iso(60), "ci_headroom_unavailable", "warning"),
            (_iso(1), "dispatch_started", "info"),
        ],
    )
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = _RecordingReport()
    alarms.check_ci_headroom_unavailable(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert report.lines[-1].startswith("OK ")
    assert "unavailable_rows=1" in report.lines[-1]
    assert "new_since_last_beat=0" in report.lines[-1]


def test_check_ci_headroom_unavailable_anomaly_when_db_missing(
    alarms: ModuleType, tmp_path: Path
) -> None:
    """Same posture as the other four checks: this check cannot vouch for a
    repo whose events.db it cannot read."""
    repo = _make_duck_repo(tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = _RecordingReport()
    alarms.check_ci_headroom_unavailable(report, repo, baseline)
    assert report.anomaly
    assert "no events.db" in report.lines[-1]
