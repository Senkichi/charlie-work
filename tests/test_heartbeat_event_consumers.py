"""Regression tests for the issue #1366 heartbeat event consumers.

``scripts/heartbeat_check.py`` gained two new checks that give the previously
consumer-less event kinds ``draft_pr_blocked`` and
``venv_editable_anchor_violation`` their first automated readers:

* ``check_draft_pr_blocked_events`` -- per-repo, periodic surfacing (WARN) of
  draft-blocked PR parks, alongside the other stuck-state detectors.
* ``check_supervisor_venv_refusal`` -- fleet-level, real-alert (ANOMALY) path
  for the supervisor's venv-anchor hard refusal (issue #974), running next to
  ``check_supervisor_heartbeat`` in the supervisor-health section.

These tests live in their own module rather than ``tests/test_heartbeat_check.py``
because that module is at its attachment-contracts ceiling (180 members > 152
baseline); the contract redirects new test growth to a non-saturated sibling
rather than bumping the baseline. The helpers below are self-contained copies
of the small fixtures ``test_heartbeat_check.py`` uses (``_iso``,
``_make_repo``, ``_write_events_db``) rather than a cross-test-module import,
so this module's correctness never depends on another test file's private
internals -- the same independence property ``test_heartbeat_check.py`` itself
maintains against ``test_event_kind_consumers.py``.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import pytest

from _script_loader import load_script_module


def _load_heartbeat_check() -> ModuleType:
    """Load scripts/heartbeat_check.py as a module without adding scripts to sys.path."""
    path = Path(__file__).parent.parent / "scripts" / "heartbeat_check.py"
    return load_script_module(path, "heartbeat_check")


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


def _iso(minutes_ago: float = 0.0, *, base: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp `minutes_ago` before `base`.

    `base` defaults to the real wall clock, sampled here. Mirrors
    ``test_heartbeat_check.py``'s ``_iso`` so the new-since-last-beat windows
    below derive test timestamps from the real clock the same way the
    production ``baseline`` comparison does.
    """
    reference = base if base is not None else datetime.now(timezone.utc)
    ts = reference - timedelta(minutes=minutes_ago)
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_repo(hb: ModuleType, tmp_path: Path) -> object:
    return hb.RepoInfo(
        slug="owner/repo",
        repo_root=tmp_path,
        state_dir=tmp_path / "state",
        config_path=tmp_path / "orchestrator.config.yaml",
    )


def _write_events_db(
    state_dir: Path,
    rows: list[tuple[str, str] | tuple[str, str, str]] | None = None,
) -> Path:
    """Create an events.db next to state.json with the production `events` schema.

    Mirrors ``test_heartbeat_check.py``'s ``_write_events_db`` by hand rather
    than importing the package, since heartbeat_check.py deliberately avoids
    that import (see ``fleet_dir``'s docstring) and these checks must be
    tested the same way.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / "events.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT    NOT NULL,
                kind            TEXT    NOT NULL,
                payload         TEXT    NOT NULL,
                repo            TEXT,
                correlation_id  TEXT,
                pr_number       INTEGER,
                issue_number    INTEGER,
                level           TEXT DEFAULT 'info'
            )
            """
        )
        for row in rows or []:
            ts, kind = row[0], row[1]
            level = row[2] if len(row) > 2 else "info"
            conn.execute(
                "INSERT INTO events (ts, kind, payload, level) VALUES (?, ?, '{}', ?)",
                (ts, kind, level),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


# ---------------------------------------------------------------------------
# check_draft_pr_blocked_events (issue #1366)
#
# Mirrors the check_infra_blocked_events coverage: a missing or unreadable
# events.db is an anomaly (this check cannot vouch for a repo it cannot
# read), and the ok/warn branching follows the production precedence -- a
# ``draft_pr_blocked`` row newer than baseline is a WARN (passive, periodic
# surfacing, never flips the exit code); otherwise OK with the row-count
# facts. A draft-blocked PR is a state to surface, not a fleet emergency.
# ---------------------------------------------------------------------------


def test_check_draft_pr_blocked_events_anomaly_when_db_missing(
    hb: ModuleType, tmp_path: Path
) -> None:
    """A missing events.db is an anomaly, matching check_infra_blocked_events:
    this check's job is to surface draft-blocked parks, and a registered repo
    with no events.db is a repo this check cannot vouch for."""
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_draft_pr_blocked_events(report, repo, baseline)
    assert report.anomaly
    assert "no events.db" in report.lines[-1]


def test_check_draft_pr_blocked_events_anomaly_when_table_missing(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    db_path = repo.state_dir / "events.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE unrelated (id INTEGER)")
        conn.commit()
    finally:
        conn.close()

    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_draft_pr_blocked_events(report, repo, baseline)
    assert report.anomaly
    assert "no events table" in report.lines[-1]


def test_check_draft_pr_blocked_events_anomaly_when_db_unreadable(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    (repo.state_dir / "events.db").write_bytes(b"not a sqlite database at all")

    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_draft_pr_blocked_events(report, repo, baseline)
    assert report.anomaly


def test_check_draft_pr_blocked_events_ok_when_no_relevant_events(
    hb: ModuleType, tmp_path: Path
) -> None:
    """An events.db with only unrelated kinds (no draft_pr_blocked) yields OK
    with the row-count facts."""
    repo = _make_repo(hb, tmp_path)
    _write_events_db(
        repo.state_dir,
        [(_iso(1), "dispatch_started", "info"), (_iso(1), "self_deploy_alarm", "error")],
    )
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_draft_pr_blocked_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "blocked_rows=0" in report.lines[-1]
    assert "new_since_last_beat=0" in report.lines[-1]
    assert report.lines[-1].startswith("OK ")


def test_check_draft_pr_blocked_events_warn_when_blocked_since_baseline(
    hb: ModuleType, tmp_path: Path
) -> None:
    """A ``draft_pr_blocked`` row newer than baseline surfaces as a WARN
    without setting ``anomaly`` -- a draft-blocked PR is a state to surface
    periodically, not a fleet emergency (issue #1366 operator comment)."""
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(1), "draft_pr_blocked", "warning")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_draft_pr_blocked_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "draft_pr_blocked since last beat" in report.lines[-1]
    assert report.lines[-1].startswith("WARN ")


def test_check_draft_pr_blocked_events_excludes_old_rows(hb: ModuleType, tmp_path: Path) -> None:
    """Rows older than baseline are excluded from the new-since-last-beat
    count, matching the check_error_events/check_infra_blocked_events
    old-row-exclusion convention."""
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(60), "draft_pr_blocked", "warning")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=5)
    report = hb.Report()
    hb.check_draft_pr_blocked_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "blocked_rows=1" in report.lines[-1]
    assert "new_since_last_beat=0" in report.lines[-1]
    assert report.lines[-1].startswith("OK ")


# ---------------------------------------------------------------------------
# check_supervisor_venv_refusal (issue #1366)
#
# The supervisor's venv-anchor refusal (issue #974) is a hard safety gate: a
# silent refusal stalls the whole supervisor with nothing but a log line as
# evidence. This check is the real-alert path -- a new
# ``venv_editable_anchor_violation`` event since the last beat is an ANOMALY
# (flips the exit code), not a passive WARN. The event is logged to a
# per-repo events.db, so the check scans every registered repo and aggregates
# into one fleet-level alert. A repo with no events.db is skipped (no refusal
# to find), not flagged -- supervisor liveness is vouched for separately by
# check_supervisor_heartbeat.
# ---------------------------------------------------------------------------


def test_check_supervisor_venv_refusal_ok_when_no_relevant_events(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Repos with events.db but no venv violation rows yield OK with the
    scanned-repo count."""
    repo = _make_repo(hb, tmp_path)
    _write_events_db(
        repo.state_dir,
        [(_iso(1), "dispatch_started", "info"), (_iso(1), "self_deploy_alarm", "error")],
    )
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_supervisor_venv_refusal(report, [repo], baseline)
    assert not report.anomaly, report.lines
    assert "refusals_since_last_beat=0" in report.lines[-1]
    assert "repos_scanned=1" in report.lines[-1]
    assert report.lines[-1].startswith("OK ")


def test_check_supervisor_venv_refusal_ok_when_no_repos(hb: ModuleType) -> None:
    """An empty fleet scans zero repos and yields OK -- no refusal to find."""
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_supervisor_venv_refusal(report, [], baseline)
    assert not report.anomaly, report.lines
    assert "repos_scanned=0" in report.lines[-1]


def test_check_supervisor_venv_refusal_skips_repo_with_no_db(
    hb: ModuleType, tmp_path: Path
) -> None:
    """A repo with no events.db is skipped, not flagged -- this check looks
    for an actual refusal event, and a repo that has never been supervised
    has none to find."""
    repo = _make_repo(hb, tmp_path)
    # deliberately no events.db created
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_supervisor_venv_refusal(report, [repo], baseline)
    assert not report.anomaly, report.lines
    assert report.lines[-1].startswith("OK ")


def test_check_supervisor_venv_refusal_anomaly_when_violation_since_baseline(
    hb: ModuleType, tmp_path: Path
) -> None:
    """A ``venv_editable_anchor_violation`` row newer than baseline is an
    ANOMALY that flips the exit code -- the real-alert path the operator
    comment reserved for this kind (issue #1366)."""
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(1), "venv_editable_anchor_violation", "error")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_supervisor_venv_refusal(report, [repo], baseline)
    assert report.anomaly, report.lines
    assert "venv_editable_anchor_violation since last beat" in report.lines[-1]
    assert report.lines[-1].startswith("ANOMALY ")


def test_check_supervisor_venv_refusal_aggregates_across_repos(
    hb: ModuleType, tmp_path: Path
) -> None:
    """A refusal in any one repo surfaces as a single fleet-level ANOMALY
    naming the offending repo; a clean repo does not dilute the alert."""
    repo_a = hb.RepoInfo(
        slug="owner/repo-a",
        repo_root=tmp_path / "a",
        state_dir=tmp_path / "a" / "state",
        config_path=tmp_path / "a" / "orchestrator.config.yaml",
    )
    repo_b = hb.RepoInfo(
        slug="owner/repo-b",
        repo_root=tmp_path / "b",
        state_dir=tmp_path / "b" / "state",
        config_path=tmp_path / "b" / "orchestrator.config.yaml",
    )
    _write_events_db(repo_a.state_dir, [(_iso(1), "venv_editable_anchor_violation", "error")])
    _write_events_db(repo_b.state_dir, [(_iso(1), "dispatch_started", "info")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_supervisor_venv_refusal(report, [repo_a, repo_b], baseline)
    assert report.anomaly, report.lines
    assert "owner/repo-a@" in report.lines[-1]
    assert "owner/repo-b@" not in report.lines[-1]


def test_check_supervisor_venv_refusal_excludes_old_rows(hb: ModuleType, tmp_path: Path) -> None:
    """A violation older than baseline is excluded -- it was already surfaced
    on the beat that followed the refusal, not a fresh refusal now."""
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(60), "venv_editable_anchor_violation", "error")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=5)
    report = hb.Report()
    hb.check_supervisor_venv_refusal(report, [repo], baseline)
    assert not report.anomaly, report.lines
    assert report.lines[-1].startswith("OK ")
