"""Event-db check tests for ``scripts/heartbeat_check.py``.

Split out of ``tests/test_heartbeat_check.py`` (issue #1556, Track-1):
``check_error_events``, ``check_warning_events`` (issue #946,
including the expected-operational bucketing and the
no-hardcoded-kinds source guard), ``report.warn`` non-anomaly
semantics, and ``check_infra_blocked_events`` (issue #1383, AC4).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import pytest

from _heartbeat_check_fixtures import (
    _iso,
    _load_heartbeat_check,
    _make_repo,
    _write_events_db,
)


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


def test_check_error_events_surfaces_seeded_self_deploy_alarm(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Regression for issue #866: error-level events are emitted, classified,
    documented, and tested -- and had NO consumer anywhere in the codebase.
    A human had to manually open events.db and know which `kind` to search
    for. `self_deploy_alarm` is a real production example of this (emitted
    from `supervise.py`, classified error-level by
    `instrumentation._classify_level` via `_ERROR_KINDS`).

    This test must fail with an AttributeError before `check_error_events`
    exists -- if it doesn't fail first, it isn't testing the gap.
    """
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(2), "self_deploy_alarm", "error")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_error_events(report, repo, baseline)
    assert report.anomaly
    assert "self_deploy_alarm" in report.lines[-1]


def test_check_error_events_ok_when_only_info_and_warning_events(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    _write_events_db(
        repo.state_dir,
        [
            (_iso(1), "dispatch_started", "info"),
            (_iso(1), "review_claim_stale", "warning"),
        ],
    )
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_error_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "error_rows=0" in report.lines[0]


def test_check_error_events_covers_synthetic_kind_not_hardcoded(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Coverage must be derived from the persisted `level` column, never a
    hardcoded list of `kind` strings in heartbeat_check.py (issue #866,
    acceptance criterion: coverage derived from `_ERROR_KINDS`, "or asserts
    the check has no literal kind list"). A `kind` that doesn't exist in
    `_ERROR_KINDS` today -- and never has -- must still be caught purely
    because its row was persisted with `level='error'`. This is also what
    makes PR #865's new `supervisor_zero_pass_alarm` kind land covered "for
    free": nothing in this check needs to change when a new alarm kind is
    added to `_ERROR_KINDS`, because it never enumerated kinds in the first
    place.
    """
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(1), "totally_novel_alarm_kind_xyz", "error")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_error_events(report, repo, baseline)
    assert report.anomaly
    assert "totally_novel_alarm_kind_xyz" in report.lines[-1]


def test_check_error_events_excludes_old_row_despite_sql_trap_shape(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Positive control for the ISO-vs-SQLite string-comparison trap, in the
    direction that matters for THIS check's query shape (`ts > baseline`,
    selecting NEW rows -- the opposite predicate from
    `check_loop_pass_freshness`'s `MAX(ts)`).

    `ts` values are `...THH:MM:SSZ`. If baseline were bound into a SQL
    predicate via a naive `str(datetime)` (space-separated, no `Z`, e.g.
    `2026-07-31 22:25:04+00:00`), a genuinely OLD row's `T`-formatted `ts`
    would still sort as "greater than" that space-formatted cutoff --
    `'T'` (0x54) sorts after `' '` (0x20) -- producing a false alarm on an
    old, already-seen event. This row is 60 minutes older than baseline and
    must be excluded; if SQL-based comparison is ever substituted for the
    Python-side `parse_iso` + `datetime` comparison, this test must go red.
    """
    repo = _make_repo(hb, tmp_path)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=5)
    _write_events_db(repo.state_dir, [(_iso(60), "self_deploy_alarm", "error")])
    report = hb.Report()
    hb.check_error_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "error_rows=1" in report.lines[0]
    assert "new_since_last_beat=0" in report.lines[0]


def test_check_error_events_excludes_row_older_than_cold_start_fallback(
    hb: ModuleType, tmp_path: Path
) -> None:
    """On a cold start (no prior heartbeat-state.json), `main()` falls
    `baseline` back to `now - LOG_FRESHNESS_STALE_MINUTES` (30m). An alarm
    older than that fallback window is deliberately out of scope on the
    very first run -- a bounded, intentional blind spot rather than an
    oversight. Pins that boundary: a 40m-old alarm against the 30m
    fallback baseline must NOT be reported.
    """
    repo = _make_repo(hb, tmp_path)
    now = datetime.now(timezone.utc)
    fallback_baseline = now - timedelta(minutes=hb.LOG_FRESHNESS_STALE_MINUTES)
    _write_events_db(repo.state_dir, [(_iso(40, base=now), "self_deploy_alarm", "error")])
    report = hb.Report()
    hb.check_error_events(report, repo, fallback_baseline)
    assert not report.anomaly, report.lines


def test_check_error_events_anomaly_when_db_missing(hb: ModuleType, tmp_path: Path) -> None:
    """Unlike `check_loop_pass_freshness` (missing db = OK, "no history
    yet"), a missing events.db here is an ANOMALY: this check's entire job
    is "did any alarm fire," and a registered repo with no events.db is a
    repo this check cannot vouch for -- reporting OK would be a silent
    false negative in exactly the direction issue #866 exists to close.
    """
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_error_events(report, repo, baseline)
    assert report.anomaly
    assert "no events.db" in report.lines[-1]


def test_check_error_events_anomaly_when_table_missing(hb: ModuleType, tmp_path: Path) -> None:
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
    hb.check_error_events(report, repo, baseline)
    assert report.anomaly
    assert "no events table" in report.lines[-1]


def test_check_error_events_anomaly_when_db_unreadable(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    (repo.state_dir / "events.db").write_bytes(b"not a sqlite database at all")

    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_error_events(report, repo, baseline)
    assert report.anomaly


# ---------------------------------------------------------------------------
# check_warning_events (issue #946)
#
# Mirrors the check_error_events tests above one level down the `level`
# column, plus the one deliberate behavioral difference: a found warning
# must surface (via `report.warn`) without setting `report.anomaly`.
# ---------------------------------------------------------------------------


def test_check_warning_events_surfaces_seeded_dispatch_stale(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Regression for issue #946: warning-level events (dispatch_stale and
    ~6 pre-existing kinds) are emitted, classified, documented, and unit
    tested -- and had NO consumer anywhere in the codebase before this
    check. `dispatch_stale` is a real production example (emitted from
    `workflow.check_dispatch_staleness`, classified warning-level by
    `instrumentation._classify_level` via `_WARNING_KINDS`).

    This test must fail with an AttributeError before `check_warning_events`
    exists -- if it doesn't fail first, it isn't testing the gap.
    """
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(2), "dispatch_stale", "warning")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_warning_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "dispatch_stale" in report.lines[-1]
    assert report.lines[-1].startswith("WARN ")


def test_check_warning_events_ok_when_only_info_and_error_events(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    _write_events_db(
        repo.state_dir,
        [
            (_iso(1), "dispatch_started", "info"),
            (_iso(1), "self_deploy_alarm", "error"),
        ],
    )
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_warning_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "warning_rows=0" in report.lines[0]


def test_check_warning_events_covers_synthetic_kind_not_hardcoded(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Coverage must be derived from the persisted `level` column, never a
    hardcoded list of `kind` strings, matching
    `test_check_error_events_covers_synthetic_kind_not_hardcoded` above."""
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(1), "totally_novel_warning_kind_xyz", "warning")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_warning_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "totally_novel_warning_kind_xyz" in report.lines[-1]


def test_check_warning_events_buckets_expected_operational_separately_from_rare(
    hb: ModuleType, tmp_path: Path
) -> None:
    """AC2 (#1271, corrected AC from the binding-decisions comment): a mixed
    fixture with expected-operational kinds at volume plus one rare genuine
    warning kind. The expected kinds must appear ONLY in the summarized
    count line; the rare kind must appear in the detailed list; and the
    COMBINED expected-operational share -- not just one kind -- must be
    absent from the detailed list."""
    repo = _make_repo(hb, tmp_path)
    rows = (
        [(_iso(1), "session_exited", "warning") for _ in range(5)]
        + [(_iso(1), "dispatch_stale", "warning") for _ in range(3)]
        + [(_iso(1), "worktree_foreign_writer", "warning")]
    )
    _write_events_db(repo.state_dir, rows)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_warning_events(report, repo, baseline)
    assert not report.anomaly, report.lines

    detail_lines = [line for line in report.lines if "new warning-level event(s)" in line]
    summary_lines = [line for line in report.lines if "routine operational warnings" in line]
    assert len(detail_lines) == 1, report.lines
    assert len(summary_lines) == 1, report.lines

    # The rare kind is in the detailed list, never the summary.
    assert "worktree_foreign_writer" in detail_lines[0]
    assert "worktree_foreign_writer" not in summary_lines[0]

    # The expected-operational kinds are in the summary, absent (combined,
    # not just one of them) from the detailed list.
    assert "session_exited" in summary_lines[0]
    assert "dispatch_stale" in summary_lines[0]
    assert "session_exited" not in detail_lines[0]
    assert "dispatch_stale" not in detail_lines[0]

    # Counts and sorted-by-kind-name ordering within the summary.
    assert "dispatch_stale=3" in summary_lines[0]
    assert "session_exited=5" in summary_lines[0]
    assert summary_lines[0].index("dispatch_stale=3") < summary_lines[0].index("session_exited=5")
    assert "8 routine operational warnings" in summary_lines[0]


def test_check_warning_events_all_expected_operational_omits_detail_line(
    hb: ModuleType, tmp_path: Path
) -> None:
    """When every new warning is expected-operational, no detailed-listing
    line is emitted at all -- only the summary. The summary line must still
    carry the `warning_rows=`/`new_since_last_beat=` facts the operator's
    digest relies on -- this is the dominant production case (#1271's own
    7-day sample: expected-operational kinds were the majority of all
    warnings), so those facts cannot be conditional on a detail line also
    firing."""
    repo = _make_repo(hb, tmp_path)
    rows = [(_iso(1), "runner_capacity_starved", "warning") for _ in range(2)] + [
        (_iso(1), "draft_pr_ready_held", "warning")
    ]
    _write_events_db(repo.state_dir, rows)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_warning_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert not any("new warning-level event(s)" in line for line in report.lines)
    assert any("routine operational warnings" in line for line in report.lines)
    assert any("warning_rows=" in line for line in report.lines)
    assert any("new_since_last_beat=3" in line for line in report.lines)


def test_check_warning_events_deterministic_across_runs(hb: ModuleType, tmp_path: Path) -> None:
    """AC4 (#1271): running check_warning_events twice over the same fixture
    yields byte-identical report lines -- this script feeds the operator's
    deterministic heartbeat digest, so no dict/set-order dependent output."""
    repo = _make_repo(hb, tmp_path)
    rows = (
        [(_iso(1), "session_exited", "warning") for _ in range(5)]
        + [(_iso(1), "dispatch_stale", "warning") for _ in range(3)]
        + [(_iso(1), "runner_capacity_starved", "warning") for _ in range(2)]
        + [(_iso(1), "draft_pr_ready_held", "warning")]
        + [(_iso(1), "worktree_foreign_writer", "warning")]
    )
    _write_events_db(repo.state_dir, rows)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)

    report1 = hb.Report()
    hb.check_warning_events(report1, repo, baseline)
    report2 = hb.Report()
    hb.check_warning_events(report2, repo, baseline)

    assert report1.lines == report2.lines
    assert not report1.anomaly


def test_heartbeat_check_source_has_no_hardcoded_expected_operational_kind_literals() -> None:
    """AC3 (#1271): heartbeat_check.py must reach every
    EXPECTED_OPERATIONAL_KINDS member only via the imported frozenset --
    never as a hardcoded literal anywhere in the file (code, comments, or
    docstrings alike). Source-derived from the live frozenset, not a
    maintained list here, so adding a member later needs no change to this
    test or to heartbeat_check.py."""
    from charlie_work.instrumentation import EXPECTED_OPERATIONAL_KINDS

    source_path = Path(__file__).parent.parent / "scripts" / "heartbeat_check.py"
    source = source_path.read_text(encoding="utf-8")

    assert EXPECTED_OPERATIONAL_KINDS, "the set must not be empty for this test to mean anything"
    for kind in EXPECTED_OPERATIONAL_KINDS:
        assert kind not in source, (
            f"{kind!r} appears as a literal in heartbeat_check.py -- it must be "
            "reached only via the imported EXPECTED_OPERATIONAL_KINDS frozenset"
        )


def test_check_warning_events_excludes_old_row(hb: ModuleType, tmp_path: Path) -> None:
    """Positive control mirroring
    `test_check_error_events_excludes_old_row_despite_sql_trap_shape`: a row
    60 minutes older than baseline must be excluded from `new_since_last_beat`."""
    repo = _make_repo(hb, tmp_path)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=5)
    _write_events_db(repo.state_dir, [(_iso(60), "dispatch_stale", "warning")])
    report = hb.Report()
    hb.check_warning_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "warning_rows=1" in report.lines[0]
    assert "new_since_last_beat=0" in report.lines[0]


def test_check_warning_events_anomaly_when_db_missing(hb: ModuleType, tmp_path: Path) -> None:
    """Unlike a found warning (non-fatal), the check's OWN inability to read
    events.db is still a genuine anomaly, matching
    `test_check_error_events_anomaly_when_db_missing`: this check cannot
    vouch for the repo at all without a readable database."""
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_warning_events(report, repo, baseline)
    assert report.anomaly
    assert "no events.db" in report.lines[-1]


def test_report_warn_does_not_set_anomaly(hb: ModuleType) -> None:
    """Direct unit test of the `Report.warn` primitive itself: it must append
    a line but never flip `anomaly`, unlike `Report.anom`."""
    report = hb.Report()
    report.warn("some-check", "some non-fatal detail")
    assert report.anomaly is False
    assert report.lines == ["WARN some-check: some non-fatal detail"]


# ---------------------------------------------------------------------------
# check_infra_blocked_events (issue #1383, AC4)
#
# Mirrors the check_error_events / check_warning_events coverage above: a
# missing or unreadable events.db is an anomaly (this check cannot vouch for
# a repo it cannot read), and the ok/warn/anom branching follows the
# production precedence -- an ``infra_blocked_escalated`` row newer than
# baseline is an anomaly; otherwise a ``check_infra_blocked`` row newer than
# baseline is a warning; otherwise OK.
# ---------------------------------------------------------------------------


def test_check_infra_blocked_events_anomaly_when_db_missing(
    hb: ModuleType, tmp_path: Path
) -> None:
    """A missing events.db is an anomaly, matching
    `test_check_error_events_anomaly_when_db_missing`: this check's entire
    job is "did any infra-blocked escalation fire," and a registered repo
    with no events.db is a repo this check cannot vouch for."""
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_infra_blocked_events(report, repo, baseline)
    assert report.anomaly
    assert "no events.db" in report.lines[-1]


def test_check_infra_blocked_events_anomaly_when_table_missing(
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
    hb.check_infra_blocked_events(report, repo, baseline)
    assert report.anomaly
    assert "no events table" in report.lines[-1]


def test_check_infra_blocked_events_anomaly_when_db_unreadable(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    (repo.state_dir / "events.db").write_bytes(b"not a sqlite database at all")

    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_infra_blocked_events(report, repo, baseline)
    assert report.anomaly


def test_check_infra_blocked_events_ok_when_no_relevant_events(
    hb: ModuleType, tmp_path: Path
) -> None:
    """An events.db with only unrelated kinds (no check_infra_blocked, no
    infra_blocked_escalated) yields OK with the row-count facts."""
    repo = _make_repo(hb, tmp_path)
    _write_events_db(
        repo.state_dir,
        [(_iso(1), "dispatch_started", "info"), (_iso(1), "self_deploy_alarm", "error")],
    )
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_infra_blocked_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "blocked_rows=0" in report.lines[-1]
    assert "escalated_rows=0" in report.lines[-1]
    assert report.lines[-1].startswith("OK ")


def test_check_infra_blocked_events_warn_when_blocked_since_baseline(
    hb: ModuleType, tmp_path: Path
) -> None:
    """A ``check_infra_blocked`` row newer than baseline (with no
    ``infra_blocked_escalated``) surfaces as a WARN without setting
    ``anomaly`` -- the infra condition is being held, not yet escalated."""
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(1), "check_infra_blocked", "warning")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_infra_blocked_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "check_infra_blocked since last beat" in report.lines[-1]
    assert report.lines[-1].startswith("WARN ")


def test_check_infra_blocked_events_anom_when_escalated_since_baseline(
    hb: ModuleType, tmp_path: Path
) -> None:
    """An ``infra_blocked_escalated`` row newer than baseline is an ANOMALY
    -- the persistence threshold was reached and an operator-facing
    escalation fired. Precedence over the warn branch: even when
    ``check_infra_blocked`` rows are also present, the escalation is the
    finding that surfaces."""
    repo = _make_repo(hb, tmp_path)
    _write_events_db(
        repo.state_dir,
        [
            (_iso(1), "check_infra_blocked", "warning"),
            (_iso(1), "infra_blocked_escalated", "error"),
        ],
    )
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_infra_blocked_events(report, repo, baseline)
    assert report.anomaly, report.lines
    assert "infra_blocked_escalated since last beat" in report.lines[-1]
    assert report.lines[-1].startswith("ANOMALY ")


def test_check_infra_blocked_events_excludes_old_rows(hb: ModuleType, tmp_path: Path) -> None:
    """Rows older than baseline are excluded from the new-since-last-beat
    counts, matching the check_error_events/check_warning_events
    old-row-exclusion convention."""
    repo = _make_repo(hb, tmp_path)
    _write_events_db(
        repo.state_dir,
        [
            (_iso(60), "check_infra_blocked", "warning"),
            (_iso(60), "infra_blocked_escalated", "error"),
        ],
    )
    baseline = datetime.now(timezone.utc) - timedelta(minutes=5)
    report = hb.Report()
    hb.check_infra_blocked_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "blocked_rows=1" in report.lines[-1]
    assert "escalated_rows=1" in report.lines[-1]
    assert report.lines[-1].startswith("OK ")
