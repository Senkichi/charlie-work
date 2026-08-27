"""Tests for issue #940: ``tripwire_status`` consuming ``unauthorized_merge_check_skipped``.

``OrchestratorApp.tripwire_status`` (``src/charlie_work/workflow.py``, around line
17936) already reported ``pending`` findings from ``unauthorized_merge_detected``
(issue #933). This change adds a second half: it now also reads
``unauthorized_merge_check_skipped`` events (issue #937 /
``test_tripwire_skip_event.py``) via ``instrumentation.query_events`` and surfaces
them as ``skipped_count``, ``skipped_window_start``, ``last_skipped_at``, and
``last_skipped_reason`` in ``CommandResult.data``, plus an appended warning clause
in ``CommandResult.message`` when at least one skip is found.

The window bound is the semantic core of the change: skips are counted only from
``armed_at`` onward (or all-time if the baseline has no usable ``armed_at``),
because a skip recorded before arming says nothing about the armed control's
coverage.

Reuses the tripwire fixtures from ``test_charlie_work.py`` (``FakeGitHub``,
``_arm_unauthorized_merge_tripwire``, ``_merged_worker_pr``) and the
``_make_app`` helper convention from ``test_tripwire_detection_record.py`` /
``test_tripwire_skip_event.py``.

Skip events carry a real-wall-clock ``ts`` with no way to backdate it through the
public API (``instrumentation.log_event`` always stamps ``_now_iso()``), so
testing the ``armed_at`` window bound requires writing rows into ``events.db``
directly. ``test_instrumentation.py`` (e.g.
``test_dedupe_existing_duplicates_on_first_access``) already does this for the
same reason -- ``_write_skip_event`` below follows that precedent, mirroring
``instrumentation._SCHEMA_SQL`` exactly so the app's later ``CREATE TABLE IF NOT
EXISTS`` is a no-op against the same file.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import OrchestratorApp, UNAUTHORIZED_MERGE_BASELINE_KEY

from _fakes_github import FakeGitHub
from _merge_tripwire_fixtures import _arm_unauthorized_merge_tripwire, _merged_worker_pr

# Fixed by ``_arm_unauthorized_merge_tripwire``'s default -- pinned here as a
# named constant so the before/boundary/after timestamps in the window test
# read as intentional relative to it, not magic strings.
_ARMED_AT = "2026-07-26T00:00:00Z"


def _make_app(tmp_path: Path, fake_gh: FakeGitHub, **kwargs) -> tuple[OrchestratorApp, object]:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, **kwargs)
    return app, paths


def _arm_without_armed_at(paths) -> None:
    """Record a baseline dict with no ``armed_at`` key.

    This is the shape ``tripwire_status`` must treat as "baseline exists but
    has no usable arming timestamp": the count falls back to all-time rather
    than being (wrongly) zeroed by a window it cannot compute.
    """
    state = load_state(paths.state_file)
    state[UNAUTHORIZED_MERGE_BASELINE_KEY] = {"pre_existing_prs": []}
    save_state(paths.state_file, state)


def _write_skip_event(paths, ts: str, reason: str, *, error_type: str = "GitHubError") -> None:
    """Insert one ``unauthorized_merge_check_skipped`` row with a caller-chosen ``ts``.

    Writes directly to ``events.db`` next to ``state.json`` -- see module
    docstring for why the public ``log_event`` API cannot be used here.
    """
    db_path = paths.state_file.parent / "events.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT    NOT NULL,
                kind            TEXT    NOT NULL,
                payload         TEXT    NOT NULL,
                repo            TEXT,
                correlation_id  TEXT,
                pr_number       INTEGER,
                issue_number    INTEGER,
                level           TEXT DEFAULT 'info'
            );
            """
        )
        payload = json.dumps({"reason": reason, "error_type": error_type}, sort_keys=True)
        conn.execute(
            """INSERT INTO events
               (ts, kind, payload, repo, correlation_id, pr_number, issue_number, level)
               VALUES (?, 'unauthorized_merge_check_skipped', ?, NULL, NULL, NULL, NULL, 'warning')""",
            (ts, payload),
        )
        conn.commit()
    finally:
        conn.close()


def test_tripwire_status_no_skips_reports_zero_and_omits_warning_clause(tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)

    result = app.tripwire_status()

    assert result.data["skipped_count"] == 0
    assert result.data["last_skipped_at"] is None
    assert result.data["last_skipped_reason"] is None
    assert "did not run" not in result.message
    assert "no pending unauthorized-merge findings" in result.message


def test_tripwire_status_counts_skips_and_reports_most_recent_not_first(tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)

    _write_skip_event(paths, "2026-07-27T00:00:00Z", "gh: command not found")
    _write_skip_event(paths, "2026-07-28T00:00:00Z", "gh: rate limited")
    _write_skip_event(paths, "2026-07-29T00:00:00Z", "gh: timeout")

    result = app.tripwire_status()

    assert result.data["skipped_count"] == 3
    assert result.data["last_skipped_at"] == "2026-07-29T00:00:00Z", (
        "must report the MOST RECENT skip, not the first one written"
    )
    assert result.data["last_skipped_reason"] == "gh: timeout"


def test_tripwire_status_window_excludes_skips_before_arming(tmp_path: Path) -> None:
    """The semantic core of #940: only skips at/after ``armed_at`` count.

    A skip recorded before the tripwire was armed says nothing about the
    armed control's coverage -- it predates there being a control to skip.
    """
    fake_gh = FakeGitHub()
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)  # armed_at == _ARMED_AT

    # Before arming -- must be excluded.
    _write_skip_event(paths, "2026-07-20T00:00:00Z", "pre-arming outage 1")
    _write_skip_event(paths, "2026-07-24T00:00:00Z", "pre-arming outage 2")
    # Exactly at armed_at -- boundary is inclusive ("at or after").
    _write_skip_event(paths, _ARMED_AT, "boundary outage")
    # After arming -- must be included.
    _write_skip_event(paths, "2026-07-27T00:00:00Z", "post-arming outage 1")
    _write_skip_event(paths, "2026-07-28T00:00:00Z", "post-arming outage 2")

    result = app.tripwire_status()

    assert result.data["skipped_count"] == 3, (
        "only the boundary + 2 post-arming skips should count; the 2 "
        "pre-arming skips must be excluded"
    )
    assert result.data["last_skipped_at"] == "2026-07-28T00:00:00Z"
    assert result.data["last_skipped_reason"] == "post-arming outage 2"


def test_tripwire_status_window_start_echoes_armed_at(tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)

    result = app.tripwire_status()

    assert result.data["skipped_window_start"] == _ARMED_AT


def test_tripwire_status_no_armed_at_falls_back_to_all_time_not_zero(tmp_path: Path) -> None:
    """Baseline present but with no ``armed_at`` -- window is unbounded, count is not zeroed."""
    fake_gh = FakeGitHub()
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_without_armed_at(paths)

    # Deliberately "old" timestamps -- if the window bound were wrongly treated
    # as "since None" == "since everything after an empty string", or if the
    # missing armed_at were coerced to "now", these would be wrongly excluded.
    _write_skip_event(paths, "2020-01-01T00:00:00Z", "ancient outage 1")
    _write_skip_event(paths, "2020-01-02T00:00:00Z", "ancient outage 2")

    result = app.tripwire_status()

    assert result.data["skipped_window_start"] is None
    assert result.data["skipped_count"] == 2, (
        "no usable armed_at must mean an ALL-TIME count, not zero"
    )


def test_tripwire_status_message_appends_warning_when_no_pending_findings(tmp_path: Path) -> None:
    fake_gh = FakeGitHub()
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)
    _write_skip_event(paths, "2026-07-27T00:00:00Z", "gh: command not found")

    result = app.tripwire_status()

    assert "no pending unauthorized-merge findings" in result.message, (
        "the pre-existing no-findings text must survive -- the clause is appended, not substituted"
    )
    assert "warning: the check did not run on 1 pass(es)" in result.message
    assert "2026-07-27T00:00:00Z" in result.message
    assert _ARMED_AT in result.message


def test_tripwire_status_message_appends_warning_alongside_pending_findings(
    tmp_path: Path,
) -> None:
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [_merged_worker_pr(1408, 1404, "sha-1408")]
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)
    app._detect_unauthorized_merges(fake_gh.prs)
    _write_skip_event(paths, "2026-07-27T00:00:00Z", "gh: rate limited")
    _write_skip_event(paths, "2026-07-28T00:00:00Z", "gh: rate limited again")

    result = app.tripwire_status()

    assert "pending unauthorized-merge finding(s) pinning ok=False" in result.message
    assert "#1408" in result.message
    assert "warning: the check did not run on 2 pass(es)" in result.message
    assert "2026-07-28T00:00:00Z" in result.message


def test_tripwire_status_not_armed_does_not_raise(tmp_path: Path) -> None:
    """No baseline at all -- the not-armed path must stay stable with the new fields."""
    fake_gh = FakeGitHub()
    app, paths = _make_app(tmp_path, fake_gh)
    # Deliberately never armed.

    result = app.tripwire_status()

    assert result.ok is True
    assert "NOT ARMED" in result.message
    assert result.data["skipped_count"] == 0
    assert result.data["skipped_window_start"] is None
    assert result.data["last_skipped_at"] is None
    assert result.data["last_skipped_reason"] is None


def test_tripwire_status_not_armed_with_skip_events_does_not_raise(tmp_path: Path) -> None:
    """Not armed AND skip events present (e.g. a very first pass that also failed open)."""
    fake_gh = FakeGitHub()
    app, paths = _make_app(tmp_path, fake_gh)
    _write_skip_event(paths, "2026-07-15T00:00:00Z", "gh unavailable on first pass")

    result = app.tripwire_status()

    assert result.ok is True
    assert "NOT ARMED" in result.message
    # No baseline means no armed_at, so the count is all-time, same as the
    # armed-but-no-armed_at case.
    assert result.data["skipped_window_start"] is None
    assert result.data["skipped_count"] == 1
