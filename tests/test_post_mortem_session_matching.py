"""Session matching tests for ``post_mortem``.

Split out of ``tests/test_post_mortem.py`` (issue #1567, Track 1):
session-window matching precision (issue #261 F1+F2 -- defensive created_at
parsing and the widened fallback window), working_directory normalization
(issue #281), and the #343 suffix-match fallback for real fleet shapes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from _post_mortem_fixtures import (
    _NOW,
    _build_sessions_db,
    _config_with_db,
    _insert_session_row,
    _make_worker,
)

from charlie_work.post_mortem import (
    classify_and_record,
    read_post_mortem,
    real_activity_for_worker,
)


# ---------------------------------------------------------------------------
# Session-window matching precision (issue #261 F1+F2): defensive created_at
# parsing (epoch int, naive string) and a widened fallback window when
# worker.started_at itself is unparseable.
# ---------------------------------------------------------------------------


def test_find_matching_session_accepts_epoch_int_created_at(tmp_path: Path) -> None:
    """An epoch-integer created_at (e.g. 1783760135) is the REAL production
    shape — sessions.created_at is declared INTEGER in the live schema.
    Must match."""
    db_path = tmp_path / "sessions.db"
    created_at_dt = datetime(2026, 7, 11, 11, 56, 0, tzinfo=UTC)
    _build_sessions_db(db_path, created_at=int(created_at_dt.timestamp()))
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None  # no message_nodes -> nothing to classify as blocked
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.session_id == "sess-1"


def test_find_matching_session_accepts_naive_iso_string_created_at(tmp_path: Path) -> None:
    """A naive ISO-8601 created_at (no tz offset) — the drift shape, since
    the live schema declares the column INTEGER but SQLite affinity stores
    a writer-inserted string as TEXT — must be treated as UTC and matched
    correctly against the tz-aware match window from worker.started_at."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        created_at="2026-07-11T11:56:00",  # naive, no tz suffix
        nodes=[],
    )
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.window_start_fallback is None  # started_at parsed fine


def test_find_matching_session_ignores_out_of_window_row_even_if_lexicographically_greater(
    tmp_path: Path,
) -> None:
    """Regression guard for the SQL-side lexicographic-string-comparison bug:
    a naive created_at string that would have sorted "greater than" the
    isoformat()'d window bound as plain text, but is chronologically outside
    the window once actually parsed, must NOT match."""
    db_path = tmp_path / "sessions.db"
    # worker started_at=11:55:00Z, margin=120s -> window [11:53:00, 12:02:00]Z.
    # A calendar day later sorts "greater" lexicographically but is far
    # outside the real time window once parsed.
    _build_sessions_db(db_path, created_at="2026-07-12T11:56:00", nodes=[])
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is False


def test_classify_and_record_unparseable_started_at_widens_fallback_window(
    tmp_path: Path,
) -> None:
    """When worker.started_at itself fails to parse, the match window must
    widen to the config lookback from "now" (not a narrow now-minus-margin
    window) — a session that started well before "now" (e.g. 2h earlier,
    outside the old ~240s window but inside the 6h default lookback) must
    still be found, and window_start_fallback must record that this
    fallback fired."""
    db_path = tmp_path / "sessions.db"
    session_created_at = _NOW.replace(hour=10)  # 2 hours before _NOW (12:00)
    _build_sessions_db(
        db_path,
        created_at=session_created_at.strftime("%Y-%m-%dT%H:%M:%S"),
        nodes=[],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(started_at="not-a-timestamp")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.window_start_fallback == "unparseable_started_at"


def test_classify_and_record_unparseable_started_at_no_match_still_records_fallback(
    tmp_path: Path,
) -> None:
    """Even when nothing matches within the widened fallback window, the
    fact that the fallback fired must still be recorded — otherwise a false
    non-match caused by an unparseable started_at is indistinguishable from
    "genuinely no session ran," which is exactly the diagnosability gap F1/F2
    close."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-999-different",
        nodes=[],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(started_at="not-a-timestamp")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is False
    assert record.window_start_fallback == "unparseable_started_at"


# ---------------------------------------------------------------------------
# working_directory normalization (issue #281)
# ---------------------------------------------------------------------------


def test_classify_and_record_matches_worktree_with_forward_slash_separator(
    tmp_path: Path,
) -> None:
    """A sessions.db row using forward slashes must match a worker whose
    worktree_path uses native Windows backslashes."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[("tool", "Tool blocked: push rejected", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.session_id == "sess-1"


def test_classify_and_record_matches_worktree_with_different_case(tmp_path: Path) -> None:
    """A sessions.db row whose working_directory differs only by case must
    match the worker's worktree_path."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="c:/repo/.var/worktrees/issue-42",
        nodes=[("tool", "Tool blocked: push rejected", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.session_id == "sess-1"


def test_classify_and_record_matches_worktree_with_trailing_separator(
    tmp_path: Path,
) -> None:
    """A sessions.db row whose working_directory has a trailing separator must
    still match the worker's worktree_path."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42/",
        nodes=[("tool", "Tool blocked: push rejected", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.session_id == "sess-1"


def test_classify_and_record_matches_worktree_with_msys_style_leading_slash(
    tmp_path: Path,
) -> None:
    r"""A sessions.db row using a POSIX/MSYS-style /c/... path must match a
    worker whose worktree_path is a native Windows C:\... path."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="/c/repo/.var/worktrees/issue-42",
        nodes=[("tool", "Tool blocked: push rejected", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.session_id == "sess-1"


def test_classify_and_record_normalized_match_outside_window_still_within_window_error(
    tmp_path: Path,
) -> None:
    """If a normalized working_directory matches a row but its created_at is
    outside the match window, the existing "within the time window" error path
    still fires exactly as today."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="c:/repo/.var/worktrees/issue-42",
        created_at="2026-07-12T11:56:00",
        nodes=[],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is False
    assert record.extraction_error is not None
    assert "within the time window" in record.extraction_error


def test_classify_and_record_no_normalized_match_surfaces_distinct_directories(
    tmp_path: Path,
) -> None:
    """When exact and normalized working_directory matching both fail, the
    extraction_error must include a sample of distinct working_directory values
    actually present in sessions.db for diagnostics."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-999-different",
        nodes=[],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result is None
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is False
    assert record.extraction_error is not None
    assert "sample distinct working_directory values" in record.extraction_error
    assert "C:/repo/.var/worktrees/issue-999-different" in record.extraction_error


def test_classify_and_record_multiple_normalized_sessions_returns_latest(
    tmp_path: Path,
) -> None:
    """A reused worktree with multiple prior dead sessions in the same
    normalized working directory must return the most recent in-window session's
    transcript, just as it does once the row is found."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="c:/repo/.var/worktrees/issue-42/",
        created_at="2026-07-11T11:54:00",
        nodes=[("tool", json.dumps({"tool": "bash"}), "2026-07-11T11:54:00")],
    )
    _insert_session_row(
        db_path,
        session_id="sess-2",
        working_directory="C:/repo/.var/worktrees/issue-42",
        created_at="2026-07-11T11:56:00",
        nodes=[("tool", "Tool blocked: push rejected", "2026-07-11T11:56:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(worktree_path=r"C:\repo\.var\worktrees\issue-42")
    sessions_dir = tmp_path / "sessions"

    result = classify_and_record(sessions_dir, config, worker, now=_NOW)

    assert result == "worker_blocked"
    record = read_post_mortem(sessions_dir, worker.issue_number)
    assert record is not None
    assert record.matched is True
    assert record.session_id == "sess-2"
    assert record.failure_kind == "worker_blocked"


# ---------------------------------------------------------------------------
# Issue #343: working_directory suffix-match fallback for real fleet shapes
# ---------------------------------------------------------------------------


def test_find_matching_session_suffix_fallback_rejects_different_issue_slug(
    tmp_path: Path,
) -> None:
    """The suffix-match fallback must not collapse two different issues'
    worktrees just because they share the same worktrees-dir parent segment.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory=(
            r"C:\Users\operator\AppData\Local\Temp\other-root"
            r"\worktrees\agent-issue-999-unrelated"
        ),
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        r"C:\Users\operator\repos\charlie-work\.var\charlie-work\worktrees\agent-issue-203-fix",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
    )

    assert probe.latest_timestamp is None
    db_source = next(s for s in probe.sources if s.name == "sessions.db")
    assert db_source.error is not None


def test_find_matching_session_suffix_fallback_requires_parent_segment_match(
    tmp_path: Path,
) -> None:
    """The suffix-match fallback must compare the segment above the
    issue-slug leaf too, not just the leaf segment alone (issue #343
    Finding 3).

    ``test_find_matching_session_suffix_fallback_rejects_different_issue_slug``
    above is satisfied by the two working_directory values having different
    trailing slug segments, so it passes at
    ``_WORKING_DIRECTORY_SUFFIX_SEGMENTS`` == 1 or == 2 alike -- it does not
    by itself pin the segment count. This test constructs a DB row whose
    working_directory shares the exact same trailing slug segment as the
    target worktree but sits under an unrelated parent directory (not the
    fleet worktrees-dir), so a 1-segment suffix would wrongly match it while
    the real 2-segment suffix correctly rejects it.

    MUTATION GATE: shrinking ``_WORKING_DIRECTORY_SUFFIX_SEGMENTS`` from 2 to
    1 makes this test fail -- the unrelated row would wrongly match on the
    shared leaf segment alone.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory=(
            r"C:\Users\operator\AppData\Local\Temp\some-unrelated-tool-cache"
            r"\agent-issue-203-fix"
        ),
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        r"C:\Users\operator\repos\charlie-work\.var\charlie-work\worktrees\agent-issue-203-fix",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
    )

    assert probe.latest_timestamp is None
    db_source = next(s for s in probe.sources if s.name == "sessions.db")
    assert db_source.error is not None
