"""``real_activity_for_worker`` / ``RealActivityProbe`` tests.

Split out of ``tests/test_post_mortem.py`` (issue #1567, Track 1): the
sessions.db / per-PID-log / worktree-files-mtime activity sources (issue
#353), probe freshness and payload serialization, normalized-path and real
fleet working_directory matching (issues #281, #343), and the worker_kind
Devin-source gating (issue #639).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from _post_mortem_fixtures import (
    _build_sessions_db,
    _config_with_db,
    _make_worker,
)

from charlie_work.config import (
    PostMortemConfig,
    WatchdogConfig,
)
from charlie_work.post_mortem import (
    ActivitySource,
    RealActivityProbe,
    real_activity_for_worker,
)


def test_real_activity_for_worker_sessions_db_source(tmp_path: Path) -> None:
    """A matched sessions.db with message_nodes is the freshest real activity
    source. Node created_at values are epoch integers here — the real
    production shape (message_nodes.created_at is declared INTEGER)."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[
            ("user", "hello", int(datetime(2026, 7, 11, 11, 55, 0, tzinfo=UTC).timestamp())),
            (
                "assistant",
                "working",
                int(datetime(2026, 7, 11, 11, 59, 0, tzinfo=UTC).timestamp()),
            ),
        ],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(pid=12345)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        worker.worktree_path,
        worker.started_at,
        worker.pid,
        now,
    )

    assert probe.latest_source == "sessions.db"
    assert probe.latest_timestamp == datetime(2026, 7, 11, 11, 59, 0, tzinfo=UTC)
    # Per-PID log directory does not exist, so it should be recorded as missing.
    per_pid = next(s for s in probe.sources if s.name == "devin_per_pid_log")
    assert per_pid.error is not None
    assert per_pid.timestamp is None


def test_real_activity_for_worker_per_pid_log_source(tmp_path: Path) -> None:
    """Per-PID Devin log mtime is picked when sessions.db is missing."""
    db_path = tmp_path / "sessions.db"
    # sessions.db is explicitly non-existent so the probe must fall back cleanly.
    config = _config_with_db(db_path)
    worker = _make_worker(pid=99999)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    log_path = logs_dir / f"devin_20260711_114500_{worker.pid}.log"
    log_path.write_text("some devin log\n")
    mtime = datetime(2026, 7, 11, 11, 58, 0, tzinfo=UTC).timestamp()
    log_path.touch()
    # Set explicit mtime after touch
    import os

    os.utime(log_path, (mtime, mtime))

    probe = real_activity_for_worker(
        config.post_mortem,
        worker.worktree_path,
        worker.started_at,
        worker.pid,
        now,
    )

    assert probe.latest_source == "devin_per_pid_log"
    assert probe.latest_timestamp == datetime(2026, 7, 11, 11, 58, 0, tzinfo=UTC)
    sessions_db = next(s for s in probe.sources if s.name == "sessions.db")
    assert sessions_db.error is not None
    assert sessions_db.timestamp is None


def test_real_activity_for_worker_prefers_latest_source(tmp_path: Path) -> None:
    """latest_timestamp and latest_source come from the freshest of both sources."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[
            ("assistant", "plan", "2026-07-11T11:57:00"),
        ],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(pid=11111)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True)
    log_path = logs_dir / f"devin_20260711_115000_{worker.pid}.log"
    log_path.write_text("log\n")
    mtime = datetime(2026, 7, 11, 11, 59, 30, tzinfo=UTC).timestamp()
    import os

    os.utime(log_path, (mtime, mtime))

    probe = real_activity_for_worker(
        config.post_mortem,
        worker.worktree_path,
        worker.started_at,
        worker.pid,
        now,
    )

    assert probe.latest_source == "devin_per_pid_log"
    assert probe.latest_timestamp == datetime(2026, 7, 11, 11, 59, 30, tzinfo=UTC)


def test_real_activity_for_worker_worktree_files_mtime_source(tmp_path: Path) -> None:
    """Issue #353: worktree file mtimes are a fourth real-activity source."""
    db_path = tmp_path / "sessions.db"
    config = _config_with_db(db_path)
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    source_file = worktree_path / "src" / "foo.py"
    source_file.parent.mkdir()
    source_file.write_text("# hello", encoding="utf-8")

    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    mtime = (now - timedelta(minutes=15)).timestamp()
    os.utime(source_file, (mtime, mtime))

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
        worktree_mtime_max_depth=4,
    )
    probe = real_activity_for_worker(
        config.post_mortem,
        str(worktree_path),
        "2026-07-11T11:30:00+00:00",
        None,
        now,
        watchdog_config=watchdog,
    )

    source = next(s for s in probe.sources if s.name == "worktree_files_mtime")
    assert source.error is None
    assert source.timestamp == datetime(2026, 7, 11, 11, 45, 0, tzinfo=UTC)
    assert source.staleness_seconds == 15 * 60
    assert source.threshold_minutes == 45
    assert probe.latest_source == "worktree_files_mtime"
    assert probe.latest_timestamp == datetime(2026, 7, 11, 11, 45, 0, tzinfo=UTC)
    # The per-source threshold lets a 15-minute-old worktree write veto a 20-minute stall window.
    assert probe.is_fresh(20) is True


def test_real_activity_for_worker_worktree_files_mtime_stale(tmp_path: Path) -> None:
    """Issue #353: worktree mtime older than its threshold is not fresh."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / "file.txt").write_text("x", encoding="utf-8")

    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    mtime = (now - timedelta(minutes=60)).timestamp()
    os.utime(worktree_path / "file.txt", (mtime, mtime))

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
    )
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        "2026-07-11T10:00:00+00:00",
        None,
        now,
        watchdog_config=watchdog,
    )

    source = next(s for s in probe.sources if s.name == "worktree_files_mtime")
    assert source.timestamp == datetime(2026, 7, 11, 11, 0, 0, tzinfo=UTC)
    assert source.threshold_minutes == 45
    assert probe.is_fresh(20) is False


def test_real_activity_for_worker_worktree_files_mtime_checkout_noise_ignored(
    tmp_path: Path,
) -> None:
    """Issue #353: checkout-time mtimes are not treated as post-start activity.

    A freshly-checked-out worktree whose files all date to session start and
    have not been written to since must not veto a stall verdict.
    """
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    source_file = worktree_path / "foo.py"
    source_file.write_text("# hello", encoding="utf-8")

    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    started_at = datetime(2026, 7, 11, 11, 30, 0, tzinfo=UTC)
    checkout_mtime = started_at.timestamp()
    os.utime(source_file, (checkout_mtime, checkout_mtime))

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
    )
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        started_at.isoformat(),
        None,
        now,
        watchdog_config=watchdog,
    )

    source = next(s for s in probe.sources if s.name == "worktree_files_mtime")
    assert source.error is None
    assert source.timestamp == started_at
    assert source.threshold_minutes == 0
    assert source.staleness_seconds == (now - started_at).total_seconds()
    assert probe.is_fresh(20) is False
    assert probe.is_fresh(5) is False


def test_real_activity_for_worker_worktree_files_mtime_depth_and_exclude(
    tmp_path: Path,
) -> None:
    """Issue #353: worktree mtime scan respects max_depth and excluded directories."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / "a.txt").write_text("a", encoding="utf-8")
    deep_dir = worktree_path / "deep" / "nested"
    deep_dir.mkdir(parents=True)
    (deep_dir / "b.txt").write_text("b", encoding="utf-8")
    git_dir = worktree_path / ".git"
    git_dir.mkdir()
    (git_dir / "c.txt").write_text("c", encoding="utf-8")

    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    os.utime(worktree_path / "a.txt", ((now - timedelta(minutes=10)).timestamp(),) * 2)
    os.utime(deep_dir / "b.txt", ((now - timedelta(minutes=5)).timestamp(),) * 2)
    os.utime(git_dir / "c.txt", ((now - timedelta(minutes=1)).timestamp(),) * 2)

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
        worktree_mtime_max_depth=1,
        worktree_mtime_exclude_dirs=(".git", ".venv"),
    )
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        "2026-07-11T11:30:00+00:00",
        None,
        now,
        watchdog_config=watchdog,
    )

    source = next(s for s in probe.sources if s.name == "worktree_files_mtime")
    # max_depth=1 includes the root and one level below it; the file at deep/nested is too deep.
    # .git is excluded, so its 1-minute-old file must not dominate the result.
    assert source.timestamp == datetime(2026, 7, 11, 11, 50, 0, tzinfo=UTC)


def test_real_activity_for_worker_worktree_files_mtime_disabled(tmp_path: Path) -> None:
    """Issue #353: worktree mtime source can be disabled."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    (worktree_path / "file.txt").write_text("x", encoding="utf-8")

    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    watchdog = WatchdogConfig(worktree_mtime_enabled=False)
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        "",
        None,
        now,
        watchdog_config=watchdog,
    )

    assert not any(s.name == "worktree_files_mtime" for s in probe.sources)


def test_real_activity_for_worker_worktree_files_mtime_missing_path(tmp_path: Path) -> None:
    """Issue #353: a missing worktree is recorded as an errored source, not a crash."""
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    watchdog = WatchdogConfig(worktree_mtime_enabled=True)
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(tmp_path / "does-not-exist"),
        "",
        None,
        now,
        watchdog_config=watchdog,
    )

    source = next(s for s in probe.sources if s.name == "worktree_files_mtime")
    assert source.error is not None
    assert source.timestamp is None


def test_real_activity_probe_is_fresh_uses_source_threshold() -> None:
    """Issue #353: RealActivityProbe.is_fresh honors per-source thresholds."""
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    stale_for_short_window = now - timedelta(minutes=30)
    worktree_source = ActivitySource(
        name="worktree_files_mtime",
        timestamp=stale_for_short_window,
        staleness_seconds=30 * 60,
        error=None,
        threshold_minutes=45,
    )
    probe = RealActivityProbe(sources=(worktree_source,))
    assert probe.is_fresh(20) is True
    assert probe.is_fresh(60) is True

    generic_source = ActivitySource(
        name="sessions.db",
        timestamp=stale_for_short_window,
        staleness_seconds=30 * 60,
        error=None,
    )
    probe_generic = RealActivityProbe(sources=(generic_source,))
    assert probe_generic.is_fresh(20) is False
    assert probe_generic.is_fresh(60) is True


def test_real_activity_probe_to_payload() -> None:
    """to_payload serializes datetimes into JSON-safe strings and lists sources."""
    ts = datetime(2026, 7, 11, 11, 59, 0, tzinfo=UTC)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(name="sessions.db", timestamp=ts, staleness_seconds=60.0, error=None),
            ActivitySource(
                name="devin_per_pid_log", timestamp=None, staleness_seconds=None, error="no pid"
            ),
        )
    )
    payload = probe.to_payload()
    assert payload["latest_timestamp"] == "2026-07-11T11:59:00+00:00"
    assert payload["latest_source"] == "sessions.db"
    assert payload["sources"][0]["timestamp"] == "2026-07-11T11:59:00+00:00"
    assert payload["sources"][0]["staleness_seconds"] == 60.0
    assert payload["sources"][1]["error"] == "no pid"


# ---------------------------------------------------------------------------
# working_directory normalization (issue #281)
# ---------------------------------------------------------------------------


def test_real_activity_for_worker_matches_normalized_worktree_path(
    tmp_path: Path,
) -> None:
    """real_activity_for_worker must also match a sessions.db row whose
    working_directory is a differently-formatted path for the same logical
    worktree."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="/c/repo/.var/worktrees/issue-42",
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker(pid=12345)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        r"C:\repo\.var\worktrees\issue-42",
        worker.started_at,
        worker.pid,
        now,
    )

    assert probe.latest_source == "sessions.db"
    assert probe.latest_timestamp == datetime(2026, 7, 11, 11, 57, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Issue #343: working_directory suffix-match fallback for real fleet shapes
# ---------------------------------------------------------------------------


def test_real_activity_for_worker_matches_real_fleet_working_directory_shape(
    tmp_path: Path,
) -> None:
    """sessions.db corroboration must match a fleet worker session even when
    the Devin CLI recorded a ``working_directory`` sharing no absolute-prefix
    relationship at all with the worktree path this process computed.

    Fixture shapes are taken verbatim from a live production post-mortem
    (issue #343, 2026-07-13, ``issue-203.post-mortem.json``): every sampled
    distinct ``working_directory`` in the real sessions.db was rooted under
    an unrelated ``AppData\\Local\\Temp\\...`` tree, never under the fleet
    worktree's own ``repos\\charlie-work\\.var\\charlie-work\\worktrees\\...``
    root -- even though the trailing worktrees-dir/issue-slug segments that
    identify the dispatch are identical. Before the suffix-match fallback,
    neither the exact nor the normalized full-path comparison in
    ``_find_matching_session`` could ever match this shape, so sessions.db
    corroboration was permanently inconclusive for real fleet dispatches.

    MUTATION GATE: reverting ``_find_matching_session``'s suffix-match tier
    (or ``_working_directory_suffix``) to fall straight from the normalized
    comparison to "no session found" makes this test fail -- the fixture row's
    working_directory has a different drive letter AND a completely different
    directory tree above the shared ``worktrees/<issue-slug>`` tail, so
    neither the exact-match nor the ``_normalize_working_directory`` tier can
    find it.
    """
    db_path = tmp_path / "sessions.db"
    real_fleet_worktree_path = (
        r"C:\Users\operator\repos\charlie-work\.var\charlie-work\worktrees"
        r"\agent-issue-203-redundant-re-dispatch"
    )
    # Recorded working_directory shares zero prefix with the worktree path
    # above -- only the trailing (worktrees-dir, issue-slug) segment pair
    # matches, exactly like the production sample values.
    recorded_working_directory = (
        r"C:\Users\operator\AppData\Local\Temp\claude\some-other-session-root"
        r"\worktrees\agent-issue-203-redundant-re-dispatch"
    )
    _build_sessions_db(
        db_path,
        working_directory=recorded_working_directory,
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        real_fleet_worktree_path,
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
    )

    assert probe.latest_source == "sessions.db"
    assert probe.latest_timestamp == datetime(2026, 7, 11, 11, 57, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Issue #639: worker_kind skips Devin sources for non-Devin workers
# ---------------------------------------------------------------------------


def test_real_activity_for_worker_skips_devin_sources_for_non_devin_kind(
    tmp_path: Path,
) -> None:
    """Issue #639: when ``worker_kind`` is a non-Devin adapter (claude-code,
    api, manual, command), the Devin-specific sources (sessions.db and
    per-PID Devin log) are skipped entirely — a non-Devin worker has no
    Devin subject to look up. The probe must not contain those sources at
    all, so no permanent "no session found" / "no pid" errors are produced.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        "C:/repo/.var/worktrees/issue-42",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
        worker_kind="claude-code",
    )

    # No Devin sources at all — they were skipped.
    source_names = {s.name for s in probe.sources}
    assert "sessions.db" not in source_names
    assert "devin_per_pid_log" not in source_names
    # No errored sources (the Devin sources that would have errored are absent).
    assert all(s.error is None for s in probe.sources)


def test_real_activity_for_worker_skips_devin_sources_for_api_kind(
    tmp_path: Path,
) -> None:
    """Issue #639: ``api``-routed workers (which delegate to claude-code) also
    have no Devin subject. The Devin sources must be skipped for
    ``worker_kind="api"`` just as for ``"claude-code"``.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        "C:/repo/.var/worktrees/issue-42",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
        worker_kind="api",
    )

    source_names = {s.name for s in probe.sources}
    assert "sessions.db" not in source_names
    assert "devin_per_pid_log" not in source_names


def test_real_activity_for_worker_consults_devin_sources_for_devin_shell(
    tmp_path: Path,
) -> None:
    """Issue #639 regression guard: ``worker_kind="devin-shell"`` must still
    consult the Devin sources. The skip only applies to non-Devin kinds.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        "C:/repo/.var/worktrees/issue-42",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
        worker_kind="devin-shell",
    )

    source_names = {s.name for s in probe.sources}
    assert "sessions.db" in source_names
    assert "devin_per_pid_log" in source_names
    # sessions.db matched and has a fresh timestamp.
    assert probe.latest_source == "sessions.db"


def test_real_activity_for_worker_consults_devin_sources_for_devin_view_kind(
    tmp_path: Path,
) -> None:
    """Issue #639: the ``WorkerView.adapter_kind`` convention uses
    ``"devin"`` (not ``"devin-shell"``). Both must be recognized as Devin
    kinds so the watchdog's ``real_activity_probe_for`` wrapper — which passes
    ``view.adapter_kind`` — does not accidentally skip Devin sources for
    Devin-shell workers.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        "C:/repo/.var/worktrees/issue-42",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
        worker_kind="devin",
    )

    source_names = {s.name for s in probe.sources}
    assert "sessions.db" in source_names
    assert "devin_per_pid_log" in source_names


def test_real_activity_for_worker_consults_all_sources_when_kind_unknown(
    tmp_path: Path,
) -> None:
    """Issue #639: ``worker_kind=None`` (unknown) preserves the pre-#639
    behavior — all sources are consulted. This is the backward-compatibility
    path for callers that have not been updated to pass ``worker_kind``.
    """
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        working_directory="C:/repo/.var/worktrees/issue-42",
        nodes=[("assistant", "working", "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    now = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)

    probe = real_activity_for_worker(
        config.post_mortem,
        "C:/repo/.var/worktrees/issue-42",
        "2026-07-11T11:55:00+00:00",
        12345,
        now,
    )

    source_names = {s.name for s in probe.sources}
    assert "sessions.db" in source_names
    assert "devin_per_pid_log" in source_names
