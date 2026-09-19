"""Stale-log activity-signal tests for ``classify_worker_health``.

Split out of ``tests/test_worker_health.py`` (issue #1568, Track-1):
log-mtime stalling and the fresh-activity overrides that rescue a
stale sidecar -- worktree-file mtimes (including checkout noise) and
claude events.jsonl recency.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from charlie_work.config import (
    OrchestratorConfig,
    PostMortemConfig,
    WatchdogConfig,
)
from charlie_work.post_mortem import ActivitySource, RealActivityProbe
from charlie_work.worker import (
    WorkerHealth,
    WorkerView,
    classify_worker_health,
    real_activity_probe_for,
)


def test_classify_worker_health_stalled_by_mtime(tmp_path: Path) -> None:
    """classify_worker_health returns STALLED for a live worker with stale log mtime."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # Set log mtime to 30 minutes ago
    old_time = datetime.now(UTC) - timedelta(minutes=30)
    import os
    import time

    os.utime(log_file, (time.time(), old_time.timestamp()))

    # Use a recent started_at to avoid triggering the wall-clock tripwire
    recent_start = datetime.now(UTC) - timedelta(minutes=10)

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=recent_start.isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    # Mock the underlying adapter liveness function that is_alive() calls
    with patch("charlie_work.worker.is_session_alive", return_value=True):
        config = OrchestratorConfig()
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.STALLED


def test_classify_worker_health_stalled_by_mtime_overridden_by_real_activity(
    tmp_path: Path,
) -> None:
    """Issue #280: stale sidecar mtime is not a kill if real activity is fresh."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=30)
    import os
    import time

    os.utime(log_file, (time.time(), old_time.timestamp()))

    recent_start = datetime.now(UTC) - timedelta(minutes=10)

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=recent_start.isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    now = datetime.now(UTC)
    fresh_timestamp = now - timedelta(minutes=1)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=fresh_timestamp,
                staleness_seconds=(now - fresh_timestamp).total_seconds(),
                error=None,
            ),
        )
    )

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        config = OrchestratorConfig()
        health = classify_worker_health(view, config, now, probe)
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_worktree_files_mtime_overrides_stale_log(
    tmp_path: Path,
) -> None:
    """Issue #353: a live worker with a stale sidecar log but recent worktree writes is healthy."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    now = datetime.now(UTC)
    old_time = now - timedelta(minutes=30)
    os.utime(log_file, (old_time.timestamp(), old_time.timestamp()))

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    source_file = worktree_path / "foo.py"
    source_file.write_text("# change", encoding="utf-8")
    worktree_mtime = now - timedelta(minutes=5)
    os.utime(source_file, (worktree_mtime.timestamp(), worktree_mtime.timestamp()))

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=(now - timedelta(minutes=10)).isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path=str(worktree_path),
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(stall_minutes=20, worktree_mtime_threshold_minutes=45),
            post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
        )
        probe = real_activity_probe_for(view, config, now)
        health = classify_worker_health(view, config, now, probe)
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_worktree_files_mtime_checkout_noise_stalls(
    tmp_path: Path,
) -> None:
    """Issue #353: checkout-time mtimes do not hide a stalled live worker."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    now = datetime.now(UTC)
    old_time = now - timedelta(minutes=30)
    os.utime(log_file, (old_time.timestamp(), old_time.timestamp()))

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    source_file = worktree_path / "foo.py"
    source_file.write_text("# change", encoding="utf-8")
    started_at = now - timedelta(minutes=30)
    os.utime(source_file, (started_at.timestamp(), started_at.timestamp()))

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=started_at.isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path=str(worktree_path),
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(stall_minutes=20, worktree_mtime_threshold_minutes=45),
            post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
        )
        probe = real_activity_probe_for(view, config, now)
        health = classify_worker_health(view, config, now, probe)
        assert health == WorkerHealth.STALLED


def test_classify_worker_health_claude_events_override_stale_log(tmp_path: Path) -> None:
    """Issue #301: a claude-code worker with frozen sidecar log but fresh events.jsonl is not stalled."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_file = sessions_dir / "issue-1.claude.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # Freeze sidecar log mtime
    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_file, (time.time(), old_time.timestamp()))

    # Fresh events.jsonl sibling
    events_file = sessions_dir / "issue-1.events.jsonl"
    fresh_time = datetime.now(UTC) - timedelta(minutes=1)
    events_file.write_text(
        f'{{"type": "tool_call", "timestamp": "{fresh_time.isoformat()}"}}\n',
        encoding="utf-8",
    )
    os.utime(events_file, (time.time(), fresh_time.timestamp()))

    recent_start = datetime.now(UTC) - timedelta(minutes=10)
    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=recent_start.isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path=str(tmp_path / "worktree"),
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
        )
        now = datetime.now(UTC)
        probe = real_activity_probe_for(view, config, now)
        health = classify_worker_health(view, config, now, probe)
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_claude_events_both_quiet_stalled(tmp_path: Path) -> None:
    """Issue #301: a claude-code worker with both stale sidecar log and stale events.jsonl is stalled."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_file = sessions_dir / "issue-1.claude.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_file, (time.time(), old_time.timestamp()))

    events_file = sessions_dir / "issue-1.events.jsonl"
    stale_time = datetime.now(UTC) - timedelta(minutes=30)
    events_file.write_text(
        f'{{"type": "tool_call", "timestamp": "{stale_time.isoformat()}"}}\n',
        encoding="utf-8",
    )
    os.utime(events_file, (time.time(), stale_time.timestamp()))

    recent_start = datetime.now(UTC) - timedelta(minutes=10)
    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=recent_start.isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path=str(tmp_path / "worktree"),
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
        )
        now = datetime.now(UTC)
        probe = real_activity_probe_for(view, config, now)
        health = classify_worker_health(view, config, now, probe)
        assert health == WorkerHealth.STALLED
