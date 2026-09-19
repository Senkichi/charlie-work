"""Wall-clock and loop-duration tripwire tests for ``classify_worker_health``.

Split out of ``tests/test_worker_health.py`` (issue #1568, Track-1):
the wall-clock-since-start tripwire and the loop-duration tripwire
(slow/runaway thresholds, stale-log skip, devin exemption, missing
events file, claude log layout).
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from charlie_work.config import (
    OrchestratorConfig,
    WatchdogConfig,
)
from charlie_work.worker import (
    WorkerHealth,
    WorkerView,
    classify_worker_health,
)


def test_classify_worker_health_wall_clock_slow_default(tmp_path: Path) -> None:
    """Wall-clock tripwire returns SLOW at default config (wall_clock_kill=False)."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # Set started_at to 241 minutes ago (past the 240-minute default)
    old_start = datetime.now(UTC) - timedelta(minutes=241)

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=old_start.isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        config = OrchestratorConfig()  # Default: wall_clock_kill=False
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.SLOW


def test_classify_worker_health_wall_clock_runaway_with_kill(tmp_path: Path) -> None:
    """Wall-clock tripwire returns RUNAWAY when wall_clock_kill=True."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # Set started_at to 241 minutes ago (past the 240-minute default)
    old_start = datetime.now(UTC) - timedelta(minutes=241)

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=old_start.isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        config = OrchestratorConfig(watchdog=WatchdogConfig(wall_clock_kill=True))
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.RUNAWAY


def test_classify_worker_health_wall_clock_within_threshold(tmp_path: Path) -> None:
    """Wall-clock tripwire does not fire when started_at is within threshold."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # Set started_at to 60 minutes ago (well within the 240-minute default)
    recent_start = datetime.now(UTC) - timedelta(minutes=60)

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

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        config = OrchestratorConfig()
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # Should be HEALTHY since wall-clock threshold not exceeded
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_loop_slow_default(tmp_path: Path) -> None:
    """Loop/no-progress tripwire returns SLOW at default config (loop_kill=False) for Claude Code."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # Create events.jsonl with last tool call 41 minutes ago (past 2 * 20 = 40 min default)
    events_file = tmp_path / "test.events.jsonl"
    old_tool_call = datetime.now(UTC) - timedelta(minutes=41)
    events_file.write_text(
        f'{{"type": "tool_call", "timestamp": "{old_tool_call.isoformat()}"}}\n',
        encoding="utf-8",
    )

    # Set log mtime to 5 minutes ago (fresh, within stall_minutes)
    import os
    import time

    recent_log_time = datetime.now(UTC) - timedelta(minutes=5)
    os.utime(log_file, (time.time(), recent_log_time.timestamp()))

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=datetime.now(UTC).isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig()  # Default: loop_kill=False
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.SLOW


def test_classify_worker_health_loop_runaway_with_kill(tmp_path: Path) -> None:
    """Loop/no-progress tripwire returns RUNAWAY when loop_kill=True for Claude Code."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # Create events.jsonl with last tool call 41 minutes ago
    events_file = tmp_path / "test.events.jsonl"
    old_tool_call = datetime.now(UTC) - timedelta(minutes=41)
    events_file.write_text(
        f'{{"type": "tool_call", "timestamp": "{old_tool_call.isoformat()}"}}\n',
        encoding="utf-8",
    )

    # Set log mtime to 5 minutes ago (fresh)
    import os
    import time

    recent_log_time = datetime.now(UTC) - timedelta(minutes=5)
    os.utime(log_file, (time.time(), recent_log_time.timestamp()))

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=datetime.now(UTC).isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(watchdog=WatchdogConfig(loop_kill=True))
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.RUNAWAY


def test_classify_worker_health_loop_skipped_when_log_stale(tmp_path: Path) -> None:
    """Loop tripwire does not fire when log is also stale (STALLED wins first)."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # Create events.jsonl with last tool call 41 minutes ago
    events_file = tmp_path / "test.events.jsonl"
    old_tool_call = datetime.now(UTC) - timedelta(minutes=41)
    events_file.write_text(
        f'{{"type": "tool_call", "timestamp": "{old_tool_call.isoformat()}"}}\n',
        encoding="utf-8",
    )

    # Set log mtime to 30 minutes ago (stale, past stall_minutes)
    import os
    import time

    stale_log_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_file, (time.time(), stale_log_time.timestamp()))

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=datetime.now(UTC).isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig()
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # STALLED wins (signal 3) before loop tripwire (signal 5)
        assert health == WorkerHealth.STALLED


def test_classify_worker_health_loop_devin_never_runaway(tmp_path: Path) -> None:
    """Devin workers never return RUNAWAY from the loop tripwire, regardless of config."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # Set log mtime to 5 minutes ago (fresh)
    import os
    import time

    recent_log_time = datetime.now(UTC) - timedelta(minutes=5)
    os.utime(log_file, (time.time(), recent_log_time.timestamp()))

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=datetime.now(UTC).isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        # Even with loop_kill=True, Devin should never return RUNAWAY from this tripwire
        config = OrchestratorConfig(watchdog=WatchdogConfig(loop_kill=True))
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # Should be HEALTHY (loop tripwire skipped entirely for Devin)
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_loop_no_events_file(tmp_path: Path) -> None:
    """Loop tripwire is skipped when events.jsonl does not exist (no error raised)."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # No events.jsonl file created

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=datetime.now(UTC).isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig()
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # Should be HEALTHY (loop tripwire skipped, falls through to signal 6)
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_loop_claude_log_layout_runaway(tmp_path: Path) -> None:
    """Issue #329: Signal 5 must find the real issue-N.events.jsonl sibling for a claude-code log.

    A claude-code log named ``issue-42.claude.log`` has a sibling named
    ``issue-42.events.jsonl`` (not ``issue-42.claude.events.jsonl``, which the
    old ``with_suffix('.events.jsonl')`` derivation produced). This regression
    test uses the real file layout and expects the loop/no-progress tripwire to
    fire and return RUNAWAY.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    issue_number = 42
    log_file = sessions_dir / f"issue-{issue_number}.claude.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # Set log mtime to 5 minutes ago (fresh, within stall_minutes)
    recent_log_time = datetime.now(UTC) - timedelta(minutes=5)
    os.utime(log_file, (time.time(), recent_log_time.timestamp()))

    # Create the real events.jsonl sibling with a stale tool call (past 2 * stall_minutes)
    events_file = sessions_dir / f"issue-{issue_number}.events.jsonl"
    old_tool_call = datetime.now(UTC) - timedelta(minutes=41)
    events_file.write_text(
        f'{{"type": "tool_call", "timestamp": "{old_tool_call.isoformat()}"}}\n',
        encoding="utf-8",
    )

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=issue_number,
        repo_key="",
        pid=12345,
        started_at=datetime.now(UTC).isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(watchdog=WatchdogConfig(loop_kill=True))
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.RUNAWAY
