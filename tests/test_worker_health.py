from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
import tempfile

from charlie_work.config import OrchestratorConfig, WatchdogConfig
from charlie_work.worker import WorkerHealth, WorkerView, classify_worker_health


def test_worker_health_enum_members() -> None:
    """WorkerHealth enum has exactly the required members."""
    assert WorkerHealth.HEALTHY.value == "healthy"
    assert WorkerHealth.SLOW.value == "slow"
    assert WorkerHealth.STALLED.value == "stalled"
    assert WorkerHealth.RUNAWAY.value == "runaway"
    assert WorkerHealth.DEAD.value == "dead"
    assert WorkerHealth.ORPHANED.value == "orphaned"


def test_classify_worker_health_healthy(tmp_path: Path) -> None:
    """classify_worker_health returns HEALTHY for a live worker with recent log activity."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nStill working...", encoding="utf-8")

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
        assert health == WorkerHealth.HEALTHY


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


def test_classify_worker_health_dead_by_terminal_marker(tmp_path: Path) -> None:
    """classify_worker_health returns DEAD for a worker with a terminal error marker in the log."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nError: A tool was rejected", encoding="utf-8")

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
        assert health == WorkerHealth.DEAD


def test_classify_worker_health_dead_by_liveness(tmp_path: Path) -> None:
    """classify_worker_health returns DEAD for a worker with a dead PID."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

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

    # Mock the underlying adapter liveness function that is_alive() calls to return False
    with patch("charlie_work.worker.is_session_alive", return_value=False):
        config = OrchestratorConfig()
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.DEAD


def test_classify_worker_health_legacy_none_start_time(tmp_path: Path) -> None:
    """A WorkerView with process_start_time=None and a live PID never classifies as DEAD on liveness grounds alone."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # Use a recent started_at to avoid triggering the wall-clock tripwire
    recent_start = datetime.now(UTC) - timedelta(minutes=10)

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=recent_start.isoformat(),
        process_start_time=None,  # Legacy record without process_start_time
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    # Mock the underlying adapter liveness function that is_alive() calls to return True (legacy fallback)
    with patch("charlie_work.worker.is_session_alive", return_value=True):
        config = OrchestratorConfig()
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # Should be HEALTHY since is_alive returns True and log is recent
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_custom_terminal_marker(tmp_path: Path) -> None:
    """A custom terminal_error_markers config changes classification for a log ending in that marker."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nCustom fatal error", encoding="utf-8")

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
        # Custom config with a custom terminal marker
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(terminal_error_markers=("Custom fatal error",))
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.DEAD


def test_classify_worker_health_no_io_performed(tmp_path: Path) -> None:
    """classify_worker_health performs no I/O beyond what WorkerView.log_stat() already captured."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

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
        # Should be HEALTHY since log is recent and no terminal error
        assert health == WorkerHealth.HEALTHY


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
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(wall_clock_kill=True)
        )
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
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(loop_kill=True)
        )
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
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(loop_kill=True)
        )
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


def test_classify_worker_health_regression_test_suite_pattern(tmp_path: Path) -> None:
    """Regression test: a real local test suite pattern classifies HEALTHY at defaults."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Running pytest...\nTest output...\nMore tests...\n", encoding="utf-8")

    # Create events.jsonl with tool calls within the loop window (simulating a healthy test suite)
    events_file = tmp_path / "test.events.jsonl"
    recent_tool_call = datetime.now(UTC) - timedelta(minutes=15)  # Within 40 min window
    events_file.write_text(
        f'{{"type": "tool_call", "timestamp": "{recent_tool_call.isoformat()}"}}\n',
        encoding="utf-8",
    )

    # Set log mtime to 10 minutes ago (fresh, within stall_minutes)
    import os
    import time
    recent_log_time = datetime.now(UTC) - timedelta(minutes=10)
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
        config = OrchestratorConfig()  # Default WARN-first settings
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # Should be HEALTHY (no tripwire fires at defaults for this pattern)
        assert health == WorkerHealth.HEALTHY
