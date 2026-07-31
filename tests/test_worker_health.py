from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from charlie_work.config import (
    ApiBudgetConfig,
    ApiWorkerConfig,
    OrchestratorConfig,
    PostMortemConfig,
    WatchdogConfig,
)
from charlie_work.post_mortem import ActivitySource, RealActivityProbe
from charlie_work.worker import (
    WorkerHealth,
    WorkerView,
    classify_worker_health,
    issue_worker_liveness,
    parse_cumulative_usage,
    real_activity_probe_for,
)


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


def test_classify_worker_health_indeterminate_liveness_does_not_bypass_deferral(
    tmp_path: Path,
) -> None:
    """Issue #360 criterion #1: an indeterminate liveness probe is not a definitive dead signal.

    When ``get_process_start_time`` returns ``None`` for a live PID,
    ``is_session_alive`` returns ``True`` (indeterminate).  A stale sidecar log
    should still classify as ``STALLED`` (not ``DEAD``), because the liveness
    signal was not definitive and the deferral cap must not be bypassed for an
    indeterminate probe.
    """
    import subprocess
    import sys
    import charlie_work.process_utils as process_utils

    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # Set log mtime to 30 minutes ago
    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_file, (time.time(), old_time.timestamp()))

    recent_start = datetime.now(UTC) - timedelta(minutes=10)

    # Spawn a real short-lived process
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )
    try:
        view = WorkerView(
            adapter_kind="devin",
            issue_number=1,
            repo_key="",
            pid=proc.pid,
            started_at=recent_start.isoformat(),
            process_start_time=123.456,
            log_path=str(log_file),
            worktree_path="",
            error=None,
            failure_kind=None,
            reclaimed=None,
        )

        # Simulate an indeterminate start-time probe while the process is alive.
        with patch.object(process_utils, "get_process_start_time", return_value=None):
            config = OrchestratorConfig()
            now = datetime.now(UTC)
            health = classify_worker_health(view, config, now)

        # Indeterminate liveness must not be treated as DEAD (bypassing the
        # deferral cap).  With a stale log and no corroborating real activity,
        # the worker is classified as STALLED.
        assert health != WorkerHealth.DEAD
        assert health == WorkerHealth.STALLED
    finally:
        proc.kill()
        proc.wait(timeout=5)


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
        assert health == WorkerHealth.HEALTHY


# Tests for parse_cumulative_usage (issue #163)


def test_parse_cumulative_usage_missing_file(tmp_path: Path) -> None:
    """parse_cumulative_usage returns None when the events file doesn't exist."""
    events_file = tmp_path / "issue-1.events.jsonl"
    usage = parse_cumulative_usage(events_file)
    assert usage is None


def test_parse_cumulative_usage_wellformed_jsonl(tmp_path: Path) -> None:
    """parse_cumulative_usage returns the latest cumulative values from well-formed JSONL."""
    events_file = tmp_path / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 1000, "cost_usd": 0.01}\n'
        '{"type": "user_message", "tokens": 2000, "cost_usd": 0.02}\n'
        '{"type": "assistant_message", "tokens": 3000, "cost_usd": 0.03}\n',
        encoding="utf-8",
    )

    usage = parse_cumulative_usage(events_file)
    assert usage is not None
    assert usage.tokens == 3000
    assert usage.cost_usd == 0.03


def test_parse_cumulative_usage_truncated_trailing_line(tmp_path: Path) -> None:
    """parse_cumulative_usage ignores a truncated trailing line and returns prior valid values."""
    events_file = tmp_path / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 1000, "cost_usd": 0.01}\n'
        '{"type": "user_message", "tokens": 2000, "cost_usd": 0.02}\n'
        '{"type": "assistant_message", "tokens": 3000, "cost_usd": 0.03',  # Truncated JSON
        encoding="utf-8",
    )

    usage = parse_cumulative_usage(events_file)
    assert usage is not None
    assert usage.tokens == 2000
    assert usage.cost_usd == 0.02


def test_parse_cumulative_usage_empty_file(tmp_path: Path) -> None:
    """parse_cumulative_usage returns None for an empty file."""
    events_file = tmp_path / "issue-1.events.jsonl"
    events_file.write_text("", encoding="utf-8")

    usage = parse_cumulative_usage(events_file)
    assert usage is None


def test_parse_cumulative_usage_no_usage_fields(tmp_path: Path) -> None:
    """parse_cumulative_usage returns None when events have no tokens/cost_usd fields."""
    events_file = tmp_path / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call"}\n{"type": "user_message"}',
        encoding="utf-8",
    )

    usage = parse_cumulative_usage(events_file)
    assert usage is None


# Tests for cost/token tripwire in classify_worker_health (issue #163)


def test_classify_worker_health_cost_budget_warn_mode(tmp_path: Path) -> None:
    """Cost budget exceeded with default warn mode returns SLOW, not RUNAWAY."""
    log_file = tmp_path / "sessions" / "issue-1.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("Working on task...", encoding="utf-8")

    events_file = tmp_path / "sessions" / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 1000, "cost_usd": 10.0}',
        encoding="utf-8",
    )

    # Use a recent started_at to avoid triggering the wall-clock/loop tripwires
    recent_start = datetime.now(UTC) - timedelta(minutes=10)

    view = WorkerView(
        adapter_kind="claude-code",
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

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(cost_budget_usd=5.0, cost_budget_action="warn")
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.SLOW


def test_classify_worker_health_cost_budget_kill_mode(tmp_path: Path) -> None:
    """Cost budget exceeded with kill mode returns RUNAWAY."""
    log_file = tmp_path / "sessions" / "issue-1.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("Working on task...", encoding="utf-8")

    events_file = tmp_path / "sessions" / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 1000, "cost_usd": 10.0}',
        encoding="utf-8",
    )

    # Use a recent started_at to avoid triggering the wall-clock/loop tripwires
    recent_start = datetime.now(UTC) - timedelta(minutes=10)

    view = WorkerView(
        adapter_kind="claude-code",
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

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(cost_budget_usd=5.0, cost_budget_action="kill")
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.RUNAWAY


def test_classify_worker_health_token_budget_warn_mode(tmp_path: Path) -> None:
    """Token budget exceeded with default warn mode returns SLOW, not RUNAWAY."""
    log_file = tmp_path / "sessions" / "issue-1.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("Working on task...", encoding="utf-8")

    events_file = tmp_path / "sessions" / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 100000, "cost_usd": 0.01}',
        encoding="utf-8",
    )

    # Use a recent started_at to avoid triggering the wall-clock/loop tripwires
    recent_start = datetime.now(UTC) - timedelta(minutes=10)

    view = WorkerView(
        adapter_kind="claude-code",
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

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(token_budget=50000, cost_budget_action="warn")
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.SLOW


def test_classify_worker_health_token_budget_kill_mode(tmp_path: Path) -> None:
    """Token budget exceeded with kill mode returns RUNAWAY."""
    log_file = tmp_path / "sessions" / "issue-1.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("Working on task...", encoding="utf-8")

    events_file = tmp_path / "sessions" / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 100000, "cost_usd": 0.01}',
        encoding="utf-8",
    )

    # Use a recent started_at to avoid triggering the wall-clock/loop tripwires
    recent_start = datetime.now(UTC) - timedelta(minutes=10)

    view = WorkerView(
        adapter_kind="claude-code",
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

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(token_budget=50000, cost_budget_action="kill")
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.RUNAWAY


def test_classify_worker_health_usage_below_budgets(tmp_path: Path) -> None:
    """Usage below both budgets does not affect classification."""
    log_file = tmp_path / "sessions" / "issue-1.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("Working on task...", encoding="utf-8")

    events_file = tmp_path / "sessions" / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 1000, "cost_usd": 0.01}',
        encoding="utf-8",
    )

    # Use a recent started_at to avoid triggering the wall-clock/loop tripwires
    recent_start = datetime.now(UTC) - timedelta(minutes=10)

    view = WorkerView(
        adapter_kind="claude-code",
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

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(cost_budget_usd=100.0, token_budget=1000000)
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_no_events_file_devin(tmp_path: Path) -> None:
    """Devin session (no events file) is not affected by cost/token budgets."""
    log_file = tmp_path / "sessions" / "issue-1.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("Working on task...", encoding="utf-8")

    # No events.jsonl file exists

    # Use a recent started_at to avoid triggering the wall-clock/loop tripwires
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

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        config = OrchestratorConfig(watchdog=WatchdogConfig(cost_budget_usd=0.01, token_budget=10))
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # Should be HEALTHY because the tripwire doesn't fire for devin
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_no_events_file_claude(tmp_path: Path) -> None:
    """Claude Code session without events file (tee disabled) is not affected by budgets."""
    log_file = tmp_path / "sessions" / "issue-1.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("Working on task...", encoding="utf-8")

    # No events.jsonl file exists

    # Use a recent started_at to avoid triggering the wall-clock/loop tripwires
    recent_start = datetime.now(UTC) - timedelta(minutes=10)

    view = WorkerView(
        adapter_kind="claude-code",
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

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(watchdog=WatchdogConfig(cost_budget_usd=0.01, token_budget=10))
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # Should be HEALTHY because the tripwire doesn't fire when events file is missing
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_budgets_disabled_by_default(tmp_path: Path) -> None:
    """Default config (None budgets) reproduces pre-#163 behavior (HEALTHY)."""
    log_file = tmp_path / "sessions" / "issue-1.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("Working on task...", encoding="utf-8")

    events_file = tmp_path / "sessions" / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 1000000, "cost_usd": 1000.0}',
        encoding="utf-8",
    )

    # Use a recent started_at to avoid triggering the wall-clock/loop tripwires
    recent_start = datetime.now(UTC) - timedelta(minutes=10)

    view = WorkerView(
        adapter_kind="claude-code",
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

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig()  # Default config has None budgets
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # Should be HEALTHY because budgets are None (disabled)
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


def test_classify_worker_health_malformed_started_at_claude_no_tool_calls(
    tmp_path: Path,
) -> None:
    """Issue #300: malformed started_at and no tool calls must not raise UnboundLocalError."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\n", encoding="utf-8")

    events_file = tmp_path / "test.events.jsonl"
    events_file.write_text('{"type": "ping"}\n', encoding="utf-8")

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="not-a-timestamp",
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
        assert isinstance(health, WorkerHealth)


def test_classify_worker_health_dead_by_liveness_deferred_by_fresh_probe(
    tmp_path: Path,
) -> None:
    """Issue #307: a dead PID with a fresh real-session activity signal is not DEAD this pass."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    # Freeze the sidecar log so Signal 3 would fire if reached.
    old_time = datetime.now(UTC) - timedelta(minutes=30)
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
    fresh_timestamp = now - timedelta(seconds=8)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=fresh_timestamp,
                staleness_seconds=(now - fresh_timestamp).total_seconds(),
                error=None,
            ),
        )
    )

    with patch("charlie_work.worker.is_session_alive", return_value=False):
        config = OrchestratorConfig()
        health = classify_worker_health(view, config, now, probe)
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_stalled_by_mtime_inconclusive_probe_deferred(
    tmp_path: Path,
) -> None:
    """Issue #307: a probe with all errored sources must not fail open to STALLED."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=30)
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
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=None,
                staleness_seconds=None,
                error="message_nodes query failed (schema drift?): no such column: id",
            ),
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error="no per-PID log found",
            ),
        )
    )

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        config = OrchestratorConfig()
        health = classify_worker_health(view, config, now, probe)
        assert health not in (WorkerHealth.DEAD, WorkerHealth.STALLED)


def test_classify_worker_health_stalled_by_mtime_no_match_yet_probe_deferred(
    tmp_path: Path,
) -> None:
    """Issue #307 scope-extension: the second inconclusive shape.

    Distinct from test_classify_worker_health_stalled_by_mtime_inconclusive_probe_deferred
    (which covers all-errored sources): here every source is error-free but
    returned no timestamp match at all (e.g. a young devin-shell session whose
    sessions.db row hasn't landed yet). This must funnel into the same defer
    branch, not fail open to STALLED.
    """
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=30)
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
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=None,
                staleness_seconds=None,
                error=None,
            ),
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error=None,
            ),
        )
    )

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        config = OrchestratorConfig()
        health = classify_worker_health(view, config, now, probe)
        assert health not in (WorkerHealth.DEAD, WorkerHealth.STALLED)


def test_classify_worker_health_terminal_marker_still_dead_with_fresh_probe(
    tmp_path: Path,
) -> None:
    """Issue #307: a terminal marker still wins even when the real probe is fresh."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nError: A tool was rejected", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=30)
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
    fresh_timestamp = now - timedelta(seconds=8)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=fresh_timestamp,
                staleness_seconds=(now - fresh_timestamp).total_seconds(),
                error=None,
            ),
        )
    )

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        config = OrchestratorConfig()
        health = classify_worker_health(view, config, now, probe)
        assert health == WorkerHealth.DEAD


def test_classify_worker_health_incident_285_routine_exit_not_stalled(
    tmp_path: Path,
) -> None:
    """Issue #307: reproduce the incident — dead PID, PR log line, fresh per-PID log."""
    log_file = tmp_path / "issue-285.log"
    log_file.write_text(
        "PR: https://github.com/Senkichi/charlie-work/pull/306\n", encoding="utf-8"
    )

    now = datetime.now(UTC)
    eight_sec_ago = now - timedelta(seconds=8)
    os.utime(log_file, (time.time(), eight_sec_ago.timestamp()))

    view = WorkerView(
        adapter_kind="devin",
        issue_number=285,
        repo_key="",
        pid=28028,
        started_at=(now - timedelta(minutes=10)).isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=None,
                staleness_seconds=None,
                error="message_nodes query failed (schema drift?): no such column: id",
            ),
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=eight_sec_ago,
                staleness_seconds=8.0,
                error=None,
            ),
        )
    )

    with patch("charlie_work.worker.is_session_alive", return_value=False):
        config = OrchestratorConfig()
        health = classify_worker_health(view, config, now, probe)
        assert health not in (WorkerHealth.DEAD, WorkerHealth.STALLED)


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


def test_classify_worker_health_budget_tripwire_rework_layout(tmp_path: Path) -> None:
    """Issue #344: Signal 6 (cost/token budget tripwire) must use the canonical
    events.jsonl derivation for rework-layout sessions too.

    A rework claude-code log named ``issue-42-rework.claude.log`` has its
    structured-events sibling at ``issue-42-rework.events.jsonl`` (not
    ``issue-42.events.jsonl``, which the old rework=False-only
    ``_events_path(sessions_dir, issue_number)`` derivation would read
    instead). This test plants a stale, under-budget ``issue-42.events.jsonl``
    from a prior (non-rework) attempt alongside the real, over-budget
    ``issue-42-rework.events.jsonl`` sibling. If Signal 6 regresses to the old
    derivation, it silently reads the stale file and never trips; the
    canonical derivation must read the rework sibling and fire.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    issue_number = 42
    log_file = sessions_dir / f"issue-{issue_number}-rework.claude.log"
    log_file.write_text("Working on rework attempt...", encoding="utf-8")

    # Stale events.jsonl from a prior (non-rework) attempt: under budget.
    stale_events_file = sessions_dir / f"issue-{issue_number}.events.jsonl"
    stale_events_file.write_text(
        '{"type": "tool_call", "tokens": 100, "cost_usd": 0.01}',
        encoding="utf-8",
    )

    # Real rework events.jsonl sibling: over the cost budget.
    events_file = sessions_dir / f"issue-{issue_number}-rework.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 1000, "cost_usd": 10.0}',
        encoding="utf-8",
    )

    # Use a recent started_at to avoid triggering the wall-clock/loop tripwires
    recent_start = datetime.now(UTC) - timedelta(minutes=10)

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=issue_number,
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

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(cost_budget_usd=5.0, cost_budget_action="kill")
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.RUNAWAY


def _dead_devin_view(tmp_path: Path, *, inconclusive_probe_deferred_count: int = 0) -> WorkerView:
    """Build a WorkerView for a dead devin worker with a stale sidecar log."""
    log_file = tmp_path / "test.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_file, (time.time(), old_time.timestamp()))

    recent_start = datetime.now(UTC) - timedelta(minutes=10)
    return WorkerView(
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
        inconclusive_probe_deferred_count=inconclusive_probe_deferred_count,
    )


def _all_errored_probe() -> RealActivityProbe:
    return RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=None,
                staleness_seconds=None,
                error="message_nodes query failed (schema drift?): no such column: id",
            ),
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error="no per-PID log found",
            ),
        )
    )


def _no_match_yet_probe() -> RealActivityProbe:
    return RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=None,
                staleness_seconds=None,
                error=None,
            ),
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error=None,
            ),
        )
    )


def test_classify_worker_health_dead_by_liveness_inconclusive_all_errored_deferred_then_escalated(
    tmp_path: Path,
) -> None:
    """Issue #338: dead PID + all-errored probe defers, then reaps after cap."""
    view = _dead_devin_view(tmp_path)
    probe = _all_errored_probe()
    now = datetime.now(UTC)

    with patch("charlie_work.worker.is_session_alive", return_value=False):
        config = OrchestratorConfig(watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=2))
        # Below the cap: defer rather than fail open to DEAD.
        health = classify_worker_health(view, config, now, probe)
        assert health == WorkerHealth.HEALTHY

        # At the cap: escalation, reap.
        capped = replace(view, inconclusive_probe_deferred_count=2)
        health = classify_worker_health(capped, config, now, probe)
        assert health == WorkerHealth.DEAD


def test_classify_worker_health_dead_by_liveness_inconclusive_no_match_yet_deferred_then_escalated(
    tmp_path: Path,
) -> None:
    """Issue #338: dead PID + no-match-yet probe defers, then reaps after cap."""
    view = _dead_devin_view(tmp_path)
    probe = _no_match_yet_probe()
    now = datetime.now(UTC)

    with patch("charlie_work.worker.is_session_alive", return_value=False):
        config = OrchestratorConfig(watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=2))
        # Below the cap: defer rather than fail open to DEAD.
        health = classify_worker_health(view, config, now, probe)
        assert health == WorkerHealth.HEALTHY

        # At the cap: escalation, reap.
        capped = replace(view, inconclusive_probe_deferred_count=2)
        health = classify_worker_health(capped, config, now, probe)
        assert health == WorkerHealth.DEAD


def test_classify_worker_health_dead_by_liveness_inconclusive_cap_zero_reaps_immediately(
    tmp_path: Path,
) -> None:
    """Issue #338: a max_inconclusive_probe_deferrals of 0 disables deferral."""
    view = _dead_devin_view(tmp_path)
    probe = _all_errored_probe()
    now = datetime.now(UTC)

    with patch("charlie_work.worker.is_session_alive", return_value=False):
        config = OrchestratorConfig(watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=0))
        health = classify_worker_health(view, config, now, probe)
        assert health == WorkerHealth.DEAD


# ---------------------------------------------------------------------------
# Issue #484: in-flight api per-session budget kill (Signal 6.5)
# ---------------------------------------------------------------------------

from _api_budget_fixtures import api_provider  # noqa: E402


def _api_worker_view(tmp_path: Path, *, provider: str = "example") -> WorkerView:
    """Build a live api WorkerView whose log_path points at issue-1.claude.log."""
    log_file = tmp_path / "sessions" / "issue-1.claude.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("Working on task...", encoding="utf-8")
    recent_start = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    return WorkerView(
        adapter_kind="api",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=recent_start,
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
        provider=provider,
    )


def _write_api_events(tmp_path: Path, cost_usd: float = 6.15) -> Path:
    """Write an events.jsonl whose usage yields ``cost_usd`` at default pricing.

    With the default ``api_provider`` pricing (input=3.0, output=15.0,
    cached=0.30 per MTok), 1M input + 0.2M output + 0.5M cached = 6.15 USD.
    """
    events_file = tmp_path / "sessions" / "issue-1.events.jsonl"
    events_file.parent.mkdir(parents=True, exist_ok=True)
    events_file.write_text(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "usage": {
                    "input_tokens": 1_000_000,
                    "output_tokens": 200_000,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 500_000,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return events_file


def test_classify_worker_health_api_budget_kill_over_cap(tmp_path: Path) -> None:
    """Issue #484: an api worker whose accumulated cost exceeds
    ``max_usd_per_session`` is classified RUNAWAY (kill)."""
    _write_api_events(tmp_path)  # 6.15 USD at default pricing
    view = _api_worker_view(tmp_path)

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            api_worker=ApiWorkerConfig(
                enabled=True,
                provider="example",
                providers={"example": api_provider()},
                budget=ApiBudgetConfig(max_usd_per_session=5.0),
            )
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.RUNAWAY


def test_classify_worker_health_api_budget_dormant_cap_unset(tmp_path: Path) -> None:
    """Issue #484: when ``max_usd_per_session`` is 0/unset the check is entirely
    dormant — no kill at any cost level (HEALTHY)."""
    _write_api_events(tmp_path)  # 6.15 USD — would exceed a 5.0 cap, but cap is 0
    view = _api_worker_view(tmp_path)

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            api_worker=ApiWorkerConfig(
                enabled=True,
                provider="example",
                providers={"example": api_provider()},
                budget=ApiBudgetConfig(max_usd_per_session=0.0),
            )
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_api_budget_below_cap_healthy(tmp_path: Path) -> None:
    """Issue #484: an api worker under the cap is HEALTHY."""
    _write_api_events(tmp_path)  # 6.15 USD
    view = _api_worker_view(tmp_path)

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            api_worker=ApiWorkerConfig(
                enabled=True,
                provider="example",
                providers={"example": api_provider()},
                budget=ApiBudgetConfig(max_usd_per_session=10.0),
            )
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_api_budget_non_api_never_evaluated(
    tmp_path: Path,
) -> None:
    """Issue #484: non-api workers are never budget-evaluated via the api
    per-session cap, even with high-cost events and a low cap configured."""
    _write_api_events(tmp_path)  # 6.15 USD
    log_file = tmp_path / "sessions" / "issue-1.claude.log"
    recent_start = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=recent_start,
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        # api budget cap is low, but the worker is claude-code, not api.
        config = OrchestratorConfig(
            api_worker=ApiWorkerConfig(
                enabled=True,
                provider="example",
                providers={"example": api_provider()},
                budget=ApiBudgetConfig(max_usd_per_session=1.0),
            )
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # Claude Code's own Signal 6 uses self-reported cost_usd; the events
        # fixture has no cost_usd field, so Signal 6 is dormant too → HEALTHY.
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_api_budget_no_provider_healthy(tmp_path: Path) -> None:
    """Issue #484: an api worker with an empty/unknown provider is not
    budget-evaluated (no pricing → cannot compute cost → HEALTHY)."""
    _write_api_events(tmp_path)
    view = _api_worker_view(tmp_path, provider="")

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            api_worker=ApiWorkerConfig(
                enabled=True,
                provider="example",
                providers={"example": api_provider()},
                budget=ApiBudgetConfig(max_usd_per_session=1.0),
            )
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_api_budget_no_events_healthy(tmp_path: Path) -> None:
    """Issue #484: an api worker with no events.jsonl yet (young session) is
    not budget-killed — absence is not over-budget."""
    view = _api_worker_view(tmp_path)
    # No events.jsonl written.

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            api_worker=ApiWorkerConfig(
                enabled=True,
                provider="example",
                providers={"example": api_provider()},
                budget=ApiBudgetConfig(max_usd_per_session=0.01),
            )
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.HEALTHY


# --- issue_worker_liveness: the two untested branches from PR #684 review ---
#
# The predicate's state-path inconclusive/wall-clock branch is exercised
# end-to-end via test_fix_unescalate.py, but its two other decisive branches
# had no direct coverage:
#   * sidecar path: an alive sidecar worker that classify_worker_health
#     classifies STALLED must yield live=False (the "closing the symmetric
#     drift" behavior described in the PR body).
#   * state path: a real (non-None) but past-stall_minutes activity timestamp
#     must yield live=False with the "alive but wedged: no real activity for
#     >Nm" reason, distinct from the inconclusive/wall-clock branch.


def test_issue_worker_liveness_sidecar_stalled_yields_not_live(tmp_path: Path) -> None:
    """PR #684 review: an alive sidecar worker classified STALLED by
    ``classify_worker_health`` must yield ``live=False``. This is the
    "closing the symmetric drift" branch -- before the predicate unified the
    two authorities, the sidecar path deferred to the watchdog (which had
    reaped the sidecar) while the state path asked only "is the PID alive?",
    so the criteria drifted. Now both route through one predicate; an
    alive-but-stalled sidecar session is wedged, not live.

    The sidecar references the live test process (``is_pid_alive`` True with
    no ``process_start_time`` fingerprint). Its log mtime is parked past
    ``stall_minutes`` so Signal 3 fires, and the real-activity probe is
    *conclusively stale* (the worktree-mtime source reports the session's
    own ``started_at`` with threshold 0 -- the "conclusively stale rather
    than inconclusive" path of ``_worktree_mtime_source``), so the STALLED
    verdict is not deferred. sessions.db / per-PID Devin log sources are
    kept inconclusive (missing db_path) so they cannot veto with a fresh
    timestamp.
    """
    import os
    import time

    now = datetime.now(UTC)
    stall_minutes = OrchestratorConfig().watchdog.stall_minutes
    started_at = (now - timedelta(minutes=stall_minutes + 10)).isoformat()

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    worktree_dir = tmp_path / "worktree"
    worktree_dir.mkdir()  # empty -> no post-start file mtimes -> conclusive-stale
    log_file = sessions_dir / "issue-123.log"
    log_file.write_text("Working on task...\nLast line", encoding="utf-8")
    old_mtime = (now - timedelta(minutes=stall_minutes + 10)).timestamp()
    os.utime(log_file, (time.time(), old_mtime))

    sidecar = sessions_dir / "issue-123.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 123,
                "branch": "",
                "worktree_path": str(worktree_dir),
                "prompt_path": "",
                "command": [],
                "pid": os.getpid(),
                "started_at": started_at,
                "log_path": str(log_file),
                "process_start_time": None,
            }
        ),
        encoding="utf-8",
    )

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )

    verdict = issue_worker_liveness(123, {}, sessions_dir, config, now)

    assert verdict.live is False
    assert verdict.source == "sidecar"
    assert verdict.pid == os.getpid()
    # The "closing the symmetric drift" reason: alive but stalled.
    assert "alive but stalled" in verdict.reason
    assert f">{stall_minutes}m" in verdict.reason
    # Conclusive-stale probe: a real last-activity timestamp is surfaced.
    assert verdict.last_activity_at is not None
    assert verdict.last_activity_source == "worktree_files_mtime"


def test_issue_worker_liveness_state_conclusive_stale_yields_not_live(tmp_path: Path) -> None:
    """PR #684 review: the state path's conclusive-stale branch -- a real
    (non-None) but past-``stall_minutes`` activity timestamp -- must yield
    ``live=False`` with the "alive but wedged: no real activity for >Nm"
    reason. This is distinct from the already-tested inconclusive/wall-clock
    branch (every source errored -> fall back to the wall-clock deadline):
    here a real activity source produced a timestamp, it is just old, so the
    conclusive-stale branch fires before the wall-clock backstop is reached.

    The state path hardcodes ``worktree_path=""`` and ``log_path=None``, so
    the only source that can produce a real timestamp is the per-PID Devin
    log (Source 2). We materialize a ``devin_*_{pid}.log`` file under the
    ``logs/`` sibling of the configured ``db_path`` with an mtime parked
    past ``stall_minutes``; sessions.db is left missing so Source 1 errors
    (timestamp=None) and cannot veto with a fresh signal. The test process
    is the alive PID; no sidecar exists for the issue.
    """
    import os
    import time

    now = datetime.now(UTC)
    stall_minutes = OrchestratorConfig().watchdog.stall_minutes
    started_at = (now - timedelta(minutes=stall_minutes + 10)).isoformat()

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()  # no sidecar for issue 123 -> source 1 skipped
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    per_pid_log = logs_dir / f"devin_cli_{os.getpid()}.log"
    per_pid_log.write_text("devin session log\n", encoding="utf-8")
    old_mtime = (now - timedelta(minutes=stall_minutes + 10)).timestamp()
    os.utime(per_pid_log, (time.time(), old_mtime))

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )

    issue_state = {
        "number": 123,
        "status": "dispatched",
        "worker_pid": os.getpid(),
        "dispatched_at": started_at,
    }

    verdict = issue_worker_liveness(123, issue_state, sessions_dir, config, now)

    assert verdict.live is False
    assert verdict.source == "state"
    assert verdict.pid == os.getpid()
    # The conclusive-stale reason, distinct from the wall-clock backstop.
    assert "alive but wedged" in verdict.reason
    assert f"no real activity for >{stall_minutes}m" in verdict.reason
    assert "wall-clock" not in verdict.reason
    # A real (non-None) activity timestamp is surfaced -- the hallmark of the
    # conclusive-stale branch, not the inconclusive branch.
    assert verdict.last_activity_at is not None
    assert verdict.last_activity_source == "devin_per_pid_log"
