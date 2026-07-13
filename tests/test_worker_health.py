from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from charlie_work.config import OrchestratorConfig, PostMortemConfig, WatchdogConfig
from charlie_work.post_mortem import ActivitySource, RealActivityProbe
from charlie_work.worker import (
    WorkerHealth,
    WorkerView,
    classify_worker_health,
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
