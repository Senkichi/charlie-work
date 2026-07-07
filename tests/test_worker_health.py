from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from charlie_work.config import OrchestratorConfig, WatchdogConfig
from charlie_work.worker import (
    UsageSnapshot,
    WorkerHealth,
    WorkerView,
    classify_worker_health,
    parse_cumulative_usage,
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

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
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

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
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

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
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

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
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

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
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

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
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

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
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

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
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

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
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

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
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

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
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

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
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

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_session_alive", return_value=True):
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(cost_budget_usd=0.01, token_budget=10)
        )
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

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(cost_budget_usd=0.01, token_budget=10)
        )
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

    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
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
