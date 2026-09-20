"""Cost/token budget tripwire tests for ``classify_worker_health``.

Split out of ``tests/test_worker_health.py`` (issue #1568, Track-1):
warn/kill modes for cost and token budgets, under-budget and
disabled-by-default paths, missing events.jsonl handling per
adapter, and the rework-layout budget tripwire (issue #163).
"""

from __future__ import annotations

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
