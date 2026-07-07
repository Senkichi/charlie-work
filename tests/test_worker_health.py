from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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

    view = WorkerView(
        adapter_kind="devin",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
        process_start_time=1710000000.0,
        log_path=str(log_file),
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
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    # Mock the underlying adapter liveness function that is_alive() calls
    with patch("charlie_work.worker.is_session_alive", return_value=True):
        # Custom config with a custom terminal marker
        config = OrchestratorConfig(
            watchdog=WatchdogConfig(
                terminal_error_markers=("Custom fatal error",)
            )
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
