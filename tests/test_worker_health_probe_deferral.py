"""Real-activity-probe deferral tests for ``classify_worker_health``.

Split out of ``tests/test_worker_health.py`` (issue #1568, Track-1):
inconclusive/fresh/no-match-yet probe outcomes deferring kill
decisions, terminal-marker precedence over a fresh probe, the
deferred-then-escalated paths, and the cap-zero immediate reap.
The ``_dead_devin_view``/``_*_probe`` helpers moved verbatim with
their only consumers.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from charlie_work.config import (
    OrchestratorConfig,
    WatchdogConfig,
)
from charlie_work.post_mortem import ActivitySource, RealActivityProbe
from charlie_work.worker import (
    WorkerHealth,
    WorkerView,
    classify_worker_health,
)


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
