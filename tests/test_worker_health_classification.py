"""Core ``WorkerHealth`` classification tests for ``charlie_work.worker``.

Split out of ``tests/test_worker_health.py`` (issue #1568, Track-1):
the ``WorkerHealth`` enum contract, the baseline healthy/dead/stalled
classification paths (terminal marker, liveness, legacy start time,
custom marker, no-IO), log/start-time regression cases, and the two
``issue_worker_liveness`` branch-covering tests.
"""

from __future__ import annotations

import json
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
    issue_worker_liveness,
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
