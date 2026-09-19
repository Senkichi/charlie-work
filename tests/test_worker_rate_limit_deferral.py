"""Rate-limit stall deferral tests (issue #247) for the stalled-session lane.

Split out of ``tests/test_worker_stalled_sessions.py`` (issue #1574, Track-1
shoulder, second-stage seam split): the ``_detect_and_handle_stalled_sessions``
rate-limit deferral lane -- defer on a rate-limit log tail, exit the deferred
state when the log resumes, reap past the defer deadline, and keep ordinary
stall-kill behavior for a quiet tail. All test bodies moved verbatim; no
renames, no fixture hoists into ``conftest.py``. The shared
``_make_stalled_devin_session``/``_stale_devin_probe``/``_wg`` helpers live in
``tests/_worker_fixtures.py``.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from _worker_fixtures import _make_stalled_devin_session, _stale_devin_probe, _wg
from charlie_work.config import OrchestratorConfig, WatchdogConfig
from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path


# ---------------------------------------------------------------------------
# Rate-limit stall deferral (issue #247)
# ---------------------------------------------------------------------------


def test_stalled_worker_with_rate_limit_signature_is_deferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled-looking worker whose log tail contains a rate-limit signature is deferred, not killed."""
    from charlie_work import workflow

    # Issue #1317: kill_process_tree / sweep_orphan_processes are called
    # bare-name from inside _sweep_orphan_processes_for_dead_sessions and
    # _classify_dead_sessions_and_update_throttle_state, both of which moved
    # (verbatim) to dead_worker_reap.py -- patch them there, not on the
    # (still-valid) workflow.py facade re-export of
    # _detect_and_handle_stalled_sessions itself.
    from charlie_work import dead_worker_reap

    issue_number = 247
    log_text = (
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n"
    )
    sessions_dir, state_file, _ = _make_stalled_devin_session(tmp_path, issue_number, log_text)

    killed = []
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, start_time=None: killed.append(pid) or [pid],
    )
    monkeypatch.setattr(dead_worker_reap, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_devin_probe)

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        )
    )

    # Issue #828: freeze the clock and inject it so the assertion below is an
    # exact equality, not a wall-clock-tolerance window that a CI stall (the
    # same failure class as PRs #700/#690) can blow through.
    frozen_now = datetime.now(UTC)
    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, write_gate=_wg(state_file), now=frozen_now
    )

    assert result == []
    assert killed == []

    sidecar = json.loads(
        (devin_sidecar_path(sessions_dir, issue_number)).read_text(encoding="utf-8")
    )
    assert sidecar["rate_limit_defer_until"] is not None
    defer_until = datetime.fromisoformat(sidecar["rate_limit_defer_until"].replace("Z", "+00:00"))
    margin_seconds = config.runtime.throttle_resume_margin_s
    expected = (frozen_now + timedelta(minutes=10 + 2, seconds=margin_seconds)).replace(
        microsecond=0
    )
    assert defer_until == expected

    state = json.loads(state_file.read_text(encoding="utf-8"))
    events = [e for e in state.get("events", []) if e.get("kind") == "session_rate_limit_deferred"]
    assert len(events) == 1
    assert events[0]["payload"]["issue_number"] == issue_number
    assert events[0]["payload"]["defer_until"] == sidecar["rate_limit_defer_until"]


def test_deferred_worker_log_resumes_exits_deferred_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deferred worker whose log resumes growing exits the deferred state and is not killed."""
    from charlie_work import workflow

    # Issue #1317: kill_process_tree / sweep_orphan_processes are called
    # bare-name from inside _sweep_orphan_processes_for_dead_sessions and
    # _classify_dead_sessions_and_update_throttle_state, both of which moved
    # (verbatim) to dead_worker_reap.py -- patch them there, not on the
    # (still-valid) workflow.py facade re-export of
    # _detect_and_handle_stalled_sessions itself.
    from charlie_work import dead_worker_reap

    issue_number = 248
    log_text = (
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n"
    )
    future_defer = (datetime.now(UTC) + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    sessions_dir, state_file, log_path = _make_stalled_devin_session(
        tmp_path, issue_number, log_text, rate_limit_defer_until=future_defer
    )

    # Now the log resumes: update mtime and size to "now".
    log_path.write_text(log_text + "Resumed work after provider window reset\n", encoding="utf-8")
    recent_time = datetime.now(UTC) - timedelta(minutes=1)
    os.utime(log_path, (recent_time.timestamp(), recent_time.timestamp()))

    killed = []
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, start_time=None: killed.append(pid) or [pid],
    )
    monkeypatch.setattr(dead_worker_reap, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        )
    )

    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, write_gate=_wg(state_file)
    )

    assert result == []
    assert killed == []

    sidecar = json.loads(
        (devin_sidecar_path(sessions_dir, issue_number)).read_text(encoding="utf-8")
    )
    assert sidecar["rate_limit_defer_until"] is None


def test_deferred_worker_past_deadline_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deferred worker still silent past the deadline is killed and classified via the rate-limit path."""
    from charlie_work import workflow

    # Issue #1317: kill_process_tree / sweep_orphan_processes are called
    # bare-name from inside _sweep_orphan_processes_for_dead_sessions and
    # _classify_dead_sessions_and_update_throttle_state, both of which moved
    # (verbatim) to dead_worker_reap.py -- patch them there, not on the
    # (still-valid) workflow.py facade re-export of
    # _detect_and_handle_stalled_sessions itself.
    from charlie_work import dead_worker_reap

    issue_number = 249
    log_text = (
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n"
    )
    # Issue #828: freeze the clock and inject it into the call below so the
    # deadline comparison is deterministic rather than depending on two
    # independently-sampled wall-clock reads (setup here vs. the sweep's own
    # internal sample) staying within 5 minutes of each other under CI load.
    frozen_now = datetime.now(UTC)
    past_defer = (frozen_now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    sessions_dir, state_file, _ = _make_stalled_devin_session(
        tmp_path, issue_number, log_text, rate_limit_defer_until=past_defer
    )

    killed = []
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, start_time=None: killed.append(pid) or [pid],
    )
    monkeypatch.setattr(dead_worker_reap, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_devin_probe)

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        )
    )

    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, write_gate=_wg(state_file), now=frozen_now
    )

    assert result == [{"issue": issue_number, "pid": 99999}]
    assert killed == [99999]

    sidecar = json.loads(
        (devin_sidecar_path(sessions_dir, issue_number)).read_text(encoding="utf-8")
    )
    assert sidecar["failure_kind"] == "rate_limited"
    # The sidecar still carries the expired defer deadline; the global throttle
    # state is set to the freshly computed cooldown from the log tail.
    assert sidecar["rate_limit_defer_until"] is not None

    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state.get("throttled_until") is not None
    throttled_until = datetime.fromisoformat(state["throttled_until"].replace("Z", "+00:00"))
    margin = timedelta(seconds=config.runtime.throttle_resume_margin_s)
    expected_min = datetime.now(UTC) + timedelta(minutes=10) + margin - timedelta(minutes=1)
    expected_max = datetime.now(UTC) + timedelta(minutes=10) + margin + timedelta(minutes=1)
    assert expected_min <= throttled_until <= expected_max


def test_stalled_worker_without_rate_limit_signature_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled worker with a genuinely quiet tail (no rate-limit signature) keeps existing stall-kill behavior."""
    from charlie_work import workflow

    # Issue #1317: kill_process_tree / sweep_orphan_processes are called
    # bare-name from inside _sweep_orphan_processes_for_dead_sessions and
    # _classify_dead_sessions_and_update_throttle_state, both of which moved
    # (verbatim) to dead_worker_reap.py -- patch them there, not on the
    # (still-valid) workflow.py facade re-export of
    # _detect_and_handle_stalled_sessions itself.
    from charlie_work import dead_worker_reap

    issue_number = 250
    log_text = "Working on task...\nLast line\n"
    sessions_dir, state_file, _ = _make_stalled_devin_session(tmp_path, issue_number, log_text)

    killed = []
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, start_time=None: killed.append(pid) or [pid],
    )
    monkeypatch.setattr(dead_worker_reap, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_devin_probe)

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        )
    )

    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, write_gate=_wg(state_file)
    )

    assert result == [{"issue": issue_number, "pid": 99999}]
    assert killed == [99999]

    sidecar = json.loads(
        (devin_sidecar_path(sessions_dir, issue_number)).read_text(encoding="utf-8")
    )
    assert sidecar["failure_kind"] == "stalled"
    assert sidecar.get("rate_limit_defer_until") is None
