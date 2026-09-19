"""Stalled/deferred worker detection tests for ``charlie_work.worker`` and
``charlie_work.workflow``/``charlie_work.dead_worker_reap``.

Split out of ``tests/test_worker.py`` (issue #1574, Track-1 shoulder): the
``_log_is_stalled_at_shim`` staleness predicate tests, the issue #247
rate-limit stall deferral lane, and the ``_detect_and_handle_stalled_sessions``
dry-run/escalation/provider-suspension tests. All test bodies moved verbatim;
no renames, no fixture hoists into ``conftest.py``. The shared ``_wg``
``WriteGate`` builder lives in ``tests/_worker_fixtures.py``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from _worker_fixtures import _wg
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.claude_code import _sidecar_path as claude_sidecar_path
from charlie_work.config import (
    OrchestratorConfig,
    PostMortemConfig,
    WatchdogConfig,
)
from charlie_work.devin_shell import (
    SessionRecord,
    _sidecar_path as devin_sidecar_path,
    _write_json,
)
from charlie_work.post_mortem import ActivitySource, RealActivityProbe, real_activity_for_worker
from charlie_work.worker import WorkerHealth, WorkerView, _log_is_stalled_at_shim


def test_log_is_stalled_at_shim_with_marker(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns True when log has shim marker and is stale."""
    log_path = tmp_path / "issue-1.log"
    # Write a log with the shim marker (typical frozen log size ~424-425 bytes)
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    # Set mtime to 10 minutes ago (past the default 5-minute grace period)
    old_time = datetime.now(UTC) - timedelta(minutes=10)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    assert _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_without_marker(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns False when log lacks shim marker."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("Some other log content\n", encoding="utf-8")

    # Set mtime to 10 minutes ago
    old_time = datetime.now(UTC) - timedelta(minutes=10)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    assert not _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_within_grace_period(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns False when log is within grace period."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    # Set mtime to 2 minutes ago (within the 5-minute grace period)
    old_time = datetime.now(UTC) - timedelta(minutes=2)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    assert not _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_large_log(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns False when log is large (>1KB)."""
    log_path = tmp_path / "issue-1.log"
    # Write a large log with the shim marker
    large_content = "[shim] .devin infra materialized\n" + "x" * 2000
    log_path.write_text(large_content, encoding="utf-8")

    # Set mtime to 10 minutes ago
    old_time = datetime.now(UTC) - timedelta(minutes=10)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    assert not _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_nonexistent_log(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns False when log file doesn't exist."""
    log_path = tmp_path / "issue-1.log"
    now = datetime.now(UTC)
    assert not _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_with_fresh_real_activity(tmp_path: Path) -> None:
    """Issue #280: frozen sidecar log is ignored when real-session activity is fresh."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

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

    assert not _log_is_stalled_at_shim(
        log_path, grace_minutes=5, now=now, real_activity_probe=probe
    )


def test_log_is_stalled_at_shim_with_stale_real_activity(tmp_path: Path) -> None:
    """Issue #280: launch stall is still detected when real activity is also stale."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    stale_timestamp = now - timedelta(minutes=10)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=stale_timestamp,
                staleness_seconds=(now - stale_timestamp).total_seconds(),
                error=None,
            ),
        )
    )

    assert _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now, real_activity_probe=probe)


def test_log_is_stalled_at_shim_with_all_errored_probe_deferred(tmp_path: Path) -> None:
    """Issue #307 scope-extension: _log_is_stalled_at_shim must not fail open on an
    all-errored probe.

    This is the site the reviewer reproduced directly: reconcile.py:264 reaches
    this function only for a CONFIRMED-ALIVE worker, and a True return here
    drives an immediate kill_process_tree (reconcile.py:288). An all-errored
    probe is insufficient evidence of a stall.
    """
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

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

    assert not _log_is_stalled_at_shim(
        log_path, grace_minutes=5, now=now, real_activity_probe=probe
    )


def test_log_is_stalled_at_shim_with_no_match_yet_probe_deferred(tmp_path: Path) -> None:
    """Issue #307 scope-extension: the second inconclusive shape at the shim site.

    Distinct from test_log_is_stalled_at_shim_with_all_errored_probe_deferred:
    here every source is error-free but returned no timestamp match at all
    (e.g. a young session within launch_stall_grace_minutes whose sessions.db
    row hasn't landed yet). This must also defer rather than report a stall.
    """
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

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

    assert not _log_is_stalled_at_shim(
        log_path, grace_minutes=5, now=now, real_activity_probe=probe
    )


def test_log_is_stalled_at_shim_worktree_files_mtime_fresh_beyond_grace(
    tmp_path: Path,
) -> None:
    """Issue #353: worktree mtime freshness uses its own generous threshold."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    now = datetime.now(UTC)
    old_time = now - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    source_file = worktree_path / "foo.py"
    source_file.write_text("# change", encoding="utf-8")
    worktree_mtime = now - timedelta(minutes=8)
    os.utime(source_file, (worktree_mtime.timestamp(), worktree_mtime.timestamp()))

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
    )
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        (now - timedelta(minutes=10)).isoformat(),
        None,
        now,
        watchdog_config=watchdog,
    )

    # 8 minutes is past the 5-minute grace, but within the 45-minute worktree threshold.
    assert not _log_is_stalled_at_shim(
        log_path, grace_minutes=5, now=now, real_activity_probe=probe
    )


def test_log_is_stalled_at_shim_worktree_files_mtime_checkout_noise_stalls(
    tmp_path: Path,
) -> None:
    """Issue #353: checkout-time mtimes do not mask a launch stall."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    now = datetime.now(UTC)
    old_time = now - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    source_file = worktree_path / "foo.py"
    source_file.write_text("# change", encoding="utf-8")
    started_at = now - timedelta(minutes=30)
    os.utime(source_file, (started_at.timestamp(), started_at.timestamp()))

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
    )
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        started_at.isoformat(),
        None,
        now,
        watchdog_config=watchdog,
    )

    assert _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now, real_activity_probe=probe)


# ---------------------------------------------------------------------------
# Rate-limit stall deferral (issue #247)
# ---------------------------------------------------------------------------


def _make_stalled_devin_session(
    tmp_path: Path,
    issue_number: int,
    log_text: str,
    *,
    rate_limit_defer_until: str | None = None,
) -> tuple[Path, Path, Path]:
    """Create a sessions directory, sidecar, and stale log for a live worker."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"issue-{issue_number}.log"
    log_path.write_text(log_text, encoding="utf-8")
    # Set mtime to 30 minutes ago so the log looks stalled at the default
    # 20-minute stall threshold.
    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    record = SessionRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("devin", "--prompt-file", str(tmp_path / "prompt.md")),
        pid=99999,
        started_at=(datetime.now(UTC) - timedelta(minutes=31)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        last_activity_at=old_time.isoformat().replace("+00:00", "Z"),
        log_bytes=log_path.stat().st_size,
        rate_limit_defer_until=rate_limit_defer_until,
    )
    _write_json(sidecar_path, record.to_dict())
    state_file = tmp_path / "state.json"
    return sessions_dir, state_file, log_path


def _stale_devin_probe(*_args: object, **_kwargs: object) -> RealActivityProbe:
    """Return a probe that is stale (not fresh) and not all-errored.

    Issue #307: a worker with a stale sidecar log and a stale real-session
    activity signal must still be classified as STALLED. Tests that exercise
    the rate-limit defer path must not be tripped up by an all-errored probe,
    which now defers to avoid the fail-open bug in Signal 3.
    """
    now = datetime.now(UTC)
    timestamp = now - timedelta(minutes=30)
    return RealActivityProbe(
        sources=(
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=timestamp,
                staleness_seconds=(now - timestamp).total_seconds(),
                error=None,
            ),
        )
    )


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


def test_detect_stalled_sessions_passes_real_activity_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #301: workflow.py stall paths must construct and pass a RealActivityProbe.

    A future edit that drops the probe argument will make this test fail because
    the spy will receive None instead of a constructed probe.
    """
    from charlie_work import workflow

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    issue_number = 301

    log_path = sessions_dir / f"issue-{issue_number}.claude.log"
    log_path.write_text("Working on task...\nLast line", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_path, (time.time(), old_time.timestamp()))

    events_path = sessions_dir / f"issue-{issue_number}.events.jsonl"
    fresh_time = datetime.now(UTC) - timedelta(minutes=1)
    events_path.write_text(
        f'{{"type": "tool_call", "timestamp": "{fresh_time.isoformat()}"}}\n',
        encoding="utf-8",
    )
    os.utime(events_path, (time.time(), fresh_time.timestamp()))

    sidecar_path = claude_sidecar_path(sessions_dir, issue_number)
    record = ClaudeWorkerRecord(
        issue_number=issue_number,
        branch="agent/issue-301",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("claude", "prompt.md"),
        pid=99999,
        started_at=(datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    captured: list[RealActivityProbe | None] = []

    def spy_classify(
        view: WorkerView,
        config: OrchestratorConfig,
        now: datetime,
        real_activity_probe: RealActivityProbe | None = None,
    ) -> WorkerHealth:
        captured.append(real_activity_probe)
        return WorkerHealth.HEALTHY

    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: True)
    monkeypatch.setattr("charlie_work.worker.classify_worker_health", spy_classify)

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    result = workflow._detect_stalled_sessions(sessions_dir, config)

    assert result == []
    assert len(captured) == 1
    assert captured[0] is not None
    assert isinstance(captured[0], RealActivityProbe)
    assert captured[0].latest_source == "claude_events_jsonl"


def test_detect_and_handle_stalled_sessions_not_killed_when_real_activity_probe_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #301 kill-path wiring: a claude-code worker whose sidecar log is frozen
    but whose events.jsonl sibling carries fresh activity must NOT be killed by
    _detect_and_handle_stalled_sessions.

    This is the kill-path counterpart to
    test_detect_stalled_sessions_passes_real_activity_probe above, which only
    exercises the read-only _detect_stalled_sessions status/digest path and
    never drives an actual kill decision. A future edit that drops the
    ``probe`` argument from the classify_worker_health call inside
    _detect_and_handle_stalled_sessions (workflow.py), or that neuters the
    claude_events_jsonl Source-3 construction in
    post_mortem.real_activity_for_worker, must make this test fail (the
    worker gets killed) rather than silently reverting to mtime-only kills.
    """
    from charlie_work import workflow

    # Issue #1317: kill_process_tree / sweep_orphan_processes are called
    # bare-name from inside _sweep_orphan_processes_for_dead_sessions and
    # _classify_dead_sessions_and_update_throttle_state, both of which moved
    # (verbatim) to dead_worker_reap.py -- patch them there, not on the
    # (still-valid) workflow.py facade re-export of
    # _detect_and_handle_stalled_sessions itself.
    from charlie_work import dead_worker_reap

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    issue_number = 302

    log_path = sessions_dir / f"issue-{issue_number}.claude.log"
    log_path.write_text("Working on task...\nLast line", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_path, (time.time(), old_time.timestamp()))

    events_path = sessions_dir / f"issue-{issue_number}.events.jsonl"
    fresh_time = datetime.now(UTC) - timedelta(minutes=1)
    events_path.write_text(
        f'{{"type": "tool_call", "timestamp": "{fresh_time.isoformat()}"}}\n',
        encoding="utf-8",
    )
    os.utime(events_path, (time.time(), fresh_time.timestamp()))

    sidecar_path = claude_sidecar_path(sessions_dir, issue_number)
    record = ClaudeWorkerRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("claude", "prompt.md"),
        pid=88888,
        started_at=(datetime.now(UTC) - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    killed: list[int] = []
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, start_time=None: killed.append(pid) or [pid],
    )
    monkeypatch.setattr(dead_worker_reap, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: True)

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    state_file = tmp_path / "state.json"

    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, write_gate=_wg(state_file)
    )

    assert result == []
    assert killed == []

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar.get("failure_kind") is None


def test_detect_and_handle_stalled_sessions_tolerates_none_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #329: workflow.py stall-event logging must tolerate a None RealActivityProbe."""
    from charlie_work import workflow

    # Issue #1317: kill_process_tree / sweep_orphan_processes are called
    # bare-name from inside _sweep_orphan_processes_for_dead_sessions and
    # _classify_dead_sessions_and_update_throttle_state, both of which moved
    # (verbatim) to dead_worker_reap.py -- patch them there, not on the
    # (still-valid) workflow.py facade re-export of
    # _detect_and_handle_stalled_sessions itself.
    from charlie_work import dead_worker_reap

    issue_number = 329
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / f"issue-{issue_number}.log"
    log_path.write_text("Working on task...\nLast line\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=30)
    os.utime(log_path, (time.time(), old_time.timestamp()))

    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    record = SessionRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("devin", "--prompt-file", str(tmp_path / "prompt.md")),
        pid=99999,
        started_at=(datetime.now(UTC) - timedelta(minutes=31)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        last_activity_at=old_time.isoformat().replace("+00:00", "Z"),
        log_bytes=log_path.stat().st_size,
        rate_limit_defer_until=None,
    )
    _write_json(sidecar_path, record.to_dict())

    killed: list[int] = []
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, start_time=None: killed.append(pid) or [pid],
    )
    monkeypatch.setattr(dead_worker_reap, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", lambda *args: None)

    state_file = tmp_path / "state.json"
    config = OrchestratorConfig()

    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, write_gate=_wg(state_file)
    )

    assert result == [{"issue": issue_number, "pid": 99999}]
    assert killed == [99999]

    state = json.loads(state_file.read_text(encoding="utf-8"))
    stalled_events = [e for e in state.get("events", []) if e.get("kind") == "session_stalled"]
    assert len(stalled_events) == 1
    payload = stalled_events[0]["payload"]
    assert payload["latest_real_activity_source"] == "probe unavailable"
    assert payload["latest_real_activity_at"] is None
    assert payload["activity_sources"] == []


def _inconclusive_probe_for_signal_1(
    view: WorkerView,
    config: OrchestratorConfig,
    now: datetime,
) -> RealActivityProbe:
    """Return an all-errored real-activity probe for issue #338 kill-path tests."""
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


def test_detect_and_handle_stalled_sessions_inconclusive_probe_deferred_then_escalated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #338 kill-path: a dead worker with an inconclusive probe is deferred,
    the sidecar counter increments, and the worker is reaped once the cap is hit.
    """
    from charlie_work import workflow

    # Issue #1317: kill_process_tree / sweep_orphan_processes are called
    # bare-name from inside _sweep_orphan_processes_for_dead_sessions and
    # _classify_dead_sessions_and_update_throttle_state, both of which moved
    # (verbatim) to dead_worker_reap.py -- patch them there, not on the
    # (still-valid) workflow.py facade re-export of
    # _detect_and_handle_stalled_sessions itself.
    from charlie_work import dead_worker_reap

    issue_number = 338
    log_text = "Working on task...\n"
    sessions_dir, state_file, _ = _make_stalled_devin_session(tmp_path, issue_number, log_text)

    killed: list[int] = []
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, start_time=None: killed.append(pid) or [pid],
    )
    monkeypatch.setattr(dead_worker_reap, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: False)
    monkeypatch.setattr(
        "charlie_work.worker.real_activity_probe_for", _inconclusive_probe_for_signal_1
    )

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=1),
    )

    # First pass: dead PID + inconclusive probe → defer, counter advances.
    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, write_gate=_wg(state_file)
    )
    assert result == []
    assert killed == []

    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar.get("inconclusive_probe_deferred_count") == 1
    assert sidecar.get("failure_kind") is None

    # Second pass: counter is now at the cap → reap.
    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, write_gate=_wg(state_file)
    )
    assert result == [{"issue": issue_number, "pid": 99999}]
    assert killed == [99999]

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["failure_kind"] == "stalled"
    assert sidecar.get("inconclusive_probe_deferred_count") == 0


def test_detect_and_handle_stalled_sessions_dry_run_suppresses_kills_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1325: under ``dry_run=True`` the stall reaper must not kill
    processes, must not write to ``state.json``/``events.db``, and must not
    mutate the worker's sidecar file.

    Before the fix, ``_detect_and_handle_stalled_sessions`` had no
    ``write_gate`` parameter: ``kill_process_tree``, ``kill_orphan_pid``,
    ``append_event``, ``save_state``, the budget-exceeded sidecar write,
    ``update_worker_log_stat``, ``classify_and_record``, and the
    failure-classification helpers were all unconditional. A ``--dry-run``
    pass could terminate live worker processes and permanently mutate
    shared state -- exactly the side effects ``--dry-run`` exists to
    suppress.

    This test drives a STALLED worker through the reaper with a
    ``dry_run=True`` gate and asserts:
      1. ``kill_process_tree`` is never called (no process kills).
      2. ``state.json`` is never written (no ``session_stalled`` event).
      3. The worker's sidecar file contents are byte-identical to the
         pre-call snapshot -- ``update_worker_log_stat``,
         ``classify_and_record``, and the failure-classification helpers
         are all gated on ``write_gate.dry_run`` and do not mutate the
         sidecar under ``--dry-run``.
      4. The function still returns the stalled entry (detection is
         read-only and must keep working under dry-run so the caller can
         reason about what *would* happen).
    """
    from charlie_work import workflow

    issue_number = 1325
    log_text = "Working on task...\nLast line\n"
    sessions_dir, state_file, _ = _make_stalled_devin_session(tmp_path, issue_number, log_text)

    # Snapshot the sidecar before the call so we can assert byte-identical
    # contents after. This is the assertion that catches ungated sidecar
    # writes (update_worker_log_stat, classify_and_record, the
    # failure-classification helpers) -- the narrower ``not state_file.exists()``
    # check alone missed them because those helpers write to the *sidecar*,
    # not to state.json.
    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    sidecar_before = sidecar_path.read_bytes()

    killed: list[int] = []
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, start_time=None: killed.append(pid) or [pid],
    )
    monkeypatch.setattr(
        "charlie_work.dead_worker_reap.sweep_orphan_processes", lambda worktree_path: []
    )
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_devin_probe)

    config = OrchestratorConfig()

    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, write_gate=_wg(state_file, dry_run=True)
    )

    # Detection still works -- the stalled entry is returned so the caller
    # can reason about what *would* have been reaped.
    assert result == [{"issue": issue_number, "pid": 99999}]

    # No process kills under dry_run.
    assert killed == []

    # No state.json was written -- the file does not exist because
    # ``write_gate.save_state`` is a no-op under dry_run and never touched
    # disk. ``load_state`` on a non-existent path returns a default empty
    # state, so the in-memory pipeline ran, but nothing persisted.
    assert not state_file.exists()

    # The sidecar file is byte-identical to the pre-call snapshot. This
    # catches ungated sidecar writes that the ``not state_file.exists()``
    # check above cannot detect (sidecar writes go to the sidecar, not
    # state.json).
    assert sidecar_path.read_bytes() == sidecar_before


def test_detect_and_handle_stalled_sessions_dry_run_suppresses_api_budget_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1325: under ``dry_run=True`` the api-budget-exceeded branch
    must not kill the process, must not mutate the sidecar, and must not
    emit a ``session_budget_exceeded`` event.

    Before the fix, the budget-exceeded branch's ``kill_process_tree`` and
    sidecar write were gated, but the ``session_budget_exceeded`` event
    write was already routed through ``write_gate.append_event`` (a no-op
    under dry-run). However, the sidecar write at the top of the branch
    (``failure_kind="budget_exceeded"``) was the only sidecar write in the
    function that *was* gated in the original PR -- the surrounding
    ``update_worker_log_stat`` calls that run unconditionally for every
    worker were not. This test asserts the full branch is suppressed under
    dry-run, matching the coverage already added for the STALLED/DEAD
    branch.
    """
    from charlie_work import workflow

    issue_number = 1326
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / f"issue-{issue_number}.claude.log"
    log_path.write_text("Working on task...\nLast line\n", encoding="utf-8")

    sidecar_path = claude_sidecar_path(sessions_dir, issue_number, "api")
    record = ClaudeWorkerRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("claude", "prompt.md"),
        pid=88888,
        started_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        adapter_kind="api",
        provider="example",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    sidecar_before = sidecar_path.read_bytes()

    killed: list[int] = []
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, start_time=None: killed.append(pid) or [pid],
    )
    monkeypatch.setattr(
        "charlie_work.dead_worker_reap.sweep_orphan_processes", lambda worktree_path: []
    )
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_devin_probe)
    # Force the budget check to fire so the api-budget-exceeded branch is
    # exercised without needing a real events.jsonl + token-usage setup.
    monkeypatch.setattr("charlie_work.worker._api_session_over_budget", lambda view, config: True)

    config = OrchestratorConfig()

    state_file = tmp_path / "state.json"
    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir,
        state_file,
        config,
        write_gate=_wg(state_file, dry_run=True),
    )

    # Detection still works -- the budget-exceeded entry is returned.
    assert result == [{"issue": issue_number, "pid": 88888}]

    # No process kills under dry_run.
    assert killed == []

    # No state.json was written.
    assert not state_file.exists()

    # The sidecar file is byte-identical -- no ``failure_kind=budget_exceeded``
    # mutation, no ``update_worker_log_stat`` progress-field mutation.
    assert sidecar_path.read_bytes() == sidecar_before


def test_detect_and_handle_stalled_sessions_dry_run_suppresses_rate_limit_defer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1325: under ``dry_run=True`` the STALLED + rate-limit-defer
    branch (workflow.py ~1671-1695) must not mutate the sidecar's
    ``rate_limit_defer_until`` field, must not emit a
    ``session_rate_limit_deferred`` event, and must not write ``state.json``.

    The non-dry-run counterpart is
    ``test_stalled_worker_with_rate_limit_signature_is_deferred``. This test
    reuses the same fixture (``_make_stalled_devin_session`` with the
    rate-limit log tail) and the same ``WatchdogConfig`` so the only
    independent variable is ``write_gate.dry_run``. A future revert of the
    ``if not write_gate.dry_run:`` guard around the
    ``update_worker_log_stat(..., rate_limit_defer_until=defer_until)`` call
    at workflow.py ~1673-1676 would set ``rate_limit_defer_until`` on the
    sidecar under dry-run and fail the byte-identical assertion below.
    """
    from charlie_work import workflow

    issue_number = 1327
    log_text = (
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n"
    )
    sessions_dir, state_file, _ = _make_stalled_devin_session(tmp_path, issue_number, log_text)

    # The fixture writes the sidecar with rate_limit_defer_until=None, which
    # is exactly the branch-entry condition (``w.rate_limit_defer_until is
    # None``). Snapshot the bytes so the assertion catches any mutation,
    # not just the rate_limit_defer_until field.
    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    sidecar_before = sidecar_path.read_bytes()
    assert json.loads(sidecar_before)["rate_limit_defer_until"] is None

    killed: list[int] = []
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, start_time=None: killed.append(pid) or [pid],
    )
    monkeypatch.setattr(
        "charlie_work.dead_worker_reap.sweep_orphan_processes", lambda worktree_path: []
    )
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_devin_probe)

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        )
    )

    frozen_now = datetime.now(UTC)
    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir,
        state_file,
        config,
        write_gate=_wg(state_file, dry_run=True),
        now=frozen_now,
    )

    # Detection still works -- the rate-limit signature is recognized and
    # the worker is deferred (not killed), so no stalled entry is returned.
    assert result == []
    assert killed == []

    # No state.json was written -- ``write_gate.save_state`` is a no-op
    # under dry_run, so the file the gate would have written does not exist.
    assert not state_file.exists()

    # No ``session_rate_limit_deferred`` event was recorded. The event is
    # routed through ``write_gate.append_event`` (a no-op under dry_run) and
    # ``write_gate.save_state`` never persists, so there is no state.json to
    # inspect -- the ``not state_file.exists()`` check above covers both.
    # Belt-and-suspenders: assert no events.db was created either.
    assert not (state_file.parent / "events.db").exists()

    # The sidecar's ``rate_limit_defer_until`` field is unchanged (still
    # None) -- the ``update_worker_log_stat(..., rate_limit_defer_until=...)``
    # call at workflow.py ~1673-1676 is gated on ``write_gate.dry_run``.
    sidecar_after = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar_after["rate_limit_defer_until"] is None

    # The sidecar file is byte-identical to the pre-call snapshot. This is
    # the assertion that catches an ungated ``update_worker_log_stat`` call
    # on this branch -- the field-level check above would miss a different
    # field being mutated (e.g. ``last_activity_at``).
    assert sidecar_path.read_bytes() == sidecar_before


def test_detect_and_handle_stalled_sessions_emits_provider_suspended_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1342: when the stall lane kills an api worker whose log carries a
    provider account-suspension signature, the dead-session lane (run in
    ``loop()``'s own order immediately after) emits a distinct
    ``api_worker_provider_suspended`` event on first detection so the operator
    learns about a billing problem in minutes, and classifies the sidecar
    ``failure_kind=provider_suspended`` so the issue escalates immediately
    instead of burning the redispatch cap."""
    from charlie_work import workflow

    # Issue #1317: kill_process_tree / sweep_orphan_processes are called
    # bare-name from inside _sweep_orphan_processes_for_dead_sessions and
    # _classify_dead_sessions_and_update_throttle_state, both of which moved
    # (verbatim) to dead_worker_reap.py -- patch them there, not on the
    # (still-valid) workflow.py facade re-export of
    # _detect_and_handle_stalled_sessions itself.
    from charlie_work import dead_worker_reap
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    issue_number = 1342
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / f"issue-{issue_number}.claude.log"
    # Fresh log (not stalled by mtime) carrying the suspension signature —
    # Signal 2.5 must fire on the tail regardless of mtime.
    log_path.write_text(
        "Working...\n"
        "Error: suspended due to insufficient balance, please recharge your "
        "account or check your plan and billing details.\n"
        "Retrying in 60s...\n",
        encoding="utf-8",
    )

    sidecar_path = claude_sidecar_path(sessions_dir, issue_number, "api")
    record = ClaudeWorkerRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("claude", "prompt.md"),
        pid=77777,
        started_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        adapter_kind="api",
        provider="example",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    killed: list[int] = []
    monkeypatch.setattr(dead_worker_reap, "sweep_orphan_processes", lambda worktree_path: [])

    # The stall lane sees a live worker (Signal 2.5 fires on the log tail);
    # after it kills the PID, the dead lane sees a dead worker. Use a mutable
    # flag flipped by kill_process_tree to model the real liveness transition.
    alive = {"yes": True}
    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: alive["yes"])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: alive["yes"])
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", lambda *args: None)

    def _kill_and_flip(pid, start_time=None):
        alive["yes"] = False
        return killed.append(pid) or [pid]

    monkeypatch.setattr("charlie_work.write_gate.kill_process_tree", _kill_and_flip)

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": [], "issues": {}}), encoding="utf-8")

    # Stall lane: kills the worker (Signal 2.5 → DEAD) and classifies the
    # sidecar failure_kind=provider_suspended.
    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, write_gate=_wg(state_file)
    )

    # The worker is killed within one supervision pass (no stall-minute wait).
    assert result == [{"issue": issue_number, "pid": 77777}]
    assert killed == [77777]

    # The sidecar is classified provider_suspended (terminal).
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["failure_kind"] == "provider_suspended"

    # Dead lane (run in loop()'s own order immediately after the stall lane):
    # reaps the sidecar, emits the distinct event, and escalates.
    class FakeGitHub:
        def __init__(self) -> None:
            self.issues = [
                {
                    "number": issue_number,
                    "title": "Test issue",
                    "url": "https://example.test/issues/1342",
                    "body": "Test",
                    "labels": [{"name": config.labels.in_progress}],
                }
            ]
            self.prs = []
            self.labels_added = []
            self.labels_removed = []

        def issue_list(self, labels=None, state=None):
            return self.issues

        def issue_view(self, number: int):
            for issue in self.issues:
                if issue["number"] == number:
                    return issue
            raise ValueError(f"Issue {number} not found")

        def pr_list(self):
            return self.prs

        def add_issue_label(self, number: int, label: str) -> bool:
            self.labels_added.append((number, label))
            return True

        def remove_issue_label(self, number: int, label: str) -> bool:
            self.labels_removed.append((number, label))
            return True

    fake_gh = FakeGitHub()

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir,
        state_file,
        fake_gh,
        config,
        write_gate=_wg(state_file),
    )

    # The sidecar is reaped by the dead lane.
    assert not sidecar_path.exists(), "Sidecar must be reaped after dead-session classification"

    # A distinct event fires on first detection.
    state = json.loads(state_file.read_text(encoding="utf-8"))
    kinds = [e.get("kind") for e in state.get("events", [])]
    assert "api_worker_provider_suspended" in kinds
    suspended_event = next(
        e for e in state["events"] if e.get("kind") == "api_worker_provider_suspended"
    )
    assert suspended_event["payload"]["issue_number"] == issue_number
    assert suspended_event["payload"]["provider"] == "example"

    # The issue is escalated (deterministic escalation), not redispatched.
    assert "session_failed_escalated" in kinds
    escalated_event = next(
        e for e in state["events"] if e.get("kind") == "session_failed_escalated"
    )
    assert escalated_event["payload"]["failure_kind"] == "provider_suspended"
