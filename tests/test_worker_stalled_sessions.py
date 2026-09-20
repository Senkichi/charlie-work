"""``_detect_and_handle_stalled_sessions`` tests for ``charlie_work.worker`` and
``charlie_work.workflow``/``charlie_work.dead_worker_reap``.

Split out of ``tests/test_worker.py`` (issue #1574, Track-1 shoulder): the
stalled-session reaper's probe wiring (#301), inconclusive-probe
defer/escalate (#329, #338), dry-run suppression (#1325), and
provider-suspension classification (#1342) tests. The
``_log_is_stalled_at_shim`` predicate tests live in
``tests/test_worker_log_is_stalled_at_shim.py`` and the issue #247
rate-limit deferral lane in ``tests/test_worker_rate_limit_deferral.py``.
All test bodies moved verbatim; no renames, no fixture hoists into
``conftest.py``. The shared ``_wg``/``_make_stalled_devin_session``/
``_stale_devin_probe`` helpers live in ``tests/_worker_fixtures.py``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from _worker_fixtures import _make_stalled_devin_session, _stale_devin_probe, _wg
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
from charlie_work.post_mortem import ActivitySource, RealActivityProbe
from charlie_work.worker import WorkerHealth, WorkerView


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
