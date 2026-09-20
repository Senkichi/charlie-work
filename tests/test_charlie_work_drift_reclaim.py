"""Cadence-gated maintenance lanes: ``_maybe_reconcile_drift`` and ``_maybe_reclaim_superseded_main_ci``.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import CommandResult, OrchestratorApp
from _dispatch_fixtures import _main_ci_reclaim_app
from _dispatch_fixtures import _reconcile_pass_app
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_maybe_reconcile_drift_noop_when_disabled(tmp_path: Path) -> None:
    """B-AC1/B-AC2: reconcile_pass.enabled=False must skip reconcile entirely --
    no call to reconcile(), no schedule armed, no summary event."""
    app = _reconcile_pass_app(tmp_path, enabled=False)

    def _fail_if_called(*, fix: bool = False) -> CommandResult:
        raise AssertionError("reconcile() must not be called when reconcile_pass is disabled")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(app, "reconcile", _fail_if_called)
    try:
        app._maybe_reconcile_drift()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    assert state.get("reconcile_pass", {}).get("next_reconcile_at") is None
    events = state.get("events", [])
    assert not [e for e in events if str(e.get("kind", "")).startswith("reconcile_pass")]


def test_maybe_reconcile_drift_runs_and_arms_schedule_when_due(tmp_path: Path) -> None:
    """B-AC1/B-AC2: enabled + due calls _reconcile_locked(fix=True) exactly
    once, arms the next-due schedule ~interval_minutes out, and records a
    single reconcile_pass_completed summary event carrying drift counts.

    Patches ``_reconcile_locked``, not ``reconcile`` (merge-lane-recovery
    D-8a): ``_maybe_reconcile_drift`` calls ``_reconcile_locked`` directly
    because loop()'s caller already holds supervisor.lock for the whole
    pass, and re-entering ``reconcile()`` would re-acquire that same
    non-reentrant lock and always no-op. Patching ``reconcile`` here would
    silently never be invoked and this test would falsely pass with 0 calls
    recorded as a bug, not a confirmation -- see
    test_maybe_reconcile_drift_runs_while_supervisor_lock_held for the test
    that actually exercises that lock-contention distinction end to end."""
    from datetime import UTC, datetime

    app = _reconcile_pass_app(tmp_path, interval_minutes=30)
    original_reconcile_locked = app._reconcile_locked
    calls: list[tuple[bool, bool]] = []

    def _counting_reconcile_locked(
        *,
        fix: bool = False,
        skip_dead_session_sweep: bool = False,
        dry_run: bool = False,
    ) -> CommandResult:
        calls.append((fix, skip_dead_session_sweep))
        return original_reconcile_locked(
            fix=fix,
            skip_dead_session_sweep=skip_dead_session_sweep,
            dry_run=dry_run,
        )

    # frozen_now (issue #828) injected so the schedule assertion below is
    # exact instead of racing _reconcile_locked's own duration (or a CI
    # stall). No downstream real-clock dependency follows in this test, so
    # no offset is needed.
    frozen_now = datetime.now(UTC)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(app, "_reconcile_locked", _counting_reconcile_locked)
    try:
        app._maybe_reconcile_drift(now=frozen_now)
    finally:
        monkeypatch.undo()

    # merge-lane-recovery §6-B follow-up: the in-loop caller must skip
    # reconcile's own dead-session sweep -- the loop's stall/dead lanes
    # (_detect_and_handle_stalled_sessions /
    # _classify_dead_sessions_and_update_throttle_state) already ran this
    # exact pass, immediately before this call, with grace-period semantics
    # (max_inconclusive_probe_deferrals) that reconcile.py's sweep does not
    # implement.
    assert calls == [(True, True)]

    state = load_state(app.paths.state_file)
    next_at = state["reconcile_pass"]["next_reconcile_at"]
    expected = (
        (frozen_now + timedelta(minutes=30))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    assert next_at == expected

    events = state.get("events", [])
    completed = [e for e in events if e.get("kind") == "reconcile_pass_completed"]
    assert len(completed) == 1
    assert completed[0]["payload"]["drift_detected"] == 0
    assert completed[0]["payload"]["drift_fixed"] == 0
    assert completed[0]["payload"]["drift_remaining"] == 0


def test_maybe_reconcile_drift_waits_until_due(tmp_path: Path) -> None:
    """The periodic cadence must not re-run reconcile before the armed
    next_reconcile_at timestamp."""
    from datetime import UTC, datetime

    from charlie_work.state import arm_reconcile_pass

    app = _reconcile_pass_app(tmp_path, interval_minutes=30)
    not_due = (
        (datetime.now(UTC) + timedelta(minutes=25))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = arm_reconcile_pass(load_state(app.paths.state_file), not_due)
    save_state(app.paths.state_file, state)

    def _fail_if_called(*, fix: bool = False) -> CommandResult:
        raise AssertionError("must not reconcile before the scheduled time")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(app, "reconcile", _fail_if_called)
    try:
        app._maybe_reconcile_drift()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    assert state["reconcile_pass"]["next_reconcile_at"] == not_due
    events = state.get("events", [])
    assert not [e for e in events if str(e.get("kind", "")).startswith("reconcile_pass")]


def test_maybe_reconcile_drift_defers_on_graphql_rate_limit(tmp_path: Path) -> None:
    """B-AC3: reconcile()'s existing GraphQL rate-limit deferral must be
    preserved, not bypassed -- surfaced as a distinguishable
    reconcile_pass_deferred event rather than a silent no-op."""

    class LowBudgetGitHub(FakeGitHub):
        def check_graphql_rate_limit(self, threshold: int) -> tuple[bool, int, int | None]:
            return (False, 50, 1234567890)

    app = _reconcile_pass_app(tmp_path, gh=LowBudgetGitHub())

    app._maybe_reconcile_drift()

    state = load_state(app.paths.state_file)
    assert state["reconcile_pass"]["next_reconcile_at"] is not None
    events = state.get("events", [])
    deferred = [e for e in events if e.get("kind") == "reconcile_pass_deferred"]
    assert len(deferred) == 1
    assert deferred[0]["payload"]["deferred_reason"] == "graphql_rate_limit"
    assert deferred[0]["payload"]["graphql_remaining"] == 50
    completed = [e for e in events if e.get("kind") == "reconcile_pass_completed"]
    assert completed == []


def test_maybe_reconcile_drift_runs_while_supervisor_lock_held(tmp_path: Path) -> None:
    """merge-lane-recovery D-8a: every production caller of loop() -- the
    cli.py bash-rats handler, fleet_dispatch.py, and supervise.py's
    `while True` -- already holds supervisor.lock for the whole call, so
    _maybe_reconcile_drift must make progress under that exact condition.

    The buggy predecessor called `self.reconcile(fix=True, ...)`, which
    re-acquires supervisor.lock as its first action. Byte-range locks taken
    via msvcrt.locking(LK_NBLCK) are per-handle and non-reentrant even
    within one process (file_lock.py keeps no reentrancy bookkeeping), so
    that reacquisition always failed and reconcile silently no-opped on
    every one of the fleet's loop passes (0 reconcile events across 9,848
    recorded events). The fix calls `self._reconcile_locked(...)` directly,
    bypassing the lock acquisition entirely, since the precondition (lock
    already held by loop()'s caller) is guaranteed by this method's callers.

    A test that does not hold supervisor.lock during the call cannot
    distinguish the two implementations -- both acquire/no-op fine when
    unlocked, which is exactly how this shipped undetected.
    """
    from charlie_work import layout
    from charlie_work.file_lock import try_acquire_byte_range_lock

    app = _reconcile_pass_app(tmp_path, interval_minutes=30)

    supervisor_lock = try_acquire_byte_range_lock(layout.supervisor_lock_path(app.paths.root))
    assert supervisor_lock is not None, "test setup failed to acquire the supervisor lock"
    try:
        app._maybe_reconcile_drift()
    finally:
        supervisor_lock.release()

    state = load_state(app.paths.state_file)
    events = state.get("events", [])

    completed = [e for e in events if e.get("kind") == "reconcile_pass_completed"]
    assert len(completed) == 1, (
        "reconcile_pass_completed must be recorded even while supervisor.lock "
        f"is held by the loop() caller; events were: {events}"
    )

    # The assertion that actually discriminates the bug: the buggy code path
    # (self.reconcile(fix=True, ...) re-acquiring the same non-reentrant
    # lock) always produced this event with this exact reason instead.
    lock_held_skips = [
        e
        for e in events
        if e.get("kind") == "reconcile_pass_skipped"
        and e.get("payload", {}).get("reason") == "supervisor_lock_held"
    ]
    assert lock_held_skips == [], (
        "reconcile must not report supervisor_lock_held when the lock is "
        "already held by this same in-process loop() call -- that reason is "
        "reserved for a genuinely concurrent mop-up --fix"
    )


def test_maybe_reclaim_superseded_main_ci_noop_when_disabled(tmp_path: Path) -> None:
    from charlie_work import workflow as workflow_module

    app = _main_ci_reclaim_app(tmp_path, enabled=False)

    def _fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "reclaim_superseded_main_ci_runs must not be called when main_ci_reclaim is disabled"
        )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", _fail_if_called)
    try:
        app._maybe_reclaim_superseded_main_ci()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    events = state.get("events", [])
    assert not [e for e in events if str(e.get("kind", "")).startswith("main_ci_reclaim")]


def test_maybe_reclaim_superseded_main_ci_records_event_on_cancellation(
    tmp_path: Path,
) -> None:
    from charlie_work import workflow as workflow_module
    from charlie_work.main_ci_reclaim import MainCiReclaimResult, ReclaimedRun

    app = _main_ci_reclaim_app(tmp_path)
    canned = MainCiReclaimResult(
        ok=True,
        tip_sha="tip-sha",
        candidates_checked=2,
        cancelled=(
            ReclaimedRun(
                run_id=42, head_sha="old-sha", status_before_cancel="queued", created_at="t1"
            ),
        ),
        skipped_not_ancestor=1,
        skipped_started_before_cancel=0,
        cancel_errors=(),
    )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", lambda *a, **k: canned)
    try:
        app._maybe_reclaim_superseded_main_ci()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    events = state.get("events", [])
    cancelled_events = [e for e in events if e.get("kind") == "main_ci_reclaim_cancelled"]
    assert len(cancelled_events) == 1
    payload = cancelled_events[0]["payload"]
    assert payload["tip_sha"] == "tip-sha"
    assert payload["cancelled_run_ids"] == [42]
    assert payload["candidates_checked"] == 2
    assert payload["skipped_not_ancestor"] == 1


def test_maybe_reclaim_superseded_main_ci_no_event_when_nothing_to_reclaim(
    tmp_path: Path,
) -> None:
    """Deliberate no-noise policy (see the method's docstring): this lane has
    no cadence gate, so a durable event on every empty pass would flood
    events.db with zero diagnostic value. Only an actual cancellation or a
    pass-level failure is worth a durable record."""
    from charlie_work import workflow as workflow_module
    from charlie_work.main_ci_reclaim import MainCiReclaimResult

    app = _main_ci_reclaim_app(tmp_path)
    canned = MainCiReclaimResult(ok=True, tip_sha="tip-sha", candidates_checked=0, cancelled=())

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", lambda *a, **k: canned)
    try:
        app._maybe_reclaim_superseded_main_ci()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    events = state.get("events", [])
    assert not [e for e in events if str(e.get("kind", "")).startswith("main_ci_reclaim")]


def test_maybe_reclaim_superseded_main_ci_records_failed_event_on_pass_failure(
    tmp_path: Path,
) -> None:
    from charlie_work import workflow as workflow_module
    from charlie_work.main_ci_reclaim import MainCiReclaimResult

    app = _main_ci_reclaim_app(tmp_path)
    canned = MainCiReclaimResult(ok=False, error="git fetch origin main failed: boom")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", lambda *a, **k: canned)
    try:
        app._maybe_reclaim_superseded_main_ci()
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    events = state.get("events", [])
    failed_events = [e for e in events if e.get("kind") == "main_ci_reclaim_failed"]
    assert len(failed_events) == 1
    assert "boom" in failed_events[0]["payload"]["error"]


def test_maybe_reclaim_superseded_main_ci_contains_exception_and_records_event(
    tmp_path: Path,
) -> None:
    """Exception containment is load-bearing: supervise.py's except Exception
    sits outside its while True, so an uncaught exception from this lane
    would kill the whole daemon rather than one pass."""
    from charlie_work import workflow as workflow_module

    app = _main_ci_reclaim_app(tmp_path)

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", _raise)
    try:
        app._maybe_reclaim_superseded_main_ci()  # must not raise
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    events = state.get("events", [])
    failed_events = [e for e in events if e.get("kind") == "main_ci_reclaim_failed"]
    assert len(failed_events) == 1
    assert "RuntimeError" in failed_events[0]["payload"]["error"]
    assert "boom" in failed_events[0]["payload"]["error"]


def test_maybe_reclaim_superseded_main_ci_dry_run_writes_nothing(
    tmp_path: Path,
) -> None:
    """Issue #1324: under dry_run=True, _maybe_reclaim_superseded_main_ci must
    not write any main_ci_reclaim_* event to state.json or events.db, and
    state.json must stay byte-identical to the pre-pass seed. Before the fix,
    _record_event called append_event directly (bypassing self.write_gate) and
    the paired save_state was also raw, so a dry-run pass that found a
    cancellation wrote a real main_ci_reclaim_cancelled event + state.json
    mutation even though nothing was actually cancelled GitHub-side."""
    from charlie_work import workflow as workflow_module
    from charlie_work.config import MainCiReclaimConfig
    from charlie_work.instrumentation import event_counts_by_kind
    from charlie_work.main_ci_reclaim import MainCiReclaimResult, ReclaimedRun
    from charlie_work.state import empty_state, save_state

    config = OrchestratorConfig(
        main_ci_reclaim=MainCiReclaimConfig(enabled=True, workflow_filename="ci.yml")
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    save_state(paths.state_file, empty_state())
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)

    canned = MainCiReclaimResult(
        ok=True,
        tip_sha="tip-sha",
        candidates_checked=2,
        cancelled=(
            ReclaimedRun(
                run_id=42,
                head_sha="old-sha",
                status_before_cancel="queued",
                created_at="t1",
            ),
        ),
        skipped_not_ancestor=1,
        skipped_started_before_cancel=0,
        cancel_errors=(),
    )

    before_bytes = paths.state_file.read_bytes()
    events_before = sum(event_counts_by_kind(paths.state_file).values())

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "reclaim_superseded_main_ci_runs", lambda *a, **k: canned)
    try:
        app._maybe_reclaim_superseded_main_ci()
    finally:
        monkeypatch.undo()

    assert paths.state_file.read_bytes() == before_bytes, (
        "dry-run main_ci_reclaim pass must leave state.json byte-identical "
        "(issue #1324 WriteGate invariant)"
    )
    events_after = sum(event_counts_by_kind(paths.state_file).values())
    assert events_after == events_before, (
        f"dry-run main_ci_reclaim pass must not write any events.db row "
        f"(before={events_before}, after={events_after})"
    )
    state = load_state(paths.state_file)
    reclaim_events = [
        e for e in state.get("events", []) if str(e.get("kind", "")).startswith("main_ci_reclaim")
    ]
    assert reclaim_events == [], (
        f"dry-run main_ci_reclaim pass must not append any main_ci_reclaim_* "
        f"event to the state.json ring, found: {reclaim_events}"
    )
