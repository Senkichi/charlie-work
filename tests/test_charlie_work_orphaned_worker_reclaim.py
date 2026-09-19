"""Orphaned-worker no-open-PR reclaim: grace-period reaping, interrupted reclaim completion, label-failure survival, terminal-label-only leave-alone, and the required-reason contract.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from unittest.mock import patch
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
)


def test_orphaned_worker_no_open_pr_mention_flag_reaped_after_grace(tmp_path: Path) -> None:
    """Issue #1230: a dead dispatched worker with no open PR whose issue already
    carries ``agent:human-needed`` (applied by a mention-flag escalation) is
    invisible to the #417 ground-truth label reclaim (no active labels to
    remove) and invisible to the redispatch cap (never re-dispatched because the
    terminal label excludes it).  The no-open-PR drift branch sets
    ``orphan_flagged_at`` but never set ``orphan_drift_at``, so the
    ``dead_dispatched_reap_minutes`` time-based backstop — which keys off
    ``orphan_drift_at`` — never fired.  The issue wedged in ``dispatched``
    indefinitely: not dispatchable (status says a worker owns it), not
    sweepable (status is not ``escalated``), and the label promises a human
    review the state machine cannot act on.

    The fix: the no-open-PR drift branch must also stamp ``orphan_drift_at`` so
    the existing ``dead_dispatched_reap_minutes`` backstop converges the status
    to ``escalated`` after the grace period, regardless of concurrent
    label-side transitions.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, dead_dispatched_reap_minutes=60),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Seed the jc #1421 shape: dispatched, dead PID, no open PR, and the issue
    # already carries agent:human-needed from a mention-flag escalation
    # (merged_pr_mention_flagged_at set).  No active labels are present, so the
    # #417 reclaim is a no-op and the issue is excluded from dispatch.
    state = load_state(paths.state_file)
    state["issues"]["1421"] = {
        "status": "dispatched",
        "worker_pid": 45292,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2026-08-09T07:33:24Z",
        "merged_pr_mention_flagged_at": "2026-08-10T12:00:00Z",
    }
    save_state(paths.state_file, state)

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues = [
        {
            "number": 1421,
            "title": "call_model cascade timeout",
            "url": "https://example.test/issues/1421",
            "body": "",
            # Only the terminal human-needed label — no active labels.
            "labels": [{"name": config.labels.human_needed}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    def _run_sweep() -> None:
        with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
            _detect_and_handle_orphaned_workers(
                sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
            )

    # Pass 1: the no-open-PR drift branch fires (no active labels to reclaim,
    # no pushed branch).  It must stamp orphan_drift_at so the time-based
    # backstop can fire on a later pass.
    _run_sweep()
    st = load_state(paths.state_file)
    entry = st["issues"]["1421"]
    assert entry.get("orphan_flagged_at") is not None
    assert entry.get("orphan_drift_at") is not None, (
        "no-open-PR drift branch must set orphan_drift_at so the "
        "dead_dispatched_reap_minutes backstop can converge the wedge"
    )
    # Status is still dispatched — the grace period has not elapsed.
    assert entry.get("status") == "dispatched"

    # Simulate the grace period elapsing: rewind orphan_drift_at to 120 minutes
    # ago (past the 60-minute reap window).
    old_drift_at = (datetime.now(UTC) - timedelta(minutes=120)).isoformat().replace("+00:00", "Z")
    st["issues"]["1421"]["orphan_drift_at"] = old_drift_at
    save_state(paths.state_file, st)

    # Pass 2: the dead_dispatched_reap_minutes backstop must fire and converge
    # status to escalated, even though the escalation label is already present.
    _run_sweep()
    st = load_state(paths.state_file)
    entry = st["issues"]["1421"]
    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "dead_dispatched_worker_reap"
    assert entry.get("reason_class") == "mechanical"

    # The reap event must be recorded.
    reaped_events = [
        e for e in st.get("events", []) if e.get("kind") == "dead_dispatched_worker_reaped"
    ]
    assert len(reaped_events) == 1
    payload = reaped_events[0]["payload"]
    assert payload["issue_number"] == 1421
    assert payload["previous_status"] == "dispatched"
    assert payload["reason"] == "dead_dispatched_worker_reap"
    assert payload["reap_minutes"] == 60


def test_orphaned_worker_no_open_pr_already_flagged_backstop_backfills(tmp_path: Path) -> None:
    """Issue #1230 regression: the real wedge precondition is an entry that was
    flagged on a prior pass (``orphan_flagged_at`` set) but never received an
    ``orphan_drift_at`` stamp -- either because it was flagged by a pre-#1230
    build (which set only ``orphan_flagged_at``) or via the reclaim-success
    branch (which deliberately omits ``orphan_drift_at``).  The original #1230
    fix placed the ``orphan_drift_at`` stamp behind the
    ``orphan_flagged_at`` early-return guard, so already-flagged entries hit
    ``continue`` before the stamp was written and remained permanently
    unreachable -- the backstop never armed and the issue never escalated.

    The fix decouples the backfill from the guard: ``orphan_drift_at`` is
    backfilled from ``orphan_flagged_at`` whenever it is missing, BEFORE the
    duplicate-event guard's early-return.  This test seeds the exact wedge
    precondition (``orphan_flagged_at`` set, ``orphan_drift_at`` absent, grace
    period already elapsed) and verifies the backstop arms on the next sweep
    pass and the issue escalates.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, dead_dispatched_reap_minutes=60),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Seed the wedge precondition: orphan_flagged_at is set to 120 minutes ago
    # (past the 60-minute reap grace) but orphan_drift_at is ABSENT.  This is
    # the state a pre-#1230-flagged entry (or a reclaim-success entry that
    # later fell through to the drift branch) would be in.
    flagged_at = (datetime.now(UTC) - timedelta(minutes=120)).isoformat().replace("+00:00", "Z")
    state = load_state(paths.state_file)
    state["issues"]["1421"] = {
        "status": "dispatched",
        "worker_pid": 45292,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": (datetime.now(UTC) - timedelta(days=4))
        .isoformat()
        .replace("+00:00", "Z"),
        "merged_pr_mention_flagged_at": (datetime.now(UTC) - timedelta(days=3))
        .isoformat()
        .replace("+00:00", "Z"),
        "orphan_flagged_at": flagged_at,
        # orphan_drift_at deliberately ABSENT -- the wedge precondition.
    }
    save_state(paths.state_file, state)

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues = [
        {
            "number": 1421,
            "title": "call_model cascade timeout",
            "url": "https://example.test/issues/1421",
            "body": "",
            # Only the terminal human-needed label -- no active labels, so the
            # #417 reclaim is a no-op and the issue is excluded from dispatch.
            "labels": [{"name": config.labels.human_needed}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    def _run_sweep() -> None:
        with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
            _detect_and_handle_orphaned_workers(
                sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
            )

    # Pass 1: the drift branch is reached (no active labels to reclaim).  The
    # duplicate-event guard sees orphan_flagged_at already set, but the
    # backfill must run BEFORE the early-return so orphan_drift_at is armed.
    # The backstop at the top of the loop already ran this pass with
    # orphan_drift_at absent, so escalation cannot happen yet -- but the stamp
    # must now be present for the next pass.
    _run_sweep()
    st = load_state(paths.state_file)
    entry = st["issues"]["1421"]
    assert entry.get("orphan_drift_at") is not None, (
        "already-flagged entry must have orphan_drift_at backfilled from "
        "orphan_flagged_at so the backstop can arm on the next sweep pass"
    )
    # Backfill must use the original flag timestamp (not utc_now()) so an
    # already-wedged entry converges immediately instead of waiting another
    # full grace window.
    assert entry.get("orphan_drift_at") == flagged_at
    # The duplicate-event guard still suppresses the drift event -- no new
    # orphaned_worker_drift for this issue on this pass.
    drift_events = [
        e
        for e in st.get("events", [])
        if e.get("kind") == "orphaned_worker_drift"
        and e.get("payload", {}).get("issue_number") == 1421
    ]
    assert len(drift_events) == 0
    # Status is still dispatched -- the backstop runs at the top of the loop,
    # before the drift branch, so it could not fire this pass.
    assert entry.get("status") == "dispatched"

    # Pass 2: the backstop at the top of the loop now sees orphan_drift_at
    # (backfilled to 120 minutes ago, past the 60-minute grace) and must
    # escalate.
    _run_sweep()
    st = load_state(paths.state_file)
    entry = st["issues"]["1421"]
    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "dead_dispatched_worker_reap"
    assert entry.get("reason_class") == "mechanical"

    reaped_events = [
        e for e in st.get("events", []) if e.get("kind") == "dead_dispatched_worker_reaped"
    ]
    assert len(reaped_events) == 1
    payload = reaped_events[0]["payload"]
    assert payload["issue_number"] == 1421
    assert payload["previous_status"] == "dispatched"
    assert payload["reason"] == "dead_dispatched_worker_reap"
    assert payload["reap_minutes"] == 60


def test_orphaned_worker_no_open_pr_completes_interrupted_reclaim(tmp_path: Path) -> None:
    """Issue #417: a reclaim interrupted between the redispatch_at state.json
    write and the GitHub label swap (e.g. by a crash/reboot) must self-heal on
    the very next orphaned-worker sweep -- reproducing job-cannon #1172/#1176's
    exact fingerprint: status still "dispatched", worker_pid dead,
    redispatch_at already has one entry, and the GitHub issue still carries
    the stale active label alongside the ready label that was never removed
    from the original dispatch.

    Also covers the non-blocking follow-up: `status` deliberately never
    advances away from "dispatched" (matching the sidecar-based lane's own
    issue #282 fingerprint preservation), so a second pass would otherwise
    re-discover this same entry and emit a spurious orphaned_worker_drift for
    an issue that is already fully fixed. It must not.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1172"] = {
        "status": "dispatched",
        "dispatched_at": "2026-07-14T16:14:40Z",
        "redispatch_at": ["2026-07-14T16:17:20.606175Z"],
        "worker_pid": 40680,
        "worker_process_start_time": 1784045680.2843266,
    }
    save_state(paths.state_file, state)

    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 1172,
            "title": "some bug",
            "url": "https://example.test/issues/1172",
            "body": "",
            "labels": [
                {"name": config.labels.in_progress},
                {"name": config.labels.ready},
            ],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []  # dead worker never opened a PR

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    # The stale active label must have been removed -- this is exactly what
    # _is_dispatchable requires (ready present, no active label) for the
    # issue to become dispatchable again.
    assert (1172, config.labels.in_progress) in fake_gh.labels_removed
    # ready was already present, so it must NOT be redundantly re-added.
    assert (1172, config.labels.ready) not in fake_gh.labels_added

    state = load_state(paths.state_file)
    entry = state["issues"]["1172"]
    # This lane must never touch the sidecar-based lane's own bookkeeping --
    # a retry here must not inflate the escalation-cap counter.
    assert entry["redispatch_at"] == ["2026-07-14T16:17:20.606175Z"]
    assert entry["worker_pid"] == 40680

    events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events) == 1
    assert events[0]["payload"]["issue_number"] == 1172
    assert events[0]["payload"]["label_write_ok"] is True
    assert events[0]["payload"]["removed_labels"] == [config.labels.in_progress]

    # A once-stranded issue must not ALSO be flagged as unresolved drift now
    # that the reclaim fully succeeded.
    drift_events = [e for e in state["events"] if e["kind"] == "orphaned_worker_drift"]
    assert drift_events == []

    # Second pass: status.json still shows status="dispatched" with the same
    # dead worker_pid (nothing advanced it). FakeGitHub's remove/add_issue_label
    # only record calls -- unlike real GitHub, they don't mutate self.issues --
    # so simulate pass 1's successful label swap actually landing: in_progress
    # is gone, ready (already present) is unchanged. With labels now fully
    # correct (no active label, ready present) this must be a quiet no-op, not
    # a second "session_failed_relabeled" event or a fresh "orphaned_worker_drift"
    # for an issue that no longer needs anything.
    fake_gh.issues[0]["labels"] = [{"name": config.labels.ready}]
    fake_gh.labels_added = []
    fake_gh.labels_removed = []
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []
    state = load_state(paths.state_file)
    assert len([e for e in state["events"] if e["kind"] == "session_failed_relabeled"]) == 1
    assert [e for e in state["events"] if e["kind"] == "orphaned_worker_drift"] == []


def test_orphaned_worker_no_open_pr_reclaim_survives_label_api_failure(tmp_path: Path) -> None:
    """Issue #417: if the label swap itself fails (gh API error), the reclaim
    must not lose state.json bookkeeping or the sidecar-independent tracking,
    and a later pass -- once the API recovers -- must complete the reclaim.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1176"] = {
        "status": "dispatched",
        "dispatched_at": "2026-07-14T17:24:55Z",
        "redispatch_at": ["2026-07-14T17:29:56.087825Z"],
        "worker_pid": 29236,
        "worker_process_start_time": 1784049895.281971,
    }
    save_state(paths.state_file, state)

    class FlakyLabelGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.fail_remove = True

        def remove_issue_label(self, number: int, label: str) -> bool:
            self.labels_removed.append((number, label))
            return not self.fail_remove

    fake_gh = FlakyLabelGitHub()
    fake_gh.issues = [
        {
            "number": 1176,
            "title": "some other bug",
            "url": "https://example.test/issues/1176",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        # Pass 1: the gh API call fails.
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1176"]
    # Nothing lost: bookkeeping and the liveness fingerprint survive intact.
    assert entry["redispatch_at"] == ["2026-07-14T17:29:56.087825Z"]
    assert entry["worker_pid"] == 29236

    events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events) == 1
    assert events[0]["payload"]["label_write_ok"] is False
    assert (1176, config.labels.in_progress) in fake_gh.labels_removed

    # Pass 2: the API recovers.
    fake_gh.fail_remove = False
    fake_gh.labels_removed = []

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    assert (1176, config.labels.in_progress) in fake_gh.labels_removed
    assert (1176, config.labels.ready) in fake_gh.labels_added

    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events) == 2
    assert events[1]["payload"]["label_write_ok"] is True
    # The redispatch_at bookkeeping must never have been touched by retries of
    # this sidecar-independent lane -- only the sidecar-based reap lane
    # (_classify_dead_sessions_and_update_throttle_state) owns that counter.
    assert state["issues"]["1176"]["redispatch_at"] == ["2026-07-14T17:29:56.087825Z"]


def test_orphaned_worker_no_open_pr_terminal_label_only_is_left_alone(tmp_path: Path) -> None:
    """Issue #417 regression: an issue in a legitimate terminal state (only
    agent:human-needed -- no active label, no ready) that ALSO happens to
    have a stale dispatched/dead-worker/no-PR state.json entry must be LEFT
    ALONE by the ground-truth label reclaim. A prior revision's early-exit
    gate (`if not active_labels and not needs_ready: continue`) proceeded
    whenever EITHER half was false, so a terminal-only issue (active_labels
    empty, needs_ready true) wrongly got `automated-ready` added back --
    producing a contradictory human-needed + automated-ready label pair and
    polluting the audit trail with a spurious `added_ready: True`. This test
    must fail against a head that regresses to that gate.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["500"] = {
        "status": "dispatched",
        "dispatched_at": "2026-07-01T00:00:00Z",
        "worker_pid": 12345,
        "worker_process_start_time": 1700000000.0,
    }
    save_state(paths.state_file, state)

    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 500,
            "title": "needs a human",
            "url": "https://example.test/issues/500",
            "body": "",
            "labels": [{"name": config.labels.human_needed}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    # No GitHub label call at all -- not even a redundant re-add of a label
    # that was already there.
    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []

    state = load_state(paths.state_file)
    assert [e for e in state["events"] if e["kind"] == "session_failed_relabeled"] == []
    # state.json bookkeeping for this issue must be untouched by the reclaim
    # (the pre-existing orphaned_worker_drift diagnostic fallback may still
    # flag it -- that part of the behavior predates issue #417 and is not
    # this test's concern).
    entry = state["issues"]["500"]
    assert entry.get("worker_pid") == 12345
    assert "redispatch_at" not in entry


def test_orphaned_worker_reclaim_carries_required_reason(tmp_path: Path) -> None:
    """Issue #978: the orphan-sweep reclaim path must emit a
    ``session_failed_relabeled`` event with ``reason`` populated. This site
    already used ``reason`` before the fix, but it is now routed through the
    shared payload builder so the invariant is enforced at one point."""
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1978"] = {
        "status": "dispatched",
        "dispatched_at": "2026-08-01T00:00:00Z",
        "redispatch_at": ["2026-08-01T00:05:00Z"],
        "worker_pid": 55555,
        "worker_process_start_time": 1784000000.0,
    }
    save_state(paths.state_file, state)

    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 1978,
            "title": "orphan reclaim",
            "url": "https://example.test/issues/1978",
            "body": "",
            "labels": [
                {"name": config.labels.in_progress},
                {"name": config.labels.ready},
            ],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "dead_worker_no_open_pr_orphan_sweep"
