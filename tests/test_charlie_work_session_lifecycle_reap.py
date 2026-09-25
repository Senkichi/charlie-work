"""Worker-session reap: dead-dispatch reap grace/disabled, session_failed_relabeled payload, stale session tmp cleanup.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8; sub-split for the PR #1750 review's 800-line cap).
"""

from __future__ import annotations

import json
import time
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
import pytest
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
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_dead_dispatched_worker_reaped_after_grace_period(tmp_path: Path) -> None:
    """Issue #654: a dead dispatched worker whose drift was already surfaced on
    a prior pass (``orphan_drift_at`` set) but whose PR state did not qualify
    for auto-reset (clean exit with no push -- the #773 no-op branch) must be
    escalated to ``agent:human-needed`` after ``dead_dispatched_reap_minutes``,
    not held in ``dispatched`` indefinitely.  This is the exact scenario from
    job-cannon #1408: the rework worker made 5 local commits, exited 0 without
    pushing, and the dispatch label held for 1+ hour because the #773 branch
    surfaces drift but never resets status or clears the label.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, dead_dispatched_reap_minutes=60),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Simulate the "second pass" state: the first pass already emitted drift
    # (dead_worker_clean_exit_no_op) and set orphan_drift_at.  The drift
    # fingerprint matches the #773 clean-exit branch so the specific sub-branch
    # would short-circuit to ``continue`` without this fix.
    old_drift_at = (datetime.now(UTC) - timedelta(minutes=120)).isoformat().replace("+00:00", "Z")
    fingerprint = json.dumps(
        {"reason": "dead_worker_clean_exit_no_op", "reviewed_head_sha": "abc123"},
        sort_keys=True,
        default=str,
    )
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "orphan_drift_at": old_drift_at,
        "orphan_drift_fingerprint": fingerprint,
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    # Issue #1229: the branch-issue validator threaded through
    # _detect_and_handle_orphaned_workers calls issue_list(state="open") and
    # rejects branch-name issue numbers absent from the open-issue set. The
    # default FakeGitHub.issues only carries #123, so #207 must be planted
    # here or the validator rejects the agent/issue-207 binding and the orphan
    # sweep cannot match the PR to the issue (pr_number resolves to None,
    # orphan_drift_at gets overwritten instead of preserved).
    fake_gh.issues.append(
        {"number": 207, "title": "test issue 207", "state": "OPEN", "labels": [], "body": ""}
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = sessions_dir / "issue-207.claude.terminal.json"
    terminal_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 0,
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:05:00Z",
                "duration_seconds": 300.0,
            }
        ),
        encoding="utf-8",
    )

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # The time-based escape must have escalated the issue.
    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "dead_dispatched_worker_reap"
    assert entry.get("reason_class") == "mechanical"
    assert entry.get("dispatched_at") is None
    # The drift fingerprint/at are cleared so a de-escalated issue does not
    # immediately re-trigger the drift path.
    assert entry.get("orphan_drift_at") is None
    assert entry.get("orphan_drift_fingerprint") is None
    # Issue #282: the liveness fingerprint is preserved.
    assert entry["worker_pid"] == 99999

    # The ``escalated`` label edge must have been applied via transition().
    # Issue #1266: dead_dispatched_worker_reap is mechanical, so it lands
    # agent:operator-queue, not agent:human-needed.
    assert (207, config.labels.operator_queue) in fake_gh.labels_added
    assert (207, config.labels.in_progress) in fake_gh.labels_removed

    # A dedicated reap event must be recorded.
    reaped_events = [
        e for e in state.get("events", []) if e.get("kind") == "dead_dispatched_worker_reaped"
    ]
    assert len(reaped_events) == 1
    payload = reaped_events[0]["payload"]
    assert payload["issue_number"] == 207
    assert payload["pr_number"] == 100
    assert payload["previous_status"] == "dispatched"
    assert payload["reason"] == "dead_dispatched_worker_reap"
    assert payload["reap_minutes"] == 60
    assert payload["exit_code"] == 0


def test_dead_dispatched_worker_not_reaped_within_grace_period(tmp_path: Path) -> None:
    """Issue #654: a dead dispatched worker whose drift was surfaced recently
    (within ``dead_dispatched_reap_minutes``) must NOT be time-escalated.  The
    existing drift-only behavior (fingerprint match short-circuits to
    ``continue``) is preserved so a freshly-dead worker is not prematurely
    escalated before its specific sub-branch has had a chance to act.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, dead_dispatched_reap_minutes=60),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    recent_drift_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    fingerprint = json.dumps(
        {"reason": "dead_worker_clean_exit_no_op", "reviewed_head_sha": "abc123"},
        sort_keys=True,
        default=str,
    )
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "orphan_drift_at": recent_drift_at,
        "orphan_drift_fingerprint": fingerprint,
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)

    # Issue #1362 Stage 1: the last-review-decision read in
    # _detect_and_handle_orphaned_workers is now file-first, so the live
    # request_changes decision must exist on disk, not only in state.json,
    # or it resolves to "missing" and misses the clean-exit-no-op fingerprint
    # short-circuit this test is exercising.
    pr_decision_dir = paths.prs / "pr-100"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "abc123"}),
        encoding="utf-8",
    )

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    # Issue #1229: the branch-issue validator threaded through
    # _detect_and_handle_orphaned_workers calls issue_list(state="open") and
    # rejects branch-name issue numbers absent from the open-issue set. The
    # default FakeGitHub.issues only carries #123, so #207 must be planted
    # here or the validator rejects the agent/issue-207 binding, the orphan
    # sweep cannot match the PR to the issue, and the fingerprint short-circuit
    # (which requires the PR to be found) never fires — orphan_drift_at gets
    # overwritten with a fresh timestamp instead of being preserved.
    fake_gh.issues.append(
        {"number": 207, "title": "test issue 207", "state": "OPEN", "labels": [], "body": ""}
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = sessions_dir / "issue-207.claude.terminal.json"
    terminal_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 0,
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:05:00Z",
                "duration_seconds": 300.0,
            }
        ),
        encoding="utf-8",
    )

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # Within the grace period: the existing drift-only behavior is preserved.
    # The fingerprint match short-circuits to ``continue`` without escalating.
    assert entry.get("status") == "dispatched"
    assert entry.get("escalation_reason") is None
    assert entry.get("orphan_drift_at") == recent_drift_at

    # No reap event, no label transition.
    reaped_events = [
        e for e in state.get("events", []) if e.get("kind") == "dead_dispatched_worker_reaped"
    ]
    assert reaped_events == []
    assert (207, config.labels.human_needed) not in fake_gh.labels_added


def test_dead_dispatched_worker_reap_disabled_by_config(tmp_path: Path) -> None:
    """Issue #654: ``dead_dispatched_reap_minutes=0`` disables the time-based
    escape, reverting to the pre-#654 hold-forever behavior.  A dead dispatched
    worker with old drift stays ``dispatched`` -- the operator explicitly opted
    out.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, dead_dispatched_reap_minutes=0),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    old_drift_at = (datetime.now(UTC) - timedelta(minutes=120)).isoformat().replace("+00:00", "Z")
    fingerprint = json.dumps(
        {"reason": "dead_worker_clean_exit_no_op", "reviewed_head_sha": "abc123"},
        sort_keys=True,
        default=str,
    )
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "orphan_drift_at": old_drift_at,
        "orphan_drift_fingerprint": fingerprint,
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = sessions_dir / "issue-207.claude.terminal.json"
    terminal_path.write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 0,
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:05:00Z",
                "duration_seconds": 300.0,
            }
        ),
        encoding="utf-8",
    )

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # With the escape disabled, the issue stays dispatched (pre-#654 behavior).
    assert entry.get("status") == "dispatched"
    assert entry.get("escalation_reason") is None
    reaped_events = [
        e for e in state.get("events", []) if e.get("kind") == "dead_dispatched_worker_reaped"
    ]
    assert reaped_events == []


def test_dead_dispatched_worker_provider_throttled_not_reaped(tmp_path: Path) -> None:
    """Issue #1917: a dead dispatched worker whose death was classified as a
    provider throttle (``failure_kind="rate_limited"``) must NOT be escalated
    by the ``dead_dispatched_reap_minutes`` timed backstop, even after the
    grace period elapses.  The classification reaches this state.json-keyed
    lane through ``dead_worker_failure_kind`` (stamped on the issue entry by
    the stall/dead reap lanes before the sidecar is reaped); the exempted
    death falls through to the normal no-open-PR handling, which reclaims the
    issue's labels back to the dispatchable pool, and the redispatch is held
    by the ``throttled_until`` governor deferral until the window passes.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, dead_dispatched_reap_minutes=60),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Second-pass state: drift was already surfaced and is past the grace
    # period -- this is exactly the shape that escalates under
    # dead_dispatched_worker_reap (see the #654 test above) -- but the
    # worker's death carried a provider-throttle classification.
    old_drift_at = (datetime.now(UTC) - timedelta(minutes=120)).isoformat().replace("+00:00", "Z")
    fingerprint = json.dumps(
        {"reason": "dead_worker_clean_exit_no_op", "reviewed_head_sha": "abc123"},
        sort_keys=True,
        default=str,
    )
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "orphan_drift_at": old_drift_at,
        "orphan_drift_fingerprint": fingerprint,
        "dead_worker_failure_kind": "rate_limited",
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)

    # Issue #1362 Stage 1: the last-review-decision read in
    # _detect_and_handle_orphaned_workers is file-first, so the live
    # request_changes decision must exist on disk for the clean-exit-no-op
    # fingerprint short-circuit to match after the timed-reap exemption
    # lets the entry fall through.
    pr_decision_dir = paths.prs / "pr-100"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "abc123"}),
        encoding="utf-8",
    )

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues.append(
        {"number": 207, "title": "test issue 207", "state": "OPEN", "labels": [], "body": ""}
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "issue-207.claude.terminal.json").write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 0,
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:05:00Z",
                "duration_seconds": 300.0,
            }
        ),
        encoding="utf-8",
    )

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    # The provider-throttle death must never reach the timed escalation.
    assert entry.get("status") == "dispatched"
    assert entry.get("escalation_reason") is None
    assert (207, config.labels.operator_queue) not in fake_gh.labels_added
    assert (207, config.labels.human_needed) not in fake_gh.labels_added

    reaped_events = [
        e for e in state.get("events", []) if e.get("kind") == "dead_dispatched_worker_reaped"
    ]
    assert reaped_events == []


def test_dead_dispatched_worker_provider_throttled_reclaimed_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1917 end-to-end: a dispatched worker that dies rate-limited is
    classified by the stall lane (which stamps ``dead_worker_failure_kind``
    on the issue entry and arms ``throttled_until``), is then NOT escalated
    by the timed dead-dispatched reap on a later pass, does not consume an
    ``orphan_redispatch_at`` attempt, gets its labels reclaimed to the
    dispatchable pool, and becomes eligible for dispatch again once
    ``throttled_until`` passes.
    """
    from unittest.mock import patch

    from _worker_fixtures import _make_stalled_devin_session, _stale_devin_probe
    from charlie_work import dead_worker_reap
    from charlie_work.state import is_throttled

    issue_number = 249
    log_text = (
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n"
    )
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(
            enabled=True,
            stall_minutes=20,
            dead_dispatched_reap_minutes=60,
            rate_limit_defer_enabled=True,
            rate_limit_defer_slack_minutes=2,
        ),
    )

    frozen_now = datetime.now(UTC)
    past_defer = (frozen_now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    sessions_dir, state_file, _ = _make_stalled_devin_session(
        tmp_path, issue_number, log_text, rate_limit_defer_until=past_defer
    )

    # Seed the issue entry the way dispatch left it, with drift already past
    # the timed-reap grace period -- the shape that escalates without #1917.
    old_drift_at = (frozen_now - timedelta(minutes=120)).isoformat().replace("+00:00", "Z")
    state = load_state(state_file)
    state["issues"][str(issue_number)] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1710000000.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "orphan_drift_at": old_drift_at,
    }
    save_state(state_file, state)

    # --- Lane 1: the stall sweep classifies the death as rate_limited ---
    killed = []
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, start_time=None: killed.append(pid) or [pid],
    )
    monkeypatch.setattr(dead_worker_reap, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: True)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_devin_probe)

    from charlie_work import workflow

    result = workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, write_gate=_wg(state_file), now=frozen_now
    )
    assert result == [{"issue": issue_number, "pid": 99999}]
    assert killed == [99999]

    state = load_state(state_file)
    entry = state["issues"][str(issue_number)]
    # The classification is stamped on the issue entry and the fleet-wide
    # cooldown is armed.
    assert entry["dead_worker_failure_kind"] == "rate_limited"
    assert state.get("throttled_until") is not None

    # --- Lane 2: the state.json-keyed orphan sweep must not escalate ---
    fake_gh = FakeGitHub()
    fake_gh.issues.append(
        {
            "number": issue_number,
            "title": "test issue 249",
            "state": "OPEN",
            "labels": [config.labels.in_progress],
            "body": "",
        }
    )

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, state_file, config, fake_gh, write_gate=_wg(state_file)
        )

    state = load_state(state_file)
    entry = state["issues"][str(issue_number)]

    # No timed escalation, no operator-queue label.
    assert entry.get("status") == "dispatched"
    assert entry.get("escalation_reason") is None
    assert (issue_number, config.labels.operator_queue) not in fake_gh.labels_added
    assert (issue_number, config.labels.human_needed) not in fake_gh.labels_added
    assert [
        e for e in state.get("events", []) if e.get("kind") == "dead_dispatched_worker_reaped"
    ] == []

    # The throttle death does not consume orphan-redispatch bookkeeping.
    assert entry.get("orphan_redispatch_at") == []

    # The issue's labels are reclaimed to the dispatchable pool.
    assert (issue_number, config.labels.in_progress) in fake_gh.labels_removed
    assert (issue_number, config.labels.ready) in fake_gh.labels_added
    relabeled = [e for e in state.get("events", []) if e.get("kind") == "session_failed_relabeled"]
    assert relabeled
    assert relabeled[0]["payload"]["reason"] == "dead_worker_no_open_pr_orphan_sweep"

    # --- While throttled_until holds, dispatch defers ---
    assert is_throttled(state)

    # --- After throttled_until passes, the issue is dispatchable again ---
    state["throttled_until"] = (
        (datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    )
    assert not is_throttled(state)


def test_dead_dispatched_worker_non_throttle_kind_still_reaped(tmp_path: Path) -> None:
    """Issue #1917 control: a stamped non-throttle classification (e.g.
    ``"stalled"``) must NOT disable the timed reap -- the exemption is
    scoped to ``PROVIDER_THROTTLE_FAILURE_KINDS``, not to any stamp.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, dead_dispatched_reap_minutes=60),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    old_drift_at = (datetime.now(UTC) - timedelta(minutes=120)).isoformat().replace("+00:00", "Z")
    fingerprint = json.dumps(
        {"reason": "dead_worker_clean_exit_no_op", "reviewed_head_sha": "abc123"},
        sort_keys=True,
        default=str,
    )
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
        "orphan_drift_at": old_drift_at,
        "orphan_drift_fingerprint": fingerprint,
        "dead_worker_failure_kind": "stalled",
    }
    state["prs"]["100"] = {
        "decision": "request_changes",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-207",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues.append(
        {"number": 207, "title": "test issue 207", "state": "OPEN", "labels": [], "body": ""}
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "issue-207.claude.terminal.json").write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 0,
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:05:00Z",
                "duration_seconds": 300.0,
            }
        ),
        encoding="utf-8",
    )

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]

    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "dead_dispatched_worker_reap"
    assert (207, config.labels.operator_queue) in fake_gh.labels_added


def test_session_failed_relabeled_payload_requires_reason() -> None:
    """Issue #978: the shared payload builder makes a relabel event without a
    ``reason`` unrepresentable -- calling it without ``reason`` raises
    TypeError, the same property ``_escalate_issue`` gives escalation."""
    from charlie_work.workflow import _session_failed_relabeled_payload

    # reason is a required keyword-only argument.
    with pytest.raises(TypeError):
        _session_failed_relabeled_payload(issue_number=42)  # type: ignore[call-arg]

    # With reason, the payload always carries it; failure_kind is optional.
    payload = _session_failed_relabeled_payload(issue_number=42, reason="dead_worker_no_open_pr")
    assert payload["reason"] == "dead_worker_no_open_pr"
    assert "failure_kind" not in payload

    payload = _session_failed_relabeled_payload(
        issue_number=42, reason="dead_worker_no_open_pr", failure_kind="stalled"
    )
    assert payload["reason"] == "dead_worker_no_open_pr"
    assert payload["failure_kind"] == "stalled"


def test_cleanup_stale_session_tmp_files_removes_stranded_tmp(
    tmp_path: Path,
) -> None:
    """Issue #1393: cleanup_stale_session_tmp_files removes stranded .json.tmp
    files from the sessions directory (left behind by an interrupted atomic
    write) without touching the valid .json sidecar.

    The tmp files are aged past the ``min_age_seconds`` threshold so the sweep
    treats them as genuinely stranded rather than in-flight.
    """
    import os

    from charlie_work.adapters import cleanup_stale_session_tmp_files

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)

    # A valid sidecar.
    (sessions_dir / "issue-100.json").write_text('{"ok": true}', encoding="utf-8")
    # A stranded tmp from an interrupted write.
    (sessions_dir / "issue-100.json.tmp").write_text('{"ok": ', encoding="utf-8")
    # Another stranded tmp for a different issue.
    (sessions_dir / "issue-200.json.tmp").write_text('{"partial": ', encoding="utf-8")

    # Age both tmp files beyond the default 60s threshold so the sweep
    # treats them as stranded, not in-flight.
    stale_mtime = time.time() - 120
    os.utime(sessions_dir / "issue-100.json.tmp", (stale_mtime, stale_mtime))
    os.utime(sessions_dir / "issue-200.json.tmp", (stale_mtime, stale_mtime))

    removed = cleanup_stale_session_tmp_files(sessions_dir)

    assert removed == 2
    assert (sessions_dir / "issue-100.json").exists()
    assert not (sessions_dir / "issue-100.json.tmp").exists()
    assert not (sessions_dir / "issue-200.json.tmp").exists()


def test_cleanup_stale_session_tmp_files_skips_fresh_tmp(tmp_path: Path) -> None:
    """Issue #1393 regression: a freshly-created (not-yet-replaced) .json.tmp
    file must survive cleanup_stale_session_tmp_files so the sweep cannot race
    a legitimate in-flight atomic write between its close() and replace()
    calls — unlinking the tmp there crashes the writer with FileNotFoundError.
    """
    from charlie_work.adapters import cleanup_stale_session_tmp_files

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)

    # A tmp file a legitimate writer just created and closed but has not yet
    # replaced — its mtime is "now", well within the 60s grace window.
    fresh = sessions_dir / "issue-300.json.tmp"
    fresh.write_text('{"in-flight": ', encoding="utf-8")

    removed = cleanup_stale_session_tmp_files(sessions_dir)

    assert removed == 0
    assert fresh.exists()
    assert fresh.read_text(encoding="utf-8") == '{"in-flight": '


def test_cleanup_stale_session_tmp_files_missing_dir(tmp_path: Path) -> None:
    """cleanup_stale_session_tmp_files is a no-op when the sessions dir
    does not exist (e.g. first-ever dispatch pass)."""
    from charlie_work.adapters import cleanup_stale_session_tmp_files

    removed = cleanup_stale_session_tmp_files(tmp_path / "nonexistent")
    assert removed == 0
