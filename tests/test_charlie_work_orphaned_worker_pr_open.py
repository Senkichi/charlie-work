"""Orphaned-worker unreviewed-open-PR advancement and drift bookkeeping: pr_open label advancement, label-failure fallback, rework-status advance-not-reset, and drift-fingerprint clearing on redispatch.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest
from _dead_session_fixtures import _write_flat_review_decision
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
    PASSIVE_OPEN_STATUS,
    load_state,
    save_state,
)
from charlie_work.workflow import OrchestratorApp


def test_orphaned_worker_approved_without_rework_status_still_drifts(tmp_path: Path) -> None:
    """Issue #1109 guard: an approved PR whose PR state does NOT carry
    ``status="rework_requested"`` has no evidence a post-approval rework lane
    dispatched this worker, so the sweep must still surface
    ``dead_worker_unsafe_to_auto_reset`` drift rather than guess.

    This is the existing test_orphaned_worker_unsafe_to_auto_reset_drift_emits_once
    scenario (approved, head unchanged, no PR-state status) -- re-asserted
    here to pin the guard's meaning: the ``pr_state_status == "rework_requested"``
    check is what separates a safe auto-reset from an unclassifiable drift.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1109"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    # Approved but NO status="rework_requested" -- no evidence a rework lane
    # dispatched this worker.
    state["prs"]["100"] = {
        "decision": "approved",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)
    _write_flat_review_decision(paths, 100, "approved", "abc123")

    class FakeGitHubForOrphan(FakeGitHub):
        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-1109",
                }
            ]

    fake_gh = FakeGitHubForOrphan()
    fake_gh.issues.append(
        {
            "number": 1109,
            "title": "Test issue",
            "url": "https://example.test/issues/1109",
            "body": "",
            "labels": [],
            "state": "OPEN",
        }
    )

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1109"]

    # No evidence of a rework lane dispatch -- must stay dispatched and drift.
    assert entry.get("status") == "dispatched"

    events = state.get("events", [])
    assert [e for e in events if e.get("kind") == "orphaned_worker_recovered"] == []
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    assert drift_events[0]["payload"]["reason"] == "dead_worker_unsafe_to_auto_reset"


def test_orphaned_worker_drift_fingerprint_cleared_on_redispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #457 review: a fresh dispatch clears the drift fingerprint.

    Without this, a worker that dies twice in the same failure class would
    recompute an identical fingerprint and the second orphaned_worker_drift
    event would be suppressed forever.
    """
    from unittest.mock import patch

    from charlie_work.adapters import SessionDispatchResult

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubForOrphan(FakeGitHub):
        def __init__(self):
            super().__init__()
            # Issue #457 must appear in issue_list() (the open-issue snapshot)
            # as well as issue_view(), so the branch-issue validator added in
            # issue #1229 accepts agent/issue-457 as a real open issue.
            self.issues = [
                *self.issues,
                {
                    "number": 457,
                    "title": "Orphan drift test",
                    "url": "https://example.test/issues/457",
                    "body": "",
                    "labels": [{"name": config.labels.needs_rework}],
                    "state": "OPEN",
                },
            ]

        def pr_list(self):
            return [
                {
                    "number": 100,
                    "headRefOid": "abc123",
                    "isCrossRepository": False,
                    "headRepository": {"owner": {"login": "test"}, "name": "repo"},
                    "headRefName": "agent/issue-457",
                }
            ]

        def issue_view(self, number: int):
            if number == 457:
                return {
                    "number": 457,
                    "title": "Orphan drift test",
                    "url": "https://example.test/issues/457",
                    "body": "",
                    "labels": [{"name": config.labels.needs_rework}],
                    "state": "OPEN",
                }
            return super().issue_view(number)

    fake_gh = FakeGitHubForOrphan()

    state = load_state(paths.state_file)
    state["issues"]["457"] = {
        "number": 457,
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["100"] = {
        "decision": "approved",
        "reviewed_head_sha": "abc123",
    }
    save_state(paths.state_file, state)

    # Issue #1229: review_decision() reads from the file, not state.json.
    # Without a review-decision.json file, last_decision is None, which
    # routes through the #1128 pr-open transition instead of the
    # dead_worker_unsafe_to_auto_reset drift path this test exercises.
    # Write the file so last_decision is "approved" as the state entry implies.
    pr_dir = paths.prs / "pr-100"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "abc123"}),
        encoding="utf-8",
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 1
    assert drift_events[0]["payload"]["reason"] == "dead_worker_unsafe_to_auto_reset"
    assert "orphan_drift_fingerprint" in state["issues"]["457"]

    # Simulate an external claim reset/redispatch: the issue returns to the
    # rework queue and dispatch_rework claims it again.
    state = load_state(paths.state_file)
    state["issues"]["457"]["status"] = "rework_requested"
    save_state(paths.state_file, state)

    rework_prompt = paths.prs / "pr-100" / "rework-prompt.md"
    rework_prompt.parent.mkdir(parents=True, exist_ok=True)
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=True,
                pid=12345,
                process_start_time=0.0,
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch_rework()

    assert result.ok is True
    assert result.data["selected_count"] == 1

    state = load_state(paths.state_file)
    assert state["issues"]["457"]["status"] == "dispatched"
    assert "orphan_drift_fingerprint" not in state["issues"]["457"]

    # Force the identical drift conditions again after the redispatch.
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 2, (
        f"Expected two orphaned_worker_drift events across dispatch generations, "
        f"got {len(drift_events)}"
    )


def test_orphaned_worker_unreviewed_open_pr_pending_file_advances_to_pr_open(
    tmp_path: Path,
) -> None:
    """Issue #1362 Stage 1 regression: a dead worker with an OPEN PR that has
    a *pending* placeholder ``review-decision.json`` (not a missing file, and
    no ``decision`` recorded in state.json either) must still be advanced
    from ``agent:in-progress`` to ``agent:pr-open`` -- a pending packet is
    "no verdict yet" exactly like a wholly-absent decision file, per the
    #1128 intent this lane implements. The single-reader predicate must check
    both ``.missing`` and a ``pending`` decision, not ``.missing`` alone --
    narrowing to ``.missing`` only would silently exclude this PR and
    re-strand the issue on ``agent:in-progress``, the precise #1128 failure
    this lane exists to fix.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    in_progress = config.labels.in_progress
    pr_open = config.labels.pr_open

    state = load_state(paths.state_file)
    state["issues"]["1578"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    # No ``decision`` key in state -- but the flat file records a pending
    # placeholder, not a missing file.
    state["prs"]["1585"] = {}
    save_state(paths.state_file, state)

    pr_dir = paths.prs / "pr-1585"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "pending"}), encoding="utf-8"
    )

    class FakeGitHubForOrphan(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 1578,
                    "title": "Salvage wedge",
                    "url": "https://example.test/issues/1578",
                    "body": "Dead worker with open unreviewed PR",
                    "labels": [{"name": in_progress}],
                    "state": "OPEN",
                }
            ]
            self.prs = [
                {
                    "number": 1585,
                    "title": "Salvaged work for #1578",
                    "url": "https://example.test/pull/1585",
                    "headRefName": "agent/issue-1578-salvage-wedge",
                    "baseRefName": "main",
                    "headRefOid": "sha-deadbeef",
                    "mergeStateStatus": "CLEAN",
                    "body": "Closes #1578\n\nTests: regression coverage added.",
                    "labels": [],
                    "isCrossRepository": False,
                    "state": "OPEN",
                }
            ]

    fake_gh = FakeGitHubForOrphan()

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1578"]
    assert entry["status"] == PASSIVE_OPEN_STATUS, (
        f"expected open_passive, got {entry['status']!r}"
    )
    assert entry.get("dispatched_at") is None

    events = state.get("events", [])
    advance_events = [e for e in events if e.get("kind") == "orphaned_worker_advanced_to_pr_open"]
    assert len(advance_events) == 1
    payload = advance_events[0]["payload"]
    assert payload["pr_number"] == 1585
    assert payload["reason"] == "dead_worker_unsafe_to_auto_reset_open_unreviewed_pr"

    # The label swap mirrors the orphaned_worker_opened_pr lane.
    assert (1578, in_progress) in fake_gh.labels_removed
    assert (1578, pr_open) in fake_gh.labels_added


def test_orphaned_worker_unreviewed_open_pr_advances_to_pr_open(tmp_path: Path) -> None:
    """Issue #1128: a dead worker with an OPEN, unreviewed PR (no decision)
    must be advanced from ``agent:in-progress`` to ``agent:pr-open`` so review
    dispatch can claim the salvage PR.  Before the fix this cell advanced no
    label and the issue sat on ``agent:in-progress`` indefinitely.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    in_progress = config.labels.in_progress
    pr_open = config.labels.pr_open

    state = load_state(paths.state_file)
    state["issues"]["1578"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    # No ``decision`` key -- the PR has not been reviewed yet.
    state["prs"]["1585"] = {
        "reviewed_head_sha": None,
    }
    save_state(paths.state_file, state)

    class FakeGitHubForOrphan(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 1578,
                    "title": "Salvage wedge",
                    "url": "https://example.test/issues/1578",
                    "body": "Dead worker with open unreviewed PR",
                    "labels": [{"name": in_progress}],
                    "state": "OPEN",
                }
            ]
            self.prs = [
                {
                    "number": 1585,
                    "title": "Salvaged work for #1578",
                    "url": "https://example.test/pull/1585",
                    "headRefName": "agent/issue-1578-salvage-wedge",
                    "baseRefName": "main",
                    "headRefOid": "sha-deadbeef",
                    "mergeStateStatus": "CLEAN",
                    "body": "Closes #1578\n\nTests: regression coverage added.",
                    "labels": [],
                    "isCrossRepository": False,
                    "state": "OPEN",
                }
            ]

    fake_gh = FakeGitHubForOrphan()

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1578"]
    assert entry["status"] == PASSIVE_OPEN_STATUS, (
        f"expected open_passive, got {entry['status']!r}"
    )
    assert entry.get("dispatched_at") is None

    events = state.get("events", [])
    advance_events = [e for e in events if e.get("kind") == "orphaned_worker_advanced_to_pr_open"]
    assert len(advance_events) == 1
    payload = advance_events[0]["payload"]
    assert payload["pr_number"] == 1585
    assert payload["previous_status"] == "dispatched"
    assert payload["new_status"] == PASSIVE_OPEN_STATUS
    assert payload["reason"] == "dead_worker_unsafe_to_auto_reset_open_unreviewed_pr"
    assert payload["label_write_ok"] is True
    assert in_progress in payload["removed_labels"]

    # No drift should be emitted -- the transition succeeded.
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert drift_events == []

    # The label swap mirrors the orphaned_worker_opened_pr lane.
    assert (1578, in_progress) in fake_gh.labels_removed
    assert (1578, pr_open) in fake_gh.labels_added

    # A second pass must not re-advance or re-emit (status is no longer
    # dispatched, so the sweep skips it entirely).
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    events = state.get("events", [])
    advance_events = [e for e in events if e.get("kind") == "orphaned_worker_advanced_to_pr_open"]
    assert len(advance_events) == 1, "advance must not be re-emitted on the second pass"


def test_orphaned_worker_unreviewed_open_pr_label_failure_falls_back_to_drift(
    tmp_path: Path,
) -> None:
    """Issue #1128: when the label write fails, the sweep must keep the
    conservative drift behavior (stay ``dispatched``, emit drift once) so the
    next pass re-attempts the transition rather than resetting the worker.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    in_progress = config.labels.in_progress

    state = load_state(paths.state_file)
    state["issues"]["1578"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    state["prs"]["1585"] = {
        "reviewed_head_sha": None,
    }
    save_state(paths.state_file, state)

    class FakeGitHubForOrphan(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 1578,
                    "title": "Salvage wedge",
                    "url": "https://example.test/issues/1578",
                    "body": "Dead worker with open unreviewed PR",
                    "labels": [{"name": in_progress}],
                    "state": "OPEN",
                }
            ]
            self.prs = [
                {
                    "number": 1585,
                    "title": "Salvaged work for #1578",
                    "url": "https://example.test/pull/1585",
                    "headRefName": "agent/issue-1578-salvage-wedge",
                    "baseRefName": "main",
                    "headRefOid": "sha-deadbeef",
                    "mergeStateStatus": "CLEAN",
                    "body": "Closes #1578\n\nTests: regression coverage added.",
                    "labels": [],
                    "isCrossRepository": False,
                    "state": "OPEN",
                }
            ]

        def remove_issue_label(self, number: int, label: str) -> bool:
            # Simulate a transient GitHub API failure on the label removal.
            return False

    fake_gh = FakeGitHubForOrphan()

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1578"]
    # Label write failed -- status must stay dispatched so the next pass
    # re-attempts rather than leaving the issue in a half-transitioned state.
    assert entry["status"] == "dispatched"

    events = state.get("events", [])
    advance_events = [e for e in events if e.get("kind") == "orphaned_worker_advanced_to_pr_open"]
    assert advance_events == [], "no advance event on label-write failure"
    drift_events = [
        e
        for e in events
        if e.get("kind") == "orphaned_worker_drift"
        and e["payload"].get("reason") == "dead_worker_unsafe_to_auto_reset"
    ]
    assert len(drift_events) == 1, "conservative drift must be emitted on failure"


def test_orphaned_worker_unreviewed_pr_with_rework_status_advances_not_resets(
    tmp_path: Path,
) -> None:
    """Issue #1128 rework after merge with #1109: an issue that could plausibly
    match both lanes -- ``last_decision`` is None (the #1128 condition) while
    ``pr_state.status`` is ``"rework_requested"`` (part of the #1109 condition) --
    must go through the #1128 advance-to-pr-open lane, NOT the #1109
    auto-reset-to-rework_requested lane.

    The ``if``/``else`` structure keys the #1109 lane on
    ``last_decision == "approved"``; a PR with no decision but a
    ``rework_requested`` status is an inconsistent state that the #1109 guard
    correctly rejects (no evidence an approved review dispatched this worker).
    The #1128 lane then advances it to ``pr-open`` so review can assess it,
    rather than guessing a re-dispatch.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    in_progress = config.labels.in_progress
    pr_open = config.labels.pr_open

    state = load_state(paths.state_file)
    state["issues"]["1578"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    # No ``decision`` key (last_decision will be None), but PR-state status
    # is ``rework_requested`` -- this is the "plausibly matches both" edge.
    state["prs"]["1585"] = {
        "status": "rework_requested",
        "reviewed_head_sha": None,
    }
    save_state(paths.state_file, state)

    class FakeGitHubForOrphan(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues = [
                {
                    "number": 1578,
                    "title": "Both-lanes edge",
                    "url": "https://example.test/issues/1578",
                    "body": "Unreviewed PR with rework_requested status",
                    "labels": [{"name": in_progress}],
                    "state": "OPEN",
                }
            ]
            self.prs = [
                {
                    "number": 1585,
                    "title": "Salvaged work for #1578",
                    "url": "https://example.test/pull/1585",
                    "headRefName": "agent/issue-1578-both-lanes-edge",
                    "baseRefName": "main",
                    "headRefOid": "sha-deadbeef",
                    "mergeStateStatus": "CLEAN",
                    "body": "Closes #1578\n\nTests: regression coverage added.",
                    "labels": [],
                    "isCrossRepository": False,
                    "state": "OPEN",
                }
            ]

    fake_gh = FakeGitHubForOrphan()

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["1578"]

    # Must advance to pr-open (#1128 lane), NOT reset to rework_requested
    # (#1109 lane).
    assert entry["status"] == PASSIVE_OPEN_STATUS, (
        f"expected open_passive (#1128 lane), got {entry['status']!r}"
    )

    events = state.get("events", [])
    advance_events = [e for e in events if e.get("kind") == "orphaned_worker_advanced_to_pr_open"]
    assert len(advance_events) == 1, "#1128 advance must fire"
    assert (
        advance_events[0]["payload"]["reason"]
        == "dead_worker_unsafe_to_auto_reset_open_unreviewed_pr"
    )

    # #1109 lane must NOT fire -- no recovered event, no clean_exit_no_op drift.
    recovered_events = [e for e in events if e.get("kind") == "orphaned_worker_recovered"]
    assert recovered_events == [], "#1109 auto-reset must not misfire on a None-decision PR"

    clean_exit_drifts = [
        e
        for e in events
        if e.get("kind") == "orphaned_worker_drift"
        and e["payload"].get("reason") == "dead_worker_clean_exit_no_op"
    ]
    assert clean_exit_drifts == [], "#1109 clean-exit-no-op drift must not misfire"

    # The label swap mirrors the #1128 lane.
    assert (1578, in_progress) in fake_gh.labels_removed
    assert (1578, pr_open) in fake_gh.labels_added
