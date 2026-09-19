"""Phantom live-worker detection and sidecar handling during dispatch.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the phantom-worker seam of ``dispatch()`` -- a live worker whose session
record is gone frees its slot and reaps or preserves the sidecar depending on
worktree/outcome state. Shared fakes and helpers in
``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path

from _fakes_github import FakeGitHub
from _worktree_fixtures import (
    _init_bare_remote_and_clone,
    _setup_completed_worktree,
)
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import (
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_dispatch_phantom_live_worker_frees_slot_and_reaps_sidecar(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #523: a live_worker_redispatch_averted result whose recorded PID is
    dead must not count as a live worker slot. The phantom slot is freed, the
    stale sidecar is reaped, active labels are stripped, ready is restored, and
    the issue is re-dispatchable on the next pass."""

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=4242,
            started_at="2026-07-02T00:00:00Z",
            log_path=str(tmp_path / "log"),
            error="probe_error",
            failure_kind="live_worker_redispatch_averted",
            process_start_time=1_234_567.0,
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)
    # Issue #523: the recorded PID is dead, so the result must not count as a
    # live worker slot. This also makes _worker_pid_alive return False so the
    # issue is selectable despite state.json recording a worker_pid.
    monkeypatch.setattr("charlie_work.workflow.is_pid_alive", lambda pid, start: False)
    # The sidecar-driven live-worker census (_issues_with_live_workers ->
    # worker.iter_workers().is_alive() -> claude_code.is_worker_alive) reads a
    # SEPARATE claude_code.is_pid_alive reference. Patch it too: whether the
    # fixture's PID reads as alive there depends on the host's current PID
    # table (process_utils.is_pid_alive treats OpenProcess/ERROR_ACCESS_DENIED
    # as indeterminate-so-alive by design), so leaving it unpatched makes the
    # test's outcome depend on host state, not the code under test.
    monkeypatch.setattr("charlie_work.claude_code.is_pid_alive", lambda pid, start: False)

    config = OrchestratorConfig(worker=WorkerRoleConfig(harness="claude-code"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.pr_list = lambda: []
    # Simulate stale label state: issue_list returns only ready (so the issue
    # is selectable), but issue_view still reports the stale in-progress label
    # that a previous dispatch left behind.
    _original_issue_view = fake_gh.issue_view

    def _patched_issue_view(number: int):
        issue = _original_issue_view(number)
        return {
            **issue,
            "labels": [
                {"name": "automated-ready"},
                {"name": "agent:in-progress"},
            ],
        }

    fake_gh.issue_view = _patched_issue_view

    # Plant a stale claude-code sidecar for the dead worker.
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sessions_dir / "issue-123.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 123,
                "branch": "agent/issue-123-fix-search",
                "worktree_path": str(tmp_path / "wt"),
                "prompt_path": "",
                "command": ["claude", "-p"],
                "pid": 4242,
                "started_at": "2026-07-02T00:00:00Z",
                "log_path": str(tmp_path / "log"),
                "error": "probe_error",
                "failure_kind": "live_worker_redispatch_averted",
                "process_start_time": 1_234_567.0,
                "session_id": "test-session-123",
            }
        ),
        encoding="utf-8",
    )

    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "branch_name": "agent/issue-123-fix-search",
        "worker_pid": 4242,
        "worker_process_start_time": 1_234_567.0,
        "title": "Fix search",
        "url": "https://example.test/issues/123",
    }
    save_state(paths.state_file, seed)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    # The phantom live worker is not counted as a live slot.
    assert result.data["live_worker_count"] == 0
    assert result.data["phantom_live_worker_count"] == 1
    assert result.data["attempted_count"] == 1
    # The stale sidecar is reaped.
    assert not sidecar_path.exists()
    # The slot is freed: worker_pid cleared, status no longer "dispatched".
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatch_failed"
    assert "worker_pid" not in state["issues"]["123"]
    assert "worker_process_start_time" not in state["issues"]["123"]
    # The stale in-progress label is removed; ready is not removed.
    assert (123, "agent:in-progress") in fake_gh.labels_removed
    assert (123, "automated-ready") not in fake_gh.labels_removed
    # A session_failed_relabeled attention event is emitted.
    assert any(
        event["kind"] == "session_failed_relabeled"
        and event["payload"]["issue_number"] == 123
        and event["payload"]["reason"] == "phantom_live_worker_pid_dead"
        for event in state.get("events", [])
    )

    # A second dispatch pass can still select the issue (slot is free).
    result2 = app.dispatch(limit=1)
    assert result2.data["attempted_count"] == 1
    assert result2.data["live_worker_count"] == 0


def test_dispatch_phantom_live_worker_no_active_labels_skips_relabel(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #523: a phantom live worker whose issue carries only a terminal
    label (no active labels) must still free the slot and reap the sidecar,
    but must NOT strip any label or add ``ready`` back -- the issue is
    terminal-only and spurious relabeling would resurrect it. A
    ``session_failed_relabeled`` event with empty ``removed_labels`` and
    ``added_ready=False`` is still recorded so the slot-free is observable."""

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=5353,
            started_at="2026-07-03T00:00:00Z",
            log_path=str(tmp_path / "log"),
            error="probe_error",
            failure_kind="live_worker_redispatch_averted",
            process_start_time=2_345_678.0,
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)
    monkeypatch.setattr("charlie_work.workflow.is_pid_alive", lambda pid, start: False)
    # The sidecar-driven live-worker census (_issues_with_live_workers ->
    # worker.iter_workers().is_alive() -> claude_code.is_worker_alive) reads a
    # SEPARATE claude_code.is_pid_alive reference. Patch it too: whether the
    # fixture's PID reads as alive there depends on the host's current PID
    # table (process_utils.is_pid_alive treats OpenProcess/ERROR_ACCESS_DENIED
    # as indeterminate-so-alive by design), so leaving it unpatched makes the
    # test's outcome depend on host state, not the code under test.
    monkeypatch.setattr("charlie_work.claude_code.is_pid_alive", lambda pid, start: False)

    config = OrchestratorConfig(worker=WorkerRoleConfig(harness="claude-code"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.pr_list = lambda: []
    # issue_list returns only ``automated-ready`` so the issue is selectable
    # (_is_dispatchable is label-only on the list view). issue_view -- the
    # full_issue the phantom router reads -- reports only a terminal label,
    # simulating an issue that was already escalated to a human while a stale
    # dispatched sidecar lingered.
    _original_issue_view = fake_gh.issue_view

    def _patched_issue_view(number: int):
        issue = _original_issue_view(number)
        return {
            **issue,
            "labels": [{"name": "agent:human-needed"}],
        }

    fake_gh.issue_view = _patched_issue_view

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sessions_dir / "issue-123.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 123,
                "branch": "agent/issue-123-fix-search",
                "worktree_path": str(tmp_path / "wt"),
                "prompt_path": "",
                "command": ["claude", "-p"],
                "pid": 5353,
                "started_at": "2026-07-03T00:00:00Z",
                "log_path": str(tmp_path / "log"),
                "error": "probe_error",
                "failure_kind": "live_worker_redispatch_averted",
                "process_start_time": 2_345_678.0,
                "session_id": "test-session-5353",
            }
        ),
        encoding="utf-8",
    )

    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "branch_name": "agent/issue-123-fix-search",
        "worker_pid": 5353,
        "worker_process_start_time": 2_345_678.0,
        "title": "Fix search",
        "url": "https://example.test/issues/123",
    }
    save_state(paths.state_file, seed)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    # The phantom live worker is not counted as a live slot.
    assert result.data["live_worker_count"] == 0
    assert result.data["phantom_live_worker_count"] == 1
    # The stale sidecar is reaped regardless of label state.
    assert not sidecar_path.exists()
    # The slot is freed: worker_pid cleared, status no longer "dispatched".
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatch_failed"
    assert "worker_pid" not in state["issues"]["123"]
    assert "worker_process_start_time" not in state["issues"]["123"]
    # No labels are touched: the terminal label is not stripped and ready is
    # not added back (the issue is terminal-only).
    assert fake_gh.labels_removed == []
    assert (123, "automated-ready") not in fake_gh.labels_added
    # A session_failed_relabeled event is still emitted with empty
    # removed_labels and added_ready=False so the slot-free is observable.
    relabel_events = [
        e
        for e in state.get("events", [])
        if e["kind"] == "session_failed_relabeled"
        and e["payload"]["issue_number"] == 123
        and e["payload"]["reason"] == "phantom_live_worker_pid_dead"
    ]
    assert len(relabel_events) == 1
    payload = relabel_events[0]["payload"]
    assert payload["removed_labels"] == []
    assert payload["added_ready"] is False
    assert payload["label_write_ok"] is True


def test_dispatch_phantom_live_worker_preserves_sidecar_for_completed_worktree(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #1122: a phantom live worker whose worktree is COMPLETED (clean,
    ahead of base) must NOT have its sidecar reaped or labels stripped. The
    sidecar is the key the reaper lane (``_classify_dead_sessions_and_update
    _throttle_state``) iterates over to salvage pushed-but-unpublished work via
    ``_attempt_salvage``. Reaping it here destroys the salvage path and
    escalates review-ready work to a human.
    """

    # Set up a real git repo with a completed worktree (commits ahead of base).
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 1122)

    # Derived, not hardcoded: this removes a latent stale-date hazard in the
    # fixture. It is NOT what fixes this test -- the phantom-live routing
    # under test does not gate on started_at at all. The actual gate that
    # made this test flip green/red across days is PID-liveness, not a time
    # window (see the claude_code.is_pid_alive patch below).
    recent_started_at = (datetime.now(UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree_path),
            prompt_path=str(worktree_path / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=6262,
            started_at=recent_started_at,
            log_path=str(tmp_path / "log"),
            error="probe_error",
            failure_kind="live_worker_redispatch_averted",
            process_start_time=3_456_789.0,
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)
    monkeypatch.setattr("charlie_work.workflow.is_pid_alive", lambda pid, start: False)
    # Issue #1122 fixture note: the sidecar's PID must also read as dead via
    # ``_issues_with_live_workers`` -> ``worker.iter_workers().is_alive()`` ->
    # ``claude_code.is_worker_alive`` -> ``claude_code.is_pid_alive`` -- a
    # SEPARATE module-level reference from ``charlie_work.workflow.is_pid_alive``
    # patched above. Without this, this fixture's specific PID (6262) can
    # spuriously read as "alive": ``process_utils.is_pid_alive`` treats
    # ``OpenProcess`` failing with ``ERROR_ACCESS_DENIED`` as indeterminate-so-
    # alive by design (process_utils.py:293-301), and on this host PID 6262
    # currently returns that error while sibling fixture PIDs (4242, 5353,
    # 7373) return ``ERROR_INVALID_PARAMETER`` (correctly read as dead) --
    # confirmed by direct ``OpenProcess`` probe, not assumed. That makes the
    # unpatched read host-PID-table-state dependent rather than deterministic,
    # which is what made this test pass on 2026-08-19 and fail on 2026-08-21
    # with no code change: it marks the issue live-dispatched and drops it out
    # of the dispatch candidate set before the phantom-live-worker routing
    # under test ever runs.
    monkeypatch.setattr("charlie_work.claude_code.is_pid_alive", lambda pid, start: False)

    config = OrchestratorConfig(worker=WorkerRoleConfig(harness="claude-code"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.pr_list = lambda: []
    _original_issue_view = fake_gh.issue_view

    def _patched_issue_view(number: int):
        issue = _original_issue_view(number)
        return {
            **issue,
            "labels": [
                {"name": "automated-ready"},
                {"name": "agent:in-progress"},
            ],
        }

    fake_gh.issue_view = _patched_issue_view

    # Plant a stale claude-code sidecar pointing at the completed worktree.
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sessions_dir / "issue-123.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 123,
                "branch": branch,
                "worktree_path": str(worktree_path),
                "prompt_path": "",
                "command": ["claude", "-p"],
                "pid": 6262,
                "started_at": recent_started_at,
                "log_path": str(tmp_path / "log"),
                "error": "probe_error",
                "failure_kind": "live_worker_redispatch_averted",
                "process_start_time": 3_456_789.0,
                "session_id": "test-session-1122",
            }
        ),
        encoding="utf-8",
    )

    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "branch_name": branch,
        "worker_pid": 6262,
        "worker_process_start_time": 3_456_789.0,
        "title": "Fix search",
        "url": "https://example.test/issues/123",
    }
    save_state(paths.state_file, seed)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    # The phantom live worker is detected but the sidecar is PRESERVED.
    assert result.data["phantom_live_worker_count"] == 1, repr(result.data)
    assert sidecar_path.exists(), "Sidecar must not be reaped so the reaper lane can salvage"

    # Labels are NOT stripped: the issue stays in-progress until salvage moves
    # it to pr_open, preventing re-dispatch into the occupied worktree.
    assert (123, "agent:in-progress") not in fake_gh.labels_removed
    assert (123, "automated-ready") not in fake_gh.labels_removed

    # A session_failed_relabeled event is emitted with the preservation reason.
    state = load_state(paths.state_file)
    preserve_events = [
        e
        for e in state.get("events", [])
        if e["kind"] == "session_failed_relabeled"
        and e["payload"]["issue_number"] == 123
        and e["payload"]["reason"] == "phantom_live_worker_completed_work_preserved"
    ]
    assert len(preserve_events) == 1
    payload = preserve_events[0]["payload"]
    assert payload["worktree_state"] == "completed"
    assert payload["removed_labels"] == []
    assert payload["added_ready"] is False


def test_dispatch_phantom_live_worker_preserves_sidecar_for_push_succeeded_outcome(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #1122: a phantom live worker whose ``.worker-outcome.json`` reports
    ``push_succeeded=true, pr_created=false`` must NOT have its sidecar reaped,
    even if the worktree inspection does not return COMPLETED (e.g. the worktree
    was reset or the commits are not visible locally). The outcome file is the
    durable signal that the worker pushed a branch the orchestrator can salvage.
    """
    from charlie_work.config import WORKER_OUTCOME_FILENAME

    # Use a non-git directory so inspect_worktree_state returns UNKNOWN, proving
    # the outcome-file check fires independently of the worktree-state check.
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir(parents=True, exist_ok=True)
    (worktree_path / WORKER_OUTCOME_FILENAME).write_text(
        json.dumps({"push_succeeded": True, "pr_created": False, "error": "gh unauthenticated"}),
        encoding="utf-8",
    )

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree_path),
            prompt_path=str(worktree_path / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=7373,
            started_at="2026-08-10T11:15:39Z",
            log_path=str(tmp_path / "log"),
            error="probe_error",
            failure_kind="live_worker_redispatch_averted",
            process_start_time=4_567_890.0,
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)
    monkeypatch.setattr("charlie_work.workflow.is_pid_alive", lambda pid, start: False)
    # See the sibling completed-worktree test above: the sidecar's PID must
    # also read as dead through claude_code.is_worker_alive's own
    # claude_code.is_pid_alive reference, not just workflow's, or the test's
    # outcome depends on the host's current PID table (ERROR_ACCESS_DENIED
    # from OpenProcess is treated as indeterminate-so-alive by design) instead
    # of the code under test.
    monkeypatch.setattr("charlie_work.claude_code.is_pid_alive", lambda pid, start: False)

    config = OrchestratorConfig(worker=WorkerRoleConfig(harness="claude-code"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.pr_list = lambda: []
    _original_issue_view = fake_gh.issue_view

    def _patched_issue_view(number: int):
        issue = _original_issue_view(number)
        return {
            **issue,
            "labels": [
                {"name": "automated-ready"},
                {"name": "agent:in-progress"},
            ],
        }

    fake_gh.issue_view = _patched_issue_view

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sessions_dir / "issue-123.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 123,
                "branch": "agent/issue-123-fix-search",
                "worktree_path": str(worktree_path),
                "prompt_path": "",
                "command": ["claude", "-p"],
                "pid": 7373,
                "started_at": "2026-08-10T11:15:39Z",
                "log_path": str(tmp_path / "log"),
                "error": "probe_error",
                "failure_kind": "live_worker_redispatch_averted",
                "process_start_time": 4_567_890.0,
                "session_id": "test-session-7373",
            }
        ),
        encoding="utf-8",
    )

    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "branch_name": "agent/issue-123-fix-search",
        "worker_pid": 7373,
        "worker_process_start_time": 4_567_890.0,
        "title": "Fix search",
        "url": "https://example.test/issues/123",
    }
    save_state(paths.state_file, seed)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    assert result.data["phantom_live_worker_count"] == 1, repr(result.data)
    assert sidecar_path.exists(), "Sidecar must not be reaped so the reaper lane can salvage"

    # Labels are NOT stripped.
    assert (123, "agent:in-progress") not in fake_gh.labels_removed

    state = load_state(paths.state_file)
    preserve_events = [
        e
        for e in state.get("events", [])
        if e["kind"] == "session_failed_relabeled"
        and e["payload"]["issue_number"] == 123
        and e["payload"]["reason"] == "phantom_live_worker_completed_work_preserved"
    ]
    assert len(preserve_events) == 1
    payload = preserve_events[0]["payload"]
    assert payload["reported_push"] is True
