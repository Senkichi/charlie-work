"""Dead rework-session immediate-escalation kinds: deterministic failure, unsafe worktree local commits, rework-branch conflict, and completed-worktree preservation.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

import sys
from pathlib import Path
from _dead_session_fixtures import (
    _write_dead_session_sidecar,
    _make_classify_state,
)
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from _worktree_fixtures import (
    _init_bare_remote_and_clone,
    _setup_completed_worktree,
)
from charlie_work.config import (
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.devin_shell import SessionRecord


def test_classify_dead_rework_session_deterministic_failure_kind_escalates_immediately(
    tmp_path: Path,
) -> None:
    """Issue #315 review finding 2b: a dead rework worker whose failure_kind is
    confirmed-deterministic (config.DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    e.g. worktree_unsafe) must escalate immediately, bypassing the redispatch
    cap entirely -- identical to the no-open-PR lane's immediate-escalation
    block (workflow.py ~line 936). That block is gated on
    `w.issue_number not in open_prs_by_issue`, so a rework worker (which
    always has an open PR) bypasses it entirely and would otherwise fall
    through to the ordinary cap-based path in _reap_restore_rework_requested.

    Mutation gate: dropping the `terminal_failure or` half of
    _reap_restore_rework_requested's `should_escalate` check makes this test
    fail (the issue would be restored to rework_requested since the
    redispatch history is empty and well under the cap).
    """
    import json
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
    # fake_gh.prs[0]["headRefOid"] defaults to "sha-abc123".

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-fix-search",
            "redispatch_at": [],  # nowhere near the cap
        }
        # Live request_changes decision matching the current head, so this
        # test isolates the deterministic-kind guard rather than finding 1's
        # gate.
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    # Issue #1362 Stage 1: the reader is now file-first, so the live
    # request_changes decision must exist on disk, not only in state.json.
    pr_decision_dir = paths.prs / "pr-456"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text("worktree contains local work, cannot reset\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-123.json"
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(paths.prs / "pr-456" / "rework-prompt.md"),
        command=("devin", "--prompt-file", "rework-prompt.md"),
        pid=None,  # Launch failure -- process never started
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="worktree creation failed: worktree contains local work",
        failure_kind="worktree_unsafe_shim_dirt",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "worktree_unsafe_shim_dirt"
    # Issue #1266: a deterministic failure_kind escalation is mechanical
    # (same _escalate_issue site as worker_death_loop/redispatch_cap_exceeded
    # in _reap_restore_rework_requested), so this lands agent:operator-queue.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added

    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 123]
    assert "session_failed_escalated" in event_kinds
    assert "rework_requeued" not in event_kinds


def test_classify_dead_rework_session_worktree_unsafe_local_commits_escalates_as_judgment(
    tmp_path: Path,
) -> None:
    """Issue #807: a dead rework worker whose failure_kind is
    ``worktree_unsafe_local_commits`` (genuine unpushed local commits on the
    worktree branch) must escalate immediately on the first occurrence — like
    the mechanical deterministic kinds — but with ``reason_class="judgment"`` so
    the de-escalation sweep never auto-clears it and the label lands
    ``agent:human-needed`` (not ``agent:operator-queue``).

    Mirrors ``test_classify_dead_rework_session_deterministic_failure_kind_escalates_immediately``
    (the mechanical sibling) and
    ``test_dispatch_worktree_unsafe_local_commits_escalates_as_judgment`` (the
    fresh-dispatch site), but covers the dead-rework-worker path in
    ``_reap_restore_rework_requested`` — the call site most relevant to the
    original #807 scenario.

    Mutation gate: reverting ``_reap_restore_rework_requested``'s label edge at
    the function tail from ``_escalation_edge("redispatch_escalated",
    reason_class)`` back to the hardcoded ``"mechanical"`` makes this test fail
    — the issue escalates with ``reason_class="judgment"`` in state but the
    label transition still lands ``agent:operator-queue``. Dropping the
    ``deterministic_judgment`` half of ``immediate_escalation`` also fails it
    (the issue is restored to ``rework_requested`` instead of escalating).
    """
    import json
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
    # fake_gh.prs[0]["headRefOid"] defaults to "sha-abc123".

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-fix-search",
            "redispatch_at": [],  # nowhere near the cap
        }
        # Live request_changes decision matching the current head, so this
        # test isolates the judgment-kind guard rather than finding 1's gate.
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    # Issue #1362 Stage 1: the reader is now file-first, so the live
    # request_changes decision must exist on disk, not only in state.json.
    pr_decision_dir = paths.prs / "pr-456"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text("worktree contains local commits, cannot reset\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-123.json"
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(paths.prs / "pr-456" / "rework-prompt.md"),
        command=("devin", "--prompt-file", "rework-prompt.md"),
        pid=None,  # Launch failure -- process never started
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="worktree creation failed: worktree contains local commits",
        failure_kind="worktree_unsafe_local_commits",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "worktree_unsafe_local_commits"
    # Issue #807: a deterministic judgment failure escalates as judgment, so
    # the label lands agent:human-needed, not agent:operator-queue.
    assert entry["reason_class"] == "judgment"
    assert (123, config.labels.human_needed) in fake_gh.labels_added
    assert (123, config.labels.operator_queue) not in fake_gh.labels_added
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added

    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 123]
    assert "session_failed_escalated" in event_kinds
    assert "rework_requeued" not in event_kinds


def test_classify_dead_rework_session_rework_branch_conflict_escalates_immediately(
    tmp_path: Path,
) -> None:
    """Issue #473: a dead rework worker whose failure_kind is rework_branch_conflict
    escalates immediately with that reason, bypassing the redispatch cap."""
    import json
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-conflict",
            "redispatch_at": [],
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    # Issue #1362 Stage 1: the reader is now file-first, so the live
    # request_changes decision must exist on disk, not only in state.json.
    pr_decision_dir = paths.prs / "pr-456"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text(
        "rework branch agent/issue-123-conflict conflicts with base origin/main; "
        "conflicted paths: file.txt\n",
        encoding="utf-8",
    )

    sidecar_path = sessions_dir / "issue-123.json"
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-conflict",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(paths.prs / "pr-456" / "rework-prompt.md"),
        command=("devin", "--prompt-file", "rework-prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="rework branch conflicts with origin/main; conflicted paths: file.txt",
        failure_kind="rework_branch_conflict",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_branch_conflict"
    # Issue #1266: same mechanical _escalate_issue site as the
    # worktree_unsafe/worker_death_loop cases above -> operator_queue.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added

    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 123]
    assert "session_failed_escalated" in event_kinds
    assert "rework_requeued" not in event_kinds


def test_classify_dead_rework_session_completed_worktree_not_rolled_back(
    tmp_path: Path,
) -> None:
    """LOW (issue #315 review): a rework worker that actually finished its
    work (worktree ahead of base and clean -- is_completed=True) must never be
    rolled back to rework_requested, even if this reap pass's PR-list
    snapshot (fetched once, at the top of
    _classify_dead_sessions_and_update_throttle_state) hasn't caught up to a
    fresh push yet and still shows the pre-rework head that matches a live
    request_changes decision. has_request_changes alone cannot catch this --
    it would look identical to a genuine, never-pushed rework -- so the
    open-PR branch must ALSO consult is_completed (issue #315 finding 1's
    second half).

    Mutation gate: removing the `if not is_completed:` guard around the
    _reap_restore_rework_requested call in the open-PR dead-session branch
    makes this test fail (the stale PR-list snapshot would incorrectly
    trigger a rollback to rework_requested).
    """
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 315)
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 315, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 315,
            "title": "Test issue",
            "url": "https://example.test/issues/315",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    # Open PR whose PR-list snapshot still shows the STALE pre-rework head --
    # simulating the race where the worker's fresh push hasn't been reflected
    # in the PR-list fetched at the top of this reap pass yet.
    gh.prs = [
        {
            "number": 900,
            "title": "Fix #315",
            "headRefName": branch,
            "headRefOid": "sha-stale-snapshot",
            "isCrossRepository": False,
            "body": "Closes #315",
            "labels": [],
            "state": "OPEN",
        }
    ]

    with state_lock(state_file):
        state = load_state(state_file)
        state["issues"]["315"] = {
            "number": 315,
            "status": "dispatched",
            "branch_name": branch,
            "worker_pid": 12345,
            "worker_process_start_time": 1111111111.0,
        }
        # This decision is still "live" against the STALE snapshot head --
        # exactly what would make has_request_changes incorrectly True if
        # is_completed weren't consulted.
        state["prs"]["900"] = {
            "number": 900,
            "issue_number": 315,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-stale-snapshot",
        }
        save_state(state_file, state)

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    state = load_state(state_file)
    entry = state["issues"]["315"]
    assert entry["status"] != "rework_requested"
    assert (315, config.labels.needs_rework) not in gh.labels_added
