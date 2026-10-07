"""Phantom live-worker dispatch: blocked and stale worker-outcome handling.

Continues ``test_charlie_work_dispatch_phantom.py`` (split for the file-size
ratchet); shared fakes live in ``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from _worktree_fixtures import (
    _init_bare_remote_and_clone,
    _init_repo,
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
from charlie_work.worktree import create_worktree
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def _dead_probe(monkeypatch: Any) -> None:
    """Install an all-dead host probe so liveness reads never hit the host PID table."""
    import dataclasses

    from charlie_work import host as host_pkg
    from charlie_work.host.fakes import FakeProcessProbe

    monkeypatch.setattr(
        host_pkg,
        "_ACTIVE",
        dataclasses.replace(host_pkg.current(), probe=FakeProcessProbe()),
    )


def _run_phantom_blocked_dispatch(
    tmp_path: Path,
    monkeypatch: Any,
    outcome_payload: dict[str, Any],
    *,
    with_commits: bool = True,
) -> tuple[Any, Path, FakeGitHub, Any]:
    """Dispatch over a phantom live worker (dead PID; local commits ahead of
    origin/main unless ``with_commits=False``) whose worktree carries
    ``outcome_payload``; returns ``(result, sidecar_path, fake_gh, paths)``."""
    from charlie_work.config import WORKER_OUTCOME_FILENAME

    # Issue #2262: the phantom lane's salvage probe runs against the app's
    # repo_root (``tmp_path`` here -- the worker's worktree lives in the
    # ``clone`` subdir). A real repo at tmp_path lets the probe reach the
    # conclusive ``no_commits`` verdict (branch ref provably absent) the
    # requeue path requires; on a non-git dir the probe is inconclusive and
    # the lane defers instead.
    _init_repo(tmp_path)
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    if with_commits:
        worktree_path, branch = _setup_completed_worktree(repo_root, 1453)
    else:
        branch = "agent/issue-1453"
        worktree_path = create_worktree(repo_root, branch, base_ref="origin/main").path
    (worktree_path / WORKER_OUTCOME_FILENAME).write_text(
        json.dumps(outcome_payload),
        encoding="utf-8",
    )

    recent_started_at = (datetime.now(UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree_path),
            prompt_path=str(worktree_path / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=8484,
            started_at=recent_started_at,
            log_path=str(tmp_path / "log"),
            error="probe_error",
            failure_kind="live_worker_redispatch_averted",
            process_start_time=5_678_901.0,
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)
    # Every liveness read — the post-launch result check and the sidecar's
    # PID through ``worker_fate.is_alive`` — goes through the one injected
    # host probe; the all-dead fake keeps the outcome off the host PID table.
    _dead_probe(monkeypatch)

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
                "branch": branch,
                "worktree_path": str(worktree_path),
                "prompt_path": "",
                "command": ["claude", "-p"],
                "pid": 8484,
                "started_at": recent_started_at,
                "log_path": str(tmp_path / "log"),
                "error": "probe_error",
                "failure_kind": "live_worker_redispatch_averted",
                "process_start_time": 5_678_901.0,
                "session_id": "test-session-1453",
            }
        ),
        encoding="utf-8",
    )

    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "branch_name": branch,
        "worker_pid": 8484,
        "worker_process_start_time": 5_678_901.0,
        "title": "Fix search",
        "url": "https://example.test/issues/123",
    }
    save_state(paths.state_file, seed)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)
    return result, sidecar_path, fake_gh, paths


def test_dispatch_phantom_live_worker_preserves_sidecar_for_blocked_outcome(
    tmp_path: Path, monkeypatch
) -> None:
    """B4 (wf-review-opus.md), rule 2: a phantom live worker with local
    commits AND a fresh ``blocked`` outcome must NOT have its sidecar reaped
    or its labels stripped/relabeled ``automated-ready``. Before the fix, a
    ``Blocked`` fate was neither ``Stranded`` nor ``PushedWithoutPr`` and
    ``declared_push`` was false, so control fell through to the reap-and-
    ``ready`` path: the sidecar the reaper lane keys salvage/escalation off
    was destroyed, and the issue was re-queued into the identical wall it
    had just declared itself blocked on. Preserving it here instead defers
    to the dead-session reaper lane, which already escalates a fresh
    ``Blocked`` fate (``workflow.py``'s ``fates.get(issue_number) ==
    Blocked`` branch) rather than redispatching.
    """
    result, sidecar_path, fake_gh, paths = _run_phantom_blocked_dispatch(
        tmp_path,
        monkeypatch,
        {
            "outcome": "blocked",
            "reason": "the task requires credentials this worker does not have",
            "push_succeeded": False,
            "pr_created": False,
        },
    )

    assert result.data["phantom_live_worker_count"] == 1, repr(result.data)
    assert sidecar_path.exists(), "Sidecar must not be reaped -- the reaper lane escalates Blocked"

    # Labels are NOT stripped and `ready` is NOT restored: redispatching would
    # re-queue the issue into the identical wall it declared itself blocked on.
    assert (123, "agent:in-progress") not in fake_gh.labels_removed
    assert (123, "automated-ready") not in fake_gh.labels_removed
    assert (123, "automated-ready") not in fake_gh.labels_added

    state = load_state(paths.state_file)
    preserve_events = [
        e
        for e in state.get("events", [])
        if e["kind"] == "session_failed_relabeled"
        and e["payload"]["issue_number"] == 123
        and e["payload"]["reason"] == "phantom_live_worker_declared_blocked_preserved"
    ]
    assert len(preserve_events) == 1
    payload = preserve_events[0]["payload"]
    assert payload["removed_labels"] == []
    assert payload["added_ready"] is False
    assert payload["worker_fate"] == "Blocked"


_PERMISSION_DENIAL_BLOCKED = {
    "outcome": "blocked",
    "detail": "Bash was denied. If you approve command execution, I can finish.",
    "push_succeeded": False,
    "pr_created": False,
}


def _phantom_reasons(paths: Any) -> list[str]:
    return [
        e["payload"]["reason"]
        for e in load_state(paths.state_file).get("events", [])
        if e["kind"] == "session_failed_relabeled" and e["payload"]["issue_number"] == 123
    ]


def test_dispatch_phantom_permission_denial_blocked_is_not_deferred_to_the_reaper_lane(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #2010: the reaper lane never escalates a permission-denial
    ``blocked`` outcome, so it is never preserved as ``declared_blocked``. But
    the exemption must not shadow the branch evidence either: a worker with
    local commits ahead is ``Stranded`` (the blocked claim is dropped and the
    fate re-resolved), so its sidecar and labels are PRESERVED for the reaper
    lane's salvage, exactly as on origin/main (wf-r5). Reaping it here would
    strand the unpublished commits. Positive control: the sibling test above,
    same bed with a genuine blocked reason, is preserved as declared_blocked;
    the no-commits sibling below is reaped-and-readied."""
    _result, sidecar_path, fake_gh, paths = _run_phantom_blocked_dispatch(
        tmp_path, monkeypatch, _PERMISSION_DENIAL_BLOCKED
    )

    reasons = _phantom_reasons(paths)
    assert "phantom_live_worker_declared_blocked_preserved" not in reasons
    assert reasons.count("phantom_live_worker_completed_work_preserved") == 1, reasons
    assert sidecar_path.exists(), "commits ahead: sidecar must be kept for salvage"
    assert (123, "agent:in-progress") not in fake_gh.labels_removed
    assert (123, "automated-ready") not in fake_gh.labels_added


def test_dispatch_phantom_permission_denial_blocked_without_commits_is_reaped(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #2010, no-commits half: an exempt permission-denial ``blocked``
    phantom worker with nothing ahead of origin/main has nothing to salvage, so
    it takes the ordinary reap-and-ready path (sidecar reaped, active label
    stripped, ``automated-ready`` restored)."""
    _result, sidecar_path, fake_gh, paths = _run_phantom_blocked_dispatch(
        tmp_path, monkeypatch, _PERMISSION_DENIAL_BLOCKED, with_commits=False
    )

    reasons = _phantom_reasons(paths)
    assert "phantom_live_worker_declared_blocked_preserved" not in reasons
    assert "phantom_live_worker_completed_work_preserved" not in reasons
    assert not sidecar_path.exists()
    assert (123, "agent:in-progress") in fake_gh.labels_removed


def test_dispatch_phantom_stale_outcome_emits_worker_evidence_stale(
    tmp_path: Path, monkeypatch
) -> None:
    """B6 (wf-r2-s6): the dispatch-time phantom-worker lane resolves a fate too.
    A ``.worker-outcome.json`` older than the phantom worker's start is ignored
    by rule 1 and must surface as a ``worker_evidence_stale`` warning. The lane
    resolves inside ``state_lock``, so the report happens after the lock is
    released -- this proves it is emitted at all, and once.
    """
    import os

    from charlie_work.config import WORKER_OUTCOME_FILENAME
    from charlie_work.instrumentation import query_events

    worktree_path = tmp_path / "wt"
    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome_path = worktree_path / WORKER_OUTCOME_FILENAME
    outcome_path.write_text(
        json.dumps({"push_succeeded": True, "pr_created": False}), encoding="utf-8"
    )
    # Written well before the sidecar's ``started_at`` (2026-08-10T11:15:39Z).
    before_start = datetime(2026, 8, 1, tzinfo=UTC).timestamp()
    os.utime(outcome_path, (before_start, before_start))

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
    _dead_probe(monkeypatch)

    config = OrchestratorConfig(worker=WorkerRoleConfig(harness="claude-code"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.pr_list = lambda: []
    _original_issue_view = fake_gh.issue_view

    def _patched_issue_view(number: int):
        issue = _original_issue_view(number)
        return {
            **issue,
            "labels": [{"name": "automated-ready"}, {"name": "agent:in-progress"}],
        }

    fake_gh.issue_view = _patched_issue_view

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "issue-123.claude.json").write_text(
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
    stale = query_events(paths.state_file, kind="worker_evidence_stale")
    assert len(stale) == 1
    assert stale[0]["level"] == "warning"
    raw = stale[0]["payload"]
    payload = json.loads(raw) if isinstance(raw, str) else raw
    assert payload["issue_number"] == 123
    assert payload["reason"] == "older_than_dispatch"
    assert payload["source"] == "worktree"
    assert load_state(paths.state_file)["issues"]["123"]["stale_evidence_reported"]
