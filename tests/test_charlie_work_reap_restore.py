"""Reap-restore dead-worker recovery: startup-death flagging, stalled real-PID vs long-runtime discrimination, and rework-requested stranded-commit salvage.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import (
    _init_repo_with_remote_inline,
    _wg,
)
from charlie_work.config import (
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.devin_shell import SessionRecord


def test_reap_restore_sets_startup_death_flag(
    tmp_path: Path,
) -> None:
    """Issue #1106: _reap_restore_rework_requested must record the
    startup-death classification in the PR state so the janitor gate can
    consult it on the next pass.

    Uses a ``launch_failed`` sidecar (pid=None, error set) — the simplest
    startup-death signature.
    """
    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]

    now = datetime.now(UTC)

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-fix-search",
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
    log_path.write_text("Refusing to run in an untrusted workspace\n", encoding="utf-8")
    sidecar_path = sessions_dir / "issue-123.json"
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(paths.prs / "pr-456" / "rework-prompt.md"),
        command=("devin", "--prompt-file", "rework-prompt.md"),
        pid=None,  # launch-failure sidecar
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="devin launch failed: untrusted workspace",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    state = load_state(paths.state_file)
    pr_state = state["prs"]["456"]
    # The startup-death flag must be set for a launch_failed session.
    assert pr_state["last_rework_was_startup_death"] is True
    assert pr_state["last_rework_failure_kind"] == "launch_failed"
    # The issue must have been restored to rework_requested.
    assert state["issues"]["123"]["status"] == "rework_requested"


def test_reap_restore_startup_death_stalled_real_pid_under_classification_delay(
    tmp_path: Path,
) -> None:
    """Issue #1106 regression: the realistic calibration incident shape.

    The 2026-08-08 incident was NOT a ``launch_failed`` (pid=None) — the Devin
    CLI launched, got a real PID, printed "Refusing to run in an untrusted
    workspace", and exited within seconds.  That is a ``stalled`` session with
    a real PID.  The orchestrator's classification pass may not run until
    minutes later (bounded by the polling interval), so the startup-death
    threshold must be checked against a signal bounded by the CLI's *actual
    death time* (log mtime), not ``runtime_seconds()`` (``now - started_at``,
    which measures elapsed time until classification).

    This test seeds a ``stalled`` sidecar with a real PID whose ``started_at``
    is 300 seconds in the past (simulating classification latency) but whose
    log file mtime is only 5 seconds after ``started_at`` (the CLI died at 5s).
    With the death-bounded runtime the session is a 5-second startup death
    (< 60s threshold); with the old ``runtime_seconds()`` it would be a
    300-second stall (> 60s) and the exemption would be silently defeated.

    Mutation gate: reverting the call site in ``_reap_restore_rework_requested``
    back to ``worker.runtime_seconds()`` makes this test fail (the flag stays
    False instead of being set True).
    """
    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.worker import WorkerView
    from charlie_work.workflow import _reap_restore_rework_requested

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]

    now = datetime.now(UTC)
    # The CLI started 300s ago and died 5s later (at 295s ago).  The
    # classification pass runs "now" — 300s after start, well past the 60s
    # threshold if measured against the wall clock.
    started_at = now - timedelta(seconds=300)
    death_at = started_at + timedelta(seconds=5)

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": started_at.timestamp(),
            "branch_name": "agent/issue-123-fix-search",
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

    # Write the log file with the calibration incident's refusal message and
    # freeze its mtime at the death moment (5s after start).
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text("Refusing to run in an untrusted workspace\n", encoding="utf-8")
    death_ts = death_at.timestamp()
    os.utime(log_path, (death_ts, death_ts))

    worker = WorkerView(
        adapter_kind="devin",
        issue_number=123,
        repo_key="",
        pid=99999,  # real PID — the CLI launched before dying
        started_at=started_at.isoformat().replace("+00:00", "Z"),
        process_start_time=started_at.timestamp(),
        log_path=str(log_path),
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        error=None,  # not a launch failure — the CLI ran and exited
        failure_kind=None,
        reclaimed=None,
        branch="agent/issue-123-fix-search",
    )

    open_prs_by_issue = {
        123: [
            {
                "number": 456,
                "title": "Fix #123: search",
                "headRefName": "agent/issue-123-fix-search",
                "headRefOid": "sha-abc123",
                "state": "OPEN",
            }
        ]
    }

    _reap_restore_rework_requested(
        paths.state_file,
        fake_gh,
        config,
        open_prs_by_issue,
        worker,
        failure_kind="stalled",
        write_gate=_wg(paths.state_file),
    )

    state = load_state(paths.state_file)
    pr_state = state["prs"]["456"]
    # The death-bounded runtime is ~5s (< 60s threshold) → startup death,
    # even though 300s elapsed between start and classification.
    assert pr_state["last_rework_was_startup_death"] is True
    assert pr_state["last_rework_failure_kind"] == "stalled"
    # The issue must have been restored to rework_requested (not escalated).
    assert state["issues"]["123"]["status"] == "rework_requested"


def test_reap_restore_stalled_long_runtime_not_startup_death(
    tmp_path: Path,
) -> None:
    """Issue #1106 negative case: a ``stalled`` session whose log shows the
    CLI genuinely ran for minutes must NOT be classified as a startup death,
    even when the classification pass runs much later.

    The log mtime is 200s after ``started_at`` (the worker ran for 200s before
    dying), well above the 60s threshold.  The death-bounded runtime correctly
    exceeds the threshold, so the cap counters should count this session.
    """
    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.worker import WorkerView
    from charlie_work.workflow import _reap_restore_rework_requested

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]

    now = datetime.now(UTC)
    started_at = now - timedelta(seconds=600)
    death_at = started_at + timedelta(seconds=200)

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": started_at.timestamp(),
            "branch_name": "agent/issue-123-fix-search",
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
    log_path.write_text("worker ran for a while then stalled\n", encoding="utf-8")
    death_ts = death_at.timestamp()
    os.utime(log_path, (death_ts, death_ts))

    worker = WorkerView(
        adapter_kind="devin",
        issue_number=123,
        repo_key="",
        pid=99999,
        started_at=started_at.isoformat().replace("+00:00", "Z"),
        process_start_time=started_at.timestamp(),
        log_path=str(log_path),
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        error=None,
        failure_kind=None,
        reclaimed=None,
        branch="agent/issue-123-fix-search",
    )

    open_prs_by_issue = {
        123: [
            {
                "number": 456,
                "title": "Fix #123: search",
                "headRefName": "agent/issue-123-fix-search",
                "headRefOid": "sha-abc123",
                "state": "OPEN",
            }
        ]
    }

    _reap_restore_rework_requested(
        paths.state_file,
        fake_gh,
        config,
        open_prs_by_issue,
        worker,
        failure_kind="stalled",
        write_gate=_wg(paths.state_file),
    )

    state = load_state(paths.state_file)
    pr_state = state["prs"]["456"]
    # 200s death-bounded runtime > 60s threshold → NOT a startup death.
    assert pr_state["last_rework_was_startup_death"] is False
    assert pr_state["last_rework_failure_kind"] == "stalled"


def test_reap_restore_rework_requested_salvages_stranded_commits(
    tmp_path: Path,
) -> None:
    """Issue #1239 Path 1: ``_reap_restore_rework_requested`` must
    salvage-push stranded commits from a dead rework worker's worktree
    BEFORE counting a death.  When the push succeeds, the issue resets to
    ``rework_requested`` WITHOUT recording a death — the PR head moved
    past the request_changes verdict, so the next ``dispatch_rework`` pass
    routes to review instead of re-dispatching into the same tail-death.
    """
    import json
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.paths import resolved_layout
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.worker import WorkerView
    from charlie_work.workflow import _reap_restore_rework_requested
    from charlie_work.worktree import push_branch, worktree_path_for_branch

    remote, repo_root = _init_repo_with_remote_inline(tmp_path)
    branch = "agent/issue-123-fix-search"

    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )

    run(["git", "branch", branch])
    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
        worker=WorkerRoleConfig(harness="command"),
    )
    layout = resolved_layout(config, repo_root)
    wt_path = worktree_path_for_branch(repo_root, branch, layout.worktrees)
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", str(wt_path), branch])

    # Push the branch so the remote branch exists at the PR head sha.
    ok, error = push_branch(repo_root, branch, worktree_path=wt_path)
    assert ok, error
    pr_head_sha = run(["git", "rev-parse", branch]).stdout.strip()

    # Add a stranded commit to the worktree.
    (wt_path / "fix.txt").write_text("fixed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "fix.txt"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "completed rework (died before push)"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )
    local_tip = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    paths = runtime_paths(repo_root, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
    fake_gh.prs[0]["headRefOid"] = pr_head_sha

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": branch,
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": pr_head_sha,
        }
        save_state(paths.state_file, state)

    # Issue #1362 Stage 1: the reader is file-first, so the live
    # request_changes decision must exist on disk.
    pr_decision_dir = paths.prs / "pr-456"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": pr_head_sha}),
        encoding="utf-8",
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text("worker died\n", encoding="utf-8")

    worker = WorkerView(
        adapter_kind="devin",
        issue_number=123,
        repo_key="",
        pid=99999,  # non-existent PID — is_alive() returns False
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        process_start_time=1234567890.0,
        log_path=str(log_path),
        worktree_path=str(wt_path),
        error=None,
        failure_kind="worker_died",
        reclaimed=None,
        branch=branch,
    )

    open_prs_by_issue = {123: [fake_gh.prs[0]]}

    _reap_restore_rework_requested(
        paths.state_file,
        fake_gh,
        config,
        open_prs_by_issue,
        worker,
        failure_kind="worker_died",
        repo_root=repo_root,
        write_gate=_wg(paths.state_file),
    )

    # The remote branch head must have advanced to the worktree's local tip.
    remote_sha = subprocess.run(
        ["git", "rev-parse", branch],
        cwd=remote,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert remote_sha == local_tip

    state = load_state(paths.state_file)
    # The issue must be reset to rework_requested (NOT escalated).
    assert state["issues"]["123"]["status"] == "rework_requested"
    # worker_death_at must NOT have been extended (no new death recorded).
    worker_death_at = state["issues"]["123"].get("worker_death_at", [])
    assert len(worker_death_at) == 0

    # An event with kind "rework_stranded_commits_salvaged" must exist.
    events = state.get("events", [])
    salvage_events = [e for e in events if e.get("kind") == "rework_stranded_commits_salvaged"]
    assert len(salvage_events) >= 1


def test_reap_restore_rework_requested_skips_salvage_when_status_not_dispatched(
    tmp_path: Path,
) -> None:
    """Issue #1239 round-2: workers are discovered from sidecar files,
    decoupled from state.json, so by the time ``_reap_restore_rework_requested``
    runs the issue's status may have already moved off ``dispatched`` (e.g. a
    concurrent loop pass re-dispatched or escalated).  In that case the
    salvage push to the shared origin remote MUST NOT be attempted — an
    unaudited push for a stale/no-longer-dispatched issue leaves no event trail
    if it succeeds.  A fresh ``status == "dispatched"`` precondition check
    (short state_lock scope, before computing the review decision and before
    any network push) gates the whole salvage path.
    """
    import json
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.paths import resolved_layout
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.worker import WorkerView
    from charlie_work.workflow import _reap_restore_rework_requested
    from charlie_work.worktree import push_branch, worktree_path_for_branch

    remote, repo_root = _init_repo_with_remote_inline(tmp_path)
    branch = "agent/issue-124-fix-search"

    run = lambda args: subprocess.run(  # noqa: E731
        args, cwd=repo_root, check=True, capture_output=True, text=True
    )

    run(["git", "branch", branch])
    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
        worker=WorkerRoleConfig(harness="command"),
    )
    layout = resolved_layout(config, repo_root)
    wt_path = worktree_path_for_branch(repo_root, branch, layout.worktrees)
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", str(wt_path), branch])

    # Push the branch so the remote branch exists at the PR head sha.
    ok, error = push_branch(repo_root, branch, worktree_path=wt_path)
    assert ok, error
    pr_head_sha = run(["git", "rev-parse", branch]).stdout.strip()

    # Add a stranded commit to the worktree — the salvage WOULD push this if
    # the precondition check were absent.
    (wt_path / "fix.txt").write_text("fixed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "fix.txt"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "completed rework (died before push)"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )

    paths = runtime_paths(repo_root, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
    fake_gh.prs[0]["headRefOid"] = pr_head_sha

    # The issue's status has ALREADY moved off "dispatched" — a concurrent
    # loop pass re-dispatched it to rework_requested.  This is the
    # sidecar/state.json decoupling the round-2 review flagged.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "rework_requested",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": branch,
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": pr_head_sha,
        }
        save_state(paths.state_file, state)

    # A LIVE request_changes verdict on disk — without the precondition check
    # the function would proceed past has_request_changes and attempt the push.
    pr_decision_dir = paths.prs / "pr-456"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": pr_head_sha}),
        encoding="utf-8",
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text("worker died\n", encoding="utf-8")

    worker = WorkerView(
        adapter_kind="devin",
        issue_number=123,
        repo_key="",
        pid=99999,  # non-existent PID — is_alive() returns False
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        process_start_time=1234567890.0,
        log_path=str(log_path),
        worktree_path=str(wt_path),
        error=None,
        failure_kind="worker_died",
        reclaimed=None,
        branch=branch,
    )

    open_prs_by_issue = {123: [fake_gh.prs[0]]}

    _reap_restore_rework_requested(
        paths.state_file,
        fake_gh,
        config,
        open_prs_by_issue,
        worker,
        failure_kind="worker_died",
        repo_root=repo_root,
        write_gate=_wg(paths.state_file),
    )

    # The remote branch head MUST NOT have advanced — no salvage push.
    remote_sha = subprocess.run(
        ["git", "rev-parse", branch],
        cwd=remote,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert remote_sha == pr_head_sha, (
        f"salvage push attempted for a non-dispatched issue: "
        f"remote {remote_sha} != pr head {pr_head_sha}"
    )

    # The issue status must be unchanged (still rework_requested, not reset).
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"

    # No salvage event must have been recorded.
    events = state.get("events", [])
    salvage_events = [e for e in events if e.get("kind") == "rework_stranded_commits_salvaged"]
    assert len(salvage_events) == 0
