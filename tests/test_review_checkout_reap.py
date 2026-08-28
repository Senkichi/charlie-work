"""Tests for the review_checkout_reap sweep (_reap_completed_review_checkouts / _reap_orphaned_review_checkouts in workflow.py), scatter-consolidated out of test_charlie_work.py (#1284) -- review_checkout_reap is the lane's working name, not a literal identifier since the production code has no 1:1 extracted module, following B7's name-the-lane precedent."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from _fakes_github import FakeGitHub
from _helpers import _init_git_repo
from charlie_work.config import OrchestratorConfig
from charlie_work.state import empty_state, load_state, save_state
from charlie_work.workflow import (
    _detect_and_handle_stalled_reviews,
    _reap_orphaned_review_checkouts,
)
from charlie_work.write_gate import WriteGate


# Issue #1264 (W6 PR2): every WriteGate the tests below construct must carry
# THAT test's own state_file as state_path -- WriteGate.save_state() writes
# to self.state_path, not to whatever path a converted function was also
# given, so a gate built with a different path would silently write to the
# wrong file while every assertion below keeps reading the real state_file.
def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


def test_reap_completed_review_checkouts_removes_checkout_once_reviewer_exited(
    tmp_path: Path,
) -> None:
    """Issue #397: once record_review has recorded a verdict
    (review_dispatch_completed) and the reviewer's own sidecar process is no
    longer alive, the isolated review checkout is reaped. Liveness must be
    checked via the sidecar in reviews_dir (iter_workers), since record_review
    already cleared state.json's reviewer_pid by this point."""
    from datetime import timedelta

    from charlie_work.workflow import _reap_completed_review_checkouts
    from charlie_work.worktree import create_review_checkout

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"

    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    checkout = create_review_checkout(repo_root, 200, head_sha, reviews_dir=reviews_dir)
    assert checkout.path.exists()

    # A sidecar recording a definitely-dead pid (record_review does not
    # delete the sidecar itself, only clears state.json's own pid fields).
    reviews_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "issue_number": 200,
        "branch": "agent/issue-20-fix",
        "worktree_path": str(checkout.path),
        "prompt_path": str(checkout.path / ".orchestrator-prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,
        "started_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "log_path": str(reviews_dir / "issue-200.claude.log"),
        "error": None,
        "process_start_time": 1.0,
    }
    (reviews_dir / "issue-200.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {},
                "prs": {
                    "200": {
                        "number": 200,
                        "review_dispatch_status": "review_dispatch_completed",
                        "reviewer_pid": None,
                        "reviewer_process_start_time": None,
                    }
                },
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    reaped = _reap_completed_review_checkouts(repo_root, reviews_dir, state_file)

    assert reaped == [200]
    assert not checkout.path.exists()


def test_reap_completed_review_checkouts_skips_while_reviewer_still_alive(
    tmp_path: Path,
) -> None:
    """A completed-verdict PR whose reviewer sidecar is still alive must NOT
    have its checkout removed out from under the exiting process.

    Uses this test process's own PID/start-time as the sidecar's recorded
    identity, so the real (non-monkeypatched) claude_code.is_worker_alive
    liveness+identity check reports it genuinely alive — matching how
    test_count_live_sessions_ghost_worker_pid_corroborated_by_state (same
    file) proves a "ghost" liveness case elsewhere in this suite.
    """
    from charlie_work.claude_code import _get_process_start_time
    from charlie_work.workflow import _reap_completed_review_checkouts
    from charlie_work.worktree import create_review_checkout

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"

    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    checkout = create_review_checkout(repo_root, 201, head_sha, reviews_dir=reviews_dir)

    reviews_dir.mkdir(parents=True, exist_ok=True)
    current_pid = os.getpid()
    sidecar = {
        "issue_number": 201,
        "branch": "agent/issue-21-fix",
        "worktree_path": str(checkout.path),
        "prompt_path": str(checkout.path / ".orchestrator-prompt.md"),
        "command": ["claude", "-p"],
        "pid": current_pid,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "log_path": str(reviews_dir / "issue-201.claude.log"),
        "error": None,
        "process_start_time": _get_process_start_time(current_pid),
    }
    (reviews_dir / "issue-201.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {},
                "prs": {
                    "201": {
                        "number": 201,
                        "review_dispatch_status": "review_dispatch_completed",
                        "reviewer_pid": None,
                        "reviewer_process_start_time": None,
                    }
                },
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    reaped = _reap_completed_review_checkouts(repo_root, reviews_dir, state_file)

    assert reaped == []
    assert checkout.path.exists()


def test_reap_orphaned_review_checkouts_clears_merged_pr_dispatch_state(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #494: the review-dispatch pass must reap checkouts and clear
    review-dispatch state for PRs that GitHub already reports as MERGED
    or CLOSED, regardless of the local claim status.
    """
    from charlie_work.state import empty_state
    from charlie_work.workflow import _reap_orphaned_review_checkouts

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    state_file = tmp_path / "state.json"

    config = OrchestratorConfig()
    state = empty_state()
    state["prs"]["100"] = {
        "number": 100,
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-20T00:00:00Z",
        "reviewer_pid": 12345,
        "reviewer_process_start_time": 1.0,
    }
    save_state(state_file, state)

    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 100,
            "title": "Fix #1",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-1-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "body": "Closes #1",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        }
    ]

    removed_calls: list[tuple[Path, int, Path | None]] = []

    def fake_remove_review_checkout(
        repo_root_arg: Path, pr_number: int, *, reviews_dir: Path | None = None
    ) -> bool:
        removed_calls.append((repo_root_arg, pr_number, reviews_dir))
        return True

    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", fake_remove_review_checkout
    )

    reaped = _reap_orphaned_review_checkouts(
        fake_gh, repo_root, reviews_dir, state_file, config, write_gate=_wg(state_file)
    )

    assert reaped == [100]
    assert len(removed_calls) == 1
    assert removed_calls[0][0] == repo_root
    assert removed_calls[0][1] == 100
    assert removed_calls[0][2] == reviews_dir

    new_state = load_state(state_file)
    assert new_state["prs"]["100"]["review_dispatch_status"] is None
    assert new_state["prs"]["100"]["review_dispatched_at"] is None
    assert new_state["prs"]["100"]["reviewer_pid"] is None
    assert new_state["prs"]["100"]["reviewer_process_start_time"] is None
    assert new_state["prs"]["100"]["status"] == "merged"


def test_reap_orphaned_review_checkouts_defers_while_reviewer_alive(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #504: a PR merged/closed externally while its reviewer is still
    alive must not have its review checkout removed or its dispatch claim
    cleared until the reviewer exits.

    Uses a sidecar in reviews_dir and monkeypatches WorkerView.is_alive to
    True/False so the test exercises the liveness gate without a real process.
    """
    from charlie_work.state import empty_state, load_state, save_state
    from charlie_work.workflow import _reap_orphaned_review_checkouts

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    state_file = tmp_path / "state.json"

    config = OrchestratorConfig()
    state = empty_state()
    state["prs"]["100"] = {
        "number": 100,
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-20T00:00:00Z",
        "reviewer_pid": 12345,
        "reviewer_process_start_time": 1.0,
    }
    save_state(state_file, state)

    sidecar = {
        "issue_number": 100,
        "branch": "agent/issue-100-fix",
        "worktree_path": str(reviews_dir / "pr-100"),
        "prompt_path": str(reviews_dir / "pr-100" / ".orchestrator-prompt.md"),
        "command": ["claude", "-p"],
        "pid": 12345,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "log_path": str(reviews_dir / "issue-100.claude.log"),
        "error": None,
        "process_start_time": 1.0,
        "adapter_kind": "claude-code",
    }
    (reviews_dir / "issue-100.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 100,
            "title": "Fix #1",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-100-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "body": "Closes #1",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        }
    ]

    removed_calls: list[tuple[Path, int, Path | None]] = []

    def fake_remove_review_checkout(
        repo_root_arg: Path, pr_number: int, *, reviews_dir: Path | None = None
    ) -> bool:
        removed_calls.append((repo_root_arg, pr_number, reviews_dir))
        return True

    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", fake_remove_review_checkout
    )

    # Live reviewer: defer the reap.
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: True)
    reaped = _reap_orphaned_review_checkouts(
        fake_gh, repo_root, reviews_dir, state_file, config, write_gate=_wg(state_file)
    )
    assert reaped == []
    assert removed_calls == []
    new_state = load_state(state_file)
    assert new_state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"

    # Dead reviewer: proceed with the reap.
    removed_calls.clear()
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: False)
    reaped = _reap_orphaned_review_checkouts(
        fake_gh, repo_root, reviews_dir, state_file, config, write_gate=_wg(state_file)
    )
    assert reaped == [100]
    assert len(removed_calls) == 1
    assert removed_calls[0][1] == 100
    new_state = load_state(state_file)
    assert new_state["prs"]["100"]["review_dispatch_status"] is None
    assert new_state["prs"]["100"]["status"] == "merged"


def test_reap_orphaned_review_checkouts_reaps_sidecar_stops_stalled_ping_pong(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue observed 07-22: reaping a merged PR's checkout without reaping
    its dead reviewer sidecar left the sidecar to resurrect as a phantom
    failed claim on every subsequent stalled sweep, which re-reaped it --
    an infinite ping-pong. The orphan sweep must delete the sidecar, and a
    following stalled-sweep pass must then see nothing to reap."""
    from charlie_work.state import empty_state
    from charlie_work.workflow import _reap_orphaned_review_checkouts

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    state_file = tmp_path / "state.json"

    config = OrchestratorConfig()
    state = empty_state()
    state["prs"]["100"] = {
        "number": 100,
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-20T00:00:00Z",
        "reviewer_pid": 12345,
        "reviewer_process_start_time": 1.0,
    }
    save_state(state_file, state)

    sidecar = {
        "issue_number": 100,
        "branch": "agent/issue-100-fix",
        "worktree_path": str(reviews_dir / "pr-100"),
        "prompt_path": str(reviews_dir / "pr-100" / ".orchestrator-prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,
        "started_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        "log_path": str(reviews_dir / "issue-100.claude.log"),
        "error": None,
        "process_start_time": 1.0,
        "adapter_kind": "claude-code",
    }
    sidecar_path = reviews_dir / "issue-100.claude.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 100,
            "title": "Fix #1",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-100-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "body": "Closes #1",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        }
    ]

    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", lambda *a, **k: True
    )
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: False)

    reaped = _reap_orphaned_review_checkouts(
        fake_gh, repo_root, reviews_dir, state_file, config, write_gate=_wg(state_file)
    )

    assert reaped == [100]
    assert not sidecar_path.exists()

    # A following stalled-sweep pass sees nothing left to reap: the sidecar
    # is gone (iter_workers yields no worker for PR 100) and the state's
    # review_dispatch_status is already None -- the ping-pong is dead.
    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    assert stalled == []
    state_after = load_state(state_file)
    assert not any(
        e.get("kind") == "review_dispatch_stalled" for e in state_after.get("events", [])
    )


def test_reap_orphaned_review_checkouts_warns_once_and_retries_on_checkout_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #526: when a merged PR's review checkout cannot be removed, the
    orphan sweep must not silently discard the failure, must not append a
    lifecycle-reaped event or claim the PR as reaped, and must emit only one
    warning per failure episode. A later successful retry clears the marker
    and records the reap."""
    from charlie_work.state import empty_state
    from charlie_work.workflow import _reap_orphaned_review_checkouts

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    checkout_dir = reviews_dir / "pr-100"
    checkout_dir.mkdir()
    state_file = tmp_path / "state.json"

    config = OrchestratorConfig()
    state = empty_state()
    state["prs"]["100"] = {
        "number": 100,
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-20T00:00:00Z",
        "reviewer_pid": 12345,
        "review_process_start_time": 1.0,
    }
    save_state(state_file, state)

    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 100,
            "title": "Fix #1",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-100-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "body": "Closes #1",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        }
    ]

    call_count = 0

    def fake_remove_review_checkout(
        repo_root_arg: Path, pr_number: int, *, reviews_dir: Path | None = None
    ) -> bool:
        nonlocal call_count
        call_count += 1
        return False

    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", fake_remove_review_checkout
    )
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: False)

    # First pass: failure is reported once, PR is not claimed as reaped.
    reaped = _reap_orphaned_review_checkouts(
        fake_gh, repo_root, reviews_dir, state_file, config, write_gate=_wg(state_file)
    )
    assert reaped == []
    assert call_count == 1
    state_after = load_state(state_file)
    assert state_after["prs"]["100"]["status"] == "merged"
    assert state_after["prs"]["100"]["review_dispatch_status"] is None
    assert state_after["prs"]["100"]["review_checkout_removal_warned"] is True
    assert not any(
        e.get("kind") == "review_dispatch_lifecycle_reaped" for e in state_after.get("events", [])
    )
    warning_events = [
        e
        for e in state_after.get("events", [])
        if e.get("kind") == "review_checkout_removal_failed"
    ]
    assert len(warning_events) == 1

    # Second pass: retry without re-emitting the warning.
    reaped = _reap_orphaned_review_checkouts(
        fake_gh, repo_root, reviews_dir, state_file, config, write_gate=_wg(state_file)
    )
    assert reaped == []
    assert call_count == 2
    state_after = load_state(state_file)
    warning_events = [
        e
        for e in state_after.get("events", [])
        if e.get("kind") == "review_checkout_removal_failed"
    ]
    assert len(warning_events) == 1

    # Third pass succeeds: the marker is cleared and the reap is recorded.
    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", lambda *a, **k: True
    )
    reaped = _reap_orphaned_review_checkouts(
        fake_gh, repo_root, reviews_dir, state_file, config, write_gate=_wg(state_file)
    )
    assert reaped == [100]
    assert call_count == 2  # the lambda does not increment the nested counter
    state_after = load_state(state_file)
    assert state_after["prs"]["100"].get("review_checkout_removal_warned") is None
    assert any(
        e.get("kind") == "review_dispatch_lifecycle_reaped" for e in state_after.get("events", [])
    )


# --- Issue #526: error-isolation hardening --------------------------------------


def test_reap_orphaned_review_checkouts_overwrites_stale_reviewing_status(
    monkeypatch, tmp_path: Path
) -> None:
    """A closed PR whose status was left as "reviewing" by the review pipeline
    must have its status overwritten to "closed" by the lifecycle reaper.

    Without this, the unclaimed-stalled sweep re-triggers every pass because
    it matches ``status is None and pr_state.get("status") == "reviewing"``,
    causing an infinite ping-pong with the lifecycle reaper.
    """
    from charlie_work.state import empty_state
    from charlie_work.workflow import _reap_orphaned_review_checkouts

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    state_file = tmp_path / "state.json"

    config = OrchestratorConfig()
    state = empty_state()
    state["prs"]["200"] = {
        "number": 200,
        "issue_number": 199,
        "status": "reviewing",
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-01T00:00:00Z",
        "reviewer_pid": 99999,
        "reviewer_process_start_time": 1.0,
        "prompt_path": str(tmp_path / "prompt.md"),
        "decision_path": str(tmp_path / "decision.json"),
    }
    save_state(state_file, state)

    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 200,
            "title": "Fix #199",
            "url": "https://example.test/pull/200",
            "headRefName": "agent/issue-199-fix",
            "baseRefName": "main",
            "headRefOid": "sha-200",
            "body": "Closes #199",
            "labels": [],
            "isCrossRepository": False,
            "state": "CLOSED",
        }
    ]

    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", lambda *a, **k: True
    )
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: False)

    reaped = _reap_orphaned_review_checkouts(
        fake_gh, repo_root, reviews_dir, state_file, config, write_gate=_wg(state_file)
    )
    assert reaped == [200]
    state_after = load_state(state_file)
    assert state_after["prs"]["200"]["status"] == "closed"
    assert state_after["prs"]["200"]["review_dispatch_status"] is None


def test_reap_orphaned_review_checkouts_aggregates_same_pass_events(
    tmp_path: Path,
) -> None:
    """Issue #525: multiple lifecycle-reaped PRs in one pass become one sweep event."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    state_file = tmp_path / "state.json"
    config = OrchestratorConfig()

    prs = [100, 200, 300]
    state = empty_state()
    for pr in prs:
        state["prs"][str(pr)] = {
            "number": pr,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": "2026-07-20T00:00:00Z",
            "reviewer_pid": 12345,
            "reviewer_process_start_time": 1.0,
        }
    save_state(state_file, state)

    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": pr,
            "title": f"Fix #{pr}",
            "url": f"https://example.test/pull/{pr}",
            "headRefName": f"agent/issue-{pr}-fix",
            "baseRefName": "main",
            "headRefOid": f"sha-{pr}",
            "body": f"Closes #{pr}",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        }
        for pr in prs
    ]

    reaped = _reap_orphaned_review_checkouts(
        fake_gh, repo_root, reviews_dir, state_file, config, write_gate=_wg(state_file)
    )

    assert reaped == prs
    state_after = load_state(state_file)
    sweep = [
        e
        for e in state_after["events"]
        if e.get("kind") == "review_dispatch_lifecycle_reaped_sweep"
    ]
    assert len(sweep) == 1
    assert sweep[0]["payload"]["count"] == len(prs)
    assert set(sweep[0]["payload"]["pr_numbers"]) == set(prs)
