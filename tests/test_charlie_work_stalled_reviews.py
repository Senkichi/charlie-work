"""``_detect_and_handle_stalled_reviews``: review-checkout removal, provider-throttle backoff, unclaimed-packet reap, and same-pass event aggregation.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from _helpers import _init_git_repo
from _review_fixtures import (
    _dispatch_reviews_app,
    _fake_claude_worker_record,
    _make_dead_review_sidecar,
    _write_review_packet,
)
from _rework_dispatch_fixtures import _wg
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.state import empty_state, load_state, save_state, state_lock
from charlie_work.workflow import _detect_and_handle_stalled_reviews
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_detect_and_handle_stalled_reviews_removes_review_checkout(tmp_path: Path) -> None:
    """Issue #397: a reaped stale-claim review must tear down that PR's
    isolated review checkout, not just free the state.json claim."""
    from datetime import timedelta

    from charlie_work.workflow import _detect_and_handle_stalled_reviews
    from charlie_work.worktree import create_review_checkout

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))

    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    checkout = create_review_checkout(repo_root, 100, head_sha, reviews_dir=reviews_dir)
    assert checkout.path.exists()

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["100"] = {
            "number": 100,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old_dispatched,
            "reviewer_pid": 999999999,  # not a real live pid
            "reviewer_process_start_time": 1.0,
        }
        save_state(state_file, state)

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    assert any(entry.get("pr") == 100 for entry in stalled)
    assert not checkout.path.exists()
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert str(checkout.path) not in result.stdout


def test_detect_and_handle_stalled_reviews_backs_off_on_provider_throttle_in_log(
    tmp_path: Path,
) -> None:
    """A dead reviewer whose own log shows a provider throttle signature
    (e.g. Claude Code CLI's "You've hit your session limit ...") must set the
    global reviewer-quota cooldown and roll back the claim, not mark it
    review_dispatch_failed. Marking it failed lets the next dispatch_reviews
    pass relaunch straight into the same limit -- job-cannon PRs #1342,
    #1343, #1344, #1346 hot-looped for 5.5-20+ hours this way on 2026-07-21
    before this reap path also learned to log-tail classify."""
    from datetime import timedelta

    from charlie_work.state import load_state as _load_state

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))

    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text(
        "You've hit your session limit · resets 4:40pm (America/Los_Angeles)\n",
        encoding="utf-8",
    )
    old_started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar = {
        "issue_number": 100,
        "branch": "agent/issue-10-fix",
        "worktree_path": str(tmp_path / "worktrees" / "issue-100"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,  # not a real live pid
        "started_at": old_started,
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    (reviews_dir / "issue-100.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["100"] = {
            "number": 100,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old_started,
            "reviewer_pid": 999999999,
            "reviewer_process_start_time": 1.0,
        }
        save_state(state_file, state)

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    assert any(
        entry.get("pr") == 100 and entry.get("reason") == "provider_throttled" for entry in stalled
    )

    state = _load_state(state_file)
    pr_state = state["prs"]["100"]
    # Rolled back, not failed -- the claim is immediately re-dispatchable
    # once the reviewer-quota gate clears.
    assert pr_state.get("review_dispatch_status") is None
    assert pr_state.get("reviewer_pid") is None

    quota = state.get("reviewer_quota", {})
    assert quota.get("throttled_until")
    assert quota.get("probe_after")
    assert any(
        event.get("kind") == "review_dispatch_stalled"
        and event.get("payload", {}).get("reason") == "provider_throttled"
        for event in state.get("events", [])
    )


def test_detect_and_handle_stalled_reviews_suppresses_backoff_after_probe_recovery(
    tmp_path: Path,
) -> None:
    """Issue #662: a dead reviewer whose log shows a throttle signature must
    NOT re-poison reviewer_quota when a green flat-interval probe already
    cleared the throttle AFTER the reviewer died. The throttle signature in a
    dead session's log tail is frozen at death time; re-applying backoff from
    it would anchor throttled_until/probe_after to "now" rather than the
    original death time, delaying the next dispatch by up to one probe cycle
    even though the quota window is open. The claim is still rolled back and
    the sidecar reaped -- the reviewer is dead regardless, and with the quota
    recovered the PR should be immediately re-dispatchable.
    """
    from datetime import timedelta

    from charlie_work.state import load_state as _load_state

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))

    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text(
        "You've hit your session limit · resets 4:40pm (America/Los_Angeles)\n",
        encoding="utf-8",
    )
    # Pin the log mtime to 10 minutes ago -- the reviewer died then, and a
    # green probe cleared the quota 1 minute ago (after death).
    death_dt = datetime.now(UTC) - timedelta(minutes=10)
    cleared_dt = datetime.now(UTC) - timedelta(minutes=1)
    os.utime(log_path, (death_dt.timestamp(), death_dt.timestamp()))

    old_started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar = {
        "issue_number": 100,
        "branch": "agent/issue-10-fix",
        "worktree_path": str(tmp_path / "worktrees" / "issue-100"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,
        "started_at": old_started,
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    (reviews_dir / "issue-100.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    cleared_iso = cleared_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["100"] = {
            "number": 100,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old_started,
            "reviewer_pid": 999999999,
            "reviewer_process_start_time": 1.0,
        }
        # Simulate a green probe that cleared the quota after the reviewer died.
        state["reviewer_quota"] = {
            "consecutive_probe_failures": 0,
            "last_probe_cleared_at": cleared_iso,
        }
        save_state(state_file, state)

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    assert any(
        entry.get("pr") == 100 and entry.get("reason") == "provider_throttled" for entry in stalled
    )

    state = _load_state(state_file)
    pr_state = state["prs"]["100"]
    # Rolled back, not failed -- immediately re-dispatchable (quota is open).
    assert pr_state.get("review_dispatch_status") is None
    assert pr_state.get("reviewer_pid") is None

    quota = state.get("reviewer_quota", {})
    # Backoff was suppressed: no throttled_until / probe_after re-poisoning.
    assert not quota.get("throttled_until")
    assert not quota.get("probe_after")
    # The recovery marker must survive the sweep unchanged.
    assert quota.get("last_probe_cleared_at") == cleared_iso

    # The event must record the suppression for observability.
    assert any(
        event.get("kind") == "review_dispatch_stalled"
        and event.get("payload", {}).get("reason") == "provider_throttled"
        and event.get("payload", {}).get("backoff_suppressed") is True
        for event in state.get("events", [])
    )
    # The sidecar was reaped (one-shot, no re-entry next sweep).
    assert not (reviews_dir / "issue-100.claude.json").exists()


def test_detect_and_handle_stalled_reviews_applies_backoff_when_probe_predates_death(
    tmp_path: Path,
) -> None:
    """Issue #662 control: when the green probe cleared BEFORE the reviewer
    died, the throttle signature is fresh evidence the quota closed again and
    backoff must still be applied. Only a probe recovery that post-dates the
    death suppresses backoff.
    """
    from datetime import timedelta

    from charlie_work.state import load_state as _load_state

    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))

    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text(
        "You've hit your session limit · resets 4:40pm (America/Los_Angeles)\n",
        encoding="utf-8",
    )
    # The reviewer died 1 minute ago; the probe cleared 10 minutes ago (before
    # death) -- the throttle signature is fresh, backoff must engage.
    death_dt = datetime.now(UTC) - timedelta(minutes=1)
    cleared_dt = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (death_dt.timestamp(), death_dt.timestamp()))

    old_started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar = {
        "issue_number": 100,
        "branch": "agent/issue-10-fix",
        "worktree_path": str(tmp_path / "worktrees" / "issue-100"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,
        "started_at": old_started,
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    (reviews_dir / "issue-100.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")

    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    cleared_iso = cleared_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["100"] = {
            "number": 100,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old_started,
            "reviewer_pid": 999999999,
            "reviewer_process_start_time": 1.0,
        }
        state["reviewer_quota"] = {
            "consecutive_probe_failures": 0,
            "last_probe_cleared_at": cleared_iso,
        }
        save_state(state_file, state)

    _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    state = _load_state(state_file)
    quota = state.get("reviewer_quota", {})
    # Backoff applied: throttled_until / probe_after are set.
    assert quota.get("throttled_until")
    assert quota.get("probe_after")
    assert any(
        event.get("kind") == "review_dispatch_stalled"
        and event.get("payload", {}).get("reason") == "provider_throttled"
        and event.get("payload", {}).get("backoff_suppressed") is False
        for event in state.get("events", [])
    )


def test_detect_and_handle_stalled_reviews_reaps_unclaimed_reviewing_packet(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #487: a reviewing PR that was never claimed/dispatched is reaped
    and then re-dispatched once its packet is past the stale-claim timeout."""
    from datetime import timedelta

    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")

    # Age the packet so the unclaimed safety net triggers on the next pass.
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-100"
    prompt_path = pr_dir / "review-prompt.md"
    old_mtime = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
    os.utime(prompt_path, (old_mtime, old_mtime))

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"] = {
            "number": 100,
            "issue_number": 10,
            "status": "reviewing",
            "prompt_path": str(prompt_path),
            "decision_path": str(pr_dir / "review-decision.json"),
        }
        save_state(app.paths.state_file, state)

    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(100, "agent/issue-10-fix")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["launched_count"] == 1
    assert launched == [100]
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert any(
        event.get("kind") == "review_dispatch_stalled"
        and event.get("payload", {}).get("status") == "unclaimed"
        for event in state.get("events", [])
    )


def test_detect_and_handle_stalled_reviews_skips_terminal_pr_reaps_sidecar(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue observed 07-22: a dead reviewer sidecar for a PR already
    lifecycle-reaped to merged/closed (review_dispatch_status None) must be
    silently reaped -- no review_dispatch_stalled event, no rewrite to
    failed. A second, non-terminal PR in the same pass still gets the normal
    reap-to-failed treatment, proving the terminal skip is scoped to that PR
    only."""
    from charlie_work.state import empty_state

    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    state_file = tmp_path / "state.json"
    config = OrchestratorConfig()

    state = empty_state()
    state["prs"]["100"] = {
        "number": 100,
        "status": "merged",
        "review_dispatch_status": None,
    }
    state["prs"]["200"] = {
        "number": 200,
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-20T00:00:00Z",
        "reviewer_pid": 12345,
        "reviewer_process_start_time": 1.0,
    }
    save_state(state_file, state)

    def _sidecar(pr_number: int, started_at: str) -> dict[str, Any]:
        return {
            "issue_number": pr_number,
            "branch": f"agent/issue-{pr_number}-fix",
            "worktree_path": str(reviews_dir / f"pr-{pr_number}"),
            "prompt_path": str(reviews_dir / f"pr-{pr_number}" / ".orchestrator-prompt.md"),
            "command": ["claude", "-p"],
            "pid": 999999999,
            "started_at": started_at,
            "log_path": str(reviews_dir / f"issue-{pr_number}.claude.log"),
            "error": None,
            "process_start_time": 1.0,
            "adapter_kind": "claude-code",
        }

    old_started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar_100 = reviews_dir / "issue-100.claude.json"
    sidecar_100.write_text(json.dumps(_sidecar(100, old_started)), encoding="utf-8")
    sidecar_200 = reviews_dir / "issue-200.claude.json"
    sidecar_200.write_text(json.dumps(_sidecar(200, old_started)), encoding="utf-8")
    # PR 200 is the non-terminal PR that should get the normal reap-to-failed
    # treatment. Give it a readable log with no throttle marker so it
    # classifies as NOT_THROTTLED (the counted-failure path). Without this the
    # log read would fail and classify as UNDETERMINED (issue #1069), which
    # rolls back instead of failing — not the path this test exercises.
    (reviews_dir / "issue-200.claude.log").write_text("ordinary crash output\n", encoding="utf-8")

    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: False)
    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", lambda *a, **k: True
    )

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    assert not sidecar_100.exists()
    assert not sidecar_200.exists()
    assert [entry["pr"] for entry in stalled] == [200]

    state_after = load_state(state_file)
    assert state_after["prs"]["100"]["review_dispatch_status"] is None
    assert state_after["prs"]["100"]["status"] == "merged"
    assert state_after["prs"]["200"]["review_dispatch_status"] == "review_dispatch_failed"

    stalled_events = [
        e for e in state_after.get("events", []) if e.get("kind") == "review_dispatch_stalled"
    ]
    assert len(stalled_events) == 1
    assert stalled_events[0]["payload"]["pr_number"] == 200


def test_detect_and_handle_stalled_reviews_warns_on_checkout_removal_failure(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #526: a genuine worktree-removal failure must not be silently
    discarded; the stalled sweep emits a one-shot warning event and sets a
    per-PR marker so the next pass can retry without flooding the event ring."""
    from datetime import timedelta

    from charlie_work.state import empty_state
    from charlie_work.workflow import _detect_and_handle_stalled_reviews

    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    state_file = tmp_path / "state.json"

    config = OrchestratorConfig()
    state = empty_state()
    old_dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    state["prs"]["100"] = {
        "number": 100,
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": old_dispatched,
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
        "started_at": old_dispatched,
        "log_path": str(reviews_dir / "issue-100.claude.log"),
        "error": None,
        "process_start_time": 1.0,
        "adapter_kind": "claude-code",
    }
    (reviews_dir / "issue-100.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")
    # Give the reviewer a readable log with no throttle marker so it
    # classifies as NOT_THROTTLED (the counted-failure path that calls
    # _remove_review_checkout_with_warning). Without this the log read would
    # fail and classify as UNDETERMINED (issue #1069), which uses a different
    # checkout-removal path — not the warning path this test exercises.
    (reviews_dir / "issue-100.claude.log").write_text("ordinary crash output\n", encoding="utf-8")

    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: False)
    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", lambda *a, **k: False
    )

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    assert [entry["pr"] for entry in stalled] == [100]
    state_after = load_state(state_file)
    assert state_after["prs"]["100"]["review_dispatch_status"] == "review_dispatch_failed"
    assert state_after["prs"]["100"]["review_checkout_removal_warned"] is True
    warning_events = [
        e
        for e in state_after.get("events", [])
        if e.get("kind") == "review_checkout_removal_failed"
    ]
    assert len(warning_events) == 1
    assert warning_events[0]["payload"]["pr_number"] == 100


def test_detect_and_handle_stalled_reviews_aggregates_same_pass_events(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #525: multiple stalled reviewer claims in one pass become one sweep event."""
    from charlie_work.worker import WorkerView

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    state_file = tmp_path / "state.json"
    config = OrchestratorConfig()

    prs = [100, 200, 300]
    old = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    state = empty_state()
    for pr in prs:
        state["prs"][str(pr)] = {
            "number": pr,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old,
            "reviewer_pid": 99999,
            "reviewer_process_start_time": 1.0,
        }
    save_state(state_file, state)

    for pr in prs:
        _make_dead_review_sidecar(reviews_dir, pr, "no verdict")

    monkeypatch.setattr(WorkerView, "is_alive", lambda self: False)
    monkeypatch.setattr("charlie_work.stalled_review_reap.is_pid_alive", lambda *_: False)
    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", lambda *a, **k: True
    )

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    assert {entry["pr"] for entry in stalled} == set(prs)
    state_after = load_state(state_file)
    events = state_after["events"]
    sweep = [e for e in events if e.get("kind") == "review_dispatch_stalled_sweep"]
    assert len(sweep) == 1
    assert sweep[0]["payload"]["count"] == len(prs)
    assert set(sweep[0]["payload"]["pr_numbers"]) == set(prs)
