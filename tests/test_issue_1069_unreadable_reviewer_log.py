"""Regression tests for issue #1069: ``_detect_and_handle_stalled_reviews``
could not distinguish "reviewer log contains no throttle marker" from
"reviewer log could not be read" (or was empty / 0-byte).

Both the ``OSError`` read-failure path and a readable-but-empty log used to
collapse to ``throttled = False`` alongside a clean read with no marker. That
burned the PR's ``review_dispatch_attempt_count`` for a death that may not
have been the PR's fault AND failed to arm the fleet-wide reviewer-quota
backoff — re-arming the #1342-1346 redispatch-into-the-wall outage mechanism
through the read-error path.

The fix introduces a three-way ``_ThrottleClassification``: the undetermined
case (unreadable or empty log) rolls back the claim and decrements the
attempt counter (like the throttle path) but does NOT arm fleet-wide backoff
(unlike the throttle path), with a distinct ``review_log_unreadable`` event
reason so the condition is diagnosable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import _detect_and_handle_stalled_reviews

from test_charlie_work import _init_git_repo


def _seed(tmp_path: Path, pr_number: int = 100, attempt_count: int = 1):
    """Seed a repo, reviews dir, config, and state with one dispatched reviewer."""
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": started,
            "reviewer_pid": 999999999,  # not a real live pid
            "reviewer_process_start_time": 1.0,
            "review_dispatch_attempt_count": attempt_count,
        }
        save_state(state_file, state)
    return repo_root, reviews_dir, config, state_file


def _write_sidecar(reviews_dir: Path, pr_number: int, tmp_path: Path, log_path: Path) -> Path:
    """Write a claude-code review sidecar pointing at ``log_path``."""
    started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar = {
        "issue_number": pr_number,
        "branch": f"agent/issue-{pr_number}-fix",
        "worktree_path": str(tmp_path / "worktrees" / f"issue-{pr_number}"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,  # not a real live pid
        "started_at": started,
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    sidecar_path = reviews_dir / f"issue-{pr_number}.claude.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return sidecar_path


def test_unreadable_log_rolls_back_without_backoff(tmp_path: Path) -> None:
    """A dead reviewer whose log file cannot be read (OSError) must be
    classified as undetermined, not as a clean non-throttle death. The claim
    is rolled back (not failed) and the attempt counter is decremented, but
    the fleet-wide reviewer-quota backoff is NOT armed — there is no evidence
    of a throttle condition (issue #1069).
    """
    repo_root, reviews_dir, config, state_file = _seed(tmp_path, attempt_count=1)
    # Point the sidecar at a log path that does not exist → read_text raises
    # OSError, which used to collapse to ``throttled = False``.
    log_path = reviews_dir / "issue-100-review.claude.log"
    sidecar_path = _write_sidecar(reviews_dir, 100, tmp_path, log_path)
    assert not log_path.exists()  # confirms the read will fail

    stalled = _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    # Distinct reason so the condition is diagnosable in events.db.
    assert any(
        entry.get("pr") == 100 and entry.get("reason") == "review_log_unreadable"
        for entry in stalled
    )

    state = load_state(state_file)
    pr_state = state["prs"]["100"]
    # Rolled back, not failed — the claim is immediately re-dispatchable.
    assert pr_state.get("review_dispatch_status") is None
    assert pr_state.get("reviewer_pid") is None
    # Attempt budget preserved (decremented from 1 to 0) — the death may not
    # be the PR's fault.
    assert pr_state.get("review_dispatch_attempt_count") == 0

    # Fleet-wide backoff NOT armed — no evidence of a throttle condition.
    quota = state.get("reviewer_quota", {})
    assert not quota.get("throttled_until")
    assert not quota.get("probe_after")

    # Sidecar reaped so the dead session does not resurface every sweep.
    assert not sidecar_path.exists()

    # A distinct event is emitted with the unreadable-log reason.
    unreadable_events = [
        e
        for e in state.get("events", [])
        if e.get("kind") == "review_dispatch_stalled"
        and e.get("payload", {}).get("reason") == "review_log_unreadable"
    ]
    assert len(unreadable_events) == 1


def test_empty_log_rolls_back_without_backoff(tmp_path: Path) -> None:
    """A dead reviewer whose log file exists but is empty (0-byte, died
    before its first flush) must also be classified as undetermined. This is
    the more likely route than the OSError case — no I/O error is required,
    just a process that died before writing anything (issue #1069).
    """
    repo_root, reviews_dir, config, state_file = _seed(tmp_path, attempt_count=1)
    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text("", encoding="utf-8")  # 0-byte log
    sidecar_path = _write_sidecar(reviews_dir, 100, tmp_path, log_path)

    stalled = _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    assert any(
        entry.get("pr") == 100 and entry.get("reason") == "review_log_unreadable"
        for entry in stalled
    )

    state = load_state(state_file)
    pr_state = state["prs"]["100"]
    assert pr_state.get("review_dispatch_status") is None
    assert pr_state.get("review_dispatch_attempt_count") == 0

    quota = state.get("reviewer_quota", {})
    assert not quota.get("throttled_until")
    assert not quota.get("probe_after")

    assert not sidecar_path.exists()


def test_readable_log_without_marker_still_fails(tmp_path: Path) -> None:
    """A dead reviewer whose log is readable and non-empty but contains no
    throttle marker must still be classified as NOT_THROTTLED and marked as a
    counted failure — this is the existing behaviour and must not regress
    (issue #1069: the fix adds a third state, it does not change the
    not-throttled branch).
    """
    repo_root, reviews_dir, config, state_file = _seed(tmp_path, attempt_count=1)
    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text("some ordinary crash output, no throttle marker here\n", encoding="utf-8")
    _write_sidecar(reviews_dir, 100, tmp_path, log_path)

    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    state = load_state(state_file)
    pr_state = state["prs"]["100"]
    # Counted failure — the attempt budget is consumed.
    assert pr_state.get("review_dispatch_status") == "review_dispatch_failed"
    assert pr_state.get("review_dispatch_attempt_count") == 1

    # No fleet-wide backoff (no throttle evidence).
    quota = state.get("reviewer_quota", {})
    assert not quota.get("throttled_until")


def test_undetermined_does_not_block_throttle_backoff_in_same_wave(tmp_path: Path) -> None:
    """An undetermined (unreadable-log) death in the same sweep as a
    confirmed throttle death must not prevent the throttle death from arming
    the fleet-wide backoff. The undetermined path does not set
    ``throttle_backoff_applied``, so a later throttled reviewer in the same
    wave still arms backoff (issue #1069).
    """
    repo_root, reviews_dir, config, state_file = _seed(tmp_path, pr_number=100, attempt_count=1)

    # Seed a second PR in the same state.
    started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["200"] = {
            "number": 200,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": started,
            "reviewer_pid": 999999999,
            "reviewer_process_start_time": 1.0,
            "review_dispatch_attempt_count": 1,
        }
        save_state(state_file, state)

    # PR 100: unreadable log (undetermined).
    log_100 = reviews_dir / "issue-100-review.claude.log"
    _write_sidecar(reviews_dir, 100, tmp_path, log_100)  # log does not exist

    # PR 200: readable log with a throttle marker.
    log_200 = reviews_dir / "issue-200-review.claude.log"
    log_200.write_text(
        "You've hit your session limit · resets 4:40pm (America/Los_Angeles)\n",
        encoding="utf-8",
    )
    _write_sidecar(reviews_dir, 200, tmp_path, log_200)

    stalled = _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    reasons = {entry.get("pr"): entry.get("reason") for entry in stalled}
    assert reasons.get(100) == "review_log_unreadable"
    assert reasons.get(200) == "provider_throttled"

    state = load_state(state_file)
    # PR 100 rolled back, attempt preserved.
    assert state["prs"]["100"].get("review_dispatch_status") is None
    assert state["prs"]["100"].get("review_dispatch_attempt_count") == 0
    # PR 200 rolled back, attempt preserved.
    assert state["prs"]["200"].get("review_dispatch_status") is None
    assert state["prs"]["200"].get("review_dispatch_attempt_count") == 0

    # Fleet-wide backoff IS armed (by the throttled PR 200, not the
    # undetermined PR 100).
    quota = state.get("reviewer_quota", {})
    assert quota.get("throttled_until")
    assert quota.get("probe_after")
    assert quota.get("consecutive_probe_failures") == 1
