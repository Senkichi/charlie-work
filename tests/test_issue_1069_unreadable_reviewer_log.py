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

from _helpers import _init_git_repo


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


def test_persistent_unreadable_log_terminates_via_streak_bound(tmp_path: Path) -> None:
    """A PR whose reviewer log is persistently unreadable/empty across many
    sweeps must eventually reach a terminal state rather than redispatching
    forever with no cap and no backoff (review finding on PR #1161: the
    original #1069 fix decremented the attempt counter on every UNDETERMINED
    death but, unlike the throttle path, did not arm fleet-wide backoff — so
    the attempt cap never fired and the PR looped indefinitely).

    The bound: a per-PR ``review_log_unreadable_streak`` counts consecutive
    UNDETERMINED deaths. The first N (``max_consecutive_review_log_unreadable``)
    are transient — roll back and decrement. After N, the death becomes a
    counted failure (attempt counter NOT decremented) so the existing
    ``max_review_dispatch_attempts`` cap converges and escalates.

    This test simulates the dispatch → death → stalled-sweep cycle manually
    (the dispatch path is a method on OrchestratorApp; the stalled sweep is
    the standalone function under test) and proves the PR reaches
    ``review_dispatch_failed`` with ``attempt_count >= max_attempts`` in a
    finite number of sweeps.
    """
    max_streak = 2
    max_attempts = 3
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(
            enabled=True,
            max_consecutive_review_log_unreadable=max_streak,
            max_review_dispatch_attempts=max_attempts,
        )
    )
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )

    started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    pr_number = 100

    def _redispatch(attempt_count: int) -> None:
        """Simulate what dispatch_reviews does: claim the PR as dispatched
        and increment the attempt counter."""
        with state_lock(state_file):
            st = load_state(state_file)
            st["prs"][str(pr_number)] = {
                **st["prs"].get(str(pr_number), {}),
                "number": pr_number,
                "review_dispatch_status": "review_dispatch_dispatched",
                "review_dispatched_at": started,
                "review_dispatch_pending_at": None,
                "review_dispatch_failed_at": None,
                "reviewer_pid": 999999999,
                "reviewer_process_start_time": 1.0,
                "review_dispatch_attempt_count": attempt_count,
            }
            save_state(state_file, st)

    # Seed the first dispatch.
    _redispatch(attempt_count=1)

    log_path = reviews_dir / "issue-100-review.claude.log"
    # Log never created → every read raises OSError → UNDETERMINED every sweep.

    max_sweeps = 20  # safety bound — must terminate well before this
    reached_terminal = False
    transient_rollback_count = 0
    persistent_failure_count = 0

    for sweep in range(max_sweeps):
        # Re-write the sidecar each sweep — the stalled sweep reaps it.
        _write_sidecar(reviews_dir, pr_number, tmp_path, log_path)
        assert not log_path.exists()  # confirms the read will fail

        _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

        state = load_state(state_file)
        pr_state = state["prs"][str(pr_number)]
        status = pr_state.get("review_dispatch_status")
        attempt_count = int(pr_state.get("review_dispatch_attempt_count", 0))
        streak = int(pr_state.get("review_log_unreadable_streak", 0))

        if streak <= max_streak:
            # Transient phase: rolled back (status None), attempt decremented.
            assert status is None, (
                f"sweep {sweep}: streak {streak} <= {max_streak} should roll back, "
                f"got status={status!r}"
            )
            transient_rollback_count += 1
        else:
            # Persistent phase: counted failure, attempt NOT decremented.
            assert status == "review_dispatch_failed", (
                f"sweep {sweep}: streak {streak} > {max_streak} should fail, got status={status!r}"
            )
            persistent_failure_count += 1

        # Check for terminal state: attempt_count >= max_attempts means the
        # dispatch cap blocks further dispatch and the escalation check fires.
        if attempt_count >= max_attempts and status == "review_dispatch_failed":
            reached_terminal = True
            break

        # Simulate re-dispatch for the next cycle.
        _redispatch(attempt_count=attempt_count + 1)

    assert reached_terminal, (
        f"PR did not reach a terminal state after {max_sweeps} sweeps — "
        f"unbounded loop (the streak bound failed to terminate)."
    )

    # Verify the phase transition actually happened (not all transient, not
    # all persistent — the bound switches from rollback to counted-failure).
    assert transient_rollback_count == max_streak, (
        f"expected {max_streak} transient rollbacks, got {transient_rollback_count}"
    )
    assert persistent_failure_count >= 1, "expected at least one persistent counted failure"

    # Final state: terminal failure at the attempt cap.
    state = load_state(state_file)
    pr_state = state["prs"][str(pr_number)]
    assert pr_state.get("review_dispatch_status") == "review_dispatch_failed"
    assert int(pr_state.get("review_dispatch_attempt_count", 0)) >= max_attempts

    # Fleet-wide backoff was never armed — UNDETERMINED never arms backoff
    # (that is the whole point of the distinct third state).
    quota = state.get("reviewer_quota", {})
    assert not quota.get("throttled_until")
    assert not quota.get("probe_after")


def test_streak_resets_on_definitive_not_throttled_outcome(tmp_path: Path) -> None:
    """If a PR had a streak of UNDETERMINED deaths and then a reviewer dies
    with a readable, non-throttle log (NOT_THROTTLED), the streak must reset
    to 0. A subsequent UNDETERMINED death starts a fresh streak rather than
    accumulating toward the persistent threshold from the prior epoch
    (issue #1069 streak-reset invariant).
    """
    repo_root, reviews_dir, config, state_file = _seed(tmp_path, attempt_count=1)

    # Pre-seed a streak of 2 (just below the default threshold of 3).
    with state_lock(state_file):
        st = load_state(state_file)
        st["prs"]["100"]["review_log_unreadable_streak"] = 2
        save_state(state_file, st)

    # Now the reviewer dies with a readable, non-throttle log → NOT_THROTTLED.
    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text("ordinary crash output, no throttle marker\n", encoding="utf-8")
    _write_sidecar(reviews_dir, 100, tmp_path, log_path)

    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    state = load_state(state_file)
    pr_state = state["prs"]["100"]
    # NOT_THROTTLED → counted failure.
    assert pr_state.get("review_dispatch_status") == "review_dispatch_failed"
    # Streak reset to 0.
    assert pr_state.get("review_log_unreadable_streak") == 0
