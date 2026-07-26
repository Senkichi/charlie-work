"""Regression tests for throttle-backoff over-counting in
``_detect_and_handle_stalled_reviews``.

The exponential reviewer-quota probe backoff counts consecutive failed
PROBES. Two defects made it count dead-session *sightings* instead:

1. The throttled path rolled back the claim and removed the review
   checkout but never reaped the sidecar. The rolled-back claim is
   deliberately non-terminal, so neither terminal guard in the sweep ever
   reaped it either -- the same dead reviewer resurfaced every sweep, its
   log tail still matched the throttle signature, and every pass
   re-applied the backoff for a session that died exactly once.
2. Within one sweep, a wave of N simultaneously throttled reviewers
   applied the backoff N times, though it is one observation of one
   closed provider window.

Observed live 2026-07-24: one quota outage killing two reviewers drove
``consecutive_probe_failures`` to 14 across ~6 passes, pushing
``probe_after`` out to the 4-hour cap while the provider window was
already open again -- a self-inflicted 4-hour review outage.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import _detect_and_handle_stalled_reviews

from test_charlie_work import _init_git_repo

_THROTTLE_LINE = "You've hit your session limit · resets 4:40pm (America/Los_Angeles)\n"


def _write_throttled_reviewer(reviews_dir: Path, pr_number: int, tmp_path: Path) -> Path:
    """Fabricate a dead reviewer sidecar whose log shows a throttle signature."""
    log_path = reviews_dir / f"issue-{pr_number}-review.claude.log"
    log_path.write_text(_THROTTLE_LINE, encoding="utf-8")
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


def _seed(tmp_path: Path, pr_numbers: list[int]):
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
        for pr in pr_numbers:
            state["prs"][str(pr)] = {
                "number": pr,
                "review_dispatch_status": "review_dispatch_dispatched",
                "review_dispatched_at": started,
                "reviewer_pid": 999999999,
                "reviewer_process_start_time": 1.0,
            }
        save_state(state_file, state)
    return repo_root, reviews_dir, config, state_file


def _stalled_event_count(state) -> int:
    return sum(
        1
        for event in state.get("events", [])
        if event.get("kind") == "review_dispatch_stalled"
        and event.get("payload", {}).get("reason") == "provider_throttled"
    )


def test_throttled_dead_reviewer_backoff_counted_once_across_sweeps(tmp_path: Path) -> None:
    repo_root, reviews_dir, config, state_file = _seed(tmp_path, [100])
    sidecar_path = _write_throttled_reviewer(reviews_dir, 100, tmp_path)

    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    state = load_state(state_file)
    quota = state.get("reviewer_quota", {})
    assert quota.get("consecutive_probe_failures") == 1
    assert not sidecar_path.exists(), (
        "the throttled path must reap the sidecar like every other handled "
        "path, or the dead session is re-counted every sweep"
    )
    assert state["prs"]["100"].get("review_dispatch_status") is None
    events_after_first = _stalled_event_count(state)
    assert events_after_first == 1

    # Second sweep: the dead session is gone; nothing may re-increment the
    # backoff or re-emit the stalled event for it.
    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    state = load_state(state_file)
    assert state.get("reviewer_quota", {}).get("consecutive_probe_failures") == 1
    assert _stalled_event_count(state) == 1


def test_wave_of_throttled_reviewers_is_one_backoff_increment(tmp_path: Path) -> None:
    repo_root, reviews_dir, config, state_file = _seed(tmp_path, [100, 200])
    path_a = _write_throttled_reviewer(reviews_dir, 100, tmp_path)
    path_b = _write_throttled_reviewer(reviews_dir, 200, tmp_path)

    stalled = _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    # Both reviewers are individually rolled back, reaped, and reported ...
    assert {entry["pr"] for entry in stalled if entry.get("reason") == "provider_throttled"} == {
        100,
        200,
    }
    state = load_state(state_file)
    assert state["prs"]["100"].get("review_dispatch_status") is None
    assert state["prs"]["200"].get("review_dispatch_status") is None
    assert not path_a.exists()
    assert not path_b.exists()
    assert _stalled_event_count(state) == 2

    # ... but the wave is ONE observation of ONE closed provider window:
    # exactly one backoff increment.
    assert state.get("reviewer_quota", {}).get("consecutive_probe_failures") == 1


# ---------------------------------------------------------------------------
# Issue #583: throttle-classified reviewer deaths that ALSO exhausted the turn
# budget must NOT roll back the attempt counter.
# ---------------------------------------------------------------------------


def _seed_with_attempt(tmp_path: Path, pr_number: int, attempt_count: int):
    """Seed state with a dispatched reviewer claim and a non-zero attempt count."""
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
            "reviewer_pid": 999999999,
            "reviewer_process_start_time": 1.0,
            "review_dispatch_attempt_count": attempt_count,
            # Set by _reap_review_verdicts (which runs before this sweep) when
            # it extracted a turn-limit miss from this same dead session.
            "review_turn_limit_summary_posted": True,
        }
        save_state(state_file, state)
    return repo_root, reviews_dir, config, state_file


def test_throttled_death_with_turn_limit_miss_counts_attempt(tmp_path: Path) -> None:
    """Issue #583: a dead reviewer whose log tail matches a throttle signature
    AND that produced a turn-limit miss (``review_turn_limit_summary_posted``
    set on the pr-state by the verdict-reaper that runs before this sweep) did
    real PR-specific work. Its death is a PR-level outcome, so the global quota
    backoff must still be applied (the throttle signal is real) but the per-PR
    attempt counter must NOT be rolled back -- otherwise a PR whose reviews
    deterministically hit the turn cap gets unlimited free retries whenever the
    account is near its limit and the 3-attempt cap can never fire.
    """
    repo_root, reviews_dir, config, state_file = _seed_with_attempt(tmp_path, 100, 1)
    sidecar_path = _write_throttled_reviewer(reviews_dir, 100, tmp_path)

    stalled = _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    state = load_state(state_file)

    # The global quota backoff IS applied -- the throttle signal is real.
    quota = state.get("reviewer_quota", {})
    assert quota.get("consecutive_probe_failures") == 1
    assert quota.get("throttled_until") is not None

    # The attempt counter is NOT rolled back: this is a counted failure.
    assert state["prs"]["100"].get("review_dispatch_attempt_count") == 1

    # The claim is moved to review_dispatch_failed (a counted, terminal
    # failure) -- NOT rolled back to None (which would make it immediately
    # re-dispatchable and never converge).
    assert state["prs"]["100"].get("review_dispatch_status") == "review_dispatch_failed"

    # The dead session's sidecar is reaped so it cannot resurrect as a phantom.
    assert not sidecar_path.exists()

    # A distinct stalled reason is emitted so the counted-but-throttled
    # outcome is observable in events.db.
    counted_events = [
        e
        for e in state.get("events", [])
        if e.get("kind") == "review_dispatch_stalled"
        and e.get("payload", {}).get("reason") == "provider_throttled_turn_limit_counted"
    ]
    assert len(counted_events) == 1
    assert any(
        entry.get("pr") == 100 and entry.get("reason") == "provider_throttled_turn_limit_counted"
        for entry in stalled
    )


def test_pure_throttle_death_without_turn_limit_still_rolls_back(tmp_path: Path) -> None:
    """Issue #583 guard: a throttled death WITHOUT a turn-limit miss still
    gets the provider-throttle rollback (claim cleared, attempt decremented).
    Only sessions that did substantial work lose the rollback.
    """
    repo_root, reviews_dir, config, state_file = _seed_with_attempt(tmp_path, 100, 1)
    # Clear the turn-limit flag: this session died purely from the provider
    # limit without exhausting its turn budget.
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"]["100"]["review_turn_limit_summary_posted"] = False
        save_state(state_file, state)
    sidecar_path = _write_throttled_reviewer(reviews_dir, 100, tmp_path)

    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    state = load_state(state_file)
    # Quota backoff applied.
    assert state.get("reviewer_quota", {}).get("consecutive_probe_failures") == 1
    # Attempt rolled back (pure provider outage -- not a PR-specific failure).
    assert state["prs"]["100"].get("review_dispatch_attempt_count") == 0
    # Claim cleared (rolled back, immediately re-dispatchable once quota opens).
    assert state["prs"]["100"].get("review_dispatch_status") is None
    assert not sidecar_path.exists()
