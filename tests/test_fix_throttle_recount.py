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
