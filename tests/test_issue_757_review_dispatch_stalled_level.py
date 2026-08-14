"""Regression tests for review_dispatch_stalled level classification.

Issues #757 and #748: ``review_dispatch_stalled`` has six emit sites in
``_detect_and_handle_stalled_reviews``. Two are benign and reclassified to
``warning``: provider-throttle reasons (``provider_throttled`` and
``provider_throttled_turn_limit_counted``, issue #757) and unclaimed packets
(issue #748 — a transient startup race, not a terminal failure). The
remaining three — dead sidecar with no throttle signature, stale pending
claim, and stale dispatched claim — are genuine failures (process death or
launch crash) and remain ``error``.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.instrumentation import close_db, query_events
from charlie_work.state import (
    _REVIEW_STALE_CLAIM_TIMEOUT_MINUTES,
    load_state,
    save_state,
    state_lock,
)
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
        "pid": 999999999,
        "started_at": started,
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    sidecar_path = reviews_dir / f"issue-{pr_number}-review.claude.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return sidecar_path


def _write_dead_reviewer(reviews_dir: Path, pr_number: int, tmp_path: Path) -> Path:
    """Fabricate a dead reviewer sidecar whose log shows no throttle signature."""
    log_path = reviews_dir / f"issue-{pr_number}-review.claude.log"
    log_path.write_text("no verdict\n", encoding="utf-8")
    started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar = {
        "issue_number": pr_number,
        "branch": f"agent/issue-{pr_number}-fix",
        "worktree_path": str(tmp_path / "worktrees" / f"issue-{pr_number}"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,
        "started_at": started,
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    sidecar_path = reviews_dir / f"issue-{pr_number}-review.claude.json"
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


def _seed_with_attempt(tmp_path: Path, pr_number: int, attempt_count: int):
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
            "review_turn_limit_summary_posted": True,
        }
        save_state(state_file, state)
    return repo_root, reviews_dir, config, state_file


def _seed_unclaimed(tmp_path: Path, pr_number: int):
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompts_dir / f"{pr_number}.md"
    prompt_path.write_text("review prompt", encoding="utf-8")
    # Backdate the prompt so it is past the stale-claim timeout.
    stale = datetime.now(UTC) - timedelta(minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES + 5)
    os.utime(prompt_path, (stale.timestamp(), stale.timestamp()))
    packet_age = stale.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "status": "reviewing",
            "prompt_path": str(prompt_path),
            "review_dispatch_status": None,
            "review_dispatch_pending_at": None,
            "review_dispatched_at": None,
            "reviewer_pid": None,
            "reviewer_process_start_time": None,
        }
        save_state(state_file, state)
    return repo_root, reviews_dir, config, state_file, packet_age


@pytest.fixture(autouse=True)
def _close_db_after_test(tmp_path: Path) -> None:
    """Ensure DB connections are closed between tests to avoid cross-test contamination."""
    yield
    close_db(tmp_path / "state.json")


def test_provider_throttled_is_warning(tmp_path: Path) -> None:
    """A provider-throttled dead reviewer emits review_dispatch_stalled at warning."""
    repo_root, reviews_dir, config, state_file = _seed(tmp_path, [100])
    _write_throttled_reviewer(reviews_dir, 100, tmp_path)

    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    stalled = query_events(
        state_file,
        kind="review_dispatch_stalled",
        level="warning",
    )
    assert len(stalled) == 1
    assert stalled[0]["payload"]["reason"] == "provider_throttled"
    assert stalled[0]["level"] == "warning"

    errors = query_events(
        state_file,
        kind="review_dispatch_stalled",
        level="error",
    )
    assert len(errors) == 0


def test_provider_throttled_turn_limit_counted_is_warning(tmp_path: Path) -> None:
    """A throttled reviewer that also exhausted its turn budget is still warning-level."""
    repo_root, reviews_dir, config, state_file = _seed_with_attempt(tmp_path, 100, 1)
    _write_throttled_reviewer(reviews_dir, 100, tmp_path)

    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    stalled = query_events(
        state_file,
        kind="review_dispatch_stalled",
        level="warning",
    )
    assert len(stalled) == 1
    assert stalled[0]["payload"]["reason"] == "provider_throttled_turn_limit_counted"
    assert stalled[0]["level"] == "warning"

    errors = query_events(
        state_file,
        kind="review_dispatch_stalled",
        level="error",
    )
    assert len(errors) == 0


def test_unclaimed_review_packet_is_warning(tmp_path: Path) -> None:
    """A never-claimed review packet is warning, not error (issue #748).

    An unclaimed packet is a transient startup race: ``review()`` generated the
    packet but ``dispatch_reviews`` has not claimed it yet. The sweep marks it
    ``review_dispatch_failed`` so the next dispatch pass retries. Measured
    against events.db (81 unclaimed events): 31/81 recovered via
    ``review_dispatch_claim`` (median 19s, 27/31 under 60s); the remaining
    50/81 were handled by fallback mechanisms. In neither case is the packet
    stuck, so ``warning`` is the right level -- not ``error`` (terminal) and
    not ``info`` (normal operation).
    """
    repo_root, reviews_dir, config, state_file, _ = _seed_unclaimed(tmp_path, 100)

    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    warnings = query_events(
        state_file,
        kind="review_dispatch_stalled",
        level="warning",
    )
    assert len(warnings) == 1
    assert warnings[0]["payload"].get("status") == "unclaimed"
    assert warnings[0]["level"] == "warning"

    errors = query_events(
        state_file,
        kind="review_dispatch_stalled",
        level="error",
    )
    assert len(errors) == 0


def test_dead_reviewer_without_throttle_stays_error(tmp_path: Path) -> None:
    """A dead reviewer that did not hit a provider throttle remains error."""
    repo_root, reviews_dir, config, state_file = _seed(tmp_path, [100])
    _write_dead_reviewer(reviews_dir, 100, tmp_path)

    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    warnings = query_events(
        state_file,
        kind="review_dispatch_stalled",
        level="warning",
    )
    assert len(warnings) == 0

    errors = query_events(
        state_file,
        kind="review_dispatch_stalled",
        level="error",
    )
    assert len(errors) == 1
    assert errors[0]["payload"].get("reason") is None
    assert errors[0]["level"] == "error"


def _seed_stale_pending(tmp_path: Path, pr_number: int):
    """Seed a PR in ``review_dispatch_pending`` past the stale timeout, no sidecar.

    This exercises the second loop's ``pending`` branch (emit site 4): the
    dispatch launch crashed before writing a sidecar, leaving a stale pending
    claim with no recoverable process.
    """
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    stale = datetime.now(UTC) - timedelta(minutes=_REVIEW_STALE_CLAIM_TIMEOUT_MINUTES + 5)
    pending_at = stale.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "review_dispatch_status": "review_dispatch_pending",
            "review_dispatch_pending_at": pending_at,
            "reviewer_pid": None,
            "reviewer_process_start_time": None,
        }
        save_state(state_file, state)
    return repo_root, reviews_dir, config, state_file


def test_stale_pending_claim_stays_error(tmp_path: Path) -> None:
    """A stale pending claim with no sidecar remains error (issue #748).

    Emit site 4: ``review_dispatch_pending`` past the stale timeout with no
    sidecar means the dispatch launch crashed before writing one. This is a
    genuine failure (launch crash), not a transient race, so it stays
    ``error``. Zero occurrences in events.db (2026-07-23 to 2026-08-13), but
    the code path exists for robustness and must remain ``error`` if it fires.
    """
    repo_root, reviews_dir, config, state_file = _seed_stale_pending(tmp_path, 100)

    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    warnings = query_events(
        state_file,
        kind="review_dispatch_stalled",
        level="warning",
    )
    assert len(warnings) == 0

    errors = query_events(
        state_file,
        kind="review_dispatch_stalled",
        level="error",
    )
    assert len(errors) == 1
    assert errors[0]["payload"].get("status") == "pending"
    assert errors[0]["level"] == "error"


def test_stale_dispatched_claim_no_sidecar_stays_error(tmp_path: Path) -> None:
    """A stale dispatched claim with dead PID and no sidecar remains error (issue #748).

    Emit site 5: ``review_dispatch_dispatched`` past the stale timeout with a
    dead PID and no sidecar means the reviewer process died after dispatch and
    its sidecar was lost. This is a genuine failure (process death), not a
    transient race, so it stays ``error``. Zero occurrences in events.db
    (2026-07-23 to 2026-08-13), but the code path exists for robustness and
    must remain ``error`` if it fires.
    """
    # _seed creates a dispatched PR with a dead PID (999999999) and a stale
    # timestamp (1 hour ago). By NOT writing a sidecar, the first loop (which
    # iterates sidecars) skips this PR, and the second loop processes it
    # through the ``review_dispatch_dispatched`` stale-claim branch.
    repo_root, reviews_dir, config, state_file = _seed(tmp_path, [100])

    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    warnings = query_events(
        state_file,
        kind="review_dispatch_stalled",
        level="warning",
    )
    assert len(warnings) == 0

    errors = query_events(
        state_file,
        kind="review_dispatch_stalled",
        level="error",
    )
    assert len(errors) == 1
    assert errors[0]["payload"].get("status") == "dispatched"
    assert errors[0]["level"] == "error"
