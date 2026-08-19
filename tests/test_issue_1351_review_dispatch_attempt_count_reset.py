"""Regression tests for issue #1351.

``review()`` used to unconditionally reset ``review_dispatch_attempt_count``
to 0 on every packet write. For a PR stuck on a stale
``decision``/``reviewed_head_sha`` pair, the ``already_approved`` branch of
``loop()`` calls ``review()`` every pass (``head_matches`` is always False),
so that reset raced with ``dispatch_reviews()``'s claim-time increment and the
counter oscillated 1->0->1->0... never reaching ``max_review_dispatch_attempts``
-- the cap was silently inert and reviewers re-dispatched indefinitely with no
escalation.

The fix baselines the head the counter is counting against in
``review_dispatch_attempt_last_head`` (mirroring
``no_op_rework_attempts_last_head`` / ``conflict_rework_attempts_last_head``):
``review()`` resets the counter only when the packetized head advances past
that baseline, not on every call.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from _fakes_github import FakeGitHub

_PR = 456
_ISSUE = 123
_HEAD = "sha-abc123"  # FakeGitHub's default PR #456 headRefOid


def _app(tmp_path: Path, *, max_attempts: int = 3) -> OrchestratorApp:
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(
            enabled=True, max_review_dispatch_attempts=max_attempts
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, FakeGitHub())


def _seed(
    app: OrchestratorApp,
    *,
    attempt_count: int,
    last_head: str | None,
    extra: dict | None = None,
) -> None:
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        entry = {
            "number": _PR,
            "issue_number": _ISSUE,
            "review_dispatch_attempt_count": attempt_count,
        }
        if last_head is not None:
            entry["review_dispatch_attempt_last_head"] = last_head
        if extra:
            entry.update(extra)
        state["prs"][str(_PR)] = entry
        save_state(app.paths.state_file, state)


def _claim_increment(app: OrchestratorApp) -> None:
    """Simulate ``dispatch_reviews()``'s claim-time increment of the counter.

    The real claim writes ``review_dispatch_attempt_count + 1`` under the state
    lock; this helper mirrors that single field update so the test can exercise
    the review()/claim race without launching a reviewer process.
    """
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        pr_state = state["prs"].get(str(_PR), {})
        pr_state["review_dispatch_attempt_count"] = (
            int(pr_state.get("review_dispatch_attempt_count", 0)) + 1
        )
        state["prs"][str(_PR)] = pr_state
        save_state(app.paths.state_file, state)


def test_review_same_head_preserves_dispatch_attempt_count(tmp_path: Path) -> None:
    """Issue #1351: rebuilding a packet for the SAME head must NOT zero the
    dispatch attempt counter. Before the fix, review() reset it to 0 on every
    call, so the counter never accumulated and max_review_dispatch_attempts
    never fired."""
    app = _app(tmp_path)
    _seed(app, attempt_count=2, last_head=_HEAD)

    result = app.review(_PR)

    assert result.ok is True
    state = load_state(app.paths.state_file)
    # The counter must be PRESERVED, not reset to 0.
    assert state["prs"][str(_PR)]["review_dispatch_attempt_count"] == 2
    assert state["prs"][str(_PR)]["review_dispatch_attempt_last_head"] == _HEAD


def test_review_new_head_resets_dispatch_attempt_count(tmp_path: Path) -> None:
    """Issue #1351 control: when the head genuinely advances past the
    baselined head, the counter resets to 0 so the fresh review cycle starts
    clean -- the legitimate case the unconditional reset used to handle."""
    app = _app(tmp_path)
    _seed(app, attempt_count=2, last_head="sha-old")

    result = app.review(_PR)

    assert result.ok is True
    state = load_state(app.paths.state_file)
    assert state["prs"][str(_PR)]["review_dispatch_attempt_count"] == 0
    assert state["prs"][str(_PR)]["review_dispatch_attempt_last_head"] == _HEAD


def test_review_first_packet_resets_dispatch_attempt_count(tmp_path: Path) -> None:
    """Issue #1351 control: a PR with no prior baseline (first packet, or a
    state.json predating the fix) resets to 0 -- the safe default that matches
    the pre-fix behavior on first contact."""
    app = _app(tmp_path)
    # Seed a counter but NO review_dispatch_attempt_last_head: a legacy state
    # entry predating the fix carries the count but not the baseline.
    _seed(app, attempt_count=2, last_head=None)

    result = app.review(_PR)

    assert result.ok is True
    state = load_state(app.paths.state_file)
    assert state["prs"][str(_PR)]["review_dispatch_attempt_count"] == 0
    assert state["prs"][str(_PR)]["review_dispatch_attempt_last_head"] == _HEAD


def test_review_repeated_same_head_calls_accumulate_to_cap(tmp_path: Path) -> None:
    """Issue #1351 end-to-end race reproduction: review() is called every loop
    pass (the already_approved / head_matches=False stuck state), with a
    dispatch claim incrementing the counter between passes. Before the fix the
    counter oscillated 1->0->1->0... (observed on PR #1235 as 1,1,1,2); after
    the fix it accumulates 1,2,3 so max_review_dispatch_attempts can fire."""
    app = _app(tmp_path, max_attempts=3)
    _seed(app, attempt_count=0, last_head=None)

    # Pass 1: first packet for this head -> reset to 0, baseline the head.
    app.review(_PR)
    _claim_increment(app)  # dispatch claim -> 1
    assert load_state(app.paths.state_file)["prs"][str(_PR)]["review_dispatch_attempt_count"] == 1

    # Pass 2: same head -> review() must PRESERVE the counter at 1.
    app.review(_PR)
    state = load_state(app.paths.state_file)
    assert state["prs"][str(_PR)]["review_dispatch_attempt_count"] == 1
    _claim_increment(app)  # dispatch claim -> 2
    assert load_state(app.paths.state_file)["prs"][str(_PR)]["review_dispatch_attempt_count"] == 2

    # Pass 3: same head -> review() must PRESERVE the counter at 2.
    app.review(_PR)
    state = load_state(app.paths.state_file)
    assert state["prs"][str(_PR)]["review_dispatch_attempt_count"] == 2
    _claim_increment(app)  # dispatch claim -> 3
    assert load_state(app.paths.state_file)["prs"][str(_PR)]["review_dispatch_attempt_count"] == 3
