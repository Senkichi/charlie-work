"""Tests for TASK W5: janitor merge-conflict / no-op-rework rework routing.

Covers cost-spirals.md Finding 1 and pr-lifecycle.md's "janitor_blocked zero
readers" finding: previously, only a CI required-check failure
(``is_check_failure_block``) routed a janitor-gated PR into the rework cycle
(``review()``, workflow.py). A genuine merge conflict (mergeable=CONFLICTING
or mergeStateStatus=DIRTY) or a no-op-rework signal (diff/head unchanged
since the last request_changes verdict) instead fell into the
``janitor_blocked`` status, which has zero readers anywhere in the codebase
-- the PR just re-logged the identical failure every pass forever, and the
one pre-existing conflict-repair path (``_request_merge_conflict_rework``)
required ``decision == "approved"``, so a ``request_changes`` PR had no
repair path at all.

These tests reuse ``FakeGitHub`` from test_charlie_work.py (the default
fixture already wires PR #456 <-> issue #123 with a linked-issue title/body
and CLEAN checks) rather than duplicating that fixture.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from charlie_work.config import DevinConfig, OrchestratorConfig, ReviewConfig, WatchdogConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import OrchestratorApp

from test_charlie_work import FakeGitHub


def _set_decision(app: OrchestratorApp, pr_number: int, decision: str) -> None:
    pr_dir = app.paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": decision}), encoding="utf-8"
    )


def _conflicting_app(tmp_path: Path, **config_kwargs) -> OrchestratorApp:
    config = OrchestratorConfig(**config_kwargs)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["mergeable"] = "CONFLICTING"
    fake_gh.prs[0]["mergeStateStatus"] = "DIRTY"
    return OrchestratorApp(tmp_path, paths, config, fake_gh)


def test_janitor_conflict_routes_to_rework_when_request_changes(tmp_path: Path) -> None:
    """A CONFLICTING PR with an outstanding request_changes verdict must get
    a rebase route -- the old ``_request_merge_conflict_rework`` gate
    required ``decision == "approved"``, so this exact shape (the common
    case: a PR under active rework that also drifted into conflict) had no
    repair path at all.
    """
    app = _conflicting_app(tmp_path)
    _set_decision(app, 456, "request_changes")

    result = app.review(456)

    assert result.ok is True
    assert result.data["routed_to_rework"] is True
    assert result.data["rework_reason"] == "merge_conflict"

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["status"] == "rework_requested"
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1
    assert (123, app.config.labels.needs_rework) in app.gh.labels_added


def test_janitor_conflict_routes_to_rework_when_approved(tmp_path: Path) -> None:
    """Routing is decision-agnostic: a conflicting branch needs a rebase
    regardless of its review verdict, so an approved PR that drifted into
    conflict must route the same way as a request_changes one.
    """
    app = _conflicting_app(tmp_path)
    _set_decision(app, 456, "approved")

    result = app.review(456)

    assert result.ok is True
    assert result.data["routed_to_rework"] is True
    assert result.data["rework_reason"] == "merge_conflict"

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1
    assert (123, app.config.labels.needs_rework) in app.gh.labels_added


def test_janitor_conflict_attempts_count_cycles_not_passes(tmp_path: Path) -> None:
    """Attempts must count completed-but-still-failing rework CYCLES (head
    movement), never loop passes: review() re-runs every pass and the
    janitor re-detects the same conflict each time, so burning an attempt
    per detection would escalate a PR whose rework worker simply hasn't run
    yet within two passes. While the routed rework is pending and the head
    hasn't moved, the gate must fall back to the passive janitor_blocked
    wait without consuming anything; a new head that STILL conflicts is a
    failed cycle and consumes one attempt (bounding the
    push-conflicted-heads-forever loop); past ``max_conflict_rework_
    attempts`` escalate via the same ``transition()`` helper the other
    escalation call sites use so ``agent:human-needed`` actually lands
    (pr-lifecycle.md Finding 3).
    """
    app = _conflicting_app(tmp_path, review=ReviewConfig(max_conflict_rework_attempts=2))
    _set_decision(app, 456, "request_changes")

    result1 = app.review(456)
    assert result1.ok is True
    assert result1.data["routed_to_rework"] is True

    # Same head, rework pending: passive wait -- no route, no attempt burn,
    # no matter how many passes go by.
    for _ in range(3):
        wait_result = app.review(456)
        assert wait_result.ok is False
        assert "janitor gate blocked" in wait_result.message
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1
    assert state["issues"]["123"]["status"] == "rework_requested"

    # A rework cycle completes but its new head still conflicts: one attempt
    # burned, no re-route (the issue is already queued; the worktree layer
    # injects the conflict notice at relaunch).
    app.gh.pr_head_shas[456] = "sha-cycle-2"
    result2 = app.review(456)
    assert result2.ok is False
    assert "janitor gate blocked" in result2.message
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["conflict_rework_attempts"] == 2
    assert state["issues"]["123"]["status"] == "rework_requested"
    cycle_events = [e for e in state["events"] if e["kind"] == "janitor_rework_cycle_failed"]
    assert len(cycle_events) == 1

    # Another failed cycle exceeds the cap: escalate.
    app.gh.pr_head_shas[456] = "sha-cycle-3"
    result3 = app.review(456)
    assert result3.ok is False
    assert result3.data["escalated"] is True

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["prs"]["456"]["conflict_rework_attempts"] == 3
    assert (123, app.config.labels.human_needed) in app.gh.labels_added

    escalated_events = [e for e in state["events"] if e["kind"] == "janitor_rework_escalated"]
    assert len(escalated_events) == 1
    assert escalated_events[0]["payload"]["reason"] == "merge_conflict"

    # Escalation is terminal-until-human: further passes (even with yet
    # another new head) are stopped by review()'s upstream escalated-issue
    # skip before the janitor gate runs -- no re-route, no re-escalation
    # event, no attempt burn. Only `charlie unescalate` re-enters the
    # machine.
    app.gh.pr_head_shas[456] = "sha-cycle-4"
    result4 = app.review(456)
    assert result4.data.get("pass_skipped") is True
    assert "escalated" in result4.message
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["conflict_rework_attempts"] == 3
    assert state["issues"]["123"]["status"] == "escalated"
    escalated_events = [e for e in state["events"] if e["kind"] == "janitor_rework_escalated"]
    assert len(escalated_events) == 1


def _force_issue_status(app: OrchestratorApp, issue_number: int, status: str | None) -> None:
    """Overwrite an issue's tracked status, simulating the orphaned/stuck
    shape (e.g. reconcile's PASSIVE_OPEN_STATUS normalization of an issue
    whose rework bookkeeping was lost) that the no-op route exists for.
    """
    state = load_state(app.paths.state_file)
    record = {**state["issues"].get(str(issue_number), {}), "number": issue_number}
    if status is None:
        record.pop("status", None)
    else:
        record["status"] = status
    state["issues"][str(issue_number)] = record
    save_state(app.paths.state_file, state)


def test_janitor_conflict_live_session_wip_pushes_do_not_burn_attempts(tmp_path: Path) -> None:
    """A live rework session (issue status ``dispatched``) may push any
    number of intermediate WIP commits before it resolves the conflict.
    Burning an attempt per observed push would escalate a PR whose worker
    is actively fixing it (2 WIP pushes + the routing attempt exceeds the
    default cap of 2). Only a SETTLED head -- the issue back in
    ``rework_requested`` after the session ends -- counts as a cycle.
    """
    app = _conflicting_app(tmp_path, review=ReviewConfig(max_conflict_rework_attempts=2))
    _set_decision(app, 456, "request_changes")

    result1 = app.review(456)
    assert result1.ok is True
    assert result1.data["routed_to_rework"] is True

    _force_issue_status(app, 123, "dispatched")
    for wip_head in ("sha-wip-1", "sha-wip-2", "sha-wip-3"):
        app.gh.pr_head_shas[456] = wip_head
        wip_result = app.review(456)
        assert wip_result.ok is False
        assert "janitor gate blocked" in wip_result.message

    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1
    assert state["issues"]["123"]["status"] == "dispatched"
    assert (123, app.config.labels.human_needed) not in app.gh.labels_added

    # Session ends (reap restores rework_requested); the settled head still
    # conflicts: exactly one attempt burned for the whole cycle.
    _force_issue_status(app, 123, "rework_requested")
    settled_result = app.review(456)
    assert settled_result.ok is False
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["conflict_rework_attempts"] == 2


def test_janitor_conflict_dispatch_pending_crash_window_waits(tmp_path: Path) -> None:
    """``dispatch_pending`` is dispatch_rework's two-phase-claim
    intermediate status; a crash between its two locks strands the issue
    there until staleness reaping reclaims it. The pending-guard must treat
    it as pending -- routing/burning during the window would thrash the
    exact rework that is already (crash-)pending.
    """
    app = _conflicting_app(tmp_path)
    _set_decision(app, 456, "request_changes")
    _force_issue_status(app, 123, "dispatch_pending")

    for _ in range(3):
        result = app.review(456)
        assert result.ok is False
        assert "janitor gate blocked" in result.message

    state = load_state(app.paths.state_file)
    assert state["prs"]["456"].get("conflict_rework_attempts", 0) == 0
    assert state["issues"]["123"]["status"] == "dispatch_pending"


def test_janitor_conflict_unknown_baseline_records_before_burning(tmp_path: Path) -> None:
    """With no recorded head baseline (the rework predates this
    bookkeeping), a moved head is not provably a completed cycle -- record
    it as the baseline instead of burning; the NEXT settled head change
    burns.
    """
    app = _conflicting_app(tmp_path)
    _set_decision(app, 456, "request_changes")
    # Rework requested without ever passing through the routing wrapper
    # (e.g. record_review's normal request_changes path).
    _force_issue_status(app, 123, "rework_requested")

    app.gh.pr_head_shas[456] = "sha-settled-1"
    result1 = app.review(456)
    assert result1.ok is False
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"].get("conflict_rework_attempts", 0) == 0
    assert state["prs"]["456"]["conflict_rework_attempts_last_head"] == "sha-settled-1"

    # Waiting on the same settled head burns nothing.
    result_wait = app.review(456)
    assert result_wait.ok is False
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"].get("conflict_rework_attempts", 0) == 0

    # A second settled head that still conflicts is a completed cycle.
    app.gh.pr_head_shas[456] = "sha-settled-2"
    result2 = app.review(456)
    assert result2.ok is False
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1
    assert state["prs"]["456"]["conflict_rework_attempts_last_head"] == "sha-settled-2"


def test_janitor_no_op_rework_routes_to_rework(tmp_path: Path) -> None:
    """The janitor's no-op-rework signal (unchanged patch-id/head since the
    last request_changes verdict) was previously detected but never
    consumed by anything -- it just re-logged ``janitor_blocked`` forever.

    The route only fires when no rework is already pending for the issue:
    a request_changes verdict normally leaves the issue in
    ``rework_requested`` (covered by test_janitor_no_op_rework_waits_while_
    rework_pending below), so this simulates the stuck shape the route is
    FOR -- verdict on record, head unchanged, but the issue's rework
    bookkeeping lost (the orphaned-issue class reconcile normalizes to the
    passive reviewing status).
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )
    _force_issue_status(app, 123, "reviewing")

    # Same head, same diff as the recorded verdict: no actual content change.
    result = app.review(456)

    assert result.ok is True
    assert result.data["routed_to_rework"] is True
    assert result.data["rework_reason"] == "no_op_rework"

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["no_op_rework_attempts"] == 1


def test_janitor_no_op_rework_waits_while_rework_pending(tmp_path: Path) -> None:
    """While the issue is already in ``rework_requested`` (the normal state
    right after a request_changes verdict), the no-op signal means the
    rework worker hasn't produced anything YET -- repeated review() passes
    must wait passively, not burn routing attempts (which would escalate a
    PR whose worker simply hasn't run within two loop passes; pending
    cycles are bounded by dispatch_rework's redispatch cap instead).
    """
    config = OrchestratorConfig(review=ReviewConfig(max_no_op_rework_attempts=2))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    for _ in range(4):
        result = app.review(456)
        assert result.ok is False
        assert "janitor gate blocked" in result.message

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"].get("no_op_rework_attempts", 0) == 0
    assert (123, app.config.labels.human_needed) not in app.gh.labels_added


def test_janitor_no_op_rework_cap_exceeded_escalates_with_label(tmp_path: Path) -> None:
    """Cap mechanics of the shared wrapper for the no-op reason: each
    re-entry from a non-pending stuck state consumes an attempt, and past
    ``max_no_op_rework_attempts`` the issue escalates with the label
    actually applied. Re-entry requires the pending status to be lost again
    (simulated here); in normal operation the pending-guard makes this
    unreachable and dispatch-side caps bound the pending cycles.
    """
    config = OrchestratorConfig(review=ReviewConfig(max_no_op_rework_attempts=2))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    _force_issue_status(app, 123, "reviewing")
    result1 = app.review(456)
    assert result1.ok is True
    _force_issue_status(app, 123, "reviewing")
    result2 = app.review(456)
    assert result2.ok is True
    _force_issue_status(app, 123, "reviewing")
    result3 = app.review(456)

    assert result3.ok is False
    assert result3.data["escalated"] is True

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["prs"]["456"]["status"] == "escalated"
    assert (123, app.config.labels.human_needed) in app.gh.labels_added


def test_janitor_conflict_with_failed_required_check_still_routes_to_conflict_rework(
    tmp_path: Path,
) -> None:
    """Mirrors the real PR #500 case from the investigation (cost-spirals.md
    Finding 1): a PR that is BOTH CONFLICTING and has a failed required
    check. ``is_check_failure_block`` is False here (janitor.py:
    ``bool(failed_required_checks) and not failures`` -- ``failures`` is
    already non-empty from the conflict check, which runs first), so this
    combination fell all the way through to the old ``janitor_blocked``
    dead end even though a check was failing. The merge-conflict route
    (unlike the no-op-rework route, which deliberately excludes this
    combination -- see test_janitor_required_check_failure_noop_does_not_
    reroute, issue #376) is NOT check-failure-gated: a conflicting branch
    needs a rebase regardless of what else is also failing, so this combo
    must still route to conflict rework.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed",),
            enabled=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubFailingCheck(FakeGitHub):
        def pr_checks(self, number: int):
            return [{"name": "Tests passed", "state": "FAILURE"}]

    fake_gh = FakeGitHubFailingCheck()
    fake_gh.prs[0]["mergeable"] = "CONFLICTING"
    fake_gh.prs[0]["mergeStateStatus"] = "DIRTY"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _set_decision(app, 456, "request_changes")

    result = app.review(456)

    assert result.ok is True
    assert result.data["routed_to_rework"] is True
    assert result.data["rework_reason"] == "merge_conflict"
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1


def test_clean_janitor_pass_resets_epoch_counters(tmp_path: Path) -> None:
    """A clean janitor pass with an AFFIRMATIVE mergeable signal ends the
    conflict/no-op epoch: stale attempts from a long-resolved conflict must
    not count against a genuinely new, unrelated conflict weeks later.
    """
    app = _conflicting_app(tmp_path)
    _set_decision(app, 456, "request_changes")

    result = app.review(456)
    assert result.data["routed_to_rework"] is True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1

    # Worker resolves the conflict and pushes: janitor goes clean.
    app.gh.prs[0]["mergeable"] = "MERGEABLE"
    app.gh.prs[0]["mergeStateStatus"] = "CLEAN"
    app.gh.pr_head_shas[456] = "sha-resolved"
    _force_issue_status(app, 123, "reviewing")
    clean_result = app.review(456)
    assert "janitor gate blocked" not in clean_result.message

    state = load_state(app.paths.state_file)
    pr = state["prs"]["456"]
    assert pr["conflict_rework_attempts"] == 0
    assert pr["conflict_rework_attempts_last_head"] is None
    assert pr["no_op_rework_attempts"] == 0
    assert pr["no_op_rework_attempts_last_head"] is None


def test_clean_janitor_pass_with_unknown_mergeable_preserves_conflict_epoch(
    tmp_path: Path,
) -> None:
    """GitHub reports mergeable UNKNOWN for a window after every push while
    it recomputes, and the janitor's conflict check only fails on
    CONFLICTING/DIRTY -- a clean pass during that window is NOT evidence the
    conflict was resolved. Resetting on it would let a flapping PR
    relitigate its attempt cap forever; the no-op epoch (whose signal is
    content movement, not mergeability) still resets.
    """
    app = _conflicting_app(tmp_path)
    _set_decision(app, 456, "request_changes")

    result = app.review(456)
    assert result.data["routed_to_rework"] is True

    # Recompute window: not CONFLICTING, not affirmatively MERGEABLE either.
    app.gh.prs[0]["mergeable"] = "UNKNOWN"
    app.gh.prs[0]["mergeStateStatus"] = "UNKNOWN"
    app.gh.pr_head_shas[456] = "sha-maybe-resolved"
    _force_issue_status(app, 123, "reviewing")
    flap_result = app.review(456)
    assert "janitor gate blocked" not in flap_result.message

    state = load_state(app.paths.state_file)
    pr = state["prs"]["456"]
    assert pr["conflict_rework_attempts"] == 1
    assert pr["no_op_rework_attempts"] == 0


# ---------------------------------------------------------------------------
# Issue #558: single-point-of-enforcement -- review() converges a CLOSED
# (unmerged) PR's state entry to "closed" at the janitor-gate boundary
# instead of falling through to janitor_blocked bookkeeping.
# ---------------------------------------------------------------------------


def _force_pr_status(app: OrchestratorApp, pr_number: int, status: str) -> None:
    state = load_state(app.paths.state_file)
    record = {**state["prs"].get(str(pr_number), {}), "number": pr_number}
    record["status"] = status
    state["prs"][str(pr_number)] = record
    save_state(app.paths.state_file, state)


def test_review_converges_closed_unmerged_pr_janitor_blocked(tmp_path: Path) -> None:
    """A PR that is CLOSED (unmerged) on GitHub with a 'janitor_blocked'
    state status must be converged to 'closed' by review() at the janitor
    gate boundary, not re-logged as janitor_blocked.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _force_pr_status(app, 456, "janitor_blocked")

    result = app.review(456)

    assert result.ok is True
    assert result.data["closed_unmerged_converged"] is True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["status"] == "closed"


def test_review_converges_closed_unmerged_pr_rework_requested(tmp_path: Path) -> None:
    """A PR that is CLOSED (unmerged) on GitHub with a 'rework_requested'
    state status must be converged to 'closed' by review().
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _force_pr_status(app, 456, "rework_requested")

    result = app.review(456)

    assert result.ok is True
    assert result.data["closed_unmerged_converged"] is True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["status"] == "closed"


def test_review_converges_closed_unmerged_pr_already_closed_is_idempotent(
    tmp_path: Path,
) -> None:
    """A PR already converged to 'closed' must not re-emit the convergence
    event on a second review() call (idempotent).

    The idempotency guard in review() skips the state write AND the
    ``closed_unmerged_pr_state_converged`` event when the PR status is
    already ``"closed"``. Asserting only the return flag / final status
    would pass even if that guard were deleted (the flag is returned
    outside the guard, and writing ``"closed"`` over ``"closed"`` is a
    value no-op), so this test pins the guard by counting convergence
    events: exactly one must land on the first call, and zero on the
    second.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _force_pr_status(app, 456, "janitor_blocked")

    # First call: status is janitor_blocked, so the guard writes and emits.
    first = app.review(456)
    assert first.ok is True
    assert first.data.get("closed_unmerged_converged") is True
    state_after_first = load_state(app.paths.state_file)
    assert state_after_first["prs"]["456"]["status"] == "closed"
    first_events = [
        e
        for e in state_after_first.get("events", [])
        if e.get("kind") == "closed_unmerged_pr_state_converged"
    ]
    assert len(first_events) == 1, "first call must emit the convergence event"

    # Second call: status is already 'closed'. The guard must skip the
    # write and the event -- deleting the guard would re-emit here.
    second = app.review(456)
    assert second.ok is True
    assert second.data.get("closed_unmerged_converged") is True
    state_after_second = load_state(app.paths.state_file)
    assert state_after_second["prs"]["456"]["status"] == "closed"
    second_events = [
        e
        for e in state_after_second.get("events", [])
        if e.get("kind") == "closed_unmerged_pr_state_converged"
    ]
    assert len(second_events) == 1, (
        "second call on an already-closed PR must NOT re-emit the "
        "convergence event (idempotency guard)"
    )


def test_review_does_not_converge_merged_pr(tmp_path: Path) -> None:
    """A MERGED PR must NOT be converged by the closed-unmerged path --
    that is merged_outside_orchestrator's job. The janitor gate should
    fall through to its normal failure handling for MERGED.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["state"] = "MERGED"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _force_pr_status(app, 456, "reviewing")

    result = app.review(456)

    # MERGED is not CLOSED, so the closed-unmerged convergence must not fire.
    assert result.data.get("closed_unmerged_converged") is None
    state = load_state(app.paths.state_file)
    # Status should not have been set to "closed" by this path.
    assert state["prs"]["456"]["status"] != "closed"


# ---------------------------------------------------------------------------
# Issue #558 rework: review()'s closed-unmerged early return yields ok=True
# with a ``closed_unmerged_converged`` data flag and NO packet. Two callers
# gate a status->"reviewing" flip on ``review_result.ok && !routed_to_rework
# && decision_unchanged`` after snapshotting the PR as OPEN earlier in the
# same pass. If the PR closes in that window, the flip must NOT fire -- it
# would strand the issue in an ACTIVE_STATE_STATUS no reconcile rule clears
# while the GitHub issue itself stays open.
# ---------------------------------------------------------------------------


class _RaceClosedGitHub(FakeGitHub):
    """Models the close-mid-pass race: ``pr_list`` (the snapshot taken
    earlier in the pass) still sees the PR as OPEN, but ``pr_view`` (called
    inside ``review()``) observes it as CLOSED. This is exactly the window
    the rework finding describes.
    """

    def pr_list(self):
        out = []
        for pr in self.prs:
            if pr["number"] != 456:
                continue
            pr_copy = dict(pr)
            pr_copy["state"] = "OPEN"
            if pr["number"] in self.pr_head_shas:
                pr_copy["headRefOid"] = self.pr_head_shas[pr["number"]]
            out.append(pr_copy)
        return out

    def pr_view(self, number: int):
        for pr in self.prs:
            if pr["number"] == number:
                pr_copy = dict(pr)
                pr_copy["state"] = "CLOSED"
                if number in self.pr_head_shas:
                    pr_copy["headRefOid"] = self.pr_head_shas[number]
                return pr_copy
        raise ValueError(f"PR {number} not found")


def test_route_rework_candidate_to_review_does_not_flip_when_pr_closes_mid_pass(
    tmp_path: Path,
) -> None:
    """``_route_rework_candidate_to_review`` (reached via ``dispatch_rework``)
    snapshots the PR as OPEN to detect a head-advance, then calls ``review()``
    which observes the PR as CLOSED and converges it. The issue must stay
    ``rework_requested`` -- NOT flip to ``reviewing`` -- because no fresh
    packet was produced and the PR is dead. Flipping would strand the issue
    in ``reviewing`` (an ACTIVE_STATE_STATUS) while the GitHub issue stays
    open, with no reconcile rule to clear it.
    """
    import sys

    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _RaceClosedGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+first"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Head advances AND the diff content genuinely changes (so dispatch_rework
    # routes to _route_rework_candidate_to_review). pr_list still reports OPEN.
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+second"
    )
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    fake_gh.labels_added.clear()
    fake_gh.labels_removed.clear()

    result = app.dispatch_rework()

    assert result.ok is True
    # The issue was NOT confirmed as routed to review (no fresh packet).
    assert result.data["routed_to_review"] == []
    assert 123 in result.data["review_blocked_retry"]

    state = load_state(paths.state_file)
    # Issue stays rework_requested -- the dead-PR convergence is not a
    # "reviewing" transition.
    assert state["issues"]["123"]["status"] == "rework_requested"
    # PR state entry converged to "closed" by review()'s janitor gate.
    assert state["prs"]["456"]["status"] == "closed"
    # No reviewing label applied (that would desync from GitHub reality).
    assert (123, app.config.labels.reviewing) not in fake_gh.labels_added

    rework_events = [e for e in state["events"] if e["kind"] == "rework_already_pushed"]
    assert len(rework_events) == 1
    assert rework_events[0]["payload"]["routed"] is False
    assert rework_events[0]["payload"]["review_ok"] is True


def test_orphan_sweep_does_not_flip_to_reviewing_when_pr_closes_mid_pass(
    tmp_path: Path,
) -> None:
    """The dead-worker orphan sweep snapshots the PR as OPEN (via
    ``pr_list``) to detect a head-advance, then calls ``review()`` which
    observes the PR as CLOSED and converges it. The orphaned ``dispatched``
    issue must stay ``dispatched`` -- NOT flip to ``reviewing`` and NOT get a
    transient-block drift fingerprint -- because the PR is permanently dead,
    not transiently blocked. The issue's disposition is left to the existing
    closed-unmerged issue-side reconcile handling.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = _RaceClosedGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Record a request_changes verdict so the orphan sweep sees a head-advance
    # and routes to review(). reviewed_head_sha pins to the default head.
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+first"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    state = load_state(paths.state_file)
    state["issues"]["123"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)

    # Head advances (pr_list snapshot sees OPEN + new head); pr_view will
    # report CLOSED when review() runs.
    fake_gh.pr_head_shas[456] = "sha-new-head"

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        from charlie_work.workflow import _detect_and_handle_orphaned_workers

        sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            fake_gh,
            review_callback=app.review,
        )

    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    # Stays dispatched -- not flipped to "reviewing".
    assert entry.get("status") == "dispatched"
    # No transient-block drift fingerprint on a permanently-dead PR.
    assert "orphan_drift_fingerprint" not in entry
    # PR state entry converged to "closed" by review()'s janitor gate.
    assert state["prs"]["456"]["status"] == "closed"

    events = state.get("events", [])
    drift_events = [e for e in events if e.get("kind") == "orphaned_worker_drift"]
    assert len(drift_events) == 0
    routed_events = [e for e in events if e.get("kind") == "orphaned_worker_routed_to_review"]
    assert len(routed_events) == 1
    assert routed_events[0]["payload"]["routed"] is False
    assert routed_events[0]["payload"]["review_ok"] is True


# ---------------------------------------------------------------------------
# Issue #765: stall bound orthogonal to the settled-head signal. The cap
# checks above (max_conflict_rework_attempts / max_no_op_rework_attempts)
# are only reachable via a SETTLED head change (merge_conflict) or a fresh
# non-pending detection (no_op_rework) -- both require the issue to leave
# rework_requested at least once. A PR whose head simply stops moving while
# queued (rework_requested, nobody dispatched) never produces that signal
# and previously waited in the passive janitor_blocked branch forever (live
# evidence: PR #696, 55 rework_already_pushed events, attempts already at
# cap, rescue tier never even attempted because the cap check was
# unreachable). These tests inject an already-elapsed
# "{attempts_key}_stall_since" timestamp directly into state -- the same
# past-timestamp-injection pattern test_worker_health.py uses for
# stall_minutes -- rather than sleeping or monkeypatching utc_now.
# ---------------------------------------------------------------------------


def test_janitor_conflict_stalled_rework_requested_escalates(tmp_path: Path) -> None:
    """A merge-conflict rework that never settles -- head unchanged while
    the issue sits in ``rework_requested`` (queued, nobody working it) --
    must escalate once ``rework_stall_minutes`` elapses, independent of the
    attempt cap. This is the exact PR #696 shape: attempts already at 1 (of
    a cap of 2), head frozen, status stuck at rework_requested.
    """
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=2, rework_stall_minutes=60),
    )
    _set_decision(app, 456, "request_changes")

    result1 = app.review(456)
    assert result1.ok is True
    assert result1.data["routed_to_rework"] is True

    # Same head, same status, but the stall clock started 61 minutes ago --
    # past the 60-minute threshold. Anchored to the current (unmoved) head,
    # matching what the real first-passive-wait code path writes -- a
    # stall_since with no matching stall_head anchor is legacy/untrusted
    # state, covered separately by
    # test_janitor_conflict_legacy_stall_since_without_head_reanchors_instead_of_escalating.
    state = load_state(app.paths.state_file)
    state["prs"]["456"]["conflict_rework_attempts_stall_since"] = (
        datetime.now(UTC) - timedelta(minutes=61)
    ).isoformat()
    state["prs"]["456"]["conflict_rework_attempts_stall_head"] = app.gh.prs[0].get("headRefOid")
    save_state(app.paths.state_file, state)

    result2 = app.review(456)

    assert result2.ok is False
    assert result2.data["escalated"] is True
    assert result2.data["escalation_reason"] == "stalled"

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["prs"]["456"]["status"] == "escalated"
    # The stall path never burns an attempt -- it is a distinct escalation
    # reason from the cap, not a disguised extra cap check.
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1
    assert (123, app.config.labels.human_needed) in app.gh.labels_added

    stalled_events = [e for e in state["events"] if e["kind"] == "janitor_rework_stalled"]
    assert len(stalled_events) == 1
    assert stalled_events[0]["payload"]["reason"] == "merge_conflict"
    assert stalled_events[0]["payload"]["pr_number"] == 456
    assert stalled_events[0]["payload"]["issue_number"] == 123
    # The stalled head must be in the payload -- without it, confirming from
    # the event stream alone that the head genuinely didn't move requires
    # reconstructing state.json snapshots after the fact (exactly what was
    # needed to forensically confirm PR #696's real escalation mechanism).
    assert stalled_events[0]["payload"]["head_sha"] == app.gh.prs[0].get("headRefOid")
    # Distinguishable in the event stream from the cap-exceeded escalation.
    cap_events = [e for e in state["events"] if e["kind"] == "janitor_rework_escalated"]
    assert len(cap_events) == 0


def test_janitor_conflict_stall_escalation_blocks_reentry_until_unescalated(
    tmp_path: Path,
) -> None:
    """Issue #776 interaction with the #765/#774 stall bound: a stall
    escalation leaves the issue in status "escalated" -- NOT
    rework_requested/dispatched/dispatch_pending -- exactly like a
    cap-exceeded escalation does. Issue #776's fix makes
    _route_janitor_gate_failure_to_rework re-enter on every pass for an
    "escalated" issue/PR (by design, so an UNRELATED lane's escalation
    doesn't wall off this lane's remediation forever). Without a
    lane-scoped escalation_reason recorded on the stall path too, that
    re-entry would fall through the (now False) rework_pending check
    straight to the attempts-increment/dispatch logic on the very next
    pass -- silently redispatching a fresh rework attempt and undoing the
    stall escalation's entire purpose (getting a human to look at a PR
    nobody is actively working) the instant it fires. This pins that the
    stall path's escalation_reason is recognized by the same guard the
    cap-exceeded path uses, and that ``charlie unescalate`` remains the
    sanctioned way back in.
    """
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=2, rework_stall_minutes=60),
    )
    _set_decision(app, 456, "request_changes")

    result1 = app.review(456)
    assert result1.ok is True
    assert result1.data["routed_to_rework"] is True

    state = load_state(app.paths.state_file)
    state["prs"]["456"]["conflict_rework_attempts_stall_since"] = (
        datetime.now(UTC) - timedelta(minutes=61)
    ).isoformat()
    state["prs"]["456"]["conflict_rework_attempts_stall_head"] = app.gh.prs[0].get("headRefOid")
    save_state(app.paths.state_file, state)

    result2 = app.review(456)
    assert result2.ok is False
    assert result2.data["escalated"] is True

    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["escalation_reason"] == "conflict_rework_attempts_stall_exceeded"
    assert state["issues"]["123"]["escalation_reason"] == "conflict_rework_attempts_stall_exceeded"
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1

    # The regression this pins: without the fix, this third pass would
    # silently redispatch (attempts_key bumped to 2, a new
    # routed_to_rework/janitor_rework_escalated event) instead of staying
    # blocked behind the escalated-issue early return.
    result3 = app.review(456)
    assert result3.ok is True
    assert result3.data.get("pass_skipped") is True
    assert result3.data.get("routed_to_rework") is not True

    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["issues"]["123"]["status"] == "escalated"
    escalated_events = [e for e in state["events"] if e["kind"] == "janitor_rework_escalated"]
    assert len(escalated_events) == 0
    stalled_events = [e for e in state["events"] if e["kind"] == "janitor_rework_stalled"]
    assert len(stalled_events) == 1  # only the original stall escalation, not re-fired

    # charlie unescalate is still the sanctioned re-arm: it clears
    # escalation_reason and the attempts counter on both records so a
    # still-conflicting PR gets a genuinely fresh budget afterward.
    unescalate_result = app.unescalate(issue_number=123)
    assert unescalate_result.ok is True

    state = load_state(app.paths.state_file)
    assert "escalation_reason" not in state["prs"]["456"]
    assert "escalation_reason" not in state["issues"]["123"]
    assert "conflict_rework_attempts" not in state["prs"]["456"]

    _set_decision(app, 456, "request_changes")
    result4 = app.review(456)
    assert result4.ok is True
    assert result4.data["routed_to_rework"] is True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1


def test_janitor_no_op_rework_stalled_escalates(tmp_path: Path) -> None:
    """The no-op-rework early return has no settled-head concept at all --
    head-unchanged IS the detection signal -- so before this fix it could
    NEVER reach the cap/rescue check no matter how long it sat in
    rework_requested doing nothing (issue #765 calls this out explicitly:
    "the current early return blocks no-op unconditionally"). The stall
    bound must fire for this reason too.
    """
    config = OrchestratorConfig(
        review=ReviewConfig(max_no_op_rework_attempts=2, rework_stall_minutes=60)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )
    # record_review leaves the issue in rework_requested directly -- the
    # common real-world shape: no_op_rework's own "fresh route" baseline
    # write is never reached because the issue is already pending by the
    # time the janitor first observes it.

    # First pass: no stall clock recorded yet -- it starts here.
    result1 = app.review(456)
    assert result1.ok is False
    assert "janitor gate blocked" in result1.message
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"].get("no_op_rework_attempts", 0) == 0
    assert state["prs"]["456"].get("no_op_rework_attempts_stall_since") is not None

    # Backdate the clock past the threshold.
    state["prs"]["456"]["no_op_rework_attempts_stall_since"] = (
        datetime.now(UTC) - timedelta(minutes=61)
    ).isoformat()
    save_state(app.paths.state_file, state)

    result2 = app.review(456)

    assert result2.ok is False
    assert result2.data["escalated"] is True
    assert result2.data["escalation_reason"] == "stalled"

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["prs"]["456"].get("no_op_rework_attempts", 0) == 0
    assert (123, app.config.labels.human_needed) in app.gh.labels_added

    stalled_events = [e for e in state["events"] if e["kind"] == "janitor_rework_stalled"]
    assert len(stalled_events) == 1
    assert stalled_events[0]["payload"]["reason"] == "no_op_rework"


def test_janitor_conflict_stall_clock_under_threshold_still_waits(tmp_path: Path) -> None:
    """Regression guard: a stall clock that has started but has not yet
    crossed ``rework_stall_minutes`` must still fall through to the passive
    wait -- this pins that the fix does not make escalation fire MORE
    eagerly, only reachable for a genuinely-stalled PR. This test passes
    both before and after the fix (before: the field is inert; after: 5min
    < 60min threshold) -- it is a guard against a regression, not proof the
    fix works.
    """
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=2, rework_stall_minutes=60),
    )
    _set_decision(app, 456, "request_changes")
    result1 = app.review(456)
    assert result1.data["routed_to_rework"] is True

    state = load_state(app.paths.state_file)
    state["prs"]["456"]["conflict_rework_attempts_stall_since"] = (
        datetime.now(UTC) - timedelta(minutes=5)
    ).isoformat()
    save_state(app.paths.state_file, state)

    result2 = app.review(456)

    assert result2.ok is False
    assert result2.data.get("escalated") is not True
    assert "janitor gate blocked" in result2.message
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert (123, app.config.labels.human_needed) not in app.gh.labels_added


def test_janitor_conflict_dispatched_status_holds_stall_clock(tmp_path: Path) -> None:
    """A live ``dispatched`` session must never TRIP the stall clock -- only
    a queued, nobody-working ``rework_requested`` PR is a stall candidate,
    and a genuinely dead ``dispatched`` session is WatchdogConfig's job
    (``stall_minutes``), not this cap's, so escalating a live worker here
    would be a false alarm.

    But with the head unchanged, ``dispatched`` must HOLD the clock rather
    than clear it (revised from an earlier draft that cleared on any
    non-``rework_requested`` status): a status-only signal is not reliable
    enough to mean "clear", because reconcile's
    ``issue_active_label_with_open_pr`` self-heal flips ``issue_status`` away
    from ``rework_requested`` on its own periodic cadence without the head
    moving at all -- if that cleared the clock, it would reset on
    essentially every reconcile pass and the bound could never fire whenever
    reconcile is enabled (the common case; see
    test_janitor_conflict_reconcile_status_oscillation_does_not_reset_stall_clock).
    So a same-head status flip -- dispatched or reconcile-driven -- holds:
    no escalation now, but the accumulated timestamp survives so idle
    dispatched time still counts toward the bound once the issue is back in
    rework_requested.
    """
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=2, rework_stall_minutes=60),
    )
    _set_decision(app, 456, "request_changes")
    result1 = app.review(456)
    assert result1.data["routed_to_rework"] is True

    _force_issue_status(app, 123, "dispatched")
    # A stale clock somehow already on record -- 10 hours old, far past the
    # 60-minute threshold -- must NOT trip escalation while dispatched...
    stale_since = (datetime.now(UTC) - timedelta(hours=10)).isoformat()
    state = load_state(app.paths.state_file)
    state["prs"]["456"]["conflict_rework_attempts_stall_since"] = stale_since
    state["prs"]["456"]["conflict_rework_attempts_stall_head"] = state["prs"]["456"].get(
        "conflict_rework_attempts_last_head"
    )
    save_state(app.paths.state_file, state)

    result2 = app.review(456)

    assert result2.ok is False
    assert result2.data.get("escalated") is not True
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"
    assert (123, app.config.labels.human_needed) not in app.gh.labels_added
    # ...and, being held rather than cleared, the timestamp survives.
    assert state["prs"]["456"].get("conflict_rework_attempts_stall_since") == stale_since


def test_janitor_conflict_dispatched_status_with_head_progress_clears_stall_clock(
    tmp_path: Path,
) -> None:
    """Unlike a same-head status flip (held, see the test above), a real
    head change while ``dispatched`` -- the worker actually pushing a commit
    -- IS progress and must clear the clock, even though ``dispatched``
    itself is never a stall candidate. This exercises
    ``_check_janitor_rework_stall``'s own head-comparison clear, distinct
    from the settled-cycle burn-attempt branch in the caller (which only
    fires for ``issue_status == "rework_requested"``, never ``dispatched``).
    """
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=2, rework_stall_minutes=60),
    )
    _set_decision(app, 456, "request_changes")
    result1 = app.review(456)
    assert result1.data["routed_to_rework"] is True

    _force_issue_status(app, 123, "dispatched")
    state = load_state(app.paths.state_file)
    state["prs"]["456"]["conflict_rework_attempts_stall_since"] = (
        datetime.now(UTC) - timedelta(hours=10)
    ).isoformat()
    state["prs"]["456"]["conflict_rework_attempts_stall_head"] = state["prs"]["456"].get(
        "conflict_rework_attempts_last_head"
    )
    save_state(app.paths.state_file, state)

    # The dispatched worker pushes a new commit.
    app.gh.pr_head_shas[456] = "sha-wip-push"
    result2 = app.review(456)

    assert result2.ok is False
    assert result2.data.get("escalated") is not True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"].get("conflict_rework_attempts_stall_since") is None
    assert state["prs"]["456"].get("conflict_rework_attempts_stall_head") is None


def test_janitor_conflict_reconcile_status_oscillation_does_not_reset_stall_clock(
    tmp_path: Path,
) -> None:
    """The discriminating regression test for the reconcile interaction
    found while building this fix: reconcile's
    ``issue_active_label_with_open_pr`` self-heal periodically normalizes a
    stale active label on an issue with an open PR, and as part of that fix
    rewrites ``state["issues"][n]["status"]`` away from ``rework_requested``
    (to the passive "reviewing" placeholder) -- WITHOUT touching the PR
    branch at all. The very next ``review()`` pass then sees a non-pending
    status and takes ``_route_janitor_gate_failure_to_rework``'s FRESH-ROUTE
    branch (not the passive-wait branch): it burns an attempt and calls the
    router, which puts the issue straight back into ``rework_requested`` --
    same head throughout, no real progress. An earlier draft of this fix
    unconditionally cleared the stall clock in that fresh-route branch
    (``route_extra_state = {attempts_key: attempts, stall_since_key: None}``),
    which this oscillation would trip on every reconcile pass (observed
    cadence ~30min in production, well under any reasonable stall
    threshold), making the bound practically unreachable whenever reconcile
    is enabled -- the common case. This test fails against that earlier
    draft (the post-flip assertion below) and passes against the
    head-keyed hold/reset design, where the fresh-route branch no longer
    touches the stall keys at all and the caller's merge (``_route_to_
    rework``, itself re-reading state fresh under its own lock) preserves
    whatever was already on record.
    """
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=2, rework_stall_minutes=60),
    )
    _set_decision(app, 456, "request_changes")
    result1 = app.review(456)
    assert result1.data["routed_to_rework"] is True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1

    # Step 1: clock starts on the first passive wait, head H, attempts still 1
    # (the not-settled branch never burns).
    result_wait = app.review(456)
    assert result_wait.ok is False
    state = load_state(app.paths.state_file)
    started_at = state["prs"]["456"].get("conflict_rework_attempts_stall_since")
    assert started_at is not None
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1

    # Step 2: reconcile-style status flip away from rework_requested, head
    # unchanged. review() now takes the fresh-route branch: it burns an
    # attempt (1 -> 2, exactly at the cap but not exceeding it) and the
    # router puts the issue straight back into rework_requested. The stall
    # clock must survive this round-trip, not reset.
    _force_issue_status(app, 123, "reviewing")
    result_flip = app.review(456)
    assert result_flip.ok is True
    assert result_flip.data["routed_to_rework"] is True
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["conflict_rework_attempts"] == 2
    assert state["prs"]["456"].get("conflict_rework_attempts_stall_since") == started_at

    # Step 3: another passive wait, status and head both unchanged since the
    # flip -- still short of the threshold, so it must keep waiting rather
    # than escalate, and must not burn a further attempt (back in the
    # not-settled branch, not fresh-route, since status is rework_requested
    # again).
    result_wait2 = app.review(456)
    assert result_wait2.ok is False
    assert result_wait2.data.get("escalated") is not True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["conflict_rework_attempts"] == 2
    assert state["prs"]["456"].get("conflict_rework_attempts_stall_since") == started_at

    # Step 4: back-date that same original timestamp past the threshold and
    # confirm escalation fires -- proving the elapsed time carried through
    # the oscillation rather than resetting when reconcile intervened.
    state["prs"]["456"]["conflict_rework_attempts_stall_since"] = (
        datetime.now(UTC) - timedelta(minutes=61)
    ).isoformat()
    save_state(app.paths.state_file, state)
    result_final = app.review(456)
    assert result_final.ok is False
    assert result_final.data["escalated"] is True
    assert result_final.data["escalation_reason"] == "stalled"
    state = load_state(app.paths.state_file)
    # The stall path escalates independent of the cap -- attempts stays at
    # whatever the fresh-route burn left it at, not bumped further.
    assert state["prs"]["456"]["conflict_rework_attempts"] == 2


def test_janitor_conflict_head_progress_clears_stall_clock(tmp_path: Path) -> None:
    """A settled head change (a rework cycle completing, even one that still
    conflicts) is real progress and must clear any stall clock that had
    started -- a later re-stall must count elapsed time from zero, not from
    a timestamp accumulated during unrelated history before this cycle.
    """
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=3, rework_stall_minutes=60),
    )
    _set_decision(app, 456, "request_changes")
    result1 = app.review(456)
    assert result1.data["routed_to_rework"] is True

    # Clock starts on the first passive wait.
    result_wait = app.review(456)
    assert result_wait.ok is False
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"].get("conflict_rework_attempts_stall_since") is not None

    # Head moves (a settled cycle completes, still conflicting): the clock
    # must clear even though the cap has not been hit.
    app.gh.pr_head_shas[456] = "sha-cycle-2"
    result2 = app.review(456)
    assert result2.ok is False
    assert result2.data.get("escalated") is not True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["conflict_rework_attempts"] == 2
    assert state["prs"]["456"].get("conflict_rework_attempts_stall_since") is None


def test_janitor_conflict_legacy_stall_since_without_head_reanchors_instead_of_escalating(
    tmp_path: Path,
) -> None:
    """A ``_stall_since`` timestamp with no matching ``_stall_head`` (state
    written before this bound existed, or any other path that could set one
    key without the other) has unknown provenance: it is impossible to tell
    whether the head moved during that already-elapsed time, since nothing
    was ever compared against. Trusting it would let a single deploy of this
    fix instantly escalate every PR that happened to have an old, unrelated
    timestamp sitting in state. The safe direction is to re-anchor (discard
    the timestamp, start fresh against the current head) rather than
    escalate off data collected before the head-comparison existed.
    """
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=2, rework_stall_minutes=60),
    )
    _set_decision(app, 456, "request_changes")
    result1 = app.review(456)
    assert result1.data["routed_to_rework"] is True

    # Simulate legacy state: a stale stall_since with no stall_head at all,
    # far past the threshold.
    state = load_state(app.paths.state_file)
    state["prs"]["456"]["conflict_rework_attempts_stall_since"] = (
        datetime.now(UTC) - timedelta(hours=10)
    ).isoformat()
    assert "conflict_rework_attempts_stall_head" not in state["prs"]["456"]
    save_state(app.paths.state_file, state)

    # Issue #828: unlike loop(now=...) (added by PR 836/838), review() has no
    # injectable-clock seam reaching `_check_janitor_rework_stall`'s
    # re-anchor write -- adding one means a new keyword-only parameter on
    # review(), a public entry point with production callers (cli.py,
    # fleet_dispatch.py) that would never pass it. Bracketing the call with
    # real before/after timestamps removes the wall-clock-tolerance window
    # without that production diff: the written timestamp is mathematically
    # bounded by the call's own start and end, so no stall between the call
    # and the assertion can widen a gap the way a fixed-N-second tolerance
    # eventually would. `before` is floored to whole seconds because the
    # re-anchor write (`utc_now()`, state.py) truncates microseconds, so a
    # `before` sampled with nonzero microseconds could otherwise sit above
    # the truncated write.
    before = datetime.now(UTC).replace(microsecond=0)
    result2 = app.review(456)
    after = datetime.now(UTC)

    # Must NOT escalate off the untrustworthy legacy timestamp.
    assert result2.ok is False
    assert result2.data.get("escalated") is not True
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert (123, app.config.labels.human_needed) not in app.gh.labels_added

    # The clock was re-anchored, not left stale or cleared to None: it must
    # now be recent (not the 10-hour-old value) and carry a head anchor.
    new_since = state["prs"]["456"].get("conflict_rework_attempts_stall_since")
    assert new_since is not None
    reanchored_at = datetime.fromisoformat(new_since)
    assert before <= reanchored_at <= after
    assert state["prs"]["456"].get("conflict_rework_attempts_stall_head") == app.gh.prs[0].get(
        "headRefOid"
    )

    # And the re-anchored clock behaves normally going forward: still under
    # threshold, no escalation yet.
    result3 = app.review(456)
    assert result3.ok is False
    assert result3.data.get("escalated") is not True


def test_janitor_no_op_rework_reconcile_status_oscillation_does_not_reset_stall_clock(
    tmp_path: Path,
) -> None:
    """Coverage-insurance sibling of
    ``test_janitor_conflict_reconcile_status_oscillation_does_not_reset_stall_clock``
    for the ``no_op_rework`` reason. The no-op detection
    (``verdict.is_no_op_rework``) is recomputed from the janitor gate every
    pass from the PR's diff/patch-id, entirely independent of
    ``issue_status`` -- so a reconcile-style status flip away from
    ``rework_requested`` does not stop no-op detection from firing on the
    next pass either. It flows through the exact same shared
    ``_route_janitor_gate_failure_to_rework`` function and the exact same
    fresh-route branch as the merge_conflict case (this is already visible
    in ``test_janitor_no_op_rework_routes_to_rework``, which forces
    ``issue_status`` to ``reviewing`` before the route fires), so the fix
    (no longer unconditionally clearing the stall keys in that branch's
    ``route_extra_state``) protects both reasons identically, not just
    ``merge_conflict``.
    """
    config = OrchestratorConfig(
        review=ReviewConfig(max_no_op_rework_attempts=2, rework_stall_minutes=60)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Clock starts on the first passive wait (issue already rework_requested).
    result_wait = app.review(456)
    assert result_wait.ok is False
    state = load_state(app.paths.state_file)
    started_at = state["prs"]["456"].get("no_op_rework_attempts_stall_since")
    assert started_at is not None
    assert state["prs"]["456"].get("no_op_rework_attempts", 0) == 0

    # Reconcile-style status flip, head/diff unchanged: fresh-route fires,
    # burns an attempt, router puts the issue back in rework_requested.
    _force_issue_status(app, 123, "reviewing")
    result_flip = app.review(456)
    assert result_flip.ok is True
    assert result_flip.data["routed_to_rework"] is True
    assert result_flip.data["rework_reason"] == "no_op_rework"
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["no_op_rework_attempts"] == 1
    assert state["prs"]["456"].get("no_op_rework_attempts_stall_since") == started_at

    # Back-date the surviving timestamp past threshold: escalation fires,
    # proving the clock carried through the oscillation instead of resetting.
    state["prs"]["456"]["no_op_rework_attempts_stall_since"] = (
        datetime.now(UTC) - timedelta(minutes=61)
    ).isoformat()
    save_state(app.paths.state_file, state)
    result_final = app.review(456)
    assert result_final.ok is False
    assert result_final.data["escalated"] is True
    assert result_final.data["escalation_reason"] == "stalled"
