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
    assert result4.data.get("skipped") is True
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
    app.record_review(456, "request_changes", summary="fix A")
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
    app.record_review(456, "request_changes", summary="fix A")

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
    app.record_review(456, "request_changes", summary="fix A")

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
    app.record_review(456, "request_changes", summary="fix A")

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
    app.record_review(456, "request_changes", summary="fix A")

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
