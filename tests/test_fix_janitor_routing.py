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

from charlie_work.config import OrchestratorConfig, ReviewConfig
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
