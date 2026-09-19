"""Review-queue stale-CI verdicts: requeues, gate-pass events, no-op rework.

Split out of ``tests/test_charlie_work.py`` (issue #1549, Track-1 wave 3/8):
the stale-CI verdict half of the ``test_review_queue_*`` seam (issue #1111) -- same-head stale-CI requeues, gate-pass events, and no-op-rework suppression. Shared fakes and helpers in ``tests/_review_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from _fakes_github import FakeGitHubWithChecks
from _helpers import (
    _STALE_CI_CONTAMINATED_REQUIRED_CHANGES,
    _STALE_CI_GREEN_CHECKS,
    _STALE_CI_RED_CHECKS,
    _STALE_CI_REQUIRED,
)
from _review_fixtures import _write_review_packet, _stale_ci_pr, _stale_ci_review_queue_app
from charlie_work.config import AutoMergeConfig, OrchestratorConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


# --------------------------------------------------------------------------
# Issue #1111: stale-CI request_changes verdict suppression in review_queue()
# --------------------------------------------------------------------------


def test_review_queue_same_head_stale_ci_requeues_as_stale(tmp_path: Path) -> None:
    """Issue #1111 (a): a same-head request_changes verdict whose only
    findings cite required checks that are all green now must be re-queued
    for a fresh review (decision "stale") with a ``stale_ci_verdict_requeued``
    event, and must NOT be routed through
    ``_reroute_stranded_request_changes`` -- that repair would re-drive the
    SAME (now-nonexistent) failure into a rework worker instead of waiting
    for the fresh review this path queues."""
    pr_number = 456
    issue_number = 123
    head = "sha-live-head"

    app = _stale_ci_review_queue_app(
        tmp_path,
        prs=[_stale_ci_pr(pr_number, issue_number, head)],
        checks=_STALE_CI_GREEN_CHECKS,
        dry_run=False,
    )
    _write_review_packet(
        tmp_path,
        pr_number,
        head,
        {
            "decision": "request_changes",
            "escalated": False,
            "reviewed_head_sha": head,
            "required_changes": _STALE_CI_CONTAMINATED_REQUIRED_CHANGES,
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": pr_number,
            "issue": issue_number,
            "packet_head_sha": head,
            "decision": "stale",
            "reviewed_head_sha": head,
            "mergeable": None,
            "mergeStateStatus": "CLEAN",
        }
    ]

    requeue_events = query_events(app.paths.state_file, kind="stale_ci_verdict_requeued")
    assert len(requeue_events) == 1
    assert requeue_events[0]["payload"]["pr_number"] == pr_number
    assert requeue_events[0]["payload"]["issue_number"] == issue_number
    assert requeue_events[0]["payload"]["reviewed_head_sha"] == head

    # _reroute_stranded_request_changes must NOT have fired: it would have
    # transitioned the issue to rework_requested via _route_to_rework.
    state = load_state(app.paths.state_file)
    assert state["issues"].get(str(issue_number), {}).get("status") != "rework_requested"


def test_review_queue_same_head_not_stale_when_checks_red(tmp_path: Path) -> None:
    """Issue #1111 control: the identical contaminated-shape verdict, but a
    required check is still red. ``is_stale_ci_verdict`` must be False, so
    the pre-#1111 behavior is preserved: the PR is not queued (same-head
    request_changes verdicts are never queued, stale or not), and the
    stranded-verdict repair fires normally since the issue was never routed
    to rework."""
    pr_number = 456
    issue_number = 123
    head = "sha-live-head"

    app = _stale_ci_review_queue_app(
        tmp_path,
        prs=[_stale_ci_pr(pr_number, issue_number, head)],
        checks=_STALE_CI_RED_CHECKS,
        dry_run=False,
    )
    _write_review_packet(
        tmp_path,
        pr_number,
        head,
        {
            "decision": "request_changes",
            "escalated": False,
            "reviewed_head_sha": head,
            "required_changes": _STALE_CI_CONTAMINATED_REQUIRED_CHANGES,
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == []

    requeue_events = query_events(app.paths.state_file, kind="stale_ci_verdict_requeued")
    assert requeue_events == []
    # Preserved existing behavior: the stranded verdict IS re-routed to rework
    # because it describes a real (still-red) failure.
    state = load_state(app.paths.state_file)
    assert state["issues"][str(issue_number)]["status"] == "rework_requested"


def test_review_queue_carry_forward_suppressed_when_stale_ci(tmp_path: Path) -> None:
    """Issue #1111 (c): carry-forward matches (identical patch-id) but the
    recorded verdict is stale-CI. The carry-forward must be skipped -- no
    ``_update_approval_head`` call, no ``verdict_carried_forward_clean_rebase``
    event -- and the PR falls through to the stale-queue path instead, so a
    fresh review supersedes the (already-resolved) failure rather than
    silently re-confirming it on the new head."""
    from charlie_work.janitor import _calculate_patch_id

    diff_text = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "+line3\n"
        " line4\n"
    )
    patch_id = _calculate_patch_id(diff_text)
    old_head = "sha-abc123"
    new_head = "sha-rebased123"
    pr_number = 456
    issue_number = 123

    app = _stale_ci_review_queue_app(
        tmp_path,
        prs=[_stale_ci_pr(pr_number, issue_number, new_head)],
        checks=_STALE_CI_GREEN_CHECKS,
        dry_run=False,
    )
    app.gh.diffs[pr_number] = diff_text

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "request_changes",
            "escalated": False,
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": patch_id,
            "carried_forward_from": [],
            "required_changes": _STALE_CI_CONTAMINATED_REQUIRED_CHANGES,
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] == [
        {
            "pr": pr_number,
            "issue": issue_number,
            "packet_head_sha": new_head,
            "decision": "stale",
            "reviewed_head_sha": old_head,
            "mergeable": None,
            "mergeStateStatus": "CLEAN",
        }
    ]

    # The decision file must NOT have been carry-forwarded onto the new head.
    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == old_head
    assert "carry_forward_tier" not in decision

    state = load_state(app.paths.state_file)
    assert "reviewed_head_sha" not in state["prs"].get(str(pr_number), {})
    carry_events = query_events(app.paths.state_file, kind="verdict_carried_forward_clean_rebase")
    assert carry_events == []
    requeue_events = query_events(app.paths.state_file, kind="stale_ci_verdict_requeued")
    assert len(requeue_events) == 1
    assert requeue_events[0]["payload"]["reviewed_head_sha"] == old_head


def test_review_queue_same_head_stale_ci_requeue_deduped_per_head(tmp_path: Path) -> None:
    """Issue #1120: ``review_queue()`` is called multiple times per loop pass
    (once by ``dispatch_reviews()``, once by
    ``_record_cross_family_verdicts()``), so the ``stale_ci_verdict_requeued``
    event double-fires for the same PR/head within one pass. Mirroring the
    ``stale_ci_gate_pass_head`` fix (PR #1117), the emission is now keyed on
    ``stale_ci_verdict_requeued_head`` stored in the PR state entry: a second
    consecutive ``review_queue()`` for the same PR/head must not emit a
    duplicate."""
    pr_number = 456
    issue_number = 123
    head = "sha-live-head"

    app = _stale_ci_review_queue_app(
        tmp_path,
        prs=[_stale_ci_pr(pr_number, issue_number, head)],
        checks=_STALE_CI_GREEN_CHECKS,
        dry_run=False,
    )
    _write_review_packet(
        tmp_path,
        pr_number,
        head,
        {
            "decision": "request_changes",
            "escalated": False,
            "reviewed_head_sha": head,
            "required_changes": _STALE_CI_CONTAMINATED_REQUIRED_CHANGES,
        },
    )

    first = app.review_queue()
    second = app.review_queue()

    assert first.ok is True
    assert second.ok is True
    # Both calls queue the PR -- the dispatch dedup is separate.
    assert len(first.data["queue"]) == 1
    assert len(second.data["queue"]) == 1

    state = load_state(app.paths.state_file)
    assert state["prs"][str(pr_number)]["stale_ci_verdict_requeued_head"] == head
    requeue_events = query_events(app.paths.state_file, kind="stale_ci_verdict_requeued")
    assert len(requeue_events) == 1


def test_review_queue_carry_forward_stale_ci_requeue_deduped_per_head(tmp_path: Path) -> None:
    """Issue #1120 (carry-forward variant): the head-advanced stale-CI
    requeue emission site must also dedup per live head. A second
    ``review_queue()`` call for the same PR/live-head must not re-emit."""
    from charlie_work.janitor import _calculate_patch_id

    diff_text = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,3 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "+line3\n"
        " line4\n"
    )
    patch_id = _calculate_patch_id(diff_text)
    old_head = "sha-abc123"
    new_head = "sha-rebased123"
    pr_number = 456
    issue_number = 123

    app = _stale_ci_review_queue_app(
        tmp_path,
        prs=[_stale_ci_pr(pr_number, issue_number, new_head)],
        checks=_STALE_CI_GREEN_CHECKS,
        dry_run=False,
    )
    app.gh.diffs[pr_number] = diff_text

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "request_changes",
            "escalated": False,
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": patch_id,
            "carried_forward_from": [],
            "required_changes": _STALE_CI_CONTAMINATED_REQUIRED_CHANGES,
        },
    )

    first = app.review_queue()
    second = app.review_queue()

    assert first.ok is True
    assert second.ok is True
    assert len(first.data["queue"]) == 1
    assert len(second.data["queue"]) == 1

    state = load_state(app.paths.state_file)
    assert state["prs"][str(pr_number)]["stale_ci_verdict_requeued_head"] == new_head
    requeue_events = query_events(app.paths.state_file, kind="stale_ci_verdict_requeued")
    assert len(requeue_events) == 1


def test_review_queue_stale_ci_requeue_re_emits_after_head_transition(tmp_path: Path) -> None:
    """Issue #1120: the dedup key is the live head, not a permanent
    suppression. When the PR advances to a new head, the next
    ``review_queue()`` call must re-emit ``stale_ci_verdict_requeued`` for
    the new head transition."""
    pr_number = 456
    issue_number = 123
    head_a = "sha-head-a"
    head_b = "sha-head-b"

    app = _stale_ci_review_queue_app(
        tmp_path,
        prs=[_stale_ci_pr(pr_number, issue_number, head_a)],
        checks=_STALE_CI_GREEN_CHECKS,
        dry_run=False,
    )
    _write_review_packet(
        tmp_path,
        pr_number,
        head_a,
        {
            "decision": "request_changes",
            "escalated": False,
            "reviewed_head_sha": head_a,
            "required_changes": _STALE_CI_CONTAMINATED_REQUIRED_CHANGES,
        },
    )

    first = app.review_queue()
    assert first.ok is True
    requeue_after_first = query_events(app.paths.state_file, kind="stale_ci_verdict_requeued")
    assert len(requeue_after_first) == 1

    # Advance to a new head: update the PR's live head, the packet head, and
    # the recorded verdict to the new head.
    app.gh.prs = [_stale_ci_pr(pr_number, issue_number, head_b)]
    _write_review_packet(
        tmp_path,
        pr_number,
        head_b,
        {
            "decision": "request_changes",
            "escalated": False,
            "reviewed_head_sha": head_b,
            "required_changes": _STALE_CI_CONTAMINATED_REQUIRED_CHANGES,
        },
    )

    second = app.review_queue()
    assert second.ok is True
    requeue_after_second = query_events(app.paths.state_file, kind="stale_ci_verdict_requeued")
    assert len(requeue_after_second) == 2
    state = load_state(app.paths.state_file)
    assert state["prs"][str(pr_number)]["stale_ci_verdict_requeued_head"] == head_b


def test_review_no_op_rework_suppressed_when_stale_ci_verdict(tmp_path: Path) -> None:
    """Issue #1111 (d): review()'s no-op-rework routing must not fire -- and
    must not burn ``no_op_rework_attempts`` -- when the on-disk verdict is a
    stale-CI request_changes (contaminated shape, all required checks
    currently green). Mirrors
    test_fix_janitor_routing.test_janitor_no_op_rework_routes_to_rework's
    same-head/same-diff setup, which is what makes ``verdict.is_no_op_rework``
    True in the first place; the only difference is the recorded verdict's
    ``required_changes`` shape and a configured, all-green ``required_checks``."""
    config = OrchestratorConfig(auto_merge=AutoMergeConfig(required_checks=_STALE_CI_REQUIRED))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(checks=_STALE_CI_GREEN_CHECKS)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456,
        "request_changes",
        summary="Tests passed: .github:18 — Process completed with exit code 1.",
        required_changes=_STALE_CI_CONTAMINATED_REQUIRED_CHANGES,
        verdict_provenance="ci_gate_auto_reject",
    )
    state = load_state(app.paths.state_file)
    record = {**state["issues"].get("123", {}), "number": 123, "status": "reviewing"}
    state["issues"]["123"] = record
    save_state(app.paths.state_file, state)

    # Same head, same diff as the recorded verdict: no actual content change,
    # so the janitor's no-op-rework signal fires -- but the verdict is
    # stale-CI, so routing must be suppressed.
    result = app.review(456)

    assert result.data.get("routed_to_rework") is not True
    state = load_state(app.paths.state_file)
    assert state["prs"].get("456", {}).get("no_op_rework_attempts", 0) == 0
    assert state["issues"]["123"]["status"] != "rework_requested"


def test_review_stale_ci_skip_emits_gate_pass_event(tmp_path: Path) -> None:
    """Issue #1116: when run_janitor's own no-op-rework check is skipped
    because the recorded verdict is stale-CI (same setup as
    test_review_no_op_rework_suppressed_when_stale_ci_verdict -- same head,
    same diff, contaminated required_changes shape, all required checks
    green), the janitor gate as a whole now PASSES (no no-op failure, no
    other failure on this green PR) and review() must log a
    ``stale_ci_verdict_gate_pass`` event via query_events -- log_event writes
    only to events.db, never state["events"], so that is the only way to
    observe it. The dry_run=True control mirrors the ``not self.dry_run``
    guard on the emission site."""
    config = OrchestratorConfig(auto_merge=AutoMergeConfig(required_checks=_STALE_CI_REQUIRED))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(checks=_STALE_CI_GREEN_CHECKS)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456,
        "request_changes",
        summary="Tests passed: .github:18 — Process completed with exit code 1.",
        required_changes=_STALE_CI_CONTAMINATED_REQUIRED_CHANGES,
        verdict_provenance="ci_gate_auto_reject",
    )
    state = load_state(app.paths.state_file)
    record = {**state["issues"].get("123", {}), "number": 123, "status": "reviewing"}
    state["issues"]["123"] = record
    save_state(app.paths.state_file, state)

    result = app.review(456)

    assert result.ok is True
    assert result.data.get("routed_to_rework") is not True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["janitor_ok"] is True

    gate_pass_events = query_events(app.paths.state_file, kind="stale_ci_verdict_gate_pass")
    assert len(gate_pass_events) == 1
    assert gate_pass_events[0]["payload"]["pr_number"] == 456
    assert gate_pass_events[0]["payload"]["head_sha"] == "sha-abc123"


def test_review_stale_ci_skip_no_gate_pass_event_in_dry_run(tmp_path: Path) -> None:
    """Control for the ``not self.dry_run`` guard on the emission site
    (workflow.py review(), issue #1116): the identical stale-CI same-head/
    same-diff scenario must not emit ``stale_ci_verdict_gate_pass`` when the
    app is running dry."""
    config = OrchestratorConfig(auto_merge=AutoMergeConfig(required_checks=_STALE_CI_REQUIRED))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(checks=_STALE_CI_GREEN_CHECKS)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456,
        "request_changes",
        summary="Tests passed: .github:18 — Process completed with exit code 1.",
        required_changes=_STALE_CI_CONTAMINATED_REQUIRED_CHANGES,
        verdict_provenance="ci_gate_auto_reject",
    )
    state = load_state(app.paths.state_file)
    record = {**state["issues"].get("123", {}), "number": 123, "status": "reviewing"}
    state["issues"]["123"] = record
    save_state(app.paths.state_file, state)

    dry_app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    dry_app.review(456)

    assert query_events(app.paths.state_file, kind="stale_ci_verdict_gate_pass") == []


def test_review_stale_ci_gate_pass_event_deduped_per_head(tmp_path: Path) -> None:
    """PR #1117 review finding: the gate re-passes on EVERY orchestrator poll
    while the PR waits on review-dispatch capacity, so without dedup the
    ``stale_ci_verdict_gate_pass`` event re-fires each pass (the log-spam
    pattern review() guards against everywhere else -- cost-spirals.md
    Finding 2). A second consecutive review() for the same PR/head must not
    emit a duplicate; the emission is keyed on ``stale_ci_gate_pass_head``
    stored in the PR state entry."""
    config = OrchestratorConfig(auto_merge=AutoMergeConfig(required_checks=_STALE_CI_REQUIRED))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(checks=_STALE_CI_GREEN_CHECKS)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456,
        "request_changes",
        summary="Tests passed: .github:18 — Process completed with exit code 1.",
        required_changes=_STALE_CI_CONTAMINATED_REQUIRED_CHANGES,
        verdict_provenance="ci_gate_auto_reject",
    )
    state = load_state(app.paths.state_file)
    record = {**state["issues"].get("123", {}), "number": 123, "status": "reviewing"}
    state["issues"]["123"] = record
    save_state(app.paths.state_file, state)

    first = app.review(456)
    second = app.review(456)

    assert first.ok is True
    assert second.ok is True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["stale_ci_gate_pass_head"] == "sha-abc123"
    gate_pass_events = query_events(app.paths.state_file, kind="stale_ci_verdict_gate_pass")
    assert len(gate_pass_events) == 1


def test_review_no_op_rework_routes_when_prose_finding(tmp_path: Path) -> None:
    """Issue #1111 control: the same same-head/same-diff no-op setup, but the
    recorded verdict's finding is real prose rather than a check citation.
    ``is_stale_ci_verdict`` must be False, so the pre-#1111 no-op-rework
    routing (and its attempt-burn) is preserved exactly as
    test_fix_janitor_routing.test_janitor_no_op_rework_routes_to_rework
    already covers -- reproduced here as the control for the suppression
    test above rather than relying on cross-file coupling."""
    config = OrchestratorConfig(auto_merge=AutoMergeConfig(required_checks=_STALE_CI_REQUIRED))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(checks=_STALE_CI_GREEN_CHECKS)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app.record_review(
        456,
        "request_changes",
        summary="fix A",
        required_changes=["src/foo.py:42 — off-by-one error in the loop bound."],
        verdict_provenance="fresh_llm_review",
    )
    state = load_state(app.paths.state_file)
    record = {**state["issues"].get("123", {}), "number": 123, "status": "reviewing"}
    state["issues"]["123"] = record
    save_state(app.paths.state_file, state)

    result = app.review(456)

    assert result.ok is True
    assert result.data["routed_to_rework"] is True
    assert result.data["rework_reason"] == "no_op_rework"

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["no_op_rework_attempts"] == 1
