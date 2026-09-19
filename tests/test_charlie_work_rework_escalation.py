"""Rework selection/escalation: lifecycle dispatch, escalated request-changes selectability, request-changes counting/cap on unchanged head.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8; sub-split for the PR #1750 review's 800-line cap).
"""

from __future__ import annotations

import sys
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    ReviewConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_standard_lifecycle_rework_dispatch_selects_issue(tmp_path: Path) -> None:
    """Issue #72 acceptance criterion 1: standard-lifecycle end-to-end rework dispatch.

    Fresh dispatch marks the issue dispatched → record_review(request_changes) →
    dispatch_rework SELECTS the issue and launches via a command-adapter fake,
    firing the rework_dispatched label transition.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Step 1: Fresh dispatch marks the issue as dispatched
    app.gh.prs[0]["state"] = "CLOSED"
    dispatch_result = app.dispatch(limit=1)
    assert dispatch_result.ok is True
    assert dispatch_result.data["selected_count"] == 1

    # Verify issue is marked as dispatched in state
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"

    # Step 2: record_review(request_changes) updates issue status to rework_requested
    # Issue #1131: record_review now refuses on a terminal-state (CLOSED) PR,
    # so restore the PR to OPEN before recording the verdict -- the CLOSED
    # state above was a fixture trick to make dispatch select the issue, not
    # a real terminal state.
    app.gh.prs[0]["state"] = "OPEN"
    review_result = app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )
    assert review_result.ok is True
    assert review_result.data["escalated"] is False

    # Verify issue status is now rework_requested
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"

    # Step 3: Create a rework prompt (normally written by record_review)
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    # Step 4: dispatch_rework SELECTS the issue and launches via command adapter
    # The issue already has needs-rework label from the request_changes transition
    app.gh.prs[0]["state"] = "OPEN"
    rework_result = app.dispatch_rework()

    # Verify dispatch_rework selected and launched the issue
    assert rework_result.ok is True
    assert rework_result.data["selected_count"] == 1
    assert rework_result.data["dispatch_results"][0]["stdout"].strip() == "123"

    # Verify the rework_dispatched label transition was fired
    # (adds in_progress, removes needs_rework)
    assert (123, "agent:in-progress") in fake_gh.labels_added
    assert (123, "agent:needs-rework") in fake_gh.labels_removed


def test_escalated_request_changes_does_not_make_issue_selectable(tmp_path: Path) -> None:
    """Issue #72 acceptance criterion 2: escalated request_changes must NOT make issue selectable.

    After an escalated verdict (request_changes_count at max), assert the issue's state status
    is NOT rework_requested and dispatch_rework does not select it.
    """
    config = OrchestratorConfig(
        review=ReviewConfig(max_rework_cycles=2),
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Step 1: Fresh dispatch marks the issue as dispatched
    app.gh.prs[0]["state"] = "CLOSED"
    dispatch_result = app.dispatch(limit=1)
    assert dispatch_result.ok is True
    assert dispatch_result.data["selected_count"] == 1

    # Step 2: Record first request_changes (count = 1, not escalated, head = "sha-1")
    # Issue #1131: restore PR to OPEN before record_review -- the CLOSED state
    # was a fixture trick for dispatch, not a real terminal state.
    app.gh.prs[0]["state"] = "OPEN"
    fake_gh.pr_head_shas[456] = "sha-1"
    review_result_1 = app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )
    assert review_result_1.ok is True
    assert review_result_1.data["escalated"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert state["issues"]["123"]["status"] == "rework_requested"

    # Step 3: Record second request_changes (count = 2, not escalated yet, head = "sha-2")
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")
    fake_gh.pr_head_shas[456] = "sha-2"

    review_result_2 = app.record_review(
        456, "request_changes", summary="fix B", verdict_provenance="fresh_llm_review"
    )
    assert review_result_2.ok is True
    assert review_result_2.data["escalated"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert state["issues"]["123"]["status"] == "rework_requested"

    # Step 4: Record third request_changes (count stays at 2, escalated because max_rework_cycles = 2, head = "sha-3")
    # When escalated, the count is NOT incremented (see workflow.py line 731-734)
    fake_gh.pr_head_shas[456] = "sha-3"
    review_result_3 = app.record_review(
        456, "request_changes", summary="fix C", verdict_provenance="fresh_llm_review"
    )
    assert review_result_3.ok is True
    assert review_result_3.data["escalated"] is True  # Should be escalated

    # Verify PR status is escalated
    state = load_state(paths.state_file)
    assert (
        state["prs"]["456"]["request_changes_count"] == 2
    )  # Count does NOT increment when escalated
    assert state["prs"]["456"]["status"] == "escalated"
    # Issue status should now be escalated (cleared from rework_requested)
    assert state["issues"]["123"]["status"] == "escalated"

    # Step 5: Verify the escalated label transition was fired (adds
    # operator_queue, removes reviewing). Issue #1266: max_rework_cycles_exceeded
    # is a mechanical escalation, so it now routes to operator_queue instead of
    # human_needed.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
    # The escalated transition removes reviewing but does NOT remove needs_rework
    # (this is by design per labels.py's redispatch_escalated/escalated edges)

    # Step 6: dispatch_rework should still NOT select the escalated issue
    # because the issue status is "escalated" (not "rework_requested")
    # The escalated issue still has needs-rework label (from previous non-escalated request_changes)
    rework_result = app.dispatch_rework()

    # Verify dispatch_rework did NOT select the escalated issue
    # (even though it has needs_rework label, the issue status is escalated so it's filtered out)
    assert rework_result.ok is True
    assert rework_result.data["selected_count"] == 0
    # No new in_progress label should have been added (rework_dispatched transition)
    # Count how many in_progress labels were added before this step
    in_progress_count_before = fake_gh.labels_added.count((123, "agent:in-progress"))
    # After the failed dispatch, the count should be the same
    in_progress_count_after = fake_gh.labels_added.count((123, "agent:in-progress"))
    assert in_progress_count_after == in_progress_count_before


def test_request_changes_count_does_not_increment_on_unchanged_head(tmp_path: Path) -> None:
    """Issue #208: request_changes_count should only increment when PR head advances.

    When a worker dies orphaned and the PR head never advances, re-issuing
    request_changes should not consume the escalation budget.
    """
    config = OrchestratorConfig(
        review=ReviewConfig(max_rework_cycles=2),
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Step 1: Fresh dispatch
    app.gh.prs[0]["state"] = "CLOSED"
    dispatch_result = app.dispatch(limit=1)
    assert dispatch_result.ok is True

    # Step 2: Record first request_changes (count = 1, head = "sha-1")
    # Issue #1131: restore PR to OPEN before record_review -- the CLOSED state
    # was a fixture trick for dispatch, not a real terminal state.
    app.gh.prs[0]["state"] = "OPEN"
    fake_gh.pr_head_shas[456] = "sha-1"
    review_result_1 = app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )
    assert review_result_1.ok is True
    assert review_result_1.data["escalated"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-1"

    # Step 3: Record second request_changes with SAME head (count should stay at 1)
    # This simulates a worker dying orphaned - no rework was actually produced
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    review_result_2 = app.record_review(
        456, "request_changes", summary="fix B", verdict_provenance="fresh_llm_review"
    )
    assert review_result_2.ok is True
    assert review_result_2.data["escalated"] is False

    state = load_state(paths.state_file)
    # Count should NOT increment because head didn't advance
    assert state["prs"]["456"]["request_changes_count"] == 1
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-1"

    # Step 4: Record third request_changes with NEW head (count should increment to 2)
    fake_gh.pr_head_shas[456] = "sha-2"
    review_result_3 = app.record_review(
        456, "request_changes", summary="fix C", verdict_provenance="fresh_llm_review"
    )
    assert review_result_3.ok is True
    assert review_result_3.data["escalated"] is False

    state = load_state(paths.state_file)
    # Count should increment because head advanced
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-2"


def test_at_cap_request_changes_on_unchanged_head_does_not_escalate(
    tmp_path: Path,
) -> None:
    """Issue #1210: an at-cap request_changes verdict on an unchanged head must not escalate.

    The head_advanced guard (issue #208) previously protected only the counter
    increment; the escalation check ran unconditionally and fired on an at-cap
    verdict even when the head was unchanged (e.g. a worker died orphaned and
    pushed nothing). Such a round should re-issue request_changes without
    escalating, mirroring what already happens below-cap. The counter must be
    unchanged and the PR/issue status must NOT be escalated.
    """
    config = OrchestratorConfig(
        review=ReviewConfig(max_rework_cycles=2),
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Step 1: Fresh dispatch
    app.gh.prs[0]["state"] = "CLOSED"
    dispatch_result = app.dispatch(limit=1)
    assert dispatch_result.ok is True

    # Step 2: Drive request_changes_count up to the cap (2) over two advancing heads.
    # Issue #1131: restore PR to OPEN before record_review -- the CLOSED state
    # was a fixture trick for dispatch, not a real terminal state.
    app.gh.prs[0]["state"] = "OPEN"
    fake_gh.pr_head_shas[456] = "sha-1"
    review_result_1 = app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )
    assert review_result_1.ok is True
    assert review_result_1.data["escalated"] is False

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")

    fake_gh.pr_head_shas[456] = "sha-2"
    review_result_2 = app.record_review(
        456, "request_changes", summary="fix B", verdict_provenance="fresh_llm_review"
    )
    assert review_result_2.ok is True
    assert review_result_2.data["escalated"] is False

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-2"

    # Step 3: A request_changes verdict on the SAME head (sha-2) — the worker
    # died without pushing anything. request_changes_count is already at the
    # cap. This must NOT escalate and must NOT mutate the counter.
    review_result_3 = app.record_review(
        456, "request_changes", summary="fix C", verdict_provenance="fresh_llm_review"
    )
    assert review_result_3.ok is True
    assert review_result_3.data["escalated"] is False

    state = load_state(paths.state_file)
    # Counter unchanged — the round consumed no escalation budget.
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert state["prs"]["456"]["reviewed_head_sha"] == "sha-2"
    # PR and issue status must reflect re-issued request_changes, not escalation.
    assert state["prs"]["456"]["status"] == "request_changes"
    assert state["issues"]["123"]["status"] == "rework_requested"
    # No human-needed label should have been added by this round.
    assert (123, "agent:human-needed") not in fake_gh.labels_added

    # Step 4: Sanity — once the head DOES advance, the at-cap verdict escalates
    # as before (existing behavior preserved for advanced heads).
    fake_gh.pr_head_shas[456] = "sha-3"
    review_result_4 = app.record_review(
        456, "request_changes", summary="fix D", verdict_provenance="fresh_llm_review"
    )
    assert review_result_4.ok is True
    assert review_result_4.data["escalated"] is True

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["request_changes_count"] == 2
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["issues"]["123"]["status"] == "escalated"
