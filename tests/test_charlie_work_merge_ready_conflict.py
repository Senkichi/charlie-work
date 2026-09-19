"""Merge-ready conflict routing to rework.

Split out of ``tests/test_charlie_work.py`` (issue #1550, Track-1
wave 4/8).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from _fakes_github import (
    FakeGitHub,
    FakeGitHubWithChecks,
)
from charlie_work.config import (
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_merge_ready_mergequeue_check_failure_still_routes_to_rework(tmp_path: Path) -> None:
    """Issue #823 (advisor-flagged gap): a silent revert caused by a genuine
    required-check failure must still reach check-failure-rework dispatch,
    not be shadowed by mergequeue_label_reverted.

    A prior pass's successful handoff leaves existing_pr_state.status ==
    "mergequeue". If a required check then goes red and Aviator strips the
    label for cause, the naive fold-in (mergequeue_label_reverted OR'd into
    mergequeue_handoff_failed with no other condition) makes
    mergequeue_handoff_failed True purely from the carried-over status. That
    would shadow the check-failure-rework dispatch gate's `not
    mergequeue_handoff_failed` term (workflow.py) and permanently block
    rework for a PR whose check failure is exactly the fixable thing rework
    exists for -- a strictly worse outcome than pre-#823 behavior. The revert
    term is gated on can_merge so a real check failure (can_merge False for a
    reason unrelated to the handoff) is still attributed and routed as a
    check failure. Proven behaviorally: the check-failure-rework dispatch
    (issue_status -> rework_requested, check_failure_rework_requested event)
    only fires when mergequeue_handoff_failed is False, so its firing here is
    direct proof the gate was not shadowed."""
    from charlie_work.config import AutoMergeConfig, DevinConfig

    required = ("Tests passed", "Lint & Format", "Pre-commit")
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=required,
            require_approved_review=True,
            mergequeue_label="mergequeue",
            failed_attempt_alarm=1,
        ),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[{"name": name, "state": "SUCCESS"} for name in required]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="ok", verdict_provenance="fresh_llm_review")

    # Pass 1: checks green, handoff succeeds, status becomes "mergequeue".
    first = app.merge_ready(456, merge=True)
    assert first.data["mergequeue_label_applied"] is True
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "mergequeue"

    # Pass 2: a required check genuinely fails. FakeGitHub.add_pr_label never
    # mutates fake_gh.prs[0]["labels"], so the live PR this pass already
    # shows the mergequeue label absent -- indistinguishable, from the
    # detector's point of view, from Aviator stripping it for cause. can_merge
    # is False this pass because of the real check failure, not a handoff
    # problem.
    fake_gh.checks = [
        {"name": "Tests passed", "state": "FAILURE"},
        {"name": "Lint & Format", "state": "SUCCESS"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    second = app.merge_ready(456, merge=True)

    assert second.data["can_merge"] is False
    assert second.data["checks"]["failed"] == ("Tests passed",)
    assert second.data["merge_attempt_alarm"] is True

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["status"] == "rework_requested"
    check_failure_events = [
        e for e in state["events"] if e["kind"] == "check_failure_rework_requested"
    ]
    assert len(check_failure_events) == 1
    assert check_failure_events[0]["payload"]["pr_number"] == 456
    assert check_failure_events[0]["payload"]["issue_number"] == 123
    assert check_failure_events[0]["payload"]["failed_checks"] == ["Tests passed"]


def test_merge_ready_conflict_rework_debounces_and_preserves_approval(tmp_path: Path) -> None:
    """Issue #456: conflict rework must not fire until N consecutive CONFLICTING
    passes and must not clobber the approved review decision.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=3,
        ),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "BEHIND",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    decision_path = paths.prs / "pr-456" / "review-decision.json"
    original_decision = json.loads(decision_path.read_text())

    # Two consecutive CONFLICTING passes must NOT dispatch rework yet.
    for _ in range(2):
        result = app.merge_ready(456, merge=False)
        assert result.ok is True
        assert result.data["can_merge"] is False
        assert result.data["merge_conflict"] is True
        assert result.data["merge_attempt_alarm"] is False
        state = load_state(paths.state_file)
        assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state["events"])
        assert state["issues"]["123"]["status"] == "approved"
        current_decision = json.loads(decision_path.read_text())
        assert current_decision["decision"] == "approved"
        assert current_decision["reviewed_head_sha"] == original_decision["reviewed_head_sha"]
        assert current_decision["reviewed_patch_id"] == original_decision["reviewed_patch_id"]

    # The third consecutive CONFLICTING pass reaches the alarm threshold and
    # dispatches conflict rework. The approved verdict must survive untouched.
    result = app.merge_ready(456, merge=False)
    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is True

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["status"] == "rework_requested"
    conflict_events = [
        e for e in state["events"] if e["kind"] == "merge_conflict_rework_requested"
    ]
    assert len(conflict_events) == 1
    assert conflict_events[0]["payload"]["pr_number"] == 456
    assert conflict_events[0]["payload"]["issue_number"] == 123
    assert "conflict_rework_requested_at" in conflict_events[0]["payload"]

    current_decision = json.loads(decision_path.read_text())
    assert current_decision["decision"] == "approved"
    assert current_decision["reviewed_head_sha"] == original_decision["reviewed_head_sha"]
    assert current_decision["reviewed_patch_id"] == original_decision["reviewed_patch_id"]
    assert state["prs"]["456"]["decision"] == "approved"
    assert state["prs"]["456"]["reviewed_head_sha"] == original_decision["reviewed_head_sha"]
    assert state["prs"]["456"]["reviewed_patch_id"] == original_decision["reviewed_patch_id"]
    assert "conflict_rework_requested_at" in state["prs"]["456"]

    prompt_path = paths.prs / "pr-456" / "rework-prompt.md"
    assert prompt_path.exists()
    assert (123, config.labels.needs_rework) in fake_gh.labels_added


def test_merge_ready_merge_conflict_routes_to_rework(tmp_path: Path) -> None:
    """Issue #371/#456: an approved PR with a genuine merge conflict is routed to rework.

    A conflict is detected from ``mergeable=CONFLICTING`` and is not retried
    with ``gh pr update-branch``. With the alarm threshold set to 1, a single
    CONFLICTING pass dispatches conflict rework. The approved verdict and its
    patch-id must survive so carry-forward can re-approve the rework push.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        ),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "BEHIND",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    decision_path = paths.prs / "pr-456" / "review-decision.json"
    original_decision = json.loads(decision_path.read_text())

    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is True
    assert result.data["merge_attempt_warning"] is not None
    # The sync step is bypassed: the PR head should not be advanced.
    assert fake_gh.prs[0]["headRefOid"] == "sha-abc123"

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["status"] == "rework_requested"
    assert state["events"][-1]["kind"] == "merge_ready"
    conflict_events = [
        e for e in state["events"] if e["kind"] == "merge_conflict_rework_requested"
    ]
    assert len(conflict_events) == 1
    assert conflict_events[0]["payload"]["pr_number"] == 456
    assert conflict_events[0]["payload"]["issue_number"] == 123

    # The approved verdict must not be clobbered by the rework request.
    current_decision = json.loads(decision_path.read_text())
    assert current_decision["decision"] == "approved"
    assert current_decision["reviewed_head_sha"] == original_decision["reviewed_head_sha"]
    assert current_decision["reviewed_patch_id"] == original_decision["reviewed_patch_id"]
    assert state["prs"]["456"]["decision"] == "approved"
    assert state["prs"]["456"]["reviewed_head_sha"] == original_decision["reviewed_head_sha"]
    assert state["prs"]["456"]["reviewed_patch_id"] == original_decision["reviewed_patch_id"]
    assert "conflict_rework_requested_at" in state["prs"]["456"]

    # The rework prompt was written and the issue was labeled for rework.
    prompt_path = paths.prs / "pr-456" / "rework-prompt.md"
    assert prompt_path.exists()
    assert (123, config.labels.needs_rework) in fake_gh.labels_added

    # dispatch_rework can pick the issue up and launch a worker.
    dispatch = app.dispatch_rework()
    assert dispatch.data["selected_count"] == 1
    assert dispatch.data["sessions"][0]["issue_number"] == 123
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"


def test_merge_ready_check_failure_routes_to_rework(tmp_path: Path) -> None:
    """Issue #674: an approved PR whose required checks genuinely fail is routed to rework.

    Once approved, loop()'s already_approved fast path never calls review()
    again, so review()'s pre-approval janitor-gate check-failure handling
    never re-fires for this PR. A completed FAILURE conclusion on a required
    check (not merely pending/missing/infra_failed/unavailable) must still
    reach rework via merge_ready's own alarm-threshold dispatch. The approved
    verdict and its patch-id must survive so carry-forward can re-approve the
    rework push.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        ),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "FAILURE"},
            {"name": "Lint & Format", "state": "SUCCESS"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
    )
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "BLOCKED",
            "mergeable": "MERGEABLE",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    decision_path = paths.prs / "pr-456" / "review-decision.json"
    original_decision = json.loads(decision_path.read_text())

    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is False
    assert result.data["checks"]["failed"] == ("Tests passed",)
    assert result.data["merge_attempt_alarm"] is True
    assert result.data["merge_attempt_warning"] is not None
    assert "Tests passed" in result.data["merge_attempt_warning"]
    # The sync step is bypassed: the PR head should not be advanced.
    assert fake_gh.prs[0]["headRefOid"] == "sha-abc123"

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["status"] == "rework_requested"
    assert state["events"][-1]["kind"] == "merge_ready"
    check_failure_events = [
        e for e in state["events"] if e["kind"] == "check_failure_rework_requested"
    ]
    assert len(check_failure_events) == 1
    assert check_failure_events[0]["payload"]["pr_number"] == 456
    assert check_failure_events[0]["payload"]["issue_number"] == 123
    assert check_failure_events[0]["payload"]["failed_checks"] == ["Tests passed"]

    # The approved verdict must not be clobbered by the rework request.
    current_decision = json.loads(decision_path.read_text())
    assert current_decision["decision"] == "approved"
    assert current_decision["reviewed_head_sha"] == original_decision["reviewed_head_sha"]
    assert current_decision["reviewed_patch_id"] == original_decision["reviewed_patch_id"]
    assert state["prs"]["456"]["decision"] == "approved"
    assert state["prs"]["456"]["reviewed_head_sha"] == original_decision["reviewed_head_sha"]
    assert state["prs"]["456"]["reviewed_patch_id"] == original_decision["reviewed_patch_id"]
    assert "check_failure_rework_requested_at" in state["prs"]["456"]

    # The rework prompt was written and the issue was labeled for rework.
    prompt_path = paths.prs / "pr-456" / "rework-prompt.md"
    assert prompt_path.exists()
    assert "Tests passed" in prompt_path.read_text(encoding="utf-8")
    assert (123, config.labels.needs_rework) in fake_gh.labels_added

    # dispatch_rework can pick the issue up and launch a worker.
    dispatch = app.dispatch_rework()
    assert dispatch.data["selected_count"] == 1
    assert dispatch.data["sessions"][0]["issue_number"] == 123
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"


def test_merge_ready_stale_base_not_routed_to_rework(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #371: a stale but fast-forwardable base is deferred, not sent to rework."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # The default base_head_sha is base-sha, which is already the merge-base of
    # sha-abc123. Advance it to a post-merge tip whose merge-base with sha-abc123
    # is still base-sha, so the freshness gate sees a stale base.
    post_merge_base = "main-merged-sha-abc123"
    fake_gh.base_head_sha = post_merge_base
    fake_gh.commits[post_merge_base] = {"parents": [{"sha": "base-sha"}, {"sha": "sha-abc123"}]}
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "BEHIND",
            "mergeable": "MERGEABLE",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    # Simulate a base-sync that reports success but does not advance the head, so
    # the merge-base freshness gate still defers the PR.
    monkeypatch.setattr(fake_gh, "pr_update_branch", lambda pr_number: True)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is False
    assert result.data.get("stale_base") is True
    assert result.data["merge_attempt_alarm"] is False
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "approved"
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state["events"])
    assert (paths.prs / "pr-456" / "rework-prompt.md").exists() is False
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added


def test_merge_ready_conflict_inflight_worker_returns_early(tmp_path: Path) -> None:
    """Issue #379: a merge conflict whose linked issue is already in-flight is not re-routed.

    The early return must not fire the alarm and must leave the worker state alone.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"]["status"] = "dispatched"
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is False
    assert result.data["merge_attempt_warning"] is None
    assert result.message == "PR #456 merge conflict is being resolved by a rework worker"

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state["events"])


def test_merge_ready_conflict_blocked_issue_not_rerouted(tmp_path: Path) -> None:
    """Issue #379 rework: a merge conflict whose linked issue is blocked (a
    human reviewer verdict, set by record_review's decision=="blocked") must
    never be rerouted to rework_requested.

    transition() has no source-state validation, so rerouting would silently
    strip that reviewer verdict and hand the issue back to automation behind
    the human's back. The PR and issue must be left untouched. Unlike
    "escalated" (see
    test_merge_ready_conflict_escalated_for_unrelated_reason_routes_to_rework
    below), "blocked" records no reason a re-entry mechanism could scope to,
    so issue #776 deliberately leaves it a one-way door.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    # Mark the linked issue as carrying the human_needed label, matching a
    # real blocked issue, so a stripped label would be observable.
    fake_gh.issues[0]["labels"] = [{"name": config.labels.human_needed}]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"]["status"] = "blocked"
        save_state(paths.state_file, state)

    labels_removed_before = list(fake_gh.labels_removed)
    labels_added_before = list(fake_gh.labels_added)

    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is False
    assert result.data["merge_attempt_warning"] is None

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "blocked"
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state["events"])
    # No label mutation must have been issued for the linked issue —
    # human_needed must stay in place.
    assert fake_gh.labels_removed == labels_removed_before
    assert fake_gh.labels_added == labels_added_before


def test_merge_ready_conflict_escalated_for_unrelated_reason_routes_to_rework(
    tmp_path: Path,
) -> None:
    """Issue #776: an issue escalated for an UNRELATED reason (e.g. a dead
    request-changes-fix worker exhausting the watchdog's redispatch cap --
    the real mechanism that escalated corpus issues #592/#648/#606 via
    _reap_restore_rework_requested) must not permanently wall off a PR that
    separately develops a merge conflict.

    This is the regression test for narrowing merge_ready()'s Guard 1
    exclusion from ``("escalated", "blocked")`` down to ``"blocked"`` only --
    it asserts the ROUTING CALL actually happened (a fresh dispatch, the
    label edge, the attempts counter), not merely that the return value
    looks different from the blocked case. It also asserts reason X's own
    budget (``redispatch_at``, the watchdog counter that produced the
    original escalation) survives untouched: re-entry must not reset X's
    cap while remediating unrelated Y.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    # Issue #1266: redispatch_cap_exceeded is mechanical, so a real prior
    # escalation for this reason would have landed agent:operator-queue, not
    # agent:human-needed -- seed the fixture to match.
    fake_gh.issues[0]["labels"] = [{"name": config.labels.operator_queue}]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    unrelated_redispatch_at = ["2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"]
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            **state["issues"]["123"],
            "status": "escalated",
            "escalation_reason": "redispatch_cap_exceeded",
            "reason_class": "mechanical",
            "redispatch_at": unrelated_redispatch_at,
        }
        save_state(paths.state_file, state)

    result = app.merge_ready(456, merge=False)

    assert result.ok is True
    assert result.data["merge_conflict"] is True

    state = load_state(paths.state_file)
    # The routing call actually happened: a fresh dispatch, not a no-op.
    # (merge_ready()'s top-level result.data doesn't expose the internal
    # routed/escalated booleans -- verify via the state.json side effects
    # the routing call actually produces instead.)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["status"] == "rework_requested"
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1
    conflict_events = [
        e for e in state["events"] if e["kind"] == "merge_conflict_rework_requested"
    ]
    assert len(conflict_events) == 1
    assert conflict_events[0]["payload"]["issue_number"] == 123
    assert (123, config.labels.needs_rework) in fake_gh.labels_added
    assert (123, config.labels.operator_queue) in fake_gh.labels_removed
    # Reason X's own budget must survive remediating unrelated Y: the
    # watchdog redispatch timestamps that drove the ORIGINAL escalation are
    # untouched, so a later re-escalation for the same reason X is not
    # handed a falsely-fresh cap.
    assert state["issues"]["123"]["redispatch_at"] == unrelated_redispatch_at
