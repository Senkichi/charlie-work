"""``_update_open_agent_prs`` approval-head recording and per-PR skip/update decisions.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_update_approval_head_records_event_for_every_tier(tmp_path: Path) -> None:
    """Issue #638: ``_update_approval_head`` must record a carry-forward event
    itself, so a new call site cannot forget it. The event kind is
    tier-dependent so the three mechanisms (patch-id, line-content,
    verified-sync) stay separately auditable, and exactly one event is
    emitted per call."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_number = 456
    issue_number = 123
    decision_dir = paths.prs / f"pr-{pr_number}"
    decision_dir.mkdir(parents=True)

    def _run(tier: str, old_head: str, new_head: str) -> dict[str, Any]:
        decision = {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": "pid-xyz",
        }
        (decision_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")
        app._update_approval_head(
            pr_number,
            decision,
            new_head,
            old_head=old_head,
            issue_number=issue_number,
            tier=tier,
        )
        return load_state(paths.state_file)

    expected_kind = {
        "patch-id": "verdict_carried_forward_clean_rebase",
        "line-content": "verdict_carried_forward_line_content",
        "verified-sync": "verdict_carried_forward_verified_sync",
    }

    for tier, kind in expected_kind.items():
        state = _run(tier, f"old-{tier}", f"new-{tier}")
        carry_events = [e for e in state["events"] if e["kind"] == kind]
        assert len(carry_events) == 1, f"tier {tier}: expected 1 {kind} event"
        payload = carry_events[0]["payload"]
        assert payload["pr_number"] == pr_number
        assert payload["issue_number"] == issue_number
        assert payload["old_reviewed_head_sha"] == f"old-{tier}"
        assert payload["new_head_sha"] == f"new-{tier}"
        assert payload["carry_forward_tier"] == tier
        assert payload["carried_forward_from"] == [f"old-{tier}"]


def test_update_open_agent_prs_front_of_train_records_verified_sync_event(
    tmp_path: Path,
) -> None:
    """Issue #638: the front-of-train ``_update_open_agent_prs`` carry-forward
    (a previously-silent ``verified-sync`` call site) must record a
    ``verdict_carried_forward_verified_sync`` event."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
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
            "mergeStateStatus": "CLEAN",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "baseRefName": "main",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.record_review(789, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    for idx, pr_number in enumerate((456, 789)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    result = app.merge_ready(456, merge=True)
    assert result.ok is True
    assert result.data["merged"] is True

    update_results = result.data["update_open_prs_results"]
    assert update_results is not None
    assert len(update_results) == 1
    assert update_results[0]["pr_number"] == 789
    assert update_results[0]["updated"] is True

    state = load_state(paths.state_file)
    sync_events = [
        e for e in state["events"] if e["kind"] == "verdict_carried_forward_verified_sync"
    ]
    assert len(sync_events) == 1
    payload = sync_events[0]["payload"]
    assert payload["pr_number"] == 789
    assert payload["issue_number"] == 124
    assert payload["carry_forward_tier"] == "verified-sync"


def test_update_open_agent_prs_reports_failure_as_value(tmp_path: Path) -> None:
    """Test that pr_update_branch failures are reported as values, not successes."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Add a second PR to test batch behavior
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "headRepository": {
                "owner": {"login": "test"},
                "name": "repo",
            },
        }
    ]
    fake_gh.issues = [
        {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "body": "Search is broken",
            "labels": [{"name": "automated-ready"}],
        },
        {
            "number": 124,
            "title": "Fix another",
            "url": "https://example.test/issues/124",
            "body": "Another issue",
            "labels": [{"name": "automated-ready"}],
        },
    ]
    # Make update-branch fail for the second PR
    fake_gh.update_branch_ok = False

    # Override prs to return two PRs
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Call _update_open_agent_prs directly
    results = app._update_open_agent_prs(merged_pr_number=456)

    # Should have results for both PRs (excluding the merged one)
    assert len(results) == 1  # Only PR 789 (456 is excluded as the merged PR)


def test_update_open_agent_prs_skips_approved_pending_ship_prs(tmp_path: Path) -> None:
    """Test that approved-pending-ship PRs are skipped to avoid invalidating approvals.

    Regression test for issue #89: when two PRs are approved in the same operator pass,
    merging the first should not base-update the second (which would move its head and
    invalidate its approval, forcing a manual re-approve loop).
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # Set up two approved PRs
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",  # Live head matches reviewed head
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-def456",  # Live head matches reviewed head
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    # Create review decision files for both PRs (approved state)
    pr_456_decision_dir = paths.prs / "pr-456"
    pr_456_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_456_decision_dir / "review-decision.json").write_text(
        json.dumps(
            {"decision": "approved", "reviewed_head_sha": "sha-abc123"},
            indent=2,
        ),
        encoding="utf-8",
    )

    pr_789_decision_dir = paths.prs / "pr-789"
    pr_789_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_789_decision_dir / "review-decision.json").write_text(
        json.dumps(
            {"decision": "approved", "reviewed_head_sha": "sha-def456"},
            indent=2,
        ),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Simulate merging PR 456: update remaining open PRs
    results = app._update_open_agent_prs(merged_pr_number=456)

    # PR 789 should be skipped (approved-pending-ship)
    assert len(results) == 1
    assert results[0]["pr_number"] == 789
    assert results[0]["updated"] is False
    assert results[0]["skipped_reason"] == "approved-pending-ship"

    # Verify pr_update_branch was NOT called for PR 789
    assert fake_gh.update_branch_ok is True  # Should still be True (never called)


def test_update_open_agent_prs_skips_request_changes_and_blocked(tmp_path: Path) -> None:
    """Issue #404: broadcast mode must not update-branch request_changes or blocked PRs.

    Rework or human intervention will replace the head, so the CI run would be
    guaranteed-wasted runner time.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
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
            "headRefOid": "sha-abc123",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 101,
            "title": "Fix #125: blocked",
            "url": "https://example.test/pull/101",
            "headRefName": "agent/issue-125-blocked",
            "headRefOid": "sha-ghi789",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #125\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    # PR 456 approved and merged
    pr_456_decision_dir = paths.prs / "pr-456"
    pr_456_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_456_decision_dir / "review-decision.json").write_text(
        json.dumps(
            {"decision": "approved", "reviewed_head_sha": "sha-abc123"},
            indent=2,
        ),
        encoding="utf-8",
    )

    # PR 789 in rework (request_changes)
    pr_789_decision_dir = paths.prs / "pr-789"
    pr_789_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_789_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes"}, indent=2),
        encoding="utf-8",
    )

    # PR 101 blocked
    pr_101_decision_dir = paths.prs / "pr-101"
    pr_101_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_101_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "blocked"}, indent=2),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    results = app._update_open_agent_prs(merged_pr_number=456)

    assert len(results) == 2
    assert all(r["updated"] is False for r in results)
    assert {r["pr_number"] for r in results} == {789, 101}
    assert all(r["skipped_reason"] == "not_approved" for r in results)
    assert fake_gh.pr_update_branch_calls == []


def test_update_open_agent_prs_updates_approved_prs_with_moved_head(tmp_path: Path) -> None:
    """Test that approved PRs with moved heads are still updated (not skipped).

    This ensures the head-moved gate remains intact for content-bearing moves.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # Set up an approved PR whose head has moved since approval
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-new456",  # Head has moved since approval
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]

    # Create review decision for PR 789 with old head
    pr_789_decision_dir = paths.prs / "pr-789"
    pr_789_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_789_decision_dir / "review-decision.json").write_text(
        json.dumps(
            {"decision": "approved", "reviewed_head_sha": "sha-def456"},  # Old head
            indent=2,
        ),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Simulate merging PR 456: update remaining open PRs
    results = app._update_open_agent_prs(merged_pr_number=456)

    # PR 789 should be updated (head moved, so not approved-pending-ship)
    assert len(results) == 1
    assert results[0]["pr_number"] == 789
    assert results[0]["updated"] is True
    assert "skipped_reason" not in results[0]


def test_update_open_agent_prs_skips_prs_with_pending_required_checks(tmp_path: Path) -> None:
    """Test that PRs with required checks in PENDING/IN_PROGRESS are skipped to avoid cancelling in-flight CI.

    Regression test for issue #209: when ship-it merges a PR with update_open_prs enabled,
    update-branch on sibling PRs cancels their in-flight CI, which can permanently wedge
    aggregate-gate checks. This test verifies the avoidance approach: skip update-branch
    for PRs whose required checks are in PENDING/IN_PROGRESS state.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # Set up a PR with PENDING required checks
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "statusCheckRollup": [
                {
                    "__typename": "CheckRun",
                    "name": "Tests passed",
                    "status": "IN_PROGRESS",  # Required check is in-flight
                    "conclusion": "",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Lint & Format",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Pre-commit",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
            ],
        },
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "headRefOid": "sha-def456",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
            "statusCheckRollup": [
                {
                    "__typename": "CheckRun",
                    "name": "Tests passed",
                    "status": "QUEUED",  # Required check is pending
                    "conclusion": "",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Lint & Format",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
                {
                    "__typename": "CheckRun",
                    "name": "Pre-commit",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
            ],
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Simulate merging a different PR: update remaining open PRs
    results = app._update_open_agent_prs(merged_pr_number=999)

    # Both PRs should be skipped due to pending required checks
    assert len(results) == 2
    assert results[0]["pr_number"] == 456
    assert results[0]["updated"] is False
    assert results[0]["skipped_reason"] == "pending-required-checks"
    assert results[1]["pr_number"] == 789
    assert results[1]["updated"] is False
    assert results[1]["skipped_reason"] == "pending-required-checks"

    # Verify update-branch was NOT called
    assert fake_gh.update_branch_ok is True  # Never set to False by a call
