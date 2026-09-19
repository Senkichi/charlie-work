"""``_update_open_agent_prs`` merge-train modes: next/all sync behavior, compare-unavailable reporting, and head-race rejection.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from charlie_work.config import AutoMergeConfig, OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_update_open_agent_prs_updates_prs_with_completed_required_checks(tmp_path: Path) -> None:
    """Test that PRs with all required checks completed are still updated normally."""
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # Set up a PR with all required checks SUCCESS
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
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
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

    # PR should be updated normally
    assert len(results) == 1
    assert results[0]["pr_number"] == 456
    assert results[0]["updated"] is True
    assert "skipped_reason" not in results[0]


def test_update_open_agent_prs_broadcast_skips_when_protection_strict_false(
    tmp_path: Path,
) -> None:
    """Issue #812, second half: the broadcast sweep's pr_update_branch write was
    previously ungated by require_current_base entirely (only merge_ready's own
    deferral gate checked it). Prove the new gate now skips the compare-API
    read and the update-branch write together when protection says freshness
    isn't required, recording a distinct skipped_reason for telemetry.
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,  # broadcast
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.branch_protection_overrides["main"] = {"required_status_checks": {"strict": False}}
    fake_gh.prs = [
        {
            "number": 789,
            "title": "Fix #124: another",
            "url": "https://example.test/pull/789",
            "headRefName": "agent/issue-124-fix-another",
            "baseRefName": "main",
            "headRefOid": "sha-new456",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        }
    ]
    decision_dir = paths.prs / "pr-789"
    decision_dir.mkdir(parents=True, exist_ok=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-def456"}, indent=2),
        encoding="utf-8",
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    results = app._update_open_agent_prs(merged_pr_number=456)

    assert len(results) == 1
    assert results[0]["pr_number"] == 789
    assert results[0]["updated"] is False
    assert results[0]["skipped_reason"] == "base_freshness_not_required"
    assert fake_gh.pr_update_branch_calls == []


def test_update_open_agent_prs_next_mode_syncs_stale_clean_base(tmp_path: Path) -> None:
    """Issue #334: next-mode update lane syncs a CLEAN-but-stale head candidate."""
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
    # The next candidate becomes stale organically once PR 456 is merged and
    # advances the fake base tip, even though mergeStateStatus reports CLEAN.
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
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456-updated"


def test_update_open_agent_prs_all_mode_syncs_stale_clean_base(tmp_path: Path) -> None:
    """Issue #334: all-mode update lane syncs a PR with a CLEAN but stale merge-base."""
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

    # Merge PR 456 first so the base tip advances and the all-mode update lane
    # sees PR 789 as stale organically.
    fake_gh.merge_pr(456, "squash")

    results = app._update_open_agent_prs(merged_pr_number=456)

    assert len(results) == 1
    assert results[0]["pr_number"] == 789
    assert results[0]["updated"] is True
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456-updated"


def test_update_open_agent_prs_next_mode_syncs_head_of_queue(tmp_path: Path) -> None:
    """In merge-train mode, post-merge only syncs the head of the approved queue."""
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
            "headRefOid": "sha-def456",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
        {
            "number": 101,
            "title": "Fix #125: third",
            "url": "https://example.test/pull/101",
            "headRefName": "agent/issue-125-third",
            "headRefOid": "sha-ghi789",
            "mergeStateStatus": "BEHIND",
            "body": "Closes #125\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Approve in order: 456 first (head), then 789, then 101.
    for pr_number in (456, 789, 101):
        app.record_review(
            pr_number, "approved", summary="lgtm", verdict_provenance="fresh_llm_review"
        )
    # Override timestamps so fast tests don't all land in the same second.
    for idx, pr_number in enumerate((456, 789, 101)):
        decision_path = paths.prs / f"pr-{pr_number}" / "review-decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["reviewed_at"] = f"2026-07-12T00:00:0{idx}Z"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")

    # Merge the head of the queue.
    result = app.merge_ready(456, merge=True)
    assert result.ok is True
    assert result.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]

    # Post-merge only the next candidate (789) should be base-synced.
    update_results = result.data["update_open_prs_results"]
    assert update_results is not None
    assert len(update_results) == 1
    assert update_results[0]["pr_number"] == 789
    assert update_results[0]["updated"] is True
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456-updated"
    # The third PR should be untouched (not the head of the queue).
    assert fake_gh.prs[2]["headRefOid"] == "sha-ghi789"


def test_update_open_agent_prs_next_mode_skips_up_to_date_head(tmp_path: Path) -> None:
    """In merge-train mode, an up-to-date head candidate is not re-synced."""
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
            "headRefOid": "sha-def456",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #124\n\nTests: added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    # Model 789 as already rebased onto the post-merge base so the next
    # candidate is genuinely up-to-date and the merge-train skip path is exercised.
    post_merge_base = "main-merged-sha-abc123"
    fake_gh.commits[post_merge_base] = {"parents": [{"sha": "base-sha"}, {"sha": "sha-abc123"}]}
    fake_gh.commits["sha-def456"] = {"parents": [{"sha": post_merge_base}]}

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    app.record_review(789, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    # Ensure 456 is the head of the queue.
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
    assert update_results[0]["updated"] is False
    assert update_results[0]["skipped_reason"] == "up-to-date"
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456"


def test_update_open_agent_prs_next_mode_reports_compare_unavailable(tmp_path: Path) -> None:
    """Issue #337 rework: a None compare() result must not be reported as up-to-date.

    When the GitHub compare API is unavailable, `_is_base_current` returns None
    and the branch is correctly never synced (fail-closed), but the reported
    reason must be distinct from a genuinely up-to-date branch — otherwise a
    compare-API outage silently masquerades as every PR being current.
    """
    from charlie_work.config import AutoMergeConfig

    class FakeGitHubCompareUnavailable(FakeGitHub):
        """compare() returns None only for the given head SHA, simulating a
        compare-API outage isolated to that candidate (the merged PR's own
        base-freshness check must still resolve normally).
        """

        def __init__(self, unavailable_head: str) -> None:
            super().__init__()
            self._unavailable_head = unavailable_head

        def compare(self, base: str, head: str) -> dict[str, Any] | None:
            if head == self._unavailable_head:
                return None
            return super().compare(base, head)

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubCompareUnavailable(unavailable_head="sha-def456")
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
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
    assert update_results[0]["updated"] is False
    assert update_results[0]["skipped_reason"] == "compare_unavailable"
    # No update-branch call should have been made for the compare-unavailable PR.
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456"


def test_update_open_agent_prs_all_mode_reports_compare_unavailable(tmp_path: Path) -> None:
    """Issue #337 rework: all-mode update lane distinguishes compare-unavailable too."""
    from charlie_work.config import AutoMergeConfig

    class FakeGitHubCompareUnavailable(FakeGitHub):
        """compare() returns None only for the given head SHA, simulating a
        compare-API outage isolated to that candidate.
        """

        def __init__(self, unavailable_head: str) -> None:
            super().__init__()
            self._unavailable_head = unavailable_head

        def compare(self, base: str, head: str) -> dict[str, Any] | None:
            if head == self._unavailable_head:
                return None
            return super().compare(base, head)

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs=True,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubCompareUnavailable(unavailable_head="sha-def456")
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

    results = app._update_open_agent_prs(merged_pr_number=456)

    assert len(results) == 1
    assert results[0]["pr_number"] == 789
    assert results[0]["updated"] is False
    assert results[0]["skipped_reason"] == "compare_unavailable"
    # No update-branch call should have been made for the compare-unavailable PR.
    assert fake_gh.prs[1]["headRefOid"] == "sha-def456"


def test_update_open_agent_prs_merge_train_post_sync_head_race_rejected(
    tmp_path: Path,
) -> None:
    """If pr_view returns a non-qualifying head after update-branch, do not bless it.

    Regression test for the _update_open_agent_prs "next" path: a racing push
    must be rejected and the approved head left unchanged.
    """
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
            "headRefOid": "sha-def456",
            "mergeStateStatus": "BEHIND",
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

    # Simulate a racing push that lands on PR 789's branch between update and view.
    fake_gh.pr_head_shas[789] = "racing-sha"
    fake_gh.commits["racing-sha"] = {
        "parents": [{"sha": "other-sha"}],
        "committer": {"login": "not-web-flow"},
        "commit": {"committer": {"name": "Not GitHub"}},
    }

    results = app._update_open_agent_prs(merged_pr_number=456)

    assert len(results) == 1
    assert results[0]["pr_number"] == 789
    assert results[0]["updated"] is False
    assert results[0]["error"] == "post-sync head verification failed"
    # The approved head must remain unchanged.
    decision = json.loads(
        (paths.prs / "pr-789" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == "sha-def456"

    # dispatch_rework doesn't include skipped_issue_numbers in its result
