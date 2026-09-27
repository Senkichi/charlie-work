"""Tests for Aviator MergeQueue parallel-mode support.

Problem 1: ``_should_update_pr_branch`` must skip PRs carrying the
configured ``mergequeue`` label -- the queue handles rebasing itself.

Problem 2: Aviator's draft PRs (authored by ``queue_bot_login`` on
``mq-bot-*`` branches) must be invisible to the fleet's reconcile loop
and mergequeue detectors.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from charlie_work.config import OrchestratorConfig
from charlie_work.reconcile import (
    detect_aviator_stale_blocked,
    detect_drift,
    detect_mergequeue_not_approved,
    detect_mergequeue_wedged,
    is_queue_bot_pr,
)
from charlie_work.state import empty_state
from _reconcile_fixtures import FakeGitHub, _pr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(
    *,
    mergequeue_label: str = "mergequeue",
    queue_bot_login: str = "aviator-app[bot]",
    branch_prefix: str = "agent/issue",
) -> OrchestratorConfig:
    return replace(
        OrchestratorConfig(),
        auto_merge=replace(
            OrchestratorConfig().auto_merge,
            mergequeue_label=mergequeue_label,
            queue_bot_login=queue_bot_login,
        ),
        dispatch=replace(
            OrchestratorConfig().dispatch,
            branch_prefix=branch_prefix,
        ),
    )


def _aviator_draft_pr(
    number: int,
    state: str = "OPEN",
    *,
    branch: str = "mq-bot-abc123",
) -> dict[str, Any]:
    """A PR created by Aviator parallel mode on a combo draft branch."""
    return {
        "number": number,
        "title": f"Aviator queue validation ({branch})",
        "url": f"https://example.test/pull/{number}",
        "headRefName": branch,
        "baseRefName": "main",
        "body": "",
        "state": state,
        "labels": [],
        "author": {"login": "aviator-app[bot]"},
        "isDraft": True,
        "isCrossRepository": False,
        "headRefOid": f"sha-{number}",
        "closedAt": None,
    }


def _fleet_pr(
    number: int,
    state: str = "OPEN",
    *,
    labels: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """A regular fleet PR on an agent/issue-* branch."""
    return {
        **_pr(number, state),
        "author": {"login": "Senkichi"},
        "isDraft": False,
        "headRefOid": f"sha-{number}",
        "labels": labels or [],
    }


# ---------------------------------------------------------------------------
# is_queue_bot_pr
# ---------------------------------------------------------------------------


class TestIsQueueBotPr:
    def test_matches_configured_bot_login(self) -> None:
        pr = _aviator_draft_pr(9001)
        assert is_queue_bot_pr(pr, _config()) is True

    def test_does_not_match_fleet_pr(self) -> None:
        pr = _fleet_pr(42)
        assert is_queue_bot_pr(pr, _config()) is False

    def test_disabled_when_bot_login_unset(self) -> None:
        config = replace(
            _config(),
            auto_merge=replace(_config().auto_merge, queue_bot_login=None),
        )
        pr = _aviator_draft_pr(9001)
        assert is_queue_bot_pr(pr, config) is False

    def test_no_author_field(self) -> None:
        pr = {**_aviator_draft_pr(9001)}
        del pr["author"]
        assert is_queue_bot_pr(pr, _config()) is False

    def test_author_not_dict(self) -> None:
        pr = {**_aviator_draft_pr(9001), "author": "aviator-app[bot]"}
        assert is_queue_bot_pr(pr, _config()) is False


# ---------------------------------------------------------------------------
# Problem 1: _should_update_pr_branch skips mergequeue-labeled PRs
# ---------------------------------------------------------------------------


class TestShouldUpdateSkipsMergequeue:
    """Verify the mergequeue-label guard in ``_should_update_pr_branch``."""

    def test_mergequeue_labeled_pr_not_updated(self, tmp_path: Path) -> None:
        """A PR carrying the mergequeue label is never synced."""
        from charlie_work.paths import runtime_paths
        from charlie_work.workflow import OrchestratorApp
        from _fakes_github import FakeGitHub as SuiteFakeGitHub

        config = _config()
        paths = runtime_paths(tmp_path, config.runtime.state_dir)
        paths.ensure()
        fake_gh = SuiteFakeGitHub()
        app = OrchestratorApp(tmp_path, paths, config, fake_gh)

        pr = {
            "number": 100,
            "headRefName": "agent/issue-100-x",
            "labels": [{"name": "mergequeue"}],
            "mergeStateStatus": "BEHIND",
        }
        assert app._should_update_pr_branch(pr) is False

    def test_non_mergequeue_pr_behind_is_updated(self, tmp_path: Path) -> None:
        """A PR without the mergequeue label that is behind is synced."""
        from charlie_work.paths import runtime_paths
        from charlie_work.workflow import OrchestratorApp
        from _fakes_github import FakeGitHub as SuiteFakeGitHub

        config = _config()
        paths = runtime_paths(tmp_path, config.runtime.state_dir)
        paths.ensure()
        fake_gh = SuiteFakeGitHub()
        app = OrchestratorApp(tmp_path, paths, config, fake_gh)

        pr = {
            "number": 100,
            "headRefName": "agent/issue-100-x",
            "labels": [],
            "mergeStateStatus": "BEHIND",
        }
        assert app._should_update_pr_branch(pr) is True

    def test_mergequeue_label_with_base_current_false(self, tmp_path: Path) -> None:
        """Even with base_current=False, a mergequeue-labeled PR is skipped."""
        from charlie_work.paths import runtime_paths
        from charlie_work.workflow import OrchestratorApp
        from _fakes_github import FakeGitHub as SuiteFakeGitHub

        config = _config()
        paths = runtime_paths(tmp_path, config.runtime.state_dir)
        paths.ensure()
        fake_gh = SuiteFakeGitHub()
        app = OrchestratorApp(tmp_path, paths, config, fake_gh)

        pr = {
            "number": 100,
            "headRefName": "agent/issue-100-x",
            "labels": [{"name": "mergequeue"}],
        }
        assert app._should_update_pr_branch(pr, base_current=False) is False

    def test_no_mergequeue_config_allows_update(self, tmp_path: Path) -> None:
        """With mergequeue_label unset, the guard is inert."""
        from charlie_work.paths import runtime_paths
        from charlie_work.workflow import OrchestratorApp
        from _fakes_github import FakeGitHub as SuiteFakeGitHub

        config = replace(
            _config(),
            auto_merge=replace(_config().auto_merge, mergequeue_label=None),
        )
        paths = runtime_paths(tmp_path, config.runtime.state_dir)
        paths.ensure()
        fake_gh = SuiteFakeGitHub()
        app = OrchestratorApp(tmp_path, paths, config, fake_gh)

        pr = {
            "number": 100,
            "headRefName": "agent/issue-100-x",
            "labels": [{"name": "mergequeue"}],
            "mergeStateStatus": "BEHIND",
        }
        # No configured label -> guard is a no-op; BEHIND triggers update
        assert app._should_update_pr_branch(pr) is True


# ---------------------------------------------------------------------------
# Problem 2: Aviator draft PRs invisible to reconcile
# ---------------------------------------------------------------------------


class TestAviatorDraftPrsInvisibleToDetectDrift:
    """Queue-bot PRs must not produce drift items in detect_drift."""

    def test_merged_aviator_pr_no_drift(self, tmp_path: Path) -> None:
        """A merged Aviator draft PR does not emit merged_outside_orchestrator."""
        config = _config()
        aviator_pr = _aviator_draft_pr(9001, "MERGED")
        fleet_pr = _fleet_pr(42, "MERGED")
        gh = FakeGitHub(
            prs=[aviator_pr, fleet_pr],
            issues=[],
        )
        state = empty_state()
        drift = detect_drift(gh, state, config, repo_root=tmp_path)
        # The fleet PR should produce a drift item, the Aviator one should not
        pr_numbers = {d.pr_number for d in drift if d.kind == "merged_outside_orchestrator"}
        assert 9001 not in pr_numbers
        assert 42 in pr_numbers

    def test_closed_aviator_pr_no_drift(self, tmp_path: Path) -> None:
        """A closed Aviator draft PR does not emit closed_unmerged_pr_* items."""
        config = _config()
        aviator_pr = _aviator_draft_pr(9001, "CLOSED")
        gh = FakeGitHub(
            prs=[aviator_pr],
            issues=[],
        )
        state = empty_state()
        drift = detect_drift(gh, state, config, repo_root=tmp_path)
        pr_numbers = {d.pr_number for d in drift}
        assert 9001 not in pr_numbers


class TestAviatorDraftPrsInvisibleToMergequeueDetectors:
    """Queue-bot PRs must not be processed by mergequeue detectors."""

    def test_detect_mergequeue_not_approved_skips_bot_pr(self, tmp_path: Path) -> None:
        config = _config()
        # A bot PR carrying the mergequeue label (shouldn't happen, but
        # defense-in-depth)
        bot_pr = {
            **_aviator_draft_pr(9001),
            "labels": [{"name": "mergequeue"}],
        }
        gh = FakeGitHub(prs=[bot_pr], issues=[])
        drift = detect_mergequeue_not_approved(gh, config, repo_root=tmp_path)
        assert drift == []

    def test_detect_mergequeue_wedged_skips_bot_pr(self) -> None:
        config = _config()
        bot_pr = {
            **_aviator_draft_pr(9001),
            "labels": [{"name": "mergequeue"}],
        }
        gh = FakeGitHub(prs=[bot_pr], issues=[])
        state = empty_state()
        drift = detect_mergequeue_wedged(gh, config, state)
        assert drift == []

    def test_detect_aviator_stale_blocked_skips_bot_pr(self) -> None:
        config = _config()
        bot_pr = {
            **_aviator_draft_pr(9001),
            "labels": [{"name": "blocked"}],
        }
        gh = FakeGitHub(prs=[bot_pr], issues=[])
        drift = detect_aviator_stale_blocked(gh, config)
        assert drift == []


# ---------------------------------------------------------------------------
# Branch-prefix structural exclusion (existing; verified here)
# ---------------------------------------------------------------------------


class TestBranchPrefixExcludesAviatorBranches:
    """The fleet's branch_prefix filter naturally excludes Aviator branches.

    This test codifies the structural guarantee: even without the
    is_queue_bot_pr filter, any Aviator-created branch (draft-PR-bearing
    ``mq-bot-*`` or internal PR-less ``mq-tmp-*``) is excluded from
    fleet operations because neither matches 'agent/issue'.
    """

    def test_linked_issue_number_returns_none_for_mq_branch(self) -> None:
        from charlie_work.issue_linking import linked_issue_number

        pr = _aviator_draft_pr(9001, branch="mq-bot-abc123")
        result = linked_issue_number(
            pr,
            is_cross_repository=False,
            branch_prefix="agent/issue",
        )
        assert result is None
