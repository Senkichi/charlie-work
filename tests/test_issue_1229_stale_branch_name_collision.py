"""Tests for issue #1229: issue-less rework episodes keyed by branch-name
number collide with unrelated PRs.

Root cause: ``linked_issue_number`` trusts a branch-name-derived issue number
unconditionally. A branch ``agent/issue-709-…`` left over from a merged
PR #709, reused by an unrelated issue-less PR (e.g. PR #1660), silently
binds the PR to issue 709. When the PR is routed to rework, the episode is
keyed under ``state["issues"]["709"]``, colliding with the unrelated
issue/PR #709's lifecycle.

Fix: ``linked_issue_number`` now accepts an optional
``branch_issue_validator`` callable. When the branch-name path produces a
candidate, the validator is called; if it returns False (the number is not
a real open issue), the binding is rejected and the function falls through
to the closing-keyword path. The rework-routing call sites
(``merge_ready``, ``review``, ``_dispatch_rework_impl``, ``loop``,
``record_review``) pass a validator built from ``issue_list(state="open")``
so a stale branch-name number can never key a rework episode under
``issues[<n>]``.
"""

from __future__ import annotations

from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work.config import AutoMergeConfig, OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp


def _stale_branch_pr() -> list[dict]:
    """A same-repo PR with a stale branch name ``agent/issue-709-…``.

    Issue #709 does not exist in the fake's issue list (it was merged long
    ago), so the branch-name validator must reject the binding. The PR body
    has no closing keyword, so the fall-through path also returns None —
    the PR is correctly treated as issue-less.
    """
    return [
        {
            "number": 1660,
            "title": "docs: update orchestrator docs",
            "url": "https://example.test/pull/1660",
            "headRefName": "agent/issue-709-job-cannon-docs-devin-orchestration",
            "baseRefName": "main",
            "headRefOid": "sha-stale-branch",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Docs maintenance. No issue — issue-less orchestrator rework.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]


def test_merge_ready_stale_branch_name_does_not_key_rework_under_wrong_issue(
    tmp_path: Path,
) -> None:
    """Issue #1229: a stale branch name must not create a rework episode
    under ``state["issues"]["709"]``.

    The PR has branch ``agent/issue-709-…`` but issue #709 is not open (not
    in the fake's issue list). Without the fix, ``linked_issue_number``
    would return 709 and the rework episode would be keyed under
    ``state["issues"]["709"]``, colliding with the unrelated PR #709. With
    the fix, the validator rejects 709, ``linked_issue_number`` returns
    None, and the PR is treated as issue-less — the same path as a fork PR
    with no linked issue.
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # The default FakeGitHub has issue #123 (OPEN) and PR #456. Replace
    # the PR list with our stale-branch PR. Issue #709 is NOT in
    # fake_gh.issues, so the validator will reject 709.
    fake_gh.prs = _stale_branch_pr()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Record an approved verdict so merge_ready enters the approved path.
    app.record_review(1660, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    result = app.merge_ready(1660, merge=False)

    assert result.ok is True
    assert result.data["can_merge"] is False
    assert result.data["merge_conflict"] is True
    assert result.data["merge_attempt_alarm"] is True
    warning = result.data["merge_attempt_warning"]
    assert warning is not None
    assert "no linked issue, cannot route to rework" in warning
    assert result.data["issue"] is None

    # The critical assertion: no rework episode under issues["709"].
    state = load_state(paths.state_file)
    assert "709" not in state.get("issues", {})
    # No rework event was emitted.
    assert not any(e["kind"] == "merge_conflict_rework_requested" for e in state.get("events", []))
    # No rework prompt was written.
    assert not (paths.prs / "pr-1660" / "rework-prompt.md").exists()


def test_make_branch_issue_validator_rejects_closed_or_nonexistent_issue(
    tmp_path: Path,
) -> None:
    """Issue #1229: ``_make_branch_issue_validator`` returns False for issue
    numbers that are not in the open-issue list."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Default issues: [{number: 123, state: OPEN}]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    validator = app._make_branch_issue_validator()
    assert validator is not None
    assert validator(123) is True  # real open issue
    assert validator(709) is False  # not in the open-issue list
    assert validator(999) is False  # doesn't exist at all


def test_make_branch_issue_validator_returns_none_on_api_failure(
    tmp_path: Path,
) -> None:
    """Issue #1229: when ``issue_list`` raises ``GitHubError``, the validator
    returns None so callers skip validation (preserve existing behavior)
    rather than blocking all rework routing during a transient outage."""
    from charlie_work.github import GitHubError

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class BrokenGitHub(FakeGitHub):
        def issue_list(self, labels=None, state=None):
            raise GitHubError("simulated API outage")

    app = OrchestratorApp(tmp_path, paths, config, BrokenGitHub())
    assert app._make_branch_issue_validator() is None
