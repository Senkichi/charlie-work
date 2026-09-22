"""Issue #1803 merged-PR mention-gate acceptance tests.

Split out of ``tests/test_charlie_work_dispatch_merged_prs.py`` so that
file stays under the repo's per-module line cap. Covers the two #1803
narrowings of ``scan_merged_pr_references`` -- the temporal hard rule (a
PR merged before the issue existed cannot have addressed it) and the
repo-qualifier suppression (``private issue #N``, ``owner/repo`` slug
qualifiers, managed-repo names) -- plus the fail-safe timestamp-unknown
path and the enriched ``dispatch_merged_pr_mention_flagged`` event
payload (``mentioning_prs`` / ``merged_at_unknown``).
"""

from __future__ import annotations

from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp


def test_dispatch_merged_pr_mention_excludes_pr_merged_before_issue_created(
    tmp_path: Path,
) -> None:
    """Issue #1803 temporal hard rule: a merged PR whose ``mergedAt`` predates
    the issue's ``createdAt`` cannot have addressed it -- the "issue #N" text
    refers to a different context (another repo, a docs example, a recycled
    number). This is the jobcannon incident verbatim: PR #43 merged
    2026-08-13 while issues #377/#391 were filed 2026-09-04, yet the mention
    gate flagged both (absence-of-disproof treated as evidence).
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    fake_gh.issues[0]["createdAt"] = "2026-09-04T12:00:00Z"
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "cleanup-unrelated-branch"
    fake_gh.prs[0]["title"] = "chore: unrelated cleanup"
    fake_gh.prs[0]["body"] = "While in the area, this also happens to fix issue #123."
    fake_gh.prs[0]["mergedAt"] = "2026-08-13T00:00:00Z"

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["merged_pr_flagged_issue_numbers"] == []
    assert result.data["merged_pr_referenced_issue_numbers"] == []
    assert result.data["merged_pr_closed_issue_numbers"] == []
    assert result.data["selected_count"] == 1
    assert (123, "agent:human-needed") not in fake_gh.labels_added
    assert 123 not in fake_gh.closed_issues
    state = load_state(paths.state_file)
    assert state["issues"].get("123", {}).get("merged_pr_mention_flagged_at") is None


def test_dispatch_merged_pr_mention_flags_pr_merged_after_issue_created(
    tmp_path: Path,
) -> None:
    """Positive control for the temporal rule: the SAME mention shape flags
    normally when the PR merged after the issue was created -- the exclusion
    only fires on a proven PR-before-issue ordering, it does not gut the
    mention gate."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    fake_gh.issues[0]["createdAt"] = "2026-09-04T12:00:00Z"
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "cleanup-unrelated-branch"
    fake_gh.prs[0]["title"] = "chore: unrelated cleanup"
    fake_gh.prs[0]["body"] = "While in the area, this also happens to fix issue #123."
    fake_gh.prs[0]["mergedAt"] = "2026-09-05T00:00:00Z"

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["merged_pr_flagged_issue_numbers"] == [123]
    assert result.data["selected_count"] == 0
    assert (123, "agent:human-needed") in fake_gh.labels_added
    state = load_state(paths.state_file)
    flagged = [
        e for e in state.get("events", []) if e.get("kind") == "dispatch_merged_pr_mention_flagged"
    ]
    assert len(flagged) == 1
    payload = flagged[0]["payload"]
    assert payload["issue_numbers"] == [123]
    # Issue #1803 observability: the event names the mentioning PR and its
    # merge timestamp, and reports that the temporal check DID run.
    assert payload["mentioning_prs"] == {
        "123": [
            {
                "number": 456,
                "merged_at": "2026-09-05T00:00:00Z",
                "merged_at_unknown": False,
            }
        ]
    }
    assert payload["merged_at_unknown"] == []


def test_dispatch_merged_pr_mention_missing_merged_at_flags_and_marks_unknown(
    tmp_path: Path,
) -> None:
    """Issue #1803 fail-safe: when the PR's merge timestamp is absent (or
    unparseable), the mention must still count -- the gate fails toward the
    pre-#1803 flagging behaviour, never silently open. The flag event marks
    the issue in ``merged_at_unknown`` so the fail-open share stays
    countable."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    fake_gh.issues[0]["createdAt"] = "2026-09-04T12:00:00Z"
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "cleanup-unrelated-branch"
    fake_gh.prs[0]["title"] = "chore: unrelated cleanup"
    fake_gh.prs[0]["body"] = "While in the area, this also happens to fix issue #123."
    # No mergedAt key at all -- the timestamp is unknown.

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["merged_pr_flagged_issue_numbers"] == [123]
    assert result.data["selected_count"] == 0
    assert (123, "agent:human-needed") in fake_gh.labels_added
    state = load_state(paths.state_file)
    flagged = [
        e for e in state.get("events", []) if e.get("kind") == "dispatch_merged_pr_mention_flagged"
    ]
    assert len(flagged) == 1
    payload = flagged[0]["payload"]
    assert payload["issue_numbers"] == [123]
    assert payload["merged_at_unknown"] == [123]
    assert payload["mentioning_prs"] == {
        "123": [{"number": 456, "merged_at": None, "merged_at_unknown": True}]
    }


def test_dispatch_merged_pr_mention_ignores_private_issue_qualifier(
    tmp_path: Path,
) -> None:
    """Issue #1803 qualifier rule: ``private issue #N`` names another repo's
    tracker (the jobcannon "private repo" docs rewrite that produced the
    false escalations) and must not count as mention coverage -- even with
    otherwise-current timestamps and no managed-repo context."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    fake_gh.issues[0]["createdAt"] = "2026-09-04T12:00:00Z"
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "cleanup-unrelated-branch"
    fake_gh.prs[0]["title"] = "docs: qualify bare issue refs"
    fake_gh.prs[0]["body"] = (
        "Rewrite bare references so the tracker is explicit: "
        "private issue #123 and private issues #124."
    )
    fake_gh.prs[0]["mergedAt"] = "2026-09-05T00:00:00Z"

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["merged_pr_flagged_issue_numbers"] == []
    assert result.data["merged_pr_referenced_issue_numbers"] == []
    assert result.data["selected_count"] == 1
    assert (123, "agent:human-needed") not in fake_gh.labels_added


def test_dispatch_merged_pr_mention_slug_qualifier_foreign_vs_self(
    tmp_path: Path,
) -> None:
    """Issue #1803 qualifier rule, ``<owner>/<repo>#N`` form: a slug
    qualifier is GitHub's own cross-repo reference syntax. A mention
    qualified with a DIFFERENT slug is suppressed; one qualified with the
    dispatching repo's own slug (``test-owner/test-repo`` on the fake) still
    counts -- the qualifier check is not a blanket slug suppression."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    fake_gh.issues[0]["createdAt"] = "2026-09-04T12:00:00Z"
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "cleanup-unrelated-branch"
    fake_gh.prs[0]["title"] = "chore: unrelated cleanup"
    fake_gh.prs[0]["body"] = (
        "Cross-references: issue test-owner/test-repo#123 and issue other-owner/elsewhere#123."
    )
    fake_gh.prs[0]["mergedAt"] = "2026-09-05T00:00:00Z"

    result = app.dispatch(limit=1)

    assert result.ok is True
    # The self-slug mention still flags; the foreign-slug mention of the
    # same number does not contribute either way.
    assert result.data["merged_pr_flagged_issue_numbers"] == [123]
    assert result.data["selected_count"] == 0


def test_dispatch_merged_pr_mention_ignores_foreign_slug_qualifier(
    tmp_path: Path,
) -> None:
    """Foreign-slug counterpart: when the ONLY mention carries another
    repo's slug, nothing flags -- the number belongs to that repo's
    tracker."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    fake_gh.issues[0]["createdAt"] = "2026-09-04T12:00:00Z"
    fake_gh.prs[0]["state"] = "MERGED"
    fake_gh.prs[0]["headRefName"] = "cleanup-unrelated-branch"
    fake_gh.prs[0]["title"] = "chore: unrelated cleanup"
    fake_gh.prs[0]["body"] = "See issue other-owner/elsewhere#123 for context."
    fake_gh.prs[0]["mergedAt"] = "2026-09-05T00:00:00Z"

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["merged_pr_flagged_issue_numbers"] == []
    assert result.data["merged_pr_referenced_issue_numbers"] == []
    assert result.data["selected_count"] == 1
    assert (123, "agent:human-needed") not in fake_gh.labels_added
