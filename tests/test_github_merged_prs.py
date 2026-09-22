"""Merged-PR enumeration tests: ``merged_pr_list`` REST pagination,
``merged_prs_for_issue`` search binding, and the ``_normalize_rest_pr``
field contract the two producers share.

Split out of ``tests/test_github.py`` (issue #1572, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_github_fixtures.py``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from charlie_work import github as github_module
from _github_fixtures import _read_fixture


def test_merged_pr_list_uses_rest_pagination_and_filters_merged(
    monkeypatch, tmp_path: Path
) -> None:
    """merged_pr_list() now paginates through the REST pulls endpoint and
    filters to merged PRs, avoiding the GraphQL query entirely.
    """
    page1 = [
        {
            "number": 1,
            "title": "x",
            "body": "",
            "head": {
                "ref": "agent/issue-1-x",
                "sha": "aaaa1111",
                "repo": {"full_name": "owner/repo"},
            },
            "base": {"repo": {"full_name": "owner/repo"}},
            "merged_at": "2026-07-21T20:00:00Z",
            "state": "closed",
        },
        {
            "number": 2,
            "title": "closed not merged",
            "body": "",
            "head": {"ref": "other", "repo": {"full_name": "owner/repo"}},
            "base": {"repo": {"full_name": "owner/repo"}},
            "merged_at": None,
            "state": "closed",
        },
    ]
    call_log: list[list[str]] = []
    pull_call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal pull_call_count
        call_log.append(cmd)
        if cmd[:2] == ["gh", "api"] and "pulls" in cmd[2]:
            pull_call_count += 1
            if pull_call_count == 1:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=json.dumps(page1), stderr=""
                )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.merged_pr_list()

    assert result == [
        {
            "number": 1,
            "title": "x",
            "body": "",
            "headRefName": "agent/issue-1-x",
            "isCrossRepository": False,
            "state": "MERGED",
            "headRefOid": "aaaa1111",
            "mergeCommitOid": None,
            "mergedAt": "2026-07-21T20:00:00Z",
        }
    ]
    assert pull_call_count >= 1
    assert any("pulls?state=closed" in c[2] for c in call_log)
    assert not any(c[:2] == ["gh", "pr"] for c in call_log)


def test_merged_pr_list_raises_on_rest_pagination_error(monkeypatch, tmp_path: Path) -> None:
    """A terminal REST failure during pagination raises GitHubError."""
    merged_pr = {
        "number": 1,
        "title": "x",
        "body": "",
        "head": {"ref": "agent/issue-1-x", "repo": {"full_name": "owner/repo"}},
        "base": {"repo": {"full_name": "owner/repo"}},
        "merged_at": "2026-07-21T20:00:00Z",
        "state": "closed",
    }
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps([merged_pr]), stderr=""
            )
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="HTTP 401: Bad credentials"
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    with pytest.raises(github_module.GitHubError):
        gh.merged_pr_list()

    assert call_count == 2


def test_merged_pr_list_raises_on_empty_stdout_not_silent_empty(
    monkeypatch, tmp_path: Path
) -> None:
    """gh exiting 0 with empty stdout is an unusable response, not an empty page.

    As of issue #756, run() itself raises GitHubError for that case (the
    ambiguous success-with-empty-stdout path). Before #756 it returned None,
    which merged_pr_list's own isinstance check also raised on — this test
    is unaffected by where the raise originates. A genuine empty page comes
    back as the JSON array ``[]`` (a list). The previous idiom
    ``result if isinstance(result, list) else []`` coerced None to [] and
    silently broke the pagination loop, returning [] as though the repository
    had no merged PRs — indistinguishable from a successful empty fetch. This
    is the silent-empty path that would arm the #502 post-merge tripwire with
    an empty baseline and leave it permanently blind (issue #633).
    """
    call_count = 0

    def fake_run(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        # gh exits 0 with empty stdout — run() now raises GitHubError directly.
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    with pytest.raises(github_module.GitHubError):
        gh.merged_pr_list()

    # The first page is where the unusable response is detected.
    assert call_count == 1


def test_merged_pr_list_empty_page_terminates_cleanly(monkeypatch, tmp_path: Path) -> None:
    """A genuine empty page (``[]``) terminates pagination without raising.

    This is the positive counterpart to test_merged_pr_list_raises_on_empty_stdout:
    a real empty page is a list (``[]``) and must keep being treated as "no more
    results", not as an unusable response.
    """
    responses = ["[]"]

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=responses.pop(0), stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.merged_pr_list()

    assert result == []


def test_merged_prs_for_issue_returns_bound_pr_without_graphql_budget_check(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #433: per-issue merged-PR lookup bypasses the 500-window cap and does
    not consume the GraphQL budget check used by merged_pr_list().
    """
    search_json = _read_fixture("gh_pr_list_search_merged.json")
    call_log: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        call_log.append(cmd)
        if cmd[:3] == ["gh", "api", "rate_limit"]:
            # Should not be called; this method is intentionally budget-agnostic.
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="{}", stderr="")
        if cmd[:2] == ["gh", "pr"] and "--search" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=search_json, stderr=""
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.merged_prs_for_issue(326, branch_prefix="agent/issue")

    assert len(result) == 1
    assert result[0]["number"] == 335
    assert result[0]["state"] == "MERGED"
    search_calls = [c for c in call_log if c[:2] == ["gh", "pr"] and "--search" in c]
    assert len(search_calls) == 1
    assert search_calls[0] == [
        "gh",
        "pr",
        "list",
        "--state",
        "merged",
        "--search",
        '"#326"',
        "--limit",
        "20",
        "--json",
        github_module.MERGED_PR_LIST_FIELDS,
    ]
    assert not any(c[:3] == ["gh", "api", "rate_limit"] for c in call_log)


def test_merged_prs_for_issue_returns_empty_when_pr_is_not_bound(
    monkeypatch, tmp_path: Path
) -> None:
    """A merged PR that only mentions the issue (no branch prefix / closing keyword)
    must not be returned, even if the issue number appears in its title/body.
    """
    prs = [
        {
            "number": 1,
            "title": "chore: unrelated",
            "body": "While in the area, this also happens to fix issue #326.",
            "headRefName": "unrelated-cleanup",
            "isCrossRepository": False,
            "state": "MERGED",
        }
    ]

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["gh", "pr"] and "--search" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=json.dumps(prs), stderr=""
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.merged_prs_for_issue(326, branch_prefix="agent/issue")

    assert result == []


def test_merged_prs_for_issue_returns_empty_on_gh_failure(monkeypatch, tmp_path: Path) -> None:
    """Per-issue lookup is best-effort; a non-zero gh exit returns an empty list."""

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["gh", "pr"] and "--search" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="HTTP 502: Bad gateway",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    result = gh.merged_prs_for_issue(326, branch_prefix="agent/issue")

    assert result == []
    assert result.ok is False


# --- merged-PR field contract: the REST normalizer must reproduce exactly the
# key set that merged_prs_for_issue() gets from `gh pr list --json`. These are
# two independent producers of the same value shape; when they drift, consumers
# reading a field the normalizer forgot silently see None on the REST path
# (which is the only path merged_pr_list() uses) while every FakeGitHub-based
# test keeps passing, because those fixtures hand-write the richer shape.


def test_normalize_rest_pr_satisfies_merged_pr_list_field_contract() -> None:
    """_normalize_rest_pr() must emit exactly MERGED_PR_LIST_FIELDS plus the
    declared REST-only extras.

    MERGED_PR_LIST_FIELDS is the single source of truth for the shared shape;
    MERGED_PR_REST_ONLY_FIELDS declares the fields gh's `--json` list cannot
    express (see the comment at its definition). This asserts the REST path
    honors both rather than restating the field lists, so adding a field to
    either contract forces the normalizer to supply it — and an undeclared
    extra still fails, keeping the two producers' drift visible.
    """
    expected: set[str] = set(github_module.MERGED_PR_LIST_FIELDS.split(",")) | set(
        github_module.MERGED_PR_REST_ONLY_FIELDS
    )

    gh = github_module.GitHub(Path("."))
    normalized = gh._normalize_rest_pr(
        {
            "number": 501,
            "title": "fix: something",
            "body": "Closes #494",
            "merged_at": "2026-07-20T20:19:07Z",
            "head": {
                "ref": "agent/issue-494-fix-something",
                "sha": "27a20fbdc0ffee0123456789abcdef0123456789",
                "repo": {"full_name": "Senkichi/charlie-work"},
            },
            "base": {"repo": {"full_name": "Senkichi/charlie-work"}},
        }
    )

    assert set(normalized) == expected, (
        "REST normalizer drifted from MERGED_PR_LIST_FIELDS: "
        f"missing={sorted(expected - set(normalized))} "
        f"extra={sorted(set(normalized) - expected)}"
    )


def test_normalize_rest_pr_maps_head_sha_to_head_ref_oid() -> None:
    """REST spells the merged head OID `head.sha`; consumers read gh's GraphQL
    name `headRefOid`. Post-merge audits use it to prove *which* commit was
    merged, so a None here silently defeats any approved-SHA comparison."""
    gh = github_module.GitHub(Path("."))

    normalized = gh._normalize_rest_pr(
        {
            "number": 1,
            "head": {"ref": "topic", "sha": "deadbeef", "repo": {"full_name": "o/r"}},
            "base": {"repo": {"full_name": "o/r"}},
        }
    )

    assert normalized["headRefOid"] == "deadbeef"


def test_normalize_rest_pr_maps_merge_commit_sha_to_merge_commit_oid() -> None:
    """Issue #1194: REST spells the landing merge commit `merge_commit_sha`;
    the #502 tripwire's queue-sync-merge recognition anchors its reachability
    check at this commit's first parent, and reads it as `mergeCommitOid`
    (gh's GraphQL-style naming, matching headRefOid's convention). A None
    here silently defeats condition 3 of `_queue_sync_merge_covered` for
    every REST-sourced merged PR."""
    gh = github_module.GitHub(Path("."))

    normalized = gh._normalize_rest_pr(
        {
            "number": 1,
            "head": {"ref": "topic", "sha": "deadbeef", "repo": {"full_name": "o/r"}},
            "base": {"repo": {"full_name": "o/r"}},
            "merge_commit_sha": "c0ffee",
        }
    )

    assert normalized["mergeCommitOid"] == "c0ffee"


def test_normalize_rest_pr_merge_commit_oid_is_none_when_absent() -> None:
    """A REST payload with no `merge_commit_sha` (should not happen for a
    genuinely merged PR, but the field is attacker/API-controlled input) must
    map to None rather than KeyError, so `_queue_sync_merge_covered` sees its
    documented fail-closed `not merge_commit_sha` branch instead of crashing
    the tripwire pass (issue #1194)."""
    gh = github_module.GitHub(Path("."))

    normalized = gh._normalize_rest_pr(
        {
            "number": 1,
            "head": {"ref": "topic", "sha": "deadbeef", "repo": {"full_name": "o/r"}},
            "base": {"repo": {"full_name": "o/r"}},
        }
    )

    assert normalized["mergeCommitOid"] is None


def test_merged_pr_list_exposes_head_ref_oid_end_to_end(monkeypatch, tmp_path: Path) -> None:
    """The full REST path — not just the normalizer — must surface headRefOid,
    since merged_pr_list() is REST-only by construction (issue #361)."""
    page = [
        {
            "number": 501,
            "title": "fix: something",
            "body": "",
            "merged_at": "2026-07-20T20:19:07Z",
            "head": {"ref": "agent/issue-494", "sha": "27a20fbd", "repo": {"full_name": "o/r"}},
            "base": {"repo": {"full_name": "o/r"}},
        }
    ]
    responses = [json.dumps(page), "[]"]

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=responses.pop(0), stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    gh = github_module.GitHub(tmp_path)
    merged = gh.merged_pr_list()

    assert len(merged) == 1
    assert merged[0]["headRefOid"] == "27a20fbd"
