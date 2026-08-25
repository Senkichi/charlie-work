"""Shared fixtures for ``test_reconcile.py``'s externally-imported surface.

Hoisted out of ``test_reconcile.py`` (issue #1284): its own minimal
reconcile-pass ``FakeGitHub`` double, PR/issue payload builders, and a
bare-remote-plus-clone / completed-worktree pair of git fixture builders,
all imported by other test modules. ``test_reconcile.py`` is one of the
three monoliths issue #1284 marks out of scope for a full split -- only
these five exported symbols move; the rest of the file (including its own
~14+ internal uses of ``FakeGitHub`` below) is untouched, via a
back-reference import.

This ``FakeGitHub`` is unrelated to ``tests/_fakes_github.py``'s
``FakeGitHub`` (the orchestrator-suite-wide fake hoisted out of
``test_charlie_work.py``) -- it is a narrower, reconcile-pass-only double
that happens to share the name. The two must never be merged, aliased to
the same bare name in one module, or made to subclass one another.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from _worktree_fixtures import _git
from charlie_work.reconcile import _LIST_LIMIT as reconcile_list_limit
from charlie_work.worktree import create_worktree


class FakeGitHub:
    """Records every call so tests can assert detect_drift never mutates."""

    def __init__(
        self,
        *,
        prs: list[dict[str, Any]],
        issues: list[dict[str, Any]],
        fail_add_labels: set[tuple[int, str]] | None = None,
        fail_remove_labels: set[tuple[int, str]] | None = None,
        repo_root: Any = None,
        pr_create_return: int | None = None,
        rate_limit_sufficient: bool = True,
        rate_limit_remaining: int = 10000,
        rate_limit_reset: int = 0,
    ) -> None:
        self._prs = prs
        self._issues = issues
        self.run_calls: list[list[str]] = []
        self.labels_added: list[tuple[int, str]] = []
        self.labels_removed: list[tuple[int, str]] = []
        self._fail_add_labels = fail_add_labels or set()
        self._fail_remove_labels = fail_remove_labels or set()
        self.repo_root = repo_root
        self.prs_created: list[dict[str, Any]] = []
        self.pr_create_return = pr_create_return
        self._rate_limit_sufficient = rate_limit_sufficient
        self._rate_limit_remaining = rate_limit_remaining
        self._rate_limit_reset = rate_limit_reset
        # PR-scoped label tracking, distinct from the issue-scoped lists above.
        self.pr_labels_added: list[tuple[int, str]] = []
        self.pr_labels_removed: list[tuple[int, str]] = []
        self._fail_add_pr_labels: set[tuple[int, str]] = set()
        self._fail_remove_pr_labels: set[tuple[int, str]] = set()
        # sha -> list of check-run dicts, for detect_aviator_stale_blocked.
        self.check_runs_by_sha: dict[str, list[dict[str, Any]]] = {}
        self.commit_check_runs_calls: list[str] = []

    def run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):
        self.run_calls.append(args)
        if args[:2] == ["pr", "list"]:
            return self._prs
        if args[:2] == ["issue", "list"]:
            # Model the real ``gh issue list --state all --limit 500`` cap for
            # mutation tests that revert _fetch_issues to the pre-#762 path.
            return self._issues[:reconcile_list_limit]
        if args[0] == "api" and "pulls?state=all" in args[1]:
            url = args[1]
            page_match = re.search(r"[?&]page=(\d+)", url)
            page = int(page_match.group(1)) if page_match else 1
            per_page_match = re.search(r"[?&]per_page=(\d+)", url)
            per_page = int(per_page_match.group(1)) if per_page_match else 100
            start = (page - 1) * per_page
            return self._prs[start : start + per_page]
        if args[0] == "api" and "issues?state=all" in args[1]:
            url = args[1]
            page_match = re.search(r"[?&]page=(\d+)", url)
            page = int(page_match.group(1)) if page_match else 1
            per_page_match = re.search(r"[?&]per_page=(\d+)", url)
            per_page = int(per_page_match.group(1)) if per_page_match else 100
            start = (page - 1) * per_page
            return self._issues[start : start + per_page]
        if args[0] == "api":
            return [] if json_output else ""
        raise AssertionError(f"unexpected gh.run call: {args}")

    def add_issue_label(self, number: int, label: str) -> bool:
        self.labels_added.append((number, label))
        return (number, label) not in self._fail_add_labels

    def remove_issue_label(self, number: int, label: str) -> bool:
        self.labels_removed.append((number, label))
        return (number, label) not in self._fail_remove_labels

    def add_pr_label(self, number: int, label: str) -> bool:
        self.pr_labels_added.append((number, label))
        return (number, label) not in self._fail_add_pr_labels

    def remove_pr_label(self, number: int, label: str) -> bool:
        self.pr_labels_removed.append((number, label))
        return (number, label) not in self._fail_remove_pr_labels

    def commit_check_runs(self, sha: str) -> list[dict[str, Any]] | None:
        self.commit_check_runs_calls.append(sha)
        return self.check_runs_by_sha.get(sha)

    def pr_create(self, head: str, base: str, title: str, body: str) -> int | None:
        self.prs_created.append({"head": head, "base": base, "title": title, "body": body})
        return self.pr_create_return

    def check_graphql_rate_limit(self, threshold: int) -> tuple[bool, int, int | None]:
        return (
            self._rate_limit_sufficient,
            self._rate_limit_remaining,
            self._rate_limit_reset,
        )

    def name_with_owner(self) -> str:
        return "owner/test-repo"


def _pr(
    number: int,
    state: str = "OPEN",
    *,
    head_ref: str | None = None,
    body: str = "",
    title: str = "",
    is_cross_repository: bool = False,
    closed_at: str | None = None,
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "url": f"https://example.test/pull/{number}",
        "headRefName": head_ref or f"agent/issue-{number}-x",
        "baseRefName": "main",
        "body": body,
        "state": state,
        "labels": [],
        "isCrossRepository": is_cross_repository,
        # Issue #1398: closedAt is part of RECONCILE_PR_FIELDS so the
        # closed-unmerged convergence rules can compare the PR's close time
        # against the issue's active-session start. None for OPEN PRs.
        "closedAt": closed_at,
    }


def _issue(number: int, labels: list[str], state: str = "OPEN") -> dict[str, Any]:
    return {
        "number": number,
        "title": f"issue {number}",
        "url": f"https://example.test/issues/{number}",
        "body": "",
        "labels": [{"name": label} for label in labels],
        "state": state,
    }


def _init_bare_remote_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare remote repo and a local clone, return (remote, clone)."""
    remote = tmp_path / "remote"
    remote.mkdir(parents=True, exist_ok=True)
    _git(remote, "init", "--bare", "--initial-branch=main")
    clone = tmp_path / "clone"
    clone.mkdir(parents=True, exist_ok=True)
    _git(clone, "init", "--initial-branch=main")
    _git(clone, "config", "user.email", "test@example.test")
    _git(clone, "config", "user.name", "Test User")
    _git(clone, "config", "commit.gpgSign", "false")
    _git(clone, "remote", "add", "origin", str(remote))
    (clone / "README.md").write_text("hello\n", encoding="utf-8")
    _git(clone, "add", "README.md")
    _git(clone, "commit", "-m", "initial commit")
    _git(clone, "push", "-u", "origin", "main")
    return remote, clone


def _setup_completed_worktree(
    repo_root: Path, issue_number: int, dirty: bool = False
) -> tuple[Path, str]:
    """Create a worktree with one commit beyond origin/main. Return (worktree_path, branch)."""
    branch = f"agent/issue-{issue_number}"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    (info.path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(info.path, "add", "feature.txt")
    _git(info.path, "commit", "-m", "feature commit")
    if dirty:
        (info.path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    return info.path, branch
