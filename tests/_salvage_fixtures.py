"""Shared fixtures for the salvage-PR-path tests.

Hoisted out of ``test_issue_956.py`` (issue #1284): a minimal fake for the
``GitHubLike`` surface the salvage helpers use, and a label-set builder,
both imported by other test modules exercising the same salvage paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work.config import OrchestratorConfig
from charlie_work.github import MergedPRSearchResult


class _SalvageTestGitHub:
    """Minimal fake for the ``GitHubLike`` surface used by the salvage helpers."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        pr_create_return: int | None = 101,
        remove_ok: bool = True,
        add_ok: bool = True,
        repo_slug: str = "owner/repo",
        closing_issue_numbers: list[int] | None = None,
        pr_view_raises: bool = False,
    ) -> None:
        self.repo_root = repo_root
        self.dry_run = False
        self.pr_create_return = pr_create_return
        self._remove_ok = remove_ok
        self._add_ok = add_ok
        self._repo_slug = repo_slug
        # None means "same as the created PR's own issue number" -- set by
        # each test via `closing_issue_numbers_override` when it needs to
        # simulate a mismatch; the default keeps existing tests (which never
        # exercise the post-create probe) unaffected.
        self._closing_issue_numbers = closing_issue_numbers
        self._pr_view_raises = pr_view_raises
        self.prs_created: list[dict[str, Any]] = []
        self.labels_removed: list[tuple[int, str]] = []
        self.labels_added: list[tuple[int, str]] = []
        self.pr_view_calls: list[int] = []

    def pr_create(self, head: str, base: str, title: str, body: str) -> int | None:
        self.prs_created.append({"head": head, "base": base, "title": title, "body": body})
        return self.pr_create_return

    def merged_prs_for_issue(self, issue_number: int, branch_prefix: str) -> MergedPRSearchResult:
        # Issue #1221: ``_attempt_salvage`` now re-checks live terminal state
        # before opening a PR, calling ``merged_prs_for_issue`` on every path.
        # Match the production shape (``MergedPRSearchResult`` carrying ``.ok``)
        # so the fake and the real GitHubCLI agree. These tests exercise the
        # salvage-proceeds path, so return an empty (ok) result -- no merged PR
        # binds to the issue, and salvage falls through to opening the PR.
        return MergedPRSearchResult([], ok=True)

    def remove_issue_label(self, number: int, label: str) -> bool:
        self.labels_removed.append((number, label))
        return self._remove_ok

    def add_issue_label(self, number: int, label: str) -> bool:
        self.labels_added.append((number, label))
        return self._add_ok

    def name_with_owner(self) -> str:
        return self._repo_slug

    def pr_view(self, number: int, *, fields: str = "") -> dict[str, Any]:
        self.pr_view_calls.append(number)
        if self._pr_view_raises:
            raise RuntimeError("gh pr view unavailable")
        numbers = self._closing_issue_numbers
        if numbers is None:
            return {"closingIssuesReferences": []}
        return {"closingIssuesReferences": [{"number": n} for n in numbers]}


def _salvage_labels(config: OrchestratorConfig) -> tuple[set[str], set[str]]:
    active = {config.labels.in_progress}
    issue = {config.labels.in_progress}
    return active, issue
