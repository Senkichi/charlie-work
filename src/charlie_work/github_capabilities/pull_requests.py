"""Pull-requests capability: PR read/create surface (Track 2, issue #1585).

Cluster E of the design doc's capability segmentation (Section 3.1):
``pr_create``, ``pr_view``, ``pr_list``, ``pr_diff``, ``pr_commits``,
``pr_ready``, ``merged_pr_list``, ``merged_prs_for_issue``.

``merged_prs_for_issue``/``merged_pr_list`` are an ambiguity call
(Section 3.1): pinned here because they return PR data even though
``merged_prs_for_issue`` is issue-keyed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ._base import CapabilityCollaborator

if TYPE_CHECKING:
    # Runtime import would cycle (github.py imports this module to build the
    # GitHubLike union); inspect.signature does not evaluate annotations
    # (eval_str=False by default), so the bare string name is all the
    # conformance test's signature comparison needs.
    #
    # GitHubRunResult itself lives in _base.py, not charlie_work.github (Track
    # 2, issue #1588/#1589; design doc Section 5, L04/L05 review finding):
    # repointed here for single-source consistency even though this leaf's
    # own members haven't moved yet -- a one-line, no-behaviour change (the
    # annotation stays a string either way under `from __future__ import
    # annotations`).
    from charlie_work.github import MergedPRSearchResult

    from ._base import GitHubRunResult


@runtime_checkable
class PullRequestsLike(Protocol):
    """Structural interface for pull-request read/create operations."""

    def pr_create(self, head: str, base: str, title: str, body: str) -> int | None: ...

    def pr_view(self, number: int, *, fields: str = ...) -> dict[str, Any]: ...

    def pr_list(self) -> list[dict[str, Any]]: ...

    def pr_diff(self, number: int) -> str: ...

    def pr_commits(self, number: int) -> list[dict[str, Any]] | None: ...

    def pr_ready(self, number: int) -> GitHubRunResult: ...

    def merged_pr_list(self) -> list[dict[str, Any]]: ...

    def merged_prs_for_issue(self, issue_number: int, branch_prefix: str) -> MergedPRSearchResult:
        """Return merged PRs binding to ``issue_number``.

        The returned object is list-like and carries an ``ok`` flag. Callers
        must check ``ok`` before treating an empty result as "no merged PRs";
        ``ok=False`` means the search itself failed (rate limit, etc.).
        """
        ...


class PullRequests(CapabilityCollaborator):
    """Pull-request read/create capability collaborator.

    Empty in L01 (pure infrastructure leaf) -- methods move here in later
    leaves per the Mikado graph (design doc Section 5).
    """
