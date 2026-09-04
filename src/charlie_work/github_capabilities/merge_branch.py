"""Merge/branch capability: PR merge and branch lifecycle (Track 2, #1585).

Cluster G of the design doc's capability segmentation (Section 3.1):
``merge_pr``, ``delete_branch``, ``pr_update_branch``, ``pr_close``,
``pr_reopen``, ``push_empty_commit``, ``branch_protection``.
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
    from ._base import GitHubRunResult


@runtime_checkable
class MergeBranchLike(Protocol):
    """Structural interface for PR-merge/branch-lifecycle operations."""

    def merge_pr(
        self, number: int, strategy: str, admin: bool = False, merge_flags: tuple[str, ...] = ()
    ) -> str: ...

    def delete_branch(self, branch: str) -> bool: ...

    def pr_update_branch(self, pr_number: int) -> bool: ...

    def pr_close(self, number: int) -> GitHubRunResult: ...

    def pr_reopen(self, number: int) -> GitHubRunResult: ...

    def push_empty_commit(self, branch: str) -> GitHubRunResult: ...

    def branch_protection(self, base: str) -> dict[str, Any] | None: ...


class MergeBranch(CapabilityCollaborator):
    """PR-merge/branch-lifecycle capability collaborator.

    Empty in L01 (pure infrastructure leaf) -- methods move here in later
    leaves per the Mikado graph (design doc Section 5).
    """
