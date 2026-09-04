"""Repo-metadata capability: identity/compare/commit lookups (Track 2, #1585).

Cluster D of the design doc's capability segmentation (Section 3.1):
``name_with_owner``, ``compare``, ``compare_diff``, ``commit``,
``invalidate_list_cache``.

``invalidate_list_cache`` is an ambiguity call (Section 3.1): it is cache
lifecycle, pinned here because RepoMeta is the smallest write-side cluster.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ._base import CapabilityCollaborator

if TYPE_CHECKING:
    # Runtime import would cycle (github.py imports this module to build the
    # GitHubLike union); inspect.signature does not evaluate annotations
    # (eval_str=False by default), so the bare string name is all the
    # conformance test's signature comparison needs.
    from charlie_work.github import GitHubRunResult


@runtime_checkable
class RepoMetaLike(Protocol):
    """Structural interface for repository-metadata operations."""

    def name_with_owner(self) -> str: ...

    def compare(self, base: str, head: str) -> dict[str, Any] | None: ...

    def compare_diff(self, base: str, head: str) -> str | None: ...

    def commit(self, sha: str) -> GitHubRunResult: ...

    def invalidate_list_cache(self) -> None: ...


class RepoMeta(CapabilityCollaborator):
    """Repository-metadata capability collaborator.

    Empty in L01 (pure infrastructure leaf) -- methods move here in later
    leaves per the Mikado graph (design doc Section 5).
    """
