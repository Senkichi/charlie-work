"""Issues capability: issue read/close and dependency surface (Track 2, #1585).

Cluster F of the design doc's capability segmentation (Section 3.1):
``close_issue``, ``issue_view``, ``issue_list``, ``issue_dependencies``,
``are_issues_open``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ._base import CapabilityCollaborator


@runtime_checkable
class IssuesLike(Protocol):
    """Structural interface for issue read/close/dependency operations."""

    def close_issue(self, number: int) -> bool: ...

    def issue_view(self, number: int) -> dict[str, Any]: ...

    def issue_list(self, labels: Any = None, state: Any = None) -> list[dict[str, Any]]: ...

    def issue_dependencies(self, issue_numbers: list[int]) -> dict[int, list[int]]: ...

    def are_issues_open(self, issue_numbers: list[int]) -> set[int]: ...


class Issues(CapabilityCollaborator):
    """Issue read/close/dependency capability collaborator.

    Empty in L01 (pure infrastructure leaf) -- methods move here in later
    leaves per the Mikado graph (design doc Section 5).
    """
