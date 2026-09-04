"""Labels capability: issue/PR label mutation (Track 2, issue #1585).

Cluster B of the design doc's capability segmentation (Section 3.1):
``add_issue_label``, ``remove_issue_label``, ``add_pr_label``,
``remove_pr_label``, ``label_list``, ``label_create``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ._base import CapabilityCollaborator


@runtime_checkable
class LabelsLike(Protocol):
    """Structural interface for issue/PR label operations."""

    def add_issue_label(self, number: int, label: str) -> bool: ...

    def remove_issue_label(self, number: int, label: str) -> bool: ...

    def add_pr_label(self, number: int, label: str) -> bool: ...

    def remove_pr_label(self, number: int, label: str) -> bool: ...

    def label_list(self) -> list[dict[str, Any]]: ...

    def label_create(self, label: str, color: str, description: str) -> None: ...


class Labels(CapabilityCollaborator):
    """Issue/PR label capability collaborator.

    Empty in L01 (pure infrastructure leaf) -- methods move here in later
    leaves per the Mikado graph (design doc Section 5).
    """
