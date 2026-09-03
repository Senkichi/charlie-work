"""Comments capability: issue/PR comment posting (Track 2, issue #1585).

Cluster A of the design doc's capability segmentation (Section 3.1):
``issue_comment``, ``pr_comment``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ._base import CapabilityCollaborator


@runtime_checkable
class CommentsLike(Protocol):
    """Structural interface for issue/PR comment operations."""

    def issue_comment(self, number: int, body_file: Path) -> None: ...

    def pr_comment(self, number: int, body_file: Path) -> None: ...


class Comments(CapabilityCollaborator):
    """Issue/PR comment capability collaborator.

    Empty in L01 (pure infrastructure leaf) -- methods move here in later
    leaves per the Mikado graph (design doc Section 5).
    """
