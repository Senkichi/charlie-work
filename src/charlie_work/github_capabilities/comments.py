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

    Moved from ``GitHub`` verbatim (Track 2, issue #1586; design doc Section
    5, L02). Bodies still say ``self.run(...)``, which resolves through
    ``CapabilityCollaborator.__getattr__`` to the owner's ``run`` (design doc
    Section 3.3).
    """

    def issue_comment(self, number: int, body_file: Path) -> None:
        self.run(["issue", "comment", str(number), "--body-file", str(body_file)])

    def pr_comment(self, number: int, body_file: Path) -> None:
        self.run(["pr", "comment", str(number), "--body-file", str(body_file)])
