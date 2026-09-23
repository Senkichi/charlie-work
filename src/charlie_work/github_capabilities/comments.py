"""Comments capability: issue/PR comment posting (Track 2, issue #1585).

Cluster A of the design doc's capability segmentation (Section 3.1):
``issue_comment``, ``pr_comment``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

# ``ci_fleet.github.GitHubError`` is imported directly here -- the same
# external, identity-sensitive source ``github.py`` re-exports from -- so the
# refusal below lands in the exception type every caller already catches.
from ci_fleet.github import GitHubError

from ..outbound_body_guard import (
    OutboundBodyGuardError,
    check_outbound_write,
    refusal_summary,
)
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

    def _guard_comment_body(self, surface: str, number: int, body_file: Path) -> None:
        """Refuse the comment before ``gh`` runs if its body carries a secret.

        Issue #1505: a comment body is permanent -- deleting it does not
        remove the text from GitHub's edit history, so the credential scan
        runs here, before the API write, on the file gh is about to submit.
        Fail-closed on both counts: a credential match raises ``GitHubError``
        (the same failure vocabulary a ``gh`` refusal produces), and an
        unreadable body file refuses rather than letting gh discover it.
        Skipped under ``dry_run`` -- nothing is written, so there is nothing
        to guard.
        """
        if self.dry_run:
            return
        try:
            body = body_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise GitHubError(f"{surface} #{number}: cannot read body file: {exc}") from exc
        try:
            matches = check_outbound_write(
                surface=surface,
                parts=(("body", body),),
                repo_root=self.repo_root,
                state_dir=getattr(self.runtime, "state_dir", None),
                issue_number=number if surface == "issue_comment" else None,
                pr_number=number if surface == "pr_comment" else None,
            )
        except OutboundBodyGuardError as exc:
            raise GitHubError(f"{surface} #{number}: {exc}") from exc
        if matches:
            raise GitHubError(refusal_summary(surface, matches))

    def issue_comment(self, number: int, body_file: Path) -> None:
        self._guard_comment_body("issue_comment", number, body_file)
        self.run(["issue", "comment", str(number), "--body-file", str(body_file)])

    def pr_comment(self, number: int, body_file: Path) -> None:
        self._guard_comment_body("pr_comment", number, body_file)
        self.run(["pr", "comment", str(number), "--body-file", str(body_file)])
