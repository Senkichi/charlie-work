"""Labels capability: issue/PR label mutation (Track 2, issue #1585).

Cluster B of the design doc's capability segmentation (Section 3.1):
``add_issue_label``, ``remove_issue_label``, ``add_pr_label``,
``remove_pr_label``, ``label_list``, ``label_create``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ._base import CapabilityCollaborator

# Moved from ``github.py`` alongside ``label_list`` (Track 2, issue #1587;
# design doc Section 5, L03). ``label_list``'s body is byte-identical to its
# former ``GitHub`` copy and still references this constant as a bare global
# name, so it must be bound in *this* module's globals (a moved function's
# free variables resolve via the module it is defined in, not the module it
# is called from -- ``self.<attr>`` forwarding through
# ``CapabilityCollaborator.__getattr__`` only covers attribute access, not
# bare-name globals). Re-exported through ``github_capabilities/__init__.py``
# and re-imported into ``github.py`` (which still uses it directly in
# ``validate_field_lists``, a Transport internal not yet moved) --  the same
# re-export pattern already used there for ``GitHubError`` and
# ``_ROUTES``/``_SIGNATURE_SOURCE``/``_make_delegate``.
LABEL_LIST_FIELDS = "name"


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

    Moved from ``GitHub`` verbatim (Track 2, issue #1587; design doc Section
    5, L03). Bodies still say ``self._run_bool(...)``/``self.run(...)``,
    which resolve through ``CapabilityCollaborator.__getattr__`` to the
    owner's ``_run_bool``/``run`` (design doc Section 3.3).
    """

    def add_issue_label(self, number: int, label: str) -> bool:
        return self._run_bool(["issue", "edit", str(number), "--add-label", label])

    def remove_issue_label(self, number: int, label: str) -> bool:
        return self._run_bool(["issue", "edit", str(number), "--remove-label", label])

    def add_pr_label(self, number: int, label: str) -> bool:
        """Add a label to a PR (PR-scoped, not the linked issue).

        Used for the Aviator MergeQueue handoff (task #10): the trigger label
        must land on the PR itself, and issue_number may be None for
        cross-repository PRs. Idempotent (gh's addLabels is a no-op if the
        label is already present) and never raises — see ``_run_bool``.
        """
        return self._run_bool(["pr", "edit", str(number), "--add-label", label])

    def remove_pr_label(self, number: int, label: str) -> bool:
        """Remove a label from a PR (PR-scoped, not the linked issue).

        Mirrors ``add_pr_label``. Used to clear Aviator's ``blocked`` label
        once it has gone stale (reconcile.py's ``aviator_stale_blocked`` drift
        kind). Idempotent and never raises — see ``_run_bool``.
        """
        return self._run_bool(["pr", "edit", str(number), "--remove-label", label])

    def label_list(self) -> list[dict[str, Any]]:
        result = self.run(
            ["label", "list", "--limit", "200", "--json", LABEL_LIST_FIELDS], json_output=True
        )
        return result if isinstance(result, list) else []

    def label_create(self, label: str, color: str, description: str) -> None:
        # --force makes this update-or-create: bootstrap must be idempotent, and
        # without it `gh label create` errors on a pre-existing label and the
        # colour/description drift silently. `--force` is a mutation but stays
        # read-only-safe under dry-run via `_is_mutating`.
        self.run(
            ["label", "create", label, "--force", "--color", color, "--description", description],
            allow_failure=True,
        )
