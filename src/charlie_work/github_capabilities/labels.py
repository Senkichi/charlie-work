"""Labels capability: issue/PR label mutation (Track 2, issue #1585).

Cluster B of the design doc's capability segmentation (Section 3.1):
``add_issue_label``, ``remove_issue_label``, ``add_pr_label``,
``remove_pr_label``, ``label_list``, ``label_create``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

from ..github_transport.outcome import Response
from ..github_transport.request import RestRequest
from ._base import CapabilityCollaborator
from ._outcome import is_success
from ._send import send

# Moved from ``github.py`` alongside ``label_list`` (Track 2, issue #1587;
# design doc Section 5, L03). ``label_list``'s body is byte-identical to its
# former ``GitHub`` copy and still references this constant as a bare global
# name, so it must be bound in *this* module's globals (a moved function's
# free variables resolve via the module it is defined in, not the module it
# is called from -- ``self.<attr>`` forwarding through
# ``CapabilityCollaborator.__getattr__`` only covers attribute access, not
# bare-name globals). Re-exported through ``github_capabilities/__init__.py``
# and re-imported into ``github.py`` (nothing there uses it directly anymore
# now that ``validate_field_lists`` moved to ``transport.py`` in L09) and
# directly into ``transport.py`` (Track 2, issue #1593; design doc Section 5,
# L09), which imports it from here rather than re-deriving a second copy --
# the same re-export pattern already used there for ``GitHubError`` and
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

    def _add_label(self, number: int, label: str) -> bool:
        # A PR is an issue: one endpoint labels both. Never raises; dry-run is
        # a typed success from the transport.
        request = RestRequest.of(
            "POST", f"repos/{{owner}}/{{repo}}/issues/{number}/labels", body={"labels": [label]}
        )
        return is_success(send(self, request))

    def _remove_label(self, number: int, label: str) -> bool:
        """Remove ``label``; an absent label counts as removed (idempotent).

        ``gh ... --remove-label`` of a label the object does not carry exits 0;
        the REST DELETE answers 404, which is therefore success here.
        """
        request = RestRequest.of(
            "DELETE",
            f"repos/{{owner}}/{{repo}}/issues/{number}/labels/{quote(label, safe='')}",
        )
        outcome = send(self, request)
        return is_success(outcome) or (isinstance(outcome, Response) and outcome.status == 404)

    def add_issue_label(self, number: int, label: str) -> bool:
        return self._add_label(number, label)

    def remove_issue_label(self, number: int, label: str) -> bool:
        return self._remove_label(number, label)

    def add_pr_label(self, number: int, label: str) -> bool:
        """Add a label to a PR (PR-scoped, not the linked issue).

        Used for the Aviator MergeQueue handoff (task #10): the trigger label
        must land on the PR itself, and issue_number may be None for
        cross-repository PRs. Idempotent (gh's addLabels is a no-op if the
        label is already present) and never raises.
        """
        return self._add_label(number, label)

    def remove_pr_label(self, number: int, label: str) -> bool:
        """Remove a label from a PR (PR-scoped, not the linked issue).

        Mirrors ``add_pr_label``. Used to clear Aviator's ``blocked`` label
        once it has gone stale (reconcile.py's ``aviator_stale_blocked`` drift
        kind). Idempotent and never raises.
        """
        return self._remove_label(number, label)

    def label_list(self) -> list[dict[str, Any]]:
        result = self.run(
            ["label", "list", "--limit", "200", "--json", LABEL_LIST_FIELDS], json_output=True
        )
        return result if isinstance(result, list) else []

    def label_create(self, label: str, color: str, description: str) -> None:
        # Update-or-create: bootstrap must be idempotent, and a plain create
        # errors on a pre-existing label so colour/description drift silently.
        # POST first; a 422 means the label exists, so PATCH it (B11: two calls
        # for an existing label). Errors are values; the result is not used.
        body = {"color": color.lstrip("#"), "description": description}
        created = send(
            self,
            RestRequest.of("POST", "repos/{owner}/{repo}/labels", body={"name": label, **body}),
        )
        if isinstance(created, Response) and created.status == 422:
            send(
                self,
                RestRequest.of(
                    "PATCH", f"repos/{{owner}}/{{repo}}/labels/{quote(label, safe='')}", body=body
                ),
            )
