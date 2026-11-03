"""Fleet ownership of a PR, resolved for the review/merge-lane act-sites (issue #2763).

``issue_linking.fleet_owns_pr`` is the pure predicate; this module is the thin
shell every act-site shares so none of them re-derives the linked issue's labels
or state entry on its own. It lives outside ``charlie_work.orchestration``
because every top-level ``def`` in that package auto-installs onto
``OrchestratorApp`` as a ``self``-taking method
(``workflow_delegation.discover_delegate_modules``) and is pinned by the member
surface baseline -- a free helper placed there would be miscalled with ``self``
bound to its first parameter.

``NOT_FLEET_OWNED_REASON`` is the ``reason`` the merge lane stamps on the
``pr_unlinked_skipped`` event for an un-owned PR (``reap_loop._loop_body``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .issue_linking import fleet_owns_pr
from .merge_path.issue_labels import open_issue_labels

if TYPE_CHECKING:
    from .config import OrchestratorConfig

#: ``pr_unlinked_skipped`` ``reason`` for a PR that binds to an issue (a closing
#: keyword) but that the fleet does not own. The absent-link case is
#: ``pr_unlinked_visibility.NO_LINKED_ISSUE_REASON``.
NOT_FLEET_OWNED_REASON = "not_fleet_owned"


def pr_fleet_owned(
    gh: Any,
    config: OrchestratorConfig,
    pr: Mapping[str, Any],
    issue_number: int,
    state: Mapping[str, Any],
) -> bool:
    """Is ``pr`` (bound to ``issue_number``) the fleet's to review and merge?

    ``state`` is an already-loaded ``state.json`` snapshot -- this reads no
    state of its own. The linked issue's labels are consulted only when the
    branch namespace and dispatch provenance do not already decide it, and then
    come from the per-pass open-issue list (``open_issue_labels``), so a normal
    pass costs no extra GitHub call. ``fleet_owns_pr`` is monotone in the label
    set (a label can only turn False into True), which is what makes the
    labels-last short-circuit equivalent to always passing them.

    An issue the open list does not show (closed, past the list limit, or an
    unreadable list) contributes no labels: ownership must then be proven by the
    branch or by provenance. That is the fail-closed direction -- an unreadable
    list never ADOPTS a PR.
    """
    entry = (state.get("issues") or {}).get(str(issue_number))
    if fleet_owns_pr(pr, (), entry, config):
        return True
    return fleet_owns_pr(pr, open_issue_labels(gh, issue_number) or (), entry, config)
