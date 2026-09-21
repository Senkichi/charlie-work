"""Unlinked-PR visibility delegate for ``OrchestratorApp``.

New functionality (not a Track 2 relocation): the merge lane's per-PR loop
(``_loop_body`` in ``reap_loop.py``) used to skip every open PR whose linked
issue could not be resolved via a bare ``continue`` fired before any
instrumentation ran -- no event, no operator-facing signal that a PR was
rotting unreviewed. ``_record_unlinked_pr_skip`` is the sole stateful side of
the fix; the pure edge-detector it calls
(``pr_unlinked_visibility.compute_unlinked_pr_transition``) lives outside
``charlie_work.orchestration`` because every top-level ``def`` in this
package auto-installs onto ``OrchestratorApp`` as a ``self``-taking method
(``workflow_delegation.discover_delegate_modules``) -- a pure helper placed
here would be miscalled with ``self`` bound to its first parameter.

``charlie_work.workflow`` is reached through the ``_wf.`` module-object seam
(the same convention every sibling ``orchestration`` module uses) only for
the shared state primitives (``state_lock``, ``load_state``, ``save_state``)
that already live there; this module's own logic is not a moved body and
does not otherwise depend on ``charlie_work.workflow``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import charlie_work.workflow as _wf
from charlie_work.pr_unlinked_visibility import (
    UNLINKED_PR_NOTICE_KEY,
    UNLINKED_PR_SKIPPED_EVENT_KIND,
    compute_unlinked_pr_transition,
    unlinked_pr_author,
    unlinked_pr_fingerprint,
)


def _record_unlinked_pr_skip(self, pr: dict[str, Any], pr_number: int, *, now: datetime) -> None:
    """Edge-triggered notice for a PR the merge lane cannot resolve an issue for.

    Called from the per-PR loop in ``reap_loop._loop_body`` at the exact site
    that used to ``continue`` with zero instrumentation. Emits
    ``pr_unlinked_skipped`` once when a PR is first seen, and again only when
    its persisted fingerprint (``mergeable``/``mergeStateStatus``/head SHA --
    the only PR fields already in hand from this pass's ``pr_list()`` call,
    so this adds no new GitHub calls) changes from the last emission. An
    unchanged pass performs a single state.json read and writes nothing at
    all -- no event, no rewrite -- so a standing backlog of unresolved PRs
    never becomes per-pass spam.
    """
    fingerprint = unlinked_pr_fingerprint(pr)
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        pr_state = dict(state["prs"].get(str(pr_number)) or {})
        marker = pr_state.get(UNLINKED_PR_NOTICE_KEY)
        new_marker, event_extra = compute_unlinked_pr_transition(marker, fingerprint, now=now)
        if event_extra is None:
            return
        author, is_bot = unlinked_pr_author(pr)
        pr_state[UNLINKED_PR_NOTICE_KEY] = new_marker
        state["prs"][str(pr_number)] = pr_state
        state = self._record_event(
            state,
            UNLINKED_PR_SKIPPED_EVENT_KIND,
            {
                "pr_number": pr_number,
                "author": author,
                "is_bot": is_bot,
                **fingerprint.as_dict(),
                **event_extra,
            },
        )
        _wf.save_state(self.paths.state_file, state)
