"""Unlinked-PR visibility delegate for ``OrchestratorApp``.

New functionality (not a Track 2 relocation): the merge lane's per-PR loop
(``_loop_body`` in ``reap_loop.py``) used to skip every open PR whose linked
issue could not be resolved via a bare ``continue`` fired before any
instrumentation ran -- no event, no operator-facing signal that a PR was
rotting unreviewed. ``_record_unlinked_pr_skips`` is the sole stateful side
of the fix; the pure edge-detector it calls
(``pr_unlinked_visibility.compute_unlinked_pr_transition``) lives outside
``charlie_work.orchestration`` because every top-level ``def`` in this
package auto-installs onto ``OrchestratorApp`` as a ``self``-taking method
(``workflow_delegation.discover_delegate_modules``) -- a pure helper placed
here would be miscalled with ``self`` bound to its first parameter.

``charlie_work.workflow`` is reached through the ``_wf.`` module-object seam
(the same convention every sibling ``orchestration`` module uses) only for
the shared state primitives (``state_lock``, ``load_state``) that already
live there; this module's own logic is not a moved body and does not
otherwise depend on ``charlie_work.workflow``.

Review of the original per-PR ``_record_unlinked_pr_skip`` (#1766 PR review)
found two defects a single-PR call site could not avoid:

- It ran outside ``_loop_body``'s per-PR isolation ``try`` -- the first
  state-lock acquisition of the pass, for a PR the lane does no other work
  on. A busy lock there raised ``StateLockBusy`` straight out of ``loop()``
  (``@_guard_state_lock`` then reports the WHOLE pass as a success-shaped
  skip), aborting review/merge of every other PR in the same batch.
- ``self.write_gate.save_state`` gates ``_record_event`` under dry-run but a
  plain ``_wf.save_state`` call right after it did not, so one ``--dry-run``
  pass durably wrote the edge-detector marker for an event that was never
  emitted, permanently suppressing the real one.

``_record_unlinked_pr_skips`` (plural) fixes both by collecting every
issue-less PR during the per-PR loop instead of writing per PR, and
recording the whole batch in ONE locked read-modify-write called after that
loop completes -- so a lock timeout here can never prevent the real
review/merge work the pass already did, and a single try/except around the
whole batch (mirroring ``_announce_unauthorized_merges``'s identical
best-effort shape) makes even that one acquisition non-fatal to the pass.
This also collapses what used to be up to N lock acquisitions and N
``state.json`` reads per pass (one per standing issue-less PR) into one of
each, regardless of backlog size.
"""

from __future__ import annotations

import logging
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
from charlie_work.state import StateLockBusy


def _record_unlinked_pr_skips(
    self, entries: list[tuple[dict[str, Any], int]], *, now: datetime
) -> None:
    """Batch edge-triggered notice for every PR the merge lane skipped this pass.

    ``entries`` is collected by the per-PR loop in ``reap_loop._loop_body``
    at the exact site that used to ``continue`` with zero instrumentation --
    one ``(pr, pr_number)`` pair per PR whose linked issue could not be
    resolved -- and this is called once, after that loop finishes. For each
    entry, emits ``pr_unlinked_skipped`` when the PR is first seen, and again
    only when its persisted fingerprint (``mergeable``/``mergeStateStatus``/
    head SHA -- already in hand from this pass's ``pr_list()`` call, so this
    adds no new GitHub calls) changes from the last emission. A pass in which
    every entry is unchanged performs one state.json read and writes nothing
    at all -- no event, no rewrite -- so a standing backlog of unresolved PRs
    never becomes per-pass spam.

    Failure here is by-value: an informational notice must never outrank the
    review/merge work this pass already completed, so lock contention or a
    write error is logged and swallowed rather than raised (matching the
    module docstring's rationale and the identical shape in
    ``_announce_unauthorized_merges``).
    """
    if not entries:
        return
    try:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            emitted = False
            for pr, pr_number in entries:
                fingerprint = unlinked_pr_fingerprint(pr)
                pr_state = dict(state["prs"].get(str(pr_number)) or {})
                marker = pr_state.get(UNLINKED_PR_NOTICE_KEY)
                new_marker, event_extra = compute_unlinked_pr_transition(
                    marker, fingerprint, now=now
                )
                if event_extra is None:
                    continue
                author, is_bot = unlinked_pr_author(pr)
                pr_state[UNLINKED_PR_NOTICE_KEY] = new_marker
                state["prs"][str(pr_number)] = pr_state
                state = self._record_event(
                    state,
                    UNLINKED_PR_SKIPPED_EVENT_KIND,  # event-consumer: audit-only -- imported constant (defined in pr_unlinked_visibility.py) rather than a same-file literal, so the AST scanner cannot resolve it; registered "warning" in _LEVEL_BY_KIND and read via query_events(kind="pr_unlinked_skipped") in tests/test_issue_1766_unlinked_pr_visibility.py.
                    {
                        "pr_number": pr_number,
                        "author": author,
                        "is_bot": is_bot,
                        **fingerprint.as_dict(),
                        **event_extra,
                    },
                    level="warning",
                )
                emitted = True
            if emitted:
                # Gated (unlike the removed direct `_wf.save_state` call this
                # replaces): under dry-run this returns `state` unchanged and
                # writes nothing, so a preview pass can never durably persist
                # the marker for an event `_record_event` above already
                # suppressed.
                self.write_gate.save_state(state)
    except (OSError, ValueError, StateLockBusy) as exc:
        logging.getLogger(__name__).warning(
            "could not record pr_unlinked_skipped for %d PR(s): %s", len(entries), exc
        )
