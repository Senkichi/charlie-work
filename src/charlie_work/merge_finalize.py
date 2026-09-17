"""Issue-side ``state.json`` finalization shared by every door that records a merged PR.

Issue #1493. Every path that persists ``prs[n].status = "merged"`` must also
advance the linked issue's record to the same terminal disposition -- the
issue/PR pair is one lifecycle, and writing only the PR half leaves the issue
frozen at whatever non-terminal status it held when review approved it
(``"approved"`` in the observed corpus), plus a stale cached-label snapshot.
The fleet's own merge (``workflow.py``'s post-``merge_pr`` block) did exactly
that: it set ``merge_alert`` but never ``status``, so reconcile's repair
sweeps -- which deliberately exclude ``"approved"`` because merge
finalization owns it -- left the record stale forever while the per-pass
session liveness checks kept re-evaluating the issue as mid-flight.

This is the mirror image of issue #1482 (de-escalation cleared the issue but
left ``prs[n].status == "escalated"``), and the fix follows the same shape
that issue established in ``tests/test_fix_unescalate.py``: the field set is
derived once here and consumed by every door, so the doors cannot diverge
from each other again. The consumers are the internal merge flow
(``workflow.py``, both the post-``merge_pr`` write and the idempotent
already-merged short-circuit), the externally-merged finalization sweep
(``orchestration/state_merge_train.py``), and the dispatch-side
merged-PR-references close-out (``orchestration/dispatch_state.py``).

``workflow.py`` re-exports ``_merged_issue_fields`` via a facade import block
so ``charlie_work.orchestration`` delegates keep reaching it through their
established ``_wf.`` import surface.
"""

from __future__ import annotations

from typing import Any


def _merged_issue_fields(issue_entry: dict[str, Any], issue_number: int) -> dict[str, Any]:
    """Return the issue record's terminal field set for a merged-PR finalization.

    Every door that marks a PR ``"merged"`` applies this same set to the
    linked issue so no door can drift into its own partial version:

    - ``"status": "closed"`` -- the terminal value the external-merge path
      already used; reconcile's ``ACTIVE_STATE_STATUSES`` sweeps treat any
      other value as repairable drift, but ``"approved"``/``"blocked"`` are
      deliberately excluded from that repair because this transition owns
      them -- so anything short of ``"closed"`` here is a permanent dead
      zone (the #1493 corpus: 120 issues pinned at ``"approved"`` while
      their PRs correctly read ``"merged"``).
    - ``"merge_alert": "OK"`` -- clears the merge-failure health latch the
      attention digest diffs against; a merged issue has no outstanding
      merge alert.
    - ``"labels"`` popped -- the intake-time cache of live GitHub labels is
      stale the moment the merge label edge runs (it adds ``agent:done``
      and strips every other workflow label). The edge runs *after* this
      state write and can partially fail, so asserting a predicted
      post-edge list could lie; clearing the snapshot records "no cache"
      instead of "wrong cache", matching the ``labels: None`` shape most
      historical entries already carry.

    Fields the caller did not put in the set are preserved verbatim from
    ``issue_entry`` -- this is a finalization overlay, not a record reset:
    worker bookkeeping (``worker_pid``, ``dispatched_at``) is cleared by
    the earlier lifecycle transitions, and escalation fields are left for
    the de-escalation doors that own them.
    """
    merged = {
        **issue_entry,
        "number": issue_number,
        "status": "closed",
        "merge_alert": "OK",
    }
    merged.pop("labels", None)
    return merged
