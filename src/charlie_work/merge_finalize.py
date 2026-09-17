"""Issue-side ``state.json`` finalization shared by every lifecycle door that
records a merged PR.

Issue #1493. Every lifecycle path that persists ``prs[n].status = "merged"``
must also advance the linked issue's record to the same terminal disposition
-- the issue/PR pair is one lifecycle, and writing only the PR half leaves
the issue frozen at whatever non-terminal status it held when review
approved it (``"approved"`` in the observed corpus), plus a stale
cached-label snapshot. The fleet's own merge (``workflow.py``'s
post-``merge_pr`` block) did exactly that: it set ``merge_alert`` but never
``status``, so reconcile's repair sweeps -- which deliberately exclude
``"approved"`` because merge finalization owns it -- left the record stale
forever while the per-pass session liveness checks kept re-evaluating the
issue as mid-flight.

This is the mirror image of issue #1482 (de-escalation cleared the issue but
left ``prs[n].status == "escalated"``), and the fix follows the same shape
that issue established in ``tests/test_fix_unescalate.py``: the field set is
derived once here and consumed by every lifecycle door, so the doors cannot
diverge from each other again. The consumers are the internal merge flow
(``workflow.py``, both the post-``merge_pr`` write and the idempotent
already-merged short-circuit -- the later ``prs_entry`` bookkeeping write in
the same ``merge_ready`` call only re-asserts the merged fact the
post-``merge_pr`` block already finalized, and is not a distinct door), the
externally-merged finalization sweep
(``orchestration/state_merge_train.py``), the dispatch-side
merged-PR-references close-out (``orchestration/dispatch_state.py``), and
reconcile's ``merged_outside_orchestrator`` drift-fix (``reconcile.py``
``apply_fixes`` -- an external merge discovered on a reconcile pass lands in
the same dead zone as #1493's internal one when the issue half is skipped).

Two other sites write ``prs[n].status = "merged"`` but are deliberately NOT
consumers:

- ``stalled_review_reap.py``'s terminal review-dispatch reap marks a reaped
  PR merged/closed on GitHub's word. Its persist goes through
  ``_merge_on_write_save``, which merges only ``prs``/``reviewer_quota``/
  ``events`` (issue #594) -- computed ``issues`` writes are silently dropped,
  so that sweep does not own the issues map. Issue-side convergence for a PR
  it marks merged is reconcile's ``merged_outside_orchestrator`` job on the
  next pass (``detect_drift`` still fires on ``issue_still_active`` even when
  the state PR status already reads ``"merged"``). Residual: an issue linked
  to the reaped PR that carries no active labels keeps its stale status --
  a narrower exposure than #1493's, accepted rather than extending the
  merge-on-write protocol into ``issues`` here.
- ``orchestration/state_operator_commands.py``'s ``unescalate`` resets an
  escalated PR entry to ``"merged"`` when GitHub says the PR merged. Its
  issue-side contract is an operator reset, not a lifecycle finalization: a
  still-escalated issue with no other open PR is dropped to the
  never-dispatched baseline via ``unescalated_requeued`` -- not pinned to
  ``"closed"`` -- and anything else is left alone, to be converged by the
  same reconcile door with the same residual.

``workflow.py`` re-exports ``_merged_issue_fields`` via a facade import block
so ``charlie_work.orchestration`` delegates keep reaching it through their
established ``_wf.`` import surface.
"""

from __future__ import annotations

from typing import Any


def _merged_issue_fields(issue_entry: dict[str, Any], issue_number: int) -> dict[str, Any]:
    """Return the issue record's terminal field set for a merged-PR finalization.

    Every lifecycle door that marks a PR ``"merged"`` applies this same set
    to the linked issue so no door can drift into its own partial version
    (the deliberately scoped-out non-consumers are listed in the module
    docstring):

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
