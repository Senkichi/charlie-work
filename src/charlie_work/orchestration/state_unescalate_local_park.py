"""Local-lane park/re-entry delegates for ``OrchestratorApp.unescalate`` (#1970).

Extracted out of ``state_operator_commands.py`` in PR #1987's second rework
round: the #1970 park logic pushed that module past its 1000-line
``file_size_ratchet_baseline`` mark. Per the established facade re-export
pattern (``workflow_delegation``, used across every submodule of this
package), the implementation lives here as plain top-level ``def``s; the
installer discovers this module the same way it discovers its siblings and
re-attaches each function onto ``OrchestratorApp`` as a class attribute, so
``self._local_finished_branch(...)`` and ``self._local_park_reentry_target(...)``
inside ``unescalate`` keep working with no change at their call sites.

The members here are the no-remote backend's half of ``unescalate``'s
no-live-PR path: on a backend that cannot publish pull requests the worker's
branch IS the deliverable, so an escalated issue whose branch still diffs
against the local base is parked for the local review lane
(``agent:review-ready``) rather than dropped back to the never-dispatched
baseline -- and a lane record that already exists for the issue is re-entered
into the lane instead of being stranded adopted-in-name-only.
"""

from __future__ import annotations

from typing import Any

from charlie_work.local_lane import (
    LOCAL_PENDING_STATUS,
    branch_diff,
    branch_head_sha,
    is_local_pr_record,
    local_base_branch,
)
from charlie_work.local_work_park import publishes_pull_requests

# Local-lane record statuses a parked issue can re-enter WITHOUT this
# command touching the record: ``_local_review_packets`` rebuilds a
# ``local_pending`` record's packet, ``_local_dispatch_reviewers`` claims a
# ``reviewing`` one, and ``_local_merge_approved`` consumes an ``approved``
# one. Every other status strands the record under an ``unescalate`` park --
# the packet pass skips ``escalated``/``blocked`` outright, and
# ``rework_requested``'s dispatch lane keys off the *issue* status the park
# just reset to ``open_passive``.
_LOCAL_LANE_REENTRY_STATUSES = frozenset({LOCAL_PENDING_STATUS, "reviewing", "approved"})


def _local_finished_branch(
    self,
    issue_number: int,
    issue_entry: dict[str, Any],
    pr_entry: dict[str, Any] | None = None,
) -> str | None:
    """The issue's worker branch when it still holds parkable work, else None.

    Issue #1970: ``unescalate``'s no-live-PR branch uses this to decide
    whether an escalated issue on a no-remote backend has a deliverable
    worth handing to the local path (``agent:review-ready``) rather than
    dropping back to the never-dispatched baseline.

    Returns None -- keep the drop -- when the backend publishes pull
    requests (there is nothing to park), when no worker branch resolves,
    or when the branch carries no diff against the local base. A
    ``branch_diff`` failure (missing ref, no merge base) returns None just
    like an empty diff: uncertainty is never proof of work, matching
    ``park_salvageable_local_orphan``'s fallback rule. The branch is
    resolved by ``_local_branch_for_issue`` -- the same resolver the
    adoption pass uses -- with ``pr_entry``'s ``branch``/``headRefName`` as
    fallback when a lane record already exists (an escalated record's
    recorded branch outlives the issue entry's ``branch_name``), so a name
    this returns is one the local path can actually adopt on the next
    pass.
    """
    if publishes_pull_requests(self.gh):
        return None
    branch = self._local_branch_for_issue(issue_number, issue_entry)
    if not branch and is_local_pr_record(pr_entry):
        branch = str(pr_entry.get("branch") or pr_entry.get("headRefName") or "")
    if not branch:
        return None
    base_branch = local_base_branch(self.repo_root) or "HEAD"
    if not branch_diff(self.repo_root, base_branch, branch):
        return None
    return branch


def _local_park_reentry_target(
    self,
    pr_number: int,
    pr_entry: dict[str, Any],
    *,
    pr_stuck: bool,
    pr_status_target: str | None,
) -> tuple[str | None, tuple[dict[str, Any], str] | None, str | None]:
    """Re-entry decision for an existing local lane record under an ``unescalate`` park.

    Issue #1970 rework: the park re-parks the issue for the local lane, but
    when a lane record already exists for the issue (``pr_number`` resolved
    through ``prs[N].issue_number`` -- e.g. the reviewer-attempt cap
    escalated it with a current packet) the plain ``open_passive`` reset
    would leave it adopted-in-name-only: the adoption pass skips issues
    that already have a record, the packet pass sees a current packet and
    does not rebuild, and ``_local_dispatch_reviewers`` /
    ``_local_merge_approved`` / ``_local_dispatch_rework`` only select
    ``reviewing`` / ``approved`` / issue-keyed ``rework_requested``
    records. Nothing would ever pick the record up again -- a silent
    strand where the old drop at least re-dispatched a worker. A parked
    local record must instead re-enter the lane.

    Returns ``(pr_status_target, still_valid_verdict, record_head)``:

    - ``pr_status_target`` -- ``"approved"`` when a still-valid approval is
      on file (the merge gate resumes it), ``local_pending`` when the
      record must re-enter (a still-valid request_changes/blocked verdict
      is on file, the record is stuck, or its status is not already
      lane-reachable) so the next packet pass rebuilds and re-dispatches a
      reviewer, else the incoming target unchanged (the record is already
      lane-reachable).
    - ``still_valid_verdict`` -- ``_still_valid_recorded_verdict``'s
      ``(decision, reason)`` pair evaluated against the resolved record
      head; the caller voids it when the target is ``local_pending``,
      exactly as the remote path voids one, or dispatch would skip the
      rebuilt record on the terminal decision.
    - ``record_head`` -- the resolved branch head. A local record has no
      PR object, so the caller pins a voided-verdict ``pending`` stub to
      this head (the same head the packet pass rebuilds against).
    """
    local_branch = str(pr_entry.get("branch") or pr_entry.get("headRefName") or "")
    record_head = branch_head_sha(self.repo_root, local_branch) if local_branch else None
    still_valid = self._still_valid_recorded_verdict(pr_number, record_head)
    if still_valid is not None and still_valid[0].get("decision") == "approved":
        return "approved", still_valid, record_head
    if (
        still_valid is not None
        or pr_stuck
        or pr_entry.get("status") not in _LOCAL_LANE_REENTRY_STATUSES
    ):
        return LOCAL_PENDING_STATUS, still_valid, record_head
    return pr_status_target, still_valid, record_head
