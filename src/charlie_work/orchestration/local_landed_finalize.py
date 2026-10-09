"""Terminal "already landed" edge for the local path (issue #2739).

An empty three-dot ``git diff <base>...<branch>`` is the legitimate answer
"everything the branch carried is on the base" -- the work landed, possibly
by another path, and the gate's own sync merge has advanced the merge base
past it. Before this module existed, both the packet phase and the merge
gate treated that shape as ordinary churn: the packet phase voided the
approval (``approval_survives_head_move`` rejected the empty diff), rebuilt
the packet, dispatched a fresh reviewer, the reviewer re-approved, the
gate's next sync merge moved the head again, and the cycle repeated
forever. A closed tracker issue with stale ``agent:pr-open`` /
``agent:reviewing`` labels converged the same way -- the drift reconciler
stripped the labels and the packet phase re-applied ``review_started``
every pass.

Three delegates live here:

* ``_local_finalize_landed`` -- the terminal edge. Kills a stray claimed
  suite (its result is moot once the record is terminal) and hands the
  record to the gate's existing merge bookkeeping, which reports ``already``
  or performs a content-free merge, lands the issue at done, and closes it.
* ``_local_landed_or_closed_skip`` -- the packet phase's pre-build check:
  finalize the landed shape; decline to rebuild a packet (or apply the
  ``review_started`` edge) for a record whose tracker issue is already
  closed when the branch cannot prove unmerged content.
* ``_local_gate_abort_streak_tick`` -- the bounded backstop. The
  livelock's signature is one "record left approved" gate abort per review
  cycle under an unchanged ``reviewed_patch_id``; the streak counts them so
  the gate can escalate mechanically at ``LOCAL_GATE_ABORT_STREAK_LIMIT``
  instead of looping forever if the shape ever re-forms.

Every top-level ``def`` is installed on ``OrchestratorApp`` by
``workflow_delegation._install_delegates``; lane members are reached
through ``self.`` and ``LOCAL_SUITE_CLAIM_FIELDS`` is imported from
``local_merge_gate.py`` directly (one-directional -- the gate never
imports this module back).
"""

from __future__ import annotations

from typing import Any

from charlie_work.local_lane import branch_diff
from charlie_work.orchestration.local_merge_gate import LOCAL_SUITE_CLAIM_FIELDS


def _local_finalize_landed(
    self,
    *,
    pr_key: str,
    record: dict[str, Any],
    entry: dict[str, Any],
    branch: str,
    base_ref: str,
    live_head: str,
    decision: dict[str, Any],
) -> bool:
    """Terminal edge for a lane record whose branch diff is empty (#2739).

    The record goes straight through the gate's terminal bookkeeping:
    ``merge_branch_into_base`` reports ``already`` or performs a
    content-free merge, the issue lands at the done label and closes, and
    the branch/worktree tear down per ``auto_merge.delete_branch``.

    An in-flight suite claim on the record is killed first: its result is
    moot once the record is terminal, and leaving it would let the next
    gate walk abort it as "record left approved" -- one round of the churn
    this issue is about.

    Returns ``_local_gate_finalize_merge``'s bool: True only when the base
    advance deferred (the gate stays held for a retry).
    """
    pr_number = int(pr_key)
    issue_number = int(record.get("issue_number") or pr_number)
    pid = record.get("local_suite_pid")
    killed: list[int] = []
    if isinstance(pid, int) and pid > 0:
        killed = self.write_gate.kill_process_tree(
            pid, record.get("local_suite_process_start_time")
        )
    # Claim clear + audit event go through one locked write so a crash can
    # never tear them apart (``_local_gate_update``'s contract).
    self._local_gate_update(
        pr_key,
        {field: None for field in LOCAL_SUITE_CLAIM_FIELDS},
        event=(
            "local_review_skipped_landed",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "branch": branch,
                "base_ref": base_ref,
                "head_sha": live_head,
                "killed_pids": killed,
            },
        ),
    )
    return self._local_gate_finalize_merge(
        pr_key=pr_key,
        record={**record, **{field: None for field in LOCAL_SUITE_CLAIM_FIELDS}},
        entry=entry,
        branch=branch,
        base_ref=base_ref,
        live_head=live_head,
        decision=decision,
    )


def _local_landed_or_closed_skip(
    self,
    *,
    pr_key: str,
    record: dict[str, Any],
    issue_number: int,
    issue_live: bool,
    branch: str,
    base_ref: str,
    head: str,
) -> dict[str, Any] | None:
    """Pre-build decision for one lane record in ``_local_review_packets``.

    Returns the ``results["skipped"]`` entry when the record is retired or
    deferred here, or ``None`` when it should proceed to the packet
    rebuild check.

    Issue #2739, two cases:

    * An empty three-dot diff is the "already landed" answer -- retire the
      record through ``_local_finalize_landed`` instead of rebuilding a
      packet (which would void an approval, dispatch a reviewer over an
      empty diff, and let the gate's next sync merge restart the cycle).
      Guarded by ``issue_live``: a worker still writing to the branch may
      be mid-push.
    * A record whose tracker issue is already ``closed`` and whose branch
      cannot prove unmerged content (empty diff under a live writer, or an
      uncomputable diff) gets no packet rebuild and no ``review_started``
      edge -- otherwise the lane re-applies the pr-open/reviewing labels
      the drift reconciler strips, forever. A non-empty diff still
      surfaces (``are_issues_open`` treats the same shape as still-open).
    """
    live_diff = branch_diff(self.repo_root, base_ref, branch)
    if live_diff == "" and not issue_live:
        entry: dict[str, Any] = {"issue": issue_number, "pr": int(pr_key)}
        self._local_finalize_landed(
            pr_key=pr_key,
            record=record,
            entry=entry,
            branch=branch,
            base_ref=base_ref,
            live_head=head,
            decision=self._review_decision(int(pr_key)),
        )
        return {
            "issue": issue_number,
            "reason": "landed",
            "outcome": entry.get("outcome"),
        }
    if not live_diff:
        issue_state = str(self.gh.issue_view(issue_number).get("state") or "")
        if issue_state.upper() == "CLOSED":
            self._local_gate_event(
                "local_review_skipped_issue_closed",
                {
                    "issue_number": issue_number,
                    "pr_number": int(pr_key),
                    "branch": branch,
                    "base_ref": base_ref,
                },
                level="warning",
            )
            return {"issue": issue_number, "reason": "issue_closed"}
    return None


def _local_gate_abort_streak_tick(self, pr_key: str, record: dict[str, Any]) -> int:
    """Bump the consecutive "record left approved" abort streak; return it.

    Issue #2739: the zero-net-diff livelock aborted the claimed suite once
    per review cycle while the approved patch-id never changed. The streak
    counts those aborts so the caller can escalate at
    ``LOCAL_GATE_ABORT_STREAK_LIMIT``. It re-baselines when the recorded
    patch-id changes (genuinely new content re-entered review) and resets
    on the terminal write in ``_local_gate_finalize_merge`` plus the
    unescalate/de-escalation reset maps.
    """
    patch_id = record.get("reviewed_patch_id")
    streak = int(record.get("local_gate_abort_streak") or 0)
    if streak and record.get("local_gate_abort_streak_patch_id") != patch_id:
        streak = 0
    streak += 1
    self._local_gate_update(
        pr_key,
        {
            "local_gate_abort_streak": streak,
            "local_gate_abort_streak_patch_id": patch_id,
        },
    )
    return streak
