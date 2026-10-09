"""Merge-queue lane helpers for the merge path (issue #2743).

``decide.py``'s five stage functions stay stage-shaped; the merge-queue
specifics they call -- park detection, stamp freshness, and the per-head
requeue ledger and cap -- live here so the stage file stays under the repo's
per-module size cap. Everything in this module is pure: facts in, verdicts
out, no clock, no ``gh``, no state file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .model import (
    AccountingFacts,
    Admission,
    HoldFacts,
    PersistedPr,
    Readiness,
)
from .rules import CHECK_ROUTE_EXCLUDED_STATUSES


def is_already_in_mergequeue(persisted: PersistedPr | None, admission: Admission) -> bool:
    """Parked in the merge queue and the label is still on (a revert voids the park)."""
    status = persisted.status if persisted is not None else None
    return status == "mergequeue" and not admission.mergequeue_label_reverted


def merge_entry(readiness: Readiness, holds: HoldFacts) -> tuple[bool, bool]:
    """``(escalated_merge_hold, enter)``: may a merge or hand-off be attempted at all.

    The escalated hold leaves ``gate.can_merge`` untouched (#840 / #777b); it
    only blocks the actions. Shared by ``decide_merge``, the merge-hold read
    guard and the preview so the three cannot drift.
    """
    can_merge = readiness.gate.can_merge
    escalated = can_merge and (holds.pr_escalated or holds.issue_escalated)
    enter = (
        can_merge
        and holds.should_merge
        and not escalated
        and not readiness.human_merge_hold
        and not readiness.human_merge_check_unavailable
    )
    return escalated, enter


def merge_hold_read_needed(
    readiness: Readiness, holds: HoldFacts, *, require_label: bool = True
) -> bool:
    """``HoldFacts.merge_hold`` is consumed: a hand-off (or, in preview, a merge) is possible.

    ``require_label=False`` is the dry-run preview's (legacy) behaviour of
    reading the hold whenever a merge could be attempted, label or not.
    """
    _, enter = merge_entry(readiness, holds)
    return bool(enter and (holds.config.mergequeue_label or not require_label))


def mergequeue_stamp_needs_now(
    locked: PersistedPr, merged: bool, live_head_sha: str | None
) -> bool:
    """``AccountingFacts.now_iso`` is consumed: a fresh mergequeue stamp will be written."""
    status = "merged" if merged else locked.status
    if status != "mergequeue" or not live_head_sha:
        return False
    return not (locked.mergequeue_head_sha == live_head_sha and bool(locked.mergequeue_since))


def is_counted_queue_revert(
    admission: Admission, *, can_merge: bool, live_head_sha: str | None
) -> bool:
    """Issue #2743: this pass observed a revert that climbs the requeue counter.

    A self-revocation never counts -- neither the ``self_revoked_stale_head``
    flag nor ANY recorded ``mergequeue_revoked_reason`` (reconcile stripped
    the label on purpose; ``"not_approved"`` is also deliberate, just normally
    masked by ``can_merge``). An unknown live head cannot anchor the count.
    """
    return (
        admission.mergequeue_label_reverted
        and can_merge
        and not admission.self_revoked_stale_head
        and admission.mergequeue_revoked_reason is None
        and bool(live_head_sha)
    )


def mergequeue_requeue_count(
    persisted: PersistedPr | None, admission: Admission, can_merge: bool, live_head_sha: str | None
) -> int:
    """Issue #2743: the same-head queue-revert count this pass projects.

    The persisted counter anchors to ``mergequeue_requeues_head_sha``: a
    revert observed under a different live head restarts from zero (a new
    head is a fresh queue budget). A revert only *counts* when the PR could
    otherwise merge (a same-pass check failure owns the revert, per the
    #823 lane), the live head is known (an unknown head cannot anchor the
    count), and it was not a self-revocation -- ANY recorded
    ``mergequeue_revoked_reason`` means reconcile stripped the label on
    purpose, a broader exclusion than ``self_revoked_stale_head``'s
    two-reason set (``"not_approved"`` is also deliberate, just normally
    masked by ``can_merge``).

    A malformed persisted counter fails toward zero rather than raising:
    this read happens on every mergequeue-configured pass, not only on a
    route like the failed-attempt debounce's (#2206 shape).
    """
    base = 0
    if (
        persisted is not None
        and live_head_sha
        and persisted.mergequeue_requeues_head_sha == live_head_sha
    ):
        try:
            base = int(persisted.mergequeue_requeues)
        except (TypeError, ValueError):
            base = 0
    counted_revert = is_counted_queue_revert(
        admission, can_merge=can_merge, live_head_sha=live_head_sha
    )
    return base + (1 if counted_revert else 0)


@dataclass(frozen=True)
class RequeueGate:
    """The projected same-head revert count and the cap verdict for ``decide_merge``."""

    requeues: int
    capped: bool
    rework: bool


def requeue_gate(readiness: Readiness, holds: HoldFacts) -> RequeueGate:
    """Issue #2743: does the projected revert count cap this pass's hand-off.

    A cap of 0 disables the gate; the count itself is still projected (and
    persisted by accounting) so enabling it later sees the accumulated
    history. The rework route needs a linked issue in a routable status and
    defers to a human-merge hand-off when one is pending.
    """
    cfg = holds.config
    adm = readiness.branch.admission
    can_merge = readiness.gate.can_merge
    requeues = mergequeue_requeue_count(holds.persisted, adm, can_merge, holds.live_head_sha)
    cap = cfg.mergequeue_requeue_cap
    capped = bool(
        cfg.mergequeue_label and can_merge and holds.live_head_sha and cap > 0 and requeues >= cap
    )
    rework = (
        capped
        and readiness.issue_number is not None
        and not readiness.human_merge_hold
        and not readiness.human_merge_check_unavailable
        and holds.issue_status not in CHECK_ROUTE_EXCLUDED_STATUSES
    )
    return RequeueGate(requeues=requeues, capped=capped, rework=rework)


@dataclass(frozen=True)
class RequeueDelta:
    """Accounting output for the requeue ledger: the new persisted count, its
    anchor head, and the ``mergequeue_requeue_capped`` payload when this pass
    crossed the cap (``None``s when the pass observed no counted revert)."""

    requeues: int | None
    head_sha: str | None
    capped_payload: Mapping[str, Any] | None


def requeue_accounting(adm: Admission, can_merge: bool, facts: AccountingFacts) -> RequeueDelta:
    """Issue #2743: persist the same-head requeue count when this pass observed
    a counted revert. A head change restarts at 1 -- the new head is a fresh
    queue budget. The cap event's payload fires exactly once, at the crossing
    pass; a count already past cap keeps climbing without re-emitting."""
    if not is_counted_queue_revert(adm, can_merge=can_merge, live_head_sha=facts.live_head_sha):
        return RequeueDelta(requeues=None, head_sha=None, capped_payload=None)
    locked_head = facts.locked.mergequeue_requeues_head_sha
    try:
        base = int(facts.locked.mergequeue_requeues)
    except (TypeError, ValueError):
        base = 0
    requeues = (base if locked_head == facts.live_head_sha else 0) + 1
    cap = facts.config.mergequeue_requeue_cap
    payload: Mapping[str, Any] | None = None
    if cap > 0 and requeues == cap:
        payload = {
            "pr_number": facts.pr_number,
            "issue_number": facts.issue_number,
            "head_sha": facts.live_head_sha,
            "requeues": requeues,
            "cap": cap,
            "mergequeue_label": facts.config.mergequeue_label,
        }
    return RequeueDelta(requeues=requeues, head_sha=facts.live_head_sha, capped_payload=payload)
