"""Re-arm a no-remote rework worker that exited without committing (issue #2094).

On a backend with no pull requests, ``local_work_park.park_unpublishable_work``
is the single edge every dead worker's branch passes through: it applies
``agent:review-ready`` and parks the issue ``open_passive``. That is right for a
worker that committed. It is wrong for a *rework* worker that died without
moving the branch: the head still equals the verdict's ``reviewed_head_sha``,
so ``_local_review_packets`` sees no new head to review, and
``_local_dispatch_rework`` selects only issue status ``rework_requested``.
Nothing selects the parked issue again -- the mdls wedge of 2026-09-29.

:func:`rearm_no_op_local_rework` is called from that park edge before the park
write. When the parked head equals ``reviewed_head_sha`` and the local PR
record's pending disposition is rework (``request_changes`` verdict, or an
approved verdict re-routed by a merge-conflict / check-failure rework, whose
record status is ``rework_requested``), it restores ``rework_requested`` instead.
The repeat is bounded by the same ``no_op_rework_attempts`` counter and
``review.max_no_op_rework_attempts`` cap the remote janitor lane uses; past the
cap the issue is escalated, never left passive.

A death the dead-worker classification resolved to a provider throttle
(``rate_limited``, ``quota_exhausted``, ... -- the kinds
``rework_attempt_exemption.is_provider_throttle_rework_death`` admits) is not a
no-op attempt: the worker never got to try the rework. It is re-armed without
advancing the counter, so it can never escalate; the provider-throttle launch
gate defers the relaunch until the window passes. The mdls #109 wedge of
2026-10-06 escalated an issue on three throttled deaths and zero real attempts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .dead_dispatched_timer import clear_local_park_deferral
from .escalation import _escalate_issue, _escalation_edge
from .github import GitHubLike
from .labels import TransitionOutcome
from .local_lane import branch_head_sha, local_pr_records
from .rework_attempt_exemption import (
    PROVIDER_THROTTLE_EXEMPTION,
    is_provider_throttle_rework_death,
)
from .state import load_state, state_lock
from .worker_fate import persisted_failure
from .write_gate import WriteGate

ATTEMPTS_KEY = "no_op_rework_attempts"
ESCALATION_REASON = f"{ATTEMPTS_KEY}_cap_exceeded"
# Local PR-record statuses meaning "a rework disposition is pending": the
# ``request_changes`` verdict value, or ``rework_requested`` (set by
# ``_route_to_rework`` for an approved verdict sent back for a conflict / CI fix).
_PENDING_REWORK_STATUSES = frozenset({"request_changes", "rework_requested"})


def _pending_rework_record(
    state: dict[str, Any], issue_number: int, head: str
) -> tuple[str, dict[str, Any]] | None:
    """The local PR record whose pending rework this unchanged ``head`` left unserved."""
    for key, record in local_pr_records(state).items():
        if int(record.get("issue_number") or key) != issue_number:
            continue
        if (
            record.get("status") in _PENDING_REWORK_STATUSES
            and record.get("reviewed_head_sha") == head
        ):
            return key, record
    return None


def rearm_no_op_local_rework(
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path,
    branch: str,
    issue_number: int,
    write_gate: WriteGate,
    failure_kind: str | None = None,
) -> tuple[bool, str | None] | None:
    """Re-arm (or escalate) a dead local rework worker that changed nothing.

    ``failure_kind`` is the dead worker's classification as the park edge
    received it. When the caller has none, the locked entry's epoch-scoped
    ``dead_worker_failure_kind`` stamp is read instead (the rework dispatch
    clears it, so it can only describe this death). A provider-throttle kind
    re-arms without counting.

    Returns ``None`` when the park should proceed as usual: the branch head
    moved (the worker committed), no local record carries a pending rework for
    this head, or the head cannot be resolved. Otherwise returns the park
    edge's ``(ok, error)`` contract -- ``ok`` means handled, do not redispatch.
    Errors come back as values; a failed label write returns ``ok=False``.
    """
    head = branch_head_sha(repo_root, branch)
    if head is None:
        return None
    state_file = write_gate.state_path
    max_attempts = config.review.max_no_op_rework_attempts
    with state_lock(state_file):
        state = load_state(state_file)
        match = _pending_rework_record(state, issue_number, head)
        if match is None:
            return None
        pr_key, record = match
        issue_entry = {**((state.get("issues") or {}).get(str(issue_number)) or {})}
        kind = failure_kind or persisted_failure(issue_entry).kind
        counted = not is_provider_throttle_rework_death(kind)
        attempts = int(record.get(ATTEMPTS_KEY) or 0) + int(counted)
        escalate = counted and attempts > max_attempts
        for field in ("orphan_flagged_at", "orphan_drift_fingerprint", "orphan_drift_at"):
            issue_entry.pop(field, None)
        clear_local_park_deferral(issue_entry)
        state.setdefault("issues", {})[str(issue_number)] = issue_entry
        pr_extra = {ATTEMPTS_KEY: attempts, f"{ATTEMPTS_KEY}_last_head": head}
        if escalate:
            state = _escalate_issue(
                state,
                issue_number,
                reason=ESCALATION_REASON,
                reason_class="mechanical",
                pr_number=int(pr_key),
                pr_extra=pr_extra,
            )
            edge = _escalation_edge("escalated", "mechanical")
        else:
            state["issues"][str(issue_number)] = {
                **issue_entry,
                "number": issue_number,
                "status": "rework_requested",
                "merge_alert": "OK",
            }
            state["prs"][pr_key] = {**record, "status": "rework_requested", **pr_extra}
            edge = "rework_requested"
        state = write_gate.append_event(
            state,
            "local_no_op_rework_rearmed",  # event-consumer: audit-only -- the actionable state (issue status rework_requested, or the escalation label) is written in this same lock; this records which no-commit exit was re-armed, whether it counted toward the cap (a provider-throttle death does not), and whether the cap escalated it
            {
                "issue_number": issue_number,
                "branch": branch,
                "head_sha": head,
                "attempts": attempts,
                "max_attempts": max_attempts,
                "escalated": escalate,
                "counted": counted,
                "failure_kind": kind,
                "reason": None if counted else PROVIDER_THROTTLE_EXEMPTION,
            },
        )
        write_gate.save_state(state)

    result = write_gate.transition(gh, config.labels, issue_number, edge)
    if not (write_gate.dry_run or result.outcome is TransitionOutcome.APPLIED):
        return False, f"{edge} label transition failed: {result.outcome.value}"
    return True, None
