"""The dead-worker sweep's pure decision entry: ``decide(facts, observed) -> SweepPlan``.

The sweep is a decide -> apply -> decide-again loop. Each phase (``pre``,
``lock``, ``post``) is a generator flow that yields *requests* (effects whose
result the decision needs) and *commits* (effects whose content is already
decided). ``decide`` replays the flow from the start against ``observed`` (the
results of every request the shell has run so far); the first request with no
result ends the round, so a plan carries at most one request plus the commits
that precede it. Commits are a growing prefix: the shell applies only the new
suffix. ``decide`` reads no clock, filesystem, GitHub or state -- only its
arguments -- so the decision table is unit-testable through this one function.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .constants import PASSIVE_OPEN_STATUS, ROUTED_OUTCOME_KEY
from .decide_common import (
    Draft,
    Flow,
    LockAcc,
    drift_fingerprint,
    emit,
    events_to_emits,
    flush,
    run_flow,
)
from .decide_no_pr import lock_no_pr_flow
from .decide_post import post_flow
from .decide_pre import pre_flow
from .decide_reap import reap_flow
from .decide_with_pr import with_pr_flow
from .model import LockOutcome, OpenPrForBranch, PreOutcome, SweepFacts, SweepPlan, UpdateIssue


class PhaseOrderError(ValueError):
    """A later phase was decided before its predecessor's flow completed."""


def _live_handoff_flow(facts: SweepFacts, pre: PreOutcome) -> Flow:
    """#1867: stamp routed outcomes and open the PR for declared live handoffs."""
    locked_issues = (facts.locked or {}).get("issues") or {}
    for number, candidate in pre.pr_already_open.items():
        entry = locked_issues.get(str(number))
        if isinstance(entry, dict) and entry.get("status") == "dispatched":
            yield UpdateIssue(number, {ROUTED_OUTCOME_KEY: candidate["outcome_at"]})
    for number, candidate in pre.live_candidates.items():
        entry = locked_issues.get(str(number), {})
        if not isinstance(entry, dict) or entry.get("status") != "dispatched":
            continue
        draft = Draft(number, entry)
        draft.work[ROUTED_OUTCOME_KEY] = candidate["outcome_at"]
        if number in pre.pr_by_issue:
            yield from flush(draft)
            continue
        opened = yield OpenPrForBranch(number, candidate["branch"], "live_handoff")
        if opened.pr_number is not None:
            draft.work["status"] = PASSIVE_OPEN_STATUS
            draft.work["pr_number"] = opened.pr_number
            yield from flush(draft)
            yield emit(
                "worker_handoff_pr_opened",
                {
                    "issue_number": number,
                    "pr_number": opened.pr_number,
                    "branch_name": candidate["branch"],
                    "worker_reported": True,
                    "previous_status": "dispatched",
                    "reason": "worker_outcome_stale_live_pid",
                    "worker_pid": candidate["worker_pid"],
                    "worker_pid_still_running": True,
                    "outcome_age_minutes": round(candidate["outcome_age_minutes"], 1),
                    "label_write_ok": opened.error is None,
                    "pr_error": opened.error,
                },
            )
            continue
        fingerprint = drift_fingerprint(
            reason="live_worker_handoff_pr_create_failed",
            branch_name=candidate["branch"],
            error=opened.error or "unknown",
        )
        if draft.work.get("orphan_drift_fingerprint") != fingerprint:
            draft.work["orphan_drift_fingerprint"] = fingerprint
            draft.work["orphan_drift_at"] = facts.stamp
            yield from flush(draft)
            yield emit(
                "pr_create_failed_branch_stranded",
                {
                    "issue_number": number,
                    "branch_name": candidate["branch"],
                    "previous_status": "dispatched",
                    "reason": "live_worker_handoff_pr_create_failed",
                    "pr_create_error": opened.error,
                    "worker_reported": True,
                    "worker_pid": candidate["worker_pid"],
                    "worker_pid_still_running": True,
                },
            )
        else:
            yield from flush(draft)


def lock_flow(facts: SweepFacts, pre: PreOutcome) -> Flow:
    """The in-lock classification of every dead dispatched worker."""
    if pre.early_exit:
        return LockOutcome((), (), (), ())
    locked = facts.locked or {}
    acc = LockAcc(throttled_until=locked.get("throttled_until"))
    for item in events_to_emits(list(pre.salvage_events)):
        yield item
    locked_issues = locked.get("issues") or {}
    for number in pre.orphans:
        entry = locked_issues.get(str(number), {})
        if not isinstance(entry, dict) or entry.get("status") != "dispatched":
            continue
        draft = Draft(number, entry)
        pr = pre.pr_by_issue.get(number)
        if (yield from reap_flow(facts, pre, draft, pr, acc)):
            acc.reap_escalations.append(number)
            continue
        if pr:
            yield from with_pr_flow(facts, pre, draft, pr, acc)
        elif (yield from lock_no_pr_flow(facts, pre, draft)):
            acc.reap_escalations.append(number)
        yield from flush(draft)
    yield from _live_handoff_flow(facts, pre)
    return LockOutcome(
        outcome_apply_routes=tuple(acc.outcome_apply_routes),
        review_routes=tuple(acc.review_routes),
        no_op_routes=tuple(acc.no_op_routes),
        reap_escalations=tuple(acc.reap_escalations),
    )


def _completed(flow: Flow, observed: Mapping[Any, Any], phase: str) -> Any:
    value, _commits, pending = run_flow(flow, observed)
    if pending is not None:
        raise PhaseOrderError(f"{phase} flow is unfinished: {pending!r} has no observed result")
    return value


def decide(facts: SweepFacts, observed: Mapping[Any, Any]) -> SweepPlan:
    if facts.phase == "pre":
        flow = pre_flow(facts)
    else:
        pre = _completed(pre_flow(facts), observed, "pre")
        if facts.phase == "lock":
            flow = lock_flow(facts, pre)
        else:
            lock = _completed(lock_flow(facts, pre), observed, "lock")
            flow = post_flow(facts, pre, lock)
    _value, commits, pending = run_flow(flow, observed)
    return SweepPlan(requests=() if pending is None else (pending,), commits=commits)
