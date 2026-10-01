"""Commit appliers: turn the decided effects into state, event and label writes.

``decide`` hands the shell a growing prefix of commits; ``apply_commit`` runs one
of them. Where it writes depends on the phase:

* ``pre`` -- no lock is held. Events go through a short ``state_lock`` on a fresh
  load (the pre phase never saves ``ctx.state``).
* ``lock`` -- inside the sweep's state lock on ``ctx.state``. Events queue on
  ``sweep_events`` (saved once at the end of the lock); issue updates mutate
  ``ctx.state`` in place with merge semantics (ADR-0001).
* ``post`` -- no lock held. Every write re-loads state under a short lock. State
  merges are not commits here: they are ``GuardedUpdate`` requests, so the flow
  learns whether the write committed (see ``apply_requests_lock.guarded_update``).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from .. import worker_fate
from . import effects_rework
from ..escalation import _escalation_edge
from ..labels import TransitionOutcome
from .apply_context import SweepContext
from .model import (
    Emit,
    Escalate,
    ReportStaleEvidence,
    RoutePreReviewRework,
    TransitionLabel,
    UpdateIssue,
)

logger = logging.getLogger(__name__)

# Commits that do network I/O; the lock phase must never carry them.
LOCK_ILLEGAL_COMMITS = (RoutePreReviewRework, TransitionLabel)
# Commits whose effect is an unguarded write to ``ctx.state``, which the post phase
# never saves; its merges go through the guarded ``GuardedUpdate`` request instead.
POST_ILLEGAL_COMMITS = (UpdateIssue, Escalate, RoutePreReviewRework)


def _emit_under_lock(ctx: SweepContext, commit: Emit) -> None:
    if commit.audit_only:
        ctx.write_gate.log_event(
            # event-consumer: audit-only -- forwards ``commit.kind`` unchanged; the consumer
            # or audit-only justification lives at the ``emit(...)`` literal in ``decide*``
            kind=commit.kind,
            payload=dict(commit.payload),
            level=commit.level,
        )
        return
    with ctx.ports.state_lock(ctx.state_file):
        state = ctx.ports.load_state(ctx.state_file)
        state = ctx.write_gate.append_event(
            state,
            # event-consumer: audit-only -- forwards ``commit.kind`` unchanged; the consumer
            # or audit-only justification lives at the ``emit(...)`` literal in ``decide*``
            commit.kind,
            dict(commit.payload),
            ctx.config.runtime.event_ring_size,
            level=commit.level,
        )
        ctx.write_gate.save_state(state)


def _apply_update(entry: dict[str, Any], commit: UpdateIssue) -> None:
    entry.update(copy.deepcopy(dict(commit.set_fields)))
    for key in commit.clear_fields:
        entry.pop(key, None)


def _update_in_memory(ctx: SweepContext, commit: UpdateIssue) -> None:
    entry = (ctx.state.get("issues") or {}).get(str(commit.issue))
    if isinstance(entry, dict):
        _apply_update(entry, commit)


def _escalate(ctx: SweepContext, commit: Escalate) -> None:
    ctx.state = ctx.ports.escalate_issue(
        ctx.state,
        commit.issue,
        reason=commit.reason,
        reason_class=commit.reason_class,
        pr_number=commit.pr_number,
        issue_extra=dict(commit.issue_extra) or None,
    )


def _route_pre_review_rework(ctx: SweepContext, commit: RoutePreReviewRework) -> None:
    enriched = ctx.pr_views.get(commit.pr_number) or ctx.pr_by_issue[commit.issue]
    effects_rework._route_dead_worker_to_pre_review_rework(
        ctx.state_file,
        ctx.gh,
        ctx.config,
        enriched,
        commit.issue,
        commit.reason,
        failure_kind=None,
        write_gate=ctx.write_gate,
    )


def _persist_label_error(ctx: SweepContext, issue: int, error: dict[str, Any]) -> None:
    with ctx.ports.state_lock(ctx.state_file):
        state = ctx.ports.load_state(ctx.state_file)
        entry = (state.get("issues") or {}).get(str(issue), {})
        if isinstance(entry, dict):
            entry["label_error"] = error
            state["issues"][str(issue)] = entry
            ctx.write_gate.save_state(state)


def _transition(ctx: SweepContext, commit: TransitionLabel) -> None:
    if commit.edge == "escalated":
        ctx.write_gate.transition(
            ctx.gh, ctx.config.labels, commit.issue, _escalation_edge("escalated", "mechanical")
        )
        return
    try:
        result = ctx.write_gate.transition(ctx.gh, ctx.config.labels, commit.issue, commit.edge)
    except Exception as exc:
        logger.exception("%s label transition escaped for issue %s", commit.edge, commit.issue)
        if commit.persist_label_error:
            _persist_label_error(
                ctx,
                commit.issue,
                {
                    "edge": commit.edge,
                    "outcome": "exception",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        return
    if result.outcome == TransitionOutcome.APPLIED or not commit.persist_label_error:
        return
    _persist_label_error(
        ctx,
        commit.issue,
        {
            "edge": commit.edge,
            "outcome": result.outcome.value,
            "add_failures": result.add_failures,
            "remove_failures": result.remove_failures,
        },
    )


def _lock_phase_event(commit: Emit) -> tuple[str, dict[str, Any]]:
    """The ``(kind, payload)`` pair ``_append_sweep_events`` writes at the end of the lock.

    The kind is whatever literal ``decide*`` chose; ``tests/test_dws_event_kinds.py``
    verifies those literals at their origin."""
    return (commit.kind, dict(commit.payload))


def apply_commit(ctx: SweepContext, phase: str, commit: Any, sweep_events: list[Any]) -> None:
    """Apply one decided commit for ``phase``."""
    if isinstance(commit, Emit):
        if phase == "lock":
            sweep_events.append(_lock_phase_event(commit))
        else:
            _emit_under_lock(ctx, commit)
    elif isinstance(commit, UpdateIssue):
        _update_in_memory(ctx, commit)
    elif isinstance(commit, Escalate):
        _escalate(ctx, commit)
    elif isinstance(commit, ReportStaleEvidence):
        worker_fate.report_stale_evidence(
            ctx.state_file, ctx.buckets[commit.bucket], write_gate=ctx.write_gate
        )
    elif isinstance(commit, RoutePreReviewRework):
        _route_pre_review_rework(ctx, commit)
    elif isinstance(commit, TransitionLabel):
        _transition(ctx, commit)
    else:  # pragma: no cover - COMMIT_TYPES is closed
        raise TypeError(f"unknown commit {commit!r}")
