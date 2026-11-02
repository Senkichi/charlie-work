"""The dead-session lane's pure per-session plans (issue #2111).

``decide_dead_sessions`` holds the small predicates; this module composes them into
the plan each arm of the lane follows: which steps fire for which session facts, in
which order. The shell (``dead_sessions`` / ``dead_sessions_reclaim``) reads the
facts, asks for the plan, and executes the steps in order.

Some facts only exist once an earlier effect has run (the worktree inspection, a
salvage attempt, the live issue's labels, the PR enrichment), so a lane is a short
sequence of *stages*: each stage is one total function over the facts read so far
and returns either a step tuple or a verdict. Nothing here reads a clock, the
filesystem, GitHub or ``state.json``; ``now`` is the pass's sample, handed in.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .decide_dead_sessions import (
    BACKGROUND_EXIT_FAILURE_KIND,
    dead_fallback_kind,
    escalation_class,
    launch_failure_escalates,
    launch_failure_redispatch_at,
    redispatch_verdict,
)

CROSS_REPO_HOP_KIND = "cross_repo_hop"
PROVIDER_SUSPENDED_KIND = "provider_suspended"
LAUNCH_FAILURE_PERSIST_SOURCE = "dead_sessions_launch_failure"
DEAD_REAP_PERSIST_SOURCE = "dead_sessions_reap"


@dataclass(frozen=True)
class PersistFailure:
    """Stamp the classification (and throttle window) on the issue entry."""

    source: str


@dataclass(frozen=True)
class EscalateLaunchFailure:
    """Salvage-or-escalate a deterministic launch failure with no open PR."""


@dataclass(frozen=True)
class ReapSidecar:
    """Delete the session sidecar (issue #113: no phantom sessions from PID recycling)."""


@dataclass(frozen=True)
class RestoreRework:
    """Issue #295: return a launch-failed rework session to ``rework_requested``."""


@dataclass(frozen=True)
class WarnLiteralTmp:
    """Issue #1780: post-hoc literal-``/tmp`` signal; the sidecar was just reaped."""


@dataclass(frozen=True)
class EmitBackgroundExit:
    """Issue #2096: warning event for a fast clean exit that left uncommitted work."""


@dataclass(frozen=True)
class EmitProviderSuspended:
    """Issue #1342: error-level event on the first detection of an account suspension."""


@dataclass(frozen=True)
class ReclaimOrRoute:
    """Hand the reaped session's issue to the reclaim / rework-routing half."""


Step = (
    PersistFailure
    | EscalateLaunchFailure
    | ReapSidecar
    | RestoreRework
    | WarnLiteralTmp
    | EmitBackgroundExit
    | EmitProviderSuspended
    | ReclaimOrRoute
)


def plan_launch_failed(failure_kind: str | None, *, has_open_pr: bool) -> tuple[Step, ...]:
    """Issue #266: a launch-failure sidecar is terminal; persist, escalate, reap, restore."""
    steps: list[Step] = []
    if failure_kind:
        steps.append(PersistFailure(LAUNCH_FAILURE_PERSIST_SOURCE))
    if launch_failure_escalates(failure_kind, has_open_pr=has_open_pr):
        steps.append(EscalateLaunchFailure())
    steps.extend((ReapSidecar(), RestoreRework()))
    return tuple(steps)


@dataclass(frozen=True)
class DeadClassification:
    """How to classify a confirmed-dead session's failure.

    ``post_mortem_first``: for a completed-but-unpublished worktree the failure kind
    must be ``unpublished_work`` even if the log tail looks like a tool rejection, so
    classify first and record the post-mortem afterwards; other sessions record the
    post-mortem first so ``worker_blocked`` still escalates. ``session_completed``
    (issue #656) skips log-tail marker matching: the inspection is ground truth.
    """

    post_mortem_first: bool
    session_completed: bool
    fallback_kind: str | None


def plan_dead_classification(
    *, is_completed: bool, worktree_unknown: bool, background_exit: bool = False
) -> DeadClassification:
    return DeadClassification(
        post_mortem_first=not is_completed,
        session_completed=is_completed,
        fallback_kind=dead_fallback_kind(
            is_completed=is_completed,
            worktree_unknown=worktree_unknown,
            background_exit=background_exit,
        ),
    )


def plan_dead_reap(failure_kind: str | None) -> tuple[Step, ...]:
    """A confirmed-dead session: persist, reap, signal, then reclaim or route."""
    steps: list[Step] = []
    if failure_kind:
        steps.append(PersistFailure(DEAD_REAP_PERSIST_SOURCE))
    steps.extend((ReapSidecar(), WarnLiteralTmp()))
    if failure_kind == BACKGROUND_EXIT_FAILURE_KIND:
        steps.append(EmitBackgroundExit())
    if failure_kind == PROVIDER_SUSPENDED_KIND:
        steps.append(EmitProviderSuspended())
    steps.append(ReclaimOrRoute())
    return tuple(steps)


class Reclaim(Enum):
    NO_OPEN_PR = "no_open_pr"
    OPEN_PR = "open_pr"


def reclaim_route(*, has_open_pr: bool) -> Reclaim:
    return Reclaim.OPEN_PR if has_open_pr else Reclaim.NO_OPEN_PR


class NoPrGate(Enum):
    """What the no-open-PR reclaim does once it has the live issue (or failed to get it)."""

    SKIP = "skip"  # issue unreadable (deleted / no access): nothing to relabel
    PARK = "park"  # no active label: only a labelless local session may be parked
    PROCEED = "proceed"


def no_pr_gate(*, issue_found: bool, active_labels: Collection[str]) -> NoPrGate:
    """Gate the WHOLE reclaim on an active label being present (agrees with reconcile.py)."""
    if not issue_found:
        return NoPrGate.SKIP
    return NoPrGate.PROCEED if active_labels else NoPrGate.PARK


def wants_publish_salvage(*, ahead_count: int, has_repo_root: bool) -> bool:
    """Issue #252/#1130: committed-but-unpublished work is salvaged, not redispatched."""
    return ahead_count > 0 and has_repo_root


def wants_salvage_from_unsafe(*, ahead_count: int) -> bool:
    """Issue #807: ``ahead_count > 0`` filters shim dirt so salvage only fires for commits."""
    return ahead_count > 0


def scope_adjusted_kind(failure_kind: str | None, *, scope_passed: bool) -> str | None:
    """Issue #1244: a cross-repo scope hop overrides the kind so it escalates at once."""
    return failure_kind if scope_passed else CROSS_REPO_HOP_KIND


@dataclass(frozen=True)
class LaunchEscalation:
    """Commit plan for escalating a launch failure (issue #266)."""

    reason: str
    reason_class: str
    redispatch_at: tuple[str, ...]
    removed_labels: tuple[str, ...]


def plan_launch_escalation(
    windowed: Sequence[str],
    failure_kind: str,
    *,
    now: datetime,
    active_labels: Collection[str],
) -> LaunchEscalation:
    """The launch-failure escalation always counts itself as a redispatch event."""
    return LaunchEscalation(
        reason=failure_kind,
        reason_class=escalation_class(failure_kind).reason_class,
        redispatch_at=launch_failure_redispatch_at(windowed, now),
        removed_labels=tuple(sorted(active_labels)),
    )


@dataclass(frozen=True)
class ReclaimCommit:
    """Commit plan for the no-open-PR relabel branch: escalate, or relabel to ``ready``."""

    escalate: bool
    reason: str | None
    reason_class: str
    redispatch_at: tuple[str, ...]
    removed_labels: tuple[str, ...]
    add_ready: bool


def plan_reclaim_commit(
    windowed: Sequence[str],
    failure_kind: str | None,
    *,
    now: datetime,
    max_auto_redispatch: int,
    active_labels: Collection[str],
    ready_label_present: bool,
) -> ReclaimCommit:
    verdict = redispatch_verdict(
        windowed, failure_kind, now=now, max_auto_redispatch=max_auto_redispatch
    )
    return ReclaimCommit(
        escalate=verdict.escalate,
        reason=verdict.reason,
        reason_class=verdict.reason_class,
        redispatch_at=verdict.redispatch_at,
        removed_labels=tuple(sorted(active_labels)),
        add_ready=not ready_label_present,
    )


class OpenPrRoute(Enum):
    SKIP = "skip"  # completed worktree: the worker finished, never roll it back
    RESTORE = "restore"  # return to ``rework_requested``
    INSPECT_PR = "inspect_pr"  # enrich the PR, then ``open_pr_candidate_route``
    ROUTE_PRE_REVIEW = "route_pre_review"  # conflict / stale-checks rework candidate


def open_pr_route(*, is_completed: bool, has_rework_pr: bool) -> OpenPrRoute:
    """Issue #295/#315: an open PR's dead session goes back to rework unless it completed."""
    if is_completed:
        return OpenPrRoute.SKIP
    return OpenPrRoute.INSPECT_PR if has_rework_pr else OpenPrRoute.RESTORE


def open_pr_candidate_route(*, is_candidate: bool) -> OpenPrRoute:
    """A pre-review rework candidate is routed (conflict / stale checks); else restored."""
    return OpenPrRoute.ROUTE_PRE_REVIEW if is_candidate else OpenPrRoute.RESTORE
