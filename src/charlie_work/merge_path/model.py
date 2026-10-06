"""Value types for the **Merge path** decision stages.

Facts are what a gather step observed; plans are what ``decide_*`` concluded.
Every type here is a frozen dataclass and every collection a tuple/frozenset
(or a read-only mapping), so a plan can be compared, logged and replayed.

Vocabulary: the outputs are ``Admission``, ``BranchGate``, ``Readiness``,
``MergePlan`` and ``Accounting`` -- never "Decision", which CONTEXT.md lists as
an avoided synonym of **Verdict**. The review verdict itself enters as a fact
(``VerdictFact``), never as a decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from ..checks import CheckSummary


class FactNotGathered(ValueError):
    """A decide function needed a fact the gather step left unread.

    Raised instead of defaulting so a gather guard that drifts from the
    decision it feeds fails loudly in the table tests.
    """


class PlanKind(StrEnum):
    """What the merge path concluded for one PR on one pass."""

    SKIP = "skip"  # already merged
    NOT_FOUND = "not_found"
    REQUEST_REWORK = "request_rework"  # re-review, stall, conflict/check/revert routes
    WAIT = "wait"  # conflict rework in flight, not train head, stale base
    HOLD = "hold"  # escalated / human-merge / merge-hold / blocked conflict
    RERUN_OR_ESCALATE = "rerun_or_escalate"  # infra remediation
    ESCALATE = "escalate"  # human-merge hand-off (policy escalation)
    HAND_OFF = "hand_off"  # merge queue hand-off (ADR-0003)
    MERGE = "merge"  # self-merge
    NONE = "none"  # evaluated, nothing to do


class StageKind(StrEnum):
    """Stage-local "keep going" value; only terminal outcomes are ``PlanKind``."""

    PROCEED = "proceed"


class BranchStop(StrEnum):
    """Why ``decide_branch`` ended the pass."""

    CONFLICT_IN_FLIGHT = "conflict_in_flight"
    CONFLICT_BLOCKED = "conflict_blocked"
    NOT_TRAIN_HEAD = "not_train_head"
    STALE_BASE = "stale_base"


class RevertStatus(StrEnum):
    CLEAN = "clean"
    DETECTED = "detected"
    UNDETERMINED = "undetermined"


class SyncOutcome(StrEnum):
    """Result of the branch-sync effect, fed back into ``decide_branch``."""

    NOT_NEEDED = "not_needed"
    SKIPPED_QUEUED = "skipped_queued"
    NOT_ATTEMPTED = "not_attempted"  # preview never applies effects
    FAILED = "failed"  # update call refused, or head could not be verified
    SAME_HEAD = "same_head"
    NEW_HEAD = "new_head"


class Hold(StrEnum):
    ESCALATED = "escalated"
    HUMAN_MERGE = "human_merge"
    HUMAN_MERGE_UNAVAILABLE = "human_merge_unavailable"
    MERGE_HOLD = "merge_hold"
    MERGE_HOLD_UNAVAILABLE = "merge_hold_unavailable"
    REVERT_UNDETERMINED = "revert_undetermined"
    CHECKS_UNAVAILABLE = "checks_unavailable"


@dataclass(frozen=True)
class Unavailable:
    """Marker for "the read failed" (errors come back as values)."""


UNAVAILABLE = Unavailable()


@dataclass(frozen=True)
class VerdictFact:
    """View of the review verdict for one PR.

    ``raw`` is kept for rendering (``review_decision`` payload key); decide
    functions never read it.
    """

    approved: bool
    reviewed_head_sha: str | None
    raw: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class PersistedPr:
    """Snapshot of ``state.prs[n]`` counters the decisions depend on."""

    status: str | None = None
    mergequeue_revoked_reason: str | None = None
    failed_attempts: int = 0
    stale_base_deferrals: int = 0
    mergequeue_since: str | None = None
    mergequeue_head_sha: str | None = None
    # Stamped by every full merge_ready pass that leaves the PR queued; the
    # queue skip (#2440) forces a re-check once it is 30 minutes old.
    mergequeue_checked_at: str | None = None


@dataclass(frozen=True)
class MergePathConfig:
    """The config slice the decisions read (built once from the orchestrator config)."""

    auto_merge_enabled: bool = True
    mergequeue_label: str | None = None
    update_branch_strategy: str = "front_of_train"
    require_approved_review: bool = True
    failed_attempt_alarm: int = 3
    max_conflict_rework_attempts: int = 2
    human_merge_labels: tuple[str, ...] = ()
    review_dispatch_enabled: bool = True
    required_checks: tuple[str, ...] = ()
    readiness_no_ci_minutes: int = 15
    merge_strategy: str = "squash"
    # TIS-CW-7: Aviator's skip-line label (None = off) and the priority label prefix.
    skip_line_label: str | None = None
    priority_prefix: str = ""


# --------------------------------------------------------------------------- #
# Stage 1: admission
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AdmissionFacts:
    pr_number: int
    config: MergePathConfig
    persisted: PersistedPr | None
    pr_found: bool
    live_labels: frozenset[str]
    live_head_sha: str | None
    issue_number: int | None
    verdict: VerdictFact
    # None = not read. Read only when approved, the head moved and the live
    # head is known; ``decide_admission`` raises FactNotGathered otherwise.
    carry_forward: bool | None
    pr_escalated: bool = False
    issue_escalated: bool = False


@dataclass(frozen=True)
class Admission:
    kind: PlanKind | StageKind
    approved: bool = False
    mergequeue_label_reverted: bool = False
    self_revoked_stale_head: bool = False
    # PROCEED that still needs the approval-head write + refetch (live) before
    # the second ``decide_admission(..., observed_carry_forward=True)`` call.
    carry_forward_needed: bool = False
    head_moved: bool = False
    # Re-review bookkeeping (only meaningful when kind is REQUEST_REWORK).
    escalated: bool = False
    stamp_reviewing: bool = False
    transition_review_started: bool = False
    record_head_moved: bool = False


# --------------------------------------------------------------------------- #
# Stage 2: branch gate
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BranchFacts:
    pr_number: int
    config: MergePathConfig
    admission: Admission
    persisted: PersistedPr | None
    merge_conflict: bool
    issue_status: str | None
    train_head_param: int | None
    # Observed train head, only when ``train_head_param`` is None and the
    # strategy is front_of_train. ``UNAVAILABLE`` = pr_list failed.
    train_head: int | None | Unavailable = None
    # None = not read (conflict / list failure / strategy "off" short-circuits).
    base_currency_gated: bool | None = None
    base_current_read: bool = False
    base_current: bool | None = None
    should_update_branch: bool | None = None


@dataclass(frozen=True)
class BranchGate:
    kind: PlanKind | StageKind
    admission: Admission
    stop: BranchStop | None = None
    merge_conflict: bool = False
    sync_failed: bool = False
    request_sync: bool = False
    already_in_mergequeue: bool = False
    stale_base_reason: str | None = None


# --------------------------------------------------------------------------- #
# Stage 3: readiness
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateInputs:
    summary_ready: bool
    approved: bool
    require_approved_review: bool
    sync_failed: bool

    @property
    def can_merge(self) -> bool:
        return (
            self.summary_ready
            and (self.approved or not self.require_approved_review)
            and not self.sync_failed
        )

    def as_payload(self) -> dict[str, bool]:
        """The four keys spread into the result data and the ``merge_ready`` event (#1060)."""
        return {
            "summary_ready": self.summary_ready,
            "approved": self.approved,
            "require_approved_review": self.require_approved_review,
            "sync_failed": self.sync_failed,
        }


@dataclass(frozen=True)
class ReadinessFacts:
    pr_number: int
    config: MergePathConfig
    branch: BranchGate
    issue_number: int | None
    issue_status: str | None
    issue_reason_class: str | None
    revert: RevertStatus
    revert_reason: str | None
    checks: CheckSummary
    checks_unavailable: bool
    # Names of the check runs GitHub reported (for the no-CI stall predicate).
    check_names_seen: frozenset[str]
    now: datetime
    pr_updated_at: str | None
    is_draft: bool
    human_merge_hold: bool
    human_merge_check_unavailable: bool
    pr_escalated: bool = False
    issue_escalated: bool = False
    containment_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RevertVerdict:
    """Outcome of the cross-PR revert gate (``decide_revert``)."""

    sync_failed: bool
    detected: bool
    undetermined: bool
    route: bool


@dataclass(frozen=True)
class Readiness:
    kind: PlanKind | StageKind
    gate: GateInputs
    branch: BranchGate
    issue_number: int | None
    summary: CheckSummary
    checks_unavailable: bool
    pending_only: bool
    cross_pr_revert_detected: bool = False
    cross_pr_revert_undetermined: bool = False
    cross_pr_revert_reason: str | None = None
    route_cross_pr_revert: bool = False
    readiness_stall: bool = False
    infra_eligible: bool = False
    deescalate: bool = False
    human_merge_hold: bool = False
    human_merge_check_unavailable: bool = False
    containment_warnings: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Stage 4: merge plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HoldFacts:
    """Escalation / hold facts, gathered AFTER the stage-3 effects (de-escalation)."""

    config: MergePathConfig
    persisted: PersistedPr | None
    issue_status: str | None
    pr_escalated: bool
    issue_escalated: bool
    should_merge: bool
    # None = not read. Read only when a hand-off could happen.
    merge_hold: bool | None = None
    merge_hold_unavailable: bool = False


@dataclass(frozen=True)
class MergePlan:
    kind: PlanKind
    readiness: Readiness
    escalated_merge_hold: bool = False
    should_merge: bool = False
    action_merge: bool = False
    action_hand_off: bool = False
    action_human_merge: bool = False
    read_merge_hold: bool = False
    merge_hold: bool = False
    merge_hold_unavailable: bool = False
    reset_failed_attempts_on_hand_off: bool = False
    holds: frozenset[Hold] = frozenset()
    conflict_rework: bool = False
    check_failure_rework: bool = False


# --------------------------------------------------------------------------- #
# Stage 5: accounting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EffectResults:
    """What the stage-4 effects reported back (only ever feeds accounting/render)."""

    mergequeue_label_applied: bool | None = None
    merge_output: str | None = None
    merged_at: str | None = None
    branch_deleted: bool | None = None
    update_open_prs_results: tuple[Mapping[str, Any], ...] | None = None
    cancel_results: Mapping[str, Any] | None = None
    human_merge_label_error: Mapping[str, Any] | None = None
    cross_pr_revert_routed: bool = False
    conflict_routed: bool = False
    conflict_escalated: bool = False
    check_failure_routed: bool = False
    rework_label_error: Mapping[str, Any] | None = None
    issue_status_after: str | None = None


@dataclass(frozen=True)
class AccountingFacts:
    """Gathered INSIDE the state lock, after the merge (``deadline_spent`` is read then)."""

    pr_number: int
    issue_number: int | None
    config: MergePathConfig
    locked: PersistedPr
    deadline_spent: bool
    now_iso: str
    live_head_sha: str | None
    mergeable: str | None = None
    merge_state_status: str | None = None


@dataclass(frozen=True)
class EventSpec:
    kind: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class Accounting:
    handoff_failed: bool
    failed_attempts: int
    stale_base_deferrals: int
    alarm: bool
    warning: str | None
    merge_alert_ok: bool
    merged: bool
    pr_status: str | None  # "merged" when we merged, else None (leave status as-is)
    mergequeue_since: str | None
    mergequeue_head_sha: str | None
    events: tuple[EventSpec, ...] = ()
