"""Pure decision stages of the **Merge path**.

Five functions, each fed by one gather step and each a pure function of its
arguments: no clock, no ``gh``, no state file, no config object beyond the
``MergePathConfig`` slice. Effects run between the stages in a thin shell; two
stages (admission, branch) are called a second time with the result of an
effect the first call requested (the carry-forward refetch, the branch sync).

    decide_admission -> decide_branch -> decide_readiness -> decide_merge
                                                         \\-> decide_accounting

Every rule below was transcribed from ``OrchestratorApp.merge_ready``; the
comments name the lane or issue that motivated it so a later edit can see what
it would undo. ADR-0003 (merge-queue hand-off) invariants are pinned in
``decide_merge`` / ``decide_accounting`` and their table tests.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from types import MappingProxyType
from typing import Any, Mapping

from .model import (
    PersistedPr,
    Accounting,
    AccountingFacts,
    Admission,
    AdmissionFacts,
    BranchFacts,
    BranchGate,
    BranchStop,
    EffectResults,
    EventSpec,
    FactNotGathered,
    GateInputs,
    Hold,
    HoldFacts,
    MergePlan,
    PlanKind,
    Readiness,
    ReadinessFacts,
    RevertStatus,
    StageKind,
    SyncOutcome,
    Unavailable,
    VerdictFact,
)
from .rules import (
    CHECK_ROUTE_EXCLUDED_STATUSES,
    CONFLICT_REWORK_IN_FLIGHT_STATUSES,
    CONFLICT_ROUTE_EXCLUDED_STATUSES,
    REWORK_ALREADY_ROUTED_STATUSES,
    format_merge_attempt_alarm_message,
    is_pending_only,
    readiness_no_ci_stall,
)

_SYNC_STRATEGIES = frozenset({"front_of_train", "broadcast"})
_CARRY_FORWARD_FROM_REVOKE = "stale_head_pending_carry_forward"
SYNC_STRATEGIES = _SYNC_STRATEGIES


# --------------------------------------------------------------------------- #
# Gather guards: the one definition of "does this stage need that fact"
# --------------------------------------------------------------------------- #
# A gather step reads a fact only when the stage below would consume it; the
# predicates live here, next to the stage, so a guard cannot drift from the
# decision it feeds (the stage raises FactNotGathered if it ever does).


def is_head_moved(verdict: VerdictFact, live_head_sha: str | None) -> bool:
    """The approved head is unknown or differs from the live head."""
    reviewed = verdict.reviewed_head_sha
    return reviewed is None or live_head_sha != reviewed


def needs_carry_forward(verdict: VerdictFact, live_head_sha: str | None) -> bool:
    """``AdmissionFacts.carry_forward`` is consumed: approved, head moved, live head known."""
    return verdict.approved and is_head_moved(verdict, live_head_sha) and bool(live_head_sha)


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


# --------------------------------------------------------------------------- #
# Stage 1: admission
# --------------------------------------------------------------------------- #


def decide_admission(
    f: AdmissionFacts,
    *,
    observed_carry_forward: bool = False,
    preview_unknown_live_head_proceeds: bool = False,
) -> Admission:
    """Skip / not-found / re-review / proceed, from the PR's persisted and live facts.

    ``observed_carry_forward`` is the second-call argument: the approval head
    was carried forward (and the PR refetched), so the head-moved check is not
    repeated -- this bounds the stage to one extra call and matches today's
    behaviour on a refetch race.

    ``preview_unknown_live_head_proceeds`` reproduces the legacy dry-run, which
    skips the carry-forward check (and so proceeds) when the live head is
    unknown, where the live path re-reviews. Only the preview driver sets it;
    it disappears with the preview flips.
    """
    persisted = f.persisted
    if persisted is not None and persisted.status == "merged":
        return Admission(kind=PlanKind.SKIP)
    if not f.pr_found:
        return Admission(kind=PlanKind.NOT_FOUND)

    label = f.config.mergequeue_label
    prior_status = persisted.status if persisted is not None else None
    reverted = bool(label and prior_status == "mergequeue" and label not in f.live_labels)
    # Only a reconcile-recorded stale-head revocation is a self-revocation; a
    # "not_approved" revocation already keeps can_merge False (issue #1402).
    revoked_reason = persisted.mergequeue_revoked_reason if persisted is not None else None
    self_revoked = bool(reverted and revoked_reason == _CARRY_FORWARD_FROM_REVOKE)
    base = Admission(
        kind=StageKind.PROCEED,
        approved=f.verdict.approved,
        mergequeue_label_reverted=reverted,
        self_revoked_stale_head=self_revoked,
    )
    if not f.verdict.approved:
        return base

    head_moved = is_head_moved(f.verdict, f.live_head_sha)
    base = replace(base, head_moved=head_moved)
    if not head_moved or observed_carry_forward:
        return base

    if f.live_head_sha:
        if f.carry_forward is None:
            raise FactNotGathered("carry_forward is needed: approved, head moved, live head known")
        if f.carry_forward:
            return replace(base, carry_forward_needed=True)
    elif preview_unknown_live_head_proceeds:
        return base

    # Head moved and not carried forward: re-review (issue #833 escalation
    # gate -- an escalated PR is told to re-review but nothing is stamped).
    escalated = f.pr_escalated or f.issue_escalated
    dispatch_enabled = f.config.review_dispatch_enabled
    return replace(
        base,
        kind=PlanKind.REQUEST_REWORK,
        escalated=escalated,
        stamp_reviewing=not escalated and dispatch_enabled,
        transition_review_started=not escalated
        and f.issue_number is not None
        and dispatch_enabled,
        record_head_moved=not escalated,
    )


# --------------------------------------------------------------------------- #
# Stage 2: branch gate
# --------------------------------------------------------------------------- #


def branch_precheck(f: BranchFacts) -> tuple[BranchGate | None, bool, bool]:
    """Conflict and train-head stages: ``(stop_gate, merge_conflict, sync_failed)``.

    The first stage of ``decide_branch``, exposed so a gather step can stop
    before the reads the later stages need (a conflict or a non-head PR returns
    before any base-currency read, as the legacy body did). ``train_head=None``
    with no ``train_head_param`` means "no other head", so a caller that has
    not yet listed PRs gets the conflict and param stops only.
    """
    adm = f.admission
    strategy = f.config.update_branch_strategy
    sync_failed = False
    merge_conflict = False

    if f.merge_conflict:
        if f.issue_status in CONFLICT_REWORK_IN_FLIGHT_STATUSES:
            return (
                BranchGate(
                    kind=PlanKind.WAIT,
                    admission=adm,
                    stop=BranchStop.CONFLICT_IN_FLIGHT,
                    merge_conflict=True,
                ),
                True,
                False,
            )
        if f.issue_status == "blocked":
            return (
                BranchGate(
                    kind=PlanKind.HOLD,
                    admission=adm,
                    stop=BranchStop.CONFLICT_BLOCKED,
                    merge_conflict=True,
                ),
                True,
                False,
            )
        # Any other status -- including "escalated" (issue #776) -- routes.
        merge_conflict = True
        sync_failed = True

    if strategy == "front_of_train":
        head_param = f.train_head_param
        if head_param is not None:
            if head_param != f.pr_number:
                return _not_train_head(adm, merge_conflict), merge_conflict, sync_failed
        elif isinstance(f.train_head, Unavailable):
            sync_failed = True
        elif f.train_head is not None and f.train_head != f.pr_number:
            return _not_train_head(adm, merge_conflict), merge_conflict, sync_failed
    return None, merge_conflict, sync_failed


def decide_branch(f: BranchFacts, sync: SyncOutcome | None = None) -> BranchGate:
    """Conflict / train-head / branch-sync / stale-base gate.

    ``sync=None`` is the first call: when a branch sync is needed the result
    carries ``request_sync=True`` and no further verdict. The shell performs
    the sync, re-gathers, and calls again with the :class:`SyncOutcome`.
    Unapproved PRs short-circuit (none of these reads ever ran for them).
    """
    adm = f.admission
    if not adm.approved:
        return BranchGate(kind=StageKind.PROCEED, admission=adm)

    cfg = f.config
    strategy = cfg.update_branch_strategy
    stop, merge_conflict, sync_failed = branch_precheck(f)
    if stop is not None:
        return stop

    already_in_mergequeue = is_already_in_mergequeue(f.persisted, adm)

    gated = False
    if not sync_failed and strategy != "off":
        if f.base_currency_gated is None:
            raise FactNotGathered("base_currency_gated is needed: strategy is not 'off'")
        gated = f.base_currency_gated

    if sync is SyncOutcome.FAILED:
        sync_failed = True

    if sync is None and not sync_failed and gated and strategy in _SYNC_STRATEGIES:
        # Already-queued PRs suppress OUR sync (Aviator owns the branch) but
        # still pass through the freshness gate below (ADR-0003).
        if not already_in_mergequeue:
            if f.should_update_branch is None:
                raise FactNotGathered("should_update_branch is needed before a branch sync")
            if f.should_update_branch:
                return BranchGate(
                    kind=StageKind.PROCEED,
                    admission=adm,
                    merge_conflict=merge_conflict,
                    sync_failed=False,
                    request_sync=True,
                    already_in_mergequeue=False,
                )

    if not sync_failed and gated:
        if not f.base_current_read:
            raise FactNotGathered("base_current is needed: the base-currency gate applies")
        if f.base_current is not True:
            reason = "compare_unavailable" if f.base_current is None else "base_stale"
            return BranchGate(
                kind=PlanKind.WAIT,
                admission=adm,
                stop=BranchStop.STALE_BASE,
                merge_conflict=merge_conflict,
                sync_failed=False,
                already_in_mergequeue=already_in_mergequeue,
                stale_base_reason=reason,
            )

    return BranchGate(
        kind=StageKind.PROCEED,
        admission=adm,
        merge_conflict=merge_conflict,
        sync_failed=sync_failed,
        already_in_mergequeue=already_in_mergequeue,
    )


def _not_train_head(adm: Admission, merge_conflict: bool) -> BranchGate:
    return BranchGate(
        kind=PlanKind.WAIT,
        admission=adm,
        stop=BranchStop.NOT_TRAIN_HEAD,
        merge_conflict=merge_conflict,
    )


# --------------------------------------------------------------------------- #
# Stage 3: readiness
# --------------------------------------------------------------------------- #


def decide_readiness(f: ReadinessFacts) -> Readiness:
    """Cross-PR revert, readiness-no-CI stall, infra remediation, gate inputs.

    ``kind`` is REQUEST_REWORK for a stall and RERUN_OR_ESCALATE for an
    eligible infra failure. Live treats the stall as terminal; it treats infra
    as terminal only if the remediation helper returns a result, and otherwise
    continues with ``dataclasses.replace(readiness, kind=StageKind.PROCEED)``.
    """
    branch = f.branch
    approved = branch.admission.approved
    cfg = f.config
    summary = f.checks
    sync_failed = branch.sync_failed

    detected = False
    undetermined = False
    route_revert = False
    if approved and not sync_failed:
        if f.revert is RevertStatus.DETECTED:
            sync_failed = True
            detected = True
            route_revert = (
                f.issue_number is not None and f.issue_status not in REWORK_ALREADY_ROUTED_STATUSES
            )
        elif f.revert is RevertStatus.UNDETERMINED:
            # Fail closed, never route.
            sync_failed = True
            undetermined = True

    pending_only = is_pending_only(summary)
    stall_window = (
        not f.checks_unavailable
        and approved
        and not sync_failed
        and not summary.ready
        and not pending_only
        and f.issue_number is not None
        and readiness_no_ci_stall(
            check_names_seen=f.check_names_seen,
            updated_at=f.pr_updated_at,
            now=f.now,
            required_checks=cfg.required_checks,
            minutes=cfg.readiness_no_ci_minutes,
        )
    )
    stall = stall_window and f.issue_status not in CHECK_ROUTE_EXCLUDED_STATUSES

    # Infra remediation: the janitor's sole-blocker predicate in this lane's
    # vocabulary (issue #1912). Pending checks are deliberately not excluded.
    infra_eligible = not (
        (not approved and cfg.require_approved_review)
        or sync_failed
        or f.issue_number is None
        or f.is_draft
        or not summary.infra_failed
        or summary.failed
        or summary.missing
        or summary.unavailable
        or summary.infra_blocked
        or f.pr_escalated
        or f.issue_escalated
    )

    kind: PlanKind | StageKind = StageKind.PROCEED
    if stall:
        kind = PlanKind.REQUEST_REWORK
    elif infra_eligible:
        kind = PlanKind.RERUN_OR_ESCALATE

    deescalate = bool(
        cfg.human_merge_labels
        and f.issue_number is not None
        and not f.human_merge_hold
        and not f.human_merge_check_unavailable
        and f.issue_status == "escalated"
        and f.issue_reason_class == "policy"
    )
    gate = GateInputs(
        summary_ready=summary.ready,
        approved=approved,
        require_approved_review=cfg.require_approved_review,
        sync_failed=sync_failed,
    )
    return Readiness(
        kind=kind,
        gate=gate,
        branch=branch,
        issue_number=f.issue_number,
        summary=summary,
        checks_unavailable=f.checks_unavailable,
        pending_only=pending_only,
        cross_pr_revert_detected=detected,
        cross_pr_revert_undetermined=undetermined,
        cross_pr_revert_reason=f.revert_reason,
        route_cross_pr_revert=route_revert,
        readiness_stall=stall,
        infra_eligible=infra_eligible,
        deescalate=deescalate,
        human_merge_hold=f.human_merge_hold,
        human_merge_check_unavailable=f.human_merge_check_unavailable,
        containment_warnings=f.containment_warnings,
    )


# --------------------------------------------------------------------------- #
# Stage 4: merge plan
# --------------------------------------------------------------------------- #


def decide_merge(readiness: Readiness, holds: HoldFacts) -> MergePlan:
    """Self-merge / hand-off / human-merge / rework-route / hold, after de-escalation.

    ``holds`` is gathered after the stage-3 effects, so the escalation flags
    are the post-de-escalation ones. The escalated hold leaves
    ``gate.can_merge`` untouched (#840 / #777b): it only blocks the actions.
    """
    cfg = holds.config
    gate = readiness.gate
    can_merge = gate.can_merge
    adm = readiness.branch.admission
    summary = readiness.summary
    human_hold = readiness.human_merge_hold
    human_unavailable = readiness.human_merge_check_unavailable

    escalated_merge_hold, enter = merge_entry(readiness, holds)
    should_merge = holds.should_merge
    label = cfg.mergequeue_label
    read_merge_hold = merge_hold_read_needed(readiness, holds)
    if read_merge_hold and holds.merge_hold is None and not holds.merge_hold_unavailable:
        raise FactNotGathered("merge_hold is needed: a mergequeue hand-off is possible")
    merge_hold = read_merge_hold and holds.merge_hold is True
    merge_hold_unavailable = read_merge_hold and holds.merge_hold_unavailable

    action_hand_off = read_merge_hold and not merge_hold and not merge_hold_unavailable
    action_merge = enter and not label
    action_human_merge = human_hold and can_merge and should_merge and not escalated_merge_hold

    prior = holds.persisted.failed_attempts if holds.persisted is not None else 0
    threshold = cfg.failed_attempt_alarm
    debounced = threshold > 0 and prior + 1 >= threshold
    issue_bound = readiness.issue_number is not None
    approved = gate.approved
    merge_conflict = readiness.branch.merge_conflict

    conflict_rework = (
        merge_conflict
        and approved
        and not can_merge
        and not readiness.pending_only
        and issue_bound
        and holds.issue_status not in CONFLICT_ROUTE_EXCLUDED_STATUSES
        and debounced
    )
    # The hand-off-failed exclusion in today's check-failure guard is implied:
    # a failed hand-off needs can_merge, and this route needs not can_merge.
    check_failure_rework = (
        not merge_conflict
        and not readiness.cross_pr_revert_detected
        and approved
        and not can_merge
        and bool(summary.failed)
        and issue_bound
        and holds.issue_status not in CHECK_ROUTE_EXCLUDED_STATUSES
        and debounced
    )
    if (conflict_rework or check_failure_rework) and (
        action_merge or action_hand_off or action_human_merge
    ):
        raise AssertionError("rework routes and merge/hand-off actions are exclusive")

    held: set[Hold] = set()
    if escalated_merge_hold:
        held.add(Hold.ESCALATED)
    if human_hold:
        held.add(Hold.HUMAN_MERGE)
    if human_unavailable:
        held.add(Hold.HUMAN_MERGE_UNAVAILABLE)
    if merge_hold:
        held.add(Hold.MERGE_HOLD)
    if merge_hold_unavailable:
        held.add(Hold.MERGE_HOLD_UNAVAILABLE)
    if readiness.cross_pr_revert_undetermined:
        held.add(Hold.REVERT_UNDETERMINED)
    if readiness.checks_unavailable:
        held.add(Hold.CHECKS_UNAVAILABLE)

    kind: PlanKind
    if readiness.kind is not StageKind.PROCEED:
        kind = PlanKind(readiness.kind)
    elif action_merge:
        kind = PlanKind.MERGE
    elif action_hand_off:
        kind = PlanKind.HAND_OFF
    elif action_human_merge:
        kind = PlanKind.ESCALATE
    elif conflict_rework or check_failure_rework:
        kind = PlanKind.REQUEST_REWORK
    elif held:
        kind = PlanKind.HOLD
    else:
        kind = PlanKind.NONE

    return MergePlan(
        kind=kind,
        readiness=readiness,
        escalated_merge_hold=escalated_merge_hold,
        should_merge=should_merge,
        action_merge=action_merge,
        action_hand_off=action_hand_off,
        action_human_merge=action_human_merge,
        read_merge_hold=read_merge_hold,
        merge_hold=merge_hold,
        merge_hold_unavailable=merge_hold_unavailable,
        # A re-apply while the label is still reverted must not reset the
        # counter (ADR-0003): the revert is what keeps it climbing.
        reset_failed_attempts_on_hand_off=not adm.mergequeue_label_reverted,
        holds=frozenset(held),
        conflict_rework=conflict_rework,
        check_failure_rework=check_failure_rework,
    )


# --------------------------------------------------------------------------- #
# Stage 5: accounting
# --------------------------------------------------------------------------- #


def _route_detail(
    issue_number: int | None,
    *,
    routed: bool,
    escalated: bool,
    label_error: Mapping[str, Any] | None,
    issue_status_after: str | None,
) -> str:
    if issue_number is None:
        return "no linked issue, cannot route to rework"
    if escalated:
        return "conflict-rework cap exhausted; escalated to a human"
    if routed:
        if label_error:
            outcome = label_error.get("outcome", label_error)
            return f"rework dispatch attempted (label update failed: {outcome})"
        return "rework dispatched"
    if issue_status_after == "rework_requested":
        return "rework already requested"
    return "rework not routed"


def _frozen(data: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(data))


def decide_accounting(
    plan: MergePlan, results: EffectResults, facts: AccountingFacts
) -> Accounting:
    """Failed-attempt counter, alarm, mergequeue stamps and events.

    ``facts`` is gathered inside the state lock after the merge, because
    ``deadline_spent`` must be read after the irreversible effect (C6). A
    hand-off never writes "merged" and never yields ``merge_succeeded``; only
    ``results.merge_output`` does (ADR-0003).
    """
    readiness = plan.readiness
    gate = readiness.gate
    adm = readiness.branch.admission
    cfg = facts.config
    summary = readiness.summary
    can_merge = gate.can_merge
    approved = gate.approved
    issue_number = facts.issue_number
    merged = bool(results.merge_output)

    handoff_failed = bool(cfg.mergequeue_label and results.mergequeue_label_applied is False) or (
        adm.mergequeue_label_reverted and can_merge and not adm.self_revoked_stale_head
    )

    attempts = facts.locked.failed_attempts
    alarm = False
    warning: str | None = None
    events: list[EventSpec] = []
    threshold = cfg.failed_attempt_alarm
    counts = (approved and not can_merge and not readiness.pending_only) or handoff_failed
    if not facts.deadline_spent and counts:
        attempts = facts.locked.failed_attempts + 1
        if threshold > 0:
            attempts = min(attempts, threshold + 1)
        alarm = threshold > 0 and attempts == threshold
        if alarm:
            warning = _alarm_warning(readiness, results, facts, attempts, handoff_failed)
            events.append(
                EventSpec(
                    "merge_failed_attempt_alarm",
                    _frozen(
                        {
                            "pr_number": facts.pr_number,
                            "issue_number": issue_number,
                            "attempts": attempts,
                            "threshold": threshold,
                            "checks_summary": asdict(summary),
                            "mergeable": facts.mergeable,
                            "merge_state_status": facts.merge_state_status,
                            "message": warning,
                        }
                    ),
                )
            )
    elif can_merge and not facts.deadline_spent:
        attempts = 0

    merge_alert_ok = (
        approved
        and can_merge
        and results.merge_output is None
        and not handoff_failed
        and not plan.merge_hold_unavailable
        and not readiness.human_merge_hold
        and not readiness.human_merge_check_unavailable
    )

    status = "merged" if merged else facts.locked.status
    since: str | None = None
    head_sha: str | None = None
    if status == "mergequeue" and facts.live_head_sha:
        if (
            facts.locked.mergequeue_head_sha == facts.live_head_sha
            and facts.locked.mergequeue_since
        ):
            since, head_sha = facts.locked.mergequeue_since, facts.locked.mergequeue_head_sha
        else:
            since, head_sha = facts.now_iso, facts.live_head_sha

    events.append(
        EventSpec(
            "merge_ready",
            _frozen(
                {
                    "pr_number": facts.pr_number,
                    "can_merge": can_merge,
                    "merged": merged,
                    "merge_hold": plan.merge_hold,
                    "merge_hold_check_unavailable": plan.merge_hold_unavailable,
                    "human_merge_hold": readiness.human_merge_hold,
                    "human_merge_check_unavailable": readiness.human_merge_check_unavailable,
                    "cancel_superseded_runs_results": results.cancel_results,
                    "mergequeue_label_applied": results.mergequeue_label_applied,
                    **gate.as_payload(),
                }
            ),
        )
    )
    if merged:
        events.append(
            EventSpec(
                "merge_succeeded",
                _frozen(
                    {
                        "pr_number": facts.pr_number,
                        "issue_number": issue_number,
                        "actor": "fleet",
                        "merge_method": cfg.merge_strategy,
                        "merged_at": results.merged_at,
                    }
                ),
            )
        )

    return Accounting(
        handoff_failed=handoff_failed,
        failed_attempts=attempts,
        stale_base_deferrals=0,
        alarm=alarm,
        warning=warning,
        merge_alert_ok=merge_alert_ok,
        merged=merged,
        pr_status="merged" if merged else None,
        mergequeue_since=since,
        mergequeue_head_sha=head_sha,
        events=tuple(events),
    )


def _alarm_warning(
    readiness: Readiness,
    results: EffectResults,
    facts: AccountingFacts,
    attempts: int,
    handoff_failed: bool,
) -> str:
    """Alarm text; priority: conflict > cross-PR revert > hand-off > failed checks > generic."""
    summary = readiness.summary
    pr_number = facts.pr_number
    issue_number = facts.issue_number
    pass_str = "pass" if attempts == 1 else "passes"
    lead = f"PR #{pr_number} approved but unmergeable for {attempts} {pass_str}: "
    if readiness.branch.merge_conflict:
        detail = _route_detail(
            issue_number,
            routed=results.conflict_routed,
            escalated=results.conflict_escalated,
            label_error=results.rework_label_error,
            issue_status_after=results.issue_status_after,
        )
        return f"{lead}merge conflict — {detail}"
    if readiness.cross_pr_revert_detected:
        detail = _route_detail(
            issue_number,
            routed=results.cross_pr_revert_routed,
            escalated=False,
            label_error=results.rework_label_error,
            issue_status_after=results.issue_status_after,
        )
        return f"{lead}cross-PR revert — {detail}"
    if handoff_failed:
        return (
            f"PR #{pr_number} approved and checks green but the mergequeue "
            f"label {facts.config.mergequeue_label!r} failed to "
            f"apply for {attempts} {pass_str} — never handed off to "
            "Aviator"
        )
    if summary.failed:
        detail = _route_detail(
            issue_number,
            routed=results.check_failure_routed,
            escalated=False,
            label_error=results.rework_label_error,
            issue_status_after=results.issue_status_after,
        )
        failed_str = ", ".join(summary.failed)
        return f"{lead}required check(s) failed ({failed_str}) — {detail}"
    return format_merge_attempt_alarm_message(
        pr_number,
        attempts,
        summary,
        mergeable=facts.mergeable,
        merge_state_status=facts.merge_state_status,
    )
