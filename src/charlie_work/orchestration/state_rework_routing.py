"""Rework-routing delegates for ``OrchestratorApp`` (connected component).

Track 2 Phase B leaf L01 batch 1 (issue #1632, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from typing import Any, Callable
import logging

from charlie_work.checks import CheckSummary
from charlie_work.labels import TransitionOutcome
from charlie_work import rescue as rescue_helpers
import charlie_work.workflow as _wf


def _route_to_rework(
    self,
    pr: dict[str, Any],
    issue_number: int,
    decision: dict[str, Any],
    summary: str,
    event_kind: str,
    extra_payload: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Route an approved PR to rework with a custom summary and event kind.

    Writes a rework prompt, transitions the linked issue to
    ``rework_requested`` (same label set as a non-escalated request_changes),
    and appends the requested state event. The PR's ``review-decision.json``
    is intentionally left untouched so the approved verdict is re-confirmed
    after the worker push moves the head. Any durable review fields already
    present in the PR state are preserved, and callers may pass additional
    keys to record without overwriting those verdict fields.
    """
    pr_number = int(pr["number"])
    self._write_rework_prompt(pr, issue_number, summary)

    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        issue_key = str(issue_number)
        issue_entry = state["issues"].get(issue_key, {})
        state["issues"][issue_key] = {
            **issue_entry,
            "number": issue_number,
            "status": "rework_requested",
            "merge_alert": "OK",
        }
        pr_entry = state["prs"].get(str(pr_number), {})
        state["prs"][str(pr_number)] = {
            **pr_entry,
            "number": pr_number,
            "issue_number": issue_number,
            "status": "rework_requested",
            "decision": pr_entry.get("decision"),
            "reviewed_head_sha": pr_entry.get("reviewed_head_sha"),
            "reviewed_patch_id": pr_entry.get("reviewed_patch_id"),
            **(extra_state or {}),
        }
        payload = {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "base_ref": pr.get("baseRefName"),
            "head_sha": pr.get("headRefOid"),
            "reviewed_head_sha": decision.get("reviewed_head_sha"),
        }
        if extra_payload:
            payload.update(extra_payload)
        state = self._record_event(state, event_kind, payload)
        _wf.save_state(self.paths.state_file, state)

    result = _wf.transition(self.gh, self.config.labels, issue_number, "rework_requested")
    if result.outcome == TransitionOutcome.APPLIED:
        return None
    return {
        "edge": "rework_requested",
        "outcome": result.outcome.value,
        "add_failures": result.add_failures,
        "remove_failures": result.remove_failures,
    }


def _request_check_failure_rework(
    self,
    pr: dict[str, Any],
    issue_number: int,
    decision: dict[str, Any],
    summary: CheckSummary,
    *,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Route an approved PR whose required checks have genuinely failed to rework.

    Called only from merge_ready's approved+alarm-threshold path (issue
    #674), mirroring ``_request_merge_conflict_rework``. Once a PR is
    approved and its head stops moving, loop()'s already_approved fast
    path never calls review() again -- it goes straight to merge_ready --
    so a required check that starts genuinely failing after approval
    (not merely pending/missing/infra_failed/unavailable) previously had
    no path back to rework at all. Debounced to the same
    failed-attempt-alarm threshold as merge-conflict rework so a
    transient failure or the mergequeue's own speculative-merge retry
    gets a chance to self-heal first.
    """
    failed_str = ", ".join(summary.failed)
    text = (
        f"CI failed on required check(s) after approval: {failed_str}. The code "
        "changes are already approved; do not re-litigate the review -- push a fix "
        "for the failing check(s)."
    )
    requested_at = _wf.utc_now()
    merged_extra_state = {"check_failure_rework_requested_at": requested_at}
    if extra_state:
        merged_extra_state.update(extra_state)
    return self._route_to_rework(
        pr,
        issue_number,
        decision,
        text,
        "check_failure_rework_requested",
        extra_payload={
            "check_failure_rework_requested_at": requested_at,
            "failed_checks": list(summary.failed),
        },
        extra_state=merged_extra_state,
    )


def _request_merge_conflict_rework(
    self,
    pr: dict[str, Any],
    issue_number: int,
    decision: dict[str, Any],
    *,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Route a PR with a genuine merge conflict to rework.

    Called from two sites: merge_ready's approved+alarm-threshold path
    (the original use -- ``decision == "approved"`` is enforced by that
    caller, not here) and review()'s janitor gate
    (``_route_janitor_gate_failure_to_rework``), which is decision-
    agnostic -- a conflicting branch needs a rebase regardless of its
    review verdict -- and passes ``extra_state`` to thread its own
    attempt counter through the same state write.
    """
    summary = (
        "The PR branch has a merge conflict with the base branch after a base update. "
        "Merge the base branch into the PR branch, resolve the conflicts, and push."
    )
    if decision.get("decision") == "approved":
        summary += " The code changes are already approved; do not re-litigate the review."
    requested_at = _wf.utc_now()
    merged_extra_state = {"conflict_rework_requested_at": requested_at}
    if extra_state:
        merged_extra_state.update(extra_state)
    return self._route_to_rework(
        pr,
        issue_number,
        decision,
        summary,
        "merge_conflict_rework_requested",
        extra_payload={"conflict_rework_requested_at": requested_at},
        extra_state=merged_extra_state,
    )


def _request_readiness_no_ci_rework(
    self,
    pr: dict[str, Any],
    issue_number: int,
    decision: dict[str, Any],
    missing_checks: tuple[str, ...],
) -> dict[str, Any] | None:
    """Route an approved PR whose required CI checks have not started to rework."""
    summary = (
        "The PR was approved but the required CI checks have not started after the "
        f"configured readiness timeout ({self.config.auto_merge.readiness_no_ci_minutes} minutes). "
        f"Required checks: {', '.join(missing_checks)}. Push an empty commit or amend the head "
        "to re-trigger CI. The code changes are already approved; do not re-litigate the review."
    )
    requested_at = _wf.utc_now()
    return self._route_to_rework(
        pr,
        issue_number,
        decision,
        summary,
        "readiness_no_ci_rework_requested",
        extra_payload={
            "missing_checks": missing_checks,
            "readiness_no_ci_rework_requested_at": requested_at,
        },
        extra_state={
            "readiness_no_ci_rework_requested_at": requested_at,
        },
    )


def _route_janitor_gate_failure_to_rework(
    self,
    pr: dict[str, Any],
    issue_number: int,
    *,
    attempts_key: str,
    max_attempts: int,
    reason: str,
    router: Callable[..., dict[str, Any] | None],
) -> _wf.CommandResult | None:
    """Shared cap/escalation wrapper for janitor-gate rework routing.

    ``router`` is ``_request_merge_conflict_rework`` or
    ``_request_no_op_rework_repair`` -- both ultimately call
    ``_route_to_rework`` and return ``None`` on a clean label transition
    or a ``label_error`` dict on partial failure.

    Returns ``None`` (caller falls through to the passive janitor_blocked
    wait) when a rework for the issue is already pending
    (``rework_requested``, ``dispatched``, or the two-phase-claim crash
    window ``dispatch_pending``): the janitor re-detects the same
    conflict/no-op every pass, and review() is re-invoked every pass, so
    routing -- and burning an attempt -- on each detection would escalate
    a PR whose rework worker simply hasn't run yet within two loop
    passes. Attempts must count completed-but-still-failing rework
    CYCLES, not loop passes and not individual pushes. For
    ``merge_conflict`` the cycle signal is a SETTLED head change: the
    issue must be back in ``rework_requested`` (a live ``dispatched``
    session may push any number of intermediate WIP commits -- burning
    per push would escalate a PR whose worker is actively fixing it) and
    the head must differ from the recorded
    ``<attempts_key>_last_head`` baseline. A truthy baseline is
    required: with no baseline on record (the rework predates this
    bookkeeping, or the head was transiently unavailable at routing
    time) the head is recorded as the new baseline instead of guessing
    that a cycle completed. A qualifying settled change burns one
    attempt (bounding the push-conflicted-heads-forever loop) but does
    not re-route -- the issue is already queued and the worktree layer
    injects the conflict notice at (re)launch. For ``no_op_rework``,
    head unchanged is the detection signal itself, so pending cycles
    are instead bounded by dispatch_rework's dead-session redispatch
    cap.

    Once ``max_attempts`` is exceeded, escalate using the same
    ``transition()`` helper the other escalation call sites use so
    ``agent:human-needed`` actually lands. The review-dispatch attempt-cap
    escalation used to skip this step, writing ``escalated`` to state
    without applying the label edge and leaving the issue without
    ``agent:human-needed`` on GitHub. This call site must not repeat that
    failure mode.
    """
    pr_number = int(pr["number"])
    head_sha = str(pr.get("headRefOid") or "")
    last_head_key = f"{attempts_key}_last_head"
    stall_since_key = f"{attempts_key}_stall_since"
    stall_head_key = f"{attempts_key}_stall_head"
    # The attempt count is read here but persisted under a later, separate
    # lock (this write for the burn path, _route_to_rework's for the
    # routing path) -- a deliberate deviation from the single-lock RMW
    # pattern used elsewhere: two overlapping passes on the same PR can
    # at worst under-count by one, which only DELAYS escalation by one
    # cycle and self-corrects on the next pass. Making the routing path
    # atomic would mean threading a counter callback through
    # _route_to_rework's shared state write; not worth it for that
    # failure direction.
    snapshot = _wf.load_state_locked(self.paths.state_file)
    existing_pr_state = snapshot.get("prs", {}).get(str(pr_number), {})
    issue_state = snapshot.get("issues", {}).get(str(issue_number), {})
    issue_status = issue_state.get("status")
    # Issue #776: this wrapper is called on every pass for as long as the
    # underlying janitor failure persists (review()'s escalated-issue
    # early return re-attempts it every pass; merge_ready()'s conflict
    # trigger no longer pre-excludes "escalated" either, by design, so
    # unrelated remediation can proceed). Once THIS lane's own cap has
    # already escalated the pair, re-running the block below on an
    # unchanged verdict would re-burn attempts_key past the cap, re-fire
    # janitor_rework_escalated, and re-call transition() every single
    # pass forever. Match on the lane-specific escalation_reason (not
    # just issue_status == "escalated") so an escalation from a
    # DIFFERENT lane -- a different attempts_key, or the unrelated
    # watchdog redispatch cap -- does NOT match here and remediation of
    # that unrelated Y still proceeds below. Checking both records
    # covers the PR-only-escalated edge case.
    #
    # Both this lane's cap-exceeded escalation AND its stall escalation
    # (_check_janitor_rework_stall, issue #765/#774) must be recognized
    # here. Without the stall reason, an issue escalated for a stalled
    # rework (nobody working it -- the whole point of that escalation is
    # to get a human to look) would fall through: its status is
    # "escalated", not "rework_requested"/"dispatched"/"dispatch_pending",
    # so `rework_pending` below is False and the function would proceed
    # straight to the attempts-increment/dispatch logic and silently
    # redispatch a fresh rework attempt on the very next pass --
    # defeating the stall escalation the moment it fires.
    #
    # Issue #1461: the check uses ``escalation_reasons_seen`` (an
    # append-only list maintained by ``_escalate_issue``) instead of the
    # single ``escalation_reason`` field. A different lane's escalation
    # clobbers the single field, which used to blind this guard on the
    # next pass -- the lane re-proceeded, re-incremented attempts_key
    # past the cap, and re-fired ``janitor_rework_escalated``. The list
    # is stable across cross-lane clobbers, so the guard reliably
    # recognizes this lane's own prior escalation regardless of what the
    # current single field says.
    current_escalation_reasons = frozenset(
        {f"{attempts_key}_cap_exceeded", f"{attempts_key}_stall_exceeded"}
    )
    issue_seen = frozenset(issue_state.get("escalation_reasons_seen") or [])
    pr_seen = frozenset(existing_pr_state.get("escalation_reasons_seen") or [])
    if current_escalation_reasons & (issue_seen | pr_seen):
        return None
    rework_pending = issue_status in ("rework_requested", "dispatched", "dispatch_pending")
    counted_head = existing_pr_state.get(last_head_key)
    if rework_pending:
        settled_new_conflicted_head = (
            reason == "merge_conflict"
            and issue_status == "rework_requested"
            and bool(head_sha)
            and head_sha != counted_head
        )
        if not settled_new_conflicted_head:
            # Issue #765: "no progress at all" and "progress still in
            # flight" both land here (a live dispatched session's WIP
            # pushes, and a stalled rework_requested queue item with an
            # unmoved head, are indistinguishable from settled_new_
            # conflicted_head's point of view). The cap/rescue check
            # below is unreachable from here -- it only runs once a
            # cycle SETTLES -- so a PR that never settles can wait here
            # forever. _check_janitor_rework_stall separates the two
            # cases, but NOT by issue status alone -- reconcile's
            # issue_active_label_with_open_pr self-heal periodically
            # flips issue_status away from rework_requested without the
            # head moving, so status is only used to decide whether to
            # ADVANCE the clock; only a head change (or escalation)
            # clears it.
            stall_result = self._check_janitor_rework_stall(
                pr_number=pr_number,
                issue_number=issue_number,
                issue_status=issue_status,
                head_sha=head_sha,
                reason=reason,
                attempts_key=attempts_key,
                stall_since_key=stall_since_key,
                stall_head_key=stall_head_key,
                stall_since=existing_pr_state.get(stall_since_key),
                stall_head=existing_pr_state.get(stall_head_key),
            )
            if stall_result is not None:
                return stall_result
            return None
        if not counted_head:
            with _wf.state_lock(self.paths.state_file):
                state = _wf.load_state(self.paths.state_file)
                state["prs"][str(pr_number)] = {
                    **state["prs"].get(str(pr_number), {}),
                    "number": pr_number,
                    "issue_number": issue_number,
                    last_head_key: head_sha,
                    # Defensive: the stall clock is only ever set from
                    # the not-settled branch above, so this should
                    # already be unset -- but a head that was
                    # transiently unavailable (see settled_new_
                    # conflicted_head's bool(head_sha) guard) could have
                    # started the clock before this baseline existed.
                    stall_since_key: None,
                    stall_head_key: None,
                }
                _wf.save_state(self.paths.state_file, state)
            return None
    # Issue #1106: if the previous rework session died at CLI startup
    # (before the worker's first tool action), this is NOT a no-op/conflict
    # rework attempt — requeue without touching the caps.  The cap counters
    # should only count sessions that *ran* and produced no useful change.
    #
    # ``head_settled`` is True only when ``rework_pending`` is True at this
    # point — the only ``rework_pending = True`` path that reaches here is
    # ``settled_new_conflicted_head = True`` (the head moved off the
    # recorded baseline), which means the worker DID push real content
    # before dying.  A startup death with no content change never reaches
    # here via the ``rework_pending = True`` branch (it returns None at the
    # stall check above), so the flag only matters on the
    # ``rework_pending = False`` path — the first detection or a re-detection
    # after the issue status left the pending set.
    head_settled = rework_pending
    if existing_pr_state.get("last_rework_was_startup_death") and not head_settled:
        decision = self._review_decision(pr_number)
        route_extra_state: dict[str, Any] = {
            "last_rework_failure_kind": None,
            "last_rework_was_startup_death": False,
        }
        if head_sha:
            route_extra_state[last_head_key] = head_sha
        label_error = router(pr, issue_number, decision, extra_state=route_extra_state)
        return _wf.CommandResult(
            True,
            f"PR #{pr_number} requeued after startup death "
            f"({existing_pr_state.get('last_rework_failure_kind')}); "
            f"{attempts_key} not incremented",
            {
                "pr": pr_number,
                "issue": issue_number,
                "routed_to_rework": True,
                "rework_reason": reason,
                "startup_death_requeue": True,
                "label_error": label_error,
            },
        )
    attempts = int(existing_pr_state.get(attempts_key, 0)) + 1

    if (
        max_attempts > 0
        and attempts > max_attempts
        and self.config.rescue.enabled
        and not existing_pr_state.get("rescue_attempted")
    ):
        # Rescue tier (issue #555): conflict-rework and no-op-rework caps
        # are both eligible ("cheap model wasn't good enough") causes.
        # Route to a bounded Opus rework via the SAME router this
        # function already uses for the non-cap-exceeded routing path
        # below, instead of escalating — never a second rescue for the
        # same PR (rescue_attempted is durable, cleared only by
        # `charlie unescalate`).
        #
        # Ordering note (reviewed, deliberate): the marker is committed
        # under this lock BEFORE `router(...)` runs its own separate
        # state_lock write below. A crash in that narrow window leaves
        # `rescue_attempted=True` with no rework actually dispatched —
        # the PR's one rescue slot is spent for nothing and the NEXT cap
        # exceedance escalates straight to a human instead of retrying
        # the rescue. This is intentional, not an oversight: the
        # alternative (write the marker AFTER `router()` succeeds) fails
        # the other way — a crash between a successful `router()` write
        # and the marker write leaves the PR eligible for a SECOND
        # rescue attempt on the next cap exceedance, violating the "one
        # rescue per PR" invariant the whole feature is built around
        # (issue #555: "no upward-migrating cost spiral"). Under-rescue
        # (safe, wasteful) is a strictly better failure mode than
        # over-rescue (invariant violation) here, so marker-first is
        # kept. This mirrors this same function's pre-existing,
        # explicitly-accepted two-lock deviation for `attempts_key`
        # itself (see the docstring above and the comment at the
        # `snapshot = load_state_locked(...)` read a few lines up) —
        # not a new risk introduced by the rescue tier, the same
        # "self-corrects, at worst delays by one cycle" tradeoff this
        # function already makes elsewhere. Making this fully atomic
        # would require plumbing an extra_payload-style rescue-marker
        # parameter through `_request_merge_conflict_rework`/
        # `_request_no_op_rework_repair` (both also called from
        # merge_ready's approved+alarm-threshold path with no rescue
        # concept), which is a larger blast-radius change than this
        # fix-forward warrants; revisit only if the wasted-slot case is
        # ever observed live.
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            state["prs"][str(pr_number)] = {
                **state["prs"].get(str(pr_number), {}),
                "number": pr_number,
                "issue_number": issue_number,
                attempts_key: attempts,
                **rescue_helpers.build_rescue_dataclass_kwargs(reason),
                "rescue_dispatched_at": _wf.utc_now(),
                # Issue #1106: clear startup-death flags.
                "last_rework_failure_kind": None,
                "last_rework_was_startup_death": False,
            }
            state = self._record_event(
                state,
                "rescue_dispatched",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "cause": reason,
                    "attempts": attempts,
                },
            )
            _wf.save_state(self.paths.state_file, state)
        decision = self._review_decision(pr_number)
        route_extra_state: dict[str, Any] = {attempts_key: attempts}
        if head_sha:
            route_extra_state[last_head_key] = head_sha
        # Issue #1106: clear startup-death flags.
        route_extra_state["last_rework_failure_kind"] = None
        route_extra_state["last_rework_was_startup_death"] = False
        label_error = router(pr, issue_number, decision, extra_state=route_extra_state)
        return _wf.CommandResult(
            True,
            f"PR #{pr_number} janitor {reason} rework cap exceeded "
            f"({attempts}/{max_attempts}); rescue tier dispatched instead of escalating",
            {
                "pr": pr_number,
                "issue": issue_number,
                "janitor_ok": False,
                "routed_to_rework": True,
                "rescue_dispatched": True,
                "rework_reason": reason,
                "label_error": label_error,
            },
        )

    if max_attempts > 0 and attempts > max_attempts:
        # Issue #776: record a structured, lane-specific escalation reason
        # (distinct from the generic "redispatch_cap_exceeded" other
        # escalation sites use for the unrelated worker-liveness
        # redispatch cap) so a human -- and any future reason-scoped
        # guard -- can tell THIS lane's cap is what's exhausted, as
        # opposed to an unrelated lane having escalated the issue. This
        # was the one escalation call site in the codebase that recorded
        # no reason at all.
        escalation_reason = f"{attempts_key}_cap_exceeded"
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            worker_launched = _wf._worker_launched_before_cap_escalation(state, issue_number)
            state = _wf._escalate_issue(
                state,
                issue_number,
                reason=escalation_reason,
                reason_class="mechanical",
                pr_number=pr_number,
                pr_extra=_wf._cap_escalation_pr_extra(attempts_key, attempts, worker_launched),
            )
            state = self._record_event(
                state,
                "janitor_rework_escalated",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "reason": reason,
                    "escalation_reason": escalation_reason,
                    "attempts": attempts,
                    "worker_launched": worker_launched,
                },
            )
            _wf.save_state(self.paths.state_file, state)
        edge = _wf._escalation_edge("escalated", "mechanical")
        result = _wf.transition(self.gh, self.config.labels, issue_number, edge)
        label_error = None
        if result.outcome != TransitionOutcome.APPLIED:
            label_error = {
                "edge": edge,
                "outcome": result.outcome.value,
                "add_failures": result.add_failures,
                "remove_failures": result.remove_failures,
            }
        return _wf.CommandResult(
            False,
            f"PR #{pr_number} janitor {reason} rework cap exceeded "
            f"({attempts}/{max_attempts}); escalated",
            {
                "pr": pr_number,
                "issue": issue_number,
                "janitor_ok": False,
                "escalated": True,
                "label_error": label_error,
            },
        )

    if rework_pending:
        # merge_conflict with a settled head that moved off the recorded
        # baseline: a rework cycle completed without clearing the
        # conflict. Burn the attempt so cycles stay bounded, but the
        # issue is already queued for rework -- record and fall through
        # to the passive janitor_blocked wait.
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            state["prs"][str(pr_number)] = {
                **state["prs"].get(str(pr_number), {}),
                "number": pr_number,
                "issue_number": issue_number,
                attempts_key: attempts,
                last_head_key: head_sha,
                # Issue #765: the head moved -- real progress -- so any
                # stall clock started while it was stuck no longer
                # applies. A later re-stall must count from zero, not
                # from a timestamp accumulated during unrelated history.
                stall_since_key: None,
                stall_head_key: None,
                # Issue #1106: clear startup-death flags — the head
                # settled, so the previous session's death (if any) is
                # now accounted for by this attempt.
                "last_rework_failure_kind": None,
                "last_rework_was_startup_death": False,
            }
            state = self._record_event(
                state,
                "janitor_rework_cycle_failed",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "reason": reason,
                    "attempts": attempts,
                    "head_sha": head_sha,
                },
            )
            _wf.save_state(self.paths.state_file, state)
        return None

    decision = self._review_decision(pr_number)
    # Issue #765: no unconditional stall_since_key/stall_head_key clear
    # here. A genuinely fresh rework request has no clock running yet, so
    # omitting the keys is a no-op; a reconcile-induced re-route (the
    # head is unchanged, only issue_status flipped and flipped back) must
    # NOT lose already-accumulated stall time, since that would let
    # reconcile's unrelated label self-heal silently defeat this bound.
    route_extra_state: dict[str, Any] = {attempts_key: attempts}
    if head_sha:
        route_extra_state[last_head_key] = head_sha
    # Issue #1106: clear startup-death flags — this attempt is being
    # counted, so the previous session's death is now accounted for.
    route_extra_state["last_rework_failure_kind"] = None
    route_extra_state["last_rework_was_startup_death"] = False
    label_error = router(pr, issue_number, decision, extra_state=route_extra_state)
    return _wf.CommandResult(
        True,
        f"PR #{pr_number} routed to rework ({reason}, attempt {attempts}/{max_attempts})",
        {
            "pr": pr_number,
            "issue": issue_number,
            "routed_to_rework": True,
            "rework_reason": reason,
            "label_error": label_error,
        },
    )


def _reroute_stranded_request_changes(
    self,
    pr: dict[str, Any],
    issue_number: int,
    decision: dict[str, Any],
) -> dict[str, Any] | None:
    """Re-drive an actionable, already-recorded ``request_changes`` verdict
    to ``rework_requested`` when the linked issue's status never reflects it.

    Issue #784 AC-8 (Case 2). ``record_review`` durably decides and
    persists the rework-vs-escalate outcome (``decision["escalated"]``,
    computed against ``max_rework_cycles`` and persisted to
    ``request_changes_count`` BEFORE any GitHub label mutation --
    ``record_review`` ~9355-9396) and applies the matching issue-status
    transition in that same call. If that label transition silently
    failed, or the issue status was later clobbered by an independent
    bug (e.g. issue #789's reconcile one-way "closed" gate), the issue
    is stranded: a real, actionable verdict sits on disk with nothing
    routing it to a rework worker (``dispatch_rework`` only ever selects
    issues already at ``status == "rework_requested"``).

    This never re-decides escalation -- the caller only invokes it when
    ``decision.get("escalated")`` is falsy, i.e. record_review already
    decided this verdict is within the rework-cycle budget -- it only
    re-applies that SAME target via the SAME generic entry point
    (``_route_to_rework``) that every other "discovered late, needs
    rework" lane in this file uses (cross-PR-revert, merge-conflict,
    check-failure).

    Idempotent: guarded by ``_REWORK_ALREADY_ROUTED_STATUSES`` -- the
    exact status set ``merge_ready``'s cross-PR-revert gate uses -- so a
    second pass over unchanged state (issue already ``rework_requested``,
    or otherwise mid-flight/escalated/blocked) is a no-op. This is also
    why the fix lives here rather than in ``reconcile.py``: routing to
    rework can dispatch a worker, and reconcile's repair pass must never
    synthesize ``rework_requested`` (reconcile.py's own invariant) --
    only a dispatch-context caller may complete this transition.

    Issue #1123: the restorer must never re-activate a CLOSED GitHub
    issue. Issue state is the source of truth (state-in-labels
    invariant), and reconcile's ``state_active_status_issue_closed``
    owns the terminal "closed" status for an issue GitHub reports as
    CLOSED. Without this guard the restorer flips the status back to
    ``rework_requested`` every pass, the no-op rework cap escalates it,
    and reconcile flips it back to "closed" -- a perpetual three-lane
    loop. A closed issue's open PR surfaces once as drift via
    reconcile's ``state_active_status_issue_closed`` for human
    adjudication instead of re-entering the rework state machine. If
    the issue fetch itself fails (transient GitHubError, missing from
    the snapshot), defer to the next pass rather than re-activating
    state that may belong to a closed issue -- the stranded repair is
    a best-effort lane, not a correctness-critical one.

    Idempotency (review rework): the closed-issue skip must fire once
    per stranded PR, not on every ``review_queue()`` pass. ``"closed"``
    cannot be added to ``_REWORK_ALREADY_ROUTED_STATUSES`` because the
    #789 repair path depends on the restorer re-activating an issue
    whose status was clobbered to ``"closed"`` by a reconcile bug while
    the issue is still OPEN on GitHub -- a shared early-return on
    ``"closed"`` would suppress that repair. Instead a per-issue
    ``stranded_skip_closed`` marker in ``state["issues"][n]`` records
    that the restorer has already confirmed the issue is CLOSED and
    skipped. The marker is checked only when ``status == "closed"`` so
    the #789 path (status ``"closed"`` but issue OPEN, marker absent)
    still fetches and re-activates. When the skip fires for an issue
    whose status is not yet ``"closed"`` (e.g. ``"reviewing"``), the
    restorer converges the status to ``"closed"`` and strips active
    labels -- mirroring reconcile's ``state_active_status_issue_closed``
    -- so the next pass sees ``status == "closed"`` + marker set and
    short-circuits without a ``gh.issue_view()`` call or a duplicate
    event. Reconcile preserves unknown keys in issue entries (spread
    copy), so the marker survives reconcile sweeps; if reconcile
    normalizes the status away from ``"closed"`` (issue re-opened),
    the marker check no longer applies and the restorer re-evaluates.
    """
    state = _wf.load_state_locked(self.paths.state_file)
    issue_entry = state.get("issues", {}).get(str(issue_number), {})
    issue_status = issue_entry.get("status")
    if issue_status in _wf._REWORK_ALREADY_ROUTED_STATUSES:
        return None
    # Local idempotency guard for the closed-issue skip (see docstring):
    # once the restorer has confirmed a CLOSED issue and recorded the
    # marker, short-circuit without re-fetching or re-emitting. Gated on
    # status == "closed" so the #789 repair (status clobbered to "closed"
    # for an OPEN issue, marker absent) still proceeds.
    if issue_status == "closed" and issue_entry.get("stranded_skip_closed"):
        return None
    try:
        issue = self.gh.issue_view(issue_number)
    except Exception:
        logging.getLogger(__name__).warning(
            "stranded_request_changes restorer for issue %s deferred "
            "(GitHub issue fetch failed); will retry next pass",
            issue_number,
            exc_info=True,
        )
        return None
    if str(issue.get("state") or "OPEN").upper() == "CLOSED":
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            issues = state.setdefault("issues", {})
            issue_entry = issues.get(str(issue_number), {})
            issue_entry["stranded_skip_closed"] = True
            if issue_entry.get("status") != "closed":
                issue_entry["status"] = "closed"
            issues[str(issue_number)] = issue_entry
            state = self._record_event(
                state,
                "stranded_request_changes_skipped_issue_closed",
                {
                    "pr_number": int(pr["number"]),
                    "issue_number": issue_number,
                    "head_sha": pr.get("headRefOid"),
                },
            )
            _wf.save_state(self.paths.state_file, state)
        # Strip active labels from the closed issue, mirroring
        # reconcile's state_active_status_issue_closed: setting
        # status to "closed" here would prevent that drift kind from
        # firing (it requires status in ACTIVE_STATE_STATUSES), so the
        # restorer must do the label cleanup itself to avoid leaving
        # active labels on a finalized issue.
        active_labels = _wf.label_names(issue) & self.config.labels.active
        for label in sorted(active_labels):
            self.gh.remove_issue_label(issue_number, label)
        return None
    return self._route_to_rework(
        pr,
        issue_number,
        decision,
        decision.get("summary") or "",
        "stranded_request_changes_rework_requested",
    )
