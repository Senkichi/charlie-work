"""The rules that decide what a recorded Verdict does (issue #3103).

Step 2 of 2 of the single Verdict module (step 1 is :mod:`charlie_work.verdict_recording`,
issue #3102). ``record_review`` (``orchestration/state_record_review.py``, the PR path) and
``record_local_review`` (``orchestration/local_lanes.py``, the Local path) each used to carry
their own copy of these rules; this module is the one copy:

* the rework counter and its ``head_advanced`` gating, the Rework cap check against
  ``max_rework_cycles`` and the Rescue tier (:func:`decide_rework`);
* the escalation reason (:func:`escalation_reason`);
* the PR-record merge, the issue status transitions for approve, request changes and
  blocked (mechanical versus judgment escalation, ``clear_escalation``, the no-op
  checkpoint, the ``worker_pid`` pops) and the Events (:func:`record_verdict`);
* the label edge chosen, the issue-closed skip and the ``label_error`` it reports
  (:func:`choose_label_edge`, :func:`label_error`). The Event for a rework label skipped
  because the issue is closed is emitted by each caller at the point it takes that branch,
  inside its own escalation-performing function.

Each declared difference is an explicit input, never a mode flag. The Rescue tier is
``rescue_enabled``: the PR path passes the configured value and the Local path passes
``False``. Extra PR-record and Event fields (``reviewed_body_sha256`` and
``external_findings_count`` on the PR path, ``local: True`` on the Local path) are passed
in as mappings, and ``utc_now`` is handed in because tests replace it on
``charlie_work.workflow``.

Writes. The only write this module makes is ``write_gate.record_event``. It never saves
``state.json`` and never calls the label seam: each path keeps its own lock, load and save
and its own transition call (``_wf.save_state`` / ``_wf.transition`` on the PR path,
``write_gate.save_state`` / ``write_gate.transition`` on the Local path), because the two
paths behave differently under a dry run and that difference is preserved, not decided
here. The callers only sequence: they branch on fields of the outcome this module returns
and never recompute a rule.

Like :mod:`charlie_work.verdict_recording`, this module imports nothing from
:mod:`charlie_work.workflow` at import time (it would be circular); names defined there
are resolved at call time through :func:`_workflow`, so every existing
``charlie_work.workflow.*`` monkeypatch site keeps intercepting. It lives at the package
root, not under ``orchestration/``, because every top-level function defined there is
auto-installed on ``OrchestratorApp``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from charlie_work import rescue as rescue_helpers
from charlie_work.labels import TransitionOutcome


def _workflow() -> Any:
    # ``charlie_work.workflow`` imports the orchestration delegates, which import
    # this module -- resolve it at call time.
    import charlie_work.workflow as wf

    return wf


@dataclass(frozen=True)
class ReworkOutcome:
    """What the rework counter and the cap decided for one verdict."""

    escalated: bool
    rescue_dispatched: bool
    request_changes_count: int
    # The summary the rework brief carries: the Rescue brief when the Rescue tier was
    # dispatched, ``summary_text`` otherwise, ``""`` when no rework is routed.
    rework_summary: str


@dataclass(frozen=True)
class LabelEdge:
    """The label edge a verdict takes, or the reason it takes none."""

    skip_issue_closed: bool
    edge: str | None
    # The name ``label_error`` reports. It is the edge itself except for a blocked
    # verdict, which reports the literal ``"blocked"``.
    error_edge: str | None


def decide_rework(
    *,
    decision: str,
    reviewed_head_sha: str,
    prior_head_sha: str | None,
    prior_count: int,
    max_rework_cycles: int,
    rescue_enabled: bool,
    rescue_attempted: bool,
    summary_text: str,
) -> ReworkOutcome:
    """Count the rework cycle, apply the Rework cap and decide the Rescue tier."""
    # Durable per-PR rework counter — NOT derived from the global events
    # log, which append_event truncates to the last 200 entries: on a busy
    # repo that eviction silently reset the count and defeated the cap
    # (a PR could rework forever instead of escalating to a human).
    request_changes_count = prior_count
    escalated = False
    rescue_dispatched = False
    # str, not str | None: it is only ever consumed by the rework-prompt writer, whose
    # dispatch_note parameter is a bare ``str`` (issue #782). Unreachable as the empty
    # default on every path that writes a brief: it is rebound whenever ``not escalated``.
    rework_summary: str = ""
    if decision == "request_changes":
        # Only count a rework cycle when the PR head has actually advanced.
        # If the head is unchanged, the prior cycle's attempt was never
        # delivered (e.g., worker died orphaned), so re-issuing
        # request_changes should not consume the escalation budget. See
        # issue #208.
        #
        # Issue #1210: the head_advanced guard previously protected only
        # the counter increment below — the escalation check ran
        # unconditionally and fired on an at-cap verdict even when the
        # head was unchanged. Gate the escalation evaluation (and the
        # rescue tier, which is itself gated on cap exceedance) on
        # head_advanced as well, so an at-cap verdict on an unchanged
        # head re-issues request_changes without escalating, mirroring
        # what already happens below-cap. Keep the existing behavior for
        # advanced heads.
        head_advanced = reviewed_head_sha != prior_head_sha
        if head_advanced:
            # Rework cap: past max_rework_cycles the evidence says
            # iteration thrashes (wrong brief or unimplementable
            # criteria) — escalate to a human instead of dispatching
            # another cycle.
            escalated = request_changes_count >= max_rework_cycles
            # Rescue tier (issue #555): a cap exceedance here is one of
            # the three verdict-driven ("cheap model wasn't good
            # enough") causes the rescue tier is gated on. If enabled
            # and this PR has not already spent its one rescue attempt,
            # route to a bounded Opus rework instead of escalating —
            # never a second rescue for the same PR (rescue_attempted
            # is durable, cleared only by `charlie unescalate`). The
            # Local path passes ``rescue_enabled=False``: at the cap a
            # Local record escalates directly.
            if escalated and rescue_enabled and not rescue_attempted:
                escalated = False
                rescue_dispatched = True
            if not escalated:
                request_changes_count += 1
        if not escalated:
            rework_summary = (
                rescue_helpers.build_rescue_rework_summary("rework_cycle_cap", summary_text)
                if rescue_dispatched
                else summary_text
            )
    return ReworkOutcome(escalated, rescue_dispatched, request_changes_count, rework_summary)


def escalation_reason(decision: str, escalated: bool) -> str | None:
    """The ``escalation_reason`` a verdict stamps on the PR record, if any."""
    return (
        "max_rework_cycles_exceeded"
        if (decision == "request_changes" and escalated)
        else ("review_blocked" if decision == "blocked" else None)
    )


def record_verdict(
    write_gate: Any,
    state: dict[str, Any],
    *,
    pr_number: int,
    issue_number: int | None,
    decision: str,
    outcome: ReworkOutcome,
    pr_state: dict[str, Any],
    decision_path: str,
    reviewed_head_sha: str,
    reviewed_patch_id: str,
    verdict_provenance: str,
    session_metrics: dict[str, Any] | None,
    summary_text: str,
    required_changes: list[str],
    findings_channel: str | None,
    reclassified_human_call: tuple[str, str] | None,
    record_extras: Mapping[str, Any],
    event_extras: Mapping[str, Any],
    utc_now: Callable[[], str],
) -> dict[str, Any]:
    """Merge the verdict into ``state`` and emit its Events; the caller saves.

    Merge-update (never in-place assignment): the caller persists BEFORE any GitHub label
    mutation, so a label-write failure or crash must not desync the durable
    decision/counter from what actually happened. Order: the PR record, the issue rules,
    then the Events. ``issue_number`` is ``None`` only on the PR path, when the PR links no
    issue: the issue rules are then skipped. ``record_extras`` and ``event_extras`` carry
    the fields only one path has.
    """
    escalated = outcome.escalated
    pr_escalation_reason = escalation_reason(decision, escalated)
    rescue_fields = (
        rescue_helpers.build_rescue_dataclass_kwargs("rework_cycle_cap")
        if outcome.rescue_dispatched
        else {}
    )
    if outcome.rescue_dispatched:
        rescue_fields["rescue_dispatched_at"] = utc_now()
    state["prs"][str(pr_number)] = {
        **pr_state,
        "number": pr_number,
        "issue_number": issue_number,
        "decision": decision,
        "decision_path": decision_path,
        "reviewed_head_sha": reviewed_head_sha,
        "reviewed_patch_id": reviewed_patch_id,
        **record_extras,
        "carried_forward_from": [],
        "request_changes_count": outcome.request_changes_count,
        "status": "escalated" if escalated else decision,
        "consecutive_failed_merge_attempts": 0,
        **rescue_fields,
        # The reviewer agent has recorded its verdict. The hub no longer
        # needs to treat this PR as having an in-flight reviewer; the next
        # stale-head review-queue entry will re-dispatch cleanly.
        "review_dispatch_status": "review_dispatch_completed",
        "reviewer_pid": None,
        "reviewer_process_start_time": None,
        # Reset the dispatch attempt counter: a verdict was produced, so
        # the PR is not stuck. If the head later advances and triggers a
        # new review cycle, the counter starts fresh.
        "review_dispatch_attempt_count": 0,
        # Reset the unreadable-log streak: a verdict proves the
        # reviewer pipeline is healthy, so any prior
        # persistent-unreadable condition is resolved (issue #1069).
        "review_log_unreadable_streak": 0,
        "review_api_error_streak": 0,  # issue #1808
        # Issue #1439: a recorded verdict ends the turn-limit miss
        # streak -- the reviewer reached a conclusion, so any prior
        # turn-limit deaths no longer count toward the cap-aware
        # backstop on a future review cycle.
        "review_turn_limit_miss_streak": 0,
        # Reviewer session token/cost telemetry (best-effort): merge-update
        # so a call without metrics (e.g. a manual `charlie verdict`)
        # never clobbers metrics recorded by an earlier automated reap.
        "review_session_metrics": (
            session_metrics
            if session_metrics is not None
            else pr_state.get("review_session_metrics")
        ),
        **(
            {"escalation_reason": pr_escalation_reason} if pr_escalation_reason is not None else {}
        ),
    }
    wf = _workflow()
    # Update the linked issue's status to reconcile out of rework_requested: the previous
    # worker session is definitionally finished, so the issue status must reflect the
    # actual decision. This prevents state-driven dispatch_rework from selecting
    # approved/blocked PRs for duplicate work. ``issue_number`` is ``None`` only on the PR
    # path, when the PR links no issue.
    if issue_number is not None:
        if decision == "request_changes":
            if not escalated:
                issue_entry = {
                    **(state["issues"].get(str(issue_number)) or {}),
                    "number": issue_number,
                    "status": "rework_requested",
                    "merge_alert": "OK",
                }
                wf.clear_escalation(issue_entry)
                wf.clear_escalation_on_issue_prs(state, issue_number)
                # Issue #1784 (job-cannon #1320 follow-up): the reset below used to gate on
                # ``head_advanced`` (a SHA comparison) alone, and its comment claimed
                # record_review only ever reaches this branch on content a reviewer actually
                # read. Neither half of that held: ``ci_gate_auto_reject``/
                # ``test_adequacy_auto_reject`` short-circuits (and the first-review case,
                # where ``reviewed_head_sha`` starts as None) reach here with no reviewer
                # involved at all, and a pure sync-merge can move the head SHA with zero
                # content change. Reached by a CI-red short-circuit specifically, that
                # combination made the cap this reset exists to protect UNREACHABLE for
                # exactly the stuck-worker population it is meant to catch (every cycle's
                # reset firing on a no-op sync-merge).
                #
                # The actual predicate the code enforces now: a *patch-id* advance (unlike a
                # SHA, immune to a sync-merge that changes nothing but the merge-base) from a
                # verdict whose ``verdict_provenance`` names a mechanism that read the diff at
                # all (``NO_OP_RESET_PROVENANCES`` -- see its docstring for why each excluded
                # value is excluded). That is evidence the issue actually moved forward since
                # the checkpoint, so the no-op/death-loop caps' windowed reads
                # (``_windowed_redispatch_at``, ``_windowed_worker_death_at``) should stop
                # counting redispatches/deaths from before it.
                #
                # This does NOT truncate ``redispatch_at``/``worker_death_at`` themselves
                # (finding 2 of the #1784 review): those two arrays are shared inputs to five
                # independent caps across ``state_dispatch_rework.py`` and
                # ``dead_worker_reap.py`` -- e.g. the plain redispatch cap and the
                # worker-death-loop cap -- not an issue-level no-op counter this function
                # alone owns. Clearing them here would silently reset all five. Instead,
                # stamp a checkpoint timestamp that ``_windowed_redispatch_at``/
                # ``_windowed_worker_death_at`` apply as a floor on top of the ordinary
                # rolling window, so every consumer still sees its own full history up to
                # the checkpoint -- only the no-op-cap's "counts from before this review no
                # longer describe the issue's CURRENT stall state" no longer needs the data
                # destroyed to say so. Reads the PRE-merge record: ``pr_state``.
                prior_patch_id = pr_state.get("reviewed_patch_id")
                content_advanced = bool(reviewed_patch_id) and reviewed_patch_id != prior_patch_id
                if content_advanced and verdict_provenance in wf.NO_OP_RESET_PROVENANCES:
                    issue_entry["no_op_checkpoint_at"] = utc_now()
                state["issues"][str(issue_number)] = issue_entry
            else:
                # Clear rework_requested status when escalated to prevent selection.
                # Issue #783: rework-cycle cap reached is a process limit, not a judgment
                # call -- mechanical.
                state = wf._escalate_issue(
                    state,
                    issue_number,
                    reason="max_rework_cycles_exceeded",
                    reason_class="mechanical",
                )
        elif decision == "approved":
            issue_entry = {
                **(state["issues"].get(str(issue_number)) or {}),
                "number": issue_number,
                "status": "approved",
                "merge_alert": "OK",
            }
            wf.clear_escalation(issue_entry)
            wf.clear_escalation_on_issue_prs(state, issue_number)
            state["issues"][str(issue_number)] = issue_entry
            # Clear worker PID when issue is approved (worker is done)
            state["issues"][str(issue_number)].pop("worker_pid", None)
            state["issues"][str(issue_number)].pop("worker_process_start_time", None)
        elif decision == "blocked":
            # Issue #783: an explicit "blocked" review decision is a human product/security
            # judgment call, never auto-cleared.
            state = wf._escalate_issue(
                state,
                issue_number,
                reason="review_blocked",
                reason_class="judgment",
                status="blocked",
            )
            # Clear worker PID when issue is blocked (worker is done)
            state["issues"][str(issue_number)].pop("worker_pid", None)
            state["issues"][str(issue_number)].pop("worker_process_start_time", None)
    event_payload: dict[str, Any] = {
        "pr_number": pr_number,
        "issue_number": issue_number,
        "decision": decision,
        "escalated": escalated,
        # Issue #1265: top-level and unconditional, distinct from (and a sibling of, not
        # nested inside) the pre-existing session_metrics.verdict_source added
        # conditionally below -- that field means "which parser found the fenced verdict
        # block"; this one means "which mechanism produced the verdict at all".
        "verdict_provenance": verdict_provenance,
        # Issue #1268 (W11), item 2: previously never assigned at all (confirmed live: 0
        # non-null across every existing record_review row) -- events.db carried no
        # verdict text. These are an observability-only, size-bounded copy; the round-K
        # archive (`rounds/round-K/review-decision.json`) is the untruncated source of
        # truth W13 reads from. ``required_changes`` is a list on the decision payload, so
        # it is newline-joined into a single string before the same byte-budget
        # truncation applies to it. ``str(item)`` guards every assignment site of
        # ``required_changes`` -- most are already ``list[str]``, but the #999
        # external-findings path is never itself read as strings elsewhere, so this join
        # must not assume str elements there.
        "summary": wf._truncate_for_event(summary_text),
        "required_changes": wf._truncate_for_event(
            "\n".join(str(item) for item in required_changes)
        ),
    }
    if session_metrics is not None:
        event_payload["session_metrics"] = session_metrics
    if findings_channel is not None:
        event_payload["findings_channel"] = findings_channel
    event_payload.update(event_extras)
    state = write_gate.record_event(state, "record_review", event_payload)
    if reclassified_human_call is not None:
        # Issue #1642: separate event so the reclassification is independently queryable
        # -- the operator-queue entry, review packets, and post-incident audits need the
        # matched finding without re-deriving it from the verdict text.
        matched_item, matched_marker = reclassified_human_call
        state = write_gate.record_event(
            state,
            "review_decision_reclassified_blocked",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "original_decision": "request_changes",
                "matched_item": matched_item,
                "matched_marker": matched_marker,
            },
        )
    if findings_channel == "vacuous":
        # Distinct from the general "record_review" event (issue #792): this is the signal
        # that a request_changes/blocked verdict arrived with nothing to act on -- neither
        # an itemized required_changes nor derivable prose -- so it can be
        # queried/alerted on independently of every other verdict.
        state = write_gate.record_event(
            state,
            "required_changes_vacuous",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "decision": decision,
            },
        )
    if outcome.rescue_dispatched:
        state = write_gate.record_event(
            state,
            "rescue_dispatched",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "cause": "rework_cycle_cap",
            },
        )
    return state


def choose_label_edge(
    gh: Any, issue_number: int, *, decision: str, escalated: bool
) -> LabelEdge | None:
    """The label edge the verdict takes (``None`` for a decision with no edge)."""
    wf = _workflow()
    if decision == "request_changes":
        # Issue #1131: never route rework onto a CLOSED issue. The observed race applied
        # ``agent:needs-rework`` to an issue that was already closed by the merge this
        # verdict chased, polluting roll-call/metrics and requiring a manual label strip.
        # GitHub's issue ``state`` is the authoritative closed signal (state.json's
        # workflow ``status`` can lag or be clobbered). Fetch is best-effort: on lookup
        # failure, proceed with the transition (label application is already best-effort
        # and reported, not fatal) -- the same fail-open posture as the
        # stranded-request_changes restorer's issue_view call. A terminal PR already
        # refuses the whole call (fix #1131); this guard covers the residual case where
        # the PR is still OPEN but its linked issue is independently CLOSED.
        issue_is_closed = False
        try:
            linked_issue = gh.issue_view(issue_number)
            issue_is_closed = str(linked_issue.get("state") or "OPEN").upper() == "CLOSED"
        except Exception:
            pass
        if issue_is_closed:
            return LabelEdge(True, None, None)
        # Issue #1266: escalated here is always max_rework_cycles_exceeded (mechanical).
        target = (
            wf._escalation_edge("escalated", "mechanical") if escalated else "rework_requested"
        )
        return LabelEdge(False, target, target)
    if decision == "blocked":
        # Issue #1266: "blocked" has no operator-queue counterpart -- a reviewer-blocked
        # verdict is judgment-only by construction -- so this is an inert pass-through,
        # kept for uniformity with every other escalation-label call site now routing
        # through the same helper.
        return LabelEdge(False, wf._escalation_edge("blocked", "judgment"), "blocked")
    if decision == "approved":
        return LabelEdge(False, "review_approved", "review_approved")
    return None


def label_error(edge: str | None, result: Any) -> dict[str, Any] | None:
    """The ``label_error`` for a transition result, ``None`` when it applied."""
    if result.outcome == TransitionOutcome.APPLIED:
        return None
    return {
        "edge": edge,
        "outcome": result.outcome.value,
        "add_failures": result.add_failures,
        "remove_failures": result.remove_failures,
    }
