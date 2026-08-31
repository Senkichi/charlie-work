"""Field sets cleared when a PR/issue is re-armed after escalation.

Pure data, no I/O. Two doors consume these: the operator command
``OrchestratorApp.unescalate`` (``UNESCALATE_*_RESET_FIELDS``) and the
automated ``_maybe_deescalate_mechanical`` sweep
(``REWORK_BUDGET_RESET_BY_ESCALATION_REASON``). They must agree on each rework
lane's companion set -- counter, ``_last_head`` baseline, and the
``_stall_since``/``_stall_head`` clock -- or a re-arm leaves one of them behind
and the janitor re-escalates on its first pass (PR #1449, stale stall clock).
``tests/test_fix_unescalate.py`` derives that agreement from the map.

Extracted from ``workflow.py`` (file-size ratchet, #1442); ``OrchestratorApp``
keeps the private class-attribute aliases so call sites are unchanged.
"""

from __future__ import annotations

# PR-record bookkeeping that must not survive an operator re-arm: attempt
# counters and caches that would otherwise instantly re-escalate the PR
# (counters at cap) or feed the pipeline frozen pre-escalation data
# (janitor/CI caches — pr-lifecycle.md: escalated PRs freeze their cached
# janitor state forever, e.g. #548 showing "Tests pending" 12h after the
# checks passed).
UNESCALATE_PR_RESET_FIELDS = (
    "review_dispatch_attempt_count",
    # Issue #1351: companion baseline to review_dispatch_attempt_count.
    # Cleared on re-arm alongside the counter so the next review() for the
    # same head starts a fresh dispatch cycle (counter is 0 either way, but
    # this keeps the pair consistent with the other _last_head baselines).
    "review_dispatch_attempt_last_head",
    "review_log_unreadable_streak",
    # Issue #1439: turn-limit miss streak must not survive a re-arm, or
    # the cap-aware backstop would re-escalate instantly on the next
    # turn-limit death.
    "review_turn_limit_miss_streak",
    "request_changes_count",
    "conflict_rework_attempts",
    "conflict_rework_attempts_last_head",
    # The stall clock (_check_janitor_rework_stall, issue #765) is the
    # third companion of each rework lane, alongside the counter and the
    # _last_head baseline. Only a head change or the stall escalation
    # itself clears it -- a cap escalation does not -- so a clock started
    # during the exhausted episode survives into the re-arm. The janitor
    # then reads the stale start on its first pass and fires
    # ``*_stall_exceeded`` at attempt 1 (PR #1449: stall_since from the
    # day before, 1927 "stalled" minutes, re-escalated 13 min after
    # ``charlie unescalate``). _REWORK_BUDGET_RESET_BY_ESCALATION_REASON
    # already resets all three companions on auto de-escalation; the
    # operator door must reset the same set (tests/test_fix_unescalate.py
    # derives the required companions from that map).
    "conflict_rework_attempts_stall_since",
    "conflict_rework_attempts_stall_head",
    "no_op_rework_attempts",
    "no_op_rework_attempts_last_head",
    "no_op_rework_attempts_stall_since",
    "no_op_rework_attempts_stall_head",
    "review_dispatch_status",
    "review_dispatch_failed_at",
    "review_dispatch_pending_at",
    "review_dispatched_at",
    "reviewer_pid",
    "reviewer_process_start_time",
    "review_turn_limit_summary_posted",
    "review_miss_summary_posted",
    "janitor_ok",
    "janitor_failures",
    "janitor_warnings",
    # Same frozen-cache hazard as janitor_ok/janitor_failures above: once
    # a head is flagged never-created, the dedup marker would otherwise
    # silently suppress a fresh event even after an operator re-arms the
    # PR and the same head is still stuck.
    "ci_run_never_created_head",
    "escalation_reason",
    # Issue #1461: clear the append-only escalation history so a re-arm
    # gives every lane a genuinely fresh dedup slate.
    "escalation_reasons_seen",
    "label_error",
    # Issue #1099: the per-head cross-family regeneration record used to hold
    # both spent-attempt budgets, and leaving it behind made the re-arm inert
    # (loop() read the spent counters and parked the PR again on the very
    # next pass). The auto-gate cross-family subsystem that wrote and read
    # this field was deleted along with cross_family.py (role-config phase 2,
    # track A); nothing produces or consumes this key anymore. The entry
    # stays so a re-arm on an old on-disk record still clears the stale key
    # rather than leaving it behind forever.
    "cross_family_regen",
    # Rescue tier (issue #555): rescue_attempted is the durable "used my
    # one shot" marker. Only charlie unescalate clears it (this tuple) —
    # every other code path treats a present marker as permanent.
    "rescue_attempted",
    "rescue_cause",
    "rescue_dispatched_at",
    # Issue #1132: ``charlie unescalate --pr`` did NOT clear the
    # ``foreign_issue_ref`` marker (observed: unescalate flipped
    # janitor_blocked -> open_passive but left the park in place, so the PR
    # stayed invisible). The marker is the third hidden layer under salvage
    # and head-keyed parks; an operator re-arm must clear it so the next
    # pass re-probes the linked issue instead of skipping with zero events.
    "foreign_issue_ref",
)
# Issue-record equivalents (dispatch-side caps and stale worker bookkeeping).
UNESCALATE_ISSUE_RESET_FIELDS = (
    "dispatch_failed_at",
    "redispatch_at",
    "worker_death_at",
    # Issue #1243: the orphan-sweep no-open-PR redispatch cap tracking
    # fields must reset on human un-escalate so the cap starts fresh
    # after the operator re-arms the issue.
    "orphan_redispatch_head_sha",
    "orphan_redispatch_at",
    "orphan_redispatch_counted_dispatch",
    "escalation_reason",
    # Issue #1461: clear the append-only escalation history so a re-arm
    # gives every lane a genuinely fresh dedup slate.
    "escalation_reasons_seen",
    # Issue #783: a human-authorized manual unescalate clears the reason
    # class (the escalation itself is gone) and resets the auto
    # de-escalation counter -- unlike the automated sweep, which never
    # resets its own counter (that is the oscillation guard; see
    # _maybe_deescalate_mechanical). The one-time cap-notification marker
    # must reset alongside the counter it gates: without this, a human
    # unescalate -> re-escalate -> re-hit-the-cap cycle would silently
    # suppress the second `deescalation_cap_exhausted` event because the
    # stale marker from the first cap-hit survived the reset, leaving
    # the oscillation guard's terminal state undiagnosable the second
    # time around.
    "reason_class",
    "auto_deescalation_count",
    "deescalation_cap_notified_at",
    # Issue #1093: the per-escalation-episode marker for the rework
    # budget reset must clear alongside the escalation it tracks, so a
    # manual re-arm gives the next sweep clear a clean slate.
    "rework_budget_reset_for_terminal_since",
    "label_error",
    "worker_pid",
    "worker_process_start_time",
    "dispatched_at",
)
# Issue #1093: the de-escalation sweep's once-per-episode rework-budget
# reset must zero the per-mechanism PR counter that ACTUALLY gates the
# cleared ``escalation_reason``, not a counter belonging to a different
# lane.  ``_route_janitor_gate_failure_to_rework`` escalates with reason
# ``f"{attempts_key}_cap_exceeded"`` (or ``_stall_exceeded``) and reads
# ``attempts_key`` itself on the next pass; ``record_review`` escalates
# with ``max_rework_cycles_exceeded`` and reads ``request_changes_count``.
# Resetting ``request_changes_count`` for a ``no_op_rework_attempts_*``
# clear (the PR's own reproduction scenario) left the real gating counter
# untouched, so the router re-escalated on the very next detection -- the
# promised "fresh rework budget" never applied to the lane it serves.
#
# Each lane's counter is reset together with its head-baseline
# (``_last_head``) and stall-clock (``_stall_since`` / ``_stall_head``)
# companions so the next detection re-baselines instead of inheriting a
# stale head/stall snapshot from the exhausted episode.  Escalation
# reasons with no per-mechanism rework counter (e.g.
# ``session_failed_escalated``, ``worktree_unsafe``) are absent from the
# map: there is no rework budget to reset for them, so the clear resets
# nothing extra.  ``auto_deescalation_count`` still independently bounds
# total clears (Issue #783 hazard (b)), so the per-episode reset cannot
# unbound the paid-session loop.
REWORK_BUDGET_RESET_BY_ESCALATION_REASON: dict[str, tuple[str, tuple[str, ...]]] = {
    "max_rework_cycles_exceeded": ("request_changes_count", ()),
    "no_op_rework_attempts_cap_exceeded": (
        "no_op_rework_attempts",
        (
            "no_op_rework_attempts_last_head",
            "no_op_rework_attempts_stall_since",
            "no_op_rework_attempts_stall_head",
        ),
    ),
    "no_op_rework_attempts_stall_exceeded": (
        "no_op_rework_attempts",
        (
            "no_op_rework_attempts_last_head",
            "no_op_rework_attempts_stall_since",
            "no_op_rework_attempts_stall_head",
        ),
    ),
    "conflict_rework_attempts_cap_exceeded": (
        "conflict_rework_attempts",
        (
            "conflict_rework_attempts_last_head",
            "conflict_rework_attempts_stall_since",
            "conflict_rework_attempts_stall_head",
        ),
    ),
    "conflict_rework_attempts_stall_exceeded": (
        "conflict_rework_attempts",
        (
            "conflict_rework_attempts_last_head",
            "conflict_rework_attempts_stall_since",
            "conflict_rework_attempts_stall_head",
        ),
    ),
}
