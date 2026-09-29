"""Local merge-gate rework-routing delegates for ``OrchestratorApp``.

Home of the issue #1972 rework cap: ``_local_route_merge_rework`` routes a
merge-gate failure (base-sync conflict or full-suite failure) to rework below
a persisted per-kind attempt cap, and ``_local_merge_rework_escalate``
escalates to the operator queue on the cap-th attempt. The pre-cap version of
``_local_route_merge_rework`` lived in ``local_lanes.py`` since issue #1844;
both members moved here when #1972's added lines would have pushed
``local_lanes.py`` past its file-size-ratchet mark -- the same reason
``local_lane_stall_alarm.py`` (#1968) is its own submodule rather than a
``local_lanes.py`` member.

``workflow_delegation._install_delegates`` re-attaches each top-level ``def``
onto ``OrchestratorApp`` like any moved delegate (``self`` binds via the
descriptor protocol). Callers (``_local_merge_approved`` in
``local_lanes.py``) reach them through ``self.<name>`` unchanged.

Workflow-defined names are reached through ``_wf.<name>`` (the module-object
form the package ``__init__`` docstring's rule 2 requires), so the
``charlie_work.workflow`` monkeypatch seams keep landing. Every other free
name is imported directly from its defining module.
"""

from __future__ import annotations

from typing import Any

import charlie_work.workflow as _wf
from charlie_work.local_lane import local_pr_dict

# Issue #1972: the local merge gate's per-failure-kind rework budgets. The
# remote lane caps conflict rework through
# ``_route_janitor_gate_failure_to_rework`` (``conflict_rework_attempts``);
# the local lane's merge gate bypasses that wrapper, so it keeps its own
# counters on the local PR record -- deliberately separate field names so the
# packet re-mint's remote-counter reset block (``_local_build_packet``) does
# not alias them, and so remote state on a shared record is never clobbered.
#
# ``reason`` (the ``_local_route_merge_rework`` argument) maps to
# ``(counter field, ReviewConfig budget field)``. The cap read is per-kind:
# conflicts use ``max_conflict_rework_attempts`` and suite failures reuse
# ``max_rework_cycles`` (the lane's generic rework cap, per the issue). The
# below-cap event kind is deliberately NOT in this table: the emit-site
# scanners (``tests/_instrumentation_kind_scanner.py`` /
# ``tests/test_event_kind_consumers.py``) only resolve literal-shaped kind
# arguments, so ``_local_route_merge_rework`` picks it with an inline
# ``reason == "merge_conflict"`` ternary the way it did before this change.
_LOCAL_MERGE_REWORK_CAPS: dict[str, tuple[str, str]] = {
    "merge_conflict": (
        "local_merge_conflict_rework_attempts",
        "max_conflict_rework_attempts",
    ),
    "suite_failed": (
        "local_suite_failed_rework_attempts",
        "max_rework_cycles",
    ),
}

# The distinct escalation reason (state) + event kind for a capped local
# merge-gate rework loop. Keeping the reason lane-scoped -- rather than
# reusing the remote ``*_cap_exceeded`` names -- lets per-lane dedup guards
# and operators tell a local gate spiral from a remote janitor one.
_LOCAL_MERGE_REWORK_CAP_REASON = "local_merge_rework_cap_exceeded"


def _local_route_merge_rework(
    self,
    pr_number: int,
    issue_number: int,
    record: dict[str, Any],
    decision: dict[str, Any],
    *,
    reason: str,
    note: str,
) -> str:
    """Route a merge-gate failure (conflict or suite failure) to rework.

    Issue #1972: this path is capped. Each failure kind carries its own
    persisted counter on the local PR record (``_LOCAL_MERGE_REWORK_CAPS``);
    the counters survive packet re-mints and are cleared only by the
    successful-merge write, ``charlie unescalate``, or the de-escalation
    reset map. An attempt below the kind's cap routes to rework exactly as
    before; the cap-th attempt escalates with the distinct
    ``local_merge_rework_cap_exceeded`` reason via the same
    ``_escalate_issue`` mechanics as ``_local_merge_error_escalate``.

    Only ``record``s whose status is ``approved`` reach this helper (the
    merge gate's entry filter), so each call is one genuinely completed
    rework cycle -- there is no pending-rework double-count hazard to
    debounce the way ``_route_janitor_gate_failure_to_rework`` must for its
    every-pass redetection.

    A non-positive cap disables escalation (the remote lane's
    ``max_attempts > 0`` disable convention); the record routes to rework
    and the counter still increments so the spiral stays diagnosable.

    Returns ``"rework"`` or ``"escalated"`` for the caller's result entry.
    """
    counter_key, budget_attr = _LOCAL_MERGE_REWORK_CAPS[reason]
    cap = int(getattr(self.config.review, budget_attr))
    snapshot = _wf.load_state_locked(self.paths.state_file)
    attempts = (
        int(((snapshot.get("prs") or {}).get(str(pr_number)) or {}).get(counter_key) or 0) + 1
    )
    if cap > 0 and attempts >= cap:
        self._local_merge_rework_escalate(
            pr_number,
            issue_number,
            reason=reason,
            counter_key=counter_key,
            attempts=attempts,
            cap=cap,
            note=note,
        )
        return "escalated"
    pr = local_pr_dict(record)
    label_error = self._route_to_rework(
        pr,
        issue_number,
        decision,
        note,
        (
            "merge_conflict_rework_requested"
            if reason == "merge_conflict"
            else "local_suite_failed"
        ),
        extra_payload={"local": True, "reason": reason, "attempts": attempts},
        extra_state={
            counter_key: attempts,
            "local_merge_rework_reason": reason,
        },
    )
    if label_error:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            issue_entry = state["issues"].get(str(issue_number), {})
            state["issues"][str(issue_number)] = {
                **issue_entry,
                "number": issue_number,
                "label_error": label_error,
            }
            self.write_gate.save_state(state)
    return "rework"


def _local_merge_rework_escalate(
    self,
    pr_number: int,
    issue_number: int,
    *,
    reason: str,
    counter_key: str,
    attempts: int,
    cap: int,
    note: str,
) -> None:
    """Escalate a local merge-gate rework loop that reached its cap (issue #1972).

    Same ``_escalate_issue`` + mechanical label-edge mechanics as
    ``_local_merge_error_escalate``, with the persisted counter carried into
    ``pr_extra`` so the record keeps the exact attempt count that tripped it.
    """
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        state = _wf._escalate_issue(
            state,
            issue_number,
            reason=_LOCAL_MERGE_REWORK_CAP_REASON,
            reason_class="mechanical",
            pr_number=pr_number,
            pr_extra={
                counter_key: attempts,
                "local_merge_rework_reason": reason,
            },
        )
        state = self._record_event(
            state,
            "local_merge_rework_escalated",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "reason": reason,
                "escalation_reason": _LOCAL_MERGE_REWORK_CAP_REASON,
                "attempts": attempts,
                "cap": cap,
                "detail": _wf._truncate_for_event(note),
            },
            level="error",
        )
        self.write_gate.save_state(state)
    self.write_gate.transition(
        self.gh,
        self.config.labels,
        issue_number,
        _wf._escalation_edge("escalated", "mechanical"),
    )
