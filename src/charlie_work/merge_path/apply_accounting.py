"""Stage 5 of the live **Merge path**: the locked accounting write.

Everything here happens inside one ``state_lock`` taken after the merge effects:
``deadline_spent`` must be read after the irreversible merge (the pass budget may
have run out during cleanup), and the counters it settles are read from the
state the hand-off or merge just wrote. ``decide_accounting`` decides; this
module only gathers its facts and writes the result.
"""

from __future__ import annotations

from typing import Any

from ..pass_deadline import pass_deadline_spent
from ..write_gate import WriteGate, require_write_gate
from .decide import decide_accounting
from .mergequeue import mergequeue_stamp_needs_now
from .gather import persisted_from
from .model import (
    MERGEQUEUE_DWELL_FIELDS,
    Accounting,
    AccountingFacts,
    EffectResults,
    EventSpec,
    MergePathConfig,
    MergePlan,
)
from .ports import MergePathPorts


def settle_accounting(
    app: Any,
    ports: MergePathPorts,
    write_gate: WriteGate,
    *,
    pr_number: int,
    issue_number: int | None,
    cfg: MergePathConfig,
    plan: MergePlan,
    results: EffectResults,
    pr: dict[str, Any],
) -> Accounting:
    """Decide and persist the failed-attempt counters, mergequeue stamps and events."""
    write_gate = require_write_gate(write_gate)
    state_file = app.paths.state_file
    live_head = pr.get("headRefOid")
    with ports.state_lock(state_file):
        state = ports.load_state(state_file)
        existing = state["prs"].get(str(pr_number), {})
        locked = persisted_from(existing)
        deadline_spent = pass_deadline_spent(app.gh)
        merged = bool(results.merge_output)
        accounting = decide_accounting(
            plan,
            results,
            AccountingFacts(
                pr_number=pr_number,
                issue_number=issue_number,
                config=cfg,
                locked=locked,
                deadline_spent=deadline_spent,
                now_iso=ports.utc_now()
                if mergequeue_stamp_needs_now(locked, merged, live_head)
                else "",
                live_head_sha=live_head,
                mergeable=pr.get("mergeable"),
                merge_state_status=pr.get("mergeStateStatus"),
            ),
        )
        if accounting.merge_alert_ok and issue_number is not None:
            issue_key = str(issue_number)
            issue_entry = state["issues"].get(issue_key, {})
            if issue_entry.get("merge_alert") != "OK":
                state["issues"][issue_key] = {**issue_entry, "merge_alert": "OK"}
        prs_entry: dict[str, Any] = {
            **existing,
            "number": pr_number,
            "issue_number": issue_number,
            "consecutive_failed_merge_attempts": accounting.failed_attempts,
            "consecutive_stale_base_deferrals": accounting.stale_base_deferrals,
        }
        if accounting.pr_status is not None:
            prs_entry["status"] = accounting.pr_status
            prs_entry["merged"] = True
        # Issue #1401: both stamps live only while the PR stays in the mergequeue.
        if accounting.mergequeue_since is not None and accounting.mergequeue_head_sha is not None:
            prs_entry["mergequeue_since"] = accounting.mergequeue_since
            prs_entry["mergequeue_head_sha"] = accounting.mergequeue_head_sha
            # #2440: every full pass re-proves the queued PR; the skip trusts it for 30 min.
            prs_entry["mergequeue_checked_at"] = ports.utc_now()
        else:
            for field_name in MERGEQUEUE_DWELL_FIELDS:
                prs_entry.pop(field_name, None)
        # Issue #2743: a counted queue revert rewrites the counter at the live
        # head; any other pass leaves both fields untouched.
        if accounting.mergequeue_requeues is not None:
            prs_entry["consecutive_mergequeue_requeues"] = accounting.mergequeue_requeues
            prs_entry["mergequeue_requeues_head_sha"] = accounting.mergequeue_requeues_head_sha
        state["prs"][str(pr_number)] = prs_entry
        for spec in accounting.events:
            state = _record(write_gate, state, spec)
        write_gate.save_state(state)
    return accounting


def _record(write_gate: WriteGate, state: dict[str, Any], spec: EventSpec) -> dict[str, Any]:
    """Emit one ``decide_accounting`` event through the gate, one literal call per kind.

    The kind is deliberately *not* forwarded as a variable: the event-kind
    guards (``test_event_kind_consumers``, ``test_instrumentation_event_kind_registry``)
    resolve literal kinds, and a forwarded ``spec.kind`` would hide these three
    kinds from them. A kind ``decide_accounting`` emits without a branch here
    raises instead of being dropped; ``tests/test_merge_path_apply.py`` pins the
    two sets equal.
    """
    payload = dict(spec.payload)
    if spec.kind == "merge_ready":
        return write_gate.record_event(state, "merge_ready", payload)
    if spec.kind == "merge_succeeded":
        return write_gate.record_event(state, "merge_succeeded", payload)
    if spec.kind == "merge_failed_attempt_alarm":
        return write_gate.record_event(state, "merge_failed_attempt_alarm", payload)
    if spec.kind == "mergequeue_requeue_capped":
        return write_gate.record_event(state, "mergequeue_requeue_capped", payload)
    raise ValueError(f"unrecognised merge-path accounting event kind: {spec.kind!r}")
