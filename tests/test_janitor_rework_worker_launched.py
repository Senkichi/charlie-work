"""Regression tests for the ``worker_launched`` diagnostic on the janitor
gate's rework-attempts-cap escalation.

Bug: ``dispatch_rework`` can fail every pass with a pre-launch worktree
refusal (``failures: {"<issue>": "worktree creation failed: worktree has
uncommitted modifications"}``) for a rework whose worktree is dirty with
launcher/worker scratch artifacts. ``dispatched_at`` stays null (the worker
never gets a PID), but ``_route_janitor_gate_failure_to_rework``'s own
``no_op_rework_attempts``/``conflict_rework_attempts`` counter is driven by
``review()`` re-detecting the same underlying no-op/conflict condition every
pass -- independent of whether a dispatch attempt ever succeeded -- so it
still eventually trips its cap and escalates with
``no_op_rework_attempts_cap_exceeded``/``conflict_rework_attempts_cap_exceeded``.
That reason string is misleading when no worker ever ran: it reads as "a
worker tried and failed repeatedly," not "the worktree layer refused to even
start one."

The fix adds a ``worker_launched`` field (derived from the issue's
``dispatched_at`` at escalation time) to both the escalated PR's state
(``pr_extra``) and the ``janitor_rework_escalated`` event payload, without
touching ``escalation_reason`` itself -- that string is matched against
``escalation_reasons_seen`` by the same-lane oscillation guard
(workflow.py, ``_route_janitor_gate_failure_to_rework``), and mutating it
would blind that guard, causing the lane to re-escalate every pass forever.

New file (not tests/test_fix_janitor_routing.py) to avoid merge conflicts
with sibling PRs also touching that file.
"""

from __future__ import annotations

from pathlib import Path

from _janitor_routing_fixtures import _conflicting_app, _set_decision
from charlie_work.config import ReviewConfig
from charlie_work.state import load_state, save_state, state_lock


def test_conflict_rework_cap_escalation_records_worker_launched_false(
    tmp_path: Path,
) -> None:
    """The default case: dispatch_rework never set dispatched_at (the worker
    never launched -- e.g. every attempt was refused at the worktree layer),
    so the cap-exceeded escalation must record worker_launched=False."""
    app = _conflicting_app(tmp_path, review=ReviewConfig(max_conflict_rework_attempts=1))
    _set_decision(app, 456, "request_changes")

    result1 = app.review(456)
    assert result1.ok is True
    assert result1.data["routed_to_rework"] is True

    # A failed rework cycle (new head, still conflicting) exceeds the cap of 1.
    app.gh.pr_head_shas[456] = "sha-cycle-2"
    result2 = app.review(456)
    assert result2.ok is False
    assert result2.data["escalated"] is True

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["prs"]["456"]["worker_launched"] is False

    escalated_events = [e for e in state["events"] if e["kind"] == "janitor_rework_escalated"]
    assert len(escalated_events) == 1
    assert escalated_events[0]["payload"]["worker_launched"] is False


def test_conflict_rework_cap_escalation_records_worker_launched_true(
    tmp_path: Path,
) -> None:
    """Positive control: when the issue's dispatched_at IS set at the moment
    the cap trips (a worker genuinely launched, ran, and its result is what
    kept failing the janitor gate), the escalation must record
    worker_launched=True -- the field is a real discriminator, not a
    constant."""
    app = _conflicting_app(tmp_path, review=ReviewConfig(max_conflict_rework_attempts=1))
    _set_decision(app, 456, "request_changes")

    result1 = app.review(456)
    assert result1.ok is True
    assert result1.data["routed_to_rework"] is True

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            **state["issues"]["123"],
            "dispatched_at": "2026-08-20T00:00:00Z",
        }
        save_state(app.paths.state_file, state)

    app.gh.pr_head_shas[456] = "sha-cycle-2"
    result2 = app.review(456)
    assert result2.ok is False
    assert result2.data["escalated"] is True

    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["worker_launched"] is True

    escalated_events = [e for e in state["events"] if e["kind"] == "janitor_rework_escalated"]
    assert len(escalated_events) == 1
    assert escalated_events[0]["payload"]["worker_launched"] is True
