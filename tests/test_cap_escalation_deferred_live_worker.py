"""Regression tests for issue #2051: cap escalations must defer while a
live worker holds the issue.

Incident (issue #2006 / PR #2027): a Devin rework worker launched at
05:03:37Z; the merge-conflict router re-set the issue to
``rework_requested`` at 05:05:43Z while that worker was still live; the
no-op redispatch cap escalated the issue at 05:08:30Z and the janitor
conflict cap escalated it again at 05:09:18Z -- handing the branch to
``agent:operator-queue`` while the worker kept writing (it committed a
merge at ~05:25Z). A manual ``unescalate`` then refused at 05:26:04Z
because the worker session was still live -- the escalation half of the
system never checked what ``unescalate`` already knew.

Root cause: both cap paths decide from counters and issue status alone,
and ``rework_requested`` is not evidence that nobody holds the issue --
the conflict/check-failure routers re-set that status while an earlier
worker remains alive. The fix routes both cap-exceeded decisions through
the same ``issue_worker_liveness`` predicate ``unescalate`` uses: a live
verdict defers the whole cap decision (counters, labels, and status
untouched) and records it once per worker via
``escalation_deferred_live_worker``; a dead or wedged verdict preserves
the existing escalation. The deferral tests below fail if the guard is
removed -- that is the mutation check for this fix.

New file (not test_janitor_rework_worker_launched.py or
test_charlie_work_dispatch_rework_caps.py) to avoid merge conflicts with
sibling PRs touching those files.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from _cap_deferral_fixtures import (
    _blocked_environment_cap_app,
    _conflicting_issue_app,
    _dead_pid,
    _events_of_kind,
    _live_worker_fields,
    _no_op_cap_app,
    _wedged_worker_fields,
    _worker_death_cap_app,
)
from _janitor_routing_fixtures import _set_decision
from charlie_work.config import RescueConfig
from charlie_work.orphaned_worker_review_drain import (
    OrphanedWorkerReviewRoute,
    drain_orphaned_worker_review_routes,
)
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import CommandResult


# ---------------------------------------------------------------------------
# dispatch_rework no-op cap (site A)
# ---------------------------------------------------------------------------


def test_no_op_cap_escalation_defers_while_worker_live(tmp_path: Path) -> None:
    """Acceptance 1: no-op cap + live worker -> no escalation, no label
    change, ``escalation_deferred_live_worker`` emitted, cap untouched."""
    app, fake_gh = _no_op_cap_app(tmp_path, issue_extra=_live_worker_fields())

    result = app.dispatch_rework()

    assert result.ok is True
    assert 123 not in result.data.get("no_op_rework_escalated", [])
    assert 123 in result.data.get("escalation_deferred_live_worker", [])

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "rework_requested"
    assert "escalation_reason" not in issue
    # The deferral must not consume or extend the cap counter -- the cap
    # is still tripped on the next pass after the worker exits.
    assert len(issue["redispatch_at"]) == 2
    marker = issue["escalation_deferred_live_worker"]
    assert marker["worker_pid"] == os.getpid()
    assert marker["reason"] == "no_op_rework_cap_exceeded"

    assert _events_of_kind(state, "session_failed_escalated") == []
    deferred_events = _events_of_kind(state, "escalation_deferred_live_worker")
    assert len(deferred_events) == 1
    payload = deferred_events[0]["payload"]
    assert payload["issue_number"] == 123
    assert payload["reason"] == "no_op_rework_cap_exceeded"
    assert payload["worker_pid"] == os.getpid()

    # No label transition at all -- the mechanical escalation edge lands
    # agent:operator-queue when it fires.
    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []


def test_no_op_cap_deferral_emits_once_per_worker(tmp_path: Path) -> None:
    """The deferral event is deduplicated per worker session: a worker
    that stays live across many passes must not fire the event every
    pass. The marker on the issue record is the dedup key."""
    app, _fake_gh = _no_op_cap_app(tmp_path, issue_extra=_live_worker_fields())

    app.dispatch_rework()
    app.dispatch_rework()

    state = load_state(app.paths.state_file)
    assert len(_events_of_kind(state, "escalation_deferred_live_worker")) == 1


def test_no_op_cap_escalates_when_worker_dead(tmp_path: Path) -> None:
    """Acceptance 3: a dead worker must not defer the cap -- the
    escalation proceeds exactly as before the guard."""
    app, fake_gh = _no_op_cap_app(
        tmp_path,
        issue_extra={
            "worker_pid": _dead_pid(),
            "dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    )

    result = app.dispatch_rework()

    assert result.ok is True
    assert 123 in result.data.get("no_op_rework_escalated", [])
    assert 123 not in result.data.get("escalation_deferred_live_worker", [])

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "escalated"
    assert issue["escalation_reason"] == "redispatch_cap_exceeded"
    assert _events_of_kind(state, "escalation_deferred_live_worker") == []
    assert (123, app.config.labels.operator_queue) in fake_gh.labels_added


def test_no_op_cap_escalates_when_worker_wedged(tmp_path: Path) -> None:
    """Acceptance 3 (wedged arm): a worker PID that is alive but past the
    wall-clock deadline with no fresh activity is wedged, not live -- the
    same standard ``unescalate`` applies -- so the cap still escalates."""
    app, fake_gh = _no_op_cap_app(tmp_path, issue_extra=_wedged_worker_fields())

    result = app.dispatch_rework()

    assert result.ok is True
    assert 123 in result.data.get("no_op_rework_escalated", [])

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert _events_of_kind(state, "escalation_deferred_live_worker") == []
    assert (123, app.config.labels.operator_queue) in fake_gh.labels_added


def test_no_op_cap_escalates_after_deferred_worker_exits(tmp_path: Path) -> None:
    """Acceptance 4: a deferral must not consume or reset the cap -- the
    pass after the worker exits escalates on the already-exhausted
    ``redispatch_at`` count."""
    app, fake_gh = _no_op_cap_app(tmp_path, issue_extra=_live_worker_fields())

    result1 = app.dispatch_rework()
    assert 123 in result1.data.get("escalation_deferred_live_worker", [])

    # The worker exits: the recorded PID is no longer alive.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"]["worker_pid"] = _dead_pid()
        save_state(app.paths.state_file, state)

    result2 = app.dispatch_rework()
    assert 123 in result2.data.get("no_op_rework_escalated", [])
    assert 123 not in result2.data.get("escalation_deferred_live_worker", [])

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "escalated"
    assert issue["escalation_reason"] == "redispatch_cap_exceeded"
    # 2 seeded + the single stamp the escalation itself appends.
    assert len(issue["redispatch_at"]) == 3
    assert (123, app.config.labels.operator_queue) in fake_gh.labels_added


# ---------------------------------------------------------------------------
# janitor conflict-cap (site B: _route_janitor_gate_failure_to_rework)
# ---------------------------------------------------------------------------


def test_conflict_cap_escalation_defers_while_worker_live(tmp_path: Path) -> None:
    """Acceptance 2: conflict cap + live worker -> no escalation, no
    ``janitor_rework_escalated``, no escalation label transition, and the
    attempt counter is not persisted."""
    app = _conflicting_issue_app(tmp_path)
    _set_decision(app, 456, "request_changes")

    result1 = app.review(456)
    assert result1.ok is True
    assert result1.data["routed_to_rework"] is True
    labels_after_route = list(app.gh.labels_added)

    # A worker launches for the rework and is still live -- the #2006
    # incident's exact precondition.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"].update(_live_worker_fields())
        save_state(app.paths.state_file, state)

    # A failed rework cycle (new head, still conflicting) trips the cap
    # of 1 -- but the escalation must defer while the worker is live.
    app.gh.pr_head_shas[456] = "sha-cycle-2"
    result2 = app.review(456)

    assert result2.ok is True
    assert result2.data.get("escalation_deferred_live_worker") is True
    assert result2.data.get("escalated") is not True

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "rework_requested"
    assert "escalation_reason" not in issue
    # The burn-the-attempt write never ran: the stored counter is still
    # the routing-pass value, so the cap stays tripped for the next pass.
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1
    marker = issue["escalation_deferred_live_worker"]
    assert marker["worker_pid"] == os.getpid()
    assert marker["reason"] == "conflict_rework_attempts_cap_exceeded"

    assert _events_of_kind(state, "janitor_rework_escalated") == []
    assert len(_events_of_kind(state, "escalation_deferred_live_worker")) == 1
    # No escalation label edge fired: needs_rework landed on the routing
    # pass, and nothing was added after it.
    assert app.gh.labels_added == labels_after_route


def test_conflict_cap_escalates_when_worker_dead(tmp_path: Path) -> None:
    """The dead-worker arm for site B: the same cap input with a dead PID
    escalates exactly as before the guard."""
    app = _conflicting_issue_app(tmp_path)
    _set_decision(app, 456, "request_changes")

    result1 = app.review(456)
    assert result1.data["routed_to_rework"] is True

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"]["worker_pid"] = _dead_pid()
        state["issues"]["123"]["dispatched_at"] = (
            datetime.now(UTC).isoformat().replace("+00:00", "Z")
        )
        save_state(app.paths.state_file, state)

    app.gh.pr_head_shas[456] = "sha-cycle-2"
    result2 = app.review(456)

    assert result2.ok is False
    assert result2.data["escalated"] is True

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == ("conflict_rework_attempts_cap_exceeded")
    assert _events_of_kind(state, "escalation_deferred_live_worker") == []
    assert len(_events_of_kind(state, "janitor_rework_escalated")) == 1
    assert (123, app.config.labels.operator_queue) in app.gh.labels_added


def test_conflict_cap_escalates_when_worker_wedged(tmp_path: Path) -> None:
    """An alive-but-wedged worker (PID alive, session past the wall-clock
    deadline, no activity) does not defer the conflict cap."""
    app = _conflicting_issue_app(tmp_path)
    _set_decision(app, 456, "request_changes")

    result1 = app.review(456)
    assert result1.data["routed_to_rework"] is True

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"].update(_wedged_worker_fields())
        save_state(app.paths.state_file, state)

    app.gh.pr_head_shas[456] = "sha-cycle-2"
    result2 = app.review(456)

    assert result2.ok is False
    assert result2.data["escalated"] is True
    state = load_state(app.paths.state_file)
    assert _events_of_kind(state, "escalation_deferred_live_worker") == []
    assert len(_events_of_kind(state, "janitor_rework_escalated")) == 1


def test_conflict_cap_escalates_after_deferred_worker_exits(tmp_path: Path) -> None:
    """Acceptance 4 for site B: defer while live, escalate on the next
    cap check after the worker exits -- the attempt counter was never
    persisted by the deferral, so the cap is still exhausted."""
    app = _conflicting_issue_app(tmp_path)
    _set_decision(app, 456, "request_changes")

    result1 = app.review(456)
    assert result1.data["routed_to_rework"] is True

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"].update(_live_worker_fields())
        save_state(app.paths.state_file, state)

    app.gh.pr_head_shas[456] = "sha-cycle-2"
    result2 = app.review(456)
    assert result2.data.get("escalation_deferred_live_worker") is True

    # The worker exits; the same settled head still differs from the
    # recorded baseline, so the cap check re-fires.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"]["worker_pid"] = _dead_pid()
        save_state(app.paths.state_file, state)

    result3 = app.review(456)
    assert result3.ok is False
    assert result3.data["escalated"] is True

    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert len(_events_of_kind(state, "janitor_rework_escalated")) == 1
    assert len(_events_of_kind(state, "escalation_deferred_live_worker")) == 1


def test_conflict_cap_deferral_suppresses_rescue_too(tmp_path: Path) -> None:
    """The deferral gates the whole cap-exceeded decision, not just the
    escalation arm: with the rescue tier enabled, a live-worker hold must
    defer rather than burn the one rescue slot on an issue a worker is
    still fixing (or dispatch a second worker that supersedes the live
    one). The rescue slot remains available for the first post-exit pass.
    """
    app = _conflicting_issue_app(tmp_path, rescue=RescueConfig(enabled=True))
    _set_decision(app, 456, "request_changes")

    result1 = app.review(456)
    assert result1.data["routed_to_rework"] is True

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"].update(_live_worker_fields())
        save_state(app.paths.state_file, state)

    app.gh.pr_head_shas[456] = "sha-cycle-2"
    result2 = app.review(456)

    assert result2.ok is True
    assert result2.data.get("escalation_deferred_live_worker") is True
    assert result2.data.get("rescue_dispatched") is not True

    state = load_state(app.paths.state_file)
    assert "rescue_attempted" not in state["prs"]["456"]
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert _events_of_kind(state, "rescue_dispatched") == []
    assert _events_of_kind(state, "janitor_rework_escalated") == []


# ---------------------------------------------------------------------------
# dispatch_rework worker-death cap (lane 2 of the three-lane guard)
# ---------------------------------------------------------------------------


def test_worker_death_cap_escalation_defers_while_worker_live(tmp_path: Path) -> None:
    """The death-loop lane shares the liveness guard: paired deaths at cap
    + a live worker -> deferral, no escalation, counters/status/labels
    untouched."""
    app, fake_gh = _worker_death_cap_app(tmp_path, issue_extra=_live_worker_fields())

    result = app.dispatch_rework()

    assert result.ok is True
    assert 123 not in result.data.get("worker_death_escalated", [])
    assert 123 in result.data.get("escalation_deferred_live_worker", [])

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "rework_requested"
    assert "escalation_reason" not in issue
    marker = issue["escalation_deferred_live_worker"]
    assert marker["worker_pid"] == os.getpid()
    assert marker["reason"] == "worker_death_loop"

    assert _events_of_kind(state, "session_failed_escalated") == []
    deferred_events = _events_of_kind(state, "escalation_deferred_live_worker")
    assert len(deferred_events) == 1
    assert deferred_events[0]["payload"]["reason"] == "worker_death_loop"
    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []


def test_worker_death_cap_escalates_when_worker_dead(tmp_path: Path) -> None:
    """The dead-worker arm for the death-loop lane: identical cap input
    with a dead PID escalates exactly as before the guard."""
    app, fake_gh = _worker_death_cap_app(
        tmp_path,
        issue_extra={
            "worker_pid": _dead_pid(),
            "dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    )

    result = app.dispatch_rework()

    assert result.ok is True
    assert 123 in result.data.get("worker_death_escalated", [])
    assert 123 not in result.data.get("escalation_deferred_live_worker", [])

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "escalated"
    assert issue["escalation_reason"] == "worker_death_loop"
    assert _events_of_kind(state, "escalation_deferred_live_worker") == []
    assert (123, app.config.labels.operator_queue) in fake_gh.labels_added


# ---------------------------------------------------------------------------
# dispatch_rework blocked-environment cap (lane 3 of the three-lane guard)
# ---------------------------------------------------------------------------


def test_blocked_environment_cap_defers_while_worker_live(tmp_path: Path) -> None:
    """The blocked-environment lane shares the liveness guard: a live
    worker holding the issue defers the escalation instead of handing a
    still-running session to the operator queue."""
    app, fake_gh = _blocked_environment_cap_app(tmp_path, issue_extra=_live_worker_fields())

    result = app.dispatch_rework()

    assert result.ok is True
    assert 123 not in result.data.get("blocked_environment_escalated", [])
    assert 123 in result.data.get("escalation_deferred_live_worker", [])

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "rework_requested"
    assert "escalation_reason" not in issue
    marker = issue["escalation_deferred_live_worker"]
    assert marker["worker_pid"] == os.getpid()
    assert marker["reason"] == "dispatch_blocked_environment"

    assert _events_of_kind(state, "session_failed_escalated") == []
    deferred_events = _events_of_kind(state, "escalation_deferred_live_worker")
    assert len(deferred_events) == 1
    assert deferred_events[0]["payload"]["reason"] == "dispatch_blocked_environment"
    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []


def test_blocked_environment_cap_escalates_when_worker_dead(tmp_path: Path) -> None:
    """The dead-worker arm for the blocked-environment lane: a dead PID
    lets the same cap input escalate exactly as before the guard."""
    app, fake_gh = _blocked_environment_cap_app(
        tmp_path,
        issue_extra={
            "worker_pid": _dead_pid(),
            "dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    )

    result = app.dispatch_rework()

    assert result.ok is True
    assert 123 in result.data.get("blocked_environment_escalated", [])
    assert 123 not in result.data.get("escalation_deferred_live_worker", [])

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "escalated"
    assert issue["escalation_reason"] == "dispatch_blocked_environment"
    assert _events_of_kind(state, "escalation_deferred_live_worker") == []
    assert (123, app.config.labels.operator_queue) in fake_gh.labels_added


# ---------------------------------------------------------------------------
# Dry-run partition, dedup key, and operator re-arm
# ---------------------------------------------------------------------------


def test_dry_run_reports_deferral_without_writing_marker_or_event(tmp_path: Path) -> None:
    """The dry-run partition reports ``escalation_deferred_live_worker``
    for the same cap input the live path defers -- but must write no
    marker to the issue and no event to state, or a preview would start
    the once-per-worker dedup window early."""
    app, _fake_gh = _no_op_cap_app(tmp_path, issue_extra=_live_worker_fields(), dry_run=True)

    result = app.dispatch_rework()

    assert result.ok is True
    assert 123 in result.data.get("escalation_deferred_live_worker", [])
    assert 123 not in result.data.get("no_op_rework_escalated", [])

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "rework_requested"
    assert "escalation_deferred_live_worker" not in issue
    assert _events_of_kind(state, "escalation_deferred_live_worker") == []
    assert _events_of_kind(state, "session_failed_escalated") == []


def test_deferral_re_emits_for_new_worker_identity(tmp_path: Path) -> None:
    """The once-per-worker dedup is keyed on worker identity, not the
    issue: after a deferral marker exists for one worker session, a
    deferral under a DIFFERENT live worker (new dispatch epoch, new PID)
    emits again rather than being silently deduped against the stale
    marker."""
    app, _fake_gh = _no_op_cap_app(tmp_path, issue_extra=_live_worker_fields())

    app.dispatch_rework()
    state = load_state(app.paths.state_file)
    assert len(_events_of_kind(state, "escalation_deferred_live_worker")) == 1

    # The first worker exits and a replacement worker launches under a
    # different PID for the same issue -- a genuinely different live
    # worker identity for the probe.
    replacement = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["issues"]["123"]["worker_pid"] = replacement.pid
            state["issues"]["123"]["dispatched_at"] = (
                datetime.now(UTC).isoformat().replace("+00:00", "Z")
            )
            save_state(app.paths.state_file, state)

        app.dispatch_rework()
    finally:
        replacement.kill()
        replacement.wait()

    state = load_state(app.paths.state_file)
    deferred_events = _events_of_kind(state, "escalation_deferred_live_worker")
    assert len(deferred_events) == 2
    assert deferred_events[-1]["payload"]["worker_pid"] == replacement.pid
    # Still no escalation -- the second worker is live too.
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert _events_of_kind(state, "session_failed_escalated") == []


def test_unescalate_clears_deferral_marker(tmp_path: Path) -> None:
    """``escalation_deferred_live_worker`` is listed in
    ``UNESCALATE_ISSUE_RESET_FIELDS``: a manual re-arm drops the per-
    episode dedup marker so the next episode's worker gets a fresh
    deferral emission instead of deduping against a stale identity."""
    app, _fake_gh = _no_op_cap_app(
        tmp_path,
        issue_extra={
            "status": "escalated",
            "escalation_reason": "redispatch_cap_exceeded",
            "escalation_deferred_live_worker": {
                "worker_pid": _dead_pid(),
                "source": "state",
                "session_started_at": "2020-01-01T00:00:00+00:00",
                "reason": "no_op_rework_cap_exceeded",
                "at": "2020-01-01T00:00:00+00:00",
            },
        },
    )
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"]["status"] = "escalated"
        state["prs"]["456"]["escalation_reason"] = "redispatch_cap_exceeded"
        save_state(app.paths.state_file, state)

    result = app.unescalate(pr_number=456)

    assert result.ok is True
    assert result.data["changed"] is True
    issue = load_state(app.paths.state_file)["issues"]["123"]
    assert "escalation_deferred_live_worker" not in issue
    assert issue["status"] != "escalated"


# ---------------------------------------------------------------------------
# review() consumer contracts (the rework finding: an ok=True deferral is
# NOT a fresh review packet)
# ---------------------------------------------------------------------------


def test_janitor_cap_deferral_does_not_flip_route_candidate_to_reviewing(
    tmp_path: Path,
) -> None:
    """``_route_rework_candidate_to_review`` must not read the janitor cap
    deferral's ``ok=True`` as a fresh packet: the head moved, the conflict
    cap is exceeded, and a live worker still holds the issue -- so
    ``review()`` defers, ``routed`` is False, and the issue stays
    ``rework_requested`` rather than flipping to ``reviewing`` while the
    worker is still writing."""
    app = _conflicting_issue_app(tmp_path)
    _set_decision(app, 456, "request_changes")
    # The head moved past the recorded verdict, which is what makes
    # _route_rework_candidate_to_review call review() at all.
    app.gh.pr_head_shas[456] = "sha-cycle-2"
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            **_live_worker_fields(),
        }
        # Settled new conflicted head: one counted attempt on a prior head
        # already at the cap of 1, and the live head differs from that
        # baseline -- the janitor gate reaches the cap-exceeded check,
        # where the live-worker deferral fires.
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-reviewed",
            "conflict_rework_attempts": 1,
            "conflict_rework_attempts_last_head": "sha-cycle-1",
        }
        save_state(app.paths.state_file, state)

    routed, review_result = app._route_rework_candidate_to_review(123, 456, "sha-reviewed")

    assert routed is False
    assert review_result.ok is True
    assert review_result.data.get("escalation_deferred_live_worker") is True
    assert not review_result.data.get("routed_to_rework")

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "rework_requested"
    assert "escalation_reason" not in issue
    assert _events_of_kind(state, "janitor_rework_escalated") == []
    deferred_events = _events_of_kind(state, "escalation_deferred_live_worker")
    assert len(deferred_events) == 1
    assert deferred_events[0]["payload"]["reason"] == "conflict_rework_attempts_cap_exceeded"
    # The routing event records the non-routing outcome, never a packet.
    pushed_events = _events_of_kind(state, "rework_already_pushed")
    assert len(pushed_events) == 1
    assert pushed_events[0]["payload"]["routed"] is False


def test_orphan_drain_does_not_report_deferral_as_routed_to_review(
    tmp_path: Path,
) -> None:
    """``drain_orphaned_worker_review_routes`` consumer contract: an
    ``ok=True`` result carrying ``escalation_deferred_live_worker`` is not
    a fresh packet either -- the issue must stay ``dispatched`` (its guard
    status) and the drain must report drift, never
    ``orphaned_worker_routed_to_review``.

    Production ``review()`` cannot currently produce this shape for a
    ``dispatched`` issue -- the janitor wrapper early-returns before the
    deferral probe for pending statuses -- so the route below drives the
    contract with a stubbed review callback returning the deferral shape,
    the same way this file's other drain coverage stubs review(). The
    guard exists so a future ``review()`` path that surfaces the flag for
    a dispatched issue cannot silently re-introduce the reviewing flip.
    """
    app, _fake_gh = _no_op_cap_app(
        tmp_path, issue_extra={"status": "dispatched", **_live_worker_fields()}
    )
    route = OrphanedWorkerReviewRoute(
        issue_number=123,
        pr_number=456,
        reviewed_head_sha="sha-abc123",
        live_head_sha="sha-abc123",
        fingerprint="fp-deferral",
        reason="dead_worker_with_head_change",
    )

    def deferred_review(_pr_number: int) -> CommandResult:
        return CommandResult(
            True,
            "rework cap escalation deferred: worker still live",
            {"pr": 456, "issue": 123, "escalation_deferred_live_worker": True},
        )

    drain_orphaned_worker_review_routes(
        [route],
        review_callback=deferred_review,
        gh=app.gh,
        config=app.config,
        state_file=app.paths.state_file,
        write_gate=app.write_gate,
    )

    state = load_state(app.paths.state_file)
    issue = state["issues"]["123"]
    assert issue["status"] == "dispatched"
    assert _events_of_kind(state, "orphaned_worker_routed_to_review") == []
    drift_events = _events_of_kind(state, "orphaned_worker_drift")
    assert len(drift_events) == 1
    assert drift_events[0]["payload"]["routed"] is False
    assert drift_events[0]["payload"]["issue_number"] == 123
