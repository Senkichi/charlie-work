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
from typing import Any

from _fakes_github import FakeGitHub
from _janitor_routing_fixtures import _conflicting_app, _set_decision
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    PostMortemConfig,
    RescueConfig,
    ReviewConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp


def _config(tmp_path: Path, **kwargs: Any) -> OrchestratorConfig:
    """Config whose post_mortem sessions.db never resolves.

    Mirrors ``tests/_unescalate_fixtures.py``'s ``_app``: the default
    ``db_path=""`` resolves to the real %APPDATA%\\devin\\cli\\sessions.db,
    which on a runner can hold a stale timestamp for a test PID and flip
    the liveness probe from inconclusive to conclusive-stale. Pointing at
    a nonexistent path makes every probe source error out, so the state-
    side verdict lands on the wall-clock-deadline branch the tests
    control via ``dispatched_at`` freshness.
    """
    return OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
        **kwargs,
    )


def _live_worker_fields() -> dict[str, Any]:
    """Issue-state fields for a genuinely live state-tracked worker.

    ``os.getpid()`` is this test process -- a real live PID -- and a fresh
    ``dispatched_at`` keeps the session within the watchdog wall-clock
    deadline, so ``issue_worker_liveness``'s inconclusive activity probe
    defers to ``live=True`` (same shape as
    ``test_unescalate_refuses_entirely_when_worker_session_alive``).
    """
    return {
        "worker_pid": os.getpid(),
        "dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def _wedged_worker_fields() -> dict[str, Any]:
    """Issue-state fields for an alive-but-wedged worker.

    The test process is alive, but ``dispatched_at`` is days old -- past
    the wall-clock deadline -- so ``issue_worker_liveness`` reports
    ``live=False`` and the cap escalation must proceed (the same shape as
    ``test_unescalate_proceeds_when_worker_alive_but_wedged``).
    """
    return {
        "worker_pid": os.getpid(),
        "dispatched_at": "2020-01-01T00:00:00+00:00",
    }


def _dead_pid() -> int:
    """A PID that is guaranteed not alive for the rest of this test."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert proc.pid is not None
    return proc.pid


def _events_of_kind(state: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in state.get("events", []) if e.get("kind") == kind]


def _no_op_cap_app(
    tmp_path: Path, issue_extra: dict[str, Any] | None = None
) -> tuple[OrchestratorApp, FakeGitHub]:
    """A ``dispatch_rework`` fixture with the no-op cap already exhausted.

    Issue 123 is ``rework_requested`` with ``redispatch_at`` at
    ``max_auto_redispatch`` and PR 456's head still at the recorded
    ``reviewed_head_sha`` -- the exact ``no_op_rework_escalated`` input
    shape from ``test_dispatch_rework_no_op_rework_cap_escalates`` --
    plus whatever worker-liveness fields ``issue_extra`` adds.
    """
    config = _config(
        tmp_path,
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    fake_gh = ReworkGitHub()
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": [now_iso, now_iso],
            **(issue_extra or {}),
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    return OrchestratorApp(tmp_path, paths, config, fake_gh), fake_gh


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


def _conflicting_issue_app(tmp_path: Path, **kwargs: Any) -> OrchestratorApp:
    """The janitor conflict-cap fixture with an isolated post_mortem db."""
    return _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=1),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
        **kwargs,
    )


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
