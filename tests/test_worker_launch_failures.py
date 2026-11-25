"""Issue #3099: what both dispatch lanes record when a launch fails.

Characterisation suite for the two failure branches shared by fresh ``dispatch``
and remote ``dispatch_rework`` -- the blocked-environment branch and the
generic / deterministic failure branch with its escalating cap. Every test drives
the public entry points with a scripted ``FakeWorkerLauncher`` and pins, per
lane, the persisted issue record, the result data, the exact Event kinds,
payloads and events.db ``repo`` column, and the label writes.

The lane differences the branches take as inputs are declared once in
:data:`LANE_FACTS`: the Event kind prefix, the status after a restore, the failure
counter (fresh ``dispatch_failed_at``, never pruned, ISO ``+00:00`` stamps;
rework ``redispatch_at``, windowed, ``Z`` stamps), the cap escalation reason and
the ``previous_status`` an escalation Event carries. The tests also pin the
differences that are not inputs but observable today: fresh clears a stale
escalation on a restore and rework does not, and fresh persists the lane-built
record on an escalation where rework persists only the counters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _launch_outcome_fixtures import (
    _FOREIGN_WRITER_ERROR,
    FRESH,
    LANES,
    REWORK,
    _app,
    _config,
    _events,
    _launch,
    _marker,
    _rework_app,
    _run,
)
from _rework_dispatch_fixtures import _blocked_env_timestamps
from charlie_work.host.fakes import FakeProcessProbe
from charlie_work.adapters import SessionDispatchResult
from charlie_work.instrumentation import query_events
from charlie_work.state import load_state
from charlie_work.unescalate_reset_fields import ISSUE_BUDGET_RESET_BY_ESCALATION_REASON
import charlie_work.worker_launch_outcome as worker_launch_outcome
from charlie_work.workflow import OrchestratorApp
from charlie_work.worker_launch_outcome import LaunchFailureOutcome
from charlie_work.write_gate import WriteGate

_BLOCKED = "worktree_foreign_writer"


@dataclass(frozen=True)
class _LaneFacts:
    """The per-lane inputs of the failure branches (issue #3099's table)."""

    restored_status: str
    blocked_kind: str
    blocked_reaped_kind: str
    counter: str
    cap_reason: str
    previous_status: str
    stamp_suffix: str


LANE_FACTS = {
    FRESH: _LaneFacts(
        restored_status="dispatch_failed",
        blocked_kind="dispatch_blocked_environment",
        blocked_reaped_kind="dispatch_blocked_environment_reaped",
        counter="dispatch_failed_at",
        cap_reason="dispatch_failed_cap_exceeded",
        previous_status="dispatch_pending",
        stamp_suffix="+00:00",
    ),
    REWORK: _LaneFacts(
        restored_status="rework_requested",
        blocked_kind="rework_dispatch_blocked_environment",
        blocked_reaped_kind="rework_dispatch_blocked_environment_reaped",
        counter="redispatch_at",
        cap_reason="redispatch_cap_exceeded",
        previous_status="rework_requested",
        stamp_suffix="Z",
    ),
}


def _issue_entry(app: OrchestratorApp) -> dict[str, Any]:
    return load_state(app.paths.state_file)["issues"]["123"]


def _stamps_ago(*hours: float) -> list[str]:
    now = datetime.now(UTC)
    return [(now - timedelta(hours=h)).isoformat().replace("+00:00", "Z") for h in hours]


def _seed(lane: str, **fields: Any) -> dict[str, Any]:
    """A seeded issue record; fresh needs a retryable ``dispatch_failed`` status."""
    seed: dict[str, Any] = {
        "escalation_reason": "stale",
        "label_error": {"edge": "stale", "outcome": "partial_failure"},
    }
    if lane == FRESH:
        seed["status"] = "dispatch_failed"
    return {**seed, **fields}


def _assert_escalated_record_keeps_claim_and_label_error(entry: dict[str, Any]) -> None:
    """Both lanes escalate onto the STORED record: the claim stamp and a stale
    ``label_error`` survive (the lane-built entry is never seated before the escalation)."""
    assert "dispatch_pending_at" in entry
    assert entry["label_error"] == {"edge": "stale", "outcome": "partial_failure"}


def _failure_data(result: Any) -> None:
    """The result data of a failed launch, identical in both lanes."""
    assert result.ok is False
    assert result.data["failed_count"] == 1
    assert result.data["failures"] == {123: _FOREIGN_WRITER_ERROR}
    assert result.data["label_errors"] == []


def _edges(app: OrchestratorApp) -> list[str]:
    return [e["payload"]["cause"] for e in _events(app, "lifecycle_transition")]


# --- blocked environment ---------------------------------------------------------


@LANES
def test_blocked_environment_below_cap_restores_and_counts(
    tmp_path: Path, fake_host, lane: str
) -> None:
    facts = LANE_FACTS[lane]
    recent, old = _blocked_env_timestamps(1)[0], _stamps_ago(10)[0]
    app = _app(
        lane,
        tmp_path,
        config=_config(max_auto_redispatch=2, max_foreign_writer_reaps=2),
        entry=_seed(lane, blocked_environment_at=[old, recent]),
    )
    _launch(fake_host, ok=False, failure_kind=_BLOCKED, pid=1234)

    result = _run(app, lane)

    _failure_data(result)
    entry = _issue_entry(app)
    assert entry["status"] == facts.restored_status
    assert entry["dispatched_at"] is None
    # The window drops the out-of-window stamp in both lanes; the new stamp is Z.
    assert entry["blocked_environment_at"][0] == recent
    assert len(entry["blocked_environment_at"]) == 2
    assert entry["blocked_environment_at"][1].endswith("Z")
    assert facts.counter not in entry  # an environment block never counts as a failure
    assert ("escalation_reason" in entry) == (lane == REWORK)  # fresh clears, rework keeps
    (event,) = _events(app, facts.blocked_kind)
    assert event["payload"] == {
        "issue_number": 123,
        "failure_kind": _BLOCKED,
        "blocking_error": _FOREIGN_WRITER_ERROR,
        "blocked_environment_count": 2,
    }
    assert event["repo"] == tmp_path.name
    assert _events(app, facts.blocked_reaped_kind) == []
    assert _events(app, "session_failed_escalated") == []
    assert app.gh.labels_added == []
    assert _edges(app) == []


def _blocked_at_cap_app(
    lane: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host: Any, *, reap: bool
) -> OrchestratorApp:
    """An app whose launch fails blocked with the cap already reached by this attempt.

    ``dispatch_rework`` escalates a spent blocked-environment counter before it
    launches, so the failure branch is only reachable once that pre-claim safety
    net has reaped an idle foreign writer (one reap, which the failure path then
    counts); fresh dispatch reaches it with an empty counter and ``cap == 0``.
    ``reap`` is what the failure path's own reap attempt returns.
    """
    app = _app(
        lane,
        tmp_path,
        config=_config(max_auto_redispatch=0, max_foreign_writer_reaps=2),
        entry=_seed(lane),
    )
    worktree = None
    if lane == REWORK:
        worktree = str(_marker(app, tmp_path))
        fake_host(probe=FakeProcessProbe({1234: None}))
        monkeypatch.setattr(
            "charlie_work.workflow._reap_idle_foreign_writer", lambda *_a, **_k: True
        )
    _launch(fake_host, ok=False, failure_kind=_BLOCKED, pid=1234, worktree_path=worktree)
    monkeypatch.setattr(
        "charlie_work.workflow._try_reap_blocked_foreign_writer", lambda *_a, **_k: reap
    )
    return app


@LANES
def test_blocked_environment_at_cap_escalates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host, lane: str
) -> None:
    facts = LANE_FACTS[lane]
    app = _blocked_at_cap_app(lane, tmp_path, monkeypatch, fake_host, reap=False)

    result = _run(app, lane)

    _failure_data(result)
    entry = _issue_entry(app)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "dispatch_blocked_environment"
    assert entry["reason_class"] == "mechanical"
    assert entry["dispatched_at"] is None
    _assert_escalated_record_keeps_claim_and_label_error(entry)
    assert len(entry["blocked_environment_at"]) == 1
    assert entry["blocked_environment_at"][-1].endswith("Z")
    assert ISSUE_BUDGET_RESET_BY_ESCALATION_REASON["dispatch_blocked_environment"] == (
        "blocked_environment_at",
    )
    # Fresh persists the record it built; rework persists only the counter.
    assert ("branch_name" in entry) == (lane == FRESH)
    assert ("prompt_path" in entry) == (lane == FRESH)
    assert _events(app, facts.blocked_kind) == []
    assert _events(app, facts.blocked_reaped_kind) == []  # the failure path's reap declined
    (event,) = _events(app, "session_failed_escalated")
    assert event["payload"] == {
        "issue_number": 123,
        "previous_status": facts.previous_status,
        "reason": "dispatch_blocked_environment",
        "failure_kind": _BLOCKED,
        "blocking_error": _FOREIGN_WRITER_ERROR,
        "blocked_environment_count": 1,
    }
    # Issue #3099: rework's blocked-environment Events now go through the write
    # gate, which always binds ``repo`` (they were raw ``append_event`` calls).
    assert event["repo"] == tmp_path.name
    assert _edges(app) == ["redispatch_operator_queued"]
    assert (123, app.config.labels.operator_queue) in app.gh.labels_added


@LANES
def test_blocked_environment_at_cap_reaps_instead_of_escalating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host, lane: str
) -> None:
    facts = LANE_FACTS[lane]
    app = _blocked_at_cap_app(lane, tmp_path, monkeypatch, fake_host, reap=True)
    prior_reaps = 1 if lane == REWORK else 0  # rework's pre-claim safety net reaped once

    result = _run(app, lane)

    _failure_data(result)
    entry = _issue_entry(app)
    assert entry["status"] == facts.restored_status
    assert entry["dispatched_at"] is None
    assert entry["blocked_environment_at"] == []
    assert len(entry["foreign_writer_reaps"]) == prior_reaps + 1
    assert ("escalation_reason" in entry) == (lane == REWORK)  # fresh clears, rework keeps
    (event,) = _events(app, facts.blocked_reaped_kind)
    assert event["payload"] == {
        "issue_number": 123,
        "failure_kind": _BLOCKED,
        "pid": 1234,
        "blocked_environment_count": 1,
        "foreign_writer_reap_count": prior_reaps + 1,
    }
    assert event["repo"] == tmp_path.name
    assert _events(app, facts.blocked_kind) == []
    assert _events(app, "session_failed_escalated") == []
    assert app.gh.labels_added == []
    assert _edges(app) == []


# --- generic and deterministic failure --------------------------------------------


@LANES
def test_generic_failure_below_cap_restores_and_counts(
    tmp_path: Path, fake_host, lane: str
) -> None:
    facts = LANE_FACTS[lane]
    old, recent = _stamps_ago(10)[0], _stamps_ago(0.1)[0]
    app = _app(
        lane,
        tmp_path,
        config=_config(max_auto_redispatch=2),
        entry=_seed(lane, **{facts.counter: [old, recent]}),
    )
    _launch(fake_host, ok=False, pid=1234)

    result = _run(app, lane)

    _failure_data(result)
    entry = _issue_entry(app)
    assert entry["status"] == facts.restored_status
    assert entry["dispatched_at"] is None
    stamps = entry[facts.counter]
    assert stamps[-1].endswith(facts.stamp_suffix)
    # Rework prunes its window on every write; fresh keeps every attempt it ever saw.
    assert stamps[:-1] == ([old, recent] if lane == FRESH else [recent])
    assert "blocked_environment_at" not in entry
    assert ("escalation_reason" in entry) == (lane == REWORK)  # fresh clears, rework keeps
    for kind in (facts.blocked_kind, facts.blocked_reaped_kind, "session_failed_escalated"):
        assert _events(app, kind) == []
    assert app.gh.labels_added == []
    assert _edges(app) == []


@LANES
def test_generic_failure_past_cap_escalates_with_the_cap_reason(
    tmp_path: Path, fake_host, lane: str
) -> None:
    facts = LANE_FACTS[lane]
    app = _app(
        lane,
        tmp_path,
        config=_config(max_auto_redispatch=2),
        entry=_seed(lane, **{facts.counter: _stamps_ago(0.2, 0.1)}),
    )
    _launch(fake_host, ok=False, pid=1234)

    result = _run(app, lane)

    _failure_data(result)
    entry = _issue_entry(app)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == facts.cap_reason
    assert entry["reason_class"] == "mechanical"
    assert entry["dispatched_at"] is None
    _assert_escalated_record_keeps_claim_and_label_error(entry)
    assert len(entry[facts.counter]) == 3
    assert entry[facts.counter][-1].endswith(facts.stamp_suffix)
    # The unescalate registry names the counter this lane just filled.
    assert ISSUE_BUDGET_RESET_BY_ESCALATION_REASON[facts.cap_reason] == (facts.counter,)
    assert ("branch_name" in entry) == (lane == FRESH)
    assert ("prompt_path" in entry) == (lane == FRESH)
    assert _events(app, "session_failed_escalated") == []
    assert _edges(app) == ["redispatch_operator_queued"]
    assert (123, app.config.labels.operator_queue) in app.gh.labels_added


@LANES
@pytest.mark.parametrize(
    ("failure_kind", "reason_class", "edge", "label"),
    [
        ("worker_blocked", "mechanical", "redispatch_operator_queued", "operator_queue"),
        ("worktree_unsafe_local_commits", "judgment", "redispatch_escalated", "human_needed"),
    ],
)
def test_deterministic_failure_escalates_at_once(
    tmp_path: Path,
    fake_host,
    lane: str,
    failure_kind: str,
    reason_class: str,
    edge: str,
    label: str,
) -> None:
    facts = LANE_FACTS[lane]
    app = _app(lane, tmp_path, config=_config(max_auto_redispatch=5), entry=_seed(lane))
    _launch(fake_host, ok=False, failure_kind=failure_kind, pid=1234)

    result = _run(app, lane)

    _failure_data(result)
    entry = _issue_entry(app)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == failure_kind  # not the cap reason
    assert entry["reason_class"] == reason_class
    assert entry["dispatched_at"] is None
    _assert_escalated_record_keeps_claim_and_label_error(entry)
    assert len(entry[facts.counter]) == 1  # the attempt is counted even when it escalates
    assert entry[facts.counter][0].endswith(facts.stamp_suffix)
    assert _edges(app) == [edge]
    assert (123, getattr(app.config.labels, label)) in app.gh.labels_added


# --- the same Events in a real run and a dry run ---------------------------------
#
# A dry ``WriteGate`` writes nothing (write_gate.py: "no event at all under
# dry-run"), and neither lane reaches a failed launch in a dry run: both return
# from their planning branch before launching. So "the same Events in a real run
# and a dry run" is pinned as: the conversion of rework's blocked-environment
# Events onto the gate changes what is written in neither mode.


def _rework_blocked_below_cap_app(tmp_path: Path, *, dry_run: bool) -> OrchestratorApp:
    seeded = _blocked_env_timestamps(1)
    return _rework_app(
        tmp_path,
        config=_config(max_auto_redispatch=2, max_foreign_writer_reaps=2),
        entry={"blocked_environment_at": seeded},
        dry_run=dry_run,
    )


def test_rework_blocked_environment_event_is_written_in_a_real_run(
    tmp_path: Path, fake_host
) -> None:
    app = _rework_blocked_below_cap_app(tmp_path, dry_run=False)
    _launch(fake_host, ok=False, failure_kind=_BLOCKED, pid=1234)

    app.dispatch_rework()

    (event,) = _events(app, "rework_dispatch_blocked_environment")
    assert event["payload"]["blocked_environment_count"] == 2
    assert event["repo"] == tmp_path.name
    ring = [e["kind"] for e in load_state(app.paths.state_file)["events"]]
    assert "rework_dispatch_blocked_environment" in ring


def test_rework_blocked_environment_event_is_not_written_in_a_dry_run(
    tmp_path: Path, fake_host
) -> None:
    app = _rework_blocked_below_cap_app(tmp_path, dry_run=True)
    before = _issue_entry(app)
    _launch(fake_host, ok=False, failure_kind=_BLOCKED, pid=1234)

    result = app.dispatch_rework()

    assert result.ok is True  # planning only: the dry run never launches
    assert result.data["dispatch_results"] == []
    assert _issue_entry(app) == before
    for kind in (
        "rework_dispatch_blocked_environment",
        "rework_dispatch_blocked_environment_reaped",
        "session_failed_escalated",
    ):
        assert _events(app, kind) == []


def _failed_launch_call(
    gate: WriteGate, state_path: Path, *, max_auto_redispatch: int
) -> LaunchFailureOutcome:
    """Drive the module directly with rework's declared inputs and ``gate``."""
    entry = {
        "number": 123,
        "status": "rework_requested",
        "blocked_environment_at": _blocked_env_timestamps(1),
    }
    state = load_state(state_path)
    state["issues"]["123"] = dict(entry)
    failed = SessionDispatchResult(
        issue_number=123,
        issue_title="Fix search",
        prompt_path="p.md",
        branch_name="agent/issue-123",
        adapter="command",
        ok=False,
        error=_FOREIGN_WRITER_ERROR,
        failure_kind=_BLOCKED,
        pid=1234,
    )
    return worker_launch_outcome.record_failed_launch(
        gate,
        state,
        entry,
        failed_result=failed,
        issue_number=123,
        now=datetime.now(UTC),
        window_minutes=240,
        max_auto_redispatch=max_auto_redispatch,
        max_reaps=0,
        try_reap=lambda: False,
        recent_attempts=None,
        rework=True,
        save_inline=True,
        clear_stale_escalation=False,
        escalation_carries_entry=False,
        z_stamps=True,
        restored_status="rework_requested",
        previous_status="rework_requested",
        cap_reason="redispatch_cap_exceeded",
    )


@pytest.mark.parametrize(
    ("max_auto_redispatch", "status", "kinds"),
    [
        (5, "rework_requested", ["rework_dispatch_blocked_environment"]),
        (0, "escalated", ["session_failed_escalated"]),
    ],
    ids=["restore", "escalate"],
)
def test_failed_launch_decides_the_same_in_a_dry_and_a_real_gate(
    tmp_path: Path, max_auto_redispatch: int, status: str, kinds: list[str]
) -> None:
    outcomes: dict[bool, LaunchFailureOutcome] = {}
    paths: dict[bool, Path] = {}
    for dry_run in (False, True):
        root = tmp_path / ("dry" if dry_run else "real")
        root.mkdir()
        paths[dry_run] = root / "state.json"
        gate = WriteGate(dry_run=dry_run, state_path=paths[dry_run], repo="repo-x")
        outcomes[dry_run] = _failed_launch_call(
            gate, paths[dry_run], max_auto_redispatch=max_auto_redispatch
        )

    real, dry = outcomes[False], outcomes[True]
    # The same decision in both modes ...
    assert (real.status, real.reason_class) == (dry.status, dry.reason_class)
    assert real.status == status
    # ... the real gate writes exactly the Events the lane declared, with the repo ...
    assert [e["kind"] for e in query_events(paths[False])] == kinds
    assert {e["repo"] for e in query_events(paths[False])} == {"repo-x"}
    assert [e["kind"] for e in real.state["events"]] == kinds
    assert paths[False].exists()
    # ... and the dry gate writes nothing: no state file, no Event rows, no ring entry.
    assert not paths[True].exists()
    assert not (paths[True].parent / "events.db").exists()
    assert dry.state.get("events", []) == []
