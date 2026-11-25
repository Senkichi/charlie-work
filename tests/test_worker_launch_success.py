"""Issue #3100: what both dispatch lanes record when a launch succeeds.

Characterisation suite for the success entry shared by fresh ``dispatch`` and
remote ``dispatch_rework``: the base issue record, the stale keys a success
drops and the ones it leaves standing, the orphan flags, the dead-worker failure
kind, the worker process stamp, Event order, save timing, label edges and
result data. Lane-only steps are pinned too: fresh's manual harness
(``manifest_written``) and escalation clear with its PR mirror-clear; rework's
PR reset, redispatch counter and cap. Every test drives the public entry points
with a scripted ``FakeWorkerLauncher`` and plants the stale keys while the launch
is in flight, so the success arm (not the claim before it) is what is observed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import charlie_work.workflow as wf
import pytest
from _launch_outcome_fixtures import (
    FRESH,
    LANES,
    REWORK,
    _amend_state,
    _app,
    _config,
    _launch,
    _run,
)
from charlie_work.config import OrchestratorConfig, WorkerRoleConfig
from charlie_work.instrumentation import query_events
from charlie_work.state import load_state

_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_ORPHANS = ("orphan_flagged_at", "orphan_drift_fingerprint", "orphan_drift_at")
_EPOCH_STALE: dict[str, Any] = {
    "orphan_flagged_at": "2026-01-01T00:00:00Z",
    "orphan_drift_fingerprint": "fp",
    "orphan_drift_at": "2026-01-01T00:00:00Z",
    "dead_worker_failure_kind": "provider_throttle",
    "dispatch_failed_at": ["2026-01-01T00:00:00+00:00"],
    "escalation_reason": "worker_blocked",
    "reason_class": "mechanical",
    "escalation_reasons_seen": ["worker_blocked"],
    "dispatch_pending_at": "2026-01-01T00:00:00Z",
    "label_error": {"edge": "stale", "outcome": "partial_failure"},
    "worker_pid": 1,
    "worker_process_start_time": 2.0,
    "kept_marker": "survives",
}


def _issue(app: Any) -> dict[str, Any]:
    return load_state(app.paths.state_file)["issues"]["123"]


def _plant(app: Any, lane: str, **extra: Any):
    """Plant stale keys (and the rework PR record) once the claim has run."""

    def _edit(state: dict[str, Any]) -> None:
        state["issues"]["123"].update({**_EPOCH_STALE, **extra})
        state.setdefault("prs", {})["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "open",
            "escalation_reason": "worker_blocked",
            "escalation_reasons_seen": ["worker_blocked"],
            "last_rework_failure_kind": "provider_throttle",
            "last_rework_was_startup_death": True,
            "kept_pr_marker": "survives",
        }

    return _amend_state(app, _edit)


class _Recorder:
    """Every ``save_state`` and ``transition`` on ``charlie_work.workflow``, in order."""

    def __init__(self, app: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[Any, ...]] = []
        real_save, real_transition = wf.save_state, wf.transition

        def save(path, state):
            issue = state["issues"]["123"]
            self.calls.append(("save", issue.get("status"), issue.get("worker_pid")))
            return real_save(path, state)

        def transition(gh, labels, number, edge, **kwargs):
            persisted = load_state(app.paths.state_file)["issues"]["123"]
            self.calls.append(("transition", edge, persisted.get("status")))
            return real_transition(gh, labels, number, edge, **kwargs)

        monkeypatch.setattr(wf, "save_state", save)
        monkeypatch.setattr(wf, "transition", transition)

    @property
    def after_claim(self) -> list[tuple[Any, ...]]:
        """Calls from the success outcome on (the claim saves before the launch)."""
        start = next(i for i, c in enumerate(self.calls) if c[0] == "save" and c[2] is not None)
        return self.calls[start:]


def _launched(app: Any, lane: str, fake_host: Any, monkeypatch: pytest.MonkeyPatch, **extra: Any):
    recorder = _Recorder(app, monkeypatch)
    _launch(
        fake_host,
        ok=True,
        pid=4242,
        process_start_time=1700000000.5,
        during=_plant(app, lane, **extra),
    )
    return _run(app, lane), recorder


# --- the base success record, both lanes ---------------------------------------


@LANES
def test_success_record_keys(tmp_path: Path, fake_host, monkeypatch, lane: str) -> None:
    """Every stored key after a launch: written, dropped, and stale-but-surviving."""
    app = _app(lane, tmp_path, config=_config(max_auto_redispatch=5))
    result, _ = _launched(app, lane, fake_host, monkeypatch)

    entry = _issue(app)
    assert result.ok is True
    assert entry["number"] == 123
    assert entry["title"] == "Fix search"
    assert entry["url"] == "https://example.test/issues/123"
    assert entry["branch_name"] == "agent/issue-123-fix-search"
    prompt = "worker-prompt.md" if lane == FRESH else "rework-prompt.md"
    assert Path(entry["prompt_path"]).name == prompt
    assert entry["status"] == "dispatched"
    assert _STAMP.match(entry["dispatched_at"])
    assert entry["worker_pid"] == 4242
    assert entry["worker_process_start_time"] == 1700000000.5
    # dropped by both lanes
    for key in ("dispatch_pending_at", "label_error", "dead_worker_failure_kind", *_ORPHANS):
        assert key not in entry, key
    assert entry["kept_marker"] == "survives"
    common = {
        "number", "title", "url", "branch_name", "prompt_path", "status",
        "dispatched_at", "worker_pid", "worker_process_start_time", "kept_marker",
    }  # fmt: skip
    if lane == FRESH:
        # fresh also supersedes the failure counter and any escalation
        assert set(entry) - common == set(), sorted(set(entry) - common)
    else:
        # rework leaves the failure counter and the escalation standing, and
        # appends this launch to the windowed redispatch counter
        assert entry["dispatch_failed_at"] == ["2026-01-01T00:00:00+00:00"]
        assert entry["escalation_reason"] == "worker_blocked"
        assert entry["reason_class"] == "mechanical"
        assert entry["escalation_reasons_seen"] == ["worker_blocked"]
        assert len(entry["redispatch_at"]) == 1
        assert set(entry) - common == {
            "dispatch_failed_at", "escalation_reason", "reason_class",
            "escalation_reasons_seen", "redispatch_at",
        }  # fmt: skip


@LANES
def test_success_record_overwrites_stale_identity(
    tmp_path: Path, fake_host, monkeypatch, lane: str
) -> None:
    """Title, url, branch and prompt come from this launch, not the stored record."""
    app = _app(lane, tmp_path, config=_config(max_auto_redispatch=5))
    _launched(
        app,
        lane,
        fake_host,
        monkeypatch,
        title="old",
        url="https://old.test",
        branch_name="old-branch",
        prompt_path="old/prompt.md",
    )

    entry = _issue(app)
    assert entry["title"] == "Fix search"
    assert entry["url"] == "https://example.test/issues/123"
    assert entry["branch_name"] == "agent/issue-123-fix-search"
    assert entry["prompt_path"] != "old/prompt.md"


@LANES
def test_success_without_pid_drops_previous_epoch_pid(
    tmp_path: Path, fake_host, monkeypatch, lane: str
) -> None:
    app = _app(lane, tmp_path, config=_config(max_auto_redispatch=5))
    _launch(fake_host, ok=True, pid=None, during=_plant(app, lane))

    _run(app, lane)

    entry = _issue(app)
    assert entry["status"] == "dispatched"
    assert "worker_pid" not in entry
    assert "worker_process_start_time" not in entry


@LANES
def test_success_uses_workflow_clock_and_stamp(
    tmp_path: Path, fake_host, monkeypatch, lane: str
) -> None:
    """``utc_now`` and ``stamp_worker_process`` are read from the lane at call time."""
    app = _app(lane, tmp_path, config=_config(max_auto_redispatch=5))
    monkeypatch.setattr(wf, "utc_now", lambda: "2031-02-03T04:05:06Z")
    _launch(fake_host, ok=True, pid=4242, during=_plant(app, lane))

    _run(app, lane)

    assert _issue(app)["dispatched_at"] == "2031-02-03T04:05:06Z"


@LANES
def test_success_events_saves_labels_and_result(
    tmp_path: Path, fake_host, monkeypatch, lane: str
) -> None:
    """Event kinds in order, save-before-label timing, label edge and result data."""
    app = _app(lane, tmp_path, config=_config(max_auto_redispatch=5))
    result, recorder = _launched(app, lane, fake_host, monkeypatch)

    edge = "dispatched" if lane == FRESH else "rework_dispatched"
    # The worker is persisted (one save) BEFORE the label transition, and the
    # transition sees the persisted record; the lane's closing save follows.
    assert recorder.after_claim == [
        ("save", "dispatched", 4242),
        ("transition", edge, "dispatched"),
        ("save", "dispatched", 4242),
    ]
    summary = "dispatch" if lane == FRESH else "dispatch_rework"
    kinds = [
        e["kind"]
        for e in query_events(app.paths.state_file)
        if e["kind"] in {"lifecycle_transition", summary, "session_failed_escalated"}
        or "escalat" in e["kind"]
        or "failed" in e["kind"]
    ]
    assert kinds == ["lifecycle_transition", summary]
    lifecycle = query_events(app.paths.state_file, kind="lifecycle_transition")
    assert len(lifecycle) == 1
    assert lifecycle[0]["issue_number"] == 123
    assert result.data["failed_count"] == 0
    assert result.data["label_errors"] == []
    assert result.data["failures"] == {}
    assert result.data["attempted_count"] == 1
    assert result.data["dispatch_results"][0]["pid"] == 4242


# --- fresh only -----------------------------------------------------------------


def test_fresh_success_clears_escalation_and_pr_mirror(
    tmp_path: Path, fake_host, monkeypatch
) -> None:
    app = _app(FRESH, tmp_path, config=_config(max_auto_redispatch=5))
    _launched(app, FRESH, fake_host, monkeypatch)

    state = load_state(app.paths.state_file)
    for key in ("escalation_reason", "reason_class", "escalation_reasons_seen"):
        assert key not in state["issues"]["123"], key
        assert key not in state["prs"]["456"], key
    assert state["prs"]["456"]["kept_pr_marker"] == "survives"
    # fresh never touches the rework resets
    assert state["prs"]["456"]["last_rework_failure_kind"] == "provider_throttle"
    assert state["prs"]["456"]["last_rework_was_startup_death"] is True
    assert "redispatch_at" not in state["issues"]["123"]


def test_fresh_manual_harness_records_manifest_written(
    tmp_path: Path, fake_host, monkeypatch
) -> None:
    config = OrchestratorConfig(worker=WorkerRoleConfig(harness="manual"))
    app = _app(FRESH, tmp_path, config=config)
    recorder = _Recorder(app, monkeypatch)

    result = app.dispatch(limit=1)

    entry = _issue(app)
    assert result.ok is True
    assert entry["status"] == "manifest_written"
    assert _STAMP.match(entry["dispatched_at"])
    assert "worker_pid" not in entry
    transitions = [c for c in recorder.calls if c[0] == "transition"]
    assert transitions == [("transition", "queued", "manifest_written")]


# --- rework only ----------------------------------------------------------------


def test_rework_success_resets_the_pr_failure_markers(
    tmp_path: Path, fake_host, monkeypatch
) -> None:
    app = _app(REWORK, tmp_path, config=_config(max_auto_redispatch=5))
    _launched(app, REWORK, fake_host, monkeypatch)

    pr = load_state(app.paths.state_file)["prs"]["456"]
    assert pr["last_rework_failure_kind"] is None
    assert pr["last_rework_was_startup_death"] is False
    # only those two keys change; the PR's escalation is not cleared by rework
    assert pr["kept_pr_marker"] == "survives"
    assert pr["escalation_reason"] == "worker_blocked"


def test_rework_success_without_pr_record_creates_none(
    tmp_path: Path, fake_host, monkeypatch
) -> None:
    app = _app(REWORK, tmp_path, config=_config(max_auto_redispatch=5))
    _launch(fake_host, ok=True, pid=4242)

    app.dispatch_rework()

    assert "456" not in load_state(app.paths.state_file).get("prs", {})


def test_rework_success_appends_to_the_windowed_redispatch_counter(
    tmp_path: Path, fake_host, monkeypatch
) -> None:
    old = "2020-01-01T00:00:00Z"  # outside the 240-minute window: pruned
    recent = wf.utc_now()
    app = _app(REWORK, tmp_path, config=_config(max_auto_redispatch=5))
    _launched(app, REWORK, fake_host, monkeypatch, redispatch_at=[old, recent])

    stored = _issue(app)["redispatch_at"]
    assert len(stored) == 2
    assert stored[0] == recent
    assert stored[1].endswith("Z") and "." in stored[1]
    assert stored[1] != old


def test_rework_success_past_the_cap_escalates_but_keeps_worker_facts(
    tmp_path: Path, fake_host, monkeypatch
) -> None:
    recent = wf.utc_now()
    app = _app(REWORK, tmp_path, config=_config(max_auto_redispatch=1))
    recorder = _Recorder(app, monkeypatch)
    _launch(
        fake_host,
        ok=True,
        pid=4242,
        process_start_time=1700000000.5,
        during=_plant(app, REWORK, redispatch_at=[recent]),
    )

    result = app.dispatch_rework()

    entry = _issue(app)
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "redispatch_cap_exceeded"
    assert entry["reason_class"] == "mechanical"
    assert entry["worker_pid"] == 4242
    assert entry["worker_process_start_time"] == 1700000000.5
    assert entry["branch_name"] == "agent/issue-123-fix-search"
    assert len(entry["redispatch_at"]) == 2
    pr = load_state(app.paths.state_file)["prs"]["456"]
    assert pr["last_rework_failure_kind"] is None
    assert pr["last_rework_was_startup_death"] is False
    transitions = [c for c in recorder.calls if c[0] == "transition"]
    assert [c[1] for c in transitions] == ["redispatch_operator_queued"]
    assert transitions[0][2] == "escalated"
    assert result.data["label_errors"] == []


@pytest.mark.parametrize("pid", [4242, None])
def test_rework_success_past_the_cap_full_stored_record(
    tmp_path: Path, fake_host, monkeypatch, pid: int | None
) -> None:
    """The escalated record merges over the STORED one: popped keys survive.

    The success arm's pops act on a copy that ``_escalate_issue`` never sees, so
    the stale orphan flags, ``dead_worker_failure_kind`` and ``label_error`` stay,
    and with no pid from the launch the previous ``worker_pid`` stays too.
    """
    recent = wf.utc_now()
    app = _app(REWORK, tmp_path, config=_config(max_auto_redispatch=1))
    _launch(
        fake_host,
        ok=True,
        pid=pid,
        process_start_time=1700000000.5 if pid else None,
        during=_plant(app, REWORK, redispatch_at=[recent]),
    )

    app.dispatch_rework()

    entry = _issue(app)
    for key in (*_ORPHANS, "dead_worker_failure_kind", "dispatch_failed_at", "kept_marker"):
        assert entry[key] == {**_EPOCH_STALE}.get(key, "survives"), key
    assert entry["dispatch_pending_at"] == _EPOCH_STALE["dispatch_pending_at"]
    assert entry["label_error"] == _EPOCH_STALE["label_error"]
    assert entry["worker_pid"] == (4242 if pid else 1)
    assert entry["worker_process_start_time"] == (1700000000.5 if pid else 2.0)
    assert sorted(entry) == sorted(
        {
            "number", "title", "url", "branch_name", "prompt_path", "status",
            "dispatched_at", "worker_pid", "worker_process_start_time", "kept_marker",
            "redispatch_at", "escalation_reason", "reason_class", "escalation_reasons_seen",
            "merge_alert", "terminal_since", *_ORPHANS,
            "dead_worker_failure_kind", "dispatch_failed_at", "dispatch_pending_at",
            "label_error",
        }
    )  # fmt: skip
