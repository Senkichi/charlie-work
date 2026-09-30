"""Regression for the wf-r3 review-1 blocker B1: a terminal record left by an
earlier dispatch must not decide a later dispatch's dead-worker route.

``find_worker_terminal_status`` returns the newest ``issue-<n>.*.terminal.json``
and nothing ever deletes one, so after a redispatch whose own watcher never ran
(devin-shell, self-deploy mid-run) the previous attempt's ``exit_code == 0`` was
still read and routed the death down ``dead_worker_clean_exit_no_op``. Rule 1
(design §3): the record's ``ended_at`` must be ``> dispatched_at`` first.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from _orphan_sweep_fixtures import _dead_worker_rework_bed, _run_orphan_sweep
from charlie_work import worker_fate
from charlie_work.orphaned_worker_sweep import maybe_reap_dead_dispatched_worker
from charlie_work.process_utils import write_worker_terminal_status
from charlie_work.state import load_state, parse_iso_timestamp


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _write_terminal(tmp_path: Path, *, ended_ago: timedelta, exit_code: int = 0) -> None:
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    write_worker_terminal_status(
        sessions_dir / "issue-207.claude-code.terminal.json",
        pid=4242,
        exit_code=exit_code,
        started_at=_iso(now - ended_ago - timedelta(minutes=5)),
        ended_at=_iso(now - ended_ago),
        duration_seconds=300.0,
    )


def _drift_reasons(state: dict[str, Any]) -> list[str]:
    return [
        e["payload"]["reason"]
        for e in state.get("events", [])
        if e.get("kind") == "orphaned_worker_drift"
    ]


def test_stale_exit_zero_record_does_not_route_to_clean_exit_no_op(tmp_path: Path) -> None:
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="request_changes"
    )
    # The bed dispatched one hour ago; this record ended three hours ago.
    _write_terminal(tmp_path, ended_ago=timedelta(hours=3))

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    assert state["issues"]["207"]["status"] == "rework_requested"
    assert "dead_worker_clean_exit_no_op" not in _drift_reasons(state)


def test_fresh_exit_zero_record_still_routes_to_clean_exit_no_op(tmp_path: Path) -> None:
    """Positive control: the same bed with a record from THIS dispatch keeps #773."""
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="request_changes"
    )
    _write_terminal(tmp_path, ended_ago=timedelta(minutes=5))

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    assert state["issues"]["207"]["status"] == "dispatched"
    assert _drift_reasons(state) == ["dead_worker_clean_exit_no_op"]


def test_dead_dispatched_reap_ignores_a_stale_terminal_exit_code(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_terminal(tmp_path, ended_ago=timedelta(hours=3))
    entry = {
        "status": "dispatched",
        "worker_pid": 99999,
        "dispatched_at": _iso(now - timedelta(hours=1)),
        "orphan_drift_at": _iso(now - timedelta(hours=2)),
    }
    events: list[tuple[str, dict[str, Any]]] = []

    _state, reaped = maybe_reap_dead_dispatched_worker(
        state={"issues": {"207": dict(entry)}, "prs": {}, "events": []},
        entry=entry,
        issue_number=207,
        sessions_dir=sessions_dir,
        pr_data=None,
        dead_dispatched_reap_minutes=60,
        now=now,
        sweep_events=events,
        max_throttle_rearms=0,
    )

    assert reaped is True
    payloads = [p for kind, p in events if kind == "dead_dispatched_worker_reaped"]
    assert [p["exit_code"] for p in payloads] == [None]


def test_fresh_terminal_record_freshness_rule() -> None:
    dispatched = parse_iso_timestamp("2026-09-29T10:00:00Z")
    stale = {"ended_at": "2026-09-29T09:00:00Z", "exit_code": 0}
    fresh = {"ended_at": "2026-09-29T10:05:00Z", "exit_code": 0}

    assert worker_fate.fresh_terminal_record(stale, dispatched) is None
    assert worker_fate.fresh_terminal_record(fresh, dispatched) is fresh
    # An unparseable ended_at cannot prove freshness.
    assert worker_fate.fresh_terminal_record({"exit_code": 0}, dispatched) is None
    # Legacy entry without dispatched_at accepts unconditionally.
    assert worker_fate.fresh_terminal_record(stale, None) is stale
    assert worker_fate.fresh_terminal_record(None, dispatched) is None


def test_n1_stderr_throttle_line_after_quoting_event_anchors_at_now(tmp_path: Path) -> None:
    """N1: the newest marker-bearing line is merged stderr, so an older user
    ``tool_result`` that merely quotes "rate limit" must not become the anchor
    (09:10 + cooldown is already past, which collapsed the window to zero)."""
    now = datetime(2026, 9, 29, 10, 0, 0, tzinfo=UTC)
    lines = [
        json.dumps(
            {
                "type": "user",
                "timestamp": "2026-09-29T09:10:00Z",
                "message": {"content": [{"type": "tool_result", "content": "hit a rate limit"}]},
            }
        ),
        json.dumps({"type": "assistant", "timestamp": "2026-09-29T09:11:00Z", "message": "ok"}),
        "API Error: 429 rate limit exceeded",
        json.dumps({"type": "result", "subtype": "error", "result": "done"}),
    ]
    log_path = tmp_path / "session.log"
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mtime = (now - timedelta(minutes=2)).timestamp()
    os.utime(log_path, (mtime, mtime))

    failure_kind, throttled_until = worker_fate.classify_for("api", log_path, now=now)

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    assert parse_iso_timestamp(throttled_until) > now


def test_n3_is_alive_reaches_a_patched_process_utils_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("charlie_work.process_utils.is_pid_alive", lambda pid, start: True)

    assert worker_fate.is_alive(2_999_999, None) is True
