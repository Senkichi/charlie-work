"""Issue #2104: a DEAD worker that already handed off is not labelled ``stalled``.

Drives ``_detect_and_handle_stalled_sessions`` end to end. The control is a live,
non-progressing worker, which keeps ``failure_kind="stalled"``.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from _worker_fixtures import _make_stalled_devin_session, _stale_devin_probe, _wg
from charlie_work import workflow
from charlie_work.config import OrchestratorConfig, PostMortemConfig
from charlie_work.dead_worker_sweep import effects_sessions

ISSUE = 2104


def _iso(delta: timedelta) -> str:
    return (datetime.now(UTC) + delta).isoformat().replace("+00:00", "Z")


def _setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    alive: bool,
    outcome: bool | dict = False,
    outcome_age: timedelta = timedelta(0),
    terminal_exit_code: int | None = None,
):
    """``outcome``: True = declared push, dict = literal payload; ``outcome_age`` backdates it."""
    sessions_dir, state_file, _log = _make_stalled_devin_session(
        tmp_path, ISSUE, "Doing work.\n.worker-outcome.json is written\n"
    )
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    if outcome:
        payload = {"push_succeeded": True, "pr_created": False} if outcome is True else outcome
        outcome_path = worktree / ".worker-outcome.json"
        outcome_path.write_text(json.dumps(payload), encoding="utf-8")
        if outcome_age:
            stamp = (datetime.now(UTC) - outcome_age).timestamp()
            os.utime(outcome_path, (stamp, stamp))
    if terminal_exit_code is not None:
        (sessions_dir / f"issue-{ISSUE}.devin.terminal.json").write_text(
            json.dumps({"exit_code": terminal_exit_code, "ended_at": _iso(timedelta(minutes=-1))}),
            encoding="utf-8",
        )
    dispatched = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    state_file.write_text(
        json.dumps({"events": [], "issues": {str(ISSUE): {"dispatched_at": dispatched}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("charlie_work.worker_fate.is_alive", lambda *_: alive)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_devin_probe)
    monkeypatch.setattr("charlie_work.write_gate.kill_process_tree", lambda *_a, **_k: [])
    monkeypatch.setattr(effects_sessions, "sweep_orphan_processes", lambda _wt: [])
    config = OrchestratorConfig(post_mortem=PostMortemConfig(db_path=str(tmp_path / "none.db")))
    workflow._detect_and_handle_stalled_sessions(
        sessions_dir, state_file, config, write_gate=_wg(state_file)
    )
    return json.loads(state_file.read_text(encoding="utf-8"))


def _reap_event(state: dict, kind: str) -> dict:
    (event,) = [e for e in state["events"] if e["kind"] == kind]
    return event["payload"]


def test_dead_completed_worker_is_not_stamped_stalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _setup(tmp_path, monkeypatch, alive=False, outcome=True)
    payload = _reap_event(state, "session_exited")
    assert payload["worker_health"] == "DEAD"
    assert payload["killed_pids"] == []
    assert payload["failure_kind"] != "stalled"
    assert "dead_worker_failure_kind" not in state["issues"][str(ISSUE)]


def test_dead_worker_without_a_handoff_still_gets_the_stalled_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _setup(tmp_path, monkeypatch, alive=False, outcome=False)
    assert _reap_event(state, "session_exited")["failure_kind"] == "stalled"


def test_live_stalled_worker_keeps_the_stalled_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _setup(tmp_path, monkeypatch, alive=True, outcome=True)
    payload = _reap_event(state, "session_stalled")
    assert payload["worker_health"] == "STALLED"
    assert payload["failure_kind"] == "stalled"


def test_dead_worker_with_clean_terminal_exit_is_not_stamped_stalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _setup(tmp_path, monkeypatch, alive=False, outcome=False, terminal_exit_code=0)
    payload = _reap_event(state, "session_exited")
    assert payload["failure_kind"] != "stalled"
    assert "dead_worker_failure_kind" not in state["issues"][str(ISSUE)]


def test_dead_worker_with_stale_outcome_still_gets_the_stalled_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A leftover .worker-outcome.json from a prior dispatch (older than dispatched_at).
    state = _setup(
        tmp_path, monkeypatch, alive=False, outcome=True, outcome_age=timedelta(hours=2)
    )
    assert _reap_event(state, "session_exited")["failure_kind"] == "stalled"


def test_dead_worker_with_blocked_outcome_still_gets_the_stalled_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = {
        "outcome": "blocked",
        "reason_kind": "ambiguous_scope",
        "detail": "needs a human",
        "push_succeeded": True,
    }
    state = _setup(tmp_path, monkeypatch, alive=False, outcome=blocked)
    assert _reap_event(state, "session_exited")["failure_kind"] == "stalled"
