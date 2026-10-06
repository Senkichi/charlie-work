"""Direct unit tests for ``live_session_count.count_live_sessions`` (issue #2230).

The shared counter is exercised end-to-end elsewhere (the worker lane through
``fleet_registry.count_fleet_live_sessions`` in
``test_fleet_registry_ghost_corroboration.py``); these tests pin the per-lane
contract at the function itself: a ``LiveSessionKind`` drives which state map,
status predicate and pid fields the ghost corroboration consults, a counted
ghost prints a ``[reconcile]`` line, and a sidecar-alive session is never
double-counted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from charlie_work.host.fakes import FakeProcessProbe
from charlie_work.live_session_count import (
    REVIEW_LANE,
    WORKER_LANE,
    count_live_sessions,
)


class _SidecarStub:
    """Minimal ``iter_workers`` record: an issue number plus a liveness flag."""

    def __init__(self, issue_number: int, alive: bool = True) -> None:
        self.issue_number = issue_number
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def _no_sidecars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("charlie_work.worker.iter_workers", lambda _dir, **_kw: [])


def _write_state(state_file: Path, **maps: Any) -> None:
    payload: dict[str, Any] = {"version": 1, "issues": {}, "prs": {}, "events": []}
    payload.update(maps)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(payload), encoding="utf-8")


def test_worker_lane_counts_ghost_and_reports_reconcile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_host: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """WORKER_LANE: a ``dispatched`` issue whose ``worker_pid`` is alive but has
    no live sidecar counts once and prints the ``[reconcile]`` ghost line."""
    _no_sidecars(monkeypatch)
    fake_host(probe=FakeProcessProbe({os.getpid(): None}))
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        issues={
            "7": {
                "status": "dispatched",
                "worker_pid": os.getpid(),
                "worker_process_start_time": None,
            }
        },
    )

    assert count_live_sessions(tmp_path / "sessions", state_file, WORKER_LANE) == 1
    out = capsys.readouterr().out
    assert "[reconcile] issue 7" in out
    assert "ghost" in out


def test_worker_lane_ignores_non_dispatched_and_dead_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_host: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """WORKER_LANE: entries not marked ``dispatched`` and dispatched entries
    whose recorded pid is dead (or absent) are not counted."""
    _no_sidecars(monkeypatch)
    live_pid = os.getpid()
    fake_host(probe=FakeProcessProbe({live_pid: None}))
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        issues={
            "8": {  # not dispatched
                "status": "in-progress",
                "worker_pid": live_pid,
                "worker_process_start_time": None,
            },
            "9": {  # dispatched but pid is dead (not in the probe map)
                "status": "dispatched",
                "worker_pid": 424242,
                "worker_process_start_time": None,
            },
            "10": {  # dispatched but no pid recorded at all
                "status": "dispatched",
            },
        },
    )

    assert count_live_sessions(tmp_path / "sessions", state_file, WORKER_LANE) == 0
    assert "[reconcile]" not in capsys.readouterr().out


def test_worker_lane_sidecar_alive_session_not_double_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_host: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """WORKER_LANE: a session with both a live sidecar and a dispatched state
    record counts once -- the corroboration pass skips sidecar-live numbers."""
    monkeypatch.setattr(
        "charlie_work.worker.iter_workers",
        lambda _dir, **_kw: [_SidecarStub(7)],
    )
    fake_host(probe=FakeProcessProbe({os.getpid(): None}))
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        issues={
            "7": {
                "status": "dispatched",
                "worker_pid": os.getpid(),
                "worker_process_start_time": None,
            }
        },
    )

    assert count_live_sessions(tmp_path / "sessions", state_file, WORKER_LANE) == 1
    assert "[reconcile]" not in capsys.readouterr().out


def test_review_lane_counts_ghost_and_reports_reconcile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_host: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REVIEW_LANE: a ``review_dispatch_dispatched`` PR whose ``reviewer_pid``
    is alive but has no live sidecar counts once, with the PR-shaped line."""
    _no_sidecars(monkeypatch)
    fake_host(probe=FakeProcessProbe({os.getpid(): None}))
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        prs={
            "11": {
                "review_dispatch_status": "review_dispatch_dispatched",
                "reviewer_pid": os.getpid(),
                "reviewer_process_start_time": None,
            }
        },
    )

    assert count_live_sessions(tmp_path / "reviews", state_file, REVIEW_LANE) == 1
    out = capsys.readouterr().out
    assert "[reconcile] PR 11" in out
    assert "review" in out


def test_review_lane_ignores_non_dispatched_and_dead_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_host: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REVIEW_LANE: entries not marked ``review_dispatch_dispatched`` and
    dispatched entries whose recorded pid is dead are not counted."""
    _no_sidecars(monkeypatch)
    fake_host(probe=FakeProcessProbe({os.getpid(): None}))
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        prs={
            "12": {
                "review_dispatch_status": "review_dispatch_done",
                "reviewer_pid": os.getpid(),
                "reviewer_process_start_time": None,
            },
            "13": {
                "review_dispatch_status": "review_dispatch_dispatched",
                "reviewer_pid": 424242,
                "reviewer_process_start_time": None,
            },
        },
    )

    assert count_live_sessions(tmp_path / "reviews", state_file, REVIEW_LANE) == 0
    assert "[reconcile]" not in capsys.readouterr().out


def test_review_lane_sidecar_alive_session_not_double_counted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_host: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REVIEW_LANE: the sidecar pass keys on ``issue_number`` regardless of
    lane, so a live review sidecar plus its dispatch record still counts once."""
    monkeypatch.setattr(
        "charlie_work.worker.iter_workers",
        lambda _dir, **_kw: [_SidecarStub(11)],
    )
    fake_host(probe=FakeProcessProbe({os.getpid(): None}))
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        prs={
            "11": {
                "review_dispatch_status": "review_dispatch_dispatched",
                "reviewer_pid": os.getpid(),
                "reviewer_process_start_time": None,
            }
        },
    )

    assert count_live_sessions(tmp_path / "reviews", state_file, REVIEW_LANE) == 1
    assert "[reconcile]" not in capsys.readouterr().out


def test_no_state_file_counts_sidecars_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``state_file=None`` the corroboration pass is skipped entirely:
    only sidecar liveness counts."""
    monkeypatch.setattr(
        "charlie_work.worker.iter_workers",
        lambda _dir, **_kw: [_SidecarStub(1), _SidecarStub(2, alive=False)],
    )

    assert count_live_sessions(tmp_path / "sessions", None, WORKER_LANE) == 1
