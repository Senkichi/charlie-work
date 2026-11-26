"""Issue #3141: one repo's long state-lock hold must not freeze the other repos' launch gates.

Two defects shared a root cause, fixed together:

1. The fleet-wide live census (``fleet_live_counts.
   count_fleet_live_sessions_with_adapter_kinds`` and
   ``fleet_registry.count_fleet_live_reviews``) read every registered repo's
   ``state.json`` through ``load_state_locked`` while holding the fleet
   launch lock -- so a lane whose OWN reconcile held its state lock for a
   long time froze every other lane's census (and therefore its launch
   gate) for the whole hold. The census now passes
   ``lock_free_state_read=True``: state writes are atomic replaces, so the
   lock-free ``load_state`` sees the last committed snapshot, which is all
   a census needs.
2. ``advisory_file_lock``'s intra-process thread-lock stage was a bare
   ``acquire()`` -- no timeout, no log line -- so the same-process wait
   above waited forever and said nothing. It now shares the single
   ``_LOCK_TIMEOUT_SECONDS`` budget with the cross-process file-lock stage
   and raises ``StateLockBusy`` with the same warning on timeout.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from charlie_work import layout
from charlie_work import live_session_count
from charlie_work import state as state_module
from charlie_work.fleet_live_counts import count_fleet_live_sessions_with_adapter_kinds
from charlie_work.fleet_registry import count_fleet_live_reviews
from charlie_work.live_session_count import WORKER_LANE, count_live_sessions


class _SidecarStub:
    """Minimal ``iter_workers`` record: an issue number plus a liveness flag."""

    def __init__(
        self, issue_number: int, alive: bool = True, adapter_kind: str = "claude-code"
    ) -> None:
        self.issue_number = issue_number
        self.adapter_kind = adapter_kind
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


_OWN_PID = os.getpid()


def _register_repo(fleet_dir: Path, tmp_path: Path, name: str) -> Path:
    """Register ``owner/<name>`` in the fleet registry with one ghost worker
    and one ghost reviewer in its ``state.json``; return the state file path.

    A ghost is a dispatched state record whose recorded pid is alive (this
    test process -- a ``None`` start time fails open on the identity check)
    with no sidecar -- the corroboration path that used to read through the
    repo's held state lock.
    """
    repo_root = tmp_path / name
    (repo_root / ".git").mkdir(parents=True)
    state_dir = repo_root / ".var" / "charlie-work"
    layout.sessions_dir_default(state_dir).mkdir(parents=True)
    layout.reviews_dir_default(state_dir).mkdir(parents=True)
    state_file = layout.state_file_path(state_dir)
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {
                    "7": {
                        "status": "dispatched",
                        "worker_pid": _OWN_PID,
                        "worker_process_start_time": None,
                    }
                },
                "prs": {
                    "900": {
                        "review_dispatch_status": "review_dispatch_dispatched",
                        "reviewer_pid": _OWN_PID,
                        "reviewer_process_start_time": None,
                    }
                },
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    fleet_dir.mkdir(parents=True, exist_ok=True)
    fleet_json = fleet_dir / "fleet.json"
    registry = (
        json.loads(fleet_json.read_text(encoding="utf-8"))
        if fleet_json.exists()
        else {"version": 1, "repos": {}}
    )
    registry["repos"][f"owner/{name}"] = {
        "repo_root": str(repo_root),
        "name_with_owner": f"owner/{name}",
        "config_path": str(repo_root / "orchestrator.config.yaml"),
        "state_dir": str(state_dir),
        "first_seen": "2024-01-01T00:00:00Z",
        "last_seen": "2024-01-01T00:00:00Z",
    }
    fleet_json.write_text(json.dumps(registry), encoding="utf-8")
    return state_file


def _hold_thread_lock_briefly(state_file: Path, seconds: float) -> threading.Timer:
    """Hold ``state_file``'s intra-process lock, auto-releasing after ``seconds``.

    The bounded hold doubles as the mutation gate: code that still waits on
    the lock acquires it when the timer fires instead of deadlocking the
    suite, and the elapsed-time assertion below then fails the test.
    """
    held = state_module._thread_lock_for(state_file)
    held.acquire()
    release = threading.Timer(seconds, held.release)
    release.start()
    return release


def _release_held_thread_lock(state_file: Path, release: threading.Timer) -> None:
    release.cancel()
    held = state_module._thread_lock_for(state_file)
    if held.locked():
        held.release()


def test_fleet_live_review_census_does_not_wait_on_a_held_state_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``count_fleet_live_reviews`` counts a repo's live sidecars and ghosts
    from its last committed state WITHOUT waiting on that repo's held state
    lock. The lock is released by timer so unfixed code (which blocks until
    the release) is caught by the elapsed assertion, not by a deadlock."""
    fleet_dir = tmp_path / "fleet"
    state_file = _register_repo(fleet_dir, tmp_path, "a")
    monkeypatch.setattr(
        "charlie_work.worker.iter_workers",
        lambda _dir, **_kw: [_SidecarStub(1)],
    )

    release = _hold_thread_lock_briefly(state_file, 2.0)
    try:
        started = time.monotonic()
        count, skipped = count_fleet_live_reviews(str(fleet_dir))
        elapsed = time.monotonic() - started
    finally:
        _release_held_thread_lock(state_file, release)

    assert elapsed < 1.0, (
        "fleet live-review census waited on the repo's held state lock "
        f"({elapsed:.2f}s) -- the ghost-corroboration read must be lock-free"
    )
    assert skipped == []
    assert count == 2  # one live review sidecar + one ghost reviewer


def test_fleet_live_session_census_does_not_wait_on_a_held_state_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``count_fleet_live_sessions_with_adapter_kinds`` is the worker-lane
    twin: same held lock, same lock-free census read."""
    fleet_dir = tmp_path / "fleet"
    state_file = _register_repo(fleet_dir, tmp_path, "a")
    monkeypatch.setattr(
        "charlie_work.worker.iter_workers",
        lambda _dir, **_kw: [_SidecarStub(1)],
    )

    release = _hold_thread_lock_briefly(state_file, 2.0)
    try:
        started = time.monotonic()
        total, kinds, skipped = count_fleet_live_sessions_with_adapter_kinds(str(fleet_dir))
        elapsed = time.monotonic() - started
    finally:
        _release_held_thread_lock(state_file, release)

    assert elapsed < 1.0, (
        "fleet live-session census waited on the repo's held state lock "
        f"({elapsed:.2f}s) -- the ghost-corroboration read must be lock-free"
    )
    assert skipped == []
    assert total == 2  # one live session sidecar + one ghost worker
    assert kinds == {"claude-code": 1}  # a ghost has no adapter kind


def test_count_live_sessions_default_read_is_still_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Callers that do not pass ``lock_free_state_read`` keep the locked read:
    the flag is opt-in for the fleet census, not a behavior change for the
    per-repo cap paths."""
    monkeypatch.setattr("charlie_work.worker.iter_workers", lambda _dir, **_kw: [])
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )

    locked_reads: list[Path] = []
    monkeypatch.setattr(
        live_session_count,
        "load_state_locked",
        lambda path: locked_reads.append(path) or {"issues": {}, "prs": {}},
    )

    def _no_unlocked_read(_path: Path) -> dict[str, Any]:
        raise AssertionError("default count_live_sessions read state without the lock")

    monkeypatch.setattr(state_module, "load_state", _no_unlocked_read)

    assert count_live_sessions(tmp_path / "sessions", state_file, WORKER_LANE) == 0
    assert locked_reads == [state_file]


def test_count_live_sessions_lock_free_read_uses_load_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``lock_free_state_read=True`` selects ``load_state`` and never touches
    the locked read."""
    monkeypatch.setattr("charlie_work.worker.iter_workers", lambda _dir, **_kw: [])
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )

    free_reads: list[Path] = []
    monkeypatch.setattr(
        state_module,
        "load_state",
        lambda path: free_reads.append(path) or {"issues": {}, "prs": {}},
    )

    def _no_locked_read(_path: Path) -> dict[str, Any]:
        raise AssertionError("lock_free_state_read took the state lock anyway")

    monkeypatch.setattr(live_session_count, "load_state_locked", _no_locked_read)

    assert (
        count_live_sessions(
            tmp_path / "sessions", state_file, WORKER_LANE, lock_free_state_read=True
        )
        == 0
    )
    assert free_reads == [state_file]


def test_thread_lock_wait_is_bounded_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The intra-process thread-lock stage honours the SAME
    ``_LOCK_TIMEOUT_SECONDS`` budget as the file-lock stage: on timeout it
    logs the identical "Failed to acquire lock" warning and raises
    ``StateLockBusy`` instead of waiting forever and saying nothing."""
    monkeypatch.setattr(state_module, "_LOCK_TIMEOUT_SECONDS", 0.3)
    state_path = tmp_path / "state.json"

    release = _hold_thread_lock_briefly(state_path, 2.0)
    try:
        with caplog.at_level(logging.WARNING, logger=state_module.__name__):
            started = time.monotonic()
            with pytest.raises(state_module.StateLockBusy):
                with state_module.advisory_file_lock(state_path):
                    pass
        elapsed = time.monotonic() - started
    finally:
        _release_held_thread_lock(state_path, release)

    assert elapsed < 1.0, (
        f"same-process lock wait ran {elapsed:.2f}s -- the thread-lock stage "
        "must be bounded by _LOCK_TIMEOUT_SECONDS, not by the holder"
    )
    assert any("Failed to acquire lock" in record.message for record in caplog.records), (
        "the thread-lock timeout logged no warning -- the silent-freeze defect"
    )


def test_thread_and_file_lock_stages_share_one_timeout_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Total wait across BOTH stages never exceeds ``_LOCK_TIMEOUT_SECONDS``:
    a thread-lock wait that consumes most of the budget leaves the file-lock
    retry loop only the remainder -- not a fresh full budget."""
    timeout = 1.0
    monkeypatch.setattr(state_module, "_LOCK_TIMEOUT_SECONDS", timeout)
    state_path = tmp_path / "state.json"

    def _always_busy(*_a: Any, **_k: Any) -> None:
        raise OSError("lock held (test stub)")

    if sys.platform == "win32":
        import msvcrt

        monkeypatch.setattr(msvcrt, "locking", _always_busy)
    else:
        import fcntl

        monkeypatch.setattr(fcntl, "flock", _always_busy)

    # Hold the thread lock for ~70% of the budget; the file stage then has
    # only ~30% left, so the raise lands near ``timeout`` after entry -- not
    # at ``hold + timeout`` as a two-budget implementation would produce.
    hold_seconds = 0.7
    release = _hold_thread_lock_briefly(state_path, hold_seconds)
    try:
        started = time.monotonic()
        with pytest.raises(state_module.StateLockBusy):
            with state_module.advisory_file_lock(state_path):
                pass
        elapsed = time.monotonic() - started
    finally:
        _release_held_thread_lock(state_path, release)

    assert elapsed >= 0.9 * timeout, (
        f"timed out after only {elapsed:.2f}s -- the waiter must still get "
        "the whole shared budget before failing"
    )
    assert elapsed < timeout + hold_seconds, (
        f"total wait {elapsed:.2f}s exceeded one shared budget -- the file "
        "stage must reuse the thread stage's start, not get its own timeout"
    )


def test_thread_lock_acquired_within_timeout_behaves_normally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A waiter whose thread-lock wait fits the budget proceeds exactly as
    today: the context manager yields with the file lock held."""
    monkeypatch.setattr(state_module, "_LOCK_TIMEOUT_SECONDS", 1.0)
    state_path = tmp_path / "state.json"

    release = _hold_thread_lock_briefly(state_path, 0.2)
    try:
        entered = False
        with state_module.advisory_file_lock(state_path):
            entered = True
    finally:
        _release_held_thread_lock(state_path, release)

    assert entered, "advisory_file_lock did not yield after a within-budget wait"
