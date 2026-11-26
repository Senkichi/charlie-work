"""Issue #3141: one repo's long state-lock hold must not freeze the other repos' launch gates.

Two defects shared a root cause, fixed together:

1. The fleet-wide live census (``fleet_live_counts.
   count_fleet_live_sessions_with_adapter_kinds`` and
   ``fleet_registry.count_fleet_live_reviews``) read every registered repo's
   ``state.json`` through ``load_state_locked`` while holding the fleet
   launch lock -- so a lane whose OWN reconcile held its state lock for a
   long time froze every other lane's census (and therefore its launch
   gate) for the whole hold. The census now reads through
   ``state.read_state_snapshot`` (via ``live_session_count.
   read_fleet_census_state``): a lock-free, non-mutating reader -- state
   writes are atomic replaces, so it sees the last committed snapshot,
   which is all a census needs. Unlike ``load_state`` it never quarantines
   the file it reads and reports an unreadable file as ``None`` instead of
   an empty state, so the census marks the repo skipped (with a warning +
   ``fleet_census_state_unreadable`` event) and counts only its live
   sidecars rather than silently counting 0 ghosts.
2. ``advisory_file_lock``'s intra-process thread-lock stage was a bare
   ``acquire()`` -- no timeout, no log line -- so the same-process wait
   above waited forever and said nothing. It now shares the single
   ``_LOCK_TIMEOUT_SECONDS`` budget with the cross-process file-lock stage
   and raises ``StateLockBusy`` with the same warning on timeout; the
   file-lock stage also always runs at least one acquisition attempt even
   when the thread stage consumed the whole budget.
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
from charlie_work.instrumentation import query_events
from charlie_work.live_session_count import (
    WORKER_LANE,
    count_live_sessions,
    read_fleet_census_state,
)
from charlie_work.state import append_event, load_state, save_state, state_lock


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
    """Callers that pass no ``state_snapshot`` keep the locked read: the
    snapshot is opt-in for the fleet census, not a behavior change for the
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
    monkeypatch.setattr(state_module, "read_state_snapshot", _no_unlocked_read)

    assert count_live_sessions(tmp_path / "sessions", state_file, WORKER_LANE) == 0
    assert locked_reads == [state_file]


def test_count_live_sessions_state_snapshot_read_is_lock_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provided ``state_snapshot`` is used verbatim: no locked read and no
    state read of any kind happens inside the counter."""
    monkeypatch.setattr("charlie_work.worker.iter_workers", lambda _dir, **_kw: [])
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )

    def _no_read(_path: Path) -> dict[str, Any]:
        raise AssertionError("state_snapshot was provided but a state read ran anyway")

    monkeypatch.setattr(live_session_count, "load_state_locked", _no_read)
    monkeypatch.setattr(state_module, "load_state", _no_read)
    monkeypatch.setattr(state_module, "read_state_snapshot", _no_read)

    snapshot = {
        "issues": {
            "9": {
                "status": "dispatched",
                "worker_pid": _OWN_PID,
                "worker_process_start_time": None,
            }
        },
        "prs": {},
    }
    assert (
        count_live_sessions(
            tmp_path / "sessions", state_file, WORKER_LANE, state_snapshot=snapshot
        )
        == 1
    )  # the ghost worker from the provided snapshot


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


def test_exhausted_budget_still_gets_one_file_lock_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file-lock stage runs at least ONE acquisition attempt even when
    the shared budget is already gone at stage entry.

    A successful ``thread_lock.acquire(timeout=...)`` always returns before
    the deadline, so the only entry shape with zero budget left is a zero
    ``_LOCK_TIMEOUT_SECONDS`` (non-blocking probe): the thread stage passes
    instantly and the file stage used to run its ``while elapsed <
    timeout`` loop ZERO times before raising. The attempt-first loop
    guarantees one real attempt instead.
    """
    monkeypatch.setattr(state_module, "_LOCK_TIMEOUT_SECONDS", 0.0)
    state_path = tmp_path / "state.json"

    attempts: list[int] = []

    def _record_busy(*_a: Any, **_k: Any) -> None:
        attempts.append(1)
        raise OSError("lock held (test stub)")

    if sys.platform == "win32":
        import msvcrt

        monkeypatch.setattr(msvcrt, "locking", _record_busy)
    else:
        import fcntl

        monkeypatch.setattr(fcntl, "flock", _record_busy)

    with pytest.raises(state_module.StateLockBusy):
        with state_module.advisory_file_lock(state_path):
            pass

    assert len(attempts) >= 1, (
        "the file-lock stage was skipped entirely once the thread-lock stage "
        "consumed the budget -- it must always make at least one attempt"
    )


def test_read_state_snapshot_returns_none_on_unreadable_state(tmp_path: Path) -> None:
    """``read_state_snapshot`` reports failure instead of returning empty
    state or quarantining: corrupt JSON and non-dict JSON both yield None,
    and the file itself is left exactly as found (issue #3141)."""
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{ not json", encoding="utf-8")
    assert state_module.read_state_snapshot(corrupt) is None
    assert corrupt.exists() and corrupt.read_text(encoding="utf-8") == "{ not json", (
        "read_state_snapshot must never rename/mutate the file it samples"
    )
    assert not list(tmp_path.glob("*.corrupt-*"))

    non_dict = tmp_path / "list.json"
    non_dict.write_text("[1, 2]", encoding="utf-8")
    assert state_module.read_state_snapshot(non_dict) is None
    assert non_dict.exists()


def test_read_state_snapshot_missing_and_healthy_files(tmp_path: Path) -> None:
    """Missing file -> ``empty_state()`` (a repo that has not run); healthy
    file -> the parsed, normalized state."""
    missing = tmp_path / "missing" / "state.json"
    assert state_module.read_state_snapshot(missing) == state_module.empty_state()

    state_file = tmp_path / "state.json"
    saved = save_state(state_file, {"issues": {"1": {"status": "dispatched"}}, "prs": {}})
    snapshot = state_module.read_state_snapshot(state_file)
    assert snapshot is not None
    assert snapshot["issues"]["1"]["status"] == "dispatched"
    assert snapshot["generated_at"] == saved["generated_at"]


def test_fleet_census_reader_never_quarantines_a_concurrent_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer looping state_lock+save_state against the lock-free census
    reader (``read_fleet_census_state``) must never produce a ``*.corrupt-*``
    quarantine file and the state must stay intact.

    Regression for the review's issue #310 finding: ``load_state`` under
    this exact contention can hit a transient read failure after its short
    retries and RENAME the healthy state file; ``read_state_snapshot``
    only ever reads (a ``None`` result is a skip signal, not an error here).
    Mirrors ``test_status_concurrent_writer_no_quarantine``.

    Real contention is not a reliable failure trigger, so the reader thread
    also gets an injected burst of exactly ``_LOAD_RETRY_ATTEMPTS``
    consecutive ``PermissionError``s on its ``state.json`` opens -- the same
    Windows sharing-violation shape a racing tmp+replace produces. That
    exhausts the retry budget deterministically: the fix returns one
    ``None`` (a skip signal), while the quarantining ``load_state`` it
    replaced renames the file -- which the ``*.corrupt-*`` assertion below
    then catches.
    """
    state_file = tmp_path / "state.json"
    save_state(state_file, {"issues": {}, "prs": {}})

    iterations = 200
    errors: list[Exception] = []
    none_reads: list[bool] = []
    injected_failures: list[int] = [state_module._LOAD_RETRY_ATTEMPTS]
    reader_thread: list[threading.Thread] = []

    orig_open = Path.open

    def _flaky_open(self: Path, *args: Any, **kwargs: Any):
        if (
            self == state_file
            and reader_thread
            and threading.current_thread() is reader_thread[0]
            and injected_failures[0] > 0
        ):
            injected_failures[0] -= 1
            raise PermissionError("injected transient sharing violation")
        return orig_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _flaky_open)

    def writer() -> None:
        for i in range(iterations):
            try:
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = append_event(state, "stress", {"i": i})
                    save_state(state_file, state)
            except Exception as exc:  # pragma: no cover - failures fail the test
                errors.append(exc)
                break

    def reader() -> None:
        for _ in range(iterations):
            try:
                # The exact reader the fleet census uses; a transient None
                # under contention (including the injected burst) is a
                # legitimate "skip" signal, not an error -- only an exception
                # or a quarantined file is a bug.
                snapshot = read_fleet_census_state(state_file, "owner/test")
                if snapshot is None:
                    none_reads.append(True)
            except Exception as exc:  # pragma: no cover - failures fail the test
                errors.append(exc)
                break

    writer_thread = threading.Thread(target=writer)
    reader_thread.append(threading.Thread(target=reader))
    writer_thread.start()
    reader_thread[0].start()
    writer_thread.join(timeout=30)
    reader_thread[0].join(timeout=30)

    assert not errors, f"exceptions during concurrent stress: {errors}"
    assert none_reads, (
        "the injected failure burst was never observed -- the test did not "
        "exercise the retry-exhaustion path it exists to pin"
    )

    quarantine_files = list(state_file.parent.glob(f"{state_file.name}.corrupt-*"))
    assert not quarantine_files, (
        f"the lock-free census read quarantined a healthy state file: {quarantine_files}"
    )

    final_state = load_state(state_file)
    assert final_state["version"] == 1
    assert len(final_state.get("events", [])) == iterations


def test_fleet_census_unreadable_state_skips_corroboration_not_destructively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo whose state.json is corrupt is marked skipped, contributes
    only its live sidecars (ghost corroboration cannot run), leaves a
    ``fleet_census_state_unreadable`` warning event, and -- crucially --
    the corrupt file is NOT renamed or quarantined away."""
    fleet_dir = tmp_path / "fleet"
    state_file = _register_repo(fleet_dir, tmp_path, "a")
    state_file.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(
        "charlie_work.worker.iter_workers",
        lambda _dir, **_kw: [_SidecarStub(1)],
    )

    total, kinds, skipped = count_fleet_live_sessions_with_adapter_kinds(str(fleet_dir))

    assert skipped == ["owner/a"]
    assert total == 1  # the live sidecar; the ghost cannot be corroborated
    assert kinds == {"claude-code": 1}
    assert state_file.exists() and state_file.read_text(encoding="utf-8") == "{ not json", (
        "the census must never mutate the state file it samples"
    )
    assert not list(state_file.parent.glob(f"{state_file.name}.corrupt-*"))
    events = query_events(state_file, kind="fleet_census_state_unreadable")
    assert events, "no fleet_census_state_unreadable event was recorded"
    assert events[-1]["level"] == "warning"
    assert events[-1]["repo"] == "owner/a"


def test_fleet_review_census_unreadable_state_skips_corroboration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review-lane twin of the worker-lane corrupt-state test: same
    skip + sidecars-only + non-destructive contract."""
    fleet_dir = tmp_path / "fleet"
    state_file = _register_repo(fleet_dir, tmp_path, "a")
    state_file.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(
        "charlie_work.worker.iter_workers",
        lambda _dir, **_kw: [_SidecarStub(1)],
    )

    count, skipped = count_fleet_live_reviews(str(fleet_dir))

    assert skipped == ["owner/a"]
    assert count == 1  # the live sidecar; the ghost reviewer is not corroborated
    assert state_file.exists() and state_file.read_text(encoding="utf-8") == "{ not json"
    assert not list(state_file.parent.glob(f"{state_file.name}.corrupt-*"))
    assert query_events(state_file, kind="fleet_census_state_unreadable")
