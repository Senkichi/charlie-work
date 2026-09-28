"""state.json persistence: round-trip, on-disk validity, concurrent-access serialization, held-lock guard.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
from pathlib import Path
from typing import Any
import pytest
from _fakes_github import FakeGitHub
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
import charlie_work.state as state_module


def test_state_round_trip(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = load_state(state_path)
    state["issues"]["123"] = {"title": "Example"}

    save_state(state_path, state)
    loaded = load_state(state_path)

    assert loaded["issues"]["123"]["title"] == "Example"
    assert loaded["version"] == 1


def test_state_json_is_valid_after_save(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    save_state(state_path, {"version": 1, "issues": {}, "prs": {}, "events": []})

    payload = json.loads(state_path.read_text(encoding="utf-8"))

    assert payload["version"] == 1
    assert payload["generated_at"].endswith("Z")


def test_concurrent_state_access_serializes_with_lock(tmp_path: Path) -> None:
    """Regression test for issue #16: concurrent load→save cycles must serialize.

    Two threads incrementing a counter should never lose updates when using
    the lock context manager. Without the lock, one thread can overwrite the
    other's update (last writer wins).
    """
    state_path = tmp_path / "state.json"
    # Initialize state with a counter
    save_state(state_path, {"version": 1, "issues": {}, "prs": {}, "events": [], "counter": 0})

    # Number of increments per thread
    increments_per_thread = 100
    errors = []

    def increment_counter(thread_id: int) -> None:
        for _ in range(increments_per_thread):
            try:
                with state_lock(state_path):
                    state = load_state(state_path)
                    current = state.get("counter", 0)
                    # Simulate some work
                    state["counter"] = current + 1
                    save_state(state_path, state)
            except Exception as exc:
                errors.append((thread_id, exc))

    # Run two threads concurrently
    thread1 = threading.Thread(target=increment_counter, args=(1,))
    thread2 = threading.Thread(target=increment_counter, args=(2,))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    # Verify no errors occurred
    assert not errors, f"Errors during concurrent access: {errors}"

    # Verify the counter is the sum of both increments (no lost updates)
    final_state = load_state(state_path)
    expected_count = increments_per_thread * 2
    assert final_state.get("counter") == expected_count, (
        f"Expected counter to be {expected_count}, got {final_state.get('counter')} "
        f"— indicates lost updates due to race condition"
    )


@contextlib.contextmanager
def _hold_state_lock(lock_path: Path) -> Any:
    """Hold a real, competing byte-range/exclusive lock on the state lock file.

    This is used to force ``state_lock`` to time out without involving another
    process, while still exercising the real platform locking primitive.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if not lock_path.exists():
        lock_path.write_bytes(b"\x00")
    handle = lock_path.open("r+b")
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


@pytest.mark.parametrize(
    "method_name,args",
    [
        ("status", ()),
        ("intake", ()),
        ("dispatch", ()),
        ("dispatch_rework", ()),
        ("review", (456,)),
        ("merge_ready", (456,)),
    ],
)
def test_state_lock_guard_returns_skip_when_lock_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    args: tuple[Any, ...],
) -> None:
    """Issue #398: if the state lock is held, public state-writing methods
    return a clean skip CommandResult and leave state.json untouched.
    """
    monkeypatch.setattr(state_module, "_LOCK_TIMEOUT_SECONDS", 0.05)

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)

    # Issue #614: merge_ready's dry-run gate sits above the state_lock and
    # returns a read-only verdict without acquiring the lock — so the
    # state-lock guard is only reachable on the non-dry-run path.  The state
    # lock is the first thing that path hits, so no side effects occur before
    # the StateLockBusy exception.
    # Issue #617: review() now has the same shape — its dry-run gate sits
    # above the state_lock and returns a read-only plan without acquiring
    # the lock.
    # Issue #618: intake's dry-run gate also sits above the state_lock —
    # dry-run skips the state merge entirely, so the lock guard never fires.
    if method_name in ("merge_ready", "review", "intake"):
        app.dry_run = False

    state_path = paths.state_file
    state_path.parent.mkdir(parents=True, exist_ok=True)
    initial_state = {
        "version": 1,
        "generated_at": "2026-01-01T00:00:00Z",
        "issues": {},
        "prs": {},
        "events": [],
    }
    state_path.write_text(json.dumps(initial_state), encoding="utf-8")
    initial_mtime = state_path.stat().st_mtime
    initial_content = state_path.read_text(encoding="utf-8")

    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    with _hold_state_lock(lock_path):
        result = getattr(app, method_name)(*args)

    assert result.ok is True
    reason = result.data.get("reason") or result.data.get("deferred_reason")
    assert reason in {"state_lock_busy", "supervisor_lock_held", "graphql_rate_limit"}
    assert result.data.get("pass_skipped") is True or result.data.get("state_lock_busy") is True
    assert state_path.stat().st_mtime == initial_mtime
    assert state_path.read_text(encoding="utf-8") == initial_content


def test_record_dead_worker_failure_kind_returns_new_state() -> None:
    """Issue #1917: ``record_dead_worker_failure_kind`` is a value-style
    update — it returns a new top-level mapping with copied ``issues`` and
    a copied issue entry, and never mutates the caller's ``data``."""
    from charlie_work.state import record_dead_worker_failure_kind

    data = {
        "version": 1,
        "issues": {
            "42": {"status": "dispatched"},
            "43": {"status": "rework_requested"},
        },
        "prs": {},
        "events": [],
    }
    issues_before = data["issues"]
    entry_before = data["issues"]["42"]

    new_state = record_dead_worker_failure_kind(data, 42, "rate_limited")

    assert new_state is not data
    assert new_state["issues"] is not issues_before
    assert new_state["issues"]["42"]["dead_worker_failure_kind"] == "rate_limited"
    # Untouched sibling entries are shared, not deep-copied.
    assert new_state["issues"]["43"] is data["issues"]["43"]
    # The caller's mapping, issues dict, and entry are all unmutated.
    assert "dead_worker_failure_kind" not in data["issues"]["42"]
    assert data["issues"] is issues_before
    assert data["issues"]["42"] is entry_before


def test_record_dead_worker_failure_kind_noop_for_missing_entry() -> None:
    """The no-op contract: an untracked issue returns the input object
    unchanged — the caller never invents an issue entry."""
    from charlie_work.state import record_dead_worker_failure_kind

    data = {"version": 1, "issues": {}, "prs": {}, "events": []}

    assert record_dead_worker_failure_kind(data, 42, "stalled") is data
    assert "42" not in data["issues"]


def test_record_dead_worker_failure_kind_noop_for_non_dict_entry() -> None:
    """Same no-op for a non-dict entry (corrupt/legacy shape) — the helper
    must not stamp onto a scalar."""
    from charlie_work.state import record_dead_worker_failure_kind

    data = {"version": 1, "issues": {"42": "dispatched"}, "prs": {}, "events": []}

    assert record_dead_worker_failure_kind(data, 42, "stalled") is data
    assert data["issues"]["42"] == "dispatched"


def test_record_dead_worker_failure_kind_noop_for_non_dict_issues() -> None:
    """And for a non-dict ``issues`` mapping itself."""
    from charlie_work.state import record_dead_worker_failure_kind

    data = {"version": 1, "issues": "corrupt", "prs": {}, "events": []}

    assert record_dead_worker_failure_kind(data, 42, "stalled") is data


def test_clear_dead_worker_failure_kind_drops_stamp() -> None:
    """Issue #1917: ``clear_dead_worker_failure_kind`` removes the field in
    place and tolerates its absence — the single point of enforcement
    every dispatch-epoch site calls."""
    from charlie_work.state import clear_dead_worker_failure_kind

    entry = {"status": "dispatched", "dead_worker_failure_kind": "rate_limited"}
    clear_dead_worker_failure_kind(entry)
    assert entry == {"status": "dispatched"}

    # Absent-field tolerance: clearing twice (or an unstamped entry) is a no-op.
    clear_dead_worker_failure_kind(entry)
    assert entry == {"status": "dispatched"}
