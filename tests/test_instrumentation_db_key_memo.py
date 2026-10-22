"""Memo contract for the resolved ``events.db`` key (issue #2566).

Split out of ``test_instrumentation.py`` (review round-1 rework on PR
#2582): that module already sat at the file-size cap, and the ratchet
(issue #1442) fails new code landing in an over-cap monolith. These
tests pin the one-``Path.resolve()``-per-``state_path`` contract both
through ``_db_key`` directly and through the public write primitives.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

import charlie_work.instrumentation as _instrumentation
from charlie_work.instrumentation import (
    _db_key,
    close_db,
    log_event,
    read_event_log,
    record_loop_pass,
)


@pytest.fixture(autouse=True)
def _close_db_after_test(tmp_path: Path) -> None:
    """Ensure DB connections are closed between tests."""
    yield
    close_db(tmp_path / "state.json")
    close_db(tmp_path / "a" / "state.json")
    close_db(tmp_path / "b" / "state.json")


def _counting_resolve(monkeypatch) -> dict[str, int]:
    """Install a ``Path.resolve`` shim that counts calls, transparently."""
    real_resolve = Path.resolve
    resolve_calls: dict[str, int] = {"count": 0}

    def counting_resolve(path: Path, *args: object, **kwargs: object) -> object:
        resolve_calls["count"] += 1
        return real_resolve(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "resolve", counting_resolve)
    return resolve_calls


def test_log_event_and_record_loop_pass_write_through_memoized_key(
    tmp_path: Path, monkeypatch
) -> None:
    """The one-resolve contract holds through the public write primitives.

    ``test_instrumentation.test_db_key_memo_resolution_across_event_writes``
    counts resolves via ``_db_key`` directly, so a revert of any single
    call site (``_get_db``, ``log_event``, ``record_loop_pass``) back to a
    raw ``Path.resolve()`` could pass while the per-event cost crept back
    in. ``instrumentation.py``'s only ``resolve()`` site lives inside
    ``_db_key``, so exercising ``log_event`` twice plus
    ``record_loop_pass`` on one fresh ``state_path`` must cost exactly one
    ``Path.resolve()`` call; reverting *any* write call site adds at least
    one more and fails this count.
    """
    state_path = tmp_path / "state.json"
    _instrumentation._resolved_db_keys.pop(str(state_path), None)

    resolve_calls = _counting_resolve(monkeypatch)

    log_event(state_path, "memo_test", {"n": 1}, repo="test-repo")
    log_event(state_path, "memo_test", {"n": 2}, repo="test-repo")
    record_loop_pass(
        state_path,
        "cid-memo",
        datetime.now(UTC).isoformat(),
    )

    assert resolve_calls["count"] == 1


def test_db_key_relative_state_path_tracks_cwd_change(tmp_path: Path, monkeypatch) -> None:
    """``_db_key`` normalizes relative ``state_path`` values per cwd.

    The memo was keyed on the raw path string, so a relative
    ``state_path`` outliving a cwd change was served the previous cwd's
    resolved form (and its connection/lock) — stale for the new cwd.
    Keys are now normalized with ``os.path.abspath`` at lookup time, so
    the same relative spelling under a different cwd yields a different
    key, re-resolves once, and returning to a previously seen cwd is a
    memo hit with no re-resolve.
    """
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    state_rel = Path("state.json")
    expected_a = str((dir_a / "events.db").resolve())
    expected_b = str((dir_b / "events.db").resolve())

    monkeypatch.chdir(dir_a)
    key_a = _db_key(state_rel)
    assert key_a == expected_a

    # Same relative spelling, different cwd -> the raw-string memo would
    # have answered with key_a; the cwd-normalized key must move to dir_b.
    monkeypatch.chdir(dir_b)
    key_b = _db_key(state_rel)
    assert key_b == expected_b
    assert key_b != key_a

    # A write from dir_b lands in dir_b's events.db, not dir_a's.
    log_event(state_rel, "cwd_b", {"where": "b"})
    assert [e["kind"] for e in read_event_log(state_rel)] == ["cwd_b"]
    monkeypatch.chdir(dir_a)
    assert read_event_log(state_rel) == []

    # close_db with the relative spelling (from dir_b's cwd, i.e. the cwd
    # that armed key_b) closes only dir_b's connection and evicts only
    # key_b's memo entry; key_a's memo and connection stay untouched.
    monkeypatch.chdir(dir_b)
    close_db(state_rel)
    assert key_b not in _instrumentation._db_connections
    assert key_a in _instrumentation._db_connections

    thread_conn_a = _instrumentation._db_connections[key_a]

    # close_db also evicted key_b's memo entry (pop keyed by the same
    # cwd-normalized spelling), so the next lookup from dir_b re-resolves.
    resolve_calls = _counting_resolve(monkeypatch)
    assert _db_key(state_rel) == key_b
    assert resolve_calls["count"] == 1

    # Returning to a previously seen cwd is a memo hit: no re-resolve.
    monkeypatch.chdir(dir_a)
    resolve_calls = _counting_resolve(monkeypatch)
    assert _db_key(state_rel) == key_a
    assert resolve_calls["count"] == 0
    assert _instrumentation._db_connections[key_a] is thread_conn_a
    # close_db tore down key_b's lock alongside its connection.
    assert key_b not in _instrumentation._db_locks
