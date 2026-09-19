"""settle_session_to_disk locked read-modify-write (issue #480, review #540).

Also carries the ``advisory_file_lock`` / ``state_lock`` primitive
regressions that pin the generic lock this settlement path routes through.

Split out of ``tests/test_api_budget.py`` (issue #1571, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_api_budget_unit_fixtures.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from charlie_work import api_budget
from charlie_work.api_budget import SessionEntry, ledger_path, load_ledger

from _api_budget_unit_fixtures import _entry


# ---------------------------------------------------------------------------
# settle_session_to_disk -- locked read-modify-write (review #540 rework)
#
# The blocking review finding: settle_session's load-modify-save cycle on
# api-budget.json had no locking, unlike state_lock for state.json --
# concurrent reaps of different api sessions could silently lose a
# settlement (lost-update race). settle_session_to_disk wraps the RMW in
# state.advisory_file_lock.
# ---------------------------------------------------------------------------


def test_settle_session_to_disk_writes_entry(tmp_path: Path) -> None:
    path = ledger_path(tmp_path)
    assert api_budget.settle_session_to_disk(path, _entry(usd=6.15)) is True
    ledger = load_ledger(path)
    assert len(ledger.sessions) == 1
    assert ledger.lifetime_usd == pytest.approx(6.15)


def test_settle_session_to_disk_idempotent(tmp_path: Path) -> None:
    path = ledger_path(tmp_path)
    api_budget.settle_session_to_disk(path, _entry(usd=3.0))
    # Same identity → no-op, still returns True.
    assert api_budget.settle_session_to_disk(path, _entry(usd=3.0)) is True
    ledger = load_ledger(path)
    assert len(ledger.sessions) == 1
    assert ledger.lifetime_usd == pytest.approx(3.0)


def test_settle_session_to_disk_concurrent_distinct_sessions_no_lost_update(
    tmp_path: Path,
) -> None:
    """Two threads settling DISTINCT sessions concurrently must both land in the
    ledger — the per-path threading.Lock in advisory_file_lock serializes the
    load→settle→save critical sections so no settlement is lost to a
    read-modify-write race (the review's blocking finding).
    """
    import threading

    path = ledger_path(tmp_path)
    entries = [
        _entry(issue=100, session_id="sess-a", usd=1.0, started_at="2026-07-22T10:00:00Z"),
        _entry(issue=101, session_id="sess-b", usd=2.0, started_at="2026-07-22T10:00:00Z"),
        _entry(issue=102, session_id="sess-c", usd=3.0, started_at="2026-07-22T10:00:00Z"),
        _entry(issue=103, session_id="sess-d", usd=4.0, started_at="2026-07-22T10:00:00Z"),
    ]
    errors: list[BaseException] = []

    def settle(entry: SessionEntry) -> None:
        try:
            api_budget.settle_session_to_disk(path, entry)
        except BaseException as exc:  # noqa: BLE001 — surface any thread failure
            errors.append(exc)

    threads = [threading.Thread(target=settle, args=(e,)) for e in entries]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    ledger = load_ledger(path)
    settled_ids = {e.session_id for e in ledger.sessions}
    assert settled_ids == {"sess-a", "sess-b", "sess-c", "sess-d"}
    # No lost update: all four settlements accumulated.
    assert ledger.lifetime_usd == pytest.approx(10.0)


def test_settle_session_to_disk_skips_when_lock_busy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch
) -> None:
    """Fail-as-a-value: if the advisory lock cannot be acquired
    (StateLockBusy), settlement is SKIPPED and logged — never written unlocked.
    """
    from charlie_work import state as state_module
    from charlie_work.state import StateLockBusy

    path = ledger_path(tmp_path)

    def _always_busy(path_arg: Path):  # noqa: ANN001
        raise StateLockBusy("simulated contention")

    monkeypatch.setattr(state_module, "advisory_file_lock", _always_busy)

    with caplog.at_level("WARNING", logger="charlie_work.api_budget"):
        result = api_budget.settle_session_to_disk(path, _entry(usd=5.0))

    assert result is False
    # Ledger was never written.
    assert not path.exists()
    assert any("lock busy" in rec.message for rec in caplog.records)


def test_settle_session_to_disk_uses_advisory_lock_not_state_lock(
    tmp_path: Path, monkeypatch
) -> None:
    """The locked RMW must go through state.advisory_file_lock (the generic
    primitive), not the state.json-specific state_lock wrapper. Asserting the
    generic primitive is the one exercised prevents a regression that re-routes
    settlement through an unrelated lock path.
    """
    from charlie_work import state as state_module

    path = ledger_path(tmp_path)
    calls: list[Path] = []
    real_lock = state_module.advisory_file_lock

    def tracking_lock(p: Path):
        calls.append(p)
        return real_lock(p)

    monkeypatch.setattr(state_module, "advisory_file_lock", tracking_lock)

    api_budget.settle_session_to_disk(path, _entry(usd=1.0))

    assert calls == [path]
    assert load_ledger(path).lifetime_usd == pytest.approx(1.0)


def test_advisory_file_lock_serializes_concurrent_threads_same_path(
    tmp_path: Path,
) -> None:
    """advisory_file_lock (the generic primitive extracted from state_lock)
    must serialize concurrent threads in this process on the same path — the
    per-path threading.Lock is what closes the intra-process lost-update race
    that file locks alone cannot (byte-range locks are owned by the process,
    not the thread).
    """
    import threading

    from charlie_work.state import advisory_file_lock

    target = tmp_path / "target.json"
    target.write_text("0", encoding="utf-8")

    # Each thread reads the int, increments, writes back — under the lock.
    # Without intra-process serialization the increments would collide.
    def bump() -> None:
        for _ in range(50):
            with advisory_file_lock(target):
                val = int(target.read_text(encoding="utf-8"))
                target.write_text(str(val + 1), encoding="utf-8")

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert target.read_text(encoding="utf-8") == "200"  # 4 threads * 50 increments


def test_state_lock_delegates_to_advisory_file_lock(tmp_path: Path, monkeypatch) -> None:
    """state_lock is now a thin wrapper over advisory_file_lock. Assert the
    delegation directly: monkeypatching the generic primitive is observed by a
    state_lock caller (regression for the extraction — keeps the two names
    bound to one mechanism, not two divergent lock implementations).
    """
    import charlie_work.state as state_module
    from charlie_work.state import state_lock

    state_path = tmp_path / "state.json"
    calls: list[Path] = []
    real_lock = state_module.advisory_file_lock

    def tracking_lock(p: Path):
        calls.append(p)

        @contextmanager
        def _cm():
            with real_lock(p):
                yield

        return _cm()

    monkeypatch.setattr(state_module, "advisory_file_lock", tracking_lock)

    with state_lock(state_path):
        pass

    assert calls == [state_path]
