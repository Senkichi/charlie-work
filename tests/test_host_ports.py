from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from charlie_work import host
from charlie_work.host.clock import RealClock, format_utc
from charlie_work.host.fakes import FakeClock

_UTC_NOW_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def test_format_utc_matches_state_utc_now_shape() -> None:
    assert _UTC_NOW_RE.match(format_utc(RealClock().now()))


def test_format_utc_drops_microseconds_and_uses_z() -> None:
    moment = datetime(2026, 1, 2, 3, 4, 5, 999999, tzinfo=UTC)
    assert format_utc(moment) == "2026-01-02T03:04:05Z"


def test_fake_clock_advance_moves_now_and_monotonic() -> None:
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC), mono=10.0)
    clock.advance(30)
    assert clock.now() == datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC)
    assert clock.monotonic() == 40.0


def test_fake_host_swaps_current_and_restores(fake_host) -> None:
    clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))
    fake = fake_host(clock=clock)
    assert host.current() is fake
    assert host.current().clock is clock


def test_fake_host_restored_after_previous_test() -> None:
    assert host.current() is host.REAL


def test_host_ports_is_frozen() -> None:
    with pytest.raises(AttributeError):
        host.REAL.clock = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))  # type: ignore[misc]


def test_fake_host_clock_freezes_state_and_workflow_utc_now(fake_host) -> None:
    from charlie_work import state, workflow

    fake_host(clock=FakeClock(datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)))
    assert state.utc_now() == "2026-03-04T05:06:07Z"
    assert workflow.utc_now() == "2026-03-04T05:06:07Z"


def test_fake_process_probe_mirrors_primitive_semantics() -> None:
    from charlie_work.host.fakes import FakeProcessProbe

    probe = FakeProcessProbe({10: 5.0, 11: None})
    assert probe.is_alive(10, 5.0) is True
    assert probe.is_alive(10, 6.0) is False  # recycled pid
    assert probe.is_alive(10, None) is True  # indeterminate -> fail-open
    assert probe.is_alive(11, 7.0) is True
    assert probe.is_alive(99) is False
    assert probe.is_alive(None) is False
    assert probe.is_alive(0) is False
    assert probe.start_time(10) == 5.0


def test_fake_host_probe_reaches_worker_fate_and_sweeps(fake_host) -> None:
    from charlie_work import worker_fate
    from charlie_work.dead_worker_sweep.effects_sessions import _worker_pid_alive
    from charlie_work.dispatch_selection import _reviewer_pid_alive
    from charlie_work.host.fakes import FakeProcessProbe

    fake_host(probe=FakeProcessProbe({4242: 1.0}))
    assert worker_fate.is_alive(4242, 1.0) is True
    assert worker_fate.is_alive(4242, 2.0) is False
    assert worker_fate.is_alive(None, None) is False
    assert _worker_pid_alive({"worker_pid": 4242, "worker_process_start_time": 1.0}) is True
    assert _reviewer_pid_alive({"reviewer_pid": 4242, "reviewer_process_start_time": 9.0}) is False


def test_real_probe_late_binds_to_process_utils(monkeypatch) -> None:
    from charlie_work.host import REAL

    monkeypatch.setattr("charlie_work.process_utils.is_pid_alive", lambda pid, st=None: pid == 7)
    assert REAL.probe.is_alive(7, None) is True
    assert REAL.probe.is_alive(8, None) is False
    assert REAL.probe.is_alive(-1, None) is False
