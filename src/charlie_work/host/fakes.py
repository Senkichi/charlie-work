"""Test doubles for host ports.

They live in src next to the Protocols so every test package can import them.
They hold mutable state on purpose, so they are not frozen (the frozen
invariant covers config and value objects; ``HostPorts`` itself is frozen).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta


class FakeClock:
    """Deterministic clock; ``advance`` moves both ``now`` and ``monotonic``."""

    def __init__(self, start: datetime, mono: float = 0.0) -> None:
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        self._now = start
        self._mono = mono

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)
        self._mono += seconds


class FakeProcessProbe:
    """Scripted liveness: ``alive`` maps pid -> recorded start time (or None).

    Mirrors ``is_pid_alive``: unknown pid or ``pid`` None/<=0 is dead; an
    indeterminate start time on either side is alive (fail-open); a mismatch
    is dead. Windows ACCESS_DENIED fail-closed is not modelled.
    """

    def __init__(self, alive: Mapping[int, float | None] | None = None) -> None:
        self._alive = dict(alive or {})
        self.calls: list[tuple[int | None, float | None]] = []

    def is_alive(self, pid: int | None, start_time: float | None = None) -> bool:
        self.calls.append((pid, start_time))
        if pid is None or pid <= 0 or pid not in self._alive:
            return False
        actual = self._alive[pid]
        if start_time is None or actual is None:
            return True
        return actual == start_time

    def start_time(self, pid: int) -> float | None:
        return self._alive.get(pid)
