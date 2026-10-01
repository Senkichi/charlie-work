"""Test doubles for host ports.

They live in src next to the Protocols so every test package can import them.
They hold mutable state on purpose, so they are not frozen (the frozen
invariant covers config and value objects; ``HostPorts`` itself is frozen).
"""

from __future__ import annotations

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
