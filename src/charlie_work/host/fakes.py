"""Test doubles for host ports.

They live in src next to the Protocols so every test package can import them.
They hold mutable state on purpose, so they are not frozen (the frozen
invariant covers config and value objects; ``HostPorts`` itself is frozen).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any


class FakeClock:
    """Deterministic clock; ``advance`` moves both ``now`` and ``monotonic``.

    ``start`` defaults to the Unix epoch for tests that only exercise the
    monotonic side. ``sleep`` doubles the injectable ``sleep`` callable of the
    supervisor loops: it appends the requested duration to ``sleep_calls`` and
    advances the clock by ``auto_advance`` per call when that is nonzero, else
    by the slept seconds.
    """

    def __init__(
        self,
        start: datetime | None = None,
        mono: float = 0.0,
        auto_advance: float = 0.0,
    ) -> None:
        if start is None:
            start = datetime(1970, 1, 1, tzinfo=UTC)
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)
        self._now = start
        self._mono = mono
        self._auto_advance = auto_advance
        self.sleep_calls: list[float] = []

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)
        self._mono += seconds

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.advance(self._auto_advance if self._auto_advance else seconds)


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


class FakeSessionCounter:
    """Scripted session counts with a call log of ``(method, args)``."""

    def __init__(
        self,
        workers: int = 0,
        reviews: int = 0,
        fleet_workers: tuple[int, list[str]] = (0, []),
        fleet_reviews: tuple[int, list[str]] = (0, []),
        issue_numbers: set[int] | None = None,
        session_pids: dict[str, int] | None = None,
    ) -> None:
        self.workers = workers
        self.reviews = reviews
        self.fleet_workers = fleet_workers
        self.fleet_reviews = fleet_reviews
        self.issue_numbers = set(issue_numbers or ())
        self.session_pids = dict(session_pids or {})
        self.calls: list[tuple[str, tuple]] = []

    def live_workers(self, sessions_dir, state_file=None) -> int:
        self.calls.append(("live_workers", (sessions_dir, state_file)))
        return self.workers

    def fleet_live_workers(self, fleet_dir_override):
        self.calls.append(("fleet_live_workers", (fleet_dir_override,)))
        return self.fleet_workers

    def live_reviews(self, reviews_dir, state_file=None) -> int:
        self.calls.append(("live_reviews", (reviews_dir, state_file)))
        return self.reviews

    def fleet_live_reviews(self, fleet_dir_override):
        self.calls.append(("fleet_live_reviews", (fleet_dir_override,)))
        return self.fleet_reviews

    def live_issue_numbers(self, sessions_dir) -> set[int]:
        self.calls.append(("live_issue_numbers", (sessions_dir,)))
        return set(self.issue_numbers)

    def live_session_pids(self, sessions_dir) -> dict[str, int]:
        self.calls.append(("live_session_pids", (sessions_dir,)))
        return dict(self.session_pids)


class FakeReviewLauncher:
    """Scripted reviewer launches; never spawns a process or sleeps.

    ``outcomes`` is consumed one per ``launch`` call (the last repeats). Each is
    a record (returned as-is), a ``str`` (a failure record with that ``.error``),
    or a callable ``(harness, kwargs) -> record``. With no outcomes every launch
    succeeds with ``pid=1``. ``requests`` logs ``(harness, kwargs)``.
    """

    def __init__(self, outcomes: Sequence[Any] = ()) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def launch(self, harness: str, **kwargs: Any) -> Any:
        from .launch import error_record

        index = len(self.requests)
        self.requests.append((harness, dict(kwargs)))
        if not self._outcomes:
            record = error_record(harness, kwargs, "")
            return _replace_record(record, error=None, pid=1)
        outcome = self._outcomes[min(index, len(self._outcomes) - 1)]
        if isinstance(outcome, str):
            return error_record(harness, kwargs, outcome)
        if callable(outcome):
            call: Callable[[str, dict[str, Any]], Any] = outcome
            return call(harness, dict(kwargs))
        return outcome


def _replace_record(record: Any, **changes: Any) -> Any:
    import dataclasses

    return dataclasses.replace(record, **changes)
