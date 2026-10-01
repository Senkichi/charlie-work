"""Thread-safe holder for the dashboard's current read model plus its collector threads.

The holder swaps a whole frozen ``ModelState`` under a lock and never mutates one.
The collector keeps the previous model when a refresh raises and records the error and
the time it started failing on the state, so the page can say "collector failing since"
instead of going blank. Both loops return errors as values and never kill their thread.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .now_model import build_now_model
from .now_types import FindingLike, NowModel, SourcesRead

log = logging.getLogger("charlie_work.dashboard")

Collect = Callable[[datetime], "tuple[SourcesRead, Sequence[FindingLike]]"]
RollupRun = Callable[[datetime], Sequence[str]]  # returns error strings (empty = ok)
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class ModelState:
    model: NowModel | None = None
    collected_at: datetime | None = None
    collector_error: str | None = None
    collector_failing_since: datetime | None = None
    rollup_errors: tuple[str, ...] = ()
    rolled_up_at: datetime | None = None


class ReadModel:
    """Swap-whole-immutable-object holder."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = ModelState()

    def get(self) -> ModelState:
        with self._lock:
            return self._state

    def update(self, **changes: Any) -> ModelState:
        with self._lock:
            self._state = dataclasses.replace(self._state, **changes)
            return self._state


def refresh_model(holder: ReadModel, collect: Collect, clock: Clock) -> None:
    """One collector pass; on failure keep the previous model and record the error."""
    now = clock()
    try:
        sources_read, findings = collect(now)
        model = build_now_model(sources_read, now, findings)
    except Exception as exc:  # collector boundary: any failure becomes a value on the state
        log.warning("dashboard collector failed: %s: %s", type(exc).__name__, exc)
        prior = holder.get()
        holder.update(
            collector_error=f"{type(exc).__name__}: {exc}",
            collector_failing_since=prior.collector_failing_since or now,
        )
        return
    holder.update(
        model=model, collected_at=now, collector_error=None, collector_failing_since=None
    )


def refresh_rollup(holder: ReadModel, rollup: RollupRun, clock: Clock) -> None:
    now = clock()
    try:
        errors = tuple(rollup(now))
    except Exception as exc:  # errors as values
        log.warning("dashboard rollup failed: %s: %s", type(exc).__name__, exc)
        errors = (f"{type(exc).__name__}: {exc}",)
    holder.update(rollup_errors=errors, rolled_up_at=now)


def _loop(stop: threading.Event, interval: float, step: Callable[[], None]) -> None:
    while True:
        step()
        if stop.wait(interval):
            return


def start_workers(
    holder: ReadModel,
    stop: threading.Event,
    *,
    collect: Collect,
    rollup: RollupRun | None,
    collector_interval: float,
    rollup_interval: float,
    clock: Clock,
) -> tuple[threading.Thread, ...]:
    """Start the ONE collector thread and (if given) the rollup thread (daemon)."""
    plan: list[tuple[str, float, Callable[[], None]]] = [
        ("dashboard-collector", collector_interval, lambda: refresh_model(holder, collect, clock))
    ]
    if rollup is not None:
        plan.append(
            ("dashboard-rollup", rollup_interval, lambda: refresh_rollup(holder, rollup, clock))
        )
    threads = tuple(
        threading.Thread(target=_loop, args=(stop, iv, step), name=name, daemon=True)
        for name, iv, step in plan
    )
    for t in threads:
        t.start()
    return threads


def to_plain(value: Any) -> Any:
    """Dataclass tree -> JSON-ready structure (datetimes become ISO strings)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value
