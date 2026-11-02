"""Axis scale and tick generation: a linear mapping and local-time ticks.

Pure arithmetic, no markup. Times are converted to the display zone (``tz``; the host's
local zone when ``None``) only for choosing boundaries and labels; positions stay on the
absolute timeline, so a DST change never bends the x axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo


@dataclass(frozen=True)
class Linear:
    """Maps ``[d0, d1]`` onto ``[r0, r1]`` (r1 < r0 for a y axis that grows upward)."""

    d0: float
    d1: float
    r0: float
    r1: float

    def __call__(self, v: float) -> float:
        span = self.d1 - self.d0
        t = 0.0 if span == 0 else (v - self.d0) / span
        return self.r0 + t * (self.r1 - self.r0)


@dataclass(frozen=True)
class TimeTick:
    at: datetime
    label: str


_HOUR = 3600
_STEPS = (_HOUR, 2 * _HOUR, 3 * _HOUR, 6 * _HOUR, 12 * _HOUR) + tuple(
    d * 86400 for d in (1, 2, 3, 4, 7, 14, 30)
)


def _day_label(moment: datetime) -> str:
    return f"{moment:%b} {moment.day}"


def time_ticks(
    t0: datetime, t1: datetime, tz: tzinfo | None = None, max_ticks: int = 6
) -> tuple[TimeTick, ...]:
    """Ticks on local-time boundaries (hours from midnight, or local midnights).

    Sub-day steps label ``HH:MM`` except at local midnight, which gets the date so the
    day is never ambiguous; day steps label the date.
    """
    if t1 <= t0:
        return (TimeTick(t0, _day_label(t0.astimezone(tz))),)
    span = (t1 - t0).total_seconds()
    step = next((s for s in _STEPS if span / s <= max_ticks), _STEPS[-1])
    local0 = t0.astimezone(tz)
    if step < 86400:
        hours = step // _HOUR
        first = local0.replace(minute=0, second=0, microsecond=0)
        first = first.replace(hour=first.hour - first.hour % hours)
    else:
        first = local0.replace(hour=0, minute=0, second=0, microsecond=0)
    ticks: list[TimeTick] = []
    cursor = first
    while cursor <= t1 and len(ticks) < 64:
        if cursor >= t0:
            midnight = cursor.hour == 0 and cursor.minute == 0
            text = _day_label(cursor) if step >= 86400 or midnight else f"{cursor:%H:%M}"
            ticks.append(TimeTick(cursor, text))
        cursor = (cursor + timedelta(seconds=step)).astimezone(tz)
    return tuple(ticks)
