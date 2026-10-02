"""Axis scales and tick generation: nice linear ticks, local-time ticks, duration ticks.

Pure arithmetic, no markup. Times are converted to the display zone (``tz``; the host's
local zone when ``None``) only for choosing boundaries and labels; positions stay on the
absolute timeline, so a DST change never bends the x axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo


def _nice(x: float, *, round_: bool) -> float:
    """Heckbert's nice number: 1, 2, 5 or 10 times a power of ten."""
    exp = math.floor(math.log10(x))
    f = x / 10**exp
    if round_:
        nf = 1.0 if f < 1.5 else 2.0 if f < 3 else 5.0 if f < 7 else 10.0
    else:
        nf = 1.0 if f <= 1 else 2.0 if f <= 2 else 5.0 if f <= 5 else 10.0
    return nf * 10**exp


def nice_ticks(lo: float, hi: float, target: int = 5, integer: bool = False) -> tuple[float, ...]:
    """Evenly spaced 1/2/5 ticks covering ``[lo, hi]``; the first and last enclose it."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    step = _nice(_nice(hi - lo, round_=False) / max(target - 1, 1), round_=True)
    if integer:
        step = max(step, 1.0)
    digits = max(0, -math.floor(math.log10(step)))
    start = math.floor(lo / step + 1e-9) * step
    end = math.ceil(hi / step - 1e-9) * step
    count = int(round((end - start) / step))
    return tuple(round(start + i * step, digits) + 0.0 for i in range(count + 1))


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


_DURATION_TICKS = (1, 10, 60, 300, 900, 3600, 4 * 3600, 86400, 7 * 86400)


def duration_label(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"


def duration_ticks(lo: float, hi: float) -> tuple[float, ...]:
    """Round durations enclosing ``[lo, hi]`` seconds, for a log axis."""
    lo, hi = max(lo, 1.0), max(hi, 1.0)
    below = [t for t in _DURATION_TICKS if t <= lo] or [_DURATION_TICKS[0]]
    above = [t for t in _DURATION_TICKS if t >= hi] or [_DURATION_TICKS[-1]]
    return tuple(t for t in _DURATION_TICKS if below[-1] <= t <= above[0])
