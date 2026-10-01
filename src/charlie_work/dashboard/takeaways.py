"""Deterministic one-sentence headlines for History charts (spec section 4).

``takeaway(series, prior)`` compares a series with the same metric queried over the equal
window immediately before it (``MetricQuery.prior()``). Rules, in order:

1. ``not_instrumented`` -> says so.
2. The series must cover both windows: if coverage starts after the prior window began, or
   ends more than one bucket before the current window ends, no trend is claimed (a gap is
   not a decline).
3. Fewer than ``MIN_SAMPLE`` observations in either window -> "not enough data".
4. Otherwise: direction and percent change, plus the repo contributing most to the change
   when one repo carries at least half of the total per-repo movement.

Counts are compared as a per-day rate; gauges, durations and ratios as the mean of the
window's bucket values. The same inputs always produce the same string.
"""

from __future__ import annotations

from datetime import timedelta

from .metrics_base import Point, Series, parse_ts

MIN_SAMPLE = 5
FLAT_BELOW = 0.05  # |change| under 5% reads as flat
DRIVER_SHARE = 0.5
_DAY = timedelta(days=1)


def _span(series: Series) -> timedelta:
    return parse_ts(series.window_end) - parse_ts(series.window_start)


def _window_label(span: timedelta) -> str:
    days = span / _DAY
    return f"{int(days)}d" if days == int(days) else f"{int(span.total_seconds() // 3600)}h"


def _value(points: tuple[Point, ...], kind: str, span: timedelta) -> float | None:
    if not points:
        return None
    total = sum(v for _, v in points)
    return total / (span / _DAY) if kind == "count" else total / len(points)


def _pct(change: float) -> int:
    return int(abs(change) * 100 + 0.5 + 1e-9)  # half-up, immune to float dust


def _headline_label(series: Series) -> str:
    base = series.label or series.name
    return f"{base}/day" if series.kind == "count" else base


def _driver(current: Series, prior: Series, delta: float, span: timedelta) -> str | None:
    """Repo with the largest same-direction share of the per-repo movement, if >= half."""
    moves: dict[str, float] = {}
    for repo in sorted(set(current.per_repo) | set(prior.per_repo)):
        cur = _value(current.per_repo.get(repo, ()), current.kind, span) or 0.0
        pri = _value(prior.per_repo.get(repo, ()), prior.kind, span) or 0.0
        moves[repo] = cur - pri
    total = sum(abs(m) for m in moves.values())
    if len(moves) < 2 or total == 0:
        return None
    repo, move = max(moves.items(), key=lambda kv: (abs(kv[1]), kv[0]))
    if (move > 0) != (delta > 0) or abs(move) / total < DRIVER_SHARE:
        return None
    return repo


def _gap(current: Series, prior: Series) -> str | None:
    """Why the series cannot support a trend claim, or None when both windows are covered."""
    tol = timedelta(seconds=current.bucket_seconds)
    start = current.coverage_start
    end = current.coverage_end
    if start is None or end is None:
        return None  # handled as "not enough data"
    if parse_ts(start) > parse_ts(prior.window_start) + tol:
        return f"history starts {start}"
    if parse_ts(end) < parse_ts(current.window_end) - tol:
        return f"data ends {end}"
    return None


def takeaway(current: Series, prior: Series, *, min_sample: int = MIN_SAMPLE) -> str:
    """One-sentence headline for ``current`` against the equal prior window ``prior``."""
    span = _span(current)
    if _span(prior) != span or prior.window_end != current.window_start:
        raise ValueError("prior must be the equal window ending where current starts")
    label = _headline_label(current)
    if current.not_instrumented:
        return f"{label}: not instrumented yet"
    gap = _gap(current, prior)
    if gap:
        return f"{label}: no trend claimed, {gap}"
    window = _window_label(span)
    cur_v = _value(current.points, current.kind, span)
    pri_v = _value(prior.points, prior.kind, span)
    if cur_v is None or pri_v is None or min(current.n, prior.n) < min_sample:
        return f"{label}: not enough data vs prior {window}"
    suffix = " (approx.)" if current.approx else ""
    if pri_v == 0:
        text = (
            f"{label} unchanged at 0 vs prior {window}"
            if cur_v == 0
            else f"{label} up from 0 vs prior {window}"
        )
        return text + suffix
    change = (cur_v - pri_v) / pri_v
    if abs(change) < FLAT_BELOW:
        sign = "+" if change >= 0 else "-"
        return f"{label} flat vs prior {window} ({sign}{_pct(change)}%){suffix}"
    arrow = "↑" if change > 0 else "↓"
    text = f"{label} {arrow}{_pct(change)}% vs prior {window}"
    driver = _driver(current, prior, cur_v - pri_v, span)
    return text + (f", driven by {driver}" if driver else "") + suffix
