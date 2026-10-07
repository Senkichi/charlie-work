"""The History page's per-metric facts: summary value, change vs prior, wrong-way flag.

Pure functions over :class:`history_data.MetricData`. Nothing here re-derives a trend: the
change a card shows is the pair of window values the takeaway compared
(``MetricData.compared``, from ``takeaways.assess``), so the card, the headline and the
takeaway sentence can never disagree about direction or size. The name, which direction
is good and how a window is summarised come from the metric registry
(``metrics.presentation``), never from the page.

Colour has one job on the page: a card is flagged (``Card.bad``) only when a metric with
a known good direction moved the other way by ``WRONG_WAY`` or more.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import timedelta

from .charts.strip import nearest_rank
from .history_data import HistoryView, MetricData
from .metrics import NEUTRAL, presentation
from .metrics_base import MetricQuery, Point, Series, iso, pooled
from .takeaways import FLAT_BELOW, Compared

WRONG_WAY = 0.20  # a move of 20% or more against the metric's good direction is flagged
_SUMMARY_WORDS = {"sum": "total", "last": "latest"}


@dataclass(frozen=True)
class Card:
    """What one metric's card and detail say. ``state``: ok | missing | error."""

    metric_id: str
    name: str
    polarity: int
    summary: str  # sum | median | mean | last
    state: str
    value: float | None = None
    change: float | None = None  # fractional change vs prior; inf: up from a real 0
    bad: bool = False
    unit: str = ""
    kind: str = "count"
    approx: bool = False
    partial: bool = False
    takeaway: str = ""
    error: str = ""
    stats: tuple[int, float, float] | None = None  # duration only: (n, median, p90)

    @property
    def summary_word(self) -> str:
        return _SUMMARY_WORDS.get(self.summary, self.summary)

    @property
    def chart(self) -> str:
        """Counts per bucket are bars; levels, ratios and durations are a line."""
        return "bars" if self.kind == "count" else "line"


def change_of(compared: Compared | None) -> float | None:
    """The takeaway's own comparison as a fraction: None when it claims none, 0.0 when it
    reads flat (``FLAT_BELOW``), ``inf`` for a rise from a real zero baseline."""
    if compared is None:
        return None
    cur, pri = compared
    if pri == 0:
        return 0.0 if cur == 0 else math.inf
    change = (cur - pri) / pri
    return 0.0 if abs(change) < FLAT_BELOW else change


def wrong_way(polarity: int, change: float | None) -> bool:
    """True when a metric with a good direction moved against it by ``WRONG_WAY`` or more."""
    if change is None or polarity == NEUTRAL or change == 0:
        return False
    return (change > 0) != (polarity > 0) and abs(change) >= WRONG_WAY


def delta_text(change: float | None) -> str:
    if change is None:
        return "no comparable prior"
    if change == 0:
        return "flat vs prior"
    if math.isinf(change):
        return "up from 0 vs prior"
    word = "up" if change > 0 else "down"
    if change >= 1:
        return f"{word} {int(1 + change + 0.5)}x vs prior"
    return f"{word} {int(abs(change) * 100 + 0.5)}% vs prior"


def _values(points: tuple[Point, ...]) -> list[float]:
    return [v for _, v in points if v is not None]


def summarize(
    points: tuple[Point, ...], how: str, samples: tuple[float, ...] = ()
) -> float | None:
    """One number for a window: its total, latest, mean, or median (of the raw samples
    when the series carries them, else of the bucket values). None: nothing observed."""
    if how == "median" and samples:
        return float(statistics.median(samples))
    vals = _values(points)
    if not vals:
        return None
    if how == "sum":
        return float(sum(vals))
    if how == "last":
        return vals[-1]
    if how == "median":
        return float(statistics.median(vals))
    return float(statistics.fmean(vals))


def fmt_value(value: float | None, unit: str) -> str:
    """A value in its unit, compactly: ``54%``, ``39m``, ``6.4h``, ``2.1d``, ``1,340``."""
    if value is None:
        return "—"
    if unit == "ratio":
        return f"{int(value * 100 + 0.5)}%"
    if unit == "hours":
        if value < 1:
            return f"{int(value * 60 + 0.5)}m"
        if value < 48:
            return f"{value:.1f}h" if value < 10 else f"{value:.0f}h"
        return f"{value / 24:.1f}d"
    if unit == "seconds":
        if value < 90:
            return f"{int(value + 0.5)}s"
        if value < 5400:
            return f"{int(value / 60 + 0.5)}m"
        return f"{value / 3600:.1f}h"
    if value >= 100:
        return f"{int(value + 0.5):,}"
    return f"{value:.1f}".rstrip("0").rstrip(".")


def card_of(metric: MetricData) -> Card:
    """The card facts for one metric (a degraded or uninstrumented one says so)."""
    mid = metric.metric_id
    if metric.error is not None:
        p = presentation(mid, None)
        return Card(mid, p.name, p.polarity, p.summary or "mean", "error", error=metric.error)
    head = metric.headline
    p = presentation(mid, head)
    summary = p.summary or "mean"
    common = dict(
        unit=head.unit,
        kind=head.kind,
        approx=head.approx,
        partial=head.partial,
        takeaway=metric.takeaways.get(head.name, ""),
    )
    if head.not_instrumented:
        return Card(mid, p.name, p.polarity, summary, "missing", **common)
    change = change_of(metric.compared.get(head.name))
    return Card(
        mid,
        p.name,
        p.polarity,
        summary,
        "ok",
        value=summarize(head.points, summary, pooled(head)),
        change=change,
        bad=wrong_way(p.polarity, change),
        stats=duration_stats(head),
        **common,
    )


def tab_cards(view: HistoryView) -> tuple[tuple[Card, MetricData], ...]:
    """The tab's cards: wrong-way movers first, then the rest, each in registry order."""
    pairs = [(card_of(m), m) for m in view.metrics]
    return tuple(sorted(pairs, key=lambda cm: not cm[0].bad))  # stable: registry order


def tab_flagged(view: HistoryView) -> bool:
    return any(card_of(m).bad for m in view.metrics)


def bucket_grid(q: MetricQuery) -> tuple[str, ...]:
    """Every bucket start of the window (ISO UTC), the x axis the points sit on."""
    return tuple(iso(q.bucket_start(i)) for i in range(q.n_buckets))


def on_grid(points: tuple[Point, ...], grid: tuple[str, ...]) -> list[float | None]:
    """Points placed on the bucket grid; a bucket with no observation is None (a gap)."""
    at = dict(points)
    return [at.get(ts) for ts in grid]


def nice_top(peak: float) -> float:
    """The axis top: ``peak`` rounded up to one significant digit (1 when nothing)."""
    if peak <= 0:
        return 1.0
    step = 10 ** math.floor(math.log10(peak))
    return math.ceil(peak / step - 1e-9) * step


def duration_stats(series: Series) -> tuple[int, float, float] | None:
    """``(n, median, p90)`` over the raw samples of a duration series (nearest-rank p90,
    in the series' own unit), or None without samples."""
    vals = pooled(series)
    if series.kind != "duration" or not vals:
        return None
    return len(vals), float(statistics.median(vals)), nearest_rank(vals, 90)


def bucket_label(seconds: int) -> str:
    step = timedelta(seconds=seconds)
    if step % timedelta(days=1) == timedelta(0):
        return "day" if step.days == 1 else f"{step.days} days"
    hours = int(step.total_seconds() // 3600)
    return "hour" if hours == 1 else f"{hours} hours"
