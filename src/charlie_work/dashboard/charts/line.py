"""Multi-series line chart on one shared y axis, as server-rendered SVG.

* Missing buckets are gaps: a ``None`` value, or two points further apart than 1.5
  buckets, ends the current path and starts a new one. A lone point is a dot.
* Series are labelled directly at their line ends (dodged apart), never via a legend.
* Approximate series are dashed and their label says "approx.".
* Each ``Coverage`` whose source starts inside the window shades the uncovered span and
  labels where the source starts; ``Marker`` draws a labelled rule (e.g. the date the
  exact series replaces the approximate one).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo

from .model import LineSpec, Point, Series
from .scale import Linear, nice_ticks, time_ticks
from .svg import (
    APPROX_DASHES,
    CHAR_PX,
    SERIES_DASH,
    caption,
    dash_attr,
    empty_figure,
    fit,
    local_label,
    num,
    svg_open,
    text,
    value_text,
)

_LABEL_GAP = 14.0  # min vertical distance between direct labels (px)


@dataclass(frozen=True)
class Frame:
    """Plot geometry: the drawing box inside the SVG and its two scales."""

    width: float
    height: float
    left: float
    right: float
    top: float
    bottom: float
    t0: datetime
    t1: datetime
    y_ticks: tuple[float, ...]

    def x(self, moment: datetime) -> float:
        return Linear(self.t0.timestamp(), self.t1.timestamp(), self.left, self.right)(
            moment.timestamp()
        )

    def y(self, value: float) -> float:
        return Linear(self.y_ticks[0], self.y_ticks[-1], self.bottom, self.top)(value)


def segments(points: tuple[Point, ...], bucket_seconds: float | None) -> list[list[Point]]:
    """Runs of consecutive present points; absent buckets split runs."""
    runs: list[list[Point]] = []
    current: list[Point] = []
    for p in sorted(points, key=lambda q: q.at):
        far = (
            bucket_seconds is not None
            and current
            and (p.at - current[-1].at).total_seconds() > 1.5 * bucket_seconds
        )
        if p.value is None or far:
            if current:
                runs.append(current)
            current = []
        if p.value is not None:
            current.append(p)
    if current:
        runs.append(current)
    return runs


def values(series: tuple[Series, ...]) -> list[float]:
    return [p.value for s in series for p in s.points if p.value is not None]


def domain(series: tuple[Series, ...], spec: LineSpec) -> tuple[datetime, datetime] | None:
    if spec.domain is not None:
        return spec.domain
    times = [p.at for s in series for p in s.points]
    if not times:
        return None
    t0, t1 = min(times), max(times)
    return (t0, t1) if t1 > t0 else (t0 - timedelta(hours=1), t1 + timedelta(hours=1))


def y_ticks_for(vals: list[float], target: int = 5) -> tuple[float, ...]:
    """Zero-based nice ticks (counts and durations share a zero baseline)."""
    lo = min([0.0, *vals])
    hi = max([0.0, *vals])
    return nice_ticks(lo, hi if hi > lo else lo + 1.0, target)


def _dodge(wanted: list[tuple[float, int]], top: float, bottom: float) -> dict[int, float]:
    """Push label baselines apart by ``_LABEL_GAP``, keeping them inside the plot."""
    placed: dict[int, float] = {}
    last = top - _LABEL_GAP
    for y, idx in sorted(wanted):
        y = max(y, last + _LABEL_GAP)
        placed[idx] = y
        last = y
    overflow = last - bottom
    if overflow > 0:
        placed = {i: y - overflow for i, y in placed.items()}
    return placed


def _axes(f: Frame, unit: str, tz: tzinfo | None, max_x_ticks: int) -> str:
    out = ['<g class="axis y">']
    for v in f.y_ticks:
        y = f.y(v)
        out.append(
            f'<line class="grid" x1="{num(f.left)}" x2="{num(f.right)}" '
            f'y1="{num(y)}" y2="{num(y)}"/>'
        )
        out.append(text(f.left - 6, y + 4, value_text(v, unit), "tick", "end"))
    out.append('</g><g class="axis x">')
    out.append(
        f'<line class="baseline" x1="{num(f.left)}" x2="{num(f.right)}" '
        f'y1="{num(f.bottom)}" y2="{num(f.bottom)}"/>'
    )
    for t in time_ticks(f.t0, f.t1, tz, max_x_ticks):
        x = f.x(t.at)
        out.append(
            f'<line class="tickmark" x1="{num(x)}" x2="{num(x)}" '
            f'y1="{num(f.bottom)}" y2="{num(f.bottom + 4)}"/>'
        )
        out.append(text(x, f.bottom + 16, t.label, "tick", "middle"))
    out.append("</g>")
    return "".join(out)


_MAX_START_LABELS = 3  # more source starts than this share one label (details in caption)


def _context(f: Frame, spec: LineSpec, tz: tzinfo | None) -> str:
    """Coverage shading + source-start labels, then annotation markers.

    ``spec.coverage`` lists every source, including ones that start before the window: only
    the span before the earliest source is shaded "uncovered" (no source could hold data
    there); each later in-window start gets a rule and a label.

    Shading and rules are drawn before any label, so a later source's shading can never
    paint over an earlier source's label; a label that would run past the plot flips to
    the left of its rule.
    """
    ordered = sorted(spec.coverage, key=lambda c: c.start)
    starts = [c for c in ordered if f.t0 < c.start <= f.t1]
    shades: list[str] = []
    labels: list[str] = []
    if starts and ordered[0].start > f.t0:  # uncovered = before the EARLIEST source only
        x = f.x(ordered[0].start)
        shades.append(
            f'<rect class="uncovered" x="{num(f.left)}" y="{num(f.top)}" '
            f'width="{num(x - f.left)}" height="{num(f.bottom - f.top)}"/>'
        )
    for cov in starts:  # a later source's start is a rule, not more shading
        x = f.x(cov.start)
        shades.append(
            f'<line class="cov-start" x1="{num(x)}" x2="{num(x)}" y1="{num(f.top)}" '
            f'y2="{num(f.bottom)}"/>'
        )
    named = [
        (c.start, f"{c.source} from {local_label(c.start, tz)[:10]}")
        for c in starts[:_MAX_START_LABELS]
    ]
    if len(starts) > _MAX_START_LABELS:
        first, last = local_label(starts[0].start, tz)[:10], local_label(starts[-1].start, tz)[:10]
        span = first if first == last else f"{first}–{last}"
        named = [(starts[-1].start, f"{len(starts)} sources start {span}")]
    for i, (at, label) in enumerate(named):
        x = f.x(at)
        room_right = f.right - x - 3
        anchor, tx = ("start", x + 3) if len(label) * CHAR_PX <= room_right else ("end", x - 3)
        room = room_right if anchor == "start" else x - 3 - f.left
        labels.append(text(tx, f.top + 11 + 13 * i, fit(label, room), "note", anchor, label))
    out = [f'<g class="coverage">{"".join(shades)}{"".join(labels)}</g>'] if starts else []
    for m in spec.markers:
        if not f.t0 <= m.at <= f.t1:
            continue
        x = f.x(m.at)
        out.append(
            f'<g class="marker"><line class="marker-rule" x1="{num(x)}" x2="{num(x)}" '
            f'y1="{num(f.top)}" y2="{num(f.bottom)}"{dash_attr("2 2")}/>'
            f"{text(x - 3, f.top + 11, m.text, 'note', 'end')}</g>"
        )
    return "".join(out)


def _series(f: Frame, series: tuple[Series, ...], spec: LineSpec, labels: bool) -> str:
    out: list[str] = []
    ends: list[tuple[float, int]] = []
    for idx, s in enumerate(series):
        cls = f"series s{idx % 3 + 1}" + (" approx" if s.approx else "")
        dash = (APPROX_DASHES if s.approx else SERIES_DASH)[idx % len(SERIES_DASH)]
        body: list[str] = []
        inside = tuple(p for p in s.points if f.t0 <= p.at <= f.t1)
        runs = segments(inside, spec.bucket_seconds)
        for run in runs:
            if len(run) == 1:
                p = run[0]
                body.append(
                    f'<circle class="dot" cx="{num(f.x(p.at))}" '
                    f'cy="{num(f.y(p.value or 0.0))}" r="2.5"/>'
                )
                continue
            d = " ".join(
                f"{'M' if i == 0 else 'L'}{num(f.x(p.at))},{num(f.y(p.value or 0.0))}"
                for i, p in enumerate(run)
            )
            body.append(f'<path class="line" d="{d}" fill="none"{dash_attr(dash)}/>')
        if runs:
            ends.append((f.y(runs[-1][-1].value or 0.0) + 4, idx))
        out.append(f'<g class="{cls}">{"".join(body)}</g>')
    if labels:
        placed = _dodge(ends, f.top + 4, f.bottom)
        for idx, y in sorted(placed.items()):
            s = series[idx]
            name = f"{s.name} approx." if s.approx else s.name
            cls = f"direct s{idx % 3 + 1}" + (" approx" if s.approx else "")
            out.append(text(f.right + 6, y, fit(name, f.width - f.right - 8), cls, full=name))
    return "".join(out)


def label_width(series: tuple[Series, ...]) -> float:
    """Room the direct labels need at the plot's right (callers cap it)."""
    longest = max((len(s.name) + (8 if s.approx else 0) for s in series), default=0)
    return 10 + CHAR_PX * longest


def plot(
    series: tuple[Series, ...],
    spec: LineSpec,
    f: Frame,
    tz: tzinfo | None,
    *,
    labels: bool,
    max_x_ticks: int = 6,
) -> str:
    """The inner SVG markup (axes, context, series) for one frame."""
    return (
        _axes(f, spec.unit, tz, max_x_ticks)
        + _context(f, spec, tz)
        + _series(f, series, spec, labels)
    )


def summary(title: str, series: tuple[Series, ...], spec: LineSpec, tz, window) -> str:
    """Deterministic aria-label: window, then per series its last value, peak and gaps."""
    parts = [f"{title}, {local_label(window[0], tz)} to {local_label(window[1], tz)} local"]
    for s in series:
        runs = segments(s.points, spec.bucket_seconds)
        name = f"{s.name} (approx.)" if s.approx else s.name
        if not runs:
            parts.append(f"{name}: no data")
            continue
        vals = [p.value for r in runs for p in r if p.value is not None]
        gaps = f", {len(runs) - 1} gap{'s' if len(runs) > 2 else ''}" if len(runs) > 1 else ""
        parts.append(
            f"{name}: last {value_text(runs[-1][-1].value or 0.0, spec.unit)}, "
            f"peak {value_text(max(vals), spec.unit)}{gaps}"
        )
    return ". ".join(parts) + "."


def sources_note(spec: LineSpec, tz: tzinfo | None) -> str:
    if not spec.coverage:
        return ""
    items = ", ".join(
        f"{c.source} from {local_label(c.start, tz)}"
        for c in sorted(spec.coverage, key=lambda c: c.start)
    )
    return f"; sources: {items}"


def line_chart(series: tuple[Series, ...], spec: LineSpec, tz: tzinfo | None = None) -> str:
    """A captioned ``<figure>`` holding one multi-series line chart."""
    window = domain(series, spec)
    if window is None:
        return empty_figure(spec.title, "no data in this window")
    width, height = spec.size.width, spec.size.height
    label_room = label_width(series)
    f = Frame(
        width=width,
        height=height,
        left=44,
        right=width - min(max(label_room, 12), width / 3),
        top=10,
        bottom=height - 26,
        t0=window[0],
        t1=window[1],
        y_ticks=y_ticks_for(values(series)),
    )
    svg = (
        svg_open(width, height, summary(spec.title, series, spec, tz, window), "line-chart")
        + plot(series, spec, f, tz, labels=True)
        + "</svg>"
    )
    return (
        '<figure class="chart">'
        + caption(spec.title, window, tz, spec.takeaway, sources_note(spec, tz))
        + f'<div class="chart-scroll">{svg}</div></figure>'
    )
