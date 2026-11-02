"""Horizontal lane chart: one lane per lifecycle stage, one band per visit (issue drill-down).

Position carries the reading (Franconeri rule 7): every lane shares one local-time x axis, so
"where did the time go" is a length comparison along a common baseline. Each lane is labelled
directly on the left and states its own total and visit count on the right (no legend,
rule 3); the takeaway names the longest stage and is computed from the same bands. Bands are
neutral ink (one loud thing per view, rule 1). Approximate bands are hollow and dashed and
say "approx."; a still-open visit runs to the "now" rule and says "open", so neither state is
colour alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, tzinfo

from ..pages.now_fmt import age
from .scale import Linear, TimeTick, time_ticks
from .svg import APPROX_DASH, caption, dash_attr, empty_figure, local_label, num, svg_open, text

ROW_H = 26.0
_LEFT = 116.0
_RIGHT_TEXT = 190.0
_TOP = 18.0
_MIN_BAND = 2.0


@dataclass(frozen=True)
class Band:
    start: datetime
    end: datetime
    open_now: bool = False


@dataclass(frozen=True)
class Lane:
    label: str
    bands: tuple[Band, ...]


def _seconds(lane: Lane) -> float:
    return sum(max(0.0, (b.end - b.start).total_seconds()) for b in lane.bands)


def lane_summary(lane: Lane, approx: bool) -> str:
    n = len(lane.bands)
    if n == 0:
        return "not visited"
    parts = [age(_seconds(lane)), f"{n} visit" + ("" if n == 1 else "s")]
    if any(b.open_now for b in lane.bands):
        parts.append("open")
    if approx:
        parts.append("approx.")
    return " · ".join(parts)


def lane_takeaway(lanes: tuple[Lane, ...], approx: bool) -> str | None:
    """Deterministic headline from the drawn bands: the stage that held the item longest."""
    timed = [lane for lane in lanes if lane.bands]
    if not timed:
        return None
    top = max(timed, key=lambda lane: (_seconds(lane), lane.label))
    total = sum(_seconds(lane) for lane in timed)
    share = f" ({round(100 * _seconds(top) / total)}% of staged time)" if total > 0 else ""
    return f"Longest in {top.label}: {age(_seconds(top))}{share}" + (", approx." if approx else "")


def _ticks(t0: datetime, t1: datetime, tz: tzinfo | None) -> tuple[TimeTick, ...]:
    ticks = tuple(t for t in time_ticks(t0, t1, tz) if t0 <= t.at <= t1)
    if len(ticks) >= 2:
        return ticks
    # A span shorter than the smallest step: label both ends so the axis is never bare.
    return (TimeTick(t0, f"{t0.astimezone(tz):%H:%M}"), TimeTick(t1, f"{t1.astimezone(tz):%H:%M}"))


def _lane(lane: Lane, y: float, x: Linear, t0: datetime, approx: bool) -> str:
    cls = "lane" + (" approx" if approx else "")
    out = [
        f'<g class="{cls}">',
        text(_LEFT - 8, y + ROW_H / 2 + 4, lane.label, "row-label", "end"),
    ]
    out.append(
        f'<line class="lane-rule" x1="{num(x.r0)}" x2="{num(x.r1)}" '
        f'y1="{num(y + ROW_H - 1)}" y2="{num(y + ROW_H - 1)}"/>'
    )
    for b in lane.bands:
        x0 = x((b.start - t0).total_seconds())
        w = max(x((b.end - t0).total_seconds()) - x0, _MIN_BAND)
        band = "band" + (" open" if b.open_now else "")
        out.append(
            f'<rect class="{band}" x="{num(x0)}" y="{num(y + 5)}" width="{num(w)}" '
            f'height="{num(ROW_H - 10)}"{dash_attr(APPROX_DASH if approx else None)}/>'
        )
    out.append(text(x.r1 + 10, y + ROW_H / 2 + 4, lane_summary(lane, approx), "row-stats"))
    out.append("</g>")
    return "".join(out)


def lane_chart(
    lanes: tuple[Lane, ...],
    window: tuple[datetime, datetime],
    title: str,
    *,
    approx: bool = False,
    now: datetime | None = None,
    width: float = 820.0,
    tz: tzinfo | None = None,
) -> str:
    """A captioned ``<figure>``: stage lanes over one shared local-time axis."""
    if not any(lane.bands for lane in lanes):
        return empty_figure(title, "no stage visits recorded for this item")
    t0, t1 = window
    span = max((t1 - t0).total_seconds(), 1.0)
    x = Linear(0.0, span, _LEFT, width - _RIGHT_TEXT)
    axis_y = _TOP + ROW_H * len(lanes)
    height = axis_y + 26
    body = ['<g class="axis x">']
    for tick in _ticks(t0, t1, tz):
        xt = x((tick.at - t0).total_seconds())
        body.append(
            f'<line class="grid" x1="{num(xt)}" x2="{num(xt)}" y1="{num(_TOP)}" '
            f'y2="{num(axis_y)}"/>'
        )
        body.append(text(xt, axis_y + 14, tick.label, "tick", "middle"))
    body.append("</g>")
    body += [_lane(lane, _TOP + ROW_H * i, x, t0, approx) for i, lane in enumerate(lanes)]
    if now is not None and t0 <= now <= t1 and any(b.open_now for la in lanes for b in la.bands):
        xn = x((now - t0).total_seconds())
        body.append(
            f'<line class="marker-rule" x1="{num(xn)}" x2="{num(xn)}" y1="{num(_TOP)}" '
            f'y2="{num(axis_y)}"/>'
        )
        body.append(text(xn, _TOP - 5, "now", "note", "middle"))
    label = f"{title}, {local_label(t0, tz)} to {local_label(t1, tz)} local. " + "; ".join(
        f"{la.label}: {lane_summary(la, approx)}" for la in lanes if la.bands
    )
    svg = svg_open(width, height, label, "lane-chart") + "".join(body) + "</svg>"
    return (
        '<figure class="chart">'
        + caption(title, window, tz, lane_takeaway(lanes, approx))
        + f'<div class="chart-scroll">{svg}</div></figure>'
    )
