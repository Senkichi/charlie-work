"""Strip plot with a box for duration distributions (lead time, loop pass duration).

Every sample is a dot (no mean bar: Franconeri rule 18), over a light inter-quartile box,
with the median and p90 as ticks and stated in text beside the row. Rows share one log
duration axis (durations span seconds to days; products rule 14) and are sorted by median,
longest first, ties by label. Approximate rows draw hollow dots and say "approx.".
Jitter is deterministic (by sample index) so a re-render never moves a dot.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, tzinfo

from ..pages.now_fmt import age
from .model import Distribution, Size
from .scale import Linear, duration_label, duration_ticks
from .svg import caption, empty_figure, num, svg_open, text

ROW_H = 28.0
_LEFT = 120.0
_RIGHT_TEXT = 210.0


def nearest_rank(values: tuple[float, ...], pct: float) -> float:
    """Nearest-rank percentile: always an observed sample, never interpolated."""
    ordered = sorted(values)
    idx = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[idx]


def _jitter(i: int) -> float:
    return float((i * 7) % 5 - 2) * 2.5


def _stats(d: Distribution) -> str:
    if not d.values:
        return "no samples"
    med = statistics.median(d.values)
    p90 = nearest_rank(d.values, 90)
    approx = " approx." if d.approx else ""
    return f"median {age(med)} · p90 {age(p90)} · n {len(d.values)}{approx}"


def _row(d: Distribution, y: float, x: Linear) -> str:
    def px(v: float) -> float:
        return x(math.log10(max(v, 1.0)))

    cls = "dist" + (" approx" if d.approx else "")
    out = [f'<g class="{cls}">', text(_LEFT - 8, y + 4, d.label, "row-label", "end")]
    if d.values:
        q1, q3 = nearest_rank(d.values, 25), nearest_rank(d.values, 75)
        out.append(
            f'<rect class="iqr" x="{num(px(q1))}" y="{num(y - 8)}" '
            f'width="{num(max(px(q3) - px(q1), 1.0))}" height="16"/>'
        )
        for i, v in enumerate(d.values):
            out.append(
                f'<circle class="sample" cx="{num(px(v))}" cy="{num(y + _jitter(i))}" r="2.5"/>'
            )
        for name, v in (
            ("median", statistics.median(d.values)),
            ("p90", nearest_rank(d.values, 90)),
        ):
            xv = px(v)
            out.append(
                f'<line class="{name}" x1="{num(xv)}" x2="{num(xv)}" '
                f'y1="{num(y - 10)}" y2="{num(y + 10)}"/>'
            )
    out.append(text(x.r1 + 10, y + 4, _stats(d), "row-stats"))
    out.append("</g>")
    return "".join(out)


def strip_plot(
    rows: tuple[Distribution, ...],
    title: str,
    *,
    window: tuple[datetime, datetime] | None = None,
    takeaway: str | None = None,
    width: float = Size().width + 160,
    tz: tzinfo | None = None,
) -> str:
    """A captioned ``<figure>``: one row per distribution on a shared log axis."""
    samples = [v for r in rows for v in r.values]
    if not samples:
        return empty_figure(title, "no samples in this window")
    ticks = duration_ticks(min(samples), max(samples))
    x = Linear(math.log10(ticks[0]), math.log10(ticks[-1]), _LEFT, width - _RIGHT_TEXT)
    ordered = sorted(
        rows,
        key=lambda r: (-(statistics.median(r.values) if r.values else -1.0), r.label),
    )
    height = 24 + ROW_H * len(ordered) + 20
    body = ['<g class="axis x">']
    axis_y = 12 + ROW_H * len(ordered)
    for t in ticks:
        xt = x(math.log10(t))
        body.append(
            f'<line class="grid" x1="{num(xt)}" x2="{num(xt)}" y1="6" y2="{num(axis_y)}"/>'
        )
        body.append(text(xt, axis_y + 14, duration_label(t), "tick", "middle"))
    body.append("</g>")
    for i, d in enumerate(ordered):
        body.append(_row(d, 12 + ROW_H * i + ROW_H / 2, x))
    label = f"{title}. " + ". ".join(f"{d.label}: {_stats(d)}" for d in ordered) + "."
    svg = svg_open(width, height, label, "strip-chart") + "".join(body) + "</svg>"
    return (
        '<figure class="chart">'
        + caption(title, window, tz, takeaway, "; log duration axis")
        + f'<div class="chart-scroll">{svg}</div></figure>'
    )
