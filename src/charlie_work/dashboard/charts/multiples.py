"""Small multiples: one line panel per repo on a SHARED y scale and the same x window.

The shared scale is the point (Franconeri rule 6): panel heights compare directly, so the
y ticks are computed once from every panel's values and reused verbatim. Panels are sorted
by their plotted total, largest first, ties broken by key (rule 5: sort by the metric).
Panel headings are HTML outside the SVG, so a heading can link to the repo drill-down
while the SVG keeps ``role="img"``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import tzinfo

from ..pages.now_fmt import link
from .line import (
    Frame,
    domain,
    label_width,
    plot,
    sources_note,
    summary,
    values,
    y_ticks_for,
)
from .model import LineSpec, Panel, Size
from .svg import caption, empty_figure, num, svg_open

PANEL = Size(width=280.0, height=130.0)


def _total(panel: Panel) -> float:
    return sum(values(panel.series))


def small_multiples(
    panels: tuple[Panel, ...],
    spec: LineSpec,
    tz: tzinfo | None = None,
    panel_size: Size = PANEL,
) -> str:
    """A captioned grid of panels; every panel shares ``y_ticks`` and the x window."""
    all_series = tuple(s for p in panels for s in p.series)
    window = domain(all_series, spec)
    if not panels or window is None:
        return empty_figure(spec.title, "no data in this window", "chart multiples")
    y_ticks = y_ticks_for(values(all_series), target=4)
    shared = LineSpec(
        title=spec.title,
        bucket_seconds=spec.bucket_seconds,
        coverage=spec.coverage,
        markers=spec.markers,
        unit=spec.unit,
        domain=window,
        size=panel_size,
    )
    # One right margin for every panel, so the shared x window lines up across the grid.
    multi = any(len(p.series) > 1 for p in panels)
    room = min(label_width(all_series), panel_size.width * 0.4) if multi else 10
    right = panel_size.width - room
    out: list[str] = []
    for panel in sorted(panels, key=lambda p: (-_total(p), p.key)):
        f = Frame(
            width=panel_size.width,
            height=panel_size.height,
            left=36,
            right=right,
            top=8,
            bottom=panel_size.height - 22,
            t0=window[0],
            t1=window[1],
            y_ticks=y_ticks,
        )
        # a panel is shaded by its own source's coverage, never by another repo's start
        own = replace(shared, coverage=tuple(c for c in shared.coverage if c.source == panel.key))
        label = summary(f"{spec.title}: {panel.key}", panel.series, shared, tz, window)
        svg = (
            svg_open(
                panel_size.width,
                panel_size.height,
                label,
                "line-chart panel-chart",
                {"data-y-max": num(y_ticks[-1])},
            )
            + plot(panel.series, own, f, tz, labels=len(panel.series) > 1, max_x_ticks=3)
            + "</svg>"
        )
        heading = link(panel.href, panel.key, cls="panel-key")
        out.append(
            f'<figure class="panel"><figcaption>{heading}</figcaption>'
            f'<div class="chart-scroll">{svg}</div></figure>'
        )
    return (
        f'<figure class="chart multiples" data-panels="{len(panels)}">'
        + caption(spec.title, window, tz, spec.takeaway, sources_note(spec, tz))
        + f'<div class="multiples-grid">{"".join(out)}</div></figure>'
    )
