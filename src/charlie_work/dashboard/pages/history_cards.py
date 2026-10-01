"""One History metric card: takeaway headline, combined chart, per-repo small multiples.

The headline is the takeaway of the series the card draws first (``MetricData.headline``),
computed by ``takeaways.takeaway`` from that same series and its prior-window twin, so the
sentence and the line can never disagree. Flags are said in words next to the chart:
approximate series are also drawn dashed, partial and not-instrumented series say so, and
the coverage note names each source's own start ("charlie-work from 07-23 · ...").
Pure functions; every dynamic value is escaped (``now_fmt.esc`` / the chart primitives).
"""

from __future__ import annotations

from datetime import datetime, timedelta, tzinfo

from ..charts import Coverage, LineSpec, Marker, Panel, Size, line_chart, small_multiples
from ..charts import Point as ChartPoint
from ..charts import Series as ChartSeries
from ..history_data import HistoryView, MetricData
from ..metrics_base import Series
from ..timeutil import iso, parse_ts
from .now_fmt import esc, repo_url, short_repo, slug

LIFECYCLE_ISSUE = "#2226"
MAX_CATEGORY_LINES = 3  # beyond this a combined chart stops supporting a comparison
COMBINED = Size(width=640.0, height=220.0)
# Usage metric -> the cap/capacity metric of the same tab drawn on the same axis (the
# comparison is "how close to the cap", Franconeri rule 4); the cap gets no card of its own.
OVERLAYS: dict[str, str] = {
    "reviewers_live": "reviewers_cap",
    "runners_running": "runners_capacity",
}
_UNIT_WORDS = {"ratio": "share"}


def _lifecycle(metric_id: str) -> bool:
    return metric_id == "lead_time" or metric_id.startswith("stage_time.")


def bucket_words(seconds: int) -> str:
    step = timedelta(seconds=seconds)
    if step % timedelta(days=1) == timedelta(0):
        return f"{step.days}d"
    return f"{int(step.total_seconds() // 3600)}h"


def title_of(series: Series) -> str:
    base = series.label or series.name
    unit = _UNIT_WORDS.get(series.unit, series.unit)
    if series.kind == "count":
        return f"{base}, {unit} per {bucket_words(series.bucket_seconds)}"
    return f"{base} ({unit})" if unit else base


def _points(points: tuple[tuple[str, float], ...]) -> tuple[ChartPoint, ...]:
    return tuple(ChartPoint(parse_ts(ts), v) for ts, v in points)


def _chart_name(series: Series, headline: Series, has_children: bool) -> str:
    if series is headline:
        return "all" if has_children else (series.label or series.name)
    return (
        series.name.rsplit(".", 1)[-1]
        if series.name.startswith(f"{headline.name}.")
        else (series.label or series.name)
    )


def drawn_series(metric: MetricData, cap: MetricData | None) -> tuple[Series, ...]:
    """The headline, its largest categories (bounded), then any overlaid cap series."""
    head = metric.headline
    children = sorted((s for s in metric.series[1:] if s.n > 0), key=lambda s: (-s.n, s.name))[
        :MAX_CATEGORY_LINES
    ]
    return (head, *children, *((cap.headline,) if cap is not None else ()))


def _chart_series(drawn: tuple[Series, ...], has_children: bool) -> tuple[ChartSeries, ...]:
    head = drawn[0]
    return tuple(
        ChartSeries(_chart_name(s, head, has_children), _points(s.points), approx=s.approx)
        for s in drawn
    )


def _local_day(ts: str, tz: tzinfo | None) -> str:
    return (
        f'<time datetime="{esc(iso(parse_ts(ts)))}">'
        f"{esc(parse_ts(ts).astimezone(tz).strftime('%m-%d'))}</time>"
    )


def coverage_note(series: Series, tz: tzinfo | None) -> str:
    """Each source's own coverage start, earliest first, plus where the data ends early."""
    spans = sorted(series.repo_coverage.items(), key=lambda kv: (kv[1][0], kv[0]))
    if not spans:
        return '<p class="cov-note">No source covers this metric yet.</p>'
    items = " · ".join(
        f"{esc(short_repo(src))} from {_local_day(lo, tz)}" for src, (lo, _) in spans
    )
    ends = ""
    tol = timedelta(seconds=series.bucket_seconds)
    if series.coverage_end and parse_ts(series.coverage_end) < parse_ts(series.window_end) - tol:
        ends = f" · data ends {_local_day(series.coverage_end, tz)}"
    return f'<p class="cov-note">Coverage: {items}{ends}</p>'


def _chart_coverage(series: Series, window: tuple[datetime, datetime]) -> tuple[Coverage, ...]:
    """Sources whose coverage starts inside the window (the chart shades before them)."""
    out = []
    for src, (lo, _) in sorted(series.repo_coverage.items()):
        start = parse_ts(lo)
        if window[0] < start <= window[1]:
            out.append(Coverage(short_repo(src), start))
    return tuple(out)


def _markers(series: Series, tz: tzinfo | None) -> tuple[Marker, ...]:
    if not series.exact_from:
        return ()
    at = parse_ts(series.exact_from)
    return (Marker(at, f"exact from {at.astimezone(tz).strftime('%m-%d')}"),)


def flag_notes(metric_id: str, series: Series) -> tuple[str, ...]:
    """The honest caveats, as words (never colour alone)."""
    notes = []
    if series.approx:
        notes.append(
            f"approx. until lifecycle instrumentation ({LIFECYCLE_ISSUE})"
            if _lifecycle(metric_id)
            else "approx.: reconstructed from proxy events"
        )
    if series.partial:
        notes.append("partial: its sources cover only part of what this metric means")
    return tuple(notes)


def _flags_html(notes: tuple[str, ...]) -> str:
    if not notes:
        return ""
    return '<ul class="flags">' + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul>"


def _not_instrumented(metric_id: str, metric: MetricData) -> str:
    head = metric.headline
    issue = f" ({LIFECYCLE_ISSUE})" if _lifecycle(metric_id) else ""
    return (
        f'<article class="mcard is-missing" id="{esc(card_id(metric_id))}">'
        f'<p class="takeaway">{esc(metric.takeaways[head.name])}</p>'
        f'<h3 class="mtitle">{esc(title_of(head))}</h3>'
        f'<p class="missing">not instrumented yet{esc(issue)}: no source has recorded the '
        "events this metric is derived from, so nothing is drawn (not a zero).</p></article>"
    )


def card_id(metric_id: str) -> str:
    return "m-" + slug(metric_id)


def _panels(drawn: tuple[Series, ...], has_children: bool) -> tuple[Panel, ...]:
    """One panel per repo for the headline (and an overlaid cap), never the categories."""
    keep = (drawn[0], *(s for s in drawn[1:] if not s.name.startswith(f"{drawn[0].name}.")))
    repos = sorted({r for s in keep for r, pts in s.per_repo.items() if pts})
    return tuple(
        Panel(
            key=short_repo(repo),
            series=tuple(
                ChartSeries(
                    _chart_name(s, drawn[0], has_children),
                    _points(s.per_repo.get(repo, ())),
                    approx=s.approx,
                )
                for s in keep
            ),
            href=repo_url(repo),
        )
        for repo in repos
    )


def render_card(view: HistoryView, metric: MetricData, tz: tzinfo | None = None) -> str:
    """One metric card, or its honest 'not instrumented yet' card."""
    mid, head = metric.metric_id, metric.headline
    if head.not_instrumented:
        return _not_instrumented(mid, metric)
    cap = overlay_for(view, mid)
    drawn = drawn_series(metric, cap)
    has_children = len(metric.series) > 1
    window = (view.query.start, view.query.end)
    spec = LineSpec(
        title=title_of(head),
        bucket_seconds=float(head.bucket_seconds),
        coverage=_chart_coverage(head, window),
        markers=_markers(head, tz),
        takeaway=metric.takeaways[head.name],
        domain=window,
        size=COMBINED,
    )
    combined = line_chart(_chart_series(drawn, has_children), spec, tz)
    panels = _panels(drawn, has_children)
    multiples = (
        small_multiples(
            panels,
            LineSpec(
                title=f"{title_of(head)}, per repo (shared y)",
                bucket_seconds=float(head.bucket_seconds),
                markers=_markers(head, tz),
                domain=window,
            ),
            tz,
        )
        if len(panels) > 1
        else ""
    )
    more = sum(1 for s in metric.series[1:] if s.n > 0) - (len(drawn) - 1 - (cap is not None))
    rest = (
        f'<p class="cov-note">{more} smaller categor{"y" if more == 1 else "ies"} '
        "counted in “all” but not drawn.</p>"
        if more > 0
        else ""
    )
    cls = "mcard" + (" is-approx" if head.approx else "")
    return (
        f'<article class="{cls}" id="{esc(card_id(mid))}">'
        f"{_flags_html(flag_notes(mid, head))}{combined}{rest}"
        f"{coverage_note(head, tz)}{multiples}</article>"
    )


def overlay_for(view: HistoryView, metric_id: str) -> MetricData | None:
    """The cap drawn on ``metric_id``'s chart, when it exists and is instrumented."""
    cap = view.get(OVERLAYS[metric_id]) if metric_id in OVERLAYS else None
    return cap if cap is not None and not cap.headline.not_instrumented else None


def render_cards(view: HistoryView, tz: tzinfo | None = None) -> str:
    """Every card of the tab in registry order; an overlaid cap is not repeated alone."""
    overlaid = {
        cap.metric_id
        for m in view.metrics
        if not m.headline.not_instrumented and (cap := overlay_for(view, m.metric_id))
    }
    return "".join(render_card(view, m, tz) for m in view.metrics if m.metric_id not in overlaid)
