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

from ..charts import (
    Coverage,
    Distribution,
    LineSpec,
    Marker,
    Panel,
    Size,
    line_chart,
    small_multiples,
    strip_plot,
)
from ..charts.model import Reference
from ..charts.strip import nearest_rank
from ..charts import Point as ChartPoint
from ..charts import Series as ChartSeries
from ..charts.svg import SERIES_DASH
from ..history_data import HistoryView, MetricData
from ..metrics_base import Series
from ..timeutil import iso, parse_ts
from .now_fmt import esc, repo_url, short_repo, slug

LIFECYCLE_ISSUE = "#2226"
# One combined chart draws at most as many lines as the renderer has distinct (colour,
# dash) styles: a 4th line would reuse the headline's style and be told apart only by its
# end label (D2). The headline and any overlaid cap take their slots first.
MAX_CHART_LINES = len(SERIES_DASH)
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
    if series.stat:  # a duration line is one statistic per bucket: name it (D8)
        base = f"{base}, {series.stat} per {bucket_words(series.bucket_seconds)}"
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
    room = MAX_CHART_LINES - 1 - (cap is not None)
    children = sorted((s for s in metric.series[1:] if s.n > 0), key=lambda s: (-s.n, s.name))[
        : max(room, 0)
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
    """Every source that starts by the window's end, including those already covering its
    start: the chart shades only before the earliest, and rules each later in-window start."""
    out = []
    for src, (lo, _) in sorted(series.repo_coverage.items()):
        start = parse_ts(lo)
        if start <= window[1]:
            out.append(Coverage(short_repo(src), start))
    return tuple(out)


def _references(
    compared: tuple[float, float] | None, range_key: str, head: Series
) -> tuple[Reference, ...]:
    """The values the headline compares as rules on the chart's axis (D1), plus the
    window's p90 beside a duration card's per-bucket medians (D8)."""
    out = (
        []
        if compared is None
        else [
            Reference(compared[1], f"prior {range_key} avg", prior=True),
            Reference(compared[0], f"this {range_key} avg"),
        ]
    )
    if head.kind == "duration":
        vals = tuple(v for vs in head.samples.values() for v in vs)
        if vals:
            out.append(Reference(nearest_rank(vals, 90), "p90"))
    return tuple(out)


_SECONDS_PER_UNIT = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}


def _distribution(head: Series, window: tuple[datetime, datetime], tz: tzinfo | None) -> str:
    """The strip plot of the raw durations behind the median line: one row per repo plus
    the fleet as a whole, each stating its median and p90 in words (D8 remainder)."""
    factor = _SECONDS_PER_UNIT.get(head.unit)  # the strip's log axis is in seconds
    if factor is None or not head.samples:
        return ""
    rows = [
        Distribution(
            short_repo(repo), tuple(v * factor for v in head.samples[repo]), approx=head.approx
        )
        for repo in sorted(head.samples)
        if head.samples[repo]
    ]
    if len(head.samples) > 1:  # "all" only adds information when several sources pool
        pooled = tuple(v * factor for vs in head.samples.values() for v in vs)
        if pooled:
            rows.insert(0, Distribution("all", pooled, approx=head.approx))
    if not rows:
        return ""
    return strip_plot(
        tuple(rows),
        f"{head.label or head.name}, per-sample distribution",
        window=window,
        width=COMBINED.width,
        tz=tz,
    )


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


def _error_card(metric_id: str, error: str) -> str:
    """A malformed stored row or a render fault degrades to this card — never the page."""
    return (
        f'<article class="mcard is-error" id="{esc(card_id(metric_id))}">'
        '<p class="takeaway">This metric could not be drawn.</p>'
        f'<h3 class="mtitle">{esc(metric_id)}</h3>'
        f'<p class="missing">a stored row it reads is malformed ({esc(error)}); '
        "the other cards are unaffected.</p></article>"
    )


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


def _panels(
    drawn: tuple[Series, ...], has_children: bool, known: frozenset[str]
) -> tuple[Panel, ...]:
    """One panel per repo for the headline (and an overlaid cap), never the categories.

    A panel heading links to ``/repo/<slug>`` only for a registry repo (``known``): a
    source the registry does not hold (e.g. a runner-allocation target) has no drill page.
    """
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
            href=repo_url(repo) if repo in known else None,
        )
        for repo in repos
    )


def render_card(
    view: HistoryView,
    metric: MetricData,
    tz: tzinfo | None = None,
    known: frozenset[str] = frozenset(),
) -> str:
    """One metric card, or its honest 'not instrumented yet' card."""
    mid = metric.metric_id
    if metric.error is not None:
        return _error_card(mid, metric.error)
    head = metric.headline
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
        references=_references(metric.compared.get(head.name), view.range_key, head),
    )
    combined = line_chart(_chart_series(drawn, has_children), spec, tz)
    panels = _panels(drawn, has_children, known)
    multiples = (
        small_multiples(
            panels,
            LineSpec(
                title=f"{title_of(head)}, per repo (shared y)",
                bucket_seconds=float(head.bucket_seconds),
                coverage=spec.coverage,  # each panel keeps only its own source's start
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
        f"{coverage_note(head, tz)}{_distribution(head, window, tz)}{multiples}</article>"
    )


def overlay_for(view: HistoryView, metric_id: str) -> MetricData | None:
    """The cap drawn on ``metric_id``'s chart, when it exists and is instrumented."""
    cap = view.get(OVERLAYS[metric_id]) if metric_id in OVERLAYS else None
    ok = cap is not None and cap.error is None and not cap.headline.not_instrumented
    return cap if ok else None


def render_cards(
    view: HistoryView, tz: tzinfo | None = None, known: frozenset[str] = frozenset()
) -> str:
    """Every card of the tab in registry order; an overlaid cap is not repeated alone.

    ``known`` is the registry's repo slugs: only those panels link to a repo page.
    A card that fails to render degrades to an error card — one bad row must never
    take down the whole page.
    """
    overlaid = {
        cap.metric_id
        for m in view.metrics
        if m.error is None
        and not m.headline.not_instrumented
        and (cap := overlay_for(view, m.metric_id))
    }
    cards: list[str] = []
    for m in view.metrics:
        if m.metric_id in overlaid:
            continue
        try:
            cards.append(render_card(view, m, tz, known))
        except Exception as exc:  # noqa: BLE001 - a bad card is a value, not a 500
            cards.append(_error_card(m.metric_id, f"{type(exc).__name__}: {exc}"))
    return "".join(cards)
