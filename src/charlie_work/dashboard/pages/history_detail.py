"""The selected metric's detail on History: headline, window drawers, chart data.

``render_head`` is the headline ("PRs merged: 315 in the last 7 days up 28% vs prior")
with the takeaway sentence from ``takeaways.py`` verbatim beneath it and the caveats in
words. ``render_lower`` is the "By repo" / "By reason" bar drawers over the whole window
plus the coverage note. ``chart_entry`` is the data ``history-chart.js`` draws from; every
number in it is pre-formatted here (``history_model.fmt_value``) so the page and the
script cannot disagree about units. There is one block of each per metric and
``history.js`` shows the selected one. Pure; every dynamic value is escaped.
"""

from __future__ import annotations

import statistics
from datetime import timedelta, tzinfo

from ..history_data import MetricData, range_phrase
from ..history_model import Card, delta_text, fmt_value, nice_top, on_grid, summarize
from ..metrics_base import Point, Series
from ..timeutil import iso, parse_ts
from .now_fmt import esc, repo_url, short_repo, slug
from .routes import routed

LIFECYCLE_ISSUE = "#2226"
MAX_ROWS = 8  # bars per drawer; the rest are counted in a note
ERROR_DETAIL = {
    "data": "a stored row it reads is malformed",
    "takeaway": "its takeaway could not be computed",
    "internal": "an internal error",
    "render": "the card failed to render",
}


def _lifecycle(metric_id: str) -> bool:
    return metric_id == "lead_time" or metric_id.startswith("stage_time.")


def flag_notes(card: Card) -> tuple[str, ...]:
    """The honest caveats, as words (never colour alone)."""
    notes = []
    if card.approx:
        notes.append(
            f"approx. until lifecycle instrumentation ({LIFECYCLE_ISSUE})"
            if _lifecycle(card.metric_id)
            else "approx.: reconstructed from proxy events"
        )
    if card.partial:
        notes.append("partial: its sources cover only part of what this metric means")
    return tuple(notes)


def stats_text(card: Card) -> str:
    """``median 39m · p90 2.1h · n 162`` for a duration with samples, else ""."""
    if card.stats is None:
        return ""
    n, median, p90 = card.stats
    return f"median {fmt_value(median, card.unit)} · p90 {fmt_value(p90, card.unit)} · n {n}"


def _local_day(ts: str, tz: tzinfo | None) -> str:
    moment = parse_ts(ts)
    day = moment.astimezone(tz).strftime("%m-%d")
    return f'<time datetime="{esc(iso(moment))}">{esc(day)}</time>'


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


def head_id(metric_id: str) -> str:
    return "head-" + slug(metric_id)


def _open(cls: str, card: Card, prefix: str) -> str:
    mid = esc(card.metric_id)
    return f'<div class="{cls}" id="{prefix}-{esc(slug(card.metric_id))}" data-k="{mid}" hidden>'


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def render_head(card: Card, range_key: str, error_kind: str = "internal") -> str:
    """The headline block for one metric (hidden until ``history.js`` selects it)."""
    name = esc(card.name)
    head = _open("mhead", card, "head")
    if card.state == "error":
        detail = ERROR_DETAIL.get(error_kind, ERROR_DETAIL["internal"])
        return (
            f"{head}<h2>{name}: could not be drawn</h2>"
            f'<p class="sub missing">This metric could not be drawn: {esc(detail)} '
            f"({esc(card.error)}); the other cards are unaffected.</p></div>"
        )
    if card.state == "missing":
        issue = f" ({LIFECYCLE_ISSUE})" if _lifecycle(card.metric_id) else ""
        return (
            f"{head}<h2>{name}: not instrumented yet</h2>"
            f'<p class="sub missing"><span class="takeaway">{esc(card.takeaway)}</span>. '
            f"Not instrumented yet{esc(issue)}: no source has recorded the events this metric "
            "is derived from, so nothing is drawn (not a zero).</p></div>"
        )
    word = "" if card.summary == "sum" else f' <span class="dim">{esc(card.summary_word)}</span>'
    bad = ' data-bad="1"' if card.bad else ""
    sentences = [_sentence(n) for n in flag_notes(card)]
    if card.stats is not None:
        sentences.append(_sentence(stats_text(card)))
    sentences.append("Click a bar to see what drove it")
    return (
        f'{head}<h2>{name}: <span class="num">{esc(fmt_value(card.value, card.unit))}</span>'
        f"{word} in the {esc(range_phrase(range_key))} "
        f'<span class="vs"{bad}>{esc(delta_text(card.change))}</span></h2>'
        f'<p class="sub"><span class="takeaway">{esc(card.takeaway)}</span>. '
        f"{esc('. '.join(sentences))}.</p></div>"
    )


def _part_key(child: Series, head: Series) -> str:
    prefix = f"{head.name}."
    key = child.name[len(prefix) :] if child.name.startswith(prefix) else child.name
    return key.replace("_", " ")


def repo_rows(card: Card, head: Series) -> list[tuple[str, float]]:
    """``(repo, window value)`` per repo, largest first; repos with nothing are left out."""
    rows = []
    for repo, pts in head.per_repo.items():
        value = summarize(pts, card.summary, head.samples.get(repo, ()))
        if value:
            rows.append((repo, value))
    return sorted(rows, key=lambda r: (-r[1], r[0]))


def part_rows(card: Card, metric: MetricData) -> list[tuple[str, float]]:
    """``(category, window value)`` per category of a category metric, largest first."""
    head = metric.headline
    rows = []
    for child in metric.series[1:]:
        value = summarize(child.points, card.summary)
        if value:
            rows.append((_part_key(child, head), value))
    return sorted(rows, key=lambda r: (-r[1], r[0]))


def _bar_row(label: str, value: float, top: float, unit: str, first: bool) -> str:
    width = max(0.0, min(100.0, value / top * 100)) if top > 0 else 0.0
    return (
        f'<div class="hbar{" lead" if first else ""}"><span class="hl">{label}</span>'
        '<svg class="hb" viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true" '
        f'focusable="false"><rect class="hb-fill" x="0" y="0" width="{width:.2f}" height="8"/>'
        f"</svg><b>{esc(fmt_value(value, unit))}</b></div>"
    )


def _repo_label(repo: str, known: frozenset[str]) -> str:
    """A registry repo links to its drill-down; any other source is plain text."""
    url = routed(repo_url(repo)) if repo in known else None
    name, full = esc(short_repo(repo)), esc(repo)
    if url is None:
        return f'<span title="{full}">{name}</span>'
    return f'<a href="{esc(url)}" title="{full}">{name}</a>'


def _drawer(title: str, rows: list[tuple[str, float]], labels: list[str], unit: str) -> str:
    if not rows:
        return ""
    top = rows[0][1]
    body = "".join(
        _bar_row(label, value, top, unit, i == 0)
        for i, ((_, value), label) in enumerate(zip(rows[:MAX_ROWS], labels, strict=False))
    )
    more = len(rows) - MAX_ROWS
    note = f'<p class="dtitle">and {more} more</p>' if more > 0 else ""
    return f'<section class="hdrawer"><h3>{esc(title)}</h3>{body}{note}</section>'


def render_lower(card: Card, metric: MetricData, known: frozenset[str], tz: tzinfo | None) -> str:
    """The window drawers and coverage note for one metric (hidden until selected)."""
    lower = _open("mlower", card, "lower")
    if card.state != "ok":
        return f"{lower}</div>"
    head = metric.headline
    repos = repo_rows(card, head)
    parts = part_rows(card, metric)
    drawers = _drawer(
        "By repo over the window",
        repos,
        [_repo_label(r, known) for r, _ in repos[:MAX_ROWS]],
        card.unit,
    ) + _drawer(
        "By reason over the window",
        parts,
        [f"<span>{esc(k)}</span>" for k, _ in parts[:MAX_ROWS]],
        card.unit,
    )
    box = f'<div class="lower">{drawers}</div>' if drawers else ""
    return f"{lower}{box}{coverage_note(head, tz)}</div>"


def _r(value: float) -> float:
    return round(value, 4)


def _cells(points: tuple[Point, ...], index: dict[str, int], unit: str) -> list[list]:
    return [[index[ts], _r(v), fmt_value(v, unit)] for ts, v in points if v and ts in index]


def chart_entry(card: Card, metric: MetricData, grid: tuple[str, ...]) -> dict:
    """What ``history-chart.js`` draws for one metric over the bucket grid ``grid``."""
    if card.state != "ok":
        word = "Not instrumented yet" if card.state == "missing" else "Could not be drawn"
        return {"chart": "none", "msg": word}
    head = metric.headline
    unit = card.unit
    values = on_grid(head.points, grid)
    seen = [v for v in values if v is not None]
    top = nice_top(max(seen, default=0.0))
    ticks = [[0, fmt_value(0.0, unit)]]
    half = top / 2
    if card.kind != "count" or half == int(half):
        ticks.append([_r(half), fmt_value(half, unit)])
    ticks.append([_r(top), fmt_value(top, unit)])
    avg = statistics.fmean(seen) if seen else None
    index = {ts: i for i, ts in enumerate(grid)}
    return {
        "chart": card.chart,
        "approx": card.approx,
        "v": [None if v is None else _r(v) for v in values],
        "f": [None if v is None else fmt_value(v, unit) for v in values],
        "top": _r(top),
        "ticks": ticks,
        "avg": None if avg is None else [_r(avg), "avg " + fmt_value(avg, unit)],
        "repo": {
            r: c for r, pts in sorted(head.per_repo.items()) if (c := _cells(pts, index, unit))
        },
        "parts": {
            _part_key(s, head): c
            for s in metric.series[1:]
            if (c := _cells(s.points, index, unit))
        },
    }
