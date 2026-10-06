"""The Progress panel: one chart, a metric toggle (Merged | Escalated), a range toggle.

All eight (metric, range) series ship in one ``data-progress`` JSON attribute (the CSP
forbids inline script); ``now.js`` / ``now-chart.js`` draw the chart, the headline, the
subtitle and the per-repo split drawer from it, and keep the choice in the URL hash so it
survives the htmx refresh. The headline and subtitle sentences are built here, once, so the
page and the script cannot disagree about wording. A missing ``dashboard.db`` is one line.
"""

from __future__ import annotations

import json

from ..now_progress_data import (
    DEFAULT_METRIC,
    DEFAULT_RANGE,
    METRICS,
    RANGES,
    ProgressData,
    ProgressResult,
    ProgressSeries,
)
from .now_fmt import esc, repo_url
from .routes import routed

# metric -> (singular, plural, toggle verb for tooltips)
_UNIT = {
    "merged": ("PR merged", "PRs merged", "merged"),
    "escalated": ("escalation", "escalations", "escalated"),
}
APPROX_NOTE = "Approximate: reconstructed from proxy events."


def _round(x: float) -> int:
    return int(x + 0.5 + 1e-9)


def headline(s: ProgressSeries) -> tuple[str, str]:
    """``(sentence, comparison)``: "315 PRs merged in the last 7 days", "up 28% vs prior"."""
    one, many, _ = _UNIT[s.metric]
    total = _round(sum(v for _, v in s.points))
    phrase = RANGES[s.range_key][3]
    text = f"{total} {one if total == 1 else many} in the {phrase}"
    if s.delta is None:
        return text, ""
    pct = _round(abs(s.delta) * 100)
    if pct == 0:
        return text, "flat vs prior"
    return text, f"{'up' if s.delta > 0 else 'down'} {pct}% vs prior"


def subtitle(s: ProgressSeries) -> str:
    if s.not_instrumented:
        return "Not instrumented yet: the orchestrator does not record this event."
    label = METRICS[s.metric][2]
    bucket = RANGES[s.range_key][2]
    approx = f" {APPROX_NOTE}" if s.approx else ""
    return f"{label} per {bucket}, all repos.{approx} Click a bar for the repo split."


def payload(data: ProgressData) -> str:
    out: dict = {"metrics": {}, "urls": {}}
    for metric, (_, _, label) in METRICS.items():
        out["metrics"][metric] = {"label": label, "unit": _UNIT[metric][2], "ranges": {}}
    repos: set[str] = set()
    for s in data.series:
        plain = s.to_plain()
        plain["head"], plain["vs"] = headline(s)
        plain["sub"] = subtitle(s)
        out["metrics"][s.metric]["ranges"][s.range_key] = plain
        repos.update(s.per_repo)
    out["urls"] = {r: routed(repo_url(r)) for r in sorted(repos)}
    return json.dumps(out, separators=(",", ":"))


def _seg(name: str, key: str, labels: list[tuple[str, str]], default: str) -> str:
    buttons = "".join(
        f'<button type="button" id="seg-{key}-{k}" data-{key}="{k}" '
        f'aria-pressed="{"true" if k == default else "false"}">{esc(label)}</button>'
        for k, label in labels
    )
    return f'<div class="seg" role="group" aria-label="{esc(name)}">{buttons}</div>'


def render_progress(result: ProgressResult | None) -> str:
    if not isinstance(result, ProgressData):
        reason = getattr(result, "reason", "history has not been read yet")
        return (
            '<section class="panel n-progress" id="progress" aria-labelledby="prog-h">'
            '<div><h2 id="prog-h">Progress over time</h2>'
            f'<p class="sub" data-unavailable="1">History is unavailable: {esc(reason)}. '
            "Run <code>charlie dashboard rollup</code> to build it.</p></div></section>"
        )
    first = result.get(DEFAULT_METRIC, DEFAULT_RANGE)
    head, vs = headline(first) if first else ("", "")
    toggles = _seg("Metric", "m", [(k, v[2]) for k, v in METRICS.items()], DEFAULT_METRIC)
    toggles += _seg("Time range", "r", [(k, k) for k in RANGES], DEFAULT_RANGE)
    return (
        '<section class="panel n-progress" id="progress" aria-labelledby="prog-h">'
        '<div class="ptop"><div><h2 id="prog-h">'
        f'<span id="prog-head">{esc(head)}</span> <span class="dim" id="prog-vs">{esc(vs)}</span></h2>'
        f'<p class="sub" id="prog-sub">{esc(subtitle(first) if first else "")}</p></div>'
        f'<span class="spacer"></span><div class="toggles">{toggles}</div></div>'
        f'<div class="chartbox" id="chart" data-progress="{esc(payload(result))}" '
        'tabindex="0" role="group" aria-label="Progress chart. Left and right arrows pick a '
        'bar, Escape clears.">'
        '<svg class="chart" aria-hidden="true"></svg><div class="tip" id="chart-tip" hidden>'
        '</div></div><div class="drawer" id="chart-drawer" hidden></div></section>'
    )
