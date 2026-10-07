"""The History page (spec section 4): tabs, a range control, metric cards, one big chart.

One server render covers one range and all four tabs (the tab dots need every tab's
cards): ``/history?range=<range>`` is the full page, ``/history/fragment?range=<range>``
the ``#hist`` region alone. ``#hist`` is the htmx target: it re-fetches itself every
rollup interval (``outerHTML``), and ``history.js`` fetches another range's fragment when
the range changes. UI state (tab, metric, range, picked bucket) lives in the URL hash,
owned by ``history.js`` and re-applied after every swap through ``aria-pressed``,
``data-*`` and ``hidden`` -- never ``class``, which htmx settles back to the server's.

Cards are wrong-way movers first; a tab holding one carries a dot. The page has no inline
script or style (CSP): layout is in now-page.css (the shared fluid frame, ``body.now``)
and history.css (``body.hist``), drawing in history-chart.js. Every dynamic value is
escaped.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import tzinfo

from ..history_data import (
    RANGES,
    TAB_KEYS,
    HistoryResult,
    HistoryUnavailable,
    HistoryView,
    MetricData,
)
from ..history_model import Card, bucket_grid, bucket_label, tab_cards
from .history_cards import render_card
from .history_detail import chart_entry, render_head, render_lower
from .nav import HEAD_BASE, THEME_BUTTON, views_nav
from .now import HTMX_CONFIG
from .now_fmt import esc, local_time, repo_url, short_repo
from .now_keyhelp import key_help
from .routes import routed

HREF = "/history"
FRAGMENT = "/history/fragment"
log = logging.getLogger("charlie_work.dashboard")

Results = Mapping[str, HistoryResult]  # tab key -> that tab's read for one range


def _fault(metric: MetricData, exc: Exception) -> Card:
    """The error card a render fault degrades one metric to."""
    error = f"{type(exc).__name__}: {exc}"
    return Card(metric.metric_id, metric.metric_id, 0, "mean", "error", error=error)


def _metric_parts(
    card: Card, metric: MetricData, view: HistoryView, grid: tuple[str, ...], known, tz
) -> tuple[str, str, str, dict]:
    """``(card, head, lower, chart entry)``; a render fault degrades this metric only."""
    try:
        kind = metric.error_kind
        return (
            render_card(card, metric, grid),
            render_head(card, view.range_key, kind),
            render_lower(card, metric, known, tz),
            chart_entry(card, metric, grid),
        )
    except Exception as exc:  # noqa: BLE001 - a bad card is a value, not a 500
        log.exception("history card failed to render: %s", metric.metric_id)
        # the fallback is fixed text only: nothing in it can fault the same way again
        bad = _fault(metric, exc)
        return (
            render_card(bad, metric, ()),
            render_head(bad, view.range_key, "render"),
            render_lower(bad, metric, known, tz),
            {"chart": "none", "msg": "Could not be drawn"},
        )


def _unavailable(result: HistoryUnavailable, compact: bool = False) -> str:
    title = "Rollup not available"
    body = (
        "History reads the dashboard rollup (<code>dashboard.db</code>), and it cannot be "
        "read right now, so nothing is drawn (no zeros are implied)."
    )
    return (
        f'<section class="hunavail" aria-label="{title}"><h2>{title}</h2><p>{body}</p>'
        f'<p class="why">{esc(result.reason)}</p>'
        + (
            ""
            if compact
            else f"<p>Checked at {local_time(result.computed_at)} (local). Run "
            "<code>charlie dashboard rollup</code> or wait for the server's rollup pass.</p>"
        )
        + "</section>"
    )


def _tabs(flags: dict[str, bool], tab: str) -> str:
    out = []
    for key, name in TAB_KEYS.items():
        dot = (
            '<span class="flag" aria-hidden="true"></span>'
            '<span class="sr"> (a metric here moved the wrong way)</span>'
            if flags.get(key)
            else ""
        )
        flagged = ' data-flag="1"' if flags.get(key) else ""
        out.append(
            f'<button type="button" id="tab-{esc(key)}" data-tab="{esc(key)}"{flagged} '
            f'aria-pressed="{"true" if key == tab else "false"}">{esc(name)}{dot}</button>'
        )
    return f'<div class="htabs" role="group" aria-label="Topic">{"".join(out)}</div>'


def _ranges(range_key: str) -> str:
    out = "".join(
        f'<button type="button" id="range-{esc(k)}" data-range="{esc(k)}" '
        f'aria-pressed="{"true" if k == range_key else "false"}">{esc(k)}</button>'
        for k in RANGES
    )
    return f'<div class="seg" role="group" aria-label="Time range (all tabs)">{out}</div>'


def _meta(views: list[HistoryView], range_key: str) -> str:
    if not views:
        return ""
    view = views[0]
    bucket = bucket_label(int(view.query.bucket.total_seconds()))
    return (
        f'<p class="hmeta">Per {esc(bucket)}, compared with the {esc(range_key)} before · '
        f"read {local_time(view.computed_at)}</p>"
    )


def _unclassified_note(views: list[HistoryView]) -> str:
    """Coverage caveat naming the event kinds the rollup counted but has neither a
    handler nor a known-ignored reason for (issue #2269): they dropped to no rows."""
    kinds = sorted({k for v in views for k in v.unclassified_kinds})
    if not kinds:
        return ""
    return (
        '<p class="cov-note">'
        f"{len(kinds)} event kind{'s' if len(kinds) != 1 else ''} not interpreted "
        f"by the rollup: {esc(', '.join(kinds))}</p>"
    )


def _tab_list(key: str, inner: str, default: str, tab: str) -> str:
    hidden = "" if key == tab else " hidden"
    return (
        f'<div class="cardlist" data-tab="{esc(key)}" data-default="{esc(default)}" '
        f'role="group" aria-label="{esc(TAB_KEYS[key])} metrics"{hidden}>{inner}</div>'
    )


def _payload(range_key: str, grid: tuple[str, ...], metrics: dict, repos: set[str]) -> str:
    out = {
        "range": range_key,
        "t": list(grid),
        "metrics": metrics,
        "names": {r: short_repo(r) for r in sorted(repos)},
        "urls": {r: u for r in sorted(repos) if (u := routed(repo_url(r)))},
    }
    return json.dumps(out, separators=(",", ":"), ensure_ascii=False)


def render_region(
    results: Results,
    range_key: str,
    tab: str,
    *,
    tz: tzinfo | None = None,
    known_repos: frozenset[str] = frozenset(),
    refresh_seconds: int = 120,
) -> str:
    """The ``#hist`` region: the htmx target and the body of the full page.

    ``known_repos`` (the registry's slugs) are the only repos whose drawer rows and
    bucket drawers link to a repo drill-down.
    """
    views = [r for r in results.values() if isinstance(r, HistoryView)]
    open_tag = (
        f'<div id="hist" class="hbody" data-range="{esc(range_key)}" data-tab="{esc(tab)}" '
        f'hx-get="{FRAGMENT}?range={esc(range_key)}" hx-trigger="every {int(refresh_seconds)}s" '
        'hx-swap="outerHTML">'
    )
    lists, heads, lowers, entries, flags = [], [], [], {}, {}
    repos: set[str] = set()
    for key in TAB_KEYS:
        result = results.get(key)
        if not isinstance(result, HistoryView):
            reason = result if isinstance(result, HistoryUnavailable) else None
            inner = _unavailable(reason, compact=True) if reason else ""
            lists.append(_tab_list(key, inner, "", tab))
            continue
        grid = bucket_grid(result.query)
        cards = tab_cards(result)
        flags[key] = any(c.bad for c, _ in cards)
        inner = []
        for card, metric in cards:
            c, h, low, entry = _metric_parts(card, metric, result, grid, known_repos, tz)
            inner.append(c)
            heads.append(h)
            lowers.append(low)
            entries[metric.metric_id] = entry
            repos.update(entry.get("repo", {}))
        default = result.metrics[0].metric_id if result.metrics else ""
        lists.append(_tab_list(key, "".join(inner), default, tab))
    controls = (
        f'<div class="hctl">{_tabs(flags, tab)}<span class="spacer"></span>'
        f"{_meta(views, range_key)}{_ranges(range_key)}</div>"
    )
    if not views:
        first = next((r for r in results.values() if isinstance(r, HistoryUnavailable)), None)
        body = _unavailable(first) if first else ""
        return f"{open_tag}{controls}{body}</div>"
    grid = bucket_grid(views[0].query)
    chart = (
        f'<div class="chartbox" id="hchart" data-hist="{esc(_payload(range_key, grid, entries, repos))}" '
        'tabindex="0" role="group" aria-label="Chart of the selected metric. Left and right '
        'arrows pick a bucket, Escape clears.">'
        '<svg class="chart" aria-hidden="true"></svg>'
        '<div class="tip" id="hchart-tip" hidden></div></div>'
    )
    return (
        f'{open_tag}{controls}<div class="hgrid">'
        f'<section class="panel hcards" id="hcards" aria-label="Metrics">{"".join(lists)}'
        f"{_unclassified_note(views)}</section>"
        f'<section class="panel hmain" id="hmain" aria-label="Selected metric">{"".join(heads)}'
        f'{chart}<div class="drawer hbucket" id="hbucket" hidden></div>{"".join(lowers)}'
        "</section></div></div>"
    )


def render_history(
    results: Results,
    range_key: str,
    tab: str,
    *,
    tz: tzinfo | None = None,
    known_repos: frozenset[str] = frozenset(),
    refresh_seconds: int = 120,
) -> str:
    """The full History page for one range (every tab)."""
    region = render_region(
        results, range_key, tab, tz=tz, known_repos=known_repos, refresh_seconds=refresh_seconds
    )
    return (
        '<!doctype html><html lang="en" data-theme="auto"><head>'
        + HEAD_BASE
        + f'<meta name="htmx-config" content="{esc(HTMX_CONFIG)}">'
        "<title>Fleet History</title>"
        '<link rel="stylesheet" href="/static/dashboard.css">'
        '<link rel="stylesheet" href="/static/now.css">'
        '<link rel="stylesheet" href="/static/now-page.css">'
        '<link rel="stylesheet" href="/static/history.css">'
        '<script src="/static/theme-init.js"></script>'
        '<script src="/static/htmx.min.js" defer></script>'
        '<script src="/static/dashboard.js" defer></script>'
        '<script src="/static/history-chart.js" defer></script>'
        '<script src="/static/history.js" defer></script></head><body class="now hist">'
        '<a class="skip" href="#hmain">Skip to the chart</a>'
        '<main id="main" class="shell"><header class="top"><div class="top-line">'
        f'<h1 class="brand">Fleet</h1>{views_nav(HREF)}<span class="spacer"></span>'
        f"{THEME_BUTTON}</div></header>{region}</main>"
        f"{key_help('history')}</body></html>"
    )
