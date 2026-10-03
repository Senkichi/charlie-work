"""The History page (spec section 4): four tabs, one shared range, one card per metric.

Server-rendered and link-driven: ``/history?tab=<tab>&range=<range>``. Tabs and range
presets are plain links (keyboard-reachable with Tab, no JS needed); the selected one
carries ``aria-current``. Changing tab keeps the range and changing range keeps the tab.
The page has no poll: it is a reading of ``dashboard.db`` as of the cached computation,
and says when that was. No inline script or style (CSP); layout is in history.css.
"""

from __future__ import annotations

from datetime import tzinfo
from urllib.parse import urlencode

from ..history_data import (
    RANGES,
    TAB_KEYS,
    HistoryResult,
    HistoryUnavailable,
    HistoryView,
)
from ..charts.svg import time_tag
from .history_cards import bucket_words, render_cards
from .nav import HEAD_BASE, THEME_BUTTON, views_nav
from .now_fmt import esc, local_time
from .now_keyhelp import key_help

HREF = "/history"


def history_url(tab: str, range_key: str) -> str:
    return f"{HREF}?{urlencode({'tab': tab, 'range': range_key})}"


def _tabs(tab: str, range_key: str) -> str:
    links = []
    for key, name in TAB_KEYS.items():
        current = ' aria-current="page"' if key == tab else ""
        links.append(
            f'<a id="tab-{esc(key)}" href="{esc(history_url(key, range_key))}"{current}>'
            f"{esc(name)}</a>"
        )
    return f'<nav class="htabs" aria-label="History tabs">{"".join(links)}</nav>'


def _ranges(tab: str, range_key: str) -> str:
    links = []
    for key in RANGES:
        current = ' aria-current="true"' if key == range_key else ""
        links.append(
            f'<a id="range-{esc(key)}" href="{esc(history_url(tab, key))}"{current}>{esc(key)}</a>'
        )
    return (
        '<nav class="hrange" aria-label="Time range (all tabs)"><span class="label">Range'
        f"</span>{''.join(links)}</nav>"
    )


def _header(tab: str, range_key: str) -> str:
    return (
        '<header class="top"><div class="top-line"><h1 class="brand">Fleet '
        f"<em>· History</em></h1>{views_nav(HREF)}{THEME_BUTTON}</div>"
        f'<div class="hcontrols">{_tabs(tab, range_key)}{_ranges(tab, range_key)}</div>'
        "</header>"
    )


def _meta(view: HistoryView, tz: tzinfo | None) -> str:
    q = view.query
    return (
        f'<p class="hmeta">{esc(TAB_KEYS[view.tab])} over the last {esc(view.range_key)}: '
        f"{time_tag(q.start, tz)} – {time_tag(q.end, tz)} (local), "
        f"{esc(bucket_words(int(q.bucket.total_seconds())))} buckets. Each headline compares "
        f"with the {esc(view.range_key)} before. Read from dashboard.db at "
        f"{local_time(view.computed_at)} (local).</p>"
    )


def _unclassified_note(view: HistoryView) -> str:
    """Coverage caveat naming the event kinds the rollup counted but has neither a
    handler nor a known-ignored reason for (issue #2269): they dropped to no rows."""
    kinds = view.unclassified_kinds
    if not kinds:
        return ""
    return (
        '<p class="cov-note">'
        f"{len(kinds)} event kind{'s' if len(kinds) != 1 else ''} not interpreted "
        f"by the rollup: {esc(', '.join(kinds))}</p>"
    )


def _unavailable(result: HistoryUnavailable) -> str:
    return (
        '<section class="hunavail" aria-label="Rollup not available">'
        "<h2>Rollup not available</h2><p>History reads the dashboard rollup "
        "(<code>dashboard.db</code>), and it cannot be read right now, so nothing is drawn "
        f'(no zeros are implied).</p><p class="why">{esc(result.reason)}</p>'
        f"<p>Checked at {local_time(result.computed_at)} (local). Run "
        "<code>charlie dashboard rollup</code> or wait for the server's rollup pass.</p>"
        "</section>"
    )


def render_history(
    result: HistoryResult,
    tab: str,
    range_key: str,
    tz: tzinfo | None = None,
    known_repos: frozenset[str] = frozenset(),
) -> str:
    """The full History page for one (tab, range) read; ``known_repos`` (the registry's
    slugs) are the only repos whose panels link to a repo drill-down."""
    if isinstance(result, HistoryView):
        cards = render_cards(result, tz, known_repos)
        body = (
            _meta(result, tz) + _unclassified_note(result) + f'<div class="hcards">{cards}</div>'
        )
    else:
        body = _unavailable(result)
    panel = (
        f'<section id="hpanel" class="hpanel" aria-labelledby="tab-{esc(tab)}">{body}</section>'
    )
    return (
        '<!doctype html><html lang="en" data-theme="auto"><head>'
        + HEAD_BASE
        + f"<title>Fleet History · {esc(TAB_KEYS[tab])}</title>"
        '<link rel="stylesheet" href="/static/dashboard.css">'
        '<link rel="stylesheet" href="/static/now.css">'
        '<link rel="stylesheet" href="/static/charts.css">'
        '<link rel="stylesheet" href="/static/history.css">'
        '<script src="/static/theme-init.js"></script>'
        '<script src="/static/dashboard.js" defer></script></head><body>'
        '<a class="skip" href="#hpanel">Skip to charts</a>'
        f'<main id="main" class="hist">{_header(tab, range_key)}{panel}</main>'
        f"{key_help('history')}</body></html>"
    )
