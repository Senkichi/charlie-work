"""Shared frame for the drill-down pages: head, header (as of, freshness), breadcrumbs, 404.

Every drill-down carries the same header as Now: the view nav, the model's "as of" time in
local time, the theme toggle, and the per-repo last-pass chips. Breadcrumbs always start at
Now. Pages carry no inline script or style (CSP); layout is in ``/static/drill.css``. Every
dynamic value is escaped here or by the helpers it calls.
"""

from __future__ import annotations

import re

from collections.abc import Sequence
from datetime import tzinfo

from ..drill import DrillError
from ..read_model import ModelState
from ..timeutil import iso, parse_ts
from .nav import THEME_BUTTON, views_nav
from .now import freshness_strip
from .now_fmt import esc, link, local_time
from .now_keyhelp import key_help

Crumb = tuple[str, str | None]  # (text, href); the last crumb is the current page

_STYLES = ("dashboard.css", "now.css", "charts.css", "drill.css")
_STATUS = {"invalid": 404, "not_found": 404, "unavailable": 503}


def ts_tag(ts: str | None, tz: tzinfo | None = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """A UTC event timestamp as local wall time; ISO UTC stays in ``datetime``."""
    if not ts:
        return '<span class="unk">—</span>'
    try:
        moment = parse_ts(ts)
    except ValueError:
        return f'<span class="unk" title="unparseable timestamp">{esc(ts)}</span>'
    return f'<time datetime="{esc(iso(moment))}">{esc(moment.astimezone(tz).strftime(fmt))}</time>'


def crumbs(trail: Sequence[Crumb]) -> str:
    items = [("Now", "/now"), *trail]
    out = []
    for i, (label, href) in enumerate(items):
        if i == len(items) - 1:
            out.append(f'<li aria-current="page">{esc(label)}</li>')
        else:
            out.append(f"<li>{link(href, label, 'crumb')}</li>")
    return f'<nav class="crumbs" aria-label="Breadcrumb"><ol>{"".join(out)}</ol></nav>'


def _header(state: ModelState, kind: str) -> str:
    model = state.model
    asof = (
        f'<span class="asof">as of <b>{local_time(model.generated_at)}</b> (local) '
        "· reload to refresh</span>"
        if model is not None
        else '<span class="asof">collecting…</span>'
    )
    fresh = freshness_strip(model) if model is not None else ""
    return (
        '<header class="top"><div class="top-line"><h1 class="brand">Fleet '
        f"<em>· {esc(kind)}</em></h1>{views_nav('')}{asof}{THEME_BUTTON}</div>{fresh}</header>"
    )


def page(state: ModelState, kind: str, title: str, trail: Sequence[Crumb], body: str) -> str:
    """A full drill-down page. ``kind`` names the view in the brand; ``title`` the tab."""
    links = "".join(f'<link rel="stylesheet" href="/static/{name}">' for name in _STYLES)
    return (
        '<!doctype html><html lang="en" data-theme="auto"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)} · Fleet</title>{links}"
        '<script src="/static/theme-init.js"></script>'
        '<script src="/static/dashboard.js" defer></script></head><body>'
        '<a class="skip" href="#content">Skip to content</a>'
        f'<main id="main" class="drill">{_header(state, kind)}{crumbs(trail)}'
        f'<div id="content" class="dcontent">{body}</div></main>'
        f"{key_help('drill')}</body></html>"
    )


def section(sid: str, heading: str, inner: str, *, note: str = "", focus: bool = False) -> str:
    cls = "dsec focus" if focus else "dsec"
    meta = f' <span class="meta">{note}</span>' if note else ""
    return (
        f'<section class="{cls}" id="{esc(sid)}" aria-labelledby="{esc(sid)}-h">'
        f'<h2 id="{esc(sid)}-h">{esc(heading)}{meta}</h2>{inner}</section>'
    )


_NUM_ATTR = ' class="num"'
_CELL = re.compile(r"<t[dh]\b([^>]*)>")


def _numeric_columns(row: str) -> frozenset[int]:
    """Column indexes whose first-row cell is ``class="num"``: their header right-aligns
    with the figures under it, derived from the cells so no caller lists them twice."""
    return frozenset(
        i for i, m in enumerate(_CELL.finditer(row)) if re.search(r'class="[^"]*\bnum\b', m[1])
    )


def table(caption: str, head: Sequence[str], rows: Sequence[str], empty: str) -> str:
    """A dense table; ``rows`` are pre-rendered ``<tr>`` strings (cells already escaped)."""
    if not rows:
        return f'<p class="calm-sm">{esc(empty)}</p>' if empty else ""
    nums = _numeric_columns(rows[0])
    th = "".join(
        f'<th scope="col"{_NUM_ATTR if i in nums else ""}>{esc(h)}</th>'
        for i, h in enumerate(head)
    )
    return (
        f'<div class="tscroll"><table class="dtable"><caption class="sr">{esc(caption)}'
        f"</caption><thead><tr>{th}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def status_for(err: DrillError) -> int:
    return _STATUS.get(err.code, 404)


def error_page(state: ModelState, err: DrillError, trail: Sequence[Crumb] = ()) -> str:
    """The house-style page for a drill-down that could not be produced (404 or 503)."""
    if status_for(err) == 404:
        heading, lead = "Not found", "Nothing in the fleet matches this address."
    else:
        heading, lead = "Not available", "The data behind this page cannot be read right now."
    body = (
        f'<section class="dmissing" aria-labelledby="miss-h"><h2 id="miss-h">{heading}</h2>'
        f'<p>{lead}</p><p class="why">{esc(err.message)}</p>'
        f"<p>{link('/now', 'Back to Now', 'crumb')}</p></section>"
    )
    return page(state, heading, heading, (*trail, (heading, None)), body)


def not_found_page(state: ModelState, message: str = "no such page") -> str:
    return error_page(state, DrillError("not_found", message))
