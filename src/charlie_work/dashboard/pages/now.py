"""The Now page (spec section 4): a window-filling dashboard in reading order.

Needs you (left, tall) -> Progress chart (top right) -> Pipeline -> Capacity. Source health
is a pill in the header, not a ledger row. The panels live in ``now_needs`` /
``now_progress`` / ``now_flow`` / ``now_capacity``; this module owns the frame, the banners
and the htmx target.

Pure: ``render_now`` / ``render_fragment`` take a ``ModelState`` (and the progress chart's
data) and return HTML. Every dynamic value goes through ``html.escape`` (``now_fmt.esc``).
The page carries no inline script, no inline style attribute and no ``<style>`` block (CSP
``script-src 'self'; style-src 'self'``): layout lives in ``/static/now.css`` (shared frame)
and ``/static/now-page.css``, behaviour in ``/static/dashboard.js`` and ``/static/now.js``.

The whole ``#now`` region is the htmx swap target. It replaces itself (``outerHTML``) on
every poll; the body is never swapped, so page scroll stays put, and every focusable
control carries a stable ``id`` so htmx restores focus to it after the swap. UI state
(active needs group, open rows, metric, range, drawers) lives in the URL hash, owned by
``now.js``, and is re-applied after each swap.

Announcements live OUTSIDE the swap target: server banners carry no ``role="alert"``
(a re-inserted alert is re-announced on every poll); they carry ``data-alert`` and
dashboard.js speaks a banner once, when the set of banners changes. ``#client-status``
is the stable polite region where dashboard.js shows "Not updating" when polls fail.
"""

from __future__ import annotations

import json

from ..now_progress_data import ProgressResult
from ..now_types import NowModel, RepoFreshness
from ..read_model import ModelState
from .nav import HEAD_BASE, THEME_BUTTON, views_nav
from .now_capacity import render_capacity
from .now_flow import render_flow
from .now_fmt import age, esc, local_time, repo_url, short_repo
from .now_health import render_health
from .now_keyhelp import key_help
from .now_needs import render_needs
from .now_progress import render_progress
from .routes import routed

# htmx runs under a strict CSP (no inline style, no eval). Settling "style" would copy a
# style attribute that a browser extension (or a test driver hiding the caret) put on an
# old element onto its swapped twin, which the CSP blocks on every poll; eval and inline
# script tags are disabled because nothing here needs them.
HTMX_CONFIG = json.dumps(
    {
        "includeIndicatorStyles": False,
        "attributesToSettle": ["class", "width", "height"],
        "allowEval": False,
        "allowScriptTags": False,
    },
    separators=(",", ":"),
)


def _fresh_chip(f: RepoFreshness) -> str:
    name = esc(short_repo(f.repo))
    when = esc(age(f.age_seconds))
    url = routed(repo_url(f.repo))
    href = f' href="{esc(url)}"' if url else ""
    tag = "a" if url else "span"
    if f.error:
        return (
            f'<{tag} class="chip is-danger text-danger"{href} '
            f'title="{esc(f.repo)}: snapshot unreadable: {esc(f.error)}">'
            f'{name} <span class="n">{when}</span> read error</{tag}>'
        )
    if f.stale:
        return (
            f'<{tag} class="chip is-warn text-warn"{href} '
            f'title="{esc(f.repo)}: no loop pass within the stale threshold">'
            f'{name} <span class="n">{when}</span> stale</{tag}>'
        )
    return (
        f'<{tag} class="chip"{href} title="{esc(f.repo)}">'
        f'{name} <span class="n">{when}</span></{tag}>'
    )


def freshness_strip(model: NowModel) -> str:
    """Last-pass chip per repo (shared by every drill-down header; Now uses the health pill)."""
    chips = "".join(_fresh_chip(f) for f in model.freshness)
    return (
        '<div class="fresh" aria-label="Last loop pass per repo"><span class="label">Last pass'
        f'</span>{chips}<span class="thr">stale after '
        f"{esc(age(model.stale_threshold_seconds))}</span></div>"
    )


def _stall_banner(stalled: str | None) -> str:
    if not stalled:
        return ""
    return (
        '<p class="banner text-danger" data-alert="stall">Numbers frozen: collector is stalled: '
        f"{esc(stalled)}. Showing the last good model.</p>"
    )


def _banner(state: ModelState) -> str:
    if not state.collector_error:
        return ""
    since = (
        f" since {local_time(state.collector_failing_since)} (local)"
        if state.collector_failing_since
        else " now"
    )
    return (
        '<p class="banner text-danger" data-alert="collector">Numbers frozen: collector failing'
        f"{since}: {esc(state.collector_error)}. Showing the last good model.</p>"
    )


def _header(state: ModelState, poll_seconds: int) -> str:
    model = state.model
    asof = (
        f'<span class="asof">as of <b>{local_time(model.generated_at)}</b> (local) '
        f"· auto-refresh {int(poll_seconds)}s</span>"
        if model is not None
        else '<span class="asof">collecting…</span>'
    )
    return (
        '<header class="top"><div class="top-line"><h1 class="brand">Fleet</h1>'
        f'{views_nav("/now")}<span class="spacer"></span>{render_health(model)}{asof}'
        f"{THEME_BUTTON}</div></header>"
    )


def render_fragment(
    state: ModelState,
    poll_seconds: int,
    stalled: str | None = None,
    progress: ProgressResult | None = None,
) -> str:
    """The ``#now`` region: the htmx poll target and the body of the full page."""
    open_tag = (
        f'<div id="now" class="shell" hx-get="/now/fragment" data-poll="{int(poll_seconds)}" '
        f'hx-trigger="every {int(poll_seconds)}s" hx-swap="outerHTML">'
    )
    banners = _stall_banner(stalled) + _banner(state)
    head = _header(state, poll_seconds)
    if banners:
        head += f'<div class="banners">{banners}</div>'
    model = state.model
    if model is None:
        return (
            f'{open_tag}{head}<div class="grid"><section class="panel n-needs" id="needs" '
            'aria-label="Needs you"><p class="calm">Collecting the first read of the fleet…'
            "</p></section></div></div>"
        )
    return (
        f'{open_tag}{head}<div class="grid">{render_needs(model)}{render_progress(progress)}'
        f'<div class="flowrow">{render_flow(model)}{render_capacity(model)}</div></div></div>'
    )


def render_now(
    state: ModelState,
    theme: str = "auto",
    *,
    poll_seconds: int = 20,
    stalled: str | None = None,
    progress: ProgressResult | None = None,
) -> str:
    return (
        f'<!doctype html><html lang="en" data-theme="{esc(theme)}"><head>'
        + HEAD_BASE
        + f'<meta name="htmx-config" content="{esc(HTMX_CONFIG)}">'
        "<title>Fleet Now</title>"
        '<link rel="stylesheet" href="/static/dashboard.css">'
        '<link rel="stylesheet" href="/static/now.css">'
        '<link rel="stylesheet" href="/static/now-page.css">'
        '<script src="/static/theme-init.js"></script>'
        '<script src="/static/htmx.min.js" defer></script>'
        '<script src="/static/staleness.js" defer></script>'
        '<script src="/static/dashboard.js" defer></script>'
        '<script src="/static/now-chart.js" defer></script>'
        '<script src="/static/now.js" defer></script></head><body class="now">'
        '<a class="skip" href="#needs">Skip to Needs you</a>'
        '<div id="client-status" class="client-status" role="status" aria-live="polite"></div>'
        f'<main id="main">{render_fragment(state, poll_seconds, stalled, progress)}</main>'
        f"{key_help()}</body></html>"
    )
