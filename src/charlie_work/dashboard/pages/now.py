"""The Now page (spec section 4): exception-first ledger on the left, quiet rail on the right.

Pure: ``render_now`` / ``render_fragment`` take a ``ModelState`` and return HTML. Every
dynamic value goes through ``html.escape`` (``now_fmt.esc``). The page carries no inline
script, no inline style attribute and no ``<style>`` block (CSP ``script-src 'self';
style-src 'self'``): layout lives in ``/static/now.css``, behaviour in
``/static/dashboard.js``.

The whole ``#now`` region is the htmx swap target. It replaces itself (``outerHTML``) on
every poll; the body is never swapped, so page scroll stays put, and every focusable
control carries a stable ``id`` so htmx restores focus to it after the swap.
"""

from __future__ import annotations

from ..now_types import NowModel, RepoFreshness
from ..read_model import ModelState
from .now_capacity import render_capacity, render_repo_ledger
from .now_flow import render_flow, render_not_dispatchable
from .now_fmt import age, esc, local_time, repo_url, short_repo
from .now_keyhelp import KEY_HELP
from .now_needs import render_needs


def _fresh_chip(f: RepoFreshness) -> str:
    name = esc(short_repo(f.repo))
    when = esc(age(f.age_seconds))
    href = esc(repo_url(f.repo))
    if f.error:
        return (
            f'<a class="chip is-danger text-danger" href="{href}" '
            f'title="{esc(f.repo)}: snapshot unreadable: {esc(f.error)}">'
            f'{name} <span class="n">{when}</span> read error</a>'
        )
    if f.stale:
        return (
            f'<a class="chip is-warn text-warn" href="{href}" '
            f'title="{esc(f.repo)}: no loop pass within the stale threshold">'
            f'{name} <span class="n">{when}</span> stale</a>'
        )
    return (
        f'<a class="chip" href="{href}" title="{esc(f.repo)}">'
        f'{name} <span class="n">{when}</span></a>'
    )


def _freshness(model: NowModel) -> str:
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
        '<p class="banner text-danger" role="alert">Numbers frozen: collector is stalled: '
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
        '<p class="banner text-danger" role="alert">Numbers frozen: collector failing'
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
    nav = (
        '<nav class="views" aria-label="Views"><a href="/now" aria-current="page">Now</a>'
        '<a href="/history" class="soon" aria-disabled="true" title="coming soon">History</a>'
        "</nav>"
    )
    fresh = _freshness(model) if model is not None else ""
    theme_btn = (
        '<button type="button" id="theme-toggle" class="themebtn" '
        'aria-label="Theme: system. Activate to change.">theme: system</button>'
    )
    return (
        '<header class="top"><div class="top-line"><span class="brand">Fleet '
        f"<em>· Now</em></span>{nav}{asof}{theme_btn}</div>{fresh}</header>"
    )


def render_fragment(state: ModelState, poll_seconds: int, stalled: str | None = None) -> str:
    """The ``#now`` region: the htmx poll target and the body of the full page."""
    open_tag = (
        f'<div id="now" class="shell" hx-get="/now/fragment" '
        f'hx-trigger="every {int(poll_seconds)}s" hx-swap="outerHTML">'
    )
    head = _header(state, poll_seconds) + _stall_banner(stalled) + _banner(state)
    model = state.model
    if model is None:
        return (
            f'{open_tag}{head}<section class="needs"><p class="calm">Collecting the first '
            "read of the fleet…</p></section></div>"
        )
    rail = (
        '<aside class="rail" aria-label="Flow and capacity">'
        f"{render_flow(model)}{render_not_dispatchable(model)}"
        f"{render_capacity(model)}{render_repo_ledger(model)}</aside>"
    )
    return f"{open_tag}{head}{render_needs(model)}{rail}</div>"


def render_now(
    state: ModelState,
    theme: str = "auto",
    *,
    poll_seconds: int = 20,
    stalled: str | None = None,
) -> str:
    return (
        f'<!doctype html><html lang="en" data-theme="{esc(theme)}"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="htmx-config" content=\'{"includeIndicatorStyles":false}\'>'
        "<title>Fleet Now</title>"
        '<link rel="stylesheet" href="/static/dashboard.css">'
        '<link rel="stylesheet" href="/static/now.css">'
        '<script src="/static/theme-init.js"></script>'
        '<script src="/static/htmx.min.js" defer></script>'
        '<script src="/static/dashboard.js" defer></script></head><body>'
        f"{render_fragment(state, poll_seconds, stalled)}{KEY_HELP}</body></html>"
    )
