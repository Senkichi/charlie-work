"""The header view nav shared by every page (``g <key>`` targets in dashboard.js).

A view whose route is registered is a link carrying ``data-go``; the current page's link
carries ``aria-current="page"``; an unbuilt view is plain text, never a link to a 404.
"""

from __future__ import annotations

from .now_fmt import esc
from .routes import VIEWS, View, live_views


def _view(v: View, current: str) -> str:
    if v.href == current:
        return f'<a href="{esc(v.href)}" aria-current="page" data-go="{v.key}">{esc(v.name)}</a>'
    if v in live_views():
        return f'<a href="{esc(v.href)}" data-go="{v.key}">{esc(v.name)}</a>'
    return f'<span class="soon">{esc(v.name)} <small>(soon)</small></span>'


def views_nav(current: str) -> str:
    """``current`` is the href of the page being rendered (e.g. ``/now``)."""
    return (
        '<nav class="views" aria-label="Views">'
        + "".join(_view(v, current) for v in VIEWS)
        + "</nav>"
    )


THEME_BUTTON = (
    '<button type="button" id="theme-toggle" class="themebtn" '
    'aria-label="Theme: system. Activate to change.">theme: system</button>'
)
