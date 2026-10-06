"""Per-repo horizontal bars for the Now page drawers.

Text is HTML (CSS-sized), the bar is an ``aria-hidden`` SVG whose geometry is an SVG
attribute (the CSP forbids inline styles). A repo name links to its drill-down only once
that route is registered (``now_fmt.link``).
"""

from __future__ import annotations

from collections.abc import Sequence

from .now_fmt import esc, fmt_float, link, repo_url, short_repo


def repo_label(repo: str) -> str:
    return link(repo_url(repo), short_repo(repo), "rl", repo)


def hbars(
    title: str, rows: Sequence[tuple[str, float, float, str]], *, empty: str = "none"
) -> str:
    """``rows``: (label markup, value, scale, shown text). Bar width is value / scale."""
    head = f'<p class="dtitle">{esc(title)}</p>'
    if not rows:
        return f'{head}<p class="dim">{esc(empty)}</p>'
    body = "".join(
        f'<div class="hbar"><span class="hl">{label}</span>'
        '<svg class="hb" viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true">'
        f'<rect class="hb-fill" x="0" y="0" width="{fmt_float(min(100.0, 100.0 * v / scale) if scale else 0.0)}" '
        f'height="8"/></svg><b>{esc(text)}</b></div>'
        for label, v, scale, text in rows
    )
    return head + body
