"""Markup helpers shared by the chart primitives.

Every helper escapes its dynamic inputs. Colour never appears here: marks carry classes
that ``theme/static/charts.css`` maps to ``--lj-*`` tokens (CSP forbids inline style).
Dash patterns are SVG presentation attributes, not style, so they survive without CSS and
act as the second channel next to colour.
"""

from __future__ import annotations

from datetime import datetime, tzinfo

from ..pages.now_fmt import esc, fmt_float
from ..timeutil import iso

# Categorical second channel: series k gets colour class s{k+1} and this dash pattern.
SERIES_DASH: tuple[str | None, ...] = (None, "9 3", "2 3")
APPROX_DASH = "4 4"
UNKNOWN_DASH = "3 3"


def num(value: float) -> str:
    """Coordinate text: at most one decimal, no trailing zeros, never ``-0``."""
    text = fmt_float(round(value, 1))
    return "0" if text in ("-0", "") else text


def value_text(value: float, unit: str = "") -> str:
    """Tick / label text for a data value: integers bare, else one decimal."""
    text = str(int(value)) if float(value).is_integer() else fmt_float(value)
    return f"{text}{unit}"


def dash_attr(pattern: str | None) -> str:
    return f' stroke-dasharray="{pattern}"' if pattern else ""


def text(x: float, y: float, body: str, cls: str, anchor: str = "start") -> str:
    """An SVG ``<text>``; ``body`` is raw and escaped here."""
    return (
        f'<text class="{esc(cls)}" x="{num(x)}" y="{num(y)}" text-anchor="{anchor}">'
        f"{esc(body)}</text>"
    )


def local_label(moment: datetime, tz: tzinfo | None) -> str:
    return moment.astimezone(tz).strftime("%Y-%m-%d %H:%M")


def time_tag(moment: datetime, tz: tzinfo | None) -> str:
    """Local wall time for reading, UTC ISO in ``datetime`` for machines."""
    return f'<time datetime="{esc(iso(moment))}">{esc(local_label(moment, tz))}</time>'


def svg_open(
    width: float, height: float, label: str, cls: str, attrs: dict[str, str] | None = None
) -> str:
    """Intrinsic-size SVG (1 user unit = 1 CSS px) so text keeps its CSS px size.

    ``role="img"`` is only correct because chart SVGs never contain links; headings that
    link live in HTML outside the SVG.
    """
    extra = "".join(f' {k}="{esc(v)}"' for k, v in sorted((attrs or {}).items()))
    return (
        f'<svg class="{esc(cls)}"{extra} width="{num(width)}" height="{num(height)}" '
        f'viewBox="0 0 {num(width)} {num(height)}" role="img" aria-label="{esc(label)}">'
    )


def caption(
    title: str,
    window: tuple[datetime, datetime] | None,
    tz: tzinfo | None,
    takeaway: str | None,
    sources: str = "",
) -> str:
    """Figure caption: takeaway headline, title, and the window the chart covers."""
    parts = []
    if takeaway:
        parts.append(f'<p class="takeaway">{esc(takeaway)}</p>')
    parts.append(f'<span class="chart-title">{esc(title)}</span>')
    if window is not None:
        parts.append(
            f'<span class="chart-window">{time_tag(window[0], tz)} – '
            f"{time_tag(window[1], tz)} local{sources}</span>"
        )
    return f"<figcaption>{''.join(parts)}</figcaption>"


def empty_figure(title: str, reason: str, cls: str = "chart") -> str:
    return (
        f'<figure class="{esc(cls)} empty-chart"><figcaption>'
        f'<span class="chart-title">{esc(title)}</span></figcaption>'
        f'<p class="empty">{esc(reason)}</p></figure>'
    )
