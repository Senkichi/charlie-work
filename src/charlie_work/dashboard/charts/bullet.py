"""Bullet bars: a current value against its cap, on one scale shared by every row.

A cap that was not reported is drawn as an open, dashed track ending in "cap not
reported" -- never a made-up scale. Over cap is said in words as well as by the ``hot``
class, so colour is never the only signal. The SVG is ``aria-hidden``: the row's HTML text
already carries the label, value and cap.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..pages.now_fmt import esc
from .svg import UNKNOWN_DASH, dash_attr, num, value_text

TRACK = 240.0


@dataclass(frozen=True)
class Bullet:
    label: str
    value: float | None
    cap: float | None
    unit: str = ""


def shared_max(rows: tuple[Bullet, ...]) -> float:
    """The one scale every row is drawn on: the largest value or cap, at least 1."""
    known = [v for r in rows for v in (r.value, r.cap) if v is not None]
    return float(max([1.0, *known]))


def _track(value: float, cap: float | None, scale: float) -> str:
    fill = TRACK * value / scale
    if cap is None:
        return (
            f'<rect class="fill" x="0" y="3" width="{num(fill)}" height="10"/>'
            f'<line class="trk-open" x1="{num(fill)}" x2="{num(TRACK)}" y1="8" y2="8"'
            f"{dash_attr(UNKNOWN_DASH)}/>"
        )
    tick = TRACK * cap / scale
    hot = " hot" if value > cap else ""
    return (
        f'<rect class="trk" x="0" y="5" width="{num(tick)}" height="6"/>'
        f'<rect class="fill{hot}" x="0" y="3" width="{num(fill)}" height="10"/>'
        f'<line class="captick" x1="{num(tick)}" x2="{num(tick)}" y1="0" y2="16"/>'
    )


def bullet_bar(row: Bullet, scale: float | None = None) -> str:
    """One ``<div class="bullet-row">``: label | bar | value text."""
    if row.value is None:
        bar = '<span class="unknown">not measured</span>'
        val = '<span class="unk">—</span>'
    else:
        top = scale if scale is not None else shared_max((row,))
        svg = (
            f'<svg class="bullet-svg" width="{num(TRACK)}" height="16" '
            f'viewBox="0 0 {num(TRACK)} 16" aria-hidden="true">'
            f"{_track(row.value, row.cap, top)}</svg>"
        )
        bar = svg + ('<span class="unknown">cap not reported</span>' if row.cap is None else "")
        cap_txt = value_text(row.cap, row.unit) if row.cap is not None else "cap ?"
        over = (
            '<span class="over text-warn">over cap</span>'
            if row.cap is not None and row.value > row.cap
            else ""
        )
        val = f"{esc(value_text(row.value, row.unit))} / {esc(cap_txt)}{over}"
    return (
        f'<div class="bullet-row"><span class="who">{esc(row.label)}</span>'
        f'<span class="bar">{bar}</span><span class="val">{val}</span></div>'
    )


def bullet_bars(rows: tuple[Bullet, ...]) -> str:
    """Several rows on one shared scale, so bar lengths compare across resources."""
    scale = shared_max(rows)
    return f'<div class="bullets">{"".join(bullet_bar(r, scale) for r in rows)}</div>'
