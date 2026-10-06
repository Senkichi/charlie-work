"""Server-rendered SVG chart primitives for the History tabs and drill-downs.

Pure functions from frozen value objects to escaped markup strings: no I/O, no clock, no
colour literals (classes map to ``--lj-*`` tokens in ``/static/charts.css``). Times are
drawn in the display zone (host-local by default) with UTC ISO kept in ``<time datetime>``.
"""

from __future__ import annotations

from .bullet import Bullet, bullet_bar, bullet_bars, bullet_svg
from .line import line_chart
from .model import Coverage, Distribution, LineSpec, Marker, Panel, Point, Series, Size
from .multiples import small_multiples
from .scale import nice_ticks, time_ticks
from .strip import strip_plot

__all__ = [
    "Bullet",
    "Coverage",
    "Distribution",
    "LineSpec",
    "Marker",
    "Panel",
    "Point",
    "Series",
    "Size",
    "bullet_bar",
    "bullet_bars",
    "bullet_svg",
    "line_chart",
    "nice_ticks",
    "small_multiples",
    "strip_plot",
    "time_ticks",
]
