"""Value objects the chart primitives draw. Frozen; the primitives never mutate them.

A ``Point`` whose ``value`` is ``None`` is a bucket the source did not cover: the line
breaks there instead of interpolating through it. ``approx`` marks a series reconstructed
from proxy events (spec §6 "Not yet recorded"); it is drawn dashed with an "approx." label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Point:
    at: datetime
    value: float | None


@dataclass(frozen=True)
class Series:
    name: str
    points: tuple[Point, ...]
    approx: bool = False


@dataclass(frozen=True)
class Coverage:
    """Where one source starts: buckets before ``start`` cannot hold its data."""

    source: str
    start: datetime


@dataclass(frozen=True)
class Marker:
    """A labelled vertical rule, e.g. "exact series starts Oct 3"."""

    at: datetime
    text: str


@dataclass(frozen=True)
class Size:
    width: float = 640.0
    height: float = 220.0


@dataclass(frozen=True)
class Panel:
    """One small-multiple panel (normally one repo). ``href`` links its heading."""

    key: str
    series: tuple[Series, ...]
    href: str | None = None


@dataclass(frozen=True)
class Distribution:
    """Duration samples (seconds) for one strip-plot row."""

    label: str
    values: tuple[float, ...]
    approx: bool = False


@dataclass(frozen=True)
class LineSpec:
    """Everything a line chart needs besides its series; shared by the small multiples."""

    title: str
    bucket_seconds: float | None = None
    coverage: tuple[Coverage, ...] = ()
    markers: tuple[Marker, ...] = ()
    unit: str = ""
    takeaway: str | None = None
    domain: tuple[datetime, datetime] | None = None
    size: Size = field(default_factory=Size)
