"""Which buckets a series may speak about (spec sections 3/4).

*Coverage* is per source (repo) and per KIND: a series derived from kinds K covers a source
from the moment every kind in K had been seen there (``any_kind`` specs, whose kinds are
alternative evidence of the same fact, from the first of them) until the source's last event
of any kind. Fleet-wide, a kind starts at its earliest sighting in any in-scope source, so
the series' ``coverage_start`` is when its LAST contributing kind began. Nothing is claimed
before it, per repo or fleet-wide: a share or count built from a kind that did not yet exist
would otherwise read as a real zero or a fake 100%.

*Liveness* separates a quiet bucket from a blackout: a count bucket inside coverage is a real
zero only if the source wrote something in it (``pulse`` rows, hourly, from the rollup);
otherwise the bucket is absent.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

from .rollup_schema import FLEET_SOURCE
from .timeutil import parse_ts

if TYPE_CHECKING:
    from .metrics_base import MetricQuery

_HOUR = timedelta(hours=1)
_SCOPE_SQL = {"repos": " AND source != ?", "fleet": " AND source = ?", "all": ""}


@dataclass(frozen=True)
class Coverage:
    start: str | None  # fleet-wide: when the last contributing kind began
    end: str | None  # latest event of any kind in scope
    spans: dict[str, tuple[str, str]] = field(default_factory=dict)  # source -> own span


def load_coverage(
    db: sqlite3.Connection, kinds: Sequence[str], sources: str, any_kind: bool = False
) -> Coverage:
    marks = ", ".join("?" for _ in kinds)
    where = _SCOPE_SQL[sources]
    args = (*kinds, *((FLEET_SOURCE,) if where else ()))
    rows = db.execute(
        f"SELECT source, kind, first_ts, last_ts FROM coverage WHERE (kind IN ({marks})"
        f" OR kind = '*'){where} AND first_ts IS NOT NULL",
        args,
    ).fetchall()
    ends = {s: last for s, k, _, last in rows if k == "*"}
    firsts: dict[str, dict[str, str]] = {}
    for source, kind, first, _ in rows:
        if kind in kinds:
            firsts.setdefault(source, {})[kind] = first
    pick = min if any_kind else max
    spans = {s: (pick(f.values()), ends[s]) for s, f in firsts.items() if s in ends}
    kind_start: dict[str, str] = {}
    for per_kind in firsts.values():
        for kind, first in per_kind.items():
            kind_start[kind] = min(first, kind_start.get(kind, first))
    if not spans:
        return Coverage(None, None)
    return Coverage(pick(kind_start.values()), max(e for _, e in spans.values()), spans)


def overlapping(q: MetricQuery, lo: str, hi: str) -> frozenset[int]:
    """Bucket indexes overlapping ``[lo, hi]``."""
    a, b = parse_ts(lo), parse_ts(hi)
    return frozenset(
        i
        for i in range(q.n_buckets)
        if q.bucket_start(i) + q.bucket > a and q.bucket_start(i) <= b
    )


def alive_buckets(
    db: sqlite3.Connection, q: MetricQuery, stream: str, sources: str
) -> dict[str, frozenset[int]]:
    """Per source, the buckets in which it wrote anything on ``stream`` (hourly resolution)."""
    where = _SCOPE_SQL[sources]
    first_hour = q.start.strftime("%Y-%m-%dT%H")
    args = (stream, first_hour, q.end_iso[:13], *((FLEET_SOURCE,) if where else ()))
    rows = db.execute(
        f"SELECT source, hour FROM pulse WHERE stream = ? AND hour >= ? AND hour <= ?{where}",
        args,
    ).fetchall()
    out: dict[str, set[int]] = {}
    for source, hour in rows:
        begin = parse_ts(f"{hour}:00:00+00:00")
        first = (begin - q.start) // q.bucket
        last = (begin + _HOUR - timedelta(seconds=1) - q.start) // q.bucket
        out.setdefault(source, set()).update(
            i for i in range(max(first, 0), min(last, q.n_buckets - 1) + 1)
        )
    return {s: frozenset(v) for s, v in out.items()}


@dataclass(frozen=True)
class Scope:
    """Per-source active (inside coverage) and alive (source wrote) bucket sets."""

    cov: Coverage
    active: dict[str, frozenset[int]]
    alive: dict[str, frozenset[int]]

    @property
    def all_active(self) -> frozenset[int]:
        return frozenset().union(*self.active.values())

    @property
    def all_alive(self) -> frozenset[int]:
        return frozenset().union(*self.alive.values())

    def active_for(self, repo: str | None) -> frozenset[int]:
        """Buckets a sample of ``repo`` may land in (None / unknown repo: the fleet's union)."""
        return self.active[repo] if repo in self.active else self.all_active

    def zero_for(self, repo: str | None) -> frozenset[int]:
        """Buckets where an empty count is a real zero for ``repo``."""
        if repo in self.active:
            return self.active[repo] & self.alive.get(repo, frozenset())
        return self.all_active & self.all_alive


def build_scope(
    db: sqlite3.Connection,
    q: MetricQuery,
    kinds: Sequence[str],
    sources: str,
    stream: str,
    any_kind: bool,
) -> Scope:
    cov = load_coverage(db, kinds, sources, any_kind)
    active = {s: overlapping(q, lo, hi) for s, (lo, hi) in cov.spans.items()}
    return Scope(cov, active, alive_buckets(db, q, stream, sources))
