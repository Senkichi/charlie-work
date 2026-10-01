"""Shared machinery for the dashboard History metrics (spec section 4 / 6).

Every metric is a pure query over ``dashboard.db`` returning a frozen ``Series``. A series
is bucketed over ``MetricQuery`` (``[start, end)`` cut into fixed ``bucket`` slices aligned
to ``start``) and only carries points for buckets that overlap the window its sources
cover: outside that coverage nothing is claimed, inside it a count with no events is a real
zero while a gauge/duration with no samples is simply absent.

Repo attribution is the ``source`` column (the events DB a row came from), never an event
payload; the per-repo breakdown is ``Series.per_repo``.
"""

from __future__ import annotations

import sqlite3
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .rollup_schema import FLEET_SOURCE

Point = tuple[str, float]  # (bucket start, ISO UTC "...Z"; value)
Sample = tuple[str, str, float]  # (event ts, repo, value)


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts).astimezone(UTC)


def iso(moment: datetime) -> str:
    return moment.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class MetricQuery:
    """Time range plus bucket size. ``prior()`` is the equal window just before it."""

    start: datetime
    end: datetime
    bucket: timedelta

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("MetricQuery bounds must be timezone-aware")
        if self.end <= self.start or self.bucket <= timedelta(0):
            raise ValueError("MetricQuery needs end > start and a positive bucket")

    @property
    def start_iso(self) -> str:
        return iso(self.start)

    @property
    def end_iso(self) -> str:
        return iso(self.end)

    @property
    def n_buckets(self) -> int:
        return -(-(self.end - self.start) // self.bucket)

    def prior(self) -> MetricQuery:
        span = self.end - self.start
        return MetricQuery(self.start - span, self.start, self.bucket)

    def bucket_start(self, i: int) -> datetime:
        return self.start + i * self.bucket

    def bucket_index(self, ts: str) -> int | None:
        moment = parse_ts(ts)
        if not self.start <= moment < self.end:
            return None
        return (moment - self.start) // self.bucket


@dataclass(frozen=True)
class Series:
    """One chart series. ``per_repo`` maps repo key -> that repo's points."""

    name: str
    unit: str
    points: tuple[Point, ...]
    per_repo: dict[str, tuple[Point, ...]]
    coverage_start: str | None
    coverage_end: str | None
    approx: bool
    not_instrumented: bool
    label: str = ""
    kind: str = "count"  # count (summed, zero-filled) | gauge | duration | ratio
    partial: bool = False  # sources cover only part of what the metric means
    exact_from: str | None = None  # ts the exact (lifecycle_transition) path takes over
    window_start: str = ""
    window_end: str = ""
    bucket_seconds: int = 0
    n: int = 0  # underlying observations inside the window


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    label: str
    unit: str
    kind: str
    kinds: tuple[str, ...] = ()  # kinds whose absence everywhere means "not instrumented"
    sources: str = "repos"  # repos | fleet | all
    check_instrumented: bool = False  # continuously-emitted kinds only: none seen => not yet
    combine: str = "samples"  # samples | repo_sum (sum of per-repo bucket means)


def open_dashboard_ro(path: Path) -> tuple[sqlite3.Connection | None, str | None]:
    """Open ``dashboard.db`` read-only; ``(conn, error)`` with exactly one None."""
    if not path.is_file():
        return None, f"missing: {path}"
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0)
        conn.execute("SELECT 1 FROM meta LIMIT 1")
    except sqlite3.Error as exc:
        return None, f"cannot open {path}: {exc}"
    return conn, None


def coverage(db: sqlite3.Connection, kinds: Sequence[str], sources: str) -> tuple[str, str] | None:
    """Earliest/latest event ts the matching sources hold for ``kinds`` (None: no data)."""
    if not kinds:
        return None
    marks = ", ".join("?" for _ in kinds)
    where = {"repos": " AND source != ?", "fleet": " AND source = ?", "all": ""}[sources]
    args = (*kinds, *((FLEET_SOURCE,) if where else ()))
    row = db.execute(
        f"SELECT MIN(first_ts), MAX(last_ts) FROM coverage WHERE kind IN ({marks}){where}", args
    ).fetchone()
    return (row[0], row[1]) if row and row[0] and row[1] else None


def _bounds(db: sqlite3.Connection, spec: SeriesSpec) -> tuple[str, str] | None:
    """The span the series' sources were being written (any kind): zeros inside it are real.

    Per-kind coverage would drop the quiet stretches around a rare event, so the bounds are
    the sources' whole span; ``spec.kinds`` only decides ``not_instrumented``.
    """
    return coverage(db, ("*",), spec.sources)


def _reduce(values: list[float], how: str) -> float:
    if how == "sum":
        return float(sum(values))
    if how == "median":
        return float(statistics.median(values))
    return float(statistics.fmean(values))


def _bucketed(q: MetricQuery, samples: Iterable[Sample]) -> dict[str, dict[int, list[float]]]:
    out: dict[str, dict[int, list[float]]] = {}
    for ts, repo, value in samples:
        i = q.bucket_index(ts)
        if i is not None:
            out.setdefault(repo, {}).setdefault(i, []).append(value)
    return out


def _merge_repos(by_repo: dict[str, dict[int, list[float]]]) -> dict[int, list[float]]:
    merged: dict[int, list[float]] = {}
    for per_bucket in by_repo.values():
        for i, vals in per_bucket.items():
            merged.setdefault(i, []).extend(vals)
    return merged


def _in_window(q: MetricQuery, samples: Sequence[Sample]) -> int:
    return sum(q.bucket_index(ts) is not None for ts, _, _ in samples)


def _active(q: MetricQuery, cov: tuple[str, str] | None) -> list[int]:
    """Bucket indexes overlapping the covered span."""
    if cov is None:
        return []
    lo, hi = parse_ts(cov[0]), parse_ts(cov[1])
    return [
        i
        for i in range(q.n_buckets)
        if q.bucket_start(i) + q.bucket > lo and q.bucket_start(i) <= hi
    ]


def _points(
    q: MetricQuery, active: list[int], by_bucket: dict[int, list[float]], how: str
) -> tuple[Point, ...]:
    out = []
    for i in active:
        vals = by_bucket.get(i)
        if vals:
            out.append((iso(q.bucket_start(i)), _reduce(vals, how)))
        elif how == "sum":
            out.append((iso(q.bucket_start(i)), 0.0))
    return tuple(out)


def _seen(db: sqlite3.Connection, spec: SeriesSpec) -> bool:
    return coverage(db, spec.kinds, spec.sources) is not None


def _finish(
    db: sqlite3.Connection,
    q: MetricQuery,
    spec: SeriesSpec,
    cov: tuple[str, str] | None,
    points: tuple[Point, ...],
    per_repo: dict[str, tuple[Point, ...]],
    n: int,
    flags: dict,
) -> Series:
    return Series(
        name=spec.name,
        unit=spec.unit,
        points=points,
        per_repo=per_repo,
        coverage_start=cov[0] if cov else None,
        coverage_end=cov[1] if cov else None,
        approx=flags.get("approx", False),
        not_instrumented=spec.check_instrumented and not _seen(db, spec),
        label=spec.label,
        kind=spec.kind,
        partial=flags.get("partial", False),
        exact_from=flags.get("exact_from"),
        window_start=q.start_iso,
        window_end=q.end_iso,
        bucket_seconds=int(q.bucket.total_seconds()),
        n=n,
    )


def make_series(
    db: sqlite3.Connection,
    q: MetricQuery,
    spec: SeriesSpec,
    combined: Sequence[Sample],
    per_repo: Sequence[Sample],
    *,
    how: str,
    approx: bool = False,
    partial: bool = False,
    exact_from: str | None = None,
) -> Series:
    """Bucket ``combined`` (the all-repos line) and ``per_repo`` samples into a Series.

    ``how`` is sum (zero-filled), mean or median. With ``spec.combine == "repo_sum"`` the
    all-repos line is instead the sum of each repo's bucket value (``combined`` unused).
    """
    cov = _bounds(db, spec)
    active = _active(q, cov)
    by_repo = _bucketed(q, per_repo)
    if spec.combine == "repo_sum":
        buckets: dict[int, list[float]] = {}
        for per_bucket in by_repo.values():
            for i, vals in per_bucket.items():
                buckets.setdefault(i, []).append(_reduce(vals, how))
        points = tuple((iso(q.bucket_start(i)), sum(buckets[i])) for i in active if i in buckets)
        n = _in_window(q, per_repo)
    else:
        points = _points(q, active, _merge_repos(_bucketed(q, combined)), how)
        n = _in_window(q, combined)
    repo_points = {r: _points(q, active, b, how) for r, b in sorted(by_repo.items())}
    flags = {"approx": approx, "partial": partial, "exact_from": exact_from}
    return _finish(db, q, spec, cov, points, repo_points, n, flags)


def make_ratio_series(
    db: sqlite3.Connection,
    q: MetricQuery,
    spec: SeriesSpec,
    num: Sequence[Sample],
    den: Sequence[Sample],
) -> Series:
    """num/den per bucket (buckets with no denominator are absent); ``n`` counts den."""
    cov = _bounds(db, spec)
    active = _active(q, cov)

    def pts(n_by: dict[int, list[float]], d_by: dict[int, list[float]]) -> tuple[Point, ...]:
        out = []
        for i in active:
            d = sum(d_by.get(i, []))
            if d > 0:
                out.append((iso(q.bucket_start(i)), sum(n_by.get(i, [])) / d))
        return tuple(out)

    n_all, d_all = _bucketed(q, num), _bucketed(q, den)
    repo_points = {r: pts(n_all.get(r, {}), d_all[r]) for r in sorted(d_all)}
    points = pts(_merge_repos(n_all), _merge_repos(d_all))
    return _finish(db, q, spec, cov, points, repo_points, _in_window(q, den), {})


def category_series(
    db: sqlite3.Connection,
    q: MetricQuery,
    spec: SeriesSpec,
    rows: Sequence[tuple[str, str, str]],
    fixed: Sequence[str] = (),
    *,
    partial: bool = False,
) -> tuple[Series, ...]:
    """Counts split by category: the total (``spec.name``) then ``name.<category>`` each.

    ``rows`` are ``(ts, repo, category)``; ``fixed`` categories always get a series.
    """
    cats = sorted({str(r[2]) for r in rows} | set(fixed))
    total: list[Sample] = [(ts, repo, 1.0) for ts, repo, _ in rows]
    out = [make_series(db, q, spec, total, total, how="sum", partial=partial)]
    for cat in cats:
        part: list[Sample] = [(ts, repo, 1.0) for ts, repo, c in rows if str(c) == cat]
        child = SeriesSpec(
            f"{spec.name}.{cat}",
            f"{spec.label}: {cat}",
            spec.unit,
            spec.kind,
            spec.kinds,
            spec.sources,
            spec.check_instrumented,
        )
        out.append(make_series(db, q, child, part, part, how="sum", partial=partial))
    return tuple(out)


def gauge_samples(rows: Iterable[tuple]) -> list[Sample]:
    """``(ts, repo, value)`` rows with a non-null value, as floats."""
    return [(ts, repo, float(v)) for ts, repo, v in rows if v is not None]
