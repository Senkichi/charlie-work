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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

from .metrics_coverage import Coverage, Scope, build_scope
from .rollup_schema import FLEET_SOURCE
from .timeutil import iso, parse_ts

__all__ = ["iso", "parse_ts"]  # re-exported: metric modules and tests import them from here

Point = tuple[str, float]  # (bucket start, ISO UTC "...Z"; value)
Sample = tuple[str, str, float]  # (event ts, repo, value)


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
    # source -> (first, last) ts of that source's own coverage for this series' kinds
    repo_coverage: dict[str, tuple[str, str]] = field(default_factory=dict)
    stat: str = ""  # per-bucket statistic a chart must name ("median"), "" when obvious
    # repo -> the raw values behind the buckets (duration kinds only), for the
    # distribution strip and its p90 reference.
    samples: dict[str, tuple[float, ...]] = field(default_factory=dict)
    # bucket-start ISO ts -> observations the combined line's point rests on (a
    # ratio's denominator mass); lets takeaways keep thin buckets out of a mean.
    bucket_n: dict[str, float] = field(default_factory=dict)
    # repo -> the same counts for that repo's points (per_repo in count form).
    repo_bucket_n: dict[str, dict[str, float]] = field(default_factory=dict)


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
    any_kind: bool = False  # kinds are alternative evidence: coverage starts at the first one
    pulse: str = "events"  # liveness stream: a quiet bucket is a zero only if this was written


def open_dashboard_ro(path: Path) -> tuple[sqlite3.Connection | None, str | None]:
    """Open ``dashboard.db`` read-only; ``(conn, error)`` with exactly one None.

    ``mode=ro`` and deliberately no ``immutable=1``: the rollup writes this database
    from the same process the readers live in (``server.py`` runs ``run_rollup``
    alongside the page handlers), and WAL auto-checkpoint plus the writer's
    close-checkpoint mutate the main file — an immutable reader takes no locks and
    can observe a checkpoint mid-write as torn pages. Reading through the WAL
    protocol may create ``-shm``/``-wal`` sidecars next to the database; that is the
    accepted price, the same tradeoff ``sources.open_events_ro`` makes for
    ``events.db``.
    """
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


def _reduce(values: list[float], how: str) -> float:
    if how == "sum":
        return float(sum(values))
    if how == "nunique":  # distinct values in the cell (e.g. PRs behind miss attempts)
        return float(len(set(values)))
    if how == "median":
        return float(statistics.median(values))
    return float(statistics.fmean(values))


def _bucketed(
    q: MetricQuery, samples: Iterable[Sample], scope: Scope
) -> dict[str, dict[int, list[float]]]:
    """Samples by repo and bucket, dropping any outside their repo's own coverage."""
    out: dict[str, dict[int, list[float]]] = {}
    for ts, repo, value in samples:
        i = q.bucket_index(ts)
        if i is not None and i in scope.active_for(repo):
            out.setdefault(repo, {}).setdefault(i, []).append(value)
    return out


def _merge_repos(by_repo: dict[str, dict[int, list[float]]]) -> dict[int, list[float]]:
    merged: dict[int, list[float]] = {}
    for per_bucket in by_repo.values():
        for i, vals in per_bucket.items():
            merged.setdefault(i, []).extend(vals)
    return merged


def _count(by_bucket: dict[int, list[float]]) -> int:
    return sum(len(v) for v in by_bucket.values())


def _bucket_counts(
    q: MetricQuery, by_bucket: dict[int, list[float]], weight: Callable = len
) -> dict[str, float]:
    """Bucket-start ts -> observations the bucket's point rests on (``weight`` over its values)."""
    return {iso(q.bucket_start(i)): weight(vals) for i, vals in sorted(by_bucket.items())}


def _points(
    q: MetricQuery,
    active: frozenset[int],
    zero_ok: frozenset[int],
    by_bucket: dict[int, list[float]],
    how: str,
) -> tuple[Point, ...]:
    """Observed buckets, plus (counts only) real zeros where the source was alive."""
    out = []
    for i in sorted(active):
        vals = by_bucket.get(i)
        if vals:
            out.append((iso(q.bucket_start(i)), _reduce(vals, how)))
        elif how in ("sum", "nunique") and i in zero_ok:
            out.append((iso(q.bucket_start(i)), 0.0))
    return tuple(out)


def _scope(db: sqlite3.Connection, q: MetricQuery, spec: SeriesSpec) -> Scope:
    return build_scope(db, q, spec.kinds, spec.sources, spec.pulse, spec.any_kind)


def _seen(db: sqlite3.Connection, spec: SeriesSpec) -> bool:
    return coverage(db, spec.kinds, spec.sources) is not None


def _finish(
    db: sqlite3.Connection,
    q: MetricQuery,
    spec: SeriesSpec,
    cov: Coverage,
    points: tuple[Point, ...],
    per_repo: dict[str, tuple[Point, ...]],
    n: int,
    flags: dict,
    *,
    samples: dict[str, tuple[float, ...]] | None = None,
    bucket_n: dict[str, float] | None = None,
    repo_bucket_n: dict[str, dict[str, float]] | None = None,
) -> Series:
    return Series(
        name=spec.name,
        unit=spec.unit,
        points=points,
        per_repo=per_repo,
        coverage_start=cov.start,
        coverage_end=cov.end,
        approx=flags.get("approx", False),
        not_instrumented=spec.check_instrumented and not _seen(db, spec),
        label=spec.label,
        kind=spec.kind,
        partial=flags.get("partial", False),
        exact_from=flags.get("exact_from"),
        stat=flags.get("stat", ""),
        window_start=q.start_iso,
        window_end=q.end_iso,
        bucket_seconds=int(q.bucket.total_seconds()),
        n=n,
        repo_coverage=dict(cov.spans),
        samples=samples or {},
        bucket_n=bucket_n or {},
        repo_bucket_n=repo_bucket_n or {},
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
    scope = _scope(db, q, spec)
    by_repo = _bucketed(q, per_repo, scope)
    if spec.combine == "repo_sum":
        buckets: dict[int, list[float]] = {}
        for per_bucket in by_repo.values():
            for i, vals in per_bucket.items():
                buckets.setdefault(i, []).append(_reduce(vals, how))
        points = tuple(
            (iso(q.bucket_start(i)), sum(buckets[i]))
            for i in sorted(scope.all_active)
            if i in buckets
        )
        n = sum(_count(b) for b in by_repo.values())
    else:
        merged = _merge_repos(_bucketed(q, combined, scope))
        points = _points(q, scope.all_active, scope.all_active & scope.all_alive, merged, how)
        n = _count(merged)
    repo_points = {
        r: _points(q, scope.active_for(r), scope.zero_for(r), b, how)
        for r, b in sorted(by_repo.items())
    }
    repo_bucket_n = {r: _bucket_counts(q, b) for r, b in sorted(by_repo.items())}
    if spec.combine == "repo_sum":
        bucket_n = _bucket_counts(q, _merge_repos(by_repo))
    else:
        bucket_n = _bucket_counts(q, merged)
    # duration series keep the per-repo raw values (scope-filtered like the buckets) so a
    # card can draw the distribution behind the per-bucket median.
    samples = (
        {r: tuple(v for i in sorted(b) for v in b[i]) for r, b in sorted(by_repo.items())}
        if spec.kind == "duration"
        else None
    )
    # a median line must say so; a repo_sum line is a sum of repo values, not one statistic
    stat = how if how == "median" and spec.combine != "repo_sum" else ""
    flags = {"approx": approx, "partial": partial, "exact_from": exact_from, "stat": stat}
    return _finish(
        db, q, spec, scope.cov, points, repo_points, n, flags,
        samples=samples, bucket_n=bucket_n, repo_bucket_n=repo_bucket_n,
    )  # fmt: skip


def make_ratio_series(
    db: sqlite3.Connection,
    q: MetricQuery,
    spec: SeriesSpec,
    num: Sequence[Sample],
    den: Sequence[Sample],
) -> Series:
    """num/den per bucket (buckets with no denominator are absent); ``n`` counts den."""
    scope = _scope(db, q, spec)

    def pts(
        active: frozenset[int], n_by: dict[int, list[float]], d_by: dict[int, list[float]]
    ) -> tuple[Point, ...]:
        out = []
        for i in sorted(active):
            d = sum(d_by.get(i, []))
            if d > 0:
                out.append((iso(q.bucket_start(i)), sum(n_by.get(i, [])) / d))
        return tuple(out)

    n_all, d_all = _bucketed(q, num, scope), _bucketed(q, den, scope)
    repo_points = {r: pts(scope.active_for(r), n_all.get(r, {}), d_all[r]) for r in sorted(d_all)}
    d_merged = _merge_repos(d_all)
    points = pts(scope.all_active, _merge_repos(n_all), d_merged)
    # a bucket's weight is its denominator mass (sum, not the count of den samples),
    # the same denominator ``pts`` divides by — unit denominators make the two equal
    repo_bucket_n = {r: _bucket_counts(q, d_all[r], sum) for r in sorted(d_all)}
    return _finish(
        db, q, spec, scope.cov, points, repo_points, _count(d_merged), {},
        bucket_n=_bucket_counts(q, d_merged, sum), repo_bucket_n=repo_bucket_n,
    )  # fmt: skip


OTHER = "other"


def top_categories(rows: Sequence[tuple[str, str, str]], top: int) -> list[tuple[str, str, str]]:
    """Keep the ``top`` most frequent categories (ties by name); fold the rest into ``other``."""
    counts: dict[str, int] = {}
    for _, _, cat in rows:
        counts[str(cat)] = counts.get(str(cat), 0) + 1
    ranked = sorted(counts, key=lambda c: (-counts[c], c))[:top]
    return [(ts, repo, c if c in ranked else OTHER) for ts, repo, c in rows]


def category_series(
    db: sqlite3.Connection,
    q: MetricQuery,
    spec: SeriesSpec,
    rows: Sequence[tuple[str, str, str]],
    fixed: Sequence[str] = (),
    *,
    partial: bool = False,
    top: int | None = None,
) -> tuple[Series, ...]:
    """Counts split by category: the total (``spec.name``) then ``name.<category>`` each.

    ``rows`` are ``(ts, repo, category)``; ``fixed`` categories always get a series. With
    ``top`` the categories are bounded to the ``top`` most frequent plus ``other``.
    """
    if top is not None:
        rows = top_categories(rows, top)
    cats = sorted({str(r[2]) for r in rows} | set(fixed))
    total: list[Sample] = [(ts, repo, 1.0) for ts, repo, _ in rows]
    out = [make_series(db, q, spec, total, total, how="sum", partial=partial)]
    for cat in cats:
        part: list[Sample] = [(ts, repo, 1.0) for ts, repo, c in rows if str(c) == cat]
        child = replace(spec, name=f"{spec.name}.{cat}", label=f"{spec.label}: {cat}")
        out.append(make_series(db, q, child, part, part, how="sum", partial=partial))
    return tuple(out)


def gauge_samples(rows: Iterable[tuple]) -> list[Sample]:
    """``(ts, repo, value)`` rows with a non-null value, as floats."""
    return [(ts, repo, float(v)) for ts, repo, v in rows if v is not None]


def pooled(series: Series) -> tuple[float, ...]:
    """Every raw sample the window holds, all repos pooled (empty without samples)."""
    return tuple(v for vs in series.samples.values() for v in vs)
