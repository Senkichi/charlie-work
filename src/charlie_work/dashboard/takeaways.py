"""Deterministic one-sentence headlines for History charts (spec section 4).

``takeaway(series, prior)`` compares a series with the same metric queried over the equal
window immediately before it (``MetricQuery.prior()``). Rules, in order:

1. ``not_instrumented`` -> says so.
2. The series must cover both windows. ``coverage_start`` is when the LAST kind the series
   is derived from began, so if it falls after the prior window began the comparison is
   refused: "not comparable: <series> starts <date>" (a kind that did not exist yet is not
   a decline, nor a surge). Coverage ending more than one bucket before the current window
   ends is "no trend claimed, data ends ...".
3. Gauges, durations and ratios need at least two populated buckets in each window, and
   every series ``MIN_SAMPLE`` observations in each, else "not enough data". A bucket is
   populated only if it holds at least ``MIN_BUCKET_N`` observations when the series
   carries per-bucket counts — a median over 4 samples must not weigh what one over 40
   does. A count whose prior window is a real zero is exempt (rule 4).
4. Otherwise: direction and percent change, plus the repo contributing most to the change
   when one repo carries at least half of the total per-repo movement. A real zero baseline
   gives the absolute change ("up from 0 to N"). A ``partial`` series ends with "(partial)".

Counts are compared as a per-day rate; a duration that carries its raw samples as the
pooled sample median; every other gauge/duration/ratio as the observation-weighted mean
of the buckets that clear ``MIN_BUCKET_N``. The same inputs always produce the same
string.
"""

from __future__ import annotations

import statistics
from datetime import timedelta

from .metrics_base import Point, Series, parse_ts

MIN_SAMPLE = 5
MIN_BUCKETS = 2  # a gauge/duration/ratio mean over one bucket is a point, not a trend
MIN_BUCKET_N = 5  # a thinner bucket never enters a mean-of-buckets statistic
FLAT_BELOW = 0.05  # |change| under 5% reads as flat
DRIVER_SHARE = 0.5
_DAY = timedelta(days=1)


def _span(series: Series) -> timedelta:
    return parse_ts(series.window_end) - parse_ts(series.window_start)


def _window_label(span: timedelta) -> str:
    days = span / _DAY
    return f"{int(days)}d" if days == int(days) else f"{int(span.total_seconds() // 3600)}h"


def _pooled(series: Series) -> tuple[float, ...]:
    """Every raw sample the window holds, all repos pooled (empty without samples)."""
    return tuple(v for vs in series.samples.values() for v in vs)


def _value(
    points: tuple[Point, ...],
    kind: str,
    span: timedelta,
    *,
    samples: tuple[float, ...] = (),
    bucket_n: dict[str, int] | None = None,
    min_bucket_n: int = MIN_BUCKET_N,
) -> float | None:
    """The window value a headline compares.

    A count is its per-day rate. A duration with raw samples is the pooled sample
    median — a thin outlier bucket then weighs its n, not 1/buckets (#2474). Every
    other kind averages its bucket values weighted by each bucket's observation
    count, skipping buckets below ``min_bucket_n`` (None when none qualify). A
    series carrying no counts keeps the plain mean — no floor can be enforced.
    """
    if not points:
        return None
    if kind == "count":
        return sum(v for _, v in points) / (span / _DAY)
    if kind == "duration" and samples:
        return float(statistics.median(samples))
    if not bucket_n:
        return statistics.fmean(v for _, v in points)
    rows = [(v, n) for ts, v in points if (n := bucket_n.get(ts, 0)) >= min_bucket_n]
    if not rows:
        return None
    return sum(v * n for v, n in rows) / sum(n for _, n in rows)


def _ready_buckets(series: Series, min_bucket_n: int) -> int:
    """How many of the window's populated buckets may enter a statistic.

    A duration with raw samples compares the pooled median, so the floor does not
    apply to it — every populated bucket counts. For a mean-of-buckets kind only
    buckets holding ``min_bucket_n`` or more observations qualify.
    """
    if not series.bucket_n or (series.kind == "duration" and series.samples):
        return len(series.points)
    return sum(1 for ts, _ in series.points if series.bucket_n.get(ts, 0) >= min_bucket_n)


def _pct(change: float) -> int:
    return int(abs(change) * 100 + 0.5 + 1e-9)  # half-up, immune to float dust


def _number(value: float) -> str:
    """At most two decimals, no trailing zeros (2.0 -> "2", 2.333 -> "2.33")."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _headline_label(series: Series) -> str:
    base = series.label or series.name
    return f"{base}/day" if series.kind == "count" else base


def _comparable(repo: str, current: Series, prior: Series) -> bool:
    """A repo can drive a change only if it was covered when the prior window began (its
    own coverage start), and -- for gauges, durations, ratios, which are not zero-filled --
    has points in both windows. An uncovered prior is no baseline, not a real 0."""
    cov = current.repo_coverage.get(repo) or prior.repo_coverage.get(repo)
    tol = timedelta(seconds=current.bucket_seconds)
    if cov is not None and parse_ts(cov[0]) > parse_ts(prior.window_start) + tol:
        return False
    if current.kind != "count":
        return bool(current.per_repo.get(repo)) and bool(prior.per_repo.get(repo))
    return True


def _repo_value(series: Series, repo: str, span: timedelta, min_bucket_n: int) -> float | None:
    """``_value`` for one repo: its own samples/counts, so the driver shares the
    headline's statistic — pooled median for durations, weighted and floored mean
    otherwise."""
    return _value(
        series.per_repo.get(repo, ()),
        series.kind,
        span,
        samples=series.samples.get(repo, ()),
        bucket_n=series.repo_bucket_n.get(repo),
        min_bucket_n=min_bucket_n,
    )


def _driver(
    current: Series, prior: Series, delta: float, span: timedelta, min_bucket_n: int
) -> str | None:
    """Repo with the largest same-direction share of the per-repo movement, if >= half.

    Only repos comparable across both windows (``_comparable``) take part, and a
    repo whose buckets all fall under ``min_bucket_n`` has no baseline to move
    from — the same thin-data rule the headline uses."""
    moves: dict[str, float] = {}
    repos = set(current.per_repo) | set(prior.per_repo)
    for repo in sorted(r for r in repos if _comparable(r, current, prior)):
        cur = _repo_value(current, repo, span, min_bucket_n)
        pri = _repo_value(prior, repo, span, min_bucket_n)
        if current.kind != "count" and (cur is None or pri is None):
            continue
        moves[repo] = (cur or 0.0) - (pri or 0.0)
    total = sum(abs(m) for m in moves.values())
    if len(moves) < 2 or total == 0:
        return None
    repo, move = max(moves.items(), key=lambda kv: (abs(kv[1]), kv[0]))
    if (move > 0) != (delta > 0) or abs(move) / total < DRIVER_SHARE:
        return None
    return repo


def _gap(current: Series, prior: Series) -> str | None:
    """Why the series cannot support a trend claim, or None when both windows are covered."""
    tol = timedelta(seconds=current.bucket_seconds)
    start = current.coverage_start
    end = current.coverage_end
    if start is None or end is None:
        return None  # handled as "not enough data"
    if parse_ts(start) > parse_ts(prior.window_start) + tol:
        return f"not comparable: {current.name} starts {start[:10]}"
    if parse_ts(end) < parse_ts(current.window_end) - tol:
        return f"data ends {end}"
    return None


def takeaway(
    current: Series,
    prior: Series,
    *,
    min_sample: int = MIN_SAMPLE,
    min_bucket_n: int = MIN_BUCKET_N,
) -> str:
    """One-sentence headline for ``current`` against the equal prior window ``prior``."""
    return assess(current, prior, min_sample=min_sample, min_bucket_n=min_bucket_n)[0]


Compared = tuple[float, float]  # (current, prior) window values, chart per-bucket units


def assess(
    current: Series,
    prior: Series,
    *,
    min_sample: int = MIN_SAMPLE,
    min_bucket_n: int = MIN_BUCKET_N,
) -> tuple[str, Compared | None]:
    """``(headline, compared)``. ``compared`` holds the two window values the headline's
    comparison rests on, scaled to the chart's per-bucket units (a count's per-day rate
    times bucket/day), or None when the headline claims no comparison -- so the chart can
    draw exactly what the words compare (D1)."""
    text, values = _headline(current, prior, min_sample, min_bucket_n)
    if values is None:
        return text, None
    scale = current.bucket_seconds / _DAY.total_seconds() if current.kind == "count" else 1.0
    return text, (values[0] * scale, values[1] * scale)


def _headline(
    current: Series, prior: Series, min_sample: int, min_bucket_n: int
) -> tuple[str, Compared | None]:
    span = _span(current)
    if _span(prior) != span or prior.window_end != current.window_start:
        raise ValueError("prior must be the equal window ending where current starts")
    label = _headline_label(current)
    if current.not_instrumented:
        return f"{label}: not instrumented yet", None
    gap = _gap(current, prior)
    if gap:
        return (
            gap if gap.startswith("not comparable") else f"{label}: no trend claimed, {gap}"
        ), None
    window = _window_label(span)
    cur_v = _value(
        current.points,
        current.kind,
        span,
        samples=_pooled(current),
        bucket_n=current.bucket_n,
        min_bucket_n=min_bucket_n,
    )
    pri_v = _value(
        prior.points,
        prior.kind,
        span,
        samples=_pooled(prior),
        bucket_n=prior.bucket_n,
        min_bucket_n=min_bucket_n,
    )
    suffix = (" (approx.)" if current.approx else "") + (" (partial)" if current.partial else "")
    if current.kind == "count" and pri_v == 0 and cur_v is not None:
        # the prior window sits inside coverage, so its zeros are real: no sample floor
        return _from_zero(label, cur_v, window, suffix), (cur_v, 0.0)
    not_enough = f"{label}: not enough data vs prior {window}"
    if (
        current.kind != "count"
        and min(_ready_buckets(current, min_bucket_n), _ready_buckets(prior, min_bucket_n))
        < MIN_BUCKETS
    ):
        return not_enough, None
    if cur_v is None or pri_v is None or min(current.n, prior.n) < min_sample:
        return not_enough, None
    if pri_v == 0:
        return _from_zero(label, cur_v, window, suffix), (cur_v, pri_v)
    change = (cur_v - pri_v) / pri_v
    if abs(change) < FLAT_BELOW:
        sign = "+" if change >= 0 else "-"
        return f"{label} flat vs prior {window} ({sign}{_pct(change)}%){suffix}", (cur_v, pri_v)
    arrow = "↑" if change > 0 else "↓"
    text = f"{label} {arrow}{_pct(change)}% vs prior {window}"
    driver = _driver(current, prior, cur_v - pri_v, span, min_bucket_n)
    return text + (f", driven by {driver}" if driver else "") + suffix, (cur_v, pri_v)


def _from_zero(label: str, cur_v: float, window: str, suffix: str) -> str:
    if cur_v == 0:
        return f"{label} unchanged at 0 vs prior {window}{suffix}"
    return f"{label} up from 0 to {_number(cur_v)} vs prior {window}{suffix}"


NO_PRIOR = "not enough data"


def paired_takeaways(current: tuple[Series, ...], prior: tuple[Series, ...]) -> dict[str, str]:
    """Takeaway per series name; category metrics may carry different categories per
    window, so the pairing is by name and a series with no prior twin gets ``NO_PRIOR``."""
    return {name: text for name, (text, _) in paired_assessments(current, prior).items()}


def paired_assessments(
    current: tuple[Series, ...], prior: tuple[Series, ...]
) -> dict[str, tuple[str, Compared | None]]:
    """``assess`` per series name, paired as ``paired_takeaways`` pairs them."""
    old = {s.name: s for s in prior}
    return {s.name: assess(s, old[s.name]) if s.name in old else (NO_PRIOR, None) for s in current}
