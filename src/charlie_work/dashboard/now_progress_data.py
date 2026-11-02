"""The Now page's progress chart data: merges and escalations over four ranges.

One cached read of ``dashboard.db`` yields every (metric, range) the chart toggles between,
so the page can switch metric or range client-side with no further request. The series come
from the same History metric functions (``metrics.TABS``: Flow ``merges_per_day`` and
Quality ``escalations``), over windows ending at the last whole bucket (``bucket_end``), and
each is compared with the equal prior window through ``takeaways.assess`` -- the one place
that decides whether a trend claim is supportable. A missing or unreadable ``dashboard.db``
comes back as :class:`ProgressUnavailable` (the panel says so in one line), never as zeros.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .history_data import bucket_end
from .metrics import TABS
from .metrics_base import MetricQuery, Series, open_dashboard_ro
from .takeaways import FLAT_BELOW, assess

log = logging.getLogger("charlie_work.dashboard")

# range key -> (window, bucket, bucket noun, window phrase)
RANGES: dict[str, tuple[timedelta, timedelta, str, str]] = {
    "24h": (timedelta(hours=24), timedelta(hours=1), "hour", "last 24 hours"),
    "7d": (timedelta(days=7), timedelta(hours=6), "6 hours", "last 7 days"),
    "30d": (timedelta(days=30), timedelta(days=1), "day", "last 30 days"),
    "90d": (timedelta(days=90), timedelta(days=1), "day", "last 90 days"),
}
DEFAULT_RANGE = "7d"

# metric key -> (History tab, metric id, toggle label)
METRICS: dict[str, tuple[str, str, str]] = {
    "merged": ("Flow", "merges_per_day", "Merged"),
    "escalated": ("Quality", "escalations", "Escalated"),
}
DEFAULT_METRIC = "merged"
ERROR_TTL_SECONDS = 15.0


@dataclass(frozen=True)
class ProgressSeries:
    """One (metric, range) chart: whole-bucket points plus the per-repo split."""

    metric: str
    range_key: str
    points: tuple[tuple[str, float], ...]  # (bucket start ISO UTC, count)
    per_repo: dict[str, tuple[tuple[int, float], ...]]  # repo -> (index into points, count)
    delta: float | None  # fractional change vs the prior window; None: no claim supportable
    approx: bool
    not_instrumented: bool

    def to_plain(self) -> dict:
        _, _, bucket_noun, phrase = RANGES[self.range_key]
        return {
            "points": [[ts, v] for ts, v in self.points],
            "repo": {r: [[i, v] for i, v in pts] for r, pts in self.per_repo.items()},
            "delta": self.delta,
            "approx": self.approx,
            "notInstrumented": self.not_instrumented,
            "bucket": bucket_noun,
            "phrase": phrase,
        }


@dataclass(frozen=True)
class ProgressData:
    series: tuple[ProgressSeries, ...]

    def get(self, metric: str, range_key: str) -> ProgressSeries | None:
        return next(
            (s for s in self.series if s.metric == metric and s.range_key == range_key), None
        )


@dataclass(frozen=True)
class ProgressUnavailable:
    reason: str


ProgressResult = ProgressData | ProgressUnavailable


def _delta(cur: Series, pri: Series) -> float | None:
    """Fractional change the takeaway rule supports (None when it claims no comparison)."""
    try:
        _, compared = assess(cur, pri)
    except ValueError:  # windows not adjacent: a programming error, never a page error
        log.exception("progress takeaway failed: %s", cur.name)
        return None
    if compared is None or compared[1] <= 0:
        return None
    change = (compared[0] - compared[1]) / compared[1]
    return 0.0 if abs(change) < FLAT_BELOW else change


def _series(db: sqlite3.Connection, metric: str, range_key: str, now: datetime) -> ProgressSeries:
    tab, metric_id, _ = METRICS[metric]
    span, bucket, _, _ = RANGES[range_key]
    end = bucket_end(now, bucket)
    query = MetricQuery(end - span, end, bucket)
    fn = TABS[tab][metric_id]

    def head(q: MetricQuery) -> Series:
        got = fn(db, q)
        return got[0] if isinstance(got, tuple) else got  # the total comes first

    cur, pri = head(query), head(query.prior())
    index = {ts: i for i, (ts, _) in enumerate(cur.points)}
    per_repo = {
        repo: tuple((index[ts], v) for ts, v in pts if v and ts in index)
        for repo, pts in cur.per_repo.items()
    }
    return ProgressSeries(
        metric,
        range_key,
        cur.points,
        {r: p for r, p in per_repo.items() if p},
        _delta(cur, pri),
        cur.approx,
        cur.not_instrumented,
    )


def load_progress(db_path: Path | None, now: datetime) -> ProgressResult:
    """Every (metric, range) series from ``dashboard.db``, or why there is none."""
    if db_path is None:
        return ProgressUnavailable("no dashboard.db is configured for this server")
    db, error = open_dashboard_ro(db_path)
    if db is None:
        return ProgressUnavailable(error or "dashboard.db unavailable")
    try:
        return ProgressData(
            tuple(_series(db, m, r, now) for m in METRICS for r in RANGES),
        )
    except sqlite3.Error as exc:  # locked / torn mid-rebuild: a value, not a 500
        return ProgressUnavailable(f"cannot read {db_path}: {exc}")
    except (ValueError, KeyError, TypeError) as exc:  # a malformed stored row
        log.exception("progress metrics failed")
        return ProgressUnavailable(f"{type(exc).__name__}: {exc}")
    finally:
        db.close()


class ProgressCache:
    """The last :class:`ProgressResult`, recomputed after ``ttl`` seconds (errors after 15s)."""

    def __init__(
        self,
        db_path: Path | None,
        ttl: float,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._db_path = db_path
        self._ttl = float(ttl)
        self._clock = clock
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._entry: tuple[float, ProgressResult] | None = None
        self.misses = 0  # real loads (tests read it to prove a hit did not query)

    def get(self) -> ProgressResult:
        with self._lock:  # one loader at a time; a concurrent poll waits, then hits
            now_m = self._monotonic()
            if self._entry is not None:
                stamp, result = self._entry
                ttl = self._ttl if isinstance(result, ProgressData) else ERROR_TTL_SECONDS
                if now_m - stamp < min(ttl, self._ttl):
                    return result
            self.misses += 1
            try:
                result = load_progress(self._db_path, self._clock())
            except Exception as exc:  # noqa: BLE001 - a metric bug is a value on the page
                log.exception("progress load failed")
                result = ProgressUnavailable(f"{type(exc).__name__}: {exc}")
            self._entry = (now_m, result)
            return result
