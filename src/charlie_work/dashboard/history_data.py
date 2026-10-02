"""The History page's query layer: one cached read of ``dashboard.db`` per (tab, range).

The page never queries per request: :class:`HistoryCache` keeps each (tab, range) result
for ``ttl`` seconds (the rollup interval, so a cached tab is at most one rollup behind).
It reads ``dashboard.db`` only, read-only, through the metrics module, and never fans out
over the per-repo ``events.db`` files. A missing or unreadable ``dashboard.db`` comes
back as a :class:`HistoryUnavailable` value, never as zeros.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, tzinfo
from pathlib import Path

from .metrics import TABS, tab_results
from .metrics_base import MetricQuery, Series, open_dashboard_ro
from .takeaways import Compared, paired_assessments

# Range key -> (window length, bucket size): about 28-30 buckets in every range.
RANGES: dict[str, tuple[timedelta, timedelta]] = {
    "7d": (timedelta(days=7), timedelta(hours=6)),
    "14d": (timedelta(days=14), timedelta(hours=12)),
    "30d": (timedelta(days=30), timedelta(days=1)),
    "90d": (timedelta(days=90), timedelta(days=3)),
}
DEFAULT_RANGE = "7d"
log = logging.getLogger("charlie_work.dashboard")

TAB_KEYS: dict[str, str] = {name.lower(): name for name in TABS}  # url key -> TABS name
DEFAULT_TAB = next(iter(TAB_KEYS))
ERROR_TTL_SECONDS = 15.0  # an unavailable db is retried sooner than a good result


def pick_tab(raw: str | None) -> str:
    """URL ``tab`` value -> a known tab key (unknown or absent: the first tab)."""
    key = (raw or "").lower()
    return key if key in TAB_KEYS else DEFAULT_TAB


def pick_range(raw: str | None) -> str:
    key = (raw or "").lower()
    return key if key in RANGES else DEFAULT_RANGE


_GRID_EPOCH = date(2026, 1, 1)  # multi-day buckets count whole days from here (stable grid)


def bucket_end(now: datetime, bucket: timedelta, tz: tzinfo | None = None) -> datetime:
    """The last local bucket boundary at or before ``now``: History draws whole buckets.

    Sub-day buckets sit on a grid from local midnight (6h: 00/06/12/18), day buckets on
    local midnights, multi-day buckets every N days from ``_GRID_EPOCH``. So a "day" bar is
    one calendar day and boundaries do not move between reloads. The still-filling bucket
    is left out (Now shows the live state): a partial newest bucket would read as a dip,
    and a window reaching past ``now`` would bias a count's per-day rate against the
    prior window. The grid uses ``now``'s UTC offset, so across a DST change older
    boundaries sit an hour off local midnight (fixed-length buckets, by design).
    """
    local = now.astimezone(tz)
    base = local.replace(hour=0, minute=0, second=0, microsecond=0)
    if bucket >= timedelta(days=1):
        base -= timedelta(days=(base.date() - _GRID_EPOCH).days % (bucket // timedelta(days=1)))
    return (base + ((local - base) // bucket) * bucket).astimezone(UTC)


def range_query(range_key: str, now: datetime, tz: tzinfo | None = None) -> MetricQuery:
    """The range's window ending at the last whole local bucket boundary (``bucket_end``)."""
    span, bucket = RANGES[range_key]
    end = bucket_end(now, bucket, tz)
    return MetricQuery(end - span, end, bucket)


@dataclass(frozen=True)
class MetricData:
    """One metric's series for the window plus each series' takeaway (by series name).

    ``error`` is set when the metric could not be computed (its exception);
    ``error_kind`` classifies the fault so the card does not blame the data for a code
    bug: ``"data"`` — a stored row the metric reads is malformed — versus a fault in
    the metric function (``"internal"``) or in the takeaway assessment
    (``"takeaway"``). The card renders as degraded and ``series``/``takeaways`` are
    empty.
    """

    metric_id: str
    series: tuple[Series, ...]
    takeaways: dict[str, str]
    # series name -> the (current, prior) values its takeaway compares, None: no claim
    compared: dict[str, Compared | None] = field(default_factory=dict)
    error: str | None = None
    error_kind: str = "internal"

    @property
    def headline(self) -> Series:
        return self.series[0]


@dataclass(frozen=True)
class HistoryView:
    tab: str  # url key, e.g. "flow"
    range_key: str
    query: MetricQuery
    metrics: tuple[MetricData, ...]
    computed_at: datetime

    def get(self, metric_id: str) -> MetricData | None:
        return next((m for m in self.metrics if m.metric_id == metric_id), None)


@dataclass(frozen=True)
class HistoryUnavailable:
    """Why no History can be shown (the page says so instead of drawing zeros)."""

    reason: str
    computed_at: datetime


HistoryResult = HistoryView | HistoryUnavailable


def load_tab(db_path: Path | None, tab: str, range_key: str, now: datetime) -> HistoryResult:
    """Every metric of one tab over the range, with takeaways vs the prior window."""
    if db_path is None:
        return HistoryUnavailable("no dashboard.db is configured for this server", now)
    query = range_query(range_key, now)
    db, error = open_dashboard_ro(db_path)
    if db is None:
        return HistoryUnavailable(error or "dashboard.db unavailable", now)
    try:
        current = tab_results(db, TAB_KEYS[tab], query)
        prior = tab_results(db, TAB_KEYS[tab], query.prior())
    except sqlite3.Error as exc:  # locked / torn mid-rebuild: a value, not a 500
        return HistoryUnavailable(f"cannot read {db_path}: {exc}", now)
    finally:
        db.close()
    metrics: list[MetricData] = []
    for mid, cur in current.items():
        pri = prior.get(mid, ())
        error = cur if isinstance(cur, Exception) else pri if isinstance(pri, Exception) else None
        # A ValueError out of a metric function is the signature of a malformed stored
        # row (float() on a TEXT value, an unparseable timestamp); a KeyError, TypeError
        # or anything else is a fault in the metric itself, not bad data.
        kind = "data" if isinstance(error, ValueError) else "internal"
        if error is None:
            try:
                got = paired_assessments(cur, pri)
            except Exception as exc:  # noqa: BLE001 - a takeaway fault degrades one card
                log.exception("takeaway assessment failed: %s %s", tab, mid)
                error, kind = exc, "takeaway"
            else:
                metrics.append(
                    MetricData(
                        mid,
                        cur,
                        {name: text for name, (text, _) in got.items()},
                        {name: values for name, (_, values) in got.items()},
                    )
                )
        if error is not None:
            metrics.append(
                MetricData(mid, (), {}, error=f"{type(error).__name__}: {error}", error_kind=kind)
            )
    return HistoryView(tab, range_key, query, tuple(metrics), now)


Loader = Callable[[str, str, datetime], HistoryResult]


class HistoryCache:
    """Per-(tab, range) results, recomputed after ``ttl`` seconds (errors after 15s).

    Each (tab, range) key has its own lock so a cold key's load never serialises a
    different key's; ``_lock`` only guards the entry map and the per-key lock table.
    ``misses`` counts real loads (the tests read it to prove a hit did not query).
    """

    def __init__(
        self,
        load: Loader,
        ttl: float,
        clock: Callable[[], datetime],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._load = load
        self._ttl = float(ttl)
        self._clock = clock
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], tuple[float, HistoryResult]] = {}
        self._key_locks: dict[tuple[str, str], threading.Lock] = {}
        self.misses = 0

    def _fresh_hit(self, key: tuple[str, str], now_m: float) -> HistoryResult | None:
        with self._lock:
            hit = self._entries.get(key)
        if hit is None:
            return None
        ttl = self._ttl if isinstance(hit[1], HistoryView) else ERROR_TTL_SECONDS
        return hit[1] if now_m - hit[0] < min(ttl, self._ttl) else None

    def get(self, tab: str, range_key: str) -> HistoryResult:
        key = (pick_tab(tab), pick_range(range_key))
        if (hit := self._fresh_hit(key, self._monotonic())) is not None:
            return hit
        with self._lock:
            lock = self._key_locks.setdefault(key, threading.Lock())
        with lock:  # only this key loads; a concurrent requester rechecks and waits
            now_m = self._monotonic()
            if (hit := self._fresh_hit(key, now_m)) is not None:
                return hit
            with self._lock:
                self.misses += 1
            try:
                result = self._load(key[0], key[1], self._clock())
            except Exception as exc:  # noqa: BLE001 - a metric bug is a value on the page
                log.exception("history load failed: %s %s", *key)
                result = HistoryUnavailable(f"{type(exc).__name__}: {exc}", self._clock())
            with self._lock:
                self._entries[key] = (now_m, result)
            return result


def history_cache(db_path: Path | None, ttl: float, clock: Callable[[], datetime]) -> HistoryCache:
    return HistoryCache(lambda tab, rng, now: load_tab(db_path, tab, rng, now), ttl, clock)
