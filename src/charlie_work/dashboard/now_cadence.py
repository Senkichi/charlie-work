"""Staleness thresholds for the Now model (pure; no I/O).

Two different clocks live here and must not be mixed up:

* **Snapshot staleness** -- a snapshot is rewritten once per *fleet* pass, which runs
  every repo sequentially and then sleeps ``full_pass_interval_seconds``. The sleep is
  not the cadence. Observed on the real fleet over 24h (133 gaps per repo between
  ``loop_passes.completed_at`` values): p50 519s, p90 1078s, p99 2451s, max 4497s. The old
  ``2 * 300 + 30 = 630s`` rule flagged a healthy fleet stale ~25% of the time.
* **Supervisor heartbeat** -- ``last_beat_at`` is rewritten at the top of every supervisor
  loop iteration, so its age reaches a full pass plus the sleep. ``heartbeat_check`` bounds
  it by ``2 x max_pass_runtime_seconds`` (issue #627); the constants come from the shared
  alarm leaf.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

# The shared alarm leaf owns the supervisor rule, so the dashboard and heartbeat_check
# cannot disagree about when the supervisor is stale.
from charlie_work.heartbeat_alarms_fleet import (
    SUPERVISOR_HEARTBEAT_DEFAULT_PASS_TIMEOUT_SECONDS,
    SUPERVISOR_HEARTBEAT_STALE_MULTIPLIER,
)

# With no heartbeat to say otherwise the fleet sleeps 300s between passes.
DEFAULT_PASS_INTERVAL_SECONDS = 300.0
# A p90 of fewer gaps than this is noise, not an observation of cadence.
MIN_GAP_SAMPLES = 10
# The snapshot threshold is this many observed p90 gaps: one full missed pass beyond the
# slowest routine one (1078s p90 -> 2156s; real worst-case healthy gaps reach ~2450s p99).
P90_MULTIPLIER = 2.0
# Pass intervals the snapshot may age before it is stale when no cadence was observed.
INTERVAL_MULTIPLIER = 3.0


def _positive(value: Any) -> float | None:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
    return float(value) if ok else None


def pass_interval_seconds(heartbeat: Mapping[str, Any] | None) -> float:
    value = _positive((heartbeat or {}).get("full_pass_interval_seconds"))
    return value if value is not None else DEFAULT_PASS_INTERVAL_SECONDS


def max_pass_runtime_seconds(heartbeat: Mapping[str, Any] | None) -> float:
    value = _positive((heartbeat or {}).get("max_pass_runtime_seconds"))
    return value if value is not None else float(SUPERVISOR_HEARTBEAT_DEFAULT_PASS_TIMEOUT_SECONDS)


def supervisor_beat_threshold_seconds(heartbeat: Mapping[str, Any] | None) -> float:
    """Beat age beyond which the supervisor is down: ``heartbeat_check``'s rule (#627).

    ``2 x max_pass_runtime_seconds`` from the heartbeat; an older heartbeat without it
    falls back to ``full_pass_interval_seconds``, then to the default pass timeout.
    """
    base = _positive((heartbeat or {}).get("max_pass_runtime_seconds"))
    if base is None:
        base = _positive((heartbeat or {}).get("full_pass_interval_seconds"))
    if base is None:
        base = float(SUPERVISOR_HEARTBEAT_DEFAULT_PASS_TIMEOUT_SECONDS)
    return SUPERVISOR_HEARTBEAT_STALE_MULTIPLIER * base


def stale_threshold_seconds(
    heartbeat: Mapping[str, Any] | None,
    collector_interval: float,
    observed_gap_p90: float | None = None,
) -> float:
    """Age beyond which a snapshot or runner event is stale, derived from observed cadence.

    ``max(3 x pass interval + collector tick, X)`` where X is ``2 x`` the observed p90 of
    completed-pass gaps when the collector measured one (p90 1078s -> 2156s), else the
    structural bound ``interval + max_pass_runtime`` (300 + 1800 = 2100s): a snapshot can
    legitimately age one whole pass plus the sleep before the next write. The collector
    tick is added so a fresh source never looks stale just because the page re-reads only
    every ``collector_interval``.
    """
    interval = pass_interval_seconds(heartbeat)
    floor = INTERVAL_MULTIPLIER * interval + collector_interval
    if observed_gap_p90 is not None and observed_gap_p90 > 0:
        return max(floor, P90_MULTIPLIER * observed_gap_p90)
    return max(floor, interval + max_pass_runtime_seconds(heartbeat))


def p90(values: Sequence[float]) -> float | None:
    """Nearest-rank 90th percentile; None below ``MIN_GAP_SAMPLES`` observations."""
    if len(values) < MIN_GAP_SAMPLES:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(0.9 * len(ordered)) - 1, 0)]


def completion_gaps(completed_at: Sequence[datetime]) -> list[float]:
    """Seconds between consecutive completions (input in any order)."""
    ordered = sorted(completed_at)
    return [(b - a).total_seconds() for a, b in zip(ordered, ordered[1:], strict=False)]
