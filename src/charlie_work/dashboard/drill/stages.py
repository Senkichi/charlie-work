"""Per-item stage and lead time from one issue's milestones (spec: accumulated across visits).

Same stage definitions and scan as the History metrics (``metrics_flow.APPROX_STAGES``), with
one addition the metric cannot make: a still-open visit is counted up to ``now`` and flagged,
because the drill-down is about one item that may be in a stage right now. An item is exact
only when it has ``lifecycle_transition`` rows (issue #2226); until then every time here is
reconstructed and flagged ``approx``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from ..metrics_flow import APPROX_STAGES, STAGES
from ..timeutil import parse_ts
from .types import StageSpan, StageTime

Milestone = tuple[str, str, bool]  # (ts, name, exact)


def _scan(
    events: Sequence[tuple[str, str]], starts: Sequence[str], ends: Sequence[str]
) -> tuple[list[tuple[str, str]], str | None]:
    """Closed ``(open, close)`` spans plus the open start when the last visit has not ended."""
    spans: list[tuple[str, str]] = []
    opened: str | None = None
    for ts, name in events:
        if opened is None and name in starts:
            opened = ts
        elif opened is not None and name in ends:
            spans.append((opened, ts))
            opened = None
    return spans, opened


def _secs(a: str, b: str) -> float:
    return max(0.0, (parse_ts(b) - parse_ts(a)).total_seconds())


def stage_spans(milestones: Sequence[Milestone], now: datetime) -> tuple[StageSpan, ...]:
    """Every visit to every stage (``STAGES`` order, visits in time order); open ends at ``now``.

    The lane chart draws these and ``stage_times`` sums them, so the bands and the totals
    beside them cannot disagree.
    """
    exact = any(e for _, _, e in milestones)
    evs = [(ts, name) for ts, name, e in milestones if e == exact]
    names = {name for _, name in evs} - {"ready_observed"}
    out: list[StageSpan] = []
    for stage in STAGES:
        if exact:
            starts, ends = (stage,), tuple(names - {stage})
        else:
            starts, ends = APPROX_STAGES[stage]
        spans, opened = _scan(evs, starts, ends)
        out += [StageSpan(stage, a, b, _secs(a, b), False) for a, b in spans]
        if opened is not None:
            end = now.isoformat()
            out.append(StageSpan(stage, opened, end, _secs(opened, end), True))
    return tuple(out)


def stage_times(
    milestones: Sequence[Milestone], now: datetime
) -> tuple[tuple[StageTime, ...], float | None, bool]:
    """``(stage times, lead seconds, approx)`` for one item's time-ordered milestones."""
    exact = any(e for _, _, e in milestones)
    evs = [(ts, name) for ts, name, e in milestones if e == exact]
    spans = stage_spans(milestones, now)
    out: list[StageTime] = []
    for stage in STAGES:
        mine = [s for s in spans if s.stage == stage]
        if mine:
            total = sum(s.seconds for s in mine)
            out.append(StageTime(stage, total, len(mine), any(s.open_now for s in mine)))
    lead_starts, lead_ends = (
        (("ready", "ready_observed"), ("done",))
        if exact
        else (
            ("dispatched",),
            ("merged",),
        )
    )
    lead_spans, _ = _scan(evs, lead_starts, lead_ends)
    lead = _secs(*lead_spans[0]) if lead_spans else None
    return tuple(out), lead, not exact
