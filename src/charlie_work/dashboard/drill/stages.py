"""Per-item stage and lead time from one issue's milestones (spec: accumulated across visits).

Same stage definitions and scan as the History metrics (``metrics_flow.APPROX_STAGES`` and
``metrics_flow._scan``), with one addition the metric cannot make: a still-open visit is
counted up to ``now`` and flagged, because the drill-down is about one item that may be in a
stage right now. An item is exact only when it has ``lifecycle_transition`` rows
(issue #2226); until then every time here is reconstructed and flagged ``approx``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from ..metrics_flow import APPROX_STAGES, STAGES, _scan
from ..timeutil import parse_ts
from .types import StageSpan, StageTime

Milestone = tuple[str, str, bool]  # (ts, name, exact)


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
        # A visit ends on any milestone that is not a start of the stage -- the
        # approx and exact paths differ only in where the starts come from. The
        # approx path restarts on a repeated start (a re-dispatch); the exact path
        # keeps the pre-#2473 first-start rule for lifecycle_transition rows.
        starts = (stage,) if exact else APPROX_STAGES[stage]
        spans, opened = _scan(evs, starts, tuple(names - set(starts)), restart=not exact)
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
    lead_spans, _ = _scan(evs, lead_starts, lead_ends, restart=False)
    lead = _secs(*lead_spans[0]) if lead_spans else None
    return tuple(out), lead, not exact
