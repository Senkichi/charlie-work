"""History metrics registry (spec section 4): every metric of the four History tabs.

``all_series`` computes the whole catalogue for one ``MetricQuery``; each metric function
is also usable on its own. Scalar metrics return one ``Series``; category metrics return a
tuple (total first, then ``<name>.<category>``). ``tab_series`` flattens either to tuples.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable

from . import metrics_capacity as cap
from . import metrics_flow as flow
from . import metrics_quality as quality
from . import metrics_reliability as rel
from .metrics_base import MetricQuery, Series, open_dashboard_ro

__all__ = ["MetricQuery", "Series", "TABS", "all_series", "open_dashboard_ro"]

Metric = Callable[[sqlite3.Connection, MetricQuery], Series | tuple[Series, ...]]


def _stage(stage: str) -> Metric:
    return lambda db, q: flow.stage_time(db, q, stage)


TABS: dict[str, dict[str, Metric]] = {
    "Flow": {
        "merges_per_day": flow.merges_per_day,
        "lead_time": flow.lead_time,
        **{f"stage_time.{s}": _stage(s) for s in flow.STAGES},
        "wip": flow.work_in_progress,
        "queue_depth": flow.queue_depth,
    },
    "Quality": {
        "verdict_mix": quality.verdict_mix,
        "rework_rate": quality.rework_rate,
        "escalations": quality.escalations,
        "worker_fate": quality.worker_fate,
        "salvage_share": quality.salvage_share,
        "verdicts_missed": quality.verdicts_missed,
    },
    "Capacity": {
        "workers_cap": cap.workers_cap,
        "reviewers_live": cap.reviewers_live,
        "reviewers_cap": cap.reviewers_cap,
        "runners_running": cap.runners_running,
        "runners_capacity": cap.runners_capacity,
        "ci_queue_wait": cap.ci_queue_wait,
        "capped_demand": cap.capped_demand,
    },
    "Reliability": {
        "loop_pass_duration": rel.loop_pass_duration,
        "loop_pass_errors": rel.loop_pass_errors,
        "launch_failures": rel.launch_failures,
        "throttles": rel.throttles,
        "self_deploys": rel.self_deploys,
    },
}


def tab_series(db: sqlite3.Connection, tab: str, q: MetricQuery) -> dict[str, tuple[Series, ...]]:
    """Every metric of one History tab, keyed by metric id."""
    out: dict[str, tuple[Series, ...]] = {}
    for metric_id, fn in TABS[tab].items():
        result = fn(db, q)
        out[metric_id] = result if isinstance(result, tuple) else (result,)
    return out


def all_series(db: sqlite3.Connection, q: MetricQuery) -> dict[str, dict[str, tuple[Series, ...]]]:
    """The full History catalogue: ``{tab: {metric id: series}}``."""
    return {tab: tab_series(db, tab, q) for tab in TABS}
