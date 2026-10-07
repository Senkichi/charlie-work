"""History metrics registry (spec section 4): every metric of the four History tabs.

``all_series`` computes the whole catalogue for one ``MetricQuery``; each metric function
is also usable on its own. Scalar metrics return one ``Series``; category metrics return a
tuple (total first, then ``<name>.<category>``). ``tab_series`` flattens either to tuples.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from . import metrics_capacity as cap
from . import metrics_flow as flow
from . import metrics_quality as quality
from . import metrics_reliability as rel
from .metrics_base import MetricQuery, Series, open_dashboard_ro

__all__ = ["MetricQuery", "Series", "TABS", "all_series", "open_dashboard_ro"]

Metric = Callable[[sqlite3.Connection, MetricQuery], Series | tuple[Series, ...]]

log = logging.getLogger("charlie_work.dashboard")


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


UP_GOOD, DOWN_GOOD, NEUTRAL = 1, -1, 0
# How a History card summarises a window, by series kind: a count is the window's total,
# a duration the median of its samples, a ratio or a level the mean of its buckets.
SUMMARY_BY_KIND: dict[str, str] = {
    "count": "sum",
    "duration": "median",
    "ratio": "mean",
    "gauge": "mean",
}


@dataclass(frozen=True)
class Presentation:
    """The presentation-only facts about one metric that its series cannot carry.

    ``polarity`` is which direction is good (``UP_GOOD``/``DOWN_GOOD``) or ``NEUTRAL`` when
    neither is (a cap, a level the operator sets); only a non-neutral metric can be
    flagged as moving the wrong way. ``summary`` overrides ``SUMMARY_BY_KIND`` for a level
    whose latest value matters more than its average (``"last"``).
    """

    name: str
    polarity: int
    summary: str | None = None


_STAGE_NAMES = {
    "in_progress": "Time in progress",
    "pr_open": "Time at PR open",
    "reviewing": "Time in review",
    "needs_rework": "Time in rework",
}

# One row per metric id in ``TABS`` (tests/test_dashboard_history_model.py holds the two
# in step): the plain name a card shows, which way is good, and any summary override.
PRESENTATION: dict[str, Presentation] = {
    "merges_per_day": Presentation("PRs merged", UP_GOOD),
    "lead_time": Presentation("Lead time", DOWN_GOOD),
    **{
        f"stage_time.{s}": Presentation(_STAGE_NAMES.get(s, flow.STAGE_NAMES.get(s, s)), DOWN_GOOD)
        for s in flow.STAGES
    },
    "wip": Presentation("Work in progress", NEUTRAL),
    "queue_depth": Presentation("Queue depth", NEUTRAL, "last"),
    "verdict_mix": Presentation("Review verdicts", UP_GOOD),
    "rework_rate": Presentation("Rework rate", DOWN_GOOD),
    "escalations": Presentation("Escalations", DOWN_GOOD),
    "worker_fate": Presentation("Worker exits", NEUTRAL),
    "salvage_share": Presentation("Salvage share", NEUTRAL),
    "verdicts_missed": Presentation("Verdicts missed", DOWN_GOOD),
    "workers_cap": Presentation("Worker cap", NEUTRAL, "last"),
    "reviewers_live": Presentation("Reviewers live", NEUTRAL),
    "reviewers_cap": Presentation("Reviewer cap", NEUTRAL, "last"),
    "runners_running": Presentation("Runners running", NEUTRAL),
    "runners_capacity": Presentation("Runner capacity", NEUTRAL),
    "ci_queue_wait": Presentation("CI queue wait", DOWN_GOOD),
    "capped_demand": Presentation("Capped demand", DOWN_GOOD),
    "loop_pass_duration": Presentation("Loop pass time", DOWN_GOOD),
    "loop_pass_errors": Presentation("Loop errors", DOWN_GOOD),
    "launch_failures": Presentation("Launch failures", DOWN_GOOD),
    "throttles": Presentation("Rate limits", DOWN_GOOD),
    "self_deploys": Presentation("Self-deploys", NEUTRAL),
}


def presentation(metric_id: str, series: Series | None) -> Presentation:
    """The metric's presentation with its summary resolved from the series kind.

    A metric missing from ``PRESENTATION`` is shown under its series label, neutral (it is
    never flagged on a guessed polarity)."""
    got = PRESENTATION.get(metric_id) or Presentation(
        (series.label or series.name) if series is not None else metric_id, NEUTRAL
    )
    kind = series.kind if series is not None else "gauge"
    return Presentation(got.name, got.polarity, got.summary or SUMMARY_BY_KIND.get(kind, "mean"))


def tab_results(
    db: sqlite3.Connection, tab: str, q: MetricQuery
) -> dict[str, tuple[Series, ...] | Exception]:
    """Every metric of one History tab, keyed by metric id.

    A metric that raises — a malformed stored row (``ValueError``) or a fault in the
    metric's own code — comes back as its exception, so that card alone degrades; a
    database-level fault (``sqlite3.Error``) still raises — that is a whole-tab problem,
    not one row's.
    """
    out: dict[str, tuple[Series, ...] | Exception] = {}
    for metric_id, fn in TABS[tab].items():
        try:
            result = fn(db, q)
        except sqlite3.Error:
            raise
        except Exception as exc:  # noqa: BLE001 - degrade this metric's card, not the tab
            log.exception("history metric failed: %s %s", tab, metric_id)
            out[metric_id] = exc
        else:
            out[metric_id] = result if isinstance(result, tuple) else (result,)
    return out


def tab_series(db: sqlite3.Connection, tab: str, q: MetricQuery) -> dict[str, tuple[Series, ...]]:
    """Every metric of one History tab, keyed by metric id — strict: bad rows raise."""
    out: dict[str, tuple[Series, ...]] = {}
    for metric_id, result in tab_results(db, tab, q).items():
        if isinstance(result, Exception):
            raise result
        out[metric_id] = result
    return out


def all_series(db: sqlite3.Connection, q: MetricQuery) -> dict[str, dict[str, tuple[Series, ...]]]:
    """The full History catalogue: ``{tab: {metric id: series}}``."""
    return {tab: tab_series(db, tab, q) for tab in TABS}
