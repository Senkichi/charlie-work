# ruff: noqa: F811  (the imported ``ro_db`` fixture is re-bound as a test parameter)
"""Takeaway headlines (``dashboard/takeaways.py``): exact strings for every rule."""

from __future__ import annotations

from dataclasses import replace

import pytest
from _dashboard_metrics_fixtures import (  # noqa: F401  (ro_db is a pytest fixture)
    CURRENT,
    PRIOR,
    day,
    ro_db,
)

from _dashboard_rollup_fixtures import ALPHA

from charlie_work.dashboard import metrics_flow as flow
from charlie_work.dashboard import metrics_quality as quality
from charlie_work.dashboard.metrics_base import MetricQuery, Series
from charlie_work.dashboard.takeaways import takeaway

DAY = 86400


def series(start: str, end: str, per_repo: dict, **kw) -> Series:
    """Synthetic count series: one point per listed value, spread from ``start``."""
    points = tuple(
        (f"{start[:8]}{int(start[8:10]) + i:02d}T00:00:00Z", v)
        for i, v in enumerate(next(iter(per_repo.values())))
    )
    total = {
        "points": points
        if len(per_repo) == 1
        else tuple(
            (t, sum(vals[i] for vals in per_repo.values())) for i, (t, _) in enumerate(points)
        ),
    }
    base = dict(
        name="merges_per_day", unit="merges", label="Merges", kind="count", approx=False,
        not_instrumented=False, window_start=start, window_end=end, bucket_seconds=DAY,
        coverage_start="2026-09-01T00:00:00Z", coverage_end=end, n=int(sum(sum(v) for v in per_repo.values())),
        per_repo={r: tuple((points[i][0], v) for i, v in enumerate(vals)) for r, vals in per_repo.items()},
    )  # fmt: skip
    return Series(**{**base, **total, **kw})


def pair(cur: dict, pri: dict, **kw) -> tuple[Series, Series]:
    return (
        series("2026-10-04T00:00:00Z", "2026-10-07T00:00:00Z", cur, **kw),
        series("2026-10-01T00:00:00Z", "2026-10-04T00:00:00Z", pri, **kw),
    )


def test_up_with_driver_and_window_label() -> None:
    cur, pri = pair({"a": [4, 4, 4], "b": [1, 1, 1]}, {"a": [1, 1, 1], "b": [1, 1, 1]})
    assert takeaway(cur, pri) == "Merges/day ↑150% vs prior 3d, driven by a"


def test_repo_uncovered_in_prior_window_never_drives() -> None:
    # "b" only began reporting mid-way through the current window: no baseline to move from
    late = {"b": ("2026-10-05T00:00:00Z", "2026-10-07T00:00:00Z")}
    cur = {"a": [6, 6, 6], "b": [9, 9, 9], "c": [2, 2, 2]}
    pri = {"a": [2, 2, 2], "b": [0, 0, 0], "c": [2, 2, 2]}
    got = takeaway(*pair(cur, pri, repo_coverage=late))
    assert got == "Merges/day ↑325% vs prior 3d, driven by a"
    # a gauge is not zero-filled: a repo with no prior points has no baseline either
    c, p = pair(cur, pri, kind="gauge", label="Lead time", n=99)
    p = replace(p, per_repo={k: v for k, v in p.per_repo.items() if k != "b"})
    assert takeaway(c, p).endswith("driven by a")


def test_no_driver_when_movement_is_spread() -> None:
    cur = {"a": [2, 2, 2], "b": [2, 2, 2], "c": [2, 2, 2]}
    pri = {"a": [1, 1, 1], "b": [1, 1, 1], "c": [1, 1, 1]}
    assert takeaway(*pair(cur, pri)) == "Merges/day ↑100% vs prior 3d"  # each repo 1/3


def test_flat_and_unchanged_and_from_zero() -> None:
    cur, pri = pair({"a": [3, 3, 3]}, {"a": [3, 3, 3]})
    assert takeaway(cur, pri) == "Merges/day flat vs prior 3d (+0%)"
    cur, pri = pair({"a": [1, 1, 3]}, {"a": [0, 0, 0]})
    assert takeaway(cur, pri) == "Merges/day up from 0 to 1.67 vs prior 3d"
    cur, pri = pair({"a": [0, 0, 0]}, {"a": [0, 0, 0]})  # real zeros inside coverage
    assert takeaway(cur, pri) == "Merges/day unchanged at 0 vs prior 3d"


def test_not_enough_data_below_minimum_sample() -> None:
    cur, pri = pair({"a": [1, 1, 1]}, {"a": [3, 3, 3]})  # n = 3 and 9
    assert takeaway(cur, pri) == "Merges/day: not enough data vs prior 3d"
    assert takeaway(cur, pri, min_sample=3).startswith("Merges/day ↓67%")
    empty, pri2 = pair({"a": []}, {"a": [3, 3, 3]})
    assert takeaway(empty, pri2) == "Merges/day: not enough data vs prior 3d"


def test_gauge_uses_bucket_mean_and_approx_suffix() -> None:
    cur, pri = pair({"a": [6, 8]}, {"a": [4, 4]}, kind="gauge", label="Lead time", approx=True)
    assert takeaway(cur, pri, min_sample=1) == "Lead time ↑75% vs prior 3d (approx.)"


def test_never_claims_a_trend_inside_a_coverage_gap() -> None:
    cur, pri = pair({"a": [9, 9, 9]}, {"a": [2, 2, 2]})
    started_late = replace(cur, coverage_start="2026-10-03T00:00:00Z")
    assert takeaway(started_late, pri) == "not comparable: merges_per_day starts 2026-10-03"
    ended_early = replace(cur, coverage_end="2026-10-05T00:00:00Z")
    assert takeaway(ended_early, pri) == (
        "Merges/day: no trend claimed, data ends 2026-10-05T00:00:00Z"
    )
    # a one-bucket lag at the end is normal (the source has not written since)
    lagging = replace(cur, coverage_end="2026-10-06T00:00:00Z")
    assert takeaway(lagging, pri) == "Merges/day ↑350% vs prior 3d"


def test_mismatched_windows_are_rejected() -> None:
    cur, pri = pair({"a": [1, 1, 1]}, {"a": [1, 1, 1]})
    with pytest.raises(ValueError):
        takeaway(pri, cur)


def test_headlines_from_a_real_rollup(ro_db) -> None:
    lead = takeaway(flow.lead_time(ro_db, CURRENT), flow.lead_time(ro_db, PRIOR))
    assert lead == "Lead time: not enough data vs prior 7d"  # prior window has no lead times
    early = MetricQuery(PRIOR.start, PRIOR.end, PRIOR.bucket)  # its prior starts 09-17
    s = flow.merges_per_day(ro_db, early)
    assert takeaway(s, flow.merges_per_day(ro_db, early.prior())) == (
        "not comparable: merges_per_day starts 2026-09-25"  # first merge evidence, 09-25T12
    )
    share = quality.salvage_share(ro_db, CURRENT)
    assert takeaway(share, quality.salvage_share(ro_db, PRIOR)) == (
        "not comparable: salvage_share starts 2026-10-03"  # worker_handoff_pr_opened first seen
    )
    assert ALPHA in flow.merges_per_day(ro_db, CURRENT).per_repo


def test_partial_series_headlines_say_so() -> None:
    cur, pri = pair(
        {"a": [4, 4, 4], "b": [1, 1, 1]}, {"a": [1, 1, 1], "b": [1, 1, 1]}, partial=True
    )
    assert takeaway(cur, pri) == "Merges/day ↑150% vs prior 3d, driven by a (partial)"
    cur, pri = pair({"a": [0, 0, 0]}, {"a": [0, 0, 0]}, partial=True, approx=True)
    assert takeaway(cur, pri) == "Merges/day unchanged at 0 vs prior 3d (approx.) (partial)"


def test_a_gauge_over_a_single_bucket_is_not_enough_data() -> None:
    cur, pri = pair({"a": [6]}, {"a": [4]}, kind="gauge", label="Lead time")
    assert takeaway(cur, pri, min_sample=1) == "Lead time: not enough data vs prior 3d"
    cur, pri = pair({"a": [6, 6]}, {"a": [4]}, kind="gauge", label="Lead time")
    assert takeaway(cur, pri, min_sample=1) == "Lead time: not enough data vs prior 3d"


def test_zero_baseline_count_states_the_absolute_change_without_a_sample_floor() -> None:
    cur, pri = pair({"a": [3, 0, 0]}, {"a": [0, 0, 0]})  # n = 3 < MIN_SAMPLE, prior n = 0
    assert takeaway(cur, pri) == "Merges/day up from 0 to 1 vs prior 3d"
    # a gauge's zero baseline still needs its samples (an idle reading is not a count of events)
    cur, pri = pair({"a": [3, 3]}, {"a": [0, 0]}, kind="gauge", label="Wip")
    assert takeaway(cur, pri, min_sample=5) == "Wip: not enough data vs prior 3d"
    assert takeaway(replace(cur, n=9), replace(pri, n=9)) == "Wip up from 0 to 3 vs prior 3d"


def test_not_comparable_names_the_series_and_start_date() -> None:
    cur, pri = pair({"a": [5, 5, 5]}, {"a": [5, 5, 5]}, name="salvage_share")
    late = replace(cur, coverage_start="2026-10-02T05:00:00Z")  # last contributing kind began
    assert takeaway(late, pri) == "not comparable: salvage_share starts 2026-10-02"
