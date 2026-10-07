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

from _dashboard_rollup_fixtures import ALPHA, BETA

from charlie_work.dashboard import metrics_flow as flow
from charlie_work.dashboard import metrics_quality as quality
from charlie_work.dashboard.metrics_base import (
    MetricQuery,
    Series,
    SeriesSpec,
    make_ratio_series,
    make_series,
)
from charlie_work.dashboard.takeaways import takeaway

DAY = 86400
KINDS = ("supervisor_started",)  # the fixture heartbeat: both repos covered in both windows


def obs(repo: str, d: int, n: int, v: float = 1.0) -> list[tuple[str, str, float]]:
    """``n`` one-per-hour samples of ``v`` on 2026-10-<d>: one bucket, n observations."""
    return [(day(d, h), repo, v) for h in range(n)]


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


def test_a_thin_outlier_bucket_cannot_swing_a_duration_headline() -> None:
    """#2474: one n=4 storm bucket (~23h median) beside five normal n=6 buckets
    (~1h). The mean of bucket medians reads ↑367%; the pooled sample median is flat."""
    cur = series(
        "2026-10-04T00:00:00Z", "2026-10-07T00:00:00Z", {"a": [23.0, 1, 1, 1, 1, 1]},
        kind="duration", label="Lead time", bucket_seconds=DAY // 2, n=34,
        samples={"a": (20.0, 22.0, 24.0, 26.0) + (1.0,) * 30},
    )  # fmt: skip
    pri = series(
        "2026-10-01T00:00:00Z", "2026-10-04T00:00:00Z", {"a": [1, 1, 1, 1, 1, 1]},
        kind="duration", label="Lead time", bucket_seconds=DAY // 2, n=30,
        samples={"a": (1.0,) * 30},
    )  # fmt: skip
    assert takeaway(cur, pri) == "Lead time flat vs prior 3d (+0%)"


def test_driver_uses_the_same_pooled_statistic() -> None:
    """#2474: repo b's apparent surge is one n=4 bucket; pooled, only repo a moved."""
    cur = series(
        "2026-10-04T00:00:00Z", "2026-10-07T00:00:00Z", {"a": [3, 3, 3], "b": [30, 1, 1]},
        kind="duration", label="Lead time", n=34,
        samples={"a": (3.0,) * 20, "b": (1.0,) * 10 + (28.0, 30.0, 31.0, 32.0)},
    )  # fmt: skip
    pri = series(
        "2026-10-01T00:00:00Z", "2026-10-04T00:00:00Z", {"a": [1, 1, 1], "b": [1, 1, 1]},
        kind="duration", label="Lead time", n=30,
        samples={"a": (1.0,) * 15, "b": (1.0,) * 15},
    )  # fmt: skip
    # b's pooled median is still 1.0; under a mean of bucket medians b (10.67) would
    # outweigh a (3.0) and steal the driver attribution.
    assert takeaway(cur, pri) == "Lead time ↑200% vs prior 3d, driven by a"


def test_thin_buckets_never_enter_a_mean_of_buckets() -> None:
    """A bucket under MIN_BUCKET_N is excluded; too few qualifiers -> no trend."""
    counts = {"2026-10-04T00:00:00Z": 4, "2026-10-05T00:00:00Z": 8, "2026-10-06T00:00:00Z": 8}
    cur = series(
        "2026-10-04T00:00:00Z", "2026-10-07T00:00:00Z", {"a": [1.0, 0.5, 0.25]},
        kind="ratio", label="Rework rate", n=48,
        bucket_n=dict(counts), repo_bucket_n={"a": dict(counts)},
    )  # fmt: skip
    pri = series(
        "2026-10-01T00:00:00Z", "2026-10-04T00:00:00Z", {"a": [0.25, 0.25, 0.25]},
        kind="ratio", label="Rework rate", n=24,
    )  # fmt: skip
    pri = replace(
        pri,
        bucket_n={ts: 8 for ts, _ in pri.points},
        repo_bucket_n={"a": {ts: 8 for ts, _ in pri.points}},
    )
    # bucket 1 (n=4) is out: the mean is (0.5*8 + 0.25*8)/16 = 0.375, not 0.58
    assert takeaway(cur, pri) == "Rework rate ↑50% vs prior 3d"
    thin = replace(
        cur,
        bucket_n={ts: 4 for ts, _ in cur.points},
        repo_bucket_n={"a": {ts: 4 for ts, _ in cur.points}},
    )
    assert takeaway(thin, pri) == "Rework rate: not enough data vs prior 3d"
    assert takeaway(thin, pri, min_bucket_n=4) == "Rework rate ↑133% vs prior 3d"


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


def test_assess_returns_the_compared_values_in_chart_units() -> None:
    from charlie_work.dashboard.takeaways import assess

    cur, pri = pair({"a": [4, 4, 4], "b": [1, 1, 1]}, {"a": [1, 1, 1], "b": [1, 1, 1]})
    text, compared = assess(cur, pri)
    assert text == takeaway(cur, pri) and compared == (5.0, 2.0)  # per 1d bucket
    hourly = replace(cur, bucket_seconds=DAY // 4), replace(pri, bucket_seconds=DAY // 4)
    assert assess(*hourly)[1] == (1.25, 0.5)  # a 6h bucket holds a quarter of a day
    late = replace(cur, coverage_start="2026-10-03T00:00:00Z")
    assert assess(late, pri)[1] is None  # "not comparable": nothing to draw


def test_producers_count_the_observations_behind_each_point(ro_db) -> None:
    """``bucket_n``/``repo_bucket_n`` key on exactly the point timestamps; the value is
    the observations the point rests on (a ratio's: the denominator mass)."""
    dur = SeriesSpec("lead_time", "Lead time", "hours", "duration", KINDS)
    samples = (
        obs(ALPHA, 1, 2, 2.0) + obs(ALPHA, 2, 6, 4.0) + obs(BETA, 2, 3, 8.0)
        + obs(BETA, 4, 5, 1.0)
    )  # fmt: skip
    s = make_series(ro_db, CURRENT, dur, samples, samples, how="median")
    assert s.points == ((day(1), 2.0), (day(2), 4.0), (day(4), 1.0))  # 9-sample bucket median: 4
    assert s.n == 16
    assert [ts for ts, _ in s.points] == sorted(s.bucket_n)
    assert s.bucket_n == {day(1): 2, day(2): 9, day(4): 5}
    assert s.repo_bucket_n[ALPHA] == {day(1): 2, day(2): 6}
    assert s.repo_bucket_n[BETA] == {day(2): 3, day(4): 5}

    gauge = SeriesSpec("wip", "Work in progress", "workers", "gauge", KINDS)
    s = make_series(ro_db, CURRENT, gauge, samples, samples, how="mean")
    assert s.points == ((day(1), 2.0), (day(2), 48 / 9), (day(4), 1.0))
    assert [ts for ts, _ in s.points] == sorted(s.bucket_n)
    assert s.bucket_n == {day(1): 2, day(2): 9, day(4): 5}
    assert s.repo_bucket_n[ALPHA] == {day(1): 2, day(2): 6}

    depth = SeriesSpec("queue_depth", "Queue depth", "issues", "gauge", KINDS, combine="repo_sum")
    s = make_series(ro_db, CURRENT, depth, samples, samples, how="mean")
    assert s.points == ((day(1), 2.0), (day(2), 12.0), (day(4), 1.0))  # sum of repo means
    assert [ts for ts, _ in s.points] == sorted(s.bucket_n)
    assert s.bucket_n == {day(1): 2, day(2): 9, day(4): 5}  # observations, not repo count
    assert s.repo_bucket_n[BETA] == {day(2): 3, day(4): 5}

    rate = SeriesSpec("rework_rate", "Rework rate", "ratio", "ratio", KINDS)
    den = obs(ALPHA, 1, 5) + obs(ALPHA, 2, 4) + obs(BETA, 2, 6)
    num = obs(ALPHA, 1, 2) + obs(ALPHA, 2, 1) + obs(BETA, 2, 2)
    s = make_ratio_series(ro_db, CURRENT, rate, num, den)
    assert s.points == ((day(1), 2 / 5), (day(2), 3 / 10))
    assert s.n == 15  # denominator samples
    assert [ts for ts, _ in s.points] == sorted(s.bucket_n)
    assert s.bucket_n == {day(1): 5, day(2): 10}  # the denominator, never num's count
    assert s.repo_bucket_n[ALPHA] == {day(1): 5, day(2): 4}
    assert s.repo_bucket_n[BETA] == {day(2): 6}
    # a denominator sample can carry more than one observation: the mass is the weight
    s = make_ratio_series(ro_db, CURRENT, rate, num, obs(ALPHA, 1, 5, 2.0) + den[5:])
    assert s.points == ((day(1), 2 / 10), (day(2), 3 / 10))
    assert s.bucket_n == {day(1): 10, day(2): 10}


def test_producer_counts_still_headline_through_takeaway(ro_db) -> None:
    """End to end: a producer-built gauge/ratio with enough observations headlines.
    A ts-key slip in ``bucket_n`` would floor every bucket to "not enough data"."""
    gauge = SeriesSpec("wip", "Work in progress", "workers", "gauge", KINDS)
    cur_s = obs(ALPHA, 1, 8, 4.0) + obs(ALPHA, 2, 6, 10.0)
    pri_s = obs(ALPHA, -6, 6, 2.0) + obs(ALPHA, -5, 8, 4.0)
    cur = make_series(ro_db, CURRENT, gauge, cur_s, cur_s, how="mean")
    pri = make_series(ro_db, PRIOR, gauge, pri_s, pri_s, how="mean")
    # observation-weighted: (4*8 + 10*6)/14 vs (2*6 + 4*8)/14; a plain mean reads 7 vs 3
    assert takeaway(cur, pri) == "Work in progress ↑109% vs prior 7d"

    rate = SeriesSpec("rework_rate", "Rework rate", "ratio", "ratio", KINDS)
    cur = make_ratio_series(
        ro_db, CURRENT, rate, obs(ALPHA, 1, 4) + obs(ALPHA, 2, 1), obs(ALPHA, 1, 5) + obs(ALPHA, 2, 10)
    )  # fmt: skip
    pri = make_ratio_series(
        ro_db, PRIOR, rate, obs(ALPHA, -6, 2) + obs(ALPHA, -5, 2), obs(ALPHA, -6, 8) + obs(ALPHA, -5, 7)
    )  # fmt: skip
    # denominator-weighted: (4+1)/15 vs (2+2)/15
    assert takeaway(cur, pri) == "Rework rate ↑25% vs prior 7d"


def test_a_duration_with_samples_is_exempt_from_the_bucket_floor() -> None:
    """``_ready_buckets``: a duration comparing pooled samples counts every populated
    bucket — buckets that would floor a mean-of-buckets still headline."""
    cur = series(
        "2026-10-04T00:00:00Z", "2026-10-07T00:00:00Z", {"a": [23.0, 1, 1]},
        kind="duration", label="Lead time", n=34,
        samples={"a": (20.0, 22.0, 24.0, 26.0) + (1.0,) * 30},
    )  # fmt: skip
    pri = series(
        "2026-10-01T00:00:00Z", "2026-10-04T00:00:00Z", {"a": [1, 1, 1]},
        kind="duration", label="Lead time", n=30, samples={"a": (1.0,) * 30},
    )  # fmt: skip

    def thin(s: Series) -> dict[str, int]:
        return {ts: 4 for ts, _ in s.points}  # every bucket under MIN_BUCKET_N

    cur = replace(cur, bucket_n=thin(cur), repo_bucket_n={"a": thin(cur)})
    pri = replace(pri, bucket_n=thin(pri), repo_bucket_n={"a": thin(pri)})
    assert takeaway(cur, pri) == "Lead time flat vs prior 3d (+0%)"  # pooled median 1.0 vs 1.0
    # drop the raw samples and the same buckets are a floored mean: nothing qualifies
    bare_cur, bare_pri = replace(cur, samples={}), replace(pri, samples={})
    assert takeaway(bare_cur, bare_pri) == "Lead time: not enough data vs prior 3d"


def test_a_repo_with_no_qualifying_bucket_cannot_drive() -> None:
    """``_driver``: a repo whose buckets all sit under ``min_bucket_n`` in either
    window has no baseline to move from; it is skipped even when its raw swing is
    the largest."""
    cur = series(
        "2026-10-04T00:00:00Z", "2026-10-07T00:00:00Z",
        {"a": [4, 4, 4], "b": [30, 30, 30], "c": [2, 2, 2]},
        kind="gauge", label="Wip", n=99,
    )  # fmt: skip
    pri = series(
        "2026-10-01T00:00:00Z", "2026-10-04T00:00:00Z",
        {"a": [2, 2, 2], "b": [1, 1, 1], "c": [1, 1, 1]},
        kind="gauge", label="Wip", n=99,
    )  # fmt: skip

    def repo_counts(s: Series, n: int) -> dict[str, dict[str, int]]:
        return {r: {ts: n for ts, _ in pts} for r, pts in s.per_repo.items()}

    cur = replace(cur, bucket_n={ts: 8 for ts, _ in cur.points}, repo_bucket_n=repo_counts(cur, 8))
    pri = replace(pri, bucket_n={ts: 8 for ts, _ in pri.points}, repo_bucket_n=repo_counts(pri, 8))
    assert takeaway(cur, pri).endswith("driven by b")  # b's +29 carries 91% of the movement
    # b thin in either window -> skipped; the headline falls to a's +2 of 3
    thin_cur = replace(cur, repo_bucket_n={**cur.repo_bucket_n, "b": repo_counts(cur, 4)["b"]})
    assert takeaway(thin_cur, pri) == "Wip ↑800% vs prior 3d, driven by a"
    thin_pri = replace(pri, repo_bucket_n={**pri.repo_bucket_n, "b": repo_counts(pri, 4)["b"]})
    assert takeaway(cur, thin_pri) == "Wip ↑800% vs prior 3d, driven by a"
