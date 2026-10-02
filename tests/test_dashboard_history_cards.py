"""History card wiring (``pages/history_cards.py``) over synthetic series."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from charlie_work.dashboard.metrics_base import Series
from charlie_work.dashboard.pages.history_cards import (
    _chart_coverage,
    _distribution,
    _panels,
    _references,
)

START = datetime(2026, 9, 1, tzinfo=UTC)
END = datetime(2026, 10, 1, tzinfo=UTC)


def _series(**kw) -> Series:
    base = dict(name="m", unit="", points=(), per_repo={}, coverage_start=None,
                coverage_end=None, approx=False, not_instrumented=False)  # fmt: skip
    return Series(**{**base, **kw})


def test_chart_coverage_keeps_sources_that_already_cover_the_window_start() -> None:
    s = _series(
        repo_coverage={
            "o/early": ("2026-07-01T00:00:00Z", "2026-10-01T00:00:00Z"),
            "o/late": ("2026-09-28T00:00:00Z", "2026-10-01T00:00:00Z"),
            "o/after": ("2026-10-02T00:00:00Z", "2026-10-03T00:00:00Z"),
        }
    )
    # the chart needs "early" to know the window start IS covered; "after" cannot matter
    assert [c.source for c in _chart_coverage(s, (START, END))] == ["early", "late"]


def test_panel_links_only_to_repos_the_registry_holds() -> None:
    pts = (("2026-09-02T00:00:00Z", 1.0),)
    s = _series(per_repo={"o/fleet-repo": pts, "o/runner-only": pts})
    got = {p.key: p.href for p in _panels((s,), False, frozenset({"o/fleet-repo"}))}
    assert got == {"fleet-repo": "/repo/o/fleet-repo", "runner-only": None}


def test_history_buckets_sit_on_the_local_calendar_grid() -> None:
    from datetime import timedelta, timezone

    from charlie_work.dashboard.history_data import bucket_end, range_query

    tz = timezone(timedelta(hours=-7))
    now = datetime(2026, 10, 1, 23, 21, tzinfo=UTC)  # 16:21 local
    q = range_query("30d", now, tz)
    assert q.end.astimezone(tz) == datetime(2026, 10, 1, tzinfo=tz)  # last whole local day
    assert q.start.astimezone(tz).time().isoformat() == "00:00:00"  # each bar = one day
    six = bucket_end(now, timedelta(hours=6), tz).astimezone(tz)
    assert (six.hour, six.minute) == (12, 0)
    # boundaries do not drift between reloads inside one bucket
    assert range_query("7d", now + timedelta(minutes=50), tz) == range_query("7d", now, tz)
    three = bucket_end(now, timedelta(days=3), tz).astimezone(tz)
    assert three.time().isoformat() == "00:00:00"
    assert (three.date() - datetime(2026, 1, 1).date()).days % 3 == 0


def test_a_combined_chart_never_repeats_a_line_style() -> None:
    import re

    from charlie_work.dashboard.history_data import MetricData, range_query
    from charlie_work.dashboard.pages.history_cards import render_card

    q = range_query("7d", END, UTC)
    pts = tuple((f"2026-09-{d:02d}T{h:02d}:00:00Z", float(d)) for d in range(24, 30)
                for h in (0, 6, 12, 18))  # fmt: skip
    kw = dict(points=pts, kind="count", n=9, window_start=q.start_iso, window_end=q.end_iso,
              bucket_seconds=int(q.bucket.total_seconds()))  # fmt: skip
    head = _series(name="esc", **kw)
    kids = tuple(_series(name=f"esc.k{i}", **{**kw, "n": 9 - i}) for i in range(5))
    view_metric = MetricData("escalations", (head, *kids), {"esc": "t"})
    from charlie_work.dashboard.history_data import HistoryView

    html = render_card(HistoryView("quality", "7d", q, (view_metric,), END), view_metric, UTC)
    combined = html[: html.index("</figure>")]
    styles = re.findall(r'<g class="(series s\d)[^"]*"><path class="line" d="[^"]+" fill="none"'
                        r'( stroke-dasharray="[^"]+")?', combined)  # fmt: skip
    assert len(styles) >= 3 and len(set(styles)) == len(styles), styles


def test_a_median_line_names_its_statistic_in_the_title() -> None:
    from charlie_work.dashboard.pages.history_cards import title_of

    s = _series(label="Lead time", unit="hours", kind="duration", bucket_seconds=86400)
    assert title_of(s) == "Lead time (hours)"
    assert title_of(_series(**{**s.__dict__, "stat": "median"})) == (
        "Lead time, median per 1d (hours)"
    )


def test_the_headline_comparison_is_drawn_on_the_chart() -> None:
    import re

    from charlie_work.dashboard.history_data import HistoryView, MetricData, range_query
    from charlie_work.dashboard.pages.history_cards import render_card

    q = range_query("7d", END, UTC)
    pts = tuple((f"2026-09-{d:02d}T{h:02d}:00:00Z", 1.0) for d in range(24, 30)
                for h in (0, 6, 12, 18))  # fmt: skip
    head = _series(name="m", points=pts, n=24, window_start=q.start_iso, window_end=q.end_iso,
                   bucket_seconds=int(q.bucket.total_seconds()))  # fmt: skip
    view = HistoryView("flow", "7d", q, (), END)
    drawn = render_card(view, MetricData("m", (head,), {"m": "M ↑300%"}, {"m": (1.0, 0.25)}), UTC)
    rules = re.findall(r'<line class="(ref(?: prior)?)"[^>]*?(stroke-dasharray="[^"]+")?/>', drawn)
    assert sorted(r[0] for r in rules) == ["ref", "ref prior"]
    assert ">prior 7d avg 0.25</text>" in drawn and ">this 7d avg 1</text>" in drawn
    assert "prior 7d avg 0.25" in re.search(r'aria-label="([^"]+)"', drawn).group(1)
    bare = render_card(view, MetricData("m", (head,), {"m": "not comparable"}, {"m": None}), UTC)
    assert 'class="ref' not in bare


def test_stage_time_titles_use_display_names() -> None:
    from charlie_work.dashboard.metrics_flow import STAGE_NAMES, STAGES

    assert set(STAGES) <= set(STAGE_NAMES)
    assert all("_" not in name for name in STAGE_NAMES.values())


def test_p90_reference_uses_the_raw_samples_in_chart_units() -> None:
    head = _series(
        kind="duration", unit="hours", samples={"o/a": tuple(float(v) for v in range(1, 11))}
    )
    refs = _references(None, "7d", head)
    # nearest-rank p90 over all repos' samples (9th of 1..10), in the series' own unit
    # (hours here — the combined chart's axis — not the strip's seconds)
    assert [(r.label, r.value, r.prior) for r in refs] == [("p90", 9.0, False)]
    refs = _references((4.0, 1.0), "7d", head)
    assert [r.label for r in refs] == ["prior 7d avg", "this 7d avg", "p90"]
    assert [r.prior for r in refs] == [True, False, False]
    # only duration kinds get the p90 rule; an empty sample map adds nothing
    assert _references(None, "7d", _series(kind="count", samples={"o/a": (1.0,)})) == ()
    assert _references(None, "7d", _series(kind="duration")) == ()


def test_distribution_scales_samples_to_seconds_and_pools_repos() -> None:
    head = _series(
        label="Lead time",
        kind="duration",
        unit="hours",
        approx=True,
        samples={"o/alpha": (1.0, 2.0), "o/beta": (3.0,)},
    )
    html = _distribution(head, (START, END), UTC)
    assert 'class="strip-chart"' in html
    # hours -> seconds on the strip's log axis: alpha's 1h/2h read as 1h30m/2h00m
    assert "median 1h30m · p90 2h00m · n 2 approx." in html
    # several sources pool an 'all' row carrying every sample (p90 of 1h,2h,3h = 3h)
    assert ">all</text>" in html and "median 2h00m · p90 3h00m · n 3 approx." in html
    assert '<g class="dist approx">' in html


def test_distribution_needs_samples_and_a_known_unit() -> None:
    assert _distribution(_series(kind="duration", unit="hours"), (START, END), UTC) == ""
    parsecs = _series(kind="duration", unit="parsecs", samples={"o/a": (1.0,)})
    assert _distribution(parsecs, (START, END), UTC) == ""
    # one repo only: no pooled 'all' row (it would repeat the same samples)
    single = _series(kind="duration", unit="seconds", samples={"o/a": (60.0, 120.0)})
    html = _distribution(single, (START, END), UTC)
    assert ">all</text>" not in html and "median 1m30s · p90 2m00s · n 2" in html


def test_a_render_fault_gets_its_own_message_and_is_logged(caplog) -> None:
    from charlie_work.dashboard.history_data import HistoryView, MetricData, range_query
    from charlie_work.dashboard.pages.history_cards import render_cards

    q = range_query("7d", END, UTC)
    # an unparseable point timestamp crashes render_card — a render fault, not a
    # malformed stored row, so the card must not claim one
    head = _series(name="m", points=(("not-a-timestamp", 1.0),), n=1,
                   window_start=q.start_iso, window_end=q.end_iso,
                   bucket_seconds=int(q.bucket.total_seconds()))  # fmt: skip
    view = HistoryView("flow", "7d", q, (MetricData("m", (head,), {"m": "t"}),), END)
    with caplog.at_level(logging.ERROR, "charlie_work.dashboard"):
        html = render_cards(view, UTC)
    assert "the card failed to render" in html and "a stored row" not in html
    assert any(r.name == "charlie_work.dashboard" and r.exc_info for r in caplog.records)
