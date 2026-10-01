"""Line charts and small multiples: structural assertions on tiny literal inputs."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone

from charlie_work.dashboard.charts import (
    Coverage,
    LineSpec,
    Marker,
    Panel,
    Point,
    Series,
    line_chart,
    nice_ticks,
    small_multiples,
    time_ticks,
)
from charlie_work.dashboard.charts.line import segments

T0 = datetime(2026, 9, 28, tzinfo=UTC)
DAY = 86400.0


def day(n: int) -> datetime:
    return T0 + timedelta(days=n)


def pts(*vals: float | None) -> tuple[Point, ...]:
    return tuple(Point(day(i), v) for i, v in enumerate(vals))


def paths(svg: str) -> list[str]:
    return re.findall(r'<path class="line" d="([^"]+)"', svg)


# --- scales ---------------------------------------------------------------------------


def test_nice_ticks_are_1_2_5_steps_enclosing_the_range() -> None:
    assert nice_ticks(0, 7) == (0, 2, 4, 6, 8)
    assert nice_ticks(0, 97) == (0, 20, 40, 60, 80, 100)
    assert nice_ticks(0, 0.34) == (0, 0.1, 0.2, 0.3, 0.4)
    assert nice_ticks(0, 3, integer=True) == (0, 1, 2, 3)
    assert nice_ticks(5, 5) == (5, 5.2, 5.4, 5.6, 5.8, 6)


def test_time_ticks_label_local_time_and_dates_at_local_midnight() -> None:
    plus2 = timezone(timedelta(hours=2))
    # 22:00Z..10:00Z next day = 00:00..12:00 at +02:00; 12h / 6 ticks -> 2h steps.
    ticks = time_ticks(
        datetime(2026, 9, 30, 22, tzinfo=UTC), datetime(2026, 10, 1, 10, tzinfo=UTC), plus2
    )
    assert [t.label for t in ticks] == [
        "Oct 1",
        "02:00",
        "04:00",
        "06:00",
        "08:00",
        "10:00",
        "12:00",
    ]
    assert ticks[0].at == datetime(2026, 9, 30, 22, tzinfo=UTC)


def test_time_ticks_day_steps_label_dates() -> None:
    ticks = time_ticks(day(0), day(4), UTC)
    assert [t.label for t in ticks] == ["Sep 28", "Sep 29", "Sep 30", "Oct 1", "Oct 2"]


# --- gaps -----------------------------------------------------------------------------


def test_none_bucket_splits_the_line_into_two_paths() -> None:
    html = line_chart((Series("merges", pts(1, 2, None, 4, 5)),), LineSpec("m"), UTC)
    assert len(paths(html)) == 2
    assert all(p.count("M") == 1 and p.count("L") == 1 for p in paths(html))


def test_absent_bucket_splits_by_distance_and_lone_point_is_a_dot() -> None:
    points = (Point(day(0), 1), Point(day(1), 2), Point(day(3), 3), Point(day(5), 1))
    runs = segments(points, DAY)
    assert [[p.value for p in r] for r in runs] == [[1, 2], [3], [1]]
    html = line_chart((Series("x", points),), LineSpec("t", bucket_seconds=DAY), UTC)
    assert len(paths(html)) == 1
    assert html.count('<circle class="dot"') == 2


def test_without_bucket_distance_does_not_split() -> None:
    points = (Point(day(0), 1), Point(day(3), 3))
    assert len(segments(points, None)) == 1


# --- styling channels, labels, escaping -----------------------------------------------


def test_approx_series_is_dashed_and_labelled_approx() -> None:
    html = line_chart(
        (Series("exact", pts(1, 2)), Series("lead", pts(3, 4), approx=True)),
        LineSpec("lead time"),
        UTC,
    )
    assert '<g class="series s2 approx"><path class="line"' in html
    assert re.search(r'class="series s2 approx"><path [^>]*stroke-dasharray="4 4"', html)
    assert re.search(r'class="series s1"><path [^>]*/>', html)
    assert "stroke-dasharray" not in re.search(r'class="series s1">(.*?)</g>', html).group(1)
    assert ">lead approx.</text>" in html
    assert "lead (approx.)" in html  # aria summary


def test_direct_labels_replace_a_legend_and_are_dodged() -> None:
    html = line_chart((Series("a", pts(5, 5)), Series("b", pts(5, 5))), LineSpec("t"), UTC)
    assert "legend" not in html
    ys = [float(y) for y in re.findall(r'<text class="direct[^"]*" x="[^"]+" y="([^"]+)"', html)]
    assert len(ys) == 2 and abs(ys[0] - ys[1]) >= 14


def test_series_names_and_titles_are_escaped_everywhere() -> None:
    html = line_chart((Series('<b>"x"&', pts(1, 2)),), LineSpec("<t>", takeaway="<k>"), UTC)
    assert "<b>" not in html and "<t>" not in html and "<k>" not in html
    assert "&lt;b&gt;&quot;x&quot;&amp;" in html
    assert 'aria-label="&lt;t&gt;' in html


def test_svg_is_role_img_without_links_and_intrinsic_size() -> None:
    html = line_chart((Series("a", pts(1, 2)),), LineSpec("t"), UTC)
    svg = re.search(r"<svg.*?</svg>", html, re.S).group(0)
    assert 'role="img"' in svg and "<a " not in svg
    assert 'width="640" height="220" viewBox="0 0 640 220"' in svg


def test_caption_carries_local_window_with_iso_datetime() -> None:
    html = line_chart((Series("a", pts(1, 2)),), LineSpec("t"), timezone(timedelta(hours=-4)))
    assert '<time datetime="2026-09-28T00:00:00Z">2026-09-27 20:00</time>' in html


# --- coverage and markers -------------------------------------------------------------


def test_coverage_shades_before_source_start_and_marker_is_drawn() -> None:
    spec = LineSpec(
        "t",
        coverage=(Coverage("events.db", day(2)), Coverage("old", day(-5))),
        markers=(Marker(day(3), "exact series starts Oct 1"),),
    )
    html = line_chart((Series("a", pts(1, 2, 3, 4)),), spec, UTC)
    assert html.count('<rect class="uncovered"') == 1  # the source covering it all draws none
    assert ">events.db from 2026-09-30</text>" in html
    assert "sources: old from 2026-09-23 00:00, events.db from 2026-09-30 00:00" in html
    assert re.search(r'class="marker-rule"[^>]*stroke-dasharray="2 2"', html)
    assert ">exact series starts Oct 1</text>" in html


def test_empty_series_renders_an_empty_state_not_an_error() -> None:
    html = line_chart((), LineSpec("t"), UTC)
    assert "no data in this window" in html and "<svg" not in html


# --- small multiples ------------------------------------------------------------------


def _panels() -> tuple[Panel, ...]:
    return (
        Panel("o/small", (Series("merges", pts(1, 2, 1)),)),
        Panel("o/big", (Series("merges", pts(3, 7, 2)),), href="/repo/o/big"),
    )


def test_multiples_share_the_y_scale_equal_to_max_of_all_panels() -> None:
    html = small_multiples(_panels(), LineSpec("merges/day"), UTC)
    # Peak over ALL panels is 7 (o/big); 4 nice ticks enclosing 0..7 are 0, 5, 10. The
    # small panel (peak 2) gets the same top, not its own 0..2 scale.
    assert nice_ticks(0, 7, 4)[-1] == 10
    assert re.findall(r'data-y-max="([^"]+)"', html) == ["10", "10"]
    tops = re.findall(
        r'<text class="tick" x="[^"]+" y="([^"]+)" text-anchor="end">10</text>', html
    )
    assert len(tops) == 2 and tops[0] == tops[1]


def test_multiples_sorted_by_total_and_share_x_window() -> None:
    html = small_multiples(_panels(), LineSpec("merges/day"), UTC)
    assert html.index("o/big") < html.index("o/small")
    firsts = [p.split(" ")[0].split(",")[0] for p in paths(html)]
    assert firsts[0] == firsts[1]  # same x for the same first bucket


def test_multiples_heading_links_only_when_routed_and_outside_svg() -> None:
    html = small_multiples(_panels(), LineSpec("m"), UTC)
    # /repo is not a registered route yet, so the heading is plain text.
    assert '<span class="panel-key">o/big</span>' in html
    for svg in re.findall(r"<svg.*?</svg>", html, re.S):
        assert "<a " not in svg
