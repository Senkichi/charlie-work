"""Line charts and small multiples: structural assertions on tiny literal inputs."""

from __future__ import annotations

import pytest

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
from charlie_work.dashboard.charts.model import Reference
from charlie_work.dashboard.charts.svg import APPROX_DASHES, CHAR_PX, fit

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
    assert re.search(
        rf'class="series s2 approx"><path [^>]*stroke-dasharray="{APPROX_DASHES[1]}"', html
    )
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
    # "old" covers the whole window, so nothing is uncovered: a late start is only a rule
    assert html.count('<rect class="uncovered"') == 0
    assert html.count('<line class="cov-start"') == 1
    assert ">events.db from 2026-09-30</text>" in html
    # coverage text lives once per card in its Coverage: note — never in the caption
    assert "sources:" not in html
    assert re.search(r'class="marker-rule"[^>]*stroke-dasharray="2 2"', html)
    assert ">exact series starts Oct 1</text>" in html


def test_uncovered_shade_stops_at_the_earliest_source_not_the_latest() -> None:
    spec = LineSpec("t", coverage=(Coverage("early", day(1)), Coverage("late", day(3))))
    html = line_chart((Series("a", pts(1, 2, 3, 4, 5)),), spec, UTC)
    rects = re.findall(r'<rect class="uncovered" x="([\d.]+)" y="[\d.]+" width="([\d.]+)"', html)
    rules = [float(x) for x in re.findall(r'<line class="cov-start" x1="([\d.]+)"', html)]
    assert len(rects) == 1 and len(rules) == 2
    x0, width = map(float, rects[0])
    assert x0 + width == pytest.approx(min(rules))  # ends at "early", not at "late"


def test_each_panel_is_shaded_by_its_own_source_only() -> None:
    spec = LineSpec("t", coverage=(Coverage("a", day(-9)), Coverage("b", day(3))))
    panels = (
        Panel("a", (Series("a", pts(5, 5, 5, 5, 5)),)),
        Panel("b", (Series("b", pts(0, 0, 0, 1, 1)),)),
    )
    html = small_multiples(panels, spec, UTC)
    figs = re.findall(r'<figure class="panel">.*?</figure>', html, re.S)
    by_key = {re.search(r"panel-key[^>]*>([ab])<", f).group(1): f for f in figs}
    assert 'class="uncovered"' not in by_key["a"]  # a covers its whole window
    assert by_key["b"].count('class="uncovered"') == 1


def test_multiples_caption_never_repeats_source_coverage() -> None:
    spec = LineSpec("t", coverage=(Coverage("a", day(-9)), Coverage("b", day(3))))
    html = small_multiples(_panels(), spec, UTC)
    # like line_chart: coverage is stated once per card in its Coverage: note, so the
    # grid caption — and no panel — may carry a "sources:" line
    assert "sources:" not in html


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


def test_multiples_heading_links_only_when_routed_and_outside_svg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from charlie_work.dashboard.pages import routes

    html = small_multiples(_panels(), LineSpec("m"), UTC)
    # /repo is a registered route, so the heading links to the repo drill-down ...
    assert '<a class="panel-key" href="/repo/o/big">o/big</a>' in html
    assert '<span class="panel-key">o/small</span>' in html  # no href: plain text
    for svg in re.findall(r"<svg.*?</svg>", html, re.S):
        assert "<a " not in svg
    # ... and an unregistered route renders the same heading as plain text.
    monkeypatch.setattr(routes, "ROUTES", frozenset({"/now"}))
    html = small_multiples(_panels(), LineSpec("m"), UTC)
    assert '<span class="panel-key">o/big</span>' in html


def test_all_approx_series_keep_distinct_dash_patterns() -> None:
    html = line_chart(
        tuple(Series(f"s{i}", pts(i, i + 1), approx=True) for i in range(3)),
        LineSpec("approx trio"),
        UTC,
    )
    dashes = re.findall(r'class="series s\d approx"><path [^>]*stroke-dasharray="([^"]+)"', html)
    assert len(dashes) == 3 and len(set(dashes)) == 3


def test_coverage_labels_follow_all_shading_and_collapse_past_three() -> None:
    covs = tuple(Coverage(f"repo{i}", day(1) + timedelta(hours=i)) for i in range(5))
    html = line_chart((Series("x", pts(1, 2, 3, 4)),), LineSpec("cov", coverage=covs), UTC)
    group = re.search(r'<g class="coverage">(.*?)</g>', html).group(1)
    assert group.rfind("<rect") < group.find("<text")  # no shading paints over a label
    notes = re.findall(r'<text class="note"[^>]*>([^<]*)</text>', group)
    assert len(notes) == 1 and notes[0].startswith("5 sources start")


def _note_boxes(html: str) -> list[tuple[str, float, float, float, float]]:
    """(class, x0, x1, y0, y1) of every in-plot annotation label, measured as the
    renderer does: ``CHAR_PX`` per character, ``_TEXT_H`` of ink above the baseline."""
    out = []
    for cls, x, y, anchor, body in re.findall(
        r'<text class="(note[^"]*)" x="(-?[\d.]+)" y="(-?[\d.]+)" text-anchor="(\w+)">'
        r"(?:<title>[^<]*</title>)?([^<]*)</text>",
        html,
    ):
        w = len(body) * CHAR_PX
        x0, x1 = (float(x) - w, float(x)) if anchor == "end" else (float(x), float(x) + w)
        out.append((cls, x0, x1, float(y) - 11.0, float(y) + 3.0))
    return out


def assert_no_overprint(html: str, width: float = 640.0) -> list:
    """No two annotation labels' estimated boxes intersect, and none leaves the plot."""
    boxes = _note_boxes(html)
    for i, (cls, x0, x1, y0, y1) in enumerate(boxes):
        assert x0 >= -0.5 and x1 <= width + 0.5, f"{cls} label left the plot: {x0:.0f}..{x1:.0f}"
        for ocls, ox0, ox1, oy0, oy1 in boxes[i + 1 :]:
            hit = x0 < ox1 and ox0 < x1 and y0 < oy1 and oy0 < y1
            assert not hit, f"{cls} overprints {ocls} (x {x0:.0f}..{x1:.0f} y {y0:.0f}..{y1:.0f})"
    return boxes


def test_annotation_labels_never_overprint_or_leave_the_plot() -> None:
    # coverage starts an hour apart, a marker between them, and two reference rules whose
    # labels sit exactly where lane 0 would be — every label must land somewhere legal.
    spec = LineSpec(
        "t",
        coverage=(Coverage("alpha", day(1)), Coverage("beta", day(1) + timedelta(hours=3))),
        markers=(Marker(day(1) + timedelta(hours=1), "exact series starts"),),
        references=(
            Reference(4.8, "prior 7d avg", prior=True),
            Reference(4.6, "this 7d avg"),
        ),
    )
    html = line_chart((Series("a", pts(1, 2, 3, 4, 5)),), spec, UTC)
    boxes = assert_no_overprint(html)
    assert any("ref-label" in cls for cls, *_ in boxes)
    # a marker's text has no card-level fallback, so it is never the label dropped
    assert "exact series starts" in html


def test_a_coverage_label_that_fits_nowhere_is_dropped_not_overprinted() -> None:
    # rules ten minutes apart (~1px): nothing fits between them, so labels drop and the
    # card's Coverage: note (not the chart) carries the names.
    covs = tuple(Coverage(f"source-{i}", day(1) + timedelta(minutes=10 * i)) for i in range(3))
    html = line_chart((Series("a", pts(1, 2, 3, 4)),), LineSpec("t", coverage=covs), UTC)
    assert_no_overprint(html)
    assert html.count('<line class="cov-start"') == 3  # the rules still draw


def test_a_coverage_label_never_crosses_the_next_rule() -> None:
    covs = (Coverage("fresh-eyes", day(1)), Coverage("beta", day(1) + timedelta(hours=2)))
    html = line_chart((Series("a", pts(1, 2, 3, 4)),), LineSpec("t", coverage=covs), UTC)
    rules = sorted(float(x) for x in re.findall(r'class="cov-start" x1="([\d.]+)"', html))
    assert len(rules) == 2
    for cls, x0, x1, _, _ in _note_boxes(html):
        for rule in rules:
            # a label box may touch a rule's side, never straddle it
            assert not (x0 < rule - 0.5 < x1), (cls, rule)


def test_week_window_with_three_ticks_gets_more_than_one_tick() -> None:
    assert len(time_ticks(day(0), day(7), UTC, max_ticks=3)) >= 2


def test_fit_truncates_with_ellipsis_and_keeps_short_text() -> None:
    assert fit("short", 200) == "short"
    cut = fit("a very long series name indeed", 70)
    assert cut.endswith("…") and len(cut) < 15
