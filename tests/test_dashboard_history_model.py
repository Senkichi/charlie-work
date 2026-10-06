"""History's per-metric facts (``history_model``) and what the page does with them: the
wrong-way flag (threshold x polarity), card order, the tab dot, chart type per kind, and
escaping in the drawers."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from html import unescape

import pytest

from charlie_work.dashboard import metrics
from charlie_work.dashboard.history_data import HistoryView, MetricData, range_query
from charlie_work.dashboard.history_model import (
    WRONG_WAY,
    card_of,
    change_of,
    delta_text,
    fmt_value,
    tab_cards,
    wrong_way,
)
from charlie_work.dashboard.metrics import DOWN_GOOD, NEUTRAL, PRESENTATION, UP_GOOD
from charlie_work.dashboard.metrics_base import Series
from charlie_work.dashboard.pages.history import render_region
from charlie_work.dashboard.pages.history_detail import render_head, render_lower

END = datetime(2026, 10, 1, tzinfo=UTC)
Q = range_query("7d", END, UTC)
PTS = (("2026-09-25T00:00:00Z", 2.0), ("2026-09-26T00:00:00Z", 4.0))


def _series(**kw) -> Series:
    base = dict(name="m", unit="", points=PTS, per_repo={}, coverage_start=None,
                coverage_end=None, approx=False, not_instrumented=False, n=2,
                window_start=Q.start_iso, window_end=Q.end_iso,
                bucket_seconds=int(Q.bucket.total_seconds()))  # fmt: skip
    return Series(**{**base, **kw})


def _metric(mid: str, compared=None, **kw) -> MetricData:
    s = _series(name=mid, **kw)
    return MetricData(mid, (s,), {mid: f"{mid} takeaway"}, {mid: compared})


def test_every_history_metric_has_a_presentation() -> None:
    ids = {mid for tab in metrics.TABS.values() for mid in tab}
    assert set(PRESENTATION) == ids, sorted(set(PRESENTATION) ^ ids)
    assert all(p.polarity in (UP_GOOD, DOWN_GOOD, NEUTRAL) for p in PRESENTATION.values())
    assert all(p.name and "_" not in p.name for p in PRESENTATION.values())


def test_change_is_the_takeaways_own_comparison() -> None:
    assert change_of(None) is None  # the takeaway claims no comparison: neither does the card
    assert change_of((12.0, 10.0)) == pytest.approx(0.2)
    assert change_of((10.0, 10.2)) == 0.0  # inside takeaways.FLAT_BELOW: flat
    assert change_of((0.0, 0.0)) == 0.0
    assert math.isinf(change_of((3.0, 0.0)))  # up from a real zero
    assert [delta_text(c) for c in (None, 0.0, math.inf, 0.25, -0.5, 6.86)] == [
        "no comparable prior",
        "flat vs prior",
        "up from 0 vs prior",
        "up 25% vs prior",
        "down 50% vs prior",
        "up 8x vs prior",
    ]


@pytest.mark.parametrize(
    ("polarity", "change", "bad"),
    [
        (DOWN_GOOD, WRONG_WAY, True),  # exactly the threshold, the wrong way: flagged
        (DOWN_GOOD, WRONG_WAY - 0.001, False),  # just under it: not
        (DOWN_GOOD, -0.9, False),  # a big move the RIGHT way is never flagged
        (UP_GOOD, -WRONG_WAY, True),
        (UP_GOOD, -0.19, False),
        (UP_GOOD, 3.0, False),
        (NEUTRAL, 5.0, False),  # no good direction: never flagged, however big
        (NEUTRAL, -5.0, False),
        (DOWN_GOOD, math.inf, True),  # up from 0 for a down-is-good metric
        (DOWN_GOOD, None, False),  # no comparison: no claim
        (DOWN_GOOD, 0.0, False),
    ],
)
def test_wrong_way_needs_the_threshold_and_a_known_bad_direction(polarity, change, bad) -> None:
    assert wrong_way(polarity, change) is bad


def test_wrong_way_movers_sort_first_and_registry_order_is_kept_otherwise() -> None:
    view = HistoryView(
        "reliability",
        "7d",
        Q,
        (
            _metric("loop_pass_errors", (5.0, 5.0)),  # flat
            _metric("self_deploys", (50.0, 1.0)),  # neutral: never first
            _metric("launch_failures", (10.0, 5.0)),  # down-good, +100%: flagged
            _metric("throttles", (6.0, 5.0)),  # +20%: flagged, at the threshold
            _metric("loop_pass_duration", (1.0, 5.0)),  # better: not flagged
        ),
        END,
    )
    order = [(c.metric_id, c.bad) for c, _ in tab_cards(view)]
    assert order == [
        ("launch_failures", True),
        ("throttles", True),
        ("loop_pass_errors", False),
        ("self_deploys", False),
        ("loop_pass_duration", False),
    ]
    html = render_region({"reliability": view}, "7d", "reliability")
    cards = re.findall(
        r'<button type="button" class="card" id="card-[^"]+" data-k="([^"]+)"', html
    )
    assert cards == [mid for mid, _ in order]  # the page keeps the model's order
    assert (
        html.count("data-bad=") >= 2 and "worth your attention" in html
    )  # said, not only coloured


def test_the_tab_dot_marks_exactly_the_tabs_holding_a_wrong_way_mover() -> None:
    calm = HistoryView("flow", "7d", Q, (_metric("merges_per_day", (12.0, 10.0)),), END)
    worse = HistoryView("quality", "7d", Q, (_metric("escalations", (30.0, 10.0)),), END)
    html = render_region({"flow": calm, "quality": worse}, "7d", "flow")
    flagged = re.findall(r'id="tab-(\w+)" data-tab="\1" data-flag="1"', html)
    assert flagged == ["quality"]
    quality = html[
        html.index('id="tab-quality"') : html.index("</button>", html.index('id="tab-quality"'))
    ]
    assert 'class="flag"' in quality and "moved the wrong way" in quality  # words for readers


@pytest.mark.parametrize(
    ("kind", "chart"),
    [("count", "bars"), ("gauge", "line"), ("duration", "line"), ("ratio", "line")],
)
def test_counts_are_bars_and_levels_are_lines(kind: str, chart: str) -> None:
    metric = _metric("m", (1.0, 1.0), kind=kind)
    assert card_of(metric).chart == chart
    html = render_region({"flow": HistoryView("flow", "7d", Q, (metric,), END)}, "7d", "flow")
    payload = json.loads(unescape(re.search(r'data-hist="([^"]*)"', html).group(1)))
    assert payload["metrics"]["m"]["chart"] == chart


def test_summary_follows_the_registry() -> None:
    gauge = card_of(_metric("queue_depth", None, kind="gauge"))
    assert (gauge.summary, gauge.value) == ("last", 4.0)  # a level's window value is its latest
    count = card_of(_metric("merges_per_day", None))
    assert (count.summary, count.value) == ("sum", 6.0)
    mean = card_of(_metric("wip", None, kind="gauge"))
    assert (mean.summary, mean.value) == ("mean", 3.0)
    dur = card_of(_metric("lead_time", None, kind="duration", unit="hours",
                          samples={"o/a": (1.0, 2.0, 30.0)}))  # fmt: skip
    assert (dur.summary, dur.value) == ("median", 2.0)  # the raw samples, not bucket values
    assert [
        fmt_value(v, u)
        for v, u in (
            (0.54, "ratio"),
            (0.5, "hours"),
            (30.0, "hours"),
            (1340.0, ""),
            (7200.0, "seconds"),
        )
    ] == [  # fmt: skip
        "54%",
        "30m",
        "30h",
        "1,340",
        "2.0h",
    ]


def test_drawers_escape_hostile_repo_and_reason_names() -> None:
    evil = '<img src=x onerror="alert(1)">'
    head = _series(name="escalations", per_repo={f"o/{evil}": PTS})
    child = _series(name=f"escalations.{evil}", per_repo={})
    metric = MetricData("escalations", (head, child), {"escalations": f"take {evil}"},
                        {"escalations": (2.0, 1.0)})  # fmt: skip
    card = card_of(metric)
    lower = render_lower(card, metric, frozenset({f"o/{evil}"}), UTC)
    page = render_region(
        {"quality": HistoryView("quality", "7d", Q, (metric,), END)}, "7d", "quality"
    )
    for html in (lower, render_head(card, "7d"), page):
        assert "<img" not in html and 'onerror="' not in html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in lower
    # the bucket drawers are built by history.js from the payload with text nodes; the
    # payload itself is an attribute value, so it must be attribute-escaped too
    attr = re.search(r'data-hist="([^"]*)"', page).group(1)
    assert "<" not in attr and json.loads(unescape(attr))["metrics"]["escalations"]["repo"]
