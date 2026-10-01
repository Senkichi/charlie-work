"""History card wiring (``pages/history_cards.py``) over synthetic series."""

from __future__ import annotations

from datetime import UTC, datetime

from charlie_work.dashboard.metrics_base import Series
from charlie_work.dashboard.pages.history_cards import _chart_coverage, _panels

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
