"""History card and detail wiring (``pages/history_cards.py``, ``pages/history_detail.py``)
over synthetic series."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from charlie_work.dashboard.history_data import HistoryView, MetricData, range_query
from charlie_work.dashboard.history_model import bucket_grid, card_of, duration_stats
from charlie_work.dashboard.metrics_base import Series
from charlie_work.dashboard.pages.history_detail import chart_entry, render_head, render_lower

START = datetime(2026, 9, 1, tzinfo=UTC)
END = datetime(2026, 10, 1, tzinfo=UTC)


def _series(**kw) -> Series:
    base = dict(name="m", unit="", points=(), per_repo={}, coverage_start=None,
                coverage_end=None, approx=False, not_instrumented=False)  # fmt: skip
    return Series(**{**base, **kw})


def _window(**kw) -> Series:
    q = range_query("7d", END, UTC)
    return _series(
        window_start=q.start_iso,
        window_end=q.end_iso,
        bucket_seconds=int(q.bucket.total_seconds()),
        **kw,
    )


def test_panel_links_only_to_repos_the_registry_holds() -> None:
    pts = (("2026-09-25T00:00:00Z", 1.0),)
    s = _window(n=2, points=pts, per_repo={"o/fleet-repo": pts, "o/runner-only": pts})
    metric = MetricData("m", (s,), {"m": "t"})
    html = render_lower(card_of(metric), metric, frozenset({"o/fleet-repo"}), UTC)
    assert '<a href="/repo/o/fleet-repo" title="o/fleet-repo">fleet-repo</a>' in html
    assert '<span title="o/runner-only">runner-only</span>' in html
    assert 'href="/repo/o/runner-only"' not in html  # never a link that would 404


def test_history_buckets_sit_on_the_local_calendar_grid() -> None:
    from datetime import timedelta, timezone

    from charlie_work.dashboard.history_data import bucket_end

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


def test_a_median_line_names_its_statistic_in_the_title() -> None:
    pts = (("2026-09-25T00:00:00Z", 2.0),)
    s = _window(name="lead_time", label="Lead time", unit="hours", kind="duration",
                stat="median", points=pts, n=1)  # fmt: skip
    card = card_of(MetricData("lead_time", (s,), {"lead_time": "t"}))
    assert (card.summary, card.summary_word) == ("median", "median")
    head = render_head(card, "7d")
    assert '<h2>Lead time: <span class="num">2.0h</span> <span class="dim">median</span>' in head
    # a count is a total, and the headline needs no word for it
    count = card_of(MetricData("m", (_window(points=pts, n=1),), {"m": "t"}))
    assert count.summary_word == "total" and '<span class="dim">' not in render_head(count, "7d")


def test_stage_time_titles_use_display_names() -> None:
    from charlie_work.dashboard.metrics_flow import STAGE_NAMES, STAGES

    assert set(STAGES) <= set(STAGE_NAMES)
    assert all("_" not in name for name in STAGE_NAMES.values())


def test_prs_companion_leads_the_card_and_causes_get_their_own_axis() -> None:
    """Issue #2476: a ``<name>.prs`` child re-expresses the count in distinct PRs
    ("3 PRs (6 attempts)"), and ``<name>.cause.*`` children are a separate axis
    from the reason parts -- in the window drawers and in the chart payload."""
    pts = (("2026-09-25T00:00:00Z", 6.0),)
    head = _window(name="verdicts_missed", label="Review verdicts missed",
                   unit="verdicts", points=pts, n=6)  # fmt: skip
    prs = _window(name="verdicts_missed.prs", label="PRs with missed verdicts",
                  points=(("2026-09-25T00:00:00Z", 3.0),), n=3)  # fmt: skip
    reason = _window(name="verdicts_missed.died_mid_session",
                     points=(("2026-09-25T00:00:00Z", 4.0),), n=4)  # fmt: skip
    cause = _window(name="verdicts_missed.cause.api_error:429",
                    points=(("2026-09-25T00:00:00Z", 4.0),), n=4)  # fmt: skip
    metric = MetricData("verdicts_missed", (head, prs, reason, cause), {})
    card = card_of(metric)
    # distinct PRs lead, attempts in parentheses -- in the card and the detail head
    assert (card.prs, card.value) == (3.0, 6.0)
    assert '<span class="num">3 PRs (6 attempts)</span>' in render_head(card, "7d")
    lower = render_lower(card, metric, frozenset(), UTC)
    assert "By reason over the window" in lower and "died mid session" in lower
    assert "By cause over the window" in lower and "api error:429" in lower
    q = range_query("7d", END, UTC)
    entry = chart_entry(card, metric, bucket_grid(q))
    assert set(entry["parts"]) == {"died mid session"}  # prs/cause are not reasons
    assert set(entry["causes"]) == {"api error:429"}
    idx = list(bucket_grid(q)).index("2026-09-25T00:00:00Z")
    assert entry["prs"] == [[idx, 3.0, "3"]]


def test_p90_reference_uses_the_raw_samples_in_chart_units() -> None:
    head = _series(
        kind="duration", unit="hours", samples={"o/a": tuple(float(v) for v in range(1, 11))}
    )
    # nearest-rank p90 over all repos' samples (9th of 1..10), in the series' own unit
    assert duration_stats(head) == (10, 5.5, 9.0)
    two = _series(kind="duration", unit="hours", samples={"o/a": (1.0, 2.0), "o/b": (3.0,)})
    assert duration_stats(two) == (3, 2.0, 3.0)  # repos are pooled
    # only duration kinds with samples get the stats
    assert duration_stats(_series(kind="count", samples={"o/a": (1.0,)})) is None
    assert duration_stats(_series(kind="duration")) is None


def test_a_multi_panel_card_states_coverage_exactly_once() -> None:
    pts = tuple((f"2026-09-{d:02d}T00:00:00Z", 1.0) for d in range(24, 30))  # fmt: skip
    head = _window(
        name="m",
        points=pts,
        n=6,
        per_repo={"o/alpha": pts, "o/beta": pts},
        repo_coverage={
            "o/alpha": ("2026-09-01T00:00:00Z", "2026-09-30T00:00:00Z"),
            "o/beta": ("2026-09-26T00:00:00Z", "2026-09-30T00:00:00Z"),  # starts in-window
        },
    )
    metric = MetricData("m", (head,), {"m": "t"})
    html = render_lower(card_of(metric), metric, frozenset(), UTC)
    assert html.count('<div class="hbar') == 2  # the per-repo drawer really rendered
    assert html.count("Coverage:") == 1
    assert "sources:" not in html


def test_a_render_fault_gets_its_own_message_and_is_logged(caplog, monkeypatch) -> None:
    from charlie_work.dashboard.pages import history

    q = range_query("7d", END, UTC)
    head = _window(name="m", points=(("2026-09-25T00:00:00Z", 1.0),), n=1)
    view = HistoryView("flow", "7d", q, (MetricData("m", (head,), {"m": "t"}),), END)

    def boom(*a, **k):
        raise ValueError("bad geometry")

    monkeypatch.setattr(history, "chart_entry", boom)
    with caplog.at_level(logging.ERROR, "charlie_work.dashboard"):
        html = history.render_region({"flow": view}, "7d", "flow")
    assert "the card failed to render" in html and "a stored row" not in html
    assert any(r.name == "charlie_work.dashboard" and r.exc_info for r in caplog.records)
