"""The History page on a synthetic dashboard.db built by the REAL rollup from synthetic events.

Takeaways are re-derived here straight from ``metrics.tab_series`` + ``takeaways.takeaway``
(not from the page's own data object) and must appear verbatim in the rendered card.
"""

from __future__ import annotations

import re
import sqlite3
from html import escape
from pathlib import Path

import pytest
from _dashboard_metrics_fixtures import (
    NOW,
    build_base,
    build_capacity,
    build_quality,
    build_reliability,
)
from _dashboard_rollup_fixtures import Fleet

from charlie_work.dashboard import history_data, metrics, rollup
from charlie_work.dashboard.history_data import (
    HistoryCache,
    HistoryUnavailable,
    HistoryView,
    load_tab,
    range_query,
)
from charlie_work.dashboard.metrics_base import open_dashboard_ro
from charlie_work.dashboard.pages.history import render_history
from charlie_work.dashboard.pages.history_cards import card_id
from charlie_work.dashboard.takeaways import takeaway
from charlie_work.dashboard.timeutil import parse_ts


def _rolled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, builders) -> Path:
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    f = Fleet(tmp_path, monkeypatch)
    for build in builders:
        build(f)
    assert rollup.run_rollup(f.sources(), NOW).errors == ()
    path = f.sources().db_path
    f.close()
    return path


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _rolled(
        tmp_path, monkeypatch, (build_base, build_quality, build_capacity, build_reliability)
    )


def _card(page: str, metric_id: str) -> str:
    start = page.index(f'id="{card_id(metric_id)}"')
    return page[start : page.index("</article>", start)]


def _expected_takeaways(path: Path, tab: str) -> dict[str, str]:
    conn, err = open_dashboard_ro(path)
    assert conn is not None, err
    try:
        q = range_query("7d", NOW)
        cur = metrics.tab_series(conn, tab, q)
        pri = metrics.tab_series(conn, tab, q.prior())
    finally:
        conn.close()
    return {mid: takeaway(series[0], pri[mid][0]) for mid, series in cur.items()}


def test_tabs_and_shared_range_are_keyboard_links(db_path: Path) -> None:
    page = render_history(load_tab(db_path, "quality", "14d", NOW), "quality", "14d")
    tabs = re.findall(r'<a id="tab-(\w+)" href="([^"]+)"( aria-current="page")?>', page)
    assert [t[0] for t in tabs] == ["flow", "quality", "capacity", "reliability"]
    assert [t[0] for t in tabs if t[2]] == ["quality"]
    assert all(href.endswith("&amp;range=14d") for _, href, _ in tabs)  # tab keeps the range
    ranges = re.findall(r'<a id="range-(\w+)" href="([^"]+)"( aria-current="true")?>', page)
    assert [r[0] for r in ranges] == ["7d", "14d", "30d", "90d"]
    assert [r[0] for r in ranges if r[2]] == ["14d"]
    assert all("tab=quality" in href for _, href, _ in ranges)  # range keeps the tab
    # g h / g n: the header nav carries the data-go links; History is the current view
    assert '<a href="/history" aria-current="page" data-go="h">History</a>' in page
    assert '<a href="/now" data-go="n">Now</a>' in page
    assert "<style" not in page and " style=" not in page and "<script>" not in page


def test_bucket_size_is_derived_from_range() -> None:
    assert [range_query(k, NOW).bucket.total_seconds() / 3600 for k in history_data.RANGES] == [
        6,
        12,
        24,
        72,
    ]
    assert history_data.pick_range("bogus") == "7d" and history_data.pick_tab("x") == "flow"


@pytest.mark.parametrize("tab", ["flow", "quality", "capacity", "reliability"])
def test_every_takeaway_equals_takeaways_output(db_path: Path, tab: str) -> None:
    view = load_tab(db_path, tab, "7d", NOW)
    assert isinstance(view, HistoryView)
    page = render_history(view, tab, "7d")
    expected = _expected_takeaways(db_path, history_data.TAB_KEYS[tab])
    assert expected
    for metric_id, text in expected.items():
        if metric_id in ("reviewers_cap", "runners_capacity"):
            continue  # drawn on its usage metric's chart, no card of its own
        assert f'<p class="takeaway">{escape(text)}</p>' in _card(page, metric_id), metric_id


def test_coverage_note_names_each_source_start_in_local_time(db_path: Path) -> None:
    page = render_history(load_tab(db_path, "flow", "7d", NOW), "flow", "7d")
    card = _card(page, "merges_per_day")
    view = load_tab(db_path, "flow", "7d", NOW)
    spans = view.get("merges_per_day").headline.repo_coverage
    assert set(spans) == {"owner/alpha", "local/beta"}
    for repo, short in (("owner/alpha", "alpha"), ("local/beta", "beta")):
        local = parse_ts(spans[repo][0]).astimezone().strftime("%m-%d")
        assert re.search(rf"{short} from <time datetime=\"[^\"]+Z\">{local}</time>", card)
    assert 'class="cov-note">Coverage: ' in card


def test_approx_series_is_dashed_and_says_why(db_path: Path) -> None:
    page = render_history(load_tab(db_path, "flow", "7d", NOW), "flow", "7d")
    card = _card(page, "lead_time")
    assert "approx. until lifecycle instrumentation (#2226)" in card
    combined = card[: card.index("</figure>")]
    assert 'class="series s1 approx"' in combined and 'stroke-dasharray="4 4"' in combined
    assert "approx." in combined  # direct label, not colour alone


def test_small_multiples_share_one_y_scale(db_path: Path) -> None:
    page = render_history(load_tab(db_path, "flow", "7d", NOW), "flow", "7d")
    card = _card(page, "merges_per_day")
    assert 'data-panels="2"' in card
    assert len(set(re.findall(r'data-y-max="([^"]+)"', card))) == 1


def test_usage_and_cap_share_one_chart(db_path: Path) -> None:
    page = render_history(load_tab(db_path, "capacity", "7d", NOW), "capacity", "7d")
    assert f'id="{card_id("reviewers_cap")}"' not in page
    card = _card(page, "reviewers_live")
    assert card.count('<g class="series') >= 2


def test_not_instrumented_card_is_honest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _rolled(tmp_path, monkeypatch, (build_base,))  # no runner_allocation anywhere
    view = load_tab(path, "capacity", "7d", NOW)
    assert isinstance(view, HistoryView) and view.get("runners_running").headline.not_instrumented
    page = render_history(view, "capacity", "7d")
    card = _card(page, "runners_running")
    assert "not instrumented yet" in card and "<svg" not in card
    assert f'id="{card_id("runners_capacity")}"' in page  # not overlaid: keeps its own card


def test_missing_db_is_rollup_not_available_never_zeros(tmp_path: Path) -> None:
    result = load_tab(tmp_path / "nope" / "dashboard.db", "flow", "7d", NOW)
    assert isinstance(result, HistoryUnavailable) and "missing" in result.reason
    page = render_history(result, "flow", "7d")
    assert "Rollup not available" in page
    assert "<svg" not in page and 'class="takeaway"' not in page and ">0<" not in page
    assert isinstance(load_tab(None, "flow", "7d", NOW), HistoryUnavailable)


def test_locked_db_is_a_value_not_a_500(db_path: Path, monkeypatch) -> None:
    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(history_data, "tab_results", locked)
    result = load_tab(db_path, "flow", "7d", NOW)
    assert isinstance(result, HistoryUnavailable) and "database is locked" in result.reason


def test_a_malformed_row_degrades_only_its_own_card(db_path: Path) -> None:
    # A TEXT value in a numeric column: SQLite stores it happily, and only the metric
    # reading that column (wip -> pass_samples.live_sessions) may fail — not its neighbours.
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO pass_samples (source, src_id, seq, ts, repo, live_sessions)"
        " VALUES ('owner/alpha', 999999, 0, '2026-10-05T00:00:00Z', 'owner/alpha',"
        " 'not-a-number')"
    )
    conn.commit()
    conn.close()
    view = load_tab(db_path, "flow", "7d", NOW)
    assert isinstance(view, HistoryView)
    assert view.get("wip").error is not None
    page = render_history(view, "flow", "7d")
    card = _card(page, "wip")
    assert "could not be drawn" in card and "malformed" in card and "<svg" not in card
    assert "<svg" in _card(page, "merges_per_day")  # the row takes down only its own card
    assert "<svg" in _card(page, "queue_depth")  # same table, other column: unaffected


def test_every_duration_card_states_median_and_p90(db_path: Path) -> None:
    for tab in ("flow", "capacity", "reliability"):
        view = load_tab(db_path, tab, "7d", NOW)
        assert isinstance(view, HistoryView)
        page = render_history(view, tab, "7d")
        for m in view.metrics:
            if m.error is not None or m.headline.kind != "duration" or not m.headline.samples:
                continue
            card = _card(page, m.metric_id)
            assert re.search(r">p90 [^<]*</text>", card), m.metric_id  # labelled ref rule
            assert 'class="strip-chart"' in card, m.metric_id
            assert re.search(r"median \S+ · p90 \S+ · n \d+", card), m.metric_id


def test_cache_keys_load_independently() -> None:
    import threading

    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def load(tab, rng, now):
        calls.append(f"{tab}:{rng}")
        if tab == "quality":
            entered.set()  # a cold quality load blocks mid-flight ...
            assert release.wait(5), "loader never released"
        return HistoryView(tab, rng, range_query(rng, now), (), now)

    cache = HistoryCache(load, ttl=60, clock=lambda: NOW)
    slow = threading.Thread(target=lambda: cache.get("quality", "7d"))
    fast = threading.Thread(target=lambda: cache.get("flow", "7d"))
    slow.start()
    assert entered.wait(5), "the quality load never started"
    fast.start()  # ... while a different key's load must NOT wait on it
    fast.join(5)
    assert not fast.is_alive(), "flow load serialised behind the quality load"
    release.set()
    slow.join(5)
    assert sorted(calls) == ["flow:7d", "quality:7d"]


def test_cache_hits_until_ttl_and_retries_errors_sooner() -> None:
    clock = {"t": 0.0}
    calls: list[tuple[str, str]] = []

    def load(tab, rng, now):
        calls.append((tab, rng))
        if tab == "quality":
            return HistoryUnavailable("down", now)
        return HistoryView(tab, rng, range_query(rng, now), (), now)

    cache = HistoryCache(load, ttl=120, clock=lambda: NOW, monotonic=lambda: clock["t"])
    first = cache.get("flow", "7d")
    assert cache.get("FLOW", "7d") is first and cache.misses == 1  # hit: no query
    cache.get("flow", "30d")
    assert cache.misses == 2  # the range is part of the key
    cache.get("quality", "7d")
    clock["t"] = history_data.ERROR_TTL_SECONDS + 1
    cache.get("quality", "7d")
    cache.get("flow", "7d")
    assert calls.count(("quality", "7d")) == 2 and calls.count(("flow", "7d")) == 1
    clock["t"] = 121
    cache.get("flow", "7d")
    assert calls.count(("flow", "7d")) == 2


def test_cache_turns_a_loader_crash_into_a_value() -> None:
    def load(tab, rng, now):
        raise KeyError("bug")

    result = HistoryCache(load, ttl=60, clock=lambda: NOW).get("flow", "7d")
    assert isinstance(result, HistoryUnavailable) and "KeyError" in result.reason


def test_history_css_uses_only_tokens() -> None:
    from charlie_work.dashboard.theme import generate_css, static_asset

    css = static_asset("history.css").read_text(encoding="utf-8")
    defined = set(re.findall(r"(--[\w-]+)\s*:", generate_css()))
    used = set(re.findall(r"var\((--[\w-]+)", css))
    assert used and used <= defined, sorted(used - defined)
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|transparent", css)
