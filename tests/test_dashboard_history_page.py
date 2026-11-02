"""The History page on a synthetic dashboard.db built by the REAL rollup from synthetic events.

Takeaways are re-derived here straight from ``metrics.tab_series`` + ``takeaways.takeaway``
(not from the page's own data object) and must appear verbatim in the rendered card.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from html import escape, unescape
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

from charlie_work import instrumentation
from charlie_work.dashboard import history_data, metrics, rollup, sources
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


def _page(path: Path | None, tab: str, range_key: str = "7d") -> str:
    """The full page as the server renders it: one range, every tab."""
    results = {k: load_tab(path, k, range_key, NOW) for k in history_data.TAB_KEYS}
    return render_history(results, range_key, tab)


def _view_page(view, tab: str) -> str:
    return render_history({tab: view}, "7d", tab)


def _block(page: str, ident: str, end: str) -> str:
    start = page.index(f'id="{ident}"')
    return page[start : page.index(end, start)]


def _card(page: str, metric_id: str) -> str:
    """Everything the page says about one metric: its card button, its headline and its
    window drawers (one block each, shown by history.js when the card is selected)."""
    slug = card_id(metric_id)[len("card-") :]
    lower_start = page.index(f'id="lower-{slug}"')
    ends = (
        page.find('class="mlower"', lower_start + 1),
        page.find("</section></div></div>", lower_start),
    )
    nxt = [i for i in ends if i != -1]
    return (
        _block(page, card_id(metric_id), "</button>")
        + _block(page, f"head-{slug}", "</div>")
        + page[lower_start : min(nxt)]
    )


def _payload(page: str) -> dict:
    raw = re.search(r'data-hist="([^"]*)"', page).group(1)
    return json.loads(unescape(raw))


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
    page = _page(db_path, "quality", "30d")
    tabs = re.findall(
        r'<button type="button" id="tab-(\w+)" data-tab="\1"[^>]*aria-pressed="(\w+)"', page
    )
    assert [t[0] for t in tabs] == ["flow", "quality", "capacity", "reliability"]
    assert [t[0] for t in tabs if t[1] == "true"] == ["quality"]
    ranges = re.findall(
        r'<button type="button" id="range-(\w+)" data-range="\1" aria-pressed="(\w+)"', page
    )
    assert [r[0] for r in ranges] == ["24h", "7d", "30d", "90d"]
    assert [r[0] for r in ranges if r[1] == "true"] == ["30d"]
    # every tab's cards are on the page; only the selected tab's list shows without JS
    lists = re.findall(r'<div class="cardlist" data-tab="(\w+)"([^>]*)>', page)
    assert [(k, " hidden" in rest) for k, rest in lists] == [
        ("flow", True),
        ("quality", False),
        ("capacity", True),
        ("reliability", True),
    ]
    # the refresh re-fetches this range's region, so the range survives an htmx swap
    assert 'hx-get="/history/fragment?range=30d"' in page and 'data-range="30d"' in page
    # g h / g n: the header nav carries the data-go links; History is the current view
    assert '<a href="/history" aria-current="page" data-go="h">History</a>' in page
    assert '<a href="/now" data-go="n">Now</a>' in page
    assert "<style" not in page and " style=" not in page and "<script>" not in page


def test_bucket_size_is_derived_from_range() -> None:
    assert [range_query(k, NOW).bucket.total_seconds() / 3600 for k in history_data.RANGES] == [
        1,
        6,
        24,
        24,
    ]
    assert history_data.pick_range("bogus") == "7d" and history_data.pick_tab("x") == "flow"


@pytest.mark.parametrize("tab", ["flow", "quality", "capacity", "reliability"])
def test_every_takeaway_equals_takeaways_output(db_path: Path, tab: str) -> None:
    view = load_tab(db_path, tab, "7d", NOW)
    assert isinstance(view, HistoryView)
    page = _view_page(view, tab)
    expected = _expected_takeaways(db_path, history_data.TAB_KEYS[tab])
    assert expected
    for metric_id, text in expected.items():
        assert f'<span class="takeaway">{escape(text)}</span>' in _card(page, metric_id), metric_id


def test_coverage_note_names_each_source_start_in_local_time(db_path: Path) -> None:
    page = _page(db_path, "flow")
    card = _card(page, "merges_per_day")
    view = load_tab(db_path, "flow", "7d", NOW)
    spans = view.get("merges_per_day").headline.repo_coverage
    assert set(spans) == {"owner/alpha", "local/beta"}
    for repo, short in (("owner/alpha", "alpha"), ("local/beta", "beta")):
        local = parse_ts(spans[repo][0]).astimezone().strftime("%m-%d")
        assert re.search(rf"{short} from <time datetime=\"[^\"]+Z\">{local}</time>", card)
    assert 'class="cov-note">Coverage: ' in card


def test_unclassified_event_kinds_are_named_on_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#2269: kinds the rollup does not interpret show up in the page coverage caveat,
    named, so a new writer kind can never drop silently again.

    #2525: one events.db with one event. ``_rolled``'s registry-plus-``build_base``
    fleet pays two ``touch_repo`` writes, ~45 WAL inserts and four closeable
    database files for an assertion that only needs the real
    ``log_event`` -> ``run_rollup`` -> ``load_tab`` chain to run once -- and the
    ledger flagged exactly that host-bound cost (0.17 s -> 1.07 s median).
    The multi-source shape is still covered by the ``db_path`` fixture's tests.
    """
    state = tmp_path / "orch" / ".var" / "charlie-work" / "state.json"
    monkeypatch.setattr(instrumentation, "_now_iso", lambda: "2026-10-01T10:30:00Z")
    instrumentation.log_event(state, "brand_new_unclassified_kind", {})
    instrumentation.close_db(state)
    (tmp_path / "fleet").mkdir()
    sources = rollup.RollupSources(
        db_path=tmp_path / "fleet" / "dashboard.db",
        fleet_events_db=state.parent / "events.db",
        repos=(),
    )
    assert rollup.run_rollup(sources, NOW).errors == ()
    view = load_tab(sources.db_path, "flow", "7d", NOW)
    assert isinstance(view, HistoryView)
    assert view.unclassified_kinds == ("brand_new_unclassified_kind",)
    page = _view_page(view, "flow")
    assert "1 event kind not interpreted by the rollup: brand_new_unclassified_kind" in page


def test_approx_series_is_dashed_and_says_why(db_path: Path) -> None:
    page = _page(db_path, "flow")
    card = _card(page, "lead_time")
    assert "Approx. until lifecycle instrumentation (#2226)" in card  # the headline, in words
    assert "median · approx." in card  # and on the card itself: not colour alone
    entry = _payload(page)["metrics"]["lead_time"]
    assert entry["chart"] == "line" and entry["approx"] is True
    # history-chart.js marks an approx line, and history.css dashes exactly that class
    from charlie_work.dashboard.theme import static_asset

    assert '"cline" + (entry.approx ? " approx" : "")' in static_asset(
        "history-chart.js"
    ).read_text(encoding="utf-8")
    assert re.search(
        r"\.cline\.approx \{[^}]*stroke-dasharray",
        static_asset("history.css").read_text(encoding="utf-8"),
    )


def test_not_instrumented_card_is_honest(tmp_path: Path) -> None:
    # #2526: the card only needs a live fleet source that never wrote
    # runner_allocation — a 3-repo Fleet + build_base's full history made this
    # test a ledger hot spot for no extra coverage.
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    heartbeat = fleet_dir / sources.HEARTBEAT_FILENAME
    instrumentation.log_event(heartbeat, "supervisor_started", {})
    instrumentation.close_db(heartbeat)
    resolved = rollup.rollup_sources(str(fleet_dir))
    assert rollup.run_rollup(resolved, NOW).errors == ()
    path = resolved.db_path  # no runner_allocation anywhere
    view = load_tab(path, "capacity", "7d", NOW)
    assert isinstance(view, HistoryView) and view.get("runners_running").headline.not_instrumented
    page = _view_page(view, "capacity")
    card = _card(page, "runners_running")
    assert "not instrumented yet" in card and "<svg" not in card
    assert f'id="{card_id("runners_capacity")}"' in page  # not overlaid: keeps its own card


def test_missing_db_is_rollup_not_available_never_zeros(tmp_path: Path) -> None:
    result = load_tab(tmp_path / "nope" / "dashboard.db", "flow", "7d", NOW)
    assert isinstance(result, HistoryUnavailable) and "missing" in result.reason
    page = render_history({k: result for k in history_data.TAB_KEYS}, "7d", "flow")
    assert "Rollup not available" in page
    assert "<svg" not in page and 'class="takeaway"' not in page and ">0<" not in page
    assert isinstance(load_tab(None, "flow", "7d", NOW), HistoryUnavailable)


def test_locked_db_is_a_value_not_a_500(db_path: Path, monkeypatch) -> None:
    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(history_data, "tab_results", locked)
    result = load_tab(db_path, "flow", "7d", NOW)
    assert isinstance(result, HistoryUnavailable) and "database is locked" in result.reason


def test_a_malformed_row_degrades_only_its_own_card(db_path: Path, caplog) -> None:
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
    with caplog.at_level(logging.ERROR, "charlie_work.dashboard"):
        view = load_tab(db_path, "flow", "7d", NOW)
    assert isinstance(view, HistoryView)
    assert view.get("wip").error is not None
    assert view.get("wip").error_kind == "data"  # float("not-a-number") raised ValueError
    # the degraded metric's swallowed exception is logged with its traceback
    logged = [r for r in caplog.records if "wip" in r.getMessage()]
    assert logged and all(r.exc_info for r in logged)
    page = _view_page(view, "flow")
    card = _card(page, "wip")
    assert "could not be drawn" in card and "malformed" in card and "<svg" not in card
    assert "<svg" in _card(page, "merges_per_day")  # the row takes down only its own card
    assert "<svg" in _card(page, "queue_depth")  # same table, other column: unaffected


def test_a_metric_code_fault_does_not_claim_a_malformed_row(
    db_path: Path, monkeypatch, caplog
) -> None:
    # A bug inside the metric function (TypeError here — KeyError behaves the same) is
    # not a malformed stored row, and the degraded card must not claim one.
    def boom(db, q):
        raise TypeError("bug in the metric")

    monkeypatch.setitem(metrics.TABS["Flow"], "wip", boom)
    with caplog.at_level(logging.ERROR, "charlie_work.dashboard"):
        view = load_tab(db_path, "flow", "7d", NOW)
    assert isinstance(view, HistoryView)
    assert view.get("wip").error == "TypeError: bug in the metric"
    assert view.get("wip").error_kind == "internal"
    assert any("wip" in r.getMessage() and r.exc_info for r in caplog.records)
    page = _view_page(view, "flow")
    card = _card(page, "wip")
    assert "could not be drawn" in card and "an internal error" in card
    assert "a stored row" not in card and "malformed" not in card
    assert "<svg" in _card(page, "merges_per_day")  # only its own card degrades


def test_a_takeaway_fault_is_logged_and_degrades_its_card(
    db_path: Path, monkeypatch, caplog
) -> None:
    def boom(cur, pri):
        raise KeyError("assessment")

    monkeypatch.setattr(history_data, "paired_assessments", boom)
    with caplog.at_level(logging.ERROR, "charlie_work.dashboard"):
        view = load_tab(db_path, "flow", "7d", NOW)
    assert isinstance(view, HistoryView)
    assert all(m.error is not None for m in view.metrics)
    assert all(m.error_kind == "takeaway" for m in view.metrics)
    assert any(
        "takeaway assessment failed" in r.getMessage() and r.exc_info for r in caplog.records
    )
    page = _view_page(view, "flow")
    card = _card(page, "wip")
    # a fault in the takeaway code is not a malformed stored row: the card says so
    assert "could not be drawn" in card and "its takeaway could not be computed" in card
    assert "a stored row" not in card and "malformed" not in card


def test_every_duration_card_states_median_and_p90(db_path: Path) -> None:
    checked = 0
    for tab in ("flow", "capacity", "reliability"):
        view = load_tab(db_path, tab, "7d", NOW)
        assert isinstance(view, HistoryView)
        page = _view_page(view, tab)
        charts = _payload(page)["metrics"]
        for m in view.metrics:
            if m.error is not None or m.headline.kind != "duration" or not m.headline.samples:
                continue
            checked += 1
            card = _card(page, m.metric_id)
            assert re.search(r"Median \S+ · p90 \S+ · n \d+", card), m.metric_id
            assert charts[m.metric_id]["chart"] == "line", m.metric_id  # a level: a line
    assert checked > 0, "no duration card carried samples — the assertions above ran zero times"


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
    # --lj-* tokens, plus the fluid --n-* scale now-page.css defines on body.now (History
    # is <body class="now hist">, so it shares that scale)
    shared = static_asset("now-page.css").read_text(encoding="utf-8")
    defined = set(re.findall(r"(--[\w-]+)\s*:", generate_css() + shared))
    used = set(re.findall(r"var\((--[\w-]+)", css))
    assert used and used <= defined, sorted(used - defined)
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|transparent", css)
