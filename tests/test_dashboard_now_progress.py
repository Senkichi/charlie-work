"""The Now page's progress chart: the data payload, the headline words and the fallbacks."""

from __future__ import annotations

import html
import json
import re
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest
from _dashboard_metrics_fixtures import NOW, build_base, build_quality
from _dashboard_page_fixtures import _model
from _dashboard_rollup_fixtures import Fleet

from charlie_work.dashboard import metrics, rollup
from charlie_work.dashboard.history_data import bucket_end
from charlie_work.dashboard.metrics_base import MetricQuery, open_dashboard_ro
from charlie_work.dashboard.now_progress_data import (
    METRICS,
    RANGES,
    ProgressCache,
    ProgressData,
    ProgressSeries,
    ProgressUnavailable,
    load_progress,
)
from charlie_work.dashboard.pages.now import render_fragment, render_now
from charlie_work.dashboard.pages.now_progress import headline, payload, render_progress, subtitle
from charlie_work.dashboard.read_model import ModelState


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    f = Fleet(tmp_path, monkeypatch)
    build_base(f)
    build_quality(f)
    assert rollup.run_rollup(f.sources(), NOW).errors == ()
    path = f.sources().db_path
    f.close()
    return path


def _data(db_path: Path) -> ProgressData:
    result = load_progress(db_path, NOW)
    assert isinstance(result, ProgressData), result
    return result


def _embedded(page: str) -> dict:
    raw = re.search(r'data-progress="([^"]*)"', page)
    assert raw is not None
    return json.loads(html.unescape(raw.group(1)))


def _series(db_path: Path, metric: str, range_key: str) -> ProgressSeries:
    got = _data(db_path).get(metric, range_key)
    assert got is not None
    return got


def _ints(*xs: float) -> tuple[tuple[str, float], ...]:
    return tuple((f"2026-10-0{i + 1}T00:00:00Z", x) for i, x in enumerate(xs))


def _fake(metric: str = "merged", range_key: str = "7d", delta: float | None = None, *pts: float):
    return ProgressSeries(metric, range_key, _ints(*pts), {}, delta, False, False)


# ---- the payload: both metrics across all four ranges ---------------------------------


@pytest.mark.parametrize("range_key", list(RANGES))
@pytest.mark.parametrize("metric", list(METRICS))
def test_every_metric_and_range_is_in_the_payload(
    db_path: Path, metric: str, range_key: str
) -> None:
    page = render_progress(_data(db_path))
    plain = _embedded(page)["metrics"][metric]["ranges"][range_key]
    span, bucket, _, _ = RANGES[range_key]
    assert 0 < len(plain["points"]) <= span // bucket  # populated buckets inside the window
    assert plain["head"].startswith(tuple("0123456789")) and "in the last" in plain["head"]
    assert plain["sub"] and plain["bucket"] and plain["phrase"]
    for repo, pairs in plain["repo"].items():
        assert repo and all(0 <= i < len(plain["points"]) and v > 0 for i, v in pairs)


@pytest.mark.parametrize("range_key", list(RANGES))
@pytest.mark.parametrize(
    ("metric", "tab", "metric_id"),
    [("merged", "Flow", "merges_per_day"), ("escalated", "Quality", "escalations")],
)
def test_points_equal_the_history_metric_over_the_same_window(
    db_path: Path, metric: str, tab: str, metric_id: str, range_key: str
) -> None:
    """Not re-derived: the chart's points are what History's own metric returns."""
    span, bucket, _, _ = RANGES[range_key]
    end = bucket_end(NOW, bucket)
    conn, err = open_dashboard_ro(db_path)
    assert conn is not None, err
    try:
        got = metrics.TABS[tab][metric_id](conn, MetricQuery(end - span, end, bucket))
    finally:
        conn.close()
    expected = got[0] if isinstance(got, tuple) else got
    assert _series(db_path, metric, range_key).points == expected.points


def test_per_repo_split_sums_to_the_bar_it_explains(db_path: Path) -> None:
    s = _series(db_path, "merged", "30d")
    assert s.per_repo, "the scenario merges in two repos"
    for i, (_, total) in enumerate(s.points):
        split = sum(v for pts in s.per_repo.values() for j, v in pts if j == i)
        assert split == pytest.approx(total)
    assert sum(v for _, v in s.points) > 0


def test_payload_urls_cover_every_split_repo(db_path: Path) -> None:
    data = _data(db_path)
    urls = json.loads(payload(data))["urls"]
    repos = {r for s in data.series for r in s.per_repo}
    assert repos and set(urls) == repos
    assert all(u is None or u.startswith("/repo/") for u in urls.values())


def test_payload_json_cannot_break_out_of_its_attribute(db_path: Path) -> None:
    page = render_progress(_data(db_path))
    attr = re.search(r'data-progress="([^"]*)"', page)
    assert attr is not None and "<" not in attr.group(1)


def test_page_default_headline_matches_the_payload(db_path: Path) -> None:
    page = render_progress(_data(db_path))
    default = _embedded(page)["metrics"]["merged"]["ranges"]["7d"]
    assert f'<span id="prog-head">{html.escape(default["head"])}</span>' in page
    assert page.count('aria-pressed="true"') == 2  # one metric, one range


# ---- headline words ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric", "delta", "expected"),
    [
        ("merged", 0.28, ("315 PRs merged in the last 7 days", "up 28% vs prior")),
        ("merged", -0.1, ("315 PRs merged in the last 7 days", "down 10% vs prior")),
        ("merged", 0.0, ("315 PRs merged in the last 7 days", "flat vs prior")),
        ("merged", None, ("315 PRs merged in the last 7 days", "")),
        ("escalated", 0.5, ("315 escalations in the last 7 days", "up 50% vs prior")),
    ],
)
def test_headline_states_total_and_comparison(metric, delta, expected) -> None:
    assert headline(_fake(metric, "7d", delta, 300, 15)) == expected


def test_headline_singular_and_zero() -> None:
    assert headline(_fake("merged", "24h", None, 1))[0] == "1 PR merged in the last 24 hours"
    assert headline(_fake("escalated", "30d", None, 1))[0] == "1 escalation in the last 30 days"
    assert headline(_fake("escalated", "30d", None, 0))[0] == "0 escalations in the last 30 days"


def test_subtitle_keeps_the_approximate_caveat_and_says_not_instrumented() -> None:
    approx = ProgressSeries("merged", "7d", _ints(1), {}, None, True, False)
    assert "Approximate" in subtitle(approx)
    assert "Approximate" not in subtitle(_fake())
    missing = ProgressSeries("escalated", "7d", (), {}, None, False, True)
    assert "Not instrumented" in subtitle(missing)


# ---- fallbacks: dashboard.db unavailable ---------------------------------------------


def test_no_db_configured_is_one_line_not_a_failure() -> None:
    result = load_progress(None, NOW)
    assert isinstance(result, ProgressUnavailable)
    panel = render_progress(result)
    assert "History is unavailable" in panel and panel.count("<p") == 1
    assert "data-progress" not in panel and "<svg" not in panel and "<button" not in panel


def test_missing_and_corrupt_db_are_values(tmp_path: Path) -> None:
    assert isinstance(load_progress(tmp_path / "nope.db", NOW), ProgressUnavailable)
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a sqlite file at all" * 50)
    got = load_progress(bad, NOW)
    assert isinstance(got, ProgressUnavailable) and got.reason


def test_db_without_the_tables_is_unavailable_not_zero(tmp_path: Path) -> None:
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    assert isinstance(load_progress(empty, NOW), ProgressUnavailable)


def test_page_survives_unavailable_progress_and_escapes_the_reason() -> None:
    page = render_now(
        ModelState(model=_model()),
        poll_seconds=15,
        progress=ProgressUnavailable("cannot read <x>: boom"),
    )
    assert "cannot read &lt;x&gt;: boom" in page and "<x>" not in page
    assert 'id="needs"' in page and 'id="flow"' in page and 'id="capacity"' in page
    assert "History is unavailable" in render_fragment(ModelState(model=_model()), 15)


# ---- the cache -----------------------------------------------------------------------


def test_cache_serves_a_hit_without_a_second_load_and_retries_errors_sooner(
    db_path: Path, tmp_path: Path
) -> None:
    tick = [0.0]
    cache = ProgressCache(db_path, 120.0, lambda: NOW, lambda: tick[0])
    first = cache.get()
    assert isinstance(first, ProgressData)
    tick[0] = 119.0
    assert cache.get() is first and cache.misses == 1
    tick[0] = 121.0
    cache.get()
    assert cache.misses == 2
    failing = ProgressCache(tmp_path / "nope.db", 120.0, lambda: NOW, lambda: tick[0])
    failing.get()
    tick[0] += 16.0  # past the 15s error TTL, far inside the 120s success TTL
    failing.get()
    assert failing.misses == 2


def test_windows_end_on_whole_buckets(db_path: Path) -> None:
    """A 30d chart ends at the last whole day, so a partial day never reads as a collapse."""
    s = _series(db_path, "merged", "30d")
    last = s.points[-1][0]
    assert last.endswith(("T00:00:00Z", ":00:00Z"))
    assert timedelta(days=30) == RANGES["30d"][0]
