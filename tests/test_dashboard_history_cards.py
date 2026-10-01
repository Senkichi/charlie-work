"""History card wiring (``pages/history_cards.py``) over synthetic series."""

from __future__ import annotations

from datetime import UTC, datetime

from charlie_work.dashboard.metrics_base import Series
from charlie_work.dashboard.pages.history_cards import _chart_coverage

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
