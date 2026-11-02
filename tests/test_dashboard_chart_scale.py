"""Local-time tick generation for the lane chart: literal assertions on tiny inputs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from charlie_work.dashboard.charts.scale import time_ticks

T0 = datetime(2026, 9, 28, tzinfo=UTC)


def day(n: int) -> datetime:
    return T0 + timedelta(days=n)


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


def test_week_window_with_three_ticks_gets_more_than_one_tick() -> None:
    assert len(time_ticks(day(0), day(7), UTC, max_ticks=3)) >= 2
