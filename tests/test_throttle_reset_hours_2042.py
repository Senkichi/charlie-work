"""Issue #2042: hour-stated provider resets and monotonic throttle writes."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from charlie_work.state import set_throttled_until
from charlie_work.throttle_signatures import match_throttle_tail

_MARKERS = ["rate limit"]
_VERBATIM = (
    "Error: Agent error: Reached free model rate limit. Upgrade to Max for "
    "higher limits, or switch to a different model. Your limit will reset "
    "in 4 hours 8 minutes."
)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        (_VERBATIM, 248),
        ("rate limit; reset in 2 hours", 120),
        ("rate limit; resets in 1 hour", 60),
        ("rate limit; resets in 30 minutes", 30),
        ("rate limit; resets in 1 hour 5 minutes", 65),
        ("rate limit; resets in 2 hours and 15 minutes", 135),
        ("rate limit; reset in 5 seconds", None),
        ("rate limit; no reset info", None),
    ],
)
def test_match_throttle_tail_parses_hours_and_minutes(tail: str, expected: int | None) -> None:
    assert match_throttle_tail(tail, _MARKERS) == (True, expected)


def test_classify_session_failure_verbatim_hours_notice(tmp_path: Path) -> None:
    from charlie_work.devin_shell import _classify_session_failure

    log_path = tmp_path / "session.log"
    log_path.write_text("work...\n" + _VERBATIM + "\n", encoding="utf-8")
    now = datetime.now(UTC).replace(microsecond=0)
    os.utime(log_path, (now.timestamp(), now.timestamp()))

    kind, until = _classify_session_failure(log_path, now=now)

    assert kind == "rate_limited"
    assert until is not None
    parsed = datetime.fromisoformat(until.replace("Z", "+00:00"))
    # 4h08m plus at most a small resume margin; never the 15-minute fallback.
    delta = parsed - now
    assert timedelta(hours=4, minutes=8) <= delta < timedelta(hours=4, minutes=20)


def test_set_throttled_until_never_shortens_active_window() -> None:
    now = datetime.now(UTC)
    long_until = _iso(now + timedelta(hours=4))
    short_until = _iso(now + timedelta(minutes=15))
    state = set_throttled_until(
        {}, long_until, source="test", reason="quota_exhausted", adapter_kind="devin"
    )

    state = set_throttled_until(
        state, short_until, source="test", reason="rate_limited", adapter_kind="claude"
    )

    assert state["throttled_until"] == long_until
    assert state["throttle_reason"] == "quota_exhausted"
    assert state["throttle_adapter_kind"] == "devin"


def test_set_throttled_until_extends_with_later_window() -> None:
    now = datetime.now(UTC)
    short_until = _iso(now + timedelta(minutes=15))
    long_until = _iso(now + timedelta(hours=4))
    state = set_throttled_until(
        {}, short_until, source="test", reason="rate_limited", adapter_kind="claude"
    )

    state = set_throttled_until(
        state, long_until, source="test", reason="quota_exhausted", adapter_kind="devin"
    )

    assert state["throttled_until"] == long_until
    assert state["throttle_reason"] == "quota_exhausted"
    assert state["throttle_adapter_kind"] == "devin"


def test_set_throttled_until_replaces_expired_window() -> None:
    now = datetime.now(UTC)
    expired = _iso(now - timedelta(hours=1))
    new = _iso(now + timedelta(minutes=5))
    state = set_throttled_until(
        {}, expired, source="test", reason="quota_exhausted", adapter_kind="devin"
    )

    state = set_throttled_until(
        state, new, source="test", reason="rate_limited", adapter_kind="claude"
    )

    assert state["throttled_until"] == new
    assert state["throttle_reason"] == "rate_limited"


def test_set_throttled_until_sets_when_none_active() -> None:
    new = _iso(datetime.now(UTC) + timedelta(minutes=5))
    state = set_throttled_until(
        {"throttled_until": None}, new, source="test", reason="rate_limited"
    )
    assert state["throttled_until"] == new
