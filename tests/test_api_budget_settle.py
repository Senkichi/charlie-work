"""settle_session pure-function semantics (issue #480).

Idempotence, distinct-session accumulation, UTC day bucketing, and
input-ledger immutability.

Split out of ``tests/test_api_budget.py`` (issue #1571, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_api_budget_unit_fixtures.py``.
"""

from __future__ import annotations

import pytest

from charlie_work.api_budget import Ledger, settle_session

from _api_budget_unit_fixtures import _entry


def test_settle_session_appends_and_bumps_day_and_lifetime() -> None:
    entry = _entry(usd=6.15, input_tokens=1_000_000, output_tokens=200_000, cached_tokens=500_000)
    ledger = settle_session(Ledger(), entry)
    assert len(ledger.sessions) == 1
    assert ledger.sessions[0] is entry
    assert ledger.lifetime_usd == pytest.approx(6.15)
    bucket = ledger.days["2026-07-22"]
    assert bucket.input_tokens == 1_000_000
    assert bucket.output_tokens == 200_000
    assert bucket.cached_tokens == 500_000
    assert bucket.usd == pytest.approx(6.15)


def test_settle_session_idempotent_double_settle_changes_nothing() -> None:
    entry = _entry()
    once = settle_session(Ledger(), entry)
    twice = settle_session(once, entry)
    assert twice is once  # no-op returns the same object
    assert twice.sessions == once.sessions
    assert twice.lifetime_usd == once.lifetime_usd
    assert twice.days == once.days


def test_settle_session_idempotent_identical_key_different_other_fields() -> None:
    """Idempotence is by (issue, started_at, session_id) — a second entry with
    the same key but different cost is still a no-op (the first settlement wins)."""
    first = _entry(usd=6.15, outcome="completed")
    ledger = settle_session(Ledger(), first)
    # Same identity, different cost/outcome — must NOT double-count.
    duplicate = _entry(usd=99.0, outcome="reaped")
    result = settle_session(ledger, duplicate)
    assert result is ledger
    assert result.lifetime_usd == pytest.approx(6.15)
    assert len(result.sessions) == 1


def test_settle_session_distinct_sessions_accumulate() -> None:
    e1 = _entry(session_id="sess-1", usd=2.0, started_at="2026-07-22T10:00:00Z")
    e2 = _entry(session_id="sess-2", usd=3.0, started_at="2026-07-22T14:00:00Z")
    ledger = settle_session(settle_session(Ledger(), e1), e2)
    assert len(ledger.sessions) == 2
    assert ledger.lifetime_usd == pytest.approx(5.0)
    assert ledger.days["2026-07-22"].usd == pytest.approx(5.0)


def test_settle_session_utc_day_bucketing_no_local_time_drift() -> None:
    """A session started at 23:30 UTC lands in the UTC day it started, not a
    local-time-rolled day. A session at 00:15 UTC lands in the new UTC day."""
    late = _entry(session_id="late", started_at="2026-07-22T23:30:00Z", usd=1.0)
    early = _entry(session_id="early", started_at="2026-07-23T00:15:00Z", usd=2.0)
    ledger = settle_session(settle_session(Ledger(), late), early)
    assert ledger.days["2026-07-22"].usd == pytest.approx(1.0)
    assert ledger.days["2026-07-23"].usd == pytest.approx(2.0)
    assert ledger.lifetime_usd == pytest.approx(3.0)


def test_settle_session_naive_timestamp_treated_as_utc() -> None:
    entry = _entry(session_id="s", started_at="2026-07-22T12:00:00", usd=1.0)
    ledger = settle_session(Ledger(), entry)
    assert "2026-07-22" in ledger.days


def test_settle_session_does_not_mutate_input_ledger() -> None:
    entry = _entry()
    original = Ledger()
    settle_session(original, entry)
    assert original.sessions == ()
    assert original.lifetime_usd == 0.0
    assert dict(original.days) == {}


def test_settle_session_malformed_started_at_skips_day_bucket() -> None:
    """A malformed started_at still settles the session + lifetime, just without a day bucket."""
    entry = _entry(session_id="s", started_at="not-a-timestamp", usd=1.5)
    ledger = settle_session(Ledger(), entry)
    assert len(ledger.sessions) == 1
    assert ledger.lifetime_usd == pytest.approx(1.5)
    assert dict(ledger.days) == {}
