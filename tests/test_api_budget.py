"""Tests for the api worker spend ledger (issue #480).

Pure-function coverage: cost math (including cached-token pricing), settlement
idempotence, corrupt-file recovery, atomicity, UTC day bucketing, and
budget_status boundary conditions. Plus the reap-path settlement wiring into
``WorkerView.reap_sidecar`` for ``adapter_kind == "api"``.

Run targeted: ``uv run --extra dev pytest -q --tb=short tests/test_api_budget.py``.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from charlie_work import api_budget
from charlie_work.api_budget import (
    DayBucket,
    Ledger,
    SessionEntry,
    Usage,
    budget_status,
    cost_usd,
    ledger_path,
    load_ledger,
    save_ledger,
    settle_session,
    usage_from_events,
)
from charlie_work.claude_code import iter_claude_events, parse_claude_events
from charlie_work.config import ApiBudgetConfig, ApiProviderConfig
from charlie_work.routing import BudgetStatus


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _provider(
    *,
    input_usd_per_mtok: float = 3.0,
    output_usd_per_mtok: float = 15.0,
    cached_input_usd_per_mtok: float = 0.30,
) -> ApiProviderConfig:
    return ApiProviderConfig(
        base_url="https://api.example.com/anthropic",
        api_key_env="EXAMPLE_API_KEY",
        model="example-model",
        input_usd_per_mtok=input_usd_per_mtok,
        output_usd_per_mtok=output_usd_per_mtok,
        cached_input_usd_per_mtok=cached_input_usd_per_mtok,
    )


def _entry(
    *,
    issue: int = 42,
    session_id: str = "sess-1",
    provider: str = "example",
    model: str = "example-model",
    started_at: str = "2026-07-22T10:00:00Z",
    ended_at: str = "2026-07-22T10:30:00Z",
    input_tokens: int = 1_000_000,
    output_tokens: int = 200_000,
    cached_tokens: int = 500_000,
    usd: float = 9.0,
    duration_s: float = 1800.0,
    outcome: str = "completed",
) -> SessionEntry:
    return SessionEntry(
        issue=issue,
        session_id=session_id,
        provider=provider,
        model=model,
        started_at=started_at,
        ended_at=ended_at,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        usd=usd,
        duration_s=duration_s,
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# cost_usd — hand-computed fixtures including cached-token pricing
# ---------------------------------------------------------------------------


def test_cost_usd_hand_computed_with_cached_pricing() -> None:
    """1M input @ $3, 0.2M output @ $15, 0.5M cached @ $0.30 = 3 + 3 + 0.15 = 6.15."""
    usage = Usage(input_tokens=1_000_000, output_tokens=200_000, cached_tokens=500_000)
    assert cost_usd(usage, _provider()) == pytest.approx(6.15)


def test_cost_usd_zero_usage() -> None:
    assert cost_usd(Usage(), _provider()) == 0.0


def test_cost_usd_cached_default_zero_rate() -> None:
    """cached_input_usd_per_mtok defaults to 0.0 → cached tokens are free."""
    provider = ApiProviderConfig(
        base_url="https://api.example.com/anthropic",
        api_key_env="EXAMPLE_API_KEY",
        model="example-model",
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        cached_input_usd_per_mtok=0.0,
    )
    usage = Usage(input_tokens=1_000_000, output_tokens=0, cached_tokens=2_000_000)
    # Only input billed: 1M * 3 = 3.0; cached 2M * 0 = 0.
    assert cost_usd(usage, provider) == pytest.approx(3.0)


def test_cost_usd_small_token_volume_precision() -> None:
    """10k input, 1k output, 5k cached at default pricing."""
    usage = Usage(input_tokens=10_000, output_tokens=1_000, cached_tokens=5_000)
    expected = (10_000 / 1e6) * 3.0 + (1_000 / 1e6) * 15.0 + (5_000 / 1e6) * 0.30
    assert cost_usd(usage, _provider()) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# usage_from_events — real stream-json shape
# ---------------------------------------------------------------------------


def _assistant_event(
    msg_id: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> dict:
    return {
        "type": "assistant",
        "session_id": "sess-1",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            },
        },
    }


def _result_event(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> dict:
    return {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "total_cost_usd": 0.0,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
        },
    }


def test_usage_from_events_prefers_result_cumulative() -> None:
    """The terminal result event's cumulative usage is authoritative."""
    events = [
        _assistant_event("msg_1", input_tokens=100, output_tokens=50),
        _assistant_event("msg_2", input_tokens=200, output_tokens=80, cache_read=40),
        _result_event(input_tokens=300, output_tokens=130, cache_creation=10, cache_read=60),
    ]
    usage = usage_from_events(events)
    # input folds in cache_creation: 300 + 10 = 310
    assert usage == Usage(input_tokens=310, output_tokens=130, cached_tokens=60)


def test_usage_from_events_falls_back_to_assistant_sum_when_no_result() -> None:
    """Killed session with no result event: sum per-turn assistant usage."""
    events = [
        _assistant_event("msg_1", input_tokens=100, output_tokens=50),
        _assistant_event("msg_2", input_tokens=200, output_tokens=80, cache_read=40),
    ]
    usage = usage_from_events(events)
    assert usage == Usage(input_tokens=300, output_tokens=130, cached_tokens=40)


def test_usage_from_events_dedups_multiline_assistant_by_message_id() -> None:
    """Claude Code writes one JSONL line per content block; input/cache are
    identical across lines, output grows. Counting per-line over-counts; the
    accumulator dedups by message.id keeping the max output_tokens."""
    events = [
        # One API call (msg_1) split across two content-block lines.
        _assistant_event("msg_1", input_tokens=100, output_tokens=30),
        _assistant_event("msg_1", input_tokens=100, output_tokens=50),  # final output
        # A second API call without a message.id (falls into the no-id bucket).
        {
            "type": "assistant",
            "message": {
                "usage": {
                    "input_tokens": 40,
                    "output_tokens": 10,
                    "cache_creation_input_tokens": 5,
                    "cache_read_input_tokens": 0,
                }
            },
        },
    ]
    usage = usage_from_events(events)
    # msg_1: input 100, output 50 (max). no-id: input 40+5=45, output 10.
    assert usage == Usage(input_tokens=145, output_tokens=60, cached_tokens=0)


def test_usage_from_events_empty() -> None:
    assert usage_from_events([]) == Usage()


def test_usage_from_events_skips_malformed_and_non_dict() -> None:
    events = [
        {"type": "assistant"},  # no message.usage
        {"type": "assistant", "message": "not-a-dict"},
        {"type": "assistant", "message": {"usage": "not-a-dict"}},
        {"type": "result"},  # no usage
        {"type": "result", "usage": "not-a-dict"},
        "not-a-dict",
        _result_event(input_tokens=42, output_tokens=7),
    ]
    usage = usage_from_events(events)
    assert usage == Usage(input_tokens=42, output_tokens=7, cached_tokens=0)


def test_usage_from_events_reuses_iter_claude_events_from_file(tmp_path: Path) -> None:
    """usage_from_events consumes iter_claude_events output (no re-implemented parsing)."""
    events_file = tmp_path / "issue-1.events.jsonl"
    events_file.write_text(
        json.dumps(_assistant_event("msg_1", input_tokens=100, output_tokens=50))
        + "\n"
        + json.dumps(_result_event(input_tokens=100, output_tokens=50))
        + "\n",
        encoding="utf-8",
    )
    usage = usage_from_events(iter_claude_events(events_file))
    assert usage == Usage(input_tokens=100, output_tokens=50, cached_tokens=0)


def test_parse_claude_events_still_works_after_refactor(tmp_path: Path) -> None:
    """parse_claude_events preserves behavior after extracting iter_claude_events."""
    events_file = tmp_path / "issue-1.events.jsonl"
    events_file.write_text(
        '{"type": "tool_call", "tokens": 1000, "cost_usd": 0.01}\n'
        '{"type": "user_message", "tokens": 2000, "cost_usd": 0.02}\n'
        '{"type": "assistant_message", "tokens": 3000, "cost_usd": 0.03}\n',
        encoding="utf-8",
    )
    progress = parse_claude_events(events_file)
    assert progress is not None
    assert progress.tool_call_count == 1
    assert progress.turn_count == 2
    assert progress.tokens == 3000
    assert progress.cost_usd == 0.03


def test_parse_claude_events_missing_file_returns_none(tmp_path: Path) -> None:
    assert parse_claude_events(tmp_path / "nope.events.jsonl") is None


# ---------------------------------------------------------------------------
# budget_status — boundary conditions (exactly-at-cap is exhausted)
# ---------------------------------------------------------------------------


def test_budget_status_returns_budget_status_value() -> None:
    status = budget_status(Ledger(), ApiBudgetConfig(), "2026-07-22")
    assert isinstance(status, BudgetStatus)
    assert status.spent_today_usd == 0.0
    assert status.lifetime_spent_usd == 0.0


def test_budget_status_daily_exactly_at_cap_is_exhausted() -> None:
    """spent_today == max_usd_per_day (with a positive reserve) → no headroom."""
    budget = ApiBudgetConfig(
        max_usd_per_session=0.0,
        preflight_reserve_usd=1.0,
        max_usd_per_day=5.0,
        lifetime_usd=15.0,
    )
    ledger = Ledger(
        days={"2026-07-22": DayBucket(usd=5.0)},
        lifetime_usd=0.0,
    )
    status = budget_status(ledger, budget, "2026-07-22")
    # 5.0 + 1.0 <= 5.0 is False → exhausted.
    assert status.daily_headroom is False
    assert status.spent_today_usd == 5.0


def test_budget_status_daily_exact_fit_launch_has_headroom() -> None:
    """spent_today + reserve == max_usd_per_day → one exact-fit launch allowed."""
    budget = ApiBudgetConfig(
        max_usd_per_session=0.0,
        preflight_reserve_usd=1.0,
        max_usd_per_day=5.0,
        lifetime_usd=15.0,
    )
    ledger = Ledger(
        days={"2026-07-22": DayBucket(usd=4.0)},
        lifetime_usd=0.0,
    )
    status = budget_status(ledger, budget, "2026-07-22")
    # 4.0 + 1.0 <= 5.0 is True → headroom.
    assert status.daily_headroom is True


def test_budget_status_daily_over_cap_exhausted() -> None:
    budget = ApiBudgetConfig(preflight_reserve_usd=1.0, max_usd_per_day=5.0, lifetime_usd=15.0)
    ledger = Ledger(days={"2026-07-22": DayBucket(usd=5.5)})
    assert budget_status(ledger, budget, "2026-07-22").daily_headroom is False


def test_budget_status_daily_uses_max_usd_per_session_as_reserve_when_set() -> None:
    budget = ApiBudgetConfig(
        max_usd_per_session=2.0,
        preflight_reserve_usd=1.0,
        max_usd_per_day=5.0,
        lifetime_usd=15.0,
    )
    ledger = Ledger(days={"2026-07-22": DayBucket(usd=3.0)})
    # 3.0 + 2.0 <= 5.0 → True (exact fit with per-session reserve).
    assert budget_status(ledger, budget, "2026-07-22").daily_headroom is True
    ledger_over = Ledger(days={"2026-07-22": DayBucket(usd=3.1)})
    # 3.1 + 2.0 <= 5.0 → False.
    assert budget_status(ledger_over, budget, "2026-07-22").daily_headroom is False


def test_budget_status_lifetime_exactly_at_cap_is_exhausted() -> None:
    """lifetime_spent == lifetime_usd cap → exhausted (strict <)."""
    budget = ApiBudgetConfig(max_usd_per_day=5.0, lifetime_usd=15.0)
    ledger = Ledger(lifetime_usd=15.0)
    status = budget_status(ledger, budget, "2026-07-22")
    assert status.lifetime_headroom is False
    assert status.lifetime_spent_usd == 15.0


def test_budget_status_lifetime_below_cap_has_headroom() -> None:
    budget = ApiBudgetConfig(max_usd_per_day=5.0, lifetime_usd=15.0)
    ledger = Ledger(lifetime_usd=14.99)
    assert budget_status(ledger, budget, "2026-07-22").lifetime_headroom is True


def test_budget_status_lifetime_over_cap_exhausted() -> None:
    budget = ApiBudgetConfig(max_usd_per_day=5.0, lifetime_usd=15.0)
    ledger = Ledger(lifetime_usd=15.01)
    assert budget_status(ledger, budget, "2026-07-22").lifetime_headroom is False


def test_budget_status_uses_today_bucket_only() -> None:
    """Spend on other days does not count toward today's headroom."""
    budget = ApiBudgetConfig(preflight_reserve_usd=1.0, max_usd_per_day=5.0, lifetime_usd=15.0)
    ledger = Ledger(
        days={
            "2026-07-21": DayBucket(usd=5.0),  # yesterday at cap
            "2026-07-22": DayBucket(usd=0.0),  # today fresh
        }
    )
    status = budget_status(ledger, budget, "2026-07-22")
    assert status.spent_today_usd == 0.0
    assert status.daily_headroom is True


# ---------------------------------------------------------------------------
# settle_session — idempotence and day bucketing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Persistence — atomicity and corrupt recovery
# ---------------------------------------------------------------------------


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    entry = _entry()
    ledger = settle_session(Ledger(), entry)
    path = ledger_path(tmp_path)
    save_ledger(path, ledger)
    loaded = load_ledger(path)
    assert loaded.lifetime_usd == pytest.approx(ledger.lifetime_usd)
    assert dict(loaded.days) == dict(ledger.days)
    assert len(loaded.sessions) == 1
    assert loaded.sessions[0] == entry


def test_load_ledger_missing_file_returns_empty(tmp_path: Path) -> None:
    ledger = load_ledger(ledger_path(tmp_path))
    assert ledger == Ledger()


def test_load_ledger_corrupt_file_preserves_original_and_recovers_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    with caplog.at_level("ERROR", logger="charlie_work.api_budget"):
        ledger = load_ledger(path)
    assert ledger == Ledger()
    # The corrupt original must be preserved on disk (moved aside, not destroyed).
    remaining = list(tmp_path.glob("api-budget.json.corrupt-*"))
    assert len(remaining) == 1
    assert remaining[0].read_text(encoding="utf-8") == "{not valid json"
    # The original path no longer holds the corrupt content.
    assert not path.exists()
    assert any("unrecoverable" in rec.message for rec in caplog.records)


def test_load_ledger_partial_json_preserves_original(tmp_path: Path) -> None:
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"days": {"2026-07-22": {', encoding="utf-8")  # truncated
    ledger = load_ledger(path)
    assert ledger == Ledger()
    assert list(tmp_path.glob("api-budget.json.corrupt-*"))


def test_load_ledger_wrong_typed_lifetime_usd_quarantines(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Valid JSON with a wrong-typed (non-numeric) ``lifetime_usd`` is structural
    corruption: ``_ledger_from_dict``'s ``float()`` coercion raises, and the
    file must go through the quarantine path (forensic log + preserved original)
    rather than propagating uncaught and wedging every future settlement.

    Regression guard for the review-#540 finding that ``load_ledger``'s guard
    only wrapped ``json.load`` and left ``_ledger_from_dict`` outside it.
    """
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = '{"lifetime_usd": "not-a-number", "days": {}, "sessions": []}'
    path.write_text(original, encoding="utf-8")
    with caplog.at_level("ERROR", logger="charlie_work.api_budget"):
        ledger = load_ledger(path)
    assert ledger == Ledger()
    # Quarantined — original preserved on disk for forensics.
    remaining = list(tmp_path.glob("api-budget.json.corrupt-*"))
    assert len(remaining) == 1
    assert remaining[0].read_text(encoding="utf-8") == original
    assert not path.exists()
    assert any("unrecoverable" in rec.message for rec in caplog.records)


def test_load_ledger_wrong_typed_day_bucket_usd_quarantines(tmp_path: Path) -> None:
    """A wrong-typed ``usd`` inside a day bucket is the same structural
    corruption class and must also quarantine (not wedge)."""
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"lifetime_usd": 1.0, "days": {"2026-07-22": {"usd": ["bad"]}}, "sessions": []}',
        encoding="utf-8",
    )
    ledger = load_ledger(path)
    assert ledger == Ledger()
    assert list(tmp_path.glob("api-budget.json.corrupt-*"))


def test_load_ledger_wrong_typed_session_usd_drops_entry_leniently(tmp_path: Path) -> None:
    """A wrong-typed ``usd`` on a session entry is caught by the per-entry
    try/except in ``_ledger_from_dict`` (which guards this exact failure class
    by dropping the bad entry, per the review-#540 note) — so it does NOT
    quarantine; the entry is dropped and the rest of the ledger loads.

    This documents the intentional split: per-entry malformation = drop the
    entry; structural (top-level / day-bucket) malformation = quarantine.
    """
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"lifetime_usd": 0.0, "days": {}, "sessions": ['
        '{"issue": 1, "session_id": "s", "usd": {"x": 1}}]}',
        encoding="utf-8",
    )
    ledger = load_ledger(path)
    # Bad entry dropped leniently; no quarantine (the file is otherwise valid).
    assert ledger == Ledger()
    assert not list(tmp_path.glob("api-budget.json.corrupt-*"))


def test_load_ledger_lenient_on_missing_keys(tmp_path: Path) -> None:
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"version": 1}', encoding="utf-8")  # valid JSON, no ledger keys
    ledger = load_ledger(path)
    assert ledger == Ledger()


def test_save_ledger_uses_atomic_temp_replace(tmp_path: Path) -> None:
    """Atomicity: the write goes through temp + replace. After save, no .tmp
    leftover remains and the file is valid JSON (a concurrent reader never
    observes a half-written file)."""
    path = ledger_path(tmp_path)
    save_ledger(path, settle_session(Ledger(), _entry()))
    # No leftover temp file.
    assert not (tmp_path / "api-budget.json.tmp").exists()
    # The file is valid, parseable JSON.
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "days" in data
    assert "lifetime_usd" in data
    assert "sessions" in data


def test_save_ledger_no_plain_open_write_on_ledger_path(tmp_path: Path) -> None:
    """Invariant: the ledger path is never written with a plain open(path, 'w').

    Inspect the module source to assert the atomic temp+replace pattern is used
    and no bare ``open(<ledger path>, "w")`` exists for the ledger file itself.
    """
    src = Path(api_budget.__file__).read_text(encoding="utf-8")
    # save_ledger opens the TMP path for writing, then replaces — never the
    # ledger path directly. Assert the canonical pattern is present.
    assert 'tmp_path = path.with_suffix(path.suffix + ".tmp")' in src
    assert "tmp_path.replace(path)" in src
    # No plain open(path, "w") on the ledger path itself.
    assert 'open(path, "w")' not in src
    assert "open(path, 'w')" not in src


def test_ledger_to_dict_and_schema(tmp_path: Path) -> None:
    entry = _entry()
    ledger = settle_session(Ledger(), entry)
    data = api_budget.ledger_to_dict(ledger)
    assert isinstance(data["days"], dict)
    day = data["days"]["2026-07-22"]
    assert set(day.keys()) == {"input_tokens", "output_tokens", "cached_tokens", "usd"}
    assert isinstance(data["sessions"], list)
    sess = data["sessions"][0]
    assert set(sess.keys()) == {
        "issue",
        "session_id",
        "provider",
        "model",
        "started_at",
        "ended_at",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "usd",
        "duration_s",
        "outcome",
    }


# ---------------------------------------------------------------------------
# Reap-path settlement wiring (WorkerView.reap_sidecar for adapter_kind == "api")
# ---------------------------------------------------------------------------


def _write_api_sidecar(sessions_dir: Path, issue_number: int, provider: str) -> Path:
    from charlie_work.claude_code import _sidecar_path

    sidecar = _sidecar_path(sessions_dir, issue_number, "api")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": issue_number,
                "branch": "agent/x",
                "worktree_path": "",
                "prompt_path": "",
                "command": [],
                "pid": None,
                "started_at": "2026-07-22T10:00:00Z",
                "log_path": str(sessions_dir / f"issue-{issue_number}.claude.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": None,
                "reclaimed": None,
                "adapter_kind": "api",
                "provider": provider,
                "session_id": "sess-1",
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def _write_events(sessions_dir: Path, issue_number: int, events: list[dict]) -> Path:
    events_path = sessions_dir / f"issue-{issue_number}.events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return events_path


def _api_worker_view(sessions_dir: Path, issue_number: int, provider: str) -> "object":
    from charlie_work.worker import WorkerView

    return WorkerView(
        adapter_kind="api",
        issue_number=issue_number,
        repo_key="",
        pid=None,
        started_at="2026-07-22T10:00:00Z",
        process_start_time=None,
        log_path=str(sessions_dir / f"issue-{issue_number}.claude.log"),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
        session_id="sess-1",
    )


def test_reap_sidecar_settles_api_session_into_ledger(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    _write_api_sidecar(sessions_dir, 42, "example")
    _write_events(
        sessions_dir,
        42,
        [_result_event(input_tokens=1_000_000, output_tokens=200_000, cache_read=500_000)],
    )
    provider = _provider()
    api_config = type("C", (), {"providers": {"example": provider}})()
    view = _api_worker_view(sessions_dir, 42, "example")

    view.reap_sidecar(sessions_dir, api_config=api_config, state_dir=state_dir)

    ledger = load_ledger(ledger_path(state_dir))
    assert len(ledger.sessions) == 1
    entry = ledger.sessions[0]
    assert entry.issue == 42
    assert entry.provider == "example"
    assert entry.model == "example-model"
    assert entry.input_tokens == 1_000_000
    assert entry.output_tokens == 200_000
    assert entry.cached_tokens == 500_000
    # 1M*3 + 0.2M*15 + 0.5M*0.30 = 3 + 3 + 0.15 = 6.15
    assert entry.usd == pytest.approx(6.15)
    assert ledger.lifetime_usd == pytest.approx(6.15)
    # The sidecar was unlinked (reap still happens after settlement).
    from charlie_work.claude_code import _sidecar_path

    assert not _sidecar_path(sessions_dir, 42, "api").exists()


def test_reap_sidecar_settlement_is_idempotent_across_reaps(tmp_path: Path) -> None:
    """Settling the same session on two reaps does not double-count (idempotence
    via the ledger's (issue, started_at, session_id) key)."""
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    # First reap settles + unlinks the sidecar.
    _write_api_sidecar(sessions_dir, 42, "example")
    _write_events(sessions_dir, 42, [_result_event(input_tokens=1_000_000, output_tokens=0)])
    provider = _provider()
    api_config = type("C", (), {"providers": {"example": provider}})()
    view = _api_worker_view(sessions_dir, 42, "example")
    view.reap_sidecar(sessions_dir, api_config=api_config, state_dir=state_dir)
    # Re-create the sidecar (e.g. a second reap cycle) and reap again.
    _write_api_sidecar(sessions_dir, 42, "example")
    view.reap_sidecar(sessions_dir, api_config=api_config, state_dir=state_dir)

    ledger = load_ledger(ledger_path(state_dir))
    assert len(ledger.sessions) == 1
    assert ledger.lifetime_usd == pytest.approx(3.0)  # 1M * 3, counted once


def test_reap_sidecar_without_api_config_skips_settlement(tmp_path: Path) -> None:
    """Legacy callers (no api_config/state_dir) still reap; no ledger written."""
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    _write_api_sidecar(sessions_dir, 42, "example")
    _write_events(sessions_dir, 42, [_result_event(input_tokens=1_000_000, output_tokens=0)])
    view = _api_worker_view(sessions_dir, 42, "example")

    view.reap_sidecar(sessions_dir)  # no kwargs

    assert not ledger_path(state_dir).exists()
    from charlie_work.claude_code import _sidecar_path

    assert not _sidecar_path(sessions_dir, 42, "api").exists()


def test_reap_sidecar_unknown_provider_skips_settlement(tmp_path: Path) -> None:
    """A provider not in the registry → no pricing → skip settlement, still reap."""
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    _write_api_sidecar(sessions_dir, 42, "ghost")
    _write_events(sessions_dir, 42, [_result_event(input_tokens=1_000_000, output_tokens=0)])
    provider = _provider()
    api_config = type("C", (), {"providers": {"example": provider}})()
    view = _api_worker_view(sessions_dir, 42, "ghost")

    view.reap_sidecar(sessions_dir, api_config=api_config, state_dir=state_dir)

    assert not ledger_path(state_dir).exists()
    from charlie_work.claude_code import _sidecar_path

    assert not _sidecar_path(sessions_dir, 42, "api").exists()


def test_reap_sidecar_settlement_failure_does_not_break_reap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If settlement raises, the sidecar is still unlinked (best-effort accounting)."""
    sessions_dir = tmp_path / "sessions"
    _write_api_sidecar(sessions_dir, 42, "example")
    _write_events(sessions_dir, 42, [_result_event(input_tokens=1_000_000, output_tokens=0)])
    provider = _provider()
    api_config = type("C", (), {"providers": {"example": provider}})()
    view = _api_worker_view(sessions_dir, 42, "example")
    # Make state_dir non-writable-ish by pointing it at a file path so save_ledger
    # raises (parent is a file, not a directory).
    state_dir_bad = tmp_path / "blocker"
    state_dir_bad.write_text("i am a file, not a dir", encoding="utf-8")

    with caplog.at_level("WARNING", logger="charlie_work.worker"):
        view.reap_sidecar(sessions_dir, api_config=api_config, state_dir=state_dir_bad)

    from charlie_work.claude_code import _sidecar_path

    assert not _sidecar_path(sessions_dir, 42, "api").exists()
    assert any("settlement failed" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Value-type invariants (frozen)
# ---------------------------------------------------------------------------


def test_value_types_are_frozen() -> None:
    for obj in (Usage(), DayBucket(), _entry(), Ledger()):
        # Frozen dataclasses raise FrozenInstanceError (an AttributeError
        # subclass) on attribute assignment.
        with pytest.raises(AttributeError):
            obj.input_tokens = 999  # type: ignore[misc]
            obj.usd = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# settle_session_to_disk — locked read-modify-write (review #540 rework)
#
# The blocking review finding: settle_session's load-modify-save cycle on
# api-budget.json had no locking, unlike state_lock for state.json — concurrent
# reaps of different api sessions could silently lose a settlement (lost-update
# race). settle_session_to_disk wraps the RMW in state.advisory_file_lock.
# ---------------------------------------------------------------------------


def test_settle_session_to_disk_writes_entry(tmp_path: Path) -> None:
    path = ledger_path(tmp_path)
    assert api_budget.settle_session_to_disk(path, _entry(usd=6.15)) is True
    ledger = load_ledger(path)
    assert len(ledger.sessions) == 1
    assert ledger.lifetime_usd == pytest.approx(6.15)


def test_settle_session_to_disk_idempotent(tmp_path: Path) -> None:
    path = ledger_path(tmp_path)
    api_budget.settle_session_to_disk(path, _entry(usd=3.0))
    # Same identity → no-op, still returns True.
    assert api_budget.settle_session_to_disk(path, _entry(usd=3.0)) is True
    ledger = load_ledger(path)
    assert len(ledger.sessions) == 1
    assert ledger.lifetime_usd == pytest.approx(3.0)


def test_settle_session_to_disk_concurrent_distinct_sessions_no_lost_update(
    tmp_path: Path,
) -> None:
    """Two threads settling DISTINCT sessions concurrently must both land in the
    ledger — the per-path threading.Lock in advisory_file_lock serializes the
    load→settle→save critical sections so no settlement is lost to a
    read-modify-write race (the review's blocking finding).
    """
    import threading

    path = ledger_path(tmp_path)
    entries = [
        _entry(issue=100, session_id="sess-a", usd=1.0, started_at="2026-07-22T10:00:00Z"),
        _entry(issue=101, session_id="sess-b", usd=2.0, started_at="2026-07-22T10:00:00Z"),
        _entry(issue=102, session_id="sess-c", usd=3.0, started_at="2026-07-22T10:00:00Z"),
        _entry(issue=103, session_id="sess-d", usd=4.0, started_at="2026-07-22T10:00:00Z"),
    ]
    errors: list[BaseException] = []

    def settle(entry: SessionEntry) -> None:
        try:
            api_budget.settle_session_to_disk(path, entry)
        except BaseException as exc:  # noqa: BLE001 — surface any thread failure
            errors.append(exc)

    threads = [threading.Thread(target=settle, args=(e,)) for e in entries]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    ledger = load_ledger(path)
    settled_ids = {e.session_id for e in ledger.sessions}
    assert settled_ids == {"sess-a", "sess-b", "sess-c", "sess-d"}
    # No lost update: all four settlements accumulated.
    assert ledger.lifetime_usd == pytest.approx(10.0)


def test_settle_session_to_disk_skips_when_lock_busy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch
) -> None:
    """Fail-as-a-value: if the advisory lock cannot be acquired
    (StateLockBusy), settlement is SKIPPED and logged — never written unlocked.
    """
    from charlie_work import state as state_module
    from charlie_work.state import StateLockBusy

    path = ledger_path(tmp_path)

    def _always_busy(path_arg: Path):  # noqa: ANN001
        raise StateLockBusy("simulated contention")

    monkeypatch.setattr(state_module, "advisory_file_lock", _always_busy)

    with caplog.at_level("WARNING", logger="charlie_work.api_budget"):
        result = api_budget.settle_session_to_disk(path, _entry(usd=5.0))

    assert result is False
    # Ledger was never written.
    assert not path.exists()
    assert any("lock busy" in rec.message for rec in caplog.records)


def test_settle_session_to_disk_uses_advisory_lock_not_state_lock(
    tmp_path: Path, monkeypatch
) -> None:
    """The locked RMW must go through state.advisory_file_lock (the generic
    primitive), not the state.json-specific state_lock wrapper. Asserting the
    generic primitive is the one exercised prevents a regression that re-routes
    settlement through an unrelated lock path.
    """
    from charlie_work import state as state_module

    path = ledger_path(tmp_path)
    calls: list[Path] = []
    real_lock = state_module.advisory_file_lock

    def tracking_lock(p: Path):
        calls.append(p)
        return real_lock(p)

    monkeypatch.setattr(state_module, "advisory_file_lock", tracking_lock)

    api_budget.settle_session_to_disk(path, _entry(usd=1.0))

    assert calls == [path]
    assert load_ledger(path).lifetime_usd == pytest.approx(1.0)


def test_reap_sidecar_settlement_uses_locked_settle_to_disk(tmp_path: Path, monkeypatch) -> None:
    """The reap wiring routes the on-disk RMW through settle_session_to_disk
    (locked), not the unlocked load_ledger/settle_session/save_ledger trio.
    Patches settle_session_to_disk and asserts it is called with the ledger path.
    """
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    _write_api_sidecar(sessions_dir, 42, "example")
    _write_events(sessions_dir, 42, [_result_event(input_tokens=1_000_000, output_tokens=0)])
    provider = _provider()
    api_config = type("C", (), {"providers": {"example": provider}})()
    view = _api_worker_view(sessions_dir, 42, "example")

    captured: list = []
    real = api_budget.settle_session_to_disk

    def spy(path: Path, entry: SessionEntry) -> bool:
        captured.append((path, entry))
        return real(path, entry)

    monkeypatch.setattr(api_budget, "settle_session_to_disk", spy)

    view.reap_sidecar(sessions_dir, api_config=api_config, state_dir=state_dir)

    assert len(captured) == 1
    assert captured[0][0] == ledger_path(state_dir)
    assert captured[0][1].issue == 42


# ---------------------------------------------------------------------------
# budget_status — reserve=0 daily-boundary characterization (minor finding)
# ---------------------------------------------------------------------------


def test_budget_status_daily_reserve_zero_at_cap_allows_headroom() -> None:
    """Minor finding characterization: when the operator sets BOTH
    max_usd_per_session=0 AND preflight_reserve_usd=0 (calibration mode with no
    headroom reserve), the daily check degenerates to ``spent_today <= max``
    and spend exactly at the cap reports headroom True.

    This is intentional and consistent with the issue's explicit formula
    ``spent_today + reserve <= max_usd_per_day``: with reserve=0 the check
    answers "have I already EXCEEDED the cap?" rather than "can I afford the
    next launch?" A 0-reserve launch is a no-cost launch, so allowing it at-cap
    is the defensible behavior. Disclosed here so the boundary is documented
    rather than latent.
    """
    budget = ApiBudgetConfig(
        max_usd_per_session=0.0,
        preflight_reserve_usd=0.0,
        max_usd_per_day=5.0,
        lifetime_usd=15.0,
    )
    ledger = Ledger(days={"2026-07-22": DayBucket(usd=5.0)})  # exactly at cap
    status = budget_status(ledger, budget, "2026-07-22")
    # 5.0 + 0.0 <= 5.0 → True (at-cap allowed with zero reserve).
    assert status.daily_headroom is True
    # Over-cap is still exhausted.
    over = Ledger(days={"2026-07-22": DayBucket(usd=5.01)})
    assert budget_status(over, budget, "2026-07-22").daily_headroom is False


# ---------------------------------------------------------------------------
# advisory_file_lock / state_lock — generic primitive regression
# ---------------------------------------------------------------------------


def test_advisory_file_lock_serializes_concurrent_threads_same_path(
    tmp_path: Path,
) -> None:
    """advisory_file_lock (the generic primitive extracted from state_lock)
    must serialize concurrent threads in this process on the same path — the
    per-path threading.Lock is what closes the intra-process lost-update race
    that file locks alone cannot (byte-range locks are owned by the process,
    not the thread).
    """
    import threading

    from charlie_work.state import advisory_file_lock

    target = tmp_path / "target.json"
    target.write_text("0", encoding="utf-8")

    # Each thread reads the int, increments, writes back — under the lock.
    # Without intra-process serialization the increments would collide.
    def bump() -> None:
        for _ in range(50):
            with advisory_file_lock(target):
                val = int(target.read_text(encoding="utf-8"))
                target.write_text(str(val + 1), encoding="utf-8")

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert target.read_text(encoding="utf-8") == "200"  # 4 threads * 50 increments


def test_state_lock_delegates_to_advisory_file_lock(tmp_path: Path, monkeypatch) -> None:
    """state_lock is now a thin wrapper over advisory_file_lock. Assert the
    delegation directly: monkeypatching the generic primitive is observed by a
    state_lock caller (regression for the extraction — keeps the two names
    bound to one mechanism, not two divergent lock implementations).
    """
    import charlie_work.state as state_module
    from charlie_work.state import state_lock

    state_path = tmp_path / "state.json"
    calls: list[Path] = []
    real_lock = state_module.advisory_file_lock

    def tracking_lock(p: Path):
        calls.append(p)

        @contextmanager
        def _cm():
            with real_lock(p):
                yield

        return _cm()

    monkeypatch.setattr(state_module, "advisory_file_lock", tracking_lock)

    with state_lock(state_path):
        pass

    assert calls == [state_path]
