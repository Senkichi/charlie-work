"""Tests for the api worker spend ledger (issue #480).

Pure-function coverage: cost math (including cached-token pricing), settlement
idempotence, corrupt-file recovery, atomicity, UTC day bucketing, and
budget_status boundary conditions. Plus the reap-path settlement wiring into
``WorkerView.reap_sidecar`` for ``adapter_kind == "api"``.

Run targeted: ``uv run --extra dev pytest -q --tb=short tests/test_api_budget.py``.
"""

from __future__ import annotations

import json
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
