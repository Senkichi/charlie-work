"""Tests for the per-issue adapter routing policy (issue #477).

Exhaustive rule-matrix coverage: every preflight failure reason, rework vs
first-pass, enabled/disabled, determinism, and the immutable
``record_adapter_choice`` state helper.
"""

from __future__ import annotations

from types import MappingProxyType

from charlie_work.config import ApiProviderConfig, ApiWorkerConfig
from charlie_work.routing import (
    AdapterChoice,
    BudgetStatus,
    record_adapter_choice,
    select_adapter,
)

PROVIDER = ApiProviderConfig(
    base_url="https://api.example.test/anthropic",
    api_key_env="EXAMPLE_API_KEY",
    model="example-model",
    input_usd_per_mtok=3.0,
    output_usd_per_mtok=15.0,
    cached_input_usd_per_mtok=0.30,
)


def _api_config(
    *,
    enabled: bool = True,
    provider: str = "example",
    max_concurrent_sessions: int = 1,
    fallback_adapter: str = "devin-shell",
    providers: dict[str, ApiProviderConfig] | None = None,
) -> ApiWorkerConfig:
    """Build an enabled-by-default ApiWorkerConfig for tests."""
    if providers is None:
        providers = {provider: PROVIDER}
    return ApiWorkerConfig(
        enabled=enabled,
        provider=provider,
        max_concurrent_sessions=max_concurrent_sessions,
        providers=MappingProxyType(providers),
        fallback_adapter=fallback_adapter,
    )


def _budget_ok() -> BudgetStatus:
    return BudgetStatus(
        spent_today_usd=0.0,
        lifetime_spent_usd=0.0,
        daily_headroom=True,
        lifetime_headroom=True,
    )


# ---------------------------------------------------------------------------
# select_adapter — first-pass (rework=False) always routes to default
# ---------------------------------------------------------------------------


def test_first_pass_routes_to_default() -> None:
    choice = select_adapter(
        rework=False,
        issue_labels=frozenset(),
        api_config=_api_config(),
        budget=_budget_ok(),
        api_key_present=True,
        provider_in_cooldown=False,
        live_api_sessions=0,
        default_adapter="devin-shell",
    )
    assert choice == AdapterChoice("devin-shell", "", "policy:default")


def test_first_pass_ignores_api_state() -> None:
    """First-pass default routing is independent of api preflight inputs."""
    choice = select_adapter(
        rework=False,
        issue_labels=frozenset(),
        api_config=_api_config(enabled=False),
        budget=BudgetStatus(0.0, 0.0, False, False),
        api_key_present=False,
        provider_in_cooldown=True,
        live_api_sessions=99,
        default_adapter="claude-code",
    )
    assert choice == AdapterChoice("claude-code", "", "policy:default")


def test_first_pass_with_labels_still_default() -> None:
    """No complexity rule exists yet; labels do not affect first-pass routing."""
    choice = select_adapter(
        rework=False,
        issue_labels=frozenset({"complexity:high", "agent:queued"}),
        api_config=_api_config(),
        budget=_budget_ok(),
        api_key_present=True,
        provider_in_cooldown=False,
        live_api_sessions=0,
        default_adapter="devin-shell",
    )
    assert choice == AdapterChoice("devin-shell", "", "policy:default")


# ---------------------------------------------------------------------------
# select_adapter — rework routes to api when preflight passes
# ---------------------------------------------------------------------------


def test_rework_routes_to_api_when_preflight_passes() -> None:
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=_api_config(provider="example"),
        budget=_budget_ok(),
        api_key_present=True,
        provider_in_cooldown=False,
        live_api_sessions=0,
        default_adapter="devin-shell",
    )
    assert choice == AdapterChoice("api", "example", "policy:rework")


def test_rework_api_with_concurrency_at_limit_minus_one() -> None:
    """live_api_sessions strictly less than max -> still passes."""
    config = _api_config(max_concurrent_sessions=3)
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=config,
        budget=_budget_ok(),
        api_key_present=True,
        provider_in_cooldown=False,
        live_api_sessions=2,
        default_adapter="devin-shell",
    )
    assert choice == AdapterChoice("api", "example", "policy:rework")


# ---------------------------------------------------------------------------
# select_adapter — preflight failure matrix (rework=True, first failing wins)
# ---------------------------------------------------------------------------


def test_preflight_disabled_fallback() -> None:
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=_api_config(enabled=False),
        budget=_budget_ok(),
        api_key_present=True,
        provider_in_cooldown=False,
        live_api_sessions=0,
        default_adapter="devin-shell",
    )
    assert choice == AdapterChoice("devin-shell", "", "fallback:disabled")


def test_preflight_auth_fallback() -> None:
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=_api_config(),
        budget=_budget_ok(),
        api_key_present=False,
        provider_in_cooldown=False,
        live_api_sessions=0,
        default_adapter="devin-shell",
    )
    assert choice == AdapterChoice("devin-shell", "", "fallback:auth")


def test_preflight_budget_daily_fallback() -> None:
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=_api_config(),
        budget=BudgetStatus(5.0, 0.0, daily_headroom=False, lifetime_headroom=True),
        api_key_present=True,
        provider_in_cooldown=False,
        live_api_sessions=0,
        default_adapter="devin-shell",
    )
    assert choice == AdapterChoice("devin-shell", "", "fallback:budget")


def test_preflight_budget_lifetime_fallback() -> None:
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=_api_config(),
        budget=BudgetStatus(0.0, 15.0, daily_headroom=True, lifetime_headroom=False),
        api_key_present=True,
        provider_in_cooldown=False,
        live_api_sessions=0,
        default_adapter="devin-shell",
    )
    assert choice == AdapterChoice("devin-shell", "", "fallback:budget")


def test_preflight_cooldown_fallback() -> None:
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=_api_config(),
        budget=_budget_ok(),
        api_key_present=True,
        provider_in_cooldown=True,
        live_api_sessions=0,
        default_adapter="devin-shell",
    )
    assert choice == AdapterChoice("devin-shell", "", "fallback:cooldown")


def test_preflight_concurrency_fallback() -> None:
    config = _api_config(max_concurrent_sessions=1)
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=config,
        budget=_budget_ok(),
        api_key_present=True,
        provider_in_cooldown=False,
        live_api_sessions=1,
        default_adapter="devin-shell",
    )
    assert choice == AdapterChoice("devin-shell", "", "fallback:concurrency")


def test_preflight_concurrency_at_exactly_max_fallback() -> None:
    """live_api_sessions == max_concurrent_sessions -> fallback (>= not >)."""
    config = _api_config(max_concurrent_sessions=3)
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=config,
        budget=_budget_ok(),
        api_key_present=True,
        provider_in_cooldown=False,
        live_api_sessions=3,
        default_adapter="devin-shell",
    )
    assert choice == AdapterChoice("devin-shell", "", "fallback:concurrency")


# ---------------------------------------------------------------------------
# First-failing-check-wins ordering
# ---------------------------------------------------------------------------


def test_disabled_wins_over_auth() -> None:
    """disabled is checked before auth."""
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=_api_config(enabled=False),
        budget=_budget_ok(),
        api_key_present=False,
        provider_in_cooldown=True,
        live_api_sessions=99,
        default_adapter="devin-shell",
    )
    assert choice.reason == "fallback:disabled"


def test_auth_wins_over_budget() -> None:
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=_api_config(),
        budget=BudgetStatus(5.0, 15.0, False, False),
        api_key_present=False,
        provider_in_cooldown=True,
        live_api_sessions=99,
        default_adapter="devin-shell",
    )
    assert choice.reason == "fallback:auth"


def test_budget_wins_over_cooldown() -> None:
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=_api_config(),
        budget=BudgetStatus(5.0, 15.0, False, False),
        api_key_present=True,
        provider_in_cooldown=True,
        live_api_sessions=99,
        default_adapter="devin-shell",
    )
    assert choice.reason == "fallback:budget"


def test_cooldown_wins_over_concurrency() -> None:
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=_api_config(max_concurrent_sessions=1),
        budget=_budget_ok(),
        api_key_present=True,
        provider_in_cooldown=True,
        live_api_sessions=99,
        default_adapter="devin-shell",
    )
    assert choice.reason == "fallback:cooldown"


# ---------------------------------------------------------------------------
# Fallback adapter is configurable
# ---------------------------------------------------------------------------


def test_fallback_adapter_is_configurable() -> None:
    config = _api_config(fallback_adapter="claude-code")
    choice = select_adapter(
        rework=True,
        issue_labels=frozenset(),
        api_config=config,
        budget=_budget_ok(),
        api_key_present=False,
        provider_in_cooldown=False,
        live_api_sessions=0,
        default_adapter="devin-shell",
    )
    assert choice == AdapterChoice("claude-code", "", "fallback:auth")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_same_inputs_same_choice() -> None:
    kwargs = dict(
        rework=True,
        issue_labels=frozenset({"agent:queued"}),
        api_config=_api_config(),
        budget=_budget_ok(),
        api_key_present=True,
        provider_in_cooldown=False,
        live_api_sessions=0,
        default_adapter="devin-shell",
    )
    assert select_adapter(**kwargs) == select_adapter(**kwargs)


def test_determinism_fallback_path() -> None:
    kwargs = dict(
        rework=True,
        issue_labels=frozenset(),
        api_config=_api_config(),
        budget=BudgetStatus(5.0, 0.0, False, True),
        api_key_present=True,
        provider_in_cooldown=False,
        live_api_sessions=0,
        default_adapter="devin-shell",
    )
    assert select_adapter(**kwargs) == select_adapter(**kwargs)


# ---------------------------------------------------------------------------
# issue_labels accepts set and frozenset
# ---------------------------------------------------------------------------


def test_issue_labels_accepts_plain_set() -> None:
    choice = select_adapter(
        rework=False,
        issue_labels={"agent:queued"},
        api_config=_api_config(),
        budget=_budget_ok(),
        api_key_present=True,
        provider_in_cooldown=False,
        live_api_sessions=0,
        default_adapter="devin-shell",
    )
    assert choice == AdapterChoice("devin-shell", "", "policy:default")


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------


def test_adapter_choice_is_frozen() -> None:
    choice = AdapterChoice("api", "example", "policy:rework")
    try:
        choice.kind = "devin-shell"  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("AdapterChoice must be frozen")


def test_budget_status_is_frozen() -> None:
    budget = _budget_ok()
    try:
        budget.daily_headroom = False  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("BudgetStatus must be frozen")


# ---------------------------------------------------------------------------
# record_adapter_choice — immutability and append semantics
# ---------------------------------------------------------------------------


def test_record_adapter_choice_appends_to_empty_state() -> None:
    state: dict = {"issues": {}}
    choice = AdapterChoice("api", "example", "policy:rework")
    new_state = record_adapter_choice(state, 42, choice, "2026-07-22T00:00:00Z")
    # Original state is untouched.
    assert state == {"issues": {}}
    history = new_state["issues"]["42"]["adapter_history"]
    assert history == [
        {
            "ts": "2026-07-22T00:00:00Z",
            "kind": "api",
            "provider": "example",
            "reason": "policy:rework",
        }
    ]


def test_record_adapter_choice_creates_issue_entry_if_absent() -> None:
    state: dict = {"issues": {"7": {"number": 7}}}
    choice = AdapterChoice("devin-shell", "", "policy:default")
    new_state = record_adapter_choice(state, 99, choice, "ts-1")
    assert "99" in new_state["issues"]
    assert new_state["issues"]["99"]["adapter_history"] == [
        {"ts": "ts-1", "kind": "devin-shell", "provider": "", "reason": "policy:default"}
    ]
    # Existing issue 7 is preserved.
    assert new_state["issues"]["7"] == {"number": 7}


def test_record_adapter_choice_appends_to_existing_history() -> None:
    state: dict = {
        "issues": {
            "5": {
                "number": 5,
                "adapter_history": [
                    {
                        "ts": "ts-0",
                        "kind": "devin-shell",
                        "provider": "",
                        "reason": "policy:default",
                    }
                ],
            }
        }
    }
    choice = AdapterChoice("api", "example", "policy:rework")
    new_state = record_adapter_choice(state, 5, choice, "ts-1")
    assert new_state["issues"]["5"]["adapter_history"] == [
        {"ts": "ts-0", "kind": "devin-shell", "provider": "", "reason": "policy:default"},
        {"ts": "ts-1", "kind": "api", "provider": "example", "reason": "policy:rework"},
    ]


def test_record_adapter_choice_does_not_mutate_input() -> None:
    state: dict = {
        "issues": {
            "3": {
                "number": 3,
                "adapter_history": [
                    {
                        "ts": "ts-0",
                        "kind": "devin-shell",
                        "provider": "",
                        "reason": "policy:default",
                    }
                ],
            }
        }
    }
    original_history = state["issues"]["3"]["adapter_history"]
    choice = AdapterChoice("api", "example", "policy:rework")
    record_adapter_choice(state, 3, choice, "ts-1")
    # The input dict, its issues sub-dict, the issue entry, and the history
    # list are all unchanged.
    assert state["issues"]["3"]["adapter_history"] is original_history
    assert len(state["issues"]["3"]["adapter_history"]) == 1


def test_record_adapter_choice_preserves_unknown_existing_keys() -> None:
    state: dict = {
        "issues": {
            "11": {
                "number": 11,
                "status": "in-progress",
                "operator_claimed_at": "ts-claim",
                "adapter_history": [
                    {
                        "ts": "ts-0",
                        "kind": "devin-shell",
                        "provider": "",
                        "reason": "policy:default",
                    }
                ],
            }
        }
    }
    choice = AdapterChoice("api", "example", "policy:rework")
    new_state = record_adapter_choice(state, 11, choice, "ts-1")
    entry = new_state["issues"]["11"]
    assert entry["number"] == 11
    assert entry["status"] == "in-progress"
    assert entry["operator_claimed_at"] == "ts-claim"
    assert len(entry["adapter_history"]) == 2


def test_record_adapter_choice_preserves_top_level_state_keys() -> None:
    state: dict = {
        "version": 1,
        "issues": {},
        "prs": {"100": {"number": 100}},
        "events": [{"at": "ts", "kind": "k", "payload": {}}],
    }
    choice = AdapterChoice("devin-shell", "", "policy:default")
    new_state = record_adapter_choice(state, 1, choice, "ts-1")
    assert new_state["version"] == 1
    assert new_state["prs"] == {"100": {"number": 100}}
    assert new_state["events"] == [{"at": "ts", "kind": "k", "payload": {}}]
    assert new_state["issues"]["1"]["adapter_history"] == [
        {"ts": "ts-1", "kind": "devin-shell", "provider": "", "reason": "policy:default"}
    ]
    # Original state untouched.
    assert state["issues"] == {}


def test_record_adapter_choice_handles_missing_issues_key() -> None:
    state: dict = {"version": 1}
    choice = AdapterChoice("api", "example", "policy:rework")
    new_state = record_adapter_choice(state, 1, choice, "ts-1")
    assert new_state["issues"]["1"]["adapter_history"] == [
        {"ts": "ts-1", "kind": "api", "provider": "example", "reason": "policy:rework"}
    ]
    assert state == {"version": 1}
