"""Per-issue adapter selection policy and adapter-history state recording.

This module is the single point of enforcement for which worker adapter an
issue is routed to. It is intentionally pure: ``select_adapter`` takes every
input as an argument and touches no filesystem, GitHub, env, or clock. The
``BudgetStatus`` value type is consumed here and produced elsewhere (the spend
ledger is a separate workstream). ``record_adapter_choice`` is an immutable
state helper — it returns a new state dict and performs no I/O; persistence
goes through the existing ``state.save_state`` atomic path at call sites.

Issue #477 introduces the policy module and the state memory only. It does NOT
wire them into dispatch (that is a later issue). No changes to workflow.py,
adapters.py, or dispatch behavior belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ApiWorkerConfig


@dataclass(frozen=True)
class AdapterChoice:
    """The adapter selected for one issue, plus a machine-readable reason.

    ``kind`` is the adapter name (``"api"``, ``"devin-shell"``, …).
    ``provider`` is the api provider key for ``kind == "api"`` and empty for
    non-api adapters. ``reason`` is either a ``policy:*`` string (a routing
    rule matched) or a ``fallback:*`` string (an api preflight check failed
    and the fallback adapter was chosen).
    """

    kind: str
    provider: str
    reason: str


@dataclass(frozen=True)
class BudgetStatus:
    """Spend snapshot consumed by routing's api preflight.

    The ledger that produces real values is a separate workstream; routing
    consumes this value type only. ``daily_headroom`` / ``lifetime_headroom``
    are pre-computed booleans so the policy stays a pure function of its
    arguments (no clock or filesystem access to recompute headroom here).
    """

    spent_today_usd: float
    lifetime_spent_usd: float
    daily_headroom: bool
    lifetime_headroom: bool


def _api_preflight(
    *,
    api_config: ApiWorkerConfig,
    budget: BudgetStatus,
    api_key_present: bool,
    provider_in_cooldown: bool,
    live_api_sessions: int,
) -> str | None:
    """Run the api preflight checks in order; return the first failing reason or None.

    First failing check wins. Each check maps to a ``fallback:*`` reason. When
    all checks pass, returns ``None`` (caller treats this as "go api").
    """
    # Ordered checks — first failure wins. Adding a check is one line here.
    if not api_config.enabled:
        return "fallback:disabled"
    if not api_key_present:
        return "fallback:auth"
    if not budget.daily_headroom or not budget.lifetime_headroom:
        return "fallback:budget"
    if provider_in_cooldown:
        return "fallback:cooldown"
    if live_api_sessions >= api_config.max_concurrent_sessions:
        return "fallback:concurrency"
    return None


def _api_choice_or_fallback(
    *,
    api_config: ApiWorkerConfig,
    budget: BudgetStatus,
    api_key_present: bool,
    provider_in_cooldown: bool,
    live_api_sessions: int,
    policy_reason: str,
) -> AdapterChoice:
    """Resolve a prospective api choice through preflight.

    On preflight pass: ``AdapterChoice("api", api_config.provider, policy_reason)``.
    On preflight failure: ``AdapterChoice(api_config.fallback_adapter, "", <fallback reason>)``.
    """
    failure = _api_preflight(
        api_config=api_config,
        budget=budget,
        api_key_present=api_key_present,
        provider_in_cooldown=provider_in_cooldown,
        live_api_sessions=live_api_sessions,
    )
    if failure is not None:
        return AdapterChoice(api_config.fallback_adapter, "", failure)
    return AdapterChoice("api", api_config.provider, policy_reason)


def select_adapter(
    *,
    rework: bool,
    issue_labels: frozenset[str] | set[str],
    complexity_high_label: str,
    api_config: ApiWorkerConfig,
    budget: BudgetStatus,
    api_key_present: bool,
    provider_in_cooldown: bool,
    live_api_sessions: int,
    default_adapter: str,
) -> AdapterChoice:
    """Pure per-issue adapter routing.

    Every input is passed in — no filesystem, GitHub, env, or clock access
    inside. Same inputs always yield the same ``AdapterChoice``.

    ``complexity_high_label`` is the label string from
    ``config.labels.complexity_high``; the literal never lives in this module
    (issue #481). It is matched against ``issue_labels`` for the first-pass
    complexity rule.

    Rules, in order (each candidate rule produces a prospective api choice
    that is then validated through api preflight; the first matching
    candidate wins):

    1. ``rework=True`` -> prospective api with ``"policy:rework"``.
    2. First pass (``rework=False``) and ``complexity_high_label`` is in
       ``issue_labels`` -> prospective api with ``"policy:complexity"``.
    3. Otherwise -> ``AdapterChoice(default_adapter, "", "policy:default")``.

    For any prospective api choice, preflight is evaluated (first failing
    check wins). On failure the fallback adapter is returned with a
    ``fallback:*`` reason.
    """
    # Candidate rules that may route to api, evaluated in order. Each entry
    # is either a policy reason string (prospective api choice) or None
    # (rule does not match -> fall through to the next). Adding a rule is
    # one entry here plus its condition; the preflight + fallback logic is
    # shared via _api_choice_or_fallback.
    candidate_reasons: list[str | None] = [
        "policy:rework" if rework else None,
        # First-pass complexity rule (issue #481). The rework rule above fires
        # first, so a rework issue carrying the label still routes to api with
        # the rework reason — the complexity rule is gated on ``not rework``.
        "policy:complexity" if (not rework) and complexity_high_label in issue_labels else None,
    ]
    for reason in candidate_reasons:
        if reason is None:
            continue
        return _api_choice_or_fallback(
            api_config=api_config,
            budget=budget,
            api_key_present=api_key_present,
            provider_in_cooldown=provider_in_cooldown,
            live_api_sessions=live_api_sessions,
            policy_reason=reason,
        )
    return AdapterChoice(default_adapter, "", "policy:default")


def record_adapter_choice(
    state: dict[str, Any],
    issue_number: int,
    choice: AdapterChoice,
    now_iso: str,
) -> dict[str, Any]:
    """Return a new state dict with the adapter choice appended to history.

    Appends ``{"ts": now_iso, "kind": choice.kind, "provider": choice.provider,
    "reason": choice.reason}`` to ``state["issues"][<n>]["adapter_history"]``,
    creating the list if absent. Does not mutate the input ``state`` dict and
    preserves all unknown existing keys on both the state and the issue entry.
    Performs no I/O — persistence goes through ``state.save_state`` at call
    sites.
    """
    issue_key = str(issue_number)
    issues = dict(state.get("issues", {}))
    entry = dict(issues.get(issue_key, {}))
    history = list(entry.get("adapter_history", []))
    history.append(
        {
            "ts": now_iso,
            "kind": choice.kind,
            "provider": choice.provider,
            "reason": choice.reason,
        }
    )
    entry["adapter_history"] = history
    issues[issue_key] = entry
    return {**state, "issues": issues}
