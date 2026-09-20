"""budget_status headroom boundary conditions (issue #480).

Split out of ``tests/test_api_budget.py`` (issue #1571, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_api_budget_unit_fixtures.py``.
"""

from __future__ import annotations

from charlie_work.api_budget import (
    BudgetStatus,
    DayBucket,
    Ledger,
    budget_status,
)
from charlie_work.config import ApiBudgetConfig


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
