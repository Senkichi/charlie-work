"""The GraphQL budget governor's tiers (issue #2442, ADR-0006).

One mapping says how much of the hour each lane may spend; one pure function
turns that and the observed ``graphql`` window into a defer-or-go decision.
Callers never compare against a threshold themselves.

The tiers are reserves, in the order lanes give way as the hour drains:

=========  =======================================  =====================
tier       lanes                                    defers below
=========  =======================================  =====================
RESERVE    reconcile, drift, review reaps, CI       ``graphql_rate_limit_threshold`` (1500)
           reclaim
SCAN       intake, dispatch scans, review           ``graphql_scan_reserve`` (800)
           dispatch, quota probe
FLOOR      everything else that is gated            ``graphql_floor_reserve`` (300)
ESSENTIAL  merge, label writes, the                 never
           unauthorized-merge tripwire
=========  =======================================  =====================

A lane absent from the mapping is FLOOR: a new gate added without a mapping
gives way last of the optional work rather than first, and never blocks the
essential lanes. The reserves are made monotone (a later tier's reserve is
never above an earlier one's), so a misordered config cannot let a lower-
priority lane outlive a higher one; a first reserve of 0 disables the guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..api_budget import GitHubRateWindow

RESOURCE = "graphql"


class Tier(IntEnum):
    RESERVE = 0
    SCAN = 1
    FLOOR = 2
    ESSENTIAL = 3  # never defers


LANE_TIERS: Mapping[str, Tier] = MappingProxyType(
    {
        "reconcile": Tier.RESERVE,
        "drift": Tier.RESERVE,
        "review_reap": Tier.RESERVE,
        "ci_reclaim": Tier.RESERVE,
        "intake": Tier.SCAN,
        "dispatch_rework": Tier.SCAN,
        "dispatch": Tier.SCAN,
        "dispatch_reviews": Tier.SCAN,
        "quota_probe": Tier.SCAN,
        "deescalate": Tier.FLOOR,
        "local_lane": Tier.FLOOR,
        "merge": Tier.ESSENTIAL,
        "label_write": Tier.ESSENTIAL,
        "unauthorized_merge_tripwire": Tier.ESSENTIAL,
    }
)


def tier_of(lane: str) -> Tier:
    return LANE_TIERS.get(lane, Tier.FLOOR)


@dataclass(frozen=True)
class BudgetReserves:
    """The three reserves (points of GraphQL budget left), monotone by construction."""

    reserve: int = 1500
    scan: int = 800
    floor: int = 300

    @classmethod
    def from_runtime(cls, runtime: Any) -> "BudgetReserves":
        reserve = int(getattr(runtime, "graphql_rate_limit_threshold", cls.reserve))
        scan = int(getattr(runtime, "graphql_scan_reserve", cls.scan))
        floor = int(getattr(runtime, "graphql_floor_reserve", cls.floor))
        scan = min(scan, reserve)
        return cls(reserve, scan, min(floor, scan))

    def threshold(self, tier: Tier) -> int:
        """Points that must remain for a *tier* lane to run (0: always runs)."""
        return (self.reserve, self.scan, self.floor, 0)[tier]


@dataclass(frozen=True)
class BudgetDecision:
    lane: str
    tier: Tier
    threshold: int
    remaining: int | None
    reset: int | None

    @property
    def defer(self) -> bool:
        """Defer only on a *known* shortfall: an unobserved budget fails open."""
        return self.remaining is not None and self.remaining < self.threshold


def decide(lane: str, reserves: BudgetReserves, window: GitHubRateWindow | None) -> BudgetDecision:
    tier = tier_of(lane)
    return BudgetDecision(
        lane,
        tier,
        reserves.threshold(tier),
        None if window is None else window.remaining,
        None if window is None else window.reset_epoch,
    )


class BudgetGovernor:
    """Decides, per lane, against the live shared ``graphql`` window."""

    def __init__(
        self,
        window: Callable[[], GitHubRateWindow | None],
        reserves: BudgetReserves,
        *,
        enabled: bool = True,
    ) -> None:
        self._window = window
        self.reserves = reserves
        self.enabled = enabled

    def check(self, lane: str) -> BudgetDecision:
        window = self._window() if self.enabled else None
        return decide(lane, self.reserves, window)


__all__ = [
    "BudgetDecision",
    "BudgetGovernor",
    "BudgetReserves",
    "LANE_TIERS",
    "RESOURCE",
    "Tier",
    "decide",
    "tier_of",
]
