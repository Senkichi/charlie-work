"""The app-side seam of the GraphQL budget governor (issue #2442).

``budget_deferral(app, lane)`` is the one question a gated lane asks. It reads
the live shared ``graphql`` window through the app's guarded transport, applies
the tier mapping in ``github_transport.governor``, and on a deferral records
one ``graphql_rate_limit_deferred`` event (the kind reconcile already emits)
per lane per repo per window. Clients that are not guarded transports (local
backends, test doubles) and a disabled governor never defer.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from charlie_work.github_transport.governor import (
    RESOURCE,
    BudgetDecision,
    BudgetGovernor,
    BudgetReserves,
)
from charlie_work.github_transport.guarded import GuardedTransport

_EVENT_KIND = "graphql_rate_limit_deferred"
_EMITTED: set[tuple[str, str, int | None]] = set()
_EMITTED_LOCK = threading.Lock()


def governor_for(app: Any) -> BudgetGovernor | None:
    """The governor over *app*'s transport, or ``None`` when it cannot apply."""
    transport = getattr(app.gh, "_transport_v2", None)
    if not isinstance(transport, GuardedTransport):
        return None
    return BudgetGovernor(
        lambda: transport.rate_window(RESOURCE),
        BudgetReserves.from_runtime(app.config.runtime),
        enabled=transport.governor_enabled(),
    )


def budget_deferral(app: Any, lane: str) -> BudgetDecision | None:
    """The decision when *lane* must defer for budget (event recorded), else ``None``."""
    governor = governor_for(app)
    if governor is None:
        return None
    decision = governor.check(lane)
    if not decision.defer:
        return None
    key = (str(app.paths.state_file), lane, decision.reset)
    with _EMITTED_LOCK:
        first = key not in _EMITTED
        _EMITTED.add(key)
    if first:
        app.write_gate.log_event(
            _EVENT_KIND,
            {
                "remaining": decision.remaining,
                "reset": decision.reset,
                "threshold": decision.threshold,
                "phase": lane,
                "tier": decision.tier.name.lower(),
            },
        )
    return decision


def budget_gate_for(app: Any) -> Callable[[str], bool]:
    """``PassDeadline``'s gate: ``True`` when *lane* defers for budget."""
    return lambda lane: budget_deferral(app, lane) is not None
