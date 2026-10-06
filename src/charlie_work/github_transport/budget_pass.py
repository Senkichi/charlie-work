"""The per-pass GitHub spend report (issue #2439).

``emit_github_budget_pass`` drains a client's spend (requests and points per
capability, from ``x-ratelimit-used`` deltas) and records one
``github_budget_pass`` event. The fleet calls it once per lane pass and once
per repo in a reap sweep; the budget governor (#2442) and the hourly ranking
read the same ``budget_pass_payload`` shape.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..api_budget import GitHubRateBudget, GitHubSpend
from ..instrumentation import log_event
from .guarded import GuardedTransport

logger = logging.getLogger(__name__)

EVENT_KIND = "github_budget_pass"


def budget_pass_payload(
    spend: GitHubSpend, budget: GitHubRateBudget, *, pass_kind: str, repo_key: str | None
) -> dict[str, Any]:
    """Event payload: totals, per-capability rows (heaviest first) and the
    remaining/limit/reset GitHub last reported per resource."""
    rows = sorted(spend.entries, key=lambda e: (-e.points, -e.requests, e.capability, e.resource))
    return {
        "pass_kind": pass_kind,
        "repo_key": repo_key,
        "requests": spend.requests,
        "points": spend.points,
        "unmetered_requests": sum(e.unmetered_requests for e in rows),
        "points_by_resource": spend.points_by_resource(),
        "capabilities": [
            {
                "capability": e.capability,
                "resource": e.resource,
                "requests": e.requests,
                "points": e.points,
                "unmetered_requests": e.unmetered_requests,
            }
            for e in rows
        ],
        "observed": {
            w.resource: {"remaining": w.remaining, "limit": w.limit, "reset": w.reset_epoch}
            for w in budget.windows
        },
    }


def emit_github_budget_pass(
    gh: object, state_path: Path, *, pass_kind: str, repo_key: str | None = None
) -> bool:
    """Record *gh*'s spend since its last take as one event. Best-effort:
    never raises. False when *gh* has no guarded transport (a local-file
    backend or a test double), so nothing was emitted."""
    transport = getattr(gh, "_transport_v2", None)
    if not isinstance(transport, GuardedTransport):
        return False
    try:
        payload = budget_pass_payload(
            transport.take_spend(), transport.budget.value, pass_kind=pass_kind, repo_key=repo_key
        )
        # write-gate-exempt(issue=2439): per-pass accounting record; no WriteGate at the client layer
        log_event(state_path, EVENT_KIND, payload, repo=repo_key)
    except Exception:  # noqa: BLE001 - accounting must never break a pass
        logger.debug("Failed to record %s for %s", EVENT_KIND, repo_key, exc_info=True)
        return False
    return True
