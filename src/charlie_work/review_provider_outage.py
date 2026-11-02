"""Provider-outage handling for a dead reviewer session (issue #1808).

A reviewer whose terminal ``result`` event is a provider-side ``api_error``
(429/500/502/503/529) died because the provider was unavailable, not because
the PR's review failed. Counting it toward ``max_review_dispatch_attempts``
walked six PRs across five repos to escalation inside one 35-minute outage.

Split out of ``stalled_review_reap`` (whose line count is pinned by a band
test) so the stalled-review sweep only classifies and delegates. The sweep's
fleet-backoff writer is injected as ``arm_backoff`` -- it lives in that module,
which imports this one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, NamedTuple

from .config import OrchestratorConfig
from .state import reviewer_quota_last_probe_cleared_at, without_review_dispatch_claim
from .verdict_parsing import _extract_terminating_cause
from .worker import WorkerView
from .write_gate import WriteGate

# ``arm_backoff(state) -> (new_state, quota_record)``: arms the fleet-wide
# exponential reviewer probe backoff.
ArmBackoff = Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]


class ProviderApiErrorOutcome(NamedTuple):
    state: dict[str, Any]
    event_payload: dict[str, Any]
    backoff_armed: bool


# Provider-side ``api_error`` statuses that mean "the provider was unavailable
# or rate-limiting", never "this PR's review failed". 429 is a rate/quota
# signal (the 2026-09-24 overnight storm); 500/502/503/529 are the 2026-09-22
# outage. 400/401 etc. are NOT here: those point at the request or credentials
# and keep counting toward the per-PR attempt cap.
PROVIDER_API_ERROR_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 529})


def provider_api_error_status(cause: dict[str, Any]) -> int | None:
    """The provider-outage HTTP status in a terminating ``cause``, else ``None``.

    Reads the ``result`` event fields ``_extract_terminating_cause`` captured:
    ``terminal_reason == "api_error"`` with ``api_error_status`` in
    ``PROVIDER_API_ERROR_STATUSES``.
    """
    if cause.get("terminal_reason") != "api_error":
        return None
    status = cause.get("api_error_status")
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    return status if status in PROVIDER_API_ERROR_STATUSES else None


def session_api_error_status(log_path: Path) -> int | None:
    """The provider-outage status that ended THIS session, else ``None``.

    Reads only the session's own tee'd stream-json log: the per-PR events
    sidecar may belong to an earlier round.
    """
    return provider_api_error_status(_extract_terminating_cause(log_path, log_path))


def _probe_cleared_since_death(state: dict[str, Any], log_mtime_dt: datetime | None) -> bool:
    """True when a probe cleared the quota gate after the session's last log
    write -- the provider has since recovered, so re-arming would re-poison it."""
    cleared = reviewer_quota_last_probe_cleared_at(state)
    if not cleared or log_mtime_dt is None:
        return False
    try:
        cleared_dt = datetime.fromisoformat(cleared.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    if cleared_dt.tzinfo is None:
        cleared_dt = cleared_dt.replace(tzinfo=UTC)
    return cleared_dt > log_mtime_dt


def apply_provider_api_error(
    state: dict[str, Any],
    pr_key: str,
    *,
    worker: WorkerView,
    config: OrchestratorConfig,
    write_gate: WriteGate,
    api_error_status: int,
    log_mtime_dt: datetime | None,
    arm_backoff: ArmBackoff,
    backoff_armed: bool,
) -> ProviderApiErrorOutcome:
    """Disposition a dead reviewer that ended on a provider ``api_error``.

    Within ``max_consecutive_review_api_errors`` consecutive deaths on the PR:
    roll the claim back and give the attempt back, and (once per sweep, unless
    a probe already cleared since the death) arm the fleet-wide backoff and
    emit one ``review_provider_outage`` warning. Past the bound the death is a
    counted failure so a persistent per-PR poison still converges on the
    attempt cap. ``backoff_armed`` is the sweep's once-per-sweep latch.
    """
    pr_state = state["prs"].get(pr_key, {})
    streak = int(pr_state.get("review_api_error_streak", 0)) + 1
    counted = streak > config.review_dispatch.max_consecutive_review_api_errors
    if counted:
        new_pr = {
            **pr_state,
            "number": worker.issue_number,
            "review_dispatch_status": "review_dispatch_failed",
            "review_dispatch_failed_at": worker.started_at,
            "review_dispatch_pending_at": None,
            "review_dispatched_at": None,
            "reviewer_pid": None,
            "reviewer_process_start_time": None,
        }
    else:
        if not backoff_armed and not _probe_cleared_since_death(state, log_mtime_dt):
            state, quota_record = arm_backoff(state)
            backoff_armed = True
            state = write_gate.append_event(
                state,
                "review_provider_outage",
                {
                    "api_error_status": api_error_status,
                    "pr_number": worker.issue_number,
                    "throttled_until": quota_record.get("throttled_until"),
                    "probe_after": quota_record.get("probe_after"),
                    "consecutive_probe_failures": quota_record.get("consecutive_probe_failures"),
                    "source": "stalled_review_sweep",
                },
                level="warning",
            )
        new_pr = without_review_dispatch_claim(pr_state)
        attempts = int(pr_state.get("review_dispatch_attempt_count", 0))
        if attempts > 0:
            new_pr["review_dispatch_attempt_count"] = attempts - 1
    new_pr["review_log_unreadable_streak"] = 0
    new_pr["review_api_error_streak"] = streak
    state["prs"][pr_key] = new_pr
    event_payload = {
        "pr_number": worker.issue_number,
        "pid": worker.pid,
        "started_at": worker.started_at,
        "reason": "provider_api_error_counted" if counted else "provider_api_error",
        "api_error_status": api_error_status,
        "streak": streak,
    }
    return ProviderApiErrorOutcome(state, event_payload, backoff_armed)
