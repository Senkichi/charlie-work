"""Is the repo's CI queue backlogged? (DD-13).

Reads the freshest ``runner_allocation`` event that ``ci_fleet``'s allocation
pass writes (the same event ``ci_headroom`` reads) and returns the target's
``oldest_queued_seconds`` when it meets the caller's threshold. The value is
per repo, not per lane: ``ci_fleet`` reports the oldest queued job across the
repo's runs.

Fails open: missing, stale, malformed or unconfigured data returns ``None``
(the caller reruns as before). Never raises and never writes an event, so it
adds no write-gate traffic to the merge driver's pass.
"""

from __future__ import annotations

from typing import Any
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work.ci_headroom import (
    ALLOCATION_EVENT_KIND,
    _is_plain_int,
    _parse_event_ts,
    _target_for_repo,
)
from charlie_work import layout
from charlie_work.command_result import CommandResult
from charlie_work.fleet_paths import fleet_dir
from charlie_work.instrumentation import query_events
from charlie_work.state import load_state, state_lock


def infra_rerun_backlog_seconds(
    repo: str,
    *,
    fleet_state_path: Path,
    threshold_seconds: int,
    now: datetime | None = None,
    max_data_age_minutes: int = 30,
) -> int | None:
    """``oldest_queued_seconds`` for ``repo`` when >= ``threshold_seconds``, else ``None``."""
    if threshold_seconds <= 0 or not repo or repo == "?":
        return None
    try:
        events = query_events(fleet_state_path, kind=ALLOCATION_EVENT_KIND, limit=1)
    except Exception:
        return None
    if not events:
        return None
    event = events[0]
    event_time = _parse_event_ts(event.get("ts"))
    if event_time is None:
        return None
    resolved_now = now if now is not None else datetime.now(UTC)
    if resolved_now - event_time > timedelta(minutes=max_data_age_minutes):
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    target = _target_for_repo(payload, repo)
    if target is None or target.get("pinned"):
        return None
    oldest = target.get("oldest_queued_seconds")
    if not _is_plain_int(oldest):
        return None
    return oldest if oldest >= threshold_seconds else None


def infra_rerun_backlog(app: Any) -> int | None:
    """Oldest queued seconds for this repo when it meets the defer threshold."""
    threshold = app.config.auto_merge.infra_rerun_backlog_defer_seconds
    if threshold <= 0:
        return None
    from charlie_work.dead_worker_sweep.effects_pr import _safe_repo_slug

    return infra_rerun_backlog_seconds(
        _safe_repo_slug(app.gh),
        fleet_state_path=layout.state_file_path(fleet_dir(override=app.fleet_dir_override)),
        threshold_seconds=threshold,
    )


def defer_infra_rerun_for_backlog(
    app: Any,
    pr_number: int,
    issue_number: int | None,
    *,
    head_key: str,
    rerun_run_ids: tuple[int, ...],
    backlog_seconds: int,
    ok: bool,
    extra_data: dict[str, Any] | None,
) -> CommandResult:
    """Defer this pass's rerun; record the marker and event on the transition only."""
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        pr_state = state["prs"].get(str(pr_number), {})
        marker = pr_state.get("infra_rerun_backlog_deferred") or {}
        if not (isinstance(marker, dict) and marker.get(head_key)):
            state["prs"][str(pr_number)] = {
                **pr_state,
                "number": pr_number,
                "issue_number": issue_number,
                "infra_rerun_backlog_deferred": {head_key: True},
            }
            state = app._record_event(
                state,
                "infra_rerun_backlog_deferred",
                {
                    "pr_number": pr_number,
                    "head_sha": head_key,
                    "run_ids": list(rerun_run_ids),
                    "oldest_queued_seconds": backlog_seconds,
                },
            )
            app.write_gate.save_state(state)
    return CommandResult(
        ok,
        f"PR #{pr_number} infra rerun deferred: CI queue backlog "
        f"(oldest queued job {backlog_seconds}s)",
        {
            "pr": pr_number,
            "issue": issue_number,
            **(extra_data or {}),
            "infra_rerun_run_ids": list(rerun_run_ids),
            "infra_rerun_backlog_deferred": True,
            "oldest_queued_seconds": backlog_seconds,
        },
    )
