"""Sustained-window capacity-starvation escalation (issue #763).

Extracted from ``fleet_dispatch.py`` and ``config.py`` so new code does not
land in an over-cap monolith (file-size ratchet, issue #1442). This module
owns the detection logic, its config dataclass, and the operator-digest
entry builder; ``fleet_dispatch.py`` wires the detector into the allocation
prologue and ``config.py`` re-exports the dataclass and calls the parser.

The allocator (``runner_allocation``) rebalances *already-registered*
listeners and can never mint a registration, so a repo whose live demand
exceeds its registered capacity while the host-wide budget has slack is
permanently unsatisfiable by allocation alone. ci_fleet's #799 lands an
edge-triggered ``runner_capacity_starved`` event the moment that condition
turns true, but a single-pass spike can look identical to a sustained
shortage, and that event never reaches the operator notify digest -- it
lives only in ``events.db``.

This module arms the *durable* half: when the same repo stays starved for a
sustained window, the fleet prologue raises a structured
``runner_capacity_starvation_escalation`` event that surfaces in the
operator attention digest (not just ``events.db``), so the next starvation
is surfaced instead of discovered by an operator reading queue times.

Scope is detection + event only. Provisioning/registration stays
operator-gated (see issue #826 for the manual-trigger actuator) -- this
module never causes a runner to be registered, started, or parked.
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .fleet_paths import warn_fleet_dir_virtualization_on_write
from . import layout
from .instrumentation import log_event
from .notify import AttentionEntry
from .state import utc_now

logger = logging.getLogger(__name__)

# Event kind for the sustained-window capacity-starvation escalation (issue
# #763). ci_fleet's #799 already writes the edge-triggered
# ``runner_capacity_starved``/``_recovered`` pair to events.db; this is the
# operator-visible escalation that fires once a starvation episode has
# persisted for the configured window, so it surfaces in the notify digest
# rather than only in events.db. Read here (not re-declared at call sites) for
# the same reason label strings are read from ``LabelConfig``: a literal
# scattered across call sites drifts silently.
CAPACITY_STARVATION_ESCALATION_KIND = "runner_capacity_starvation_escalation"


@dataclass(frozen=True)
class RunnerCapacityEscalationConfig:
    """Sustained-window escalation for runner capacity starvation (issue #763).

    ``enabled`` defaults True because the feature is pure observability: it
    reads the allocation plan the prologue already computes and writes an
    event, with no actuation side effect. It is inert on any host where
    ``runner_allocation.enabled`` is false, since the prologue returns before
    reaching it. The knob exists for rollback, not opt-in.

    ``starvation_escalation_minutes`` is the sustained window. A repo must
    stay starved (``demand > capacity`` while host-wide budget has slack) for
    at least this many minutes before the escalation fires, so a transient
    spike that clears within one or two fleet passes (default cadence 5 min)
    does not raise a false alarm. The window is measured wall-clock from the
    first starved pass, not by counting passes, so it is robust to the
    supervisor's respawn/restart cadence and to a pass that was skipped.
    Default 15 min = three default-cadence passes.
    """

    enabled: bool = True
    starvation_escalation_minutes: int = 15


def parse_runner_capacity_escalation(data: dict[str, Any]) -> RunnerCapacityEscalationConfig:
    """Validate and build the ``runner_capacity_escalation`` config section.

    Extracted from ``config.build_config_from_data`` so the validation lives
    alongside the dataclass it guards. ``ConfigError`` is imported lazily to
    avoid a circular import (``config.py`` imports this module for the
    dataclass re-export; this module needs ``ConfigError`` from ``config.py``).
    The same lazy-import pattern ``github.py`` already uses.
    """
    from .config import ConfigError

    section_data = data.get("runner_capacity_escalation")
    if not isinstance(section_data, dict):
        section_data = {}
    rce_enabled = section_data.get("enabled")
    if rce_enabled is not None and not isinstance(rce_enabled, bool):
        raise ConfigError(
            "config section 'runner_capacity_escalation' key 'enabled' must be a bool, "
            f"got {type(rce_enabled).__name__}"
        )
    rce_minutes = section_data.get("starvation_escalation_minutes")
    if rce_minutes is not None and (
        isinstance(rce_minutes, bool) or not isinstance(rce_minutes, int)
    ):
        raise ConfigError(
            "config section 'runner_capacity_escalation' key 'starvation_escalation_minutes' "
            f"must be an int, got {type(rce_minutes).__name__}"
        )
    if isinstance(rce_minutes, int) and not isinstance(rce_minutes, bool) and rce_minutes <= 0:
        raise ConfigError(
            "config section 'runner_capacity_escalation' key 'starvation_escalation_minutes' "
            f"must be > 0, got {rce_minutes}"
        )
    valid = {f.name for f in fields(RunnerCapacityEscalationConfig)}
    unknown = sorted(set(section_data) - valid)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in config section 'runner_capacity_escalation': "
            f"{', '.join(unknown)} (valid: {', '.join(sorted(valid))})"
        )
    return RunnerCapacityEscalationConfig(**section_data)


def _starved_repos_from_plan(plan: Any) -> list[dict[str, Any]]:
    """Repos where ``demand > capacity`` while host-wide budget has slack.

    Mirrors ``ci_fleet.runner_allocation.starved_repos`` (issue #799) but is
    computed here from the plan the prologue already holds, so the escalation
    detection does not couple to a ci_fleet import that lives in a separate
    repo on its own release cadence. The condition is identical: a repo is
    starved when its live demand exceeds its registered runner capacity *and*
    the host still has unused budget elsewhere -- the second clause is what
    makes the signal worth raising, since a fully-subscribed budget means no
    idle headroom a bigger registration could fill.

    Returns a list of dicts (``repo``, ``demand``, ``capacity``, ``running``,
    ``spare_budget``) sorted by repo for deterministic output.
    """
    targets = tuple(getattr(plan, "targets", ()) or ())
    if not targets:
        return []
    spare_budget = getattr(plan, "budget", 0) - sum(getattr(t, "running", 0) for t in targets)
    if spare_budget <= 0:
        return []
    starved = [
        {
            "repo": t.repo,
            "demand": t.demand,
            "capacity": t.capacity,
            "running": t.running,
            "spare_budget": spare_budget,
        }
        for t in targets
        if t.demand > t.capacity
    ]
    starved.sort(key=lambda s: s["repo"])
    return starved


def _load_capacity_starvation_state(path: Path) -> dict[str, dict[str, Any]]:
    """Load the fleet capacity-starvation escalation sidecar (issue #763).

    Returns a ``{repo: {"starved_since": iso, "escalated": bool}}`` mapping.
    A missing or corrupt file is non-fatal: the worst case is one extra
    episode-start record on the next pass, which re-arms the sustained window
    from scratch rather than escalating on stale data -- the safe direction.
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, LookupError, ValueError, OSError):
        logger.warning("Capacity starvation state %s unreadable; starting fresh", path)
        return {}
    repos = data.get("repos") if isinstance(data, dict) else None
    if not isinstance(repos, dict):
        return {}
    # Coerce to the expected shape; drop anything malformed so a corrupt entry
    # cannot crash the escalation pass.
    cleaned: dict[str, dict[str, Any]] = {}
    for repo, entry in repos.items():
        if not isinstance(entry, dict) or not isinstance(repo, str):
            continue
        since = entry.get("starved_since")
        if not isinstance(since, str):
            continue
        cleaned[repo] = {"starved_since": since, "escalated": bool(entry.get("escalated"))}
    return cleaned


def _save_capacity_starvation_state(path: Path, repos: dict[str, dict[str, Any]]) -> None:
    """Atomically persist the capacity-starvation escalation sidecar.

    Temp-file + ``replace()`` per the project's atomic-write invariant. Warns
    on fleet-dir virtualization (issue #624) but never blocks the write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    warn_fleet_dir_virtualization_on_write(
        path.parent, context="writing capacity_starvation_state.json"
    )
    payload = {"version": 1, "generated_at": utc_now(), "repos": repos}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _iso_utc(dt: datetime.datetime) -> str:
    """ISO-8601 UTC timestamp with second precision and a ``Z`` suffix.

    Matches the convention used elsewhere in the fleet event store so a
    round-tripped ``starved_since`` compares cleanly against event ``ts``
    values.
    """
    return dt.astimezone(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def detect_capacity_starvation_escalation(
    plan: Any,
    *,
    fleet_dir_override: str | None,
    fleet_state_path: Path,
    escalation_config: RunnerCapacityEscalationConfig,
    dry_run: bool,
    now: datetime.datetime | None = None,
) -> list[dict[str, Any]]:
    """Raise a structured escalation when capacity starvation is sustained (#763).

    Runs after the allocation pass, on the plan it already computed. For each
    repo that is starved (``demand > capacity`` while host-wide budget has
    slack), the sidecar records the episode start on the first starved pass;
    once the starvation has persisted for
    ``escalation_config.starvation_escalation_minutes`` (wall-clock, so robust
    to the supervisor's respawn cadence and to skipped passes), this emits a
    single ``runner_capacity_starvation_escalation`` event to the fleet-level
    ``events.db`` and returns a matching attention-event dict for the operator
    digest. The escalation is edge-triggered per episode: the sidecar's
    ``escalated`` flag suppresses re-firing every subsequent pass, and a repo
    that recovers (no longer starved) is dropped from the sidecar so the next
    episode starts a fresh window.

    Gated on ``not dry_run`` for the same reason the allocation pass's
    hysteresis persist and #799's edge events are: a dry-run previews the plan
    and must not have side effects, and an event write plus a sidecar update
    are both side effects that would also consume the rising edge the next
    real pass needs to see.

    Returns a list of attention-event dicts (``type`` =
    ``runner_capacity_starvation_escalation``) for aggregation into the fleet
    digest. Empty when nothing escalated this pass.
    """
    if dry_run or not escalation_config.enabled:
        return []
    starved = _starved_repos_from_plan(plan)

    resolved_now = now if now is not None else datetime.datetime.now(datetime.UTC)
    state_path = layout.capacity_starvation_state_path(override=fleet_dir_override)
    prior = _load_capacity_starvation_state(state_path)

    threshold = datetime.timedelta(minutes=escalation_config.starvation_escalation_minutes)
    next_state: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []

    for s in starved:
        repo = s["repo"]
        entry = prior.get(repo)
        if entry is None:
            # Rising edge: first starved pass for this episode. Record the
            # start but do not escalate yet -- a single-pass spike must not
            # raise a false alarm.
            next_state[repo] = {"starved_since": _iso_utc(resolved_now), "escalated": False}
            continue
        starved_since_iso = entry["starved_since"]
        next_state[repo] = {
            "starved_since": starved_since_iso,
            "escalated": entry["escalated"],
        }
        if entry["escalated"]:
            # Already escalated this episode; stay silent until recovery.
            continue
        try:
            starved_since = datetime.datetime.fromisoformat(starved_since_iso)
        except ValueError:
            # A corrupt timestamp cannot be trusted to measure the window;
            # re-arm from now rather than escalate on garbage.
            next_state[repo] = {"starved_since": _iso_utc(resolved_now), "escalated": False}
            continue
        if starved_since.tzinfo is None:
            starved_since = starved_since.replace(tzinfo=datetime.UTC)
        if resolved_now - starved_since >= threshold:
            # Sustained window crossed: escalate once for this episode.
            duration_seconds = int((resolved_now - starved_since).total_seconds())
            payload = {
                "repo": repo,
                "demand": s["demand"],
                "capacity": s["capacity"],
                "running": s["running"],
                "spare_budget": s["spare_budget"],
                "starved_since": starved_since_iso,
                "duration_seconds": duration_seconds,
                "escalation_minutes": escalation_config.starvation_escalation_minutes,
                "remedy": (
                    "provision more runners for this repo (runner_scaling or "
                    "config.cmd); runner_allocation cannot mint registrations"
                ),
            }
            log_event(
                fleet_state_path,
                CAPACITY_STARVATION_ESCALATION_KIND,
                payload,
                repo=repo,
                level="error",
            )
            events.append(
                {
                    "repo_key": "fleet",
                    "type": CAPACITY_STARVATION_ESCALATION_KIND,
                    "repo": repo,
                    "demand": s["demand"],
                    "capacity": s["capacity"],
                    "running": s["running"],
                    "spare_budget": s["spare_budget"],
                    "starved_since": starved_since_iso,
                    "duration_seconds": duration_seconds,
                    "reason": (
                        f"{repo}: CI demand {s['demand']} exceeds its "
                        f"{s['capacity']} registered runner(s) for "
                        f"{duration_seconds // 60} min while "
                        f"{s['spare_budget']} budget slot(s) sit idle -- "
                        "provision more runners; allocation cannot"
                    ),
                }
            )
            next_state[repo]["escalated"] = True

    # Repos that recovered this pass drop out of the sidecar so the next
    # starvation starts a fresh sustained window. ci_fleet's #799
    # ``runner_capacity_recovered`` event already records the recovery in
    # events.db, so no separate recovery event is needed here.
    # Repos not observed this pass (absent from the plan's targets entirely)
    # are also dropped: a plan that no longer lists a repo means its runners
    # were de-registered, and carrying a stale episode forward would escalate
    # against a repo the host no longer serves.
    _save_capacity_starvation_state(state_path, next_state)
    return events


def build_capacity_starvation_attention_entry(event: dict[str, Any]) -> AttentionEntry:
    """Build the operator-digest ``AttentionEntry`` for an escalation event.

    This is occurrence-style, not persistent-health: the detector is
    edge-triggered per episode (its sidecar's ``escalated`` flag suppresses
    re-firing), so the event only reaches the digest list on the single pass
    the window is crossed -- it must not be deduped by
    ``_filter_fleet_health_transitions``, which would collapse the one real
    rising edge into silence. ``adapter_kind`` carries the starved repo so an
    operator can see *which* repo is starving, not just "fleet".
    """
    return AttentionEntry(
        issue_number=-1,
        adapter_kind=event.get("repo") or event.get("repo_key", "fleet"),
        health="ERROR",
        previous_health=None,
        last_log_line=event.get("reason"),
        pid=None,
    )
