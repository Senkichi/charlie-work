from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .config import ApiWorkerConfig, ConfigError, OrchestratorConfig, SupervisorConfig
from .fleet_paths import fleet_dir, warn_fleet_dir_virtualization_on_write
from .fleet_registry import _load_registry, count_fleet_runners
from . import layout
from .github import GitHub, GitHubError
from .global_config import describe_config_file, load_layered_config
from .instrumentation import log_event
from .notify import AttentionDigest, AttentionEntry, emit_digest
from .paths import RepoNotFoundError, runtime_paths
from .supervise import (
    LocalSnapshot,
    orchestrator_root,
    read_head_sha,
    record_zero_pass_streak,
    self_deploy,
    take_snapshot,
    try_acquire_supervisor_lock,
)
from ci_fleet.charlie_work_adapter import (
    FleetTotals,
    ScaleAction,
    UNATTENDED_ALLOCATION_SOURCE,
    decide_autoscale,
    is_in_cooldown,
    is_pool_idle_for_minutes,
    observe_runner_pool,
    provision_runner,
    record_scale_event,
    run_allocation_pass,
    save_allocation_skip,
    scale_down_idle_runners,
)
from .state import state_lock, utc_now
from .subprocess_runner import no_console_window_kwargs
from .supervise_loop import (
    DEFAULT_MAX_RELAUNCHES,
    EXIT_RESTART_REQUESTED,
    SuperviseLoopResult,
    run_supervise_relaunch_loop,
)
from .supervisor_lifecycle import (
    detect_prior_abnormal_exit,
    is_exit_alertable,
    record_prior_abnormal_exit,
    record_supervisor_exit,
    record_supervisor_started,
    update_supervisor_heartbeat,
)
from .workflow import CommandResult, OrchestratorApp

logger = logging.getLogger(__name__)


def _select_repos(
    registry: dict[str, Any],
    repos: tuple[str, ...] | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Select and order repos for a fleet pass.

    If repos is provided, use exactly that subset in the given order.
    Otherwise, return all repos sorted by oldest last_seen first.

    Args:
        registry: The fleet registry dict with a "repos" map.
        repos: Optional tuple of repo keys to select explicitly.

    Returns:
        A list of (repo_key, entry) tuples in the order to process.
    """
    repos_map = registry.get("repos", {})
    if repos:
        # Explicit subset: use exactly the given keys in the given order
        # Skip keys that don't exist in the registry
        selected = [(key, repos_map[key]) for key in repos if key in repos_map]
        return selected
    else:
        # All repos: sort by oldest last_seen first
        all_repos = list(repos_map.items())

        # Sort by last_seen ascending (oldest first)
        # Repos without last_seen go last (treated as newest)
        def last_seen_key(item: tuple[str, dict[str, Any]]) -> tuple[bool, str]:
            key, entry = item
            last_seen = entry.get("last_seen", "")
            # Repos with last_seen sort before those without
            # (False < True, so False comes first)
            has_last_seen = last_seen != ""
            return (not has_last_seen, last_seen)

        all_repos.sort(key=last_seen_key)
        return all_repos


@dataclass(frozen=True)
class ApiWorkerFleetReport:
    """Fleet-wide api-worker observability line (issue #483).

    Read-only snapshot rendered in the fleet status / pass summary. The
    ``provider``, ``today_usd``, ``lifetime_usd``, and ``cap_usd`` fields come
    from a representative repo (the first enabled repo, or the first configured
    repo when none are enabled). ``live`` is the fleet-wide count of alive
    ``adapter_kind == "api"`` workers. ``enabled_k``/``enabled_m`` is the
    fleet-wide enablement ratio across all repos that configure the section.
    """

    provider: str
    today_usd: float
    lifetime_usd: float
    cap_usd: float
    live: int
    enabled_k: int
    enabled_m: int

    def format_line(self) -> str:
        return (
            f"api-worker: {self.provider}, "
            f"${self.today_usd:.2f} today / ${self.lifetime_usd:.2f} lifetime "
            f"of ${self.cap_usd:.2f}, {self.live} live, "
            f"enabled {self.enabled_k}/{self.enabled_m} repos"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "today_usd": self.today_usd,
            "lifetime_usd": self.lifetime_usd,
            "cap_usd": self.cap_usd,
            "live": self.live,
            "enabled_k": self.enabled_k,
            "enabled_m": self.enabled_m,
            "line": self.format_line(),
        }


def _api_worker_configured(config: OrchestratorConfig) -> bool:
    """Return True when the api_worker section is non-default (configured).

    A bare ``api_worker: {enabled: false}`` with no providers/budget is the
    package default and carries nothing to report. A section with providers
    set but ``enabled: false`` is the built-but-dormant case.
    """
    return config.api_worker != ApiWorkerConfig()


def compute_api_worker_fleet_report(
    *,
    fleet_dir_override: str | None = None,
    preloaded_configs: dict[str, OrchestratorConfig] | None = None,
    now: datetime.datetime | None = None,
) -> ApiWorkerFleetReport | None:
    """Compute the fleet-wide api-worker report line (issue #483).

    Returns ``None`` when no registered repo configures the ``api_worker``
    section (the line is omitted entirely in that case). Otherwise returns an
    ``ApiWorkerFleetReport`` with spend from a representative repo's ledger and
    a fleet-wide live-worker count.

    ``preloaded_configs`` is an optional ``repo_key -> config`` mapping for
    repos whose layered config the caller already loaded (e.g. ``fleet_loop``
    loads each selected repo's config for dispatch). When a repo_key is present
    in this map, its config is reused instead of re-loading from disk — this
    avoids a redundant per-repo config reload on every fleet pass. Repos absent
    from the map (unselected, or callers that don't preload) fall back to
    ``load_layered_config``. The preloaded config must be the raw layered
    config (the report only reads ``api_worker`` fields, so a caller-side
    ``replace(...)`` on unrelated fields is harmless).

    Near read-only: never settles or writes the ledger itself, but
    ``api_budget.load_ledger`` quarantines a corrupt ledger file (renames it to
    a ``.corrupt-*`` sibling) as a side effect of detecting it. That is the only
    filesystem mutation. All errors surface as report values (zeroed spend,
    zero live), never raised.

    ``now`` is the injectable clock used to derive the ledger's ``today`` key
    (issue #828): defaults to ``datetime.now(UTC)`` when not supplied, so
    production behavior is byte-identical. Without it, the day-granularity
    lookup below can miss a UTC-midnight boundary crossed between a caller
    writing the ledger and this function reading it.
    """
    from datetime import UTC, datetime

    from .api_budget import budget_status, ledger_path, load_ledger
    from .worker import iter_workers

    fleet_json_path = layout.fleet_registry_path(override=fleet_dir_override)
    registry = _load_registry(fleet_json_path)
    repos = registry.get("repos", {})
    if not repos:
        return None

    enabled_k = 0
    configured_m = 0
    representative: tuple[str, OrchestratorConfig, dict[str, Any]] | None = None

    for repo_key, entry in repos.items():
        repo_root = Path(entry.get("repo_root", ""))
        if not repo_root.is_dir():
            continue
        config = preloaded_configs.get(repo_key) if preloaded_configs else None
        if config is None:
            try:
                explicit_cfg = entry.get("config_path")
                config = load_layered_config(
                    repo_root,
                    Path(explicit_cfg) if explicit_cfg else None,
                    fleet_dir_override=fleet_dir_override,
                )
            except (ConfigError, GitHubError, OSError):
                continue
        if not _api_worker_configured(config):
            continue
        configured_m += 1
        if config.api_worker.enabled:
            enabled_k += 1
            if representative is None:
                representative = (repo_key, config, entry)
        elif representative is None:
            representative = (repo_key, config, entry)

    if configured_m == 0:
        return None

    assert representative is not None  # configured_m > 0 guarantees this
    _, rep_config, rep_entry = representative
    provider_name = rep_config.api_worker.provider

    # Spend from the representative repo's ledger (read-only).
    state_dir_str = rep_entry.get("state_dir")
    today_usd = 0.0
    lifetime_usd = 0.0
    cap_usd = rep_config.api_worker.budget.lifetime_usd
    if state_dir_str:
        try:
            ledger_file = ledger_path(Path(state_dir_str))
            ledger = load_ledger(ledger_file)
            resolved_now = now if now is not None else datetime.now(UTC)
            today = resolved_now.strftime("%Y-%m-%d")
            status = budget_status(ledger, rep_config.api_worker.budget, today)
            today_usd = status.spent_today_usd
            lifetime_usd = status.lifetime_spent_usd
        except Exception:
            pass

    # Fleet-wide live api worker count.
    live = 0
    for _repo_key, entry in repos.items():
        state_dir_str = entry.get("state_dir")
        if not state_dir_str:
            continue
        sessions_dir = layout.sessions_dir_default(Path(state_dir_str))
        if not sessions_dir.is_dir():
            continue
        try:
            for w in iter_workers(sessions_dir):
                if w.adapter_kind == "api" and w.is_alive():
                    live += 1
        except Exception:
            pass

    return ApiWorkerFleetReport(
        provider=provider_name,
        today_usd=today_usd,
        lifetime_usd=lifetime_usd,
        cap_usd=cap_usd,
        live=live,
        enabled_k=enabled_k,
        enabled_m=configured_m,
    )


def _run_fleet_allocation_prologue(
    fleet_dir_override: str | None,
    global_config: Any,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Rebalance the host's running runner listeners across repos by demand.

    Runs before the autoscale prologue on purpose: reallocating existing
    capacity is free and instant, so it should be tried before deciding that
    the host needs *more* runners registered. A repo that looks starved often
    only needs a slot another repo is sitting on idle.

    This is host-wide rather than per-repo — the repo set comes from the
    ``.runner`` files under the managed root, not from the fleet registry, so a
    repo with runners on this host is covered whether or not it is registered
    for dispatch. Gated on ``runner_allocation.enabled``.

    Args:
        fleet_dir_override: Optional override for the fleet directory path.
        global_config: Global fleet configuration.
        dry_run: If True, plan without starting or parking anything.

    Returns:
        A list of attention event dicts for aggregation into the fleet digest.
    """
    events: list[dict[str, Any]] = []

    allocation = getattr(global_config, "runner_allocation", None)

    # Unconditional, before any branch. This line is the whole difference
    # between "the prologue ran and declined" and "the prologue was never
    # reached" — and #590 stayed unisolated for hours precisely because every
    # skip path lived inside a branch, so an absent log line was consistent
    # with both readings and could not be used as evidence either way. Logging
    # the resolved decision inputs here, where nothing can short-circuit past
    # them, means the next occurrence names its own cause.
    logger.info(
        "Fleet allocation prologue: entered (enabled=%s, budget=%s, managed_root=%s, "
        "fleet_dir=%s, dry_run=%s)",
        getattr(allocation, "enabled", None),
        getattr(allocation, "max_running_runners", None),
        getattr(allocation, "managed_root", None) or "(unset)",
        fleet_dir(override=fleet_dir_override),
        dry_run,
    )

    def skipped(reason: str) -> list[dict[str, Any]]:
        """Record an anomalous skip in the digest, not only in the log.

        The log is a host-local file that rotates and has been demonstrably
        lossy here; the digest is the structured surface operators actually
        read. A prologue that cannot act for a reason nobody chose belongs in
        the same place its errors go.
        """
        events.append(
            {
                "repo_key": "fleet",
                "type": "runner_allocation_skipped",
                "reason": reason,
            }
        )
        return events

    if global_config is None:
        # No global fleet config supplied at all — this is the documented default
        # of fleet_loop's parameter, not a disagreement between code and config,
        # so it must not be reported as an anomaly. Host-wide allocation has
        # nothing to read here; the entry line above already records that the
        # prologue was reached, which is the distinction #590 needed.
        logger.info("Fleet allocation prologue: not run — no global fleet config was supplied")
        return events
    if allocation is None:
        # The config object has no such section at all. That is not "the
        # operator left it off" — it means this process is holding a config
        # built by different code, or a load failure already fell back to
        # defaults. Either way the feature is silently absent, so say so.
        logger.warning(
            "Fleet allocation prologue: config object has no runner_allocation "
            "section (%s); allocation cannot run in this process",
            type(global_config).__name__,
        )
        return skipped(
            f"config object has no runner_allocation section ({type(global_config).__name__})"
        )
    if not allocation.enabled:
        # INFO, not DEBUG. This is the branch that makes the entire feature inert,
        # and the daemon runs at INFO, so a DEBUG line here is never written at
        # all: during issue #590 a host where allocation never ran was
        # indistinguishable from a converged one, and the absence of a log line
        # could not be used as evidence either way.
        # Name the fleet directory too — it identifies *which* config.yaml
        # governs, which is the thing an operator gets wrong when they set a
        # host-wide knob in a per-repo layer.
        # Deliberately log-only, unlike the other early returns: this is a state
        # the operator chose, so a digest entry every pass would be recurring
        # noise on any host that simply runs without allocation. The entry line
        # above already distinguishes it from "never reached", which was the
        # evidence gap that mattered.
        logger.info(
            "Fleet allocation prologue: not run — runner_allocation.enabled is false "
            "in the resolved config (fleet dir: %s)",
            fleet_dir(override=fleet_dir_override),
        )
        return events

    # Any existing repo root works as the gh working directory: the allocation
    # pass addresses every repo by explicit owner/name slug, so the cwd's git
    # identity is irrelevant. Only auth and a valid directory are needed.
    fleet_json_path = layout.fleet_registry_path(override=fleet_dir_override)
    registry = _load_registry(fleet_json_path)
    anchor_root: Path | None = None
    anchor_state: Path | None = None
    for entry in registry.get("repos", {}).values():
        candidate = Path(entry.get("repo_root", ""))
        if candidate.is_dir():
            anchor_root = candidate
            state_dir = entry.get("state_dir")
            anchor_state = layout.state_file_path(Path(state_dir)) if state_dir else None
            break

    full_pass_interval_seconds = getattr(
        getattr(global_config, "supervisor", None),
        "full_pass_interval_seconds",
        SupervisorConfig().full_pass_interval_seconds,
    )

    if anchor_root is None:
        logger.info("Fleet allocation prologue: no usable repo root in registry, skipping")
        if not dry_run:
            save_allocation_skip(
                fleet_dir(override=fleet_dir_override),
                source=UNATTENDED_ALLOCATION_SOURCE,
                full_pass_interval_seconds=full_pass_interval_seconds,
                skip_reason="no usable repo root in the fleet registry",
            )
        return skipped("no usable repo root in the fleet registry")

    runtime = getattr(global_config, "runtime", None)
    runner_scaling = getattr(global_config, "runner_scaling", None)
    gh = GitHub(repo_root=anchor_root, runtime=runtime, dry_run=False)

    result = run_allocation_pass(
        gh,
        allocation,
        managed_root_fallback=getattr(runner_scaling, "managed_root", "") or "",
        fleet_dir_override=fleet_dir_override,
        state_path=anchor_state,
        dry_run=dry_run,
        source=UNATTENDED_ALLOCATION_SOURCE,
        full_pass_interval_seconds=full_pass_interval_seconds,
    )

    if result.error:
        logger.warning("Fleet allocation prologue: %s", result.error)
        events.append(
            {
                "repo_key": "fleet",
                "type": "runner_allocation_error",
                "error": result.error,
            }
        )
        return events

    # Report the inputs alongside the outcome: "started=0 parked=0" is the correct
    # result for a converged host *and* for one pointed at the wrong managed_root
    # or running under a budget the operator did not intend. Without the budget and
    # root, a healthy no-op and a misconfigured no-op read identically.
    logger.info(
        "Fleet allocation prologue: started=%d parked=%d notes=%d (budget=%d, managed_root=%s)",
        result.started,
        result.parked,
        len(result.notes),
        allocation.max_running_runners,
        allocation.managed_root or getattr(runner_scaling, "managed_root", "") or "(unset)",
    )

    # Only surface an event when something actually moved or a bound was hit;
    # a balanced host should not add noise to every digest.
    if result.started or result.parked or result.notes:
        events.append(
            {
                "repo_key": "fleet",
                "type": "runner_allocation",
                "started": result.started,
                "parked": result.parked,
                "budget": result.plan.budget if result.plan else 0,
                "notes": list(result.notes),
                "dry_run": dry_run,
            }
        )

    for slot in result.results:
        if not slot.ok:
            events.append(
                {
                    "repo_key": "fleet",
                    "type": "runner_allocation_slot_error",
                    "runner": slot.change.runner_name,
                    "action": slot.change.action.value,
                    "message": slot.message,
                }
            )

    return events


def _run_fleet_autoscale_prologue(
    fleet_dir_override: str | None,
    global_config: Any,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Run autoscale decision as a prologue to fleet bash-rats.

    This is an opt-in prologue phase that:
    - Loads fleet-wide runner totals
    - Runs the autoscale decision function
    - Executes scale actions if not in dry-run mode
    - Returns attention events for the digest

    Args:
        fleet_dir_override: Optional override for the fleet directory path.
        global_config: Global fleet configuration.
        dry_run: If True, print decisions without executing.

    Returns:
        A list of attention event dicts for aggregation into the fleet digest.
    """
    events: list[dict[str, Any]] = []

    # Check if autoscale prologue is enabled
    runners_config = getattr(global_config, "runners", None)
    if not runners_config or not getattr(runners_config, "fleet_autoscale_prologue", False):
        return events

    # Check if runner_scaling is enabled
    runner_scaling = getattr(global_config, "runner_scaling", None)
    if not runner_scaling or not runner_scaling.enabled:
        logger.info("Fleet autoscale prologue: runner_scaling not enabled, skipping")
        return events

    # Pick a representative repo with runner_scaling enabled. Its runtime config
    # is used for the fleet-wide runner count (all repos share the same retry
    # knobs) and for the subsequent autoscale observation/actions.
    fleet_json_path = layout.fleet_registry_path(override=fleet_dir_override)
    registry = _load_registry(fleet_json_path)
    repos_map = registry.get("repos", {})

    representative_repo = None
    for repo_key, entry in repos_map.items():
        repo_root = Path(entry.get("repo_root"))
        if not repo_root.exists():
            continue

        try:
            # Load per-repo config to check if runner_scaling is enabled
            explicit_cfg = entry.get("config_path")
            config = load_layered_config(
                repo_root,
                Path(explicit_cfg) if explicit_cfg else None,
                fleet_dir_override=fleet_dir_override,
            )
            if config.runner_scaling.enabled:
                representative_repo = (repo_key, repo_root, config)
                break
        except (ConfigError, GitHubError):
            continue

    if not representative_repo:
        logger.info("Fleet autoscale prologue: no repo with runner_scaling enabled found")
        return events

    repo_key, repo_root, config = representative_repo

    # Load fleet-wide runner totals using the representative repo's runtime config
    total_runners, total_busy_runners, skipped_repos = count_fleet_runners(
        fleet_dir_override, runtime=config.runtime
    )
    fleet_totals = FleetTotals(
        total_runners=total_runners,
        total_busy_runners=total_busy_runners,
    )

    paths = runtime_paths(repo_root, config.runtime.state_dir)
    gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=dry_run)

    # Observe current pool state
    state = observe_runner_pool(gh, config.runner_scaling, state_dir=paths.root, dry_run=dry_run)

    # Check cooldown and idle duration
    in_cooldown = is_in_cooldown(paths.root, config.runner_scaling.cooldown_minutes)
    is_idle_for_duration = is_pool_idle_for_minutes(
        paths.root, config.runner_scaling.idle_scale_down_minutes
    )

    # Run the pure decision function
    decision = decide_autoscale(
        state,
        config.runner_scaling,
        fleet_totals=fleet_totals,
        in_cooldown=in_cooldown,
        is_idle_for_duration=is_idle_for_duration,
    )

    logger.info(
        f"Fleet autoscale prologue: decision={decision.action.value}({decision.count}) reason={decision.reason}"
    )

    # In dry-run mode, just log the decision
    if dry_run:
        events.append(
            {
                "repo_key": "fleet",
                "type": "autoscale_decision",
                "action": decision.action.value,
                "count": decision.count,
                "reason": decision.reason,
                "dry_run": True,
            }
        )
        return events

    # Execute the decision
    if decision.action == ScaleAction.UP:
        result = provision_runner(
            gh,
            config.runner_scaling,
            state.busy_runners,
            dry_run=False,
        )
        if result.ok:
            record_scale_event(paths.root, "up")
            events.append(
                {
                    "repo_key": repo_key,
                    "type": "autoscale_up",
                    "runner_name": result.runner_name,
                    "reason": decision.reason,
                }
            )
        else:
            events.append(
                {
                    "repo_key": repo_key,
                    "type": "autoscale_error",
                    "action": "up",
                    "error": result.error,
                    "reason": decision.reason,
                }
            )
    elif decision.action == ScaleAction.DOWN:
        managed_root = Path(config.runner_scaling.managed_root)
        if managed_root.exists():
            removed_count, errors = scale_down_idle_runners(
                managed_root,
                config.runner_scaling.runner_dir_prefix,
                gh,
                config.runner_scaling,
                paths.root,
                dry_run=False,
            )
            events.append(
                {
                    "repo_key": repo_key,
                    "type": "autoscale_down",
                    "removed_count": removed_count,
                    "errors": errors,
                    "reason": decision.reason,
                }
            )
        else:
            events.append(
                {
                    "repo_key": repo_key,
                    "type": "autoscale_error",
                    "action": "down",
                    "error": f"managed_root does not exist: {managed_root}",
                    "reason": decision.reason,
                }
            )

    return events


def _extract_attention_events(
    repo_key: str,
    result: CommandResult,
) -> list[dict[str, Any]]:
    """Extract attention-worthy events from a per-repo CommandResult.

    This reads only the already-returned CommandResult.data from each per-repo
    loop() call (e.g. stalled, errors, health-transition fields) and does not
    re-query anything.

    Args:
        repo_key: The repo key for this result.
        result: The CommandResult from a per-repo loop() call.

    Returns:
        A list of attention event dicts for aggregation into the fleet digest.
    """
    events: list[dict[str, Any]] = []
    data = result.data

    # Extract stalled sessions
    stalled = data.get("stalled", [])
    for stall in stalled:
        events.append(
            {
                "repo_key": repo_key,
                "type": "stalled",
                "session_id": stall.get("session_id"),
                "issue_number": stall.get("issue_number"),
                "reason": stall.get("reason"),
            }
        )

    # Extract errors
    errors = data.get("errors", [])
    for error in errors:
        events.append(
            {
                "repo_key": repo_key,
                "type": "error",
                "pr": error.get("pr"),
                "issue_number": error.get("issue") or error.get("pr"),
                "error": error.get("error"),
            }
        )

    # Extract launch failures from dispatch lanes. `loop()` nests per-stage
    # CommandResult data under "dispatch"/"dispatch_rework"/"dispatch_reviews";
    # `fleet_loop(work_only=True)` passes a dispatch sub-dict directly. Walk both
    # the result itself and the nested sub-results so worker/rework/reviewer
    # launch failures reach the fleet attention digest.
    launch_failures = _collect_launch_failures(repo_key, data)
    events.extend(launch_failures)

    # Extract live-worker redispatch averted outcomes (issue #506)
    # These are reported at the top level for work_only dispatch() and nested
    # under "dispatch" for full loop() results.
    averted = data.get("live_worker_redispatch_averted", [])
    if not averted and "dispatch" in data:
        averted = data["dispatch"].get("live_worker_redispatch_averted", [])
    for event in averted:
        events.append(
            {
                "repo_key": repo_key,
                "type": "live_worker_redispatch_averted",
                "issue_number": event.get("issue_number"),
                "reason": event.get("probe_result"),
                "pid": event.get("pid"),
                "adapter_kind": event.get("adapter_kind", "unknown"),
            }
        )
    # Extract review verdict reaper results. `dispatch_reviews` carries
    # `recorded_verdicts`/`missed_verdicts` lists; `loop()` nests them under the
    # "dispatch_reviews" key. Surfacing both recorded and missed verdicts in the
    # digest makes a silent 0%-recording-rate regression visible to the heartbeat.
    verdict_events = _collect_review_verdict_events(repo_key, data)
    events.extend(verdict_events)

    # Extract health transitions (if present from #161/#165)
    health_transitions = data.get("health_transitions", [])
    for transition in health_transitions:
        events.append(
            {
                "repo_key": repo_key,
                "type": "health_transition",
                "session_id": transition.get("session_id"),
                "from_state": transition.get("from_state"),
                "to_state": transition.get("to_state"),
            }
        )

    # Extract lock/deferral skips (state_lock_busy, supervisor_lock_held,
    # graphql_rate_limit) surfaced by OrchestratorApp public methods. loop()
    # nests the per-stage CommandResult data under "intake"/"dispatch"/
    # "dispatch_rework"/"dispatch_reviews", so we collect skip reasons from
    # the top-level result as well as those nested sub-results.
    skip_reasons = _collect_skip_reasons(data)
    for reason in sorted(skip_reasons):
        events.append(
            {
                "repo_key": repo_key,
                "type": "skipped",
                "reason": reason,
            }
        )

    return events


_SKIP_REASONS = frozenset({"state_lock_busy", "supervisor_lock_held", "graphql_rate_limit"})


def _collect_skip_reasons(data: Any) -> set[str]:
    """Collect all lock/deferral skip reasons from a result and its nested sub-results."""
    reasons: set[str] = set()
    _add_skip_reasons(data, reasons)
    for sub_key in ("intake", "dispatch", "dispatch_rework", "dispatch_reviews"):
        sub_data = data.get(sub_key) if isinstance(data, dict) else None
        if isinstance(sub_data, dict):
            _add_skip_reasons(sub_data, reasons)
    return reasons


def _add_skip_reasons(data: dict[str, Any], reasons: set[str]) -> None:
    """Add any skip reason present in a flat result dict to the set."""
    skip_reason = data.get("reason") or data.get("deferred_reason")
    if data.get("skipped") or data.get("state_lock_busy") or skip_reason in _SKIP_REASONS:
        reasons.add(skip_reason or "state_lock_busy")


def _collect_review_verdict_events(repo_key: str, data: Any) -> list[dict[str, Any]]:
    """Collect review verdict reaper results from a result and nested dispatch_reviews.

    ``dispatch_reviews`` returns ``recorded_verdicts`` and ``missed_verdicts``;
    ``loop()`` nests them under the ``dispatch_reviews`` key. Both recorded
    (heartbeat confirmation) and missed (regression signal) verdicts become
    attention events so a future silent 0%-recording-rate regression is caught.
    """
    events: list[dict[str, Any]] = []
    if isinstance(data, dict):
        _add_review_verdict_events(repo_key, data, events)
        sub_data = data.get("dispatch_reviews")
        if isinstance(sub_data, dict):
            _add_review_verdict_events(repo_key, sub_data, events)
    return events


def _add_review_verdict_events(
    repo_key: str,
    data: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:
    """Add recorded/missed review verdict events from a flat result dict."""
    for verdict in data.get("recorded_verdicts", []):
        if not isinstance(verdict, dict):
            continue
        events.append(
            {
                "repo_key": repo_key,
                "type": "review_verdict_recorded",
                "issue_number": verdict.get("issue") or verdict.get("pr"),
                "pr": verdict.get("pr"),
                "decision": verdict.get("decision"),
            }
        )

    for verdict in data.get("missed_verdicts", []):
        if not isinstance(verdict, dict):
            continue
        events.append(
            {
                "repo_key": repo_key,
                "type": "review_verdict_missed",
                "issue_number": verdict.get("issue") or verdict.get("pr"),
                "pr": verdict.get("pr"),
                "reason": verdict.get("reason", "verdict not recorded"),
            }
        )


def _collect_launch_failures(repo_key: str, data: Any) -> list[dict[str, Any]]:
    """Collect worker/rework/reviewer launch failures from a result and its nested sub-results.

    Mirrors ``_collect_skip_reasons`` but walks the ``dispatch``/``dispatch_rework``/
    ``dispatch_reviews`` sub-dicts that carry per-issue/per-PR failure text.
    """
    failures: list[dict[str, Any]] = []
    if isinstance(data, dict):
        _add_launch_failures(repo_key, data, failures)
        for sub_key in ("dispatch", "dispatch_rework", "dispatch_reviews"):
            sub_data = data.get(sub_key)
            if isinstance(sub_data, dict):
                _add_launch_failures(repo_key, sub_data, failures)
    return failures


def _add_launch_failures(
    repo_key: str,
    data: dict[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    """Add any worker/rework/reviewer launch failure present in a flat result dict.

    ``dispatch`` and ``dispatch_rework`` store issue-keyed ``failures`` maps;
    ``dispatch_reviews`` stores a PR-keyed ``failed`` list. Both shapes are
    normalized to ``{"repo_key": ..., "type": "error", "issue_number"/"pr": ..., "error": ...}``.

    Issues deferred by the concurrency cap are not launch failures, so they are
    excluded from ``failures`` maps via the ``deferred_by_concurrency`` list.
    """
    deferred_issue_numbers: set[int] = set()
    for d in data.get("deferred_by_concurrency", []):
        try:
            deferred_issue_numbers.add(int(d))
        except (TypeError, ValueError):
            continue

    failures_map = data.get("failures")
    if isinstance(failures_map, dict):
        for key, error in failures_map.items():
            if not isinstance(error, str):
                continue
            try:
                issue_number = int(key)
            except (TypeError, ValueError):
                continue
            if issue_number in deferred_issue_numbers:
                continue
            failures.append(
                {
                    "repo_key": repo_key,
                    "type": "error",
                    "issue_number": issue_number,
                    "error": error,
                }
            )

    failed_list = data.get("failed")
    if isinstance(failed_list, list):
        for entry in failed_list:
            if not isinstance(entry, dict):
                continue
            error = entry.get("error")
            if not error:
                continue
            event: dict[str, Any] = {
                "repo_key": repo_key,
                "type": "error",
                "error": error,
            }
            issue_number = entry.get("issue") or entry.get("issue_number")
            pr = entry.get("pr")
            if issue_number is not None:
                try:
                    event["issue_number"] = int(issue_number)
                except (TypeError, ValueError):
                    pass
            if pr is not None:
                try:
                    event["pr"] = int(pr)
                except (TypeError, ValueError):
                    pass
            if "issue_number" in event or "pr" in event:
                failures.append(event)


# Event types whose AttentionEntry represents a persistent health state of an
# issue/worker (STALLED/ERROR/REPAIRED-style). Only these are subject to
# cross-pass transition dedup: a persistent ERROR must not re-fire with
# ``previous_health: null`` every pass (issue #554). Occurrence-style events
# (review_verdict_recorded, review_verdict_missed, skipped,
# live_worker_redispatch_averted, and the operational fallback entries) are
# one-shot confirmations whose health string is constant by construction, so
# deduping them would collapse the recorded-verdict heartbeat and silently drop
# repeat missed-verdict signals (PR #669 review).
_PERSISTENT_HEALTH_EVENT_TYPES = frozenset({"stalled", "error", "health_transition"})


def _fleet_health_state_path(fleet_dir_override: str | None) -> Path:
    """Return the fleet-level notify health baseline sidecar path.

    This sidecar persists the last-known health per (adapter_kind, issue_number)
    so the fleet digest can emit only on real transitions instead of re-firing
    every pass with ``previous_health: null`` (issue #554). It lives in the
    fleet directory alongside ``fleet.json``.
    """
    return layout.notify_health_state_path(override=fleet_dir_override)


def _load_fleet_health_state(path: Path) -> dict[str, str]:
    """Load the fleet health baseline sidecar.

    Returns an empty dict when the file is missing or unparseable (a corrupt
    sidecar is non-fatal: the worst case is one extra transition emission on
    the next pass, which is exactly the degraded mode we are fixing away from).
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, LookupError, ValueError, OSError):
        logger.warning("Fleet health state %s unreadable; starting fresh", path)
        return {}
    issues = data.get("issues") if isinstance(data, dict) else None
    if not isinstance(issues, dict):
        return {}
    # Coerce values to str; keys are already "adapter_kind:issue_number" strings.
    return {str(k): str(v) for k, v in issues.items() if isinstance(v, str)}


def _save_fleet_health_state(path: Path, issues: dict[str, str]) -> None:
    """Atomically persist the fleet health baseline sidecar.

    Temp-file + ``replace()`` per the project's atomic-write invariant. Warns
    on fleet-dir virtualization (issue #624) but never blocks the write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    warn_fleet_dir_virtualization_on_write(path.parent, context="writing notify_health_state.json")
    payload = {"version": 1, "generated_at": utc_now(), "issues": issues}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def _filter_fleet_health_transitions(
    entries: list[AttentionEntry],
    state_file: Path,
    *,
    persistent_mask: list[bool] | None = None,
    observed_repo_keys: frozenset[str] | None = None,
) -> list[AttentionEntry]:
    """Stateful filter: keep only persistent-health entries whose health changed.

    Reads the fleet health baseline sidecar (``notify_health_state.json``)
    mapping ``"adapter_kind:issue_number"`` to the last-emitted health. For
    each *persistent* entry, only keeps it when the health differs from the
    persisted baseline, setting ``previous_health`` to that prior value.
    Updates and atomically persists the new baselines.

    This is the fleet-level analogue of ``workflow._build_attention_digest``'s
    per-issue transition dedup. Without it, every fleet pass re-emits the same
    ERROR/STALLED entries with ``previous_health: null`` (issue #554).

    ``persistent_mask`` (parallel to ``entries``) marks which entries represent
    a persistent health state subject to cross-pass dedup. Entries marked
    ``False`` are occurrence-style events (review_verdict_recorded/missed,
    skipped, live_worker_redispatch_averted, operational fallback) and pass
    through unchanged every call — they never consult nor update the baseline,
    so a constant-health confirmation keeps firing as a heartbeat (PR #669
    review). When ``persistent_mask`` is ``None`` every entry is treated as
    persistent (the self-deploy ERROR/REPAIRED path, which is always
    persistent).

    Entries are processed in order so a within-pass health change (e.g.
    ERROR then OK for the same issue) emits both transitions; the persisted
    baseline ends at the final health.

    ``observed_repo_keys`` (issue #817 item 2) reconciles the baseline
    against issues that were genuinely re-checked this pass and found
    healthy. Persistent-health entries only ever exist for *unhealthy*
    observations (stalled/error/health_transition) -- a healthy issue
    produces no event at all, so without this the baseline can only ever
    move *into* an unhealthy value and never back out, latching every
    tracked issue at its first failure forever (the same defect item 1 fixes
    for self-deploy, whose producer always has a distinct success value to
    feed; issue health has no such value to hang a recovery entry on). After
    the loop above, any *other* persisted key whose ``adapter_kind`` (the
    part before the first ``:``) is in ``observed_repo_keys`` -- meaning
    that repo's lane ran to completion this pass, so every issue it tracks
    was genuinely looked at -- and that was not itself re-affirmed unhealthy
    in this same call is silently cleared, not re-emitted as a recovery
    entry (there is no "issue confirmed healthy" event to attach one to).
    This does not create a false transition; it lets the *next* unhealthy
    observation for that key read ``previous_health: null`` and emit as a
    fresh incident instead of being suppressed by a latch that could never
    leave its last value. A repo lane that did not run this pass (missing
    repo_root, supervisor lock held, an unhandled per-repo exception) is not
    in ``observed_repo_keys``, so its keys are left untouched -- absence of
    a check is not evidence of health.
    """
    emitted: list[AttentionEntry] = []
    touched_keys: set[str] = set()
    # Ensure the parent directory exists before state_lock tries to create the
    # sibling .lock file (advisory_file_lock touches it directly).
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_lock(state_file):
        baselines = _load_fleet_health_state(state_file)
        for idx, entry in enumerate(entries):
            is_persistent = persistent_mask[idx] if persistent_mask is not None else True
            if not is_persistent:
                # Occurrence-style event: emit every time and leave the
                # persisted baseline untouched (it tracks persistent health
                # only, so a one-shot confirmation must not poison it).
                emitted.append(entry)
                continue
            key = f"{entry.adapter_kind}:{entry.issue_number}"
            touched_keys.add(key)
            last = baselines.get(key)
            if last == entry.health:
                continue
            emitted.append(replace(entry, previous_health=last))
            baselines[key] = entry.health
        if observed_repo_keys:
            for key in list(baselines):
                if key in touched_keys:
                    continue
                adapter_kind = key.split(":", 1)[0]
                if adapter_kind in observed_repo_keys:
                    del baselines[key]
        _save_fleet_health_state(state_file, baselines)
    return emitted


def _emit_fleet_transition(
    notify_config: Any,
    entry: AttentionEntry,
    fleet_dir_override: str | None,
    *,
    persistent: bool = True,
) -> None:
    """Emit a single fleet attention entry as a digest.

    By default this wraps ``_filter_fleet_health_transitions`` so a
    persistent condition (e.g. a recurring self-deploy ERROR) does not
    re-fire with ``previous_health: null`` every supervisor pass (issue #554).
    Only emits when the entry represents a real transition vs the persisted
    baseline.

    Set ``persistent=False`` for occurrence-style events (e.g. supervisor
    lifecycle kills) that must alert every time instead of being deduped.
    """
    state_file = _fleet_health_state_path(fleet_dir_override)
    persistent_mask = [persistent] if persistent is not None else None
    filtered = _filter_fleet_health_transitions(
        [entry], state_file, persistent_mask=persistent_mask
    )
    if not filtered:
        return
    digest = AttentionDigest(
        generated_at=utc_now(),
        repo="fleet",
        transitions=tuple(filtered),
    )
    emit_digest(notify_config, digest)


def _build_fleet_attention_digest(
    attention_events: list[dict[str, Any]],
    state_file: Path | None = None,
    observed_repo_keys: frozenset[str] | None = None,
) -> AttentionDigest:
    """Convert fleet-aggregated event dicts into a single AttentionDigest.

    Fleet events are already-flattened per-repo dicts (stalled / error /
    health_transition / review_verdict_recorded / review_verdict_missed)
    produced by ``_extract_attention_events``. This maps
    each one onto the real #166 ``AttentionEntry`` schema so the fleet pass
    can go through the same ``emit_digest`` sink pipeline as a single-repo
    pass, rather than re-deriving its own notification format.

    ``issue_number`` is required by ``AttentionEntry``; events that carry no
    issue number (e.g. PR errors) fall back to ``-1`` as a sentinel so they
    still surface in the digest instead of being silently dropped.

    When ``state_file`` is provided, the entries are filtered through
    ``_filter_fleet_health_transitions`` so persistent-health entries
    (``stalled``/``error``/``health_transition``) emit only on real transitions
    vs the last-persisted baseline, while occurrence-style entries
    (``review_verdict_recorded``/``review_verdict_missed``, ``skipped``,
    ``live_worker_redispatch_averted``, operational fallback) pass through
    every pass — collapsing those would defeat the recorded-verdict heartbeat
    (PR #669 review). Without ``state_file`` every pass re-fires the same
    entries with ``previous_health: null`` (issue #554). The returned digest
    always carries the post-filter entries; callers that pass ``state_file``
    should still check ``digest.transitions`` for emptiness before emitting.

    ``observed_repo_keys`` (issue #817 item 2) is forwarded to
    ``_filter_fleet_health_transitions`` so a persistent baseline key whose
    issue was genuinely re-checked this pass and found healthy is reconciled
    away instead of latching at its last unhealthy value forever -- see that
    function's docstring for the exact semantics.
    """
    entries: list[AttentionEntry] = []
    # Parallel to ``entries``: True for persistent-health event types subject
    # to cross-pass dedup, False for occurrence-style events that emit every
    # pass. See ``_PERSISTENT_HEALTH_EVENT_TYPES``.
    persistent_mask: list[bool] = []
    for event in attention_events:
        event_type = event["type"]
        is_persistent = event_type in _PERSISTENT_HEALTH_EVENT_TYPES
        if event_type == "stalled":
            entries.append(
                AttentionEntry(
                    issue_number=event.get("issue_number") or -1,
                    adapter_kind=event["repo_key"],
                    health="STALLED",
                    previous_health=None,
                    last_log_line=event.get("reason"),
                    pid=None,
                )
            )
            persistent_mask.append(is_persistent)
        elif event_type == "error":
            entries.append(
                AttentionEntry(
                    issue_number=event.get("issue_number") or event.get("pr") or -1,
                    adapter_kind=event["repo_key"],
                    health="ERROR",
                    previous_health=None,
                    last_log_line=event.get("error"),
                    pid=None,
                )
            )
            persistent_mask.append(is_persistent)
        elif event_type == "health_transition":
            entries.append(
                AttentionEntry(
                    issue_number=-1,
                    adapter_kind=event["repo_key"],
                    health=event.get("to_state") or "UNKNOWN",
                    previous_health=event.get("from_state"),
                    last_log_line=None,
                    pid=None,
                )
            )
            persistent_mask.append(is_persistent)
        elif event_type == "skipped":
            entries.append(
                AttentionEntry(
                    issue_number=-1,
                    adapter_kind=event["repo_key"],
                    health="SKIPPED",
                    previous_health=None,
                    last_log_line=event.get("reason"),
                    pid=None,
                )
            )
            persistent_mask.append(is_persistent)
        elif event_type == "live_worker_redispatch_averted":
            entries.append(
                AttentionEntry(
                    issue_number=event.get("issue_number") or -1,
                    adapter_kind=event.get("adapter_kind", event["repo_key"]),
                    health="DISPATCH_AVERTED",
                    previous_health=None,
                    last_log_line=event.get("reason"),
                    pid=event.get("pid"),
                )
            )
            persistent_mask.append(is_persistent)
        elif event_type == "review_verdict_recorded":
            entries.append(
                AttentionEntry(
                    issue_number=event.get("issue_number") or event.get("pr") or -1,
                    adapter_kind=event["repo_key"],
                    health="OK",
                    previous_health=None,
                    last_log_line=f"{event.get('decision')} recorded for PR {event.get('pr')}",
                    pid=None,
                )
            )
            persistent_mask.append(is_persistent)
        elif event_type == "review_verdict_missed":
            entries.append(
                AttentionEntry(
                    issue_number=event.get("issue_number") or event.get("pr") or -1,
                    adapter_kind=event["repo_key"],
                    health="ERROR",
                    previous_health=None,
                    last_log_line=event.get("reason"),
                    pid=None,
                )
            )
            persistent_mask.append(is_persistent)
        elif event_type == "runner_allocation":
            # Deliberately not in the digest — unlike the accidental drops the
            # fallback below exists to prevent. This is the allocator's *success*
            # event, and the attention digest is for things needing attention.
            #
            # It also cannot be left to the fallback: the prologue emits it when
            # anything moved *or any note was produced*, and the notes include
            # standing advisory conditions that persist for as long as the condition
            # does ("holding 4 surplus slot(s) — slack for 0/3 pass(es)", "demand 7
            # exceeds its 2 registered runner(s)"). Verified against this host's
            # events.db: every recorded pass carried a note while moving no slots, so
            # rendering it would put a near-identical entry in every 5-minute digest.
            # The event stays in events.db and the prologue logs its inputs at INFO,
            # which is where standing conditions belong.
            continue
        else:
            # Visible by default. This chain used to end here, so any event type
            # without an explicit branch was dropped silently — which is how the
            # prologue's own ``runner_allocation_error`` never reached the digest
            # despite being emitted correctly (issue #590). Making the fallback
            # generic means adding an event type can no longer make it invisible;
            # forgetting a branch costs a less specific entry, not a lost signal.
            entries.append(
                AttentionEntry(
                    issue_number=event.get("issue_number") or event.get("pr") or -1,
                    adapter_kind=event.get("repo_key", "fleet"),
                    health="ERROR" if "error" in event_type else "INFO",
                    previous_health=None,
                    last_log_line=event.get("error") or event.get("reason") or event_type,
                    pid=event.get("pid"),
                )
            )
            persistent_mask.append(is_persistent)

    if state_file is not None:
        entries = _filter_fleet_health_transitions(
            entries,
            state_file,
            persistent_mask=persistent_mask,
            observed_repo_keys=observed_repo_keys,
        )

    return AttentionDigest(
        generated_at=utc_now(),
        repo="fleet",
        transitions=tuple(entries),
    )


def _lane_failure_state_path(repo_root: Path, entry: dict[str, Any]) -> Path:
    """Resolve the per-repo ``state.json`` path for a lane that failed to start.

    A lane failure (e.g. ``load_layered_config`` raising ``ConfigError``) can
    happen before ``runtime_paths()`` is reachable, so this cannot use the
    configured ``runtime.state_dir`` the way a healthy pass does. Instead it
    mirrors ``_take_fleet_snapshot``'s precedent: prefer the fleet registry's
    recorded ``state_dir`` (written by a prior successful ``touch_repo()``),
    falling back to the conventional default for a repo that has never
    registered successfully.

    Note this can diverge from ``runtime_paths(...).state_file`` for a repo
    whose config overrides ``runtime.state_dir`` to a non-default path *and*
    has no registry ``state_dir`` recorded yet (e.g. its very first pass) —
    the event would be recorded at the default location, and a doctor check
    reading the configured path would not find it until the repo registers
    successfully once. This is the same divergence class layout.py's
    docstrings warn about; it is not fully closable here since the whole
    point of this helper is to work when config load has failed.
    """
    state_dir_str = entry.get("state_dir")
    state_root = Path(state_dir_str) if state_dir_str else layout.default_state_root(repo_root)
    return layout.state_file_path(state_root)


def _record_lane_failure_event(
    repo_root: Path,
    repo_key: str,
    entry: dict[str, Any],
    error_message: str,
) -> None:
    """Durably record a lane-startup failure to the repo's own events.db (#6-G).

    This is best-effort and must never escape: ``log_event`` already guards
    its own sqlite calls, but the directory-creation call it makes before
    that guard is not caught by that guard's exception type, so this wraps
    the whole call in its own try/except. A failure to *record* a lane
    failure must never itself break the per-repo isolation boundary this is
    called from (D-4) — the caller has already logged the real error via
    ``logger.exception`` regardless of what happens here.
    """
    try:
        state_path = _lane_failure_state_path(repo_root, entry)
        log_event(
            state_path,
            "fleet_pass_config_error",
            {"repo_key": repo_key, "error": error_message},
            repo=repo_key,
        )
    except Exception:
        logger.exception("Failed to record lane failure event for repo %s", repo_key)


def fleet_loop(
    fleet_dir_override: str | None = None,
    global_config: Any = None,  # GlobalConfig from #159, but we don't have the type yet
    *,
    repos: tuple[str, ...] | None = None,
    limit: int | None = None,
    merge: bool | None = None,
    dry_run: bool = False,
    work_only: bool = False,
) -> CommandResult:
    """Run a fleet pass across all (or selected) registered repos.

    This composes the existing single-repo pass (intake -> dispatch -> review -> merge)
    across multiple repos under one global concurrency budget, ending in one
    consolidated attention digest emitted via the notifier.

    Args:
        fleet_dir_override: Optional override for the fleet directory path.
        global_config: GlobalConfig from #159 (optional, for #166 notifier integration).
        repos: Optional tuple of repo keys to select explicitly. If None, all
            registered repos are processed in oldest-last_seen order.
        limit: Optional per-repo limit for dispatch.
        merge: Whether to merge ready PRs (None = use config default).
        dry_run: If True, pass dry_run to every per-repo GitHub/OrchestratorApp.
        work_only: If True, run dispatch-only path (no review/merge), analogous
            to single-repo 'work' vs 'bash-rats'.

    Returns:
        A CommandResult with per-repo results and the consolidated digest.
    """
    # Load fleet registry with state_lock guard
    fleet_json_path = layout.fleet_registry_path(override=fleet_dir_override)
    registry = _load_registry(fleet_json_path)

    # Select repos in the appropriate order
    selected = _select_repos(registry, repos)

    per_repo_results: dict[str, CommandResult] = {}
    attention_events: list[dict[str, Any]] = []
    # repo_keys whose lane actually ran app.loop()/app.dispatch() to
    # completion this pass -- i.e. every issue that repo tracks was genuinely
    # looked at. Threaded into _build_fleet_attention_digest so a healthy
    # issue's stale health-baseline entry can be reconciled away instead of
    # latching forever (issue #817 item 2). A repo that was skipped (missing
    # repo_root, supervisor lock held) or raised is deliberately excluded --
    # absence of a check is not evidence of health.
    observed_repo_keys: set[str] = set()
    orphan_sweep_calls = 0
    # Collect each selected repo's raw layered config so the api-worker fleet
    # report below can reuse it instead of re-loading every config each pass
    # (issue #483 review: redundant per-repo config reload each fleet pass).
    # The raw config is captured before the notify-silencing replace() below;
    # the report only reads api_worker fields, which that replace never touches.
    loaded_configs: dict[str, OrchestratorConfig] = {}

    # Run runner prologues if enabled (only for full loop, not work-only).
    # Allocation first: moving an idle slot to a starved repo is free, so it
    # runs before autoscale decides the host needs more runners registered.
    if not work_only:
        attention_events.extend(
            _run_fleet_allocation_prologue(fleet_dir_override, global_config, dry_run)
        )
        autoscale_events = _run_fleet_autoscale_prologue(
            fleet_dir_override, global_config, dry_run
        )
        attention_events.extend(autoscale_events)

    for repo_key, entry in selected:
        # Default to "" rather than None: a registry entry missing repo_root
        # entirely would make Path(None) raise, where the is_dir() check below
        # already has the right answer for a bad path.
        repo_root = Path(entry.get("repo_root") or "")
        if not repo_root.is_dir():
            # Tolerate vanished/moved repo (#169 precedent)
            per_repo_results[repo_key] = CommandResult(
                False, f"repo_root missing, skipped: {repo_root}", {}
            )
            continue

        try:
            # Load per-repo config through the global fleet layer so a fleet-wide
            # default (e.g. fleet.global_max_concurrent_sessions, watchdog knobs)
            # set once in <fleet_dir>/config.yaml applies here too; the per-repo
            # orchestrator.config.yaml still wins on any overlapping key.
            explicit_cfg = entry.get("config_path")
            config = load_layered_config(
                repo_root,
                Path(explicit_cfg) if explicit_cfg else None,
                fleet_dir_override=fleet_dir_override,
            )
            # Cache the raw layered config for the api-worker fleet report
            # (captured before the notify-silencing replace below).
            loaded_configs[repo_key] = config
            # Fleet mode is the single notification authority: the aggregate
            # digest below emits once for the whole pass. Silence per-repo
            # dispatch()/loop() emission so one health transition doesn't fire
            # both a per-repo and a fleet-level notification.
            config = replace(config, notify=replace(config.notify, enabled=False))
            paths = runtime_paths(repo_root, config.runtime.state_dir)

            # Non-blocking supervisor lock: fleet passes must be mutually exclusive
            # with a supervised bash-rats loop on the same repo to avoid double-
            # dispatching through the governor's read-then-launch window.
            lock = try_acquire_supervisor_lock(layout.supervisor_lock_path(paths.root))
            if lock is None:
                per_repo_results[repo_key] = CommandResult(
                    True,
                    "supervisor lock held, skipped",
                    {"skipped": True, "reason": "supervisor_lock_held"},
                )
                attention_events.append(
                    {
                        "repo_key": repo_key,
                        "type": "skipped",
                        "reason": "supervisor_lock_held",
                    }
                )
                continue

            try:
                gh = GitHub(repo_root=repo_root, runtime=config.runtime, dry_run=dry_run)
                app = OrchestratorApp(
                    repo_root,
                    paths,
                    config,
                    gh,
                    dry_run=dry_run,
                    fleet_dir_override=fleet_dir_override,
                )

                # Call the appropriate per-repo method
                if work_only:
                    # Dispatch-only path (worker dispatch + optional review dispatch)
                    result = app.dispatch(limit)
                    if config.review_dispatch.enabled:
                        review_dispatch_result = app.dispatch_reviews(limit)
                        ok = result.ok and review_dispatch_result.ok
                        message = (
                            "work-only dispatch: "
                            f"workers={result.data.get('selected_count', 0)}, "
                            f"reviews={review_dispatch_result.data.get('selected_count', 0)}"
                        )
                        combined_data = dict(result.data)
                        combined_data["dispatch_reviews"] = review_dispatch_result.data
                        result = CommandResult(ok, message, combined_data)
                else:
                    # Full loop (intake -> dispatch -> review -> merge)
                    result = app.loop(limit, merge=merge)

                per_repo_results[repo_key] = result
                attention_events.extend(_extract_attention_events(repo_key, result))
                observed_repo_keys.add(repo_key)

                # Count orphan sweep calls (B6a interaction)
                # Each loop() call internally triggers orphan sweep via
                # _sweep_orphan_processes_for_dead_sessions
                # We count this as a metric for the follow-up optimization
                if not work_only:
                    orphan_sweep_calls += 1
            finally:
                lock.release()

        except Exception as exc:
            # Per-repo isolation: catch any provider/logic failure at the
            # iteration boundary and continue. Keep the rest of the fleet
            # pass alive instead of crashing on one unclassified exception.
            # The exception type is part of the message and the full traceback
            # goes to the log — an unclassified failure must stay diagnosable.
            error_message = f"{type(exc).__name__}: {exc}"
            per_repo_results[repo_key] = CommandResult(
                False, f"fleet pass error: {error_message}", {}
            )
            logger.exception("Error processing repo %s", repo_key)
            # #6-G: the two lines above are an in-process dict and a line in a
            # dated flat-text log — neither reaches events.db, state.json, or
            # the fleet digest, since _extract_attention_events() above only
            # runs after a successful app.loop(). A repo whose lane fails on
            # every pass (e.g. a config-load ConfigError, cw#... 2026-07-29)
            # was previously invisible to everything except that log line.
            # Reuse the existing "error" AttentionEntry branch (health=ERROR,
            # already desktop-toast-eligible via _DESKTOP_SEVERITIES) so this
            # works on exactly the path where app.loop() never ran, and
            # durably record it so `charlie doctor` and
            # query_events(level="error") can see it even after the digest's
            # cross-pass dedup stops re-emitting the same standing failure.
            attention_events.append(
                {"repo_key": repo_key, "type": "error", "error": error_message}
            )
            _record_lane_failure_event(repo_root, repo_key, entry, error_message)

    # Call the notifier digest sink exactly once per fleet pass, via the real
    # #166 notify.py implementation (AttentionDigest + emit_digest).
    notify_config = getattr(global_config, "notify", None) if global_config else None
    digest: dict[str, Any] = {
        "events": attention_events,
        "count": len(attention_events),
        "orphan_sweep_calls": orphan_sweep_calls,
        "emitted": False,
    }
    # Build the digest whenever notify is on, then gate the *emission* on the
    # built digest's transitions (the inner ``if attention_digest.transitions``
    # added by #669). The previous outer ``and attention_events`` test was
    # redundant with that inner gate and tested a different, rawer signal — a
    # converged pass (every event routed to an explicit ``continue``, e.g. a
    # healthy ``runner_allocation``) has a non-empty ``attention_events`` list
    # but an empty ``transitions`` tuple, so the two tests disagreed on whether
    # to enter this block. #669's inner gate already prevented the
    # ``emit_digest(transitions=())`` empty-envelope call for that case; this
    # drop removes the contradictory outer condition rather than fixing a
    # currently-reproducing emission (issue #610). The ``notify_config`` check
    # stays outermost so no digest is built when notify is off.
    if notify_config is not None and getattr(notify_config, "enabled", False):
        health_state_file = _fleet_health_state_path(fleet_dir_override)
        attention_digest = _build_fleet_attention_digest(
            attention_events,
            state_file=health_state_file,
            observed_repo_keys=frozenset(observed_repo_keys),
        )
        if attention_digest.transitions:
            notify_result = emit_digest(notify_config, attention_digest)
            digest["emitted"] = notify_result.ok
            if notify_result.error:
                digest["notify_error"] = notify_result.error

    ok = all(r.ok for r in per_repo_results.values())
    message = f"fleet pass complete: {len(per_repo_results)} repo(s) processed"
    if not ok:
        failed_count = sum(1 for r in per_repo_results.values() if not r.ok)
        message += f", {failed_count} failed"

    # Build repos data with ok/message fields included for CLI rendering
    repos_data: dict[str, dict[str, Any]] = {}
    for k, r in per_repo_results.items():
        repo_data = dict(r.data)  # Copy to avoid mutation
        repo_data["ok"] = r.ok  # Add ok field for CLI rendering
        repo_data["message"] = r.message  # Surface per-repo failure message
        repos_data[k] = repo_data

    # api-worker fleet report line (issue #483): near read-only, never raises.
    # Reuse the configs already loaded this pass instead of re-loading every
    # repo config (review: redundant per-repo config reload each fleet pass).
    api_worker_report = compute_api_worker_fleet_report(
        fleet_dir_override=fleet_dir_override,
        preloaded_configs=loaded_configs,
    )

    return CommandResult(
        ok,
        message,
        {
            "repos": repos_data,
            "digest": digest,
            "api_worker_report": api_worker_report.to_dict()
            if api_worker_report is not None
            else None,
        },
    )


def _is_fleet_pass_active(pass_result: CommandResult) -> bool:
    """Return True when a fleet pass produced actionable work.

    Activity is any dispatch, any successful merge, any generated review, or
    any attention event (stalled worker, error, health transition, skip).
    A skipped repo (another supervisor holds its lock) also counts as activity
    because the fleet is not idle.
    """
    data = pass_result.data
    if not isinstance(data, dict):
        return False
    digest = data.get("digest") or {}
    if isinstance(digest, dict) and digest.get("count", 0) > 0:
        return True
    for repo_data in data.get("repos", {}).values():
        if not isinstance(repo_data, dict):
            continue
        if repo_data.get("skipped") is True:
            return True
        for section_key in ("dispatch", "dispatch_rework", "dispatch_reviews"):
            section = repo_data.get(section_key) or {}
            if isinstance(section, dict) and section.get("selected_count", 0) > 0:
                return True
        merges = repo_data.get("merges", [])
        if isinstance(merges, list) and any(
            isinstance(m, dict) and m.get("merged") for m in merges
        ):
            return True
        if isinstance(repo_data.get("reviews", []), list) and repo_data.get("reviews"):
            return True
    return False


@dataclass(frozen=True)
class FleetLocalSnapshot:
    """Aggregate of per-repo ``LocalSnapshot`` values for cheap fleet-wide delta detection.

    Each entry is keyed by the repo's name-with-owner so adding/removing a repo
    is also a delta.  The snapshots themselves include live session counts, so
    worker birth/death is detected even when sidecar file mtimes do not change.
    """

    repo_snapshots: frozenset[tuple[str, LocalSnapshot]]


def _repo_state_dirs(state_dir: Path) -> tuple[Path, Path]:
    """Return the (sessions_dir, prs_dir) for a repo given its state dir."""
    sessions_dir = layout.sessions_dir_default(state_dir)
    prs_dir = state_dir / "prs"
    return sessions_dir, prs_dir


def _take_fleet_snapshot(
    *,
    fleet_dir_override: str | None = None,
) -> FleetLocalSnapshot:
    """Capture a cheap, network-free snapshot across all registered fleet repos."""
    fleet_json_path = layout.fleet_registry_path(override=fleet_dir_override)
    registry = _load_registry(fleet_json_path)
    repos = registry.get("repos", {})

    repo_snapshots: set[tuple[str, LocalSnapshot]] = set()
    for repo_key, entry in repos.items():
        state_dir_str = entry.get("state_dir")
        if not state_dir_str:
            continue
        state_dir = Path(state_dir_str)
        if not state_dir.exists():
            continue
        sessions_dir, prs_dir = _repo_state_dirs(state_dir)
        repo_snapshots.add((repo_key, take_snapshot(sessions_dir, prs_dir)))

    return FleetLocalSnapshot(frozenset(repo_snapshots))


def _has_fleet_delta(
    before: FleetLocalSnapshot,
    after: FleetLocalSnapshot,
) -> bool:
    """Return True if any per-repo local signal changed between snapshots."""
    return before.repo_snapshots != after.repo_snapshots


def _fleet_has_configured_repos(
    fleet_dir_override: str | None, repos: tuple[str, ...] | None
) -> bool:
    """Return True when at least one repo is configured/selected for this scope.

    Gates the zero-repo-pass streak (issue #855 acceptance criterion 4): a
    fleet with zero registered repos -- or an explicit ``repos`` filter that
    matches none of them -- is a configuration state, not an incident, and
    must never move the streak counter. Reads the registry directly rather
    than deriving the answer from a completed ``fleet_loop`` call, because
    the #851 exit shape this alarm targets breaks out of the supervisor loop
    *before* ``fleet_loop`` is ever called.
    """
    registry = _load_registry(layout.fleet_registry_path(override=fleet_dir_override))
    return bool(_select_repos(registry, repos))


# Exit reasons that mean "the code on disk changed underneath this process", so
# the supervisor must be replaced *now* rather than after a watchdog interval
# (#862). Every other reason is a deliberate stop and must not relaunch.
#
# This is a reason vocabulary rather than a `restart_requested` boolean set at
# the two restarting break sites, because the failure directions differ. A new
# break site that forgets to set a boolean reads as "normal exit" and silently
# reintroduces #862's downtime hole; one that forgets to set a reason reports
# "unknown", which is visible in the exit event and the launcher log. #855's
# zero-pass alarm also cannot currently tell these cases apart.
RESTART_EXIT_REASONS = frozenset({"self_deploy", "head_drift"})

# Bound on the per-repo failure reasons appended to the pass-summary line
# (#893). ``message`` is operator-facing free text (e.g. "loop completed with
# N PR error(s)", a tracebacks-derived string from an unclassified exception
# path) with no length contract, so a single pathological message must not be
# able to make the summary line unbounded.
_PASS_SUMMARY_REASON_MAX_CHARS = 500


def run_fleet_supervise(
    *,
    fleet_dir_override: str | None = None,
    repos: tuple[str, ...] | None = None,
    limit: int | None = None,
    merge: bool | None = None,
    dry_run: bool = False,
    poll_interval_override: int | None = None,
    max_runtime_override: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    max_passes: int | None = None,
) -> CommandResult:
    """Run a continuous fleet supervisor loop.

    This is the fleet-wide equivalent of ``charlie bash-rats``/``run_supervised``:
    it polls cheap local signals (per-repo sidecar and verdict mtimes) across the
    registered fleet and calls ``fleet_loop`` only when something actionable
    changed or ``supervisor.full_pass_interval_seconds`` has elapsed since the
    last full pass. It sleeps for ``supervisor.poll_interval_seconds`` after an
    idle pass (or poll with no pass) and ``supervisor.active_cooldown_seconds``
    after an active pass. The loop continues until ``max_runtime_minutes``
    expires, ``max_passes`` is reached, or the operator interrupts it.

    A single ``fleet-supervisor.lock`` in the fleet directory prevents two
    ``charlie fleet supervise`` invocations from overlapping.
    """
    try:
        global_config = load_layered_config(
            Path.cwd(),
            None,
            fleet_dir_override=fleet_dir_override,
            require_global=True,
        )
    except (ConfigError, RepoNotFoundError) as exc:
        # Falling back to defaults silently is how a whole feature disappears
        # without a trace: every config-gated behavior (notify, labels, the
        # runner prologues) reverts to off while passes keep reporting success.
        # A typo in the global layer must be loud.
        logger.warning(
            "Fleet supervisor could not load the global config layer; "
            "continuing with per-repo config only (fleet-wide knobs fall back "
            "to per-repo values or defaults): %s",
            exc,
        )
        print(f"config load failed, continuing on per-repo config: {exc}", flush=True)
        # The global layer is required, but the per-repo config is still valid
        # and must not be discarded with it -- discarding both regresses the
        # #623 silent-disable failure (every per-repo knob reverting to its
        # dataclass default while passes keep reporting success). Reload
        # without the global requirement so per-repo settings survive; only
        # fall back to pristine defaults if the per-repo load itself fails.
        try:
            global_config = load_layered_config(
                Path.cwd(), None, fleet_dir_override=fleet_dir_override
            )
        except (ConfigError, RepoNotFoundError):
            global_config = OrchestratorConfig()

    # Provenance of the layer every fleet-wide knob comes from, logged once per
    # supervisor start (not per pass -- this is startup, so it costs one stat).
    # The prologue already logs what runner_allocation *resolved to*; the fact
    # missing from #590 is whether the file that declares it was read at all.
    #
    # Deliberately one stat() rather than an exists() flag: exists() collapses a
    # missing file, an unready device and an unresolvable path into the same bare
    # False, and all of them take load_layered_config's silent-{} branch. A bare
    # exists=False here would reproduce the exact ambiguity this line exists to
    # remove. See describe_config_file for which errors are and are not hidden.
    global_config_path = layout.global_config_path(override=fleet_dir_override)
    logger.info(
        "Fleet supervisor global config: path=%s %s",
        global_config_path,
        describe_config_file(global_config_path),
    )

    overrides: dict[str, int] = {}
    if poll_interval_override is not None:
        overrides["poll_interval_seconds"] = poll_interval_override
    if max_runtime_override is not None:
        overrides["max_runtime_minutes"] = max_runtime_override
    cfg = replace(global_config.supervisor, **overrides)

    lock_path = layout.fleet_supervisor_lock_path(override=fleet_dir_override)
    lock = try_acquire_supervisor_lock(lock_path)
    if lock is None:
        return CommandResult(
            False,
            "fleet supervisor already running (fleet-supervisor.lock held)",
            # Reason carried here too so every exit route answers the same field.
            # Not a relaunch: another supervisor already holds the lock, so the
            # right move is to stand down, not to spawn a second one.
            {"exit_reason": "lock_held", "restart_requested": False},
        )

    pass_number = 0
    total_repo_passes = 0
    total_attention_events = 0
    total_failed_repos = 0
    start_time = clock()
    # Set at every route out of the loop below; see RESTART_EXIT_REASONS.
    exit_reason: str | None = None
    full_pass_interval = cfg.full_pass_interval_seconds
    last_full_pass_at = start_time - full_pass_interval
    snapshot = _take_fleet_snapshot(fleet_dir_override=fleet_dir_override)

    # Supervisor lifecycle instrumentation (issue #627): record started/exited
    # events to the fleet-level events.db and maintain a heartbeat sidecar so
    # a killed supervisor (TerminateProcess, exit=-1) leaves a diagnosable gap
    # instead of only a launcher text marker. Acquiring the lock proves any
    # prior supervisor is gone, so a stale heartbeat with no ``exited_at`` here
    # means the prior one was killed — emit a retroactive supervisor_exited and
    # alert on it before this supervisor records its own start.
    notify_config = getattr(global_config, "notify", None)
    prior = detect_prior_abnormal_exit(fleet_dir_override)
    if prior is not None:
        record_prior_abnormal_exit(fleet_dir_override, prior)
        print(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] supervisor: prior "
            f"supervisor terminated without an exit event (last beat "
            f"{prior.get('prior_last_beat_at')}); recorded retroactive "
            f"supervisor_exited",
            flush=True,
        )
        if notify_config is not None and getattr(notify_config, "enabled", False):
            _emit_fleet_transition(
                notify_config,
                AttentionEntry(
                    issue_number=-1,
                    adapter_kind="fleet-supervisor",
                    health="ERROR",
                    previous_health=None,
                    last_log_line=(
                        f"prior supervisor terminated without an exit event "
                        f"(last beat {prior.get('prior_last_beat_at')})"
                    ),
                    pid=prior.get("prior_pid"),
                ),
                fleet_dir_override,
                persistent=False,
            )

    started_at_iso = utc_now()
    record_supervisor_started(
        fleet_dir_override,
        pid=os.getpid(),
        started_at=started_at_iso,
        full_pass_interval_seconds=full_pass_interval,
        max_pass_runtime_seconds=cfg.max_pass_runtime_seconds,
    )

    # Exit tracking: ``_exit_code`` is 0 for every in-control clean exit (drain,
    # max_runtime, max_passes, HEAD-drift restart, self-deploy restart,
    # KeyboardInterrupt) and 1 for an uncaught exception. ``_exit_reason`` names
    # the branch. Both are consumed by the ``finally`` below to emit
    # ``supervisor_exited``. A TerminateProcess kill skips ``finally`` entirely,
    # leaving the heartbeat's ``exited_at`` null for the next start to detect.
    _exit_code = 0
    _exit_reason = "completed"

    # Capture the HEAD SHA at process startup so we can detect drift caused
    # by an external actor (operator pull, another process) between passes.
    # self_deploy only reports from_sha/to_sha for pulls *it* performed; a
    # HEAD moved out-of-band shows as "already up to date" and the daemon
    # silently runs stale code forever (observed 2026-07-23: ~90 minutes of
    # ConfigError crashes after an operator pulled origin/main manually while
    # the daemon was already running).
    startup_head = read_head_sha(orchestrator_root())

    try:
        while True:
            now = clock()
            # Refresh the heartbeat at the top of every iteration so the
            # freshness signal stays at most one ``max_pass_runtime_seconds``
            # (plus the post-pass sleep) old on a live supervisor (issue #627).
            # A killed supervisor stops updating it, which is exactly the gap
            # the heartbeat check detects.
            update_supervisor_heartbeat(
                fleet_dir_override,
                pass_number=pass_number,
                last_beat_at=utc_now(),
            )
            if cfg.max_runtime_minutes is not None and cfg.max_runtime_minutes > 0:
                elapsed_minutes = (now - start_time) / 60.0
                if elapsed_minutes >= cfg.max_runtime_minutes:
                    exit_reason = "max_runtime"
                    _exit_reason = "max_runtime"
                    break
            if max_passes is not None and pass_number >= max_passes:
                exit_reason = "max_passes"
                _exit_reason = "max_passes"
                break

            new_snapshot = _take_fleet_snapshot(fleet_dir_override=fleet_dir_override)
            fallback_due = (now - last_full_pass_at) >= full_pass_interval
            run_pass = _has_fleet_delta(snapshot, new_snapshot) or fallback_due

            if not run_pass:
                snapshot = new_snapshot
                sleep(float(cfg.poll_interval_seconds))
                continue

            now_str = datetime.datetime.now().strftime("%H:%M:%S")

            pass_number += 1
            last_full_pass_at = now

            # Self-deploy before running the pass: FF-pull origin/main and sync
            # dependencies when pyproject.toml/uv.lock changed.  Non-fatal on a
            # diverged or dirty tree.
            deploy = self_deploy(
                orchestrator_root(),
                fleet_dir_override=fleet_dir_override,
                dry_run=dry_run,
                failure_alarm_threshold=cfg.self_deploy_failure_alarm,
            )
            notify_config = getattr(global_config, "notify", None)
            notify_enabled = notify_config is not None and getattr(notify_config, "enabled", False)
            if not deploy.ok:
                print(
                    f"[{now_str}] self-deploy skipped: {deploy.error}",
                    flush=True,
                )
                if deploy.alertable and notify_enabled:
                    entry = AttentionEntry(
                        issue_number=-1,
                        adapter_kind="self-deploy",
                        health="ERROR",
                        previous_health=None,
                        last_log_line=deploy.error,
                        pid=None,
                    )
                    _emit_fleet_transition(notify_config, entry, fleet_dir_override)
            elif deploy.previewed:
                print(f"[{now_str}] self-deploy: {deploy.message}", flush=True)
            else:
                # Real (non-previewed) success. Console output is unchanged from
                # before this fix -- print only on the previously-notable
                # outcomes -- but the notify digest gets a health-OK entry
                # unconditionally, on *every* real success, not only when the
                # venv was repaired. This is the single-point fix for issue
                # #817: _filter_fleet_health_transitions is a correct
                # recovery-edge detector, but it can only detect a recovery if
                # *some* entry carries a non-ERROR health value -- before this
                # fix only the venv_repaired branch ever produced one, so a
                # plain successful deploy fed the baseline nothing and the
                # first failure latched "self-deploy:-1" at ERROR forever
                # (34/34 tracked keys were latched this way).
                if deploy.synced or deploy.venv_repaired:
                    print(f"[{now_str}] self-deploy: {deploy.message}", flush=True)
                if notify_enabled:
                    entry = AttentionEntry(
                        issue_number=-1,
                        adapter_kind="self-deploy",
                        health="REPAIRED" if deploy.venv_repaired else "OK",
                        previous_health=None,
                        last_log_line=deploy.message,
                        pid=None,
                    )
                    _emit_fleet_transition(notify_config, entry, fleet_dir_override)

            # A successful pull that actually moved HEAD updated the files on
            # disk, but this process already imported every charlie_work
            # module at startup -- Python does not hot-reload modules just
            # because git changed them underneath it. Left running, the
            # supervisor would keep executing whatever code was live at
            # process start for its entire (max-runtime-0 == unbounded)
            # lifetime, silently ignoring every fix merged to main afterward
            # (observed 2026-07-22: ~40 minutes on stale code, including a
            # dispatch-rework redispatch-cap fix and a worker model-pin fix
            # that had already landed on main). Exit cleanly here so the
            # scheduled-task watchdog (5-minute trigger, MultipleInstancesPolicy
            # =IgnoreNew) relaunches a fresh process with the new commit
            # actually imported. Safe to do before this pass's fleet_loop
            # call: no dispatch/state mutation has happened yet this
            # iteration, and state.json is disk-persisted, not in-memory, so
            # the next process resumes from exactly where this one left off.
            # Bind the shas to locals so the non-None guard survives into the
            # message below; folding the check into a bool() loses it.
            from_sha = deploy.from_sha
            to_sha = deploy.to_sha
            if deploy.ok and deploy.pulled and from_sha and to_sha and deploy.head_changed:
                print(
                    f"[{now_str}] self-deploy: HEAD moved {from_sha[:12]} -> "
                    f"{to_sha[:12]}; exiting for watchdog restart to pick up new code",
                    flush=True,
                )
                exit_reason = "self_deploy"
                _exit_reason = "self_deploy_head_moved"
                break

            # Independent drift check: even when self_deploy reports "already
            # up to date" (head_changed=False), HEAD may have been moved by an
            # external actor (operator pull, another process) since this
            # process started. Python does not hot-reload modules on disk
            # changes, so we must exit and let the watchdog relaunch with the
            # new code (observed 2026-07-23: ~90 minutes of ConfigError crashes
            # after an operator pulled origin/main while the daemon was
            # already running — self_deploy saw "already up to date" every
            # pass because HEAD was already at the new commit).
            current_head = read_head_sha(orchestrator_root())
            if startup_head and current_head and current_head != startup_head:
                print(
                    f"[{now_str}] HEAD drift detected: startup={startup_head[:12]} "
                    f"current={current_head[:12]} (moved externally); exiting for "
                    f"watchdog restart to pick up new code",
                    flush=True,
                )
                exit_reason = "head_drift"
                _exit_reason = "head_drift_restart"
                break

            pass_result = fleet_loop(
                fleet_dir_override=fleet_dir_override,
                global_config=global_config,
                repos=repos,
                limit=limit,
                merge=merge,
                dry_run=dry_run,
                work_only=False,
            )

            data = pass_result.data
            repos_data = data.get("repos", {}) if isinstance(data, dict) else {}
            digest = data.get("digest", {}) if isinstance(data, dict) else {}
            repo_count = len(repos_data)
            failed = sum(
                1 for r in repos_data.values() if isinstance(r, dict) and not r.get("ok", True)
            )
            attention_count = digest.get("count", 0) if isinstance(digest, dict) else 0

            total_repo_passes += repo_count
            total_attention_events += attention_count
            total_failed_repos += failed

            summary = (
                f"[{now_str}] fleet pass {pass_number}: {repo_count} repo(s), "
                f"{repo_count - failed} ok, {failed} failed, "
                f"{attention_count} attention event(s)"
            )
            # #893: "N failed" alone is indistinguishable from a real outage --
            # the per-repo failure reason (e.g. a known, acked-releasable
            # control like the unauthorized-merge tripwire) is already sitting
            # in repos_data[key]["message"] (set by fleet_loop) but was never
            # read here. Append it, on the existing line (an attribute of the
            # pass, not a separate report -- see the api_worker_report line
            # below for the precedent on when a *second* line is warranted
            # instead).
            if failed:
                # Strip before filtering, not after: a whitespace-only message
                # is falsy-after-strip but truthy-before, so filtering on the
                # raw value would let an empty-looking reason through as "[]".
                raw_reasons = (
                    str(r.get("message") or "").strip()
                    for r in repos_data.values()
                    if isinstance(r, dict) and not r.get("ok", True)
                )
                reasons = sorted({m for m in raw_reasons if m})
                if reasons:
                    reason_text = "; ".join(reasons)
                    if len(reason_text) > _PASS_SUMMARY_REASON_MAX_CHARS:
                        # Plain ASCII ellipsis, not U+2026: this print() runs
                        # through whatever locale-derived stdout encoding the
                        # supervisor's launch chain picked (wscript -> ps ->
                        # cmd -> uv), and a codepage without U+2026 (cp437,
                        # cp850) would turn a truncation marker into a
                        # UnicodeEncodeError that crashes the whole pass.
                        reason_text = reason_text[: _PASS_SUMMARY_REASON_MAX_CHARS - 3] + "..."
                    summary += f" [{reason_text}]"

            print(summary, flush=True)

            # api-worker fleet report line (issue #483): one line per pass
            # when any repo configures the section, keeping partial rollout
            # visible until fleet-wide enablement completes.
            api_worker_report = data.get("api_worker_report") if isinstance(data, dict) else None
            if isinstance(api_worker_report, dict) and api_worker_report.get("line"):
                print(f"[{now_str}] {api_worker_report['line']}", flush=True)

            # Snapshot after the pass becomes the baseline for the next delta
            # check; this avoids a spurious extra pass when this pass's own
            # side-effect writes (new session sidecars, verdict files, etc.)
            # show up as a "delta" on the very next poll.
            snapshot = _take_fleet_snapshot(fleet_dir_override=fleet_dir_override)

            sleep(
                float(
                    cfg.active_cooldown_seconds
                    if _is_fleet_pass_active(pass_result)
                    else cfg.poll_interval_seconds
                )
            )

        # Issue #855 (the general shape behind #851): a cycle can complete
        # with exit code 0 having done zero repo passes -- e.g. every restart
        # exits via the self-deploy HEAD-moved break above before ever
        # reaching fleet_loop -- and every existing health signal reads
        # green because, by its own accounting, nothing failed. Record this
        # process's aggregate total_repo_passes (already computed and
        # printed in the summary below) against the persisted streak so N
        # consecutive such cycles escalate instead of repeating silently.
        # Placed inside the try block, after the loop, so a bookkeeping
        # failure here is caught by the except Exception handler below
        # instead of propagating out of run_fleet_supervise uncaught.
        # Deliberately skipped on KeyboardInterrupt (falls through the
        # except clause below without reaching this line): an
        # operator-initiated stop is not the automatic-failure pattern this
        # alarm targets.
        # Contained locally rather than left to the handler below. This call does
        # real file I/O (mkdir, state_lock, log_event) and its own docstring says
        # it can raise. If it did, the handler's `aborted` reason replaced an
        # already-set `self_deploy`/`head_drift`, restart_requested went False,
        # and the wrapper did NOT relaunch -- reintroducing #862 through a
        # secondary bookkeeping failure. A zero-pass counter that failed to
        # record is worth a log line; it is not worth suppressing a restart.
        try:
            record_zero_pass_streak(
                orchestrator_root(),
                repo_passes=total_repo_passes,
                repos_configured=_fleet_has_configured_repos(fleet_dir_override, repos),
                threshold=cfg.zero_pass_alarm,
            )
        except Exception as streak_exc:  # noqa: BLE001 - bookkeeping must not alter exit semantics
            logger.warning(
                "zero-pass streak bookkeeping failed (exit_reason=%s): %s",
                exit_reason or "unknown",
                streak_exc,
            )
    except KeyboardInterrupt:
        _exit_reason = "keyboard_interrupt"
        # An operator stop must never relaunch. Named rather than left unset so
        # the exit event distinguishes it from a break site that forgot a reason.
        exit_reason = "interrupted"
    except Exception as exc:
        _exit_code = 1
        _exit_reason = "exception"
        elapsed_s = clock() - start_time
        # Defence in depth behind the contained streak call above: if any future
        # post-break statement raises, an already-set restarting reason must
        # survive. Overwriting it with "aborted" is what made a bookkeeping
        # failure able to cancel a self-deploy relaunch.
        reason = exit_reason or "aborted"
        return CommandResult(
            False,
            f"fleet supervisor aborted on pass {pass_number}: {exc}",
            {
                "exit_reason": reason,
                "restart_requested": reason in RESTART_EXIT_REASONS,
                "passes": pass_number,
                "total_repo_passes": total_repo_passes,
                "total_attention_events": total_attention_events,
                "total_failed_repos": total_failed_repos,
                "elapsed_seconds": elapsed_s,
            },
        )
    finally:
        lock.release()
        # Record supervisor_exited for every in-control exit (issue #627). A
        # TerminateProcess kill never reaches here — that is the gap the next
        # start's detect_prior_abnormal_exit recovers. Best-effort: wrapped so
        # an instrumentation failure cannot break the exit path.
        try:
            record_supervisor_exit(
                fleet_dir_override,
                exit_code=_exit_code,
                passes=pass_number,
                started_at=started_at_iso,
                reason=_exit_reason,
            )
            if (
                is_exit_alertable(_exit_code)
                and notify_config is not None
                and getattr(notify_config, "enabled", False)
            ):
                _emit_fleet_transition(
                    notify_config,
                    AttentionEntry(
                        issue_number=-1,
                        adapter_kind="fleet-supervisor",
                        health="ERROR",
                        previous_health=None,
                        last_log_line=(
                            f"supervisor exited abnormally (exit_code={_exit_code}, "
                            f"reason={_exit_reason}, passes={pass_number})"
                        ),
                        pid=os.getpid(),
                    ),
                    fleet_dir_override,
                    persistent=False,
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("supervisor exit instrumentation failed: %s", exc)

    elapsed_s = clock() - start_time
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_s)))
    return CommandResult(
        True,
        f"fleet supervisor complete: {pass_number} pass(es) in {elapsed_str}, "
        f"{total_repo_passes} repo pass(es), {total_attention_events} attention "
        f"event(s), {total_failed_repos} failed repo(s) "
        f"[exit_reason={exit_reason or 'unknown'}]",
        {
            "passes": pass_number,
            "total_repo_passes": total_repo_passes,
            "total_attention_events": total_attention_events,
            "total_failed_repos": total_failed_repos,
            "elapsed_seconds": elapsed_s,
            "exit_reason": exit_reason or "unknown",
            "restart_requested": exit_reason in RESTART_EXIT_REASONS,
        },
    )


def _record_supervise_loop_cap_event(result: SuperviseLoopResult) -> None:
    """Best-effort fleet-level record that the relaunch bound refused a restart.

    Same never-escape discipline as ``_record_lane_failure_event``: the whole
    point of the cap is to exit cleanly so the scheduled task's tick regains
    restart authority, and a failure to *record* that must not turn the clean
    exit into a crash.
    """
    try:
        state_path = layout.state_file_path(layout.default_state_root(orchestrator_root()))
        log_event(
            state_path,
            "supervise_relaunch_cap_reached",
            {
                "launches": result.launches,
                "relaunches": result.relaunches,
                "last_exit_code": result.last_exit_code,
                # #903: which of the two cap conditions this was -- "retirement"
                # (healthy wrapper stepping aside after sustained uptime) or
                # "non_convergence" (every child died on startup). Without it a
                # reader cannot tell an expected event from an incident, and
                # this function's own contract above claims the event says
                # exactly which condition occurred.
                "cap_cause": result.cap_cause,
            },
        )
    except Exception:
        logger.exception("Failed to record supervise relaunch cap event")


def _spawn_supervise_child(supervise_args: Sequence[str]) -> int:
    """Run one ``fleet supervise`` child to completion; return its exit code.

    ``Popen`` + ``wait`` with *inherited* stdio, deliberately not
    ``subprocess_runner.run_captured``: the child runs for hours and
    ``capture_output=True`` would buffer its entire log in memory. Inheriting
    also leaves the launcher's own ``>> $log`` redirection as the single place
    output lands, unchanged from before this wrapper existed.

    This is a wrapper waiting on its own child, so it is outside the adapter
    "never block on worker completion" invariant — blocking here is the point.

    ``sys.executable`` keeps the child in the venv the wrapper was started in;
    ``-m charlie_work`` rather than the console script because of #854 (the
    running ``charlie.exe`` is the file ``uv sync`` must delete on self-deploy).

    ``no_console_window_kwargs`` and *not* ``hidden_console_kwargs``, even though
    this is a long-lived worker-shaped spawn: ``CREATE_NEW_CONSOLE`` gives the
    child that console's std handles instead of the ones it inherits, which
    would silently redirect the supervisor's entire log away from the launcher's
    ``>> $log`` and into a hidden console nobody can read. ``CREATE_NO_WINDOW``
    suppresses the window while leaving inherited handles intact.
    """
    command = [sys.executable, "-m", "charlie_work", "fleet", "supervise", *supervise_args]
    process = subprocess.Popen(
        command,
        cwd=str(orchestrator_root()),
        **no_console_window_kwargs(),
    )
    return process.wait()


def run_fleet_supervise_loop(
    *,
    supervise_args: Sequence[str] = (),
    max_relaunches: int = DEFAULT_MAX_RELAUNCHES,
    spawn: Callable[[int], int] | None = None,
    on_cap_reached: Callable[[SuperviseLoopResult], None] | None = None,
) -> CommandResult:
    """Run ``fleet supervise``, relaunching immediately on a restart request.

    ``supervise_args`` is forwarded to the child verbatim rather than being
    re-declared flag by flag here: a per-flag list would silently stop
    forwarding any option added to ``fleet supervise`` later.

    ``on_cap_reached`` is injectable for the same reason ``spawn`` is. The
    default writes a real event to the real state file resolved from
    ``orchestrator_root()``, so a test exercising the cap path with defaults
    writes into the **live** ``events.db`` that the running supervisor owns --
    both polluting production data and contending for its state lock. Tests
    pass their own recorder and assert on it.
    """
    args = tuple(supervise_args)

    def _default_spawn(_launch_number: int) -> int:
        return _spawn_supervise_child(args)

    result = run_supervise_relaunch_loop(
        spawn if spawn is not None else _default_spawn,
        max_relaunches=max_relaunches,
        log=lambda message: print(message, flush=True),
        on_cap_reached=(
            on_cap_reached if on_cap_reached is not None else _record_supervise_loop_cap_event
        ),
    )

    # A cap is a *clean handoff*, not a failure: the wrapper deliberately gives
    # restart authority back to the 5-minute trigger rather than spinning. So the
    # last exit being EXIT_RESTART_REQUESTED is ok, exactly like a plain 0.
    #
    # Reporting the cap as ok=False (exit 1) was the first draft, and it is wrong
    # for the same reason #862 itself is: it makes two different conditions
    # indistinguishable. `except Exception` in `run_fleet_supervise` also exits 1,
    # so an operator seeing LastTaskResult=1 could not tell "supervisor crashed"
    # from "self-deploy is not converging" — the ambiguity just moves up a layer
    # instead of being removed. The cap's signal is the distinct log line and the
    # `supervise_relaunch_cap_reached` event, both of which say exactly which
    # condition occurred; the exit code does not need to carry it too.
    ok = result.last_exit_code in (0, EXIT_RESTART_REQUESTED)
    return CommandResult(
        ok,
        f"supervise-loop: {result.launches} launch(es), {result.relaunches} relaunch(es), "
        f"last exit {result.last_exit_code}"
        + (" (relaunch cap reached)" if result.cap_reached else ""),
        {
            "launches": result.launches,
            "relaunches": result.relaunches,
            "last_exit_code": result.last_exit_code,
            "cap_reached": result.cap_reached,
            "cap_cause": result.cap_cause,
        },
    )
