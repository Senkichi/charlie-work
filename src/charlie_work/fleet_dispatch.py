from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .config import ConfigError, OrchestratorConfig
from .fleet_paths import fleet_dir
from .fleet_registry import _load_registry, count_fleet_runners
from .github import GitHub, GitHubError
from .global_config import load_layered_config
from .notify import AttentionDigest, AttentionEntry, emit_digest
from .paths import RepoNotFoundError, runtime_paths
from .supervise import (
    LocalSnapshot,
    orchestrator_root,
    self_deploy,
    take_snapshot,
    try_acquire_supervisor_lock,
)
from .runners import (
    decide_autoscale,
    FleetTotals,
    is_in_cooldown,
    is_pool_idle_for_minutes,
    observe_runner_pool,
    provision_runner,
    record_scale_event,
    scale_down_idle_runners,
    ScaleAction,
)
from .state import utc_now
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
    fleet_json_path = fleet_dir(override=fleet_dir_override) / "fleet.json"
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
    state = observe_runner_pool(gh, config.runner_scaling, state_dir=paths.root)

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


def _build_fleet_attention_digest(
    attention_events: list[dict[str, Any]],
) -> AttentionDigest:
    """Convert fleet-aggregated event dicts into a single AttentionDigest.

    Fleet events are already-flattened per-repo dicts (stalled / error /
    health_transition) produced by ``_extract_attention_events``. This maps
    each one onto the real #166 ``AttentionEntry`` schema so the fleet pass
    can go through the same ``emit_digest`` sink pipeline as a single-repo
    pass, rather than re-deriving its own notification format.

    ``issue_number`` is required by ``AttentionEntry``; events that carry no
    issue number (e.g. PR errors) fall back to ``-1`` as a sentinel so they
    still surface in the digest instead of being silently dropped.
    """
    entries: list[AttentionEntry] = []
    for event in attention_events:
        event_type = event["type"]
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

    return AttentionDigest(
        generated_at=utc_now(),
        repo="fleet",
        transitions=tuple(entries),
    )


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
    fleet_json_path = fleet_dir(override=fleet_dir_override) / "fleet.json"
    registry = _load_registry(fleet_json_path)

    # Select repos in the appropriate order
    selected = _select_repos(registry, repos)

    per_repo_results: dict[str, CommandResult] = {}
    attention_events: list[dict[str, Any]] = []
    orphan_sweep_calls = 0

    # Run autoscale prologue if enabled (only for full loop, not work-only)
    if not work_only:
        autoscale_events = _run_fleet_autoscale_prologue(
            fleet_dir_override, global_config, dry_run
        )
        attention_events.extend(autoscale_events)

    for repo_key, entry in selected:
        repo_root = Path(entry.get("repo_root"))
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
            # Fleet mode is the single notification authority: the aggregate
            # digest below emits once for the whole pass. Silence per-repo
            # dispatch()/loop() emission so one health transition doesn't fire
            # both a per-repo and a fleet-level notification.
            config = replace(config, notify=replace(config.notify, enabled=False))
            paths = runtime_paths(repo_root, config.runtime.state_dir)

            # Non-blocking supervisor lock: fleet passes must be mutually exclusive
            # with a supervised bash-rats loop on the same repo to avoid double-
            # dispatching through the governor's read-then-launch window.
            lock = try_acquire_supervisor_lock(paths.root / "supervisor.lock")
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
            per_repo_results[repo_key] = CommandResult(
                False, f"fleet pass error: {type(exc).__name__}: {exc}", {}
            )
            logger.exception("Error processing repo %s", repo_key)

    # Call the notifier digest sink exactly once per fleet pass, via the real
    # #166 notify.py implementation (AttentionDigest + emit_digest).
    notify_config = getattr(global_config, "notify", None) if global_config else None
    digest: dict[str, Any] = {
        "events": attention_events,
        "count": len(attention_events),
        "orphan_sweep_calls": orphan_sweep_calls,
        "emitted": False,
    }
    if notify_config is not None and getattr(notify_config, "enabled", False) and attention_events:
        attention_digest = _build_fleet_attention_digest(attention_events)
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

    return CommandResult(
        ok,
        message,
        {
            "repos": repos_data,
            "digest": digest,
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
    sessions_dir = state_dir / "dispatches" / "sessions"
    prs_dir = state_dir / "prs"
    return sessions_dir, prs_dir


def _take_fleet_snapshot(
    *,
    fleet_dir_override: str | None = None,
) -> FleetLocalSnapshot:
    """Capture a cheap, network-free snapshot across all registered fleet repos."""
    fleet_json_path = fleet_dir(override=fleet_dir_override) / "fleet.json"
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
            Path.cwd(), None, fleet_dir_override=fleet_dir_override
        )
    except (ConfigError, RepoNotFoundError):
        global_config = OrchestratorConfig()

    overrides: dict[str, int] = {}
    if poll_interval_override is not None:
        overrides["poll_interval_seconds"] = poll_interval_override
    if max_runtime_override is not None:
        overrides["max_runtime_minutes"] = max_runtime_override
    cfg = replace(global_config.supervisor, **overrides)

    lock_path = fleet_dir(override=fleet_dir_override) / "fleet-supervisor.lock"
    lock = try_acquire_supervisor_lock(lock_path)
    if lock is None:
        return CommandResult(
            False,
            "fleet supervisor already running (fleet-supervisor.lock held)",
            {},
        )

    pass_number = 0
    total_repo_passes = 0
    total_attention_events = 0
    total_failed_repos = 0
    start_time = clock()
    full_pass_interval = cfg.full_pass_interval_seconds
    last_full_pass_at = start_time - full_pass_interval
    snapshot = _take_fleet_snapshot(fleet_dir_override=fleet_dir_override)

    try:
        while True:
            now = clock()
            if cfg.max_runtime_minutes is not None and cfg.max_runtime_minutes > 0:
                elapsed_minutes = (now - start_time) / 60.0
                if elapsed_minutes >= cfg.max_runtime_minutes:
                    break
            if max_passes is not None and pass_number >= max_passes:
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
            deploy = self_deploy(orchestrator_root(), fleet_dir_override=fleet_dir_override)
            if not deploy.ok:
                print(
                    f"[{now_str}] self-deploy skipped: {deploy.error}",
                    flush=True,
                )
                notify_config = getattr(global_config, "notify", None)
                if notify_config is not None and getattr(notify_config, "enabled", False):
                    attention_digest = AttentionDigest(
                        generated_at=utc_now(),
                        repo="fleet",
                        transitions=(
                            AttentionEntry(
                                issue_number=-1,
                                adapter_kind="self-deploy",
                                health="ERROR",
                                previous_health=None,
                                last_log_line=deploy.error,
                                pid=None,
                            ),
                        ),
                    )
                    emit_digest(notify_config, attention_digest)
            elif deploy.synced:
                print(f"[{now_str}] self-deploy: {deploy.message}", flush=True)
            elif deploy.venv_repaired:
                print(f"[{now_str}] self-deploy: {deploy.message}", flush=True)
                notify_config = getattr(global_config, "notify", None)
                if notify_config is not None and getattr(notify_config, "enabled", False):
                    attention_digest = AttentionDigest(
                        generated_at=utc_now(),
                        repo="fleet",
                        transitions=(
                            AttentionEntry(
                                issue_number=-1,
                                adapter_kind="self-deploy",
                                health="REPAIRED",
                                previous_health=None,
                                last_log_line=deploy.message,
                                pid=None,
                            ),
                        ),
                    )
                    emit_digest(notify_config, attention_digest)

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

            print(
                f"[{now_str}] fleet pass {pass_number}: {repo_count} repo(s), "
                f"{repo_count - failed} ok, {failed} failed, "
                f"{attention_count} attention event(s)",
                flush=True,
            )

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
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        elapsed_s = clock() - start_time
        return CommandResult(
            False,
            f"fleet supervisor aborted on pass {pass_number}: {exc}",
            {
                "passes": pass_number,
                "total_repo_passes": total_repo_passes,
                "total_attention_events": total_attention_events,
                "total_failed_repos": total_failed_repos,
                "elapsed_seconds": elapsed_s,
            },
        )
    finally:
        lock.release()

    elapsed_s = clock() - start_time
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_s)))
    return CommandResult(
        True,
        f"fleet supervisor complete: {pass_number} pass(es) in {elapsed_str}, "
        f"{total_repo_passes} repo pass(es), {total_attention_events} attention "
        f"event(s), {total_failed_repos} failed repo(s)",
        {
            "passes": pass_number,
            "total_repo_passes": total_repo_passes,
            "total_attention_events": total_attention_events,
            "total_failed_repos": total_failed_repos,
            "elapsed_seconds": elapsed_s,
        },
    )
