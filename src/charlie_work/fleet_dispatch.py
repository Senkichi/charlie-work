from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import ConfigError, find_config_path, load_config
from .fleet_paths import fleet_dir
from .fleet_registry import _load_registry
from .github import GitHub, GitHubError
from .notify import AttentionDigest, AttentionEntry, emit_digest
from .paths import runtime_paths
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

    return events


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
                    issue_number=event.get("pr") or -1,
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

    for repo_key, entry in selected:
        repo_root = Path(entry.get("repo_root"))
        if not repo_root.is_dir():
            # Tolerate vanished/moved repo (#169 precedent)
            per_repo_results[repo_key] = CommandResult(
                False, f"repo_root missing, skipped: {repo_root}", {}
            )
            continue

        try:
            # Load per-repo config
            config = load_config(find_config_path(repo_root, entry.get("config_path")))
            paths = runtime_paths(repo_root, config.runtime.state_dir)
            gh = GitHub(repo_root=repo_root, dry_run=dry_run)
            app = OrchestratorApp(
                repo_root, paths, config, gh, dry_run=dry_run, fleet_dir_override=None
            )

            # Call the appropriate per-repo method
            if work_only:
                # Dispatch-only path (no review/merge)
                result = app.dispatch(limit)
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

        except (GitHubError, ConfigError) as exc:
            # Per-repo isolation: catch at iteration boundary and continue
            per_repo_results[repo_key] = CommandResult(False, f"fleet pass error: {exc}", {})
            logger.error(f"Error processing repo {repo_key}: {exc}")

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

    # Build repos data with ok field included for CLI rendering
    repos_data: dict[str, dict[str, Any]] = {}
    for k, r in per_repo_results.items():
        repo_data = dict(r.data)  # Copy to avoid mutation
        repo_data["ok"] = r.ok  # Add ok field for CLI rendering
        repos_data[k] = repo_data

    return CommandResult(
        ok,
        message,
        {
            "repos": repos_data,
            "digest": digest,
        },
    )
