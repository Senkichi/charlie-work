from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import layout
from .config import DEFAULT_CONFIG_FILENAME
from .global_config import load_layered_config
from .file_lock import ByteRangeFileLock, try_acquire_byte_range_lock
from .fleet_paths import warn_fleet_dir_virtualization_on_write
from .github import GitHub, GitHubError, GitHubLike
from .paths import RuntimePaths
from .state import save_state, state_lock
from .worker import iter_workers

if TYPE_CHECKING:
    from .config import RuntimeConfig

logger = logging.getLogger(__name__)

# Intra-process serialization for try_acquire_fleet_lock.
#
# File locks (msvcrt.locking / fcntl.flock) serialize across PROCESSES, but
# byte-range file locks are owned by the process, not the thread — two threads
# in the SAME process may not be serialized by the file lock alone. A per-path
# threading.Lock, acquired before the file lock attempt, restores deterministic
# intra-process serialization.
_FLEET_THREAD_LOCKS: dict[str, threading.Lock] = {}
_FLEET_THREAD_LOCKS_GUARD = threading.Lock()


def _fleet_thread_lock_for(path: Path) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(path))
    with _FLEET_THREAD_LOCKS_GUARD:
        lock = _FLEET_THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _FLEET_THREAD_LOCKS[key] = lock
        return lock


FLEET_REGISTRY_VERSION = 1


def _empty_registry() -> dict[str, Any]:
    """Return an empty fleet registry dict."""
    return {"version": FLEET_REGISTRY_VERSION, "repos": {}}


def _load_registry(fleet_json_path: Path) -> dict[str, Any]:
    """Load the fleet registry from disk, returning an empty registry if missing.

    This is a local loader for fleet.json specifically — it does not use
    state.load_state because that assumes the state.json schema.
    """
    if not fleet_json_path.exists():
        return _empty_registry()
    try:
        with fleet_json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        # Corrupt registry — start fresh (best-effort, same as state.py)
        return _empty_registry()
    if not isinstance(data, dict):
        return _empty_registry()
    data.setdefault("version", FLEET_REGISTRY_VERSION)
    data.setdefault("repos", {})
    return data


def touch_repo(
    fleet_dir_override: str | None,
    repo_root: Path,
    paths: RuntimePaths,
    gh: GitHubLike,
) -> dict[str, Any]:
    """Register or update a repo in the fleet registry.

    Resolves nameWithOwner via gh repo view, then creates or updates the
    registry entry with:
    - repo_root (updated if moved)
    - name_with_owner (key)
    - config_path
    - state_dir
    - first_seen (set on create only)
    - last_seen (always bumped)

    If gh repo view fails (offline, not a GitHub repo, gh missing), the
    registration is silently skipped — the command proceeds normally and
    the registry is not updated for that invocation (errors-as-values invariant).

    Args:
        fleet_dir_override: Optional override for the fleet directory path.
        repo_root: The repository root path.
        paths: The RuntimePaths for this repo.
        gh: The GitHub client instance.

    Returns:
        The updated registry dict (or the unchanged registry if registration
        was skipped due to gh error).
    """
    # Resolve nameWithOwner — best-effort, skip on failure
    try:
        name_with_owner = gh.name_with_owner()
    except GitHubError as exc:
        logger.debug(f"Skipping fleet registration: gh repo view failed: {exc}")
        # Return the current registry unchanged (or empty if missing)
        fleet_json_path = layout.fleet_registry_path(override=fleet_dir_override)
        return _load_registry(fleet_json_path)

    fleet_json_path = layout.fleet_registry_path(override=fleet_dir_override)

    # Issue #624: a virtualized fleet dir forks a private copy on this write,
    # so registering a repo would land where the fleet supervisor never reads.
    # Warn before the write; never block it.
    warn_fleet_dir_virtualization_on_write(fleet_json_path.parent, context="writing fleet.json")

    # Ensure fleet directory exists before locking
    fleet_json_path.parent.mkdir(parents=True, exist_ok=True)

    with state_lock(fleet_json_path):
        data = _load_registry(fleet_json_path)
        repos = data.setdefault("repos", {})

        entry = repos.get(name_with_owner, {})

        # Build/update the entry
        now = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        # Preserve first_seen if this is the same repo (same nameWithOwner), even if repo_root changed
        # This handles the "moved repo" case from the acceptance criteria
        updated_entry = {
            "repo_root": str(repo_root),
            "name_with_owner": name_with_owner,
            "config_path": str(repo_root / DEFAULT_CONFIG_FILENAME),
            "state_dir": str(paths.root),
            "first_seen": entry.get("first_seen", now) if entry else now,
            "last_seen": now,
        }

        repos[name_with_owner] = updated_entry
        data["repos"] = repos

        return save_state(fleet_json_path, data)


def count_fleet_live_sessions(
    fleet_dir_override: str | None,
) -> tuple[int, list[str]]:
    """Count live worker sessions across all registered repos in the fleet.

    Reads the fleet registry, iterates over each registered repo, resolves its
    sessions_dir, and counts live workers using the adapter-agnostic iter_workers
    from worker.py. Tolerates per-repo problems by skipping them and returning a
    list of skipped repo keys for operator visibility.

    Args:
        fleet_dir_override: Optional override for the fleet directory path.

    Returns:
        A tuple of (total_live_count, skipped_repos) where skipped_repos is a
        list of name_with_owner keys for any of:
        - repo_root missing or not a git worktree,
        - state_dir missing,
        - config load failure (missing/unreadable/malformed/invalid), or
        - resolved sessions_dir missing.
    """
    fleet_json_path = layout.fleet_registry_path(override=fleet_dir_override)
    data = _load_registry(fleet_json_path)
    repos = data.get("repos", {})

    total_live_count = 0
    skipped_repos: list[str] = []

    for name_with_owner, entry in repos.items():
        repo_root_str = entry.get("repo_root")
        if not repo_root_str:
            continue

        repo_root = Path(repo_root_str)

        # Skip if repo_root no longer exists
        if not repo_root.exists():
            logger.warning(
                f"Skipping fleet live-count for {name_with_owner}: repo_root {repo_root} does not exist"
            )
            skipped_repos.append(name_with_owner)
            continue

        # Skip if not a git worktree (basic sanity check)
        if not (repo_root / ".git").exists():
            logger.warning(
                f"Skipping fleet live-count for {name_with_owner}: repo_root {repo_root} is not a git worktree"
            )
            skipped_repos.append(name_with_owner)
            continue

        # Resolve sessions_dir from the registry entry's state_dir
        # The state_dir is the .var/charlie-work root for that repo
        state_dir = Path(entry.get("state_dir", ""))
        if not state_dir.exists():
            logger.warning(
                f"Skipping fleet live-count for {name_with_owner}: state_dir {state_dir} does not exist"
            )
            skipped_repos.append(name_with_owner)
            continue

        # Resolve sessions_dir from the repo's effective layered config.
        # The registry's state_dir is the resolved state root, but the repo's
        # per-repo orchestrator.config.yaml may not declare devin.sessions_dir;
        # the fleet-wide <fleet_dir>/config.yaml layer is also allowed to set it
        # (known_config_sections() includes "devin"). Use load_layered_config so
        # both layers are merged and an explicit devin.sessions_dir is resolved
        # against repo_root via the single sentinel resolver in layout.py.
        explicit_cfg = entry.get("config_path")
        try:
            repo_config = load_layered_config(
                repo_root,
                Path(explicit_cfg) if explicit_cfg else None,
                fleet_dir_override=fleet_dir_override,
            )
        except Exception as exc:  # noqa: BLE001 - containment is deliberate
            logger.warning(
                f"Skipping fleet live-count for {name_with_owner}: "
                f"failed to load repo config for {repo_root}: {exc}"
            )
            skipped_repos.append(name_with_owner)
            continue

        sessions_dir = layout.resolve_state_child(
            repo_config.devin.sessions_dir,
            repo_root=repo_root,
            default=layout.sessions_dir_default(state_dir),
        )
        if not sessions_dir.exists():
            logger.warning(
                f"Skipping fleet live-count for {name_with_owner}: "
                f"sessions_dir {sessions_dir} does not exist"
            )
            skipped_repos.append(name_with_owner)
            continue

        # Count live workers using adapter-agnostic iter_workers
        workers = iter_workers(sessions_dir, repo_key=name_with_owner)
        live_count = sum(1 for worker in workers if worker.is_alive())
        total_live_count += live_count

    return total_live_count, skipped_repos


FleetLock = ByteRangeFileLock


def try_acquire_fleet_lock(fleet_dir_override: str | None) -> FleetLock | None:
    """Try to acquire the fleet-wide dispatch lock non-blocking.

    Returns a ``FleetLock`` if acquired; ``None`` if another process (or thread)
    holds it. Never raises. The lock is held across the read→dispatch window so
    that independently-running supervised loops cannot over-dispatch against a
    shared fleet-wide cap.
    """
    try:
        lock_path = layout.fleet_lock_path(override=fleet_dir_override)

        thread_lock = _fleet_thread_lock_for(lock_path)
        if not thread_lock.acquire(blocking=False):
            return None
        try:
            return try_acquire_byte_range_lock(lock_path)
        finally:
            thread_lock.release()
    except OSError:
        return None


def count_fleet_runners(
    fleet_dir_override: str | None,
    runtime: RuntimeConfig | None = None,
) -> tuple[int, int, list[str]]:
    """Count fleet-wide runners across all registered repos.

    Reads the fleet registry, iterates over each registered repo, and counts
    total and busy runners using the GitHub API. Tolerates vanished/moved repo dirs
    by skipping them and returning a list of skipped repo keys for operator visibility.

    Args:
        fleet_dir_override: Optional override for the fleet directory path.

    Returns:
        A tuple of (total_runners, total_busy_runners, skipped_repos) where
        skipped_repos is a list of name_with_owner keys whose repo_root no longer
        exists or is not a git worktree.
    """
    fleet_json_path = layout.fleet_registry_path(override=fleet_dir_override)
    data = _load_registry(fleet_json_path)
    repos = data.get("repos", {})

    total_runners = 0
    total_busy_runners = 0
    skipped_repos: list[str] = []

    for name_with_owner, entry in repos.items():
        repo_root_str = entry.get("repo_root")
        if not repo_root_str:
            continue

        repo_root = Path(repo_root_str)

        # Skip if repo_root no longer exists
        if not repo_root.exists():
            logger.warning(
                f"Skipping fleet runner-count for {name_with_owner}: repo_root {repo_root} does not exist"
            )
            skipped_repos.append(name_with_owner)
            continue

        # Skip if not a git worktree (basic sanity check)
        if not (repo_root / ".git").exists():
            logger.warning(
                f"Skipping fleet runner-count for {name_with_owner}: repo_root {repo_root} is not a git worktree"
            )
            skipped_repos.append(name_with_owner)
            continue

        # Create a GitHub client for this repo
        try:
            repo_gh = GitHub(repo_root=repo_root, runtime=runtime)
        except GitHubError:
            logger.warning(
                f"Skipping fleet runner-count for {name_with_owner}: failed to create GitHub client"
            )
            skipped_repos.append(name_with_owner)
            continue

        # Query runners for this repo
        try:
            runners_data = repo_gh.run(
                ["api", "repos/{owner}/{repo}/actions/runners"], json_output=True
            )
            runners = runners_data.get("runners", []) if runners_data else []
            total_runners += len(runners)
            total_busy_runners += sum(1 for r in runners if r.get("busy") is True)
        except (GitHubError, Exception) as exc:
            logger.warning(
                f"Skipping fleet runner-count for {name_with_owner}: failed to query runners: {exc}"
            )
            skipped_repos.append(name_with_owner)
            continue

    return total_runners, total_busy_runners, skipped_repos
