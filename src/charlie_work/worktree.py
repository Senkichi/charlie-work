"""Git worktree lifecycle for isolated per-branch worker environments.

Ports the battle-tested job-cannon shell scripts (``setup_worker.sh`` /
``finish_worker.sh``) into library code. The critical invariant this module
exists to enforce: worktrees may share ONE dev+eval virtualenv via a Windows
junction (or a symlink elsewhere) at ``<worktree>/.venv``, and naive removal
(``git worktree remove --force`` / ``rm -rf``) FOLLOWS that reparse point and
recursively deletes the shared venv's contents — corrupting every other live
worktree. ``remove_worktree`` orders teardown so the junction itself is
unlinked (never the target it points at) before the worktree is removed.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from .attempt_refs import AttemptSnapshot, snapshot_attempt_ref
from .config import OrchestratorConfig
from .github import GitHub, GitHubRunResult, PR_VIEW_MERGED_FIELDS, linked_issue_number
from .post_mortem import real_activity_for_worker
from .process_utils import is_pid_alive
from .subprocess_runner import run_captured

_DEFAULT_TIMEOUT_SECONDS = 60


class WorktreeUnsafeError(RuntimeError):
    """Raised when ``create_worktree`` is about to reset a worktree that is
    CONFIRMED to contain local work (uncommitted modifications, or local
    commits not on the remote branch). This is a deterministic finding — the
    launch shim surfaces it as ``failure_kind="worktree_unsafe"``, which sits
    in ``config.DETERMINISTIC_ESCALATION_FAILURE_KINDS`` and escalates to a
    human on first occurrence. Do not raise this for a probe that merely
    *failed to determine* dirtiness — see ``WorktreeProbeFailedError``.
    """


class WorktreeProbeFailedError(RuntimeError):
    """Raised when the safety probe used to decide whether a worktree/branch
    reset would destroy local work itself failed (e.g. ``git status
    --porcelain`` hit an index lock, I/O error, or permissions issue) — NOT
    when the probe ran and confirmed the worktree is dirty.

    This is transient contention that an ordinary redispatch-cap retry would
    plausibly heal, unlike a confirmed-dirty worktree. The launch shim
    surfaces it as ``failure_kind="worktree_probe_failed"``, which must stay
    OFF ``config.DETERMINISTIC_ESCALATION_FAILURE_KINDS`` so it takes the
    normal redispatch-cap path instead of escalating on first occurrence
    (issue #288 follow-up review finding, PR #314).

    The reset is still refused (the caller cannot confirm safety), so this
    is a sibling of ``WorktreeUnsafeError`` rather than a subclass of it —
    keeping the two independent avoids an ``isinstance`` check on the more
    general class silently reclassifying a probe failure as confirmed-dirty.
    """


class LiveWorkerRedispatchError(RuntimeError):
    """Raised when a recovery/redispatch path is about to destroy a worktree
    but the recorded worker process is still alive (or sessions.db shows fresh
    activity). Carries the probe result so the orchestrator can restore labels
    and emit a diagnostic event.
    """

    def __init__(
        self,
        *,
        issue_number: int | None,
        pid: int | None,
        process_start_time: float | None,
        probe_result: str,
    ) -> None:
        self.issue_number = issue_number
        self.pid = pid
        self.process_start_time = process_start_time
        self.probe_result = probe_result
        super().__init__(probe_result)


# Known porcelain flag keys that may appear as space-less lines (value=True)
# These are the only keys that map to True in git worktree --porcelain output
KNOWN_FLAG_KEYS = frozenset({"bare", "detached", "locked", "prunable"})


@dataclass(frozen=True)
class WorktreeInfo:
    path: Path
    branch: str
    venv_junction: Path | None
    reclaimed: str | None = None  # "fetch-fallback" | "pruned" | "salvaged" | None
    # Set when a redispatch reset a branch tip that had commits worth
    # preserving (issue #261) — see attempt_refs.snapshot_attempt_ref.
    attempt_snapshot: AttemptSnapshot | None = None


class WorktreeState(str, Enum):
    """Classification of a worker worktree after the process dies.

    - completed: branch is ahead of the dispatch base and the working tree is clean.
    - partial: the worktree has uncommitted changes (with or without commits).
    - no_commits: the working tree is clean and has no commits beyond the base.
    - unknown: git probing failed; treat as partial and avoid destructive actions.
    """

    COMPLETED = "completed"
    PARTIAL = "partial"
    NO_COMMITS = "no_commits"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class WorktreeInspection:
    """Result of inspecting a worker worktree after the process dies."""

    state: WorktreeState
    ahead_count: int = 0
    dirty: bool = False
    resolved_base_ref: str | None = None
    error: str | None = None


def _slugify(value: str, *, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:max_length].rstrip("-") or "worktree"


def _default_worktrees_dir(repo_root: Path) -> Path:
    return repo_root / ".var" / "charlie-work" / "worktrees"


def _read_origin_head_symref(repo_root: Path) -> str | None:
    """Read refs/remotes/origin/HEAD and return "origin/<branch>", or None."""
    result = run_captured(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if result.ok and result.stdout.strip():
        # Output is "refs/remotes/origin/main" -> extract "origin/main"
        ref = result.stdout.strip()
        if ref.startswith("refs/remotes/"):
            return ref[len("refs/remotes/") :]
    return None


def _resolve_default_branch_ref(repo_root: Path) -> str:
    """Resolve the repository's default branch as a remote-tracking ref.

    Returns a string like "origin/main" or "origin/master". Only when the repo
    has no origin remote at all (pure-local repos, e.g. test fixtures) does it
    return "HEAD".

    Uses git symbolic-ref refs/remotes/origin/HEAD, the standard way to read
    the default branch without the GitHub CLI. That ref is only written at
    clone time (or by git remote set-head), so when it is missing, heal it
    with ``git remote set-head origin --auto`` and retry once.

    Raises:
        RuntimeError: an origin remote exists but the default branch cannot be
            determined even after the set-head heal. Falling back to local
            HEAD here is how a stale/dirty operator checkout leaks into fresh
            worker branches (issue #239: two agent PRs shipped the operator's
            unpublished docs commit). The adapter boundary converts this raise
            into a SessionRecord error value, failing the one dispatch loudly.
    """
    if not _has_origin_remote(repo_root):
        # Pure-local repo: local HEAD is the only meaningful base.
        return "HEAD"

    resolved = _read_origin_head_symref(repo_root)
    if resolved is not None:
        return resolved

    # origin/HEAD is unset on this clone; set-head persists it in-repo so both
    # future dispatches and operator git usage benefit. Failure surfaces via
    # the retried symref read below, not an exception (run_captured contract).
    run_captured(
        ["git", "remote", "set-head", "origin", "--auto"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    resolved = _read_origin_head_symref(repo_root)
    if resolved is not None:
        return resolved

    raise RuntimeError(
        "Cannot resolve origin's default branch: refs/remotes/origin/HEAD is "
        "unset and 'git remote set-head origin --auto' did not heal it. "
        "Refusing to base a fresh worktree on local HEAD (issue #239)."
    )


def _has_origin_remote(repo_root: Path) -> bool:
    """Check if the repo has an 'origin' remote configured.

    Returns True if 'git remote get-url origin' succeeds (exit code 0),
    False otherwise. This is a deterministic check for remote existence
    before attempting fetch operations.
    """
    result = run_captured(
        ["git", "remote", "get-url", "origin"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    return result.ok


def _remote_branch_exists(repo_root: Path, branch: str) -> bool | None:
    """Check if a branch exists on the origin remote.

    Returns True if the branch exists, False if it does not exist, or None if the probe
    failed (network/auth error). This distinguishes 'remote ref missing' from 'broken remote':
    - exists: exit 0 AND non-empty stdout
    - missing: exit 0 AND empty stdout
    - probe-failed: nonzero exit (e.g., network error, auth failure)
    """
    result = run_captured(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not result.ok:
        # Probe failed (network error, auth failure, etc.)
        return None
    if result.stdout.strip():
        # Branch exists
        return True
    # Branch does not exist (exit 0 with empty stdout)
    return False


def _worker_authored_dirty(
    worktree_path: Path,
    injected_paths: tuple[str, ...] = (),
) -> bool:
    """Return True if the worktree has uncommitted changes that are NOT
    orchestrator-injected prompt files.

    ``injected_paths`` are worktree-relative paths (files or directories) that
    the orchestrator writes into the worktree (e.g. rendered worker/rework
    prompts). They are excluded from the dirty check so completed worker work
    is not stranded by prompt-injection noise (issue #381).

    Raises:
        WorktreeProbeFailedError: if the ``git status --porcelain`` probe itself
            fails (index lock, corruption, permissions).
    """
    status_result = run_captured(
        ["git", "-c", "core.quotePath=off", "status", "--porcelain"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not status_result.ok:
        raise WorktreeProbeFailedError("worktree status probe failed; treating as dirty")

    injected = [PurePosixPath(str(p)) for p in injected_paths]
    for line in status_result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Porcelain format: two status chars, a space, then the path.
        # Renames include "old -> new"; the right-hand side is the current path.
        if len(line) < 4:
            continue
        raw_path = line[3:]
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ")[-1]
        # Git may emit backslashes on Windows; normalize for comparison.
        path = PurePosixPath(str(raw_path).replace("\\", "/"))
        if any(
            path == injected_path or injected_path in path.parents or path in injected_path.parents
            for injected_path in injected
        ):
            continue
        return True
    return False


def _worktree_refuse_to_reset_reason(
    repo_root: Path,
    branch: str,
    base_ref: str,
    worktree_path: Path | None = None,
    injected_paths: tuple[str, ...] = (),
) -> str | None:
    """Return a human-readable reason if resetting the branch/worktree would destroy
    local work, otherwise ``None``.

    Checks for:
      - uncommitted modifications in ``worktree_path`` (if it exists)
      - local commits that are not present on the remote branch (``git ls-remote``
        comparison, falling back to the merge-base with ``base_ref`` when the remote
        branch does not exist)

    This is read-only: it never commits, fetches, or resets.

    Raises:
        WorktreeProbeFailedError: if the ``git status --porcelain`` probe itself
            fails (index lock, corruption, permissions). The reset is still
            refused (we cannot confirm the worktree is clean), but this is
            classified distinctly from a confirmed-dirty worktree — see the
            class docstring for why callers must not conflate the two under
            the same ``failure_kind``.
    """
    # Uncommitted modifications are only meaningful when the worktree directory exists.
    if worktree_path is not None and worktree_path.is_dir():
        if _worker_authored_dirty(worktree_path, injected_paths):
            return "worktree has uncommitted modifications"
        local_tip_result = run_captured(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
    else:
        local_tip_result = run_captured(
            ["git", "rev-parse", "--verify", branch],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )

    if not local_tip_result.ok or not local_tip_result.stdout.strip():
        # No branch or no commit; nothing to lose.
        return None

    local_sha = local_tip_result.stdout.strip()

    # Compare against the remote branch via git ls-remote.
    remote_sha: str | None = None
    if _has_origin_remote(repo_root):
        ls_remote_result = run_captured(
            ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if ls_remote_result.ok and ls_remote_result.stdout.strip():
            remote_sha = ls_remote_result.stdout.strip().split()[0]

    if remote_sha is None:
        # Branch does not exist on origin (or the probe failed). Any commits beyond
        # the base ref are unpushed and must not be discarded.
        base = base_ref if base_ref else _resolve_default_branch_ref(repo_root)
        merge_base_result = run_captured(
            ["git", "merge-base", base, local_sha],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not merge_base_result.ok:
            return "worktree has local commits not on remote branch"
        merge_base = merge_base_result.stdout.strip()
        rev_list_result = run_captured(
            ["git", "rev-list", "--count", f"{merge_base}..{local_sha}"],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if (
            rev_list_result.ok
            and rev_list_result.stdout.strip().isdigit()
            and int(rev_list_result.stdout.strip()) > 0
        ):
            return f"worktree has {rev_list_result.stdout.strip()} local commit(s) not on remote branch"
        return None

    # Remote branch exists. If local tip matches the remote tip, there are no
    # local-only commits.
    if local_sha == remote_sha:
        return None

    # local_sha != remote_sha. If local_sha is an ancestor of remote_sha, the local
    # branch is behind the remote and has no local commits not on remote. Otherwise
    # local is ahead or has diverged, which means local commits not on remote.
    ancestor_result = run_captured(
        ["git", "merge-base", "--is-ancestor", local_sha, remote_sha],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if ancestor_result.ok:
        return None

    return "worktree has local commits not on remote branch"


def _worktree_dirty_reason(
    worktree_path: Path,
    injected_paths: tuple[str, ...] = (),
) -> str | None:
    """Return a reason string if ``worktree_path`` has uncommitted modifications.

    Used by the merged-PR cleanup path (``clean_worktrees``), which decides
    eligibility by comparing the worktree's local HEAD against the merged PR's
    ``headRefOid`` rather than by comparing against the (possibly already
    deleted) remote branch — see ``clean_worktrees`` for why
    ``_worktree_refuse_to_reset_reason``'s remote-branch comparison does not
    apply once a PR has been squash-merged with its branch deleted.

    Raises:
        WorktreeProbeFailedError: if the ``git status --porcelain`` probe itself
            fails (index lock, corruption, permissions) — mirrors
            ``_worktree_refuse_to_reset_reason``'s handling of the same failure
            mode.
    """
    if not worktree_path.is_dir():
        return None
    if _worker_authored_dirty(worktree_path, injected_paths):
        return "worktree has uncommitted modifications"
    return None


def _salvage_worktree(repo_root: Path, worktree_path: Path, branch: str) -> str | None:
    """Salvage a worktree with uncommitted changes or unpushed commits.

    Commits the current state to a salvage ref, pushes it to origin, and returns
    the salvage ref name. Returns None if the worktree is clean (nothing to salvage).
    Raises RuntimeError if the salvage push fails.
    """
    from .state import utc_now

    # Check if there's anything to salvage
    dirty_result = run_captured(
        ["git", "status", "--porcelain"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    # If the probe fails (index lock, corruption, permissions), treat as dirty to be safe
    has_dirty = not dirty_result.ok or bool(dirty_result.stdout.strip())

    # Check for unpushed commits (only if branch exists on origin)
    has_unpushed = False
    remote_exists = None
    if _has_origin_remote(repo_root):
        remote_exists = _remote_branch_exists(repo_root, branch)
    if remote_exists is True:
        unpushed_result = run_captured(
            ["git", "log", f"origin/{branch}..HEAD", "--oneline"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        has_unpushed = unpushed_result.ok and bool(unpushed_result.stdout.strip())
    else:
        # Branch doesn't exist on origin: any commits are considered "unpushed"
        # (killed before first push scenario)
        # Check if HEAD has any commits beyond the default branch tip
        default_ref = _resolve_default_branch_ref(repo_root)
        if default_ref == "HEAD":
            # No origin remote: there is no authoritative default-branch tip to compare against.
            # Conservatively assume there are unpushed commits to preserve work.
            # This replaces the previous accidental behavior (failed merge-base → same default)
            # with an intentional, documented trade-off.
            has_unpushed = True
        else:
            # Origin exists: compare against the remote-tracking tip directly.
            # Using the remote-tracking ref (e.g., "origin/master") is the correct salvage
            # semantic — we want to know if there are commits beyond what is actually on
            # the shared default branch. No local-name conversion is needed.
            merge_base_result = run_captured(
                ["git", "merge-base", "HEAD", default_ref],
                cwd=worktree_path,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            )
            if merge_base_result.ok:
                merge_base = merge_base_result.stdout.strip()
                rev_list_result = run_captured(
                    ["git", "rev-list", "--count", f"{merge_base}..HEAD"],
                    cwd=worktree_path,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                has_unpushed = rev_list_result.ok and int(rev_list_result.stdout.strip()) > 0
            else:
                # If merge-base fails, assume has commits to be safe
                has_unpushed = True

    if not has_dirty and not has_unpushed:
        return None

    # Create salvage ref name
    timestamp = utc_now().replace(":", "-").replace("+00:00", "Z")
    salvage_ref = f"salvage/{branch.replace('/', '-')}-{timestamp}"

    # Commit dirty changes if any
    if has_dirty:
        run_captured(
            ["git", "add", "-A"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        run_captured(
            ["git", "commit", "-m", f"Salvage before worktree cleanup: {timestamp}"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )

    # Create the salvage ref
    run_captured(
        ["git", "update-ref", f"refs/{salvage_ref}", "HEAD"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )

    # Push the salvage ref if origin exists
    if _has_origin_remote(repo_root):
        push_result = run_captured(
            ["git", "push", "origin", f"refs/{salvage_ref}:refs/{salvage_ref}"],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not push_result.ok:
            raise RuntimeError(
                f"Failed to push salvage ref {salvage_ref!r} to origin: "
                f"{push_result.error or push_result.stderr}"
            )

    return salvage_ref


def is_junction(path: Path) -> bool:
    """Return True if ``path`` is a reparse point (Windows junction/symlink)
    or, on non-Windows platforms, a symlink. ``os.path.islink()`` alone is
    unreliable for junctions on some Windows Python builds, so the Windows
    path checks the reparse-point file attribute directly."""
    if os.name == "nt":
        try:
            result = os.stat(path, follow_symlinks=False)
        except OSError:
            return False
        return bool(result.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return os.path.islink(path)


def _unlink_reparse_point(path: Path) -> None:
    """Remove a reparse point (Windows junction/symlink) or POSIX symlink.

    ``os.rmdir`` is used on Windows because it removes the reparse point
    itself without following into the target directory. ``os.unlink`` is used
    on POSIX because ``os.rmdir`` raises on a symlink. In both cases the
    target is left untouched.
    """
    if os.name == "nt":
        os.rmdir(path)
    else:
        os.unlink(path)


def _create_junction_or_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or is_junction(link_path):
        raise RuntimeError(f"venv link target already exists: {link_path}")
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target_path), str(link_path))
    else:
        try:
            os.symlink(target_path, link_path, target_is_directory=True)
        except FileExistsError as exc:
            raise RuntimeError(f"venv link target already exists: {link_path}") from exc


def _is_git_tracked(repo_root: Path, path: Path) -> bool:
    """Check if a path is tracked by git in the repo.

    Returns True if the path is tracked, False if it's untracked or doesn't exist.
    """
    result = run_captured(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(repo_root))],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    return result.ok


def _materialize_directory(repo_root: Path, worktree_path: Path, dir_path: str) -> None:
    """Copy a directory from repo_root to worktree_path if it's not tracked.

    Skip if the path is already tracked by git (it will be in the worktree).
    Copy-not-link (workers may write marker files). Errors surface as OSError.
    """
    source = repo_root / dir_path
    if not source.exists():
        return  # Source doesn't exist, nothing to copy

    # Check if the path is tracked by git
    if _is_git_tracked(repo_root, source):
        return  # Tracked paths are already in the worktree

    # Copy the directory to the worktree
    target = worktree_path / dir_path
    if target.exists():
        return  # Already exists in worktree (shouldn't happen, but be safe)

    shutil.copytree(source, target)


def _probe_recovery_liveness(
    recovery: dict[str, Any],
    worktree_path: Path,
    config: OrchestratorConfig | None,
    issue_number: int | None,
) -> None:
    """Abort recovery/redispatch if the recorded worker is still alive.

    Checks the recorded PID with start-time fingerprint matching first. If the
    PID is dead/wrong, corroborate with the Devin CLI's real-activity probe
    (``real_activity_for_worker``) for devin-shell sessions — a recent
    ``sessions.db`` message_nodes row OR a fresh per-PID Devin log for this
    worktree means the real leaf session is still moving even if the wrapper
    PID is gone.

    The probe is best-effort and never raises by design: a schema-drifted or
    locked sessions.db, or an unreadable per-PID log, comes back as an
    ``ActivitySource`` with ``error`` set rather than a timestamp. That is an
    INCONCLUSIVE result, not a confirmed-dead one — treating it as "no
    activity" would fail OPEN into the same destructive reset issue #282 exists
    to prevent (this reintroduced that exact shape once already; see the
    review that required this fix). So any source with a non-null ``error``
    aborts recovery just as certainly as confirmed fresh activity does.

    Raises ``LiveWorkerRedispatchError`` when a live signal is detected, or
    when liveness could not be determined.
    """
    resolved_config = config or OrchestratorConfig()

    worker_pid = recovery.get("worker_pid")
    worker_process_start_time = recovery.get("worker_process_start_time")
    if worker_pid is None:
        worker_pid = recovery.get("last_known_worker_pid")
        worker_process_start_time = recovery.get("last_known_worker_process_start_time")

    try:
        worker_pid = int(worker_pid) if worker_pid is not None else None
    except (TypeError, ValueError):
        worker_pid = None

    if worker_pid is not None and is_pid_alive(worker_pid, worker_process_start_time):
        raise LiveWorkerRedispatchError(
            issue_number=issue_number,
            pid=worker_pid,
            process_start_time=worker_process_start_time,
            probe_result="pid_alive",
        )

    # For devin-shell sessions, the real-activity probe (sessions.db +
    # per-PID Devin log) is the source of truth even when the wrapper PID is
    # gone or has been recycled.
    if resolved_config.devin.adapter == "devin-shell":
        started_at = recovery.get("started_at") or recovery.get("dispatched_at") or ""
        pm_config = resolved_config.post_mortem
        now = datetime.now(UTC)
        try:
            probe = real_activity_for_worker(
                pm_config,
                str(worktree_path),
                started_at,
                worker_pid,
                now,
            )
        except Exception:
            # The probe is best-effort; never let it crash the recovery path.
            return

        # Any errored source means liveness is UNKNOWN for that signal, not
        # confirmed dead — never let an inconclusive probe fall through to a
        # destructive reset (issue #282 rework).
        for source in probe.sources:
            if source.error is not None:
                raise LiveWorkerRedispatchError(
                    issue_number=issue_number,
                    pid=worker_pid,
                    process_start_time=worker_process_start_time,
                    probe_result="probe_error",
                )

        # Consult BOTH activity sources via the probe's own freshest-signal
        # aggregation (the same corroboration classify_worker_health uses for
        # the stall watchdog, issue #280) instead of a sessions.db-only check
        # that would ignore fresh devin_per_pid_log activity.
        stall_minutes = resolved_config.watchdog.stall_minutes or 20
        stall_seconds = stall_minutes * 60
        if probe.latest_timestamp is not None:
            staleness_seconds = (now - probe.latest_timestamp).total_seconds()
            if staleness_seconds <= stall_seconds:
                source_label = (probe.latest_source or "real_activity").replace(".", "_")
                raise LiveWorkerRedispatchError(
                    issue_number=issue_number,
                    pid=worker_pid,
                    process_start_time=worker_process_start_time,
                    probe_result=f"{source_label}_activity",
                )


def create_worktree(
    repo_root: Path,
    branch: str,
    *,
    base_ref: str = "HEAD",
    worktrees_dir: Path | None = None,
    venv_source: Path | None = None,
    materialize_dirs: tuple[str, ...] = (),
    rework: bool = False,
    recovery: dict[str, Any] | None = None,
    issue_number: int | None = None,
    config: OrchestratorConfig | None = None,
) -> WorktreeInfo:
    """Create a git worktree for ``branch`` (a new branch) off ``base_ref``.

    If ``venv_source`` is given, a Windows junction (symlink elsewhere) is
    created at ``<worktree>/.venv`` pointing at it, so workers share one
    dev+eval virtualenv instead of cold-building their own. Raises
    RuntimeError if that link target already exists in the fresh worktree
    (programmer error / stale state — fail loudly rather than silently
    reusing or overwriting it).

    If ``rework`` is True, the branch is assumed to already exist (from a
    previous PR cycle). In rework mode:
    - If a worktree for the branch already exists, fetch and fast-forward it
      to the origin tip instead of failing.
    - Otherwise, use ``git worktree add <path> <branch>`` (no ``-b``) to
      attach to the existing branch at its origin tip.

    If ``recovery`` is provided (a dict with state file dispatch record),
    this is a dead-worker recovery re-dispatch. The dict must contain
    ``branch_name`` matching the requested ``branch``. Recovery mode:
    - If the leftover worktree/branch has NO commits beyond the merge-base
      with ``base_ref`` (crashed before committing): remove worktree + branch
      and create fresh — a clean restart.
    - If it HAS commits or a dirty tree (crashed mid-work): reuse via the
      existing rework-style attach (fetch/ff if possible), so the relaunched
      worker continues from the partial work.
    - If the branch exists WITHOUT a matching state record (foreign state):
      fail loudly — that protects against clobbering anything that is not ours.

    The ``base_ref`` parameter controls where the new branch bases off:
    - Empty string ("") means auto-resolve to the repository's default branch
      as a remote-tracking ref (e.g., "origin/main"). This is the recommended
      setting for production to ensure fresh worktrees base off the latest
      remote tip instead of a potentially stale local HEAD.
    - A remote-tracking ref like "origin/main" or "origin/master" will trigger
      a git fetch before worktree creation to ensure the ref is up-to-date.
    - Any other ref (e.g., "HEAD", a commit SHA, or a local branch name) is
      used as-is without fetching.

    ``issue_number``, when given, enables attempt-tip preservation (issue
    #261): immediately before any ``git branch -D`` in this function that
    could discard commits a dead worker made but never got to push, the old
    tip is snapshotted to ``refs/charlie/attempts/issue-<n>/attempt-<k>``
    (see ``attempt_refs.snapshot_attempt_ref``) and surfaced on the returned
    ``WorktreeInfo.attempt_snapshot``. Best-effort — a snapshot failure never
    blocks the branch reset. When ``issue_number`` is None, no snapshot is
    attempted (existing callers/tests are unaffected).
    """
    # Resolve base_ref: empty string means auto-resolve to origin/<default>
    resolved_base_ref = base_ref
    if base_ref == "":
        resolved_base_ref = _resolve_default_branch_ref(repo_root)

    # Fetch if the resolved base_ref is a remote-tracking ref (origin/<branch>)
    # Only do this for fresh dispatch (not rework/recovery) to avoid moving existing tips
    if not rework and recovery is None and resolved_base_ref.startswith("origin/"):
        # Extract the branch name from "origin/<branch>"
        remote_branch = resolved_base_ref[len("origin/") :]
        fetch_result = run_captured(
            ["git", "fetch", "origin", remote_branch],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not fetch_result.ok:
            raise RuntimeError(
                f"Failed to fetch base ref {resolved_base_ref!r} before worktree creation: "
                f"{fetch_result.error or fetch_result.stderr}"
            )

    target_dir = worktrees_dir or _default_worktrees_dir(repo_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    worktree_path = target_dir / _slugify(branch)

    # Worktree-relative paths that the orchestrator injects (e.g. rendered
    # prompts). These are excluded from "is this worktree dirty?" checks.
    injected_paths = config.dispatch.injected_paths if config is not None else ()

    # Recovery mode: dead-worker re-dispatch with leftover worktree/branch
    reclaimed: str | None = None
    attempt_snapshot: AttemptSnapshot | None = None

    def _snapshot_before_delete(target_branch: str) -> None:
        """Best-effort attempt-tip snapshot immediately before a branch reset.

        Updates the enclosing ``attempt_snapshot`` (last snapshot wins — only
        one branch-delete path fires per call). No-op when ``issue_number``
        was not provided.
        """
        nonlocal attempt_snapshot
        if issue_number is None:
            return
        attempt_snapshot = snapshot_attempt_ref(
            repo_root, target_branch, issue_number, base_ref=resolved_base_ref
        )

    def _raise_if_unsafe_to_reset(target_path: Path | None = None) -> None:
        """Hard-refuse to reset if the worktree/branch contains local work."""
        check_path = target_path or worktree_path
        reason = _worktree_refuse_to_reset_reason(
            repo_root, branch, resolved_base_ref, check_path, injected_paths
        )
        if reason:
            raise WorktreeUnsafeError(reason)

    if recovery is not None:
        # Validate that the recovery record matches the requested branch
        recovery_branch = recovery.get("branch_name")
        if recovery_branch != branch:
            raise RuntimeError(
                f"Recovery record branch_name {recovery_branch!r} does not match "
                f"requested branch {branch!r}"
            )

        # Issue #282: verify the prior worker is actually dead before any
        # destructive reset. If the recorded PID (or sessions.db real activity)
        # says the worker is still alive, abort the redispatch and let the
        # orchestrator restore the in-progress label.
        _probe_recovery_liveness(recovery, worktree_path, config, issue_number)

        # Issue #110: Check if the branch exists on origin before attempting fetch
        # If the branch doesn't exist on origin (killed before first push), fall through
        # to fresh dispatch instead of failing with fetch error 128
        # If the probe fails (network/auth error), abort dispatch to avoid data loss
        remote_exists = None
        if _has_origin_remote(repo_root):
            remote_exists = _remote_branch_exists(repo_root, branch)
            if remote_exists is None:
                # Probe failed: abort dispatch to avoid triggering fallback on transient error
                raise RuntimeError(
                    f"Failed to probe remote branch {branch!r} for recovery: "
                    f"transient network or auth error. Aborting dispatch to avoid data loss."
                )
        if remote_exists is False:
            # Branch doesn't exist on origin - this is a killed-before-push session
            # Fall through to fresh dispatch (rework=False) after cleaning up local state
            reclaimed = "fetch-fallback"
            # Clean up any local worktree/branch that might exist
            existing_worktrees = list_worktrees(repo_root)
            existing_wt = next(
                (wt for wt in existing_worktrees if Path(wt["worktree"]) == worktree_path),
                None,
            )
            if existing_wt:
                wt_path = Path(existing_wt["worktree"])
                # Check if it's on the correct branch (should be, since it's our recovery record)
                wt_branch = existing_wt.get("branch", "")
                normalized_wt_branch = wt_branch.replace("refs/heads/", "")
                if (
                    normalized_wt_branch == branch
                    or normalized_wt_branch == f"refs/heads/{branch}"
                ):
                    # It's our worktree - refuse to reset it if it holds local work
                    _raise_if_unsafe_to_reset(wt_path)
                    # Remove the worktree (junction-safe)
                    if not remove_worktree(repo_root, wt_path, force=True):
                        raise RuntimeError(
                            f"Failed to remove leftover worktree {wt_path} for recovery"
                        )
            # Delete the local branch if it exists
            branch_result = run_captured(
                ["git", "branch", "--list", branch],
                cwd=repo_root,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            )
            if branch_result.ok and branch_result.stdout.strip():
                # fetch-fallback: refuse to reset if the branch holds local work.
                _raise_if_unsafe_to_reset(worktree_path)
                # The branch never made it to origin, so any commits here are
                # local-only. If the guard allowed the reset, the worktree is
                # clean and the branch has no local commits, so we can safely
                # remove the branch. Snapshot the tip before deleting it, best-
                # effort, as a defensive artifact for post-mortem.
                _snapshot_before_delete(branch)
                branch_delete_result = run_captured(
                    ["git", "branch", "-D", branch],
                    cwd=repo_root,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                if not branch_delete_result.ok:
                    raise RuntimeError(
                        f"Failed to delete branch {branch!r} for recovery: "
                        f"{branch_delete_result.error or branch_delete_result.stderr}"
                    )
            # Fall through to fresh dispatch below (rework=False)
        else:
            # Branch exists on origin or no origin - proceed with normal recovery logic
            # Check if a worktree exists at the expected path (by slug)
            existing_worktrees = list_worktrees(repo_root)
            existing_wt = next(
                (wt for wt in existing_worktrees if Path(wt["worktree"]) == worktree_path),
                None,
            )

            if existing_wt:
                # AC #3: Fail loudly if the worktree at the expected path is on a FOREIGN branch
                # (i.e., a worktree whose branch does not match our recovery record).
                # This protects against clobbering work that is not ours.
                wt_branch = existing_wt.get("branch", "")
                # Normalize branch names for comparison (strip refs/heads/ prefix)
                normalized_wt_branch = wt_branch.replace("refs/heads/", "")
                if (
                    normalized_wt_branch != branch
                    and normalized_wt_branch != f"refs/heads/{branch}"
                ):
                    raise RuntimeError(
                        f"Recovery mode found leftover worktree at {worktree_path} on foreign branch {normalized_wt_branch!r}, "
                        f"but recovery record specifies branch {branch!r}. "
                        f"This is not our crashed worker — refusing to clobber foreign work."
                    )

                # Worktree exists on the correct branch: check if it has commits beyond the merge-base
                wt_path = Path(existing_wt["worktree"])
                # Check for dirty working tree, ignoring orchestrator-injected files.
                try:
                    has_dirty = _worker_authored_dirty(wt_path, injected_paths)
                except WorktreeProbeFailedError:
                    # If the probe fails (index lock, corruption, permissions), treat as dirty to be safe
                    has_dirty = True

                # Check for commits beyond merge-base with resolved_base_ref
                merge_base_result = run_captured(
                    ["git", "merge-base", resolved_base_ref, branch],
                    cwd=repo_root,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                if merge_base_result.ok:
                    merge_base = merge_base_result.stdout.strip()
                    # Count commits from merge-base to branch tip
                    rev_list_result = run_captured(
                        ["git", "rev-list", "--count", f"{merge_base}..{branch}"],
                        cwd=repo_root,
                        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                    )
                    has_commits = rev_list_result.ok and int(rev_list_result.stdout.strip()) > 0
                else:
                    # If merge-base fails, assume has commits to be safe
                    has_commits = True

                if has_commits or has_dirty:
                    # Has work: reuse via rework-style attach
                    # Fall through to rework logic below by setting rework=True
                    rework = True
                else:
                    # Clean: remove worktree and branch, then create fresh
                    _raise_if_unsafe_to_reset(wt_path)
                    if not remove_worktree(repo_root, wt_path, force=True):
                        raise RuntimeError(
                            f"Failed to remove leftover worktree {wt_path} for recovery"
                        )
                    # Delete the branch and check the result
                    _raise_if_unsafe_to_reset(worktree_path)
                    _snapshot_before_delete(branch)
                    branch_delete_result = run_captured(
                        ["git", "branch", "-D", branch],
                        cwd=repo_root,
                        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                    )
                    if not branch_delete_result.ok:
                        raise RuntimeError(
                            f"Failed to delete branch {branch!r} for recovery: "
                            f"{branch_delete_result.error or branch_delete_result.stderr}"
                        )
                    reclaimed = "pruned"
                    # Fall through to fresh dispatch below (rework=False)
            else:
                # No worktree exists, but branch might exist
                # Check if branch exists
                branch_result = run_captured(
                    ["git", "branch", "--list", branch],
                    cwd=repo_root,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                if branch_result.ok and branch_result.stdout.strip():
                    # Branch exists without worktree: check commits and reuse or delete
                    merge_base_result = run_captured(
                        ["git", "merge-base", resolved_base_ref, branch],
                        cwd=repo_root,
                        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                    )
                    if merge_base_result.ok:
                        merge_base = merge_base_result.stdout.strip()
                        rev_list_result = run_captured(
                            ["git", "rev-list", "--count", f"{merge_base}..{branch}"],
                            cwd=repo_root,
                            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                        )
                        has_commits = (
                            rev_list_result.ok and int(rev_list_result.stdout.strip()) > 0
                        )
                    else:
                        has_commits = True

                    if has_commits:
                        # Has commits: reuse via rework-style attach
                        rework = True
                    else:
                        # Clean: delete branch and create fresh
                        _raise_if_unsafe_to_reset(worktree_path)
                        _snapshot_before_delete(branch)
                        branch_delete_result = run_captured(
                            ["git", "branch", "-D", branch],
                            cwd=repo_root,
                            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                        )
                        if not branch_delete_result.ok:
                            raise RuntimeError(
                                f"Failed to delete branch {branch!r} for recovery: "
                                f"{branch_delete_result.error or branch_delete_result.stderr}"
                            )
                        reclaimed = "pruned"
                        # Fall through to fresh dispatch below (rework=False)

    if rework:
        # Rework mode: branch already exists, reuse or attach to it
        existing_worktrees = list_worktrees(repo_root)
        # Branch names in git worktree list may have refs/heads/ prefix
        existing_wt = next(
            (
                wt
                for wt in existing_worktrees
                if wt.get("branch", "").endswith(f"/{branch}") or wt.get("branch") == branch
            ),
            None,
        )

        if existing_wt:
            # Reuse existing worktree: fetch and fast-forward to origin tip
            worktree_path = Path(existing_wt["worktree"])
            # Only fetch if origin remote exists (deterministic check)
            if _has_origin_remote(repo_root):
                # Fetch the remote-tracking ref only (branch:<branch> fails when branch is checked out)
                fetch_result = run_captured(
                    ["git", "fetch", "origin", branch],
                    cwd=repo_root,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                # Fast-forward inside the worktree if fetch succeeded
                if fetch_result.ok:
                    ff_result = run_captured(
                        ["git", "merge", "--ff-only", f"origin/{branch}"],
                        cwd=worktree_path,
                        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                    )
                    # If fast-forward fails (diverged history), fail the launch
                    if not ff_result.ok:
                        raise RuntimeError(
                            f"Cannot fast-forward rework branch {branch!r} to origin tip: "
                            f"{ff_result.error or ff_result.stderr}"
                        )
                # If fetch failed with origin present, raise (real network/error failure)
                if not fetch_result.ok:
                    raise RuntimeError(
                        f"Fetch failed for rework branch {branch!r}: "
                        f"{fetch_result.error or fetch_result.stderr}"
                    )
            venv_link = worktree_path / ".venv"
            venv_junction: Path | None = None
            if venv_source is not None:
                if venv_link.exists() or is_junction(venv_link):
                    venv_junction = venv_link
                else:
                    try:
                        _create_junction_or_symlink(venv_link, venv_source)
                        venv_junction = venv_link
                    except (OSError, RuntimeError):
                        # Clean up the orphan worktree (but not the branch, which already exists in rework mode)
                        remove_worktree(repo_root, worktree_path, force=True, branch=None)
                        raise
            else:
                # If a pre-existing .venv junction from the previous default era
                # is present, unlink it so the worker's uv sync creates a local
                # .venv instead of writing through the reparse point.
                if is_junction(venv_link):
                    _unlink_reparse_point(venv_link)
            return WorktreeInfo(
                path=worktree_path,
                branch=branch,
                venv_junction=venv_junction,
                reclaimed=reclaimed,
                attempt_snapshot=attempt_snapshot,
            )
        else:
            # No existing worktree: attach to existing branch (no -b flag)
            # Fetch first to ensure we materialize at the origin tip, but only if origin exists
            if _has_origin_remote(repo_root):
                fetch_result = run_captured(
                    ["git", "fetch", "origin", f"{branch}:{branch}"],
                    cwd=repo_root,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                # If fetch failed with origin present, raise (real network/error failure)
                if not fetch_result.ok:
                    raise RuntimeError(
                        f"Fetch failed for rework branch {branch!r}: "
                        f"{fetch_result.error or fetch_result.stderr}"
                    )
            result = run_captured(
                ["git", "worktree", "add", str(worktree_path), branch],
                cwd=repo_root,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            )
            if not result.ok:
                raise RuntimeError(
                    f"git worktree add failed for rework branch {branch!r}: {result.error or result.stderr}"
                )
    else:
        # Fresh dispatch: create new branch off base_ref
        # Issue #110: Stale worktree reclamation before git worktree add
        existing_worktrees = list_worktrees(repo_root)
        existing_wt = next(
            (wt for wt in existing_worktrees if Path(wt["worktree"]) == worktree_path),
            None,
        )
        if existing_wt:
            wt_path = Path(existing_wt["worktree"])
            if not wt_path.exists():
                # Directory missing but worktree still registered: prune it
                run_captured(
                    ["git", "worktree", "prune"],
                    cwd=repo_root,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                reclaimed = "pruned"
            elif wt_path.exists():
                # Refuse to reset a worktree that still contains local work.
                # Partial/dirty worktrees are the redispatch case, not the fresh
                # dispatch case (issue #257).
                _raise_if_unsafe_to_reset(wt_path)
                # Directory is clean and has no local commits: remove and recreate
                if not remove_worktree(repo_root, wt_path, force=True):
                    raise RuntimeError(
                        f"Failed to remove stale worktree {wt_path} for fresh dispatch"
                    )
                reclaimed = "pruned"

        # Delete the branch if it exists (it might be leftover from a killed session)
        branch_result = run_captured(
            ["git", "branch", "--list", branch],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if branch_result.ok and branch_result.stdout.strip():
            # Refuse to reset a branch that still contains local commits.
            _raise_if_unsafe_to_reset(worktree_path)
            # Leftover branch from a killed session may hold real, unpushed
            # commits (e.g. it died between committing and pushing) — snapshot
            # before reclaiming it, same as the fetch-fallback recovery path.
            _snapshot_before_delete(branch)
            branch_delete_result = run_captured(
                ["git", "branch", "-D", branch],
                cwd=repo_root,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            )
            if not branch_delete_result.ok:
                raise RuntimeError(
                    f"Failed to delete branch {branch!r} for fresh dispatch: "
                    f"{branch_delete_result.error or branch_delete_result.stderr}"
                )

        result = run_captured(
            ["git", "worktree", "add", "-b", branch, str(worktree_path), resolved_base_ref],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not result.ok:
            raise RuntimeError(
                f"git worktree add failed for branch {branch!r}: {result.error or result.stderr}"
            )

    venv_junction: Path | None = None
    if venv_source is not None:
        venv_link = worktree_path / ".venv"
        try:
            _create_junction_or_symlink(venv_link, venv_source)
            venv_junction = venv_link
        except (OSError, RuntimeError):
            # Clean up the orphan worktree and branch if junction creation fails
            # (to prevent leaks on stale .venv or bad venv_source)
            # In rework mode, the branch already exists, so don't delete it
            delete_branch = None if rework else branch
            remove_worktree(repo_root, worktree_path, force=True, branch=delete_branch)
            raise

    # Materialize git-excluded directories into the worktree
    for dir_path in materialize_dirs:
        try:
            _materialize_directory(repo_root, worktree_path, dir_path)
        except OSError as exc:
            # Clean up the worktree and branch if materialization fails
            delete_branch = None if rework else branch
            remove_worktree(repo_root, worktree_path, force=True, branch=delete_branch)
            raise RuntimeError(
                f"Failed to materialize directory {dir_path} into worktree: {exc}"
            ) from exc

    return WorktreeInfo(
        path=worktree_path,
        branch=branch,
        venv_junction=venv_junction,
        reclaimed=reclaimed,
        attempt_snapshot=attempt_snapshot,
    )


def remove_worktree(
    repo_root: Path, worktree_path: Path, *, force: bool = False, branch: str | None = None
) -> bool:
    """Remove a worktree, taking care never to follow a ``.venv`` junction
    into a shared virtualenv.

    Teardown order is mandatory:
      1. If ``<worktree>/.venv`` exists and is a real directory (not a
         junction/symlink), ABORT and return False unless ``force=True`` —
         and even then, only the worktree-local directory is removed, never
         a junction target.
      2. If it is a junction/symlink, unlink the reparse point itself
         (``os.rmdir`` on Windows; ``os.unlink`` on POSIX — never follows
         into the target).
      3. ``git worktree remove``.
      4. On failure, ``git worktree prune`` to clear stale metadata.
      5. If ``branch`` is provided, delete the branch with ``git branch -D``.

    Returns False for expected failures (real .venv dir without force, git
    command failure); never raises for those. Programmer errors (e.g. a
    nonexistent repo_root) surface as False via a failed git command, since
    git itself reports the error rather than crashing this function.
    """
    venv_path = worktree_path / ".venv"
    if venv_path.exists() or is_junction(venv_path):
        # Any OS-level failure removing the reparse point / local venv is an
        # "expected failure" per this function's contract (a locked/open file
        # under force=True raises PermissionError/WinError 32) — return False,
        # never raise, so one worktree's teardown can't crash the whole batch.
        try:
            if is_junction(venv_path):
                # Unlink the reparse point itself; never follow into the target.
                _unlink_reparse_point(venv_path)
            elif venv_path.is_dir():
                if not force:
                    return False
                shutil.rmtree(venv_path)
            else:
                venv_path.unlink()
        except OSError:
            return False

    args = ["git", "worktree", "remove", str(worktree_path)]
    if force:
        args.append("--force")
    result = run_captured(args, cwd=repo_root, timeout_seconds=_DEFAULT_TIMEOUT_SECONDS)
    worktree_removed = result.ok
    if not worktree_removed:
        run_captured(
            ["git", "worktree", "prune"], cwd=repo_root, timeout_seconds=_DEFAULT_TIMEOUT_SECONDS
        )

    # Delete the branch if provided (to prevent branch leaks on launch failure)
    # Attempt branch deletion independently of worktree-removal success to avoid
    # leaking branches when worktree removal itself fails (e.g., Windows file locks)
    branch_deleted = True
    if branch is not None:
        branch_result = run_captured(
            ["git", "branch", "-D", branch],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        branch_deleted = branch_result.ok

    return worktree_removed and branch_deleted


def inspect_worktree_state(
    worktree_path: Path,
    base_ref: str = "",
    injected_paths: tuple[str, ...] = (),
) -> WorktreeInspection:
    """Inspect a worker worktree after the process dies.

    Returns a ``WorktreeInspection`` describing whether the worker has:
    - completed work (clean working tree and commits ahead of the dispatch base)
    - partial work (uncommitted changes)
    - no commits (clean and at the base)
    - unknown (git probing failed)

    This is the single enforcement point for worktree inspection; it is used by
    both the workflow dead-session lane and reconcile.py drift detection.
    """
    if not worktree_path.is_dir():
        return WorktreeInspection(
            WorktreeState.UNKNOWN,
            error=f"worktree path does not exist: {worktree_path}",
        )

    try:
        dirty = _worker_authored_dirty(worktree_path, injected_paths)
    except WorktreeProbeFailedError as exc:
        return WorktreeInspection(
            WorktreeState.UNKNOWN,
            error=str(exc),
        )

    if base_ref == "":
        try:
            resolved_base_ref = _resolve_default_branch_ref(worktree_path)
        except RuntimeError as exc:
            return WorktreeInspection(
                WorktreeState.UNKNOWN,
                error=f"failed to resolve base ref: {exc}",
            )
    else:
        resolved_base_ref = base_ref

    merge_base_result = run_captured(
        ["git", "merge-base", resolved_base_ref, "HEAD"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not merge_base_result.ok:
        return WorktreeInspection(
            WorktreeState.UNKNOWN,
            error=merge_base_result.error or merge_base_result.stderr or "merge-base failed",
        )

    merge_base = merge_base_result.stdout.strip()
    rev_list_result = run_captured(
        ["git", "rev-list", "--count", f"{merge_base}..HEAD"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not rev_list_result.ok:
        return WorktreeInspection(
            WorktreeState.UNKNOWN,
            error=rev_list_result.error or rev_list_result.stderr or "rev-list failed",
        )

    try:
        ahead_count = int(rev_list_result.stdout.strip())
    except ValueError:
        return WorktreeInspection(
            WorktreeState.UNKNOWN,
            error=f"invalid rev-list count: {rev_list_result.stdout!r}",
        )

    if ahead_count > 0 and not dirty:
        return WorktreeInspection(
            WorktreeState.COMPLETED,
            ahead_count=ahead_count,
            dirty=False,
            resolved_base_ref=resolved_base_ref,
        )
    if ahead_count > 0 and dirty:
        return WorktreeInspection(
            WorktreeState.PARTIAL,
            ahead_count=ahead_count,
            dirty=True,
            resolved_base_ref=resolved_base_ref,
        )
    if dirty:
        return WorktreeInspection(
            WorktreeState.PARTIAL,
            ahead_count=0,
            dirty=True,
            resolved_base_ref=resolved_base_ref,
        )
    return WorktreeInspection(
        WorktreeState.NO_COMMITS,
        ahead_count=0,
        dirty=False,
        resolved_base_ref=resolved_base_ref,
    )


def push_branch(
    repo_root: Path, branch: str, worktree_path: Path | None = None
) -> tuple[bool, str | None]:
    """Push ``branch`` to origin and verify via ``git ls-remote``.

    Returns ``(ok, error)``. Pushes can fail silently on some transports, so the
    remote branch tip is explicitly checked and compared to the local branch tip.
    """
    cwd = worktree_path if worktree_path else repo_root
    push_result = run_captured(
        ["git", "push", "origin", branch],
        cwd=cwd,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not push_result.ok:
        return False, push_result.error or push_result.stderr or "git push failed"

    ls_remote_result = run_captured(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=cwd,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not ls_remote_result.ok:
        return False, ls_remote_result.error or ls_remote_result.stderr or "git ls-remote failed"

    remote_line = ls_remote_result.stdout.strip()
    if not remote_line:
        return False, f"remote branch {branch} not found after push"

    remote_sha = remote_line.split()[0]
    local_sha_result = run_captured(
        ["git", "rev-parse", "--verify", branch],
        cwd=cwd,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not local_sha_result.ok:
        return False, local_sha_result.error or local_sha_result.stderr or "git rev-parse failed"

    if local_sha_result.stdout.strip() == remote_sha:
        return True, None
    return False, f"remote branch {branch} does not match local tip after push"


def resolve_base_branch_name(repo_root: Path, base_ref: str) -> str:
    """Convert a base ref (e.g. ``origin/main`` or ``HEAD``) into a branch name.

    ``gh pr create --base`` expects a simple branch name. Remote-tracking refs
    are stripped to their local branch name; ``HEAD`` falls back to the current
    branch or ``main``.
    """
    if base_ref.startswith("refs/remotes/origin/"):
        return base_ref[len("refs/remotes/origin/") :]
    if base_ref.startswith("refs/heads/"):
        return base_ref[len("refs/heads/") :]
    if base_ref.startswith("origin/"):
        return base_ref[len("origin/") :]
    if base_ref == "HEAD":
        current_branch = run_captured(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if current_branch.ok and current_branch.stdout.strip():
            return current_branch.stdout.strip()
    return "main"


def list_worktrees(repo_root: Path) -> list[dict]:
    """Parse ``git worktree list --porcelain`` into one dict per worktree.

    Invalid entries (missing required 'worktree' key or unknown flag keys) are
    dropped entirely - every returned dict is guaranteed to have a 'worktree' key
    with a Path value. This makes all downstream consumers safe by construction.
    """
    result = run_captured(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not result.ok:
        return []

    worktrees: list[dict] = []
    current: dict = {}
    entry_malformed = False

    for line in result.stdout.splitlines():
        if not line.strip():
            # Entry boundary: flush current entry if valid
            if current and not entry_malformed and "worktree" in current:
                worktrees.append(current)
            current = {}
            entry_malformed = False
            continue

        if " " in line:
            key, _, value = line.partition(" ")
        else:
            # Space-less line: must be a known flag key
            key, value = line, True
            if key not in KNOWN_FLAG_KEYS:
                # Unknown flag key marks this entry as malformed
                entry_malformed = True

        if key == "worktree":
            # Worktree lines must have a path (str), not True
            if isinstance(value, str):
                current[key] = Path(value)
            else:
                # Malformed worktree line (bare "worktree" with no path)
                entry_malformed = True
        elif not entry_malformed:
            # Only add other keys if entry is not already malformed
            current[key] = value

    # Flush final entry if valid
    if current and not entry_malformed and "worktree" in current:
        worktrees.append(current)

    return worktrees


def _site_packages_dir(venv_path: Path) -> Path | None:
    """Locate the ``site-packages`` directory inside a virtualenv.

    Tries Windows and POSIX layouts, then falls back to a recursive search.
    """
    candidates = [
        venv_path / "Lib" / "site-packages",
        venv_path / "lib" / "site-packages",
    ]
    lib_dir = venv_path / "lib"
    if lib_dir.is_dir():
        candidates.extend(lib_dir.glob("python*/site-packages"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    for found in venv_path.rglob("site-packages"):
        if found.is_dir():
            return found
    return None


def _top_level_package_names(repo_root: Path) -> frozenset[str]:
    """Return the top-level importable package names under ``repo_root/src``.

    Used to identify which ``.pth`` files in the shared venv belong to this
    project without hardcoding the project name.
    """
    src = repo_root / "src"
    if not src.is_dir():
        return frozenset()
    names: set[str] = set()
    for child in src.iterdir():
        if child.is_dir() and (child / "__init__.py").is_file():
            names.add(child.name)
        elif child.is_file() and child.suffix == ".py":
            names.add(child.stem)
    return frozenset(names)


def _venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _resolve_pth_line(site_packages: Path, line: str) -> Path:
    """Resolve a single line from a ``.pth`` file.

    Returns an empty path for executable/comment/empty lines. Relative paths
    are resolved against the ``site-packages`` directory containing the file.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("import"):
        return Path()
    p = Path(stripped)
    if p.is_absolute():
        return p.resolve()
    return (site_packages / p).resolve()


def _verify_shared_venv_by_import(repo_root: Path, venv_path: Path) -> tuple[bool, str]:
    """Fallback import-based verification of the shared venv."""
    python = _venv_python(venv_path)
    if not python.exists():
        return False, f"venv python not found at {python} (hint: uv sync --all-extras)"
    main_src = (repo_root / "src").resolve()
    script = "import charlie_work, pathlib; print(pathlib.Path(charlie_work.__file__).resolve())"
    result = run_captured([str(python), "-c", script], cwd=repo_root, timeout_seconds=60)
    if not result.ok:
        return False, (
            f"shared venv cannot import charlie_work: {result.error or result.stderr} "
            "(hint: uv sync --all-extras --reinstall-package charlie-work)"
        )
    try:
        imported = Path(result.stdout.strip()).resolve()
    except OSError:
        return False, (
            f"shared venv charlie_work __file__ is not a valid path: {result.stdout!r} "
            "(hint: uv sync --all-extras --reinstall-package charlie-work)"
        )
    if not imported.is_relative_to(main_src):
        return False, (
            f"shared venv imports charlie_work from {imported}, not main checkout {main_src} "
            "(hint: uv sync --all-extras --reinstall-package charlie-work)"
        )
    return True, "shared venv imports charlie_work from main checkout src"


def verify_shared_venv(repo_root: Path, venv_path: Path) -> tuple[bool, str]:
    """Verify the shared venv's editable ``.pth`` points to the main checkout src.

    Searches ``site-packages`` for ``.pth`` files whose names contain a top-level
    package name from ``repo_root/src``. For each matching ``.pth``, every path
    line must resolve to ``repo_root/src``; a path pointing anywhere else is the
    poisoned-editable-pth case and surfaces the ``--reinstall-package`` recovery
    hint.
    """
    site_packages = _site_packages_dir(venv_path)
    if not site_packages:
        return False, "could not locate site-packages in shared venv"
    main_src = (repo_root / "src").resolve()
    package_names = _top_level_package_names(repo_root)
    project_pth_files: list[Path] = []
    for pth in site_packages.glob("*.pth"):
        if any(name in pth.name for name in package_names):
            project_pth_files.append(pth)
    if not project_pth_files:
        return _verify_shared_venv_by_import(repo_root, venv_path)
    for pth in project_pth_files:
        content = pth.read_text(encoding="utf-8", errors="replace")
        for raw_line in content.splitlines():
            target = _resolve_pth_line(site_packages, raw_line)
            if target == Path() or target == main_src:
                continue
            return False, (
                f"editable .pth {pth.name} points outside main checkout: {target} "
                "(hint: uv sync --all-extras --reinstall-package charlie-work)"
            )
    return True, "shared venv editable .pth points to main checkout src"


def _find_linked_pr_number(
    issue_number: int, issue_state: dict[str, Any], state_prs: dict[str, Any]
) -> int | None:
    """Resolve the PR number linked to ``issue_number`` from state.json."""
    if issue_state.get("pr_number"):
        return int(issue_state["pr_number"])
    for pr_key, pr_entry in state_prs.items():
        if pr_entry.get("issue_number") == issue_number:
            return int(pr_key)
    return None


@dataclass(frozen=True)
class WorktreeCleanResult:
    """Result of a ``clean_worktrees`` run.

    ``data`` carries ``planned``/``removed``/``skipped``/``failed`` (lists of
    per-worktree dicts) plus ``venv_ok``/``venv_message``. Kept as a dict
    rather than further nested dataclasses since callers (``CommandResult``)
    consume it as a JSON-able blob for CLI output.
    """

    ok: bool
    message: str
    data: dict[str, Any]


def clean_worktrees(
    repo_root: Path,
    worktrees_dir: Path,
    state: dict[str, Any],
    config: OrchestratorConfig,
    gh: GitHub,
    *,
    dry_run: bool = False,
) -> WorktreeCleanResult:
    """Junction-safe cleanup of worker worktrees for merged PRs.

    Enumerates worktrees under ``worktrees_dir`` whose linked issue/PR resolves
    to a PR number in ``state.json``, then applies a merged-PR-aware
    eligibility check per worktree:

      1. The PR must be confirmed ``MERGED`` by a *live* ``gh pr view`` call.
         ``state.json`` claiming "merged" is corroboration only -- it is never
         sufficient on its own (state.json reliability history: #285/#309/
         #310), and an unavailable/erroring ``gh`` call fails CLOSED (skip)
         rather than falling back to trusting state.json.
      2. No live worker (see ``_probe_recovery_liveness``).
      3. The worktree's working tree is clean (no uncommitted modifications).
      4. The worktree's local HEAD equals the merged PR's ``headRefOid``.

    Check 4 is deliberately a standalone comparison rather than a reuse of
    ``_worktree_refuse_to_reset_reason``'s remote-branch-ahead check. This
    repo's production ``AutoMergeConfig`` defaults to ``strategy="squash"``
    with ``delete_branch=True``, so after a real merge the remote branch is
    gone and the squash commit landed on the base branch is NOT an ancestor of
    the worker branch's commits. Reusing the remote-branch-ahead helper here
    would count every legitimately merged worktree's own work as "local
    commit(s) not on remote branch" and skip it forever (issue #286 rework,
    PR #340 review finding 1) -- local-ahead-of-a-deleted-remote-branch is the
    EXPECTED shape for a squash-merged worktree, not a sign of unpushed work.
    A worktree whose local HEAD does NOT match the merged PR's head (a stray
    post-merge commit) is skipped with a distinct reason instead of being
    silently folded into the same "unsafe" bucket.

    Eligible worktrees are removed with ``remove_worktree`` (junction-safe)
    and the local branch is deleted. After removals, the shared venv is
    checked for a poisoned editable ``.pth``.
    """
    state_issues = state.get("issues", {})
    state_prs = state.get("prs", {})
    planned: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for wt in list_worktrees(repo_root):
        wt_path = wt.get("worktree")
        if not isinstance(wt_path, Path) or not wt_path.is_relative_to(worktrees_dir):
            continue
        raw_branch = str(wt.get("branch", ""))
        branch = raw_branch.removeprefix("refs/heads/")
        if not branch.startswith(config.dispatch.branch_prefix):
            continue
        issue_number = linked_issue_number(
            {"headRefName": branch},
            is_cross_repository=False,
            branch_prefix=config.dispatch.branch_prefix,
        )
        if issue_number is None:
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "reason": "could not parse issue number from branch",
                }
            )
            continue
        issue_state = state_issues.get(str(issue_number), {})
        pr_number = _find_linked_pr_number(issue_number, issue_state, state_prs)
        if not pr_number:
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "reason": "no linked PR in state.json",
                }
            )
            continue
        pr_state = state_prs.get(str(pr_number), {})
        state_merged = bool(pr_state.get("status") == "merged" or pr_state.get("merged") is True)

        # `allow_failure=True` means this never raises (see GitHub.run): errors
        # come back as GitHubRunResult(ok=False, error=...), not exceptions.
        gh_result = gh.run(
            ["pr", "view", str(pr_number), "--json", PR_VIEW_MERGED_FIELDS],
            json_output=True,
            allow_failure=True,
        )
        gh_ok = isinstance(gh_result, GitHubRunResult) and gh_result.ok
        gh_merged = False
        merged_head_sha: str | None = None
        if gh_ok and isinstance(gh_result.value, dict):
            gh_merged = gh_result.value.get("state") == "MERGED"
            merged_head_sha = gh_result.value.get("headRefOid")

        if not gh_merged:
            # Fail-closed: only a live `gh pr view` MERGED confirmation may
            # authorize destructive removal. state.json is corroboration at
            # best -- never sufficient on its own (state.json reliability
            # history: #285/#309/#310). An unavailable/erroring `gh` call
            # falls into this branch too and is DISTINGUISHED from a
            # confirmed-not-merged PR rather than falling back to trusting
            # state.json.
            if not gh_ok:
                gh_error = gh_result.error if isinstance(gh_result, GitHubRunResult) else "unknown"
                reason = f"gh pr view unavailable; cannot confirm merge status: {gh_error}"
            elif state_merged:
                reason = "state.json says merged but gh pr view did not confirm MERGED"
            else:
                reason = "PR not merged"
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": reason,
                }
            )
            continue
        try:
            _probe_recovery_liveness(issue_state, wt_path, config, issue_number)
        except LiveWorkerRedispatchError as exc:
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": f"live worker detected: {exc.probe_result}",
                }
            )
            continue
        try:
            dirty_reason = _worktree_dirty_reason(wt_path, config.dispatch.injected_paths)
        except WorktreeProbeFailedError as exc:
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": f"worktree status probe failed: {exc}",
                }
            )
            continue
        if dirty_reason:
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": dirty_reason,
                }
            )
            continue
        if not merged_head_sha:
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": "gh pr view did not return headRefOid for the merged PR",
                }
            )
            continue
        head_result = run_captured(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=wt_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not head_result.ok or not head_result.stdout.strip():
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": "could not resolve worktree HEAD",
                }
            )
            continue
        local_head_sha = head_result.stdout.strip()
        if local_head_sha != merged_head_sha:
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": (
                        f"worktree HEAD ({local_head_sha[:8]}) does not match merged PR head "
                        f"({merged_head_sha[:8]}); stray post-merge commit(s)"
                    ),
                }
            )
            continue
        if dry_run:
            planned.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                }
            )
            continue
        if not remove_worktree(repo_root, wt_path, force=True, branch=branch):
            failed.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": "remove_worktree failed",
                }
            )
            continue
        removed.append(
            {
                "worktree": str(wt_path),
                "branch": branch,
                "issue_number": issue_number,
                "pr_number": pr_number,
            }
        )

    venv_source = config.devin.venv_source or config.claude_code.venv_source
    venv_ok = True
    venv_message = "no shared venv configured; pth verification skipped"
    if venv_source:
        venv_path = repo_root / venv_source
        if venv_path.is_dir():
            venv_ok, venv_message = verify_shared_venv(repo_root, venv_path)
        else:
            venv_ok = False
            venv_message = f"shared venv not found: {venv_path}"

    data = {
        "planned": planned,
        "removed": removed,
        "skipped": skipped,
        "failed": failed,
        "venv_ok": venv_ok,
        "venv_message": venv_message,
    }
    ok = not failed and venv_ok
    if dry_run:
        message = f"worktree-clean (dry-run): {len(planned)} eligible, {len(skipped)} skipped"
    else:
        message = (
            f"worktree-clean: {len(removed)} removed, {len(skipped)} skipped, {len(failed)} failed"
        )
    if not venv_ok:
        message = f"{message}; {venv_message}"
    return WorktreeCleanResult(ok=ok, message=message, data=data)
