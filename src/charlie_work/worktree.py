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
from pathlib import Path
from typing import Any

from .subprocess_runner import run_captured

_DEFAULT_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class WorktreeInfo:
    path: Path
    branch: str
    venv_junction: Path | None
    reclaimed: str | None = None  # "fetch-fallback" | "pruned" | "salvaged" | None


def _slugify(value: str, *, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:max_length].rstrip("-") or "worktree"


def _default_worktrees_dir(repo_root: Path) -> Path:
    return repo_root / ".var" / "charlie-work" / "worktrees"


def _resolve_default_branch_ref(repo_root: Path) -> str:
    """Resolve the repository's default branch as a remote-tracking ref.

    Returns a string like "origin/main" or "origin/master". If the repo has no
    origin remote or the default branch cannot be determined, returns "HEAD"
    (fallback to local behavior).

    Uses git symbolic-ref refs/remotes/origin/HEAD which is the standard way
    to get the default branch without needing the GitHub CLI.
    """
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
    # Fallback to HEAD if we can't determine the default branch
    return "HEAD"


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
        # Check if HEAD has any commits beyond the initial commit
        merge_base_result = run_captured(
            ["git", "merge-base", "HEAD", "main"],
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


def create_worktree(
    repo_root: Path,
    branch: str,
    *,
    base_ref: str = "HEAD",
    worktrees_dir: Path | None = None,
    venv_source: Path | None = None,
    rework: bool = False,
    recovery: dict[str, Any] | None = None,
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

    # Recovery mode: dead-worker re-dispatch with leftover worktree/branch
    reclaimed: str | None = None
    if recovery is not None:
        # Validate that the recovery record matches the requested branch
        recovery_branch = recovery.get("branch_name")
        if recovery_branch != branch:
            raise RuntimeError(
                f"Recovery record branch_name {recovery_branch!r} does not match "
                f"requested branch {branch!r}"
            )

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
                    # It's our worktree - remove it
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
                # Check for dirty working tree
                dirty_result = run_captured(
                    ["git", "status", "--porcelain"],
                    cwd=wt_path,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                # If the probe fails (index lock, corruption, permissions), treat as dirty to be safe
                has_dirty = not dirty_result.ok or bool(dirty_result.stdout.strip())

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
                    if not remove_worktree(repo_root, wt_path, force=True):
                        raise RuntimeError(
                            f"Failed to remove leftover worktree {wt_path} for recovery"
                        )
                    # Delete the branch and check the result
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
            # Skip venv junction creation for reused worktrees (already exists)
            venv_junction = None
            if venv_source is not None:
                venv_link = worktree_path / ".venv"
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
            return WorktreeInfo(
                path=worktree_path, branch=branch, venv_junction=venv_junction, reclaimed=reclaimed
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
                # Directory exists: check if it's clean at the recorded base
                dirty_result = run_captured(
                    ["git", "status", "--porcelain"],
                    cwd=wt_path,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                # If the probe fails (index lock, corruption, permissions), treat as dirty to be safe
                has_dirty = not dirty_result.ok or bool(dirty_result.stdout.strip())
                if has_dirty:
                    # Dirty tree: salvage it
                    try:
                        salvage_ref = _salvage_worktree(repo_root, wt_path, branch)
                        if salvage_ref:
                            reclaimed = "salvaged"
                    except RuntimeError as salvage_error:
                        # Salvage push failed: surface the error and leave worktree intact
                        raise RuntimeError(
                            f"Failed to salvage stale worktree {wt_path} for fresh dispatch: {salvage_error}"
                        ) from salvage_error
                    # Remove the worktree (junction-safe)
                    if not remove_worktree(repo_root, wt_path, force=True):
                        raise RuntimeError(
                            f"Failed to remove stale worktree {wt_path} for fresh dispatch"
                        )
                else:
                    # Clean at base: junction-safe remove and recreate
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

    return WorktreeInfo(
        path=worktree_path, branch=branch, venv_junction=venv_junction, reclaimed=reclaimed
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
                # Windows junctions (reparse points) are removed with os.rmdir,
                # which unlinks only the reparse point — never follows into the
                # target. On POSIX, symlinks must be removed with os.unlink;
                # os.rmdir raises NotADirectoryError/OSError on a symlink.
                if os.name == "nt":
                    os.rmdir(venv_path)
                else:
                    os.unlink(venv_path)
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


def list_worktrees(repo_root: Path) -> list[dict]:
    """Parse ``git worktree list --porcelain`` into one dict per worktree."""
    result = run_captured(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not result.ok:
        return []

    worktrees: list[dict] = []
    current: dict = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            if current:
                worktrees.append(current)
                current = {}
            continue
        if " " in line:
            key, _, value = line.partition(" ")
        else:
            key, value = line, True
        if key == "worktree":
            current[key] = Path(value)
        else:
            current[key] = value
    if current:
        worktrees.append(current)
    return worktrees
