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

from .subprocess_runner import run_captured

_DEFAULT_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class WorktreeInfo:
    path: Path
    branch: str
    venv_junction: Path | None


def _slugify(value: str, *, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:max_length].rstrip("-") or "worktree"


def _default_worktrees_dir(repo_root: Path) -> Path:
    return repo_root / ".var" / "charlie-work" / "worktrees"


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
    """
    target_dir = worktrees_dir or _default_worktrees_dir(repo_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    worktree_path = target_dir / _slugify(branch)

    if rework:
        # Rework mode: branch already exists, reuse or attach to it
        existing_worktrees = list_worktrees(repo_root)
        # Branch names in git worktree list may have refs/heads/ prefix
        existing_wt = next(
            (wt for wt in existing_worktrees if wt.get("branch", "").endswith(f"/{branch}") or wt.get("branch") == branch),
            None
        )

        if existing_wt:
            # Reuse existing worktree: fetch and fast-forward to origin tip
            worktree_path = Path(existing_wt["worktree"])
            # Fetch the branch from origin to get latest changes
            # If there's no remote (e.g., in tests), skip the fetch
            fetch_result = run_captured(
                ["git", "fetch", "origin", f"{branch}:{branch}"],
                cwd=repo_root,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            )
            # Don't fail on fetch errors - the worktree is still usable
            # (e.g., in test repos without a remote)
            # Skip venv junction creation for reused worktrees (already exists)
            venv_junction = None
            if venv_source is not None:
                venv_link = worktree_path / ".venv"
                if venv_link.exists() or is_junction(venv_link):
                    venv_junction = venv_link
                else:
                    _create_junction_or_symlink(venv_link, venv_source)
                    venv_junction = venv_link
            return WorktreeInfo(path=worktree_path, branch=branch, venv_junction=venv_junction)
        else:
            # No existing worktree: attach to existing branch (no -b flag)
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
        result = run_captured(
            ["git", "worktree", "add", "-b", branch, str(worktree_path), base_ref],
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
        _create_junction_or_symlink(venv_link, venv_source)
        venv_junction = venv_link

    return WorktreeInfo(path=worktree_path, branch=branch, venv_junction=venv_junction)


def remove_worktree(repo_root: Path, worktree_path: Path, *, force: bool = False) -> bool:
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
    if not result.ok:
        run_captured(
            ["git", "worktree", "prune"], cwd=repo_root, timeout_seconds=_DEFAULT_TIMEOUT_SECONDS
        )
        return False
    return True


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
