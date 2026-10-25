"""Git reads and effects for main-sync (docs/specs/main-sync.md).

Every function returns a value and never raises: failures come back as ``None``, a
skip-reason string, ``False`` or a ``RunResult`` -- the project's external-process
invariant.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from .git_retry import run_git_with_retry
from .main_sync_classify import CheckoutState, IncomingCommit
from .subprocess_runner import RunResult, run_captured

GIT_TIMEOUT_SECONDS = 30
NETWORK_TIMEOUT_SECONDS = 60
_OPERATIONS = (
    ("MERGE_HEAD", "merge_in_progress"),
    ("rebase-merge", "rebase_in_progress"),
    ("rebase-apply", "rebase_in_progress"),
    ("CHERRY_PICK_HEAD", "cherry_pick_in_progress"),
    ("REVERT_HEAD", "revert_in_progress"),
)


def _git(repo_root: Path, *args: str, timeout: int = GIT_TIMEOUT_SECONDS) -> RunResult:
    return run_captured(["git", *args], cwd=repo_root, timeout_seconds=timeout)


def _line(repo_root: Path, *args: str) -> str | None:
    result = _git(repo_root, *args)
    return (result.stdout.strip() or None) if result.ok else None


def _names(repo_root: Path, *args: str) -> tuple[str, ...] | None:
    """A ``-z`` path list as a tuple; None when git failed."""
    result = _git(repo_root, *args)
    if not result.ok:
        return None
    return tuple(name for name in result.stdout.split("\0") if name)


def _git_dirs(repo_root: Path) -> tuple[Path, Path] | None:
    out = _line(repo_root, "rev-parse", "--path-format=absolute", "--git-dir", "--git-common-dir")
    lines = out.splitlines() if out else []
    return (Path(lines[0]), Path(lines[1])) if len(lines) == 2 else None


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(str(a.resolve())) == os.path.normcase(str(b.resolve()))


def has_origin(repo_root: Path) -> bool:
    return _git(repo_root, "remote", "get-url", "origin").ok


def is_primary_checkout(repo_root: Path) -> bool:
    """True for a repo's main working tree; False for a linked worktree (its git dir is
    ``<common>/worktrees/<name>``) or when git cannot answer."""
    dirs = _git_dirs(repo_root)
    return dirs is not None and _same_path(*dirs)


def git_common_dir(repo_root: Path) -> Path | None:
    dirs = _git_dirs(repo_root)
    return dirs[1] if dirs else None


def resolve_default_branch(repo_root: Path, *, dry_run: bool) -> str | None:
    """``main`` from ``refs/remotes/origin/HEAD``. Outside dry run a missing symref is
    healed once with ``git remote set-head origin --auto`` (a network call)."""
    ref = _line(repo_root, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD")
    if ref is None and not dry_run:
        healed = _git(
            repo_root, "remote", "set-head", "origin", "--auto", timeout=NETWORK_TIMEOUT_SECONDS
        )
        if healed.ok:
            ref = _line(
                repo_root, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"
            )
    if ref is None or not ref.startswith("origin/"):
        return None
    return ref.removeprefix("origin/")


def fetch_default(repo_root: Path, branch: str) -> RunResult:
    """``git fetch origin <branch>``: moves only ``refs/remotes/origin/<branch>``."""
    return run_git_with_retry(
        ["git", "fetch", "--quiet", "origin", branch],
        cwd=repo_root,
        timeout_seconds=NETWORK_TIMEOUT_SECONDS,
    )


def read_checkout_state(repo_root: Path, branch: str) -> CheckoutState | str:
    """Snapshot for ``classify``, or a skip reason when git cannot provide one."""
    local = _line(repo_root, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    if local is None:
        return "no_local_branch"
    origin = _line(repo_root, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}")
    if origin is None:
        return "no_remote_branch"
    dirs = _git_dirs(repo_root)
    counts = _line(repo_root, "rev-list", "--left-right", "--count", f"{local}...{origin}")
    try:
        ahead, behind = (int(n) for n in (counts or "").split())
    except ValueError:
        return "git_error"
    if dirs is None:
        return "git_error"
    git_dir = dirs[0]
    state = CheckoutState(
        default_branch=branch,
        current_branch=_line(repo_root, "symbolic-ref", "--quiet", "--short", "HEAD"),
        local_sha=local,
        origin_sha=origin,
        merge_base=local if local == origin else _line(repo_root, "merge-base", local, origin),
        ahead=ahead,
        behind=behind,
        operation_in_progress=next(
            (reason for name, reason in _OPERATIONS if (git_dir / name).exists()), None
        ),
        index_locked=(git_dir / "index.lock").exists(),
    )
    if state.current_branch != branch:
        elsewhere = _checked_out_elsewhere(repo_root, branch)
        return replace(state, default_checked_out_elsewhere=elsewhere)
    dirty = _names(repo_root, "diff", "--name-only", "-z", "--no-renames", "HEAD")
    untracked = _names(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    incoming = _names(repo_root, "diff", "--name-only", "-z", "--no-renames", local, origin)
    if dirty is None or untracked is None or incoming is None:
        return "git_error"
    return replace(state, dirty_tracked=dirty, untracked=untracked, incoming_paths=incoming)


def _checked_out_elsewhere(repo_root: Path, branch: str) -> bool:
    """True when a linked worktree has ``branch`` checked out, or when git cannot say:
    moving a ref under someone's working tree must fail closed."""
    result = _git(repo_root, "worktree", "list", "--porcelain")
    if not result.ok:
        return True
    return f"branch refs/heads/{branch}" in result.stdout.splitlines()


def list_incoming_commits(
    repo_root: Path, start: str, end: str
) -> tuple[IncomingCommit, ...] | None:
    """First-parent commits in ``start..end``, oldest first, each with the paths it
    changes against its first parent (merge commits included)."""
    result = _git(
        repo_root,
        "-c",
        "core.quotePath=false",
        "log",
        "--first-parent",
        "-m",
        "--reverse",
        "--no-renames",
        "--name-only",
        "--format=%x01%H",
        f"{start}..{end}",
    )
    if not result.ok:
        return None
    commits = []
    for chunk in result.stdout.split("\x01"):
        lines = [line for line in chunk.splitlines() if line]
        if lines:
            commits.append(IncomingCommit(sha=lines[0], paths=tuple(lines[1:])))
    return tuple(commits)


def fast_forward(repo_root: Path, sha: str) -> RunResult:
    return _git(repo_root, "merge", "--ff-only", "--quiet", sha)


def update_ref(repo_root: Path, branch: str, new: str, old: str) -> RunResult:
    """Compare-and-swap ``refs/heads/<branch>`` from ``old`` to ``new``; no working tree."""
    ref = f"refs/heads/{branch}"
    return _git(repo_root, "update-ref", "-m", "main-sync: fast-forward", ref, new, old)


def reset_keep(repo_root: Path, sha: str) -> RunResult:
    """Move the checked-out branch back to ``sha``; fails rather than discard local edits."""
    return _git(repo_root, "reset", "--keep", "--quiet", sha)


def is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    return _git(repo_root, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def changed_paths(repo_root: Path, start: str, end: str) -> tuple[str, ...]:
    return _names(repo_root, "diff", "--name-only", "-z", "--no-renames", start, end) or ()


def count_commits(repo_root: Path, start: str, end: str) -> int:
    out = _line(repo_root, "rev-list", "--count", f"{start}..{end}")
    return int(out) if out and out.isdigit() else 0


def run_post_sync(repo_root: Path, command: tuple[str, ...], timeout_seconds: int) -> RunResult:
    return run_captured(list(command), cwd=repo_root, timeout_seconds=timeout_seconds)
