from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .subprocess_runner import hidden_console_kwargs

logger = logging.getLogger(__name__)


class RepoNotFoundError(ValueError):
    """Raised when ``--repo`` points at a path that is not a git work tree."""


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    issues: Path
    prs: Path
    dispatches: Path
    logs: Path
    state_file: Path

    def ensure(self) -> None:
        for path in (self.root, self.issues, self.prs, self.dispatches, self.logs):
            path.mkdir(parents=True, exist_ok=True)


def _main_worktree_root(start: Path) -> Path | None:
    """Resolve the shared (main) worktree root for *start* via git.

    Uses ``git rev-parse --git-common-dir`` to locate the shared ``.git``
    directory, then returns its parent — the main worktree root — which is
    the same regardless of which linked worktree *start* is inside. Returns
    None if git is unavailable, *start* is not inside a work tree, or the
    resolved common dir does not yield a usable parent (e.g. a bare repo,
    whose common dir is the repo itself rather than ``<root>/.git``).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=start,
            text=True,
            capture_output=True,
            check=True,
            **hidden_console_kwargs(),
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    common_dir = Path(raw)
    if not common_dir.is_absolute():
        common_dir = (start / common_dir).resolve()
    else:
        common_dir = common_dir.resolve()
    # The common dir is ``<main root>/.git``; its parent is the main worktree
    # root. A bare repo (common dir is the repo itself, no ``.git`` basename)
    # has no worktree root — bail out and let the caller fall back.
    if common_dir.name != ".git":
        return None
    parent = common_dir.parent
    if not parent.exists() or not parent.is_dir():
        return None
    return parent


def find_repo_root(cwd: Path | None = None, *, explicit: bool = False) -> Path:
    """Return the git work-tree root for *cwd* (defaults to ``Path.cwd()``).

    When *explicit* is True the caller supplied ``cwd`` directly from a
    user-facing ``--repo`` flag.  In that case the path must exist and must
    be inside a git work tree; a clear :class:`RepoNotFoundError` is raised
    otherwise so the operator sees the mistake instead of a silent phantom repo.

    When *explicit* is False (the default ``--repo``-omitted case) the
    *shared* repo root is resolved via ``git rev-parse --git-common-dir``
    rather than ``--show-toplevel``.  For a linked git worktree
    ``--show-toplevel`` returns the worktree's own root, but the
    orchestrator's runtime state lives under the *main* worktree's
    ``.var/charlie-work/``.  Resolving the shared root prevents a cwd
    inside a linked worktree from silently targeting a phantom,
    never-populated state directory (issue #648).
    """
    start = (cwd or Path.cwd()).resolve()
    if explicit:
        if not start.exists():
            raise RepoNotFoundError(f"--repo path does not exist: {start}")
        if not start.is_dir():
            raise RepoNotFoundError(f"--repo path is not a directory: {start}")
    else:
        # Prefer the shared/main worktree root so that a cwd inside a linked
        # worktree does not resolve to that worktree's own toplevel (which
        # would point at a phantom state dir). Fall back to --show-toplevel
        # if common-dir resolution is unavailable.
        main_root = _main_worktree_root(start)
        if main_root is not None:
            return main_root
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            text=True,
            capture_output=True,
            check=True,
            **hidden_console_kwargs(),
        )
        return Path(result.stdout.strip()).resolve()
    except (OSError, subprocess.CalledProcessError):
        for candidate in (start, *start.parents):
            if (candidate / ".git").exists():
                return candidate
    if explicit:
        raise RepoNotFoundError(f"--repo path is not inside a git work tree: {start}")
    return start


def runtime_paths(repo_root: Path, state_dir: str) -> RuntimePaths:
    root = Path(state_dir)
    if not root.is_absolute():
        root = repo_root / root
    root = root.resolve()
    _warn_if_phantom_state_dir(root)
    return RuntimePaths(
        root=root,
        issues=root / "issues",
        prs=root / "prs",
        dispatches=root / "dispatches",
        logs=root / "logs",
        state_file=root / "state.json",
    )


def _warn_if_phantom_state_dir(root: Path) -> None:
    """Warn when *root* exists but has no ``state.json`` (issue #648).

    A state directory that already exists without a ``state.json`` is a
    strong signal that an earlier invocation resolved the wrong repo root
    (e.g. a linked worktree's own toplevel) and left sibling artifacts
    (``events.db``, lock files) without ever initializing real state. The
    warning is non-blocking; a genuine first run has not created *root* yet
    at the point ``runtime_paths`` is called, so this does not fire on
    legitimate initialization.
    """
    if not root.exists() or (root / "state.json").exists():
        return
    logger.warning(
        "Runtime state directory %s exists but has no state.json — this may "
        "be a phantom directory left by an earlier invocation that resolved a "
        "linked worktree's own toplevel instead of the shared repo root. "
        "State will be initialized fresh here; if this is unexpected, pass "
        "--repo pointing at the main checkout.",
        root,
    )
