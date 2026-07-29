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


def _resolve_git_path(start: Path, rev_parse_arg: str) -> Path | None:
    """Run ``git rev-parse <rev_parse_arg>`` in *start* and return the resolved path."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", rev_parse_arg],
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
    path = Path(raw)
    if not path.is_absolute():
        path = (start / path).resolve()
    else:
        path = path.resolve()
    return path


def _main_worktree_root(start: Path) -> Path | None:
    """Resolve the shared (main) worktree root for *start* via git.

    When *start* is inside a **linked** git worktree, ``git rev-parse
    --show-toplevel`` returns the linked worktree's own root rather than the
    main repository root.  This function returns the main worktree root
    instead, so that runtime state resolves to the shared
    ``.var/charlie-work/`` directory regardless of which linked worktree
    *start* is inside.

    Returns None (letting the caller fall back to ``--show-toplevel``) when:

    - *start* is in the **main** worktree — ``--show-toplevel`` is already
      correct there, including for ``--separate-git-dir`` repos whose common
      dir lives elsewhere (issue #648 review: returning ``common_dir.parent``
      for those would place state under the external git dir's container, not
      the working tree).
    - git is unavailable or *start* is not inside a work tree.
    - The common dir does not yield a usable parent (e.g. a bare repo, whose
      common dir is the repo itself rather than ``<root>/.git``).

    Detection uses ``git rev-parse --git-dir`` vs ``--git-common-dir``: they
    are equal inside the main worktree and differ inside a linked worktree
    (whose per-worktree git dir lives under ``<common>/worktrees/<name>/``).
    """
    git_dir = _resolve_git_path(start, "--git-dir")
    common_dir = _resolve_git_path(start, "--git-common-dir")
    if git_dir is None or common_dir is None:
        return None
    # Main worktree (normal OR --separate-git-dir): --show-toplevel is
    # already correct.  Returning None here avoids the regression where
    # common_dir.parent is the external git dir's container for
    # --separate-git-dir repos rather than the working tree root.
    if git_dir == common_dir:
        return None
    # Linked worktree: the common dir is ``<main root>/.git`` for a normal
    # repo; its parent is the main worktree root.  A bare repo (common dir is
    # the repo itself, no ``.git`` basename) has no worktree root — bail out.
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

    Regardless of *explicit*, the *shared* (main) worktree root is resolved
    via ``git rev-parse --git-dir`` / ``--git-common-dir`` before falling back
    to ``--show-toplevel``.  For a linked git worktree ``--show-toplevel``
    returns the worktree's own root, but the orchestrator's runtime state
    lives under the *main* worktree's ``.var/charlie-work/``.  Resolving the
    shared root prevents a cwd — or an explicit ``--repo <linked-worktree>``
    — from silently targeting a phantom, never-populated state directory
    (issue #648).  In the main worktree (where ``--git-dir`` equals
    ``--git-common-dir``) the shared-root resolution returns None and
    ``--show-toplevel`` is used, which is correct for normal and
    ``--separate-git-dir`` repos alike.
    """
    start = (cwd or Path.cwd()).resolve()
    if explicit:
        if not start.exists():
            raise RepoNotFoundError(f"--repo path does not exist: {start}")
        if not start.is_dir():
            raise RepoNotFoundError(f"--repo path is not a directory: {start}")
    # Prefer the shared/main worktree root so that a cwd (or an explicit
    # --repo) inside a linked worktree does not resolve to that worktree's
    # own toplevel (which would point at a phantom state dir). Returns None
    # in the main worktree, where --show-toplevel is already correct.
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
    """Warn when *root* exists with phantom sibling artifacts but no ``state.json``.

    A state directory that already exists without a ``state.json`` *and* with
    at least one of the sibling artifacts (``events.db``, ``state.json.lock``)
    that the orchestrator creates alongside state is a strong signal that an
    earlier invocation resolved the wrong repo root (e.g. a linked worktree's
    own toplevel) and left those artifacts without ever initializing real
    state.  The warning is non-blocking.

    Requiring a sibling artifact avoids false positives from callers that
    invoke ``runtime_paths`` with an absolute ``state_dir`` pointing at a
    pre-existing directory used for an unrelated purpose.  The
    corruption-quarantine path (``state.json`` moved aside but ``events.db``
    remains) still fires the warning, which is appropriate — the operator
    should know state is being re-initialized.
    """
    if not root.exists() or (root / "state.json").exists():
        return
    # Require at least one sibling artifact so the warning is specific to the
    # phantom-directory signature rather than any pre-existing directory.
    sibling_artifacts = ("events.db", "state.json.lock")
    if not any((root / name).exists() for name in sibling_artifacts):
        return
    logger.warning(
        "Runtime state directory %s exists with sibling artifacts but no "
        "state.json — state will be initialized fresh here. If this is "
        "unexpected, verify the resolved repo root is correct.",
        root,
    )
