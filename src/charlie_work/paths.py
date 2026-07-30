from __future__ import annotations

import dataclasses
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import layout
from .config import NotifyConfig, OrchestratorConfig
from .subprocess_runner import hidden_console_kwargs


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
    #: Plain ``root/worktrees`` -- ``root`` already honours a
    #: ``runtime.state_dir`` override (see :func:`runtime_paths`), but this
    #: member deliberately does NOT honour ``claude_code.worktrees_dir``.
    #: Required by the state-dir-centralization plan's step A2 as part of
    #: ``RuntimePaths``'s general shape; no production call site reads it.
    #: Both dispatch and ``worktree-clean`` resolve their *actual* worktrees
    #: root via ``resolved_layout(config, repo_root).worktrees`` instead (see
    #: that function's docstring) -- using this member for a doctor
    #: dispatch-vs-clean comparison would be comparing the same computation
    #: to itself once both call sites are unified, and would false-positive
    #: on a legitimately configured ``claude_code.worktrees_dir`` override.
    worktrees: Path
    cross_family: Path

    def ensure(self) -> None:
        for path in (self.root, self.issues, self.prs, self.dispatches, self.logs):
            path.mkdir(parents=True, exist_ok=True)


def find_repo_root(cwd: Path | None = None, *, explicit: bool = False) -> Path:
    """Return the git work-tree root for *cwd* (defaults to ``Path.cwd()``).

    When *explicit* is True the caller supplied ``cwd`` directly from a
    user-facing ``--repo`` flag.  In that case the path must exist and must
    be inside a git work tree; a clear :class:`RepoNotFoundError` is raised
    otherwise so the operator sees the mistake instead of a silent phantom repo.
    """
    start = (cwd or Path.cwd()).resolve()
    if explicit:
        if not start.exists():
            raise RepoNotFoundError(f"--repo path does not exist: {start}")
        if not start.is_dir():
            raise RepoNotFoundError(f"--repo path is not a directory: {start}")
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
    return RuntimePaths(
        root=root,
        issues=root / layout.ISSUES_DIRNAME,
        prs=root / layout.PRS_DIRNAME,
        dispatches=root / layout.DISPATCHES_DIRNAME,
        logs=root / layout.LOGS_DIRNAME,
        state_file=layout.state_file_path(root),
        worktrees=layout.worktrees_dir(root),
        cross_family=layout.cross_family_dir(root),
    )


@dataclass(frozen=True)
class ResolvedLayout:
    """Every sentinel-resolved state-child path/config for one (config, repo_root).

    Several config fields use an empty string to mean "derive this from
    ``runtime.state_dir``" instead of re-spelling the historical default in
    the dataclass itself (``devin.sessions_dir``, ``devin.session_manifest``,
    ``devin.session_results``, ``review_dispatch.reviews_dir``,
    ``notify.file_path``), and ``claude_code.worktrees_dir`` uses ``None`` for
    the same purpose. This is the single object that resolves all of them --
    call sites should use its fields rather than re-implementing the "empty
    means derive" check inline (that duplication is exactly what caused the
    74-uncollected-worktrees production incident documented in layout.py).

    Unlike :class:`RuntimePaths`, ``worktrees`` here DOES honour
    ``claude_code.worktrees_dir`` -- this is the value both dispatch and
    ``charlie worktree-clean`` must use so the create and sweep sides can
    never diverge, regardless of which of the two independent overrides
    (``runtime.state_dir``, ``claude_code.worktrees_dir``) is in play.

    ``notify`` is a full, ready-to-pass ``NotifyConfig`` (not just a bare
    ``Path``) since :func:`charlie_work.notify.emit_digest` takes the config
    object directly; the source ``NotifyConfig`` is never mutated, only
    copied via ``dataclasses.replace`` with ``file_path`` resolved.
    """

    sessions_dir: Path
    session_manifest: Path
    session_results: Path
    reviews_dir: Path
    worktrees: Path
    cross_family: Path
    notify: NotifyConfig


def resolved_layout(config: OrchestratorConfig, repo_root: Path) -> ResolvedLayout:
    """Resolve every sentinel-style state-child config value for *config*.

    See :class:`ResolvedLayout`. Frozen dataclasses are never mutated here --
    this builds a fresh, derived view each call from ``config`` and
    *repo_root* alone.
    """
    root = runtime_paths(repo_root, config.runtime.state_dir).root
    sessions_dir = layout.resolve_state_child(
        config.devin.sessions_dir,
        repo_root=repo_root,
        default=layout.sessions_dir_default(root),
    )
    session_manifest = layout.resolve_state_child(
        config.devin.session_manifest,
        repo_root=repo_root,
        default=layout.session_manifest_default(root),
    )
    session_results = layout.resolve_state_child(
        config.devin.session_results,
        repo_root=repo_root,
        default=layout.session_results_default(root),
    )
    reviews_dir = layout.resolve_state_child(
        config.review_dispatch.reviews_dir,
        repo_root=repo_root,
        default=layout.reviews_dir_default(root),
    )
    worktrees = layout.resolve_state_child(
        config.claude_code.worktrees_dir or "",
        repo_root=repo_root,
        default=layout.worktrees_dir(root),
    )
    notify_file_path = layout.resolve_state_child(
        config.notify.file_path,
        repo_root=repo_root,
        default=layout.notify_digest_default(root),
    )
    return ResolvedLayout(
        sessions_dir=sessions_dir,
        session_manifest=session_manifest,
        session_results=session_results,
        reviews_dir=reviews_dir,
        worktrees=worktrees,
        cross_family=layout.cross_family_dir(root),
        notify=dataclasses.replace(config.notify, file_path=str(notify_file_path)),
    )
