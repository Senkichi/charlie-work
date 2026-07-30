from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import layout
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
    )
