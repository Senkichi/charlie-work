from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


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


def find_repo_root(cwd: Path | None = None) -> Path:
    start = (cwd or Path.cwd()).resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            text=True,
            capture_output=True,
            check=True,
        )
        return Path(result.stdout.strip()).resolve()
    except (OSError, subprocess.CalledProcessError):
        for candidate in (start, *start.parents):
            if (candidate / ".git").exists():
                return candidate
    return start


def runtime_paths(repo_root: Path, state_dir: str) -> RuntimePaths:
    root = Path(state_dir)
    if not root.is_absolute():
        root = repo_root / root
    root = root.resolve()
    return RuntimePaths(
        root=root,
        issues=root / "issues",
        prs=root / "prs",
        dispatches=root / "dispatches",
        logs=root / "logs",
        state_file=root / "state.json",
    )
