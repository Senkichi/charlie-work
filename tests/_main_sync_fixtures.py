"""Real-git fixtures for main-sync tests: a bare origin, an upstream clone that pushes,
and the checkout under test. Underscore-prefixed: a shared helper, not a test module.

``tests/conftest.py`` already isolates git (ceiling dirs, ``GIT_CONFIG_*``); identity
is set per repo so commits never depend on the host's global config.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


def git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _identity(repo: Path) -> None:
    git(repo, "config", "user.name", "Main Sync Test")
    git(repo, "config", "user.email", "main-sync@example.invalid")
    git(repo, "config", "commit.gpgsign", "false")


@dataclass(frozen=True)
class SyncRepos:
    origin: Path
    upstream: Path
    clone: Path

    def push_commit(self, relpath: str, content: str, message: str | None = None) -> str:
        """Commit ``relpath`` in upstream, push it to origin's main, return its sha."""
        target = self.upstream / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        git(self.upstream, "add", relpath)
        git(self.upstream, "commit", "-q", "-m", message or f"edit {relpath}")
        git(self.upstream, "push", "-q", "origin", "HEAD:main")
        return git(self.upstream, "rev-parse", "HEAD")

    def head(self, path: Path | None = None) -> str:
        return git(path or self.clone, "rev-parse", "HEAD")

    def fetch(self) -> None:
        git(self.clone, "fetch", "-q", "origin")

    def graft_origin_tip(self) -> str:
        """Push two commits, then ``fetch --depth 1``: the full clone turns shallow at
        origin's new tip, which hides the merge base with local main (cw#2849 live)."""
        self.push_commit("graft-a.txt", "a\n")
        tip = self.push_commit("graft-b.txt", "b\n")
        git(self.clone, "fetch", "-q", "--depth", "1", f"file://{self.origin.as_posix()}", "main")
        git(self.clone, "update-ref", "refs/remotes/origin/main", tip)
        return tip


def make_sync_repos(tmp_path: Path) -> SyncRepos:
    tmp_path.mkdir(parents=True, exist_ok=True)
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    upstream = tmp_path / "upstream"
    git(tmp_path, "init", "-q", "-b", "main", str(upstream))
    _identity(upstream)
    git(upstream, "remote", "add", "origin", str(origin))
    (upstream / "README.md").write_text("seed\n", encoding="utf-8")
    git(upstream, "add", "README.md")
    git(upstream, "commit", "-q", "-m", "seed")
    git(upstream, "push", "-q", "origin", "HEAD:main")
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(origin), str(clone))
    _identity(clone)
    return SyncRepos(origin=origin, upstream=upstream, clone=clone)
