"""G3: exclude-set resolution for tree scans and backtest replay.

Combines the one sanctioned config surface (pyproject.toml
``[tool.attachment-contracts] exclude_globs``) with always-on structural
excludes and (for backtest use) git-blame-ignore-revs / codemod-shape
detection. No hand-maintained module or file lists live here — the only
literals are the structural directory names named in the spec.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

# Structural excludes: always on, independent of any config. These are the
# directory names the spec names explicitly, not a project-specific list.
_STRUCTURAL_DIR_NAMES: tuple[str, ...] = (
    ".venv",
    ".var",
    "node_modules",
    "__pycache__",
)
_STRUCTURAL_DIR_SUFFIXES: tuple[str, ...] = (
    "generated",
    "vendor",
)
_STRUCTURAL_PATH_FRAGMENTS: tuple[str, ...] = (".claude/worktrees",)


@dataclass(frozen=True)
class Excludes:
    """Resolved exclude set for one scan/backtest run."""

    exclude_globs: tuple[str, ...] = field(default_factory=tuple)
    blame_ignore_shas: frozenset[str] = field(default_factory=frozenset)

    def is_excluded_dir(self, name: str) -> bool:
        """Whether a bare directory name is always-excluded (structural)."""
        return name in _STRUCTURAL_DIR_NAMES or name in _STRUCTURAL_DIR_SUFFIXES

    def is_excluded_path(self, rel_posix: str) -> bool:
        """Whether a repo-relative posix path should be skipped entirely."""
        parts = PurePosixPath(rel_posix).parts
        if any(part in _STRUCTURAL_DIR_NAMES for part in parts):
            return True
        if any(part in _STRUCTURAL_DIR_SUFFIXES for part in parts):
            return True
        if any(fragment in rel_posix for fragment in _STRUCTURAL_PATH_FRAGMENTS):
            return True
        return any(fnmatch(rel_posix, g) for g in self.exclude_globs)

    def is_codemod_commit(self, changed_file_count: int) -> bool:
        """Backtest-only: a commit touching many files is a bulk-reformat candidate."""
        return changed_file_count > 20


def _load_exclude_globs(root: Path) -> tuple[str, ...]:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return ()
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    section = data.get("tool", {}).get("attachment-contracts", {})
    globs = section.get("exclude_globs", [])
    if not isinstance(globs, list):
        return ()
    return tuple(str(g) for g in globs)


def _load_blame_ignore_revs(root: Path) -> frozenset[str]:
    ignore_file = root / ".git-blame-ignore-revs"
    if not ignore_file.is_file():
        return frozenset()
    shas: set[str] = set()
    for line in ignore_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        shas.add(stripped)
    return frozenset(shas)


def load_excludes(root: Path) -> Excludes:
    """Resolve the full exclude set for `root` (repo root)."""
    return Excludes(
        exclude_globs=_load_exclude_globs(root),
        blame_ignore_shas=_load_blame_ignore_revs(root),
    )
