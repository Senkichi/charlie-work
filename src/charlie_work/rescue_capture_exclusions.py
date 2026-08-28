"""Exclusion-pathspec filtering for the rescue-capture ``git add -A``.

Extracted from ``worktree.py`` (issue #1442 file-size ratchet: that module is
already over its 800-line high-water mark, so new code must land in a small
domain module and be re-exported through ``worktree.py``'s facade-import
block rather than growing the capped file directly).

On git 2.45.1.windows.1, ``git add -A -- . ':(exclude)<path>'`` exits 1 with
"The following paths are ignored by one of your .gitignore files" advice
whenever ``<path>`` both exists and is already gitignored -- even though the
resulting staged tree is correct either way (an already-ignored path was
never going to be staged by ``-A`` regardless of the exclude pathspec). This
module filters out exactly those redundant-and-harmful literal exclusions
before they reach ``git add``, so a rescue capture of a worktree with a real,
gitignored ``.venv`` (every live charlie-work worktree) no longer fails.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .config import LAUNCHER_OWNED_DIRS
from .subprocess_runner import run_captured

# Local to this module: the rescue-capture ignore-check is a fast,
# already-local `git check-ignore` invocation, not the network-touching
# commands worktree.py's own timeout constants are tuned for. Kept as a
# plain literal (rather than imported from worktree.py) to avoid a circular
# import -- worktree.py imports the functions below from this module.
_CHECK_IGNORE_TIMEOUT_SECONDS = 60


def _is_glob_pathspec(target: str) -> bool:
    """Return True if ``target`` contains a git pathspec glob metacharacter.

    Glob exclusions (e.g. ``PR_BODY*.md``) do not trigger the ignored-file
    advice/error below (empirically confirmed: ``git add -A -- .
    ':(exclude)PR_BODY*.md'`` exits 0 even when a matching file is
    gitignored), so they pass through ``_filter_redundant_add_exclusions``
    unfiltered rather than being expanded and checked per-match.
    """
    return any(ch in target for ch in ("*", "?", "["))


def _filter_redundant_add_exclusions(worktree_path: Path, targets: Sequence[str]) -> list[str]:
    """Drop literal (non-glob) exclusion targets that would trip git's
    ignored-file advice/error without changing what gets staged (issue: git
    2.45.1.windows.1's ``git add -A -- . ':(exclude)<path>'`` exits 1 with
    "The following paths are ignored by one of your .gitignore files" when
    ``<path>`` both exists and is already gitignored — even though the
    resulting staged tree is correct either way, since an already-ignored
    path was never going to be staged by the ``-A`` in the first place).

    A literal target is dropped when it does not exist under
    ``worktree_path``, or when ``git check-ignore -q`` reports it as
    ignored (returncode 0). Glob targets (``PR_BODY*.md``) are always kept:
    they do not trigger the same advice/error (verified empirically), and
    expanding them to check ignore-status per match is unnecessary.

    Never raises: a ``git check-ignore`` invocation that itself fails to run
    (missing binary, timeout) is treated as "not ignored" so the exclusion
    is conservatively kept — worst case is the pre-existing rc=1 bug this
    helper exists to avoid, not a correctness regression in what gets
    captured.
    """
    kept: list[str] = []
    for target in targets:
        if _is_glob_pathspec(target):
            kept.append(target)
            continue
        candidate = worktree_path / target
        if not candidate.exists():
            continue
        ignore_result = run_captured(
            ["git", "check-ignore", "-q", target],
            cwd=worktree_path,
            timeout_seconds=_CHECK_IGNORE_TIMEOUT_SECONDS,
        )
        if ignore_result.error is None and ignore_result.returncode == 0:
            continue
        kept.append(target)
    return kept


def _build_rescue_capture_exclusions(
    worktree_path: Path,
    injected_paths: tuple[str, ...],
    materialize_dirs: tuple[str, ...],
) -> list[str]:
    """Build the ``:(exclude)...`` pathspec list for the rescue-capture
    ``git add -A``, pruning any exclusion that would trip the ignored-file
    advice/error via ``_filter_redundant_add_exclusions``.

    ``.venv`` is always a candidate: it is either a junction into the shared
    virtualenv (following it would add every other worktree's venv
    contents) or a local venv that is not worker content. Launcher-owned
    directories (``.devin``, ``.git_worktree_dir``) are also candidates:
    they are shim residue, not worker output, and including them in the
    rescue tree pollutes the salvage commit (issue #1391). PR body scratch
    files are excluded via glob patterns (kept unconditionally — see
    ``_is_glob_pathspec``) so they do not pollute the rescue tree either.
    """
    literal_targets: list[str] = [
        ".venv",
        *LAUNCHER_OWNED_DIRS,
        ".worker-pr-body.md",
        "_pr_body.md",
    ]
    for p in (*injected_paths, *materialize_dirs):
        normalized = str(p).replace("\\", "/").strip("/")
        if normalized:
            literal_targets.append(normalized)

    # True glob patterns (contain a pathspec metacharacter) are kept
    # unconditionally by _filter_redundant_add_exclusions itself, so they
    # can be passed through the same filter call rather than special-cased
    # here.
    glob_targets = ["PR_BODY*.md", ".pr_body*.md"]

    filtered = _filter_redundant_add_exclusions(worktree_path, [*literal_targets, *glob_targets])
    return [f":(exclude){t}" for t in filtered]
