"""Git worktree-toplevel predicate for the cross-repo gate (issue #1854).

Extracted from ``cross_repo_gate.py`` so that module stays under its
file-size high-water mark (issue #1442) — the gate keeps the call-site
guards and their failure fallbacks; this module owns the probe itself.

``git`` resolves the repository for a cwd by walking UP the directory
tree, so a git command run in a plain directory nested inside an enclosing
checkout answers with that enclosing repository's data — ``check-ignore``
consults its ignore rules, ``ls-files`` lists its tracked files. The
consumers in ``cross_repo_gate`` (``_is_gitignored``,
``_repo_tracked_files``) cannot detect that substitution from the
command's output alone: the foreign answer is well-formed, just about the
wrong repository. Both gate on :func:`_repo_is_toplevel` so a
``repo_root`` that is not its own repo root — a fixture directory, a
removed ``.git``, a nested non-repo subtree — degrades to its documented
failure fallback rather than trusting a foreign repo.
"""

from __future__ import annotations

import os
from pathlib import Path

from .subprocess_runner import run_captured

#: Timeout for the ``git rev-parse --show-toplevel`` probe. A single
#: rev-parse is sub-second in practice; this is a backstop against a
#: wedged index lock, the same role ``_CHECK_IGNORE_TIMEOUT_SECONDS``
#: plays for the gate's other git invocations.
_TOPLEVEL_TIMEOUT_SECONDS = 5


def _repo_is_toplevel(repo_root: Path) -> bool:
    """Return ``True`` when ``repo_root`` is itself the toplevel of a git
    worktree.

    ``git`` resolves the repository for a cwd by walking UP the directory
    tree, so any git command run in a plain directory nested inside an
    enclosing checkout answers with that enclosing repository's data —
    ``check-ignore`` consults its ignore rules, ``ls-files`` lists its
    tracked files. The callers of this check (``cross_repo_gate``'s
    ``_is_gitignored`` and ``_repo_tracked_files``) cannot detect that
    substitution from the command's output alone: the foreign answer is
    well-formed, just about the wrong repository. Both therefore gate on
    this check so a ``repo_root`` that is not its own repo root — a
    fixture directory, a removed ``.git``, a nested non-repo subtree —
    degrades to its documented failure fallback rather than trusting a
    foreign repo.

    Comparison is against ``rev-parse --show-toplevel`` (not ``--git-dir``
    or a ``.git`` existence check) so a linked-worktree root — whose git
    dir lives under the main checkout's ``.git/worktrees/`` — still
    counts as its own toplevel, and a subdirectory of a repo (whose
    ``.git`` lives above it) does not. ``normcase``/``normpath`` on both
    sides keeps the comparison stable across git's forward-slash output,
    drive-letter case, and symlink/junction resolution.
    """
    result = run_captured(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=repo_root,
        timeout_seconds=_TOPLEVEL_TIMEOUT_SECONDS,
    )
    if not result.ok:
        return False
    try:
        toplevel = os.path.normcase(os.path.normpath(result.stdout.strip()))
        root = os.path.normcase(os.path.normpath(str(repo_root.resolve())))
    except OSError:
        return False
    return toplevel == root
