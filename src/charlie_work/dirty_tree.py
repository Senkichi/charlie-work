"""Pre-flight gate: prove the working tree matches the reviewed (HEAD) tree.

Issue #729: state-mutating CLI commands such as ``migrate-state-dir --apply``
execute the *working* tree, but CI only ever reviews the *committed* tree. A
guard neutered only in the working tree (not on ``HEAD``, not on
``origin/main``) is invisible to every review and test run that validated the
commit -- green-on-main is not a statement about the bytes the command is
about to execute. This module closes that gap at the boundary where the
irreversible action happens, rather than relying on an operator to remember a
procedural ``git status`` check whose absence is invisible.

The check is deliberately scoped to **tracked** files
(``git status --porcelain --untracked-files=no``): the hazard is a
modification to a file that is already in the reviewed tree, changing the
behavior of code CI saw. Untracked files (state dirs, scratch copies) are
excluded so a normal development tree does not trip the gate -- the operator
iterating on plan-only output is the normal loop, and a plan-only run never
reaches this check at all.

Errors as values (per CLAUDE.md): a git failure comes back as
``DirtyTreeReport(ok=False, error=...)`` -- never raised -- so the calling
command can refuse closed rather than crashing or silently proceeding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .subprocess_runner import run_captured

#: Timeout for the ``git status`` invocation. A local porcelain query is
#: sub-second in practice; this is a backstop against a wedged index lock.
_STATUS_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class DirtyTreeReport:
    """Result of :func:`check_working_tree_clean`.

    ``ok`` is ``True`` when the check ran to completion (the tree is clean or
    dirty); ``False`` when git itself could not be run (missing binary, index
    lock, timeout). A caller must refuse closed on ``ok=False`` rather than
    treating it as "clean" -- a probe that cannot determine cleanliness is not
    evidence of cleanliness, the same fail-closed principle
    :mod:`charlie_work.quiesce` applies to a quiescence probe failure.

    ``dirty_paths`` lists every tracked path that differs from ``HEAD``
    (modified, staged, deleted, renamed, ...), in git's own output order.
    Untracked files are excluded by construction (``--untracked-files=no``).
    Each entry is the path portion of a porcelain v1 line (everything after
    the two-character ``XY`` status and its separating space), so a rename
    shows its new path and a copy shows its destination -- the strings an
    operator reads to understand *what* diverged, not just *that* it did.
    """

    ok: bool
    dirty_paths: tuple[str, ...] = ()
    error: str | None = None

    @property
    def clean(self) -> bool:
        """``True`` when the check ran and found no tracked-file divergence."""
        return self.ok and not self.dirty_paths


def check_working_tree_clean(*, repo_root: Path) -> DirtyTreeReport:
    """Return whether *repo_root*'s tracked working tree matches ``HEAD``.

    Runs ``git status --porcelain --untracked-files=no`` in *repo_root* and
    parses the output into :class:`DirtyTreeReport`. Never raises: a git
    failure (non-zero exit, missing binary, timeout) is returned as
    ``ok=False`` with ``.error`` set, so the caller can refuse closed rather
    than crashing or silently proceeding as though the tree were clean.

    Untracked files are excluded (``--untracked-files=no``) so a normal
    development tree with scratch state does not trip the gate -- the hazard
    this gate exists to catch is a *modification* to a reviewed file, not the
    presence of untracked work.
    """
    result = run_captured(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        timeout_seconds=_STATUS_TIMEOUT_SECONDS,
    )
    if not result.ok:
        return DirtyTreeReport(
            ok=False,
            error=(
                "could not check working tree cleanliness: "
                f"{result.error or result.stderr or 'git status failed'}"
            ),
        )

    # Porcelain v1 format: ``XY <path>`` -- two status columns, one space,
    # then the path. ``line[3:]`` extracts the path; for a rename the path
    # column carries the new name (the old name is only shown with ``-z`` or
    # in the long format), which is still the string an operator needs to see.
    dirty_paths = tuple(line[3:] for line in result.stdout.splitlines() if line.strip())
    return DirtyTreeReport(ok=True, dirty_paths=dirty_paths)
