"""The "not worker product" path predicates: orchestrator scaffolding and
launcher-owned residue.

Extracted from ``worktree.py`` under the issue #1442 file-size ratchet: the
#1688 fix (pre-merge repair must recognize launcher-owned PR-body files as
repair-eligible residue, not just declared scaffolding) added the combined
predicate family to an already-over-cap monolith, so the family moves here
and is re-exported through ``worktree.py``'s facade import block (the same
``from .X import (...)  # noqa: F401 (deliberate re-export)`` pattern the
#1283 Phase-A extraction lineage established; ``rescue_capture_exclusions``
and ``base_branch`` are the prior worktree.py extractions). Callers keep
importing the names from ``charlie_work.worktree`` unchanged.

Two sides of one definition live here:

* The *ignore* side — ``_worker_authored_dirty`` treats orchestrator-declared
  scaffolding (``injected_paths`` + ``materialize_dirs``) and launcher-owned
  residue (the ``LAUNCHER_OWNED_DIRS`` directories plus the root-level
  PR-body scratch-file family) as not worker product, so their presence in a
  worktree is not dirt.
* The *destructive* side — the pre-merge repair path may discard the same
  residue when it blocks a merge, minus the launcher-owned *directories*:
  those can be junctions or other reparse points on Windows, and a
  ``git clean -f -d`` that follows one escapes the worktree (issue #1688).

``_non_worker_product_matcher`` is the single predicate both sides share;
``include_launcher_dirs`` is the load-bearing split between them. Keeping
the whole family in one module is the same drift-prevention argument
``_declared_scaffolding_matcher`` makes in its own docstring: two copies of
a rule that decides what may be destroyed are two chances to disagree.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import PurePosixPath

from .config import LAUNCHER_OWNED_DIRS

# PR body scratch files: workers ad-hoc draft PR bodies into root-level .md
# files with varying naming conventions (``PR_BODY.md``, ``PR_BODY_<issue>.md``,
# ``.worker-pr-body.md``, ``_pr_body.md``, ``.pr_body_<issue>.md``). All are
# launcher/protocol residue, not worker output (issue #1391). The regex
# matches any root-level filename in this family so a new ad-hoc variant does
# not re-trip the unsafe check.
_LAUNCHER_OWNED_PR_BODY_RE = re.compile(
    r"^(?:PR_BODY.*|\.worker-pr-body|_pr_body|\.pr_body.*)\.md$", re.IGNORECASE
)


def _launcher_owned_file_matcher() -> Callable[[str], bool]:
    """Build a predicate matching ONLY the root-level launcher-owned PR-body
    scratch files — the file half of :func:`_launcher_owned_matcher`.

    Split out for issue #1688: the pre-merge repair path may discard the
    launcher-owned *files* (a stale local ``PR_BODY.md`` edit is
    launcher-regenerable residue, and letting it stand wedges the merge
    forever — the #1477 loop), but must never extend the same license to the
    launcher-owned *directories*, which can be junctions or other reparse
    points on Windows. ``git clean``/``git checkout`` eligibility therefore
    uses this narrower predicate while the dirty check uses the full one.
    """

    def _is_launcher_owned_file(raw_path: str) -> bool:
        path = PurePosixPath(str(raw_path).replace("\\", "/"))
        # PR body scratch file match: root-level file (no path separator)
        # whose name matches the PR body family pattern.
        return len(path.parts) == 1 and _LAUNCHER_OWNED_PR_BODY_RE.match(path.name) is not None

    return _is_launcher_owned_file


def _launcher_owned_matcher() -> Callable[[str], bool]:
    """Build a predicate matching worktree-relative paths owned by the
    worker launch shim (not worker output).

    The shim materializes ``.devin/`` (the Devin CLI config directory) and
    ``.git_worktree_dir/`` into each worktree on every dispatch; workers
    also ad-hoc draft PR bodies into root-level ``.md`` scratch files.
    None of this is worker product — it is launcher/protocol residue that
    the shim re-materializes on the next dispatch — so it is excluded from
    the dirty check alongside declared scaffolding (issue #1391).

    Semantically distinct from :func:`_declared_scaffolding_matcher`:
    declared scaffolding is what the *orchestrator* itself writes
    (``injected_paths`` + ``materialize_dirs``); launcher-owned paths are
    what the *shim* writes. Both are "not worker product", but they have
    different sources and different re-materialization guarantees, so they
    are kept as separate predicates.
    """
    excluded_dirs = [PurePosixPath(d) for d in LAUNCHER_OWNED_DIRS]
    is_launcher_owned_file = _launcher_owned_file_matcher()

    def _is_launcher_owned(raw_path: str) -> bool:
        path = PurePosixPath(str(raw_path).replace("\\", "/"))
        # Directory match: path is at or under a launcher-owned directory.
        if any(path == d or d in path.parents for d in excluded_dirs):
            return True
        return is_launcher_owned_file(raw_path)

    return _is_launcher_owned


def _declared_scaffolding_matcher(
    injected_paths: tuple[str, ...] = (),
    materialize_dirs: tuple[str, ...] = (),
) -> Callable[[str], bool]:
    """Build a predicate matching worktree-relative paths the orchestrator
    itself declares it writes (``injected_paths`` + ``materialize_dirs``).

    Extracted so the dirty-check and the pre-merge collision cleanup share one
    definition of "orchestrator scaffolding, not worker product". Two copies of
    this rule that drift apart would let the cleanup delete something the dirty
    check considers worker-authored — the one outcome that must never happen.
    """
    # Normalize the configured side too, so a Windows-style backslash override
    # still matches git's forward-slash path reporting.
    excluded = [
        PurePosixPath(str(p).replace("\\", "/")) for p in (*injected_paths, *materialize_dirs)
    ]

    def _is_declared(raw_path: str) -> bool:
        # Git may emit backslashes on Windows; normalize for comparison.
        path = PurePosixPath(str(raw_path).replace("\\", "/"))
        return any(
            path == excluded_path or excluded_path in path.parents for excluded_path in excluded
        )

    return _is_declared


def _non_worker_product_matcher(
    injected_paths: tuple[str, ...] = (),
    materialize_dirs: tuple[str, ...] = (),
    *,
    include_launcher_dirs: bool,
) -> Callable[[str], bool]:
    """Build the single combined "not worker product" predicate shared by the
    dirty check and the pre-merge repair path (issue #1688).

    ``_worker_authored_dirty`` already ignored both orchestrator-declared
    scaffolding and launcher-owned residue, but the pre-merge repair path
    recognized only the declared half — so a locally-modified tracked
    ``PR_BODY.md`` colliding with the base's own moving copy could never be
    repaired and wedged the rework branch into an infinite pre-merge failure
    loop (the #1477 reproduction). Sharing one combined predicate keeps the
    two call sites from drifting apart again, the same rationale
    :func:`_declared_scaffolding_matcher` gives for existing as one function.

    ``include_launcher_dirs`` splits the launcher-owned family in two, and
    the split is load-bearing in opposite directions:

    - The *ignore* path (dirty check) passes ``True``: launcher-owned
      directories are not worker product, so residue under them is not dirt.
    - The *destructive* path (pre-merge clear/restore) passes ``False``: only
      the root-level PR-body FILE family is repair-eligible. A launcher-owned
      directory may be a junction or other reparse point on Windows, and a
      ``git clean -f -d`` that follows one escapes the worktree — the same
      hazard the ``.venv`` guard exists for.
    """
    is_declared = _declared_scaffolding_matcher(injected_paths, materialize_dirs)
    is_launcher_owned = (
        _launcher_owned_matcher() if include_launcher_dirs else _launcher_owned_file_matcher()
    )

    def _is_non_worker_product(raw_path: str) -> bool:
        return is_declared(raw_path) or is_launcher_owned(raw_path)

    return _is_non_worker_product
