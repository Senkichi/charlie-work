"""Default-branch resolution for ``gh pr create --base`` (issue #1250).

Extracted out of the ``worktree.py`` monolith by the file-size ratchet
(issue #1442): ``resolve_base_branch_name`` grew past ``worktree.py``'s
recorded high-water mark when the #1250 fix added the derive-from-repo
fallback, so the function moves here and is re-exported through
``worktree.py``'s facade import block (the same ``from .X import (...)  #
noqa: F401 (deliberate re-export)`` pattern the #1283 Phase-A extraction
lineage established). Callers in ``workflow.py`` / ``reconcile.py`` /
``worktree.py`` keep importing it from ``charlie_work.worktree`` unchanged.

The private helpers this calls (``_resolve_default_branch_ref`` and the
``_DEFAULT_TIMEOUT_SECONDS`` constant) stay in ``worktree.py`` -- they are
shared by ~30 other call sites there -- and are imported lazily inside the
function to avoid a top-level import cycle (``worktree`` re-exports this
module's ``resolve_base_branch_name``). The lazy ``from .worktree import ...``
pattern is already used at ``workflow.py:5076`` and ``process_utils.py:502``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .subprocess_runner import run_captured


def resolve_base_branch_name(repo_root: Path, base_ref: str) -> str:
    """Convert a base ref (e.g. ``origin/main`` or ``HEAD``) into a branch name.

    ``gh pr create --base`` expects a simple branch name. Remote-tracking refs
    are stripped to their local branch name; ``HEAD`` falls back to the current
    branch. When ``base_ref`` is empty or unrecognized and the ``HEAD`` probe
    fails, the default branch is derived from the repository's remote HEAD
    (reusing ``_resolve_default_branch_ref``, which reads
    ``git symbolic-ref refs/remotes/origin/HEAD`` and heals an unset symref via
    ``git remote set-head origin --auto``) instead of a hardcoded literal, so a
    repo whose default branch is ``master``/``trunk`` is not silently compared
    against a nonexistent ``origin/main`` — including the incident state from
    #239/#1250 where the symref was deleted after clone. The literal ``"main"``
    survives only as a last resort when the repo itself provides no answer (no
    origin remote, or an origin whose default branch cannot be healed), and its
    use is logged so the guess is visible. This function never raises.
    """
    # Lazy import: ``_resolve_default_branch_ref`` and ``_DEFAULT_TIMEOUT_SECONDS``
    # live in ``worktree.py`` (shared by ~30 call sites there), and ``worktree``
    # re-exports this function -- a top-level mutual import would cycle. The
    # lazy form matches the precedent at workflow.py:5076 / process_utils.py:502.
    from .worktree import _DEFAULT_TIMEOUT_SECONDS, _resolve_default_branch_ref

    if base_ref.startswith("refs/remotes/origin/"):
        return base_ref[len("refs/remotes/origin/") :]
    if base_ref.startswith("refs/heads/"):
        return base_ref[len("refs/heads/") :]
    if base_ref.startswith("origin/"):
        return base_ref[len("origin/") :]
    if base_ref == "HEAD":
        current_branch = run_captured(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if current_branch.ok and current_branch.stdout.strip():
            return current_branch.stdout.strip()
    # Derive the default branch from the repository's remote HEAD instead of a
    # hardcoded literal. Reuse ``_resolve_default_branch_ref`` rather than
    # re-reading the symref here so the unset-symref case is healed via
    # ``git remote set-head origin --auto`` (issue #239 / #1250): a clone whose
    # ``refs/remotes/origin/HEAD`` was deleted still resolves to the real
    # default branch instead of silently falling back to ``main``. That helper
    # returns ``"origin/<branch>"`` when origin is present, ``"HEAD"`` for a
    # pure-local repo with no origin, and raises ``RuntimeError`` when an origin
    # exists but its default branch cannot be healed. This function never
    # raises, so the RuntimeError is caught and logged here.
    try:
        default_ref = _resolve_default_branch_ref(repo_root)
    except RuntimeError as exc:
        logging.getLogger(__name__).warning(
            "resolve_base_branch_name: could not derive default branch from repo "
            "at %s (base_ref=%r): %s; falling back to hardcoded 'main'.",
            repo_root,
            base_ref,
            exc,
        )
        return "main"
    if default_ref.startswith("origin/"):
        derived = default_ref[len("origin/") :]
        if derived:
            return derived
    # ``default_ref == "HEAD"``: pure-local repo with no origin remote, so there
    # is no remote default branch to derive. The literal ``"main"`` is the last
    # resort and its use is logged so the guess is visible.
    logging.getLogger(__name__).warning(
        "resolve_base_branch_name: could not derive default branch from repo "
        "at %s (base_ref=%r); falling back to hardcoded 'main'.",
        repo_root,
        base_ref,
    )
    return "main"
