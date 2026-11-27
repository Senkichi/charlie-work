"""``tree/``: the foreman's detached worktree of its registered repo.

The foreman's Decide step reads the repo through a dedicated detached
worktree under ``foremen/<name>/tree/`` — never through the operator's own
checkout. This module keeps that worktree pointed at the default branch tip.
It routes every git call through the existing worktree helpers
(``run_captured``, ``list_worktrees``, ``_resolve_default_branch_ref``), and
removes a stale ``tree/`` only via ``git worktree remove`` — never a
recursive delete.

Failures come back as ``Result.error`` values: a missing origin symref, a
refused checkout, a leftover directory — each names what git said.
"""

from __future__ import annotations

from pathlib import Path

from ..subprocess_runner import run_captured
from ..worktree import (
    _has_origin_remote,
    _resolve_default_branch_ref,
    _run_remote_captured,
    list_worktrees,
)
from . import Result
from .paths import tree_path

_TIMEOUT_SECONDS = 60


def _registered(tree: Path, repo_root: Path) -> bool:
    """True when ``tree`` is a registered worktree of ``repo_root``."""
    resolved = tree.resolve()
    return any(w["worktree"].resolve() == resolved for w in list_worktrees(repo_root))


def _default_ref(repo_root: Path) -> str:
    """The default branch tip to park ``tree/`` on.

    ``_resolve_default_branch_ref`` returns ``origin/<branch>`` when origin
    exists and ``"HEAD"`` for a pure-local repo. For the local case a named
    mainline branch (``main``/``master``) beats whatever the operator happens
    to have checked out, so we prefer it when it resolves.
    """
    ref = _resolve_default_branch_ref(repo_root)
    if ref == "HEAD":
        for candidate in ("main", "master"):
            probe = run_captured(
                ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{candidate}"],
                cwd=repo_root,
                timeout_seconds=_TIMEOUT_SECONDS,
            )
            if probe.ok and probe.stdout.strip():
                return candidate
    return ref


def _live_default_ref(repo_root: Path) -> Result[str]:
    """The default branch tip as a *fetched* ref.

    ``origin/<branch>`` is a remote-tracking ref — without a fetch it resolves
    to whatever the last fetch left behind, and ``tree/`` would park on a
    stale tip. Fetch origin's copy of the branch first (the same
    ``_run_remote_captured`` path the worktree helpers use) so the ref names
    the live tip; a failed fetch is an error value, not a stale resolve.
    """
    try:
        ref = _default_ref(repo_root)
    except RuntimeError as exc:
        return Result(error=f"tree: cannot resolve default branch: {exc}")
    if ref.startswith("origin/") and _has_origin_remote(repo_root):
        fetched = _run_remote_captured(
            ["git", "fetch", "origin", ref[len("origin/") :]], cwd=repo_root
        )
        if not fetched.ok:
            return Result(
                error=f"tree: git fetch origin {ref[len('origin/') :]} failed: "
                f"{fetched.error or fetched.stderr.strip()}"
            )
    return Result(value=ref)


def ensure_tree(foreman_root: Path, repo_root: Path) -> Result[Path]:
    """Create ``tree/`` as a detached worktree at the default branch tip.

    Idempotent: an already-registered ``tree/`` is returned as-is. A ``tree/``
    directory that is not a registered worktree is removed via
    ``git worktree remove`` and recreated; anything that cannot be removed by
    git is an error value — never a recursive delete.
    """
    try:
        tree = tree_path(foreman_root)
    except ValueError as exc:
        return Result(error=str(exc))
    resolved = _live_default_ref(repo_root)
    if not resolved.ok:
        return Result(error=resolved.error)
    ref = resolved.value
    if _registered(tree, repo_root):
        return Result(value=tree)
    if tree.exists():
        removed = run_captured(
            ["git", "worktree", "remove", "--force", str(tree)],
            cwd=repo_root,
            timeout_seconds=_TIMEOUT_SECONDS,
        )
        if not removed.ok or tree.exists():
            detail = removed.error or removed.stderr.strip() or "directory still present"
            return Result(
                error=f"tree: stale directory at {tree} not removable by "
                f"git worktree remove: {detail}"
            )
    added = run_captured(
        ["git", "worktree", "add", "--detach", str(tree), ref],
        cwd=repo_root,
        timeout_seconds=_TIMEOUT_SECONDS,
    )
    if not added.ok:
        return Result(
            error=f"tree: git worktree add failed: {added.error or added.stderr.strip()}"
        )
    return Result(value=tree)


def sync_tree(foreman_root: Path, repo_root: Path) -> Result[Path]:
    """Move ``tree/`` to the default branch tip.

    A dirty ``tree/`` refuses the checkout — the error value carries git's
    message rather than clobbering whatever landed there.
    """
    try:
        tree = tree_path(foreman_root)
    except ValueError as exc:
        return Result(error=str(exc))
    if not _registered(tree, repo_root):
        return Result(error=f"tree: {tree} is not a registered worktree of {repo_root}")
    resolved = _live_default_ref(repo_root)
    if not resolved.ok:
        return Result(error=resolved.error)
    ref = resolved.value
    checkout = run_captured(
        ["git", "checkout", "--detach", ref],
        cwd=tree,
        timeout_seconds=_TIMEOUT_SECONDS,
    )
    if not checkout.ok:
        return Result(
            error=f"tree: git checkout --detach {ref} failed: "
            f"{checkout.error or checkout.stderr.strip()}"
        )
    return Result(value=tree)
