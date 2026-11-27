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

from ..subprocess_runner import RunResult, run_captured
from ..worktree import (
    _has_origin_remote,
    _resolve_default_branch_ref,
    _run_remote_captured,
    list_worktrees,
)
from . import Result
from .paths import done_check_path, tree_path

_TIMEOUT_SECONDS = 60

#: Bound on the whole ``done_when.check`` run (design §8.3: "with a
#: timeout"). Operator-authored argv like a pytest shard can take minutes;
#: ten minutes is generous for a check meant to answer "is the outcome met".
DONE_CHECK_TIMEOUT_SECONDS = 600


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


def _remove_worktree(repo_root: Path, worktree: Path) -> str | None:
    """``git worktree remove --force``; the failure detail or None on success."""
    removed = run_captured(
        ["git", "worktree", "remove", "--force", str(worktree)],
        cwd=repo_root,
        timeout_seconds=_TIMEOUT_SECONDS,
    )
    if not removed.ok or worktree.exists():
        return removed.error or removed.stderr.strip() or "directory still present"
    return None


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
        detail = _remove_worktree(repo_root, tree)
        if detail is not None:
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


def _output_tail(result: RunResult, *, lines: int = 8) -> str:
    """The last ``lines`` of a check's combined output — the briefing's tail."""
    text = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    return "\n".join(text.splitlines()[-lines:])


def run_done_check(
    foreman_root: Path,
    repo_root: Path,
    argv: tuple[str, ...],
    *,
    timeout_seconds: int = DONE_CHECK_TIMEOUT_SECONDS,
) -> Result[RunResult]:
    """Run the charter's ``done_when.check`` argv in a fresh detached worktree.

    Design §8.3: operator-authored argv, a timeout, and a worktree created
    and removed around the run — never the operator's own checkout and never
    ``tree/``, which belongs to Decide. The value is the check's
    ``RunResult``: ``.ok`` means done, anything else carries the output tail
    for the next briefing. Worktree setup/teardown failures are error
    values; a check that ran but left an unremovable worktree is an error
    carrying the ``RunResult`` as the value.
    """
    try:
        worktree = done_check_path(foreman_root)
    except ValueError as exc:
        return Result(error=str(exc))
    resolved = _live_default_ref(repo_root)
    if not resolved.ok:
        return Result(error=resolved.error)
    ref = resolved.value
    if worktree.exists():
        detail = _remove_worktree(repo_root, worktree)
        if detail is not None:
            return Result(
                error=f"done_when.check: stale worktree at {worktree} not "
                f"removable by git worktree remove: {detail}"
            )
    added = run_captured(
        ["git", "worktree", "add", "--detach", str(worktree), ref],
        cwd=repo_root,
        timeout_seconds=_TIMEOUT_SECONDS,
    )
    if not added.ok:
        return Result(
            error=f"done_when.check: git worktree add failed: "
            f"{added.error or added.stderr.strip()}"
        )
    removal: str | None = None
    try:
        result = run_captured(list(argv), cwd=worktree, timeout_seconds=timeout_seconds)
    finally:
        removal = _remove_worktree(repo_root, worktree)
    if removal is not None:
        return Result(
            value=result,
            error=f"done_when.check ran (rc={result.returncode}) but its "
            f"worktree could not be removed: {removal}",
        )
    return Result(value=result)
