"""Attempt-tip preservation for redispatch.

Motivation (issue #261, 2026-07-11 incident): every redispatch of a dead
worker resets its branch, which silently destroys the strongest diagnostic
signal available — a worker can complete several real commits and die only
at the final `git push` (e.g. blocked by a `.devin` push-gate hook). The
recovery path in ``worktree.create_worktree`` legitimately deletes a local
branch in several places (a killed-before-push session has no remote copy,
so ``git branch -D`` is the only way to reclaim the worktree slot for the
next attempt) — but until now nothing preserved the tip before deleting it.

This module snapshots a branch's current tip to a local, never-pushed ref
(``refs/charlie/attempts/issue-<n>/attempt-<k>``) immediately before any
such reset. Refs are cheap and local — they live only in the main repo's
object store (shared by every worktree) and are safe to garbage-collect
later; nothing here ever pushes them or blocks a redispatch on failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .subprocess_runner import run_captured

_DEFAULT_TIMEOUT_SECONDS = 60

# Never pushed; charlie-work owns this namespace exclusively so a collision
# with an operator's own refs is not a concern.
ATTEMPT_REF_PREFIX = "refs/charlie/attempts"
_ATTEMPT_NUM_PATTERN = re.compile(r"/attempt-(\d+)$")


@dataclass(frozen=True)
class AttemptSnapshot:
    """Result of attempting to preserve a branch tip before it is reset.

    ``ref_name``/``old_tip`` are None when there was nothing to preserve
    (the branch does not currently resolve to a commit) or when the git
    plumbing itself failed (``error`` set) — either way, callers must never
    treat this as a reason to abort the redispatch that triggered it.
    """

    ref_name: str | None
    old_tip: str | None
    ahead_of_main_count: int | None
    error: str | None = None


def _next_attempt_number(repo_root: Path, issue_number: int) -> int:
    """Return 1 + the highest existing attempt-K for this issue, or 1 if none."""
    result = run_captured(
        [
            "git",
            "for-each-ref",
            "--format=%(refname)",
            f"{ATTEMPT_REF_PREFIX}/issue-{issue_number}",
        ],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not result.ok:
        return 1
    max_seen = 0
    for line in result.stdout.splitlines():
        match = _ATTEMPT_NUM_PATTERN.search(line.strip())
        if match:
            max_seen = max(max_seen, int(match.group(1)))
    return max_seen + 1


def list_attempt_refs(repo_root: Path, issue_number: int) -> tuple[str, ...]:
    """Return every attempt ref name recorded for ``issue_number``, sorted.

    Best-effort: any git failure returns an empty tuple rather than raising —
    this is a surfacing/observability helper (doctor, digests), never a gate.
    """
    result = run_captured(
        [
            "git",
            "for-each-ref",
            "--format=%(refname)",
            f"{ATTEMPT_REF_PREFIX}/issue-{issue_number}",
        ],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not result.ok:
        return ()
    return tuple(sorted(line.strip() for line in result.stdout.splitlines() if line.strip()))


def snapshot_attempt_ref(
    repo_root: Path,
    branch: str,
    issue_number: int,
    *,
    base_ref: str = "",
) -> AttemptSnapshot:
    """Snapshot ``branch``'s current tip under refs/charlie/attempts/issue-<n>/attempt-<k>.

    ``base_ref`` (when given) is used only to compute ``ahead_of_main_count``
    for the post-mortem record — a best-effort diagnostic, not required for
    the snapshot itself.

    Never raises: every git invocation goes through ``run_captured`` (errors
    as values), and any failure returns an ``AttemptSnapshot`` with ``error``
    set. Callers must treat this as fire-and-forget insurance — a failed
    snapshot must never block or fail the redispatch that triggered it.

    Returns an all-None snapshot (no ref written) when ``branch`` does not
    currently resolve to a commit — nothing to preserve.
    """
    rev_parse = run_captured(
        ["git", "rev-parse", "--verify", branch],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not rev_parse.ok or not rev_parse.stdout.strip():
        return AttemptSnapshot(ref_name=None, old_tip=None, ahead_of_main_count=None)

    old_tip = rev_parse.stdout.strip()

    ahead_of_main_count: int | None = None
    if base_ref:
        rev_list = run_captured(
            ["git", "rev-list", "--count", f"{base_ref}..{old_tip}"],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if rev_list.ok and rev_list.stdout.strip().isdigit():
            ahead_of_main_count = int(rev_list.stdout.strip())

    attempt_num = _next_attempt_number(repo_root, issue_number)
    ref_name = f"{ATTEMPT_REF_PREFIX}/issue-{issue_number}/attempt-{attempt_num}"

    update_ref = run_captured(
        ["git", "update-ref", ref_name, old_tip],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not update_ref.ok:
        return AttemptSnapshot(
            ref_name=None,
            old_tip=old_tip,
            ahead_of_main_count=ahead_of_main_count,
            error=update_ref.error or update_ref.stderr,
        )

    return AttemptSnapshot(
        ref_name=ref_name, old_tip=old_tip, ahead_of_main_count=ahead_of_main_count
    )


__all__ = [
    "ATTEMPT_REF_PREFIX",
    "AttemptSnapshot",
    "list_attempt_refs",
    "snapshot_attempt_ref",
]
