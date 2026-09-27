"""Archive unreachable diverged branch tips on no-remote repos (issue #1944).

On a repo with no origin remote, an agent branch's commits past the default
branch can never become "pushed" — there is nowhere to push to — so the
refuse-to-reset guard's ``worktree_unsafe_local_commits`` verdict can never
resolve and every requeue escalates to human-needed forever (mdls #144).
These helpers preserve the diverged tip on a local
``archive/<branch>-<utc-date>`` branch so ``create_worktree`` can permit the
reset instead; repos WITH a remote keep refusing because a salvage push is
real there.

Extracted from ``worktree.py`` under the file-size ratchet (issue #1442):
new code must not land in the over-cap monolith, so the archive helpers —
and ``_is_confirmed_missing_ref``, the shared confirmed-absent-ref probe
predicate both they and ``_worktree_refuse_to_reset_reason`` rely on — live
here and are re-imported by ``worktree.py``, the same shape
``worktree_pr_lookup.py`` took for the #1713 fallback lookup.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .subprocess_runner import RunResult, run_captured

# Same value as ``worktree._DEFAULT_TIMEOUT_SECONDS``; declared locally like
# ``attempt_refs`` does — a cross-module import back into worktree.py would
# close an import cycle (worktree.py imports this module).
_DEFAULT_TIMEOUT_SECONDS = 60


def _is_confirmed_missing_ref(result: RunResult) -> bool:
    """True only when ``git rev-parse --verify -q <ref>`` ran to completion and
    definitively reported that ``<ref>`` does not resolve to a single
    revision (unborn ``HEAD`` in an empty repo, or a branch/tag/sha that does
    not exist) -- the one non-``ok`` outcome where "nothing to lose" is a
    sound conclusion.

    This is an allow-list on git's exit code, not a deny-list on failure
    reasons: with ``-q``/``--quiet``, git reserves exit code 1 exclusively
    for "the given ref does not resolve" and suppresses the fatal message
    entirely (confirmed against git 2.45 for an empty repo's unborn ``HEAD``
    and for a missing branch name -- both produce ``returncode=1`` with empty
    stdout/stderr). Any other non-zero outcome -- ``returncode=128`` (not a
    git repository, corrupted refs, permissions error), a git binary missing
    from PATH entirely (``RunResult.error`` set, ``returncode is None``), or
    the probe timing out -- fails this check and is therefore treated as a
    probe failure by the caller, not as a confirmed-absent ref.

    Exit code, not a stderr string match, is the discriminator on purpose:
    git's fatal messages are locale-translatable, so matching on message text
    would silently stop working (fail closed forever, not loudly) on a host
    with a non-English git locale. The exit-code contract for ``--verify -q``
    is part of git's documented plumbing behavior and does not vary with
    locale. This is the safe default either way: we only ever fail OPEN
    (report "nothing to lose") on a positive match, never on the absence of
    one.
    """
    return not result.timed_out and result.returncode == 1


def _resolve_reset_target_tip(
    repo_root: Path, branch: str, worktree_path: Path | None
) -> str | None:
    """Resolve the commit the refuse-to-reset probe measured (issue #1944).

    Mirrors ``_worktree_refuse_to_reset_reason``'s own tip resolution: the
    worktree's ``HEAD`` when the directory exists, else the ``branch`` tip.
    Returns None when the ref does not resolve — the caller keeps the refusal.
    """
    if worktree_path is not None and worktree_path.is_dir():
        result = run_captured(
            ["git", "rev-parse", "--verify", "-q", "HEAD"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
    else:
        result = run_captured(
            ["git", "rev-parse", "--verify", "-q", branch],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
    if result.ok and result.stdout.strip():
        return result.stdout.strip()
    return None


# Local-only archive namespace for diverged branch tips on repos with no
# remote (issue #1944). A plain branch — never pushed, there is nowhere to
# push to — distinct from the refs/charlie/* namespaces, which stay out of
# ``git branch`` listings on purpose; an archive the operator must be able
# to find and merge by hand is deliberately a first-class branch.
_ARCHIVE_BRANCH_PREFIX = "archive"
_ARCHIVE_NAME_ATTEMPTS = 100


def _archive_unreachable_branch_tip(repo_root: Path, branch: str, tip_sha: str) -> str | None:
    """Preserve ``tip_sha`` on branch ``archive/<branch>-<utc-date>``.

    Returns the archive branch name, or None when no archive could be
    created or verified — the caller keeps the refusal in that case, so a
    failed archive never downgrades the safety property. When a ref with the
    same name already points at the same tip, the existing ref is reused
    (the second refuse-to-reset probe of one dispatch sees the archive the
    first created). A same-named ref at a different tip gets a ``-2``,
    ``-3``, … suffix so two diverged tips on one day never overwrite.
    """
    date = datetime.now(UTC).strftime("%Y%m%d")
    base_name = f"{_ARCHIVE_BRANCH_PREFIX}/{branch}-{date}"
    for suffix in range(1, _ARCHIVE_NAME_ATTEMPTS + 1):
        name = base_name if suffix == 1 else f"{base_name}-{suffix}"
        existing = run_captured(
            ["git", "rev-parse", "--verify", "-q", f"refs/heads/{name}"],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if existing.ok and existing.stdout.strip():
            if existing.stdout.strip() == tip_sha:
                return name
            continue
        if not _is_confirmed_missing_ref(existing):
            # Not a clean "ref absent" verdict — a probe failure or a
            # verify that printed nothing. Refuse rather than guess.
            return None
        create = run_captured(
            ["git", "branch", name, tip_sha],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        return name if create.ok else None
    return None
