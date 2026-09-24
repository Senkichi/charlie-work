"""Local (no-remote) review + merge lane primitives (issue #1844).

A ``LocalFileGitHub`` repo has no GitHub PR objects, no CI backend, and no
``gh`` CLI. This module holds the git-only building blocks the orchestrator's
local lane drives: reading a branch's head, diffing a parked branch against
the local base, running the full test suite inside the branch worktree, and
finally advancing the base branch (fast-forward when possible, ``--no-ff``
otherwise) once review + suite both pass.

Everything here is remote-free by construction: only local git plumbing is
used, so none of these helpers can accidentally reach for a remote that does
not exist. The orchestration-side state machine that calls these lives in
``orchestration/local_lanes.py`` -- this module deliberately carries no
``OrchestratorApp`` dependencies so it stays a pure, testable leaf.

Local review records reuse ``state["prs"]`` keyed by issue number with a
``"local": True`` marker (:func:`is_local_pr_record`), so the existing
decision-file/rework machinery operates on them unchanged while every
remote-only consumer can tell them apart.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .prompt_test_command import PYTEST_FLAGS, derive_pytest_runner
from .subprocess_runner import run_captured
from .worktree import list_worktrees, remove_worktree, worktree_path_for_branch

# Generous bound for a full-suite run inside a branch worktree. The suite
# failing is an ordinary outcome (routes to rework); the suite *hanging* must
# not stall the loop pass forever.
SUITE_TIMEOUT_SECONDS = 3600

# Temporary-worktree prefix for the non-fast-forward merge path. The worktree
# is created under ``worktrees_dir``, merged, and removed in the same call.
_MERGE_WORKTREE_PREFIX = "local-merge"

# Bound for ordinary git plumbing calls (rev-parse, diff, merge, worktree).
# These are local and fast; the bound exists so a wedged git process cannot
# stall the loop pass forever.
GIT_OP_TIMEOUT_SECONDS = 120


def is_local_pr_record(entry: Any) -> bool:
    """Whether a ``state["prs"]`` entry belongs to the local (no-remote) lane.

    Local records are keyed by issue number (there is no PR number to use) and
    carry ``"local": True``. Remote-only consumers that would otherwise call
    ``pr_view``/``pr_diff``/``pr_checks`` on the record use this predicate to
    skip them; the local lane uses it to select its own records.
    """
    return isinstance(entry, dict) and bool(entry.get("local"))


def local_pr_records(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """All ``state["prs"]`` entries carrying the local marker, keyed by str number."""
    return {
        key: entry for key, entry in (state.get("prs") or {}).items() if is_local_pr_record(entry)
    }


def branch_head_sha(repo_root: Path, branch: str) -> str | None:
    """Resolve ``refs/heads/<branch>`` to a commit SHA, or None when absent."""
    result = run_captured(
        ["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
        cwd=repo_root,
        timeout_seconds=GIT_OP_TIMEOUT_SECONDS,
    )
    sha = result.stdout.strip() if result.ok else ""
    return sha or None


def local_base_branch(repo_root: Path) -> str | None:
    """The branch the main worktree currently has checked out, or None detached.

    A no-remote repo's "base" is whatever the main worktree sits on --
    typically ``main``. When HEAD is detached there is no branch to advance,
    so the caller treats None as "no base" and skips/defers rather than
    guessing.
    """
    result = run_captured(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=repo_root,
        timeout_seconds=GIT_OP_TIMEOUT_SECONDS,
    )
    name = result.stdout.strip() if result.ok else ""
    return name or None


def branch_diff(repo_root: Path, base_ref: str, branch: str) -> str | None:
    """``git diff <base_ref>...<branch>`` -- the three-dot branch diff.

    Returns the diff text, or None when git reports an error (missing ref,
    no merge base). An empty string is a legitimate result (branch has no
    content delta vs. base) and is returned as-is -- callers decide whether an
    empty diff is dispatchable.
    """
    result = run_captured(
        ["git", "diff", f"{base_ref}...{branch}"],
        cwd=repo_root,
        timeout_seconds=GIT_OP_TIMEOUT_SECONDS,
    )
    if not result.ok:
        return None
    return result.stdout


def is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    """``git merge-base --is-ancestor`` -- True when ancestor is contained in descendant."""
    return run_captured(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        timeout_seconds=GIT_OP_TIMEOUT_SECONDS,
    ).ok


def worktree_for_branch(repo_root: Path, branch: str) -> Path | None:
    """The worktree path that has ``branch`` checked out, if any."""
    want = f"refs/heads/{branch}"
    for entry in list_worktrees(repo_root):
        if entry.get("branch") == want:
            return entry["worktree"]
    return None


def ensure_branch_worktree(repo_root: Path, branch: str, worktrees_dir: Path) -> Path | None:
    """Return the worktree serving ``branch``, creating one when missing.

    A merge-gate pass can legitimately find no worktree (the worker's was
    already reclaimed, or the issue predates worktree tracking). A plain
    ``git worktree add <path> <branch>`` attaches the existing branch --
    deliberately not ``create_worktree``'s fresh-branch path, which would
    refuse or reset the branch we are about to merge.
    """
    existing = worktree_for_branch(repo_root, branch)
    if existing is not None:
        return existing
    target = worktree_path_for_branch(repo_root, branch, worktrees_dir)
    result = run_captured(
        ["git", "worktree", "add", str(target), branch],
        cwd=repo_root,
        timeout_seconds=GIT_OP_TIMEOUT_SECONDS,
    )
    if not result.ok:
        return None
    # Return the spelling git recorded, matching the reuse path above: git
    # canonicalizes the worktree path at registration (8.3 short names
    # expand to long names on Windows), so returning ``target`` verbatim
    # hands callers a different spelling of the same directory depending on
    # whether the worktree was created or found.
    return worktree_for_branch(repo_root, branch) or target.resolve()


def suite_command_argv(configured_runner: str, repo_root: Path) -> list[str] | None:
    """Resolve the full-suite argv for the local merge gate.

    Same resolution as the worker prompt's test command (config override,
    else derived from the consumer's pyproject.toml) so the gate runs exactly
    what the templates tell workers to run. Returns None when nothing is
    resolvable -- the caller treats that as an escalation-worthy config gap,
    never as "suite passed".
    """
    runner = configured_runner.strip() or derive_pytest_runner(repo_root)
    if not runner:
        return None
    return shlex.split(runner) + shlex.split(PYTEST_FLAGS)


@dataclass(frozen=True)
class LocalSuiteResult:
    """Outcome of running the full test suite in a branch worktree."""

    ok: bool
    argv: tuple[str, ...]
    returncode: int | None
    # Bounded tail of combined stdout+stderr for the rework brief / event
    # payload. Never the full log -- suites can emit megabytes.
    tail: str


def run_full_suite(
    worktree_path: Path,
    argv: list[str],
    *,
    timeout_seconds: int = SUITE_TIMEOUT_SECONDS,
) -> LocalSuiteResult:
    """Run ``argv`` in ``worktree_path`` and report pass/fail as a value."""
    result = run_captured(list(argv), cwd=worktree_path, timeout_seconds=timeout_seconds)
    combined = (
        (result.stdout or "")
        + ("\n" if result.stdout and result.stderr else "")
        + (result.stderr or "")
    )
    if result.error:
        combined = f"{combined}\n{result.error}".strip()
    return LocalSuiteResult(
        ok=result.ok,
        argv=tuple(argv),
        returncode=result.returncode,
        tail=combined[-4000:],
    )


def _conflicted_paths(worktree_path: Path) -> tuple[str, ...]:
    result = run_captured(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=worktree_path,
        timeout_seconds=GIT_OP_TIMEOUT_SECONDS,
    )
    if not result.ok:
        return ()
    return tuple(sorted(line.strip() for line in result.stdout.splitlines() if line.strip()))


def _abort_merge(worktree_path: Path) -> None:
    run_captured(
        ["git", "merge", "--abort"],
        cwd=worktree_path,
        timeout_seconds=GIT_OP_TIMEOUT_SECONDS,
    )


@dataclass(frozen=True)
class LocalMergeOutcome:
    """Result of merging a reviewed branch into the local base branch."""

    # "merged" -- base now contains the branch head (ff or --no-ff).
    # "already" -- base already contained the branch head; nothing to do.
    # "conflict" -- the merge conflicted and was aborted; routes to rework.
    # "deferred" -- base checkout is dirty / merge target unsafe this pass;
    #               retry next pass (not a rework -- nothing is wrong yet).
    # "error" -- a git failure that is neither conflict nor deferral; the
    #            caller escalates.
    status: str
    fast_forward: bool
    merged_sha: str | None
    detail: str
    conflicted_paths: tuple[str, ...] = field(default_factory=tuple)


def merge_branch_into_base(
    repo_root: Path,
    base_branch: str,
    branch: str,
    *,
    worktrees_dir: Path,
) -> LocalMergeOutcome:
    """Advance ``base_branch`` to contain ``branch``, without an operator.

    The caller has already merged the base INTO the branch and run the suite
    there, so the common case is a fast-forward. When the base moved between
    the sync-merge and this call, a ``--no-ff`` merge commits the same
    integration in a temporary worktree (or the base's own checkout when one
    exists and is clean).

    Fast-forward updates the ref via ``git merge --ff-only`` inside the base
    checkout when one exists, else ``git update-ref`` directly -- both are
    atomic against a racing push of the base (the ref moves only if it still
    points where we resolved it).
    """
    head = branch_head_sha(repo_root, branch)
    if head is None:
        return LocalMergeOutcome(
            status="error",
            fast_forward=False,
            merged_sha=None,
            detail=f"branch {branch!r} does not resolve to a commit",
        )
    base_sha = branch_head_sha(repo_root, base_branch)
    if base_sha is None:
        return LocalMergeOutcome(
            status="error",
            fast_forward=False,
            merged_sha=None,
            detail=f"base branch {base_branch!r} does not resolve to a commit",
        )
    if is_ancestor(repo_root, head, base_sha):
        return LocalMergeOutcome(
            status="already",
            fast_forward=False,
            merged_sha=base_sha,
            detail=f"base {base_branch!r} already contains {branch!r} head {head[:12]}",
        )
    fast_forward = is_ancestor(repo_root, base_sha, head)
    base_wt = worktree_for_branch(repo_root, base_branch)

    if fast_forward:
        if base_wt is not None:
            # No "any dirt -> defer" pre-check: a local repo's main checkout
            # legitimately carries unrelated WIP/untracked files, and refusing
            # on those would deadlock the lane. ``git merge --ff-only`` is
            # itself the precise guard -- it fails only when the update would
            # overwrite locally-dirty paths, which IS a deferral.
            result = run_captured(
                ["git", "merge", "--ff-only", head],
                cwd=base_wt,
                timeout_seconds=GIT_OP_TIMEOUT_SECONDS,
            )
            if not result.ok:
                return LocalMergeOutcome(
                    status="deferred",
                    fast_forward=True,
                    merged_sha=None,
                    detail=(
                        f"git merge --ff-only into dirty/racing base checkout "
                        f"failed: {result.stderr or result.error}; retry next pass"
                    ),
                )
        else:
            # Base not checked out anywhere: move the ref atomically. The old
            # value guard means a concurrent advance wins instead of being
            # overwritten.
            result = run_captured(
                ["git", "update-ref", f"refs/heads/{base_branch}", head, base_sha],
                cwd=repo_root,
                timeout_seconds=GIT_OP_TIMEOUT_SECONDS,
            )
            if not result.ok:
                return LocalMergeOutcome(
                    status="deferred",
                    fast_forward=True,
                    merged_sha=None,
                    detail=(
                        f"update-ref for {base_branch!r} raced or failed: "
                        f"{result.stderr or result.error}; retry next pass"
                    ),
                )
        return LocalMergeOutcome(
            status="merged",
            fast_forward=True,
            merged_sha=head,
            detail=f"fast-forwarded {base_branch!r} to {head[:12]}",
        )

    # Non-fast-forward: needs a real merge commit in a checkout of the base.
    # Same no-pre-check rule as the fast-forward path: git merge refuses to
    # run over locally-dirty paths it would touch, so a dirty-but-unrelated
    # base checkout still merges and a dirty-and-overlapping one surfaces as
    # a merge failure we classify below (conflicted paths -> conflict, none
    # -> deferred).
    if base_wt is not None:
        merge_wt = base_wt
        remove_after = False
    else:
        merge_wt = worktrees_dir / f"{_MERGE_WORKTREE_PREFIX}-{base_branch}-{head[:12]}"
        add = run_captured(
            ["git", "worktree", "add", str(merge_wt), base_branch],
            cwd=repo_root,
            timeout_seconds=GIT_OP_TIMEOUT_SECONDS,
        )
        if not add.ok:
            return LocalMergeOutcome(
                status="error",
                fast_forward=False,
                merged_sha=None,
                detail=(f"git worktree add for merge failed: {add.stderr or add.error}"),
            )
        remove_after = True
    try:
        result = run_captured(
            ["git", "merge", "--no-ff", "--no-edit", branch],
            cwd=merge_wt,
            timeout_seconds=GIT_OP_TIMEOUT_SECONDS,
        )
        if result.ok:
            merged_sha_result = run_captured(
                ["git", "rev-parse", "HEAD"],
                cwd=merge_wt,
                timeout_seconds=GIT_OP_TIMEOUT_SECONDS,
            )
            merged_sha = merged_sha_result.stdout.strip() if merged_sha_result.ok else None
            return LocalMergeOutcome(
                status="merged",
                fast_forward=False,
                merged_sha=merged_sha,
                detail=f"merged {branch!r} into {base_branch!r} (--no-ff)",
            )
        conflicted = _conflicted_paths(merge_wt)
        _abort_merge(merge_wt)
        if not conflicted:
            # Merge refused before starting (dirty paths it would touch, or a
            # racing ref) rather than mid-merge -- nothing to rework, retry.
            return LocalMergeOutcome(
                status="deferred",
                fast_forward=False,
                merged_sha=None,
                detail=(
                    f"merge of {branch!r} into {base_branch!r} did not start: "
                    f"{result.stderr or result.error}; retry next pass"
                ),
            )
        return LocalMergeOutcome(
            status="conflict",
            fast_forward=False,
            merged_sha=None,
            detail=(f"merge of {branch!r} into {base_branch!r} conflicted; aborted"),
            conflicted_paths=conflicted,
        )
    finally:
        if remove_after:
            remove_worktree(repo_root, merge_wt)


def local_pr_dict(local_entry: dict[str, Any]) -> dict[str, Any]:
    """Project a local-lane record into the ``pr`` dict shape shared helpers read.

    ``_write_rework_prompt``/``_route_to_rework``/``check_test_adequacy`` all
    consume the GitHub ``pr`` dict shape (``number``/``title``/``headRefName``/
    ``headRefOid``/``baseRefName``/``url``). There is no ``pr_view`` to call on
    a no-remote backend, so the lane's own state record is the source.
    """
    branch = str(local_entry.get("branch") or local_entry.get("headRefName") or "")
    return {
        "number": int(local_entry.get("number") or local_entry.get("issue_number") or 0),
        "title": local_entry.get("title") or "",
        "url": "",
        "state": "OPEN",
        "headRefName": branch,
        "headRefOid": local_entry.get("headRefOid"),
        "baseRefName": local_entry.get("baseRefName"),
        "isCrossRepository": False,
        "local": True,
    }


def synthesize_open_pr(local_entry: dict[str, Any]) -> dict[str, Any] | None:
    """Project a local-lane record into the open-PR dict shape reconcile consumes.

    ``reconcile.detect_drift`` derives "does this issue have a live PR" from
    the fetched PR list. A no-remote repo's lane record IS the live review
    unit, so the local lane injects these synthetic entries into that list:
    ``state_pr_missing_on_github`` stays quiet, ``open_prs_by_issue`` stays
    populated, and ``issue_active_label_no_open_pr`` never strips a local
    issue's ``reviewing``/``needs_rework`` labels. Only terminal records
    (merged/closed) produce no entry -- there is nothing live left to protect.

    Returns None for terminal/absent fields so the caller can skip cleanly.
    """
    if not is_local_pr_record(local_entry):
        return None
    if local_entry.get("status") in ("merged", "closed"):
        return None
    number = local_entry.get("number") or local_entry.get("issue_number")
    branch = local_entry.get("branch") or local_entry.get("headRefName")
    if number is None or not branch:
        return None
    return {
        "number": int(number),
        "state": "OPEN",
        "headRefName": branch,
        "headRefOid": local_entry.get("headRefOid"),
        "baseRefName": local_entry.get("baseRefName"),
        "isCrossRepository": False,
        "title": local_entry.get("title") or "",
        "body": "",
        "labels": [],
    }
