"""Preserve a branch tip whose worktree directory no longer exists (issue #2904).

``create_worktree`` refuses to reset a branch whose commits are not on the
remote (``worktree_unsafe_local_commits``) and, before refusing, tries to
rescue the work onto a ref. The rescue is a working-tree operation
(``git add`` / ``git write-tree`` with ``cwd`` set to the worktree), so when
the worktree directory is gone every call fails -- ``[WinError 267] The
directory name is invalid`` on Windows -- and the issue escalates again on
every requeue until someone deletes the branch by hand.

A missing directory has no dirty content to capture. The only work at risk is
the committed branch tip, which already sits on a local ref, and the refusal
message itself names the push that keeps it. This module performs that push:
it publishes the tip to ``refs/heads/salvage/<branch>`` -- or, when the remote
already holds the tip, reports where instead of pushing again.

Extracted from ``worktree.py`` under the file-size ratchet (issue #1442): new
code must not land in the over-cap monolith. The read-only network runner is
injected (``run_remote``) rather than imported, because ``worktree.py`` imports
this module at top level and a back-import would close the cycle -- the same
reason ``worktree_archive`` declares its own timeout constant. ``run_remote`` is
used for ``ls-remote`` only; the one remote-mutating call, the salvage push, is
deliberately not routed through it (see ``preserve_stranded_branch_tip``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .safe_ref import require_valid_ref_name
from .subprocess_runner import RunResult, command_failure_message, run_captured

# Same value as ``worktree._DEFAULT_TIMEOUT_SECONDS``; declared locally for the
# import-cycle reason in the module docstring.
_DEFAULT_TIMEOUT_SECONDS = 60

# The remote namespace the refuse-to-reset message tells the operator to push
# to. ``salvage_ref_for_branch`` is the one place that spells it, so the
# message (in ``worktree._worktree_refuse_to_reset_reason``) and the push this
# module performs cannot drift apart.
SALVAGE_REF_PREFIX = "refs/heads/salvage"

# Remote refs whose tip means "this work is on the remote": branches, and PR
# heads -- which outlive a deleted branch, so a closed PR's commits stay
# reachable from ``refs/pull/<n>/head``.
_REMOTE_REF_PATTERNS = ("refs/heads/*", "refs/pull/*/head")

_TRACKING_PREFIX = "refs/remotes/origin/"

# ``(command, cwd) -> RunResult`` -- ``worktree._run_remote_captured``. Retrying
# and 20s-bounded, so it is for read-only ``ls-remote`` probes and nothing that
# mutates the remote.
RunRemote = Callable[[list[str], Path], RunResult]


def salvage_ref_for_branch(branch: str) -> str:
    """Return the remote salvage ref for ``branch`` (``refs/heads/salvage/<branch>``)."""
    return f"{SALVAGE_REF_PREFIX}/{branch}"


@dataclass(frozen=True)
class StrandedTipOutcome:
    """Result of preserving a branch tip that has no worktree directory.

    ``ref_name`` is the REMOTE ref that now holds (or already held) the tip --
    the salvage ref after a push, or whichever branch / PR-head ref already
    contained it. ``pushed`` tells those apart. ``ref_name`` is None exactly
    when ``error`` is set; the caller must treat that as a hard refusal.
    ``commit_sha`` is the tip whenever it resolved, even on failure.
    """

    ref_name: str | None
    commit_sha: str | None
    pushed: bool = False
    error: str | None = None


def preserve_stranded_branch_tip(
    repo_root: Path, branch: str, run_remote: RunRemote
) -> StrandedTipOutcome:
    """Make ``refs/heads/<branch>``'s tip durable on the remote; never raises.

    Order, with the safety property each step carries:

    1. Resolve the tip from the local branch ref. An unresolvable tip is an
       error -- there is nothing to claim durable.
    2. If the remote already advertises the tip (a branch or a PR head), skip
       the push and report that ref. A probe that fails here only forfeits the
       shortcut: the push below is still attempted.
    3. Otherwise push ``<tip>:refs/heads/salvage/<branch>`` -- a plain push,
       never forced, so an existing salvage ref at a different tip is rejected
       rather than overwritten -- and read the ref back from the remote. The
       work counts as durable only when the remote advertises the tip. The
       push runs once through ``run_captured`` (never ``run_remote``: a
       remote-mutating command must not sit behind a retry loop); only the
       read-only ``ls-remote`` probes use ``run_remote``.

    Every failure comes back as ``error`` so the caller keeps its
    ``WorktreeUnsafeError``: a failed rescue must not weaken the refusal.
    """
    try:
        branch = require_valid_ref_name(branch, context="stranded-tip branch")
    except ValueError as exc:
        return StrandedTipOutcome(ref_name=None, commit_sha=None, error=str(exc))

    local_ref = f"refs/heads/{branch}"
    tip_result = run_captured(
        ["git", "rev-parse", "--verify", "-q", local_ref],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    tip = tip_result.stdout.strip()
    if not tip_result.ok or not tip:
        return StrandedTipOutcome(
            ref_name=None,
            commit_sha=None,
            error=f"cannot resolve {local_ref} to a commit to preserve",
        )

    advertised = _advertised_remote_refs(repo_root, run_remote)
    if advertised is not None:
        holder = _remote_ref_holding_tip(repo_root, tip, advertised)
        if holder is not None:
            return StrandedTipOutcome(ref_name=holder, commit_sha=tip)

    salvage_ref = salvage_ref_for_branch(branch)
    push_command = ["git", "push", "origin", f"{tip}:{salvage_ref}"]
    # The push mutates the remote, so it must NOT go through ``run_remote``:
    # that is ``worktree._run_remote_captured``, whose retry loop
    # (``git_retry.run_git_with_retry``) is for read-only/idempotent commands
    # only and whose 20s read timeout is sized for ``ls-remote``. A plain
    # ``run_captured`` is one attempt with the module's normal budget. A push
    # that timed out after landing is still safe: the next call's probe finds
    # the tip on the salvage ref and reports it without pushing again.
    push_result = run_captured(
        push_command, cwd=repo_root, timeout_seconds=_DEFAULT_TIMEOUT_SECONDS
    )
    if not push_result.ok:
        detail = command_failure_message(push_command, push_result, "git push failed")
        return StrandedTipOutcome(
            ref_name=None, commit_sha=tip, error=f"push to {salvage_ref} failed: {detail}"
        )

    readback = run_remote(["git", "ls-remote", "origin", salvage_ref], repo_root)
    remote_sha = readback.stdout.split()[0] if readback.ok and readback.stdout.strip() else None
    if remote_sha != tip:
        return StrandedTipOutcome(
            ref_name=None,
            commit_sha=tip,
            error=(
                f"push to {salvage_ref} reported success but the remote does not "
                f"advertise {tip[:12]} there (saw {remote_sha or 'nothing'})"
            ),
        )
    return StrandedTipOutcome(ref_name=salvage_ref, commit_sha=tip, pushed=True)


def _advertised_remote_refs(repo_root: Path, run_remote: RunRemote) -> dict[str, str] | None:
    """Return ``{ref: sha}`` for the remote's branches and PR heads, or None.

    None means the probe failed (no origin, network, auth) -- the caller cannot
    prove anything from it and must not read it as "the remote holds nothing".
    """
    result = run_remote(["git", "ls-remote", "origin", *_REMOTE_REF_PATTERNS], repo_root)
    if not result.ok:
        return None
    refs: dict[str, str] = {}
    for line in result.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if sha.strip() and ref.strip():
            refs[ref.strip()] = sha.strip()
    return refs


def _remote_ref_holding_tip(repo_root: Path, tip: str, advertised: dict[str, str]) -> str | None:
    """Name a remote branch or PR head that already contains ``tip``, else None.

    Two sound proofs, both anchored on what the remote advertises *now*:

    - **Exact**: a branch or PR head whose advertised sha IS the tip. This is
      the closed-PR shape -- the branch was deleted, ``refs/pull/<n>/head``
      still points at the final tip.
    - **Containing**: a local remote-tracking ref whose commit contains the
      tip AND whose remote counterpart still advertises that same commit, so a
      tracking ref that went stale cannot vouch for a remote that moved on.

    A tip that is only an *ancestor* of a PR head the repo never fetched is not
    detected; that costs one redundant salvage push, never a lost commit.
    """
    exact = sorted(
        (ref for ref, sha in advertised.items() if sha == tip),
        key=lambda ref: (not ref.startswith("refs/heads/"), ref),
    )
    if exact:
        return exact[0]

    contains = run_captured(
        [
            "git",
            "for-each-ref",
            "--contains",
            tip,
            "--format=%(objectname) %(refname)",
            _TRACKING_PREFIX.rstrip("/"),
        ],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not contains.ok:
        return None
    for line in sorted(contains.stdout.splitlines()):
        sha, _, tracking = line.partition(" ")
        if not tracking.startswith(_TRACKING_PREFIX):
            continue
        name = tracking[len(_TRACKING_PREFIX) :]
        remote_ref = f"refs/heads/{name}"
        if name and name != "HEAD" and advertised.get(remote_ref) == sha:
            return remote_ref
    return None
