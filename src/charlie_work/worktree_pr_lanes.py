"""Resolved-PR lane verdicts for ``clean_worktrees`` (issue #2487 rework).

Extracted from ``worktree.py`` under the file-size ratchet (issue #1442):
new code must not land in the over-cap monolith, and the #2487 lane-order
fix needs the resolved-PR eligibility check as a *value* -- a verdict the
caller can decline on -- rather than the inline ``skipped.append`` +
``continue`` it used to be, so the whole block moved here. Same shape
``worktree_pr_lookup.py`` took for the #1713 fallback lookup and
``worktree_closed_reclaim.py`` for the closed-issue lane itself.

Lane order is the load-bearing property the extraction serves: a closing
keyword auto-closes an issue when its PR merges, so issue-CLOSED +
merged-PR is the COMMON merged shape, not an edge. ``clean_worktrees``
must let the resolved-PR lanes evaluate every candidate first -- the
merged lane is the one that deletes the branch -- and hand the
closed-issue lane only the candidates the PR lanes declined. This module
is the PR side of that dispatch: :func:`pr_lane_verdict` returns the
decline reason ``clean_worktrees`` would previously have appended to
``skipped`` directly (``None`` when the candidate is eligible for the
shared liveness/dry-run/removal tail), plus whether the live
``gh pr view`` confirmed a terminal PR state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .github import GitHubRunResult, PR_VIEW_MERGED_FIELDS
from .worktree_pr_lookup import WorktreeCleanGH

if TYPE_CHECKING:
    from .config import OrchestratorConfig


@dataclass(frozen=True)
class PrLaneVerdict:
    """Outcome of the resolved-PR eligibility check for one worktree.

    ``decline_reason`` is ``None`` when the worktree is eligible for the
    shared liveness/dry-run/removal tail of ``clean_worktrees``; otherwise
    it is the human-readable reason the PR lanes produced for skipping it.

    ``terminal`` records whether the live ``gh pr view`` confirmed a
    terminal PR state (``MERGED``, or ``CLOSED``-and-never-merged). The
    closed-issue lane may only reclaim a declined worktree whose resolved
    PR is terminal -- a still-open or unconfirmable PR keeps the wait even
    when the linked issue is closed, because an issue can be closed by
    hand while its PR is still under review.
    """

    decline_reason: str | None
    terminal: bool


def pr_lane_verdict(
    repo_root: Path,
    wt_path: Path,
    branch: str,
    pr_number: int,
    state_merged: bool,
    config: OrchestratorConfig,
    gh: WorktreeCleanGH,
) -> PrLaneVerdict:
    """Evaluate the merged / closed-unmerged PR lanes for one candidate.

    Runs the live ``gh pr view`` confirmation and, for a confirmed
    ``MERGED`` PR, the dirty-tree + merged-head-containment gates; for a
    confirmed closed-never-merged PR, the ``_worktree_refuse_to_reset_reason``
    nothing-would-be-lost gate. Returns ``PrLaneVerdict(None, True)`` when
    the candidate is eligible for removal by the caller's shared tail;
    otherwise the decline reason and whether the PR itself is terminal.

    Every gate here is fail-closed exactly as it was inline in
    ``clean_worktrees``: an erroring or ambiguous ``gh`` call, a failed
    probe, or an unrecognized eligibility state all produce a decline,
    never an eligible verdict.
    """
    # Deferred: worktree.py imports this module at top level, so its
    # helpers are reached only after it has finished initializing -- the
    # same cycle break ``worktree_closed_reclaim.py`` uses. Resolving them
    # through the worktree module object (rather than importing the
    # subprocess_runner originals directly) also preserves the
    # ``worktree.run_captured`` monkeypatch seam the existing suite uses.
    from .worktree import (
        _DEFAULT_TIMEOUT_SECONDS,
        WorktreeProbeFailedError,
        _has_origin_remote,
        _is_ancestor,
        _object_exists,
        _run_remote_captured,
        _worktree_dirty_reason,
        _worktree_refuse_to_reset_reason,
        run_captured,
    )

    # `allow_failure=True` means this never raises (see GitHub.run): errors
    # come back as GitHubRunResult(ok=False, error=...), not exceptions.
    gh_result = gh.run(
        ["pr", "view", str(pr_number), "--json", PR_VIEW_MERGED_FIELDS],
        json_output=True,
        allow_failure=True,
    )
    gh_ok = isinstance(gh_result, GitHubRunResult) and gh_result.ok
    gh_merged = False
    gh_closed_unmerged = False
    merged_head_sha: str | None = None
    if gh_ok and isinstance(gh_result.value, dict):
        gh_pr_state = gh_result.value.get("state")
        gh_merged = gh_pr_state == "MERGED"
        merged_head_sha = gh_result.value.get("headRefOid")
        # Terminal, not pending: CLOSED-and-never-merged is a decision, not
        # a not-yet-merged state, so it must not collapse into the
        # "PR not merged" wait-forever branch below (issue #990). This is
        # still positive proof from the SAME live `gh pr view` call the
        # merged path already makes -- no extra quota -- and it is exactly
        # as fail-closed: an erroring/ambiguous `gh` call still falls
        # through to the generic "not merged" branch, which skips.
        gh_closed_unmerged = gh_pr_state == "CLOSED" and gh_result.value.get("mergedAt") is None

    if not gh_merged and not gh_closed_unmerged:
        # Fail-closed: only a live `gh pr view` MERGED or CLOSED-unmerged
        # confirmation may authorize destructive removal. state.json is
        # corroboration at best -- never sufficient on its own (state.json
        # reliability history: #285/#309/#310). An unavailable/erroring
        # `gh` call falls into this branch too and is DISTINGUISHED from a
        # confirmed-not-merged PR rather than falling back to trusting
        # state.json.
        if not gh_ok:
            gh_error = gh_result.error if isinstance(gh_result, GitHubRunResult) else "unknown"
            decline_reason = f"gh pr view unavailable; cannot confirm merge status: {gh_error}"
        elif state_merged:
            decline_reason = "state.json says merged but gh pr view did not confirm MERGED"
        else:
            decline_reason = "PR not merged"
        return PrLaneVerdict(decline_reason=decline_reason, terminal=False)

    if gh_merged:
        try:
            dirty_reason = _worktree_dirty_reason(
                wt_path,
                config.dispatch.injected_paths,
                config.dispatch.materialize_dirs,
            )
        except WorktreeProbeFailedError as exc:
            return PrLaneVerdict(
                decline_reason=f"worktree status probe failed: {exc}", terminal=True
            )
        if dirty_reason:
            return PrLaneVerdict(decline_reason=dirty_reason, terminal=True)
        if not merged_head_sha:
            return PrLaneVerdict(
                decline_reason="gh pr view did not return headRefOid for the merged PR",
                terminal=True,
            )
        head_result = run_captured(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=wt_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not head_result.ok or not head_result.stdout.strip():
            return PrLaneVerdict(decline_reason="could not resolve worktree HEAD", terminal=True)
        local_head_sha = head_result.stdout.strip()
        if local_head_sha != merged_head_sha:
            # Containment, NOT equality. The question this gate exists to ask
            # is "does the worktree hold commits that did not get merged?",
            # and an equality test cannot tell the two directions apart:
            #
            #   local BEHIND merged  -> everything here is reachable from the
            #       merged head; nothing to lose. This is the ORDINARY shape,
            #       because the merge path advances the PR branch after the
            #       worker's last local commit (Aviator merge-queue rebases,
            #       merge-train updates, base-into-branch merges). 46 of 47
            #       mismatching worktrees on this host were this shape and were
            #       all reported as "stray post-merge commit(s)".
            #   local AHEAD/DIVERGED -> real unmerged work; refuse.
            #
            # This is the same class of error the note above records for
            # `_worktree_refuse_to_reset_reason`: a check whose shape counts
            # the expected post-merge topology as danger.
            #
            # The containment test needs `merged_head_sha` to still be in
            # the local object store. For a squash-merged PR whose remote
            # branch was deleted, nothing references that SHA once the
            # local branch sits behind it, so a `git gc` can prune it (or
            # this repo may simply never have fetched a branch that ever
            # pointed at it), and the object-presence gate below starts
            # refusing every otherwise-reclaimable worktree, permanently
            # (37 of 63 candidates in one live sweep; root growing without
            # bound). GitHub still serves the commit via
            # `refs/pull/<N>/head` even after the head branch is deleted —
            # unlike a raw-SHA fetch, which most servers (GitHub included)
            # refuse to advertise — so attempt one bounded, non-interactive
            # fetch of that ref and re-check before failing closed. The
            # fetch only adds objects: it never moves a branch, touches a
            # worktree, or leaves FETCH_HEAD in a state some other call
            # site depends on, so it is safe to run unconditionally,
            # including under `--dry-run` — every other eligibility probe
            # in this function (gh pr view, rev-parse, the ancestor check
            # below) already runs regardless of `dry_run`, so withholding
            # just this fetch would make a dry-run preview under-report
            # what a real run could actually reclaim.
            if not _object_exists(repo_root, merged_head_sha) and _has_origin_remote(repo_root):
                _run_remote_captured(
                    ["git", "fetch", "origin", f"refs/pull/{pr_number}/head"],
                    cwd=repo_root,
                    extra_env={"GIT_TERMINAL_PROMPT": "0"},
                )
            if not _object_exists(repo_root, merged_head_sha):
                # Still absent -- no origin remote, no such PR ref on the
                # remote, or the fetch itself failed/timed out.
                # `_run_remote_captured` never raises, so a network
                # failure here degrades to this same fail-closed skip
                # rather than aborting the sweep.
                return PrLaneVerdict(
                    decline_reason=(
                        f"merged PR head ({merged_head_sha[:8]}) is not present in the "
                        "local object store; cannot verify the worktree adds nothing "
                        "beyond it"
                    ),
                    terminal=True,
                )
            if not _is_ancestor(repo_root, local_head_sha, merged_head_sha):
                return PrLaneVerdict(
                    decline_reason=(
                        f"worktree HEAD ({local_head_sha[:8]}) is not contained in "
                        f"merged PR head ({merged_head_sha[:8]}); stray post-merge "
                        "commit(s)"
                    ),
                    terminal=True,
                )
        return PrLaneVerdict(decline_reason=None, terminal=True)

    if gh_closed_unmerged:
        # A closed-and-never-merged PR is a terminal decision (issue
        # #990), not a pending state -- nothing will ever advance it to
        # MERGED, so the worktree cannot wait on that. There is no
        # "merged head" to compare against here (the PR never landed), so
        # eligibility is the ORDINARY "would removing this lose anything"
        # question instead of the merged path's special
        # squash-with-deleted-branch containment check: is the working
        # tree clean, and does the branch have any commits that are not
        # also on its remote copy? `_worktree_refuse_to_reset_reason`
        # already answers exactly that (dirty-tree check + remote-ahead
        # check, positive-proof throughout) for the redispatch lane, and a
        # closed-unmerged PR's branch is the ordinary case it was built
        # for -- closing without merging does not *automatically* delete
        # the remote branch the way GitHub's merge-time auto-delete does
        # (an operator or Aviator can still delete it by hand), so this is
        # not the squash-merge special case documented above; the helper's
        # remote-ahead check already refuses when the remote branch is
        # gone and the local tip carries commits beyond base. Reused as-is
        # rather than re-derived.
        try:
            closed_unsafe_reason = _worktree_refuse_to_reset_reason(
                repo_root,
                branch,
                config.dispatch.base_ref,
                wt_path,
                config.dispatch.injected_paths,
                config.dispatch.materialize_dirs,
            )
        except (WorktreeProbeFailedError, RuntimeError) as exc:
            return PrLaneVerdict(
                decline_reason=f"closed-unmerged PR reclaim safety probe failed: {exc}",
                terminal=True,
            )
        if closed_unsafe_reason:
            return PrLaneVerdict(
                decline_reason=f"closed-unmerged PR: {closed_unsafe_reason}",
                terminal=True,
            )
        return PrLaneVerdict(decline_reason=None, terminal=True)

    # Unreachable today: the `not gh_merged and not gh_closed_unmerged`
    # guard above already returns for every state that is neither.
    # Kept as an explicit fail-closed branch rather than relying on
    # that guard alone, so a future third eligibility state added
    # above this `if` cannot silently fall through into either
    # removal path via a bare `else`.
    return PrLaneVerdict(
        decline_reason=(
            "PR eligibility state not recognized (neither merged nor closed-unmerged)"
        ),
        terminal=False,
    )
