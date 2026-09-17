"""Base-currency / merge-readiness gating delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L02 batch 1 (issue #1652, parent #1633, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from typing import Any

from charlie_work.github import GitHubRunResult
import charlie_work.workflow as _wf


def _is_base_currency_gated(self, base_ref: str) -> bool:
    """Return whether ``merge_ready`` must refuse to merge a stale-base PR.

    This is the *merge gate* policy. It differs deliberately from
    ``_is_base_freshness_required`` (the *broadcast sweep* policy) and the
    difference is the entire point of issue #875.

    Issue #812 made base-freshness policy protection-derived, so that a
    repo with ``required_status_checks.strict: false`` would not burn CI
    re-running every open PR against a base GitHub does not require it to
    be current with. That reasoning is correct for the broadcast sweep,
    which writes to *N* open PRs per merge.

    Applying it to the merge gate too was the defect. This repo runs
    ``strict: false``, so the gate derived "currency not required" and
    disabled *itself*. ``_is_base_current`` was still correct and still
    failed closed; it was simply never reached. An approved PR whose base
    had advanced merged without its merged tree ever being tested, which
    is how ``main`` went red (both parents green, merge untested -- the
    same class as #853 x #865).

    The ``strict: false`` setting itself predates the hosted-CI migration
    (#1500): it was a capacity decision on the two-runner self-hosted pool
    (strict mode re-tests every open PR on every base advance, which that
    pool could not absorb). On hosted runners the capacity constraint is
    void, but ``strict: false`` is retained because the #875 merge gate
    (``require_current_base=True``, the default) already enforces
    base-freshness at merge time regardless of ``strict`` -- so
    ``strict: false`` no longer trades away safety, it only avoids the
    broadcast re-test cost. Whether to flip protection to ``strict: true``
    and let GitHub enforce base-freshness natively (redundant with this
    gate, and Aviator's queue already rebases) is an operator decision on
    a setting that lives outside this repo.

    So protection may *raise* the requirement, never *lower* it below what
    the operator asked for:

        gated = require_current_base OR protection_strict

    - ``strict: true`` + ``require_current_base=False`` -> gated (unchanged
      from #812: protection-derived enforcement still works for operators
      who opted out in config).
    - ``strict: false`` + ``require_current_base=True`` -> gated. **This is
      the #875 fix.** Previously ungated.
    - protection unreadable -> falls through to
      ``_is_base_freshness_required``'s fail-closed contract, which returns
      ``require_current_base``.

    Cost: one ``pr_update_branch`` + one CI cycle for the single PR that is
    actually merging, not N cycles across every open PR -- the broadcast
    sweep keeps calling ``_is_base_freshness_required`` and keeps #812's
    saving. The real cost is serialization: each merge staleness-invalidates
    the next PR in the lane, which must then rebase and wait for CI before
    it can land. That is the intended trade -- a slower lane against a lane
    that merges untested trees.

    Scope limit: this gate cannot retract a PR already handed to Aviator
    MergeQueue (``already_in_mergequeue`` suppresses the write, and the
    deferral does not remove the label), and it cannot stop a human
    pressing merge in the GitHub UI. It prevents the *initial* handoff of a
    stale PR, which is the path the orchestrator controls.
    """
    if self.config.auto_merge.require_current_base:
        return True
    return self._is_base_freshness_required(base_ref)


def _is_base_current(self, pr: dict[str, Any]) -> bool | None:
    """Return True if the PR's merge-base is the current tip of its base.

    Uses the GitHub compare API to derive ancestry rather than timestamps
    or mergeStateStatus, which can lag. Returns ``None`` when the comparison
    cannot be completed so callers can decide how to fail-safe.
    """
    base_ref = pr.get("baseRefName") or self.config.runners.default_branch
    head_sha = pr.get("headRefOid")
    if not base_ref or not head_sha:
        return None
    comparison = self.gh.compare(base_ref, head_sha)
    if not comparison:
        return None
    base_commit = comparison.get("base_commit")
    merge_base_commit = comparison.get("merge_base_commit")
    if not isinstance(base_commit, dict) or not isinstance(merge_base_commit, dict):
        return None
    base_sha = base_commit.get("sha")
    merge_base_sha = merge_base_commit.get("sha")
    if not base_sha or not merge_base_sha:
        return None
    return bool(base_sha == merge_base_sha)


def _is_base_freshness_required(self, base_ref: str) -> bool:
    """Return whether base currency is actually required for ``base_ref``.

    Derives base-freshness policy from GitHub branch protection
    (``required_status_checks.strict``) instead of trusting the
    hardcoded ``auto_merge.require_current_base`` config constant
    (issue #812). A repo whose protection sets ``strict: false`` does
    not require branches to be current with their base before merging,
    so forcing ``pr_update_branch`` on every open PR and deferring
    merges on staleness is pure CI-capacity waste with zero
    mergeability benefit.

    The read is cached per orchestrator pass by
    ``GitHub.branch_protection`` (cleared each pass by
    ``_loop_body``'s ``invalidate_list_cache()`` call), so PRs sharing a
    base ref within one pass cost exactly one API call in total.

    FAILS CLOSED: this is the single most important safety property of
    the derivation. If the protection read raises, returns an error
    value (``None``), 404s, is rate-limited, or
    ``required_status_checks.strict`` is absent or not a bool for any
    reason, this falls back to ``auto_merge.require_current_base``
    (default ``True``) rather than treating the failure as "no
    freshness required". An API hiccup must never silently disable base
    checking.
    """
    try:
        protection = self.gh.branch_protection(base_ref)
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "branch_protection(%s) raised; failing closed to auto_merge.require_current_base=%s",
            base_ref,
            self.config.auto_merge.require_current_base,
            exc_info=True,
        )
        return self.config.auto_merge.require_current_base
    if isinstance(protection, dict):
        required_status_checks = protection.get("required_status_checks")
        if isinstance(required_status_checks, dict):
            strict = required_status_checks.get("strict")
            if isinstance(strict, bool):
                return strict
    return self.config.auto_merge.require_current_base


def _should_update_pr_branch(
    self,
    pr: dict[str, Any],
    base_current: bool | None | _wf._BaseCurrentUnset = _wf._BASE_CURRENT_UNSET,
) -> bool:
    """Return True if the PR branch should be synced against its base.

    When a ``base_current`` signal is supplied, it is authoritative:
    ``True`` means the branch is already up-to-date and no sync is needed;
    ``False`` means the branch is stale and should be synced; ``None`` means
    the compare API is unavailable, so fail-closed and do not sync.

    If no ``base_current`` signal is supplied, fall back to the legacy
    mergeStateStatus heuristic for backward compatibility.
    """
    if isinstance(base_current, _wf._BaseCurrentUnset):
        status = str(pr.get("mergeStateStatus") or "").upper()
        return status not in {"CLEAN", "UNSTABLE", "HAS_HOOKS"}
    return base_current is False


def _verify_synced_head(self, pr_number: int, old_head_sha: str | None) -> str | None:
    """Verify that the new head of a PR is a valid base-sync merge commit.

    After ``gh pr update-branch`` advances the PR head, we must not bless the
    new SHA until we confirm it is a GitHub-generated merge commit (web-flow)
    whose parents include the previously approved head. This closes the
    approval-integrity TOCTOU: a racing push to the PR branch could otherwise
    be mistaken for a base update and auto-merged without review.

    ``old_head_sha`` must be a real SHA the caller observed before the sync
    attempt. A ``None``/falsy value means the caller never had a valid
    pre-sync head to verify against, so verification cannot succeed;
    return early rather than falling through to the parent-SHA check below,
    where ``None not in parent_shas`` would otherwise be vacuously true for
    any two-parent web-flow merge commit and mask this as a normal miss.
    """
    if not old_head_sha:
        return None
    pr = self.gh.pr_view(pr_number)
    if not pr:
        return None
    new_head_sha = pr.get("headRefOid")
    if not new_head_sha:
        return None
    if new_head_sha == old_head_sha:
        return new_head_sha

    commit_result = self.gh.commit(new_head_sha)
    if isinstance(commit_result, GitHubRunResult):
        commit = (
            commit_result.value
            if commit_result.ok and isinstance(commit_result.value, dict)
            else None
        )
    elif isinstance(commit_result, dict):
        commit = commit_result
    else:
        commit = None
    if not commit:
        return None

    parents = commit.get("parents") or []
    if len(parents) != 2:
        return None
    parent_shas = [str(p.get("sha")) for p in parents if isinstance(p, dict)]
    if old_head_sha not in parent_shas:
        return None

    committer = commit.get("committer") or {}
    if not isinstance(committer, dict):
        committer = {}
    commit_committer = commit.get("commit", {}).get("committer") or {}
    if not isinstance(commit_committer, dict):
        commit_committer = {}
    committer_login = committer.get("login")
    committer_name = commit_committer.get("name")
    # Both identity signals must match a GitHub-generated merge. The git
    # metadata name is pusher-settable, and the account login can be
    # spoofed via the committer email, so accepting either alone would let
    # a crafted racing push get blessed as a base sync. Fail closed.
    if committer_login != "web-flow" or committer_name != "GitHub":
        return None

    return new_head_sha


def _is_merge_conflict(self, pr: dict[str, Any]) -> bool:
    """Detect a genuine content conflict that gh pr update-branch cannot resolve.

    GitHub exposes this through ``mergeable`` (CONFLICTING) and through
    ``mergeStateStatus`` (DIRTY). Both are already fetched by ``pr_view``.
    """
    return (
        str(pr.get("mergeable") or "").upper() == "CONFLICTING"
        or str(pr.get("mergeStateStatus") or "").upper() == "DIRTY"
    )


def _merge_train_head(self, prs: list[dict[str, Any]] | None = None) -> int | None:
    """Return the PR number of the head of the merge-train queue, or None.

    The head is the earliest approved-pending-ship PR (same-repo, matching
    the configured branch prefix) ordered by reviewed_at (falling back to
    updatedAt), then by PR number for determinism.
    """
    candidates = self._merge_train_candidates(prs=prs)
    return candidates[0][1] if candidates else None
