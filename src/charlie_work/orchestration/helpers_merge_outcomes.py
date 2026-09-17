"""Merge preflight / outcome-recording delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L02 batch 3 (issue #1654, parent #1633, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import charlie_work.workflow as _wf


def _record_event(
    self,
    state: dict[str, Any],
    kind: str,
    payload: dict[str, Any],
    *,
    level: str | None = None,
) -> dict[str, Any]:
    """Append an event to state.json and the unlimited events.db log.

    This is the single instrumentation entry point for OrchestratorApp
    methods. It forwards to ``self.write_gate.record_event`` (issue #1324)
    so every one of its ~70 call sites is dry-run-gated by construction:
    under ``dry_run=True`` the gate returns ``state`` unchanged with zero
    writes to ``events.db`` and zero mutation of the in-memory event ring
    (the WriteGate invariant -- "no event at all under dry-run"). Under
    ``dry_run=False`` the gate is a pure passthrough to ``append_event``
    with ``self.paths.state_file`` and the repo name auto-bound, dual-writing
    each event to ``state.json``'s bounded ring (``EVENT_RING_SIZE``, default
    2000) and the append-only ``events.db`` audit log. ``level`` is forwarded
    to ``append_event`` so the emit site can declare it explicitly.
    """
    return self.write_gate.record_event(state, kind, payload, level=level)


def _resolve(self, value: str) -> Path:
    # pathlib keeps an absolute right-hand side as-is, so this handles
    # both repo-relative and absolute config paths.
    return self.repo_root / value


def merge_check(self, pr_number: int) -> _wf.CommandResult:
    """Answer "is this PR merge-authorized *right now*?" without merging it.

    Issue #894. Merge authorization was enforced only on the paths that
    merge through this codebase -- ``merge_ready`` (``ship-it``) and, for
    the Aviator re-queue, ``reconcile._pr_review_approved_at_head``. A raw
    ``gh pr merge`` bypassed both, and the #502 tripwire only reports the
    bypass *after* the merge is irreversible. That is what happened to PR
    #759, whose merge was justified from GitHub's review state while
    ``review-decision.json`` recorded ``request_changes``.

    This is the preflight those paths never exposed: same invariant, no
    side effects, callable before the merge rather than after it. It is
    the single command a ``PreToolUse`` hook can shell out to, so the
    interception logic stays in the repo (versioned, CI-covered) instead
    of in unversioned agent configuration.

    **Fails closed.** Every unreadable, absent, malformed, or ambiguous
    input yields ``ok=False``. An authorization preflight that answers
    "yes" when it cannot tell is worse than no preflight, because it
    launders uncertainty into permission. ``data["reason"]`` names which
    condition fired so the caller can act on it; the distinction between
    ``not_approved`` and ``head_moved`` is the difference between "get a
    review" and "get a re-review".

    Deliberately does *not* reuse ``merge_ready``'s inline gate: that one
    may **mutate** state via approval carry-forward (``_update_approval_head``).
    A preflight must be a pure question. The shared invariant is the pair
    ``decision == "approved"`` and ``reviewed_head_sha == headRefOid``,
    asserted identically here and in ``_pr_review_approved_at_head``.
    """
    pr = self.gh.pr_view(pr_number)
    if not isinstance(pr, dict) or not pr:
        return _wf.CommandResult(
            False,
            f"PR #{pr_number}: cannot read PR from GitHub — refusing to authorize",
            {"pr": pr_number, "authorized": False, "reason": "pr_unreadable"},
        )
    if str(pr.get("state", "")).upper() == "MERGED":
        return _wf.CommandResult(
            False,
            f"PR #{pr_number} is already merged — nothing to authorize",
            {"pr": pr_number, "authorized": False, "reason": "already_merged"},
        )

    live_head_sha = pr.get("headRefOid")
    decision = self._review_decision(pr_number)
    decision_value = decision.get("decision")
    reviewed_head_sha = decision.get("reviewed_head_sha")
    base = {
        "pr": pr_number,
        "decision": decision_value,
        "reviewed_head_sha": reviewed_head_sha,
        "live_head_sha": live_head_sha,
    }

    if not live_head_sha:
        return _wf.CommandResult(
            False,
            f"PR #{pr_number}: no live head sha — refusing to authorize",
            {**base, "authorized": False, "reason": "no_live_head"},
        )
    # Issue #934: an explicit operator authorization recorded via
    # ``merge_authorize`` is as authoritative as an approved review
    # decision. Checked before the missing/invalid/not-approved/head-moved
    # gates so a valid override authorizes regardless of the recorded
    # review verdict — that is the whole point, since the override exists
    # for PRs whose verdict is stale, absent, or pending. A malformed or
    # SHA-mismatched override falls through to the existing fail-closed
    # checks below, so this adds a way to record authorization without
    # adding a way to skip the control.
    if _wf._authorized_override_matches(decision, live_head_sha):
        override = decision["authorized_override"]
        return _wf.CommandResult(
            True,
            f"PR #{pr_number}: authorized by operator override at head "
            f"{live_head_sha} (by {override.get('by') or 'unknown'})",
            {
                **base,
                "authorized": True,
                "reason": "authorized_override",
                "authorized_by": override.get("by"),
                "authorized_at": override.get("authorized_at"),
                "authorized_sha": override.get("authorized_sha"),
            },
        )
    if decision_value == "missing":
        # Issue #1362 Stage 1: resolve_decision_payload collapses both
        # "no decision file at all" and "flat file corrupt, no round
        # fallback" into this same "missing" sentinel (the old distinct
        # "invalid" sentinel no longer exists) -- both are equally
        # non-terminal for authorization purposes, so this one reason
        # covers what used to be two.
        return _wf.CommandResult(
            False,
            f"PR #{pr_number}: no readable review-decision.json — not authorized",
            {**base, "authorized": False, "reason": "no_decision"},
        )
    if decision_value != "approved":
        return _wf.CommandResult(
            False,
            f"PR #{pr_number}: recorded decision is {decision_value!r}, not 'approved'",
            {**base, "authorized": False, "reason": "not_approved"},
        )
    if reviewed_head_sha != live_head_sha:
        return _wf.CommandResult(
            False,
            (
                f"PR #{pr_number}: approved at {reviewed_head_sha} but head is now "
                f"{live_head_sha} — re-review required"
            ),
            {**base, "authorized": False, "reason": "head_moved"},
        )
    return _wf.CommandResult(
        True,
        f"PR #{pr_number}: approved at current head {live_head_sha}",
        {**base, "authorized": True, "reason": "approved_at_head"},
    )


def _record_review_or_error(
    self,
    review_result: _wf.CommandResult,
    errors: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> bool:
    """Append a review result to reviews or errors if checks are unavailable.

    Returns True if the caller should continue to the next PR (the unavailable
    case has already been recorded as an error).
    """
    if review_result.data.get("checks_unavailable"):
        errors.append({"pr": review_result.data.get("pr"), "error": review_result.message})
        return True
    reviews.append(review_result.data)
    return False


def _record_merge_or_error(
    self,
    merge_result: _wf.CommandResult,
    errors: list[dict[str, Any]],
    merges: list[dict[str, Any]],
) -> None:
    """Append a merge_ready result to merges or errors if a gh check failed."""
    if merge_result.data.get("checks_unavailable") or merge_result.data.get(
        "merge_hold_check_unavailable"
    ):
        errors.append({"pr": merge_result.data.get("pr"), "error": merge_result.message})
    else:
        merges.append(merge_result.data)
