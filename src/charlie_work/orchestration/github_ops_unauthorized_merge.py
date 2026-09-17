"""Unauthorized-merge detection delegate moved out of ``OrchestratorApp``.

Track 2 Phase B, L04 batch 1 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each top-level ``def``
unwrapped onto ``OrchestratorApp``. ``linked_issue_number`` and
``_authorized_override_matches`` are reached through ``_wf.``:
``linked_issue_number`` is patched on ``charlie_work.workflow`` by the suite
(Tier D, via the ``workflow_mod`` alias), and ``_authorized_override_matches``
is a ``charlie_work.workflow`` module-level def. All other free names are
imported directly (no test patches them on ``charlie_work.workflow``).
"""

from __future__ import annotations

import charlie_work.workflow as _wf
from typing import Any
from charlie_work.github import GitHubError


def _detect_unauthorized_merges(
    self, merged_prs: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Scan recently-merged PRs for worker branches whose merged head does not match an approved review decision.

    Workers are forbidden from merging their own PRs (issue #502). This
    post-merge tripwire catches a bypass after it happens: any merged PR
    whose head branch matches the configured worker branch prefix and whose
    head SHA is not covered by an approved ``.var/charlie-work/prs/pr-N/review-decision.json``
    was merged without the orchestrator's adversarial review gate. GitHub
    errors are swallowed so a transient ``gh`` failure cannot crash the
    fleet pass.

    Findings are bounded to merges this control could actually have governed
    — see ``_apply_unauthorized_merge_baseline``, which every return path
    goes through so the bound cannot be bypassed by a future caller.
    """
    candidates: list[dict[str, Any]] = []
    prefix = self.config.dispatch.branch_prefix
    # Capture the review-dispatch gate state at detection time so the finding
    # describes the moment it was made, not the moment it is later read back
    # (issue #975). Config is mutable and unversioned in events.db, so once
    # the operator flips ``enabled`` to True, every historical
    # ``decision: "missing"`` finding becomes permanently ambiguous -- was
    # the gate off (expected steady state) or on (a genuine bypass)? This
    # boolean is the only point where the answer is knowable, and it is a
    # record, not a suppressor: the finding still fires, still pins
    # ok=False, and still requires an explicit ack.
    review_dispatch_enabled = self.config.review_dispatch.enabled
    if merged_prs is None:
        try:
            merged_prs = self.gh.merged_pr_list()
        except GitHubError as exc:
            # No list means nothing observed, not "nothing wrong" — return
            # empty without arming, so a transient gh failure on the very
            # first pass cannot bake an empty baseline and permanently
            # exempt the real history it never saw.
            #
            # This covers every raising failure in merged_pr_list(): gh
            # missing, unparseable JSON, non-zero exit, AND gh exiting 0
            # with empty stdout (the silent-empty case #633 closed —
            # merged_pr_list() now raises GitHubError on a non-list result
            # instead of coercing None to []). The empty-stdout path no
            # longer arms an empty baseline silently; it is reported here
            # as a raising failure and the next pass re-arms from real
            # data.
            #
            # Failing open is correct; failing open *silently* is not. Record
            # that the control did not run (issue #937) — without this the
            # pass reports ok=True and is byte-identical in every stored
            # artifact to a pass where the tripwire ran and found nothing,
            # discarding the very distinction #633 created upstream when it
            # made merged_pr_list() raise instead of coercing to [].
            self._record_unauthorized_merge_skip(exc)
            return []

    for pr in merged_prs:
        if str(pr.get("state") or "").upper() != "MERGED":
            continue
        if pr.get("isCrossRepository") is True:
            continue
        head = str(pr.get("headRefName") or "")
        if not head.startswith(prefix):
            continue
        raw_number = pr.get("number")
        try:
            pr_number = int(raw_number) if raw_number is not None else None
        except (TypeError, ValueError):
            continue
        if pr_number is None:
            continue
        decision = self._review_decision(pr_number)
        decision_value = decision.get("decision")
        reviewed_head_sha = decision.get("reviewed_head_sha")
        live_head_sha = pr.get("headRefOid")
        approved = decision_value == "approved"
        head_matches = (
            reviewed_head_sha is not None
            and live_head_sha is not None
            and reviewed_head_sha == live_head_sha
        )
        # Issue #934: an explicit operator authorization recorded at merge
        # time is as authoritative as an approved review decision. Without
        # this, every legitimate operator merge of a worker PR (stale
        # decision, absent decision, or pending after rebase) becomes a
        # finding that pins ok=False until someone writes a retrospective
        # ack. The override is SHA-bound and reason-bearing — see
        # ``_authorized_override_matches`` — so it does not weaken the
        # control: an unrecorded merge is still a finding, and a rebase
        # after authorization invalidates the override exactly as it
        # invalidates an approved decision.
        override_authorized = _wf._authorized_override_matches(decision, live_head_sha)
        # Issue #1194: an approved decision whose head moved may be an
        # Aviator queue sync-merge of the approved head — structurally a
        # false positive. Recognized only under the fail-closed
        # four-condition predicate in _queue_sync_merge_covered, and
        # evaluated lazily: only for approved decisions with a genuine
        # head mismatch and no override, so the common paths (matching
        # head, unapproved, overridden) never pay its API calls.
        coverage_result = (
            self._queue_sync_merge_covered(pr, reviewed_head_sha, live_head_sha)
            if approved and not head_matches and not override_authorized
            else None
        )
        queue_sync_covered = coverage_result is not None and coverage_result.covered
        if (
            (not approved or not head_matches)
            and not override_authorized
            and not queue_sync_covered
        ):
            # Issue #1229 scoping decision: this call site is deliberately
            # NOT threaded through branch_issue_validator. The resolved
            # ``issue_number`` is attached to the unauthorized-merge
            # finding dict for correlation/reporting only; the finding is
            # keyed off ``pr_number`` and the review-decision mismatch,
            # and no issue-label transition or state escalation keys off
            # the ``issue`` field. A stale branch-name binding can at
            # worst mislabel the finding's ``issue`` field, not escalate
            # the wrong issue. (Contrast ``detect_mergequeue_wedged``,
            # whose ``issue_number`` DOES drive ``_escalate_issue`` and is
            # validator-threaded.)
            issue_number = _wf.linked_issue_number(
                pr,
                is_cross_repository=pr.get("isCrossRepository"),
                branch_prefix=prefix,
            )
            candidate: dict[str, Any] = {
                "pr": pr_number,
                "issue": issue_number,
                "head": head,
                "decision": decision_value,
                "reviewed_head_sha": reviewed_head_sha,
                "live_head_sha": live_head_sha,
                "review_dispatch_enabled": review_dispatch_enabled,
            }
            # coverage_result is only non-None when _queue_sync_merge_covered
            # actually ran (approved + head mismatch + no override) and did
            # not return covered=True -- i.e. exactly the cases this
            # candidate is being appended for. Distinguish "the API never
            # answered" (indeterminate) from "the shape was checked and
            # rejected" (not_covered) in the finding itself, so triage does
            # not have to re-derive it from events.db.
            if coverage_result is not None:
                if coverage_result.indeterminate:
                    candidate["coverage_check"] = "indeterminate"
                    candidate["coverage_check_error"] = coverage_result.reason
                else:
                    candidate["coverage_check"] = "not_covered"
                    candidate["coverage_reason"] = coverage_result.reason
            candidates.append(candidate)
    reported = self._apply_unauthorized_merge_baseline(candidates)
    # Announce on the bounded set, never on raw candidates: the arming pass
    # deliberately reports nothing, and an acked finding is deliberately
    # silent. Emitting before the bound would re-introduce exactly the noise
    # the baseline exists to suppress (issue #933).
    self._announce_unauthorized_merges(reported)
    return reported
