"""Merge-train and externally-merged-finalization delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L01 batch 3 (issue #1646, parent #1632, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from charlie_work.github import GitHubError
import charlie_work.workflow as _wf


def _merge_train_candidates(
    self,
    prs: list[dict[str, Any]] | None = None,
    exclude_pr_number: int | None = None,
) -> list[tuple[str, int, dict[str, Any], dict[str, Any], str]]:
    """Return approved-pending-ship candidates sorted by approval time.

    Each tuple contains (sort_key, pr_number, pr, decision, head_ref).
    """
    if prs is None:
        prs = self.gh.pr_list()

    branch_prefix = self.config.dispatch.branch_prefix
    human_merge_labels = self.config.dispatch.human_merge_labels
    # Aviator MergeQueue handoff (task #10): a PR already parked in
    # Aviator's queue (state status "mergequeue") must never occupy
    # charlie's merge-train head — Aviator now owns serialization for it.
    # Without this exclusion the parked PR keeps winning "earliest
    # reviewed" on every poll until GitHub reports it merged, so under
    # front_of_train no other approved PR is ever attempted while Aviator
    # is still processing it. Reading state is only necessary when the
    # mergequeue handoff feature is actually configured.
    state_prs = (
        _wf.load_state_locked(self.paths.state_file).get("prs", {})
        if self.config.auto_merge.mergequeue_label
        else {}
    )
    candidates: list[tuple[str, int, dict[str, Any], dict[str, Any], str]] = []
    for pr in prs:
        pr_number = int(pr.get("number", 0))
        if pr_number == exclude_pr_number:
            continue
        if pr.get("isCrossRepository"):
            continue
        head = str(pr.get("headRefName") or "")
        if not head.startswith(branch_prefix):
            continue
        if (state_prs.get(str(pr_number)) or {}).get("status") == "mergequeue":
            continue
        decision = self._review_decision(pr_number)
        if decision.get("decision") != "approved":
            continue
        reviewed_head_sha = decision.get("reviewed_head_sha")
        live_head_sha = pr.get("headRefOid")
        if reviewed_head_sha is None or live_head_sha != reviewed_head_sha:
            continue
        # Issue #1598: a bound PR whose issue carries a configured
        # human_merge_labels label is never a merge-train candidate —
        # it is human-merged, not fleet-merged. The check reads live
        # issue labels at decision time so an operator adding or
        # removing the label mid-flight takes effect on the next pass.
        # Skipped entirely when human_merge_labels is empty (default),
        # preserving current behaviour with zero overhead.
        if human_merge_labels and self._pr_bound_issue_has_human_merge_label(pr, branch_prefix):
            continue
        reviewed_at = decision.get("reviewed_at") or pr.get("updatedAt") or ""
        candidates.append((str(reviewed_at), pr_number, pr, decision, head))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates


def _merge_deferred_stale_base_result(
    self,
    pr_number: int,
    issue_number: int | None,
    decision: dict[str, Any],
    base_ref: str | None,
    head_sha: str | None,
    reason: str = "base_stale",
) -> _wf.CommandResult:
    """Return a non-mergeable result for an approved PR whose base is stale.

    Records a ``merge_deferred_stale_base`` event so operators can see that
    the merge was deferred because the PR's merge-base is not the current
    base branch tip.
    """
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        existing = state["prs"].get(str(pr_number), {})
        new_stale_base_deferrals = int(existing.get("consecutive_stale_base_deferrals", 0)) + 1
        threshold = self.config.auto_merge.failed_attempt_alarm
        stale_base_alarm = threshold > 0 and new_stale_base_deferrals == threshold
        stale_base_warning: str | None = None
        if stale_base_alarm:
            stale_base_warning = _wf._format_stale_base_alarm_message(
                pr_number, new_stale_base_deferrals, reason
            )
            state = _wf.append_event(
                state,
                "merge_deferred_stale_base_alarm",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "base_ref": base_ref,
                    "head_sha": head_sha,
                    "reason": reason,
                    "attempts": new_stale_base_deferrals,
                    "threshold": threshold,
                    "message": stale_base_warning,
                },
                state_path=self.paths.state_file,
            )
        state = _wf.append_event(
            state,
            "merge_deferred_stale_base",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "base_ref": base_ref,
                "head_sha": head_sha,
                "reason": reason,
            },
            state_path=self.paths.state_file,
        )
        state["prs"][str(pr_number)] = {
            **existing,
            "number": pr_number,
            "issue_number": issue_number,
            "consecutive_stale_base_deferrals": new_stale_base_deferrals,
        }
        _wf.save_state(self.paths.state_file, state)
    return _wf.CommandResult(
        True,
        f"PR #{pr_number} base is stale; merge deferred until base is current",
        {
            "pr": pr_number,
            "issue": issue_number,
            "can_merge": False,
            "auto_merge_enabled": self.config.auto_merge.enabled,
            "merged": False,
            "merge_output": None,
            "branch_deleted": None,
            "review_decision": decision,
            "checks": asdict(_wf.summarize_checks([], self.config.auto_merge.required_checks)),
            "checks_unavailable": False,
            "label_error": None,
            "update_open_prs_results": None,
            "cancel_superseded_runs_results": None,
            "containment_warnings": [],
            "stale_base": True,
            "consecutive_failed_merge_attempts": existing.get(
                "consecutive_failed_merge_attempts", 0
            ),
            "consecutive_stale_base_deferrals": new_stale_base_deferrals,
            "merge_attempt_alarm": stale_base_alarm,
            "merge_attempt_warning": stale_base_warning,
            "merge_conflict": False,
        },
    )


def _finalize_externally_merged_issues(
    self,
    ready_issues: list[dict[str, Any]] | None = None,
) -> tuple[set[int], list[dict[str, Any]], _wf._MergedPRListOutcome]:
    """Finalize closed ready-labeled issues whose linked PR merged externally,
    and strip the ready/active labels from closed ready issues that have no
    merged PR binding them (issue #429/#433).

    Runs before dispatch capacity guards (fleet lock, GraphQL budget, provider
    throttle) so a pass that defers new work still drains the backlog of
    externally-merged issues (e.g. Aviator MergeQueue handoffs).  It first
    binds candidates against the cheap most-recent-500 ``merged_pr_list()``;
    only issues whose merged PR falls outside that window incur a per-issue
    ``gh pr list --search`` lookup.

    Per-issue lookups are capped at ``dispatch.finalize_limit`` and processed
    oldest-first (by ``createdAt``, then issue number). A consecutive-failure
    circuit breaker stops the pass after 3 failed lookups so a transient
    Search API rate limit does not monopolize the shared token.
    """
    if ready_issues is None:
        ready_issues = self.gh.issue_list(
            labels=[self.config.labels.ready],
            state="all",
        )
    closed_ready = [
        issue for issue in ready_issues if str(issue.get("state") or "OPEN").upper() == "CLOSED"
    ]
    if not closed_ready:
        return set(), ready_issues, _wf._MergedPRListOutcome()

    finalize_limit = self.config.dispatch.finalize_limit
    if finalize_limit <= 0:
        return set(), ready_issues, _wf._MergedPRListOutcome()

    # Try the cheap 500-window binding first; if the GraphQL-budget guard
    # refuses the call, fall back to per-issue search for all candidates.
    bound_issue_numbers: set[int] = set()
    mention_only_issue_numbers: set[int] = set()
    merged_pr_outcome = _wf._MergedPRListOutcome()
    try:
        merged_prs = self.gh.merged_pr_list()
    except GitHubError as exc:
        merged_pr_outcome = _wf._MergedPRListOutcome([], exc, called=True)
        merged_prs = []
    else:
        merged_pr_outcome = _wf._MergedPRListOutcome(merged_prs, called=True)
    for pr in merged_prs:
        if str(pr.get("state") or "").upper() != "MERGED":
            continue
        # Issue #1229 scoping decision: this call site is deliberately NOT
        # threaded through branch_issue_validator. ``bound_issue_numbers``
        # only gates which CLOSED ready issues have a merged PR binding
        # them (so they are not stripped as closed-unmerged); no
        # issue-label transition or state escalation keys off it. A stale
        # branch-name binding can at worst add a wrong number to
        # ``bound_issue_numbers``, causing a missed label-strip on an
        # already-CLOSED (terminal) issue that the next pass recovers --
        # not the "escalate the wrong issue" failure class the validator
        # exists to prevent. (Contrast ``detect_mergequeue_wedged``, whose
        # ``issue_number`` DOES drive ``_escalate_issue`` and is
        # validator-threaded.)
        bound = _wf.linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=self.config.dispatch.branch_prefix,
        )
        if bound is not None:
            bound_issue_numbers.add(bound)
        # isCrossRepository describes the PR's own head-branch provenance
        # (fork vs. same-repo). It cannot fully guard a cross-repo mention
        # collision, but it does guard the common case of a fork PR's text
        # being trusted at all.
        if pr.get("isCrossRepository") is False:
            for mentioned in _wf.issue_numbers_mentioned_by_pr(pr):
                mention_only_issue_numbers.add(mentioned)

    # Mention-only references are advisory; they are not a binding, but
    # they also must not be stripped as "unmerged" — dispatch() will flag
    # them for a human decision.
    mention_only_issue_numbers -= bound_issue_numbers

    # Only unbound closed issues are candidates for per-issue search or strip.
    unbound_issues = [
        issue for issue in closed_ready if int(issue["number"]) not in bound_issue_numbers
    ]

    def _finalization_order(issue: dict[str, Any]) -> tuple[str, int]:
        return (str(issue.get("createdAt") or ""), int(issue["number"]))

    # Slice BEFORE any per-issue lookup so a large backlog cannot exhaust
    # the GitHub Search API bucket in a single pass.
    candidates = sorted(unbound_issues, key=_finalization_order)[:finalize_limit]

    issue_pr_map: dict[int, list[dict[str, Any]]] = {}
    closed_unmerged_ready_issues: set[int] = set()
    consecutive_failures = 0
    for issue in candidates:
        if consecutive_failures >= 3:
            break
        issue_number = int(issue["number"])
        merged_prs = self.gh.merged_prs_for_issue(
            issue_number,
            self.config.dispatch.branch_prefix,
        )
        if not getattr(merged_prs, "ok", True):
            consecutive_failures += 1
            continue
        consecutive_failures = 0
        if merged_prs:
            issue_pr_map[issue_number] = list(merged_prs)
        elif issue_number not in mention_only_issue_numbers:
            # Confirmed closed ready issue with no merged PR binding it.
            closed_unmerged_ready_issues.add(issue_number)

    # Persist state first, then apply labels outside the lock.
    if issue_pr_map or closed_unmerged_ready_issues:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            for issue_number, prs in issue_pr_map.items():
                issue_key = str(issue_number)
                issue_entry = state["issues"].get(issue_key, {})
                state["issues"][issue_key] = {
                    **issue_entry,
                    "number": issue_number,
                    "status": "closed",
                }
                for pr in prs:
                    pr_number = int(pr["number"])
                    pr_key = str(pr_number)
                    pr_entry = state["prs"].get(pr_key, {})
                    _new_pr_state = {
                        **pr_entry,
                        "number": pr_number,
                        "status": "merged",
                        "merged": True,
                        "issue_number": issue_number,
                    }
                    # Issue #747: stamp ``merged_at`` only on a genuine
                    # non-merged -> merged transition so the original
                    # observation time is preserved across re-finalization
                    # passes and existing entries are never back-dated.
                    if pr_entry.get("status") != "merged":
                        _new_pr_state["merged_at"] = _wf.utc_now()
                    state["prs"][pr_key] = _new_pr_state
            if issue_pr_map:
                state = self._record_event(
                    state,
                    # event-consumer: audit-only -- records the PR-status "merged"
                    # finalization already applied inline above; no separate consumer needed
                    "finalize_externally_merged",
                    {
                        "issue_numbers": sorted(issue_pr_map.keys()),
                        "pr_numbers": sorted(
                            {int(pr["number"]) for prs in issue_pr_map.values() for pr in prs}
                        ),
                    },
                )
            for issue_number in closed_unmerged_ready_issues:
                issue_key = str(issue_number)
                issue_entry = state["issues"].get(issue_key, {})
                state["issues"][issue_key] = {
                    **issue_entry,
                    "number": issue_number,
                    "status": "closed",
                }
            if closed_unmerged_ready_issues:
                state = self._record_event(
                    state,
                    "dispatch_closed_unmerged_ready_stripped",
                    {"issue_numbers": sorted(closed_unmerged_ready_issues)},
                )
            _wf.save_state(self.paths.state_file, state)

    for issue_number in issue_pr_map:
        _wf.transition(self.gh, self.config.labels, issue_number, "merged")
        self.gh.close_issue(issue_number)

    for issue_number in closed_unmerged_ready_issues:
        _wf.transition(self.gh, self.config.labels, issue_number, "closed_unmerged")

    finalized: set[int] = set(issue_pr_map.keys())
    removed = finalized | closed_unmerged_ready_issues
    remaining = [issue for issue in ready_issues if int(issue["number"]) not in removed]
    return finalized, remaining, merged_pr_outcome
