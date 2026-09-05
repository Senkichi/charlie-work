"""Rework-dispatch delegate for ``OrchestratorApp``.

Track 2 Phase B leaf L01 batch 2 (issue #1645, parent #1632, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from charlie_work.adapters import (
    SessionDispatchResult,
    SessionRequest,
    manifest_adapter_label,
    write_session_results,
)
from charlie_work.config import (
    DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS,
    PRE_LAUNCH_BLOCKED_ENVIRONMENT_FAILURE_KINDS,
)
from charlie_work.github import GitHubError
from charlie_work.labels import TransitionOutcome
from charlie_work.state import StateLockBusy
from charlie_work.worktree import worktree_ahead_of_sha
import charlie_work.workflow as _wf


def _dispatch_rework_impl(
    self,
    limit: int | None = None,
    *,
    only_issues: str | None = None,
    stalled_entries: list[dict[str, int]] | None = None,
) -> _wf.CommandResult:
    """Dispatch rework workers for issues in needs-rework state with open PRs.

    This is only for non-manual adapters. The manual adapter's human-paste
    path remains intact.

    Candidate selection is STATE-DRIVEN: an issue is a rework candidate iff
    state["issues"][n]["status"] == "rework_requested" and it has an open PR.
    The label is used for display only and never for selection.
    """
    if self.config.worker.harness == "manual":
        return _wf.CommandResult(
            True,
            "rework dispatch skipped for manual adapter",
            {"adapter": "manual", "selected_count": 0},
        )

    sessions_dir = self._layout.sessions_dir
    # Issue #1393: clean up stranded .json.tmp session sidecar files from
    # interrupted atomic writes (e.g. a watchdog kill during a launch-
    # refusal write) before this pass writes new ones.
    _wf.cleanup_stale_session_tmp_files(sessions_dir)
    # Unconditional stall reaper call, matching dispatch()'s — previously this
    # only ran when max_concurrent_sessions > 0 via the governor. Skipped
    # only when the caller (loop()) already ran the sweep this pass and
    # handed its result down — see dispatch_rework()'s docstring.
    if stalled_entries is None:
        _wf._detect_and_handle_stalled_sessions(
            sessions_dir,
            self.paths.state_file,
            self.config,
            write_gate=self.write_gate,
        )

    # Note: orphaned-worker detection is intentionally NOT re-run here.
    # loop() already runs _detect_and_handle_orphaned_workers once per pass
    # (with the review callback needed to route head-advanced findings to
    # review). Re-running it here produced duplicate drift events (#457).

    # Load state to find rework_requested issues (state-driven selection)
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)

    operator_claimed = _wf.operator_claimed_issues(state)
    operator_claimed_skipped: list[int] = []

    # Build the open-PR index BEFORE any per-issue gh.issue_view() fetch.
    # pr_list() returns only OPEN PRs by contract (--state open), so an
    # issue whose PR is closed-unmerged (or that never had one) is absent
    # from pr_by_issue. Issue #558: without this ordering, the candidate
    # scan below called gh.issue_view() for every rework_requested issue
    # every pass -- including issues whose PR closed-unmerged between
    # reconcile sweeps -- a permanent per-pass GitHub fetch with no
    # terminal exit (the exact slow-cost-spiral shape #556/#558 exist to
    # eliminate). Filtering by open PR first cuts the fetch to genuine
    # candidates only. pr_list() is cached within a pass, so calling it
    # here vs. later is the same GitHub call.
    prs = self.gh.pr_list()
    pr_by_issue: dict[int, dict[str, Any]] = {}
    branch_validator = self._make_branch_issue_validator()
    for pr in prs:
        issue_number = _wf.linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=self.config.dispatch.branch_prefix,
            branch_issue_validator=branch_validator,
        )
        if issue_number is not None:
            # If multiple PRs link to the same issue, keep the lowest PR number
            if issue_number not in pr_by_issue or int(pr["number"]) < int(
                pr_by_issue[issue_number]["number"]
            ):
                pr_by_issue[issue_number] = pr

    # Find issues with rework_requested status. Only fetch the full issue
    # from GitHub for issues that actually have an open PR -- a
    # rework_requested issue with no open PR is not a launch candidate
    # (the PR was closed-unmerged or never existed), so the per-issue
    # gh.issue_view fetch is skipped entirely.
    import logging

    logger = logging.getLogger(__name__)
    rework_issues: list[dict[str, Any]] = []
    failed_issue_fetches: list[tuple[int, Exception]] = []
    for number, entry in state.get("issues", {}).items():
        if not isinstance(entry, dict):
            continue
        if entry.get("status") == "rework_requested":
            issue_number = int(number)
            # Issue #400: operator-claimed issues are not rework-dispatchable.
            if issue_number in operator_claimed:
                operator_claimed_skipped.append(issue_number)
                continue
            # Issue #558: skip the gh.issue_view fetch for issues with no
            # open PR -- not a rework candidate, and the fetch is the
            # permanent per-pass cost this gate exists to eliminate.
            if issue_number not in pr_by_issue:
                continue
            # Fetch the full issue from GitHub to get labels and other metadata
            try:
                full_issue = self.gh.issue_view(issue_number)
                rework_issues.append(full_issue)
            except GitHubError as exc:
                # Skip issues that can't be fetched (deleted, transient
                # outage, etc.), but record the degradation so a later
                # stall escalation has a reason (issue #939).
                failed_issue_fetches.append((issue_number, exc))
                continue

    if failed_issue_fetches:
        payload = _wf._build_rework_issue_fetch_skip_payload(failed_issue_fetches)
        logger.warning(
            "rework dispatch skipped fetching %d issue(s) this pass: %s",
            len(failed_issue_fetches),
            payload["reason"],
        )
        if not self.dry_run:
            try:
                with _wf.state_lock(self.paths.state_file):
                    event_state = _wf.load_state(self.paths.state_file)
                    event_state = self._record_event(
                        event_state,
                        "rework_issue_fetch_skipped",
                        payload,
                    )
                    _wf.save_state(self.paths.state_file, event_state)
            except (OSError, ValueError, StateLockBusy) as write_exc:
                # StateLockBusy is a RuntimeError, so it is not covered by the
                # two above. Without it here the *diagnostic* write can abort
                # the pass it was only meant to describe: it would propagate to
                # dispatch_rework's own `except StateLockBusy`, which defers the
                # whole call and discards the legitimate candidates already
                # scanned. #939 asked for observation without a control-flow
                # change; letting a best-effort write decide the pass outcome
                # is exactly the control-flow change it ruled out.
                logger.warning("could not record rework_issue_fetch_skipped: %s", write_exc)

    rework_limit = limit if limit is not None else self.config.dispatch.default_limit

    # Apply global concurrency governor cap
    gov = self._apply_concurrency_governor(rework_limit)
    rework_limit = gov.dispatch_limit

    # Apply provider throttle cooldown check
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        if _wf.is_throttled(state):
            throttled_until = state.get("throttled_until")
            # Return immediately with deferral reason
            data = {
                "adapter": self.config.worker.harness,
                "selected_count": 0,
                "deferred_reason": "provider_throttled",
                "throttled_until": throttled_until,
            }
            if gov.enabled or gov.fleet_enabled or gov.open_pr_enabled:
                data.update(gov.report_fields())
            return _wf.CommandResult(
                False,
                f"rework dispatch deferred: provider throttled until {throttled_until}",
                data,
            )

    # Dry-run: read-only planning — compute selection and would-be
    # SessionRequests, but skip all state writes, label transitions,
    # escalations, review routing, and worker launches. Mirrors
    # _dispatch_impl's dry-run branch. Without this, the fabricated
    # _dry_run_result objects (adapters.py, ok=True for every request)
    # are consumed as ground truth: issues marked "dispatched" with real
    # dispatched_at, redispatch_at counters advanced, orphan flags
    # cleared, and auto-escalation at the redispatch cap — all on zero
    # actual work (issue #616). The no-op-rework and worker-death
    # escalation paths also write state + transition GitHub labels, and
    # the review-routing path calls self.review() which writes state;
    # all are skipped here.
    if self.dry_run:
        dry_candidates = [issue for issue in rework_issues if int(issue["number"]) in pr_by_issue]

        # Head-check filtering (read-only): identify candidates that
        # would be routed to review or escalated, but do NOT perform the
        # routing or escalation — both write state and transition GitHub
        # labels.
        dry_head_check_state = _wf.load_state_locked(self.paths.state_file)
        dry_routed_to_review: list[int] = []
        dry_head_indeterminate: list[int] = []
        dry_no_op_rework_escalated: list[int] = []
        dry_worker_death_escalated: list[int] = []
        dry_blocked_environment_escalated: list[int] = []
        dry_filtered_candidates: list[dict[str, Any]] = []
        for issue in dry_candidates:
            issue_number = int(issue["number"])
            pr_data = pr_by_issue[issue_number]
            pr_number = int(pr_data["number"])
            live_head_sha = pr_data.get("headRefOid")
            pr_state = dry_head_check_state.get("prs", {}).get(str(pr_number), {})
            reviewed_head_sha = pr_state.get("reviewed_head_sha")

            # Issue #1393: mirror the live path's blocked-environment cap
            # check (read-only here).
            dry_issue_entry_pre = dry_head_check_state.get("issues", {}).get(str(issue_number), {})
            if isinstance(dry_issue_entry_pre, dict):
                dry_prior_blocked = _wf._windowed_blocked_environment_at(
                    dry_issue_entry_pre,
                    window_minutes=self.config.watchdog.redispatch_window_minutes,
                )
                if len(dry_prior_blocked) >= self.config.watchdog.max_auto_redispatch:
                    dry_blocked_environment_escalated.append(issue_number)
                    continue

            if not reviewed_head_sha:
                dry_filtered_candidates.append(issue)
                continue
            if not live_head_sha:
                dry_head_indeterminate.append(issue_number)
                continue
            if live_head_sha == reviewed_head_sha:
                issue_entry = dry_head_check_state.get("issues", {}).get(str(issue_number), {})
                if isinstance(issue_entry, dict):
                    prior_redispatch = _wf._windowed_redispatch_at(
                        issue_entry,
                        window_minutes=self.config.watchdog.redispatch_window_minutes,
                    )
                    prior_deaths = _wf._windowed_worker_death_at(
                        issue_entry,
                        window_minutes=self.config.watchdog.redispatch_window_minutes,
                    )
                    no_op_count = max(0, len(prior_redispatch) - len(prior_deaths))
                    if no_op_count >= self.config.watchdog.max_auto_redispatch:
                        dry_no_op_rework_escalated.append(issue_number)
                        continue
                    if len(prior_deaths) >= self.config.watchdog.max_auto_redispatch:
                        dry_worker_death_escalated.append(issue_number)
                        continue
                dry_filtered_candidates.append(issue)
                continue

            reviewed_patch_id = pr_state.get("reviewed_patch_id")
            diff = self.gh.pr_diff(pr_number)
            live_patch_id = _wf._calculate_patch_id(diff) if diff else ""
            if not reviewed_patch_id or not live_patch_id:
                dry_head_indeterminate.append(issue_number)
                continue
            if live_patch_id == reviewed_patch_id:
                dry_filtered_candidates.append(issue)
                continue
            # Issue #1349: a patch-id-advanced head on a CONFLICTING/DIRTY
            # PR is still a legitimate launch candidate -- the head advance
            # did not resolve the conflict the rework was requested for.
            # Mirror the live path's conflict check (read-only here) via
            # the shared helper so the dry-run report does not misreport
            # these as routed_to_review.
            if self._rework_candidate_conflict_blocked(pr_data, pr_number):
                dry_filtered_candidates.append(issue)
                continue
            dry_routed_to_review.append(issue_number)

        dry_candidates = dry_filtered_candidates

        # Apply only_issues filter and concurrency cap (read-only).
        # Issue #1014 (mirroring #1005 in the fresh-dispatch path): compute
        # deferred_by_concurrency uniformly across both the only_issues and
        # automatic branches -- the automatic branch used to unconditionally
        # report [] even when the concurrency governor dropped candidates,
        # making a saturated governor indistinguishable from an empty
        # backlog. Shares the selection helper with the live branch below
        # so the two cannot drift.
        (
            dry_selected,
            dry_deferred_by_concurrency_full,
            dry_deferred_by_concurrency,
            dry_deferred_by_concurrency_count,
        ) = _wf._select_rework_candidates(dry_candidates, rework_limit, only_issues=only_issues)

        # Compute would-be SessionRequests without state mutation, label
        # transitions, or worker launches. Rework-prompt re-rendering
        # (which writes files) is skipped; the existing on-disk prompt
        # is used as-is for the planning report.
        dry_session_requests: list[SessionRequest] = []
        dry_skipped_issue_numbers: list[int] = []
        dry_missing_prompt_failures: dict[int, str] = {}
        dry_rescue_issue_numbers: set[int] = set()
        for issue in dry_selected:
            issue_number = int(issue["number"])
            full_issue = self.gh.issue_view(issue_number)
            pr = pr_by_issue[issue_number]
            pr_number = int(pr["number"])
            branch_name = pr.get("headRefName", "")
            rework_prompt_path = self.paths.prs / f"pr-{pr_number}" / "rework-prompt.md"
            if not rework_prompt_path.exists():
                dry_skipped_issue_numbers.append(issue_number)
                dry_missing_prompt_failures[issue_number] = (
                    f"missing rework prompt: {rework_prompt_path}"
                )
                continue
            pr_state_for_rescue = dry_head_check_state.get("prs", {}).get(str(pr_number), {})
            if pr_state_for_rescue.get("rescue_attempted"):
                dry_rescue_issue_numbers.add(issue_number)
            dry_session_requests.append(
                SessionRequest(
                    issue_number=issue_number,
                    issue_title=str(full_issue.get("title") or ""),
                    prompt_path=rework_prompt_path,
                    branch_name=branch_name,
                    rework=True,
                )
            )

        data = {
            "adapter": self.config.worker.harness,
            "selected_count": len(dry_session_requests),
            "attempted_count": len(dry_session_requests),
            "failed_count": 0,
            "failures": _wf._build_failure_map(
                [],
                set(),
                dry_deferred_by_concurrency_full,
                rework_limit,
                extra_failures=dry_missing_prompt_failures,
            ),
            "deferred_by_concurrency": dry_deferred_by_concurrency,
            "deferred_by_concurrency_count": dry_deferred_by_concurrency_count,
            "skipped_issue_numbers": sorted(dry_skipped_issue_numbers),
            "sessions": [asdict(request) for request in dry_session_requests],
            "dispatch_results": [],
            "routed_to_review": sorted(dry_routed_to_review),
            "skipped_head_indeterminate": sorted(dry_head_indeterminate),
            "operator_claimed_skipped": sorted(operator_claimed_skipped),
            "no_op_rework_escalated": sorted(dry_no_op_rework_escalated),
            "worker_death_escalated": sorted(dry_worker_death_escalated),
            "blocked_environment_escalated": sorted(dry_blocked_environment_escalated),
            "rescue_issue_numbers": sorted(dry_rescue_issue_numbers),
        }
        if gov.enabled or gov.fleet_enabled or gov.open_pr_enabled:
            data.update(gov.report_fields())
        return _wf.CommandResult(
            True,
            f"dry-run: would dispatch rework for {len(dry_session_requests)} issue(s)",
            data,
        )

    # pr_list() returns only open PRs by contract (--state open); its field
    # list does not include "state", so no per-PR state check here.
    # rework_issues already contains only issues with an open PR (the
    # fetch loop above skipped issues absent from pr_by_issue), so this
    # filter is now a no-op kept for clarity/defense-in-depth.
    candidates = [issue for issue in rework_issues if int(issue["number"]) in pr_by_issue]

    # Issue #339: a rework worker relaunched onto a PR whose rework was
    # already pushed (PR head moved past the last request_changes verdict)
    # finds nothing to do, idles, and is watchdog-reaped — burning a
    # session and a concurrency slot. Filter those candidates out here,
    # before any dispatch_pending claim, and route them to the review
    # lane instead. A sync-merge-only head advance (patch-id unchanged)
    # is NOT treated as "already reworked" — the same patch still needs a
    # genuine rework cycle, so it remains a legitimate launch candidate.
    head_check_state = _wf.load_state_locked(self.paths.state_file)
    routed_to_review: list[int] = []
    head_indeterminate: list[int] = []
    no_op_rework_escalated: list[int] = []
    worker_death_escalated: list[int] = []
    # Issue #1239: issues salvaged out of the death-loop gate (stranded
    # commits pushed) and routed to review instead of escalated.
    salvaged_to_review: list[int] = []
    blocked_environment_escalated: list[int] = []
    filtered_candidates = []
    for issue in candidates:
        issue_number = int(issue["number"])
        pr_data = pr_by_issue[issue_number]
        pr_number = int(pr_data["number"])
        live_head_sha = pr_data.get("headRefOid")
        pr_state = head_check_state.get("prs", {}).get(str(pr_number), {})
        reviewed_head_sha = pr_state.get("reviewed_head_sha")

        # Issue #1393: if the previous dispatch was blocked by a
        # pre-launch environment conflict (e.g. worktree_foreign_writer)
        # and the cap is already exhausted, escalate here instead of
        # attempting another launch that will fail identically.  This
        # is a safety net for when the dispatch failure path's
        # escalation didn't stick (race/crash between the escalation
        # and the state write), parallel to the no_op/death checks
        # below.
        issue_entry_pre = head_check_state.get("issues", {}).get(str(issue_number), {})
        if isinstance(issue_entry_pre, dict):
            prior_blocked = _wf._windowed_blocked_environment_at(
                issue_entry_pre,
                window_minutes=self.config.watchdog.redispatch_window_minutes,
            )
            if len(prior_blocked) >= self.config.watchdog.max_auto_redispatch:
                # Issue #1423: before escalating a stuck blocked-environment
                # cap, attempt to reap an idle foreign writer from the
                # worktree. If the writer has gone idle since the last
                # blocked pass, reaping it clears the path for dispatch
                # instead of escalating a zombie. The marker is read from
                # the worktree path derived from the PR's head ref.
                #
                # Review finding: bound the number of auto-reaps per issue
                # before falling back to escalation (see the fresh-dispatch
                # site for the full rationale). The cap is checked against
                # the snapshot entry here; the reap timestamp is appended
                # inside the lock below.
                max_reaps = self.config.watchdog.max_foreign_writer_reaps
                prior_reaps = _wf._windowed_foreign_writer_reaps(
                    issue_entry_pre,
                    window_minutes=self.config.watchdog.redispatch_window_minutes,
                )
                reap_cap_ok = len(prior_reaps) < max_reaps
                branch_pre = str(pr_data.get("headRefName") or "")
                if branch_pre and reap_cap_ok:
                    wt_path_pre = _wf.worktree_path_for_branch(
                        self.repo_root, branch_pre, self._layout.worktrees
                    )
                    marker_pre = _wf.read_worktree_marker(wt_path_pre)
                    # Issue #1443: the reap-candidate guards (operator,
                    # stale-pid, own-live-session) and the pid are enforced
                    # inside ``_reap_idle_foreign_writer`` (single point of
                    # enforcement) so this pre-filter cannot drop one and
                    # reap one of our own idle workers as
                    # ``foreign_writer_reaped``. The pid for the audit event
                    # is read from the marker (same source of truth as the
                    # kill path).
                    if marker_pre is not None and _wf._reap_idle_foreign_writer(
                        wt_path_pre,
                        marker_pre,
                        self.config,
                        sessions_dir,
                        state_file=self.paths.state_file,
                        issue_number=issue_number,
                    ):
                        # Writer reaped: clear the counter so the issue
                        # proceeds as a legitimate candidate, and record
                        # the reap so a persistently-blocked worktree
                        # eventually escalates instead of looping.
                        with _wf.state_lock(self.paths.state_file):
                            state = _wf.load_state(self.paths.state_file)
                            issue_entry = state["issues"].get(str(issue_number), {})
                            if isinstance(issue_entry, dict):
                                issue_entry["blocked_environment_at"] = []
                                existing_reaps = _wf._windowed_foreign_writer_reaps(
                                    issue_entry,
                                    window_minutes=self.config.watchdog.redispatch_window_minutes,
                                )
                                issue_entry["foreign_writer_reaps"] = existing_reaps + [
                                    datetime.now(UTC).isoformat().replace("+00:00", "Z")
                                ]
                                state["issues"][str(issue_number)] = issue_entry
                                state = _wf.append_event(  # event-consumer: audit-only -- records a pre-escalation foreign-writer reap (issue #1423) already enforced by the blocked_environment_at reset and foreign_writer_reaps counter; consumed by tests/test_charlie_work.py regression tests.
                                    state,
                                    "dispatch_blocked_environment_reaped",
                                    {
                                        "issue_number": issue_number,
                                        "pid": marker_pre.get("pid"),
                                        "blocked_environment_count": len(prior_blocked),
                                        "foreign_writer_reap_count": len(existing_reaps) + 1,
                                    },
                                    state_path=self.paths.state_file,
                                )
                                _wf.save_state(self.paths.state_file, state)
                        filtered_candidates.append(issue)
                        continue
                blocked_environment_escalated.append(issue_number)
                continue

        if not reviewed_head_sha:
            # No recorded request_changes head to compare against —
            # nothing to disambiguate; proceed as a legitimate candidate.
            filtered_candidates.append(issue)
            continue
        if not live_head_sha:
            # Live head cannot be determined — fail closed against a
            # wasted launch, but don't strand the issue: status is left
            # untouched so the next pass retries with fresh PR data.
            head_indeterminate.append(issue_number)
            continue
        if live_head_sha == reviewed_head_sha:
            # Head hasn't moved since request_changes. Check if previous
            # rework attempts for this head already exhausted the
            # redispatch cap — if so, escalate immediately instead of
            # dispatching another worker that will also produce no
            # changes. This is a safety net for cases where the restore
            # path's escalation didn't stick (race/crash between the
            # restore and the state write).
            #
            # Issue #1134: a worker that died before pushing leaves the
            # PR head unchanged, but that is NOT a no-op — the worker
            # may have completed its work and died mid-push with
            # salvageable stranded commits.  The orphan sweep records
            # each death in ``worker_death_at``; here we subtract death
            # redispatches from the total to get the genuine no-op
            # count.  A death-loop still escalates, but with
            # ``worker_death_loop`` (triage: "check the worktree for
            # stranded work") instead of ``no_op_rework_cap_exceeded``
            # (triage: "worker is spinning").
            issue_entry = head_check_state.get("issues", {}).get(str(issue_number), {})
            if isinstance(issue_entry, dict):
                prior_redispatch = _wf._windowed_redispatch_at(
                    issue_entry,
                    window_minutes=self.config.watchdog.redispatch_window_minutes,
                )
                prior_deaths = _wf._windowed_worker_death_at(
                    issue_entry,
                    window_minutes=self.config.watchdog.redispatch_window_minutes,
                )
                no_op_count = max(0, len(prior_redispatch) - len(prior_deaths))
                if no_op_count >= self.config.watchdog.max_auto_redispatch:
                    no_op_rework_escalated.append(issue_number)
                    continue
                if len(prior_deaths) >= self.config.watchdog.max_auto_redispatch:
                    # Issue #1239: before escalating a death-loop, attempt
                    # to salvage-push stranded commits from the dead
                    # worker's worktree — the same sanctioned-git path the
                    # fresh-dispatch lane uses (#1248).  A successful push
                    # means the worker completed the rework and died at the
                    # final push step; the PR head moves past the
                    # request_changes verdict and the next pass routes to
                    # review (packet regeneration supersedes the verdict),
                    # so this death does NOT count toward the death-loop
                    # cap.  Only a death that produced NO pushable commit
                    # escalates.  ``salvage_push_stranded_commits`` runs
                    # ls-remote before pushing, never force-pushes, and is
                    # fast-forward only — never trust the sidecar's
                    # ``push_succeeded`` as proof of a push.
                    if self._salvage_rework_stranded_commits(issue_number, pr_data, issue_entry):
                        salvaged_to_review.append(issue_number)
                        continue
                    worker_death_escalated.append(issue_number)
                    continue
            filtered_candidates.append(issue)
            continue

        # Head moved since the request_changes verdict. Disambiguate a
        # real content push from a sync-merge-only advance using the same
        # patch-id helper the janitor's no-op-rework gate relies on
        # (issue #222).
        reviewed_patch_id = pr_state.get("reviewed_patch_id")
        diff = self.gh.pr_diff(pr_number)
        live_patch_id = _wf._calculate_patch_id(diff) if diff else ""
        if not reviewed_patch_id or not live_patch_id:
            # Can't establish content identity (no recorded baseline, or
            # the diff fetch itself failed) — fail closed rather than
            # guess; retry next pass instead of stranding the issue.
            head_indeterminate.append(issue_number)
            continue
        if live_patch_id == reviewed_patch_id:
            # Sync-merge only: the patch itself is unchanged, so the
            # rework is still genuinely outstanding.
            filtered_candidates.append(issue)
            continue

        # Issue #1349: a patch-id-advanced head does NOT mean the
        # rework was already pushed when the PR is still
        # CONFLICTING/DIRTY. The head advance was sync-merges (or
        # conflict-laden merges) that changed the patch-id without
        # resolving the conflict the rework was requested for --
        # whatever was pushed since the last verdict did NOT resolve
        # the conflict, so the rework is still outstanding. Routing
        # such a PR to review() just bounces off the janitor gate's
        # conflict check back to rework_requested, deadlocking the
        # issue between dispatch_rework and review() forever (the
        # only exit being the #765 stall escalation to a human, not a
        # dispatch). Keep it as a legitimate launch candidate so a
        # conflict-rework worker actually dispatches, subject to
        # max_conflict_rework_attempts via the janitor gate on the
        # next settled head change.
        if self._rework_candidate_conflict_blocked(pr_data, pr_number):
            filtered_candidates.append(issue)
            continue

        routed_to_review.append(issue_number)

    candidates = filtered_candidates
    # A content change was detected above, but review() may itself fail to
    # produce a packet (deterministic janitor gate: conflicting/draft/red
    # CI — see review()'s early-return before any packet/label write).
    # Only issues review() actually routed get reported as routed_to_review;
    # the rest keep their rework_requested status and are retried next pass
    # (issue #339 finding 1: never report a routing that didn't happen —
    # doing so desyncs state.json from GitHub labels/PR state with no
    # automated recovery path).
    confirmed_routed_to_review: list[int] = []
    review_blocked_retry: list[int] = []
    for routed_issue_number in routed_to_review:
        routed_pr_number = int(pr_by_issue[routed_issue_number]["number"])
        reviewed_head_sha_before = (
            head_check_state.get("prs", {}).get(str(routed_pr_number), {}).get("reviewed_head_sha")
        )
        routed, _review_result = self._route_rework_candidate_to_review(
            routed_issue_number, routed_pr_number, reviewed_head_sha_before
        )
        if routed:
            confirmed_routed_to_review.append(routed_issue_number)
        else:
            review_blocked_retry.append(routed_issue_number)
    routed_to_review = confirmed_routed_to_review

    # Issue #1239: route salvaged death-loop issues to review.  The
    # salvage push already advanced the PR head past the request_changes
    # verdict, so review() generates a fresh packet that supersedes the
    # stale verdict — the same machinery the head-moved branch above uses.
    # A salvage that review() blocks (deterministic janitor gate) stays
    # rework_requested for the next pass, same as a blocked head-moved
    # routing; the death was NOT counted, so the death-loop cap is not
    # consumed.
    salvaged_confirmed: list[int] = []
    salvaged_blocked: list[int] = []
    for salvaged_issue_number in salvaged_to_review:
        salvaged_pr_number = int(pr_by_issue[salvaged_issue_number]["number"])
        reviewed_head_sha_before = (
            head_check_state.get("prs", {})
            .get(str(salvaged_pr_number), {})
            .get("reviewed_head_sha")
        )
        routed, _review_result = self._route_rework_candidate_to_review(
            salvaged_issue_number, salvaged_pr_number, reviewed_head_sha_before
        )
        if routed:
            salvaged_confirmed.append(salvaged_issue_number)
        else:
            salvaged_blocked.append(salvaged_issue_number)
    routed_to_review.extend(salvaged_confirmed)
    review_blocked_retry.extend(salvaged_blocked)

    # Escalate no-op rework issues that have exhausted the redispatch cap
    # without the PR head ever advancing. Each of these would have burned
    # another worker session on an unchanged diff.
    if no_op_rework_escalated:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            for issue_number in no_op_rework_escalated:
                entry = state.get("issues", {}).get(str(issue_number), {})
                if not isinstance(entry, dict):
                    entry = {}
                current_status = entry.get("status")
                if current_status == "escalated":
                    continue
                redispatch_at = _wf._windowed_redispatch_at(
                    entry,
                    window_minutes=self.config.watchdog.redispatch_window_minutes,
                ) + [datetime.now(UTC).isoformat().replace("+00:00", "Z")]
                # Issue #783: no-op rework redispatch cap is a process
                # failure, not a judgment call -- mechanical.
                state = _wf._escalate_issue(
                    state,
                    issue_number,
                    reason="redispatch_cap_exceeded",
                    reason_class="mechanical",
                    issue_extra={
                        "redispatch_at": redispatch_at,
                        "dispatched_at": None,
                    },
                )
                state = _wf.append_event(
                    state,
                    "session_failed_escalated",
                    {
                        "issue_number": issue_number,
                        "previous_status": "rework_requested",
                        "reason": "no_op_rework_cap_exceeded",
                        "redispatch_count": len(redispatch_at),
                    },
                    state_path=self.paths.state_file,
                )
            _wf.save_state(self.paths.state_file, state)
        for issue_number in no_op_rework_escalated:
            _wf.transition(
                self.gh,
                self.config.labels,
                issue_number,
                _wf._escalation_edge("redispatch_escalated", "mechanical"),
            )

    # Issue #1134: escalate worker-death loops separately from no-op
    # rework loops.  A death loop means the worker keeps dying before
    # pushing — the work may be complete but stranded in the worktree.
    # The operator triage for ``worker_death_loop`` is "check the
    # worktree for stranded commits," not "worker is spinning."
    # Pre-compute stranded-commits counts outside the state lock because
    # the git probe touches the filesystem.
    if worker_death_escalated:
        stranded_counts: dict[int, int | None] = {}
        for issue_number in worker_death_escalated:
            pr_data = pr_by_issue.get(issue_number)
            if pr_data is None:
                stranded_counts[issue_number] = None
                continue
            live_head = pr_data.get("headRefOid")
            if not live_head:
                stranded_counts[issue_number] = None
                continue
            issue_entry = head_check_state.get("issues", {}).get(str(issue_number), {})
            branch = issue_entry.get("branch_name") if isinstance(issue_entry, dict) else None
            if not branch:
                stranded_counts[issue_number] = None
                continue
            wt_path = _wf.worktree_path_for_branch(self.repo_root, branch, self._layout.worktrees)
            ahead, _err = worktree_ahead_of_sha(wt_path, live_head)
            stranded_counts[issue_number] = ahead
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            for issue_number in worker_death_escalated:
                entry = state.get("issues", {}).get(str(issue_number), {})
                if not isinstance(entry, dict):
                    entry = {}
                current_status = entry.get("status")
                if current_status == "escalated":
                    continue
                prior_deaths = _wf._windowed_worker_death_at(
                    entry,
                    window_minutes=self.config.watchdog.redispatch_window_minutes,
                )
                # Issue #783: worker death loop is a process failure,
                # not a judgment call -- mechanical.
                issue_extra: dict[str, Any] = {
                    "worker_death_at": prior_deaths,
                    "dispatched_at": None,
                }
                stranded = stranded_counts.get(issue_number)
                if stranded is not None:
                    issue_extra["stranded_commits"] = stranded
                state = _wf._escalate_issue(
                    state,
                    issue_number,
                    reason="worker_death_loop",
                    reason_class="mechanical",
                    issue_extra=issue_extra,
                )
                event_payload: dict[str, Any] = {
                    "issue_number": issue_number,
                    "previous_status": "rework_requested",
                    "reason": "worker_death_loop",
                    "worker_death_count": len(prior_deaths),
                }
                if stranded is not None:
                    event_payload["stranded_commits"] = stranded
                state = _wf.append_event(
                    state,
                    "session_failed_escalated",
                    event_payload,
                    state_path=self.paths.state_file,
                )
            _wf.save_state(self.paths.state_file, state)
        for issue_number in worker_death_escalated:
            _wf.transition(
                self.gh,
                self.config.labels,
                issue_number,
                _wf._escalation_edge("redispatch_escalated", "mechanical"),
            )

    # Issue #1393: escalate issues whose pre-launch environment has
    # blocked every dispatch attempt (e.g. a stale foreign worktree).
    # The operator triage for ``dispatch_blocked_environment`` is
    # "remove the stale checkout at <path>," not "worker quality cap
    # exceeded."  Parallel to the no_op/death escalation blocks above.
    if blocked_environment_escalated:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            for issue_number in blocked_environment_escalated:
                entry = state.get("issues", {}).get(str(issue_number), {})
                if not isinstance(entry, dict):
                    entry = {}
                current_status = entry.get("status")
                if current_status == "escalated":
                    continue
                prior_blocked = _wf._windowed_blocked_environment_at(
                    entry,
                    window_minutes=self.config.watchdog.redispatch_window_minutes,
                )
                state = _wf._escalate_issue(
                    state,
                    issue_number,
                    reason="dispatch_blocked_environment",
                    reason_class="mechanical",
                    issue_extra={
                        "blocked_environment_at": prior_blocked,
                        "dispatched_at": None,
                    },
                )
                state = _wf.append_event(
                    state,
                    "session_failed_escalated",
                    {
                        "issue_number": issue_number,
                        "previous_status": "rework_requested",
                        "reason": "dispatch_blocked_environment",
                        "blocked_environment_count": len(prior_blocked),
                    },
                    state_path=self.paths.state_file,
                )
            _wf.save_state(self.paths.state_file, state)
        for issue_number in blocked_environment_escalated:
            _wf.transition(
                self.gh,
                self.config.labels,
                issue_number,
                _wf._escalation_edge("redispatch_escalated", "mechanical"),
            )

    # Issue #1014 (mirroring #1005 in the fresh-dispatch path): compute
    # deferred_by_concurrency uniformly across both the only_issues and
    # automatic branches -- the automatic branch used to unconditionally
    # report [] even when the concurrency governor dropped candidates,
    # making a saturated governor indistinguishable from an empty backlog.
    # Shares the selection helper with the dry-run branch above so the two
    # cannot drift.
    (
        selected,
        deferred_by_concurrency_full,
        deferred_by_concurrency,
        deferred_by_concurrency_count,
    ) = _wf._select_rework_candidates(candidates, rework_limit, only_issues=only_issues)

    if not selected:
        data = {
            "adapter": self.config.worker.harness,
            "selected_count": 0,
            "failures": _wf._build_failure_map(
                [], set(), deferred_by_concurrency_full, rework_limit
            ),
            "deferred_by_concurrency": deferred_by_concurrency,
            "deferred_by_concurrency_count": deferred_by_concurrency_count,
            "routed_to_review": sorted(routed_to_review),
            "skipped_head_indeterminate": sorted(head_indeterminate),
            "review_blocked_retry": sorted(review_blocked_retry),
            "operator_claimed_skipped": sorted(operator_claimed_skipped),
            "no_op_rework_escalated": sorted(no_op_rework_escalated),
            "worker_death_escalated": sorted(worker_death_escalated),
            "salvaged_to_review": sorted(salvaged_to_review),
            "blocked_environment_escalated": sorted(blocked_environment_escalated),
        }
        if gov.enabled or gov.fleet_enabled or gov.open_pr_enabled:
            data.update(gov.report_fields())
        return _wf.CommandResult(
            True,
            "no rework candidates found",
            data,
        )

    # First lock: claim issues by marking them as dispatch_pending
    selected_issue_numbers: list[int] = []
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        # Filter out issues whose PR is in escalated state (rework cap exhausted)
        selected = [
            issue
            for issue in selected
            if state["prs"].get(str(pr_by_issue[int(issue["number"])]["number"]), {}).get("status")
            != "escalated"
        ]
        live_dispatched = set()
        for number, entry in state.get("issues", {}).items():
            if not isinstance(entry, dict):
                continue
            status = entry.get("status")
            if status == "dispatched":
                live_dispatched.add(int(number))
            elif status == "dispatch_pending" and not _wf.is_claim_stale(
                entry.get("dispatch_pending_at")
            ):
                live_dispatched.add(int(number))
        # Filter out already-dispatched issues
        selected = [issue for issue in selected if int(issue["number"]) not in live_dispatched]
        selected_issue_numbers = [int(issue["number"]) for issue in selected]
        # Mark selected issues as "dispatch_pending"
        for issue_number in selected_issue_numbers:
            entry = {
                **state["issues"].get(str(issue_number), {}),
                "number": issue_number,
                "status": "dispatch_pending",
                "dispatch_pending_at": _wf.utc_now(),
            }
            # A fresh dispatch supersedes any previous orphan flag.
            entry.pop("orphan_flagged_at", None)
            entry.pop("orphan_drift_fingerprint", None)
            entry.pop("orphan_drift_at", None)
            state["issues"][str(issue_number)] = entry
        _wf.save_state(self.paths.state_file, state)

    if not selected_issue_numbers:
        data = {
            "adapter": self.config.worker.harness,
            "selected_count": 0,
            "failures": _wf._build_failure_map(
                [], set(), deferred_by_concurrency_full, rework_limit
            ),
            "deferred_by_concurrency": deferred_by_concurrency,
            "deferred_by_concurrency_count": deferred_by_concurrency_count,
            "routed_to_review": sorted(routed_to_review),
            "skipped_head_indeterminate": sorted(head_indeterminate),
            "review_blocked_retry": sorted(review_blocked_retry),
            "no_op_rework_escalated": sorted(no_op_rework_escalated),
            "worker_death_escalated": sorted(worker_death_escalated),
            "salvaged_to_review": sorted(salvaged_to_review),
            "blocked_environment_escalated": sorted(blocked_environment_escalated),
        }
        if gov.enabled or gov.fleet_enabled or gov.open_pr_enabled:
            data.update(gov.report_fields())
        return _wf.CommandResult(
            True,
            "all rework candidates already dispatched",
            data,
        )

    # Representative PR for the aggregate dispatch_rework event. The events table
    # has a single pr_number column, so the first selected issue's open PR is used
    # as the indexed representative (mirroring the singular pr_number in
    # rework_already_pushed).
    first_selected_issue_number = selected_issue_numbers[0]
    first_rework_pr_number = (
        int(pr_by_issue[first_selected_issue_number]["number"])
        if first_selected_issue_number in pr_by_issue
        else None
    )

    # Do all network calls, file writes, and worker launches outside the lock
    session_requests: list[SessionRequest] = []
    full_issues: dict[int, dict[str, Any]] = {}
    skipped_issue_numbers: list[int] = []
    missing_prompt_failures: dict[int, str] = {}
    # Rescue tier (issue #555): an issue is rescue-marked when the rescue
    # interception sites (record_review / _route_janitor_gate_failure_to_
    # rework) already stamped `rescue_attempted` on its PR record instead
    # of escalating. Those issues must launch via the claude-code adapter
    # pinned to `rescue.worker_model`, regardless of the primary
    # configured `worker.harness` — tracked separately here so the SAME
    # candidate-selection/session-request/state-bookkeeping code below
    # handles them, only the final dispatch_sessions() call differs.
    #
    # Always loaded (never gated on self.config.rescue.enabled): routing
    # of a PR that already carries the durable marker must not depend on
    # the current config value. If an operator flips rescue.enabled off
    # while a rescue rework is queued (rework_requested + marker set,
    # worker not yet launched), it must still launch via the rescue
    # adapter/model -- enabled only gates NEW rescue entry at the three
    # cap sites, never routing of an already-marked PR.
    rescue_issue_numbers: set[int] = set()
    rescue_state_snapshot = _wf.load_state_locked(self.paths.state_file)
    for issue_number in selected_issue_numbers:
        full_issue = self.gh.issue_view(issue_number)
        full_issues[issue_number] = full_issue
        pr = pr_by_issue[issue_number]
        pr_number = int(pr["number"])
        # Use the existing PR branch instead of creating a new one
        branch_name = pr.get("headRefName", "")
        # Use the rework prompt from the PR directory
        rework_prompt_path = self.paths.prs / f"pr-{pr_number}" / "rework-prompt.md"
        if not rework_prompt_path.exists():
            # Skip if rework prompt doesn't exist — record as rework_requested
            # to release the claim and allow retry (issue #116)
            skipped_issue_numbers.append(issue_number)
            missing_prompt_failures[issue_number] = f"missing rework prompt: {rework_prompt_path}"
            continue
        # The brief on disk is authoritative and dispatch_rework reads it
        # verbatim, so it must be checked for staleness before dispatch.
        # It can go stale along two independent axes:
        #
        #   1. The verdict content changed under it — an operator
        #      hand-editing review-decision.json is the #510 case, and
        #      issue #632 added the mtime comparison that catches it.
        #   2. The *renderer* changed under an untouched verdict — issue
        #      #800. No timestamp moves, so axis 1's check cannot see it.
        #
        # Axis 2 subsumes axis 1 wherever the dispatch note can be
        # replayed from its sidecar, so the sidecar decides which check
        # runs; see the two branches below.
        decision_path = self.paths.prs / f"pr-{pr_number}" / "review-decision.json"
        note_path = self.paths.prs / f"pr-{pr_number}" / "rework-dispatch-note.txt"
        # The sidecar is what makes re-rendering non-lossy, so an
        # unreadable one is treated exactly like an absent one: fall back
        # to the mtime gate rather than regenerate with an empty note.
        try:
            dispatch_note = note_path.read_text(encoding="utf-8") if note_path.exists() else None
        except (OSError, UnicodeDecodeError):
            dispatch_note = None
        # A missing verdict is the other way re-rendering could lose
        # information: _render_required_changes_section would fall to its
        # Tier 3 "REVIEWER FINDINGS UNAVAILABLE" placeholder and overwrite
        # a brief that still has real findings in it. Without the verdict
        # there is no way to produce a *better* brief, only a worse one,
        # so leave it alone — same failure direction as the sidecar guard.
        if dispatch_note is not None and decision_path.exists():
            # Issue #800: the mtime comparison above only sees one of the
            # two ways a brief goes stale. It catches *verdict content*
            # drift, but a brief also goes stale when the **renderer**
            # changes underneath an untouched verdict — 35c072d adding
            # tier-2/3 fallbacks to _render_required_changes_section, or
            # #883 rewriting rework.md's fences. Nothing about that makes
            # the verdict newer, so every brief already on disk kept
            # rendering through the old code indefinitely.
            #
            # A template/renderer digest (the #592 approach) would not fix
            # this: #800's own trigger was a change to Python, not to a
            # template. So re-render unconditionally and diff instead —
            # that is axis-agnostic by construction and cannot miss a
            # staleness source nobody has enumerated yet.
            #
            # Writing stays conditional on the content actually differing.
            # An unconditional write would churn the brief mtime that
            # _is_verdict_newer_than_brief reads, on every pass.
            #
            # Reading the brief is new failure surface on a path that
            # previously never opened it, and this loop body has no
            # per-issue exception handling — an unreadable brief would
            # abort the whole pass for every *other* issue too. Treat it
            # as "differs": the note is in hand, so regenerating is both
            # safe and the right repair for an unreadable file.
            try:
                current_brief: str | None = rework_prompt_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                current_brief = None
            expected_brief = _wf._render_rework_prompt(
                self.paths.state_file,
                pr,
                issue_number,
                dispatch_note,
                self.config,
                repo_root=self.repo_root,
            )
            if expected_brief != current_brief:
                verdict_newer = _wf._is_verdict_newer_than_brief(decision_path, rework_prompt_path)
                self._write_rework_prompt(pr, issue_number, dispatch_note)
                _wf.log_event(
                    self.paths.state_file,
                    "rework_brief_regenerated",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        # Which axis moved: a newer verdict is the #632
                        # case the mtime gate already covered; anything
                        # else is renderer/template drift (#800), which
                        # was previously undetectable.
                        "reason": "verdict_newer" if verdict_newer else "renderer_drift",
                    },
                    repo=self.repo_root.name,
                )
        elif _wf._is_verdict_newer_than_brief(decision_path, rework_prompt_path):
            # No usable sidecar: a brief predating #632, whose dispatch note
            # exists only inside the rendered markdown. Re-rendering it
            # would silently drop that note, so the unconditional path
            # above is gated on the inputs being reproducible. Fall back
            # to the mtime gate, which regenerates with an empty note
            # only when a newer verdict makes the findings worth more
            # than the lost prose.
            self._write_rework_prompt(pr, issue_number, "")
        # Rescue tier (issue #555): rescue-marked PRs always launch via
        # the claude-code adapter pinned to rescue.worker_model,
        # regardless of the primary configured worker.harness.
        pr_state_for_rescue = rescue_state_snapshot.get("prs", {}).get(str(pr_number), {})
        if pr_state_for_rescue.get("rescue_attempted"):
            rescue_issue_numbers.add(issue_number)
        session_requests.append(
            SessionRequest(
                issue_number=issue_number,
                issue_title=str(full_issue.get("title") or ""),
                prompt_path=rework_prompt_path,
                branch_name=branch_name,
                rework=True,
            )
        )

    if not session_requests:
        # Release the dispatch_pending claims for all skipped issues
        no_session_failure_map = _wf._build_failure_map(
            [],
            set(),
            deferred_by_concurrency_full,
            rework_limit,
            extra_failures=missing_prompt_failures,
        )
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            for issue_number in skipped_issue_numbers:
                full_issue = full_issues[issue_number]
                entry = {
                    **state["issues"].get(str(issue_number), {}),
                    "number": issue_number,
                    "title": full_issue.get("title"),
                    "url": full_issue.get("url"),
                    # Issue #116: restore to rework_requested for retry (missing prompt may be transient)
                    "status": "rework_requested",
                    "dispatched_at": None,
                }
                entry.pop("dispatch_pending_at", None)
                entry.pop("label_error", None)
                state["issues"][str(issue_number)] = entry
            state = _wf.append_event(
                state,
                "dispatch_rework",
                {
                    "pr_number": first_rework_pr_number,
                    "issue_numbers": [],
                    "failed_issue_numbers": [],
                    "skipped_issue_numbers": sorted(skipped_issue_numbers),
                    "deferred_by_concurrency": deferred_by_concurrency,
                    "deferred_by_concurrency_count": deferred_by_concurrency_count,
                    "label_errors": [],
                    "operator_claimed_skipped": sorted(operator_claimed_skipped),
                    "failures": no_session_failure_map,
                    # Issue #1014 (mirroring #1005): the capacity axis.
                    # Always present -- gov.report_fields() is safe to call
                    # unclamped -- and explicit about `clamped` so a reader
                    # does not have to redo the arithmetic. `dispatch_limit`
                    # is included explicitly (report_fields() does not carry
                    # it) because it is the only field that reflects a
                    # fleet-cap clamp.
                    "concurrency_governor": {
                        "clamped": gov.clamped,
                        "dispatch_limit": gov.dispatch_limit,
                        **gov.report_fields(),
                    },
                },
                state_path=self.paths.state_file,
            )
            _wf.save_state(self.paths.state_file, state)
        data = {
            "adapter": self.config.worker.harness,
            "selected_count": 0,
            "failures": no_session_failure_map,
            "deferred_by_concurrency": deferred_by_concurrency,
            "deferred_by_concurrency_count": deferred_by_concurrency_count,
            "routed_to_review": sorted(routed_to_review),
            "skipped_head_indeterminate": sorted(head_indeterminate),
            "review_blocked_retry": sorted(review_blocked_retry),
            "operator_claimed_skipped": sorted(operator_claimed_skipped),
            "no_op_rework_escalated": sorted(no_op_rework_escalated),
            "worker_death_escalated": sorted(worker_death_escalated),
            "salvaged_to_review": sorted(salvaged_to_review),
            "blocked_environment_escalated": sorted(blocked_environment_escalated),
        }
        if gov.enabled or gov.fleet_enabled or gov.open_pr_enabled:
            data.update(gov.report_fields())
        return _wf.CommandResult(
            True,
            "no valid rework prompts found",
            data,
        )

    manifest_path = self._layout.session_manifest
    results_path = self._layout.session_results
    # Rescue tier (issue #555): split the batch so rescue-marked issues
    # launch via the claude-code adapter pinned to rescue.worker_model
    # (see _rescue_adapter_settings), while every other candidate
    # dispatches via the pass's single configured worker harness. Reuses
    # the same dispatch_sessions()/launch_claude_worker() path for both;
    # the only difference is which AdapterSettings/config is passed in.
    normal_requests = [r for r in session_requests if r.issue_number not in rescue_issue_numbers]
    rescue_requests = [r for r in session_requests if r.issue_number in rescue_issue_numbers]
    dispatch_results: list[SessionDispatchResult] = []
    if normal_requests:
        dispatch_results.extend(
            _wf.dispatch_sessions(
                self.repo_root,
                manifest_path,
                results_path,
                self._adapter_settings(),
                normal_requests,
            )
        )
    if rescue_requests:
        dispatch_results.extend(
            _wf.dispatch_sessions(
                self.repo_root,
                manifest_path,
                results_path,
                self._rescue_adapter_settings(),
                rescue_requests,
            )
        )
    # When both normal and rescue tiers dispatched, each sub-call's
    # write_session_manifest/write_session_results overwrote the files with
    # only its subset. Rewrite once with the combined batch so the on-disk
    # observability files reflect the full pass. When only one tier ran,
    # its sub-call already wrote the correct manifest, so the combined
    # manifest write is skipped (it was redundant and, before #626, used
    # the wrong label).
    if normal_requests and rescue_requests:
        # Every normal-tier issue uses the single configured worker
        # harness; the rescue tier always adds "claude-code" (issue #626).
        # A homogeneous batch is labeled with its single kind via
        # manifest_adapter_label, not "mixed" — "mixed" still occurs
        # routinely here whenever the worker harness differs from
        # claude-code (e.g. a devin-shell primary worker + claude-code
        # rescue).
        combined_kinds = {self.config.worker.harness}
        combined_kinds.add("claude-code")
        _wf.write_session_manifest(
            manifest_path, session_requests, adapter=manifest_adapter_label(combined_kinds)
        )
    write_session_results(results_path, dispatch_results)

    successful_issue_numbers = {result.issue_number for result in dispatch_results if result.ok}
    failed_issue_numbers = {result.issue_number for result in dispatch_results if not result.ok}

    # Second lock: upgrade claim from dispatch_pending to dispatched/dispatch_failed
    label_errors: list[int] = []
    label_error_failures: dict[int, str] = {}
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        # Record skipped issues (missing rework prompt) as rework_requested
        # This handles the mixed case where some issues have prompts and some don't.
        # Missing rework-prompt.md may be transient (review agent hasn't written it yet),
        # so restore to rework_requested for retry (issue #116).
        for issue_number in skipped_issue_numbers:
            full_issue = full_issues[issue_number]
            entry = {
                **state["issues"].get(str(issue_number), {}),
                "number": issue_number,
                "title": full_issue.get("title"),
                "url": full_issue.get("url"),
                "status": "rework_requested",
                "dispatched_at": None,
            }
            entry.pop("dispatch_pending_at", None)
            entry.pop("label_error", None)
            state["issues"][str(issue_number)] = entry
        for request in session_requests:
            full_issue = full_issues[request.issue_number]
            ok = request.issue_number in successful_issue_numbers
            entry = {
                **state["issues"].get(str(request.issue_number), {}),
                "number": request.issue_number,
                "title": full_issue.get("title"),
                "url": full_issue.get("url"),
                "branch_name": request.branch_name,
                "prompt_path": str(request.prompt_path),
                # On failure, restore to rework_requested so the issue can be retried
                # in the next pass (issue #116). On success, mark as dispatched.
                "status": "dispatched" if ok else "rework_requested",
                "dispatched_at": _wf.utc_now() if ok else None,
            }
            entry.pop("dispatch_pending_at", None)
            entry.pop("label_error", None)
            # A successful dispatch supersedes any previous orphan flag.
            if ok:
                entry.pop("orphan_flagged_at", None)
                entry.pop("orphan_drift_fingerprint", None)
                entry.pop("orphan_drift_at", None)
                # Issue #1106: a new rework dispatch supersedes any
                # prior startup-death classification — the new session
                # is the one whose outcome the next janitor pass will
                # attribute, so the stale flag must not survive.
                dispatched_pr = pr_by_issue.get(request.issue_number)
                if dispatched_pr is not None:
                    dispatched_pr_number = int(dispatched_pr["number"])
                    pr_state = state.get("prs", {}).get(str(dispatched_pr_number), {})
                    if pr_state:
                        state["prs"][str(dispatched_pr_number)] = {
                            **pr_state,
                            "last_rework_failure_kind": None,
                            "last_rework_was_startup_death": False,
                        }
            # Store worker PID and process start time for state-based liveness detection
            # This allows recovery even when session sidecar files are orphaned (issue #207)
            if ok:
                result = next(
                    (r for r in dispatch_results if r.issue_number == request.issue_number),
                    None,
                )
                if result and result.pid is not None:
                    entry["worker_pid"] = result.pid
                    entry["worker_process_start_time"] = result.process_start_time
            if ok:
                # Track redispatch count for escalation cap (issue #165)
                now = datetime.now(UTC)
                redispatch_at = _wf._windowed_redispatch_at(
                    entry, window_minutes=self.config.watchdog.redispatch_window_minutes
                ) + [now.isoformat().replace("+00:00", "Z")]
                if len(redispatch_at) > self.config.watchdog.max_auto_redispatch:
                    # Escalate to human review
                    # Issue #783: rework dispatch redispatch cap is a
                    # process failure, not a judgment call -- mechanical.
                    state = _wf._escalate_issue(
                        state,
                        request.issue_number,
                        reason="redispatch_cap_exceeded",
                        reason_class="mechanical",
                        issue_extra={"redispatch_at": redispatch_at},
                    )
                    entry = state["issues"][str(request.issue_number)]
                    _wf.save_state(self.paths.state_file, state)
                    edge = _wf._escalation_edge("redispatch_escalated", "mechanical")
                    result = _wf.transition(
                        self.gh,
                        self.config.labels,
                        request.issue_number,
                        edge,
                    )
                    if result.outcome != TransitionOutcome.APPLIED:
                        label_error = {
                            "edge": edge,
                            "outcome": result.outcome.value,
                            "add_failures": result.add_failures,
                            "remove_failures": result.remove_failures,
                        }
                        entry["label_error"] = label_error
                        label_errors.append(request.issue_number)
                        label_error_failures[request.issue_number] = _wf._label_error_reason(
                            label_error
                        )
                        _wf.save_state(self.paths.state_file, state)
                    continue
                else:
                    entry["redispatch_at"] = redispatch_at
                    state["issues"][str(request.issue_number)] = entry
                    _wf.save_state(self.paths.state_file, state)
                    result = _wf.transition(
                        self.gh,
                        self.config.labels,
                        request.issue_number,
                        "rework_dispatched",
                    )
                    if result.outcome != TransitionOutcome.APPLIED:
                        label_error = {
                            "edge": "rework_dispatched",
                            "outcome": result.outcome.value,
                            "add_failures": result.add_failures,
                            "remove_failures": result.remove_failures,
                        }
                        entry["label_error"] = label_error
                        label_errors.append(request.issue_number)
                        label_error_failures[request.issue_number] = _wf._label_error_reason(
                            label_error
                        )
                        _wf.save_state(self.paths.state_file, state)
            else:
                # Track every rework-dispatch attempt, successful or not,
                # against the same redispatch window used on the success path.
                # Failed attempts that repeat without ever succeeding
                # eventually trip max_auto_redispatch and escalate instead of
                # looping forever (issue #515).
                failed_result = next(
                    (r for r in dispatch_results if r.issue_number == request.issue_number),
                    None,
                )
                failure_kind = failed_result.failure_kind if failed_result else None
                now = datetime.now(UTC)
                # Issue #1393: a pre-launch environment block (e.g.
                # worktree_foreign_writer) never started a worker session,
                # so it must NOT count against the redispatch cap (which
                # measures worker output, not environment hygiene).  Use a
                # separate blocked_environment_at counter and escalate with
                # the correct reason + blocking path after the same cap.
                blocked_environment = failure_kind in PRE_LAUNCH_BLOCKED_ENVIRONMENT_FAILURE_KINDS
                if blocked_environment:
                    blocked_environment_at = _wf._windowed_blocked_environment_at(
                        entry,
                        window_minutes=self.config.watchdog.redispatch_window_minutes,
                    ) + [now.isoformat().replace("+00:00", "Z")]
                    blocking_error = failed_result.error if failed_result else None
                    if len(blocked_environment_at) > self.config.watchdog.max_auto_redispatch:
                        # Issue #1423: before escalating a blocked-environment
                        # cap exhaustion for a foreign writer, attempt to reap
                        # it one more time. A writer that was active on earlier
                        # passes but has since gone idle is reaped here instead
                        # of escalating a zombie to a human. Escalation is
                        # reserved for a writer that is alive *and* active.
                        #
                        # Review finding: bound the number of auto-reaps per
                        # issue before falling back to escalation (see the
                        # fresh-dispatch site for the full rationale).
                        max_reaps = self.config.watchdog.max_foreign_writer_reaps
                        prior_reaps = _wf._windowed_foreign_writer_reaps(
                            entry,
                            window_minutes=self.config.watchdog.redispatch_window_minutes,
                        )
                        if len(prior_reaps) < max_reaps and _wf._try_reap_blocked_foreign_writer(
                            failed_result,
                            self.config,
                            self.paths.state_file,
                            request.issue_number,
                            sessions_dir,
                        ):
                            entry["status"] = "rework_requested"
                            entry["dispatched_at"] = None
                            entry["blocked_environment_at"] = []
                            entry["foreign_writer_reaps"] = prior_reaps + [
                                now.isoformat().replace("+00:00", "Z")
                            ]
                            state["issues"][str(request.issue_number)] = entry
                            state = _wf.append_event(  # event-consumer: audit-only -- records a rework-dispatch foreign-writer reap (issue #1423) already enforced by the blocked_environment_at reset and foreign_writer_reaps counter; consumed by tests/test_charlie_work.py regression tests.
                                state,
                                "rework_dispatch_blocked_environment_reaped",
                                {
                                    "issue_number": request.issue_number,
                                    "failure_kind": failure_kind,
                                    "pid": failed_result.pid if failed_result else None,
                                    "blocked_environment_count": len(blocked_environment_at),
                                    "foreign_writer_reap_count": len(prior_reaps) + 1,
                                },
                                state_path=self.paths.state_file,
                            )
                            _wf.save_state(self.paths.state_file, state)
                            continue
                        # Escalate with the environment reason and the
                        # blocking path so the operator sees "remove
                        # C:\...\wt", not "worker quality cap exceeded."
                        state = _wf._escalate_issue(
                            state,
                            request.issue_number,
                            reason="dispatch_blocked_environment",
                            reason_class="mechanical",
                            issue_extra={
                                "blocked_environment_at": blocked_environment_at,
                                "dispatched_at": None,
                            },
                        )
                        entry = state["issues"][str(request.issue_number)]
                        state = _wf.append_event(
                            state,
                            "session_failed_escalated",
                            {
                                "issue_number": request.issue_number,
                                "previous_status": "rework_requested",
                                "reason": "dispatch_blocked_environment",
                                "failure_kind": failure_kind,
                                "blocking_error": blocking_error,
                                "blocked_environment_count": len(blocked_environment_at),
                            },
                            state_path=self.paths.state_file,
                        )
                        _wf.save_state(self.paths.state_file, state)
                        edge = _wf._escalation_edge("redispatch_escalated", "mechanical")
                        result = _wf.transition(
                            self.gh,
                            self.config.labels,
                            request.issue_number,
                            edge,
                        )
                        if result.outcome != TransitionOutcome.APPLIED:
                            label_error = {
                                "edge": edge,
                                "outcome": result.outcome.value,
                                "add_failures": result.add_failures,
                                "remove_failures": result.remove_failures,
                            }
                            entry["label_error"] = label_error
                            label_errors.append(request.issue_number)
                            label_error_failures[request.issue_number] = _wf._label_error_reason(
                                label_error
                            )
                            _wf.save_state(self.paths.state_file, state)
                        continue
                    # Cap not exceeded: restore to rework_requested without
                    # incrementing redispatch_at, and emit a distinct event
                    # so the operator can see the environment conflict
                    # before it escalates.
                    entry["status"] = "rework_requested"
                    entry["dispatched_at"] = None
                    entry["blocked_environment_at"] = blocked_environment_at
                    state["issues"][str(request.issue_number)] = entry
                    state = _wf.append_event(  # event-consumer: audit-only -- records a pre-launch environment block (issue #1393) already enforced by the blocked_environment_at counter and the dispatch_blocked_environment escalation; consumed by tests/test_charlie_work.py regression tests.
                        state,
                        "rework_dispatch_blocked_environment",
                        {
                            "issue_number": request.issue_number,
                            "failure_kind": failure_kind,
                            "blocking_error": blocking_error,
                            "blocked_environment_count": len(blocked_environment_at),
                        },
                        state_path=self.paths.state_file,
                    )
                    _wf.save_state(self.paths.state_file, state)
                    continue
                redispatch_at = _wf._windowed_redispatch_at(
                    entry, window_minutes=self.config.watchdog.redispatch_window_minutes
                ) + [now.isoformat().replace("+00:00", "Z")]
                terminal_failure = failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
                # Issue #807: a deterministic judgment failure escalates
                # immediately but as ``reason_class="judgment"``.
                deterministic_judgment = (
                    failure_kind in DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS
                )
                immediate_escalation = terminal_failure or deterministic_judgment
                if (
                    immediate_escalation
                    or len(redispatch_at) > self.config.watchdog.max_auto_redispatch
                ):
                    # Escalate to human review
                    reason = failure_kind if immediate_escalation else "redispatch_cap_exceeded"
                    # Issue #783: dead worker session / redispatch cap is a
                    # process failure, not a judgment call -- mechanical.
                    # Issue #807: a deterministic judgment failure (genuine
                    # local commits) is a judgment call -- judgment.
                    reason_class = "judgment" if deterministic_judgment else "mechanical"
                    state = _wf._escalate_issue(
                        state,
                        request.issue_number,
                        reason=reason,
                        reason_class=reason_class,
                        issue_extra={
                            "redispatch_at": redispatch_at,
                            "dispatched_at": None,
                        },
                    )
                    entry = state["issues"][str(request.issue_number)]
                    _wf.save_state(self.paths.state_file, state)
                    edge = _wf._escalation_edge("redispatch_escalated", reason_class)
                    result = _wf.transition(
                        self.gh,
                        self.config.labels,
                        request.issue_number,
                        edge,
                    )
                    if result.outcome != TransitionOutcome.APPLIED:
                        label_error = {
                            "edge": edge,
                            "outcome": result.outcome.value,
                            "add_failures": result.add_failures,
                            "remove_failures": result.remove_failures,
                        }
                        entry["label_error"] = label_error
                        label_errors.append(request.issue_number)
                        label_error_failures[request.issue_number] = _wf._label_error_reason(
                            label_error
                        )
                        _wf.save_state(self.paths.state_file, state)
                    continue
                entry["status"] = "rework_requested"
                entry["dispatched_at"] = None
                entry["redispatch_at"] = redispatch_at
                state["issues"][str(request.issue_number)] = entry
                _wf.save_state(self.paths.state_file, state)
        rework_failure_map = _wf._build_failure_map(
            dispatch_results,
            failed_issue_numbers,
            deferred_by_concurrency_full,
            rework_limit,
            extra_failures={**missing_prompt_failures, **label_error_failures},
        )
        state = _wf.append_event(
            state,
            "dispatch_rework",
            {
                "pr_number": first_rework_pr_number,
                "issue_numbers": sorted(successful_issue_numbers),
                "failed_issue_numbers": sorted(failed_issue_numbers),
                "skipped_issue_numbers": sorted(skipped_issue_numbers),
                "deferred_by_concurrency": deferred_by_concurrency,
                "deferred_by_concurrency_count": deferred_by_concurrency_count,
                "label_errors": sorted(label_errors),
                "operator_claimed_skipped": sorted(operator_claimed_skipped),
                "failures": rework_failure_map,
                # Issue #1014 (mirroring #1005): the capacity axis.
                # Always present -- gov.report_fields() is safe to call
                # unclamped -- and explicit about `clamped` so a reader
                # does not have to redo the arithmetic. `dispatch_limit`
                # is included explicitly (report_fields() does not carry
                # it) because it is the only field that reflects a
                # fleet-cap clamp.
                "concurrency_governor": {
                    "clamped": gov.clamped,
                    "dispatch_limit": gov.dispatch_limit,
                    **gov.report_fields(),
                },
            },
            state_path=self.paths.state_file,
        )
        _wf.save_state(self.paths.state_file, state)

    result_dicts = [result.to_dict() for result in dispatch_results]
    message = "rework dispatch complete"
    if failed_issue_numbers:
        entries = ", ".join(
            f"#{issue} ({rework_failure_map[issue]})" for issue in sorted(failed_issue_numbers)
        )
        message = f"rework dispatch failures: {entries}"
    if label_errors:
        message += f" (launched but label write failed: {sorted(label_errors)})"
    data = {
        "selected_count": len(successful_issue_numbers),
        "attempted_count": len(session_requests),
        "failed_count": len(failed_issue_numbers),
        "failures": rework_failure_map,
        "deferred_by_concurrency": deferred_by_concurrency,
        "deferred_by_concurrency_count": deferred_by_concurrency_count,
        "label_errors": sorted(label_errors),
        "session_manifest": str(manifest_path),
        "session_results": str(results_path),
        "sessions": [asdict(request) for request in session_requests],
        "dispatch_results": result_dicts,
        "routed_to_review": sorted(routed_to_review),
        "skipped_head_indeterminate": sorted(head_indeterminate),
        "review_blocked_retry": sorted(review_blocked_retry),
        "operator_claimed_skipped": sorted(operator_claimed_skipped),
        "no_op_rework_escalated": sorted(no_op_rework_escalated),
        "salvaged_to_review": sorted(salvaged_to_review),
        "blocked_environment_escalated": sorted(blocked_environment_escalated),
    }
    if gov.enabled or gov.fleet_enabled or gov.open_pr_enabled:
        data.update(gov.report_fields())

    # Emit notification digest if there are health transitions (stalled sessions)
    # This will be enhanced by #165 to include RUNAWAY/DEAD/escalated transitions
    sessions_dir = self._layout.sessions_dir
    stalled_entries = _wf._detect_stalled_sessions(sessions_dir, self.config)
    if stalled_entries and self.config.notify.enabled:
        health_transitions: dict[int, dict[str, Any]] = {}
        for entry in stalled_entries:
            health_transitions[entry["issue"]] = {
                "adapter_kind": "unknown",  # Will be filled by #165's full supervisor
                "health": entry.get("health", "STALLED"),
                "last_log_line": None,
                "pid": entry.get("pid"),
                "terminal_tool": entry.get("terminal_tool"),
                "terminal_reason": entry.get("terminal_reason"),
            }
        digest = _wf._build_attention_digest(
            self.paths.state_file,
            health_transitions,
            repo=self.repo_root.name,
        )
        if digest:
            _wf.emit_digest(self._layout.notify, digest)

    return _wf.CommandResult(
        not failed_issue_numbers,
        message,
        data,
    )
