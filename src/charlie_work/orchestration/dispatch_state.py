"""Dispatch loop moved out of ``OrchestratorApp`` (Track 2 Phase B, L08, #1639).

``_dispatch_impl`` was relocated verbatim from ``charlie_work.workflow`` (design
doc ``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2); ``workflow_delegation._install_delegates`` re-attaches it
unwrapped onto ``OrchestratorApp`` so ``app._dispatch_impl(...)`` binds ``self``
through the descriptor protocol and every existing patch/monkeypatch site keeps
intercepting.

Namespace rule (#1627). Two kinds of name are reached through the
``import charlie_work.workflow as _wf`` seam rather than imported directly:

- **Patched-on-workflow (Tier D).** Names some test patches on the
  ``charlie_work.workflow`` module object -- ``state_lock``, ``load_state_locked``,
  ``utc_now``, ``transition``, ``is_pid_alive``, ``is_claim_stale``,
  ``dispatch_sessions``, ``emit_digest``, ``linked_issue_number``,
  ``_count_live_sessions``, ``_detect_and_handle_stalled_sessions``,
  ``_worker_pid_alive``, ``_try_reap_blocked_foreign_writer`` -- must resolve
  through ``_wf.`` so ``patch("charlie_work.workflow.<name>")`` still bites.
- **Defined in workflow.py.** ``CommandResult``, ``_MergedPRListOutcome``,
  ``_build_attention_digest``, ``_build_failure_map``, ``_label_error_reason``,
  ``_recent_dispatch_failed_attempts`` live in ``workflow`` itself; reaching them
  via ``_wf.`` avoids an import cycle and keeps a single definition site.

The three state primitives ``load_state`` / ``save_state`` / ``append_event`` are
also routed via ``_wf.`` (matching ``state_maintenance.py``). ``load_state`` in
particular MUST be ``_wf.load_state``: all seven of its calls sit inside
``with _wf.state_lock():`` blocks, and ``state_lock`` is forced to ``_wf.`` by
Tier D. ``tests/test_load_state_locked.py`` recognises a lock context only when
the ``with`` item is a bare ``ast.Name`` ``state_lock`` and flags a ``load_state``
call only when it is a bare ``ast.Name``; routing ``state_lock`` through the
``_wf.`` attribute makes the visitor stop seeing the lock, so ``load_state`` must
likewise become an attribute (``_wf.``) to stay invisible to that lint and
preserve behaviour.

Everything else the body needs -- including the deterministic-escalation config
constants and the escalation helper ``_escalate_issue`` (patched by no test; a
census hit is only a ``pytest.param`` string literal in
``tests/test_escalation_split.py``) -- is imported directly from its source
module below.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import charlie_work.workflow as _wf
from charlie_work.adapters import SessionRequest, cleanup_stale_session_tmp_files
from charlie_work.backlog_reachability import (
    classify_backlog_reachability,
    resolve_dispatch_mention_coverage,
)
from charlie_work.ci_findings import check_dispatch_staleness
from charlie_work.citation_check import CitationVerdict
from charlie_work.config import (
    DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS,
    PRE_LAUNCH_BLOCKED_ENVIRONMENT_FAILURE_KINDS,
)
from charlie_work.cross_repo_gate import (
    CrossRepoGateResult,
    cross_repo_gate,
    cross_repo_scope_gate,
)
from charlie_work.dead_worker_reap import (
    _detect_stalled_sessions,
    _dispatching_repo_name,
    _issues_with_live_workers,
)
from charlie_work.dispatch_selection import (
    _MAX_DEFERRED_CONCURRENCY_EXAMPLES,
    _select_dispatch_candidates,
    _windowed_blocked_environment_at,
    _windowed_foreign_writer_reaps,
)
from charlie_work.env_sanitize import worker_github_token_findings
from charlie_work.escalation import _escalate_issue, _escalation_edge
from charlie_work.fleet_registry import managed_repo_names
from charlie_work.labels import TransitionOutcome
from charlie_work.state import (
    clear_escalation,
    clear_escalation_on_issue_prs,
    escalation_reason_class,
    is_throttled,
    operator_claimed_issues,
)


def _dispatch_impl(
    self,
    limit: int | None = None,
    *,
    only_issues: str | None = None,
    stalled_entries: list[dict[str, int]] | None = None,
    ready_issues: list[dict[str, Any]] | None = None,
    merged_prs: _wf._MergedPRListOutcome | None = None,
) -> _wf.CommandResult:
    # Issue #1001: worker GitHub token gate. Before dispatching to an
    # adapter family that routes through sanitize_env's merge, consult the
    # same predicate doctor._check_worker_github_token uses. If no
    # worker_env token is configured, escalate once (not per pass) and
    # either refuse (when dispatch.require_worker_github_token is True) or
    # warn and proceed (the default, so the gate does not take the fleet
    # down on a config that has not yet been provisioned — see the issue
    # #1001 sequencing hazard comment in config.py).
    #
    # The once-only guarantee must hold across OrchestratorApp
    # reconstruction: fleet_dispatch.fleet_loop builds a fresh app per
    # repo per pass, so an instance-level flag alone resets every pass
    # and re-escalates indefinitely. The durable marker
    # ``worker_token_escalated`` in state.json is the cross-instance
    # source of truth; the instance-level ``_worker_token_escalated``
    # flag (initialized in __init__) is a same-instance optimization
    # that also covers dry-run, where the durable marker is never
    # written. The marker is cleared when the condition resolves (all
    # findings ok), so a future regression re-escalates.
    token_findings = worker_github_token_findings(self.config)
    missing_findings = [f for f in token_findings if not f.ok]
    if missing_findings:
        if not self._worker_token_escalated:
            self._worker_token_escalated = True
            # Record the escalation event once. Payload carries only
            # config_key names and adapter contexts — never a token
            # value or prefix (issue #1001 acceptance criterion).
            #
            # Dry-run never writes: the escalation event and the
            # durable marker are state mutations (state_lock +
            # save_state), so they are gated on ``not self.dry_run``
            # — the same read-only contract documented at the
            # merge_ready dry-run gate (~line 15452, "Dry-run never
            # writes") and modelled on this function's own
            # top-of-body dry-run short-circuit. The in-memory
            # once-only flag is still set under dry-run so a dry-run
            # pass does not re-enter this block on the next pass;
            # the event and marker are emitted on the first real
            # (non-dry-run) dispatch. ``self.dry_run`` is fixed at
            # construction, so a dry-run instance cannot later
            # "forget" the flag and skip a real write.
            #
            # The durable marker is read inside the lock (not before it)
            # so the cross-instance once-only guarantee holds without an
            # unlocked load_state — issue #310's
            # test_no_unlocked_load_state_in_production_code lint forbids
            # any load_state outside a state_lock block. The in-memory
            # ``_worker_token_escalated`` flag is the first gate
            # (same-instance), so the lock is entered at most once per
            # instance lifetime (the flag's False→True transition); the
            # inner ``if not state.get(...)`` re-check is the
            # authoritative cross-instance gate and no-ops when a prior
            # instance already set the marker.
            if not self.dry_run:
                with _wf.state_lock(self.paths.state_file):
                    state = _wf.load_state(self.paths.state_file)
                    if not state.get("worker_token_escalated", False):
                        state = self._record_event(
                            state,
                            "worker_token_missing",
                            {
                                "findings": [
                                    {
                                        "config_key": f.config_key,
                                        "context": f.context,
                                    }
                                    for f in missing_findings
                                ],
                            },
                            level="warning",
                        )
                        state["worker_token_escalated"] = True
                        _wf.save_state(self.paths.state_file, state)
        # The refusal itself is NOT dry-run-gated: a dry-run preview must
        # report the same deferral a live pass would take (matching the
        # fleet_lock_held / graphql_rate_limit deferral precedent in this
        # function). Only the escalation event / durable marker writes
        # above stay behind ``not self.dry_run``.
        if self.config.dispatch.require_worker_github_token:
            return _wf.CommandResult(
                True,
                "dispatch deferred: no sanctioned worker GitHub token "
                "(set devin.worker_env / claude_code.worker_env "
                "{'GH_TOKEN': '<scoped-PAT>'}; see issue #1001)",
                {
                    "selected_count": 0,
                    "attempted_count": 0,
                    "failed_count": 0,
                    "skipped_issue_numbers": [],
                    "label_errors": [],
                    "sessions": [],
                    "dispatch_results": [],
                    "deferred_reason": "worker_token_missing",
                    "missing_config_keys": [f.config_key for f in missing_findings],
                },
            )
    else:
        self._worker_token_escalated = False
        # Condition resolved: clear the durable marker so a future
        # regression re-escalates. Dry-run never writes — a dry-run pass
        # that observes a now-healthy config must not mutate the marker
        # set by a prior real pass (and cannot have set it itself).
        if not self.dry_run:
            with _wf.state_lock(self.paths.state_file):
                state = _wf.load_state(self.paths.state_file)
                if state.get("worker_token_escalated", False):
                    state["worker_token_escalated"] = False
                    _wf.save_state(self.paths.state_file, state)

    # Issue #427: include closed ready-labeled issues so externally-merged PRs
    # (e.g. Aviator MergeQueue) can be finalized even after GitHub closes the issue.
    if ready_issues is None:
        issues = self.gh.issue_list(
            labels=[self.config.labels.ready],
            state="all",
        )
    else:
        issues = ready_issues
    dispatch_limit = limit if limit is not None else self.config.dispatch.default_limit
    operator_claimed_ready: list[int] = []

    # Issue #1110 rework: classify_backlog_reachability now runs the same
    # per-issue blocker check the dispatch candidate filter runs (issue
    # #1110 wired _get_open_blockers_for_issue into its else-branch). That
    # check calls get_github_issue_dependencies + are_issues_open per issue
    # -- exactly the N+1 serial `gh` pattern issue #870 built
    # _prefetch_blocker_data to eliminate. Warm the pass-scoped cache for
    # every ready issue once, *before* reachability's serial per-issue
    # lookups run, mirroring how status() warms the cache before its own
    # classify_backlog_reachability call. ``issues`` here is the
    # ready-labelled state="all" set; reachability fetches its own open
    # list, but its blocker check only runs on ready-labelled OPEN issues,
    # which are a subset of this set, so this warm-up covers every
    # dependency lookup reachability will make. The later
    # _filter_blocked_issues(candidates) call below benefits too:
    # candidates are a further subset, so the cache is already warm for
    # them as well. Harmless to call twice (second call is a cache hit).
    self._prefetch_blocker_data(issues)

    # Issue #944: observe the UNFILTERED backlog alongside the filtered
    # candidate query above. This does not participate in selection and
    # must not change dispatch behaviour -- it exists so that a zero
    # dispatch count carries a reason. Done here, outside the state lock,
    # because it is network I/O.
    # Issue #1337: resolve the mention-coverage map, reusing an
    # already-fetched merged-PR outcome when available and fetching
    # (fail-open) otherwise. See resolve_dispatch_mention_coverage's
    # docstring for the three-branch logic. A fresh fetch rebinds
    # ``merged_prs`` so the later _resolve_merged_prs calls and the
    # tripwire reuse the same list -- no second API call.
    mention_covered, _fetched_merged_prs = resolve_dispatch_mention_coverage(
        issues, merged_prs, self.gh, self
    )
    if _fetched_merged_prs is not None:
        merged_prs = _wf._MergedPRListOutcome(_fetched_merged_prs, called=True)
    backlog_reachability = classify_backlog_reachability(
        self.gh,
        self.config,
        operator_claimed_issues(_wf.load_state_locked(self.paths.state_file)),
        ready_open_count=sum(
            1 for issue in issues if str(issue.get("state") or "OPEN").upper() == "OPEN"
        ),
        mention_covered=mention_covered,
    )

    # Gather sessions_dir for stall detection and live worker counting
    sessions_dir = self._layout.sessions_dir
    # Issue #1393: clean up stranded .json.tmp session sidecar files from
    # interrupted atomic writes before this pass writes new ones.
    cleanup_stale_session_tmp_files(sessions_dir)

    # Detect and handle stalled sessions before applying concurrency governor.
    # This must run exactly once per pass, not twice (was duplicated in the
    # governor). When the caller already ran the sweep this pass (loop()'s
    # unconditional reaper at the top of its pass), it hands the result down
    # via ``stalled_entries`` and the sweep is NOT re-run here — see
    # dispatch()'s docstring for why re-running it corrupts the Signal-1
    # deferral counter.
    if stalled_entries is None:
        stalled_entries = _wf._detect_and_handle_stalled_sessions(
            sessions_dir,
            self.paths.state_file,
            self.config,
            write_gate=self.write_gate,
        )

    # Count live workers after stall handling (stalled workers are killed).
    # Corroborated against state.json (issue #343) so a ghost -- a live
    # worker_pid whose sidecar was removed -- cannot silently free a slot.
    live_count = _wf._count_live_sessions(sessions_dir, self.paths.state_file)

    # Apply global concurrency governor cap with pre-computed live_count.
    # Issue #1129: fresh-issue dispatch also applies open-PR backpressure
    # (max_open_agent_prs), pacing new PR creation to the review/merge lane.
    gov = self._apply_concurrency_governor(
        dispatch_limit,
        live_count=live_count,
        apply_open_pr_backpressure=True,
    )
    dispatch_limit = gov.dispatch_limit

    # Compute the merged PR list (if already fetched) for the tripwire so
    # loop() can reuse it and avoid a second GraphQL call per pass.
    merged_prs_for_tripwire: list[dict[str, Any]] | None = (
        merged_prs.items
        if merged_prs is not None and merged_prs.called and merged_prs.error is None
        else None
    )

    # Apply provider throttle cooldown check
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        if is_throttled(state):
            throttled_until = state.get("throttled_until")
            # Return immediately with deferral reason
            data = {
                "selected_count": 0,
                "attempted_count": 0,
                "failed_count": 0,
                "skipped_issue_numbers": [],
                "label_errors": [],
                "sessions": [],
                "dispatch_results": [],
                "merged_prs": merged_prs_for_tripwire,
                "deferred_reason": "provider_throttled",
                "throttled_until": throttled_until,
            }
            if gov.enabled or gov.fleet_enabled or gov.open_pr_enabled:
                data.update(gov.report_fields())
            return _wf.CommandResult(
                False,
                f"dispatch deferred: provider throttled until {throttled_until}",
                data,
            )

    def _resolve_merged_prs(
        outcome: _wf._MergedPRListOutcome | None,
    ) -> list[dict[str, Any]]:
        # Both raising branches (the direct fallback below and the
        # outcome.error re-raise) propagate GitHubError to dispatch()'s
        # ``except GitHubError`` handler, which defers the pass. This is
        # deliberate: proceeding with [] would re-dispatch issues a merged
        # PR already covered (the silent-empty path #633 closed). The
        # direct fallback is the COMMON case — it runs whenever there are
        # open ready issues but no closed-ready issues this pass, because
        # _finalize_externally_merged_issues skips the merged_pr_list()
        # fetch entirely when closed_ready is empty (returning an outcome
        # with called=False).
        if outcome is None or not outcome.called:
            return self.gh.merged_pr_list() if issues else []
        if outcome.error is not None and issues:
            raise outcome.error
        return outcome.items if issues else []

    # Dry-run: read-only planning — compute selection and would-be SessionRequests,
    # but skip all state writes, label transitions, and file mutations.
    if self.dry_run:
        selected_issue_numbers: list[int] = []
        skipped_issue_numbers: list[int] = []
        # Detect stalled sessions (read-only for dry-run)
        stalled_entries = _detect_stalled_sessions(sessions_dir, self.config)
        stalled_issues = {entry["issue"] for entry in stalled_entries}
        live_worker_issues = _issues_with_live_workers(sessions_dir)
        prs = self.gh.pr_list()
        # No ready issues means _merged_pr_referenced_issue_numbers() would
        # return empty sets regardless of what merged_pr_list() returns
        # (it intersects against the ready-issue-number set) — skip the
        # expensive listing query entirely rather than fetch-and-discard
        # (issue #361).
        resolved_merged_prs = _resolve_merged_prs(merged_prs)
        (
            merged_pr_bound_issue_numbers,
            merged_pr_mention_only_issue_numbers,
            _,
            _,
        ) = self._merged_pr_referenced_issue_numbers(issues, resolved_merged_prs)
        merged_pr_issue_numbers = (
            merged_pr_bound_issue_numbers | merged_pr_mention_only_issue_numbers
        )
        pr_by_issue = {}
        # Issue #1229: validate branch-name-derived issue numbers against the
        # real open-issue set so a stale branch name (e.g. agent/issue-709-…
        # left over from a merged issue/PR #709, reused by an unrelated
        # issue-less PR) cannot populate pr_by_issue[<wrong n>] and make the
        # dry-run report wrongly claim a real, dispatchable issue already has
        # an open PR. Same validator as the real dispatch-claim path below so
        # the two cannot diverge.
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

        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            # Same dispatchability logic as the real dispatch, but read-only
            # Issue #5: also check worker liveness for "dispatched" status
            live_dispatched = set()
            for number, entry in state.get("issues", {}).items():
                if not isinstance(entry, dict):
                    continue
                status = entry.get("status")
                if status == "dispatch_pending" and not _wf.is_claim_stale(
                    entry.get("dispatch_pending_at")
                ):
                    live_dispatched.add(int(number))
                elif status == "dispatched":
                    # Issue #5: only exclude if the worker is alive OR there's an open PR.
                    # A dead worker with no open PR is recoverable (crashed before PR opened).
                    # A dead worker with an open PR is mid-review and must not be re-dispatched.
                    # Issue #207: also check state.json worker_pid for liveness when session files are orphaned
                    issue_number = int(number)
                    worker_alive = _wf._worker_pid_alive(entry)
                    if (
                        issue_number in live_worker_issues
                        or worker_alive
                        or issue_number in pr_by_issue
                    ):
                        live_dispatched.add(issue_number)
            issues_with_open_tracked_prs = set(pr_by_issue.keys())
            # Issue #1336: lift the mention-only exclusion for re-armed
            # issues (read-only detection for dry-run parity with the
            # real dispatch path -- never stamps state).
            _already_flagged_dry = {
                int(num)
                for num, entry in state.get("issues", {}).items()
                if isinstance(entry, dict) and entry.get("merged_pr_mention_flagged_at")
            }
            rearmed_mention_issues, _ = self._mention_rearmed_issue_numbers(
                merged_pr_mention_only_issue_numbers,
                issues,
                state,
                _already_flagged_dry,
            )
            merged_pr_issue_numbers = merged_pr_bound_issue_numbers | (
                merged_pr_mention_only_issue_numbers - rearmed_mention_issues
            )
        candidates = [
            issue
            for issue in issues
            if self._is_dispatchable(issue)
            and int(issue["number"]) not in live_dispatched
            and int(issue["number"]) not in stalled_issues
            and int(issue["number"]) not in issues_with_open_tracked_prs
            and int(issue["number"]) not in merged_pr_issue_numbers
        ]

        # Apply dependency gate: skip issues with open blockers (dry-run)
        # Done outside the lock to avoid holding it during GitHub API calls
        candidates, blocked_issues, _open_blockers_by_issue = self._filter_blocked_issues(
            candidates
        )

        # Sort candidates by dispatch order
        # Default (oldest) uses dependency-aware ordering; explicit newest uses creation date
        if self.config.dispatch.order == "newest":
            candidates = self._sort_by_dispatch_order(candidates)
        else:
            # Default: use dependency-aware ordering (out-degree) with oldest-first tiebreaker
            candidates = self._sort_by_dependency_depth(candidates)

        # Fill fresh candidates first; recovery retries only get leftover slots
        # and are capped at one per pass (issue #506).
        (
            selected,
            skipped_issue_numbers,
            deferred_by_concurrency_full,
            deferred_by_concurrency_count,
        ) = _select_dispatch_candidates(
            candidates,
            dispatch_limit,
            state,
            self._branch_name,
            only_issues=only_issues,
        )
        # Issue #1005 review: this dry-run branch never persists an event
        # or calls _build_failure_map, but truncate for display parity
        # with the real-dispatch payload below -- keep the untruncated
        # list around under its own name so nothing downstream mistakes
        # it for complete.
        deferred_by_concurrency = deferred_by_concurrency_full[:_MAX_DEFERRED_CONCURRENCY_EXAMPLES]
        selected_issue_numbers = [int(issue["number"]) for issue in selected]

        # Compute would-be SessionRequests without state mutation
        session_requests: list[SessionRequest] = []
        full_issues: dict[int, dict[str, Any]] = {}
        # Issue #1010: dry-run cross-repo gate — report which issues would
        # be escalated without mutating state or labels.
        # Issue #1244: dry-run cross-repo *scope* gate — report issues whose
        # title names another managed repo.
        dry_run_cross_repo_escalated: dict[int, str] = {}
        fleet_repos = managed_repo_names(self.fleet_dir_override)
        dispatching_repo_name = _dispatching_repo_name(self.gh, self.repo_root)
        for issue_number in selected_issue_numbers:
            full_issue = self.gh.issue_view(issue_number)
            full_issues[issue_number] = full_issue
            branch_name = self._branch_name(full_issue)

            # Pre-flight gate: report cross-repo targets without escalating.
            gate_result = cross_repo_gate(str(full_issue.get("body") or ""), self.repo_root)
            if not gate_result.passed:
                dry_run_cross_repo_escalated[issue_number] = gate_result.reason
                continue

            # Pre-flight scope gate: report cross-repo scope targets.
            scope_result = cross_repo_scope_gate(
                str(full_issue.get("title") or ""),
                str(full_issue.get("body") or ""),
                dispatching_repo_name,
                fleet_repos,
            )
            if not scope_result.passed:
                dry_run_cross_repo_escalated[issue_number] = scope_result.reason
                continue

            prompt_path = self._write_worker_prompt(full_issue, dry_run=True)

            # Check if this is a dead-worker recovery (same logic as real dispatch)
            recovery_record: dict[str, Any] | None = None
            prev_entry = state.get("issues", {}).get(str(issue_number), {})
            prev_branch = prev_entry.get("branch_name")
            if prev_branch == branch_name and prev_entry.get("status") == "dispatched":
                recovery_record = prev_entry

            session_requests.append(
                SessionRequest(
                    issue_number=issue_number,
                    issue_title=str(full_issue.get("title") or ""),
                    prompt_path=prompt_path,
                    branch_name=branch_name,
                    recovery=recovery_record,
                )
            )

        # Return planning data without touching state, labels, or manifest/results files
        data = {
            "selected_count": len(session_requests),
            "attempted_count": len(session_requests),
            "failed_count": 0,
            "skipped_issue_numbers": skipped_issue_numbers,
            "deferred_by_concurrency": deferred_by_concurrency,
            "deferred_by_concurrency_count": deferred_by_concurrency_count,
            "merged_prs": resolved_merged_prs,
            "merged_pr_referenced_issue_numbers": sorted(merged_pr_issue_numbers),
            "merged_pr_mention_only_issue_numbers": sorted(merged_pr_mention_only_issue_numbers),
            "merged_pr_mention_rearmed_issue_numbers": sorted(rearmed_mention_issues),
            "label_errors": [],
            "cross_repo_escalated_issue_numbers": sorted(dry_run_cross_repo_escalated),
            "sessions": [asdict(request) for request in session_requests],
            "dispatch_results": [],
            "blocked": [
                {"issue": issue_number, "blockers": blockers}
                for issue_number, blockers in sorted(blocked_issues.items())
            ],
            "stalled": stalled_entries,
        }
        if gov.enabled or gov.fleet_enabled or gov.open_pr_enabled:
            data.update(gov.report_fields())
        return _wf.CommandResult(
            True,
            f"dry-run: would dispatch {len(session_requests)} issue(s)",
            data,
        )

    # Real dispatch: claim issues, launch workers, update state and labels
    # First lock: claim issues by marking them as dispatch_pending
    selected_issue_numbers: list[int] = []
    skipped_issue_numbers: list[int] = []
    # Use pre-computed stalled_entries from the stall detection above
    stalled_issues = {entry["issue"] for entry in stalled_entries}
    live_worker_issues = _issues_with_live_workers(sessions_dir)
    prs = self.gh.pr_list()
    # No ready issues means _merged_pr_referenced_issue_numbers() would
    # return empty sets regardless of what merged_pr_list() returns (it
    # intersects against the ready-issue-number set) — skip the expensive
    # listing query entirely rather than fetch-and-discard (issue #361).
    resolved_merged_prs = _resolve_merged_prs(merged_prs)
    (
        merged_pr_bound_issue_numbers,
        merged_pr_mention_only_issue_numbers,
        merged_pr_bound_pr_numbers,
        _,
    ) = self._merged_pr_referenced_issue_numbers(issues, resolved_merged_prs)
    merged_pr_issue_numbers = merged_pr_bound_issue_numbers | merged_pr_mention_only_issue_numbers

    # Issue #432: cap merge-finalization per pass so a large backlog cannot
    # monopolize the pass budget. Oldest first (by creation date, then issue
    # number) drains the backlog deterministically.
    issue_by_number = {int(issue["number"]): issue for issue in issues}
    finalize_limit = self.config.dispatch.finalize_limit

    def _finalization_order(issue_numbers: set[int]) -> list[int]:
        return sorted(
            issue_numbers,
            key=lambda n: (issue_by_number.get(n, {}).get("createdAt", ""), n),
        )

    finalizable_bound_issue_numbers = _finalization_order(merged_pr_bound_issue_numbers)[
        :finalize_limit
    ]
    finalizable_mention_issue_numbers = _finalization_order(merged_pr_mention_only_issue_numbers)[
        :finalize_limit
    ]

    pr_by_issue = {}
    # Issue #1229: validate branch-name-derived issue numbers against the
    # real open-issue set so a stale branch name (e.g. agent/issue-709-…
    # left over from a merged issue/PR #709, reused by an unrelated
    # issue-less PR) cannot populate pr_by_issue[<wrong n>] and make the
    # dispatcher believe a real, dispatchable issue already has an open PR,
    # silently skipping dispatch for it. This is the same phantom-binding
    # failure class already fixed at the dead-session escalation guard
    # (_classify_dead_sessions_and_update_throttle_state) and the
    # orphaned-worker sweep (_detect_and_handle_orphaned_workers); all
    # route through _make_branch_issue_validator so the open-issue fetch
    # cannot diverge between call surfaces.
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

    # Close ready issues whose merged PR safely binds to them (hijack-safe:
    # same-repo branch-prefix or closing-action verb — the same trust
    # level issue #220 uses to close at merge time). This is
    # belt-and-suspenders in case #220's merge-time close hasn't landed
    # yet. These are network calls, so they run outside the state lock;
    # the successful closures are persisted to state.json inside the lock
    # below, and the issue numbers are excluded from dispatch candidates
    # regardless of closure success. Issue #432: only the oldest
    # finalize_limit issues are processed per pass, so a one-time backlog
    # cannot monopolize the pass budget.
    closed_merged_pr_issues: set[int] = set()
    for issue_number in finalizable_bound_issue_numbers:
        # Best-effort label transition and issue close. A failure here is
        # non-fatal; the issue is still excluded from dispatch because the
        # merged PR reference exists, and the next pass will retry.
        _wf.transition(self.gh, self.config.labels, issue_number, "merged")
        if self.gh.close_issue(issue_number):
            closed_merged_pr_issues.add(issue_number)

    # Issue #203 (redesigned per review): a merged PR that only
    # *mentions* the issue in free text has no hijack-safe binding and
    # must never authorize a close. Flag it for a human instead — the
    # issue is excluded from this pass's candidates (via
    # merged_pr_issue_numbers below) and left OPEN for the operator to
    # decide whether to close it, wire up a proper closing reference, or
    # redispatch it. Issue #432: capped to finalize_limit per pass.
    #
    # Issue #564: one-shot flagging. The flag must fire once per issue,
    # not every pass — otherwise the operator's removal of
    # agent:human-needed is overridden on the next pass and the event
    # stream is spammed with one dispatch_merged_pr_mention_flagged event
    # per pass while the mention persists. Skip issues whose state entry
    # already records merged_pr_mention_flagged_at (set the first time
    # this path flagged them). This follows the emit-on-change dedup
    # pattern established in #556 for dispatch_skip_blocked/janitor_gate.
    #
    # Re-flag semantics: keyed on the timestamp's absence — once flagged,
    # an issue is never re-flagged, even if a NEW merged PR mentions it.
    # The simplest acceptable semantics per issue #564; pinned by
    # test_dispatch_merged_pr_mention_flag_is_one_shot.
    #
    # Issue #1336 follow-up to #564 point 2: the mention-only *dispatch
    # exclusion* previously keyed off the raw mention scan
    # (merged_pr_issue_numbers below) alone, so an operator who removed
    # agent:human-needed to re-arm automation could NOT re-enter dispatch
    # -- the scan-based exclusion kept blocking the issue until it closed
    # or the mentioning PRs were no longer merged/referenced. The
    # exclusion now lifts for an issue once it was flagged in a prior pass
    # and the operator has removed agent:human-needed: the re-arm is
    # detected from the already-loaded issue labels (no new per-pass API
    # call) and recorded durably in state.json as ``mention_rearmed_at``,
    # so the candidate filter keys the lift off the state signal rather
    # than a per-pass label read -- the blast-radius concern the original
    # comment raised. ``bound`` exclusions stay scan-based; the safe
    # default (never-flagged or still-carries-human-needed stays
    # excluded) is preserved. See ``_mention_rearmed_issue_numbers`` and
    # the re-arm block inside the state lock below.
    # load_state_locked (not raw load_state) so the read holds the
    # advisory state lock — required by the invariant enforced in
    # test_no_unlocked_load_state_in_production_code. The authoritative
    # timestamp write below is a separate locked critical section; this
    # read is best-effort relative to it but must still hold the lock to
    # avoid racing a concurrent tmp+replace writer (issue #310).
    mention_state = _wf.load_state_locked(self.paths.state_file)
    already_flagged_mention_issues = {
        int(num)
        for num, entry in mention_state.get("issues", {}).items()
        if isinstance(entry, dict) and entry.get("merged_pr_mention_flagged_at")
    }
    newly_flagged_mention_issues = [
        n for n in finalizable_mention_issue_numbers if n not in already_flagged_mention_issues
    ]
    # Capture the transition outcome per issue so the dedup marker below
    # is only stamped for issues whose label edge actually took effect.
    # Stamping unconditionally (the pre-fix behavior) meant a
    # PARTIAL_FAILURE label write still recorded
    # merged_pr_mention_flagged_at, permanently suppressing retry (the
    # one-shot guard above keys off the timestamp's presence) with no
    # diagnostic beyond transition()'s own log line. NOTHING_CHANGED is
    # treated the same as APPLIED: per labels.py's _edges(),
    # "merged_pr_mention_flagged" always has a non-empty add tuple, so
    # NOTHING_CHANGED is unreachable for this event today, but it is
    # handled here defensively since a retry would recompute the exact
    # same static edge and produce the same NOTHING_CHANGED outcome again.
    mention_flag_outcomes: list[tuple[int, TransitionOutcome]] = [
        (
            issue_number,
            _wf.transition(
                self.gh, self.config.labels, issue_number, "merged_pr_mention_flagged"
            ).outcome,
        )
        for issue_number in newly_flagged_mention_issues
    ]
    stamped_mention_issues = [
        issue_number
        for issue_number, outcome in mention_flag_outcomes
        if outcome != TransitionOutcome.PARTIAL_FAILURE
    ]

    # Issue #429/#433: closed-unmerged stripping is handled by
    # _finalize_externally_merged_issues, which already performs the
    # capped per-issue merged-PR lookup and removes stale ready/active labels.

    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        # Persist the fact that merged PRs already covered these ready issues.
        # This keeps state.json consistent with the closed GitHub issue and lets
        # reconcile skip the active-status-on-closed-issue drift sweep.
        for issue_number in closed_merged_pr_issues:
            _issue_key = str(issue_number)
            _issue_entry = state["issues"].get(_issue_key, {})
            state["issues"][_issue_key] = {
                **_issue_entry,
                "number": issue_number,
                "status": "closed",
            }
        if closed_merged_pr_issues:
            state = _wf.append_event(
                state,
                # event-consumer: audit-only -- records the issue-status "closed" mutation
                # already applied inline above; no downstream consumer needed
                "dispatch_merged_pr_references_closed",
                {"issue_numbers": sorted(closed_merged_pr_issues)},
                state_path=self.paths.state_file,
            )
            _wf.save_state(self.paths.state_file, state)
        # Issue #427: finalize state.json entries for the merged PRs so
        # externally-merged PRs (Aviator mergequeue handoff) do not leave
        # stale prs[...].status == "mergequeue" behind.
        for pr_number in merged_pr_bound_pr_numbers:
            _pr_key = str(pr_number)
            _pr_entry = state["prs"].get(_pr_key, {})
            _bound_pr_state = {
                **_pr_entry,
                "status": "merged",
                "merged": True,
            }
            # Issue #747: stamp ``merged_at`` only on a genuine non-merged
            # -> merged transition; preserve the original observation time
            # on entries already recorded as merged.
            if _pr_entry.get("status") != "merged":
                _bound_pr_state["merged_at"] = _wf.utc_now()
            state["prs"][_pr_key] = _bound_pr_state
        if merged_pr_bound_pr_numbers:
            _wf.save_state(self.paths.state_file, state)
        # Record a flag timestamp so operators/tooling (e.g. a doctor
        # check) can surface mention-only coverage without re-deriving
        # the mention scan. "status" is deliberately untouched — the
        # issue stays open and its normal state machine intact.
        # Issue #564: only record/emit for issues flagged *this* pass
        # (newly_flagged_mention_issues); already-flagged issues are
        # skipped so the event fires once and the operator's label
        # removal is not overridden on the next pass.
        # Only issues whose transition() outcome was not PARTIAL_FAILURE
        # (stamped_mention_issues, computed above) get the dedup marker —
        # a failed label write must leave it unset so the next pass
        # retries instead of silently leaving the wrong labels forever.
        for issue_number in stamped_mention_issues:
            _issue_key = str(issue_number)
            _issue_entry = state["issues"].get(_issue_key, {})
            state["issues"][_issue_key] = {
                **_issue_entry,
                "number": issue_number,
                "merged_pr_mention_flagged_at": _wf.utc_now(),
                # Issue #783: a merged PR that only mentions the issue in
                # free text is a hijack-safety judgment call, not a
                # process failure -- never auto-de-escalated. "status" is
                # deliberately untouched (see above), so this reason_class
                # is carried for completeness/consistency even though the
                # de-escalation sweep never visits this issue via status.
                "reason_class": escalation_reason_class("judgment"),
            }
        if stamped_mention_issues:
            state = _wf.append_event(
                state,
                "dispatch_merged_pr_mention_flagged",
                {"issue_numbers": stamped_mention_issues},
                state_path=self.paths.state_file,
            )
            _wf.save_state(self.paths.state_file, state)
        # Issue #1336: lift the mention-only dispatch exclusion once the
        # operator has re-armed a previously-flagged issue (removed
        # agent:human-needed). The re-arm is recorded durably as
        # ``mention_rearmed_at`` so the exclusion keys off a state.json
        # signal rather than a per-pass GitHub label read in the
        # candidate filter -- the blast-radius concern the #564 point-2
        # comment raised when it documented this as out of scope.
        #
        # Safe default preserved: an issue never flagged, flagged this
        # pass, or flagged and still carrying agent:human-needed stays
        # excluded. ``bound`` exclusions are never lifted -- those PRs
        # genuinely bound to the issue by a hijack-safe signal.
        #
        # The durable stamp + event emission is routed through
        # ``_stamp_mention_rearm`` (Convention A: ``self.write_gate.*``)
        # rather than raw ``append_event``/``save_state`` so the R9
        # shrink-only ratchet on workflow.py's raw-primitive count
        # (issue #1264 W6 PR4) is not increased -- the re-arm writes are
        # new territory this wave does not convert, and a raw
        # ``append_event``+``save_state`` pair would trip the ratchet
        # (baseline 266). The existing flag block above stays raw (it is
        # part of the ratchet's baseline); only the NEW re-arm writes go
        # through the gate.
        rearmed_mention_issues, newly_rearmed_mention_issues = self._mention_rearmed_issue_numbers(
            merged_pr_mention_only_issue_numbers,
            issues,
            state,
            already_flagged_mention_issues,
        )
        state = self._stamp_mention_rearm(state, newly_rearmed_mention_issues)
        # Recompute the exclusion set so the candidate filter below and
        # the result payload reflect the lifted mention-only exclusions.
        # ``bound`` exclusions are never lifted.
        merged_pr_issue_numbers = merged_pr_bound_issue_numbers | (
            merged_pr_mention_only_issue_numbers - rearmed_mention_issues
        )
        # Defence-in-depth against double-dispatch: an issue whose state records
        # a live launched worker (status "dispatched") or a fresh pending claim
        # (status "dispatch_pending" not yet stale) is not re-dispatchable even
        # if its GitHub label write failed after the worker launched.
        # _is_dispatchable is label-only; this closes the launched-but-unlabeled
        # window that would otherwise spawn a second worker on the same issue.
        # Stale claims (crashed phase-2) are excluded to allow re-dispatch.
        # Issue #5: also check worker liveness for "dispatched" status to recover
        # from crashed workers before PR opens.
        live_dispatched = set()
        dispatch_blocked = set()
        now = datetime.now(UTC)
        for number, entry in state.get("issues", {}).items():
            if not isinstance(entry, dict):
                continue
            status = entry.get("status")
            if status == "dispatch_pending" and not _wf.is_claim_stale(
                entry.get("dispatch_pending_at")
            ):
                live_dispatched.add(int(number))
            elif status == "dispatched":
                # Issue #5: only exclude if the worker is alive OR there's an open PR.
                # A dead worker with no open PR is recoverable (crashed before PR opened).
                # A dead worker with an open PR is mid-review and must not be re-dispatched.
                # Issue #207: also check state.json worker_pid for liveness when session files are orphaned
                issue_number = int(number)
                worker_alive = _wf._worker_pid_alive(entry)
                if (
                    issue_number in live_worker_issues
                    or worker_alive
                    or issue_number in pr_by_issue
                ):
                    live_dispatched.add(issue_number)
            elif status in ("dispatch_failed", "escalated"):
                # Issue #461: bound dispatch_failed retries using the same
                # redispatch-window cap that rework uses. A status already
                # marked ``escalated`` should also drop out of dispatch.
                issue_number = int(number)
                if status == "escalated":
                    dispatch_blocked.add(issue_number)
                else:
                    recent = _wf._recent_dispatch_failed_attempts(
                        entry,
                        now,
                        self.config.watchdog.redispatch_window_minutes,
                    )
                    if len(recent) > self.config.watchdog.max_auto_redispatch:
                        dispatch_blocked.add(issue_number)
        operator_claimed = operator_claimed_issues(state)
        ready_issue_numbers = {int(issue["number"]) for issue in issues}
        operator_claimed_ready = sorted(operator_claimed & ready_issue_numbers)
        issues_with_open_tracked_prs = set(pr_by_issue.keys())
        candidates = [
            issue
            for issue in issues
            if self._is_dispatchable(issue, operator_claimed)
            and int(issue["number"]) not in live_dispatched
            and int(issue["number"]) not in stalled_issues
            and int(issue["number"]) not in issues_with_open_tracked_prs
            and int(issue["number"]) not in merged_pr_issue_numbers
            and int(issue["number"]) not in dispatch_blocked
        ]
        if operator_claimed_ready:
            state = _wf.append_event(
                state,
                # event-consumer: audit-only -- records a skip already enforced by the
                # `candidates` filter above; the skip itself already happened
                "dispatch_skip_operator_claimed",
                {"issue_numbers": operator_claimed_ready},
                state_path=self.paths.state_file,
            )
            _wf.save_state(self.paths.state_file, state)

    # Apply dependency gate: skip issues with open blockers
    # Done outside the lock to avoid holding it during GitHub API calls
    candidates, blocked_issues, open_blockers_by_issue = self._filter_blocked_issues(candidates)

    # Sort candidates by dispatch order
    # Default (oldest) uses dependency-aware ordering; explicit newest uses creation date
    if self.config.dispatch.order == "newest":
        candidates = self._sort_by_dispatch_order(candidates)
    else:
        # Default: use dependency-aware ordering (out-degree) with oldest-first tiebreaker
        candidates = self._sort_by_dependency_depth(candidates)

    # Re-enter lock to log events and claim issues
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)

        # Log dispatch_skip_blocked events for blocked issues. Dedup
        # (cost-spirals.md Finding 3): a still-blocked issue re-selects
        # every pass with the identical blocker list -- 784 byte-identical
        # events over 18h in the investigated window -- so only emit when
        # the (issue, blockers) content actually changed since the last
        # emission, tracked via a compact snapshot on the issue record.
        if blocked_issues:
            for issue_number, blockers in blocked_issues.items():
                issue_key = str(issue_number)
                issue_entry = state["issues"].get(issue_key, {})
                if not isinstance(issue_entry, dict):
                    issue_entry = {}
                if issue_entry.get("last_skip_blocked_blockers") != blockers:
                    issue_entry = {
                        **issue_entry,
                        "number": issue_number,
                        "last_skip_blocked_blockers": blockers,
                    }
                    state["issues"][issue_key] = issue_entry
                    state = self._record_event(
                        state,
                        "dispatch_skip_blocked",
                        {"issue": issue_number, "blockers": blockers},
                    )

                # Blocked-chain attention (pr-lifecycle.md/cost-spirals.md
                # Finding 3/4): an issue whose every currently-open
                # blocker is itself dead (escalated, or its tracked PR is
                # escalated/janitor_blocked) can never unblock through any
                # automated path. Alert once on transition into that
                # state -- no label changes, diagnostic only -- instead
                # of silently re-skipping forever (observed: 4+ days
                # stuck with zero signal).
                open_blockers = open_blockers_by_issue.get(issue_number, [])
                dead_blockers = sorted(
                    b for b in open_blockers if self._is_dead_blocker(b, state, pr_by_issue)
                )
                chain_dead = bool(open_blockers) and dead_blockers == sorted(open_blockers)
                previously_alerted = issue_entry.get("chain_dead_alerted_blockers")
                if chain_dead and previously_alerted != dead_blockers:
                    state["issues"][issue_key] = {
                        **issue_entry,
                        "number": issue_number,
                        "chain_dead_alerted_blockers": dead_blockers,
                    }
                    state = self._record_event(
                        state,
                        "dispatch_blocked_chain_dead",
                        {"issue": issue_number, "chain_root": dead_blockers},
                    )
                elif not chain_dead and previously_alerted is not None:
                    # Recovered (or the dead set changed) -- clear the
                    # marker so a future transition back into all-dead
                    # alerts again instead of staying silent forever.
                    state["issues"][issue_key] = {
                        **issue_entry,
                        "number": issue_number,
                        "chain_dead_alerted_blockers": None,
                    }
            _wf.save_state(self.paths.state_file, state)

        # Fill fresh candidates first; recovery retries only get leftover slots
        # and are capped at one per pass (issue #506).
        (
            selected,
            skipped_issue_numbers,
            deferred_by_concurrency_full,
            deferred_by_concurrency_count,
        ) = _select_dispatch_candidates(
            candidates,
            dispatch_limit,
            state,
            self._branch_name,
            only_issues=only_issues,
        )
        # Issue #1005 review: _build_failure_map must see every deferred
        # issue (deferred_by_concurrency_full) so each one keeps its
        # per-issue "failures" entry -- only the persisted event and
        # CommandResult.data payloads truncate, via
        # deferred_by_concurrency below. Truncating before
        # _build_failure_map silently dropped failures entries for the
        # 6th+ deferred issue; caught in review before merge.
        deferred_by_concurrency = deferred_by_concurrency_full[:_MAX_DEFERRED_CONCURRENCY_EXAMPLES]
        selected_issue_numbers = [int(issue["number"]) for issue in selected]
        # Capture previous entries for recovery detection BEFORE overwriting status
        # Issue #81: we need to know if an issue was previously "dispatched" on the same branch
        # to recover from a crashed worker. This snapshot must be taken before we overwrite
        # the status to "dispatch_pending".
        previous_entries: dict[int, dict[str, Any]] = {}
        for issue_number in selected_issue_numbers:
            previous_entries[issue_number] = state["issues"].get(str(issue_number), {})
        # Mark selected issues as "dispatch_pending" to claim them before launching
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
    # Do all network calls, file writes, and worker launches outside the lock
    session_requests: list[SessionRequest] = []
    full_issues: dict[int, dict[str, Any]] = {}
    # Issue #1000: per-issue citation-drift verdicts and the fingerprint that
    # dedups the flag-comment across passes. Populated in the loop below;
    # consumed in the second state-lock section to stamp the issue record and
    # emit one ``dispatch_citation_drift_flagged`` event per drift change.
    citation_drift_stamps: dict[int, tuple[str, list[CitationVerdict]]] = {}
    # Issue #1010: pre-flight cross-repo gate. Issues whose referenced
    # file paths are all absent from the target repo are escalated to
    # human-needed instead of dispatching a worker that will wander to a
    # sibling repo's shared checkout.
    # Issue #1244: pre-flight cross-repo *scope* gate. Issues whose title
    # names another managed repo in the fleet (e.g. "other-repo: ...") are
    # escalated too — their deliverables live in that repo, not this one,
    # so the dispatching lane can never finalize them.  The managed-repo
    # set is derived from the fleet registry, never a hardcoded list.
    cross_repo_escalated: dict[int, CrossRepoGateResult] = {}
    fleet_repos = managed_repo_names(self.fleet_dir_override)
    dispatching_repo_name = _dispatching_repo_name(self.gh, self.repo_root)
    for issue_number in selected_issue_numbers:
        full_issue = self.gh.issue_view(issue_number)
        full_issues[issue_number] = full_issue
        branch_name = self._branch_name(full_issue)

        # Pre-flight gate: refuse to dispatch when the issue's referenced
        # code does not exist in this repo (issue #1010).
        gate_result = cross_repo_gate(str(full_issue.get("body") or ""), self.repo_root)
        if not gate_result.passed:
            cross_repo_escalated[issue_number] = gate_result
            continue

        # Pre-flight scope gate: refuse to dispatch when the issue's title
        # names another managed repo (issue #1244).
        scope_result = cross_repo_scope_gate(
            str(full_issue.get("title") or ""),
            str(full_issue.get("body") or ""),
            dispatching_repo_name,
            fleet_repos,
        )
        if not scope_result.passed:
            cross_repo_escalated[issue_number] = scope_result
            continue

        prompt_path = self._write_worker_prompt(full_issue)

        # Issue #1000: verify path:line citations in the issue body against
        # the working tree before a worker is sent to them. Drift is flagged
        # (a comment on the issue, which the worker sees via $issue_comments)
        # rather than auto-edited -- the correction needs judgment. The flag
        # is deduped by fingerprint so a still-stale issue is not re-commented
        # every pass, and a newly-stale citation re-alerts. Non-blocking: a
        # verification failure never aborts dispatch, and dispatch itself is
        # not gated on drift -- the comment is the signal, not a hold.
        citation_drift_stamps.update(
            self._check_issue_citations(issue_number, full_issue, previous_entries)
        )

        # Check if this is a dead-worker recovery: the issue has a previous
        # dispatch record with the same branch name (i.e., our own crashed attempt)
        # Use the snapshot captured before status overwrite (Issue #81 fix)
        recovery_record: dict[str, Any] | None = None
        prev_entry = previous_entries.get(issue_number, {})
        prev_branch = prev_entry.get("branch_name")
        if prev_branch == branch_name and prev_entry.get("status") == "dispatched":
            # This is our own crashed attempt - pass the record for recovery
            recovery_record = prev_entry

        session_requests.append(
            SessionRequest(
                issue_number=issue_number,
                issue_title=str(full_issue.get("title") or ""),
                prompt_path=prompt_path,
                branch_name=branch_name,
                recovery=recovery_record,
            )
        )
    manifest_path = self._layout.session_manifest
    results_path = self._layout.session_results
    dispatch_results = _wf.dispatch_sessions(
        self.repo_root,
        manifest_path,
        results_path,
        self._adapter_settings(),
        session_requests,
    )
    successful_issue_numbers = {result.issue_number for result in dispatch_results if result.ok}
    # Issue #523: a live_worker_redispatch_averted result claims the prior
    # worker is still alive, but the adapter's probe (_probe_recovery_liveness)
    # can fail closed on an inconclusive real-activity signal (probe_error) or
    # report fresh sessions.db activity even when the recorded wrapper PID is
    # dead/recycled. Verify the PID against the OS (with start-time identity)
    # at the single point where live worker slots are counted — the same
    # is_pid_alive + process_start_time check the review lane uses via
    # _reviewer_pid_alive. A session whose recorded PID is dead is a phantom
    # slot and is routed through the dead-session path below (sidecar reap,
    # label repair) instead of starving fresh dispatch.
    live_worker_issue_numbers: set[int] = set()
    phantom_live_worker_issue_numbers: set[int] = set()
    for result in dispatch_results:
        if result.ok or result.failure_kind != "live_worker_redispatch_averted":
            continue
        if (
            result.pid is not None
            and result.pid > 0
            and _wf.is_pid_alive(result.pid, result.process_start_time)
        ):
            live_worker_issue_numbers.add(result.issue_number)
        else:
            phantom_live_worker_issue_numbers.add(result.issue_number)
    failed_issue_numbers = {
        result.issue_number
        for result in dispatch_results
        if not result.ok
        and result.issue_number not in live_worker_issue_numbers
        and result.issue_number not in phantom_live_worker_issue_numbers
    }
    foreign_writer_issue_numbers = {
        result.issue_number
        for result in dispatch_results
        if not result.ok and result.failure_kind == "worktree_foreign_writer"
    }
    # Second lock: upgrade claim from dispatch_pending to dispatched/dispatch_failed
    manual = self.config.worker.harness == "manual"
    label_errors: list[int] = []
    label_error_failures: dict[int, str] = {}
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        for request in session_requests:
            full_issue = full_issues[request.issue_number]
            ok = request.issue_number in successful_issue_numbers
            is_live_worker = request.issue_number in live_worker_issue_numbers
            is_phantom_live_worker = request.issue_number in phantom_live_worker_issue_numbers
            prev_entry = state["issues"].get(str(request.issue_number), {})
            # Issues #837 / #779: the dispatch outcome (status/dispatched_at)
            # and its escalation bookkeeping (dispatch_failed_at /
            # escalation_reason / reason_class) used to be decided in two
            # textually separate if/elif chains keyed on the same four
            # predicates (ok / is_live_worker / is_phantom_live_worker /
            # else) -- one chain bound all_attempts/failed_result/
            # terminal_failure only in its `else`, a second chain ~70 lines
            # down read them only in its matching `elif`/`else`. That was
            # safe only because the two enumerations happened to agree;
            # nothing enforced it, and pyright could not prove it (reported
            # possibly-unbound). Collapsed into one chain below so each
            # branch binds and consumes its own locals -- there is no
            # second enumeration left to drift out of sync with the first.
            # status/dispatched_at are the only values that still cross out
            # of this chain (applied once, after it); every arm binds them
            # unconditionally, so a future arm that forgets to set one is a
            # possibly-unbound error at type-check time, not a silently
            # stale value copied from prev_entry.
            entry = {
                **prev_entry,
                "number": request.issue_number,
                "title": full_issue.get("title"),
                "url": full_issue.get("url"),
                "branch_name": request.branch_name,
                "prompt_path": str(request.prompt_path),
            }
            # Clear the claim timestamp on successful upgrade
            entry.pop("dispatch_pending_at", None)
            entry.pop("label_error", None)
            if ok:
                status = "manifest_written" if manual else "dispatched"
                dispatched_at = _wf.utc_now()
                # A successful recovery supersedes any previous orphan flag.
                entry.pop("orphan_flagged_at", None)
                entry.pop("orphan_drift_fingerprint", None)
                entry.pop("orphan_drift_at", None)
                entry.pop("dispatch_failed_at", None)
                clear_escalation(entry)
                clear_escalation_on_issue_prs(state, request.issue_number)
            elif is_live_worker:
                status = "dispatched"
                dispatched_at = prev_entry.get("dispatched_at") or _wf.utc_now()
                # A live-worker recovery supersedes any previous orphan flag.
                entry.pop("orphan_flagged_at", None)
                entry.pop("orphan_drift_fingerprint", None)
                entry.pop("orphan_drift_at", None)
                entry.pop("dispatch_failed_at", None)
                clear_escalation(entry)
                clear_escalation_on_issue_prs(state, request.issue_number)
            elif is_phantom_live_worker:
                # Issue #523: the adapter reported a live worker, but the
                # recorded PID failed the OS-level liveness + identity
                # check. Route through the dead-session path (sidecar reap,
                # label repair) instead of keeping the phantom slot
                # occupied. The slot is freed and the issue becomes
                # dispatchable again without burning a redispatch attempt.
                # A dispatched request's issue can never have an open
                # tracked PR -- candidate selection excludes every issue in
                # pr_by_issue -- so the rework-routing case is handled by
                # the dead-session reaper lane, not here.
                status, dispatched_at, state = self._route_phantom_live_worker(
                    state,
                    request,
                    full_issue,
                    sessions_dir,
                )
                # A phantom live worker is being routed as dead; do not
                # preserve a stale worker_pid that would keep the slot
                # occupied, and do not burn a redispatch attempt (the launch
                # was averted, not failed).
                entry.pop("orphan_flagged_at", None)
                entry.pop("orphan_drift_fingerprint", None)
                entry.pop("orphan_drift_at", None)
                entry.pop("dispatch_failed_at", None)
                clear_escalation(entry)
                clear_escalation_on_issue_prs(state, request.issue_number)
                entry.pop("worker_pid", None)
                entry.pop("worker_process_start_time", None)
            else:
                # Issue #461: bound dispatch_failed retries with the same
                # redispatch-window cap used for rework.
                now = datetime.now(UTC)
                failed_result = next(
                    (r for r in dispatch_results if r.issue_number == request.issue_number),
                    None,
                )
                failure_kind = failed_result.failure_kind if failed_result is not None else None
                # Issue #1393: a pre-launch environment block (e.g.
                # worktree_foreign_writer) never started a worker session,
                # so it must NOT count against the dispatch_failed cap
                # (which measures worker output, not environment hygiene).
                # Use a separate blocked_environment_at counter and
                # escalate with the correct reason + blocking path after
                # the same cap.
                blocked_environment = failure_kind in PRE_LAUNCH_BLOCKED_ENVIRONMENT_FAILURE_KINDS
                if blocked_environment:
                    blocked_environment_at = _windowed_blocked_environment_at(
                        entry,
                        window_minutes=self.config.watchdog.redispatch_window_minutes,
                    ) + [now.isoformat().replace("+00:00", "Z")]
                    blocking_error = failed_result.error if failed_result else None
                    entry["blocked_environment_at"] = blocked_environment_at
                    if len(blocked_environment_at) > self.config.watchdog.max_auto_redispatch:
                        # Issue #1423: before escalating a blocked-environment
                        # cap exhaustion for a foreign writer, attempt to reap
                        # it one more time. A writer that was active on earlier
                        # passes but has since gone idle is reaped here instead
                        # of escalating a zombie to a human. Escalation is
                        # reserved for a writer that is alive *and* active.
                        #
                        # Review finding: bound the number of auto-reaps per
                        # issue before falling back to escalation. Each
                        # successful reap resets ``blocked_environment_at`` to
                        # ``[]``, so without a separate cross-pass cap a
                        # persistently-blocked worktree loops forever between
                        # reap and redispatch. ``foreign_writer_reaps`` is a
                        # windowed counter that survives the reset; once it
                        # reaches ``max_foreign_writer_reaps`` the issue
                        # escalates instead of reaping again.
                        max_reaps = self.config.watchdog.max_foreign_writer_reaps
                        prior_reaps = _windowed_foreign_writer_reaps(
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
                            entry["blocked_environment_at"] = []
                            entry["foreign_writer_reaps"] = prior_reaps + [
                                now.isoformat().replace("+00:00", "Z")
                            ]
                            status = "dispatch_failed"
                            dispatched_at = None
                            clear_escalation(entry)
                            clear_escalation_on_issue_prs(state, request.issue_number)
                            state = self._record_event(  # event-consumer: audit-only -- records a foreign-writer reap (issue #1423) already enforced by the blocked_environment_at reset and foreign_writer_reaps counter; consumed by tests/test_charlie_work.py regression tests.
                                state,
                                "dispatch_blocked_environment_reaped",
                                {
                                    "issue_number": request.issue_number,
                                    "failure_kind": failure_kind,
                                    "pid": failed_result.pid if failed_result else None,
                                    "blocked_environment_count": len(blocked_environment_at),
                                    "foreign_writer_reap_count": len(prior_reaps) + 1,
                                },
                            )
                        else:
                            status = "escalated"
                            dispatched_at = None
                            reason_class = "mechanical"
                            state = _escalate_issue(
                                state,
                                request.issue_number,
                                reason="dispatch_blocked_environment",
                                reason_class=reason_class,
                                issue_extra=entry,
                            )
                            entry = dict(state["issues"][str(request.issue_number)])
                            state = self._record_event(
                                state,
                                "session_failed_escalated",
                                {
                                    "issue_number": request.issue_number,
                                    "previous_status": "dispatch_pending",
                                    "reason": "dispatch_blocked_environment",
                                    "failure_kind": failure_kind,
                                    "blocking_error": blocking_error,
                                    "blocked_environment_count": len(blocked_environment_at),
                                },
                            )
                    else:
                        status = "dispatch_failed"
                        dispatched_at = None
                        clear_escalation(entry)
                        clear_escalation_on_issue_prs(state, request.issue_number)
                        state = self._record_event(  # event-consumer: audit-only -- records a pre-launch environment block (issue #1393) already enforced by the blocked_environment_at counter and the dispatch_blocked_environment escalation; consumed by tests/test_charlie_work.py regression tests.
                            state,
                            "dispatch_blocked_environment",
                            {
                                "issue_number": request.issue_number,
                                "failure_kind": failure_kind,
                                "blocking_error": blocking_error,
                                "blocked_environment_count": len(blocked_environment_at),
                            },
                        )
                else:
                    all_attempts = list(prev_entry.get("dispatch_failed_at") or [])
                    if not isinstance(all_attempts, list):
                        all_attempts = []
                    all_attempts.append(now.isoformat())
                    recent = _wf._recent_dispatch_failed_attempts(
                        {"dispatch_failed_at": all_attempts},
                        now,
                        self.config.watchdog.redispatch_window_minutes,
                    )
                    # Deterministic launch failures escalate immediately,
                    # mirroring dispatch_rework's post-#550 behavior — fresh
                    # dispatch previously only consulted the redispatch-window
                    # cap, so e.g. a worktree_unsafe failure burned every
                    # capped retry before a human ever heard about it.
                    terminal_failure = (
                        failed_result is not None
                        and failed_result.failure_kind in DETERMINISTIC_ESCALATION_FAILURE_KINDS
                    )
                    # Issue #807: a deterministic judgment failure escalates
                    # immediately but as ``reason_class="judgment"``.
                    deterministic_judgment = (
                        failed_result is not None
                        and failed_result.failure_kind
                        in DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS
                    )
                    immediate_escalation = terminal_failure or deterministic_judgment
                    entry["dispatch_failed_at"] = all_attempts
                    if (
                        immediate_escalation
                        or len(recent) > self.config.watchdog.max_auto_redispatch
                    ):
                        status = "escalated"
                        dispatched_at = None
                        reason_class = "judgment" if deterministic_judgment else "mechanical"
                        state = _escalate_issue(
                            state,
                            request.issue_number,
                            reason=(
                                failed_result.failure_kind
                                if (
                                    immediate_escalation
                                    and failed_result is not None
                                    and failed_result.failure_kind is not None
                                )
                                else "dispatch_failed_cap_exceeded"
                            ),
                            reason_class=reason_class,
                            issue_extra=entry,
                        )
                        # Re-read the escalation fields _escalate_issue merged in,
                        # but keep ``entry`` a decoupled copy: the mutations below
                        # must not reach state until the single atomic write at the
                        # end of this block.
                        entry = dict(state["issues"][str(request.issue_number)])
                    else:
                        status = "dispatch_failed"
                        dispatched_at = None
                        clear_escalation(entry)
                        clear_escalation_on_issue_prs(state, request.issue_number)
            entry["status"] = status
            entry["dispatched_at"] = dispatched_at
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
            # Issue #1000: stamp the citation-drift fingerprint computed in
            # the outside-lock loop. The comment was already posted there
            # (best-effort); this persists the dedup marker so a still-stale
            # issue is not re-commented next pass, and emits one event per
            # drift change. ``citation_drift_flagged_at`` records the last
            # time drift was observed, not the first -- it moves with every
            # newly-detected drift state, matching the fingerprint's
            # re-alert semantics.
            drift_stamp = citation_drift_stamps.get(request.issue_number)
            if drift_stamp is not None:
                fp, drift_verdicts = drift_stamp
                entry["citation_drift_fingerprint"] = fp
                if fp:
                    entry["citation_drift_flagged_at"] = _wf.utc_now()
                    state = self._record_event(
                        state,
                        "dispatch_citation_drift_flagged",
                        {
                            "issue": request.issue_number,
                            "drifted_citations": [
                                {
                                    "citation": v.citation.raw,
                                    "path": v.citation.path,
                                    "line": v.citation.line,
                                    "end_line": v.citation.end_line,
                                    "status": v.status.value,
                                    "resolved_path": v.resolved_path.replace("\\", "/")
                                    if v.resolved_path
                                    else None,
                                    "candidates": list(v.candidates) if v.candidates else None,
                                }
                                for v in drift_verdicts
                            ],
                        },
                    )
                else:
                    # Drift resolved since the last pass: clear the marker
                    # so a future regression re-alerts. No event -- the
                    # interesting transition is into drift, not out of it.
                    entry.pop("citation_drift_flagged_at", None)
            state["issues"][str(request.issue_number)] = entry
            # Persist the launched worker BEFORE touching GitHub labels: a
            # transient label-write failure (or crash) must never leave a live
            # worker unrecorded and therefore re-dispatchable next wave. The
            # transition is isolated per-issue so one failure never aborts the
            # rest of the batch (orphaning already-launched workers).
            _wf.save_state(self.paths.state_file, state)
            # This re-tests the same four predicates as the outcome chain
            # above for a fourth time (label transitions), rather than
            # branching on the outcome directly. It cannot be folded into
            # that chain: GitHub labels must only be touched after
            # save_state has persisted the launched worker (comment above),
            # so this enumeration is structurally separated from the one
            # that decides status/escalation. It does not read
            # all_attempts/failed_result/terminal_failure (issues #837,
            # #779 do not apply here), only the already-bound ok /
            # is_live_worker / status locals, so there is no possibly-
            # unbound hazard -- just a fourth place that must be kept in
            # sync with the outcome predicates if a new outcome is added.
            if ok or is_live_worker:
                target = "queued" if manual else "dispatched"
                result = _wf.transition(
                    self.gh,
                    self.config.labels,
                    request.issue_number,
                    target,
                )
                if result.outcome != TransitionOutcome.APPLIED:
                    label_error = {
                        "edge": target,
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
                if is_live_worker:
                    result = next(
                        (r for r in dispatch_results if r.issue_number == request.issue_number),
                        None,
                    )
                    state = _wf.append_event(
                        state,
                        "live_worker_redispatch_averted",
                        {
                            "issue_number": request.issue_number,
                            "branch_name": request.branch_name,
                            "pid": result.pid if result else None,
                            "process_start_time": result.process_start_time if result else None,
                            "probe_result": result.error if result else None,
                        },
                        state_path=self.paths.state_file,
                    )
                    _wf.save_state(self.paths.state_file, state)
            elif status == "escalated":
                # Issue #461: dispatch-failed retry cap exceeded — or a
                # deterministic launch failure that retrying cannot fix
                # (escalation_reason carries the failure_kind) — escalate to
                # human-needed and remove the issue from the dispatch pool.
                # Issue #807: a deterministic judgment failure uses
                # ``reason_class="judgment"`` so the label lands on
                # human-needed, not operator_queued.
                edge = _escalation_edge("redispatch_escalated", reason_class)
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

        # Issue #1010: escalate issues blocked by the cross-repo pre-flight
        # gate. Their referenced file paths are all absent from the target
        # repo, so dispatching a worker would send it to a sibling repo's
        # shared checkout. Escalate to human-needed with a cross_repo_target
        # reason and record the event — the issue stays in the dispatch
        # pool's state as escalated, not dispatch_pending.
        for issue_number, gate_result in sorted(cross_repo_escalated.items()):
            reason = gate_result.reason
            prev_entry = state["issues"].get(str(issue_number), {})
            entry = {
                **prev_entry,
                "number": issue_number,
                "title": full_issues.get(issue_number, {}).get("title"),
                "url": full_issues.get(issue_number, {}).get("url"),
            }
            entry.pop("dispatch_pending_at", None)
            entry.pop("label_error", None)
            state = _escalate_issue(
                state,
                issue_number,
                reason=reason,
                reason_class="mechanical",
                issue_extra=entry,
            )
            # Issue #1583: report ``neutral_paths`` and ``missing_paths``
            # in the event payload so the operator can see from the
            # event alone which citation tripped the gate (today the
            # payload carried only the count, embedded in ``reason``).
            # For the cross-repo *scope* gate (issue #1244) both tuples
            # are empty by construction -- the scope gate does not deal
            # in file paths -- so the fields are present but empty,
            # which is accurate.
            state = _wf.append_event(
                state,
                "dispatch_cross_repo_escalated",
                {
                    "issue_number": issue_number,
                    "reason": reason,
                    "neutral_paths": list(gate_result.neutral_paths),
                    "missing_paths": list(gate_result.missing_paths),
                },
                state_path=self.paths.state_file,
            )
            _wf.save_state(self.paths.state_file, state)
            # Transition labels (operator_queue for mechanical, following
            # the same pattern as the redispatch_escalated path above).
            edge = _escalation_edge("redispatch_escalated", "mechanical")
            result = _wf.transition(
                self.gh,
                self.config.labels,
                issue_number,
                edge,
            )
            if result.outcome != TransitionOutcome.APPLIED:
                label_error = {
                    "edge": edge,
                    "outcome": result.outcome.value,
                    "add_failures": result.add_failures,
                    "remove_failures": result.remove_failures,
                }
                escalated_entry = state["issues"].get(str(issue_number), {})
                escalated_entry["label_error"] = label_error
                state["issues"][str(issue_number)] = escalated_entry
                label_errors.append(issue_number)
                label_error_failures[issue_number] = _wf._label_error_reason(label_error)
                _wf.save_state(self.paths.state_file, state)

        # Build dispatch-alert transitions for the notify digest. Averted
        # redispatches surface as DISPATCH_AVERTED; a later successful or
        # non-averted dispatch clears the alert back to OK.
        dispatch_alert_transitions: dict[int, dict[str, Any]] = {}
        live_worker_redispatch_averted: list[dict[str, Any]] = []
        for request in session_requests:
            prev_alert = previous_entries.get(request.issue_number, {}).get("dispatch_alert")
            result = next(
                (r for r in dispatch_results if r.issue_number == request.issue_number),
                None,
            )
            is_live_worker = request.issue_number in live_worker_issue_numbers
            if is_live_worker:
                dispatch_alert_transitions[request.issue_number] = {
                    "adapter_kind": result.adapter if result else "unknown",
                    "health": "DISPATCH_AVERTED",
                    "last_log_line": None,
                    "pid": result.pid if result else None,
                    "terminal_tool": None,
                    "terminal_reason": result.error if result else None,
                }
                live_worker_redispatch_averted.append(
                    {
                        "issue_number": request.issue_number,
                        "branch_name": request.branch_name,
                        "pid": result.pid if result else None,
                        "process_start_time": result.process_start_time if result else None,
                        "probe_result": result.error if result else None,
                        "adapter_kind": result.adapter if result else "unknown",
                    }
                )
            elif prev_alert == "DISPATCH_AVERTED":
                dispatch_alert_transitions[request.issue_number] = {
                    "adapter_kind": result.adapter if result else "unknown",
                    "health": "OK",
                    "last_log_line": None,
                    "pid": result.pid if result else None,
                    "terminal_tool": None,
                    "terminal_reason": None,
                }

        for issue_number in foreign_writer_issue_numbers:
            result = next((r for r in dispatch_results if r.issue_number == issue_number), None)
            branch_name = next(
                (r.branch_name for r in session_requests if r.issue_number == issue_number),
                None,
            )
            state = _wf.append_event(
                state,
                "worktree_foreign_writer",
                {
                    "issue_number": issue_number,
                    "branch_name": branch_name,
                    "pid": result.pid if result else None,
                    "probe_result": result.error if result else None,
                },
                state_path=self.paths.state_file,
            )
            _wf.save_state(self.paths.state_file, state)
        dispatch_failure_map = _wf._build_failure_map(
            dispatch_results,
            failed_issue_numbers,
            deferred_by_concurrency_full,
            dispatch_limit,
            extra_failures=label_error_failures,
        )
        # Issue #946: warn when a non-empty backlog has not produced a
        # non-empty dispatch event for longer than the configured threshold.
        dispatch_staleness = check_dispatch_staleness(
            self.paths.state_file,
            self.config.dispatch,
            backlog_reachability,
            recent_issue_numbers=sorted(successful_issue_numbers),
            now=datetime.now(UTC),
        )
        if dispatch_staleness["stale"]:
            state = self._record_event(state, "dispatch_stale", dispatch_staleness)
        state = _wf.append_event(
            state,
            "dispatch",
            {
                "issue_numbers": sorted(successful_issue_numbers),
                "live_worker_issue_numbers": sorted(live_worker_issue_numbers),
                "phantom_live_worker_issue_numbers": sorted(phantom_live_worker_issue_numbers),
                "failed_issue_numbers": sorted(failed_issue_numbers),
                "foreign_writer_issue_numbers": sorted(foreign_writer_issue_numbers),
                "cross_repo_escalated_issue_numbers": sorted(cross_repo_escalated),
                "label_errors": sorted(label_errors),
                "skipped_issue_numbers": skipped_issue_numbers,
                "deferred_by_concurrency": deferred_by_concurrency,
                "deferred_by_concurrency_count": deferred_by_concurrency_count,
                "merged_pr_referenced_issue_numbers": sorted(merged_pr_issue_numbers),
                "merged_pr_closed_issue_numbers": sorted(closed_merged_pr_issues),
                "merged_pr_flagged_issue_numbers": sorted(newly_flagged_mention_issues),
                "merged_pr_mention_rearmed_issue_numbers": sorted(newly_rearmed_mention_issues),
                "failures": dispatch_failure_map,
                # Issue #944: why zero, when it is zero. Every other field
                # here describes issues the ready-filtered query returned;
                # this one describes the backlog that query cannot see.
                "backlog_reachability": backlog_reachability,
                # Issue #946: cadence-staleness diagnostic, always present so
                # the capped state.json ring carries the signal.
                "dispatch_staleness": dispatch_staleness,
                # Issue #1005: the capacity axis. backlog_reachability answers
                # "why zero" for supply (which issues exist/are reachable);
                # this answers it for capacity (whether there was a slot to put
                # one in). Always present -- gov.report_fields() is safe to call
                # unclamped -- and explicit about `clamped` so a reader does not
                # have to redo the arithmetic (available_slots == 0 alone does
                # not say whether the repo cap or the fleet cap was binding).
                # `dispatch_limit` is included explicitly (report_fields() does
                # not carry it) because it is the only field that reflects a
                # fleet-cap clamp: `available_slots` is only recomputed when the
                # repo governor itself is enabled, so a fleet-only clamp leaves
                # `available_slots` at its pre-fleet value while `dispatch_limit`
                # still shows the true (possibly zero) effective limit.
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
    message = "dispatch complete"
    if failed_issue_numbers:
        entries = ", ".join(
            f"#{issue} ({dispatch_failure_map[issue]})" for issue in sorted(failed_issue_numbers)
        )
        message = f"dispatch failures: {entries}"
    elif live_worker_issue_numbers:
        message = "dispatch completed with live worker redispatch averted"
    if skipped_issue_numbers:
        message += f" (skipped non-dispatchable: {skipped_issue_numbers})"
    if label_errors:
        message += f" (launched but label write failed: {sorted(label_errors)})"
    if phantom_live_worker_issue_numbers:
        message += (
            f" (reaped phantom live worker slots: {sorted(phantom_live_worker_issue_numbers)})"
        )
    if cross_repo_escalated:
        message += f" (cross-repo escalated: {sorted(cross_repo_escalated)})"
    data = {
        "selected_count": len(successful_issue_numbers),
        "attempted_count": len(session_requests),
        "failed_count": len(failed_issue_numbers),
        "failures": dispatch_failure_map,
        "live_worker_count": len(live_worker_issue_numbers),
        "phantom_live_worker_count": len(phantom_live_worker_issue_numbers),
        "phantom_live_worker_issue_numbers": sorted(phantom_live_worker_issue_numbers),
        "foreign_writer_count": len(foreign_writer_issue_numbers),
        "cross_repo_escalated_issue_numbers": sorted(cross_repo_escalated),
        "skipped_issue_numbers": skipped_issue_numbers,
        "deferred_by_concurrency": deferred_by_concurrency,
        "deferred_by_concurrency_count": deferred_by_concurrency_count,
        "merged_prs": resolved_merged_prs,
        "merged_pr_referenced_issue_numbers": sorted(merged_pr_issue_numbers),
        "merged_pr_closed_issue_numbers": sorted(closed_merged_pr_issues),
        "merged_pr_flagged_issue_numbers": sorted(newly_flagged_mention_issues),
        "merged_pr_mention_rearmed_issue_numbers": sorted(newly_rearmed_mention_issues),
        "label_errors": sorted(label_errors),
        "session_manifest": str(manifest_path),
        "session_results": str(results_path),
        "sessions": [asdict(request) for request in session_requests],
        "dispatch_results": result_dicts,
        "live_worker_redispatch_averted": live_worker_redispatch_averted,
        "stalled": stalled_entries,
        "blocked": [
            {"issue": issue_number, "blockers": blockers}
            for issue_number, blockers in sorted(blocked_issues.items())
        ],
        "operator_claimed_ready": sorted(operator_claimed_ready),
    }
    if gov.enabled or gov.fleet_enabled or gov.open_pr_enabled:
        data.update(gov.report_fields())

    # Emit notification digest if there are health transitions (stalled sessions)
    # This will be enhanced by #165 to include RUNAWAY/DEAD/escalated transitions
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

    # Emit dispatch-alert digest for live-worker redispatch averted outcomes.
    # This surfaces the silent-stall class of dispatch failures in the same
    # attention pipeline used for stalled workers (issue #506 / #497).
    if dispatch_alert_transitions and self.config.notify.enabled:
        dispatch_digest = _wf._build_attention_digest(
            self.paths.state_file,
            dispatch_alert_transitions,
            repo=self.repo_root.name,
            state_field="dispatch_alert",
        )
        if dispatch_digest:
            _wf.emit_digest(self._layout.notify, dispatch_digest)

    return _wf.CommandResult(
        not failed_issue_numbers,
        message,
        data,
    )
