"""Dead-worker-reap loop-body delegate moved out of ``OrchestratorApp``.

Track 2 Phase B, leaf L06 (issue #1637; design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). ``_loop_body`` relocated verbatim from
``charlie_work.workflow``; ``workflow_delegation._install_delegates``
re-attaches the top-level ``def`` unwrapped onto ``OrchestratorApp`` (``self``
binds through the descriptor protocol exactly as a lexical method did).

Names reached through ``_wf.`` (module-object seam, design Section 3.1 rule 2,
#1627): ``charlie_work.workflow`` module-level definitions ``CommandResult``
(class), ``_build_attention_digest``, ``_clear_foreign_issue_ref_marker``,
``_detect_and_handle_orphaned_workers``, ``_should_reprobe_foreign_marker``,
``_touch_foreign_issue_ref_marker`` (free functions); and Tier-D names patched
on ``charlie_work.workflow`` by the suite:
``_classify_dead_sessions_and_update_throttle_state``,
``_detect_and_handle_stalled_sessions``,
``_sweep_orphan_processes_for_dead_sessions``, ``emit_digest``,
``linked_issue_number``, ``load_state_locked``, ``utc_now``. All other free
names are imported directly from their defining module (a three-form,
six-alias patch census confirms no test patches any of them on
``charlie_work.workflow``); note ``_detect_stalled_sessions`` (log-mtime stall
detector, unpatched) is the decoy sibling of the ``_wf.``-rebound
``_detect_and_handle_stalled_sessions`` and is imported directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import charlie_work.workflow as _wf
from charlie_work.dead_worker_reap import _detect_stalled_sessions
from charlie_work.escalation import _escalation_flags
from charlie_work.github import (
    GitHubError,
    GitHubNotFoundError,
    is_transient_repo_resolution_failure,
)
from charlie_work.instrumentation import log_event
from charlie_work.notify import AttentionDigest, AttentionEntry
from charlie_work.review_decision import review_decision


def _loop_body(
    self, limit: int | None, *, merge: bool | None, now: datetime | None = None
) -> _wf.CommandResult:
    # Every pass must observe a fresh GitHub snapshot. The list cache
    # dedupes calls within one pass, but a long-running supervisor
    # (charlie fleet supervise) reuses this app -- and therefore one
    # GitHub instance -- across many passes; without this, issues filed
    # or PRs opened after the first pass stay invisible until the daemon
    # restarts (observed live: intake frozen at a stale issue set for the
    # daemon's entire lifetime).
    self.gh.invalidate_list_cache()
    sessions_dir = self._layout.sessions_dir
    # Issue #646: the worker census now logs from inside dispatch() itself
    # (the one chokepoint every dispatch path funnels through, including
    # standalone `work`/`fleet work` which never reach this method) --
    # see dispatch()'s docstring. Not re-logged here to avoid a duplicate
    # census line within the same pass.
    # Unconditional sweep: reap stalled/orphaned sessions even when this pass
    # has zero ready/rework candidates and never reaches dispatch()'s reaper call.
    # The result is handed down to dispatch_rework()/dispatch() below so the
    # sweep runs exactly once per pass — it is the sole writer of Signal-1's
    # inconclusive-probe deferral counter, and re-running it inside each
    # dispatch lane advanced the counter up to 3x per pass, collapsing the
    # max_inconclusive_probe_deferrals "N passes of grace" into a single pass
    # (issue #343 Finding 2).
    # now=now (issue #828): forward this pass's clock so the reaper's
    # rate-limit-defer classification shares one sample with the other
    # cadence-gated lanes below instead of re-sampling independently.
    loop_stalled_entries = _wf._detect_and_handle_stalled_sessions(
        sessions_dir,
        self.paths.state_file,
        self.config,
        write_gate=self.write_gate,
        now=now,
    )
    intake = self.intake()
    # Share a single wave budget between fresh and rework dispatch
    # Rework-first, then fresh fills the remainder
    # Resolve the effective budget once
    effective_limit = limit if limit is not None else self.config.dispatch.default_limit

    # Apply global concurrency governor cap to the total wave budget
    gov = self._apply_concurrency_governor(effective_limit)
    effective_limit = gov.dispatch_limit

    # Classify dead sessions and update throttle state (production loop path)
    # This detects provider throttling from worker deaths and sets cooldown
    # Also reconciles labels for dead sessions with no open PR (issue #118)
    sessions_dir = self._layout.sessions_dir
    # Issue #343 Finding 2: the stall lane at the top of this method
    # (line ~4100) already ran this pass and is the sole writer of the
    # inconclusive-probe deferral counter for a not-alive worker -- tell
    # this lane not to persist it again on top of that write.
    reaped = _wf._classify_dead_sessions_and_update_throttle_state(
        sessions_dir,
        self.paths.state_file,
        self.gh,
        self.config,
        write_gate=self.write_gate,
        persist_inconclusive_probe_counter=False,
        now=now,
        fleet_dir_override=self.fleet_dir_override,
    )

    # Flat-interval Haiku probe for early quota/rate-limit recovery (see
    # docstring): only does real work when a throttle indicator is active.
    # `now` (issue #828) is this pass's single injected clock, forwarded
    # so the probe's own cadence-scheduling samples stay consistent with
    # the rest of this pass instead of independently racing wall clock.
    self._maybe_probe_quota_recovery(now=now)

    # Periodic in-loop reconcile (merge-lane-recovery §6-B): repairs
    # GitHub label / state.json divergence on a fixed cadence instead of
    # only when an operator runs `charlie mop-up --fix`. Placed before
    # the dispatch calls below so labels it repairs (e.g. a stale
    # `needs-rework` on an issue state already marked `escalated`) are
    # visible to this same pass's dispatch decisions, not just the next.
    self._maybe_reconcile_drift(now=now)

    # Issues #863/#815: reclaim superseded, not-yet-started main CI runs
    # every pass -- no runner needed, so unlike the workflow-based
    # reaper this cannot lose the race for the capacity it exists to
    # free. See _maybe_reclaim_superseded_main_ci's docstring.
    self._maybe_reclaim_superseded_main_ci()

    # Issue #783: periodic re-evaluation of `mechanical` escalations --
    # the only automated re-entry from `agent:human-needed` for pure
    # process failures (a dead rework worker, a redispatch cap, a
    # stalled janitor-gate rework, ...) whose underlying PR artifact has
    # since become mergeable and janitor-clean. `judgment` escalations
    # and any pre-existing escalation with no recorded reason_class are
    # untouched by construction (see _maybe_deescalate_mechanical).
    self._maybe_deescalate_mechanical()

    # Sweep for orphan processes in dead session worktrees (issue #139)
    # This catches detached/daemonized processes that survived session kills
    _wf._sweep_orphan_processes_for_dead_sessions(
        sessions_dir, self.paths.state_file, self.config, write_gate=self.write_gate
    )

    # Detect and handle orphaned workers using state.json PID records (issue #207)
    # This fallback detects dead workers even when session sidecar files are orphaned.
    # Pass the review callback so a head-advanced request_changes finding can be
    # routed to the review-pending path instead of being re-emitted as drift.
    _wf._detect_and_handle_orphaned_workers(
        sessions_dir,
        self.paths.state_file,
        self.config,
        self.gh,
        write_gate=self.write_gate,
        review_callback=self.review,
        fleet_dir_override=self.fleet_dir_override,
    )

    # Detect stalled sessions for notification (read-only, stateful via _build_attention_digest)
    stalled_entries = _detect_stalled_sessions(sessions_dir, self.config)
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

    # Issue #706: feed reaped dead-session transitions into the notify
    # digest. ``_detect_stalled_sessions`` is gated on
    # ``watchdog.enabled`` (it returns ``[]`` immediately when watchdog is
    # off), so a deployment that disables watchdog -- e.g. to work around
    # a shim log-mtime blindness, as a sibling repo does -- gets zero stalled
    # entries and the notify sink never fires, even though dead workers
    # ARE reaped by ``_classify_dead_sessions_and_update_throttle_state``
    # above (which is NOT watchdog-gated; see issue #1122). Without this,
    # an operator monitoring ``notify/digest.jsonl`` gets zero signal
    # silently -- the exact symptom in #706. ``setdefault`` preserves any
    # stalled transition already recorded for the same issue (the stall
    # lane's post-mortem terminal_tool/terminal_reason are richer than the
    # reaped entry's failure_kind); in practice the two are mutually
    # exclusive (a session is either alive-stalled or dead-reaped, not
    # both), so this only fills in for sessions the watchdog-gated
    # read-only detection skipped.
    for reaped_entry in reaped:
        reaped_issue = reaped_entry.get("issue_number")
        if reaped_issue is None:
            continue
        health_transitions.setdefault(
            reaped_issue,
            {
                "adapter_kind": reaped_entry.get("adapter_kind", "unknown"),
                "health": "DEAD",
                "last_log_line": None,
                "pid": reaped_entry.get("pid"),
                "terminal_tool": None,
                "terminal_reason": reaped_entry.get("failure_kind"),
            },
        )

    # Emit notification digest if there are health transitions
    if health_transitions and self.config.notify.enabled:
        digest = _wf._build_attention_digest(
            self.paths.state_file,
            health_transitions,
            repo=self.repo_root.name,
        )
        if digest:
            _wf.emit_digest(self._layout.notify, digest)

    dispatch_rework = self.dispatch_rework(effective_limit, stalled_entries=loop_stalled_entries)
    rework_count = dispatch_rework.data.get("selected_count", 0)
    fresh_limit = max(0, effective_limit - rework_count)
    dispatch = self.dispatch(fresh_limit, stalled_entries=loop_stalled_entries)

    # Issue #370: launch reviewers for queued PRs. This runs after worker
    # dispatch so a completed worker's review packet can be picked up by the
    # same loop pass only if the reviewer finishes immediately (tests); in
    # production the per-PR merge lane below fires on the next poll.
    # `now` (issue #822/#828) is this pass's injectable clock, threaded
    # through so dispatch_reviews's is_claim_stale checks share the same
    # instant as the rest of this pass instead of resampling.
    dispatch_reviews = self.dispatch_reviews(now=now)

    reviews: list[dict[str, Any]] = []
    merges: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    # Issue #502: post-merge tripwire. Detect any merged worker PR that was
    # not approved by the orchestrator's adversarial review gate.
    #
    # Reuse the merged PR list already fetched by dispatch() to avoid a
    # redundant fetch per loop pass. dispatch() returns an empty list (not
    # the fetched merged PRs) when there are no ready issues, so coerce an
    # empty reuse list back to None — the tripwire must then fetch its own
    # list to stay armed even when the queue is idle (a worker self-merge can
    # land regardless of whether issues are ready).
    #
    # Deliberate trade-off: because empty-means-unknown is indistinguishable
    # from empty-means-no-merged-PRs, an idle queue costs one extra
    # merged_pr_list() call per pass. That is a paginated REST fetch, not a
    # GraphQL check-run walk (merged_pr_list is REST-only by construction —
    # issue #361), so the cost is bounded and does not risk the gateway 502s
    # that motivated #361. Staying armed while idle is worth it: the whole
    # point of the tripwire is to catch merges the orchestrator did not
    # perform, which are exactly the ones that can happen on a quiet pass.
    # Replacing this with an explicit "not fetched" sentinel threaded out of
    # dispatch() would remove the extra call; that is tracked in #446 with
    # the rest of the per-pass fetch consolidation, and is deliberately not
    # done here to keep this security fix narrow.
    merged_prs_for_tripwire: list[dict[str, Any]] | None = dispatch.data.get("merged_prs")
    if not merged_prs_for_tripwire:
        merged_prs_for_tripwire = None
    for unauthorized in self._detect_unauthorized_merges(merged_prs_for_tripwire):
        reviewed_sha = unauthorized.get("reviewed_head_sha")
        live_sha = unauthorized.get("live_head_sha")
        if (
            unauthorized["decision"] == "approved"
            and reviewed_sha is not None
            and live_sha is not None
            and reviewed_sha != live_sha
        ):
            reason = f"approved for head {reviewed_sha!r} but merged head is {live_sha!r}"
        else:
            reason = f"without an approved review decision (decision={unauthorized['decision']!r})"
        errors.append(
            {
                "pr": unauthorized["pr"],
                "issue": unauthorized["issue"],
                "error": (
                    f"PR #{unauthorized['pr']} ({unauthorized['head']}) is MERGED "
                    f"{reason}; possible worker self-merge"
                ),
            }
        )

    foreign_transitions: dict[int, dict[str, Any]] = {}
    open_tracked_prs = 0
    skipped_reviews = 0
    parked_prs: list[int] = []
    prs = self.gh.pr_list()
    # Snapshot for foreign-PR markers only: markers change at most once
    # per PR, so a single point-in-time read at loop start is sufficient.
    state_snapshot = _wf.load_state_locked(self.paths.state_file)
    merge_train_head = (
        self._merge_train_head(prs)
        if self.config.auto_merge.update_branch_strategy == "front_of_train"
        else None
    )
    # Issue #1229: validate branch-name-derived issue numbers against the
    # actual open-issue set so a stale branch name (e.g. agent/issue-709-…
    # left over from a merged PR #709, reused by an unrelated issue-less
    # PR) cannot bind the PR to a non-existent or closed issue and corrupt
    # state["issues"][<n>] with an unrelated rework episode.
    branch_validator = self._make_branch_issue_validator()
    fir_confirm_passes = self.config.review.foreign_issue_ref_confirm_passes
    fir_reprobe_hours = self.config.review.foreign_issue_ref_reprobe_hours
    loop_now = now or datetime.now(UTC)
    for pr in prs:
        issue_number = _wf.linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=self.config.dispatch.branch_prefix,
            branch_issue_validator=branch_validator,
        )
        if issue_number is None:
            continue
        pr_number = int(pr["number"])
        parked = (state_snapshot["prs"].get(str(pr_number)) or {}).get("foreign_issue_ref") or {}
        if parked.get("issue") == issue_number:
            # Issue #1132: a marker is only "confirmed" (skip all per-PR
            # work) once confirmations reach the configured threshold.
            # Legacy markers without a ``confirmations`` field are treated
            # as confirmed so existing parks are not re-processed.
            confirmations = parked.get("confirmations")
            confirmed = confirmations is None or confirmations >= fir_confirm_passes
            if confirmed:
                # Issue #1132: bounded self-heal. Re-probe parked markers
                # on a slow cadence; if the issue now resolves, clear the
                # marker and resume per-PR processing instead of skipping.
                # A wrong park (e.g. from a transient failure that slipped
                # past classification) costs hours, not forever.
                if _wf._should_reprobe_foreign_marker(parked, loop_now, fir_reprobe_hours):
                    try:
                        self.gh.issue_view(issue_number)
                        # Issue now resolves — clear marker, emit event,
                        # and fall through to the try block.
                        _wf._clear_foreign_issue_ref_marker(self.paths.state_file, pr_number)
                        log_event(
                            self.paths.state_file,
                            "foreign_issue_ref_cleared",
                            {
                                "pr_number": pr_number,
                                "issue_number": issue_number,
                                "reason": "re-probe resolved; marker cleared",
                            },
                            repo=self.repo_root.name,
                        )
                        # Fall through to per-PR processing below; the
                        # marker has been cleared so the try block runs.
                    except GitHubNotFoundError as exc:
                        # Issue #1132: ``GitHubNotFoundError`` conflates a
                        # permanent issue-level 404 with a transient
                        # repository-level resolution failure (same
                        # conflation the main per-PR park decision above
                        # guards against). A transient repo-resolution
                        # failure during reprobe is NOT evidence the issue
                        # is absent — leave the marker AND the re-probe
                        # clock untouched so the next cadence window retries
                        # from the same anchor, matching the
                        # ``GitHubError`` handler below. Only a permanent
                        # issue-level 404 resets the re-probe clock.
                        if is_transient_repo_resolution_failure(str(exc)):
                            parked_prs.append(pr_number)
                            continue
                        # Still genuinely not found — reset the re-probe
                        # clock so the next check is gated from now.
                        _wf._touch_foreign_issue_ref_marker(
                            self.paths.state_file, pr_number, issue_number
                        )
                        parked_prs.append(pr_number)
                        continue
                    except GitHubError:
                        # Transient failure during re-probe — leave the
                        # marker in place; the next cadence window will
                        # retry. Do not clear or touch the clock: a
                        # transient failure is not evidence the issue is
                        # absent, nor evidence it is present.
                        parked_prs.append(pr_number)
                        continue
                else:
                    # Foreign/unlinked PR: its claimed issue does not
                    # exist in this repo (e.g. opened against the wrong
                    # fleet repo). Skip all per-PR work with zero GitHub
                    # calls until the marker is cleared or the PR's
                    # linked-issue ref changes.
                    parked_prs.append(pr_number)
                    continue
            # Not yet confirmed — fall through to the try block for
            # another confirmation pass. The PR is still tracked.
        # Count every PR with a resolvable linked issue (includes skipped ones)
        open_tracked_prs += 1
        is_merge_head = merge_train_head is None or pr_number == merge_train_head
        # Per-PR isolation: one PR's merge conflict or gh failure must not
        # abort review/merge of every remaining PR in the batch.
        try:
            # Idempotence: if the PR already has an approved decision in
            # state and isn't in a rework/blocked state, skip the expensive
            # review() pass (packet regeneration + label transitions) and
            # go straight to merge_ready. This prevents a second loop() pass
            # from rewriting the review packet or re-firing labels for a PR
            # that's simply waiting on pending checks.
            state = _wf.load_state_locked(self.paths.state_file)
            pr_state = state["prs"].get(str(pr_number), {})
            pr_dir_for_decision = self.paths.prs / f"pr-{pr_number}"
            live_head_sha = pr.get("headRefOid")
            # Issue #1362 Stage 1 (#1340 regression): the FILE is
            # authoritative for the decision itself -- state.json's
            # ``decision`` can lag a concurrent void/record_review (the
            # #1340 shape: state still says "approved" at an old head
            # while the file has already been reset to "pending").
            # ``status`` is a distinct workflow-state field, not one of
            # the three decision stores, so it still reads pr_state.
            resolved_decision = review_decision(pr_dir_for_decision, None, live_head_sha)
            # Issue #1362 Stage 3: state.json's decision fields are a
            # declared cache of the file, refreshed here at the start of
            # this PR's evaluation so any OTHER reader of ``state["prs"]``
            # later in this same pass (merge_ready, janitor, the reap
            # sweep, ...) sees a current decision/reviewed_head_sha/
            # decision_path without waiting for one of the four writer-
            # adjacent mirrors. Never touches ``status`` (checked just
            # below), so the already-loaded ``pr_state`` stays valid.
            self._refresh_pr_decision_cache(
                pr_number,
                resolved_decision,
                pr_dir_for_decision / "review-decision.json",
            )
            already_approved = resolved_decision.decision == "approved" and pr_state.get(
                "status"
            ) not in ("request_changes", "escalated", "blocked")
            if already_approved:
                # ``resolved_decision.stale`` already encodes exactly the
                # reviewed-head-vs-live-head comparison ``head_matches``
                # used to hand-roll.
                head_matches = not resolved_decision.stale
                if head_matches and is_merge_head:
                    merge_result = self.merge_ready(
                        pr_number, merge=merge, merge_train_head=merge_train_head
                    )
                    self._record_merge_or_error(merge_result, errors, merges)
                elif not head_matches:
                    review = self.review(pr_number)
                    if self._record_review_or_error(review, errors, reviews):
                        continue
                    post_review_decision = review_decision(
                        pr_dir_for_decision, None, pr.get("headRefOid")
                    )
                    if (
                        post_review_decision.decision == "approved"
                        and not post_review_decision.stale
                        and is_merge_head
                    ):
                        merge_result = self.merge_ready(
                            pr_number, merge=merge, merge_train_head=merge_train_head
                        )
                        self._record_merge_or_error(merge_result, errors, merges)
            else:
                # Same-head packet skip: if we already have a review packet
                # for this exact head SHA and no decision has been recorded
                # yet, skip regenerating the packet. This prevents repeated
                # supervised passes from re-firing review_started transitions
                # and regenerating packets every poll cycle while the operator
                # is still reading. The packet remains current; verdict file
                # appearance triggers a delta → the merge lane fires normally.
                #
                # Issue #592: the prompt template is as load-bearing an
                # input to the packet as the head SHA. A template edit
                # must reach static-head PRs, otherwise reviewers keep
                # receiving a packet rendered from the old template and
                # fail identically until they escalate. Treat a digest
                # mismatch as staleness alongside a head mismatch. A
                # missing digest is a legacy packet (pre-#592); treat it
                # as current so the upgrade does not force a one-time
                # fleet-wide regeneration burst -- packets rendered after
                # this fix always carry the digest.
                live_head_sha = pr.get("headRefOid")
                packet_head_sha = self._read_packet_head_oid(pr_number)
                head_current = (
                    live_head_sha is not None
                    and packet_head_sha is not None
                    and live_head_sha == packet_head_sha
                )
                current_template_sha = self._review_template_sha()
                packet_template_sha = self._read_packet_template_sha(pr_number)
                template_current = packet_template_sha is None or (
                    packet_template_sha == current_template_sha
                )
                if head_current and template_current:
                    # Packet is current — skip regenerating it. The
                    # already_approved branch above is evaluated earlier
                    # in this same pass and may not see a decision file an
                    # operator wrote after that check ran, so the verdict
                    # would otherwise stay invisible until the head moves.
                    # Re-check the decision file here and proceed to merge
                    # on approval, same as the decided path.
                    skipped_reviews += 1
                    packet_skip_decision = review_decision(
                        pr_dir_for_decision, None, pr.get("headRefOid")
                    )
                    if (
                        packet_skip_decision.decision == "approved"
                        and not packet_skip_decision.stale
                        and is_merge_head
                    ):
                        merge_result = self.merge_ready(
                            pr_number, merge=merge, merge_train_head=merge_train_head
                        )
                        self._record_merge_or_error(merge_result, errors, merges)
                else:
                    # Issue #1338: an escalated PR's packet regeneration is
                    # unreachable -- review() early-returns "escalated;
                    # review skipped" before its regen path -- so the
                    # staleness WARNING below would re-emit an identical
                    # review_packet_template_stale event every pass without
                    # ever converging. (The cross-family regen-budget charge
                    # this comment used to also call out here -- attempts_before /
                    # _charge_cross_family_regen_not_reached -- was deleted along
                    # with the auto-gate cross-family subsystem in role-config
                    # phase 2, track A; there is no second side effect left to
                    # skip.) Escalation means "awaiting a human", and the recovery
                    # procedure (unescalate + why-charlie-hate) already
                    # regenerates the packet with the current template, so skip
                    # ONLY that meaningless side effect while escalated.
                    #
                    # self.review(pr_number) is still called: its own
                    # _escalation_flags entry gate no-ops packet regen and
                    # label transitions, but it is the ONLY per-pass path
                    # that refreshes janitor_ok/janitor_failures/
                    # ci_run_never_created and runs the #776
                    # merge-conflict/no-op-rework remediation for
                    # judgment-class escalations (PRs #1397/#1443). Skipping
                    # it entirely would reintroduce the exact frozen-
                    # diagnostics staleness class the sibling-repo fix
                    # addressed, just via a different trigger. A
                    # non-escalated stale-template PR still regenerates
                    # exactly as today (#592 preserved).
                    issue_state_for_esc = (
                        state.get("issues", {}).get(str(issue_number), {})
                        if issue_number is not None
                        else None
                    )
                    pr_escalated_now, issue_escalated_now = _escalation_flags(
                        pr_state, issue_state_for_esc
                    )
                    escalated_now = pr_escalated_now or issue_escalated_now
                    # Emit a distinct event when regeneration fires
                    # because the template changed while the head stayed
                    # put, so a fleet-wide template edit is visible as a
                    # burst rather than unexplained review churn. Suppressed
                    # while escalated (#1338): the regen is unreachable, so
                    # the WARNING would fire identically every pass without
                    # converging.
                    if head_current and not template_current and not escalated_now:
                        log_event(
                            self.paths.state_file,
                            "review_packet_template_stale",
                            {
                                "pr_number": pr_number,
                                "issue_number": issue_number,
                                "packet_template_sha": packet_template_sha,
                                "current_template_sha": current_template_sha,
                            },
                            repo=self.repo_root.name,
                        )
                    review = self.review(pr_number)
                    if self._record_review_or_error(review, errors, reviews):
                        continue
                    decision = self._review_decision(pr_number)
                    if decision.get("decision") == "approved" and is_merge_head:
                        merge_result = self.merge_ready(
                            pr_number, merge=merge, merge_train_head=merge_train_head
                        )
                        self._record_merge_or_error(merge_result, errors, merges)
        except GitHubNotFoundError as exc:
            # Issue #1132: ``GitHubNotFoundError`` conflates a permanent
            # issue-level 404 ("Could not resolve to a Issue") with a
            # transient repository-level resolution failure ("Could not
            # resolve to a Repository"). The latter is transient — the
            # orchestrator just successfully listed PRs from this repo
            # (``pr_list`` at loop start), so the repo *did* resolve
            # moments ago. Route transient repo-resolution failures to
            # the retry path (same as ``GitHubError``) instead of parking
            # durably, which would wedge the PR forever with no self-heal.
            if is_transient_repo_resolution_failure(str(exc)):
                log_event(
                    self.paths.state_file,
                    "github_error",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "error": str(exc),
                        "transient_repo_resolution": True,
                    },
                    repo=self.repo_root.name,
                )
                errors.append({"pr": pr_number, "error": str(exc)})
            else:
                # Permanent: the PR's claimed issue (or another object it
                # references) does not exist in this repo. Park it durably
                # and alert once instead of failing the pass every 5
                # minutes forever — retrying can never succeed. Issue
                # #1132: parking now requires ``confirm_passes``
                # consecutive not-founds before the marker is confirmed,
                # so a transient window that produces an issue-level 404
                # shape (rare but possible) still clears before 2 passes.
                log_event(
                    self.paths.state_file,
                    # event-consumer: audit-only -- handled inline via _mark_foreign_issue_ref
                    # below (parked durably, alerted once); no separate downstream consumer needed
                    "github_not_found_error",
                    {"pr_number": pr_number, "issue_number": issue_number, "error": str(exc)},
                    repo=self.repo_root.name,
                )
                if self._mark_foreign_issue_ref(pr_number, issue_number, str(exc)):
                    foreign_transitions[pr_number] = {
                        "adapter_kind": "unknown",
                        "health": "FOREIGN_ISSUE_REF",
                        "last_log_line": str(exc),
                        "terminal_reason": (
                            f"linked issue #{issue_number} not found in this repo; "
                            f"PR #{pr_number} parked until the marker is cleared"
                        ),
                    }
        except GitHubError as exc:
            log_event(
                self.paths.state_file,
                "github_error",
                {"pr_number": pr_number, "issue_number": issue_number, "error": str(exc)},
                repo=self.repo_root.name,
            )
            errors.append({"pr": pr_number, "error": str(exc)})
    warnings: list[str] = []
    merge_alert_transitions: dict[int, dict[str, Any]] = {}
    for merge_entry in merges:
        warning = merge_entry.get("merge_attempt_warning")
        if warning:
            warnings.append(warning)
        if merge_entry.get("merge_attempt_alarm") and merge_entry.get("issue") is not None:
            issue = merge_entry["issue"]
            merge_alert_transitions[issue] = {
                "adapter_kind": "unknown",
                "health": "MERGE_BLOCKED",
                "last_log_line": None,
                "pid": None,
                "terminal_tool": None,
                "terminal_reason": warning,
            }

    # Emit a merge-lane alert digest when a PR crosses the threshold.
    if merge_alert_transitions and self.config.notify.enabled:
        digest = _wf._build_attention_digest(
            self.paths.state_file,
            merge_alert_transitions,
            repo=self.repo_root.name,
            state_field="merge_alert",
        )
        if digest:
            _wf.emit_digest(self._layout.notify, digest)

    # One-shot alert for newly parked foreign PRs. Dedupe comes from the
    # durable state marker (_mark_foreign_issue_ref returns True exactly
    # once per (pr, issue) pair), so this digest is built directly rather
    # than through the per-issue health-baseline machinery.
    if foreign_transitions and self.config.notify.enabled:
        _wf.emit_digest(
            self._layout.notify,
            AttentionDigest(
                generated_at=_wf.utc_now(),
                repo=self.repo_root.name,
                transitions=tuple(
                    AttentionEntry(
                        issue_number=pr_num,
                        adapter_kind=t["adapter_kind"],
                        health=t["health"],
                        previous_health=None,
                        last_log_line=t["last_log_line"],
                        pid=None,
                        terminal_tool=None,
                        terminal_reason=t["terminal_reason"],
                    )
                    for pr_num, t in foreign_transitions.items()
                ),
            ),
        )

    ok = intake.ok and dispatch.ok and dispatch_rework.ok and dispatch_reviews.ok and not errors
    message = "loop complete"
    if errors:
        message = f"loop completed with {len(errors)} PR error(s)"
    elif not intake.ok:
        message = "loop completed with intake failures"
    elif not dispatch.ok:
        message = dispatch.message
    elif not dispatch_rework.ok:
        message = dispatch_rework.message
    elif not dispatch_reviews.ok:
        message = "loop completed with review dispatch failures"
    data = {
        "intake": intake.data,
        "dispatch": dispatch.data,
        "dispatch_rework": dispatch_rework.data,
        "dispatch_reviews": dispatch_reviews.data,
        "reviews": reviews,
        "merges": merges,
        "errors": errors,
        "warnings": warnings,
        "open_tracked_prs": open_tracked_prs,
        "skipped_reviews": skipped_reviews,
        "reaped": reaped,
        # Issue #1132: parked PRs were previously invisible — the
        # early-continue emitted zero events, so "PR untouched for days"
        # was unattributable from events.db. Surface the count and PR
        # numbers in loop_completed so a parked PR is diagnosable.
        "parked_prs": parked_prs,
    }
    # Propagate concurrency info from dispatch results
    if gov.enabled or gov.fleet_enabled or gov.open_pr_enabled:
        data.update(gov.report_fields())
    # Prefer the dispatch-scoped governor values (they reflect sidecars
    # written by this pass and the most accurate fleet-wide live count).
    for lane in ("dispatch", "dispatch_rework"):
        for key in (
            "concurrency_limit",
            "live_session_count",
            "available_slots",
            "fleet_concurrency_limit",
            "fleet_live_session_count",
            "open_pr_count",
            "open_pr_max",
        ):
            if key in data[lane]:
                data[key] = data[lane][key]
    # Cadence-gated merged-PR worktree reclamation (issue #636). Runs at
    # the END of the pass so the per-candidate `gh pr view` fan-out never
    # contends with the dispatch/review/merge lanes for state_lock or
    # GitHub quota during the critical window. Gated by
    # worktree_reclamation.interval_minutes, so it fires at most once per
    # interval regardless of poll frequency or backlog size. `now`
    # (issue #828) is this pass's single injected clock -- see
    # `_loop_body`'s other cadence-gated calls above.
    reclamation = self._maybe_reclaim_worktrees(now=now)
    if reclamation is not None:
        data["worktrees_reclaimed"] = reclamation
    return _wf.CommandResult(
        ok,
        message,
        data,
    )
