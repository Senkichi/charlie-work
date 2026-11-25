"""Review-recording delegate for ``OrchestratorApp``.

Track 2 Phase B leaf L01 batch 2 (issue #1645, parent #1632, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from charlie_work import rescue as rescue_helpers
from charlie_work.github import GitHubError
from charlie_work.labels import TransitionOutcome
from charlie_work.local_work_park import publishes_pull_requests
import charlie_work.verdict_recording as verdict_recording
import charlie_work.workflow as _wf
from charlie_work.command_result import CommandResult

# How the PR path differs from the Local path when it records a Verdict (issue #3102).
_PR_DIFFERENCES = verdict_recording.VerdictDifferences(
    wording=verdict_recording.VerdictWording(
        invalid_decision="decision must be approved, request_changes, or blocked",
        invalid_provenance="verdict_provenance must be one of: {allowed}",
        head_noun="PR",
        reviewed_head_flag="--reviewed-head",
        choose_head_hint="use --reviewed-head to choose",
    ),
    stringify_required_changes=False,
    local_decision_fields=False,
    comment_repeats_derived_summary=False,
)


def record_review(
    self,
    pr_number: int,
    decision: str,
    summary: str = "",
    summary_file: Path | None = None,
    comment: bool = False,
    reviewed_head: str | None = None,
    required_changes: Sequence[str] | None = None,
    session_metrics: dict[str, Any] | None = None,
    *,
    verdict_provenance: str,
    allow_stale_head: bool = False,
    verdict_source: str | None = None,
) -> CommandResult:
    # Issue #2095: a backend that publishes no pull requests has nothing for
    # ``pr_view`` to read, so the operator ``charlie verdict`` path must take
    # the same local ingestion as the reviewer-verdict and stranded-verdict
    # sites (#1844). ``record_local_review`` owns the merged/closed terminal
    # guard, the record-authoritative issue/branch, and fails closed (never
    # mints a non-local ``prs[N]`` record) when no lane record exists.
    if not publishes_pull_requests(self.gh):
        return self.record_local_review(
            pr_number,
            decision,
            summary,
            summary_file,
            comment,
            reviewed_head,
            required_changes,
            session_metrics,
            verdict_provenance=verdict_provenance,
            allow_stale_head=allow_stale_head,
            verdict_source=verdict_source,
        )
    verdict = verdict_recording.derive_verdict(
        decision,
        verdict_provenance=verdict_provenance,
        summary=summary,
        summary_file=summary_file,
        required_changes=required_changes,
        human_decision_markers=self.config.review.human_decision_markers,
        differences=_PR_DIFFERENCES,
    )
    if isinstance(verdict, CommandResult):
        return verdict
    decision = verdict.decision
    summary_text = verdict.summary_text
    effective_required_changes = verdict.required_changes
    findings_channel = verdict.findings_channel
    reclassified_human_call = verdict.reclassified_human_call

    pr = self.gh.pr_view(pr_number)
    issue_number = (
        _wf.linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=self.config.dispatch.branch_prefix,
            branch_issue_validator=self._make_branch_issue_validator(),
        )
        if pr
        else None
    )

    # GitHub's PR ``state`` is the authoritative terminal-state signal; ``pr`` is falsy
    # only when ``pr_view`` itself failed, in which case the head resolution below fails
    # safe on its own and this guard correctly does not fire on missing data.
    pr_github_state = str(pr.get("state") or "").upper() if pr else ""
    terminal_refusal = (
        CommandResult(
            False,
            f"PR #{pr_number} is {pr_github_state}; verdict not recorded "
            f"(terminal-state PR decision file is preserved)",
            {
                "pr": pr_number,
                "issue": issue_number,
                "reason": "pr_terminal_state",
                "terminal_state": pr_github_state,
            },
        )
        if pr and pr_github_state in ("MERGED", "CLOSED")
        else None
    )

    def _post_pr_comment(body: str) -> str | None:
        try:
            self._comment_pr(pr_number, body)
        except GitHubError as exc:
            return str(exc)
        return None

    subject = verdict_recording.VerdictSubject(
        pr_number=pr_number,
        issue_number=issue_number,
        terminal_refusal=terminal_refusal,
        escalated_refusal=CommandResult(
            False,
            f"PR #{pr_number} is escalated; verdict not recorded "
            f"(run `charlie unescalate --pr {pr_number}` to re-arm it first)",
            {"pr": pr_number, "issue": issue_number, "reason": "pr_escalated", "escalated": True},
        ),
        guard_state=lambda: _wf.load_state_locked(self.paths.state_file),
        live_head=lambda: pr.get("headRefOid") if pr else None,
        post_comment=_post_pr_comment,
    )
    refusal = verdict_recording.check_recordable(subject)
    if refusal is not None:
        return refusal

    pr_dir = self.paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)

    packet_head_sha = self._read_packet_head_oid(pr_number)
    pinned = verdict_recording.pin_reviewed_head(
        subject,
        packet_head_sha=packet_head_sha,
        reviewed_head=reviewed_head,
        allow_stale_head=allow_stale_head,
        differences=_PR_DIFFERENCES,
    )
    if isinstance(pinned, CommandResult):
        return pinned
    reviewed_head_sha = pinned.sha
    reviewed_head_source = pinned.source
    live_head_sha = pinned.live_head_sha

    # The diff the verdict applies to: the packet's when the verdict is pinned to the
    # packet, the live PR diff otherwise.
    diff: str | None = None
    if pr:
        if reviewed_head_source == "packet":
            diff = self._read_packet_diff(pr_number)
            if diff is None:
                diff = self.gh.pr_diff(pr_number)
        else:
            diff = self.gh.pr_diff(pr_number)
    content = verdict_recording.reviewed_content(diff)
    reviewed_patch_id = content.patch_id

    # Issue #1939: stamp a hash of the PR body the reviewer read alongside
    # reviewed_patch_id. A request_changes finding whose required fix lives
    # in the PR description produces no code delta, so the janitor's
    # no-op-rework gate compares the live body against this baseline to
    # tell a body-only rework apart from a genuine no-op. Same source rule
    # as reviewed_head_sha/reviewed_patch_id above: the packet's pr.json
    # when the verdict is pinned to the packet, the live PR body otherwise
    # (a body edit landing between packet generation and verdict recording
    # is content the reviewer has not seen and must NOT be stamped as
    # reviewed). "" when neither source yields a body -- the gate fails
    # closed on a missing baseline, preserving the pre-#1939 behavior.
    packet_body = self._read_packet_body(pr_number) if reviewed_head_source == "packet" else None
    if packet_body is not None:
        reviewed_body_sha256 = _wf._body_content_sha256(packet_body)
    elif isinstance(pr, dict) and "body" in pr:
        reviewed_body_sha256 = _wf._body_content_sha256(pr.get("body"))
    else:
        reviewed_body_sha256 = ""

    # Issue #950 / #998: fold non-bot findings from the PR's own external
    # review surfaces into the verdict at write time. This is a live fetch
    # at record_review time, not at render time, so _render_rework_prompt
    # stays a pure function of the durable verdict file.
    #
    # The ingestion window is bounded on both sides by *time*, not by
    # author identity (the orchestrator and workers both post through user
    # tokens, so identity cannot separate their output from a genuine
    # human's -- see ORCHESTRATOR_COMMENT_MARKER and issue #998):
    #   * lower bound ``since``  = the previous round's *upper* bound
    #     (``before``), falling back to its ``reviewed_at`` only when no
    #     ``before`` was recorded -- see the contiguity note below;
    #   * upper bound ``before`` = the current reviewed_head_sha's committer
    #     date -- a worker's own rework reply is posted *after* the rework
    #     commit it describes (which is the head the reviewer is now
    #     reading), so it falls outside the window and is not fed back to
    #     the worker as a "required change". This also closes the
    #     pre-ORCHESTRATOR_COMMENT_MARKER comment gap (#998 comment): those
    #     comments predate every future reviewed_head_sha and are excluded
    #     by the lower bound from the second round on.
    # Both bounds fail toward ingestion when their timestamp is missing or
    # unparsable: losing a genuine human finding is the expensive direction.
    # This block runs after reviewed_head_sha is resolved (and after the
    # --reviewed-head validation that can return early), so the upper bound
    # always reflects the exact head the verdict is pinned to.
    #
    # Contiguity across rounds (issue #998 rework): the next round's
    # ``since`` MUST be this round's ``before``, not this round's
    # ``reviewed_at``. A genuine human comment posted in the gap
    # ``(before, reviewed_at]`` -- between the reviewed head commit landing
    # and the verdict being written -- is excluded by ``before`` this round
    # (item_dt > before). If the next round used ``reviewed_at`` as its
    # ``since``, that comment would satisfy ``item_dt <= reviewed_at`` and
    # be dropped by the lower bound *forever* -- a silent hole in the
    # window that violates the "fail toward ingestion" invariant above.
    # Deriving ``since`` from the previous round's persisted ``before``
    # makes the per-round windows contiguous: every comment falls in
    # exactly one round's ``(since, before]`` window, so nothing is
    # permanently lost. The ``reviewed_at`` fallback is only taken when no
    # ``before`` was recorded -- which happens precisely when the upper
    # bound was not applied (the commit timestamp could not be resolved),
    # so nothing was excluded by ``before`` and ``reviewed_at`` remains the
    # correct lower bound.
    ingestion_before: str | None = None
    # Issue #999: external findings no longer merge into
    # ``required_changes``. They ride in their own ``external_findings``
    # field so ``findings_channel`` keeps describing *only* the reviewer's
    # list -- ``"derived"`` is never overwritten and keeps its tier-2
    # verbatim rendering. The one exception is ``"vacuous"``: a
    # content-free reviewer summary has nothing worth rendering above the
    # external items, so that case still *replaces* ``required_changes``
    # (flipping the channel to ``"external"``) exactly as before #999 --
    # no ``external_findings`` field is written, and the renderer's
    # old-shape ``"external"`` path handles it unchanged.
    recorded_external_findings: list[str] | None = None
    if decision in {"request_changes", "blocked"}:
        previous_decision = self._review_decision(pr_number)
        previous_before = previous_decision.get("before")
        previous_reviewed_at = previous_decision.get("reviewed_at")
        since = (
            previous_before
            if isinstance(previous_before, str) and previous_before
            else (previous_reviewed_at if isinstance(previous_reviewed_at, str) else None)
        )
        before = _wf._commit_timestamp(self.gh, reviewed_head_sha)
        ingestion_before = before
        external_findings = _wf._collect_external_findings(
            self.gh, pr_number, since=since, before=before
        )
        if external_findings:
            if findings_channel == "vacuous":
                effective_required_changes = list(external_findings)
                findings_channel = "external"
            else:
                recorded_external_findings = list(external_findings)
    reviewed_at = _wf.utc_now()
    decision_path = pr_dir / "review-decision.json"
    round_number: int = 0
    # Merge-update (never in-place assignment) and persist BEFORE any GitHub
    # label mutation: a label-write failure or crash must not desync the
    # durable decision/counter from what actually happened.
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        pr_state = state["prs"].get(str(pr_number), {})
        rework_path: str | None = None
        # str, not str | None: unlike rework_path (used as-is, Optional,
        # in the returned CommandResult payload), rework_summary is only
        # ever consumed at the _write_rework_prompt call below, whose
        # dispatch_note parameter is a bare `str` (issue #782). A None
        # default would relocate reportPossiblyUnboundVariable into a
        # reportArgumentType mismatch instead of eliminating it. This
        # default is unreachable dead code on every path (see #782):
        # rework_summary is always rebound before use because the
        # binding guard (`if not escalated:`, above the write) and the
        # use guard (`decision == "request_changes" and not escalated`,
        # at the write) are the same conjunction and `escalated` is not
        # reassigned between them.
        rework_summary: str = ""
        escalated = False
        rescue_dispatched = False
        # Durable per-PR rework counter — NOT derived from the global events
        # log, which append_event truncates to the last 200 entries: on a busy
        # repo that eviction silently reset the count and defeated the cap
        # (a PR could rework forever instead of escalating to a human).
        request_changes_count = int(pr_state.get("request_changes_count", 0))
        if decision == "request_changes":
            # Only count a rework cycle when the PR head has actually advanced.
            # If the head is unchanged, the prior cycle's attempt was never
            # delivered (e.g., worker died orphaned), so re-issuing
            # request_changes should not consume the escalation budget. See
            # issue #208.
            #
            # Issue #1210: the head_advanced guard previously protected only
            # the counter increment below — the escalation check ran
            # unconditionally and fired on an at-cap verdict even when the
            # head was unchanged. Gate the escalation evaluation (and the
            # rescue tier, which is itself gated on cap exceedance) on
            # head_advanced as well, so an at-cap verdict on an unchanged
            # head re-issues request_changes without escalating, mirroring
            # what already happens below-cap. Keep the existing behavior for
            # advanced heads.
            head_advanced = reviewed_head_sha != pr_state.get("reviewed_head_sha")
            if head_advanced:
                # Rework cap: past max_rework_cycles the evidence says
                # iteration thrashes (wrong brief or unimplementable
                # criteria) — escalate to a human instead of dispatching
                # another cycle.
                escalated = request_changes_count >= self.config.review.max_rework_cycles
                # Rescue tier (issue #555): a cap exceedance here is one of
                # the three verdict-driven ("cheap model wasn't good
                # enough") causes the rescue tier is gated on. If enabled
                # and this PR has not already spent its one rescue attempt,
                # route to a bounded Opus rework instead of escalating —
                # never a second rescue for the same PR (rescue_attempted
                # is durable, cleared only by `charlie unescalate`).
                if (
                    escalated
                    and self.config.rescue.enabled
                    and not pr_state.get("rescue_attempted")
                ):
                    escalated = False
                    rescue_dispatched = True
                if not escalated:
                    request_changes_count += 1
            if not escalated:
                rework_summary = (
                    rescue_helpers.build_rescue_rework_summary("rework_cycle_cap", summary_text)
                    if rescue_dispatched
                    else summary_text
                )
        pr_escalation_reason = (
            "max_rework_cycles_exceeded"
            if (decision == "request_changes" and escalated)
            else ("review_blocked" if decision == "blocked" else None)
        )
        # Persist the verdict BEFORE rendering the rework brief: the brief reads
        # review-decision.json itself (issue #632), so the decision file must be on disk
        # first. A label-write failure or crash after this point leaves a durable verdict.
        decision_round = verdict_recording.write_decision_round(
            pr_dir,
            subject,
            decision=decision,
            summary_text=summary_text,
            required_changes=effective_required_changes,
            findings_channel=findings_channel,
            reclassified_human_call=reclassified_human_call,
            verdict_provenance=verdict_provenance,
            verdict_source=verdict_source,
            pinned=pinned,
            content=content,
            body_sha256=reviewed_body_sha256,
            external_findings=recorded_external_findings,
            ingestion_before=ingestion_before,
            escalated=escalated,
            reviewed_at=reviewed_at,
            next_round_number=_wf._next_round_number,
            differences=_PR_DIFFERENCES,
        )
        round_number = decision_round.round_number
        round_dir = decision_round.round_dir
        provenance_caveat = decision_round.provenance_caveat
        if decision == "request_changes" and not escalated:
            rework_path = str(self._write_rework_prompt(pr, issue_number, rework_summary))
            # Archived alongside the decision above, same round_dir: this
            # branch is the only one that produces these two live files
            # (an approved/blocked round has no rework brief), so only
            # archive them here. Read back the just-written live files
            # rather than re-deriving the rendered text, so the archive
            # copy is guaranteed byte-identical to what actually shipped.
            _wf._write_text_atomic(
                round_dir / "rework-prompt.md",
                Path(rework_path).read_text(encoding="utf-8"),
            )
            _wf._write_text_atomic(
                round_dir / "rework-dispatch-note.txt",
                (pr_dir / "rework-dispatch-note.txt").read_text(encoding="utf-8"),
            )
        rescue_fields = (
            rescue_helpers.build_rescue_dataclass_kwargs("rework_cycle_cap")
            if rescue_dispatched
            else {}
        )
        if rescue_dispatched:
            rescue_fields["rescue_dispatched_at"] = _wf.utc_now()
        state["prs"][str(pr_number)] = {
            **pr_state,
            "number": pr_number,
            "issue_number": issue_number,
            "decision": decision,
            "decision_path": str(decision_path),
            "reviewed_head_sha": reviewed_head_sha,
            "reviewed_patch_id": reviewed_patch_id,
            "reviewed_body_sha256": reviewed_body_sha256,
            "carried_forward_from": [],
            "request_changes_count": request_changes_count,
            "status": "escalated" if escalated else decision,
            "consecutive_failed_merge_attempts": 0,
            **rescue_fields,
            # The reviewer agent has recorded its verdict. The hub no longer
            # needs to treat this PR as having an in-flight reviewer; the next
            # stale-head review-queue entry will re-dispatch cleanly.
            "review_dispatch_status": "review_dispatch_completed",
            "reviewer_pid": None,
            "reviewer_process_start_time": None,
            # Reset the dispatch attempt counter: a verdict was produced, so
            # the PR is not stuck. If the head later advances and triggers a
            # new review cycle, the counter starts fresh.
            "review_dispatch_attempt_count": 0,
            # Reset the unreadable-log streak: a verdict proves the
            # reviewer pipeline is healthy, so any prior
            # persistent-unreadable condition is resolved (issue #1069).
            "review_log_unreadable_streak": 0,
            "review_api_error_streak": 0,  # issue #1808
            # Issue #1439: a recorded verdict ends the turn-limit miss
            # streak -- the reviewer reached a conclusion, so any prior
            # turn-limit deaths no longer count toward the cap-aware
            # backstop on a future review cycle.
            "review_turn_limit_miss_streak": 0,
            # Reviewer session token/cost telemetry (best-effort): merge-update
            # so a call without metrics (e.g. a manual `charlie verdict`)
            # never clobbers metrics recorded by an earlier automated reap.
            "review_session_metrics": (
                session_metrics
                if session_metrics is not None
                else pr_state.get("review_session_metrics")
            ),
            **(
                {"escalation_reason": pr_escalation_reason}
                if pr_escalation_reason is not None
                else {}
            ),
        }
        # Update the linked issue's status to reconcile out of rework_requested:
        # the previous worker session is definitionally finished, so the issue
        # status must reflect the actual decision. This prevents state-driven
        # dispatch_rework from selecting approved/blocked PRs for duplicate work.
        if issue_number is not None:
            if decision == "request_changes":
                if not escalated:
                    issue_entry = {
                        **state["issues"].get(str(issue_number), {}),
                        "number": issue_number,
                        "status": "rework_requested",
                        "merge_alert": "OK",
                    }
                    _wf.clear_escalation(issue_entry)
                    _wf.clear_escalation_on_issue_prs(state, issue_number)
                    # Issue #1784 (job-cannon #1320 follow-up): the reset
                    # below used to gate on ``head_advanced`` (a SHA
                    # comparison) alone, and its comment claimed
                    # record_review only ever reaches this branch on
                    # content a reviewer actually read. Neither half of
                    # that held: ``ci_gate_auto_reject``/
                    # ``test_adequacy_auto_reject`` short-circuits (and the
                    # first-review case, where ``reviewed_head_sha`` starts
                    # as None) reach here with no reviewer involved at all,
                    # and a pure sync-merge can move the head SHA with zero
                    # content change. Reached by a CI-red short-circuit
                    # specifically, that combination made the cap this
                    # reset exists to protect UNREACHABLE for exactly the
                    # stuck-worker population it is meant to catch (every
                    # cycle's reset firing on a no-op sync-merge).
                    #
                    # The actual predicate the code enforces now: a
                    # *patch-id* advance (unlike a SHA, immune to a
                    # sync-merge that changes nothing but the merge-base)
                    # from a verdict whose ``verdict_provenance`` names a
                    # mechanism that read the diff at all
                    # (``NO_OP_RESET_PROVENANCES`` -- see its docstring for
                    # why each excluded value is excluded). That is
                    # evidence the issue actually moved forward since the
                    # checkpoint, so the no-op/death-loop caps'
                    # windowed reads (``_windowed_redispatch_at``,
                    # ``_windowed_worker_death_at``) should stop counting
                    # redispatches/deaths from before it.
                    #
                    # This does NOT truncate ``redispatch_at``/
                    # ``worker_death_at`` themselves (finding 2 of the
                    # #1784 review): those two arrays are shared inputs to
                    # five independent caps across
                    # ``state_dispatch_rework.py`` and
                    # ``dead_worker_reap.py`` -- e.g. the plain redispatch
                    # cap and the worker-death-loop cap -- not an
                    # issue-level no-op counter this function alone owns.
                    # Clearing them here would silently reset all five.
                    # Instead, stamp a checkpoint timestamp that
                    # ``_windowed_redispatch_at``/``_windowed_worker_death_at``
                    # apply as a floor on top of the ordinary rolling
                    # window, so every consumer still sees its own full
                    # history up to the checkpoint -- only the no-op-cap's
                    # "counts from before this review no longer describe
                    # the issue's CURRENT stall state" no longer needs the
                    # data destroyed to say so.
                    prior_patch_id = pr_state.get("reviewed_patch_id")
                    content_advanced = (
                        bool(reviewed_patch_id) and reviewed_patch_id != prior_patch_id
                    )
                    if content_advanced and verdict_provenance in _wf.NO_OP_RESET_PROVENANCES:
                        issue_entry["no_op_checkpoint_at"] = _wf.utc_now()
                    state["issues"][str(issue_number)] = issue_entry
                else:
                    # Clear rework_requested status when escalated to prevent selection
                    # Issue #783: rework-cycle cap reached is a
                    # process limit, not a judgment call --
                    # mechanical.
                    state = _wf._escalate_issue(
                        state,
                        issue_number,
                        reason="max_rework_cycles_exceeded",
                        reason_class="mechanical",
                    )
            elif decision == "approved":
                issue_entry = {
                    **state["issues"].get(str(issue_number), {}),
                    "number": issue_number,
                    "status": "approved",
                    "merge_alert": "OK",
                }
                _wf.clear_escalation(issue_entry)
                _wf.clear_escalation_on_issue_prs(state, issue_number)
                state["issues"][str(issue_number)] = issue_entry
                # Clear worker PID when issue is approved (worker is done)
                state["issues"][str(issue_number)].pop("worker_pid", None)
                state["issues"][str(issue_number)].pop("worker_process_start_time", None)
            elif decision == "blocked":
                # Issue #783: an explicit "blocked" review decision is
                # a human product/security judgment call, never
                # auto-cleared.
                state = _wf._escalate_issue(
                    state,
                    issue_number,
                    reason="review_blocked",
                    reason_class="judgment",
                    status="blocked",
                )
                # Clear worker PID when issue is blocked (worker is done)
                state["issues"][str(issue_number)].pop("worker_pid", None)
                state["issues"][str(issue_number)].pop("worker_process_start_time", None)
        event_payload: dict[str, Any] = {
            "pr_number": pr_number,
            "issue_number": issue_number,
            "decision": decision,
            "escalated": escalated,
            # Issue #1265: top-level and unconditional, distinct from
            # (and a sibling of, not nested inside) the pre-existing
            # session_metrics.verdict_source added conditionally below --
            # that field means "which parser found the fenced verdict
            # block"; this one means "which mechanism produced the
            # verdict at all".
            "verdict_provenance": verdict_provenance,
            # Issue #1268 (W11), item 2: previously never assigned at all
            # (confirmed live: 0 non-null across every existing
            # record_review row) -- events.db carried no verdict text.
            # These are an observability-only, size-bounded copy; the
            # round-K archive (`rounds/round-K/review-decision.json`,
            # above) is the untruncated source of truth W13 reads from.
            # ``required_changes`` is a list on ``decision_payload``, so
            # it is newline-joined into a single string before the same
            # byte-budget truncation applies to it. ``str(item)`` guards
            # every assignment site of ``effective_required_changes`` --
            # most are already ``list[str]``, but the #999 external-
            # findings path (``effective_required_changes =
            # list(external_findings)``) is never itself read as
            # strings elsewhere, so this join must not assume str
            # elements there. Coercing costs nothing for the str case
            # (round-trips identically) and makes this key correct for
            # every present and future assignment site rather than
            # only the ones exercised by today's tests.
            "summary": _wf._truncate_for_event(summary_text),
            "required_changes": _wf._truncate_for_event(
                "\n".join(str(item) for item in effective_required_changes)
            ),
        }
        if session_metrics is not None:
            event_payload["session_metrics"] = session_metrics
        if findings_channel is not None:
            event_payload["findings_channel"] = findings_channel
        if recorded_external_findings is not None:
            event_payload["external_findings_count"] = len(recorded_external_findings)
        state = self._record_event(state, "record_review", event_payload)
        if reclassified_human_call is not None:
            # Issue #1642: separate event so the reclassification is
            # independently queryable -- the operator-queue entry, review
            # packets, and post-incident audits need the matched finding
            # without re-deriving it from the verdict text.
            matched_item, matched_marker = reclassified_human_call
            state = self._record_event(
                state,
                "review_decision_reclassified_blocked",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "original_decision": "request_changes",
                    "matched_item": matched_item,
                    "matched_marker": matched_marker,
                },
            )
        if findings_channel == "vacuous":
            # Distinct from the general "record_review" event (issue
            # #792): this is the signal that a request_changes/blocked
            # verdict arrived with nothing to act on -- neither an
            # itemized required_changes nor derivable prose -- so it can
            # be queried/alerted on independently of every other verdict.
            state = self._record_event(
                state,
                "required_changes_vacuous",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "decision": decision,
                },
            )
        if rescue_dispatched:
            state = self._record_event(
                state,
                "rescue_dispatched",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "cause": "rework_cycle_cap",
                },
            )
        _wf.save_state(self.paths.state_file, state)
    # GitHub label side effects are best-effort and isolated: the durable
    # decision above is the authority; a label failure is reported, not fatal.
    label_error: dict[str, Any] | None = None
    if issue_number is not None:
        if decision == "request_changes":
            # Issue #1131: never route rework onto a CLOSED issue. The
            # observed race applied ``agent:needs-rework`` to an issue
            # that was already closed by the merge this verdict chased,
            # polluting roll-call/metrics and requiring a manual label
            # strip. GitHub's issue ``state`` is the authoritative
            # closed signal (state.json's workflow ``status`` can lag or
            # be clobbered). Fetch is best-effort: on lookup failure,
            # proceed with the transition (label application is already
            # best-effort and reported, not fatal) -- the same fail-open
            # posture as the stranded-request_changes restorer's
            # issue_view call. ``terminal_state`` on the PR (fix #1131
            # above) already refuses the whole call for a merged/closed
            # PR; this guard covers the residual case where the PR is
            # still OPEN but its linked issue is independently CLOSED.
            issue_is_closed = False
            try:
                linked_issue = self.gh.issue_view(issue_number)
                issue_is_closed = str(linked_issue.get("state") or "OPEN").upper() == "CLOSED"
            except Exception:
                pass
            if issue_is_closed:
                with _wf.state_lock(self.paths.state_file):
                    state = _wf.load_state(self.paths.state_file)
                    state = self._record_event(
                        state,
                        "rework_label_skipped_issue_closed",
                        {
                            "pr_number": pr_number,
                            "issue_number": issue_number,
                            "decision": decision,
                            "escalated": escalated,
                        },
                        level="warning",
                    )
                    _wf.save_state(self.paths.state_file, state)
            else:
                # Issue #1266: escalated here is always max_rework_cycles_
                # exceeded (mechanical) -- see the _escalate_issue call above.
                target = (
                    _wf._escalation_edge("escalated", "mechanical")
                    if escalated
                    else "rework_requested"
                )
                result = _wf.transition(
                    self.gh,
                    self.config.labels,
                    issue_number,
                    target,
                    state_path=self.paths.state_file,
                    repo=self.repo_root.name,
                    pr_number=pr_number,
                )
                if result.outcome != TransitionOutcome.APPLIED:
                    label_error = {
                        "edge": target,
                        "outcome": result.outcome.value,
                        "add_failures": result.add_failures,
                        "remove_failures": result.remove_failures,
                    }
        elif decision == "blocked":
            # Issue #1266: "blocked" has no operator-queue counterpart --
            # a reviewer-blocked verdict is judgment-only by
            # construction -- so this is an inert pass-through, kept
            # here for uniformity with every other escalation-label
            # call site now routing through the same helper.
            result = _wf.transition(
                self.gh,
                self.config.labels,
                issue_number,
                _wf._escalation_edge("blocked", "judgment"),
                state_path=self.paths.state_file,
                repo=self.repo_root.name,
                pr_number=pr_number,
            )
            if result.outcome != TransitionOutcome.APPLIED:
                label_error = {
                    "edge": "blocked",
                    "outcome": result.outcome.value,
                    "add_failures": result.add_failures,
                    "remove_failures": result.remove_failures,
                }
        elif decision == "approved":
            result = _wf.transition(
                self.gh,
                self.config.labels,
                issue_number,
                "review_approved",
                state_path=self.paths.state_file,
                repo=self.repo_root.name,
                pr_number=pr_number,
            )
            if result.outcome != TransitionOutcome.APPLIED:
                label_error = {
                    "edge": "review_approved",
                    "outcome": result.outcome.value,
                    "add_failures": result.add_failures,
                    "remove_failures": result.remove_failures,
                }
    # Issue #1268 (W11), item 3: was `decision == "request_changes" and
    # comment and summary_text` -- request_changes-only, and silent
    # unless a caller explicitly passed comment=True. Widened to fire
    # for every terminal decision this call actually recorded
    # (approved/request_changes/blocked), gated on the config default
    # (`review.post_verdict_comment`, True) OR the existing `comment`
    # force-on override -- the CLI's `--comment` flag stays a strict
    # superset of the config default, never replaced by it. Still
    # excludes `escalated`: that is the SAME in-call local flipped by
    # the rework-cap logic above (not the early state-guard escalated
    # return at the top of this function, which never reaches this far)
    # -- the rescue tier and the rework-cap escalation path already post
    # their own comment for that case (_process_rescue_review,
    # rescue_helpers), so leaving this gate open on escalated=True would
    # double-post. `summary_text` is no longer required to be truthy:
    # it is guaranteed non-empty for request_changes/blocked (the reject
    # gate near the top of this function), but NOT for approved (no
    # validation requires it) -- dropping the clause means an
    # empty-summary approval still gets its header + required-changes
    # section posted rather than silently skipping the comment for a
    # decision type this gate previously never even reached.
    if not escalated and (comment or self.config.review.post_verdict_comment):
        comment_error = verdict_recording.post_verdict_comment(
            subject,
            round_number=round_number,
            decision=decision,
            summary_text=summary_text,
            provenance_caveat=provenance_caveat,
            required_changes=effective_required_changes,
            findings_channel=findings_channel,
            differences=_PR_DIFFERENCES,
        )
        if comment_error is not None:
            label_error = verdict_recording.with_comment_error(label_error, comment_error)
    source_note = f"head from {reviewed_head_source}"
    if rescue_dispatched:
        message = (
            f"review recorded — rework cap ({self.config.review.max_rework_cycles}) "
            f"reached, rescue tier dispatched instead of escalating ({source_note})"
        )
    elif escalated:
        message = (
            f"review recorded — rework cap ({self.config.review.max_rework_cycles}) reached, "
            f"escalated to human ({source_note})"
        )
    else:
        message = f"review recorded ({source_note})"
    if reclassified_human_call is not None:
        # Issue #1642: caller-visible explanation so `charlie verdict` /
        # orchestration logs show why a recorded request_changes landed as
        # blocked in the operator queue.
        message += (
            " — request_changes reclassified as blocked: finding "
            f"{reclassified_human_call[0]!r} asks for a human/operator decision"
        )
    if label_error:
        message += f" (label update failed: {label_error.get('outcome', label_error)})"
    return CommandResult(
        True,
        message,
        {
            "pr": pr_number,
            "decision": decision,
            "decision_path": str(decision_path),
            "reviewed_head_sha": reviewed_head_sha,
            "reviewed_head_source": reviewed_head_source,
            "live_head_sha": live_head_sha,
            "packet_head_sha": packet_head_sha,
            "rework_path": rework_path,
            "escalated": escalated,
            "rescue_dispatched": rescue_dispatched,
            "request_changes_count": request_changes_count,
            "reclassified_from": (
                "request_changes" if reclassified_human_call is not None else None
            ),
            "label_error": label_error,
        },
    )
