"""Review-recording delegate for ``OrchestratorApp``.

Track 2 Phase B leaf L01 batch 2 (issue #1645, parent #1632, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from charlie_work.github import GitHubError
from charlie_work.local_work_park import publishes_pull_requests
import charlie_work.verdict_recording as verdict_recording
import charlie_work.verdict_rules as verdict_rules
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
    # Persist BEFORE any GitHub label mutation: a label-write failure or crash must not
    # desync the durable decision/counter from what actually happened.
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        pr_state = state["prs"].get(str(pr_number), {})
        rework_path: str | None = None
        # The Rescue tier is the configured value on the PR path (the Local path passes False).
        outcome = verdict_rules.decide_rework(
            decision=decision,
            reviewed_head_sha=reviewed_head_sha,
            prior_head_sha=pr_state.get("reviewed_head_sha"),
            prior_count=int(pr_state.get("request_changes_count", 0)),
            max_rework_cycles=self.config.review.max_rework_cycles,
            rescue_enabled=self.config.rescue.enabled,
            rescue_attempted=bool(pr_state.get("rescue_attempted")),
            summary_text=summary_text,
        )
        escalated = outcome.escalated
        rescue_dispatched = outcome.rescue_dispatched
        request_changes_count = outcome.request_changes_count
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
            rework_path = str(self._write_rework_prompt(pr, issue_number, outcome.rework_summary))
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
        state = verdict_rules.record_verdict(
            self.write_gate,
            state,
            pr_number=pr_number,
            issue_number=issue_number,
            decision=decision,
            outcome=outcome,
            pr_state=pr_state,
            decision_path=str(decision_path),
            reviewed_head_sha=reviewed_head_sha,
            reviewed_patch_id=reviewed_patch_id,
            verdict_provenance=verdict_provenance,
            session_metrics=session_metrics,
            summary_text=summary_text,
            required_changes=effective_required_changes,
            findings_channel=findings_channel,
            reclassified_human_call=reclassified_human_call,
            record_extras={"reviewed_body_sha256": reviewed_body_sha256},
            event_extras=(
                {"external_findings_count": len(recorded_external_findings)}
                if recorded_external_findings is not None
                else {}
            ),
            utc_now=_wf.utc_now,
        )
        _wf.save_state(self.paths.state_file, state)
    # GitHub label side effects are best-effort and isolated: the durable
    # decision above is the authority; a label failure is reported, not fatal.
    label_error: dict[str, Any] | None = None
    if issue_number is not None:
        label_edge = verdict_rules.choose_label_edge(
            self.gh, issue_number, decision=decision, escalated=escalated
        )
        if label_edge is not None and label_edge.skip_issue_closed:
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
        elif label_edge is not None:
            result = _wf.transition(
                self.gh,
                self.config.labels,
                issue_number,
                label_edge.edge,
                state_path=self.paths.state_file,
                repo=self.repo_root.name,
                pr_number=pr_number,
            )
            label_error = verdict_rules.label_error(label_edge.error_edge, result)
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
