"""Review-recording delegate for ``OrchestratorApp``.

Track 2 Phase B leaf L01 batch 2 (issue #1645, parent #1632, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json

from charlie_work import rescue as rescue_helpers
from charlie_work.github import GitHubError
from charlie_work.janitor import DiffContentSignature
from charlie_work.labels import TransitionOutcome
import charlie_work.workflow as _wf


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
) -> _wf.CommandResult:
    if decision not in {"approved", "request_changes", "blocked"}:
        return _wf.CommandResult(
            False, "decision must be approved, request_changes, or blocked", {}
        )
    # Issue #1265: no default above -- every caller must say where the
    # verdict came from. This membership check catches a garbage/typo'd
    # literal (a missing argument is already caught earlier, at the call
    # boundary, by the keyword-only parameter having no default).
    if verdict_provenance not in _wf.VERDICT_PROVENANCE_VALUES:
        return _wf.CommandResult(
            False,
            "verdict_provenance must be one of: "
            + ", ".join(sorted(_wf.VERDICT_PROVENANCE_VALUES)),
            {},
        )
    summary_text = summary_file.read_text(encoding="utf-8") if summary_file else summary
    # Issue #11: reject empty summary for request_changes/blocked decisions
    # before any state/label mutation
    if decision in {"request_changes", "blocked"} and not summary_text.strip():
        return _wf.CommandResult(
            False,
            f"--summary or --summary-file is required for decision '{decision}'",
            {},
        )

    # Issue #792: required_changes has a near-0% fill rate because
    # prompts/review.md historically documented it as optional, so
    # reviewers reliably fill in `summary` and skip the structured list.
    # `_render_required_changes_section` used to paper over this at
    # *render* time, falling back to `summary` when the list was empty --
    # but that runs long after the reviewer is gone and the verdict is
    # frozen, so a bad or missing derivation there can never be
    # corrected. Derive here instead, at the point of record, so
    # whatever `required_changes` lands on disk is exactly what a
    # worker's rework brief will show.
    #
    # This step never *rejects* a verdict for an empty required_changes
    # -- see test_record_review_never_rejects_for_empty_required_changes.
    # A reject path here is the tempting wrong fix and that test is what
    # stops a future session from "tightening" it: a False return writes
    # no review-decision.json, the caller logs review_verdict_missed and
    # moves on, the next pass still sees the PR as pending, and it gets
    # re-dispatched to a fresh reviewer -- an unbounded re-review loop
    # with no cap and no escalation. Contrast with issue #597's reject
    # gate on a contentless *approved* verdict in
    # `_validate_review_verdict`: that gate is safe because rejecting an
    # approval just means "not merged yet", not an infinite loop.
    effective_required_changes = list(required_changes) if required_changes else []
    findings_channel: str | None = None
    if decision in {"request_changes", "blocked"} and not effective_required_changes:
        if _wf._summary_is_vacuous(summary_text):
            # Nothing derivable. Persist the empty list anyway (never
            # reject) with an explicit marker so
            # `_render_required_changes_section` can tell "genuinely
            # nothing was recorded" apart from "the reviewer wrote
            # prose instead of a list" and render its loud tier-3
            # warning instead of silently treating this the same as a
            # real derivation.
            findings_channel = "vacuous"
        else:
            # The F1 extractor: `_render_required_changes_section`'s own
            # tier-2 fallback, lifted here unchanged (issue #792) so the
            # producer -- not a consumer with no way to correct a bad
            # call -- makes this decision. `findings_channel = "derived"`
            # tells the renderer this single item is reviewer prose, not
            # an itemized list, so it renders as prose (tier 2) rather
            # than a one-item bullet list built from a multi-paragraph
            # summary.
            effective_required_changes = [summary_text.strip()]
            findings_channel = "derived"

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

    # Issue #1131: a terminal-state PR (MERGED or CLOSED) must never have
    # its decision file overwritten or rework routed. The observed race:
    # an operator unescalate->approve->merge interleaved with a dispatch
    # pass, so a reviewer's request_changes landed ~8 minutes after the
    # merge and clobbered ``review-decision.json`` (approved ->
    # request_changes), destroying the audit record that authorized the
    # merge, firing a false unauthorized-merge tripwire, and applying
    # ``agent:needs-rework`` to an already-CLOSED issue. Refuse here --
    # the durable verdict that authorized the merge is preserved, and the
    # late verdict is dropped (the caller logs ``review_verdict_missed``,
    # same as any refused verdict). This is unconditional (not gated on
    # ``allow_stale_head``): even the operator CLI must not overwrite a
    # merge-authorizing decision file. GitHub's PR ``state`` is the
    # authoritative terminal-state signal; ``pr`` is falsy only when
    # ``pr_view`` itself failed, in which case the existing head-resolution
    # logic below fails safe on its own and this guard correctly does not
    # fire on missing data.
    pr_github_state = str(pr.get("state") or "").upper() if pr else ""
    if pr and pr_github_state in ("MERGED", "CLOSED"):
        return _wf.CommandResult(
            False,
            f"PR #{pr_number} is {pr_github_state}; verdict not recorded "
            f"(terminal-state PR decision file is preserved)",
            {
                "pr": pr_number,
                "issue": issue_number,
                "terminal_state": pr_github_state,
            },
        )

    # Escalation is terminal for verdict recording too, mirroring review()'s
    # guard: without this, a late-arriving verdict (a reviewer that finished
    # after the attempt-cap escalation fired, or a stale reap) silently
    # overwrites status="escalated" and re-enters the PR into the pipeline,
    # which is exactly how escalated PRs were observed re-escalating 2-3x
    # (pr-lifecycle.md: non-durable escalation). A human re-arms the PR with
    # `charlie unescalate`, after which verdicts record normally again.
    guard_state = _wf.load_state_locked(self.paths.state_file)
    guard_pr_state = guard_state.get("prs", {}).get(str(pr_number), {})
    guard_issue_state = (
        guard_state.get("issues", {}).get(str(issue_number), {})
        if issue_number is not None
        else {}
    )
    if (
        guard_pr_state.get("status") == "escalated"
        or guard_issue_state.get("status") == "escalated"
    ):
        return _wf.CommandResult(
            False,
            f"PR #{pr_number} is escalated; verdict not recorded "
            f"(run `charlie unescalate --pr {pr_number}` to re-arm it first)",
            {"pr": pr_number, "issue": issue_number, "escalated": True},
        )

    pr_dir = self.paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)

    # reviewed_head_sha/reviewed_patch_id must reflect the packet the reviewer
    # actually read (review()'s pr.json/diff.patch), not a fresh fetch made
    # here at verdict time: a commit landing between packet generation and
    # verdict recording would otherwise silently reattribute the decision to
    # a head/diff that was never reviewed. Fall back to a live fetch only
    # when no packet exists (e.g. a decision recorded without a prior
    # review() call).
    packet_head_sha = self._read_packet_head_oid(pr_number)
    live_head_sha = pr.get("headRefOid") if pr else None

    # Issue #467: do not silently pin a verdict to a stale packet when the PR
    # head has advanced since the packet was generated. If the packet head
    # disagrees with the live PR head, require an explicit --reviewed-head
    # choice. When they agree (or no packet exists), preserve the existing
    # packet-first / live-fallback semantics and record where the SHA came from.
    #
    # Issue #1072: the #467 check only fired when *no* --reviewed-head was
    # passed. When an automated caller (review()'s CI-failure exit,
    # test-adequacy exit, dispatch_reviews, cross-family) passed
    # --reviewed-head matching the packet head, the check was bypassed and
    # the verdict was pinned to a head that had moved mid-build. The
    # compare-and-swap guard added by #1036 covered only review()'s
    # packet-commit tail exit; the two earlier exits returned through
    # record_review() and bypassed it. Pushing the guard here — into the
    # single choke point every verdict write passes through — makes the
    # invariant hold by construction for every caller, not just the one
    # exit #1036 happened to guard. The operator CLI (``charlie verdict``)
    # is the one caller that may legitimately pin to a superseded head
    # (issue #467's explicit-choice design), so it passes
    # ``allow_stale_head=True``; every automated caller uses the default
    # ``False`` and is refused. The failure direction is toward redundant
    # work (the PR is re-reviewed next pass), never toward authorizing a
    # merge on the wrong head — ``reviewed_head_sha`` is compared against
    # the current head by every downstream consumer.
    if reviewed_head is not None:
        if packet_head_sha is not None and reviewed_head == packet_head_sha:
            if (
                not allow_stale_head
                and live_head_sha is not None
                and packet_head_sha != live_head_sha
            ):
                return _wf.CommandResult(
                    False,
                    f"reviewed head ({reviewed_head}) matches packet head "
                    f"({packet_head_sha}) but live PR head has moved to "
                    f"({live_head_sha}); verdict not recorded — head moved "
                    "during build, will re-review next pass",
                    {
                        "pr": pr_number,
                        "issue": issue_number,
                        "reason": "head_moved_during_build",
                        "packet_head_sha": packet_head_sha,
                        "live_head_sha": live_head_sha,
                    },
                )
            reviewed_head_sha = reviewed_head
            reviewed_head_source = "packet"
        elif live_head_sha is not None and reviewed_head == live_head_sha:
            reviewed_head_sha = reviewed_head
            reviewed_head_source = "live"
        else:
            options: list[str] = []
            if packet_head_sha is not None:
                options.append(f"packet head {packet_head_sha}")
            if live_head_sha is not None:
                options.append(f"live head {live_head_sha}")
            options_str = " or ".join(options) if options else "any available head"
            return _wf.CommandResult(
                False,
                f"--reviewed-head {reviewed_head} does not match {options_str}",
                {},
            )
    elif (
        packet_head_sha is not None
        and live_head_sha is not None
        and packet_head_sha != live_head_sha
    ):
        return _wf.CommandResult(
            False,
            f"review packet head ({packet_head_sha}) differs from live PR head ({live_head_sha}); "
            "use --reviewed-head to choose the head the verdict applies to",
            {},
        )
    elif packet_head_sha is not None:
        reviewed_head_sha = packet_head_sha
        reviewed_head_source = "packet"
    elif live_head_sha is not None:
        reviewed_head_sha = live_head_sha
        reviewed_head_source = "live"
    else:
        return _wf.CommandResult(False, "no packet or live PR head available", {})

    # Calculate patch-id for the PR diff to detect actual content changes
    # (issue #222: base-update merges can advance head SHA without changing diff content).
    # All terminal decisions (approved/request_changes/blocked) persist reviewed_patch_id
    # so the review-queue enumerator can carry them forward on content-identical heads.
    # Also record the tier-2 content signature (issue #414): patch-id is
    # unstable across every main advance (the merge-base moves), so the
    # ordered +/- line stream and changed-file set are persisted here,
    # at the moment the diff is freshly known, so a later carry-forward
    # check never needs to reconstruct this historical diff.
    reviewed_patch_id = ""
    reviewed_signature = DiffContentSignature((), frozenset())
    diff: str | None = None
    if pr:
        if reviewed_head_source == "packet":
            diff = self._read_packet_diff(pr_number)
            if diff is None:
                diff = self.gh.pr_diff(pr_number)
        else:
            diff = self.gh.pr_diff(pr_number)
        reviewed_patch_id = _wf._calculate_patch_id(diff)
        if diff:
            reviewed_signature = _wf._diff_content_signature(diff)

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
    decision_payload = {
        "pr_number": pr_number,
        "issue_number": issue_number,
        "decision": decision,
        "summary": summary_text,
        "required_changes": effective_required_changes,
        # Issue #1265: unconditional -- every record_review call already
        # requires this argument (no default, see the keyword-only
        # parameter above), so it is always present, never a
        # conditionally-added key the way findings_channel/
        # external_findings are.
        "verdict_provenance": verdict_provenance,
        "reviewed_head_sha": reviewed_head_sha,
        "reviewed_head_source": reviewed_head_source,
        "reviewed_patch_id": reviewed_patch_id,
        "reviewed_changed_lines": list(reviewed_signature.changed_lines),
        "reviewed_changed_files": sorted(reviewed_signature.changed_files),
        "reviewed_has_binary": reviewed_signature.has_binary,
        "carried_forward_from": [],
        "reviewed_at": _wf.utc_now(),
    }
    # Only present when the derivation above ran (issue #792): a verdict
    # that already carried a populated required_changes, or an
    # `approved` verdict, passes through with no new key at all.
    if findings_channel is not None:
        decision_payload["findings_channel"] = findings_channel
    # Issue #999: external findings live in their own field, separate
    # from the reviewer's required_changes. Absent on old-shape records
    # (pre-#999, or the vacuous-replace case) -- the renderer treats
    # absence as old-shape and renders exactly as before.
    if recorded_external_findings is not None:
        decision_payload["external_findings"] = recorded_external_findings
    # Persist the ingestion upper bound so the next round's ``since`` can
    # be derived from it (issue #998 rework: contiguous windowing across
    # rounds -- see the contiguity note above). Only present when ingestion
    # actually ran AND the upper bound was resolved; an ``approved`` verdict
    # has no next ingestion round, and a None ``before`` means the upper
    # bound was not applied (so there is nothing for the next round's
    # ``since`` to recover -- it falls back to ``reviewed_at``).
    if ingestion_before is not None:
        decision_payload["before"] = ingestion_before
    # Issue #1340: persist the parser-level provenance (which extractor
    # found the verdict block: "log", "events", "file:<source>") into the
    # decision file so a later reader can distinguish a log-extracted
    # verdict (the reviewer process died before emitting a structured
    # result) from a clean structured completion. Distinct from
    # ``verdict_provenance`` (the mechanism that produced the verdict --
    # "fresh_llm_review", "ci_gate_auto_reject", ...). Only present when a
    # caller supplied it; the reap path threads ``session_metrics``'s
    # ``verdict_source`` here, while direct operator/CI-gate callers leave
    # it absent (no parser was involved).
    if verdict_source is not None:
        decision_payload["verdict_source"] = verdict_source
    # Issue #1485: when a verdict was extracted from a dead reviewer's
    # session artifacts (verdict_source = "log"/"events"/"file:*") rather
    # than emitted as a clean structured completion, it may be incomplete
    # or contain objectively wrong factual claims (e.g. "already merged",
    # "duplicate"). Persist a provenance caveat into the decision file so
    # every downstream surface (PR comment, rework brief, round-history)
    # can render it and a human knows to re-verify before acting on a
    # destructive-adjacent action like closing a PR. Only applies to
    # ``blocked``/``request_changes`` -- an ``approved`` verdict has no
    # required_changes to caveat, and the issue scopes to verdicts that
    # route to operator instructions.
    provenance_caveat: str | None = None
    if decision in {"request_changes", "blocked"}:
        provenance_caveat = _wf.provenance_caveat_for(verdict_source)
        if provenance_caveat is not None:
            decision_payload["provenance_caveat"] = provenance_caveat
    decision_path = pr_dir / "review-decision.json"
    # Issue #1268 (W11): per-round archive directory. ``round_number`` is
    # declared here (rather than inside the lock below) so it survives
    # past the lock for later use -- e.g. a comment header rendered after
    # the lock releases needs to reference the same round this call
    # recorded. It is always reassigned inside the lock before use; the
    # 0 default is never observed downstream.
    rounds_dir = pr_dir / "rounds"
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
        decision_payload["escalated"] = escalated
        pr_escalation_reason = (
            "max_rework_cycles_exceeded"
            if (decision == "request_changes" and escalated)
            else ("review_blocked" if decision == "blocked" else None)
        )
        # Persist the verdict BEFORE rendering the rework brief: the brief
        # reads review-decision.json itself (issue #632, single point of
        # enforcement) to surface required_changes, so the decision file
        # must be on disk first. A label-write failure or crash after this
        # point leaves a durable verdict and a brief consistent with it.
        #
        # Issue #934: ``decision_payload`` is a fresh dict that would
        # silently discard an ``authorized_override`` written by
        # ``merge_authorize``. The read-modify-write here is inside the
        # ``state_lock`` (which ``merge_authorize`` also holds), so re-reading
        # the file at this point is atomic against concurrent writers. Carry
        # the override forward so a subsequent ``record_review`` for the same
        # PR does not resurrect the false-positive tripwire finding this PR
        # exists to eliminate.
        if decision_path.exists():
            try:
                with decision_path.open("r", encoding="utf-8") as _handle:
                    _existing_decision = json.load(_handle)
                if isinstance(_existing_decision, dict) and isinstance(
                    _existing_decision.get("authorized_override"), dict
                ):
                    decision_payload["authorized_override"] = _existing_decision[
                        "authorized_override"
                    ]
            except (OSError, json.JSONDecodeError):
                pass
        # Issue #1268 (W11): archive this round's artifacts before a
        # later round's call can overwrite the live files above in
        # place. round_number is derived from the archive already on
        # disk -- never from pr_state or request_changes_count (see
        # _next_round_number's docstring) -- so a crash between this
        # write and save_state() below cannot desync the round this call
        # recorded from what is actually on disk. Computed here (not just
        # inside record_decision) because round_dir is also the archive
        # target for the rework-prompt/dispatch-note siblings below;
        # record_decision (issue #1362 Stage 2) derives the identical
        # number from the same rounds_dir/decision_payload pair, so the
        # two never disagree.
        round_number = _wf._next_round_number(rounds_dir, decision_payload)
        round_dir = rounds_dir / f"round-{round_number}"
        # Single writer (issue #1362 Stage 2): round-file-then-flat, so a
        # crash between the two writes still leaves a durable, readable
        # verdict via review_decision()'s round-fallback. head_sha=None:
        # decision_payload["reviewed_head_sha"] is already resolved above.
        _wf.record_decision(pr_dir, decision_payload, None)
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
            )
            if result.outcome != TransitionOutcome.APPLIED:
                label_error = {
                    "edge": "blocked",
                    "outcome": result.outcome.value,
                    "add_failures": result.add_failures,
                    "remove_failures": result.remove_failures,
                }
        elif decision == "approved":
            result = _wf.transition(self.gh, self.config.labels, issue_number, "review_approved")
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
        header = f"## Fleet review - round {round_number} - {decision}"
        body_parts = [header]
        if summary_text:
            body_parts.append(summary_text)
        # Issue #1485: surface the provenance caveat in the PR comment
        # before the required-changes section so an operator reading it
        # knows to re-verify factual claims before acting. ``provenance_
        # caveat`` is computed above (near the decision_payload build)
        # and is only set for log-extracted blocked/request_changes
        # verdicts.
        if provenance_caveat is not None:
            body_parts.append(provenance_caveat)
        # Issue #792's `findings_channel == "derived"` marker (set above,
        # ~15326) means `effective_required_changes` is not a real
        # structured list -- it is `[summary_text.strip()]`, the exact
        # same text just appended above as the summary paragraph.
        # `_render_required_changes_section` (the rework-brief renderer)
        # already treats this case specially for that reason -- tier-2
        # prose, never a one-item bullet wrapping a whole paragraph
        # (see its docstring) -- and this gate must honor the same
        # invariant for the same reason: rendering both here would post
        # the reviewer's entire summary twice in one comment. Every
        # other channel (a real structured list, or `"external"``'s
        # replacement list) is unaffected -- only `"derived"` is ever a
        # verbatim copy of `summary_text`.
        if effective_required_changes and findings_channel != "derived":
            body_parts.append(
                "### Required changes\n"
                + "\n".join(f"- {rc}" for rc in effective_required_changes)
            )
        comment_body = "\n\n".join(body_parts)
        try:
            self._comment_pr(pr_number, comment_body)
        except GitHubError as exc:
            # Comment failure is separate from label transition failure
            if label_error is None:
                label_error = {"comment_error": str(exc)}
            else:
                label_error["comment_error"] = str(exc)
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
    if label_error:
        message += f" (label update failed: {label_error.get('outcome', label_error)})"
    return _wf.CommandResult(
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
            "label_error": label_error,
        },
    )
