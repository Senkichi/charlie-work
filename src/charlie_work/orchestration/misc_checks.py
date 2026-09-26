"""Check-evaluation and merge-readiness delegates moved out of ``OrchestratorApp``.

Track 2 Phase B, L03 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each top-level ``def``
here unwrapped onto ``OrchestratorApp`` (``self`` binds via the descriptor
protocol exactly as the lexical methods did).

Workflow-defined names are reached through ``_wf.<name>`` so every existing
``charlie_work.workflow`` monkeypatch seam keeps landing: the ``CarryForwardCheck``
and ``CommandResult`` classes, and ``_calculate_patch_id`` (a helper a test patches
on the ``charlie_work.workflow`` namespace). Every other free name is imported
directly from its defining module (none is patched on ``charlie_work.workflow``).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import charlie_work.workflow as _wf
from charlie_work.checks import (
    CheckSummary,
    classify_infra_failures,
    is_infra_blocked_check,
    summarize_checks,
)
from charlie_work.janitor import (
    DiffContentSignature,
    _diff_content_signature,
    is_stale_ci_verdict,
    required_check_citation_names,
)


def _check_carry_forward(self, pr_number: int, decision: dict[str, Any]) -> _wf.CarryForwardCheck:
    """Determine whether ``decision``'s verdict can carry forward to the
    PR's live head, and via which tier (issues #411/#412, #414).

    Tier 1 (fast path): the live diff's stable patch-id equals the
    recorded ``reviewed_patch_id``. Issue #1187: ``git patch-id --stable``
    strips leading whitespace from ``+``/``-`` content lines, so two
    diffs that differ ONLY in indentation depth produce the identical
    patch-id — and in an indentation-sensitive language (Python) an
    indentation-only change can alter control flow (e.g. moving a
    ``return`` into or out of an ``if`` block). A tier-1 patch-id match
    alone is therefore NOT sufficient to carry forward a verdict: when
    a tier-2 line-content signature (which preserves whitespace verbatim)
    was recorded at review time, it is also validated and must match.
    If the tier-2 signatures differ — a whitespace-only change that
    patch-id collapsed — the check fails closed to stale rather than
    carrying forward an approved verdict across a semantically
    different, unreviewed head. Decisions that predate tier-2 (no
    signature stored) have no tier-2 baseline to consult and preserve
    #412's original patch-id-only carry-forward behavior. The tier-2
    binary eligibility gate does NOT apply here: patch-id already
    proved binary content identity, and tier-2 is consulted only for
    its text-line view, which is unaffected by binary payloads.

    Tier 2 (line-content, issue #414): patch-ids differ — which happens
    on every ordinary main advance, since ``git patch-id --stable``
    hashes hunk-boundary context and the merge-base moves whenever main
    does — but the ordered ``+``/``-`` line stream and changed-file set
    recorded at review time are identical to the live diff's. Reordered,
    added, removed, or altered lines, or a changed file set, are real
    content changes and do NOT carry forward. Tier 2 is INELIGIBLE
    whenever either side's diff touched a binary file: a binary payload
    emits no ``+``/``-`` lines, so the signature is blind to it — two
    diffs with genuinely different binary content at the same path
    would otherwise compare equal (review follow-up on issue #414).

    Fails closed (``tier=None``) on any missing data, diff-fetch
    failure, or binary content: a decision recorded before tier-2
    existed (no signature stored), a PR whose diff cannot be fetched,
    a binary file on either side, or a genuine content difference all
    report "cannot carry forward" — never carry forward on
    uncertainty. Tier 2 is pure string parsing of the diff text already
    fetched for tier 1 — it needs no additional git/gh calls and so has
    no failure mode of its own beyond that shared fetch.

    Eligibility for BOTH tiers is gated on ``reviewed_patch_id`` being
    recorded at all (matching #412's original behavior exactly): a
    "blocked" verdict, or any other decision that never computed one,
    has no baseline to compare against, full stop. A pure-rename or
    mode-only diff also has an empty ``reviewed_patch_id`` (no ``@@``
    hunk) despite having a valid tier-2 signature on file — that
    specific case is intentionally left conservative (stays stale)
    rather than gating on the signature fields' presence instead, which
    was tried and reverted: ``record_review`` unconditionally records a
    (possibly trivially-empty) signature for every approved/
    request_changes decision, so gating on "signature present" instead
    of "patch-id present" made an unrelated placeholder/no-op diff look
    like a valid tier-2 baseline and wrongly carried forward verdicts
    whose head had genuinely moved to unrelated content. Tracked as a
    narrow follow-up, not fixed here.
    """
    live_diff = self.gh.pr_diff(pr_number) or ""
    if not live_diff:
        return _wf.CarryForwardCheck(None, "", DiffContentSignature((), frozenset()))

    live_patch_id = _wf._calculate_patch_id(live_diff)
    live_signature = _diff_content_signature(live_diff)

    reviewed_patch_id = decision.get("reviewed_patch_id") or ""
    if not reviewed_patch_id:
        # No baseline recorded at all (e.g. a "blocked" verdict never
        # computes a patch-id) — nothing to compare against.
        return _wf.CarryForwardCheck(None, live_patch_id, live_signature)

    if live_patch_id and live_patch_id == reviewed_patch_id:
        # Issue #1187: ``git patch-id --stable`` strips leading
        # whitespace from ``+``/``-`` content lines, so two diffs that
        # differ only in indentation depth produce the identical
        # patch-id. In an indentation-sensitive language (Python), an
        # indentation-only change can alter control flow. A patch-id
        # match alone must not carry forward a verdict: validate the
        # tier-2 line-content signature (which preserves whitespace
        # verbatim) when one was recorded. The tier-2 binary gate does
        # NOT apply here — patch-id already proved binary content
        # identity, and tier-2 is consulted only for its text-line view.
        reviewed_changed_lines = decision.get("reviewed_changed_lines")
        reviewed_changed_files = decision.get("reviewed_changed_files")
        if reviewed_changed_lines is None or reviewed_changed_files is None:
            # Decision predates tier-2 (no signature recorded) —
            # patch-id is the only available signal; preserve #412's
            # original carry-forward behavior for legacy decisions.
            return _wf.CarryForwardCheck("patch-id", live_patch_id, live_signature)
        lines_match = tuple(reviewed_changed_lines) == live_signature.changed_lines
        files_match = frozenset(reviewed_changed_files) == live_signature.changed_files
        if lines_match and files_match:
            return _wf.CarryForwardCheck("patch-id", live_patch_id, live_signature)
        # Patch-id matched but tier-2 signatures differ — a
        # whitespace-only change that patch-id collapsed (issue #1187).
        # Fail closed to stale rather than carrying forward an approved
        # verdict across a semantically different, unreviewed head.
        return _wf.CarryForwardCheck(None, live_patch_id, live_signature)

    reviewed_changed_lines = decision.get("reviewed_changed_lines")
    reviewed_changed_files = decision.get("reviewed_changed_files")
    if reviewed_changed_lines is None or reviewed_changed_files is None:
        # Decision predates tier-2 (no signature recorded) — cannot
        # establish content identity; fail closed to stale.
        return _wf.CarryForwardCheck(None, live_patch_id, live_signature)

    if decision.get("reviewed_has_binary") or live_signature.has_binary:
        # A binary payload emits no +/- content lines, so the signature
        # cannot see it — never rely on its silence for content it
        # never observed (issue #414 review follow-up).
        return _wf.CarryForwardCheck(None, live_patch_id, live_signature)

    lines_match = tuple(reviewed_changed_lines) == live_signature.changed_lines
    files_match = frozenset(reviewed_changed_files) == live_signature.changed_files
    if lines_match and files_match:
        return _wf.CarryForwardCheck("line-content", live_patch_id, live_signature)

    return _wf.CarryForwardCheck(None, live_patch_id, live_signature)


def _still_valid_recorded_verdict(
    self, pr_number: int, live_head_sha: str | None
) -> tuple[dict[str, Any], str] | None:
    """Return ``(decision, reason)`` when PR ``pr_number`` carries a terminal
    review verdict (approved/request_changes/blocked) still valid at
    ``live_head_sha`` -- pinned to it exactly, or carrying forward to it via
    ``_check_carry_forward`` -- else ``None`` (no verdict, a ``pending``
    placeholder, or a genuinely stale verdict the carry-forward check
    rejects).

    Single point of enforcement for "is the recorded verdict still live",
    shared by two callers that both need to know before mutating anything:
    ``review_verdict_guard`` (refuses a destructive ``why-charlie-hate``
    re-review while one exists) and ``unescalate`` (issue #1765: must void
    one before re-arming a PR to the passive-open state, or the freshly
    reset PR is invisible to ``review_queue`` -- the exact rc=0-silent-no-op
    class ``unescalate`` exists to fix, moved one hop downstream).
    """
    decision = self._review_decision(pr_number)
    decision_value = decision.get("decision")
    if decision_value not in ("approved", "request_changes", "blocked"):
        return None
    reviewed_head_sha = decision.get("reviewed_head_sha")
    if live_head_sha is not None and reviewed_head_sha == live_head_sha:
        return (
            decision,
            f"the recorded {decision_value} verdict is still valid at the "
            f"live head ({live_head_sha})",
        )
    check = self._check_carry_forward(pr_number, decision)
    if not check.carry_forward:
        return None
    return (
        decision,
        f"the recorded {decision_value} verdict at {reviewed_head_sha} "
        f"carries forward to the live head ({live_head_sha}) via "
        f"{check.tier} carry-forward",
    )


def review_verdict_guard(self, pr_number: int) -> _wf.CommandResult | None:
    """CLI-boundary guard for ``charlie why-charlie-hate`` (issue #1695).

    ``review()`` unconditionally regenerates the review packet: when the
    recorded verdict is pinned to a superseded head it is voided back to a
    ``pending`` stub -- destroying the carry-forward baseline
    (``reviewed_patch_id`` and the tier-2 signature) -- the PR's state
    entry flips to ``reviewing``, and the ``review_started`` label
    transition fires. Run from the operator CLI that is destructive when
    the recorded verdict is still valid: a content-unchanged PR falls back
    to a full fresh review, burning a reviewer session and delaying a
    merge the queue would otherwise carry forward.

    Returns a refusal ``CommandResult`` (``ok=False``) when
    ``_still_valid_recorded_verdict`` finds a still-valid verdict, and
    ``None`` when there is nothing worth preserving. The loop's internal
    ``review()`` callers never consult this guard; it exists only at the
    CLI boundary, where a bare ``why-charlie-hate --pr N`` must not
    silently discard a verdict. The operator opts out explicitly with
    ``--force-rereview``.
    """
    pr = self.gh.pr_view(pr_number)
    if not pr:
        # review() itself reports the not-found case; nothing to preserve.
        return None
    result = self._still_valid_recorded_verdict(pr_number, pr.get("headRefOid"))
    if result is None:
        return None
    _decision, reason = result
    refusal = f"refusing to regenerate the review packet for PR #{pr_number}: {reason}"
    return _wf.CommandResult(
        False,
        # Issue #1695 acceptance: the refusal reason must survive `head -1`
        # AND `tail -1` of the rendered output, so it is restated on the
        # last line. ``data`` stays empty -- print_result appends a JSON
        # block for a non-empty payload, whose closing brace would become
        # the last output line and hide the reason.
        f"{refusal}\n{refusal} -- pass --force-rereview to discard it",
        {},
    )


def _enrich_checks_infra_blocked(
    self, checks: list[dict[str, Any]] | None, required: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Reclassify FAILURE required checks as ``INFRA_BLOCKED`` at the
    check-ingestion data boundary (issue #1383).

    Single point of enforcement for the infra_blocked classification: a
    required check whose FAILED job shows structural evidence of a
    non-started job (zero non-setup steps / instant-fail) or a
    config-listed billing annotation is rewritten to the
    ``INFRA_BLOCKED`` marker state before ``summarize_checks`` (and
    therefore the janitor gate / rework-routing decision) ever sees it.
    ``summarize_checks`` then routes it to ``CheckSummary.infra_blocked``
    rather than ``failed``, so it never enters the "required checks
    failed -> dispatch rework" path and never burns a rework attempt.

    Kept as an OrchestratorApp method (not a pure function in
    ``checks.py``) because the structural + annotation signals require
    two ``gh`` API calls per FAILURE check (``actions_job`` /
    ``check_run_annotations``) -- I/O that the pure
    ``summarize_checks``/``is_infra_blocked_check`` layer must not
    perform. Both API methods return safe empty values (``None`` / ``[]``)
    on any GitHub failure and never raise, so an unenrichable check
    degrades to ordinary ``failed`` routing rather than crashing
    ingestion -- mirroring the existing ``merge_ready`` enrichment this
    replaces.

    Called before the janitor gate in ``review()`` (the rework-routing
    path) and in ``merge_ready()`` (the merge-execution path) so both
    paths classify budget-failed checks identically.
    """
    if not checks or not required:
        return list(checks or [])
    cfg = self.config.auto_merge.infra_blocked
    if not cfg.enabled:
        return list(checks)
    required_set = set(required)
    enriched: list[dict[str, Any]] = []
    for check in checks:
        name = str(check.get("name") or "")
        if name in required_set and str(check.get("state") or "").upper() == "FAILURE":
            check_run_id = check.get("databaseId")
            if isinstance(check_run_id, int):
                job = self.gh.actions_job(check_run_id)
                annotations = self.gh.check_run_annotations(check_run_id)
                if job is not None and is_infra_blocked_check(job, annotations, cfg):
                    check = {**check, "state": "INFRA_BLOCKED"}
        enriched.append(check)
    return enriched


def _is_stale_ci_request_changes(self, pr_number: int, decision: dict[str, Any]) -> bool:
    """True when ``decision`` is a request_changes verdict whose only
    findings cite required checks that are all green on the live head
    (issue #1111 staleness predicate, network half).

    The pure text-shape check (``required_check_citation_names``) runs
    first so ``gh pr checks`` is only fetched for the small set of
    verdicts that could possibly be stale. Checks-unavailable (``None``)
    fails closed to False — the verdict keeps its normal lifecycle.
    """
    required = self.config.auto_merge.required_checks
    if not required:
        return False
    if required_check_citation_names(decision, required) is None:
        return False
    checks = self.gh.pr_checks(pr_number)
    if checks is None:
        return False
    return is_stale_ci_verdict(decision, summarize_checks(checks, required))


def _merge_not_ready_result(
    self,
    pr_number: int,
    issue_number: int | None,
    decision: dict[str, Any],
    existing_pr_state: dict[str, Any],
) -> _wf.CommandResult:
    """Return a non-mergeable result for an approved PR that is not the train head."""
    return _wf.CommandResult(
        True,
        f"PR #{pr_number} is not the head of the merge-train queue",
        {
            "pr": pr_number,
            "issue": issue_number,
            "can_merge": False,
            "auto_merge_enabled": self.config.auto_merge.enabled,
            "merged": False,
            "merge_output": None,
            "branch_deleted": None,
            "review_decision": decision,
            "checks": asdict(summarize_checks([], self.config.auto_merge.required_checks)),
            "checks_unavailable": False,
            "label_error": None,
            "update_open_prs_results": None,
            "cancel_superseded_runs_results": None,
            "containment_warnings": [],
            "consecutive_failed_merge_attempts": existing_pr_state.get(
                "consecutive_failed_merge_attempts", 0
            ),
            "merge_attempt_alarm": False,
            "merge_attempt_warning": None,
            "merge_conflict": False,
        },
    )


def _drive_infra_rerun_or_escalate(
    self,
    pr_number: int,
    issue_number: int | None,
    *,
    head_sha: Any,
    rerun_run_ids: tuple[int, ...],
    infra_rerun_attempts: dict[str, Any],
    definitive_failed: tuple[str, ...],
    escalate_exhausted: bool,
    ok: bool,
    extra_data: dict[str, Any] | None = None,
) -> _wf.CommandResult | None:
    """Drive the #841 infra-failure remediation mechanics: bounded
    ``gh run rerun`` per eligible run id, then operator-queue escalation
    once a check's run ids are exhausted (or none was parseable).

    Extracted out of ``review()`` for issue #1912 so ``merge_ready()`` can
    run the identical mechanics on carried-forward approved verdicts --
    previously this block was review()-only, so an approved PR whose
    verdict carried forward to the live head (the loop() already_approved
    fast path routes it straight to ``merge_ready``) got no rerun and no
    escalation for a cancelled required check, looping forever behind only
    a diagnostic ``merge_failed_attempt_alarm`` event (live instance:
    swole PR #349 / issue #174). A single shared driver keeps the two
    lanes from drifting apart again.

    A job-level ``timeout-minutes`` kill is an infra failure (CANCELLED on
    the self-hosted-era runners, possibly TIMED_OUT on hosted runners),
    which ``summarize_checks`` buckets into ``infra_failed`` via
    ``_classify_check_run`` -- both conclusions route to the infra bucket,
    so the rerun path matches regardless of which runner reports. There is
    no code-fix rework path for an infra kill.

    ``gh run rerun RUN_ID`` is dispatched WITHOUT ``--failed``: the job
    never completed (cancelled/timed out, not failed), so ``--failed``'s
    "rerun the failed jobs in this run" semantics do not apply -- omitting
    it reruns the whole run, which is the correct behavior for a run that
    never produced a completed job to target.

    ``rerun_run_ids`` / ``infra_rerun_attempts`` / ``definitive_failed``
    are the caller's ``classify_infra_failures`` output (``review()`` reads
    them off the ``JanitorVerdict``; ``merge_ready()`` classifies itself).
    ``escalate_exhausted`` is the caller's sole-blocker flag
    (``verdict.is_infra_failure_block`` in ``review()``) -- when False,
    ``definitive_failed`` names checks that were deliberately not retried
    this pass because other blockers co-occur, which must not escalate.

    Returns the ``CommandResult`` the caller should return early when a
    rerun was dispatched or the cap was escalated, else ``None`` -- a
    rerun API error records ``infra_rerun_failed`` without consuming the
    attempt and falls through so the caller's own bookkeeping can record
    the still-blocked pass. ``ok`` and ``extra_data`` preserve each
    caller's result shape: ``review()`` returns ``ok=False`` (no review
    produced this pass) while ``merge_ready()`` returns ``ok=True`` (a
    healthy, still-unmergeable pass).
    """
    if rerun_run_ids:
        infra_rerun_errors: list[str] = []
        infra_triggered_run_ids: list[int] = []
        for run_id in rerun_run_ids:
            result = self.gh.run(["run", "rerun", str(run_id)], allow_failure=True)
            if isinstance(result, _wf.GitHubRunResult):
                if result.ok:
                    infra_triggered_run_ids.append(run_id)
                else:
                    infra_rerun_errors.append(
                        result.error or f"gh run rerun {run_id} exited {result.returncode}"
                    )
            elif isinstance(result, str):
                # Dry-run returns a descriptive string; treat as success.
                infra_triggered_run_ids.append(run_id)
            else:
                infra_rerun_errors.append(
                    f"unexpected result from gh run rerun {run_id}: {result!r}"
                )

        if infra_triggered_run_ids and not infra_rerun_errors:
            with _wf.state_lock(self.paths.state_file):
                state = _wf.load_state(self.paths.state_file)
                state["prs"][str(pr_number)] = {
                    **state["prs"].get(str(pr_number), {}),
                    "number": pr_number,
                    "issue_number": issue_number,
                    "infra_rerun_attempts": infra_rerun_attempts,
                }
                state = self._record_event(
                    state,
                    "infra_rerun_triggered",
                    {
                        "pr_number": pr_number,
                        "run_ids": infra_triggered_run_ids,
                        "head_sha": head_sha,
                    },
                )
                self.write_gate.save_state(state)
            return _wf.CommandResult(
                ok,
                f"infra rerun triggered for PR #{pr_number}: run(s) "
                + ", ".join(str(rid) for rid in infra_triggered_run_ids),
                {
                    "pr": pr_number,
                    "issue": issue_number,
                    **(extra_data or {}),
                    "infra_rerun_run_ids": infra_triggered_run_ids,
                },
            )

        # Rerun API error: record it, but do not consume the attempt.
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            state = self._record_event(
                state,
                "infra_rerun_failed",
                {
                    "pr_number": pr_number,
                    "run_ids": list(rerun_run_ids),
                    "errors": infra_rerun_errors,
                },
            )
            self.write_gate.save_state(state)

    if escalate_exhausted and issue_number is not None and definitive_failed:
        # Attempt cap exhausted (or no parseable run id at all): there is no
        # code-fix rework path for an infra failure, so escalate straight to
        # a human instead of looping forever on a PR that can never clear
        # the gate on its own -- this is the bug issue #841 fixes
        # (previously: a diagnostic merge_failed_attempt_alarm event and
        # nothing else).
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            state = _wf._escalate_issue(
                state,
                issue_number,
                reason="infra_rerun_cap_exceeded",
                reason_class="mechanical",
                pr_number=pr_number,
                pr_extra={"infra_rerun_attempts": infra_rerun_attempts},
            )
            state = self._record_event(
                state,
                "infra_rerun_escalated",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "checks": list(definitive_failed),
                },
            )
            self.write_gate.save_state(state)
        edge = _wf._escalation_edge("escalated", "mechanical")
        transition_result = self.write_gate.transition(
            self.gh, self.config.labels, issue_number, edge
        )
        label_error = None
        if transition_result.outcome != _wf.TransitionOutcome.APPLIED:
            label_error = {
                "edge": edge,
                "outcome": transition_result.outcome.value,
                "add_failures": transition_result.add_failures,
                "remove_failures": transition_result.remove_failures,
            }
        return _wf.CommandResult(
            ok,
            f"PR #{pr_number} infra-failed check(s) exhausted rerun cap: "
            + ", ".join(definitive_failed),
            {
                "pr": pr_number,
                "issue": issue_number,
                **(extra_data or {}),
                "infra_escalated": True,
                "label_error": label_error,
            },
        )

    return None


def _merge_ready_infra_remediation(
    self,
    pr_number: int,
    pr: dict[str, Any],
    issue_number: int | None,
    decision: dict[str, Any],
    enriched_checks: list[dict[str, Any]],
    summary: CheckSummary,
    *,
    approved: bool,
    sync_failed: bool,
    merge_conflict: bool,
) -> _wf.CommandResult | None:
    """Infra-failure remediation for the merge lane (issue #1912).

    The #841 rerun/escalate mechanics lived only in ``review()`` -- fed by
    ``run_janitor``'s ``classify_infra_failures`` call -- but an approved
    PR whose verdict carries forward to the live head never re-enters
    ``review()`` (``loop()``'s already_approved fast path calls
    ``merge_ready`` directly), so a CANCELLED/TIMED_OUT required check on
    that head was retried by nothing and escalated to nobody; the PR
    looped forever behind only a diagnostic ``merge_failed_attempt_alarm``
    (live instance: swole PR #349 / issue #174). This lane gives
    ``merge_ready()`` the identical behavior: classify the live head's
    checks the same way the janitor does, then drive the shared
    ``_drive_infra_rerun_or_escalate`` mechanics.

    The gate mirrors the janitor's ``is_infra_failure_block`` sole-blocker
    predicate, translated to merge_ready's own blocker vocabulary:

    - ``summary.infra_failed`` non-empty (a CANCELLED/INFRA_FAILURE/
      TIMED_OUT required check on the live head);
    - no co-occurring ``summary.failed`` (a genuine code failure is owned
      by the check-failure rework lane), ``summary.missing`` (owned by the
      readiness-no-CI stall gate), ``summary.unavailable`` (covers a
      ``gh pr checks`` fetch failure -- every required check lands there),
      or ``summary.infra_blocked`` (fleet-wide Actions budget/runner
      outage, #1383 -- held by its own lane, never per-run rerun);
    - pending checks are deliberately NOT excluded, matching the janitor
      predicate -- rerunning a cancelled check while a sibling is still
      in flight heals it sooner, and the merge gate stays closed either
      way;
    - the janitor's non-check blockers map to merge_ready's own:
      approval (``approved``, or approval not required), ``sync_failed``
      (merge conflict, failed branch sync, and cross-PR revert all resolve
      through lanes that move the head, making a rerun against this one
      wasted), draft (the janitor counts draft as a blocker), and a linked
      issue (the janitor counts its absence as a blocker -- and escalation
      needs the target);
    - neither the PR nor the linked issue already escalated -- review()'s
      entry gate treats escalation as terminal for automated remediation
      (the pass_skipped early return), and without the same exclusion here
      a cap-exhausted check would re-fire ``infra_rerun_escalated`` plus
      its label transition every pass after this lane's own escalation.

    ``classify_infra_failures`` is called with ``record_attempts=True`` --
    the gate above is exactly the condition under which the janitor passes
    ``record_attempts=is_infra_failure_block`` -- and is fed
    ``enriched_checks`` (post-#1383 data-boundary checks, the same input
    the janitor sees in ``review()``); the enrichment only rewrites
    FAILURE conclusions, never the CANCELLED/INFRA_FAILURE/TIMED_OUT
    conclusions this lane consumes.

    Returns the ``CommandResult`` ``merge_ready()`` should return early
    when a rerun was dispatched or the cap escalated, else ``None`` -- a
    rerun API error already recorded ``infra_rerun_failed`` without
    consuming the attempt, so the caller's normal bookkeeping records the
    still-blocked pass (counter/alarm path unchanged).
    """
    if (
        (not approved and self.config.auto_merge.require_approved_review)
        or sync_failed
        or issue_number is None
        or bool(pr.get("isDraft"))
        or not summary.infra_failed
        or summary.failed
        or summary.missing
        or summary.unavailable
        or summary.infra_blocked
    ):
        return None
    snap = _wf.load_state_locked(self.paths.state_file)
    pr_escalated, issue_escalated = _wf._escalation_flags(
        snap.get("prs", {}).get(str(pr_number), {}),
        snap.get("issues", {}).get(str(issue_number), {}),
    )
    if pr_escalated or issue_escalated:
        return None
    pr_state = snap.get("prs", {}).get(str(pr_number), {})
    debounce = classify_infra_failures(
        enriched_checks,
        self.config.auto_merge.required_checks,
        pr_state,
        str(pr.get("headRefOid") or "") or None,
        record_attempts=True,
        attempt_cap=self.config.auto_merge.infra_rerun_attempt_cap,
    )
    return self._drive_infra_rerun_or_escalate(
        pr_number,
        issue_number,
        head_sha=pr.get("headRefOid"),
        rerun_run_ids=debounce.rerun_run_ids,
        infra_rerun_attempts=debounce.infra_rerun_attempts,
        definitive_failed=debounce.definitive_failed,
        escalate_exhausted=True,
        ok=True,
        extra_data={
            "can_merge": False,
            "merged": False,
            "review_decision": decision,
            "checks": asdict(summary),
            "checks_unavailable": False,
            "consecutive_failed_merge_attempts": int(
                pr_state.get("consecutive_failed_merge_attempts", 0)
            ),
            "merge_attempt_alarm": False,
            "merge_attempt_warning": None,
            "merge_conflict": merge_conflict,
            "label_error": None,
        },
    )
