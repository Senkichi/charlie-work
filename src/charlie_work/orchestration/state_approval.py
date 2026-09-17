"""Approval-head and unauthorized-merge announcement delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L01 batch 4 (issue #1647, parent #1632, umbrella
#1582). Method bodies moved verbatim from ``OrchestratorApp`` in
``charlie_work.workflow``; the ``workflow_delegation`` installer re-attaches
each ``def`` onto the class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work.janitor import DiffContentSignature
from charlie_work.review_decision import ReviewDecision, record_decision
from charlie_work.state import StateLockBusy
import charlie_work.workflow as _wf


def _update_approval_head(
    self,
    pr_number: int,
    decision: dict[str, Any],
    new_head: str,
    old_head: str | None = None,
    *,
    issue_number: int | None = None,
    tier: str = "verified-sync",
    new_patch_id: str | None = None,
    new_signature: DiffContentSignature | None = None,
) -> bool:
    """Persist an updated review head for a PR whose branch was synced.

    Keeps the verdict valid when the branch was base-updated or rebased
    without content changes. Updates both review-decision.json and
    state.json, appending the old head to ``carried_forward_from`` for audit.

    ``tier`` records which mechanism justified the update, for audit:
    ``"patch-id"``/``"line-content"`` from :meth:`_check_carry_forward`
    (issues #412/#414), or the default ``"verified-sync"`` for the
    structurally-verified (``_verify_synced_head``) front-of-train/
    broadcast sync callers, which never compared patch-ids at all.

    When ``new_patch_id``/``new_signature`` are supplied — they are
    already computed as part of the tier check that led here — they
    replace the stored baseline so ``reviewed_head_sha`` and the
    patch-id/content-signature fields always describe the SAME head
    consistently. Leaving a patch-id recorded against a stale head would
    pointlessly defeat that head's own future tier-1 fast path: patch-id
    is unstable across every main advance, not just the one just
    carried past.

    A carry-forward event is recorded here, inside the locked section
    that already persists the transition, so every call site is
    instrumented by construction — a new caller cannot forget it
    (issue #638). The event kind is tier-dependent so the three
    mechanisms (``verdict_carried_forward_clean_rebase`` for patch-id,
    ``verdict_carried_forward_line_content`` for line-content, and
    ``verdict_carried_forward_verified_sync`` for the structurally-
    verified sync that never compared patch-ids) stay separately
    auditable in the events log. Callers that previously recorded the
    event themselves must NOT do so anymore, or the transition would be
    double-counted.

    Issue #1038: ``decision`` is the caller's copy, read before one or
    more network round-trips (``pr_update_branch``, ``_verify_synced_head``,
    ``_check_carry_forward``'s diff fetch). This used to write the whole
    of ``review-decision.json`` from that copy, outside any lock, so a
    verdict recorded on disk during the round-trips (e.g. an operator's
    ``record_review`` landing a fresh ``request_changes``) was silently
    replaced by the stale copy and re-pinned to the new head — an
    approval could be resurrected over a contemporaneous rejection,
    authorizing a merge that should not happen.

    Now the on-disk decision is re-read inside the same ``state_lock``
    that guards the ``state.json`` half of this transition (previously
    the decision-file write sat entirely outside that lock), and the
    write is refused unless the on-disk verdict's identity — its
    ``decision`` value, plus ``reviewed_head_sha`` matching ``old_head``
    when the caller supplied one — still matches what the caller
    observed. Only the fields this function owns (``reviewed_head_sha``,
    ``carry_forward_tier``, the patch-id/signature fields,
    ``carried_forward_from``) are patched onto the on-disk copy;
    ``decision``, ``summary``, and ``required_changes`` always come from
    disk, never from the caller's stale parameter — this is a
    read-modify-write, never a whole-dict replace.

    Returns True if the carry-forward was applied, False if it was
    skipped because the on-disk verdict had already changed underneath
    it (an event is still recorded in that case so the skip is
    greppable rather than silent).
    """
    decision_path = self.paths.prs / f"pr-{pr_number}" / "review-decision.json"
    expected_decision_value = decision.get("decision")

    with _wf.state_lock(self.paths.state_file):
        current_decision = self._review_decision(pr_number)
        decision_changed = current_decision.get("decision") != expected_decision_value
        head_changed = (
            old_head is not None and current_decision.get("reviewed_head_sha") != old_head
        )
        if decision_changed or head_changed:
            state = _wf.load_state(self.paths.state_file)
            state = self._record_event(
                state,
                "verdict_carry_forward_skipped_stale",
                {
                    "pr_number": pr_number,
                    "issue_number": issue_number,
                    "expected_decision": expected_decision_value,
                    "expected_reviewed_head_sha": old_head,
                    "on_disk_decision": current_decision.get("decision"),
                    "on_disk_reviewed_head_sha": current_decision.get("reviewed_head_sha"),
                    "attempted_new_head_sha": new_head,
                    "carry_forward_tier": tier,
                },
                level="warning",
            )
            _wf.save_state(self.paths.state_file, state)
            return False

        updated_decision = dict(current_decision)
        updated_decision["reviewed_head_sha"] = new_head
        updated_decision["carry_forward_tier"] = tier
        # Issue #1265: carry-forward always stamps "carried_forward",
        # unconditionally overwriting whatever provenance the verdict
        # being carried had -- this write's own mechanism IS the
        # provenance of the resulting (still-valid) verdict, regardless
        # of how the original decision was produced.
        updated_decision["verdict_provenance"] = "carried_forward"
        if new_patch_id is not None:
            updated_decision["reviewed_patch_id"] = new_patch_id
        if new_signature is not None:
            updated_decision["reviewed_changed_lines"] = list(new_signature.changed_lines)
            updated_decision["reviewed_changed_files"] = sorted(new_signature.changed_files)
            updated_decision["reviewed_has_binary"] = new_signature.has_binary
        carried_forward: list[str] = list(updated_decision.get("carried_forward_from", []))
        if old_head is not None and old_head != new_head and old_head not in carried_forward:
            carried_forward.append(old_head)
        updated_decision["carried_forward_from"] = carried_forward
        # Single writer (issue #1362 Stage 2): archive_round=False -- a
        # carry-forward is a mechanical head-rebind, not a new reviewer
        # verdict, even though it changes reviewed_head_sha (one of
        # _ROUND_COMPARE_KEYS). Routing it through the default round-mint
        # logic would archive a content-free duplicate "round" (see
        # record_decision's archive_round docstring). head_sha=None:
        # reviewed_head_sha is already set to new_head above.
        record_decision(decision_path.parent, updated_decision, None, archive_round=False)

        if tier == "patch-id":
            event_kind = "verdict_carried_forward_clean_rebase"
        elif tier == "line-content":
            event_kind = "verdict_carried_forward_line_content"
        else:
            event_kind = "verdict_carried_forward_verified_sync"

        state = _wf.load_state(self.paths.state_file)
        pr_state = state["prs"].get(str(pr_number), {})
        state["prs"][str(pr_number)] = {
            **pr_state,
            "number": pr_number,
            "decision": pr_state.get("decision") or updated_decision.get("decision") or "approved",
            "status": pr_state.get("status") or updated_decision.get("decision") or "approved",
            "reviewed_head_sha": new_head,
            "reviewed_patch_id": new_patch_id
            if new_patch_id is not None
            else (
                pr_state.get("reviewed_patch_id")
                or updated_decision.get("reviewed_patch_id")
                or ""
            ),
            "carry_forward_tier": tier,
            "carried_forward_from": carried_forward,
        }
        state = self._record_event(
            state,
            event_kind,
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "old_reviewed_head_sha": old_head,
                "new_head_sha": new_head,
                "patch_id": new_patch_id,
                "carry_forward_tier": tier,
                "carried_forward_from": carried_forward,
                "verdict_provenance": "carried_forward",
            },
        )
        _wf.save_state(self.paths.state_file, state)
    return True


def _refresh_pr_decision_cache(
    self, pr_number: int, decision: ReviewDecision, decision_path: Path
) -> None:
    """Mirror the file-first decision into ``state["prs"][pr_number]`` (issue #1362 Stage 3).

    ``state.json``'s ``decision``/``reviewed_head_sha``/``decision_path``
    fields are a declared *cache* of the file, refreshed from
    ``review_decision()`` at exactly one boundary: the start of each PR's
    evaluation in ``loop()``'s per-PR dispatch block (this method's only
    call site), plus the four writer-adjacent mirrors that already run
    immediately after a ``record_decision()`` call in the same pass
    (``record_review``, ``_route_to_rework``, ``_update_approval_head``,
    ``merge_ready``'s carry-forward branch, and ``review``'s new-dispatch
    placeholder write) -- those keep a PR whose verdict changed THIS pass
    current without waiting for the NEXT pass's boundary refresh; this
    method covers every OTHER *already-tracked* PR (one already present
    in ``state["prs"]``), so the cache never goes stale between
    verdict-producing passes for a PR this method is willing to touch at
    all -- see the ``pr_key not in state["prs"]`` early-return below for
    the one case (a PR not yet dispatched this pass) that instead waits
    for its normal dispatch path to create the full entry first. See
    ``tests/test_state_decision_cache_enforcement.py`` for the enforcement
    that no other production site may set these keys.

    Skips the write entirely when the cache already agrees with the file
    (the common case: no verdict activity since the last pass), so a
    healthy fleet does not pay a state.json write per tracked PR per pass.
    Uses ``self.write_gate.save_state`` (never the raw ``save_state``
    primitive) so this method is itself WriteGate-exclusive under issue
    #1264's R9 predicate -- it must never gain a second, ungated write.
    """
    decision_path_str = str(decision_path)
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        pr_key = str(pr_number)
        if pr_key not in state["prs"]:
            # No PR-state entry exists yet for this PR (it has not been
            # dispatched/tracked this pass) -- refreshing here would
            # materialize a decision-only partial entry with no
            # status/counters, a state shape the four writer-adjacent
            # mirrors never produce. Let the PR's normal dispatch path
            # create the full entry; the boundary refresh will cover it
            # on a later pass once that entry exists.
            return
        pr_state = state["prs"][pr_key]
        if (
            pr_state.get("decision") == decision.decision
            and pr_state.get("reviewed_head_sha") == decision.reviewed_head_sha
            and pr_state.get("decision_path") == decision_path_str
        ):
            return
        state["prs"][pr_key] = {
            **pr_state,
            "decision": decision.decision,
            "reviewed_head_sha": decision.reviewed_head_sha,
            "decision_path": decision_path_str,
        }
        self.write_gate.save_state(state)


def _stamp_mention_rearm(
    self,
    state: dict[str, Any],
    newly_rearmed: list[int],
) -> dict[str, Any]:
    """Issue #1336: durably stamp ``mention_rearmed_at`` and emit the
    ``dispatch_merged_pr_mention_rearmed`` event for issues the operator
    re-armed this pass.

    Routed through WriteGate (Convention A: ``self.write_gate.*``) so the
    R9 shrink-only ratchet on workflow.py's raw-primitive count (issue
    #1264 W6 PR4) is not increased -- the re-arm writes are new territory
    this wave does not convert, and a raw ``append_event``/``save_state``
    pair would trip the ratchet (baseline 266). The existing flag block in
    ``_dispatch_impl`` stays raw (it is part of the ratchet's baseline);
    only the NEW re-arm writes go through the gate.

    Returns ``state`` (possibly updated by ``append_event``). In dry-run
    mode the gate no-ops and ``state`` is returned unchanged -- but this
    method is only called from the non-dry-run dispatch path (the dry-run
    early return precedes it), so the dry-run no-op is defence-in-depth.
    """
    for issue_number in newly_rearmed:
        _ra_key = str(issue_number)
        _ra_entry = state["issues"].get(_ra_key, {})
        state["issues"][_ra_key] = {
            **_ra_entry,
            "number": issue_number,
            "mention_rearmed_at": _wf.utc_now(),
        }
    if newly_rearmed:
        state = self.write_gate.append_event(
            state,
            "dispatch_merged_pr_mention_rearmed",
            {"issue_numbers": sorted(newly_rearmed)},
        )
        self.write_gate.save_state(state)
    return state


def _record_unauthorized_merge_skip(self, exc: Exception) -> None:
    """Record that the unauthorized-merge tripwire did not run this pass (issue #937).

    ``_detect_unauthorized_merges`` fails open on a ``GitHubError``: it returns
    ``[]`` without arming, which is the right call — an unusable response must
    never bake an empty baseline. But returning ``[]`` silently makes a blinded
    pass byte-identical to a clean one in every stored artifact, discarding the
    distinction ``merged_pr_list`` was deliberately changed to create (#633).

    Unlike ``unauthorized_merge_detected``, this fires **once per pass**, not
    once per PR, and that difference is deliberate rather than an inconsistency.
    A finding is one fact that stays true across passes, so repeating it is
    noise. A skipped check is a *distinct occurrence each time*: two blinded
    passes are two windows in which an unauthorized merge could have landed
    unseen, and collapsing them would destroy exactly the count that makes the
    gap measurable.

    Classified ``warning``, not ``error``: nothing is broken and no finding is
    being suppressed — a control simply did not run. That matches
    ``session_rate_limit_deferred`` / ``review_quota_exhausted`` (handled
    degradations) rather than the error tier, which is reserved for things
    needing triage.

    Best-effort by construction: this is the reporting path for a failure that
    has already happened, so it must never turn a degraded pass into a crashed
    one.
    """
    import logging

    logger = logging.getLogger(__name__)
    reason = str(exc) or exc.__class__.__name__
    logger.warning(
        "unauthorized-merge tripwire did not run this pass: %s "
        "(failing open; no findings will be reported until gh recovers)",
        reason,
    )
    if self.dry_run:
        return
    try:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            state = self._record_event(
                state,
                "unauthorized_merge_check_skipped",
                {"reason": reason, "error_type": exc.__class__.__name__},
            )
            _wf.save_state(self.paths.state_file, state)
    except (OSError, ValueError, StateLockBusy) as write_exc:  # pragma: no cover
        # StateLockBusy is a RuntimeError and so escapes the other two. This
        # handler's whole promise is that recording a degraded pass never
        # crashes it; contention on the state lock is the single most likely
        # reason this write fails, so omitting it left the promise unmet in
        # exactly its expected case.
        logger.warning("could not record unauthorized_merge_check_skipped: %s", write_exc)


def _announce_unauthorized_merges(self, reported: list[dict[str, Any]]) -> None:
    """Emit ``unauthorized_merge_detected`` once per PR, the first time it is reported.

    The acknowledgement of a finding was durably recorded
    (``unauthorized_merge_acknowledged``); the *finding itself* was not. A
    control whose triage is auditable but whose alarm is not is only half a
    control — and it is why a live finding took a source-reading session to
    identify instead of one query (issue #933).

    Fires on the post-baseline, post-ack set — exactly the candidates that
    become ``errors`` entries and pin ``ok=False`` — and is keyed durably in
    ``UNAUTHORIZED_MERGE_DETECTED_KEY`` so a finding that persists for 21
    passes produces one event, not 21. See that constant for why the record
    deliberately does not suppress the finding itself.

    Classified ``error`` by ``instrumentation._classify_level``, which closes
    the second of the two misreads in #933: a session reported "0 errors
    since restart" from ``query_events(level='error')`` while this very
    finding was pinning every pass, because nothing on the finding path ever
    set that level.

    Best-effort by construction: a state failure here must not crash a fleet
    pass on the *reporting* path of a security control. The finding is
    returned to the caller regardless.
    """
    import logging

    if not reported or self.dry_run:
        # dry_run: a preview must not write state (issues #609/#613/#621),
        # matching _apply_unauthorized_merge_baseline's arming path.
        return

    logger = logging.getLogger(__name__)
    key = _wf.UNAUTHORIZED_MERGE_DETECTED_KEY
    try:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            record = state.get(key)
            if not isinstance(record, dict):
                record = {}
            fresh = [c for c in reported if str(c["pr"]) not in record]
            if not fresh:
                return
            detected_at = _wf.utc_now()
            for candidate in fresh:
                record[str(candidate["pr"])] = {
                    "detected_at": detected_at,
                    "issue": candidate.get("issue"),
                    "head": candidate.get("head"),
                    "decision": candidate.get("decision"),
                    "reviewed_head_sha": candidate.get("reviewed_head_sha"),
                    "live_head_sha": candidate.get("live_head_sha"),
                    "review_dispatch_enabled": candidate.get("review_dispatch_enabled"),
                    # Issue #1194 retry fix: present only for candidates that
                    # went through _queue_sync_merge_covered (approved +
                    # head-mismatch + no override) and were not covered --
                    # None for every other finding shape (unapproved,
                    # missing decision, ...). See _QueueSyncCoverageResult.
                    "coverage_check": candidate.get("coverage_check"),
                    "coverage_check_error": candidate.get("coverage_check_error"),
                    "coverage_reason": candidate.get("coverage_reason"),
                }
            state[key] = record
            for candidate in fresh:
                state = self._record_event(
                    state,
                    "unauthorized_merge_detected",
                    {
                        "pr": candidate["pr"],
                        "issue": candidate.get("issue"),
                        "head": candidate.get("head"),
                        "decision": candidate.get("decision"),
                        "reviewed_head_sha": candidate.get("reviewed_head_sha"),
                        "live_head_sha": candidate.get("live_head_sha"),
                        "review_dispatch_enabled": candidate.get("review_dispatch_enabled"),
                        "coverage_check": candidate.get("coverage_check"),
                        "coverage_check_error": candidate.get("coverage_check_error"),
                        "coverage_reason": candidate.get("coverage_reason"),
                    },
                )
            _wf.save_state(self.paths.state_file, state)
    except (OSError, ValueError) as exc:
        logger.warning("unauthorized-merge detection record not persisted: %s", exc)
        return

    logger.error(
        "unauthorized-merge tripwire: %d new uncovered merge(s) detected (%s); "
        "ack with `charlie tripwire ack <pr> --reason ...` once triaged",
        len(fresh),
        ", ".join(f"#{c['pr']}" for c in fresh),
    )
