"""Stale-checks retrigger/escalation delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L01 batch 1 (issue #1632, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from charlie_work.labels import TransitionOutcome
import charlie_work.workflow as _wf


def _attempt_stale_checks_retrigger(
    self,
    pr: dict[str, Any],
    *,
    pr_number: int,
    issue_number: int,
    head_sha: str,
    existing_pr_state: dict[str, Any],
) -> _wf.CommandResult | None:
    """Follow-up retrigger policy for ``_detect_ci_run_never_created``
    (issue #1274, W17).

    Called from ``review()``'s main (non-escalated) janitor-gate path
    only, once the caller has already established the current head is
    marked (persisted or freshly detected) as never having had an
    Actions run created for it, and that ``verdict.missing_required_checks``
    is still non-empty this pass. This method owns only the *policy*
    layer on top of that: the attempt cap, the post-retrigger grace
    wait, and the mechanical retrigger itself. It never re-derives the
    "never created" signal on its own -- that stays the sole
    responsibility of ``_detect_ci_run_never_created`` (binding comment
    item 2 on issue #1274: two independently-tuned grace periods gating
    the same underlying condition is the invalid-state smell this
    codebase's design explicitly avoids).

    Mechanism, in order (binding comment item 5): close then reopen the
    PR; an empty-commit push to its branch is a fallback used only when
    close/reopen does not mechanically succeed. Both paths increment the
    SAME ``stale_checks_retrigger_attempts`` counter -- one shared
    budget, not two. A transient ``gh`` API error consumes no attempt
    (mirrors the flake-rerun block's "record the error but do not
    consume the attempt" convention a few hundred lines up in
    ``review()``) -- it simply returns None so the caller falls through
    to the existing passive ``janitor_blocked`` bookkeeping, unchanged.

    NOTE (unverified -- see ``GitHub.pr_reopen``'s docstring, flagged
    per issue #1274's binding comment item 6): whether reopening a PR
    actually causes GitHub Actions to create a fresh check-suite run for
    the PR's CURRENT head has not been confirmed against a live
    repository; it cannot be verified with a real ``gh`` call inside a
    sandboxed/mocked test environment. The empty-commit fallback exists
    specifically because of that uncertainty.

    Makes GitHub calls (via ``self.gh``) only when eligibility (cap,
    grace wait) has already been decided with no lock held -- mirrors
    every other rerun/retrigger block in this method (flake-aware
    rerun, infra rerun): external I/O never happens while
    ``state_lock`` is held.

    Returns a ``CommandResult`` in two cases: a retrigger attempt was
    actually made (mechanically succeeded, bookkeeping persisted), or
    the attempt cap was already exhausted on entry and this call escalates
    to the W9 operator queue (issue #1274 item 7) instead. Returns None
    in every other case (still inside the post-retrigger grace wait, or a
    mechanical failure that consumed no attempt) so the caller falls
    through unchanged to the pre-existing janitor-gate bookkeeping below.

    Exhaustion -> escalation routing (issue #1274 item 7): once
    ``stale_checks_retrigger_attempts >= stale_checks_max_retriggers``
    AND the caller has already established the check suite is still
    missing this pass (a precondition of even being called -- see
    ``review()``'s ``stale_checks_retrigger_in_scope`` gate), escalate
    via ``_escalate_issue`` (``reason="stale_checks_retrigger_exhausted"``,
    ``reason_class="mechanical"``) followed by
    ``transition(..., "escalated")`` -- the same mechanism every other
    cap-exhaustion escalation in this file relies on to actually land
    ``agent:human-needed`` (mirrors the infra-rerun exhaustion block,
    a few hundred lines up in ``review()``: mechanical cap exceeded, no
    code-fix rework path, straight to a human instead of looping
    forever). Re-firing on a later pass is guarded explicitly below,
    although it is also structurally prevented: once escalated,
    ``review()``'s top-of-function escalated-visibility early return
    (``status == "escalated"``) makes this whole method unreachable on
    every subsequent pass, the same way it already does for the
    retrigger action itself (scope fence item 3/b on issue #1274).

    Issue #1451: the mergeable state of the PR is checked BEFORE any
    remedy is chosen -- single point of enforcement for the
    discrimination. On a CONFLICTING PR, close/reopen (and the
    empty-commit fallback) can never create a CI run: GitHub cannot
    build ``refs/pull/N/merge`` while the branch conflicts, so no
    ``pull_request`` workflow run is created for ANY event (opened,
    reopened, synchronize) until the conflict is resolved. Retriggering
    there burns a ``stale_checks_retrigger_attempts`` slot and a
    close/reopen notification cycle with zero possible effect. Route to
    the existing merge-conflict rework path instead -- the same
    ``_route_janitor_gate_failure_to_rework`` wrapper the janitor gate
    itself uses for ``is_merge_conflict_block`` -- recording a
    ``ci_retrigger_skipped_conflicting`` event (deduped per head) for
    diagnosability. ``UNKNOWN`` (GitHub still computing mergeable)
    defers to the next pass rather than retriggering blind -- a
    close/reopen on a PR whose mergeable state is not yet settled could
    either waste the attempt (if it resolves to CONFLICTING) or fire
    prematurely (if it resolves to MERGEABLE but GitHub has not yet
    created the run). ``MERGEABLE`` and any other value proceed with the
    current behavior below.
    """
    mergeable = str(pr.get("mergeable") or "").upper()
    if mergeable == "CONFLICTING":
        # Dedup per head: a CONFLICTING PR can sit for many passes while
        # the rework worker resolves the conflict; emit the skip event
        # only when the head changes (mirrors the ci_run_never_created
        # per-head dedup pattern) so events.db stays diagnosable without
        # a per-pass spam loop.
        last_skipped_head = existing_pr_state.get("ci_retrigger_skipped_conflicting_head")
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            state["prs"][str(pr_number)] = {
                **state["prs"].get(str(pr_number), {}),
                "ci_retrigger_skipped_conflicting_head": head_sha,
            }
            if last_skipped_head != head_sha:
                state = self._record_event(
                    state,
                    "ci_retrigger_skipped_conflicting",
                    {
                        "pr_number": pr_number,
                        "issue_number": issue_number,
                        "head_sha": head_sha,
                    },
                )
            _wf.save_state(self.paths.state_file, state)
        return self._route_janitor_gate_failure_to_rework(
            pr,
            issue_number,
            attempts_key="conflict_rework_attempts",
            max_attempts=self.config.review.max_conflict_rework_attempts,
            reason="merge_conflict",
            router=self._request_merge_conflict_rework,
        )
    if mergeable == "UNKNOWN":
        # GitHub is still computing mergeable -- defer to the next pass
        # rather than retriggering blind (issue #1451).
        return None

    raw_attempts = existing_pr_state.get("stale_checks_retrigger_attempts", 0)
    attempts = raw_attempts if isinstance(raw_attempts, int) else 0
    max_retriggers = self.config.review.stale_checks_max_retriggers
    if attempts >= max_retriggers:
        return self._escalate_stale_checks_exhaustion(
            pr_number=pr_number,
            issue_number=issue_number,
            head_sha=head_sha,
            attempts=attempts,
            max_retriggers=max_retriggers,
            existing_pr_state=existing_pr_state,
        )

    last_retrigger_at = existing_pr_state.get("stale_checks_last_retrigger_at")
    if last_retrigger_at:
        last_dt = _wf._parse_iso_timestamp(str(last_retrigger_at))
        if last_dt is not None:
            grace_minutes = self.config.review.stale_checks_grace_minutes
            elapsed_minutes = (datetime.now(UTC) - last_dt).total_seconds() / 60.0
            if elapsed_minutes < grace_minutes:
                # Still inside the post-retrigger grace wait: skip this
                # pass silently -- no event, no bookkeeping change
                # (binding comment item 5 on issue #1274).
                return None

    close_result = self.gh.pr_close(pr_number)
    if close_result.ok:
        reopen_result = self.gh.pr_reopen(pr_number)
        mechanically_succeeded = reopen_result.ok
    else:
        mechanically_succeeded = False
    method = "close_reopen"
    if not mechanically_succeeded:
        method = "empty_commit"
        branch = str(pr.get("headRefName") or "")
        empty_commit_result = self.gh.push_empty_commit(branch) if branch else None
        mechanically_succeeded = empty_commit_result is not None and empty_commit_result.ok

    if not mechanically_succeeded:
        # Record it, but do not consume the attempt (same convention as
        # the flake-rerun / infra-rerun API-error branches above): a
        # transient gh error must not burn the bounded retrigger budget.
        return None

    new_attempts = attempts + 1
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        state["prs"][str(pr_number)] = {
            **state["prs"].get(str(pr_number), {}),
            "number": pr_number,
            "issue_number": issue_number,
            "stale_checks_retrigger_attempts": new_attempts,
            "stale_checks_last_retrigger_at": now_iso,
        }
        state = self._record_event(
            state,
            "ci_retriggered_stale_checks",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "head_sha": head_sha,
                "method": method,
                "attempt": new_attempts,
            },
        )
        _wf.save_state(self.paths.state_file, state)
    return _wf.CommandResult(
        False,
        f"CI retrigger attempted for PR #{pr_number} "
        f"(attempt {new_attempts}/{max_retriggers}, method={method})",
        {
            "pr": pr_number,
            "issue": issue_number,
            "stale_checks_retriggered": True,
            "stale_checks_retrigger_method": method,
            "stale_checks_retrigger_attempts": new_attempts,
        },
    )


def _escalate_stale_checks_exhaustion(
    self,
    *,
    pr_number: int,
    issue_number: int,
    head_sha: str,
    attempts: int,
    max_retriggers: int,
    existing_pr_state: dict[str, Any],
) -> _wf.CommandResult | None:
    """Exhaustion -> escalation routing for the stale-checks retrigger
    lane (issue #1274 item 7, the W9 operator queue).

    Called only from ``_attempt_stale_checks_retrigger`` once
    ``attempts >= max_retriggers`` on entry, with the check suite
    confirmed still missing this pass by the caller's own
    ``stale_checks_retrigger_in_scope`` precondition. Escalates via
    ``_escalate_issue`` + ``transition(..., "escalated")`` -- the same
    pair every other ``reason_class="mechanical"`` cap-exhaustion
    escalation in this file (``dead_dispatched_worker_reap``,
    ``orphan_sweep_redispatch_cap_exceeded``, ``redispatch_cap_exceeded``,
    ``infra_rerun_cap_exceeded``) uses to actually land
    ``agent:human-needed`` -- never a second queue or a hardcoded label.

    Dedup guard: skips re-escalating once this PR/issue already carries
    ``escalation_reason == "stale_checks_retrigger_exhausted"``. This
    mirrors ``_route_janitor_gate_failure_to_rework``'s
    ``current_escalation_reasons`` dedup convention, but is
    belt-and-suspenders here rather than load-bearing: once escalated,
    ``status`` becomes ``"escalated"``, and ``review()``'s top-of-function
    escalated-visibility early return (``_escalation_flags``, matched on
    status alone) makes this entire method structurally unreachable on
    every later pass -- the same structural guarantee that already
    excludes the retrigger action itself from that branch (scope fence
    item 3/b). The explicit check here only matters if something resets
    ``status`` while ``escalation_reason`` survives; it is kept for
    readability parity with the rest of this file and because it is
    directly testable independent of that structural argument.

    Issue #1461: the check uses ``escalation_reasons_seen`` (the
    append-only list maintained by ``_escalate_issue``) instead of the
    single ``escalation_reason`` field, so a cross-lane clobber that
    overwrites the single field does not blind this guard.

    Returns None (never re-escalates, never emits a duplicate event) when
    the dedup guard trips; otherwise always returns a ``CommandResult``
    (``ok=False``) describing the escalation, mirroring the infra-rerun
    exhaustion block's return shape.
    """
    exhaustion_reason = "stale_checks_retrigger_exhausted"
    if exhaustion_reason in frozenset(existing_pr_state.get("escalation_reasons_seen") or []):
        return None

    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        state = _wf._escalate_issue(
            state,
            issue_number,
            reason=exhaustion_reason,
            reason_class="mechanical",
            pr_number=pr_number,
            pr_extra={"stale_checks_retrigger_attempts": attempts},
        )
        state = self._record_event(
            state,
            "stale_checks_retrigger_exhausted",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "head_sha": head_sha,
                "attempts": attempts,
                "max_retriggers": max_retriggers,
            },
        )
        _wf.save_state(self.paths.state_file, state)

    edge = _wf._escalation_edge("escalated", "mechanical")
    result = _wf.transition(self.gh, self.config.labels, issue_number, edge)
    label_error = None
    if result.outcome != TransitionOutcome.APPLIED:
        label_error = {
            "edge": edge,
            "outcome": result.outcome.value,
            "add_failures": result.add_failures,
            "remove_failures": result.remove_failures,
        }
    return _wf.CommandResult(
        False,
        f"PR #{pr_number} exhausted stale-checks retrigger cap "
        f"({attempts}/{max_retriggers}); escalated to human",
        {
            "pr": pr_number,
            "issue": issue_number,
            "stale_checks_retrigger_exhausted": True,
            "label_error": label_error,
        },
    )


def _check_janitor_rework_stall(
    self,
    *,
    pr_number: int,
    issue_number: int,
    issue_status: str | None,
    head_sha: str,
    reason: str,
    attempts_key: str,
    stall_since_key: str,
    stall_head_key: str,
    stall_since: Any,
    stall_head: Any,
) -> _wf.CommandResult | None:
    """Stall bound orthogonal to the settled-head signal (issue #765).

    Called only from ``_route_janitor_gate_failure_to_rework``'s
    not-settled branch, where a rework is pending but has not produced
    that function's own progress signal. That combination is ambiguous
    on its own -- it covers both a live ``dispatched``/``dispatch_
    pending`` session still doing real work (must NOT escalate here; a
    genuinely dead session is WatchdogConfig.stall_minutes's job, not
    this function's) and a PR that is queued with nobody working it at
    all (``rework_requested``, no progress). The latter can never reach
    the cap/rescue check in the caller, because that check only runs
    once a cycle SETTLES, and a stalled PR's cycle never settles -- so
    without this, it waits here forever (issue #765: PR #696, 55
    ``rework_already_pushed`` events, attempts already at cap, rescue
    never attempted).

    HOLD vs. RESET is deliberately split on two different signals:

    - Only a HEAD CHANGE since the clock started clears it. This is
      checked regardless of ``issue_status`` -- a live ``dispatched``
      session pushing WIP commits is real progress and must reset the
      clock even though it isn't the "stalled" status itself, and (the
      motivating case, found while building this fix) reconcile's
      ``issue_active_label_with_open_pr`` self-heal flips ``issue_status``
      away from ``rework_requested`` on its own periodic cadence
      *without touching the branch* -- an earlier draft keyed the clear
      on status alone, which reconcile would have reset roughly every
      reconcile pass, making the bound practically unreachable whenever
      reconcile is enabled (the common case). Head-only clearing makes
      the clock immune to that oscillation.
    - Only ``issue_status == "rework_requested"`` ADVANCES the clock
      (starts it, or lets elapsed time accrue toward the threshold). Any
      other pending status (``dispatched``, ``dispatch_pending``, or a
      transient reconcile-driven flip with the head unchanged) HOLDS --
      returns ``None`` without escalating and without touching the
      clock, so already-accumulated stall time survives status churn
      that isn't accompanied by real progress.

    Returns ``None`` while waiting/holding, or after clearing a clock
    whose head moved. Returns a terminal ``CommandResult`` --
    ``agent:human-needed`` applied via the same ``transition()`` helper
    the cap-exceeded path uses -- once the threshold is exceeded.
    """
    progressed = (
        stall_since is not None
        and stall_head is not None
        and bool(head_sha)
        and head_sha != stall_head
    )
    if progressed:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            state["prs"][str(pr_number)] = {
                **state["prs"].get(str(pr_number), {}),
                "number": pr_number,
                "issue_number": issue_number,
                stall_since_key: None,
                stall_head_key: None,
            }
            _wf.save_state(self.paths.state_file, state)
        stall_since = None
        stall_head = None

    stalled_candidate = issue_status == "rework_requested"
    if not stalled_candidate:
        # HOLD: do not clear. Only a head change (above) clears the
        # clock; a live worker or a transient status flip with the head
        # unchanged must not lose accumulated stall time.
        return None

    started = _wf._parse_iso_timestamp(stall_since) if stall_since is not None else None
    if started is None or stall_head is None:
        # (Re)start the clock. Covers: no clock running yet; an
        # unparseable/corrupt timestamp; and a clock inherited from
        # before this bound had a head anchor at all (stall_since set,
        # stall_head missing -- pre-redesign state, or any other write
        # path that recorded one key without the other). In that last
        # case we cannot tell whether the head moved during the
        # already-elapsed time, so we discard it and re-anchor rather
        # than trust elapsed time of unknown provenance -- the safe
        # direction on rollout is one extra window before the bound can
        # fire, never a spurious escalation from state that never
        # actually observed a stalled head.
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            state["prs"][str(pr_number)] = {
                **state["prs"].get(str(pr_number), {}),
                "number": pr_number,
                "issue_number": issue_number,
                stall_since_key: _wf.utc_now(),
                stall_head_key: head_sha or None,
            }
            _wf.save_state(self.paths.state_file, state)
        return None

    threshold_minutes = self.config.review.rework_stall_minutes
    if threshold_minutes <= 0:
        return None
    elapsed_minutes = (datetime.now(UTC) - started).total_seconds() / 60
    if elapsed_minutes < threshold_minutes:
        return None

    # Issue #776 follow-up: record a lane-scoped escalation_reason here
    # too (parallel to the cap-exceeded branch above), not just a status
    # flip. Without it, _route_janitor_gate_failure_to_rework's same-lane
    # guard has nothing to match on the next pass -- the issue's status
    # is "escalated" (so rework_pending is False, skipping straight past
    # the not-settled branch) and the wrapper would silently redispatch a
    # fresh rework attempt (or re-escalate) immediately, undoing the stall
    # escalation's entire purpose the moment it fires.
    escalation_reason = f"{attempts_key}_stall_exceeded"

    # Surface ``rework_issue_fetch_skipped`` events from this stall window
    # in the escalation payload (issue #970). The stall escalation exists
    # to answer "why was this issue never dispatched for rework?", and
    # ``rework_issue_fetch_skipped`` (issue #939) is the event that records
    # exactly that -- but until this change the escalation's payload did
    # not correlate with it, so the signal lived in events.db and was
    # absent from the one report built to explain the very condition it
    # describes. Mirrors #940's wiring of ``unauthorized_merge_check_skipped``
    # into ``tripwire_status`` as ``last_skipped_reason``.
    #
    # The window bound is ``stall_since`` -- the stall clock start -- not a
    # fixed lookback, for the same reason #940 bounds by ``armed_at``: a
    # skip recorded before the stall began says nothing about why *this*
    # stall never progressed. Reading events.db here does not compromise
    # the state-lock section below -- it is local SQLite, independent of
    # state.json, and ``query_events`` takes no state lock.
    #
    # Scope to the escalating ``issue_number``. ``rework_issue_fetch_skipped``
    # is one event per dispatch pass and bundles every issue that failed to
    # fetch in that pass into one payload's ``issue_numbers`` list
    # (``_build_rework_issue_fetch_skip_payload`` collects all
    # ``failed_issue_fetches``). An unscoped ``kind``+``since`` query would
    # attribute a *different* PR's/issue's fetch failure to this PR's stall
    # escalation. The indexed ``issue_number`` column cannot be used for the
    # filter either: ``_extract_payload_refs`` backfills it with only the
    # *first* entry of ``issue_numbers``, so an event whose list contains
    # this issue but not as the first element would be silently missed.
    # Filter in Python on the full ``issue_numbers`` list instead.
    raw_skips = _wf.query_events(
        self.paths.state_file,
        kind="rework_issue_fetch_skipped",
        since=stall_since,
    )
    stall_skips = [
        e
        for e in raw_skips
        if isinstance(e, dict)
        and isinstance(e.get("payload"), dict)
        and issue_number in (e["payload"].get("issue_numbers") or [])
    ]
    last_skip = stall_skips[-1] if stall_skips else None
    last_skip_payload = last_skip.get("payload") if isinstance(last_skip, dict) else None
    last_skip_payload_dict = last_skip_payload if isinstance(last_skip_payload, dict) else {}
    last_skip_reason = last_skip_payload_dict.get("reason") or None
    last_skip_issue_numbers = last_skip_payload_dict.get("issue_numbers")
    last_skip_reasons = last_skip_payload_dict.get("reasons")
    last_skip_at = last_skip["ts"] if isinstance(last_skip, dict) else None
    stall_skip_summary = {
        "rework_fetch_skips": len(stall_skips),
        "last_rework_fetch_skip_at": last_skip_at,
        "last_rework_fetch_skip_reason": last_skip_reason,
        "last_rework_fetch_skip_issue_numbers": last_skip_issue_numbers,
        "last_rework_fetch_skip_reasons": last_skip_reasons,
    }

    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        attempts_so_far = int(state["prs"].get(str(pr_number), {}).get(attempts_key, 0))
        state = _wf._escalate_issue(
            state,
            issue_number,
            reason=escalation_reason,
            reason_class="mechanical",
            pr_number=pr_number,
            pr_extra={
                stall_since_key: None,
                stall_head_key: None,
            },
        )
        state = self._record_event(
            state,
            "janitor_rework_stalled",
            {
                "pr_number": pr_number,
                "issue_number": issue_number,
                "reason": reason,
                "escalation_reason": escalation_reason,
                "attempts": attempts_so_far,
                "stalled_minutes": round(elapsed_minutes, 1),
                "stall_since": stall_since,
                "head_sha": head_sha,
                **stall_skip_summary,
            },
        )
        _wf.save_state(self.paths.state_file, state)
    edge = _wf._escalation_edge("escalated", "mechanical")
    result = _wf.transition(self.gh, self.config.labels, issue_number, edge)
    label_error = None
    if result.outcome != TransitionOutcome.APPLIED:
        label_error = {
            "edge": edge,
            "outcome": result.outcome.value,
            "add_failures": result.add_failures,
            "remove_failures": result.remove_failures,
        }
    message = (
        f"PR #{pr_number} janitor {reason} rework stalled "
        f"({round(elapsed_minutes)}m with no progress while {issue_status}); escalated"
    )
    # Appended, not substituted, so the base escalation text stays true --
    # the whole problem is that it reads as a self-contained explanation
    # on its own while a fetch skip is the actual root cause. Conditioned
    # on ``stall_skips`` so a stall with no fetch skips is not decorated
    # with a vacuous "0 passes" clause.
    if stall_skips:
        issue_list = last_skip_issue_numbers
        issue_clause = (
            f" issue(s) {issue_list}" if isinstance(issue_list, list) and issue_list else ""
        )
        message += (
            f" (warning: {len(stall_skips)} rework pass(es) since {stall_since}"
            f" could not fetch{issue_clause}; most recent {last_skip_at}"
            f", reason: {last_skip_reason})"
        )
    return _wf.CommandResult(
        False,
        message,
        {
            "pr": pr_number,
            "issue": issue_number,
            "janitor_ok": False,
            "escalated": True,
            "escalation_reason": "stalled",
            "label_error": label_error,
            **stall_skip_summary,
        },
    )
