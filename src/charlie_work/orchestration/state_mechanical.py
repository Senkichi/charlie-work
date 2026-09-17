"""Mechanical-issue de-escalation delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L01 batch 2 (issue #1645, parent #1632, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from charlie_work.labels import TransitionOutcome
from charlie_work.state import PASSIVE_OPEN_STATUS, arm_deescalation_pass, is_deescalation_due
from charlie_work.worktree import WORKTREE_UNSAFE_KINDS
import charlie_work.workflow as _wf


def _deescalate_mechanical_issue(self, issue_number: int) -> dict[str, Any]:
    """Re-evaluate one ``mechanical`` escalation and clear it if safe.

    Called only from ``_maybe_deescalate_mechanical`` for issues whose
    snapshot already matched the selection query (``status in
    ("escalated", "blocked")`` and ``reason_class == "mechanical"``).
    Every check here is re-run against FRESH state/GitHub data -- the
    snapshot may be stale by the time this runs (a concurrent
    ``charlie unescalate``, a re-escalation with a new reason, or a
    parallel loop lane already touched the entry).

    Always returns a dict, never ``None``. Returns
    ``{"skipped": "<reason>", "issue_number": ...}`` when the issue was
    left escalated -- the entry no longer qualifies, a worker is still
    live, no open PR is bound to the issue, or the fresh
    mergeable/janitor check failed (this is the expected, common
    outcome; the reason string is what the caller histograms into
    ``deescalation_pass_completed``, see ``_deescalation_skip``). Returns
    ``{"cap_exhausted": True, ...}`` the first time
    ``auto_deescalation_count`` has already reached
    ``config.deescalation.max_auto_deescalations``. Returns
    ``{"cleared": True, ...}`` after a successful auto-de-escalation.

    Issue #783 hazard (a) -- oscillation guard: ``auto_deescalation_count``
    is incremented here on every clear and is NEVER reset by this
    method -- only a human-invoked ``charlie unescalate`` resets it (see
    ``_UNESCALATE_ISSUE_RESET_FIELDS``). Once the count reaches
    ``max_auto_deescalations`` (default 2), this method permanently
    stops clearing that issue and instead emits
    ``deescalation_cap_exhausted`` exactly once (guarded by the
    ``deescalation_cap_notified_at`` marker, also reset only by
    ``unescalate()``). The issue stays on ``agent:human-needed`` --  this
    is a diagnosable terminal state (the event names the exhausted
    count), not a silent re-creation of the one-way door under a new
    name: a human is still required to re-arm it, exactly once more
    than before.

    Issue #783 hazard (b) -- unbounded paid-session loop: this method
    resets ONLY the per-mechanism attempt/cap counter that gates the
    CLEARED ``escalation_reason`` (see
    ``_REWORK_BUDGET_RESET_BY_ESCALATION_REASON``), and only once per
    escalation episode (tracked via ``rework_budget_reset_for_terminal_since``).
    It does NOT reset the other lanes' counters, nor the cross-lane
    bookkeeping that a full ``charlie unescalate`` clears
    (``redispatch_at``, ``review_dispatch_attempt_count``, etc.). If the
    same mechanical condition recurs after a clear, the lane's counter
    has been zeroed so the cap re-trips only after a fresh
    ``max_attempts``/``max_rework_cycles`` worth of completed-but-still-
    failing cycles, which re-escalates the issue through one of the
    S1-S14 call sites and re-enters this same accounting. That recurrence
    also consumes one more slot of ``auto_deescalation_count``, so the
    two counters compound: the mechanism-specific cap bounds how fast a
    single recurring failure can re-escalate, and
    ``auto_deescalation_count``'s cap independently bounds how many times
    this sweep will ever clear the SAME issue. The per-episode counter
    reset cannot unbound the loop because ``auto_deescalation_count``
    still caps total clears -- the sweep cannot re-dispatch a paid worker
    session more than ``max_auto_deescalations`` times for one recurring
    failure before falling permanently back to human review.

    A pre-PR dispatch failure (e.g. ``dispatch_failed_cap_exceeded`` --
    the launch never produced a PR at all) has no artifact AC3's
    "mergeable AND janitor_ok" can be checked against, so it is left
    untouched here (skip reason ``no_open_pr``) rather than guessed at; a
    human still recovers it via ``charlie unescalate --issue``.
    """
    # Issue #1327: under dry-run the paired GitHub label transition is
    # already gated at the sink, so the state.json write that records the
    # clear must be suppressed in lockstep -- otherwise a single dry-run
    # pass leaves state.json reporting the issue de-escalated while GitHub
    # still shows the ``escalated`` label. Returning a skip (rather than
    # proceeding and letting ``self.write_gate`` suppress each write)
    # also avoids pointless GitHub reads and a misleading ``cleared``
    # return value that would diverge from the (unwritten) state. The
    # write-gate routing below is the structural backstop for any future
    # caller that bypasses this guard.
    if self.dry_run:
        return _wf._deescalation_skip("dry_run", issue_number)
    state = _wf.load_state_locked(self.paths.state_file)
    issue_key = str(issue_number)
    issue_entry = state.get("issues", {}).get(issue_key, {})
    if not isinstance(issue_entry, dict):
        return _wf._deescalation_skip("invalid_issue_entry", issue_number)
    if issue_entry.get("status") not in ("escalated", "blocked"):
        # already resolved by something else since the snapshot
        return _wf._deescalation_skip("not_escalated", issue_number)
    if issue_entry.get("reason_class") != "mechanical":
        # fail closed: judgment (or re-classified) since the snapshot
        return _wf._deescalation_skip("not_mechanical", issue_number)

    max_auto = self.config.deescalation.max_auto_deescalations
    auto_count = int(issue_entry.get("auto_deescalation_count", 0) or 0)
    if auto_count >= max_auto:
        if issue_entry.get("deescalation_cap_notified_at"):
            # already reported once; do not re-notify every pass
            return _wf._deescalation_skip("cap_already_notified", issue_number)
        with _wf.state_lock(self.paths.state_file):
            fresh_state = _wf.load_state(self.paths.state_file)
            fresh_entry = fresh_state["issues"].get(issue_key, {})
            if not isinstance(fresh_entry, dict) or fresh_entry.get(
                "deescalation_cap_notified_at"
            ):
                return _wf._deescalation_skip("cap_notify_raced", issue_number)
            fresh_state["issues"][issue_key] = {
                **fresh_entry,
                "number": issue_number,
                "deescalation_cap_notified_at": _wf.utc_now(),
            }
            fresh_state = self.write_gate.record_event(
                fresh_state,
                "deescalation_cap_exhausted",
                {
                    "issue_number": issue_number,
                    "auto_deescalation_count": auto_count,
                    "max_auto_deescalations": max_auto,
                },
            )
            self.write_gate.save_state(fresh_state)
        # Issue #1266: the auto-clearing sweep has given up on this
        # mechanical escalation -- move it off operator_queue onto
        # human_needed so a human actually sees it (operator_queue is
        # meant to be a self-clearing holding area, not a place a
        # capped-out issue silently stays forever). The "escalated" edge
        # already does this unconditionally: it adds human_needed and
        # strips every other workflow_labels member, operator_queue
        # included. Best-effort outside the lock, matching every other
        # label transition here -- the label-repair self-heal sweep
        # (_repair_escalated_labels) backstops a failed write, and it
        # correctly re-targets human_needed once reason_class stays
        # "mechanical" but this cap-exhaustion path has already run
        # (repair reads state fresh, not this function's return value).
        self.write_gate.transition(self.gh, self.config.labels, issue_number, "escalated")
        return {"cap_exhausted": True, "issue_number": issue_number}

    pr_number: int | None = None
    for key, entry in state.get("prs", {}).items():
        if (
            isinstance(entry, dict)
            and entry.get("issue_number") == issue_number
            and entry.get("status") not in ("merged", "closed")
            and key.isdigit()
        ):
            pr_number = int(key)
            break
    if pr_number is None:
        return _wf._deescalation_skip("no_open_pr", issue_number)

    from charlie_work.worker import issue_worker_liveness

    liveness = issue_worker_liveness(
        issue_number, issue_entry, self._layout.sessions_dir, self.config, datetime.now(UTC)
    )
    if liveness.live:
        # a live worker is using this issue; not stuck
        return _wf._deescalation_skip("worker_live", issue_number)

    # Issue #849: a ``worktree_unsafe`` escalation is caused by bytes on
    # disk. Clearing it based on PR-level health (OPEN, not CONFLICTING,
    # janitor_ok) without inspecting the worktree reports success for an
    # operation that changed nothing causal — the next rework dispatch
    # reproduces the escalation deterministically. Re-run the safety
    # check and skip clearing while the worktree still fails it.
    # Issue #807: ``worktree_unsafe`` is split into
    # ``worktree_unsafe_shim_dirt`` and ``worktree_unsafe_local_commits``;
    # both are covered by ``WORKTREE_UNSAFE_KINDS`` so the safety re-check
    # fires for either kind.
    if issue_entry.get("escalation_reason") in WORKTREE_UNSAFE_KINDS:
        unsafe_reason = self._worktree_still_unsafe(issue_number, state)
        if unsafe_reason:
            return _wf._deescalation_skip("worktree_still_unsafe", issue_number)

    pr = self.gh.pr_view(pr_number)
    pr_state_str = str(pr.get("state") or "").upper()
    # Mirror janitor._check_mergeable's own permissive definition of
    # "mergeable" exactly: it fails a PR only on the literal
    # "CONFLICTING" value and treats "UNKNOWN" (the common transient
    # value in the minutes after any push, before GitHub finishes
    # computing mergeability) or a missing field as passing. Requiring
    # an exact "MERGEABLE" match here would be a *stricter*, competing
    # definition of mergeable than the one the janitor gate itself uses
    # -- and would make this sweep silently inert on exactly the
    # freshly-pushed green PRs (#690/#699/#759) it exists to unblock.
    mergeable = str(pr.get("mergeable") or "").upper()
    checks = self.gh.pr_checks(pr_number)
    diff = self.gh.pr_diff(pr_number)
    pr_entry_for_janitor = state.get("prs", {}).get(str(pr_number))
    janitor_verdict = _wf.run_janitor(
        pr,
        checks,
        self.config,
        pr_state=pr_entry_for_janitor if isinstance(pr_entry_for_janitor, dict) else None,
        repo_root=self.repo_root,
        pr_diff=diff,
        review_decision=self._review_decision(pr_number),
    )

    with _wf.state_lock(self.paths.state_file):
        fresh_state = _wf.load_state(self.paths.state_file)
        fresh_pr_entry = fresh_state["prs"].get(str(pr_number))
        if isinstance(fresh_pr_entry, dict):
            # Visibility update mirrors review()'s escalated-PR early
            # return (workflow.py): the cached janitor fields stay fresh
            # even on a pass that does not end up clearing anything.
            fresh_state["prs"][str(pr_number)] = {
                **fresh_pr_entry,
                "janitor_ok": janitor_verdict.ok,
                "janitor_failures": list(janitor_verdict.failures),
                "is_missing_checks_only_block": janitor_verdict.is_missing_checks_only_block,
            }
        fresh_issue_entry = fresh_state["issues"].get(issue_key)
        if not isinstance(fresh_issue_entry, dict) or (
            fresh_issue_entry.get("status") not in ("escalated", "blocked")
            or fresh_issue_entry.get("reason_class") != "mechanical"
        ):
            # Changed concurrently since the snapshot (human unescalate,
            # re-escalation with a different reason_class, etc.) -- do
            # not act on stale intent.
            self.write_gate.save_state(fresh_state)
            return _wf._deescalation_skip("changed_concurrently", issue_number)

        # Split from a single composite condition purely so each blocking
        # reason is attributable in the histogram -- the three checks below
        # are the same test, in the same order, with the same outcome. The
        # only ``skipped`` reason that means "the sweep works but the PR is
        # not ready" is ``janitor_blocked``; ``pr_not_open`` and
        # ``pr_conflicting`` mean the escalation should never have been a
        # sweep candidate at all.
        if pr_state_str != "OPEN":
            self.write_gate.save_state(fresh_state)
            return _wf._deescalation_skip("pr_not_open", issue_number)
        if mergeable == "CONFLICTING":
            self.write_gate.save_state(fresh_state)
            return _wf._deescalation_skip("pr_conflicting", issue_number)
        if not janitor_verdict.ok:
            self.write_gate.save_state(fresh_state)
            return _wf._deescalation_skip("janitor_blocked", issue_number)

        cleared_condition = fresh_issue_entry.get("escalation_reason")
        cleared_auto_count = auto_count + 1
        # Issue #1093: reset the over-cap rework counter once per
        # escalation episode so the issue gets a fresh rework budget on
        # the first clear after an escalation.  ``terminal_since`` is
        # refreshed on every ``_escalate_issue`` call, so it identifies
        # the current episode; the marker records which episode was
        # last reset.  A re-escalation produces a new ``terminal_since``,
        # so the next clear resets again.  Repeated clears within the
        # same episode (same ``terminal_since``) do not re-reset.
        current_terminal_since = fresh_issue_entry.get("terminal_since")
        budget_reset_needed = (
            "rework_budget_reset_for_terminal_since" not in fresh_issue_entry
            or fresh_issue_entry.get("rework_budget_reset_for_terminal_since")
            != current_terminal_since
        )
        updated_issue_entry = {
            **fresh_issue_entry,
            "number": issue_number,
            "status": PASSIVE_OPEN_STATUS,
            "auto_deescalation_count": cleared_auto_count,
        }
        if budget_reset_needed:
            updated_issue_entry["rework_budget_reset_for_terminal_since"] = current_terminal_since
        _wf.clear_escalation(updated_issue_entry)
        updated_issue_entry.pop("label_error", None)
        fresh_state["issues"][issue_key] = updated_issue_entry
        # Issue #1093: mirror-clear the PR record's escalation fields so
        # the rework router's short-circuit on
        # ``existing_pr_state.get("escalation_reasons_seen")`` (issue
        # #1461: was ``escalation_reason``) no longer fires after the
        # sweep clears the issue.  Also reset the per-mechanism
        # rework counter that ACTUALLY gates the cleared escalation_reason
        # on the open PR once per escalation episode -- resetting only
        # ``request_changes_count`` left the no_op/conflict lanes' real
        # gating counter untouched, so the router re-escalated on the next
        # detection.  See ``_REWORK_BUDGET_RESET_BY_ESCALATION_REASON``.
        _wf.clear_escalation_on_issue_prs(fresh_state, issue_number)
        _wf._reset_linked_pr_status_to_passive_open(fresh_state, pr_number)
        if budget_reset_needed:
            fresh_pr = fresh_state["prs"].get(str(pr_number))
            if isinstance(fresh_pr, dict):
                reset_spec = self._REWORK_BUDGET_RESET_BY_ESCALATION_REASON.get(cleared_condition)
                if reset_spec is not None:
                    counter_field, companion_fields = reset_spec
                    fresh_pr[counter_field] = 0
                    for _field in companion_fields:
                        fresh_pr.pop(_field, None)
        fresh_state = self.write_gate.record_event(
            fresh_state,
            "deescalation_cleared",
            {
                "issue_number": issue_number,
                "pr_number": pr_number,
                "reason_class": "mechanical",
                "cleared_condition": cleared_condition,
                "pr_mergeable": mergeable,
                "janitor_ok": janitor_verdict.ok,
                "auto_deescalation_count": cleared_auto_count,
                "rework_budget_reset": budget_reset_needed,
            },
        )
        self.write_gate.save_state(fresh_state)

    result = self.write_gate.transition(
        self.gh, self.config.labels, issue_number, "unescalated_pr_open"
    )
    if result.outcome != TransitionOutcome.APPLIED:
        with _wf.state_lock(self.paths.state_file):
            fresh_state = _wf.load_state(self.paths.state_file)
            entry = fresh_state["issues"].get(issue_key, {})
            fresh_state["issues"][issue_key] = {
                **(entry if isinstance(entry, dict) else {}),
                "number": issue_number,
                "label_error": {
                    "edge": "unescalated_pr_open",
                    "outcome": result.outcome.value,
                    "add_failures": result.add_failures,
                    "remove_failures": result.remove_failures,
                },
            }
            self.write_gate.save_state(fresh_state)

    return {
        "cleared": True,
        "issue_number": issue_number,
        "pr_number": pr_number,
        "cleared_condition": cleared_condition,
    }


def _maybe_deescalate_mechanical(self) -> None:
    """Periodic sweep that re-evaluates ``mechanical`` escalations (issue #783).

    Escalation to ``agent:human-needed`` was a one-way door: four
    ``labels.py`` edges (``escalated``, ``blocked``, ``redispatch_escalated``,
    ``merged_pr_mention_flagged``) add the label; nothing in the automated
    loop ever removed it. All recovery went through the operator-invoked
    ``charlie unescalate`` command -- including for PRs whose underlying
    artifact was already fine (pushed, open, CI green, ``janitor_ok``) and
    whose escalation was purely a process failure (e.g. a rework worker's
    session dying -- ``session_failed_escalated``), not a human decision.

    This method is the automated re-entry point for exactly that
    process-failure class. It is scoped narrowly by construction:

    - Issues whose entry carries ``reason_class == "mechanical"`` --
      written atomically alongside every ``status -> escalated/blocked``
      transition at its call site (S1-S14 in the issue #783 implementation)
      -- are considered directly. ``judgment`` escalations stay terminal.
      A PRE-EXISTING escalation with no recorded ``reason_class`` (every
      escalation before issue #783 shipped) is first backfilled from its
      most recent escalation-transition event in ``events.db`` (issue #797):
      only kinds that unambiguously denote a process failure become
      ``mechanical``; ambiguous or unknown kinds, and issues with no event,
      stay fail-closed and invisible to the selection query.
    - Clearing additionally requires a live, freshly-fetched PR that is
      still OPEN, does not report ``mergeable == "CONFLICTING"`` (mirrors
      ``janitor._check_mergeable``'s own permissive definition exactly --
      a transient ``"UNKNOWN"`` value is not a conflict), and passes a
      freshly-computed ``run_janitor().ok`` -- reason-clearing alone
      (e.g. simply having a ``reason_class``) is never sufficient. See
      ``_deescalate_mechanical_issue`` for the full per-issue algorithm
      and this issue's two required hazard analyses (oscillation guard /
      unbounded-loop guard), both enforced there via
      ``auto_deescalation_count`` and by never resetting the original
      per-mechanism attempt counters.

    Two-phase lock pattern, mirroring ``_maybe_reconcile_drift``:
      1. Under our own (short) lock: decide whether this sweep is due at
         all, and take a snapshot of candidate issue numbers. If not
         due, return without calling out to GitHub.
      2. Outside any lock held by this method: process each candidate
         (``_deescalate_mechanical_issue`` acquires its own locks as
         needed, mirroring ``unescalate()``'s snapshot-then-reapply
         pattern so a slow ``gh`` call never holds the file lock).
      3. Under our own (short) lock again: persist the next-due
         timestamp and emit exactly one summary event for this pass.

    Wrapped in exception containment per-issue (one issue's failure must
    not sink the whole pass) AND around the snapshot/scheduling itself
    (mirroring ``_maybe_reconcile_drift``'s docstring: ``supervise.py``'s
    daemon loop has no outer per-pass exception boundary of its own, so
    an uncaught exception here would kill the daemon, not just this
    pass).
    """
    if not self.config.deescalation.enabled:
        return

    state_file = self.paths.state_file
    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        if not is_deescalation_due(state):
            return
        state = self._backfill_missing_reason_classes(state)
        self.write_gate.save_state(state)
        candidates = sorted(
            int(num)
            for num, entry in state.get("issues", {}).items()
            if isinstance(entry, dict)
            and entry.get("status") in ("escalated", "blocked")
            and entry.get("reason_class") == "mechanical"
            and str(num).isdigit()
        )

    next_deescalation_at = (
        (datetime.now(UTC) + timedelta(minutes=self.config.deescalation.interval_minutes))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    cleared: list[dict[str, Any]] = []
    cap_exhausted: list[int] = []
    errors: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for issue_number in candidates:
        try:
            outcome = self._deescalate_mechanical_issue(issue_number)
        except Exception as exc:  # noqa: BLE001 - containment is deliberate; see docstring
            errors.append({"issue_number": issue_number, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if outcome.get("cap_exhausted"):
            cap_exhausted.append(issue_number)
        elif outcome.get("cleared"):
            cleared.append(outcome)
        else:
            # ``unknown`` is unreachable via ``_deescalation_skip`` (which
            # always sets a reason); it exists so a future outcome shape
            # that is neither cleared nor cap-exhausted still lands in the
            # histogram as a countable bucket instead of vanishing.
            skipped[str(outcome.get("skipped") or "unknown")] += 1

    with _wf.state_lock(state_file):
        state = _wf.load_state(state_file)
        state = arm_deescalation_pass(state, next_deescalation_at)
        state = self.write_gate.record_event(
            state,
            "deescalation_pass_completed",
            {
                "candidates": len(candidates),
                "cleared": cleared,
                "cap_exhausted": cap_exhausted,
                # Attribution for the difference between ``candidates`` and
                # everything else in this payload. Before issue #1090 that
                # difference was silent: a sweep that considered 29 issues
                # and cleared none emitted the same event as a sweep with
                # no candidates at all, so "the sweep is inert" and "the
                # sweep ran and every PR was legitimately blocked" were
                # indistinguishable from events.db alone. Sorted for a
                # stable diff across passes.
                "skipped": dict(sorted(skipped.items())),
                "errors": errors,
            },
        )
        self.write_gate.save_state(state)
