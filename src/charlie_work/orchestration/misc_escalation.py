"""Escalated-label repair delegate moved out of ``OrchestratorApp``.

Track 2 Phase B, L03 batch 3 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Body relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches the top-level ``def``
unwrapped onto ``OrchestratorApp``.

Two Tier D names -- ``transition`` and ``_collect_escalated_label_subjects`` --
are patched on the ``charlie_work.workflow`` namespace by the suite, so the moved
body reaches them through ``_wf.``. Every other free name is imported directly.

Behavior delta: ``logging.getLogger(__name__)`` now names the logger
``charlie_work.orchestration.misc_escalation`` instead of ``charlie_work.workflow``
(``__name__`` is a module global that moves with the body). No test asserts on
that logger name.
"""

from __future__ import annotations

import charlie_work.workflow as _wf
import logging
from typing import Any

from charlie_work.escalation import (
    _escalated_label_needs_repair,
    _escalation_edge,
    _escalation_label,
    _repair_reason_class,
)
from charlie_work.github import label_names
from charlie_work.labels import TransitionOutcome


def _repair_escalated_labels(self) -> dict[str, Any]:
    """Re-apply the ``agent:human-needed`` edge for escalated issues that lack it.

    Issue #586 established this self-heal: the edge is applied once at
    escalation time, and if that ``transition()`` failed -- or the issue was
    escalated by a path predating the edge -- the issue sits ``escalated`` in
    ``state.json`` and invisible on GitHub, permanently excluded from dispatch
    with no human-visible signal that operator action is required.

    Issue #1088 is that the sweep is unreachable whenever review dispatch is
    off. Its subject set was built inside ``dispatch_reviews``'
    candidate-filter loop, below the ``review_dispatch.enabled`` early
    return, and both deployed fleets run that flag false -- so the set is
    empty and the loop a no-op for as long as the flag stays off, which at
    the time of the fix was ~8 days and counting. It is not dead code in the
    absolute sense: it fired once (a sibling repo, 2026-07-28) and repaired 10
    issues, which is exactly why the inertness matters.
    Deriving the subjects from ``state`` (see
    ``_collect_escalated_label_subjects``) is what makes the guarantee real,
    and this method is called *above* that early return.

    Two properties worth preserving:

    - **Steady state costs nothing.** Once an edge verifies, ``label_error``
      is ``None`` and ``_escalated_label_needs_repair`` answers with a dict
      lookup and no GitHub call. Without that, the sweep would re-apply every
      label on every pass forever.
    - **The per-pass cap is mandatory, not defensive.** Every subject in the
      absent-key arm costs a live ``issue_view``, and when this was written
      *every* escalated subject was in that arm (8 in charlie-work, 49 in
      the other repo). Sweeping all 57 in one pass would add that many sequential
      ``gh`` calls to a loop shared sequentially between both repos, which is
      #1078's starvation mechanism. The cap converges over a few passes
      instead; subjects are visited in issue order so progress is monotonic
      rather than re-rolling the same head of the list.
    """
    empty: dict[str, Any] = {
        "issue_numbers": [],
        "failures": [],
        "errored": [],
        "deferred": 0,
    }
    # --dry-run must not perform live GitHub label mutations or state.json
    # writes (review finding on PR #670).
    if self.dry_run:
        return empty
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        subjects = [
            (pr_number, issue_number)
            for pr_number, issue_number in _wf._collect_escalated_label_subjects(state)
            if _escalated_label_needs_repair(state, pr_number=pr_number, issue_number=issue_number)
        ]
    if not subjects:
        return empty
    cap = self.config.runtime.escalated_label_repair_max_per_pass
    deferred = 0
    if cap > 0 and len(subjects) > cap:
        deferred = len(subjects) - cap
        subjects = subjects[:cap]

    outcomes: list[tuple[int, dict[str, Any] | None]] = []
    # Subjects whose GitHub call raised. Tracked separately from `failures`
    # (a transition() that returned non-APPLIED) because they are a different
    # operational state: nothing was written for them, so they retry next
    # pass. They MUST still be reported -- see the summary construction below.
    errored: list[int] = []
    for pr_number, issue_number in subjects:
        # Re-evaluate per item, immediately before touching GitHub, rather
        # than trusting the batch read above. #586 did this for a reason: a
        # concurrent unescalate() can free the issue at any point, and it
        # clears label_error, so the freed issue lands in the absent-key
        # ("never attempted") arm and would be silently re-escalated. Doing
        # the check once for the whole batch would widen that window from
        # "immediately before the call" to "up to `cap` GitHub round-trips
        # earlier" -- which is a real regression, and the existing race test
        # catches it.
        with _wf.state_lock(self.paths.state_file):
            fresh = _wf.load_state(self.paths.state_file)
        if not _escalated_label_needs_repair(
            fresh, pr_number=pr_number, issue_number=issue_number
        ):
            continue
        # issue_view/transition make GitHub API calls that can raise on
        # transient errors; a failure to verify one escalated issue must not
        # abort the pass. Skip it and retry next pass -- the escalated status
        # is already durable in state, so deferring loses no ground truth.
        try:
            # Issue #1266: a mechanical escalation's correct label is
            # operator_queue, not human_needed -- re-deriving from the
            # issue's reason_class (rather than hardcoding "escalated"
            # here) is what stops this sweep from clobbering a correctly
            # operator-queued issue back to human_needed every pass.
            fresh_issue_entry = fresh.get("issues", {}).get(str(issue_number), {})
            repair_reason_class = _repair_reason_class(fresh_issue_entry)
            edge = _escalation_edge("escalated", repair_reason_class)
            expected_label = _escalation_label(self.config.labels, edge)
            issue_view = self.gh.issue_view(int(issue_number))
            if expected_label is not None and expected_label in label_names(issue_view):
                outcomes.append((int(issue_number), None))
                continue
            result = _wf.transition(self.gh, self.config.labels, int(issue_number), edge)
        except Exception:
            logging.getLogger(__name__).warning(
                "escalated label repair for issue %s deferred (GitHub fetch "
                "failed); will retry next pass",
                issue_number,
                exc_info=True,
            )
            errored.append(int(issue_number))
            continue
        if result.outcome == TransitionOutcome.APPLIED:
            outcomes.append((int(issue_number), None))
        else:
            outcomes.append(
                (
                    int(issue_number),
                    {
                        "edge": edge,
                        "outcome": result.outcome.value,
                        "add_failures": result.add_failures,
                        "remove_failures": result.remove_failures,
                    },
                )
            )
    # `errored` gates this too, not just `outcomes`. If every subject's
    # GitHub call raised, `outcomes` is empty -- and returning `empty` here
    # would hand back {"issue_numbers": [], "failures": [], "deferred": 0},
    # byte-identical to "there was nothing to repair", while emitting no
    # event at all. An operator could then not tell a healthy quiet fleet
    # from N subjects failing on every single pass, and events.db (the audit
    # trail) would show nothing either. The logger.warning above is real but
    # log-only; the reported summary has to carry it as well.
    if not outcomes and not errored:
        return {**empty, "deferred": deferred}
    with _wf.state_lock(self.paths.state_file):
        state = _wf.load_state(self.paths.state_file)
        for issue_number, label_error in outcomes:
            entry = state["issues"].get(str(issue_number), {})
            state["issues"][str(issue_number)] = {
                **(entry if isinstance(entry, dict) else {}),
                "number": issue_number,
                "label_error": label_error,
            }
        summary = {
            "issue_numbers": [i for i, _ in outcomes],
            # transition() ran and did not fully apply -- label_error IS
            # written for these, so they are diagnosable from state.
            "failures": [i for i, e in outcomes if e is not None],
            # The GitHub call raised -- nothing was written, so these retry
            # next pass and are invisible in state. This list is the only
            # durable record that they were attempted at all.
            "errored": errored,
            "deferred": deferred,
        }
        state = _wf.append_event(
            state,
            "escalated_label_repaired",
            summary,
            state_path=self.paths.state_file,
            # A pass that repaired cleanly is routine, and the registry's
            # default ("info") is right for it. A pass that could not reach
            # GitHub, or whose transition() did not apply, is precisely what
            # an operator filters for with query_events(level="warning") --
            # and for an all-errored pass this event is the ONLY durable
            # record, since nothing is written to state for those subjects.
            # Leaving it at "info" would bury it among every routine pass.
            # None falls back to the registry (log_event's contract).
            level="warning" if (errored or summary["failures"]) else None,
        )
        _wf.save_state(self.paths.state_file, state)
    return summary
