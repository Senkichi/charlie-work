"""Escalation free-function family (issue #1283 Phase A).

Extracted verbatim from ``workflow.py``: the shared predicates and mutator
that escalate an issue/PR pair (or check/repair their escalated-label state),
plus the de-escalation skip-outcome builder. ``workflow.py`` re-exports every
symbol here via a facade import block (mirroring ``config.py``'s
``RunnerAllocationConfig`` re-export pattern and this repo's own
``dispatch_selection.py`` precedent), so existing import paths and
monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from .config import LabelConfig
from .labels import _edges
from .state import ESCALATION_REASON_CLASSES, escalation_reason_class, utc_now

# Issue #1266: the two labels.py edges every escalation call site otherwise
# hardcodes ("escalated" and "redispatch_escalated") get a mechanical
# counterpart that lands LabelConfig.operator_queue instead of human_needed.
# This table is the ONLY place that pairing is recorded — _escalation_edge()
# is the single point every escalation call site (and both label-repair
# consumers) goes through to pick between them.
_MECHANICAL_ESCALATION_EDGES: MappingProxyType[str, str] = MappingProxyType(
    {
        "escalated": "operator_queued",
        "redispatch_escalated": "redispatch_operator_queued",
    }
)


def _escalation_edge(edge: str, reason_class: str) -> str:
    """Map a labels.py transition edge + escalation reason_class to the edge to call.

    ``edge`` is the event name a call site would otherwise hardcode --
    ``"escalated"`` or ``"redispatch_escalated"``, the two edges every
    escalation call site in workflow.py resolves to before this helper
    existed. For ``reason_class == "mechanical"``, returns the
    operator-queue counterpart edge instead (issue #1266), so a mechanical
    escalation lands ``LabelConfig.operator_queue`` and reserves
    ``agent:human-needed`` for judgment calls. Any other ``edge`` (e.g.
    ``"blocked"``, which state.py's taxonomy makes judgment-only by
    construction -- the rework-cycle-cap "blocked" status is always a
    reviewer decision, never an automated one) has no operator-queue
    counterpart and passes through unchanged for every reason_class,
    including ``"mechanical"`` -- there is deliberately no "blocked but
    mechanical" cell in this mapping.

    Validates ``reason_class`` via ``escalation_reason_class`` first, so an
    unrecognized value raises ``ValueError`` here rather than silently
    falling through to the identity return -- the same fail-loud contract
    state.py documents for the persisted field. Callers that read a
    possibly-missing/legacy ``reason_class`` out of state (the label-repair
    self-heal sweep, reconcile's drift converger) must normalize it to a
    valid value themselves before calling this helper -- normalizing inside
    the helper would let an unclassified escalation silently default to
    either label instead of the caller making that choice explicitly.
    """
    escalation_reason_class(reason_class)
    if reason_class == "mechanical":
        return _MECHANICAL_ESCALATION_EDGES.get(edge, edge)
    return edge


def _escalation_label(labels: LabelConfig, edge: str) -> str | None:
    """The label a labels.py transition ``edge`` adds, derived from ``_edges()``.

    Both label-repair consumers (workflow.py's ``_repair_escalated_labels``
    self-heal sweep and reconcile.py's ``escalated_labels_converged`` drift
    check) need to know "is the right label already present" without
    re-deciding what "right" means -- that decision belongs to
    ``_edges()`` alone. Returns ``None`` for an edge whose add-tuple is
    empty (no escalation edge is one of these today, but the contract holds
    for any future edge that only removes).
    """
    add, _ = _edges(labels)[edge]
    return add[0] if add else None


def _repair_reason_class(issue_entry: dict[str, Any] | None) -> str:
    """The label-repair target reason_class for an escalated issue's state entry.

    The label-repair self-heal sweep (workflow.py's
    ``_repair_escalated_labels``) and reconcile's ``escalated_labels_converged``
    drift check both need "what label should this issue carry right now",
    which is not always the same as its stored ``reason_class``:

    - Missing/legacy ``reason_class`` (an escalation predating issue #797's
      field, not yet touched by ``_backfill_missing_reason_classes``) or any
      other unrecognized value falls back to ``"judgment"`` (``human_needed``)
      -- the same fail-closed default state.py's own
      ``DELIBERATELY_UNCLASSIFIED_ESCALATION_EVENT_KINDS`` handling uses: an
      unclassified escalation must never be silently assumed mechanical.
    - ``deescalation_cap_notified_at`` set means ``_deescalate_mechanical_issue``'s
      cap-exhaustion branch already moved this issue off ``operator_queue``
      onto ``human_needed`` (issue #1266) -- deliberately treated as
      ``"judgment"`` here even though the stored ``reason_class`` is still
      ``"mechanical"``, so neither repair consumer re-applies operator_queue
      to an issue the auto-clear sweep has already given up on.

    Centralized here (rather than inlined at each of the two call sites) so
    both consumers share the exact same rule instead of two
    independently-maintained copies of it.
    """
    if not isinstance(issue_entry, dict):
        return "judgment"
    if issue_entry.get("deescalation_cap_notified_at"):
        return "judgment"
    raw = issue_entry.get("reason_class")
    return raw if raw in ESCALATION_REASON_CLASSES else "judgment"


def _escalation_flags(
    pr_state: dict[str, Any], issue_state: dict[str, Any] | None
) -> tuple[bool, bool]:
    """Return ``(pr_escalated, issue_escalated)`` -- the shared escalation check.

    A PR or its linked issue is "escalated" when its state entry's ``status``
    is exactly ``"escalated"`` (a human has been asked to intervene via
    ``agent:human-needed``). ``issue_state`` may be ``None`` when the PR has
    no resolvable linked issue (cross-repo/fork PRs, or branches outside the
    configured prefix) -- in that case only ``pr_escalated`` can be True.

    This is the single predicate shared by ``review()`` (issue #384: gates
    ALL packet generation and label transitions at entry, unconditionally,
    once escalated) and ``merge_ready()`` (issue #833: gates only the
    specific mutation sites that had NO escalation awareness at all).
    ``merge_ready()`` deliberately does NOT call this at function entry --
    several of its remediation lanes must keep running while escalated for
    an unrelated reason (issue #776; real corpus: issues #592/#648/#606), so
    a blanket entry gate would silently re-break that fix. See the
    "escalation policy is per-route" comment inside ``merge_ready()`` for the
    full breakdown of which lanes exclude "escalated" and which don't.
    """
    pr_escalated = pr_state.get("status") == "escalated"
    issue_escalated = issue_state is not None and issue_state.get("status") == "escalated"
    return pr_escalated, issue_escalated


def _deescalation_skip(reason: str, issue_number: int) -> dict[str, Any]:
    """Build the "this candidate was not acted on" outcome of the de-escalation sweep.

    ``_deescalate_mechanical_issue`` used to signal every one of its
    not-acted-on branches by returning ``None``, which
    ``_maybe_deescalate_mechanical`` collapsed into a bare ``continue``. The
    resulting ``deescalation_pass_completed`` event recorded a candidate count
    and nothing else, so "48 candidates, 0 cleared" -- the steady state on the
    job-cannon fleet for 274 consecutive passes -- was not diagnosable from
    ``events.db`` at all and cost a full manual investigation to explain.

    Naming the reason at the branch (rather than having the caller re-derive
    it from a parallel table it maintains) is what keeps the histogram
    honest: the caller counts whatever string arrives, so a skip branch added
    later is reported without anyone remembering to register it.
    ``tests/test_deescalation.py`` enforces the one way that can still go
    wrong -- it asserts, over workflow.py's AST, that
    ``_deescalate_mechanical_issue`` keeps its non-Optional return annotation
    and contains no ``return None``, so a new branch cannot fall back into
    the unattributed bucket this helper exists to eliminate.
    """
    return {"skipped": reason, "issue_number": issue_number}


def _escalate_issue(
    state: dict[str, Any],
    issue_number: int | None,
    *,
    reason: str,
    reason_class: str,
    status: str = "escalated",
    pr_number: int | None = None,
    issue_extra: dict[str, Any] | None = None,
    pr_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Escalate an issue (and optionally its linked PR) with a required reason.

    ``reason`` is keyword-only and required, making ``status="escalated"``
    (or any other terminal status such as ``"blocked"``) without a reason
    unrepresentable at the call site. The issue record receives the paired
    ``escalation_reason`` and ``reason_class``; the PR record receives
    ``escalation_reason`` only (``reason_class`` is not part of the PR
    unescalate reset list and is not consulted by the mechanical
    de-escalation sweep).

    ``issue_number`` may be ``None`` when a PR is being escalated but its
    linked issue is not resolvable (cross-repo/fork PRs). In that case only
    the PR record is updated.

    Callers may supply ``issue_extra`` and ``pr_extra`` to merge additional
    lane-specific fields (attempt counters, dispatch claim state, etc.). The
    helper overwrites ``number``, ``issue_number``, ``status``, and
    ``escalation_reason`` on the PR, and ``number``, ``status``,
    ``escalation_reason``, ``reason_class``, ``merge_alert``, and
    ``terminal_since`` on the issue, so a caller cannot accidentally omit the
    required reason.

    ``terminal_since`` (issue #947) records when this issue most recently
    entered a terminal state via this helper, so a periodic sweep can alert
    on an issue parked in ``agent:human-needed`` past a configurable age
    instead of that state being silently invisible. It is unconditionally
    refreshed on every call (not just the first): a re-escalation after a
    de-escalation is a fresh terminal episode, not a continuation of the old
    one.
    """
    state.setdefault("issues", {})
    state.setdefault("prs", {})

    if issue_number is not None:
        issue_key = str(issue_number)
        issue_entry: dict[str, Any] = {
            **state["issues"].get(issue_key, {}),
            "number": issue_number,
            "status": status,
            "escalation_reason": reason,
            "reason_class": escalation_reason_class(reason_class),
            "merge_alert": "OK",
            "terminal_since": utc_now(),
        }
        if issue_extra:
            issue_entry.update(issue_extra)
        # Re-assert the paired escalation fields after any caller-supplied
        # extras, so stale/caller data can never silently drop the required
        # reason or misclassify its reason class.
        issue_entry["status"] = status
        issue_entry["escalation_reason"] = reason
        issue_entry["reason_class"] = escalation_reason_class(reason_class)
        issue_entry["merge_alert"] = "OK"
        issue_entry["terminal_since"] = utc_now()
        state["issues"][issue_key] = issue_entry

    if pr_number is not None:
        pr_key = str(pr_number)
        pr_entry: dict[str, Any] = {**state["prs"].get(pr_key, {})}
        if pr_extra:
            pr_entry.update(pr_extra)
        pr_entry["number"] = pr_number
        if issue_number is not None:
            pr_entry["issue_number"] = issue_number
        # Re-assert after caller-supplied extras so the PR always carries the
        # same required reason that justifies the status.
        pr_entry["status"] = status
        pr_entry["escalation_reason"] = reason
        state["prs"][pr_key] = pr_entry

    return state


def _escalated_label_needs_repair(
    state: dict[str, Any],
    *,
    pr_number: int | None,
    issue_number: int | None,
) -> bool:
    """Should the ``escalated`` label edge be re-applied for an already-escalated PR?

    ``pr_number`` may be ``None``: an issue can be escalated by a path that never
    produced a PR, and ``_collect_escalated_label_subjects`` yields those with no
    PR to pair them with. The PR arm of the status check then simply never
    matches, and the issue's own status decides.

    Same three-state ``label_error`` contract #586 established for
    ``dispatch_reviews``' self-heal sweep. This predicate is the shared
    definition both that sweep and the cross-family escalation call site now
    evaluate, so the two cannot drift apart:

    - ``None``        -> applied and verified on a prior pass; nothing to do.
                         This is the steady state, and answering it costs one
                         dict lookup rather than a GitHub label fetch.
    - a ``dict``      -> a prior ``transition()`` failed; retry.
    - key absent      -> the edge was never attempted; apply it.

    The status re-check is a race guard, not belt-and-braces: a concurrent
    ``unescalate()`` may have freed the issue since it was escalated, and it
    clears ``label_error`` when it does. Without the check, the absent-key arm
    would read that cleared state as "never attempted" and silently re-escalate
    the issue the unescalate had just released.
    """
    if issue_number is None:
        return False
    pr_entry = state.get("prs", {}).get(str(pr_number), {}) if pr_number is not None else {}
    issue_entry = state.get("issues", {}).get(str(issue_number), {})
    if not isinstance(pr_entry, dict):
        pr_entry = {}
    if not isinstance(issue_entry, dict):
        issue_entry = {}
    if pr_entry.get("status") != "escalated" and issue_entry.get("status") != "escalated":
        return False
    return not ("label_error" in issue_entry and issue_entry["label_error"] is None)


def _collect_escalated_label_subjects(
    state: dict[str, Any],
) -> list[tuple[int | None, int]]:
    """``(pr_number, issue_number)`` pairs whose ``escalated`` label edge may be owed.

    Issue #1088. The set #586's self-heal sweep iterated was built inside
    ``dispatch_reviews``' candidate-filter loop, so its members were whatever the
    *dispatch selection* happened to consider that pass. That is what made the
    sweep unreachable: candidate selection runs below the
    ``review_dispatch.enabled`` early return, and both deployed fleets run that
    flag false, so the set is empty and the sweep has nothing to do for as long
    as the flag stays off -- which, measured at the time of the fix, was every
    pass for ~8 days. (Not "never": ``escalated_label_repaired`` did fire once,
    job-cannon 2026-07-28T21:36:56Z, repairing 10 issues. That single run is the
    argument *for* the fix, not against it.) Deriving the subjects from ``state``
    instead is the whole fix -- the repair set must not depend on the dispatch
    lane being on.

    An escalated PR contributes its linked issue; an escalated issue contributes
    itself even when no PR points at it, because ``_escalate_issue`` and the
    rework-cycle cap can escalate an issue that has no PR at all. Pairs are
    deduplicated by issue, preferring the PR-derived pair so the caller's status
    re-check can see both records.
    """
    subjects: dict[int, int | None] = {}
    prs = state.get("prs", {})
    if isinstance(prs, dict):
        for pr_key, pr_entry in prs.items():
            if not isinstance(pr_entry, dict) or pr_entry.get("status") != "escalated":
                continue
            issue_number = pr_entry.get("issue_number")
            if issue_number is None:
                continue
            try:
                subjects[int(issue_number)] = int(pr_key)
            except (TypeError, ValueError):
                continue
    issues = state.get("issues", {})
    if isinstance(issues, dict):
        for issue_key, issue_entry in issues.items():
            if not isinstance(issue_entry, dict) or issue_entry.get("status") != "escalated":
                continue
            try:
                issue_number = int(issue_key)
            except (TypeError, ValueError):
                continue
            subjects.setdefault(issue_number, None)
    return [(pr_number, issue_number) for issue_number, pr_number in sorted(subjects.items())]
