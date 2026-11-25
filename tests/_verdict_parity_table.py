"""The scenario table and the declared-differences list for ``test_verdict_parity.py``.

Kept apart from the tests so each file stays under the file-size cap.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from _verdict_parity_driver import MISSING, Diff, Scenario
from charlie_work.rescue_review import LEGACY_VACUOUS_SUMMARY

HUMAN_CALL_FINDING = "This needs a human decision before it can ship."


def _refused(reason: str) -> tuple[tuple[str, Any], ...]:
    return (("result.ok", False), ("result.data.reason", reason), ("events.order", ()))


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "approve",
        "approved",
        summary="Looks good.",
        expect=(
            ("result.ok", True),
            ("issue.status", "approved"),
            ("pr_record.status", "approved"),
            ("labels.added", ("agent:pr-open",)),
            ("events.order", ("record_review", "lifecycle_transition")),
        ),
    ),
    Scenario(
        "approve_no_summary_no_comment",
        "approved",
        summary="",
        post_verdict_comment=False,
        expect=(("result.ok", True), ("verdict_comment", None)),
    ),
    # ``max_rework_cycles`` is 2: a count of 1 is under the cap, 2 is at it.
    Scenario(
        "rework_under_cap",
        "request_changes",
        prior_count=1,
        prior_head="older",
        expect=(
            ("result.data.escalated", False),
            ("result.data.request_changes_count", 2),
            ("issue.status", "rework_requested"),
            ("labels.added", ("agent:needs-rework",)),
            ("rework_prompt", True),
        ),
    ),
    Scenario(
        "rework_first_verdict",
        "request_changes",
        expect=(
            ("result.data.request_changes_count", 1),
            ("issue.status", "rework_requested"),
            ("rework_prompt", True),
        ),
    ),
    Scenario(
        "rework_at_cap_new_head",
        "request_changes",
        prior_count=2,
        prior_head="older",
        expect=(
            ("result.data.escalated", True),
            ("result.data.request_changes_count", 2),
            ("issue.status", "escalated"),
            ("pr_record.status", "escalated"),
            ("pr_record.escalation_reason", "max_rework_cycles_exceeded"),
            ("labels.added", ("agent:operator-queue",)),
            ("rework_prompt", False),
            ("verdict_comment", None),
        ),
    ),
    Scenario(
        "rework_at_cap_same_head",
        "request_changes",
        prior_count=2,
        prior_head="same",
        expect=(
            ("result.data.escalated", False),
            ("result.data.request_changes_count", 2),
            ("issue.status", "rework_requested"),
            ("labels.added", ("agent:needs-rework",)),
            ("rework_prompt", True),
        ),
    ),
    Scenario(
        "rework_under_cap_same_head",
        "request_changes",
        prior_count=1,
        prior_head="same",
        expect=(
            ("result.data.escalated", False),
            ("result.data.request_changes_count", 1),
            ("issue.status", "rework_requested"),
        ),
    ),
    # The two paths part ways here by design (declared: no Rescue tier locally).
    Scenario(
        "rework_at_cap_new_head_rescue_enabled",
        "request_changes",
        prior_count=2,
        prior_head="older",
        rescue_enabled=True,
        # The PR path spends its one rescue attempt; the Local path has no Rescue
        # tier and escalates directly at the cap.
        expect_remote=(
            ("result.data.escalated", False),
            ("result.data.rescue_dispatched", True),
            ("result.data.request_changes_count", 3),
            ("issue.status", "rework_requested"),
            ("pr_record.rescue_attempted", True),
            ("pr_record.status", "request_changes"),
            ("rework_prompt", True),
            ("labels.added", ("agent:needs-rework",)),
        ),
        expect_local=(
            ("result.data.escalated", True),
            ("result.data.rescue_dispatched", MISSING),
            ("result.data.request_changes_count", 2),
            ("issue.status", "escalated"),
            ("pr_record.rescue_attempted", MISSING),
            ("pr_record.status", "escalated"),
            ("rework_prompt", False),
            ("labels.added", ("agent:operator-queue",)),
        ),
    ),
    # The one rescue attempt is already spent: both paths escalate at the cap.
    Scenario(
        "rework_at_cap_new_head_rescue_spent",
        "request_changes",
        prior_count=2,
        prior_head="older",
        rescue_enabled=True,
        rescue_attempted=True,
        expect=(
            ("result.data.escalated", True),
            ("issue.status", "escalated"),
            ("labels.added", ("agent:operator-queue",)),
        ),
    ),
    Scenario(
        "blocked",
        "blocked",
        summary="Needs a product call on the API shape.",
        expect=(
            ("issue.status", "blocked"),
            ("pr_record.status", "blocked"),
            ("pr_record.escalation_reason", "review_blocked"),
            ("labels.added", ("agent:human-needed",)),
        ),
    ),
    Scenario(
        "human_call_reclassified",
        "request_changes",
        summary="One finding for the operator.",
        required_changes=(HUMAN_CALL_FINDING,),
        expect=(
            ("result.data.decision", "blocked"),
            ("result.data.reclassified_from", "request_changes"),
            ("issue.status", "blocked"),
            ("labels.added", ("agent:human-needed",)),
            ("rework_prompt", False),
            (
                "events.order",
                ("record_review", "review_decision_reclassified_blocked", "lifecycle_transition"),
            ),
        ),
    ),
    Scenario(
        "rework_structured_required_changes",
        "request_changes",
        summary="Two things to fix.",
        required_changes=("Add a test.", "Rename the helper."),
        expect=(("decision_file.required_changes", ["Add a test.", "Rename the helper."]),),
    ),
    Scenario(
        "rework_vacuous_summary",
        "request_changes",
        summary=LEGACY_VACUOUS_SUMMARY,
        expect=(("decision_file.findings_channel", "vacuous"),),
    ),
    Scenario(
        "rework_forced_comment",
        "request_changes",
        comment=True,
        post_verdict_comment=False,
        expect=(("result.ok", True),),
    ),
    Scenario(
        "rework_issue_closed",
        "request_changes",
        issue_closed=True,
        expect=(
            ("issue.status", "rework_requested"),
            ("labels.added", ()),
            (
                "events.order",
                (
                    "worker_test_selection_disabled",
                    "record_review",
                    "rework_label_skipped_issue_closed",
                ),
            ),
        ),
    ),
    Scenario(
        "rework_ci_gate_provenance",
        "request_changes",
        provenance="ci_gate_auto_reject",
        expect=(
            ("issue.no_op_checkpoint_at", MISSING),
            ("decision_file.verdict_provenance", "ci_gate_auto_reject"),
        ),
    ),
    Scenario(
        "rework_external_findings",
        "request_changes",
        external_comment=True,
        expect_remote=(
            ("decision_file.external_findings", ["Please also handle the empty-input case."]),
            ("events.record_review#0.payload.external_findings_count", 1),
        ),
        expect_local=(("decision_file.external_findings", MISSING),),
    ),
    # A verdict extracted from a dead reviewer's log: the caveat and the session
    # metrics must be persisted identically.
    Scenario(
        "rework_log_extracted_verdict",
        "request_changes",
        verdict_source="log",
        session_metrics=(("cost_usd", 0.5), ("turns", 3)),
        expect=(
            ("decision_file.verdict_source", "log"),
            ("pr_record.review_session_metrics.cost_usd", 0.5),
            ("events.record_review#0.payload.session_metrics.turns", 3),
        ),
    ),
    Scenario(
        "blocked_log_extracted_verdict",
        "blocked",
        summary="Needs a product call.",
        verdict_source="events",
        expect=(("decision_file.verdict_source", "events"),),
    ),
    Scenario(
        "stale_head_packet_behind_live",
        "approved",
        packet="stale",
        expect=_refused("head_moved_during_build"),
    ),
    Scenario(
        "stale_head_reviewed_head_is_packet",
        "approved",
        packet="stale",
        reviewed_head="packet",
        expect=_refused("head_moved_during_build"),
    ),
    Scenario(
        "stale_head_operator_pins_packet",
        "approved",
        packet="stale",
        reviewed_head="packet",
        allow_stale_head=True,
        expect=(("result.ok", True), ("result.data.reviewed_head_source", "packet")),
    ),
    # The packet is behind the live head and the operator pins the LIVE head: the
    # patch-id must come from the live head's diff, not from the stale packet's.
    Scenario(
        "stale_head_operator_pins_live",
        "approved",
        packet="stale",
        reviewed_head="live",
        expect=(("result.ok", True), ("result.data.reviewed_head_source", "live")),
    ),
    Scenario(
        "stale_head_reviewed_head_matches_nothing",
        "approved",
        reviewed_head="bogus",
        expect=_refused("reviewed_head_mismatch"),
    ),
    Scenario(
        "no_packet_pins_live_head",
        "approved",
        packet="none",
        expect=(("result.ok", True), ("result.data.reviewed_head_source", "live")),
    ),
    Scenario(
        "already_escalated_record",
        "approved",
        pr_status="escalated",
        expect=_refused("pr_escalated"),
    ),
    Scenario(
        "already_escalated_issue",
        "request_changes",
        issue_status="escalated",
        expect=_refused("pr_escalated"),
    ),
    Scenario(
        "already_merged",
        "request_changes",
        pr_status="merged",
        expect=_refused("pr_terminal_state"),
    ),
    Scenario(
        "already_closed",
        "approved",
        pr_status="closed",
        expect=_refused("pr_terminal_state"),
    ),
    Scenario("invalid_decision", "maybe", expect=_refused("invalid_decision")),
    Scenario(
        "invalid_provenance",
        "approved",
        provenance="made_up",
        expect=_refused("invalid_verdict_provenance"),
    ),
    Scenario("missing_summary", "blocked", summary="  ", expect=_refused("missing_summary")),
)


# ---------------------------------------------------------------------------
# Declared differences
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Declared differences
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclaredDifference:
    """One known difference, as narrow as the row names it.

    ``paths`` are the outcome paths the row may exempt: an exact path, or a
    prefix ending in ``.``. ``when`` further limits it by scenario and values.
    The staleness test requires EVERY named path to be exempted somewhere, so a
    path that stops differing cannot linger in the list.
    """

    name: str
    reason: str
    paths: tuple[str, ...]
    when: Callable[[Diff], bool] = lambda diff: True
    note: str = ""
    observable: bool = True  # False: these two functions cannot show it on their own

    def path_hit(self, path: str) -> str | None:
        for named in self.paths:
            if path == named or (named.endswith(".") and path.startswith(named)):
                return named
        return None

    def matches(self, diff: Diff) -> bool:
        return self.path_hit(diff.path) is not None and self.when(diff)


def _remote_missing(diff: Diff) -> bool:
    return diff.remote is MISSING


def _local_missing(diff: Diff) -> bool:
    return diff.local is MISSING


_RESCUE_SCENARIO = "rework_at_cap_new_head_rescue_enabled"
_SUMMARY = "Add a regression test for the new branch."
# path -> (record_review value, record_local_review value), exactly, in the rescue scenario.
_RESCUE_PINNED: dict[str, tuple[Any, Any]] = {
    "decision_file.escalated": (False, True),
    "events.lifecycle_transition#0.payload.cause": ("rework_requested", "operator_queued"),
    "events.lifecycle_transition#0.payload.to_state": ("needs_rework", "operator_queue"),
    "events.order": (
        (
            "worker_test_selection_disabled",
            "record_review",
            "rescue_dispatched",
            "lifecycle_transition",
        ),
        ("record_review", "lifecycle_transition"),
    ),
    "events.record_review#0.payload.escalated": (False, True),
    "events.rescue_dispatched#0.level": ("info", MISSING),
    "events.rescue_dispatched#0.payload.cause": ("rework_cycle_cap", MISSING),
    "events.rescue_dispatched#0.payload.issue_number": (7, MISSING),
    "events.rescue_dispatched#0.payload.pr_number": (7, MISSING),
    "events.worker_test_selection_disabled#0.level": ("info", MISSING),
    "events.worker_test_selection_disabled#0.payload.issue_number": (7, MISSING),
    "events.worker_test_selection_disabled#0.payload.pr_number": (7, MISSING),
    "events.worker_test_selection_disabled#0.payload.detail": (
        "ci-fleet.toml sets enabled = false",
        MISSING,
    ),
    "events.worker_test_selection_disabled#0.payload.reason": ("disabled", MISSING),
    "issue.escalation_reason": (MISSING, "max_rework_cycles_exceeded"),
    "issue.escalation_reasons_seen": (MISSING, ["max_rework_cycles_exceeded"]),
    "issue.no_op_checkpoint_at": ("<set>", MISSING),
    "issue.reason_class": (MISSING, "mechanical"),
    "issue.status": ("rework_requested", "escalated"),
    "issue.terminal_since": (MISSING, "<set>"),
    "label_edge": (
        '[{"cause": "rework_requested", "from_state": "review_ready", "to_state": "needs_rework"}]',
        '[{"cause": "operator_queued", "from_state": "review_ready", "to_state": "operator_queue"}]',
    ),
    "labels.added": (("agent:needs-rework",), ("agent:operator-queue",)),
    "labels.final": (
        ("agent:needs-rework", "automated-ready"),
        ("agent:operator-queue", "automated-ready"),
    ),
    "pr_record.escalation_reason": (MISSING, "max_rework_cycles_exceeded"),
    "pr_record.request_changes_count": (3, 2),
    "pr_record.rescue_attempted": (True, MISSING),
    "pr_record.rescue_cause": ("rework_cycle_cap", MISSING),
    "pr_record.rescue_dispatched_at": ("<set>", MISSING),
    "pr_record.status": ("request_changes", "escalated"),
    "result.data.escalated": (False, True),
    "result.data.request_changes_count": (3, 2),
    "result.data.rework_path": (True, False),
    "result.message": (
        "review recorded \u2014 rework cap (2) reached, rescue tier dispatched instead of "
        "escalating (head from packet)",
        "review recorded \u2014 rework cap (2) reached, escalated to human (head from packet)",
    ),
    "rework_prompt": (True, False),
    "rounds": (
        (
            "round-1/diff.patch",
            "round-1/review-decision.json",
            "round-1/rework-dispatch-note.txt",
            "round-1/rework-prompt.md",
        ),
        ("round-1/diff.patch", "round-1/review-decision.json"),
    ),
    "verdict_comment": (f"## Fleet review - round 1 - request_changes\n\n{_SUMMARY}", None),
    "verdict_comment_channel": ("pr", None),
}
_REFUSALS = frozenset(sc.name for sc in SCENARIOS if ("result.ok", False) in sc.expect)


def _rescue_when(diff: Diff) -> bool:
    if diff.scenario != _RESCUE_SCENARIO:
        return False
    if diff.path == "rework_dispatch_note":
        return (
            isinstance(diff.remote, str)
            and diff.remote.startswith("## Rescue-tier rework (issue #555)")
            and diff.local is None
        )
    return _RESCUE_PINNED[diff.path] == (diff.remote, diff.local)


DECLARED_DIFFERENCES: tuple[DeclaredDifference, ...] = (
    # --- the issue's table rows (seven remain): difference and reason exactly as its table gives them ---
    DeclaredDifference(
        "No Rescue tier on the Local path",
        "stated in a comment in `record_local_review`; at the cap a local record escalates directly",
        (
            *_RESCUE_PINNED,
            "rework_dispatch_note",
        ),
        _rescue_when,
        note="Pinned to exact values in the one rescue scenario; each side's outcome is "
        "also asserted by that scenario's expect_remote / expect_local.",
    ),
    DeclaredDifference(
        "No external-findings intake",
        "there are no PR comments to read",
        (
            "decision_file.external_findings",
            "decision_file.before",
            "events.record_review#0.payload.external_findings_count",
        ),
        _local_missing,
    ),
    DeclaredDifference(
        "No CI auto-reject",
        "there is no CI",
        (),
        note="Not observable at these two functions: a ci_gate_auto_reject provenance reaches "
        "both identically (scenario rework_ci_gate_provenance is the evidence).",
        observable=False,
    ),
    DeclaredDifference(
        "Verdict comment posted on the issue",
        "there is no PR to comment on",
        ("verdict_comment_channel",),
        lambda diff: (diff.remote, diff.local) == ("pr", "issue"),
    ),
    DeclaredDifference(
        "Local-only decision fields (`human_call_reclassified_from`, `human_call_marker`, `local: True`)",
        "local bookkeeping",
        (
            "decision_file.human_call_reclassified_from",
            "decision_file.human_call_marker.",
            "decision_file.local",
        ),
        _remote_missing,
    ),
    DeclaredDifference(
        "`reviewed_body_sha256` not recorded locally",
        "drift, tracked separately",
        ("decision_file.reviewed_body_sha256", "pr_record.reviewed_body_sha256"),
        _local_missing,
    ),
    DeclaredDifference(
        "Local lifecycle transitions omit the record number",
        "drift, tracked separately",
        ("events.lifecycle_transition#0.payload.pr_number",),
        lambda diff: diff.remote == 7 and diff.local in (MISSING, None),
    ),
    # --- differences the table surfaced that the issue does not name ---
    DeclaredDifference(
        "record_review Event payload carries `local: True` on the Local path",
        "undecided",
        ("events.record_review#0.payload.local",),
        _remote_missing,
    ),
    DeclaredDifference(
        "Local result data carries the `issue` and `local` keys",
        "undecided",
        ("result.data.issue", "result.data.local"),
        _remote_missing,
    ),
    DeclaredDifference(
        "PR result data carries `rescue_dispatched`, Local result data has no such key",
        "undecided",
        ("result.data.rescue_dispatched",),
        lambda diff: isinstance(diff.remote, bool) and diff.local is MISSING,
    ),
    DeclaredDifference(
        "Terminal-state refusal names a different data key",
        "undecided",
        ("result.data.terminal", "result.data.terminal_state"),
        lambda diff: (
            diff.scenario in {"already_merged", "already_closed"}
            and (
                (diff.path == "result.data.terminal" and diff.remote is MISSING)
                or (diff.path == "result.data.terminal_state" and diff.local is MISSING)
            )
        ),
    ),
    DeclaredDifference(
        "Local PR record carries `local: True`",
        "undecided",
        ("pr_record.local",),
        _remote_missing,
    ),
    DeclaredDifference(
        "Refusal wording differs between the two paths",
        "undecided",
        ("result.message",),
        lambda diff: diff.scenario in _REFUSALS,
    ),
)
