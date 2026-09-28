"""Precision corpus for ``review.human_decision_markers`` (issue #1642 follow-up).

Every ``review_decision_reclassified_blocked`` event recorded fleet-wide up
to 2026-09-27 (swole, charlie-work, fresh-eyes), verbatim, hand-labelled.
The bare-noun marker set ("human", "operator", "sign-off") reclassified all
16; 12 were ordinary rework findings that merely used those words as domain
vocabulary, and they parked agent-doable PRs under agent:human-needed for
days. The default set must keep every genuine human call and drop the rest.
"""

from __future__ import annotations

import pytest

from charlie_work.config import ReviewConfig
from charlie_work.review_decision import human_decision_marker_match

# Findings that genuinely ask a human to decide -- must still reclassify.
_HUMAN_CALLS = [
    # swole PR #244
    "HUMAN DECISION (plan-mandated, not worker-resolvable): confirm whether the "
    "replay-after-load() design in w0-05c is acceptable as a schema-only stepping "
    "stone for #174, or whether the plan should change so rejections are captured "
    "where load() fails. Merge should wait on this call.",
    # swole PR #333
    "Human check (plan-mandated, no code change): confirm issue #281 (CLI consumer) "
    "is filed and open, since none of the new public functions has a src/ caller yet.",
    # fresh-eyes PR #39
    "In src/fresh_eyes/config.py load_config, catch ValueError (covers "
    "UnicodeDecodeError, TOMLDecodeError and pydantic ValidationError) alongside "
    "OSError so a non-UTF-8 or UTF-16 operator config returns "
    "Failure('config_invalid', ...) instead of raising. Note this is plan-mandated "
    "code, so the human should confirm the deviation from the plan.",
    # fresh-eyes PR #57
    "Include finding content (e.g. normalized observation) in the group_shared key "
    "so distinct findings on the same shared subtree are not merged or dropped; add "
    "a test with two different observations on one shared subtree (plan-mandated "
    "behaviour, flag for human/plan amendment).",
]

# Ordinary rework findings that mention "operator"/"human"/"sign-off" as
# domain vocabulary -- must route to automated rework, not the human queue.
_REWORK_FINDINGS = [
    # charlie-work PR #1818
    "Add a supervisor-level test for the idle-fleet drain exit: a drain marker "
    "landing while live workers == 0, no snapshot delta and no fallback pass due "
    "must exit with exit_reason=operator_stop_drained without calling fleet_loop.",
    # charlie-work PR #1901
    "Either (a) keep the arm as deliberate one-shot silencing, mandated by the "
    "#1894 amendment comment, and say so truthfully in the PR body so the human "
    "can confirm, or (b) narrow the blocked arm to the regen-unreachable case.",
    # charlie-work PR #1898
    "Either restrict the check to the branch(es) the issue actually recorded, or "
    "add a test showing the intended behaviour and an operator escape hatch.",
    # charlie-work PR #1925
    "Move the unescalate_cleared_reason/unescalate_cleared_at stamping logic into "
    "unescalate_reset_fields.py, so state_operator_commands.py (over the 800-line "
    "cap) adds only a few lines.",
    # charlie-work PR #1924
    "Add recovery-gate tests showing an operator-kind marker, an operator-* "
    "session_id marker, and a pid<=0 marker are each refused with 'cannot prove "
    "prior orchestrator ownership'.",
    # swole PR #268
    "Provide a real recovery path for the delete half-failure, for example an "
    "orphan sweep, or a documented operator cleanup, with a test.",
    # swole PR #298
    "State explicitly that founder read-and-approval before w1-h4 is a "
    "precondition and this PR is not legal sign-off.",
    # swole PR #267
    "int('') raises ValueError, so any in-session swole command crashes when the "
    "operator has set a limit override such as SWOLE_COACH_IDLE_AFTER_DAYS.",
    # swole PR #305
    "That dashboard router is bound to the local/founder tenant repo and gated by "
    "operator HTTP Basic, so intake tenants get a dead or wrong-tenant editor.",
    # swole PR #328
    "Replace permissions_path=coach-permissions.json with the shadow allow-list. "
    "If the human keeps the spec-literal deny-only setup, they must say so explicitly.",
    # swole PR #343
    "Keep the CANCELLATION -> free note in the PR body, stating that it is mandated "
    "by w2-05a and flagged for the human, since RevenueCat advises revoking on EXPIRATION.",
]


@pytest.mark.parametrize("finding", _HUMAN_CALLS)
def test_default_markers_keep_genuine_human_calls(finding: str) -> None:
    assert human_decision_marker_match([finding], ReviewConfig().human_decision_markers)


@pytest.mark.parametrize("finding", _REWORK_FINDINGS)
def test_default_markers_ignore_domain_vocabulary(finding: str) -> None:
    assert human_decision_marker_match([finding], ReviewConfig().human_decision_markers) is None
