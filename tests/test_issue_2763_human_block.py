"""Issue #2763 part (b): a human's outstanding request-changes holds the merge path.

The merge lane approved on the fleet's own verdict only; on 2026-10-19 that let a
PR merge over an unresolved human review. ``merge_path/human_block.py`` adds a
lazily evaluated hold at the point a merge or hand-off is imminent:

* ``reviewDecision == CHANGES_REQUESTED`` on the payload in hand blocks;
* otherwise the newest PR comment matching a verdict pattern decides (the
  orchestrator's own marker-bearing comments are skipped), approved clearing;
* unreadable comments hold (fail closed);
* it is a hold, not an action: no ``mergequeue`` label, no escalation, and it
  clears itself on the next pass.

Sections: the pure classifier, the gate, ``decide_merge``, and end-to-end passes
through ``apply.py`` / ``preview.py`` using the merge-path characterization
harness. Every negative has a positive control on the same fixture.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import _merge_path_facts as mf
import pytest
from _fakes_github import FakeGitHub
from _merge_path_characterization_harness import _PR, _cfg, _run

from charlie_work.config import AutoMergeConfig, OrchestratorConfig, build_config_from_data
from charlie_work.config_validation import ConfigError
from charlie_work.github import GitHubRunResult
from charlie_work.merge_path.decide import decide_merge
from charlie_work.merge_path.gather import config_slice
from charlie_work.merge_path.human_block import (
    CHANGES_REQUESTED,
    HUMAN_BLOCK_SOURCE_COMMENT,
    HUMAN_BLOCK_SOURCE_REVIEW_DECISION,
    HumanBlockRead,
    classify_human_block,
    compile_verdict_pattern,
    human_block_gate,
)
from charlie_work.merge_path.model import FactNotGathered, Hold, MergePathConfig, PlanKind
from charlie_work.workflow import ORCHESTRATOR_COMMENT_MARKER, _is_orchestrator_comment

_AUTO = AutoMergeConfig()
REQUEST_CHANGES = compile_verdict_pattern(_AUTO.human_block_request_changes_pattern)
APPROVED = compile_verdict_pattern(_AUTO.human_block_approved_pattern)

RC_MARKER = "<!-- review-verdict: request-changes -->"
OK_MARKER = "<!-- review-verdict: approved -->"
RC_PLAIN = "**Verdict: request changes.** The retry path swallows the error."
OK_PLAIN = "**Verdict: approved.** Addressed."


def _c(body: str, *, orchestrator: bool = False) -> dict[str, Any]:
    prefix = f"{ORCHESTRATOR_COMMENT_MARKER}\n" if orchestrator else ""
    return {"body": prefix + body, "user": {"login": "someone", "type": "User"}}


def _classify(
    comments: list[dict[str, Any]], review_decision: str | None = None
) -> HumanBlockRead:
    return classify_human_block(
        review_decision,
        comments,
        request_changes=REQUEST_CHANGES,
        approved=APPROVED,
        is_orchestrator_comment=_is_orchestrator_comment,
    )


# ---------------------------------------------------------------------------
# 1. The pure classifier
# ---------------------------------------------------------------------------


def test_shipped_patterns_are_on_by_default_and_compile() -> None:
    assert _AUTO.honor_human_block is True
    assert OrchestratorConfig().auto_merge.honor_human_block is True
    assert REQUEST_CHANGES.flags & re.IGNORECASE


@pytest.mark.parametrize(
    "body", [RC_MARKER, RC_PLAIN, "VERDICT: Request-Changes", "**Verdict:** request changes"]
)
def test_a_request_changes_comment_blocks(body: str) -> None:
    assert _classify([_c(body)]) == HumanBlockRead(True, source=HUMAN_BLOCK_SOURCE_COMMENT)


@pytest.mark.parametrize("body", [OK_MARKER, OK_PLAIN, "verdict: approved"])
def test_an_approved_comment_does_not_block(body: str) -> None:
    assert _classify([_c(body)]) == HumanBlockRead(False)


def test_the_newest_verdict_decides_so_a_later_approve_clears_the_block() -> None:
    assert _classify([_c(RC_PLAIN), _c(OK_PLAIN)]).blocked is False
    # Control: reversed, the same two comments block.
    assert _classify([_c(OK_PLAIN), _c(RC_PLAIN)]).blocked is True


def test_non_verdict_comments_are_skipped_not_decisive() -> None:
    comments = [_c(RC_PLAIN), _c("thanks, looking now"), _c("+1")]
    assert _classify(comments).blocked is True


def test_a_comment_matching_both_patterns_resolves_to_block() -> None:
    assert _classify([_c(f"{OK_PLAIN}\n{RC_PLAIN}")]).blocked is True


def test_orchestrator_marker_comments_are_ignored() -> None:
    """The fleet's own verdict comments carry the provenance marker: its
    ``approved`` must not clear a human's block (the 2026-10-19 shape), and its
    ``request-changes`` must not block on a human's approve."""
    human_block_then_fleet_approve = [_c(RC_PLAIN), _c(OK_MARKER, orchestrator=True)]
    assert _classify(human_block_then_fleet_approve).blocked is True

    human_approve_then_fleet_block = [_c(OK_PLAIN), _c(RC_MARKER, orchestrator=True)]
    assert _classify(human_approve_then_fleet_block).blocked is False


def test_a_quote_reply_that_merely_contains_the_marker_is_still_a_human_comment() -> None:
    """Same prefix-only semantics as ``_is_orchestrator_comment``: a quote reply
    embeds the marker without starting with it, and must not be dropped."""
    quote_reply = _c(f"> {ORCHESTRATOR_COMMENT_MARKER}\n> approved\n\n{RC_PLAIN}")
    assert _classify([quote_reply]).blocked is True


def test_changes_requested_review_decision_blocks_without_reading_comments() -> None:
    result = _classify([_c(OK_PLAIN)], review_decision=CHANGES_REQUESTED)
    assert result == HumanBlockRead(True, source=HUMAN_BLOCK_SOURCE_REVIEW_DECISION)
    # Control: any other decision defers to the comments.
    assert _classify([_c(OK_PLAIN)], review_decision="APPROVED").blocked is False


def test_no_comments_and_malformed_items_do_not_block() -> None:
    assert _classify([]).blocked is False
    assert _classify([{"body": None}, {"no_body": True}, {"body": 7}]).blocked is False  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 2. The gate
# ---------------------------------------------------------------------------


def _holds(**over: Any) -> Any:
    return mf.hold_facts(config=mf.cfg(honor_human_block=True), **over)


def test_gate_is_not_read_when_the_feature_is_off_or_nothing_can_merge() -> None:
    assert human_block_gate(True, mf.hold_facts()).read_needed is False  # fixture default: off
    assert human_block_gate(False, _holds()).read_needed is False  # no merge/hand-off imminent


def test_gate_refuses_to_treat_an_unread_signal_as_no_block() -> None:
    with pytest.raises(FactNotGathered):
        human_block_gate(True, _holds())
    # Control: read (clear) -> a gate that does not hold.
    gate = human_block_gate(True, _holds(human_block=False))
    assert (gate.read_needed, gate.held) == (True, False)


def test_gate_holds_on_a_block_and_on_an_unavailable_read() -> None:
    blocked = human_block_gate(True, _holds(human_block=True, human_block_source="comment"))
    assert (blocked.held, blocked.source, blocked.holds) == (True, "comment", {Hold.HUMAN_BLOCK})
    unavailable = human_block_gate(True, _holds(human_block=False, human_block_unavailable=True))
    assert (unavailable.held, unavailable.holds) == (True, {Hold.HUMAN_BLOCK_UNAVAILABLE})


# ---------------------------------------------------------------------------
# 3. decide_merge
# ---------------------------------------------------------------------------


def _queue_holds(**over: Any) -> Any:
    """Mergequeue hand-off configured, merge-hold label read clear."""
    return mf.hold_facts(
        config=mf.cfg(mergequeue_label=mf.LABEL, honor_human_block=True),
        merge_hold=False,
        live_head_sha=mf.HEAD,
        **over,
    )


def test_decide_hands_off_when_no_human_block() -> None:
    plan = decide_merge(mf.readiness(), _queue_holds(human_block=False))
    assert (plan.kind, plan.action_hand_off) == (PlanKind.HAND_OFF, True)
    assert plan.read_human_block is True
    assert plan.human_block is False


def test_decide_holds_a_hand_off_on_a_human_block() -> None:
    plan = decide_merge(
        mf.readiness(), _queue_holds(human_block=True, human_block_source="comment")
    )
    assert (plan.kind, plan.action_hand_off, plan.action_merge) == (PlanKind.HOLD, False, False)
    assert Hold.HUMAN_BLOCK in plan.holds
    assert (plan.human_block, plan.human_block_source) == (True, "comment")


def test_decide_holds_on_an_unavailable_read_fail_closed() -> None:
    plan = decide_merge(
        mf.readiness(), _queue_holds(human_block=False, human_block_unavailable=True)
    )
    assert plan.kind is PlanKind.HOLD
    assert plan.action_hand_off is False
    assert Hold.HUMAN_BLOCK_UNAVAILABLE in plan.holds
    assert plan.human_block_unavailable is True


def test_decide_holds_a_direct_self_merge_too() -> None:
    """Unlike the merge-hold label, the block is not gated on the queue label:
    with no ``mergequeue_label`` the fleet would merge directly."""
    holds = mf.hold_facts(config=mf.cfg(honor_human_block=True))
    merged = decide_merge(mf.readiness(), dataclasses.replace(holds, human_block=False))
    assert (merged.kind, merged.action_merge) == (PlanKind.MERGE, True)

    held = decide_merge(mf.readiness(), dataclasses.replace(holds, human_block=True))
    assert (held.kind, held.action_merge) == (PlanKind.HOLD, False)
    assert Hold.HUMAN_BLOCK in held.holds


def test_decide_kill_switch_ignores_the_block() -> None:
    """``auto_merge.honor_human_block: false`` restores the old behaviour even
    when stale block facts are present."""
    holds = mf.hold_facts(
        config=mf.cfg(mergequeue_label=mf.LABEL, honor_human_block=False),
        merge_hold=False,
        human_block=True,
    )
    plan = decide_merge(mf.readiness(), holds)
    assert (plan.kind, plan.read_human_block) == (PlanKind.HAND_OFF, False)


def test_decide_reads_nothing_when_a_merge_is_not_imminent() -> None:
    """should_merge=False: no merge/hand-off can happen, so no read is demanded
    (no FactNotGathered) and the plan carries no human-block state."""
    plan = decide_merge(mf.readiness(), _queue_holds(should_merge=False))
    assert plan.read_human_block is False
    assert Hold.HUMAN_BLOCK not in plan.holds


def test_decide_raises_when_a_needed_read_is_missing() -> None:
    with pytest.raises(FactNotGathered):
        decide_merge(mf.readiness(), _queue_holds())


def test_a_human_blocked_pass_is_never_capped_or_routed_to_rework() -> None:
    """A held pass neither caps nor routes: an outstanding request-changes must
    not be answered with a rework dispatch or an escalation."""
    at_cap = mf.persisted(mergequeue_requeues=3, mergequeue_requeues_head_sha=mf.HEAD)
    capped = decide_merge(mf.readiness(), _queue_holds(human_block=False, persisted=at_cap))
    assert capped.mergequeue_requeue_capped is True  # control: the fixture arms the cap

    blocked = decide_merge(mf.readiness(), _queue_holds(human_block=True, persisted=at_cap))
    assert blocked.mergequeue_requeue_capped is False
    assert blocked.requeue_rework is False
    assert blocked.kind is PlanKind.HOLD


# ---------------------------------------------------------------------------
# 4. End to end through apply.py / preview.py
# ---------------------------------------------------------------------------


class _Gh(FakeGitHub):
    """FakeGitHub that counts PR-comment reads and can fail them."""

    def __init__(self) -> None:
        super().__init__()
        self.comment_reads = 0
        self.comments_unreadable = False

    def run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):
        joined = " ".join(args)
        if re.search(r"/issues/\d+/comments$", joined):
            self.comment_reads += 1
            if self.comments_unreadable:
                return GitHubRunResult(
                    ok=False, returncode=1, stdout="", stderr="boom", value=None, error="boom"
                )
        return super().run(args, json_output=json_output, allow_failure=allow_failure)


def _config(*, honor: bool = True, mergequeue: str | None = "mergequeue") -> OrchestratorConfig:
    config = _cfg(mergequeue=mergequeue)
    return dataclasses.replace(
        config, auto_merge=dataclasses.replace(config.auto_merge, honor_human_block=honor)
    )


def _pass(
    tmp_path: Path,
    *,
    comments: list[dict[str, Any]] | None = None,
    review_decision: str | None = None,
    unreadable: bool = False,
    config: OrchestratorConfig | None = None,
    dry_run: bool = False,
    **run_kwargs: Any,
):
    def pre(_app: Any, _paths: Any, gh: _Gh) -> None:
        # Setup (record_review) may itself read comments; count only the pass.
        gh.comment_reads = 0
        gh.comments_unreadable = unreadable
        gh.pr_external_issue_comments[_PR] = comments or []
        for pr in gh.prs:
            if pr["number"] == _PR and review_decision is not None:
                pr["reviewDecision"] = review_decision

    return _run(
        tmp_path / ("dry" if dry_run else "live"),
        config or _config(),
        _Gh,
        dry_run=dry_run,
        pre=pre,
        **run_kwargs,
    )


def _assert_held(out: Any) -> None:
    assert out.gh.pr_labels_added == []
    assert out.gh.merged == []
    assert out.data["mergequeue_label_applied"] is None
    assert out.pr_entry.get("status") != "mergequeue"
    # A hold is not an escalation.
    assert out.pr_entry.get("status") != "escalated"
    assert out.gh.labels_added == []


def test_control_a_clear_pass_hands_off_and_reads_the_comments_once(tmp_path: Path) -> None:
    out = _pass(tmp_path)

    assert out.gh.pr_labels_added == [(_PR, "mergequeue")]
    assert out.data["mergequeue_label_applied"] is True
    assert "human_block" not in out.data  # an un-fired hold leaves the pinned result shape alone
    assert out.gh.comment_reads == 1


def test_changes_requested_blocks_the_hand_off_without_a_comment_read(tmp_path: Path) -> None:
    out = _pass(tmp_path, review_decision=CHANGES_REQUESTED)

    _assert_held(out)
    assert out.result.ok is True
    assert out.data["human_block"] is True
    assert out.data["human_block_source"] == HUMAN_BLOCK_SOURCE_REVIEW_DECISION
    assert "human request-changes outstanding" in out.result.message
    assert out.gh.comment_reads == 0


def test_a_newest_human_request_changes_comment_holds_the_hand_off(tmp_path: Path) -> None:
    out = _pass(tmp_path, comments=[_c(OK_PLAIN), _c(RC_PLAIN)])

    _assert_held(out)
    assert out.data["human_block"] is True
    assert out.data["human_block_source"] == HUMAN_BLOCK_SOURCE_COMMENT
    event = next(e for e in out.state_after["events"] if e["kind"] == "merge_ready")
    assert event["payload"]["human_block"] is True
    assert event["payload"]["human_block_source"] == HUMAN_BLOCK_SOURCE_COMMENT


def test_a_later_human_approve_comment_clears_the_hold(tmp_path: Path) -> None:
    out = _pass(tmp_path, comments=[_c(RC_PLAIN), _c(OK_PLAIN)])

    assert out.gh.pr_labels_added == [(_PR, "mergequeue")]
    assert "human_block" not in out.data


def test_the_hold_self_clears_on_the_next_pass(tmp_path: Path) -> None:
    blocked = _pass(tmp_path, comments=[_c(RC_PLAIN)])
    _assert_held(blocked)

    # Same root (state persisted): the human approves; no operator step needed.
    cleared = _pass(tmp_path, comments=[_c(RC_PLAIN), _c(OK_PLAIN)], approve=False)
    assert cleared.gh.pr_labels_added == [(_PR, "mergequeue")]


def test_a_fleet_approved_marker_comment_does_not_clear_a_human_block(tmp_path: Path) -> None:
    """The 2026-10-19 incident shape: fleet verdict approved, a human's
    request-changes outstanding, the fleet's own marker-bearing approve newest."""
    out = _pass(tmp_path, comments=[_c(RC_PLAIN), _c(OK_MARKER, orchestrator=True)])

    _assert_held(out)


def test_a_marker_bearing_request_changes_comment_is_ignored(tmp_path: Path) -> None:
    out = _pass(tmp_path, comments=[_c(OK_PLAIN), _c(RC_MARKER, orchestrator=True)])

    assert out.gh.pr_labels_added == [(_PR, "mergequeue")]


def test_unreadable_comments_hold_the_pr_fail_closed(tmp_path: Path) -> None:
    out = _pass(tmp_path, unreadable=True)

    _assert_held(out)
    assert out.data["human_block_check_unavailable"] is True
    assert out.result.ok is False
    assert "human-review check unavailable" in out.result.message
    assert out.gh.comment_reads == 1


def test_kill_switch_restores_the_old_behaviour_and_reads_nothing(tmp_path: Path) -> None:
    out = _pass(
        tmp_path,
        comments=[_c(RC_PLAIN)],
        review_decision=CHANGES_REQUESTED,
        config=_config(honor=False),
    )

    assert out.gh.pr_labels_added == [(_PR, "mergequeue")]
    assert out.gh.comment_reads == 0


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param({"approve": False}, id="no-approved-verdict"),
        pytest.param({"merge": False}, id="merge-disabled-for-the-pass"),
    ],
)
def test_no_extra_gh_call_when_no_hand_off_is_imminent(
    tmp_path: Path, scenario: dict[str, Any]
) -> None:
    out = _pass(tmp_path, comments=[_c(RC_PLAIN)], **scenario)

    assert out.gh.comment_reads == 0
    assert out.gh.pr_labels_added == []


def test_a_direct_self_merge_is_held_too(tmp_path: Path) -> None:
    config = _config(mergequeue=None)

    held = _pass(tmp_path / "blocked", comments=[_c(RC_PLAIN)], config=config)
    assert held.gh.merged == []
    assert held.data["merged"] is False
    assert held.data["human_block"] is True

    # Control: the same pass with no block merges.
    merged = _pass(tmp_path / "clear", config=config)
    assert merged.data["merged"] is True
    assert merged.gh.merged != []


def test_the_dry_run_preview_reports_the_hold_and_mutates_nothing(tmp_path: Path) -> None:
    out = _pass(tmp_path, comments=[_c(RC_PLAIN)], dry_run=True)

    assert "human request-changes outstanding" in out.result.message
    assert out.data["human_block"] is True
    assert out.gh.pr_labels_added == []

    control = _pass(tmp_path / "control", comments=[_c(OK_PLAIN)], dry_run=True)
    assert "would hand off" in control.result.message


# ---------------------------------------------------------------------------
# 5. Config surface: ships ON, validated, and copied into the decide slice
# ---------------------------------------------------------------------------


def test_both_new_switches_ship_on() -> None:
    config = OrchestratorConfig()
    assert config.auto_merge.honor_human_block is True
    assert config.review.require_fleet_ownership is True


@pytest.mark.parametrize(
    "key", ["human_block_request_changes_pattern", "human_block_approved_pattern"]
)
def test_a_pattern_that_is_not_a_valid_regex_is_rejected_at_load(key: str) -> None:
    with pytest.raises(ConfigError):
        build_config_from_data({"auto_merge": {key: "["}})
    # Control: the same key with a valid regex loads and is honoured.
    loaded = build_config_from_data({"auto_merge": {key: r"nope"}})
    assert getattr(loaded.auto_merge, key) == r"nope"


@pytest.mark.parametrize("value", ["", 3, True])
def test_a_pattern_must_be_a_nonempty_string(value: object) -> None:
    with pytest.raises(ConfigError):
        build_config_from_data({"auto_merge": {"human_block_approved_pattern": value}})


def test_the_kill_switches_load_from_yaml_data() -> None:
    loaded = build_config_from_data(
        {"auto_merge": {"honor_human_block": False}, "review": {"require_fleet_ownership": False}}
    )
    assert loaded.auto_merge.honor_human_block is False
    assert loaded.review.require_fleet_ownership is False


def test_config_slice_copies_the_switch_and_the_fixture_default_is_off() -> None:
    on = OrchestratorConfig()
    off = dataclasses.replace(
        on, auto_merge=dataclasses.replace(on.auto_merge, honor_human_block=False)
    )
    assert config_slice(on).honor_human_block is True
    assert config_slice(off).honor_human_block is False
    # Fixture-only default: every pre-#2763 decide test builds a MergePathConfig
    # directly and must keep its exact behaviour; the shipped config is ON above.
    assert MergePathConfig().honor_human_block is False
