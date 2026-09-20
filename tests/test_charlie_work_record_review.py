"""Recorded review outcomes: verdict transitions, head-sha capture, persistence.

Split out of ``tests/test_charlie_work.py`` (issue #1549, Track-1 wave 3/8):
the recorded-review-outcome seam -- label/status transitions on each verdict, reviewed-head-sha capture, session metrics, and decision-file persistence. Shared fakes and helpers in ``tests/_review_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig, ReviewConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_record_review_approved_transitions_labels(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # review_approved clears reviewing/needs-rework so the issue isn't stuck.
    assert (123, "agent:reviewing") in fake_gh.labels_removed
    assert load_state(paths.state_file)["prs"]["456"]["status"] == "approved"


def test_record_review_transition_failure_recorded(tmp_path: Path) -> None:
    """Issue #135: PARTIAL_FAILURE during record_review transition must be recorded."""
    from charlie_work.labels import TransitionOutcome

    config = OrchestratorConfig()

    class LabelFailGitHub(FakeGitHub):
        def remove_issue_label(self, number: int, label: str) -> bool:
            # Return False to simulate remove failure (error-as-value)
            return False

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review"
    )

    assert result.ok is True
    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "review_approved"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert len(label_error["remove_failures"]) > 0


def test_record_review_request_changes_transition_failure_recorded(tmp_path: Path) -> None:
    """Issue #135: PARTIAL_FAILURE during record_review request_changes transition must be recorded."""
    from charlie_work.labels import TransitionOutcome

    config = OrchestratorConfig()

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            # Return False to simulate add failure (error-as-value)
            return False

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456, "request_changes", summary="fix it", verdict_provenance="fresh_llm_review"
    )

    assert result.ok is True
    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "rework_requested"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert len(label_error["add_failures"]) > 0


def test_record_review_blocked_transitions_to_human_needed(tmp_path: Path) -> None:
    """Issue #1266 behavior preservation: "blocked" is judgment-only by
    construction (a reviewer-flagged security/product concern is never
    mechanical), so record_review's "blocked" decision must keep routing to
    agent:human-needed unchanged -- it has no operator-queue counterpart in
    _MECHANICAL_ESCALATION_EDGES. Representative judgment site #2 (site #1 is
    test_cross_family_regen_reachability.py's
    test_the_escalating_pass_does_not_strip_the_human_needed_label)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456, "blocked", summary="security concern", verdict_provenance="fresh_llm_review"
    )

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["reason_class"] == "judgment"
    assert (123, config.labels.human_needed) in fake_gh.labels_added
    assert (123, config.labels.operator_queue) not in fake_gh.labels_added


def test_record_review_blocked_transition_failure_recorded(tmp_path: Path) -> None:
    """Issue #135: PARTIAL_FAILURE during record_review blocked transition must be recorded."""
    from charlie_work.labels import TransitionOutcome

    config = OrchestratorConfig()

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            # Return False to simulate add failure (error-as-value)
            return False

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456, "blocked", summary="security issue", verdict_provenance="fresh_llm_review"
    )

    assert result.ok is True
    label_error = result.data["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "blocked"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value
    assert len(label_error["add_failures"]) > 0


# --- Issue #31: approvals pinned to PR head SHA --------------------------------


def test_record_review_captures_reviewed_head_sha(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.record_review(
        456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review"
    )

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["reviewed_head_sha"] == "sha-abc123"
    assert decision["reviewed_head_source"] == "live"
    assert load_state(paths.state_file)["prs"]["456"]["reviewed_head_sha"] == "sha-abc123"
    assert result.data["reviewed_head_sha"] == "sha-abc123"
    assert result.data["reviewed_head_source"] == "live"


def test_record_review_requires_explicit_head_when_packet_and_live_differ(
    tmp_path: Path,
) -> None:
    """Issue #467: when the packet head and live PR head differ, record_review
    must refuse to silently choose a source and must record provenance.

    A commit landing between review() (packet generation) and record_review()
    (verdict recording) now requires an explicit --reviewed-head choice.
    Selecting the packet head preserves the original packet SHA/diff; selecting
    the live head records the new SHA/diff and provenance.
    """
    from charlie_work.janitor import _calculate_patch_id

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = "diff --git a/file b/file\n+packet diff"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    review_result = app.review(456)
    assert review_result.ok is True
    packet_patch_id = _calculate_patch_id(fake_gh.diffs[456])

    # Simulate a new commit landing after the packet was generated.
    fake_gh.pr_head_shas[456] = "sha-new789"
    fake_gh.diffs[456] = "diff --git a/file b/file\n+unreviewed change"
    live_patch_id = _calculate_patch_id(fake_gh.diffs[456])

    decision_path = paths.prs / "pr-456" / "review-decision.json"

    # Without an explicit choice, the verdict must fail loudly and must not
    # overwrite the pending decision file written by review().
    result = app.record_review(
        456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review"
    )
    assert result.ok is False
    assert "sha-abc123" in result.message
    assert "sha-new789" in result.message
    assert "--reviewed-head" in result.message
    assert json.loads(decision_path.read_text(encoding="utf-8")).get("decision") == "pending"

    # Choosing the original packet head records the packet SHA, patch, and source.
    # Issue #1072: allow_stale_head=True is the operator CLI's explicit exemption
    # — a human deliberately choosing to pin a verdict to a superseded head.
    # Automated callers (review() exits, dispatch_reviews) use the default False
    # and are refused by record_review()'s compare-and-swap guard.
    result = app.record_review(
        456,
        "approved",
        summary="lgtm",
        reviewed_head="sha-abc123",
        allow_stale_head=True,
        verdict_provenance="operator_manual",
    )
    assert result.ok is True
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["reviewed_head_sha"] == "sha-abc123"
    assert decision["reviewed_head_source"] == "packet"
    assert decision["reviewed_patch_id"] == packet_patch_id
    assert result.data["reviewed_head_source"] == "packet"
    assert load_state(paths.state_file)["prs"]["456"]["reviewed_head_sha"] == "sha-abc123"

    # Choosing the live head records the live SHA, live patch, and source.
    decision_path.unlink()
    state = load_state(paths.state_file)
    state["prs"].pop("456", None)
    save_state(paths.state_file, state)

    result = app.record_review(
        456,
        "approved",
        summary="lgtm",
        reviewed_head="sha-new789",
        verdict_provenance="fresh_llm_review",
    )
    assert result.ok is True
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["reviewed_head_sha"] == "sha-new789"
    assert decision["reviewed_head_source"] == "live"
    assert decision["reviewed_patch_id"] == live_patch_id
    assert result.data["reviewed_head_source"] == "live"


def test_record_review_blocked_persists_reviewed_patch_id(tmp_path: Path) -> None:
    """Issue #413: blocked decisions must persist reviewed_patch_id so the
    review-queue enumerator can carry them forward on content-identical heads."""
    from charlie_work.janitor import _calculate_patch_id

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 blocked\n"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    review_result = app.review(456)
    assert review_result.ok is True
    packet_patch_id = _calculate_patch_id(fake_gh.diffs[456])

    result = app.record_review(
        456, "blocked", summary="security concern", verdict_provenance="fresh_llm_review"
    )

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "blocked"
    assert decision["reviewed_patch_id"] == packet_patch_id
    assert load_state(paths.state_file)["prs"]["456"]["reviewed_patch_id"] == packet_patch_id
    assert result.data["reviewed_head_sha"] == "sha-abc123"


def test_record_review_request_changes_updates_issue_status_to_rework_requested(
    tmp_path: Path,
) -> None:
    """Issue #72: request_changes (non-escalated) updates issue status to rework_requested
    so dispatch_rework can select it."""
    from charlie_work.issue_linking import linked_issue_number

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Verify that linked_issue_number returns the correct issue number
    issue_number = linked_issue_number(
        fake_gh.prs[0],
        is_cross_repository=fake_gh.prs[0].get("isCrossRepository"),
        branch_prefix=config.dispatch.branch_prefix,
    )
    assert issue_number == 123

    # Record a non-escalated request_changes decision
    result = app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    assert result.ok is True
    assert result.data["escalated"] is False

    # Assert the actual state change: issue status should be rework_requested
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"


def test_record_review_persists_escalated_in_decision_file(tmp_path: Path) -> None:
    """Issue #407: review-decision.json must include the correct escalated value.

    The decision payload is fully built before the single atomic write, so
    re-reading the persisted file returns the same escalated flag as the
    in-memory result. Non-escalated request_changes and escalated
    request_changes must both persist the correct value.
    """
    config = OrchestratorConfig(review=ReviewConfig(max_rework_cycles=2))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    decision_path = paths.prs / "pr-456" / "review-decision.json"

    fake_gh.pr_head_shas[456] = "sha-1"
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert "escalated" in decision
    assert decision["escalated"] is False
    assert app._review_decision(456)["escalated"] is False

    fake_gh.pr_head_shas[456] = "sha-2"
    app.record_review(
        456, "request_changes", summary="fix B", verdict_provenance="fresh_llm_review"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["escalated"] is False

    fake_gh.pr_head_shas[456] = "sha-3"
    app.record_review(
        456, "request_changes", summary="fix C", verdict_provenance="fresh_llm_review"
    )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["escalated"] is True
    # _review_decision is the reader used by merge_ready and merge-train
    # eligibility; it must see the persisted escalated value.
    assert app._review_decision(456)["escalated"] is True


def test_record_review_session_metrics_none_preserves_prior_metrics(tmp_path: Path) -> None:
    """A manual `charlie verdict` call (cli.py's record_review invocation shape)
    passes session_metrics=None -- this must never clobber metrics recorded by
    an earlier automated reap (the merge-update guard in record_review)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    prior_metrics = {
        "tokens": 500,
        "cost_usd": 0.5,
        "turn_count": 2,
        "tool_call_count": 1,
        "verdict_source": "log",
    }
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            **state["prs"].get("456", {}),
            "review_session_metrics": prior_metrics,
        }
        save_state(paths.state_file, state)

    result = app.record_review(
        456,
        "approved",
        summary="lgtm",
        session_metrics=None,
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["review_session_metrics"] == prior_metrics


def test_record_review_session_metrics_replaces_prior_metrics(tmp_path: Path) -> None:
    """A fresh non-None session_metrics call (the automated reap shape) DOES
    replace whatever metrics were recorded previously."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    prior_metrics = {
        "tokens": 500,
        "cost_usd": 0.5,
        "turn_count": 2,
        "tool_call_count": 1,
        "verdict_source": "log",
    }
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"]["456"] = {
            **state["prs"].get("456", {}),
            "review_session_metrics": prior_metrics,
        }
        save_state(paths.state_file, state)

    new_metrics = {
        "tokens": 900,
        "cost_usd": 0.9,
        "turn_count": 3,
        "tool_call_count": 2,
        "verdict_source": "events",
    }
    result = app.record_review(
        456,
        "approved",
        summary="lgtm",
        session_metrics=new_metrics,
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["review_session_metrics"] == new_metrics


def test_record_review_persists_required_changes(tmp_path: Path) -> None:
    """Issue #507: record_review accepts and persists required_changes."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.record_review(
        456,
        "request_changes",
        summary="fix A",
        required_changes=["add null check", "update tests"],
        verdict_provenance="fresh_llm_review",
    )

    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["required_changes"] == ["add null check", "update tests"]
