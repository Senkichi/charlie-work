"""Mergequeue requeue cap (issue #2743): bounded re-add after repeated
same-head queue-branch failures.

A PR whose Aviator queue branch (``mq-bot-*``) fails deterministically used to
be re-labelled forever: Aviator strips ``mergequeue``, the next merge_ready
pass sees the label gone and re-adds it, and the queue CI bill grows without
bound (PR #2649: ~106 queue runs in 13 hours). The cap counts queue reverts
per PR head; at ``auto_merge.mergequeue_requeue_cap`` the hand-off stops being
admissible and the PR routes to rework (or escalates when the rework budget is
spent).

``FakeGitHub.add_pr_label`` only records the call -- it never mutates the
served PR's ``labels`` -- so every pass after the first hand-off observes the
label absent, which is exactly what a real Aviator revert looks like to the
detector.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from _fakes_github import FakeGitHub
from _merge_ready_fixtures import _mergequeue_automerge
from charlie_work.config import DevinConfig, OrchestratorConfig, WorkerRoleConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

_PR = 456
_ISSUE = 123


def _app(tmp_path: Path, cap: int = 3) -> tuple[OrchestratorApp, FakeGitHub, object]:
    """An app whose rework routes land without launching a real worker."""
    config = OrchestratorConfig(
        auto_merge=_mergequeue_automerge(),
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )
    if cap != config.auto_merge.mergequeue_requeue_cap:
        config = replace(
            config,
            auto_merge=replace(config.auto_merge, mergequeue_requeue_cap=cap),
        )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(_PR, "approved", summary="ok", verdict_provenance="fresh_llm_review")
    return app, fake_gh, paths


def _mergequeue_adds(gh: FakeGitHub) -> int:
    return sum(1 for _, label in gh.pr_labels_added if label == "mergequeue")


def _events(paths, kind: str) -> list[dict]:
    return [e for e in load_state(paths.state_file)["events"] if e["kind"] == kind]


def _seed_pr(paths, **fields) -> None:
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"][str(_PR)] = {**state["prs"].get(str(_PR), {}), **fields}
        save_state(paths.state_file, state)


def test_requeue_cap_stops_the_relabel_loop_and_routes_to_rework(tmp_path: Path) -> None:
    """Issue #2743 AC-1/AC-2: after the cap-th same-head queue revert the label
    is not re-added, the distinct event fires, and the PR routes to rework."""
    app, gh, paths = _app(tmp_path)

    for _ in range(3):
        app.merge_ready(_PR, merge=True)
    # Three queue attempts so far: the initial hand-off plus two counted
    # reverts that were still below the cap.
    assert _mergequeue_adds(gh) == 3
    assert load_state(paths.state_file)["prs"][str(_PR)]["consecutive_mergequeue_requeues"] == 2

    # The third observed revert reaches the cap: no fourth label POST.
    capped = app.merge_ready(_PR, merge=True)
    assert _mergequeue_adds(gh) == 3
    assert capped.data["mergequeue_label_applied"] is None
    assert capped.data["mergequeue_requeue_capped"] is True
    assert capped.data["consecutive_mergequeue_requeues"] == 3

    state = load_state(paths.state_file)
    pr_entry = state["prs"][str(_PR)]
    assert pr_entry["mergequeue_requeues_head_sha"] == "sha-abc123"
    assert pr_entry["status"] == "rework_requested"
    assert state["issues"][str(_ISSUE)]["status"] == "rework_requested"

    capped_events = _events(paths, "mergequeue_requeue_capped")
    assert len(capped_events) == 1
    payload = capped_events[0]["payload"]
    assert payload["pr_number"] == _PR
    assert payload["issue_number"] == _ISSUE
    assert payload["head_sha"] == "sha-abc123"
    assert payload["requeues"] == 3
    assert payload["cap"] == 3

    routed_events = _events(paths, "mergequeue_requeue_rework_requested")
    assert len(routed_events) == 1
    assert routed_events[0]["payload"]["issue_number"] == _ISSUE


def test_requeue_cap_is_durable_while_the_head_is_unchanged(tmp_path: Path) -> None:
    """Once capped, later passes must not sneak the label back on even when
    the pass is not itself a revert (e.g. status left 'mergequeue')."""
    app, gh, paths = _app(tmp_path, cap=1)

    app.merge_ready(_PR, merge=True)  # hand-off #1
    capped = app.merge_ready(_PR, merge=True)  # first revert -> capped + routed
    assert capped.data["mergequeue_requeue_capped"] is True
    assert _mergequeue_adds(gh) == 1

    # The routing left status='rework_requested': the next pass sees no revert
    # (prior status is not 'mergequeue') but the cap must still hold.
    again = app.merge_ready(_PR, merge=True)
    assert _mergequeue_adds(gh) == 1
    assert again.data["mergequeue_requeue_capped"] is True
    assert len(_events(paths, "mergequeue_requeue_capped")) == 1


def test_a_new_head_resets_the_requeue_cap(tmp_path: Path) -> None:
    """Issue #2743 AC-3: a pushed head gets a fresh queue budget -- the fix the
    rework worker pushed must be allowed back into the queue."""
    app, gh, paths = _app(tmp_path, cap=1)
    app.merge_ready(_PR, merge=True)
    capped = app.merge_ready(_PR, merge=True)
    assert capped.data["mergequeue_requeue_capped"] is True

    gh.pr_head_shas[_PR] = "sha-fixed"
    moved = app.merge_ready(_PR, merge=True)
    assert moved.data["head_moved"] is True
    app.record_review(
        _PR,
        "approved",
        summary="ok",
        verdict_provenance="fresh_llm_review",
        reviewed_head="sha-fixed",
    )
    requeued = app.merge_ready(_PR, merge=True)
    assert requeued.data["mergequeue_label_applied"] is True
    assert requeued.data["mergequeue_requeue_capped"] is False

    # One revert at the new head counts 1 -- under the cap even at cap=1's
    # sibling head it would have stayed blocked.
    reverted = app.merge_ready(_PR, merge=True)
    pr_entry = load_state(paths.state_file)["prs"][str(_PR)]
    assert reverted.data["mergequeue_requeue_capped"] is True
    assert pr_entry["consecutive_mergequeue_requeues"] == 1
    assert pr_entry["mergequeue_requeues_head_sha"] == "sha-fixed"


def test_a_self_revocation_does_not_count_toward_the_cap(tmp_path: Path) -> None:
    """Issue #2743 AC-4: a mergequeue_revoked_reason reconcile wrote on purpose
    (here: the wedge watchdog) is a self-revocation, not a queue failure."""
    app, gh, paths = _app(tmp_path, cap=1)
    app.merge_ready(_PR, merge=True)
    _seed_pr(paths, mergequeue_revoked_reason="mergequeue_wedged")

    result = app.merge_ready(_PR, merge=True)

    # The revert detector sees the stripped label but classifies it as a
    # self-revocation: the requeue counter must stay at zero and the cap must
    # never engage -- this is also existing handoff_failed behaviour.
    assert result.data["consecutive_mergequeue_requeues"] == 0
    assert result.data["mergequeue_requeue_capped"] is False
    assert _events(paths, "mergequeue_requeue_capped") == []


def test_requeue_cap_zero_disables_the_gate(tmp_path: Path) -> None:
    """The kill switch: cap 0 counts reverts but never gates the hand-off."""
    app, gh, paths = _app(tmp_path, cap=0)
    for _ in range(4):
        app.merge_ready(_PR, merge=True)
    assert _mergequeue_adds(gh) == 4
    pr_entry = load_state(paths.state_file)["prs"][str(_PR)]
    assert pr_entry["consecutive_mergequeue_requeues"] == 3
    assert _events(paths, "mergequeue_requeue_capped") == []


def test_requeue_cap_escalates_when_the_rework_budget_is_spent(tmp_path: Path) -> None:
    """Issue #2743 AC-2: an exhausted mergequeue_rework_attempts budget turns
    the capped PR into an escalation, not another rework dispatch."""
    app, gh, paths = _app(tmp_path, cap=1)
    _seed_pr(
        paths,
        mergequeue_rework_attempts=2,  # == review.max_conflict_rework_attempts
        mergequeue_rework_attempts_last_head="sha-abc123",
    )
    app.merge_ready(_PR, merge=True)

    capped = app.merge_ready(_PR, merge=True)

    assert _mergequeue_adds(gh) == 1
    assert capped.data["mergequeue_requeue_capped"] is True
    state = load_state(paths.state_file)
    assert state["issues"][str(_ISSUE)]["status"] == "escalated"
    assert state["prs"][str(_PR)]["status"] == "escalated"
    escalations = _events(paths, "janitor_rework_escalated")
    assert len(escalations) == 1
    assert escalations[0]["payload"]["reason"] == "mergequeue_requeue"
    assert escalations[0]["payload"]["escalation_reason"] == (
        "mergequeue_rework_attempts_cap_exceeded"
    )


def test_requeue_rework_prompt_carries_the_queue_branch_evidence(tmp_path: Path) -> None:
    """The routed rework tells the worker WHY: the failure lives on Aviator's
    queue branch, not on the PR's own green checks."""
    app, gh, paths = _app(tmp_path, cap=1)
    app.merge_ready(_PR, merge=True)
    app.merge_ready(_PR, merge=True)

    state = load_state(paths.state_file)
    pr_entry = state["prs"][str(_PR)]
    assert pr_entry["status"] == "rework_requested"
    assert "mergequeue_requeue_rework_requested_at" in pr_entry
    prompt_path = paths.prs / f"pr-{_PR}" / "rework-prompt.md"
    prompt = prompt_path.read_text(encoding="utf-8")
    assert "merge queue" in prompt.lower() or "aviator" in prompt.lower()


def test_unescalate_resets_the_requeue_lane(tmp_path: Path) -> None:
    """An operator re-arm after the cap escalation clears the counter, its
    head anchor, and the lane's attempt bookkeeping, so the PR can requeue."""
    app, gh, paths = _app(tmp_path, cap=1)
    _seed_pr(
        paths,
        status="escalated",
        escalation_reason="mergequeue_rework_attempts_cap_exceeded",
        consecutive_mergequeue_requeues=3,
        mergequeue_requeues_head_sha="sha-abc123",
        mergequeue_rework_attempts=3,
        mergequeue_rework_attempts_last_head="sha-abc123",
        mergequeue_rework_attempts_stall_since="2026-01-01T00:00:00Z",
        mergequeue_rework_attempts_stall_head="sha-abc123",
    )
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"][str(_ISSUE)] = {"number": _ISSUE, "status": "escalated"}
        save_state(paths.state_file, state)

    result = app.unescalate(pr_number=_PR)
    assert result.ok is True

    pr_entry = load_state(paths.state_file)["prs"][str(_PR)]
    for field in (
        "consecutive_mergequeue_requeues",
        "mergequeue_requeues_head_sha",
        "mergequeue_rework_attempts",
        "mergequeue_rework_attempts_last_head",
        "mergequeue_rework_attempts_stall_since",
        "mergequeue_rework_attempts_stall_head",
    ):
        assert pr_entry.get(field) in (None, 0), field
