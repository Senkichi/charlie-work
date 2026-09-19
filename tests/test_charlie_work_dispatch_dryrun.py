"""Dry-run dispatch: rework passes must leave state and side effects untouched.

Split out of ``tests/test_charlie_work.py`` (issue #1547, Track-1 wave 1/8):
the ``test_dry_run_dispatch_rework_*`` seam -- dry-run passes leave state,
escalations, review routing, and conflict bypass untouched. Shared fakes
and helpers in ``tests/_rework_dispatch_fixtures.py``.
"""

from __future__ import annotations

import sys
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp


def test_dry_run_dispatch_rework_leaves_state_unchanged(tmp_path: Path) -> None:
    """Issue #616: --dry-run rework dispatch must not advance the state machine
    off fabricated adapter results.

    Without the dry-run short-circuit in _dispatch_rework_impl, the fabricated
    _dry_run_result objects (ok=True for every request) are consumed as ground
    truth: the issue is marked "dispatched" with a real dispatched_at,
    redispatch_at counters advance, orphan flags clear, and at the redispatch
    cap the issue is auto-escalated with escalation_reason="redispatch_cap_exceeded"
    — all on zero actual work. The no-op-rework escalation path also writes
    state + transitions GitHub labels, and the review-routing path calls
    self.review() which writes state; all must be skipped under dry-run.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    # The result should indicate a dry-run planning outcome, not a real dispatch.
    assert result.ok is True
    assert "dry-run" in result.message.lower()
    assert result.data["selected_count"] == 1
    assert len(result.data["sessions"]) == 1
    assert result.data["sessions"][0]["issue_number"] == 123
    assert result.data["dispatch_results"] == []

    # State must be unchanged: no status transition, no dispatched_at, no
    # redispatch_at counter, no events.
    with state_lock(paths.state_file):
        final_state = load_state(paths.state_file)
    issue_entry = final_state["issues"]["123"]
    assert issue_entry["status"] == "rework_requested", (
        "dry-run must not advance the state machine to 'dispatched'"
    )
    assert issue_entry.get("dispatched_at") is None, "dry-run must not stamp a real dispatched_at"
    assert "redispatch_at" not in issue_entry, "dry-run must not advance the redispatch counter"
    assert final_state["events"] == [], "dry-run must not record dispatch_rework events"

    # No GitHub label transitions should have been attempted.
    assert fake_gh.labels_added == [], "dry-run must not add GitHub labels"
    assert fake_gh.labels_removed == [], "dry-run must not remove GitHub labels"


def test_dry_run_dispatch_rework_no_op_escalation_does_not_escalate(
    tmp_path: Path,
) -> None:
    """Issue #616: under --dry-run, a rework issue at the redispatch cap must
    NOT be escalated. The no-op-rework escalation path writes state and
    transitions GitHub labels; the dry-run short-circuit must skip it and
    instead report the issue as *would-be* escalated in the planning data.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    # Seed state with the issue at the redispatch cap: head unchanged,
    # redispatch_at already at max_auto_redispatch, so the real dispatch
    # would escalate it as no_op_rework_cap_exceeded.
    now = datetime.now(UTC)
    redispatch_ts = [
        (now - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
    ]
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": redispatch_ts,
        }
        state["prs"]["456"] = {
            "number": 456,
            "status": "needs_rework",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is True
    assert "dry-run" in result.message.lower()
    # The issue should be reported as would-be escalated, not dispatched.
    assert 123 in result.data["no_op_rework_escalated"]
    assert result.data["selected_count"] == 0

    # State must be unchanged: still rework_requested, NOT escalated.
    with state_lock(paths.state_file):
        final_state = load_state(paths.state_file)
    assert final_state["issues"]["123"]["status"] == "rework_requested", (
        "dry-run must not escalate the issue to 'escalated'"
    )
    assert "escalation_reason" not in final_state["issues"]["123"], (
        "dry-run must not stamp an escalation_reason"
    )
    assert final_state["events"] == [], "dry-run must not record escalation events"
    assert fake_gh.labels_added == [], "dry-run must not add GitHub labels"
    assert fake_gh.labels_removed == [], "dry-run must not remove GitHub labels"


def test_dry_run_dispatch_rework_review_routing_does_not_invoke_review(
    tmp_path: Path,
) -> None:
    """Issue #616: under --dry-run, a rework candidate whose PR head has
    diverged from reviewed_head_sha with a genuine content change must be
    classified into routed_to_review (or skipped_head_indeterminate if the
    patch-id can't be established), but the dry-run short-circuit must NOT
    invoke self.review()/_route_rework_candidate_to_review — both write state
    and transition GitHub labels. State.json and GitHub labels must be
    unchanged.

    Mirrors test_dispatch_rework_routes_to_review_instead_of_relaunch_when_head_moved
    but under dry-run: the real (non-dry-run) path routes the issue to the
    review lane (writing state + relabeling); the dry-run path must only
    *report* the would-be routing without performing it.
    """
    from charlie_work.janitor import _calculate_patch_id

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    fake_gh = ReworkGitHub()

    # Record a request_changes baseline: reviewed_head_sha pins the
    # pre-rework head, reviewed_patch_id pins the pre-rework patch content.
    reviewed_diff = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+first"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    assert reviewed_patch_id, "fixture must produce a non-empty reviewed_patch_id"

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        state["prs"]["456"] = {
            "number": 456,
            "status": "needs_rework",
            "reviewed_head_sha": "sha-abc123",
            "reviewed_patch_id": reviewed_patch_id,
        }
        save_state(paths.state_file, state)

    # Simulate the rework already having been pushed: head advances AND the
    # diff content genuinely changes (different patch-id, not just a
    # sync-merge). This is exactly the condition that makes the real dispatch
    # route to review.
    live_diff = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+second"
    live_patch_id = _calculate_patch_id(live_diff)
    assert live_patch_id != reviewed_patch_id, (
        "fixture must reproduce a genuine content change (distinct patch-ids)"
    )
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = live_diff

    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # Spy on _route_rework_candidate_to_review to prove the dry-run path
    # never invokes it. Any call would write state and transition GitHub
    # labels — the exact mutation dry-run must avoid.
    routing_calls: list[tuple[int, int, str | None]] = []

    def _spy_route_rework_candidate_to_review(
        issue_number: int, pr_number: int, reviewed_head_sha_before: str | None
    ):
        routing_calls.append((issue_number, pr_number, reviewed_head_sha_before))
        raise AssertionError(
            "dry-run must not invoke _route_rework_candidate_to_review "
            f"(called for issue {issue_number}, pr {pr_number})"
        )

    app._route_rework_candidate_to_review = _spy_route_rework_candidate_to_review  # type: ignore[method-assign]

    # A rework prompt exists so the candidate is otherwise dispatch-eligible.
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is True
    assert "dry-run" in result.message.lower()
    # The issue is classified as a would-be review routing (genuine content
    # change, patch-ids differ), NOT dispatched.
    assert 123 in result.data["routed_to_review"], (
        "dry-run must report the would-be review routing"
    )
    assert result.data["selected_count"] == 0
    assert result.data["sessions"] == []

    # The routing helper was never invoked — dry-run only *reports* the
    # classification; it does not perform the routing.
    assert routing_calls == [], "dry-run must not invoke _route_rework_candidate_to_review"

    # State must be unchanged: still rework_requested, no review transition,
    # no events.
    with state_lock(paths.state_file):
        final_state = load_state(paths.state_file)
    assert final_state["issues"]["123"]["status"] == "rework_requested", (
        "dry-run must not transition the issue to 'reviewing'"
    )
    assert final_state["prs"]["456"]["reviewed_head_sha"] == "sha-abc123", (
        "dry-run must not advance reviewed_head_sha"
    )
    assert final_state["events"] == [], "dry-run must not record review-routing events"

    # No GitHub label transitions should have been attempted.
    assert fake_gh.labels_added == [], "dry-run must not add GitHub labels"
    assert fake_gh.labels_removed == [], "dry-run must not remove GitHub labels"


def test_dry_run_dispatch_rework_worker_death_escalation_does_not_escalate(
    tmp_path: Path,
) -> None:
    """Issue #616: under --dry-run, a rework issue whose worker_death_at
    entries have reached max_auto_redispatch (with an unchanged head) must
    NOT be escalated. The worker-death escalation path writes state and
    transitions GitHub labels; the dry-run short-circuit must skip it and
    instead report the issue as *would-be* escalated in the planning data.

    Mirrors test_dispatch_rework_worker_deaths_dont_count_as_no_op but under
    dry-run: the real (non-dry-run) path escalates with
    escalation_reason="worker_death_loop"; the dry-run path must only
    *report* the would-be escalation without performing it.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    # Seed state with the issue at the worker-death cap: head unchanged,
    # redispatch_at and worker_death_at both at max_auto_redispatch. The
    # no-op count (redispatch - deaths) is 0, so it does NOT escalate as
    # no_op_rework_cap_exceeded; the death count itself hits the cap, so the
    # real dispatch would escalate as worker_death_loop.
    now = datetime.now(UTC)
    ts = [
        (now - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
    ]
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": ts,
            "worker_death_at": ts,
        }
        state["prs"]["456"] = {
            "number": 456,
            "status": "needs_rework",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is True
    assert "dry-run" in result.message.lower()
    # The issue should be reported as would-be worker-death escalated, not
    # dispatched, and NOT as no-op escalated (deaths are not no-ops).
    assert 123 in result.data["worker_death_escalated"], (
        "dry-run must report the would-be worker-death escalation"
    )
    assert 123 not in result.data.get("no_op_rework_escalated", []), (
        "deaths must not be classified as no-op escalations"
    )
    assert result.data["selected_count"] == 0

    # State must be unchanged: still rework_requested, NOT escalated.
    with state_lock(paths.state_file):
        final_state = load_state(paths.state_file)
    assert final_state["issues"]["123"]["status"] == "rework_requested", (
        "dry-run must not escalate the issue to 'escalated'"
    )
    assert "escalation_reason" not in final_state["issues"]["123"], (
        "dry-run must not stamp an escalation_reason"
    )
    assert final_state["events"] == [], "dry-run must not record escalation events"
    assert fake_gh.labels_added == [], "dry-run must not add GitHub labels"
    assert fake_gh.labels_removed == [], "dry-run must not remove GitHub labels"


def test_dry_run_dispatch_rework_conflict_bypass_direct_conflicting(
    tmp_path: Path,
) -> None:
    """Issue #1349 dry-run mirror: a rework candidate whose PR head moved with
    a real content change AND whose PR is directly CONFLICTING/DIRTY in
    ``pr_list`` must be classified as a launch candidate (not routed_to_review)
    even under --dry-run. The dry-run path must only *report* the
    classification without performing the dispatch or the review routing.
    """
    from charlie_work.janitor import _calculate_patch_id

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    fake_gh = ReworkGitHub()

    reviewed_diff = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+first"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    assert reviewed_patch_id, "fixture must produce a non-empty reviewed_patch_id"

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        state["prs"]["456"] = {
            "number": 456,
            "status": "needs_rework",
            "reviewed_head_sha": "sha-abc123",
            "reviewed_patch_id": reviewed_patch_id,
        }
        save_state(paths.state_file, state)

    live_diff = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+second"
    live_patch_id = _calculate_patch_id(live_diff)
    assert live_patch_id != reviewed_patch_id, (
        "fixture must reproduce a genuine content change (distinct patch-ids)"
    )
    # Head advances with a real content change AND the PR is directly
    # CONFLICTING/DIRTY in pr_list -- the conflict-bypass applies.
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.prs[0]["mergeable"] = "CONFLICTING"
    fake_gh.prs[0]["mergeStateStatus"] = "DIRTY"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = live_diff

    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    routing_calls: list[tuple[int, int, str | None]] = []

    def _spy_route_rework_candidate_to_review(
        issue_number: int, pr_number: int, reviewed_head_sha_before: str | None
    ):
        routing_calls.append((issue_number, pr_number, reviewed_head_sha_before))
        raise AssertionError(
            "dry-run must not invoke _route_rework_candidate_to_review "
            f"(called for issue {issue_number}, pr {pr_number})"
        )

    app._route_rework_candidate_to_review = _spy_route_rework_candidate_to_review  # type: ignore[method-assign]

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is True
    assert "dry-run" in result.message.lower()
    # The conflict-bypass keeps the issue as a launch candidate, NOT routed
    # to the review lane.
    assert result.data["routed_to_review"] == [], (
        "dry-run must not report a conflict-bypassed candidate as routed_to_review"
    )
    assert result.data["selected_count"] == 1, (
        "dry-run must report the conflict-bypassed candidate as dispatch-eligible"
    )
    assert result.data["sessions"][0]["issue_number"] == 123

    # The routing helper was never invoked.
    assert routing_calls == [], "dry-run must not invoke _route_rework_candidate_to_review"

    # State must be unchanged.
    with state_lock(paths.state_file):
        final_state = load_state(paths.state_file)
    assert final_state["issues"]["123"]["status"] == "rework_requested"
    assert final_state["prs"]["456"]["reviewed_head_sha"] == "sha-abc123"
    assert final_state["events"] == []

    assert fake_gh.labels_added == [], "dry-run must not add GitHub labels"
    assert fake_gh.labels_removed == [], "dry-run must not remove GitHub labels"


def test_dry_run_dispatch_rework_conflict_bypass_unknown_mergeable_pr_view_fallback(
    tmp_path: Path,
) -> None:
    """Issue #1349 dry-run mirror: when ``pr_list``'s ``mergeable`` is UNKNOWN
    (indeterminate) and a fresh ``pr_view`` reveals the conflict, the dry-run
    path must classify the candidate as a launch candidate (not
    routed_to_review) via the pr_view fallback. The dry-run path must only
    *report* the classification without performing the dispatch or the review
    routing.
    """
    from charlie_work.janitor import _calculate_patch_id

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class PrViewConflictingGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

        def pr_view(self, number: int):
            pr_copy = dict(super().pr_view(number))
            # pr_list reports UNKNOWN; the authoritative pr_view reveals
            # the conflict.
            pr_copy["mergeable"] = "CONFLICTING"
            pr_copy["mergeStateStatus"] = "DIRTY"
            return pr_copy

    fake_gh = PrViewConflictingGitHub()

    reviewed_diff = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+first"
    )
    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    assert reviewed_patch_id, "fixture must produce a non-empty reviewed_patch_id"

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        state["prs"]["456"] = {
            "number": 456,
            "status": "needs_rework",
            "reviewed_head_sha": "sha-abc123",
            "reviewed_patch_id": reviewed_patch_id,
        }
        save_state(paths.state_file, state)

    live_diff = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+second"
    live_patch_id = _calculate_patch_id(live_diff)
    assert live_patch_id != reviewed_patch_id, (
        "fixture must reproduce a genuine content change (distinct patch-ids)"
    )
    # pr_list's mergeable is UNKNOWN (not a definite CONFLICTING/MERGEABLE)
    # and mergeStateStatus is CLEAN, so the pr_view fallback path is taken.
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.prs[0]["mergeable"] = "UNKNOWN"
    fake_gh.prs[0]["mergeStateStatus"] = "CLEAN"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = live_diff

    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    routing_calls: list[tuple[int, int, str | None]] = []

    def _spy_route_rework_candidate_to_review(
        issue_number: int, pr_number: int, reviewed_head_sha_before: str | None
    ):
        routing_calls.append((issue_number, pr_number, reviewed_head_sha_before))
        raise AssertionError(
            "dry-run must not invoke _route_rework_candidate_to_review "
            f"(called for issue {issue_number}, pr {pr_number})"
        )

    app._route_rework_candidate_to_review = _spy_route_rework_candidate_to_review  # type: ignore[method-assign]

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    result = app.dispatch_rework()

    assert result.ok is True
    assert "dry-run" in result.message.lower()
    # The pr_view fallback detects the conflict, so the issue is kept as a
    # launch candidate, NOT routed to the review lane.
    assert result.data["routed_to_review"] == [], (
        "dry-run must not report a pr_view-fallback conflict-bypassed "
        "candidate as routed_to_review"
    )
    assert result.data["selected_count"] == 1, (
        "dry-run must report the pr_view-fallback conflict-bypassed candidate as dispatch-eligible"
    )
    assert result.data["sessions"][0]["issue_number"] == 123

    assert routing_calls == [], "dry-run must not invoke _route_rework_candidate_to_review"

    with state_lock(paths.state_file):
        final_state = load_state(paths.state_file)
    assert final_state["issues"]["123"]["status"] == "rework_requested"
    assert final_state["prs"]["456"]["reviewed_head_sha"] == "sha-abc123"
    assert final_state["events"] == []

    assert fake_gh.labels_added == [], "dry-run must not add GitHub labels"
    assert fake_gh.labels_removed == [], "dry-run must not remove GitHub labels"
