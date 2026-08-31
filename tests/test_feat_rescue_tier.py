"""Tests for the rescue tier (issue #555): one bounded strong-model rework +
cross-family review attempt inserted between "cheap-worker cap exhausted"
and escalating to a human.

Covers the three eligible interception sites (record_review's
max_rework_cycles cap; the shared _route_janitor_gate_failure_to_rework
escalation branch for max_conflict_rework_attempts/max_no_op_rework_attempts),
the ineligible sites (which must be untouched regardless of rescue.enabled),
the durable rescue_attempted marker (one rescue per PR, cleared only by
`charlie unescalate`), the rescue rework dispatch's adapter/model override
(_dispatch_rework_impl / _rescue_adapter_settings), and the rescue review's
exit semantics (_process_rescue_review).

Reuses ``FakeGitHub`` from test_charlie_work.py (PR #456 <-> issue #123),
mirroring test_fix_escalation_paths.py and test_fix_janitor_routing.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work.config import (
    OrchestratorConfig,
    RescueConfig,
    ReviewConfig,
    WorkerRoleConfig,
)
from charlie_work.rescue_review import CrossFamilyResult
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from _fakes_github import FakeGitHub


def _events(state, kind: str) -> list[dict]:
    return [e for e in state.get("events", []) if e.get("kind") == kind]


def _seed_pr_state(paths, pr_number: int, issue_number: int, **fields) -> None:
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": issue_number,
            **fields,
        }
        save_state(paths.state_file, state)


def _set_decision(app: OrchestratorApp, pr_number: int, decision: str) -> None:
    pr_dir = app.paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": decision}), encoding="utf-8"
    )


def _write_review_packet(paths, pr_number: int, head_sha: str) -> None:
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        f'{{"number": {pr_number}, "headRefOid": "{head_sha}"}}', encoding="utf-8"
    )
    (pr_dir / "review-prompt.md").write_text("review prompt", encoding="utf-8")


# --- Site 1a: record_review's max_rework_cycles cap ---


def test_rework_cycle_cap_dispatches_rescue_instead_of_escalating(tmp_path: Path) -> None:
    config = OrchestratorConfig(rescue=RescueConfig(enabled=True, worker_model="claude-opus-4-1"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _seed_pr_state(paths, 456, 123, request_changes_count=config.review.max_rework_cycles)

    result = app.record_review(
        456, "request_changes", summary="needs another pass", verdict_provenance="fresh_llm_review"
    )

    assert result.ok is True
    assert result.data["escalated"] is False
    assert result.data["rescue_dispatched"] is True

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "request_changes"
    assert state["prs"]["456"]["rescue_attempted"] is True
    assert state["prs"]["456"]["rescue_cause"] == "rework_cycle_cap"
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert (123, config.labels.needs_rework) in fake_gh.labels_added
    assert (123, config.labels.human_needed) not in fake_gh.labels_added

    rescue_events = _events(state, "rescue_dispatched")
    assert len(rescue_events) == 1
    assert rescue_events[0]["payload"] == {
        "pr_number": 456,
        "issue_number": 123,
        "cause": "rework_cycle_cap",
    }

    rework_prompt = (paths.prs / "pr-456" / "rework-prompt.md").read_text(encoding="utf-8")
    assert "Rescue-tier rework" in rework_prompt
    assert "needs another pass" in rework_prompt


def test_rework_cycle_cap_with_rescue_already_attempted_escalates_normally(
    tmp_path: Path,
) -> None:
    """One rescue per PR: a PR that already spent its rescue attempt must
    escalate exactly as it would with rescue disabled. Issue #1266:
    max_rework_cycles_exceeded is a mechanical reason, so it lands
    agent:operator-queue, not agent:human-needed."""
    config = OrchestratorConfig(rescue=RescueConfig(enabled=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _seed_pr_state(
        paths,
        456,
        123,
        request_changes_count=config.review.max_rework_cycles,
        rescue_attempted=True,
        rescue_cause="rework_cycle_cap",
    )

    result = app.record_review(
        456, "request_changes", summary="still broken", verdict_provenance="fresh_llm_review"
    )

    assert result.data["escalated"] is True
    assert result.data["rescue_dispatched"] is False
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["issues"]["123"]["status"] == "escalated"
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
    assert _events(state, "rescue_dispatched") == []


def test_rework_cycle_cap_disabled_config_matches_legacy_escalation(tmp_path: Path) -> None:
    """rescue.enabled defaults False: an absent/disabled config block must
    reproduce byte-for-byte the pre-rescue escalation behavior. Issue
    #1266: max_rework_cycles_exceeded is a mechanical reason, so it lands
    agent:operator-queue, not agent:human-needed."""
    config = OrchestratorConfig()  # rescue.enabled defaults False
    assert config.rescue.enabled is False
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _seed_pr_state(paths, 456, 123, request_changes_count=config.review.max_rework_cycles)

    result = app.record_review(
        456, "request_changes", summary="needs another pass", verdict_provenance="fresh_llm_review"
    )

    assert result.data["escalated"] is True
    assert result.data["rescue_dispatched"] is False
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["issues"]["123"]["status"] == "escalated"
    assert "rescue_attempted" not in state["prs"]["456"]
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


# --- Site 1b/1c: _route_janitor_gate_failure_to_rework (conflict/no-op caps) ---


def _conflicting_app(tmp_path: Path, **config_kwargs) -> OrchestratorApp:
    config = OrchestratorConfig(**config_kwargs)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["mergeable"] = "CONFLICTING"
    fake_gh.prs[0]["mergeStateStatus"] = "DIRTY"
    return OrchestratorApp(tmp_path, paths, config, fake_gh)


def test_conflict_rework_cap_dispatches_rescue_instead_of_escalating(tmp_path: Path) -> None:
    app = _conflicting_app(
        tmp_path,
        review=ReviewConfig(max_conflict_rework_attempts=1),
        rescue=RescueConfig(enabled=True),
    )
    _set_decision(app, 456, "request_changes")

    # First cycle: routes to rework normally (attempts=1, not yet over the cap).
    result1 = app.review(456)
    assert result1.data["routed_to_rework"] is True
    assert result1.data.get("rescue_dispatched") is not True

    # Second cycle (new head, still conflicting): attempts=2 exceeds
    # max_conflict_rework_attempts=1 -- rescue-eligible, dispatched instead
    # of escalating.
    app.gh.pr_head_shas[456] = "sha-cycle-2"
    result2 = app.review(456)

    assert result2.ok is True
    assert result2.data["routed_to_rework"] is True
    assert result2.data["rescue_dispatched"] is True

    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["conflict_rework_attempts"] == 2
    assert state["prs"]["456"]["rescue_attempted"] is True
    assert state["prs"]["456"]["rescue_cause"] == "merge_conflict"
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert (123, app.config.labels.human_needed) not in app.gh.labels_added

    rescue_events = _events(state, "rescue_dispatched")
    assert len(rescue_events) == 1
    assert rescue_events[0]["payload"]["cause"] == "merge_conflict"
    assert _events(state, "janitor_rework_escalated") == []


def test_review_dispatch_attempt_cap_ineligible_still_escalates_with_rescue_enabled(
    tmp_path: Path,
) -> None:
    """Infra-driven caps are NOT gated by cause -- review-dispatch attempt
    cap must escalate straight to human even when rescue.enabled is True."""
    from charlie_work.config import ReviewDispatchConfig

    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True, max_review_dispatch_attempts=2),
        rescue=RescueConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        '{"number": 456, "headRefOid": "sha-abc123"}', encoding="utf-8"
    )
    (pr_dir / "review-prompt.md").write_text("review prompt", encoding="utf-8")
    _seed_pr_state(paths, 456, 123, review_dispatch_attempt_count=2)

    result = app.dispatch_reviews()

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["issues"]["123"]["status"] == "escalated"
    assert "rescue_attempted" not in state["prs"]["456"]
    # Issue #1266: max_review_dispatch_attempts_exceeded is a mechanical
    # reason (an attempt-cap limit), so it lands agent:operator-queue,
    # not agent:human-needed -- "escalates straight to human" here means
    # "skips the rescue tier", not "lands the judgment label".
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
    assert len(_events(state, "review_dispatch_escalated")) == 1
    assert _events(state, "rescue_dispatched") == []


# --- Rescue rework dispatch: adapter/model override reuses the existing path ---


def test_dispatch_rework_routes_rescue_marked_issue_via_rescue_adapter_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = OrchestratorConfig(
        # worker.harness must not be the default "manual" -- dispatch_rework's
        # manual-adapter skip gate reads worker.harness directly.
        worker=WorkerRoleConfig(harness="claude-code"),
        rescue=RescueConfig(enabled=True, worker_model="claude-opus-4-1"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    # Seed rework-requested state with the rescue marker already set, as
    # record_review's interception would leave it -- BEFORE creating the app
    # (mirrors test_dispatch_rework_transitions_to_rework_dispatched in
    # test_charlie_work.py: app construction can touch paths.ensure()).
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "rework_requested"}
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "rescue_attempted": True,
            "rescue_cause": "rework_cycle_cap",
        }
        save_state(paths.state_file, state)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "rework-prompt.md").write_text("rescue rework prompt", encoding="utf-8")

    captured_settings = []

    def fake_dispatch_sessions(_repo_root, _manifest, _results, settings, requests):
        captured_settings.append(settings)
        from charlie_work.adapters import SessionDispatchResult

        return [
            SessionDispatchResult(
                issue_number=r.issue_number,
                issue_title=r.issue_title,
                prompt_path=str(r.prompt_path),
                branch_name=r.branch_name,
                adapter=settings.adapter,
                ok=True,
                pid=4242,
                process_start_time=1.0,
            )
            for r in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    result = app.dispatch_rework(limit=5)

    assert result.ok is True
    assert len(captured_settings) == 1
    settings = captured_settings[0]
    assert settings.adapter == "claude-code"
    assert settings.config is not None
    assert settings.config.worker.model == "claude-opus-4-1"

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"


def test_dispatch_rework_normal_issue_unaffected_by_rescue_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A normal (non-rescue-marked) rework candidate must keep using the
    primary configured adapter settings even when rescue.enabled is True."""
    config = OrchestratorConfig(
        worker=WorkerRoleConfig(harness="claude-code"),
        rescue=RescueConfig(enabled=True, worker_model="claude-opus-4-1"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "rework_requested"}
        state["prs"]["456"] = {"number": 456, "issue_number": 123}
        save_state(paths.state_file, state)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "rework-prompt.md").write_text("normal rework prompt", encoding="utf-8")

    captured_settings = []

    def fake_dispatch_sessions(_repo_root, _manifest, _results, settings, requests):
        captured_settings.append(settings)
        from charlie_work.adapters import SessionDispatchResult

        return [
            SessionDispatchResult(
                issue_number=r.issue_number,
                issue_title=r.issue_title,
                prompt_path=str(r.prompt_path),
                branch_name=r.branch_name,
                adapter=settings.adapter,
                ok=True,
                pid=4242,
                process_start_time=1.0,
            )
            for r in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    app.dispatch_rework(limit=5)

    assert len(captured_settings) == 1
    settings = captured_settings[0]
    assert settings.config.worker.model != "claude-opus-4-1"


# --- Rescue review exit semantics: _process_rescue_review ---


def _rescue_marked_app(tmp_path: Path, **config_kwargs) -> OrchestratorApp:
    config = OrchestratorConfig(rescue=RescueConfig(enabled=True), **config_kwargs)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _seed_pr_state(paths, 456, 123, rescue_attempted=True, rescue_cause="rework_cycle_cap")
    return app


def _fake_cross_family_review(decision: str, summary: str):
    def fake(
        *,
        model,
        command,
        repo_root,
        prompt_text,
        prompt_path,
        report_path,
        timeout_seconds,
        head_ref_oid=None,
        **_,
    ):
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt_text, encoding="utf-8")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        verdict = {"decision": decision, "summary": summary, "required_changes": []}
        report_path.write_text(
            "# Cross-family adversarial review\n\n---\n\n"
            f"analysis text\n\n```json\n{json.dumps(verdict)}\n```\n",
            encoding="utf-8",
        )
        return CrossFamilyResult(ok=True, report_path=str(report_path), model=model, returncode=0)

    return fake


def test_process_rescue_review_request_changes_escalates_with_both_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1266: a "request_changes" rescue verdict is not a "blocked"
    judgment call -- the rescue tier structurally cannot loop again, so
    this is mechanical and lands agent:operator-queue, not
    agent:human-needed."""
    captured_comments: list[tuple[int, str]] = []

    class CapturingGitHub(FakeGitHub):
        def pr_comment(self, number: int, body_file: Path) -> None:
            captured_comments.append((number, body_file.read_text(encoding="utf-8")))

    config = OrchestratorConfig(rescue=RescueConfig(enabled=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = CapturingGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _seed_pr_state(paths, 456, 123, rescue_attempted=True, rescue_cause="rework_cycle_cap")

    monkeypatch.setattr(
        "charlie_work.workflow.run_cross_family_review",
        _fake_cross_family_review("request_changes", "still has a bug"),
    )

    result = app._process_rescue_review({"pr": 456, "issue": 123})

    assert result.data["rescue_review_decision"] == "request_changes"
    assert result.data["escalated"] is True

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["issues"]["123"]["status"] == "escalated"
    assert (123, config.labels.operator_queue) in fake_gh.labels_added

    events = _events(state, "rescue_review_escalated")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["cause"] == "rework_cycle_cap"
    assert payload["rescue_branch"] == "agent/issue-123-fix-search"
    assert payload["rescue_head_sha"] == "sha-abc123"
    assert payload["cross_family_report"]
    assert Path(payload["cross_family_report"]).exists()
    assert payload["verdict_decision"] == "request_changes"

    assert len(captured_comments) == 1
    comment_number, comment_body = captured_comments[0]
    assert comment_number == 456
    assert "agent/issue-123-fix-search" in comment_body
    assert "sha-abc123" in comment_body
    assert "rescue-review-report.md" in comment_body


def test_process_rescue_review_blocked_verdict_still_escalates_to_human_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1266 behavior preservation: an explicit "blocked" rescue
    verdict is a human product/security decision (the same judgment call as
    record_review's own "blocked" path per issue #783), so it must keep
    routing to agent:human-needed unchanged -- unlike "request_changes" and
    an unparseable report (both mechanical, covered above), this is the one
    rescue-review outcome _process_rescue_review classifies as reason_class
    "judgment" (see the ``rescue_reason_class`` derivation in
    ``_process_rescue_review``)."""
    app = _rescue_marked_app(tmp_path)
    monkeypatch.setattr(
        "charlie_work.workflow.run_cross_family_review",
        _fake_cross_family_review("blocked", "security concern, needs a human"),
    )

    result = app._process_rescue_review({"pr": 456, "issue": 123})

    assert result.data["rescue_review_decision"] == "blocked"
    assert result.data["escalated"] is True

    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["reason_class"] == "judgment"
    assert (123, app.config.labels.human_needed) in app.gh.labels_added
    assert (123, app.config.labels.operator_queue) not in app.gh.labels_added

    events = _events(state, "rescue_review_escalated")
    assert len(events) == 1
    assert events[0]["payload"]["verdict_decision"] == "blocked"


def test_process_rescue_review_approved_takes_normal_merge_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _rescue_marked_app(tmp_path)
    monkeypatch.setattr(
        "charlie_work.workflow.run_cross_family_review",
        _fake_cross_family_review("approved", "looks good now"),
    )

    result = app._process_rescue_review({"pr": 456, "issue": 123})

    assert result.data["rescue_review_decision"] == "approved"
    state = load_state(app.paths.state_file)
    # record_review's normal approved path took over.
    assert state["prs"]["456"]["decision"] == "approved"
    assert state["prs"]["456"]["status"] == "approved"
    assert state["issues"]["123"]["status"] == "approved"
    # rescue_attempted is durable -- it must survive an approval too, so a
    # PR can never spend a second rescue attempt.
    assert state["prs"]["456"]["rescue_attempted"] is True
    assert _events(state, "rescue_review_escalated") == []


def test_process_rescue_review_unparseable_report_escalates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cross-family failure/unparseable report must fail closed to
    escalation, never silently loop or silently approve."""
    app = _rescue_marked_app(tmp_path)

    def fake_failure(*, report_path, model, **_):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("(UNAVAILABLE)\n", encoding="utf-8")
        return CrossFamilyResult(
            ok=False, report_path=str(report_path), model=model, error="provider outage"
        )

    monkeypatch.setattr("charlie_work.workflow.run_cross_family_review", fake_failure)

    result = app._process_rescue_review({"pr": 456, "issue": 123})

    assert result.data["rescue_review_decision"] == "unparseable"
    assert result.data["escalated"] is True
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["status"] == "escalated"


# --- charlie unescalate clears the rescue marker ---


def test_unescalate_clears_rescue_marker(tmp_path: Path) -> None:
    config = OrchestratorConfig(rescue=RescueConfig(enabled=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _seed_pr_state(
        paths,
        456,
        123,
        status="escalated",
        rescue_attempted=True,
        rescue_cause="rework_cycle_cap",
        rescue_dispatched_at="2026-07-24T00:00:00Z",
        request_changes_count=2,
    )
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "escalated"}
        save_state(paths.state_file, state)

    result = app.unescalate(pr_number=456)

    assert result.ok is True
    state = load_state(paths.state_file)
    pr_state = state["prs"]["456"]
    assert "rescue_attempted" not in pr_state
    assert "rescue_cause" not in pr_state
    assert "rescue_dispatched_at" not in pr_state
    assert pr_state["request_changes_count"] == 0

    # A fresh cap exceedance after unescalate gets a fresh rescue attempt.
    result2 = app.record_review(
        456, "request_changes", summary="needs work again", verdict_provenance="fresh_llm_review"
    )
    assert result2.data.get("rescue_dispatched") is not True  # only 1 request_changes so far


# --- Adversarial-review fix 1 (CRITICAL): escalated rescue PRs must not
# re-enter _process_rescue_review every pass ---


def test_dispatch_reviews_skips_escalated_rescue_pr_no_reprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rescue-marked PR that already escalated (status == "escalated")
    must be dropped from the queue entirely -- neither reprocessed through
    _process_rescue_review (which would re-run the blocking cross-family
    review, repost the escalation PR comment, and re-fire the escalation
    event every pass) nor routed to the normal Claude reviewer."""
    from charlie_work.config import ReviewDispatchConfig

    captured_comments: list[tuple[int, str]] = []

    class CapturingGitHub(FakeGitHub):
        def pr_comment(self, number: int, body_file: Path) -> None:
            captured_comments.append((number, body_file.read_text(encoding="utf-8")))

    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
        rescue=RescueConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = CapturingGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_pr_state(
        paths,
        456,
        123,
        status="escalated",
        rescue_attempted=True,
        rescue_cause="rework_cycle_cap",
    )
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "escalated"}
        save_state(paths.state_file, state)

    call_count = {"n": 0}
    original = OrchestratorApp._process_rescue_review

    def spy(self, candidate):
        call_count["n"] += 1
        return original(self, candidate)

    monkeypatch.setattr(OrchestratorApp, "_process_rescue_review", spy)
    monkeypatch.setattr(
        "charlie_work.workflow.run_cross_family_review",
        _fake_cross_family_review("request_changes", "would re-escalate"),
    )

    result = app.dispatch_reviews()

    assert call_count["n"] == 0
    assert captured_comments == []
    assert result.data.get("selected_count", 0) == 0
    state = load_state(paths.state_file)
    assert _events(state, "rescue_review_escalated") == []

    # Run a second pass to confirm this is stable, not a one-shot fluke.
    result2 = app.dispatch_reviews()
    assert call_count["n"] == 0
    assert captured_comments == []
    state2 = load_state(paths.state_file)
    assert _events(state2, "rescue_review_escalated") == []
    assert result2.ok is True


# --- Adversarial-review fix 2 (IMPORTANT): Claude quota deferral must not
# freeze rescue reviews (they run on the cross-family/Devin adapter) ---


def test_quota_deferred_rescue_candidate_still_processed_normal_still_deferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import UTC, datetime, timedelta

    from charlie_work.config import ReviewDispatchConfig

    class TwoPrGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues.append(
                {
                    "number": 124,
                    "title": "Fix other thing",
                    "url": "https://example.test/issues/124",
                    "body": "Other thing is broken",
                    "labels": [{"name": "automated-ready"}],
                    "state": "OPEN",
                }
            )
            self.prs.append(
                {
                    "number": 789,
                    "title": "Fix #124: other thing",
                    "url": "https://example.test/pull/789",
                    "headRefName": "agent/issue-124-fix-other",
                    "baseRefName": "main",
                    "headRefOid": "sha-def456",
                    "mergeStateStatus": "CLEAN",
                    "body": "Closes #124\n\nTests: regression coverage added.",
                    "labels": [],
                    "isCrossRepository": False,
                    "state": "OPEN",
                }
            )

    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
        rescue=RescueConfig(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = TwoPrGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")  # rescue-marked candidate
    _write_review_packet(paths, 789, "sha-def456")  # normal candidate
    _seed_pr_state(paths, 456, 123, rescue_attempted=True, rescue_cause="rework_cycle_cap")

    # Force the Claude reviewer-quota gate into deferred (exhausted, probe
    # not ready) state.
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["reviewer_quota"] = {"throttled_until": future, "probe_after": future}
        save_state(paths.state_file, state)

    monkeypatch.setattr(
        "charlie_work.workflow.run_cross_family_review",
        _fake_cross_family_review("approved", "rescue review ran during quota deferral"),
    )

    result = app.dispatch_reviews()

    assert result.data.get("deferred_reason") == "reviewer_quota_probe_backoff"
    # Normal candidate (789) stayed deferred: no dispatch attempted for it.
    assert result.data.get("selected_count", 0) == 0
    assert result.data.get("launched_count", 0) == 0

    # Rescue candidate (456) was still processed despite the Claude quota
    # deferral -- it runs on the cross-family adapter, not Claude.
    rescue_results = result.data.get("rescue_review_results", [])
    assert len(rescue_results) == 1
    assert rescue_results[0]["pr"] == 456
    assert rescue_results[0]["rescue_review_decision"] == "approved"

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["decision"] == "approved"
    # The normal candidate's PR record is untouched by this pass.
    assert "decision" not in state["prs"].get("789", {})


# --- Adversarial-review fix 3 (IMPORTANT): routing of an already-marked PR
# must key on the durable marker alone, never on config.rescue.enabled ---


def test_rescue_marker_routes_correctly_even_when_rescue_disabled_in_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rescue.enabled only gates NEW rescue entry at the three cap sites.
    A PR that already carries rescue_attempted must keep routing through
    the rescue paths (review AND rework dispatch) even if an operator flips
    rescue.enabled off while the rescue is in flight."""
    from charlie_work.adapters import SessionDispatchResult
    from charlie_work.config import ReviewDispatchConfig

    # -- dispatch_reviews: rescue-marked PR must still get the cross-family
    # rescue review, never the normal Claude reviewer dispatch path.
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True),
        rescue=RescueConfig(enabled=False, worker_model="claude-opus-4-1"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_pr_state(paths, 456, 123, rescue_attempted=True, rescue_cause="rework_cycle_cap")

    call_count = {"n": 0}
    original = OrchestratorApp._process_rescue_review

    def spy(self, candidate):
        call_count["n"] += 1
        return original(self, candidate)

    monkeypatch.setattr(OrchestratorApp, "_process_rescue_review", spy)
    monkeypatch.setattr(
        "charlie_work.workflow.run_cross_family_review",
        _fake_cross_family_review("approved", "rescue still ran with rescue.enabled=False"),
    )

    result = app.dispatch_reviews()

    assert call_count["n"] == 1
    assert result.data.get("rescue_review_results")
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["decision"] == "approved"

    # -- dispatch_rework: rescue-marked issue must still launch via the
    # claude-code adapter pinned to rescue.worker_model, never the primary
    # configured adapter/model.
    rework_config = OrchestratorConfig(
        worker=WorkerRoleConfig(harness="command"),
        rescue=RescueConfig(enabled=False, worker_model="claude-opus-4-1"),
    )
    rework_paths = runtime_paths(tmp_path / "rework", rework_config.runtime.state_dir)
    rework_paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(rework_paths.state_file):
        state = load_state(rework_paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "rework_requested"}
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "rescue_attempted": True,
            "rescue_cause": "rework_cycle_cap",
        }
        save_state(rework_paths.state_file, state)
    rework_gh = FakeGitHub()
    rework_app = OrchestratorApp(tmp_path / "rework", rework_paths, rework_config, rework_gh)
    pr_dir = rework_paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "rework-prompt.md").write_text("rescue rework prompt", encoding="utf-8")

    captured_settings = []

    def fake_dispatch_sessions(_repo_root, _manifest, _results, settings, requests):
        captured_settings.append(settings)
        return [
            SessionDispatchResult(
                issue_number=r.issue_number,
                issue_title=r.issue_title,
                prompt_path=str(r.prompt_path),
                branch_name=r.branch_name,
                adapter=settings.adapter,
                ok=True,
                pid=4242,
                process_start_time=1.0,
            )
            for r in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    rework_app.dispatch_rework(limit=5)

    assert len(captured_settings) == 1
    settings = captured_settings[0]
    # devin.adapter is "command" and rescue.enabled is False -- but the
    # marker alone must still route this issue through the rescue adapter.
    assert settings.adapter == "claude-code"
    assert settings.config.worker.model == "claude-opus-4-1"


# --- Issue #618-D: dry-run short-circuit for _process_rescue_review -----------


def test_process_rescue_review_dry_run_short_circuits_without_escalating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #618-D: ``_process_rescue_review`` in dry-run must short-circuit
    BEFORE any writes or mutations. Threading ``dry_run`` to
    ``run_cross_family_review`` alone is harmful: the dry-run branch returns a
    synthetic failure (``ok=False``), which drives the function into its
    escalation arm and would mark a PR escalated during a preview.
    """
    config = OrchestratorConfig(rescue=RescueConfig(enabled=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    _seed_pr_state(paths, 456, 123, rescue_attempted=True, rescue_cause="rework_cycle_cap")

    # If the short-circuit fails, run_cross_family_review would be called.
    # Plant a sentinel that raises if the function is reached at all.
    def _must_not_run(**kwargs):
        raise AssertionError("run_cross_family_review must not be called in dry-run")

    monkeypatch.setattr("charlie_work.workflow.run_cross_family_review", _must_not_run)

    result = app._process_rescue_review({"pr": 456, "issue": 123})

    # The dry-run result should indicate success (the preview itself worked)
    assert result.ok is True
    assert "dry-run" in result.message.lower()
    assert result.data["rescue_review_decision"] == "dry-run"

    # No state mutation — the PR must NOT be escalated
    state = load_state(paths.state_file)
    assert state["prs"]["456"].get("status") != "escalated"
    assert state["issues"].get("123", {}).get("status") != "escalated"

    # No escalation event recorded
    assert _events(state, "rescue_review_escalated") == []

    # No escalation label added
    assert (123, config.labels.human_needed) not in fake_gh.labels_added

    # No PR comment posted (no pr_dir created)
    assert not (paths.prs / "pr-456").exists()
