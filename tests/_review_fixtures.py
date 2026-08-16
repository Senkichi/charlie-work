"""Shared review/dispatch and per-round-archive fixtures.

Two related fixture families hoisted out of ``test_charlie_work.py`` and
``test_review_round_archive.py`` (issue #1284) because both are imported by
other test modules: general review-dispatch/loop app builders and reviewer
sidecar writers first, then the W11 round-archive helpers (PR #456 fixture,
``_record``/``_round_dir``/``_pr_dir``) that ``test_review_event_payload.py``
and ``test_review_pr_comment.py`` build their own tests on top of.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import CrossFamilyConfig, OrchestratorConfig, ReviewDispatchConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp


def _dispatch_reviews_app(
    tmp_path: Path,
    *,
    prs: list[dict[str, Any]] | None = None,
    enabled: bool = True,
    dry_run: bool = False,
) -> OrchestratorApp:
    """Build an OrchestratorApp with review_dispatch enabled (by default) and an empty state file.

    Issue #868: ``enabled`` is overridable so tests can exercise
    ``dispatch_reviews()``'s disabled-gate path with the same PR/state seeding
    helpers used by the enabled-path tests.

    Issue #1251: ``dry_run`` is overridable so tests can exercise the dry-run
    preview branch of ``dispatch_reviews()`` (read-only selection + empty-diff
    pre-flight mirror) with the same PR/state seeding helpers.
    """
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=enabled),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    (paths.root).mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    if prs is not None:
        fake_gh.prs = prs
    return OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=dry_run)


def _write_review_packet(
    tmp_path: Path,
    pr_number: int,
    packet_head_sha: str,
    decision: dict[str, Any] | None = None,
) -> Path:
    """Create a review packet fixture for a PR."""
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        json.dumps({"number": pr_number, "headRefOid": packet_head_sha}),
        encoding="utf-8",
    )
    (pr_dir / "review-prompt.md").write_text(
        f"review prompt for PR #{pr_number}",
        encoding="utf-8",
    )
    if decision is not None:
        (pr_dir / "review-decision.json").write_text(
            json.dumps(decision),
            encoding="utf-8",
        )
    return pr_dir


def _make_dead_review_sidecar(
    reviews_dir: Path,
    pr_number: int,
    log_text: str,
    *,
    started_at: str | None = None,
) -> Path:
    """Create a claude-code review sidecar + log file for a dead reviewer."""
    reviews_dir.mkdir(parents=True, exist_ok=True)
    log_path = reviews_dir / f"issue-{pr_number}-review.claude.log"
    log_path.write_text(log_text, encoding="utf-8")
    sidecar = {
        "issue_number": pr_number,
        "branch": f"agent/issue-{pr_number}-fix",
        "worktree_path": str(reviews_dir / f"issue-{pr_number}"),
        "prompt_path": str(reviews_dir / f"issue-{pr_number}-review-prompt.md"),
        "command": ["claude", "-p", "--permission-mode", "plan"],
        "pid": 99999,
        "started_at": started_at or "2026-07-06T12:00:00Z",
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    sidecar_path = reviews_dir / f"issue-{pr_number}.claude.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return sidecar_path


def _set_review_dispatched_state(
    app: OrchestratorApp,
    pr_number: int,
    issue_number: int,
    dispatched_at: str,
) -> None:
    """Seed state.json with a review_dispatch_dispatched claim."""
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": issue_number,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": dispatched_at,
            "review_dispatch_pending_at": None,
            "review_dispatch_failed_at": None,
            "reviewer_pid": 99999,
            "reviewer_process_start_time": 1.0,
        }
        save_state(app.paths.state_file, state)


def _write_review_events(
    reviews_dir: Path, pr_number: int, *, turns: int, tool_calls: int = 0
) -> Path:
    """Write a stream-json events sidecar for a reviewer that ran ``turns`` turns.

    ``parse_claude_events`` counts one turn per ``assistant`` event and one tool
    call per ``tool_use`` content block. Without this sidecar a dead reviewer
    has zero turns and zero tool calls, which classifies as a launch failure
    rather than a turn-limit death (issue #588) -- so any test asserting
    turn-limit behaviour must seed real session telemetry.
    """
    reviews_dir.mkdir(parents=True, exist_ok=True)
    events_path = reviews_dir / f"issue-{pr_number}-review.events.jsonl"
    lines: list[str] = []
    for index in range(turns):
        content: list[dict[str, Any]] = [{"type": "text", "text": f"Analysis step {index + 1}."}]
        if index < tool_calls:
            content.append({"type": "tool_use", "id": f"t{index}", "name": "Read", "input": {}})
        lines.append(json.dumps({"type": "assistant", "message": {"content": content}}))
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return events_path


def _fake_claude_worker_record(pr_number: int, branch: str) -> ClaudeWorkerRecord:
    """Return a successful Claude worker record for monkeypatched launches."""
    return ClaudeWorkerRecord(
        issue_number=pr_number,
        branch=branch,
        worktree_path="/fake/worktree",
        prompt_path="/fake/prompt.md",
        command=("claude", "-p"),
        pid=12345,
        started_at="2026-07-06T12:00:00Z",
        log_path="/fake/log.log",
        error=None,
        process_start_time=1.0,
    )


def _make_loop_app(tmp_path: Path, *, prs: list[dict]) -> tuple[OrchestratorApp, FakeGitHub]:
    """Build a minimal OrchestratorApp with the given open PRs for loop() tests."""
    from test_charlie_work import _approved_automerge

    config = OrchestratorConfig(
        cross_family=CrossFamilyConfig(enabled=False),
        auto_merge=_approved_automerge(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Ensure PRs passed into loop tests are treated as open even if callers omit state.
    for pr in prs:
        pr.setdefault("state", "OPEN")
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app, fake_gh


def _required_checks_config(**kwargs) -> OrchestratorConfig:
    from charlie_work.config import AutoMergeConfig

    auto_merge = AutoMergeConfig(
        required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
        enabled=True,  # Ensure auto_merge is enabled for merge tests
        **kwargs,
    )
    return OrchestratorConfig(auto_merge=auto_merge)


_PR_NUMBER = 456


def _round_archive_app(tmp_path: Path) -> tuple[OrchestratorApp, Any]:
    """Same minimal-fixture shape as test_verdict_provenance_enforcement's
    ``_carry_forward_app``: bare state.json, default config, default
    FakeGitHub (PR #456, linked to issue #123, head ``sha-abc123``)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
    return app, paths


def _record(app: OrchestratorApp, *, head: str, summary: str, required_changes: list[str]):
    # FakeGitHub.pr_view honors pr_head_shas[number] as an override for the
    # PR's headRefOid -- the same mechanism test_charlie_work.py's own
    # rework-cap tests use to control reviewed_head_sha between calls.
    app.gh.pr_head_shas[_PR_NUMBER] = head
    result = app.record_review(
        _PR_NUMBER,
        "request_changes",
        summary=summary,
        required_changes=required_changes,
        verdict_provenance="fresh_llm_review",
    )
    assert result.ok is True, result.message
    return result


def _round_dir(paths: Any, round_number: int) -> Path:
    return paths.prs / f"pr-{_PR_NUMBER}" / "rounds" / f"round-{round_number}"


def _pr_dir(paths: Any) -> Path:
    return paths.prs / f"pr-{_PR_NUMBER}"
