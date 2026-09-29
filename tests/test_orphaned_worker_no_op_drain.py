"""App-level coverage for the #2034 no-op rework disposition.

A rework worker that exits without changing the PR used to leave the issue
``dispatched`` on ``agent:in-progress`` forever (one fingerprinted drift event,
no consumer). ``orphaned_worker_no_op_drain`` gives every such finding a
disposition; these tests drive the real sweep with the real
``OrchestratorApp.record_review`` / ``review`` and pin one test per branch:

- live CI red -> CI-failure rework whose brief names the failing step (the #2005
  shape: exit 0, only a merge commit pushed, Lint red);
- CI green + the worker rebutted the verdict -> ``review()`` once per head;
- otherwise, or a repeat no-op on the same head -> ``rework_no_op`` escalation;
- CI state unknown -> the one deliberate deferral (retry, #654-bounded).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHubWithChecksAndAnnotations
from _orphan_sweep_fixtures import _dead_worker_rework_bed, _write_outcome
from _rework_dispatch_fixtures import _wg
from charlie_work.config import (
    AutoMergeConfig,
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.janitor import _calculate_patch_id
from charlie_work.state import load_state, save_state
from charlie_work.workflow import CommandResult, OrchestratorApp

_LINT = "Lint & Format"
_LINT_LINK = "https://github.com/o/r/actions/runs/5/job/77"
_LOG = "\n".join(f"noise line {i}" for i in range(80)) + "\nsrc/foo.py:3:1: E501 line too long\n"
_DIFF = "diff --git a/file b/file"


class _CiGh(FakeGitHubWithChecksAndAnnotations):
    """FakeGitHub with a failing Lint job: step list and log tail available."""

    def __init__(self, checks: list[dict[str, Any]] | None) -> None:
        super().__init__(checks=checks)
        self.checks_unavailable = False

    def pr_checks(self, number: int):
        return None if self.checks_unavailable else super().pr_checks(number)

    def actions_job(self, job_id: int) -> dict[str, Any] | None:
        return {
            "conclusion": "failure",
            "steps": [
                {"name": "Set up job", "conclusion": "success"},
                {"name": "Run ruff check", "conclusion": "failure"},
            ],
        }

    def run(self, args, *, json_output: bool = False, allow_failure: bool = False):
        if any(str(a).endswith("/logs") for a in args):
            return _LOG
        return super().run(args, json_output=json_output, allow_failure=allow_failure)


def _bed(tmp_path: Path, *, lint_red: bool, live_head: str = "abc123"):
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
        auto_merge=AutoMergeConfig(required_checks=(_LINT,)),
    )
    config, paths, bed_gh, _ = _dead_worker_rework_bed(tmp_path, config=config, janitor_green=True)
    gh = _CiGh(
        [{"name": _LINT, "state": "FAILURE" if lint_red else "SUCCESS", "link": _LINT_LINK}]
    )
    gh.repo_root = tmp_path
    gh.issues = bed_gh.issues
    gh.prs = bed_gh.prs
    gh.prs[0]["headRefOid"] = live_head
    gh.prs[0]["body"] = "Closes #207\n\n## Tests\nuv run --extra dev pytest -q"
    # The reviewed verdict pinned this diff, so an unchanged diff is a no-op.
    state = load_state(paths.state_file)
    state["prs"]["100"]["reviewed_patch_id"] = _calculate_patch_id(_DIFF)
    save_state(paths.state_file, state)
    return config, paths, gh


def _terminal_exit_zero(tmp_path: Path) -> None:
    sessions = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / "issue-207.claude.terminal.json").write_text(
        json.dumps({"pid": 99999, "exit_code": 0, "duration_seconds": 300.0}), encoding="utf-8"
    )


def _sweep(tmp_path: Path, paths, config, gh, app: OrchestratorApp, review=None) -> None:
    from unittest.mock import patch

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    sessions = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions,
            paths.state_file,
            config,
            gh,
            review_callback=review or app.review,
            record_review_callback=app.record_review,
            enrich_checks_callback=app._enrich_checks_infra_blocked,
            write_gate=_wg(paths.state_file),
        )


def _redispatch(paths) -> None:
    """Put issue 207 back to a dead ``dispatched`` worker, as a second no-op would."""
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        **state["issues"]["207"],
        "status": "dispatched",
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)


def _decision(paths) -> dict[str, Any]:
    return json.loads((paths.prs / "pr-100" / "review-decision.json").read_text(encoding="utf-8"))


def _assert_ci_rework_brief(paths, tmp_path: Path, app: OrchestratorApp, gh) -> None:
    changes = _decision(paths)["required_changes"]
    assert any("failing step(s): Run ruff check" in c for c in changes)
    assert any("E501 line too long" in c for c in changes)
    brief = app._write_rework_prompt(gh.prs[0], 207, "note").read_text(encoding="utf-8")
    assert "Run ruff check" in brief
    assert "E501 line too long" in brief


def test_2005_shape_merge_only_push_lint_red_reaches_rework_naming_failing_step(
    tmp_path: Path,
) -> None:
    """Regression: exit 0, only a merge commit pushed (head moved, diff unchanged),
    Lint red. The janitor refuses the head-change review; the issue must still reach
    a rework whose prompt contains the failing step."""
    config, paths, gh = _bed(tmp_path, lint_red=True, live_head="def456")
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(tmp_path, paths, config, gh, app)

    state = load_state(paths.state_file)
    assert state["issues"]["207"]["status"] == "rework_requested"
    assert state["issues"]["207"]["no_op_handled_head"] == "def456"
    assert _decision(paths)["decision"] == "request_changes"
    _assert_ci_rework_brief(paths, tmp_path, app, gh)
    kinds = [e["kind"] for e in state["events"]]
    assert "rework_no_op_ci_rework_requested" in kinds


def test_clean_exit_with_ci_red_routes_to_ci_rework(tmp_path: Path) -> None:
    config, paths, gh = _bed(tmp_path, lint_red=True)
    _terminal_exit_zero(tmp_path)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(tmp_path, paths, config, gh, app)

    state = load_state(paths.state_file)
    assert state["issues"]["207"]["status"] == "rework_requested"
    _assert_ci_rework_brief(paths, tmp_path, app, gh)


def test_repeat_no_op_on_same_head_escalates_after_ci_rework(tmp_path: Path) -> None:
    config, paths, gh = _bed(tmp_path, lint_red=True)
    _terminal_exit_zero(tmp_path)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(tmp_path, paths, config, gh, app)
    assert load_state(paths.state_file)["issues"]["207"]["status"] == "rework_requested"

    _redispatch(paths)  # the CI-rework worker no-oped on the same head
    _sweep(tmp_path, paths, config, gh, app)

    entry = load_state(paths.state_file)["issues"]["207"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    assert (207, config.labels.human_needed) in gh.labels_added


def test_ci_green_with_rebuttal_returns_to_review_once_per_head(tmp_path: Path) -> None:
    config, paths, gh = _bed(tmp_path, lint_red=False)
    _terminal_exit_zero(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {"push_succeeded": True, "head_sha": "abc123", "pr_comment": "Head already fixes it."},
    )
    app = OrchestratorApp(tmp_path, paths, config, gh)
    calls: list[int] = []

    def review(pr_number: int) -> CommandResult:
        calls.append(pr_number)
        return CommandResult(True, "packet", {"pr": pr_number})

    _sweep(tmp_path, paths, config, gh, app, review=review)
    entry = load_state(paths.state_file)["issues"]["207"]
    assert calls == [100]
    assert entry["status"] == "reviewing"

    _redispatch(paths)  # second no-op on the same head: no second rebuttal review
    _sweep(tmp_path, paths, config, gh, app, review=review)
    entry = load_state(paths.state_file)["issues"]["207"]
    assert calls == [100]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    assert entry["no_op_worker_comment"] == "Head already fixes it."


def test_ci_green_rebuttal_review_refused_escalates_with_worker_comment(tmp_path: Path) -> None:
    config, paths, gh = _bed(tmp_path, lint_red=False)
    _terminal_exit_zero(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {"push_succeeded": True, "head_sha": "abc123", "pr_comment": "Nothing left to fix."},
    )
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(
        tmp_path,
        paths,
        config,
        gh,
        app,
        review=lambda n: CommandResult(False, "janitor refused", {"pr": n}),
    )
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    escalated = [e for e in state["events"] if e["kind"] == "rework_no_op_escalated"]
    assert escalated[0]["payload"]["worker_comment"] == "Nothing left to fix."


def test_ci_green_without_rebuttal_escalates_human_needed(tmp_path: Path) -> None:
    config, paths, gh = _bed(tmp_path, lint_red=False)
    _terminal_exit_zero(tmp_path)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(tmp_path, paths, config, gh, app)

    entry = load_state(paths.state_file)["issues"]["207"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    assert (207, config.labels.human_needed) in gh.labels_added


def test_unknown_ci_state_defers_instead_of_escalating(tmp_path: Path) -> None:
    config, paths, gh = _bed(tmp_path, lint_red=True)
    gh.checks_unavailable = True
    _terminal_exit_zero(tmp_path)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(tmp_path, paths, config, gh, app)

    entry = load_state(paths.state_file)["issues"]["207"]
    assert entry["status"] == "dispatched"
    assert entry["orphan_drift_at"]  # the #654 backstop is armed and bounds the deferral
