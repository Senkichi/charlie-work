"""Shared deadline-aware fake and harnesses for the issue #1948 refusal tests.

``FakeGitHub`` never reaches ``GitHub.run()``, where the production
``_pass_deadline_exceeded`` hook lives -- so a test that wants real
armed/spent/refuse behavior through the *real* call graph needs a double
that consults the hook itself. ``_DeadlineAwareGitHub`` is that double:
every gh-touching method is gated the same way the real yield points are,
raising the same ``PassDeadlineExceeded`` refusal, and recording both the
call order (``gh_calls``) and the refused calls (``deadline_refusals``) so a
test can pin *which* call a trip point hits.

Harness helpers shared by ``test_fleet_lane_deadline_refusal.py`` and
``test_fleet_lane_deadline_suspend.py`` (test modules cannot import each
other, so the shared pieces live here): ``_build_app`` (minimal real
OrchestratorApp in the shape of test_write_gate_dry_run_loop's harness),
``_merge_ready_app`` / ``_assert_merge_finalized`` (the merge_ready refusal
matrix), and ``_dispatch_pending_claim`` / ``_ok_dispatch_sessions`` (the
claim->launch refusal tests).
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from charlie_work.config import (
    AutoMergeConfig,
    DeescalationConfig,
    DevinConfig,
    MainCiReclaimConfig,
    OrchestratorConfig,
    ReconcilePassConfig,
    RunnersConfig,
    WorkerRoleConfig,
    WorktreeReclamationConfig,
)
from charlie_work.pass_deadline import PassDeadlineExceeded
from charlie_work.paths import runtime_paths
from charlie_work.state import empty_state, load_state, save_state
from charlie_work.workflow import OrchestratorApp


class _DeadlineAwareGitHub(FakeGitHub):
    """FakeGitHub whose gh-touching methods honor the armed deadline hook."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gh_calls: list[str] = []
        self.deadline_refusals: list[str] = []

    def _deadline_gate(self, command: str) -> None:
        self.gh_calls.append(command)
        check = getattr(self, "_pass_deadline_exceeded", None)
        if check is not None and check():
            self.deadline_refusals.append(command)
            raise PassDeadlineExceeded(f"in-pass deadline exceeded; gh call refused: {command}")


def _gate_method(name: str):
    original = getattr(FakeGitHub, name)

    @functools.wraps(original)
    def _gated(self: _DeadlineAwareGitHub, *args: Any, **kwargs: Any) -> Any:
        self._deadline_gate(name)
        return original(self, *args, **kwargs)

    return _gated


for _method_name in (
    "issue_list",
    "issue_view",
    "pr_list",
    "merged_pr_list",
    "merged_prs_for_issue",
    "pr_view",
    "pr_create",
    "pr_commits",
    "pr_checks",
    "check_run_annotations",
    "pr_diff",
    "add_issue_label",
    "remove_issue_label",
    "add_pr_label",
    "remove_pr_label",
    "close_issue",
    "issue_comment",
    "name_with_owner",
    "merge_pr",
    "delete_branch",
    "pr_ready",
    "pr_close",
    "pr_reopen",
    "push_empty_commit",
    "pr_update_branch",
    "are_issues_open",
    "issue_dependencies",
    "run",
    "commit",
    "compare",
    "compare_diff",
    "branch_protection",
    "label_create",
    "label_list",
    "pr_comment",
    "pr_edit",
    "actions_job",
    "commit_check_runs",
    "workflow_runs_for_head",
    "check_graphql_rate_limit",
):
    setattr(_DeadlineAwareGitHub, _method_name, _gate_method(_method_name))
del _method_name


def _build_app(
    root: Path, gh: FakeGitHub | None = None
) -> tuple[OrchestratorApp, Any, FakeGitHub]:
    """Minimal real-app harness in the shape of test_write_gate_dry_run_loop's."""
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        deescalation=DeescalationConfig(enabled=False),
        worktree_reclamation=WorktreeReclamationConfig(enabled=False),
        main_ci_reclaim=MainCiReclaimConfig(enabled=False),
        reconcile_pass=ReconcilePassConfig(enabled=False),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(root, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    save_state(paths.state_file, empty_state())

    fake_gh = gh if gh is not None else FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = []

    app = OrchestratorApp(root, paths, config, fake_gh)
    (root / ".var" / "charlie-work" / "dispatches" / "sessions").mkdir(parents=True, exist_ok=True)
    return app, paths, fake_gh


def _merge_ready_app(
    tmp_path: Path,
    *,
    runners: RunnersConfig | None = None,
    **auto_merge_kwargs: Any,
) -> tuple[OrchestratorApp, Any, _DeadlineAwareGitHub]:
    """An app whose approved PR 456 (linked to issue 123) is merge-eligible.

    ``update_branch_strategy="off"`` keeps the post-merge deferral tail
    (``_update_open_agent_prs``) out of the finalize-trio tests so the
    armed predicate isolates exactly transition / close_issue /
    delete_branch -- the trio is what the review matrix names. (The config
    cross-check requires ``require_current_base=False`` alongside "off".)
    Pass ``update_branch_strategy="front_of_train"`` (the production
    default) to keep the tail in play, and ``runners=RunnersConfig(...)``
    to arm ``cancel_superseded_runs``.
    """
    auto_merge: dict[str, Any] = {
        "required_checks": ("Tests passed", "Lint & Format", "Pre-commit"),
        "require_approved_review": True,
        "failed_attempt_alarm": 1,
        "update_branch_strategy": "off",
        "require_current_base": False,
    }
    auto_merge.update(auto_merge_kwargs)
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(**auto_merge),
        runners=runners if runners is not None else RunnersConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    save_state(paths.state_file, empty_state())
    fake_gh = _DeadlineAwareGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")
    return app, paths, fake_gh


def _assert_merge_finalized(
    app: OrchestratorApp, paths: Any, fake_gh: _DeadlineAwareGitHub
) -> None:
    """Shared assertions for the suspend-window tests: the whole trio ran."""
    config = app.config
    # merge_pr ran and the merged fact is durable.
    assert fake_gh.merged == [(456, "squash")]
    state = load_state(paths.state_file)
    pr_state = state["prs"]["456"]
    assert pr_state["status"] == "merged"
    assert pr_state["merged"] is True
    assert pr_state["consecutive_failed_merge_attempts"] == 0
    # The finalize trio ran despite the spent predicate.
    assert (123, config.labels.done) in fake_gh.labels_added
    assert (123, config.labels.ready) in fake_gh.labels_removed
    assert fake_gh.closed_issues == [123]
    assert fake_gh.deleted_branches == ["agent/issue-123-fix-search"]
    # No merge-failure counter or alarm: a refused-or-failed finalize call is
    # neither (and under the suspend, the calls simply ran).
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert alarm_events == []
    # The issue record converged to the merged terminal disposition
    # (_merged_issue_fields writes "closed" -- the terminal issue status).
    assert state["issues"]["123"]["status"] == "closed"
    # Re-entry is a no-op: the next pass short-circuits on the durable record.
    second = app.merge_ready(456, merge=True)
    assert second.data.get("already_merged") is True


def _dispatch_pending_claim(paths: Any, issue_number: int = 123) -> bool:
    """True once the issue's dispatch_pending claim is durably written."""
    state = load_state(paths.state_file)
    return state.get("issues", {}).get(str(issue_number), {}).get("status") == "dispatch_pending"


def _ok_dispatch_sessions(requests: list[Any]) -> list[Any]:
    from charlie_work.adapters import SessionDispatchResult

    return [
        SessionDispatchResult(
            issue_number=request.issue_number,
            issue_title=request.issue_title,
            prompt_path=str(request.prompt_path),
            branch_name=request.branch_name,
            adapter="command",
            ok=True,
        )
        for request in requests
    ]
