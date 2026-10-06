"""Shared harness for the merge_ready characterization suites.

See ``test_merge_path_characterization.py`` for the pinning contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


from _fakes_github import FakeGitHub
from charlie_work.config import (
    AutoMergeConfig,
    DispatchConfig,
    OrchestratorConfig,
    ReviewDispatchConfig,
    WorkerRoleConfig,
    DevinConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.command_result import CommandResult
from charlie_work.workflow import OrchestratorApp

_REQUIRED = ("Tests passed", "Lint & Format", "Pre-commit")
_PR = 456
_ISSUE = 123

# ---------------------------------------------------------------------------
# Return-shape key sets (the exact ``CommandResult.data`` contract)
# ---------------------------------------------------------------------------

# Keys shared by the full-evaluation result of BOTH paths.
_COMMON_FINAL_KEYS = frozenset(
    {
        "pr",
        "issue",
        "can_merge",
        "auto_merge_enabled",
        "merged",
        "merge_output",
        "branch_deleted",
        "review_decision",
        "checks",
        "checks_unavailable",
        "label_error",
        "update_open_prs_results",
        "cancel_superseded_runs_results",
        "containment_warnings",
        "consecutive_failed_merge_attempts",
        "consecutive_stale_base_deferrals",
        "merge_attempt_alarm",
        "merge_attempt_warning",
        "merge_conflict",
        "cross_pr_revert_detected",
        "cross_pr_revert_reason",
        "cross_pr_revert_routed",
        "cross_pr_revert_undetermined",
        "mergequeue_label_applied",
        "merge_hold",
        "merge_hold_check_unavailable",
        "human_merge_hold",
        "human_merge_check_unavailable",
        "escalated_merge_hold",
        # merge_gate_inputs (#1060)
        "summary_ready",
        "approved",
        "require_approved_review",
        "sync_failed",
    }
)
_GATE_INPUT_KEYS = frozenset(
    {"summary_ready", "approved", "require_approved_review", "sync_failed"}
)
_LIVE_FINAL_KEYS = _COMMON_FINAL_KEYS | {"human_merge_label_error"}
_DRY_FINAL_KEYS = _COMMON_FINAL_KEYS | {"dry_run"}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Outcome:
    """One ``merge_ready`` call plus everything observable about it."""

    result: CommandResult
    gh: FakeGitHub
    paths: Any
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    # (mtime_ns, size) of state.json, or None when absent: ``save_state`` always
    # restamps ``generated_at``, so a stray write of unchanged content is only
    # visible here, not in the content comparison.
    stat_before: tuple[int, int] | None = None
    stat_after: tuple[int, int] | None = None

    @property
    def data(self) -> dict[str, Any]:
        return self.result.data

    @property
    def pr_entry(self) -> dict[str, Any]:
        return self.state_after["prs"].get(str(_PR), {})

    @property
    def issue_entry(self) -> dict[str, Any]:
        return self.state_after["issues"].get(str(_ISSUE), {})

    def event_kinds(self) -> list[str]:
        return [e["kind"] for e in self.state_after.get("events", [])]

    def new_event_kinds(self) -> list[str]:
        before = len(self.state_before.get("events", []))
        return [e["kind"] for e in self.state_after.get("events", [])[before:]]


@dataclass(frozen=True)
class _Pair:
    live: _Outcome
    dry: _Outcome


def _cfg(
    *,
    required: tuple[str, ...] = (),
    strategy: str = "off",
    mergequeue: str | None = None,
    alarm: int = 3,
    require_current_base: bool | None = None,
    **extra: Any,
) -> OrchestratorConfig:
    """Config builder.  ``strategy="off"`` keeps the base-freshness lane inert
    unless a scenario opts in; ``DevinConfig`` + ``harness="command"`` make
    rework routing land without launching a real worker."""
    am_kwargs: dict[str, Any] = {
        "required_checks": required,
        "require_approved_review": True,
        "update_branch_strategy": strategy,
        "failed_attempt_alarm": alarm,
        "require_current_base": (strategy != "off")
        if require_current_base is None
        else require_current_base,
    }
    if mergequeue is not None:
        am_kwargs["mergequeue_label"] = mergequeue
    dispatch = extra.pop("dispatch", DispatchConfig())
    review_dispatch = extra.pop("review_dispatch", ReviewDispatchConfig())
    return OrchestratorConfig(
        auto_merge=AutoMergeConfig(**am_kwargs),
        dispatch=dispatch,
        review_dispatch=review_dispatch,
        devin=DevinConfig(dispatch_command="exit 0"),
        worker=WorkerRoleConfig(harness="command"),
    )


def _run(
    root: Path,
    config: OrchestratorConfig,
    make_gh: Callable[[], FakeGitHub],
    *,
    dry_run: bool,
    approve: bool = True,
    pre: Callable[[OrchestratorApp, Any, FakeGitHub], None] | None = None,
    merge: bool | None = True,
    merge_train_head: int | None = None,
) -> _Outcome:
    root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(root, config.runtime.state_dir)
    gh = make_gh()
    app = OrchestratorApp(root, paths, config, gh, dry_run=dry_run)
    if approve:
        app.record_review(_PR, "approved", summary="ok", verdict_provenance="fresh_llm_review")
    if pre is not None:
        pre(app, paths, gh)
    # Setup (record_review) stamps labels; the pins below cover merge_ready only.
    for recorder in (
        gh.labels_added,
        gh.labels_removed,
        gh.pr_labels_added,
        gh.pr_comments_posted,
        gh.merged,
        gh.deleted_branches,
        gh.closed_issues,
        gh.pr_update_branch_calls,
    ):
        recorder.clear()
    before = load_state(paths.state_file)
    stat_before = _state_stat(paths.state_file)
    kwargs: dict[str, Any] = {"merge": merge}
    if merge_train_head is not None:
        kwargs["merge_train_head"] = merge_train_head
    result = app.merge_ready(_PR, **kwargs)
    return _Outcome(
        result,
        gh,
        paths,
        before,
        load_state(paths.state_file),
        stat_before,
        _state_stat(paths.state_file),
    )


def _state_stat(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return st.st_mtime_ns, st.st_size


def _pair(
    tmp_path: Path,
    config: OrchestratorConfig,
    make_gh: Callable[[], FakeGitHub] = FakeGitHub,
    **kwargs: Any,
) -> _Pair:
    """Same facts through the live path and the dry-run path (separate roots)."""
    live = _run(tmp_path / "live", config, make_gh, dry_run=False, **kwargs)
    dry = _run(tmp_path / "dry", config, make_gh, dry_run=True, **kwargs)
    return _Pair(live, dry)


def _seed_pr_entry(paths: Any, **fields: Any) -> None:
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"][str(_PR)] = {
            **state["prs"].get(str(_PR), {}),
            "number": _PR,
            "issue_number": _ISSUE,
            **fields,
        }
        save_state(paths.state_file, state)


def _seed_issue_entry(paths: Any, **fields: Any) -> None:
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"][str(_ISSUE)] = {
            **state["issues"].get(str(_ISSUE), {}),
            "number": _ISSUE,
            **fields,
        }
        save_state(paths.state_file, state)


def _assert_dry_run_inert(out: _Outcome) -> None:
    """The dry-run invariant: state.json (counters and events included) is
    untouched and nothing mutating reached GitHub."""

    # ``generated_at`` is a wall-clock stamp: with no state file on disk
    # ``load_state`` mints a fresh default each call, so two loads straddling
    # a second boundary differ without any write having happened.
    def _stable(state: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in state.items() if k != "generated_at"}

    assert _stable(out.state_after) == _stable(out.state_before)
    # A dry-run never writes the file at all (not even an unchanged re-save).
    assert out.stat_after == out.stat_before
    gh = out.gh
    assert gh.merged == []
    assert gh.pr_labels_added == []
    assert gh.labels_added == []
    assert gh.labels_removed == []
    assert gh.pr_update_branch_calls == []
    assert gh.deleted_branches == []
    assert gh.closed_issues == []
    assert gh.pr_comments_posted == []


def _conflict_pr() -> list[dict[str, Any]]:
    return [
        {
            "number": _PR,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]


def _conflicted_gh() -> FakeGitHub:
    gh = FakeGitHub()
    gh.prs = _conflict_pr()
    return gh
