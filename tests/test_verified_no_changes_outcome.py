"""Issue #2185: ``verified_no_changes`` -- the worker's zero-diff success outcome.

A verification-only ticket completes correctly with no commit. The sweep must
close it (not reap it as a dead worker and redispatch), but only when the
worktree proves nothing was left undone.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
from _dead_worker_sweep_characterization_fixtures import (
    events_of,
    iso,
    issue_entry,
    no_pr_bed,
    run_sweep,
    sessions_dir_for,
)
from charlie_work.config import DispatchConfig, OrchestratorConfig, RuntimeConfig
from charlie_work.dead_worker_sweep.model import REQUEST_TYPES, CloseVerifiedNoChanges
from charlie_work.paths import resolved_layout, runtime_paths
from charlie_work.verified_no_changes import (
    VERIFIED_NO_CHANGES_OUTCOME,
    is_verified_no_changes,
    resolution_comment,
)
from charlie_work.workflow import OrchestratorApp as _App
from charlie_work.worktree import worktree_path_for_branch

_SECTION = (
    Path(__file__).parents[1]
    / "src"
    / "charlie_work"
    / "prompts"
    / "worker_sections"
    / "verified_no_changes_outcome.md"
)
_BRANCH = "agent/issue-56-integration-verification"
_NUMBER = 56
_DETAIL = "ran synthetic, contract, adversarial and slow sets: 412 passed"


# ---------------------------------------------------------------------------
# Prompt <-> consumer pin
# ---------------------------------------------------------------------------


def _json_examples(text: str) -> list[dict[str, Any]]:
    return [json.loads(block) for block in re.findall(r"```json\n(.*?)\n```", text, re.DOTALL)]


def test_prompt_section_example_uses_the_consumer_literal() -> None:
    (example,) = _json_examples(_SECTION.read_text(encoding="utf-8"))
    assert example["outcome"] == VERIFIED_NO_CHANGES_OUTCOME
    # The consumer accepts exactly what the brief tells the worker to write.
    assert is_verified_no_changes(example)
    assert "detail" in example


@pytest.mark.parametrize("template_name", ["worker.md", "worker_claude_code.md"])
def test_rendered_worker_brief_carries_the_outcome_next_to_blocked(
    tmp_path: Path, template_name: str
) -> None:
    config = OrchestratorConfig(
        dispatch=DispatchConfig(worker_template=template_name),
        runtime=RuntimeConfig(state_dir="custom-state"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = _App(tmp_path, paths, config, gh=None)

    rendered = app._write_worker_prompt(
        {"number": 7, "title": "t", "body": "b", "url": "https://example.test/7"}
    ).read_text(encoding="utf-8")

    assert f'"outcome": "{VERIFIED_NO_CHANGES_OUTCOME}"' in rendered
    assert rendered.index("Worker-declared blocked outcome") < rendered.index(
        "Worker-declared verified-no-changes outcome"
    )
    assert "$section_" not in rendered


def test_non_matching_outcomes_are_not_verified() -> None:
    assert not is_verified_no_changes(None)
    assert not is_verified_no_changes({"outcome": "blocked"})
    assert not is_verified_no_changes({"push_succeeded": True})
    assert VERIFIED_NO_CHANGES_OUTCOME in resolution_comment("x")


def test_close_request_is_a_registered_request_type() -> None:
    assert CloseVerifiedNoChanges in REQUEST_TYPES


# ---------------------------------------------------------------------------
# Sweep behaviour (real git worktree)
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _repo_with_worktree(tmp_path: Path, config: OrchestratorConfig) -> Path:
    """Repo + bare origin, and ``_BRANCH``'s worktree cut at the base (no commits)."""
    remote = tmp_path / "remote"
    remote.mkdir()
    _git(["init", "--bare", str(remote)], cwd=tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "--initial-branch=main", str(repo)], cwd=tmp_path)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["add", "README.md"], cwd=repo)
    _git(["commit", "-m", "initial"], cwd=repo)
    _git(["remote", "add", "origin", str(remote)], cwd=repo)
    _git(["push", "-u", "origin", "main"], cwd=repo)
    _git(["remote", "set-head", "origin", "main"], cwd=repo)
    worktree = worktree_path_for_branch(repo, _BRANCH, resolved_layout(config, repo).worktrees)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    _git(["worktree", "add", "-b", _BRANCH, str(worktree), "main"], cwd=repo)
    (worktree / ".worker-outcome.json").write_text(
        json.dumps({"outcome": VERIFIED_NO_CHANGES_OUTCOME, "detail": _DETAIL}),
        encoding="utf-8",
    )
    return worktree


def _bed(tmp_path: Path, *, labels: tuple[str, ...] | None = None):
    config, paths, gh = no_pr_bed(
        tmp_path,
        _NUMBER,
        repo_root=tmp_path / "repo",
        labels=labels,
        branch_name=_BRANCH,
        dispatched_at=iso(minutes_ago=30),
    )
    worktree = _repo_with_worktree(tmp_path, config)
    return config, paths, gh, worktree


def test_clean_zero_diff_completion_closes_the_issue_without_redispatch(tmp_path: Path) -> None:
    config, paths, gh, _ = _bed(tmp_path)

    run_sweep(tmp_path, paths, config, gh)

    assert gh.closed_issues == [_NUMBER]
    (comment_number, body) = gh.issue_comments_posted[0]
    assert comment_number == _NUMBER
    assert _DETAIL in body
    (event,) = events_of(paths, "worker_verified_no_changes")
    assert event["payload"]["issue_number"] == _NUMBER
    assert event["payload"]["detail"] == _DETAIL
    assert (_NUMBER, config.labels.in_progress) in gh.labels_removed
    assert events_of(paths, "session_failed_relabeled") == []
    assert events_of(paths, "worker_verified_no_changes_ignored") == []
    entry = issue_entry(paths, _NUMBER)
    assert entry["status"] == "closed"
    # No redispatch bookkeeping: the #1243 cap never saw this issue.
    assert "orphan_redispatch_at" not in entry
    assert (_NUMBER, config.labels.ready) not in gh.labels_added


def _assert_refused(paths: Any, gh: Any, reason: str) -> None:
    assert gh.closed_issues == []
    assert events_of(paths, "worker_verified_no_changes") == []
    (ignored,) = events_of(paths, "worker_verified_no_changes_ignored")
    assert ignored["payload"]["reason"] == reason


def test_commits_ahead_of_base_are_ignored(tmp_path: Path) -> None:
    config, paths, gh, worktree = _bed(tmp_path)
    (worktree / "fix.txt").write_text("fix\n", encoding="utf-8")
    _git(["add", "fix.txt"], cwd=worktree)
    _git(["commit", "-m", "fix: real work"], cwd=worktree)

    run_sweep(tmp_path, paths, config, gh)

    _assert_refused(paths, gh, "worktree_completed")


def test_uncommitted_source_changes_are_ignored(tmp_path: Path) -> None:
    config, paths, gh, worktree = _bed(tmp_path)
    (worktree / "README.md").write_text("edited\n", encoding="utf-8")

    run_sweep(tmp_path, paths, config, gh)

    _assert_refused(paths, gh, "worktree_partial")


def test_escalated_issue_is_ignored(tmp_path: Path) -> None:
    config, paths, gh, _ = _bed(tmp_path, labels=("agent:in-progress", "agent:human-needed"))
    assert config.labels.human_needed == "agent:human-needed"

    run_sweep(tmp_path, paths, config, gh)

    _assert_refused(paths, gh, "escalated")


def test_missing_worktree_is_ignored(tmp_path: Path) -> None:
    config, paths, gh, worktree = _bed(tmp_path)
    # The outcome survives only via the terminal record; the worktree is gone.
    _git(["worktree", "remove", "--force", str(worktree)], cwd=tmp_path / "repo")
    terminal = sessions_dir_for(tmp_path) / f"issue-{_NUMBER}.claude.terminal.json"
    terminal.write_text(
        json.dumps(
            {
                "pid": 99999,
                "exit_code": 0,
                "started_at": iso(minutes_ago=25),
                "ended_at": iso(minutes_ago=1),
                "worker_outcome": {"outcome": VERIFIED_NO_CHANGES_OUTCOME, "detail": _DETAIL},
                "worker_outcome_written_at": iso(minutes_ago=2),
            }
        ),
        encoding="utf-8",
    )

    run_sweep(tmp_path, paths, config, gh)
    _assert_refused(paths, gh, "worktree_unavailable")


def test_verified_close_wins_over_a_stale_pushed_branch(tmp_path: Path) -> None:
    """A fresh claim plus a remote branch ahead of base yields ONE winner: the close.

    The pushed-orphan candidate lane must not open a salvage PR on the issue the
    claim just closed (the stale branch is left for the operator).
    """
    config, paths, gh, worktree = _bed(tmp_path)
    (worktree / "stale.txt").write_text("stale\n", encoding="utf-8")
    _git(["add", "stale.txt"], cwd=worktree)
    _git(["commit", "-m", "wip: stale push"], cwd=worktree)
    _git(["push", "origin", _BRANCH], cwd=worktree)
    # The worktree is back at the base; only the remote branch is ahead.
    _git(["reset", "--hard", "main"], cwd=worktree)

    run_sweep(tmp_path, paths, config, gh)

    assert gh.closed_issues == [_NUMBER]
    assert len(events_of(paths, "worker_verified_no_changes")) == 1
    assert events_of(paths, "orphaned_worker_opened_pr") == []
    assert events_of(paths, "worker_handoff_pr_opened") == []
    assert events_of(paths, "pr_create_failed_branch_stranded") == []
    entry = issue_entry(paths, _NUMBER)
    assert entry["status"] == "closed"
    assert "pr_number" not in entry
