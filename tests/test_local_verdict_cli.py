"""``charlie verdict`` on a no-remote (local-file) repo (issue #2095).

``record_review`` reads the PR through ``gh.pr_view``; ``LocalFileGitHub``
returned ``{}``, so ``_write_rework_prompt`` raised ``KeyError: 'number'``.
The fix routes ``record_review`` to ``record_local_review`` on a backend that
publishes no pull requests (the #1844 pattern), so it never reads ``pr_view``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _local_lane_fixtures import (
    _app,
    _init_repo,
    _make_branch,
    _parked_issue,
    new_repo_root,
)
from charlie_work import cli
from charlie_work.state import load_state


@pytest.fixture
def repo() -> Path:
    return new_repo_root()


def _verdict_args(pr: int, head: str, summary_file: Path):
    return cli.build_parser().parse_args(
        [
            "verdict",
            "--pr",
            str(pr),
            "--decision",
            "request_changes",
            "--reviewed-head",
            head,
            "--summary-file",
            str(summary_file),
        ]
    )


def test_verdict_request_changes_writes_rework_prompt(repo: Path, tmp_path: Path) -> None:
    _init_repo(repo)
    issues_dir = repo / "docs" / "issues"
    head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _app(repo, issues_dir)
    _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
    app._local_review_packets()
    summary_file = tmp_path / "f.md"
    summary_file.write_text("Add a test for the new function.\n", encoding="utf-8")

    args = cli.build_parser().parse_args(
        [
            "verdict",
            "--pr",
            "7",
            "--decision",
            "request_changes",
            "--reviewed-head",
            head,
            "--summary-file",
            str(summary_file),
        ]
    )
    result = cli.run_command(app, args)

    assert result.ok, result.message
    assert (app.paths.prs / "pr-7" / "rework-prompt.md").is_file()
    decision = json.loads(
        (app.paths.prs / "pr-7" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["decision"] == "request_changes"


def test_verdict_on_merged_local_record_is_refused(repo: Path, tmp_path: Path) -> None:
    _init_repo(repo)
    issues_dir = repo / "docs" / "issues"
    head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _app(repo, issues_dir)
    _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
    app._local_review_packets()
    state = load_state(app.paths.state_file)
    state["prs"]["7"]["status"] = "merged"
    app.write_gate.save_state(state)
    summary_file = tmp_path / "f.md"
    summary_file.write_text("Add a test.\n", encoding="utf-8")

    decision_path = app.paths.prs / "pr-7" / "review-decision.json"
    before = decision_path.read_bytes()  # the pending decision the packet pass wrote

    result = cli.run_command(app, _verdict_args(7, head, summary_file))

    assert not result.ok
    assert result.data.get("terminal") is True
    assert decision_path.read_bytes() == before


def test_verdict_without_lane_record_does_not_mint_pr_record(repo: Path, tmp_path: Path) -> None:
    _init_repo(repo)
    issues_dir = repo / "docs" / "issues"
    head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _app(repo, issues_dir)
    _parked_issue(app, issues_dir, 7, "agent/issue-7-x")  # parked, no lane record yet
    summary_file = tmp_path / "f.md"
    summary_file.write_text("Add a test.\n", encoding="utf-8")

    result = cli.run_command(app, _verdict_args(7, head, summary_file))

    assert not result.ok
    assert "7" not in (load_state(app.paths.state_file).get("prs") or {})
