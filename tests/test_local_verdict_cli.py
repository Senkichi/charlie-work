"""``charlie verdict`` on a no-remote (local-file) repo (issue #2095).

``record_review`` reads the PR through ``gh.pr_view``; ``LocalFileGitHub``
returned ``{}``, so ``_write_rework_prompt`` raised ``KeyError: 'number'``.
The fix is at the seam: ``pr_view`` returns the ``gh pr view`` shape.
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


@pytest.fixture
def repo() -> Path:
    return new_repo_root()


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


def test_pr_view_shape_and_absent_issue(repo: Path) -> None:
    _init_repo(repo)
    issues_dir = repo / "docs" / "issues"
    head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
    app = _app(repo, issues_dir)
    _parked_issue(app, issues_dir, 7, "agent/issue-7-x")

    pr = app.gh.pr_view(7)

    assert pr["number"] == 7
    assert pr["headRefName"] == "agent/issue-7-x"
    assert pr["headRefOid"] == head
    assert app.gh.pr_view(999) == {}
