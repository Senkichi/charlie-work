"""Shared fixtures for CLI-driving tests.

Hoisted out of ``test_cli.py`` (issue #1284): a stub GitHub client
sufficient to drive ``cli.main`` through the verdict path, and a minimal
repo-with-state.json builder, both imported by other test modules that
drive the CLI the same way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from charlie_work import github as github_module


class _FakeGitHub:
    """Stub GitHub client sufficient to drive cli.main through the verdict path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.pr_comment_calls: list[tuple[int, str]] = []

    def name_with_owner(self) -> str:
        return "owner/repo"

    def pr_view(self, number: int) -> dict[str, Any]:
        return {
            "number": number,
            "title": "Fix search",
            "url": "https://example.test/pull/1",
            "headRefName": "agent/issue-1-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #1\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }

    def pr_diff(self, number: int) -> str:
        return "diff content"

    def pr_comment(self, number: int, body_file: Path) -> None:
        # Issue #1268 (W11): record_review's post_verdict_comment default
        # (True) means every CLI verdict now posts a PR comment via this
        # method -- mirrors _CapturingGitHub in test_review_pr_comment.py.
        self.pr_comment_calls.append((number, body_file.read_text(encoding="utf-8")))

    def run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):
        return [] if json_output else ""

    def add_issue_label(self, number: int, label: str) -> bool:
        return True

    def remove_issue_label(self, number: int, label: str) -> bool:
        return True

    def commit(self, sha: str) -> github_module.GitHubRunResult:
        # This stub does not model commit metadata, so the committer-date
        # timestamp cannot be resolved. Returning a failed GitHubRunResult
        # (errors-as-values invariant) makes _commit_timestamp yield None,
        # which tells _collect_external_findings to skip the upper bound and
        # fail toward ingestion -- a no-op here, since ``run`` returns ``[]``
        # for JSON output so no external comments are ever surfaced. These
        # tests exercise required_changes derivation, not external-findings
        # filtering (see test_charlie_work.py for that coverage).
        return github_module.GitHubRunResult(
            ok=False,
            returncode=1,
            stdout="",
            stderr="",
            value=None,
            error=f"commit {sha} not modeled by _FakeGitHub",
        )


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    state_dir = tmp_path / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    return tmp_path
