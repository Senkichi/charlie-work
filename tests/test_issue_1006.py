"""Regression coverage for issue #1006.

Issue #1006: three call sites in ``workflow.py`` pass a possibly-``None`` value
into a non-Optional parameter. The fix makes the invariants structural and
removes the pyright ``reportArgumentType`` findings, without using ``cast`` or
``# type: ignore``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from charlie_work.config import (
    DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    OrchestratorConfig,
)
from charlie_work.workflow import _detect_and_handle_orphaned_workers

from test_charlie_work import FakeGitHub


def test_none_not_in_deterministic_escalation_failure_kinds() -> None:
    """The two ``reason = failure_kind if terminal_failure else ...`` sites in
    ``workflow.py`` would pass ``None`` into ``_escalate_issue(..., reason: str)``
    if ``None`` were ever a member of this set. Pin the invariant explicitly so
    future edits to the set cannot silently reopen the type hole.
    """
    assert None not in DETERMINISTIC_ESCALATION_FAILURE_KINDS


@pytest.mark.parametrize(
    ("repo_root", "expect_called"),
    [
        (None, False),
        ("a/string/path", False),
        ("a_real_path", True),
    ],
    ids=["repo_root_none", "repo_root_string", "repo_root_path"],
)
def test_orphan_salvage_repo_root_guard(
    repo_root: Any,
    expect_called: bool,
    tmp_path: Path,
) -> None:
    """The no-open-PR orphan salvage path only reaches
    ``_open_pr_for_orphaned_branch`` when ``repo_root`` is an actual,
    existing ``pathlib.Path``.

    The two invalid cases verify the structural guard; removing the guard
    causes ``_open_pr_for_orphaned_branch`` to be called with a non-``Path``
    value and the patched helper below raises. The ``path`` case covers the
    reachable, valid path and asserts the real path is passed through.
    """
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    state_file = tmp_path / "state.json"
    config = OrchestratorConfig()

    issue_number = 1006
    branch = "agent/issue-1006-test"
    state = {
        "issues": {
            str(issue_number): {
                "status": "dispatched",
                "worker_pid": 99999,
                "branch_name": branch,
            }
        }
    }
    state_file.write_text(json.dumps(state), encoding="utf-8")

    terminal = {
        "pid": 99999,
        "exit_code": 0,
        "started_at": "2024-01-01T00:00:00Z",
        "ended_at": "2024-01-01T00:00:01Z",
        "duration_seconds": 1.0,
        "worker_outcome": {
            "push_succeeded": True,
            "pr_created": False,
        },
    }
    (sessions_dir / f"issue-{issue_number}.claude-code.terminal.json").write_text(
        json.dumps(terminal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if repo_root == "a_real_path":
        repo_root_value: Any = tmp_path
    else:
        repo_root_value = repo_root

    fake_gh = FakeGitHub(repo_root=repo_root_value)
    fake_gh.issues = [
        {
            "number": issue_number,
            "title": "Test issue",
            "url": f"https://example.test/issues/{issue_number}",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    calls: list[tuple[Path, ...]] = []

    def fake_open_pr(*, repo_root: Path, **kwargs: Any) -> tuple[int | None, str | None]:
        if not isinstance(repo_root, Path):
            raise AssertionError(
                f"_open_pr_for_orphaned_branch called with non-Path repo_root: {repo_root!r}"
            )
        if not repo_root.exists():
            raise AssertionError(
                f"_open_pr_for_orphaned_branch called with non-existent repo_root: {repo_root}"
            )
        calls.append((repo_root,))
        return (101, None)

    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch("charlie_work.workflow._open_pr_for_orphaned_branch", side_effect=fake_open_pr),
    ):
        _detect_and_handle_orphaned_workers(sessions_dir, state_file, config, fake_gh)

    assert len(calls) == (1 if expect_called else 0)
    if expect_called:
        assert calls[0][0] == tmp_path
