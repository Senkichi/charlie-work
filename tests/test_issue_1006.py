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
from charlie_work.state import PASSIVE_OPEN_STATUS, load_state
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
    ("repo_root_value", "is_valid_path"),
    [
        (None, False),
        ("a/string/path", False),
        ("tmp_path", True),
    ],
    ids=["repo_root_none", "repo_root_string", "repo_root_path"],
)
def test_orphan_salvage_repo_root_guard(
    repo_root_value: Any,
    is_valid_path: bool,
    tmp_path: Path,
) -> None:
    """The no-open-PR orphan salvage path narrows ``repo_root`` to ``Path | None``.

    ``getattr(gh, "repo_root", None)`` is not statically typed, so a non-``Path``
    value is treated the same as ``None``: ``_open_pr_for_orphaned_branch`` is
    still called, but with ``repo_root=None``. The helper already handles ``None``
    by returning an error, which preserves the pre-#1041 drift/hold semantics for
    a missing repo root.

    The invalid cases also serve as a guard test: if the ``isinstance`` narrowing
    were removed and a string were passed through, the patched helper below
    raises. The ``path`` case verifies the real ``Path`` is passed through and the
    worker branch is salvaged into a passively-opened PR.
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

    if repo_root_value == "tmp_path":
        actual_repo_root: Any = tmp_path
    else:
        actual_repo_root = repo_root_value

    fake_gh = FakeGitHub(repo_root=actual_repo_root)
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

    calls: list[Any] = []

    def fake_open_pr(
        *,
        repo_root: Any,
        **kwargs: Any,
    ) -> tuple[int | None, str | None, Any]:
        # Record every invocation first, before any guard logic, so the test can
        # assert the actual value that reached the helper.
        calls.append(repo_root)
        if repo_root is not None and not isinstance(repo_root, Path):
            raise AssertionError(
                f"_open_pr_for_orphaned_branch called with non-Path repo_root: {repo_root!r}"
            )
        if repo_root is None:
            # Mirror the real helper's None handling: it returns an error so the
            # caller follows the existing salvage-failure drift path.
            return (None, "repo_root is required to open a salvage PR", None)
        if not repo_root.exists():
            raise AssertionError(
                f"_open_pr_for_orphaned_branch called with non-existent repo_root: {repo_root}"
            )
        return (101, None, None)

    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch("charlie_work.workflow._open_pr_for_orphaned_branch", side_effect=fake_open_pr),
    ):
        _detect_and_handle_orphaned_workers(sessions_dir, state_file, config, fake_gh)

    state = load_state(state_file)
    issue_state = state["issues"][str(issue_number)]

    # cw#1273: this specific reason now emits its own kind
    # (pr_create_failed_branch_stranded) instead of the generic
    # orphaned_worker_drift, after the bounded outer retry exhausted.
    drift_events = [
        e
        for e in state.get("events", [])
        if e.get("kind") == "pr_create_failed_branch_stranded"
        and e.get("payload", {}).get("reason") == "dead_worker_branch_pushed_pr_create_failed"
    ]
    opened_events = [
        e for e in state.get("events", []) if e.get("kind") == "orphaned_worker_opened_pr"
    ]
    relabel_events = [
        e for e in state.get("events", []) if e.get("kind") == "session_failed_relabeled"
    ]

    if is_valid_path:
        assert calls == [tmp_path]
        assert len(opened_events) == 1
        assert opened_events[0]["payload"]["pr_number"] == 101
        assert opened_events[0]["payload"]["issue_number"] == issue_number
        assert len(drift_events) == 0
        assert len(relabel_events) == 0
        assert issue_state["status"] == PASSIVE_OPEN_STATUS
        assert issue_state["pr_number"] == 101
    else:
        assert calls == [None]
        assert len(opened_events) == 0
        assert len(relabel_events) == 0
        assert len(drift_events) == 1
        assert drift_events[0]["payload"]["issue_number"] == issue_number
        # The issue is held as drift, not silently relabeled/reopened.
        assert issue_state["status"] == "dispatched"
        assert issue_state.get("orphan_drift_fingerprint") is not None
