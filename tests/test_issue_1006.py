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
from charlie_work.write_gate import WriteGate

from _fakes_github import FakeGitHub
from _host_fixtures import host_probe


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


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
        pytest.param(None, False, id="repo_root_none"),
        pytest.param("a/string/path", False, id="repo_root_string"),
        pytest.param("tmp_path", True, id="repo_root_path"),
    ],
)
def test_orphan_salvage_repo_root_guard(
    repo_root_value: Any, is_valid_path: bool, tmp_path: Path, monkeypatch
) -> None:
    """The no-open-PR orphan salvage path narrows ``repo_root`` to ``Path | None``.

    ``getattr(gh, "repo_root", None)`` is not statically typed, so a non-``Path``
    value is treated the same as ``None``. Since worker-fate rule 9, a missing
    or invalid repo root means no git evidence can be gathered, and per
    ``wf-design.md`` section 4 ("An unknown value never proves a push") the
    worker's self-reported push is not credited: ``_open_pr_for_orphaned_branch``
    is never called for these cases.

    The invalid cases also serve as a guard test: if the fate ever credited the
    self-report without git evidence, the patched helper below would be called
    (and raise for a non-``Path``). The ``path`` case verifies the real ``Path``
    is passed through and the worker branch is salvaged into a passively-opened
    PR.

    worker-fate rule 9 (design doc Sec 9): the ``path`` case needs a real
    pushed branch -- admission into the pushed-branch salvage lane now
    requires git-confirmed evidence, not the worker's bare self-report, so
    ``repo_root`` must point at an actual repo with the branch actually ahead
    of base for ``resolve_fate`` to reach ``PushedWithoutPr`` at all. The
    ``none``/``string`` cases are unaffected: they were never meant to reach
    real git evidence, only to prove the ``isinstance`` narrowing.
    """
    import subprocess

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
        actual_repo_root: Any = tmp_path / "repo"
        remote_repo = tmp_path / "remote"
        remote_repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init", "--bare", str(remote_repo)],
            check=True,
            capture_output=True,
            text=True,
        )
        actual_repo_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init", "--initial-branch=main", str(actual_repo_root)],
            check=True,
            capture_output=True,
            text=True,
        )
        for cmd in (
            ["git", "config", "user.email", "test@example.test"],
            ["git", "config", "user.name", "Test User"],
        ):
            subprocess.run(cmd, cwd=actual_repo_root, check=True, capture_output=True, text=True)
        (actual_repo_root / "README.md").write_text("hello\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "README.md"],
            cwd=actual_repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=actual_repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "remote", "add", "origin", str(remote_repo)],
            cwd=actual_repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=actual_repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "checkout", "-b", branch],
            cwd=actual_repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        (actual_repo_root / "fix.txt").write_text("fix\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "fix.txt"],
            cwd=actual_repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "fix"],
            cwd=actual_repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "push", "-u", "origin", branch],
            cwd=actual_repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=actual_repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
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
        host_probe(monkeypatch, alive=False),
        patch("charlie_work.workflow._open_pr_for_orphaned_branch", side_effect=fake_open_pr),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir, state_file, config, fake_gh, write_gate=_wg(state_file)
        )

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
    # cw#1771 steps 4-6: this scenario's worker_outcome confirms
    # push_succeeded=True/pr_created=False, so a successful open now emits the
    # honestly-named additive ``worker_handoff_pr_opened`` kind rather than
    # the anomaly-path ``orphaned_worker_opened_pr`` (which stays reserved for
    # the no-outcome-file, ahead-count-only case).
    handoff_events = [
        e for e in state.get("events", []) if e.get("kind") == "worker_handoff_pr_opened"
    ]
    opened_events = [
        e for e in state.get("events", []) if e.get("kind") == "orphaned_worker_opened_pr"
    ]
    relabel_events = [
        e for e in state.get("events", []) if e.get("kind") == "session_failed_relabeled"
    ]

    if is_valid_path:
        assert calls == [actual_repo_root]
        assert len(handoff_events) == 1
        assert handoff_events[0]["payload"]["pr_number"] == 101
        assert handoff_events[0]["payload"]["issue_number"] == issue_number
        assert handoff_events[0]["payload"]["reason"] == "worker_handoff_clean_exit"
        assert len(opened_events) == 0
        assert len(drift_events) == 0
        assert len(relabel_events) == 0
        assert issue_state["status"] == PASSIVE_OPEN_STATUS
        assert issue_state["pr_number"] == 101
    else:
        # wf-design.md section 4: "An unknown value never proves a push."
        # Without a usable repo_root no git evidence can exist, so the
        # worker's bare self-report (push_succeeded=True) is not credited:
        # the fate resolves to Crashed, the PR-open helper is never
        # attempted, and the issue is relabelled for redispatch instead of
        # being held as pr-create drift.
        assert calls == []
        assert len(handoff_events) == 0
        assert len(opened_events) == 0
        assert len(drift_events) == 0
        assert len(relabel_events) == 1
        assert relabel_events[0]["payload"]["issue_number"] == issue_number
        assert relabel_events[0]["payload"]["reason"] == "dead_worker_no_open_pr_orphan_sweep"
        assert issue_state.get("orphan_drift_fingerprint") is None
