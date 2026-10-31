"""Shared builders for ``tests/test_harness_auth_failure.py`` and
``tests/test_harness_auth_resume.py`` (issue #2810).

A three-issue fake repo whose worker harness is ``devin-shell`` (so a harness
block on it is what the launch gate must react to), a registry helper, the
ledger-side helpers that arm / read a harness block, and a dead devin session
laid out the way the real launcher leaves it (sidecar + log + optional terminal
record). Nothing here fakes the code under test: the ledger, the selection, the
permit and the fleet pass all run for real.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from charlie_work import layout, role_quota_ledger
from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import (
    DevinConfig,
    DispatchConfig,
    FleetConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.role_chain import RoleEntry
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

HARNESS = "devin-shell"
ISSUES = (123, 124, 125)
DEVIN_PRIMARY = RoleEntry(HARNESS, "swe-2")
CLAUDE_FALLBACK = RoleEntry("claude-code", "claude-sonnet-5-5")
CHAIN = (DEVIN_PRIMARY, CLAUDE_FALLBACK)
NOT_LOGGED_IN = "Error: Not logged in. Run `devin auth login` to authenticate"
INVALID_KEY = "Error: Authentication required: invalid api key"
EVIDENCE = NOT_LOGGED_IN


class MultiIssueGitHub(FakeGitHub):
    """Three eligible issues; open PRs (rework) or closed PRs (fresh dispatch)."""

    def __init__(self, rework_label: str | None = None) -> None:
        super().__init__()
        label = rework_label or self.issues[0]["labels"][0]["name"]
        self.issues = [
            {**self.issues[0], "number": n, "title": f"Issue {n}", "labels": [{"name": label}]}
            for n in ISSUES
        ]
        self.prs = [
            {
                **self.prs[0],
                "number": n + 333,
                "title": f"Fix #{n}",
                "headRefName": f"agent/issue-{n}-fix",
                "headRefOid": f"sha-{n}",
                "body": f"Closes #{n}\n\nTests: covered.",
                "state": "CLOSED" if rework_label is None else "OPEN",
            }
            for n in ISSUES
        ]


def make_config(
    worker: WorkerRoleConfig | None = None, *, enabled: bool = True
) -> OrchestratorConfig:
    return OrchestratorConfig(
        # The host-load terms default to a function of ``os.cpu_count()`` (trees =
        # cpu // 2): on a 4-vCPU runner that is 2, which clamps a 3-issue wave to 2
        # launches and fails every "all three launched" assertion. Off, the launch
        # count is the host-independent one these tests mean to pin.
        dispatch=DispatchConfig(
            default_limit=3, host_load_max_pytest_processes=0, host_load_max_pytest_trees=0
        ),
        devin=DevinConfig(),
        worker=worker or WorkerRoleConfig(harness=HARNESS),
        fleet=FleetConfig(harness_auth_block_enabled=enabled),
    )


def make_app(
    tmp_path: Path,
    name: str,
    fleet: Path,
    *,
    rework: bool = False,
    worker: WorkerRoleConfig | None = None,
    enabled: bool = True,
):
    """A repo ``name`` with three launchable issues; returns ``(app, paths)``."""
    root = tmp_path / name
    root.mkdir()
    config = make_config(worker, enabled=enabled)
    paths = runtime_paths(root, config.runtime.state_dir)
    gh = MultiIssueGitHub(config.labels.needs_rework if rework else None)
    if rework:
        paths.root.mkdir(parents=True, exist_ok=True)
        with state_lock(paths.state_file):
            state = load_state(paths.state_file)
            for n in ISSUES:
                state["issues"][str(n)] = {
                    "number": n,
                    "title": f"Issue {n}",
                    "url": f"https://example.test/issues/{n}",
                    "status": "rework_requested",
                    "branch_name": f"agent/issue-{n}-fix",
                }
                state["prs"][str(n + 333)] = {
                    "number": n + 333,
                    "issue_number": n,
                    "decision": "request_changes",
                    "reviewed_head_sha": f"sha-{n}",
                }
            save_state(paths.state_file, state)
        for n in ISSUES:
            pr_dir = root / ".var" / "charlie-work" / "prs" / f"pr-{n + 333}"
            pr_dir.mkdir(parents=True)
            (pr_dir / "rework-prompt.md").write_text("Fix it", encoding="utf-8")
    app = OrchestratorApp(root, paths, config, gh, fleet_dir_override=str(fleet))
    return app, paths


def register(fleet: Path, *paths_list: Any) -> None:
    fleet.mkdir(parents=True, exist_ok=True)
    repos = {f"o/r{i}": {"state_dir": str(p.state_file.parent)} for i, p in enumerate(paths_list)}
    (fleet / "fleet.json").write_text(json.dumps({"repos": repos}), encoding="utf-8")


def run_lane(app: Any, lane: str) -> Any:
    return app.dispatch(limit=3) if lane == "dispatch" else app.dispatch_rework(limit=3)


LANES = pytest.mark.parametrize("lane", ["dispatch", "rework"])


def fake_launch(launched: list[tuple[int, str]]):
    """A ``dispatch_sessions`` fake recording ``(issue, adapter)``; every launch lives."""

    def _dispatch(_root, _manifest, _results, settings, requests):
        launched.extend((r.issue_number, settings.adapter) for r in requests)
        return [
            SessionDispatchResult(
                issue_number=r.issue_number,
                issue_title=r.issue_title,
                prompt_path=str(r.prompt_path),
                branch_name=r.branch_name,
                adapter=settings.adapter,
                ok=True,
                pid=os.getpid(),
            )
            for r in requests
        ]

    return _dispatch


def arm(
    harness: str = HARNESS,
    *,
    age: timedelta = timedelta(0),
    escalate: bool = True,
    repo: str | None = "o/r0",
    issue: int | None = 123,
    evidence: str | None = EVIDENCE,
) -> role_quota_ledger.HarnessBlock:
    """Arm a harness block ``age`` ago: ``age`` past the base window means it has lapsed."""
    outcome = role_quota_ledger.arm_harness_block(
        harness,
        base_window=role_quota_ledger.HARNESS_AUTH_BASE_WINDOW,
        evidence=evidence,
        source="test",
        now=datetime.now(UTC) - age,
        escalate=escalate,
        affected=(repo, issue) if repo is not None and issue is not None else None,
    )
    assert outcome.block is not None
    return outcome.block


LAPSED = role_quota_ledger.HARNESS_AUTH_BASE_WINDOW + timedelta(minutes=30)


def block() -> role_quota_ledger.HarnessBlock:
    found = role_quota_ledger.read_harness_block(HARNESS)
    assert found is not None
    return found


def dead_devin_session(
    tmp_path: Path,
    log_text: str,
    *,
    pid: int = 4242,
    terminal: dict[str, Any] | None = None,
    terminal_pid: int | None = None,
    issue: int = 123,
    worktrees_root: Path | None = None,
) -> Path:
    """Lay out an exited devin session; returns its ``sessions_dir``.

    ``terminal`` is the terminal-status record's body (``exit_code`` /
    ``worker_outcome``); it is written for ``terminal_pid`` (default: ``pid``).
    ``worktrees_root`` is where the session's worktree lives; by default the
    repo's real default-layout root, ``<repo>/.var/charlie-work/worktrees``
    (what ``paths.runtime_paths`` hands dispatch), not a shape no launch produces.
    """
    state_root = layout.default_state_root(tmp_path / "repo")
    worktrees_root = worktrees_root or layout.worktrees_dir(state_root)
    sessions_dir = layout.sessions_dir_default(state_root)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log = sessions_dir / f"issue-{issue}.log"
    log.write_text(log_text, encoding="utf-8")
    (sessions_dir / f"issue-{issue}.json").write_text(
        json.dumps(
            {
                "issue_number": issue,
                "pid": pid,
                "log_path": str(log),
                "worktree_path": str(worktrees_root / f"agent-issue-{issue}"),
                "role_entry": role_quota_ledger.session_stamp("worker", HARNESS, "swe-2", 0),
            }
        ),
        encoding="utf-8",
    )
    if terminal is not None:
        record = {"pid": terminal_pid if terminal_pid is not None else pid, **terminal}
        (sessions_dir / f"issue-{issue}.devin.terminal.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    return sessions_dir
