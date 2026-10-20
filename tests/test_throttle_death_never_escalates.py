"""Issue #1993 point 2: a provider-throttle death never escalates.

A ``rate_limited`` / ``quota_exhausted`` death says nothing about the work,
yet on 2026-09-29 three lanes escalated such deaths after the #1917
window-bounded exemption had lapsed:

- the #654 timed backstop (``dead_dispatched_worker_reap``: #1976, #1993),
- the #1153 zero-artifact loop guard (``zero_artifact_dispatch_loop``: #1983),
- the rework death-loop cap (``worker_death_loop``: #1971).

These tests pin each lane's throttle handling, both while the window is open
and after it has expired, plus the bounded re-arm that keeps the #654 wedge
guarantee.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _dws_facts import run_reap
from _host_fixtures import host_probe

from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import clear_dead_worker_failure_kind, load_state, save_state
from charlie_work.unescalate_reset_fields import UNESCALATE_ISSUE_RESET_FIELDS
from charlie_work.write_gate import WriteGate

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
REAP_MINUTES = 60
MAX_REARMS = 3


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _reap(
    entry: dict[str, Any],
    *,
    throttled_until: datetime | None,
    pr_data: dict[str, Any] | None = None,
    tmp_path: Path,
) -> tuple[dict[str, Any], bool, list[tuple[str, dict[str, Any]]]]:
    """Decide the #654 backstop for ``entry``; return (state view, reaped, events).

    The state view is the entry after the decision: re-arm mutations from the
    draft, plus the escalation the ``Escalate`` commit asks the shell to apply.
    """
    from charlie_work.dead_worker_sweep.model import Escalate

    run = run_reap(
        entry,
        issue=42,
        pr_data=pr_data,
        reap_minutes=REAP_MINUTES,
        max_rearms=MAX_REARMS,
        throttled_until=None if throttled_until is None else _iso(throttled_until),
        now=NOW,
    )
    stored = dict(run.entry)
    for commit in run.commits:
        if isinstance(commit, Escalate):
            stored.update(status="escalated", escalation_reason=commit.reason)
    return {"issues": {"42": stored}, "prs": {}}, run.reaped, [(k, dict(p)) for k, p in run.events]


def _entry(kind: str | None, *, drift_minutes_ago: int, rearms: int = 0) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "orphan_drift_at": _iso(NOW - timedelta(minutes=drift_minutes_ago)),
    }
    if kind is not None:
        entry["dead_worker_failure_kind"] = kind
    if rearms:
        entry["throttle_reap_rearm_count"] = rearms
    return entry


# -- #654 timed backstop -----------------------------------------------------


@pytest.mark.parametrize("kind", ["rate_limited", "quota_exhausted"])
def test_backstop_grace_runs_from_window_close_not_drift(tmp_path: Path, kind: str) -> None:
    """#1976's shape: drift 05:30, window closed 06:28, reaped 06:32.

    Four dispatchable minutes is not a 60-minute grace, so the backstop
    must measure from the window close, with or without a linked PR.
    """
    for pr_data in (None, {"number": 7}):
        entry = _entry(kind, drift_minutes_ago=120)
        state, reaped, events = _reap(
            entry,
            throttled_until=NOW - timedelta(minutes=4),
            pr_data=pr_data,
            tmp_path=tmp_path,
        )
        assert reaped is False
        assert events == []
        assert state["issues"]["42"].get("status") == "dispatched"


def test_backstop_open_window_still_exempt(tmp_path: Path) -> None:
    entry = _entry("rate_limited", drift_minutes_ago=500)
    _, reaped, events = _reap(
        entry, throttled_until=NOW + timedelta(minutes=30), tmp_path=tmp_path
    )
    assert reaped is False
    assert events == []


@pytest.mark.parametrize("window_closed_minutes_ago", [90, None], ids=["expired", "never-armed"])
def test_backstop_rearms_instead_of_escalating_without_pr(
    tmp_path: Path, window_closed_minutes_ago: int | None
) -> None:
    """No PR: the issue is back in the pool, so the clock re-arms."""
    entry = _entry("rate_limited", drift_minutes_ago=120)
    throttled_until = (
        None
        if window_closed_minutes_ago is None
        else NOW - timedelta(minutes=window_closed_minutes_ago)
    )
    state, reaped, events = _reap(entry, throttled_until=throttled_until, tmp_path=tmp_path)

    assert reaped is False
    stored = state["issues"]["42"]
    assert stored.get("status") == "dispatched"
    assert "escalation_reason" not in stored
    assert stored["orphan_drift_at"] == _iso(NOW)
    assert stored["throttle_reap_rearm_count"] == 1
    assert [kind for kind, _ in events] == ["dead_dispatched_throttle_rearmed"]
    payload = events[0][1]
    assert payload["issue_number"] == 42
    assert payload["rearm_count"] == 1
    assert payload["max_rearms"] == MAX_REARMS


def test_backstop_escalates_once_rearms_exhausted(tmp_path: Path) -> None:
    """The #654 wedge guarantee survives: a throttle-held issue that no
    dispatch picks up across ``max_throttle_rearms`` grace windows escalates.
    """
    entry = _entry("rate_limited", drift_minutes_ago=120, rearms=MAX_REARMS)
    state, reaped, events = _reap(
        entry, throttled_until=NOW - timedelta(minutes=90), tmp_path=tmp_path
    )
    assert reaped is True
    assert state["issues"]["42"]["escalation_reason"] == "dead_dispatched_worker_reap"
    assert [kind for kind, _ in events] == ["dead_dispatched_worker_reaped"]


def test_backstop_pr_linked_throttle_escalates_after_anchored_grace(tmp_path: Path) -> None:
    """PR-linked: the branch mutex is held, so re-arming would only postpone
    the wedge. Past the anchored grace it escalates as before (#1917)."""
    entry = _entry("rate_limited", drift_minutes_ago=180)
    _, reaped, events = _reap(
        entry,
        throttled_until=NOW - timedelta(minutes=90),
        pr_data={"number": 7},
        tmp_path=tmp_path,
    )
    assert reaped is True
    assert [kind for kind, _ in events] == ["dead_dispatched_worker_reaped"]


@pytest.mark.parametrize("kind", [None, "stalled", "worker_blocked"])
def test_backstop_non_throttle_death_escalates_without_rearm(
    tmp_path: Path, kind: str | None
) -> None:
    """Control: nothing changes for a death that is not a provider throttle,
    even with a (stale) window stamped on the repo."""
    entry = _entry(kind, drift_minutes_ago=120)
    state, reaped, events = _reap(
        entry, throttled_until=NOW - timedelta(minutes=4), tmp_path=tmp_path
    )
    assert reaped is True
    assert "throttle_reap_rearm_count" not in state["issues"]["42"]
    assert [k for k, _ in events] == ["dead_dispatched_worker_reaped"]


def test_rearm_counter_dies_with_the_classification() -> None:
    """The counter is scoped to one throttle death: the dispatch epoch and
    an operator unescalate both drop it along with the stamp."""
    entry = {"dead_worker_failure_kind": "rate_limited", "throttle_reap_rearm_count": 2}
    clear_dead_worker_failure_kind(entry)
    assert entry == {}
    assert "throttle_reap_rearm_count" in UNESCALATE_ISSUE_RESET_FIELDS


# -- #1153 zero-artifact loop guard ------------------------------------------


def _write_zero_artifact_post_mortem(sessions_dir: Path, issue_number: int) -> None:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "issue_number": issue_number,
        "generated_at": _iso(datetime.now(UTC)),
        "db_path": "",
        "matched": False,
        "attempts": [
            {"ref": f"refs/charlie/attempts/{n}", "ahead_of_main": 0, "recorded_at": _iso(NOW)}
            for n in (1, 2)
        ],
    }
    (sessions_dir / f"issue-{issue_number}.post-mortem.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("kind", "expect_escalated"),
    [("rate_limited", False), ("quota_exhausted", False), (None, True)],
    ids=["rate_limited", "quota_exhausted", "control-unclassified"],
)
def test_zero_artifact_guard_skips_throttle_death(
    tmp_path: Path, kind: str | None, expect_escalated: bool, monkeypatch
) -> None:
    """#1983's shape: two zero-artifact attempts, the latest a rate-limit
    death. The guard must relabel for redispatch, not escalate; the
    unclassified control proves the fixture does trip the guard."""
    from _fakes_github import FakeGitHub

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    state = load_state(paths.state_file)
    entry: dict[str, Any] = {
        "status": "dispatched",
        "dispatched_at": "2026-09-29T12:43:54Z",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
    }
    if kind is not None:
        entry["dead_worker_failure_kind"] = kind
    state["issues"]["1983"] = entry
    save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_zero_artifact_post_mortem(sessions_dir, 1983)

    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 1983,
            "title": "some in-repo fix",
            "url": "https://example.test/issues/1983",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    with host_probe(monkeypatch, alive=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            fake_gh,
            write_gate=WriteGate(dry_run=False, state_path=paths.state_file, repo="charlie-work"),
        )

    stored = load_state(paths.state_file)["issues"]["1983"]
    if expect_escalated:
        assert stored["status"] == "escalated"
        assert stored["escalation_reason"] == "zero_artifact_dispatch_loop"
        assert (1983, config.labels.human_needed) in fake_gh.labels_added
    else:
        assert stored.get("status") != "escalated"
        assert stored.get("escalation_reason") is None
        assert (1983, config.labels.human_needed) not in fake_gh.labels_added
        assert (1983, config.labels.in_progress) in fake_gh.labels_removed


# -- rework caps (pre-dispatch safety net) -----------------------------------


@pytest.mark.parametrize(
    ("kind", "expect_escalated"),
    [("rate_limited", False), ("provider_auth", False), (None, True)],
    ids=["rate_limited", "provider_auth", "control-unclassified"],
)
def test_dispatch_rework_death_loop_skips_throttle_death(
    tmp_path: Path, kind: str | None, expect_escalated: bool
) -> None:
    """#1971's shape: the death-loop cap is already full when the latest
    death is a rate limit. That death must not be the one that escalates;
    the unclassified control proves the fixture does hit the cap.
    """
    import subprocess
    import sys

    from _fakes_github import FakeGitHub
    from _rework_dispatch_fixtures import _init_repo_with_remote_inline

    from charlie_work.paths import resolved_layout
    from charlie_work.state import state_lock
    from charlie_work.workflow import OrchestratorApp
    from charlie_work.worktree import push_branch, worktree_path_for_branch

    remote, repo_root = _init_repo_with_remote_inline(tmp_path)
    branch = "agent/issue-123-fix-search"

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, cwd=repo_root, check=True, capture_output=True, text=True)

    run(["git", "branch", branch])
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    layout = resolved_layout(config, repo_root)
    wt_path = worktree_path_for_branch(repo_root, branch, layout.worktrees)
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "worktree", "add", str(wt_path), branch])
    ok, error = push_branch(repo_root, branch, worktree_path=wt_path)
    assert ok, error
    pr_head_sha = run(["git", "rev-parse", branch]).stdout.strip()

    paths = runtime_paths(repo_root, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.repo_root = repo_root
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]
            self.prs[0]["headRefOid"] = pr_head_sha

    fake_gh = ReworkGitHub()
    now_iso = _iso(datetime.now(UTC))
    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        entry: dict[str, Any] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "redispatch_at": [now_iso, now_iso],
            "worker_death_at": [now_iso, now_iso],
            "branch_name": branch,
        }
        if kind is not None:
            entry["dead_worker_failure_kind"] = kind
        state["issues"]["123"] = entry
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": pr_head_sha,
        }
        save_state(paths.state_file, state)

    result = OrchestratorApp(repo_root, paths, config, fake_gh).dispatch_rework()
    assert result.ok is True

    stored = load_state(paths.state_file)["issues"]["123"]
    if expect_escalated:
        assert 123 in result.data.get("worker_death_escalated", [])
        assert stored["escalation_reason"] == "worker_death_loop"
    else:
        assert 123 not in result.data.get("worker_death_escalated", [])
        assert 123 not in result.data.get("no_op_rework_escalated", [])
        assert stored.get("escalation_reason") is None
