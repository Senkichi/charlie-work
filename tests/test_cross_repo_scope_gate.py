"""Tests for the cross-repo scope gate and fleet-registry helper (issue #1244).

The scope gate checks an issue's *title* for a ``<repo-name>:`` prefix that
names a managed repo other than the dispatching one — the clearest signal
that the issue's deliverables live in that repo, not this one.  The
managed-repo set is derived from the fleet registry, never a hardcoded list.

The first section tests the isolated helper functions.  The second section
tests the *wiring* — that the four ``workflow.py`` call sites (orphan-sweep
no-PR path, dead-session classification, and the live + dry-run dispatch
paths) actually invoke the scope gate and act on its verdict.  A regression
in the wiring that drops the call would silently reintroduce the #709
infinite-loop bug with every unit test green, so these tests drive the real
workflow functions, not ``cross_repo_scope_gate`` in isolation.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from charlie_work.config import (
    DETERMINISTIC_ESCALATION_FAILURE_KINDS,
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.cross_repo_gate import (
    CrossRepoGateResult,
    cross_repo_scope_gate,
)
from charlie_work.devin_shell import SessionRecord
from charlie_work.fleet_registry import managed_repo_names
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.worktree import create_worktree
from charlie_work.write_gate import WriteGate

from _fakes_github import FakeGitHub


# ---------------------------------------------------------------------------
# cross_repo_scope_gate
# ---------------------------------------------------------------------------


def test_empty_managed_repos_passes() -> None:
    """No fleet registry / single-repo deployment → nothing to check, passes."""
    result = cross_repo_scope_gate("job-cannon: fix the docs", "", "charlie-work", frozenset())
    assert result.passed
    assert "no other managed repos" in result.reason


def test_title_names_other_managed_repo_blocks() -> None:
    """The #709 pattern: title starts with another managed repo's name."""
    result = cross_repo_scope_gate(
        "job-cannon: docs/devin-orchestration/ ... stale",
        "Body text about job-cannon files.",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon"}),
    )
    assert not result.passed
    assert "cross_repo_scope" in result.reason
    assert "job-cannon" in result.reason
    assert "charlie-work" in result.reason


def test_title_names_dispatching_repo_passes() -> None:
    """An issue whose title starts with the dispatching repo's own name passes."""
    result = cross_repo_scope_gate(
        "charlie-work: fix the dispatch logic",
        "",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon"}),
    )
    assert result.passed


def test_title_does_not_name_any_repo_passes() -> None:
    """An issue with a generic title passes even when other repos exist."""
    result = cross_repo_scope_gate(
        "Fix the search function in the worker",
        "",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon"}),
    )
    assert result.passed


def test_title_mentions_repo_not_as_prefix_passes() -> None:
    """A repo name appearing in the title but not as a prefix passes.

    ``coordinate with job-cannon on this`` mentions the repo but is not
    evidence of a cross-repo scope — the issue's deliverables may still
    live in the dispatching repo.
    """
    result = cross_repo_scope_gate(
        "Coordinate with job-cannon on the shared API",
        "",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon"}),
    )
    assert result.passed


def test_case_insensitive_title_match() -> None:
    """Title-prefix matching is case-insensitive."""
    result = cross_repo_scope_gate(
        "Job-Cannon: fix the docs",
        "",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon"}),
    )
    assert not result.passed


def test_leading_whitespace_in_title_handled() -> None:
    """Leading whitespace before the repo-name prefix does not defeat the check."""
    result = cross_repo_scope_gate(
        "  job-cannon: fix the docs",
        "",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon"}),
    )
    assert not result.passed


def test_dispatching_repo_not_in_managed_set_blocks() -> None:
    """When the dispatching repo is not in the managed set, other repos still block."""
    result = cross_repo_scope_gate(
        "job-cannon: fix the docs",
        "",
        "some-other-repo",
        frozenset({"job-cannon"}),
    )
    assert not result.passed


def test_multiple_other_repos_one_matches_blocks() -> None:
    """When multiple other repos exist, matching any one blocks."""
    result = cross_repo_scope_gate(
        "ci-fleet: fix the runner allocation",
        "",
        "charlie-work",
        frozenset({"charlie-work", "job-cannon", "ci-fleet"}),
    )
    assert not result.passed
    assert "ci-fleet" in result.reason


def test_result_type_is_cross_repo_gate_result() -> None:
    """The scope gate returns the same result type as the file-path gate."""
    result = cross_repo_scope_gate(
        "job-cannon: fix", "", "charlie-work", frozenset({"charlie-work", "job-cannon"})
    )
    assert isinstance(result, CrossRepoGateResult)


# ---------------------------------------------------------------------------
# managed_repo_names
# ---------------------------------------------------------------------------


def _write_fleet_registry(fleet_dir: Path, repos: dict[str, dict[str, str]]) -> None:
    """Write a fleet.json with the given repo entries."""
    fleet_dir.mkdir(parents=True, exist_ok=True)
    fleet_json = fleet_dir / "fleet.json"
    data = {"version": 1, "repos": repos}
    fleet_json.write_text(json.dumps(data), encoding="utf-8")


def test_managed_repo_names_extracts_repo_segments(tmp_path: Path) -> None:
    """Repo names are the last segment of owner/repo keys."""
    _write_fleet_registry(
        tmp_path,
        {
            "Senkichi/charlie-work": {"repo_root": "/tmp/cw"},
            "Senkichi/job-cannon": {"repo_root": "/tmp/jc"},
        },
    )
    names = managed_repo_names(str(tmp_path))
    assert names == frozenset({"charlie-work", "job-cannon"})


def test_managed_repo_names_empty_registry(tmp_path: Path) -> None:
    """A missing fleet registry returns an empty set."""
    names = managed_repo_names(str(tmp_path))
    assert names == frozenset()


def test_managed_repo_names_single_repo(tmp_path: Path) -> None:
    """A single-repo fleet returns just that repo's name."""
    _write_fleet_registry(
        tmp_path,
        {"Senkichi/charlie-work": {"repo_root": "/tmp/cw"}},
    )
    names = managed_repo_names(str(tmp_path))
    assert names == frozenset({"charlie-work"})


def test_managed_repo_names_corrupt_registry_returns_empty(tmp_path: Path) -> None:
    """A corrupt fleet.json returns an empty set, not an exception."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "fleet.json").write_text("not valid json", encoding="utf-8")
    names = managed_repo_names(str(tmp_path))
    assert names == frozenset()


# ---------------------------------------------------------------------------
# Wiring tests: the four workflow.py call sites (issue #1244)
#
# The isolated helper tests above prove the scope gate's logic.  These tests
# prove the *wiring* — that the real workflow functions actually call the
# scope gate and act on its verdict.  A regression that drops the call would
# silently reintroduce the #709 infinite-loop bug with every unit test green.
# ---------------------------------------------------------------------------


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_bare_remote_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare remote repo and a local clone, return (remote, clone)."""
    remote = tmp_path / "remote"
    remote.mkdir(parents=True, exist_ok=True)
    _git(remote, "init", "--bare", "--initial-branch=main")
    clone = tmp_path / "clone"
    clone.mkdir(parents=True, exist_ok=True)
    _git(clone, "init", "--initial-branch=main")
    _git(clone, "config", "user.email", "test@example.test")
    _git(clone, "config", "user.name", "Test User")
    _git(clone, "config", "commit.gpgSign", "false")
    _git(clone, "remote", "add", "origin", str(remote))
    (clone / "README.md").write_text("hello\n", encoding="utf-8")
    _git(clone, "add", "README.md")
    _git(clone, "commit", "-m", "initial commit")
    _git(clone, "push", "-u", "origin", "main")
    return remote, clone


def _make_classify_state(tmp_path: Path) -> tuple[Path, Path]:
    """Create a state file and sessions dir under tmp_path, return (sessions_dir, state_file)."""
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    return sessions_dir, state_file


def _write_dead_session_sidecar(
    sessions_dir: Path, issue_number: int, branch: str, worktree_path: Path
) -> None:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    record = SessionRecord(
        issue_number=issue_number,
        branch=branch,
        worktree_path=str(worktree_path),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(sessions_dir / f"issue-{issue_number}.log"),
        error=None,
    )
    sidecar_path = sessions_dir / f"issue-{issue_number}.json"
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")


def _make_fleet_dir(tmp_path: Path) -> Path:
    """Write a fleet.json with charlie-work + job-cannon, return the fleet dir path."""
    fleet_dir = tmp_path / "fleet"
    _write_fleet_registry(
        fleet_dir,
        {
            "Senkichi/charlie-work": {"repo_root": "/tmp/cw"},
            "Senkichi/job-cannon": {"repo_root": "/tmp/jc"},
        },
    )
    return fleet_dir


def _charlie_work_gh(repo_root: Path) -> FakeGitHub:
    """A FakeGitHub whose name_with_owner returns Senkichi/charlie-work."""
    gh = FakeGitHub(repo_root=repo_root)
    gh.name_with_owner = lambda: "Senkichi/charlie-work"  # type: ignore[method-assign]
    return gh


# ---------------------------------------------------------------------------
# Wiring test 1: orphan-sweep no-PR path
# ---------------------------------------------------------------------------


def test_orphan_sweep_escalates_cross_repo_scoped_issue(tmp_path: Path) -> None:
    """The orphan-sweep no-PR path escalates a cross-repo-scoped issue to
    ``agent:human-needed`` (removing active labels, NOT adding
    ``automated-ready``) instead of redispatching.

    Drives ``_detect_and_handle_orphaned_workers`` — the real sweep function —
    not ``cross_repo_scope_gate`` in isolation.  A dead worker whose issue
    title names another managed repo (``job-cannon:``) hopped to that repo's
    worktree; redispatching repeats the hop forever.  The scope gate must
    catch it before the relabel-to-ready path and escalate instead.
    """
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fleet_dir = _make_fleet_dir(tmp_path)

    # Seed a dispatched issue with a dead worker and no open PR.
    issue_number = 709
    state = load_state(paths.state_file)
    state["issues"][str(issue_number)] = {
        "status": "dispatched",
        "dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
    }
    save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    fake_gh = _charlie_work_gh(tmp_path)
    fake_gh.issues = [
        {
            "number": issue_number,
            "title": "job-cannon: docs/devin-orchestration/ ... stale",
            "url": f"https://example.test/issues/{issue_number}",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            paths.state_file,
            config,
            fake_gh,
            write_gate=_wg(paths.state_file),
            fleet_dir_override=str(fleet_dir),
        )

    state = load_state(paths.state_file)
    entry = state["issues"][str(issue_number)]

    # Escalated with cross_repo_hop, not relabeled to ready.
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "cross_repo_hop"

    # Active labels removed, human-needed added, ready NOT added.
    assert (issue_number, config.labels.in_progress) in fake_gh.labels_removed
    assert (issue_number, config.labels.human_needed) in fake_gh.labels_added
    assert (issue_number, config.labels.ready) not in fake_gh.labels_added

    # A session_failed_escalated event was emitted with cross_repo_hop.
    events = [e for e in state["events"] if e["kind"] == "session_failed_escalated"]
    assert len(events) == 1
    assert events[0]["payload"]["reason"] == "cross_repo_hop"


# ---------------------------------------------------------------------------
# Wiring test 2: _classify_dead_sessions_and_update_throttle_state
# ---------------------------------------------------------------------------


def test_classify_dead_sessions_cross_repo_hop_escalates_on_first_occurrence(
    tmp_path: Path,
) -> None:
    """``_classify_dead_sessions_and_update_throttle_state`` overrides
    ``failure_kind`` to ``cross_repo_hop`` and escalates on the FIRST
    occurrence for a dead session whose issue title names another managed
    repo.

    Without the scope gate, a dead session with a non-terminal failure kind
    (``stalled``) would relabel to ``automated-ready`` for redispatch —
    repeating the hop forever.  The scope gate override makes
    ``cross_repo_hop`` a deterministic escalation kind, so the terminal-failure
    check fires on the first occurrence instead of waiting for the redispatch
    cap.

    Drives the real classification function with a real git worktree (no
    commits → no salvage, ``fallback_kind="stalled"``) and a fleet registry
    on disk, not a monkeypatched helper.
    """
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path / "repo")
    branch = "agent/issue-709"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 709, branch, info.path)

    fleet_dir = _make_fleet_dir(tmp_path)
    config = OrchestratorConfig()
    gh = _charlie_work_gh(repo_root)
    gh.issues = [
        {
            "number": 709,
            "title": "job-cannon: docs/devin-orchestration/ ... stale",
            "url": "https://example.test/issues/709",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.prs = []

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir,
        state_file,
        gh,
        config,
        write_gate=_wg(state_file),
        fleet_dir_override=str(fleet_dir),
    )

    state = json.loads(state_file.read_text(encoding="utf-8"))
    entry = state["issues"]["709"]

    # Escalated on the first occurrence — no redispatch cap needed.
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "cross_repo_hop"

    # Active labels removed, ready NOT added (it would be added if relabeled).
    assert (709, config.labels.in_progress) in gh.labels_removed
    assert (709, config.labels.ready) not in gh.labels_added

    # The escalation event records the overridden failure_kind.
    events = [e for e in state["events"] if e["kind"] == "session_failed_escalated"]
    assert len(events) == 1
    assert events[0]["payload"]["failure_kind"] == "cross_repo_hop"

    # Sanity: cross_repo_hop is a deterministic escalation kind, so the
    # terminal-failure check (not the redispatch cap) is what fired.
    assert "cross_repo_hop" in DETERMINISTIC_ESCALATION_FAILURE_KINDS


# ---------------------------------------------------------------------------
# Wiring test 3: dispatch path (the _finalization_order / dispatch method)
# ---------------------------------------------------------------------------


def test_dispatch_refuses_cross_repo_scoped_issue_via_scope_gate(
    tmp_path: Path,
) -> None:
    """``OrchestratorApp.dispatch`` refuses to dispatch (and records) a
    cross-repo-scoped issue via the pre-flight scope gate.

    The scope gate runs after the file-path gate (issue #1010) and before
    adapter routing.  An issue whose title names another managed repo
    (``job-cannon:``) is added to ``cross_repo_escalated`` and escalated to
    ``agent:operator-queue`` — no session is requested, no worker is launched.

    Drives the real ``dispatch`` method (the ``_finalization_order`` path),
    not ``cross_repo_scope_gate`` in isolation.
    """
    from charlie_work.workflow import OrchestratorApp

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fleet_dir = _make_fleet_dir(tmp_path)

    fake_gh = _charlie_work_gh(tmp_path)
    # Close the default fixture PR so issue #123 is selectable.
    fake_gh.prs[0]["state"] = "CLOSED"
    # Title names another managed repo; body has no file paths (so the
    # file-path gate passes and the scope gate is what blocks).
    fake_gh.issues[0]["title"] = "job-cannon: fix the docs"
    fake_gh.issues[0]["body"] = "Docs are stale."

    app = OrchestratorApp(
        tmp_path,
        paths,
        config,
        fake_gh,
        fleet_dir_override=str(fleet_dir),
    )

    result = app.dispatch(limit=1)

    # The issue was escalated, not dispatched: no session was requested.
    assert result.ok is True
    assert result.data["selected_count"] == 0
    assert 123 in result.data["cross_repo_escalated_issue_numbers"]
    assert result.data["dispatch_results"] == []

    # Label transition to operator-queue (mechanical reason_class).
    assert (123, config.labels.operator_queue) in fake_gh.labels_added

    # The dispatch_cross_repo_escalated event was recorded with a
    # cross_repo_scope reason.
    state = load_state(paths.state_file)
    escalated_events = [e for e in state["events"] if e["kind"] == "dispatch_cross_repo_escalated"]
    assert len(escalated_events) == 1
    assert escalated_events[0]["payload"]["issue_number"] == 123
    assert "cross_repo_scope" in escalated_events[0]["payload"]["reason"]

    # The issue is terminal in state, not left dispatch_pending.
    assert state["issues"]["123"]["status"] == "escalated"
