"""Issue #2262: the phantom-live-worker lane must park a dead local worker's
committed branch for review — never requeue the issue into a worktree reset
that archives the commits.

The incident: on a ``local_issues`` backend (no remote, the branch IS the
deliverable) a worker committed its fix and died. The next dispatch attempt
hit the phantom-live-worker path — the adapter refused the launch because a
sidecar claimed a live worker, but the recorded PID was dead — and that path
requeued the issue (strip active labels, re-add ``automated-ready``) without
ever asking whether the branch carried work. The redispatch then archived
the committed worktree. ``test_orphan_sweep_local_park.py`` covers the same
fix for the state/PID orphan sweep and the sidecar reaper; this file is the
phantom-lane half.

Covered here:

- happy path: phantom + worktree with commits ahead -> parked
  (``agent:review-ready``, status ``open_passive``, ``local_work_ready``
  event), sidecar reaped, no redispatch;
- the incident shape: NO sidecar left but the branch ref still carries the
  commits -> the branch-diff fallback still parks;
- worktree present but zero commits ahead -> the pre-#2262 reap/requeue
  behavior is preserved (control);
- PR-capable backend: the same probe routes through the existing
  ``_attempt_salvage`` push+PR path instead of parking;
- park label-write failure -> defer (status ``dispatched``, labels kept,
  no requeue) so the sweep retries instead of archiving work on a transient
  failure.

Helpers are self-contained per this repo's test-file convention (see
``test_orphan_sweep_local_park.py`` / ``test_charlie_work_dispatch_phantom.py``
— same fixture shapes, kept local so this file does not depend on another
test module's private surface).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from _fakes_github import FakeGitHub
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.host.fakes import FakeProcessProbe
from charlie_work.config import OrchestratorConfig, WorkerRoleConfig, load_config
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import resolved_layout, runtime_paths
from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state
from charlie_work.worktree import worktree_path_for_branch
from charlie_work.workflow import OrchestratorApp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ISSUE_NUMBER = 12  # the incident's issue number


def _git(repo_root: Path, *args: str, cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd or repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo with one commit and NO origin remote — the
    ``local_issues`` shape where the worker branch is the deliverable."""
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "core.longpaths", "true")
    _git(
        repo_root,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "--allow-empty",
        "-m",
        "chore: seed",
    )


def _write_issue(issues_dir: Path, number: int, *, labels: tuple[str, ...]) -> Path:
    issues_dir.mkdir(parents=True, exist_ok=True)
    labels_yaml = "[" + ", ".join(labels) + "]"
    path = issues_dir / f"{number:03d}_2026-09-17_fix-search.md"
    path.write_text(
        "---\n"
        'title: "Fix search"\n'
        "state: open\n"
        f"labels: {labels_yaml}\n"
        'created: "2026-09-17"\n'
        'author: "test"\n'
        "---\n"
        "Body text.\n",
        encoding="utf-8",
    )
    return path


def _local_config(repo_root: Path, worktrees_dir: Path) -> OrchestratorConfig:
    """Config for a local_issues repo with worktrees at ``worktrees_dir`` --
    the ``claude_code.worktrees_dir`` override is the production knob
    ``resolved_layout`` reads, so the test exercises real path derivation."""
    config_file = repo_root / "orchestrator.config.yaml"
    config_file.write_text(
        "local_issues:\n"
        "  enabled: true\n"
        "worker:\n"
        '  harness: "claude-code"\n'
        "claude_code:\n"
        f'  worktrees_dir: "{worktrees_dir.as_posix()}"\n',
        encoding="utf-8",
    )
    return load_config(config_file)


@pytest.fixture
def shallow_wts(tmp_path: Path) -> Iterator[Path]:
    """Worktrees root OUTSIDE pytest's deep basetemp (``$GIT_DIR`` path cap
    on ``git worktree add`` -- same hazard test_orphan_sweep_local_park's
    fixture documents)."""
    root = Path(tempfile.mkdtemp(prefix="cwwt2262-"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _add_worktree_commit(repo_root: Path, config: OrchestratorConfig, branch: str) -> Path:
    """Create the worker's worktree at the production-resolved location and
    commit real work on ``branch`` inside it."""
    worktrees_dir = resolved_layout(config, repo_root).worktrees
    worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "worktree", "add", "-b", branch, str(worktree_path))
    (worktree_path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(repo_root, "add", "feature.txt", cwd=worktree_path)
    _git(
        repo_root,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-m",
        "feat: work",
        cwd=worktree_path,
    )
    return worktree_path


def _seed_dead_dispatched(state_file: Path, issue_number: int, branch: str) -> None:
    """The dead worker's state entry: status dispatched + a dead PID."""
    state = load_state(state_file)
    state["issues"][str(issue_number)] = {
        "number": issue_number,
        "status": "dispatched",
        "branch_name": branch,
        "worker_pid": 424242,
        "worker_process_start_time": 1700000000.0,
        "dispatched_at": "2026-09-17T00:00:00Z",
        "title": "Fix search",
    }
    save_state(state_file, state)


def _write_dead_sidecar(
    sessions_dir: Path, issue_number: int, branch: str, worktree_path: Path
) -> Path:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sessions_dir / f"issue-{issue_number}.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": issue_number,
                "branch": branch,
                "worktree_path": str(worktree_path),
                "prompt_path": "",
                "command": ["claude", "-p"],
                "pid": 424242,
                "started_at": (datetime.now(UTC) - timedelta(hours=1)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "log_path": "",
                "error": "probe_error",
                "failure_kind": "live_worker_redispatch_averted",
                "process_start_time": 1700000000.0,
                "session_id": f"test-session-{issue_number}",
            }
        ),
        encoding="utf-8",
    )
    return sidecar_path


def _phantom_launch(worktree_path: Path) -> object:
    """A launch that reports ``live_worker_redispatch_averted`` for a dead PID."""

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree_path),
            prompt_path=str(worktree_path / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=424242,
            started_at="2026-09-17T00:00:00Z",
            log_path="",
            error="probe_error",
            failure_kind="live_worker_redispatch_averted",
            process_start_time=1700000000.0,
        )

    return _fake_launch


def _patch_dead_pids(monkeypatch, fake_host) -> None:
    """Every liveness read the dispatch path consults must read dead.

    The post-launch phantom classifier's recorded-PID check, the sidecar
    census (``_issues_with_live_workers`` via ``worker_fate.is_alive``), and
    the state-entry liveness check in candidate selection all read the one
    injected host probe — faked all-dead here so the outcome does not depend
    on the host's PID table, which is exactly the flakiness
    test_charlie_work_dispatch_phantom.py documents for PID 6262.
    """
    fake_host(probe=FakeProcessProbe())


def _label_names(gh: LocalFileGitHub, issue_number: int) -> set[str]:
    return {entry["name"] for entry in gh.issue_view(issue_number)["labels"]}


def _events(state_file: Path, kind: str, issue_number: int | None = None):
    state = load_state(state_file)
    return [
        event
        for event in state.get("events", [])
        if event.get("kind") == kind
        and (issue_number is None or event.get("payload", {}).get("issue_number") == issue_number)
    ]


# ---------------------------------------------------------------------------
# Issue #2262 acceptance: local backend + commits ahead -> park for review.
# ---------------------------------------------------------------------------


def test_phantom_parks_committed_local_work(
    tmp_path: Path, shallow_wts: Path, monkeypatch, fake_host
) -> None:
    """A phantom live-worker result on a local backend whose sidecar'd
    worktree carries commits ahead of base must park the branch
    ``agent:review-ready`` (status ``open_passive``) instead of requeueing
    the issue. The requeue is what handed the issue back to dispatch, where
    the next launch's worktree reset archived the commits."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)
    issues_dir = repo_root / "issues"
    branch = f"agent/issue-{ISSUE_NUMBER}-fix-search"
    worktree_path = _add_worktree_commit(repo_root, config, branch)

    # The incident shape: the issue was requeued once already, so it carries
    # ``automated-ready`` and is dispatch-selectable while the dead worker's
    # state entry still says dispatched.
    _write_issue(issues_dir, ISSUE_NUMBER, labels=("automated-ready",))
    gh = LocalFileGitHub(repo_root, issues_dir)

    paths = runtime_paths(repo_root, config.runtime.state_dir)
    sessions_dir = resolved_layout(config, repo_root).sessions_dir
    sidecar_path = _write_dead_sidecar(sessions_dir, ISSUE_NUMBER, branch, worktree_path)
    _seed_dead_dispatched(paths.state_file, ISSUE_NUMBER, branch)

    monkeypatch.setattr(
        "charlie_work.claude_code.launch_claude_worker", _phantom_launch(worktree_path)
    )
    _patch_dead_pids(monkeypatch, fake_host)

    app = OrchestratorApp(repo_root, paths, config, gh)
    result = app.dispatch(limit=1)

    assert result.data["phantom_live_worker_count"] == 1, repr(result.data)

    # Parked, not requeued: review-ready applied, ready stripped of nothing
    # it already lacked, and crucially the issue is no longer dispatchable
    # (review_ready is terminal).
    labels = _label_names(gh, ISSUE_NUMBER)
    assert "agent:review-ready" in labels
    assert "agent:in-progress" not in labels

    state = load_state(paths.state_file)
    entry = state["issues"][str(ISSUE_NUMBER)]
    assert entry["status"] == PASSIVE_OPEN_STATUS
    assert "worker_pid" not in entry

    # The stale sidecar no longer occupies a slot.
    assert not sidecar_path.exists()

    # Observable contract: the park event, the phantom-salvage relabel
    # record, and NONE of the archive/requeue signals.
    assert _events(paths.state_file, "local_work_ready", ISSUE_NUMBER)
    salvaged = _events(paths.state_file, "session_failed_relabeled", ISSUE_NUMBER)
    assert [e["payload"]["reason"] for e in salvaged] == ["phantom_live_worker_commits_salvaged"]
    assert not _events(paths.state_file, "worktree_local_commits_archived")

    # No redispatch: a second pass finds nothing to select.
    result2 = app.dispatch(limit=1)
    assert result2.data["attempted_count"] == 0


def test_phantom_parks_committed_local_work_without_sidecar(
    tmp_path: Path, shallow_wts: Path, monkeypatch, fake_host
) -> None:
    """The incident's exact shape: the dead worker's sidecar is ALREADY gone
    (reaped by an earlier lane) but its branch ref still carries the
    commits. The probe must find the work via the branch-diff fallback —
    with no sidecar there is no worktree inspection to save it, and the
    pre-#2262 path simply requeued."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)
    issues_dir = repo_root / "issues"
    branch = f"agent/issue-{ISSUE_NUMBER}-fix-search"
    worktree_path = _add_worktree_commit(repo_root, config, branch)
    # The worktree is gone too — only the branch ref holds the work.
    _git(repo_root, "worktree", "remove", "--force", str(worktree_path))

    _write_issue(issues_dir, ISSUE_NUMBER, labels=("automated-ready",))
    gh = LocalFileGitHub(repo_root, issues_dir)

    paths = runtime_paths(repo_root, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, ISSUE_NUMBER, branch)

    monkeypatch.setattr(
        "charlie_work.claude_code.launch_claude_worker",
        _phantom_launch(tmp_path / "wt"),
    )
    _patch_dead_pids(monkeypatch, fake_host)

    app = OrchestratorApp(repo_root, paths, config, gh)
    result = app.dispatch(limit=1)

    assert result.data["phantom_live_worker_count"] == 1, repr(result.data)
    labels = _label_names(gh, ISSUE_NUMBER)
    assert "agent:review-ready" in labels

    state = load_state(paths.state_file)
    assert state["issues"][str(ISSUE_NUMBER)]["status"] == PASSIVE_OPEN_STATUS
    assert _events(paths.state_file, "local_work_ready", ISSUE_NUMBER)
    assert not _events(paths.state_file, "worktree_local_commits_archived")

    result2 = app.dispatch(limit=1)
    assert result2.data["attempted_count"] == 0


def test_phantom_no_commits_still_requeues(
    tmp_path: Path, shallow_wts: Path, monkeypatch, fake_host
) -> None:
    """Control: a local-backend phantom whose branch carries NO commits keeps
    the pre-#2262 behavior — sidecar reaped, ``session_failed_relabeled``
    with ``phantom_live_worker_pid_dead``, no park, issue left dispatchable."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)
    issues_dir = repo_root / "issues"
    branch = f"agent/issue-{ISSUE_NUMBER}-fix-search"
    # Worktree exists on the branch but with zero commits ahead.
    worktrees_dir = resolved_layout(config, repo_root).worktrees
    worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "worktree", "add", "-b", branch, str(worktree_path))

    _write_issue(issues_dir, ISSUE_NUMBER, labels=("automated-ready",))
    gh = LocalFileGitHub(repo_root, issues_dir)

    paths = runtime_paths(repo_root, config.runtime.state_dir)
    sessions_dir = resolved_layout(config, repo_root).sessions_dir
    sidecar_path = _write_dead_sidecar(sessions_dir, ISSUE_NUMBER, branch, worktree_path)
    _seed_dead_dispatched(paths.state_file, ISSUE_NUMBER, branch)

    monkeypatch.setattr(
        "charlie_work.claude_code.launch_claude_worker", _phantom_launch(worktree_path)
    )
    _patch_dead_pids(monkeypatch, fake_host)

    app = OrchestratorApp(repo_root, paths, config, gh)
    result = app.dispatch(limit=1)

    assert result.data["phantom_live_worker_count"] == 1, repr(result.data)
    assert not sidecar_path.exists()

    state = load_state(paths.state_file)
    assert state["issues"][str(ISSUE_NUMBER)]["status"] == "dispatch_failed"

    labels = _label_names(gh, ISSUE_NUMBER)
    # Nothing parked: no review-ready, and the issue remains dispatchable
    # (ready-only labels) — exactly the pre-#2262 empty-worktree outcome.
    assert "agent:review-ready" not in labels
    relabeled = _events(paths.state_file, "session_failed_relabeled", ISSUE_NUMBER)
    assert [e["payload"]["reason"] for e in relabeled] == ["phantom_live_worker_pid_dead"]
    assert not _events(paths.state_file, "local_work_ready")
    assert not _events(paths.state_file, "worktree_local_commits_archived")

    # The issue is dispatchable again — requeue preserved.
    result2 = app.dispatch(limit=1)
    assert result2.data["attempted_count"] == 1


def test_phantom_pr_capable_backend_routes_through_attempt_salvage(
    tmp_path: Path, shallow_wts: Path, monkeypatch, fake_host
) -> None:
    """PR-capable backend, no preserving fate: the probe finds commits on the
    dead worker's branch and routes them through ``_attempt_salvage`` — the
    same push+PR seam the sidecar reaper and the orphan sweep already use —
    instead of relabeling the issue back to ``ready``.

    Uses FakeGitHub's built-in issue #123 (``automated-ready`` → selectable;
    ``pr_list=[]`` removes the default linked PR)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = OrchestratorConfig(worker=WorkerRoleConfig(harness="claude-code"))
    branch = "agent/issue-123-fix-search"
    # Commit on the branch ref only — no sidecar, so no preserving fate and
    # the probe must find the work via the branch-diff fallback.
    worktree_path = _add_worktree_commit(repo_root, config, branch)
    _git(repo_root, "worktree", "remove", "--force", str(worktree_path))

    paths = runtime_paths(repo_root, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, 123, branch)

    fake_gh = FakeGitHub()
    fake_gh.pr_list = lambda: []

    salvage_calls: list[dict] = []

    def _spy_salvage(**kwargs):
        salvage_calls.append(kwargs)
        return True, None

    monkeypatch.setattr("charlie_work.workflow._attempt_salvage", _spy_salvage)
    monkeypatch.setattr(
        "charlie_work.claude_code.launch_claude_worker",
        _phantom_launch(tmp_path / "wt"),
    )
    _patch_dead_pids(monkeypatch, fake_host)

    app = OrchestratorApp(repo_root, paths, config, fake_gh)
    result = app.dispatch(limit=1)

    assert result.data["phantom_live_worker_count"] == 1, repr(result.data)
    assert len(salvage_calls) == 1
    assert salvage_calls[0]["branch"] == branch
    assert salvage_calls[0]["issue_number"] == 123

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == PASSIVE_OPEN_STATUS
    salvaged = _events(paths.state_file, "session_failed_relabeled", 123)
    assert [e["payload"]["reason"] for e in salvaged] == ["phantom_live_worker_commits_salvaged"]


def test_phantom_salvage_failure_defers_without_requeue(
    tmp_path: Path, shallow_wts: Path, monkeypatch, fake_host
) -> None:
    """A probe that finds commits but fails the park write must NOT fall back
    to requeueing — that is the archive path. With no sidecar (the incident
    shape) there is no preserving fate to defer through, so the route's own
    deferral fires: the entry stays ``dispatched`` so the dead-worker sweep's
    state-orphan lane retries the park probe next pass, and the deferral is
    recorded on the event. (With a sidecar still present, the pre-#2262
    preserve branch already defers through the reaper lane — covered by the
    existing phantom tests.)"""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = _local_config(repo_root, shallow_wts)
    issues_dir = repo_root / "issues"
    branch = f"agent/issue-{ISSUE_NUMBER}-fix-search"
    worktree_path = _add_worktree_commit(repo_root, config, branch)
    # Sidecar AND worktree both gone — only the branch ref holds the work.
    _git(repo_root, "worktree", "remove", "--force", str(worktree_path))

    _write_issue(issues_dir, ISSUE_NUMBER, labels=("automated-ready",))

    class FailingLabelGitHub(LocalFileGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            if label == "agent:review-ready":
                return False
            return super().add_issue_label(number, label)

    gh = FailingLabelGitHub(repo_root, issues_dir)

    paths = runtime_paths(repo_root, config.runtime.state_dir)
    _seed_dead_dispatched(paths.state_file, ISSUE_NUMBER, branch)

    monkeypatch.setattr(
        "charlie_work.claude_code.launch_claude_worker",
        _phantom_launch(tmp_path / "wt"),
    )
    _patch_dead_pids(monkeypatch, fake_host)

    app = OrchestratorApp(repo_root, paths, config, gh)
    result = app.dispatch(limit=1)

    assert result.data["phantom_live_worker_count"] == 1, repr(result.data)

    state = load_state(paths.state_file)
    entry = state["issues"][str(ISSUE_NUMBER)]
    # Deferral: still dispatched (the sweep owns the retry), not requeued.
    assert entry["status"] == "dispatched"
    # review-ready add failed (park write), and the deferral is observable.
    labels = _label_names(gh, ISSUE_NUMBER)
    assert "agent:review-ready" not in labels
    # ``ready`` is stripped so the deferred entry is NOT dispatchable while
    # the park is pending — otherwise a ready-only label set plus a
    # dead/popped PID would re-select the issue and relaunch before the
    # sweep's retry, the same requeue-into-reset shape the fix removes.
    assert "automated-ready" not in labels
    deferred = _events(paths.state_file, "session_failed_relabeled", ISSUE_NUMBER)
    assert [e["payload"]["reason"] for e in deferred] == ["phantom_live_worker_salvage_deferred"]
    # The attempted park records its own audit event marked as a failed write.
    ready_events = _events(paths.state_file, "local_work_ready", ISSUE_NUMBER)
    assert all(e["payload"].get("label_write_ok") is False for e in ready_events)
    assert not _events(paths.state_file, "worktree_local_commits_archived")

    # Non-dispatchable while deferred: a second pass selects nothing.
    result2 = app.dispatch(limit=1)
    assert result2.data["attempted_count"] == 0
