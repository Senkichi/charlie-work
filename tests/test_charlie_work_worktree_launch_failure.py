"""Worktree-unsafe launch failures: escalation, redispatch suppression, commit salvage.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from _dead_session_fixtures import _git
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from _worktree_fixtures import (
    _init_bare_remote_and_clone,
    _setup_completed_worktree,
)
from charlie_work.config import (
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.devin_shell import SessionRecord
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from charlie_work.worktree import create_worktree
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_worktree_unsafe_launch_failure_escalates_and_suppresses_redispatch(
    tmp_path: Path,
) -> None:
    """Issue #288: a launch result whose sidecar carries failure_kind=worktree_unsafe
    must escalate immediately, bypass the redispatch cap, and not be relabeled to ready.
    A subsequent dispatch pass must not select the issue.
    """
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state

    now = datetime.now(UTC)

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.ready}],
        }
    ]
    fake_gh.prs = []  # No open PR — the ordinary relabel path would fire here.

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("worktree contains local work, cannot reset\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Launch failure — process never started
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="worktree creation failed: worktree contains local work",
        failure_kind="worktree_unsafe_shim_dirt",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # No hot relabel-to-ready.
    assert (42, config.labels.ready) not in fake_gh.labels_added
    # Escalation transition added operator_queue. Issue #1266: worktree_unsafe
    # is mechanical, so it lands agent:operator-queue, not agent:human-needed.
    assert (42, config.labels.operator_queue) in fake_gh.labels_added
    # The launch never succeeded, so the issue should not be marked in_progress.
    assert (42, config.labels.in_progress) not in fake_gh.labels_added

    state = load_state(paths.state_file)
    issue_entry = state["issues"]["42"]
    assert issue_entry["status"] == "escalated"
    assert issue_entry["escalation_reason"] == "worktree_unsafe_shim_dirt"

    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 42]
    assert "session_failed_relabeled" not in event_kinds
    assert "session_failed_escalated" in event_kinds

    fake_gh.issues[0]["labels"].append({"name": config.labels.operator_queue})
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)
    assert result.data["selected_count"] == 0


def test_worktree_probe_failed_launch_failure_does_not_escalate(
    tmp_path: Path,
) -> None:
    """PR #314 review follow-up to issue #288: a launch result whose sidecar
    carries failure_kind=worktree_probe_failed (the git status --porcelain
    safety probe itself failed -- index lock, I/O error, etc. -- NOT a
    confirmed-dirty worktree) must NOT escalate on first occurrence. It must
    take the ordinary redispatch-cap path so a subsequent dispatch pass can
    still select the issue.

    This is the mirror of
    test_worktree_unsafe_launch_failure_escalates_and_suppresses_redispatch:
    confirmed-dirty (worktree_unsafe) escalates immediately; a failed probe
    (worktree_probe_failed) must not, because it is transient contention an
    ordinary redispatch retry would plausibly heal.
    """
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state

    now = datetime.now(UTC)

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.ready}],
        }
    ]
    fake_gh.prs = []  # No open PR — the ordinary relabel path would fire here.

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("index.lock: File exists\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Launch failure — process never started
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="worktree creation failed: worktree status probe failed; treating as dirty",
        failure_kind="worktree_probe_failed",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # No escalation transition — human_needed must NOT be added, and the
    # issue must not be marked in_progress (the launch never succeeded).
    assert (42, config.labels.human_needed) not in fake_gh.labels_added
    assert (42, config.labels.in_progress) not in fake_gh.labels_added
    assert (42, config.labels.ready) not in fake_gh.labels_removed

    # No escalated status recorded in state for this issue.
    state = load_state(paths.state_file)
    issue_entry = state["issues"].get("42")
    assert issue_entry is None or issue_entry.get("status") != "escalated"

    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 42]
    assert "session_failed_escalated" not in event_kinds

    # Because nothing removed the "ready" label or marked the issue escalated,
    # a subsequent dispatch pass must still be able to select it — the
    # opposite of the confirmed-dirty (worktree_unsafe) case above.
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch(limit=1)
    assert result.data["selected_count"] == 1


def test_worktree_unsafe_launch_failure_with_commits_salvages_before_escalation(
    tmp_path: Path,
) -> None:
    """Issue #1130: a ``worktree_unsafe`` launch failure whose worktree has
    commits ahead of base must attempt salvage (push + PR) before escalating
    to ``agent:human-needed``. Salvage-the-commit is the cheap safe action;
    human adjudication is the fallback only when salvage fails."""

    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state

    now = datetime.now(UTC)

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    # Create a worktree with one commit beyond origin/main — the stranded
    # work that ``worktree_unsafe`` refused to reset.
    worktree_path, branch = _setup_completed_worktree(repo_root, 1130)

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub(repo_root=repo_root)
    fake_gh.issues = [
        {
            "number": 1130,
            "title": "Salvage test",
            "url": "https://example.test/issues/1130",
            "body": "Salvage does not fire",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []  # No open PR.
    fake_gh.pr_create_return = 200

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-1130.log"
    log_path.write_text("worktree contains local work, cannot reset\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-1130.json"
    record = SessionRecord(
        issue_number=1130,
        branch=branch,
        worktree_path=str(worktree_path),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Launch failure — process never started
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="worktree creation failed: worktree contains local work",
        failure_kind="worktree_unsafe_local_commits",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # Salvage fired: branch pushed and PR created.
    remote_refs = _git(remote, "show-ref")
    assert branch in remote_refs.stdout
    assert len(fake_gh.prs_created) == 1
    assert fake_gh.prs_created[0]["head"] == branch

    # Labels moved to pr_open, NOT human_needed.
    assert (1130, config.labels.in_progress) in fake_gh.labels_removed
    assert (1130, config.labels.pr_open) in fake_gh.labels_added
    assert (1130, config.labels.human_needed) not in fake_gh.labels_added

    state = load_state(paths.state_file)
    salvage_events = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert len(salvage_events) == 1
    assert salvage_events[0]["payload"]["issue_number"] == 1130
    # No escalation event.
    escalate_events = [e for e in state["events"] if e["kind"] == "session_failed_escalated"]
    assert not escalate_events


def test_worktree_unsafe_launch_failure_no_commits_still_escalates(
    tmp_path: Path,
) -> None:
    """Issue #1130: a ``worktree_unsafe`` launch failure whose worktree has NO
    commits ahead of base (e.g. dirty working tree with no commits) still
    escalates to ``agent:human-needed``. Salvage is only attempted when there
    is committed work to push."""

    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state

    now = datetime.now(UTC)

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    # A worktree with no commits ahead, just a dirty file.
    branch = "agent/issue-1130-no-commits"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    worktree_path = info.path
    (worktree_path / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub(repo_root=repo_root)
    fake_gh.issues = [
        {
            "number": 1130,
            "title": "Salvage test",
            "url": "https://example.test/issues/1130",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []
    fake_gh.pr_create_return = 200

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-1130.log"
    log_path.write_text("worktree contains local work, cannot reset\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-1130.json"
    record = SessionRecord(
        issue_number=1130,
        branch=branch,
        worktree_path=str(worktree_path),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="worktree creation failed: worktree contains local work",
        failure_kind="worktree_unsafe_shim_dirt",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # No salvage: no PR created.
    assert not fake_gh.prs_created
    # Escalation fired: a deterministic (mechanical) launch failure parks on
    # operator_queue (issue #1266), not human_needed.
    assert (1130, config.labels.operator_queue) in fake_gh.labels_added

    state = load_state(paths.state_file)
    salvage_events = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert not salvage_events
    escalate_events = [e for e in state["events"] if e["kind"] == "session_failed_escalated"]
    assert len(escalate_events) == 1
