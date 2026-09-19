"""Dead-session classification relabel path: idempotent relabel, closed/open-PR suppression, dispatch recovery, terminal-label-only leave-alone, no-commits relabel, and the required-reason contract.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from _dead_session_fixtures import (
    _write_dead_session_sidecar,
    _make_classify_state,
)
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from _worktree_fixtures import _init_bare_remote_and_clone
from charlie_work.config import (
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
from charlie_work.worktree import create_worktree


def test_classify_dead_sessions_relabel_idempotent(tmp_path: Path) -> None:
    """Issue #118 AC3: classification pass relabel is idempotent - two-pass test.

    This test runs the classification pass twice on the same dead session and
    verifies that (a) no error occurs, (b) no duplicate event is emitted, and
    (c) the issue remains in the correct label state after the second pass.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    # Use command adapter to avoid needing real devin binary
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Issue starts with in_progress label (active)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]

    # Ensure state directory exists
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a sessions directory with a dead session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a session record for a dead session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # First pass: run classification directly (not via loop to avoid review logic)
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # Verify first pass relabeled the issue
    assert (42, config.labels.in_progress) in fake_gh.labels_removed
    assert (42, config.labels.ready) in fake_gh.labels_added

    # Verify event was emitted
    state = load_state(paths.state_file)
    events_after_first = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events_after_first) == 1
    assert events_after_first[0]["payload"]["issue_number"] == 42

    # Update fake GitHub to reflect the relabeled state (ready label, no active labels)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.ready}],
        }
    ]
    # Clear label tracking for second pass
    fake_gh.labels_added = []
    fake_gh.labels_removed = []

    # Second pass: run classification again
    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # Verify second pass did NOT emit duplicate event (idempotency)
    state_after_second = load_state(paths.state_file)
    events_after_second = [
        e for e in state_after_second["events"] if e["kind"] == "session_failed_relabeled"
    ]
    assert len(events_after_second) == 1, "Second pass should not emit duplicate event"

    # Verify second pass did not attempt to remove in_progress (already gone)
    assert (42, config.labels.in_progress) not in fake_gh.labels_removed

    # Verify second pass did not attempt to add ready (already present)
    assert (42, config.labels.ready) not in fake_gh.labels_added


def test_classify_dead_sessions_preserves_state_record_branch(tmp_path: Path) -> None:
    """Issue #118 AC1: classification pass preserves state record branch/worktree fields.

    This test ensures that the relabel logic does not clobber the branch or
    worktree_path fields in the state record for the issue. Mutation gate:
    clobbering branch MUST fail this test.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    # Use command adapter to avoid needing real devin binary
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
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
            "labels": [{"name": config.labels.in_progress}],
        }
    ]

    # Ensure state directory exists
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize state with a branch/worktree entry for issue 42
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["42"] = {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "labels": [config.labels.in_progress],
            "branch": "agent/issue-42-fix-search",
            "worktree_path": "/tmp/worktree-issue-42",
        }
        save_state(paths.state_file, state)

    # Create a sessions directory with a dead session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a session record for a dead session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Run classification pass directly
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # Verify branch and worktree_path are preserved byte-identical
    state_after = load_state(paths.state_file)
    assert state_after["issues"]["42"]["branch"] == "agent/issue-42-fix-search"
    assert state_after["issues"]["42"]["worktree_path"] == "/tmp/worktree-issue-42"


def test_classify_dead_sessions_dispatch_recovery_integration(tmp_path: Path) -> None:
    """Issue #118 AC4: full chain integration test - classified-dead + relabeled → dispatch.

    This test drives the complete workflow:
    1. A session dies and is classified by the automated pass
    2. The issue is relabeled to dispatchable (ready label)
    3. The next dispatch pass selects the issue
    4. The recovery dict (branch/worktree) is passed to create_worktree
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    # Use command adapter to avoid needing real devin binary
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Issue starts with in_progress label (active)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Ensure state directory exists
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize state with a branch/worktree entry for issue 42 (recovery dict)
    # The recovery matcher expects branch_name and status: "dispatched"
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["42"] = {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "labels": [config.labels.in_progress],
            "branch_name": "agent/issue-42-fix-search",
            "worktree_path": "/tmp/worktree-issue-42",
            "status": "dispatched",
        }
        save_state(paths.state_file, state)

    # Create a sessions directory with a dead session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a session record for a dead session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Step 1: Run classification pass directly
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # Verify the issue was relabeled to ready
    assert (42, config.labels.in_progress) in fake_gh.labels_removed
    assert (42, config.labels.ready) in fake_gh.labels_added

    # Update fake GitHub to reflect the relabeled state
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.ready}],
        }
    ]
    # Clear label tracking
    fake_gh.labels_added = []
    fake_gh.labels_removed = []

    # Clear throttle state so dispatch is not deferred
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state.pop("throttled_until", None)
        save_state(paths.state_file, state)

    # Step 2: Run dispatch pass - should select the relabeled issue
    app.gh.prs[0]["state"] = "CLOSED"
    dispatch_result = app.dispatch(limit=1)

    # Verify dispatch selected the issue
    assert dispatch_result.ok is True
    assert dispatch_result.data["selected_count"] == 1
    assert dispatch_result.data["sessions"][0]["issue_number"] == 42

    # Verify the recovery dict (branch_name/worktree_path) was preserved in state
    # The dispatch should have used the existing branch from state
    state_after_dispatch = load_state(paths.state_file)
    assert state_after_dispatch["issues"]["42"]["branch_name"] == "agent/issue-42-fix-search"
    assert state_after_dispatch["issues"]["42"]["worktree_path"] == "/tmp/worktree-issue-42"
    # Verify the recovery dict was actually passed to the adapter (non-None)
    # The test should fail if recovery is hardcoded to None at dispatch call sites
    assert dispatch_result.data["sessions"][0].get("recovery") is not None


def test_classify_dead_sessions_with_closed_pr_triggers_relabel(tmp_path: Path) -> None:
    """Issue #118 R2: dead session + prior CLOSED PR only ⇒ relabel fires.

    This is a workflow-level test driving _classify_dead_sessions_and_update_throttle_state
    to ensure the OPEN filter is enforced. Mutation gate: dropping the OPEN filter fails this test.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    # Use command adapter to avoid needing real devin binary
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Issue starts with in_progress label (active)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    # Prior CLOSED PR (not OPEN) - should NOT suppress relabel
    fake_gh.prs = [
        {
            "number": 1,
            "title": "Fix #42: search",
            "url": "https://example.test/pull/1",
            "headRefName": "agent/issue-42-fix-search",
            "baseRefName": "main",
            "body": "Closes #42",
            "state": "CLOSED",
            "labels": [],
            "isCrossRepository": False,
        }
    ]

    # Ensure state directory exists
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a sessions directory with a dead session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a session record for a dead session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Run classification pass directly
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # Verify relabel fired despite CLOSED PR (OPEN filter works)
    assert (42, config.labels.in_progress) in fake_gh.labels_removed
    assert (42, config.labels.ready) in fake_gh.labels_added


def test_classify_dead_sessions_with_open_pr_suppresses_relabel(tmp_path: Path) -> None:
    """Issue #118 R2: dead session + OPEN PR ⇒ no relabel.

    This is a workflow-level test driving _classify_dead_sessions_and_update_throttle_state
    to ensure the OPEN PR guard is enforced. Mutation gate: deleting the guard fails this test.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    # Use command adapter to avoid needing real devin binary
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Issue starts with in_progress label (active)
    fake_gh.issues = [
        {
            "number": 42,
            "title": "Fix search",
            "url": "https://example.test/issues/42",
            "body": "Search is broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    # OPEN PR - should suppress relabel
    fake_gh.prs = [
        {
            "number": 1,
            "title": "Fix #42: search",
            "url": "https://example.test/pull/1",
            "headRefName": "agent/issue-42-fix-search",
            "baseRefName": "main",
            "body": "Closes #42",
            "state": "OPEN",
            "labels": [],
            "isCrossRepository": False,
        }
    ]

    # Ensure state directory exists
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a sessions directory with a dead session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log with rate-limit signature
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    # Write a session record for a dead session (pid=None to simulate dead)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Run classification pass directly
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # Verify relabel did NOT fire (OPEN PR guard works)
    assert (42, config.labels.in_progress) not in fake_gh.labels_removed
    assert (42, config.labels.ready) not in fake_gh.labels_added


def test_classify_dead_sessions_terminal_label_only_is_left_alone(tmp_path: Path) -> None:
    """Issue #417 regression: same bug as the orphaned-worker sweep's, but
    for the sidecar-based reap lane -- a dead session whose issue carries
    ONLY a terminal label must be left alone: no labels touched, and no
    redispatch_at bump (which would otherwise spend down the
    max_auto_redispatch escalation cap for an issue that needs no automatic
    recovery at all). This test must fail against a head that regresses to
    the `if not active_labels and not needs_ready: continue` gate.
    """
    from charlie_work.config import AutoMergeConfig, DevinConfig as DevinCfg
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinCfg(
            dispatch_command=(sys.executable, "-c", "import sys; print('ok')"),
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 501,
            "title": "needs a human too",
            "url": "https://example.test/issues/501",
            "body": "",
            "labels": [{"name": config.labels.human_needed}],
        }
    ]

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-501.log"
    log_path.write_text("Some work done, then the process died.\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-501.json"
    record = SessionRecord(
        issue_number=501,
        branch="agent/issue-501-x",
        worktree_path="/tmp/worktree-501",
        prompt_path="/tmp/prompt-501.md",
        command=("devin", "--prompt-file", "/tmp/prompt-501.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    assert fake_gh.labels_added == []
    assert fake_gh.labels_removed == []

    state = load_state(paths.state_file)
    assert [e for e in state["events"] if e["kind"] == "session_failed_relabeled"] == []
    # No redispatch_at bookkeeping should have been written at all for this
    # issue -- the escalation-cap counter must not spend down on an issue
    # that needed no automatic recovery.
    entry = state["issues"].get("501", {})
    assert entry.get("redispatch_at") is None

    # The sidecar is still reaped -- issue #113 phantom-session protection is
    # unrelated to whether there was anything to relabel.
    assert not sidecar_path.exists()


def test_classify_dead_sessions_relabel_carries_required_reason(tmp_path: Path) -> None:
    """Issue #978: the dead-worker no-open-PR relabel path must emit a
    ``session_failed_relabeled`` event whose ``reason`` is always populated.
    Previously this site passed ``failure_kind`` only (which can be ``None``
    when classification is inconclusive), producing rows with neither
    ``reason`` nor a populated ``failure_kind`` -- the "neither field" shape."""
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    # Issue #1130: use a no-commits worktree so the relabel path (not salvage)
    # is taken. A worktree with commits (even dirty) is now salvaged.
    branch = "agent/issue-978"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    worktree_path = info.path
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 978, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 978,
            "title": "Test issue",
            "url": "https://example.test/issues/978",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.pr_create_return = 101

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    state = json.loads(state_file.read_text(encoding="utf-8"))
    events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events) == 1
    # The canonical "why" field is always present -- the "neither field"
    # shape cannot recur regardless of whether failure_kind was classified.
    assert events[0]["payload"]["reason"] == "dead_worker_no_open_pr"


def test_classify_dead_sessions_no_commits_relabels_to_ready_issue_1130(
    tmp_path: Path,
) -> None:
    """Issue #1130: a dead session whose worktree has NO commits ahead of base
    is not salvaged (there is nothing to push); it relabels to ready. This
    guards the relaxed ``ahead_count > 0`` condition against false positives
    on empty worktrees."""
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    branch = "agent/issue-1130-empty"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    worktree_path = info.path
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 1130, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 1130,
            "title": "Test issue",
            "url": "https://example.test/issues/1130",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.pr_create_return = 101

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    # No PR created — nothing to salvage.
    assert not gh.prs_created
    # Active label removed, ready label added — the ordinary relabel path.
    assert (1130, config.labels.in_progress) in gh.labels_removed
    assert (1130, config.labels.ready) in gh.labels_added

    state = json.loads(state_file.read_text(encoding="utf-8"))
    salvage_events = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert not salvage_events
    relabel_events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(relabel_events) == 1


def test_classify_dead_sessions_no_commits_relabels_to_ready(tmp_path: Path) -> None:
    """Issue #252: a clean worktree with no commits relabels to ready."""
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    branch = "agent/issue-254"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    sessions_dir, state_file = _make_classify_state(tmp_path)
    _write_dead_session_sidecar(sessions_dir, 254, branch, info.path)

    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 254,
            "title": "Test issue",
            "url": "https://example.test/issues/254",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.pr_create_return = 101

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    assert not gh.prs_created
    assert (254, config.labels.in_progress) in gh.labels_removed
    assert (254, config.labels.ready) in gh.labels_added
