"""Orphan-sweep redispatch caps: no-progress escalation and counter resets.

Split out of ``tests/test_charlie_work.py`` (issue #1547, Track-1 wave 1/8):
the ``test_orphan_sweep_redispatch_cap_*`` seam's trigger half -- a
no-progress redispatch loop escalates, head movement and stranded local
commits reset the counter, and repeated observations dedupe. Identity and
history bookkeeping lives in
``tests/test_charlie_work_orphan_sweep_identity.py``; shared fakes and
helpers in ``tests/_rework_dispatch_fixtures.py``.
"""

from __future__ import annotations

import subprocess
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import (
    _wg,
)
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
)


def test_orphan_sweep_redispatch_cap_escalates_after_no_progress_loop(
    tmp_path: Path,
) -> None:
    """Issue #1243: 3+ no-progress orphan-sweep redispatches must escalate
    instead of a 4th dispatch. The branch head is unchanged across attempts
    (no remote push, no local commits), so the cap fires parallel to the
    rework lane's worker_death_loop (death_count > max_auto_redispatch).

    The counter is a timestamp list (``orphan_redispatch_at``) appended once
    per *distinct dead dispatch* -- keyed by
    ``orphan_redispatch_counted_dispatch`` (a fingerprint of
    ``dispatched_at``/``worker_pid``) -- not ``len(adapter_history)``, which
    only grows when ``api_worker.enabled`` is ``True``. This test deliberately
    does NOT append to ``adapter_history`` between sweeps, and gives each
    simulated redispatch a distinct ``dispatched_at``, proving the cap fires
    on genuine redispatch attempts in the default (non-API-routed)
    configuration.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1243"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2026-08-14T00:00:00Z",
    }
    save_state(paths.state_file, state)

    class FakeGitHubNoPR(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubNoPR()
    fake_gh.issues = [
        {
            "number": 1243,
            "title": "test issue",
            "url": "https://example.test/issues/1243",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    def _run_sweep() -> None:
        with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
            _detect_and_handle_orphaned_workers(
                sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
            )

    def _simulate_redispatch(dispatch_index: int) -> None:
        """Simulate a dispatch between sweeps: reset the issue's GitHub labels
        to in_progress (the dispatch transition would do this), clear orphan
        flags (the dispatch clears them on success), and give the dispatch a
        fresh ``dispatched_at`` (a real redispatch always assigns a new
        timestamp). Crucially, this does NOT append to adapter_history -- in
        the default config (api_worker.enabled=False), the per-issue adapter
        selector that wrote it was deleted in Phase 2 Track B (PR #1517), so
        adapter_history never grows. The cap must fire without it.
        """
        st = load_state(paths.state_file)
        entry = st["issues"]["1243"]
        entry.pop("orphan_flagged_at", None)
        entry.pop("orphan_drift_fingerprint", None)
        entry.pop("orphan_drift_at", None)
        # A genuine redispatch always assigns a new dispatched_at timestamp,
        # which changes the dead-dispatch identity the cap dedupes on.
        entry["dispatched_at"] = f"2026-08-14T00:0{dispatch_index}:00Z"
        # FakeGitHub's label ops only record calls; simulate the dispatch
        # transition landing in_progress on GitHub.
        fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
        fake_gh.labels_added = []
        fake_gh.labels_removed = []
        save_state(paths.state_file, st)

    # Pass 1: first observation. orphan_redispatch_at = [now], count=1 <= 3,
    # proceed with reclaim.
    _run_sweep()
    st = load_state(paths.state_file)
    assert st["issues"]["1243"].get("status") == "dispatched"
    assert st["issues"]["1243"].get("orphan_redispatch_head_sha") == "none:none"
    assert len(st["issues"]["1243"].get("orphan_redispatch_at", [])) == 1
    # Simulate the reclaim's label change landing on GitHub.
    fake_gh.issues[0]["labels"] = [{"name": config.labels.ready}]

    # Simulate dispatch 2
    _simulate_redispatch(2)

    # Pass 2: head unchanged. count=2 <= 3, proceed.
    _run_sweep()
    st = load_state(paths.state_file)
    assert st["issues"]["1243"].get("status") == "dispatched"
    assert len(st["issues"]["1243"].get("orphan_redispatch_at", [])) == 2
    fake_gh.issues[0]["labels"] = [{"name": config.labels.ready}]

    # Simulate dispatch 3
    _simulate_redispatch(3)

    # Pass 3: head unchanged. count=3 <= 3, proceed.
    _run_sweep()
    st = load_state(paths.state_file)
    assert st["issues"]["1243"].get("status") == "dispatched"
    assert len(st["issues"]["1243"].get("orphan_redispatch_at", [])) == 3
    fake_gh.issues[0]["labels"] = [{"name": config.labels.ready}]

    # Simulate dispatch 4
    _simulate_redispatch(4)

    # Pass 4: head unchanged. count=4 > 3, ESCALATE!
    _run_sweep()
    st = load_state(paths.state_file)
    entry = st["issues"]["1243"]
    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "orphan_sweep_redispatch_cap_exceeded"
    assert entry.get("reason_class") == "mechanical"

    # Verify the dedicated event was emitted (not session_failed_escalated).
    escalated_events = [
        e for e in st.get("events", []) if e.get("kind") == "orphan_sweep_redispatch_escalated"
    ]
    assert len(escalated_events) == 1
    assert escalated_events[0]["payload"]["issue_number"] == 1243
    assert escalated_events[0]["payload"]["redispatch_count"] == 4
    assert escalated_events[0]["payload"]["reason"] == "orphan_sweep_redispatch_cap_exceeded"

    # The relabel event must NOT be emitted for the cap-exceeded pass.
    relabel_events = [
        e
        for e in st.get("events", [])
        if e.get("kind") == "session_failed_relabeled"
        and e["payload"].get("reason") == "dead_worker_no_open_pr_orphan_sweep"
    ]
    # Passes 1-3 each emitted one relabel event; pass 4 must not add another.
    assert len(relabel_events) == 3


def test_orphan_sweep_redispatch_cap_resets_on_moving_head(tmp_path: Path) -> None:
    """Issue #1243: a moving branch head (remote push or local stranded
    commits) resets the redispatch counter, so the cap does not fire even
    after max_auto_redispatch+1 attempts. A moving head with a dead worker
    is the salvage path's job, not escalation.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Start with 4 orphan_redispatch_at timestamps and a prior head
    # fingerprint, which would normally trigger the cap (count=4 > 3).
    base_time = datetime.now(UTC)
    state = load_state(paths.state_file)
    state["issues"]["1243"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": base_time.isoformat().replace("+00:00", "Z"),
        "branch_name": "agent/issue-1243-test",
        "orphan_redispatch_head_sha": "none:none",
        "orphan_redispatch_at": [
            (base_time - timedelta(minutes=i * 10)).isoformat().replace("+00:00", "Z")
            for i in range(4)
        ],
    }
    save_state(paths.state_file, state)

    class FakeGitHubNoPR(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubNoPR(repo_root=tmp_path)
    fake_gh.issues = [
        {
            "number": 1243,
            "title": "test issue",
            "url": "https://example.test/issues/1243",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    # Mock remote_branch_head_sha to return a NEW SHA (the worker pushed
    # something since the last orphan-sweep). The head fingerprint changes,
    # so the counter resets and the cap does not fire.
    new_sha = "abc123def456"
    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch("charlie_work.workflow.remote_branch_head_sha", return_value=new_sha),
        patch("charlie_work.workflow.worktree_head_sha", return_value=None),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    st = load_state(paths.state_file)
    entry = st["issues"]["1243"]

    # The cap must NOT fire -- the head moved, which is progress.
    assert entry.get("status") == "dispatched"
    assert entry.get("escalation_reason") is None

    # The head fingerprint must be updated to the new SHA.
    assert entry.get("orphan_redispatch_head_sha") == f"{new_sha}:none"

    # The orphan_redispatch_at list must be reset to a single entry (this
    # pass), so the redispatch_count starts from 1 again.
    assert len(entry.get("orphan_redispatch_at", [])) == 1

    # No escalation event must have been emitted.
    escalated_events = [
        e for e in st.get("events", []) if e.get("kind") == "orphan_sweep_redispatch_escalated"
    ]
    assert len(escalated_events) == 0


def test_orphan_sweep_redispatch_cap_resets_on_stranded_local_commits(
    tmp_path: Path,
) -> None:
    """Issue #1243: stranded LOCAL commits (worktree head moved, remote head
    unchanged) reset the redispatch counter. Unlike the moving-remote-head
    test above, this exercises the real ``worktree_head_sha`` against a real
    git worktree -- no mock -- so silently breaking the local half of the
    progress fingerprint fails here.
    """
    from unittest.mock import patch

    from charlie_work.paths import resolved_layout
    from charlie_work.worktree import worktree_path_for_branch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Build a REAL git repo at the exact path the sweep derives for this
    # branch, with one commit -- the stranded work.
    branch = "agent/issue-1243-test"
    worktrees_dir = resolved_layout(config, tmp_path).worktrees
    wt_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    wt_path.mkdir(parents=True, exist_ok=True)
    for args in (
        ["git", "init", "--initial-branch", branch],
        ["git", "config", "user.email", "test@example.test"],
        ["git", "config", "user.name", "Test User"],
    ):
        subprocess.run(args, cwd=wt_path, check=True, capture_output=True, text=True)
    (wt_path / "work.txt").write_text("stranded\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "work.txt"], cwd=wt_path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "stranded work"],
        cwd=wt_path,
        check=True,
        capture_output=True,
        text=True,
    )
    local_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=wt_path, check=True, capture_output=True, text=True
    ).stdout.strip()

    # Seed over-cap timestamps and a prior fingerprint whose REMOTE half
    # matches this pass (unchanged) but whose LOCAL half is an older SHA --
    # only the stranded local commit distinguishes this pass from the last.
    remote_sha = "feedfeedfeed"
    base_time = datetime.now(UTC)
    state = load_state(paths.state_file)
    state["issues"]["1243"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": base_time.isoformat().replace("+00:00", "Z"),
        "branch_name": branch,
        "orphan_redispatch_head_sha": f"{remote_sha}:0000000000000000",
        "orphan_redispatch_at": [
            (base_time - timedelta(minutes=i * 10)).isoformat().replace("+00:00", "Z")
            for i in range(4)
        ],
    }
    save_state(paths.state_file, state)

    class FakeGitHubNoPR(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubNoPR(repo_root=tmp_path)
    fake_gh.issues = [
        {
            "number": 1243,
            "title": "test issue",
            "url": "https://example.test/issues/1243",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    # remote_branch_head_sha is pinned to the SAME value as the prior
    # fingerprint's remote half; worktree_head_sha is deliberately NOT
    # patched -- the real implementation must read the real repo above.
    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch("charlie_work.workflow.remote_branch_head_sha", return_value=remote_sha),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    st = load_state(paths.state_file)
    entry = st["issues"]["1243"]

    # The cap must NOT fire -- the stranded local commit is progress.
    assert entry.get("status") == "dispatched"
    assert entry.get("escalation_reason") is None

    # The fingerprint's local half must be the REAL HEAD of the real repo.
    assert entry.get("orphan_redispatch_head_sha") == f"{remote_sha}:{local_sha}"

    # Counter reset to this single pass.
    assert len(entry.get("orphan_redispatch_at", [])) == 1

    escalated_events = [
        e for e in st.get("events", []) if e.get("kind") == "orphan_sweep_redispatch_escalated"
    ]
    assert len(escalated_events) == 0


def test_orphan_sweep_redispatch_cap_dedupes_repeated_observation(tmp_path: Path) -> None:
    """Issue #1243 round-3 fix: re-observing the SAME dead dispatch across
    multiple orphan-sweep passes must not grow the redispatch counter. The
    #417 reclaim deliberately leaves ``status``/``worker_pid`` stale on the
    dead entry, so without dedup on dispatch identity, a few sweep passes
    with zero real redispatch attempts (e.g. fleet-capacity dispatch delays)
    would escalate. Here the dispatch identity (``dispatched_at``:
    ``worker_pid``) never changes between passes, so the count must stay at
    1 and the cap must never fire, no matter how many passes run.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1243"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2026-08-14T00:00:00Z",
    }
    save_state(paths.state_file, state)

    class FakeGitHubNoPR(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubNoPR()
    fake_gh.issues = [
        {
            "number": 1243,
            "title": "test issue",
            "url": "https://example.test/issues/1243",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    def _run_sweep() -> None:
        with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
            _detect_and_handle_orphaned_workers(
                sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
            )

    # Run 5 sweep passes -- well past max_auto_redispatch=3 -- WITHOUT ever
    # changing dispatched_at/worker_pid, simulating the #417 reclaim leaving
    # the dead entry's identity untouched pass after pass.
    for pass_number in range(1, 6):
        _run_sweep()
        st = load_state(paths.state_file)
        entry = st["issues"]["1243"]
        assert entry.get("status") == "dispatched", f"must not escalate on pass {pass_number}"
        assert entry.get("escalation_reason") is None
        # Count must stay at 1 -- the same dead dispatch is being
        # re-observed, not a new redispatch attempt.
        assert len(entry.get("orphan_redispatch_at", [])) == 1, (
            f"orphan_redispatch_at must stay at 1 entry on pass {pass_number}"
        )
        assert entry.get("orphan_redispatch_counted_dispatch") == "2026-08-14T00:00:00Z:99999"
        # Reset the label the reclaim would have flipped, so the next pass
        # re-observes the same dead entry (still labeled in_progress on
        # GitHub in the real #417 scenario the reclaim tolerates).
        fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]

    escalated_events = [
        e
        for e in load_state(paths.state_file).get("events", [])
        if e.get("kind") == "orphan_sweep_redispatch_escalated"
    ]
    assert len(escalated_events) == 0
