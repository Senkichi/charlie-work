"""Dead rework-session classification: return to rework_requested, stale-prompt no-reopen, death-cap escalation, and no-op-cap accounting.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

import sys
from pathlib import Path
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from charlie_work.config import (
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from charlie_work.devin_shell import SessionRecord


def test_classify_dead_rework_session_returns_to_rework_requested(
    tmp_path: Path,
) -> None:
    """Issue #295: a dead/launch-failed rework session with an open PR and a
    LIVE request_changes verdict (still matching the PR's current head) must
    be restored to rework_requested so the next dispatch_rework can re-select
    it.

    Issue #315 review rework: a bare rework-prompt.md on disk is no longer
    sufficient by itself (see the stale-prompt regression test below) — the
    prompt file is never deleted, so has_request_changes is now the single
    signal that gates the restore. This test records a live request_changes
    decision (matching production: _write_rework_prompt and the
    decision/reviewed_head_sha state write happen in the same review() call)
    so it keeps exercising the restore path under the corrected semantics.
    It also exercises finding 2's window-filtered redispatch_at bookkeeping
    (previously this lane preserved redispatch_at unchanged and never grew
    it, which is exactly why the escalation cap could never trip).

    Mutation gate: dropping the request_changes check or the rework_requested
    status rollback from _reap_restore_rework_requested fails this test.
    """
    import json
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    # Issue is stuck in the dispatched state with the rework worker label.
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]

    # PR state records a LIVE request_changes decision matching the PR's
    # current head (fake_gh.prs[0]["headRefOid"] == "sha-abc123" by default) —
    # the only signal _reap_restore_rework_requested now honors (issue #315
    # finding 1). The on-disk rework-prompt.md below is still written (it's
    # what a real request_changes cycle produces) but is a diagnostic
    # supplement only, not required for the restore.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-fix-search",
            "prompt_path": str(paths.prs / "pr-456" / "rework-prompt.md"),
            "redispatch_at": ["2020-01-01T00:00:00Z"],
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    # Issue #1362 Stage 1: the reader is now file-first, so the live
    # request_changes decision must exist on disk, not only in state.json.
    pr_decision_dir = paths.prs / "pr-456"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    # Create the rework prompt on disk (the rework brief).
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    # Create a sessions directory with a launch-failure sidecar carrying an
    # ordinary (non-throttle) failure signature — under issue #1684 a
    # provider-throttle-classified death is exempt from cap bookkeeping, so
    # this fixture must be an ordinary failure to exercise it.
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text(
        "Error: worker exited before its first turn.\n",
        encoding="utf-8",
    )

    sidecar_path = sessions_dir / "issue-123.json"
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(rework_prompt),
        command=("devin", "--prompt-file", str(rework_prompt)),
        pid=None,  # launch-failure sidecar
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="devin launch failed: worker exited",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Run the reap pass.
    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # Verify state was restored to rework_requested for the owning lane.
    state = load_state(paths.state_file)
    entry = state["issues"].get("123")
    assert entry is not None
    assert entry.get("status") == "rework_requested"
    assert entry.get("dispatched_at") is None
    # Issue #315 finding 2: the old 2020 entry is outside the redispatch
    # window (default 240 minutes) and is dropped; a fresh entry is appended
    # in its place — proof the cap bookkeeping this lane previously skipped
    # now actually runs, while staying under config.watchdog.max_auto_redispatch
    # (default 3) so the restore (not escalation) path is taken.
    redispatch_at = entry.get("redispatch_at")
    assert redispatch_at is not None
    assert len(redispatch_at) == 1
    assert redispatch_at[0] != "2020-01-01T00:00:00Z"
    # Liveness fingerprint preserved for recovery path (issue #282)
    assert entry.get("worker_pid") == 99999
    assert entry.get("worker_process_start_time") == 1234567890.0
    # Label transitioned from in_progress to needs_rework
    assert (123, config.labels.in_progress) in fake_gh.labels_removed
    assert (123, config.labels.needs_rework) in fake_gh.labels_added

    # Next dispatch_rework should re-select the issue. Clear the throttle
    # window first so the test verifies the rework restore path, not the
    # provider-throttle deferral behavior (issue #499 adds a resume margin
    # that makes a "0 minutes" reset window a real 90-second deferral).
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state.pop("throttled_until", None)
        save_state(paths.state_file, state)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.dispatch_rework()
    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["sessions"][0]["issue_number"] == 123
    assert str(result.data["sessions"][0]["prompt_path"]).endswith("rework-prompt.md")


def test_classify_dead_rework_session_stale_prompt_does_not_reopen_approved_head(
    tmp_path: Path,
) -> None:
    """Issue #315 review finding 1: a stale rework-prompt.md left over from an
    earlier cycle must NOT roll a PR whose CURRENT head is already approved
    back to rework_requested. The prompt file is written once per PR
    (workflow._write_rework_prompt) and is never deleted, so its mere
    existence cannot distinguish "still awaiting this cycle's rework" from
    "leftover from a cycle that has since been approved" the way
    has_request_changes can (it re-derives from the PR's live review record
    on every call).

    Mutation gate: reverting _reap_restore_rework_requested's gate from
    `if not has_request_changes: return` back to
    `if not has_request_changes and not has_rework_prompt: return` makes this
    test fail — the stale prompt alone would incorrectly trigger the restore.
    """
    import json
    from datetime import UTC, datetime

    from charlie_work.config import DevinConfig
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
    # The PR's CURRENT head is approved (a fresh review cycle already ran and
    # passed) -- not request_changes.
    fake_gh.prs[0]["headRefOid"] = "sha-approved-head"

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-fix-search",
            "redispatch_at": [],
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "approved",
            "reviewed_head_sha": "sha-approved-head",
        }
        save_state(paths.state_file, state)

    # Stale rework-prompt.md left over from an EARLIER cycle (never deleted).
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the old issues", encoding="utf-8")

    # Dead worker that exited normally (no launch error). The worktree is
    # never created (is_completed=False), isolating this test to finding 1's
    # has_request_changes fix rather than finding 1's is_completed guard
    # (covered by test_classify_dead_rework_session_completed_worktree_not_rolled_back).
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = sessions_dir / "issue-123.json"
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),  # never created
        prompt_path=str(rework_prompt),
        command=("devin", "--prompt-file", str(rework_prompt)),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(sessions_dir / "issue-123.log"),
        error=None,  # exited normally
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    # Must NOT be rolled back -- the approved head is live, the prompt is stale.
    assert entry["status"] != "rework_requested"
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added


def test_classify_dead_rework_session_escalates_at_death_cap(
    tmp_path: Path,
) -> None:
    """Issue #315 review finding 2a + #1134: a dead rework worker must be
    escalated (not restored to rework_requested) once its death history hits
    config.watchdog.max_auto_redispatch.  Since every reap from this lane is
    a worker death (non-terminal failure), the prior redispatches are all
    deaths — seeded in ``worker_death_at`` to match.  The fourth death trips
    the death cap and escalates with ``worker_death_loop`` (not
    ``redispatch_cap_exceeded``), because a death is not a no-op.

    Mutation gate: dropping the ``death_loop`` half of
    _reap_restore_rework_requested's ``should_escalate`` check makes this
    test fail (the issue would be restored to rework_requested indefinitely
    instead of escalating).
    """
    import json
    from datetime import UTC, datetime, timedelta

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
    # fake_gh.prs[0]["headRefOid"] defaults to "sha-abc123".

    now = datetime.now(UTC)
    # Three recent deaths, all inside the default 240-minute window --
    # max_auto_redispatch defaults to 3, so a fourth death trips the cap.
    recent = [(now - timedelta(minutes=m)).isoformat().replace("+00:00", "Z") for m in (6, 4, 2)]

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-fix-search",
            "redispatch_at": recent,
            "worker_death_at": recent,  # all prior redispatches were deaths
        }
        # Live request_changes decision matching the current head, so this
        # test isolates the cap check rather than finding 1's gate.
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    # Issue #1362 Stage 1: the reader is now file-first, so the live
    # request_changes decision must exist on disk, not only in state.json.
    pr_decision_dir = paths.prs / "pr-456"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    # Launch-failure sidecar with a non-deterministic, non-throttle failure
    # signature -- isolates the cap check from finding 2b's deterministic-kind
    # guard (covered by the worktree_unsafe test below). A provider-throttle
    # signature would be exempt from cap bookkeeping under issue #1684, so the
    # fixture is an ordinary failure.
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text(
        "Error: worker exited before its first turn.\n",
        encoding="utf-8",
    )
    sidecar_path = sessions_dir / "issue-123.json"
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(paths.prs / "pr-456" / "rework-prompt.md"),
        command=("devin", "--prompt-file", "rework-prompt.md"),
        pid=None,  # launch-failure sidecar
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="devin launch failed: worker exited",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    assert entry["status"] == "escalated"
    # Issue #1134: deaths escalate as worker_death_loop, not redispatch_cap_exceeded.
    assert entry["escalation_reason"] == "worker_death_loop"
    assert len(entry["redispatch_at"]) == 4
    assert len(entry["worker_death_at"]) == 4
    # Issue #1266: worker_death_loop is mechanical, so this lands
    # agent:operator-queue, not agent:human-needed.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added

    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 123]
    assert "session_failed_escalated" in event_kinds
    assert "rework_requeued" not in event_kinds


def test_classify_dead_rework_session_no_op_cap_with_prior_no_ops(
    tmp_path: Path,
) -> None:
    """Issue #1134: when there are enough prior *genuine* no-op redispatches
    (no worker deaths), the no-op cap still fires even though the current
    reap is a death.  With 4 prior no-op redispatches (no ``worker_death_at``)
    and cap=3, the reap adds 1 redispatch + 1 death, making no_op_count = 4
    (5 redispatches - 1 death) which exceeds the cap → ``redispatch_cap_exceeded``.
    """
    import json
    from datetime import UTC, datetime, timedelta

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]

    now = datetime.now(UTC)
    # Four recent no-op redispatches (no worker_death_at — these were genuine
    # no-ops from the dispatch path, not deaths).  cap=3, so after the reap
    # adds 1 redispatch + 1 death: no_op_count = 5-1 = 4 > 3.
    recent_no_ops = [
        (now - timedelta(minutes=m)).isoformat().replace("+00:00", "Z") for m in (8, 6, 4, 2)
    ]

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-fix-search",
            "redispatch_at": recent_no_ops,
            # No worker_death_at — these were genuine no-ops.
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    # Issue #1362 Stage 1: the reader is now file-first, so the live
    # request_changes decision must exist on disk, not only in state.json.
    pr_decision_dir = paths.prs / "pr-456"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text(
        "Error: worker exited before its first turn.\n",
        encoding="utf-8",
    )
    sidecar_path = sessions_dir / "issue-123.json"
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(paths.prs / "pr-456" / "rework-prompt.md"),
        command=("devin", "--prompt-file", "rework-prompt.md"),
        pid=None,
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="devin launch failed: worker exited",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "redispatch_cap_exceeded"
    assert len(entry["redispatch_at"]) == 5
    # The reap recorded this as a death too, but the no-ops dominate.
    assert len(entry["worker_death_at"]) == 1


def test_classify_dead_rework_session_deaths_below_cap_not_escalated(
    tmp_path: Path,
) -> None:
    """Issue #1134: when both the death count and the no-op count are below
    the cap, the issue must NOT be escalated — it is restored to
    rework_requested for re-dispatch.  With 2 prior deaths (cap=3), the reap
    adds 1 death making death_count=3 (not > 3) and no_op_count=0.
    """
    import json
    from datetime import UTC, datetime, timedelta

    from charlie_work.config import DevinConfig
    from charlie_work.state import load_state, save_state, state_lock
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]

    now = datetime.now(UTC)
    recent = [(now - timedelta(minutes=m)).isoformat().replace("+00:00", "Z") for m in (6, 4)]

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-fix-search",
            "redispatch_at": recent,
            "worker_death_at": recent,  # both prior redispatches were deaths
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    # Issue #1362 Stage 1: the reader is now file-first, so the live
    # request_changes decision must exist on disk, not only in state.json.
    pr_decision_dir = paths.prs / "pr-456"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "request_changes", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text(
        "Error: worker exited before its first turn.\n",
        encoding="utf-8",
    )
    sidecar_path = sessions_dir / "issue-123.json"
    record = SessionRecord(
        issue_number=123,
        branch="agent/issue-123-fix-search",
        worktree_path=str(tmp_path / "worktrees" / "agent-123"),
        prompt_path=str(paths.prs / "pr-456" / "rework-prompt.md"),
        command=("devin", "--prompt-file", "rework-prompt.md"),
        pid=None,
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error="devin launch failed: worker exited",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    # Not escalated — death_count=3 is not > cap(3), no_op_count=0.
    assert entry["status"] == "rework_requested"
    assert len(entry["worker_death_at"]) == 3
    assert (123, config.labels.needs_rework) in fake_gh.labels_added
    assert (123, config.labels.human_needed) not in fake_gh.labels_added
