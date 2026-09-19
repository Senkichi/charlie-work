"""Session-liveness drift tests for ``reconcile.detect_drift`` and
``reconcile.apply_fixes``.

Split out of ``tests/test_reconcile.py`` (issue #1559, Track-1):
dead/stalled/live session detection via sidecars and ``repo_root``,
provider-throttle classification, api-worker budget-ledger settlement at
the reap sites, and the provider_throttle fix lane. The ``session_failed``
relabel/escalation tests live in ``tests/test_reconcile_drift_session_failed.py``;
the worktree-salvage lane lives in ``tests/test_reconcile_drift_salvage.py``.
"""

from __future__ import annotations

import json
import pytest
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from _reconcile_fixtures import (
    FakeGitHub,
    _issue,
    _pr,
)
from _sessions_db_fixtures import make_sessions_db
from charlie_work.config import (
    OrchestratorConfig,
    PostMortemConfig,
)
from charlie_work.paths import (
    resolved_layout,
    runtime_paths,
)
from charlie_work.reconcile import (
    DriftItem,
    apply_fixes,
    detect_drift,
)
from charlie_work.state import empty_state


def test_detect_drift_provider_throttle_detected_with_repo_root(tmp_path: Path) -> None:
    """Test that detect_drift with repo_root detects dead sessions and classifies throttling."""
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()

    # Create a sessions directory with a dead session that has a rate-limit log
    # Use the default sessions_dir path from config
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

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect provider throttle
    throttle_drift = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert len(throttle_drift) == 1
    assert throttle_drift[0].issue_number == 42
    assert "rate_limited" in throttle_drift[0].detail
    assert "throttled_until" in throttle_drift[0].fix_actions[0]
    # Regression: reason/adapter_kind must survive on the DriftItem itself
    # (not just embedded in the detail string) so apply_fixes can thread them
    # into set_throttled_until -- see test_apply_fixes_provider_throttle_threads_reason_and_adapter_kind.
    assert throttle_drift[0].throttle_reason == "rate_limited"
    assert throttle_drift[0].throttle_adapter_kind == "devin"


def test_detect_drift_defers_dead_session_on_inconclusive_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #755: detect_drift must not reap a not-alive worker on the first
    inconclusive probe; it must respect ``max_inconclusive_probe_deferrals``.
    """
    from charlie_work.config import WatchdogConfig
    from charlie_work.devin_shell import SessionRecord, _sidecar_path as devin_sidecar_path
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(max_inconclusive_probe_deferrals=1),
    )
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("Session log\n", encoding="utf-8")

    sidecar_path = devin_sidecar_path(sessions_dir, 42)
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=99999,  # Dead PID
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    def _inconclusive_probe(*_args: object, **_kwargs: object) -> RealActivityProbe:
        return RealActivityProbe(
            sources=(
                ActivitySource(
                    name="sessions.db",
                    timestamp=None,
                    staleness_seconds=None,
                    error="no session found matching working_directory",
                ),
                ActivitySource(
                    name="devin_per_pid_log",
                    timestamp=None,
                    staleness_seconds=None,
                    error="no per-PID log found",
                ),
            )
        )

    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda _record: False)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _inconclusive_probe)

    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Sidecar must still be present and the deferral counter advanced.
    assert sidecar_path.exists(), "detect_drift should defer, not reap, on an inconclusive probe"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar.get("inconclusive_probe_deferred_count") == 1

    # No drift item should propose a reap/relabel/throttle for this session yet.
    assert not any(d.issue_number == 42 for d in drift)
    assert gh.labels_added == []
    assert gh.labels_removed == []


def test_apply_fixes_provider_throttle_sets_throttled_until() -> None:
    """Test that apply_fixes correctly sets throttled_until for provider throttle drift."""
    from datetime import UTC, datetime, timedelta

    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()

    # Create a provider throttle drift item
    throttled_until = (
        (datetime.now(UTC) + timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    )
    drift = [
        DriftItem(
            kind="provider_throttle_detected",
            issue_number=42,
            pr_number=None,
            detail="issue #42 session died with rate_limited",
            fix_actions=(f"set throttled_until={throttled_until}",),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    # Verify throttled_until is set in the new state
    assert new_state.get("throttled_until") == throttled_until
    # Original state should be unchanged
    assert state.get("throttled_until") is None


def test_apply_fixes_provider_throttle_threads_reason_and_adapter_kind() -> None:
    """A ``provider_throttle_detected`` drift item's reason/adapter_kind must
    reach ``set_throttled_until`` -- otherwise ``clear_quota_throttles``
    treats a devin/provider_auth throttle applied via ``reconcile --fix`` as
    claude-code-shaped (the field-unset default) and a later green ambient-CLI
    probe wrongly clears a throttle it never actually tested."""
    from datetime import UTC, datetime, timedelta

    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()

    throttled_until = (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    drift = [
        DriftItem(
            kind="provider_throttle_detected",
            issue_number=42,
            pr_number=None,
            detail="issue #42 session died with provider_auth",
            fix_actions=(f"set throttled_until={throttled_until}",),
            throttle_reason="provider_auth",
            throttle_adapter_kind="devin",
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    assert new_state.get("throttled_until") == throttled_until
    assert new_state.get("throttle_reason") == "provider_auth"
    assert new_state.get("throttle_adapter_kind") == "devin"


def test_detect_drift_without_repo_root_skips_session_check() -> None:
    """Test that detect_drift without repo_root does not check sessions."""
    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()

    # Run detect_drift without repo_root
    drift = detect_drift(gh, state, config)

    # Should not detect any session-related drift
    assert [d for d in drift if d.kind == "provider_throttle_detected"] == []
    assert [d for d in drift if d.kind == "session_failed_relabeled"] == []


def test_detect_drift_claude_code_session_collision_with_unrelated_open_pr(tmp_path: Path) -> None:
    """Issue #118: dead claude-code session issue 42 with unrelated open PR #42 should relabel.

    This is the collision test: issues and PRs share one number sequence, so a dead
    claude-code session for issue N plus any unrelated OPEN PR numbered N must still
    trigger relabel (the guard is keyed by issue, not PR number).
    """
    from charlie_work.claude_code import ClaudeWorkerRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    # Unrelated open PR #42 (does NOT link to issue 42 via branch or closing keyword)
    gh = FakeGitHub(
        prs=[_pr(42, "OPEN", head_ref="some-unrelated-branch")],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    # Create a sessions directory with a dead claude-code session
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

    # Write a claude-code worker record for a dead session (pid=None to simulate dead).
    # Filename must match claude_code.py's real sidecar convention (issue-{N}.claude.json,
    # see _sidecar_path in claude_code.py) so claude_code.read_worker_records actually
    # picks it up. The old "issue-42-claude-code.json" name never matched that glob and
    # only produced a drift item because devin_shell.py's pre-issue-#343-fix exclusion
    # check let it slip through as a phantom devin session.
    sidecar_path = sessions_dir / "issue-42.claude.json"
    record = ClaudeWorkerRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("claude", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,  # No launch error - exited normally
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect session_failed_relabeled despite unrelated open PR #42
    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) >= 1  # May be multiple if both adapters read the same issue
    assert all(d.issue_number == 42 for d in relabel_drift)
    assert config.labels.in_progress in relabel_drift[0].remove_labels


def test_detect_drift_live_session_no_pr_no_issue_active_drift(tmp_path: Path) -> None:
    """Issue #214: live session with no open PR should NOT trigger issue_active_label_no_open_pr.

    This test ensures that the drift rule checks session liveness before proposing
    label removal. A worker that is still running (is_alive() returns True) should
    not have its labels stripped even if it hasn't opened a PR yet.
    """
    import os
    from charlie_work.devin_shell import SessionRecord, _get_process_start_time
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    # Create a sessions directory with a LIVE session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("Worker is running...\n", encoding="utf-8")

    # Write a session record for a LIVE session (with a real PID that we'll mock as alive)
    # We use the current process's PID to ensure is_alive() returns True
    current_pid = os.getpid()
    current_start_time = _get_process_start_time(current_pid)
    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path="/tmp/worktree",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=current_pid,  # Use current PID to simulate live session
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        process_start_time=current_start_time,  # Use actual process start time
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should NOT detect issue_active_label_no_open_pr for live session
    issue_active_drift = [d for d in drift if d.kind == "issue_active_label_no_open_pr"]
    assert len(issue_active_drift) == 0, (
        "Should not emit issue_active_label_no_open_pr when session is still alive"
    )

    # Should also not emit session_failed_relabeled (session is alive)
    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) == 0, "Should not emit session_failed_relabeled for live session"


def test_detect_drift_dead_session_no_pr_still_triggers_issue_active_drift(tmp_path: Path) -> None:
    """Issue #214: dead session with no open PR should still trigger issue_active_label_no_open_pr.

    This test ensures that the drift rule still works correctly for dead sessions.
    When a session is dead (is_alive() returns False) and has no open PR, the drift
    rule should still propose label removal.
    """
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    # Create a sessions directory with a DEAD session
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a session log
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("Worker died...\n", encoding="utf-8")

    # Write a session record for a DEAD session (pid=None to simulate dead)
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
        error=None,
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect session_failed_relabeled (dead session with no open PR)
    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) >= 1, "Should emit session_failed_relabeled for dead session"

    # Should NOT detect issue_active_label_no_open_pr (mutually exclusive with session_failed_relabeled)
    issue_active_drift = [d for d in drift if d.kind == "issue_active_label_no_open_pr"]
    assert len(issue_active_drift) == 0, (
        "Should not emit issue_active_label_no_open_pr when session_failed_relabeled handles it"
    )


def test_detect_drift_launch_stalled_session(tmp_path: Path) -> None:
    """Issue #221: detect launch_stalled sessions (alive but hung at shim marker)."""
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.worker import _log_is_stalled_at_shim

    config = OrchestratorConfig()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()

    # Create a sessions directory with a launch_stalled session
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Write a log with the shim marker (frozen at ~424-425 bytes)
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    # Set mtime to 10 minutes ago (past the default 5-minute grace period)
    old_time = datetime.now(UTC) - timedelta(minutes=10)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    # Verify the log is detected as stalled
    now = datetime.now(UTC)
    assert _log_is_stalled_at_shim(log_path, config.watchdog.launch_stall_grace_minutes, now)

    # Write a session record for a dead session (non-existent PID)
    # The launch_stalled check only runs for alive sessions, so we test the helper directly
    issue_number = 42
    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path

    sidecar_path = devin_sidecar_path(sessions_dir, issue_number)
    record = SessionRecord(
        issue_number=issue_number,
        branch="agent/issue-42",
        worktree_path="/tmp/worktree-42",
        prompt_path="/tmp/prompt-42.md",
        command=("devin", "prompt.md"),
        pid=None,  # Dead session
        started_at="2026-07-09T00:00:00Z",
        log_path=str(log_path),
        error=None,
        process_start_time=None,
    )
    import json

    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Run detect_drift with repo_root
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Since the session is dead (pid=None), it won't be detected as launch_stalled
    # but the helper function test above confirms the detection logic works
    # This test verifies the integration doesn't crash with the new code
    assert len(drift) == 0  # No drift for dead sessions without open PRs


def test_detect_drift_launch_stalled_calls_kill_process_tree(tmp_path: Path) -> None:
    """Issue #221: launch_stalled path must call kill_process_tree with pid and process_start_time.

    Mutation check: this test FAILS against the old inline-kill code (which calls
    os.killpg / ctypes.TerminateProcess directly and never touches kill_process_tree)
    and PASSES against the fix (which calls kill_process_tree from process_utils).

    Issue #307: the real-activity probe must be conclusive (a genuinely stale,
    non-None timestamp from sessions.db) rather than left to hit the host's real
    sessions.db, which would produce an all-errored/inconclusive probe for this
    fake worktree and now correctly defer instead of killing.
    """
    import json
    import os
    from unittest.mock import patch

    from charlie_work.devin_shell import SessionRecord

    worktree_path = "/tmp/worktree-55"
    now = datetime.now(UTC)

    db_path = tmp_path / "sessions.db"
    make_sessions_db(
        db_path,
        session_id="sess-55",
        working_directory=worktree_path,
        created_at=now.isoformat(),
        rows=[
            {
                "role": "assistant",
                "content": "still working",
                # Stale past the launch-stall grace period: conclusive evidence
                # of a real stall, not the no-match-yet shape.
                "created_at": (now - timedelta(minutes=20)).isoformat(),
            }
        ],
    )

    config = OrchestratorConfig(post_mortem=PostMortemConfig(db_path=str(db_path)))
    gh = FakeGitHub(prs=[], issues=[_issue(55, [config.labels.in_progress])])
    state = empty_state()

    # detect_drift resolves the sessions dir through paths.resolved_layout
    # (config.devin.sessions_dir is a "" sentinel resolved against runtime.state_dir).
    sessions_dir = resolved_layout(config, tmp_path).sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Write a small log with only the shim marker — frozen well past grace period
    log_path = sessions_dir / "issue-55.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")
    old_time = now - timedelta(minutes=20)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    # Use a fake PID that passes is_alive() without actually checking the OS.
    # We patch is_session_alive so the worker reads as alive.
    fake_pid = 99999
    fake_start_time = 1700000000.0

    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path

    sidecar_path = devin_sidecar_path(sessions_dir, 55)
    record = SessionRecord(
        issue_number=55,
        branch="agent/issue-55",
        worktree_path=worktree_path,
        prompt_path="/tmp/prompt-55.md",
        command=("devin", "prompt.md"),
        pid=fake_pid,
        started_at="2026-07-09T00:00:00Z",
        log_path=str(log_path),
        error=None,
        process_start_time=fake_start_time,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    # Ensure no claude-code sidecar interferes
    (sessions_dir / "issue-55.claude.json").unlink(missing_ok=True)

    kill_calls: list[tuple[int, float | None]] = []

    def fake_kill(pid: int, expected_start_time: float | None = None) -> list[int]:
        kill_calls.append((pid, expected_start_time))
        return [pid]

    with (
        patch("charlie_work.worker.is_session_alive", return_value=True),
        patch("charlie_work.reconcile.kill_process_tree", fake_kill),
    ):
        detect_drift(gh, state, config, repo_root=tmp_path)

    assert len(kill_calls) == 1, (
        f"Expected kill_process_tree to be called exactly once, got {kill_calls}"
    )


# ---------------------------------------------------------------------------
# Issue #480: api-worker budget settlement wiring at the reconcile reap sites
# ---------------------------------------------------------------------------
#
# detect_drift has two production reap_sidecar call sites that wire
# ``api_config=config.api_worker, state_dir=state_dir_root`` so an api worker's
# spend is settled into the ledger before its sidecar is unlinked:
#   - the dead-session lane (~reconcile.py:440)
#   - the launch_stalled lane (~reconcile.py:304)
# Neither had any test coverage. A wiring regression at either site would
# silently disable budget tracking with no test failing. These two tests drive
# the real detect_drift path and assert the ledger is populated.
def test_detect_drift_dead_api_session_settles_budget_ledger(tmp_path: Path) -> None:
    """Dead api-worker session: detect_drift reaps and settles spend (issue #480).

    Covers the dead-session reap call site (~reconcile.py:440). A wiring
    regression that drops ``api_config``/``state_dir`` from that call leaves
    the sidecar reaped but the ledger empty — this assertion fails.
    """
    from _api_budget_fixtures import (
        api_worker_config,
        ledger_entries,
        write_api_events,
        write_api_sidecar,
    )

    config = OrchestratorConfig(
        api_worker=api_worker_config(),
        # Disable sessions.db post-mortem so the test does not touch a real
        # sessions.db; the wiring under test is the budget reap, not post-mortem.
        post_mortem=PostMortemConfig(enabled=False),
    )
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    sessions_dir = resolved_layout(config, tmp_path).sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    write_api_sidecar(sessions_dir, 42, provider="example")
    write_api_events(sessions_dir, 42)

    state_dir_root = runtime_paths(tmp_path, config.runtime.state_dir).root

    detect_drift(gh, state, config, repo_root=tmp_path)

    sessions = ledger_entries(state_dir_root)
    assert len(sessions) == 1, "dead api session must settle into the ledger"
    entry = sessions[0]
    assert entry.issue == 42
    assert entry.provider == "example"
    assert entry.model == "example-model"
    # 1M*3 + 0.2M*15 + 0.5M*0.30 = 6.15
    assert entry.usd == pytest.approx(6.15)


def test_detect_drift_launch_stalled_api_session_settles_budget_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch-stalled api-worker session: the launch_stalled reap settles spend.

    Covers the launch_stalled reap call site (~reconcile.py:304). That lane
    fires only for an alive-but-shim-frozen worker corroborated by a
    conclusive-stale real-activity probe. We patch ``is_worker_alive`` to True
    and ``real_activity_probe_for`` to a conclusive-stale probe so the lane
    runs for an api sidecar without spawning a real process. A wiring
    regression that drops the api kwargs from this call site leaves the ledger
    empty — this assertion fails.
    """
    import os as _os

    from _api_budget_fixtures import (
        api_worker_config,
        ledger_entries,
        write_api_events,
        write_api_sidecar,
    )
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe

    config = OrchestratorConfig(
        api_worker=api_worker_config(),
        post_mortem=PostMortemConfig(enabled=False),
    )
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(55, [config.labels.in_progress])],
    )
    state = empty_state()

    sessions_dir = resolved_layout(config, tmp_path).sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    write_api_sidecar(sessions_dir, 55, provider="example")
    write_api_events(sessions_dir, 55)

    # A shim-frozen log: small, contains the marker, stale past the grace period.
    log_path = sessions_dir / "issue-55.claude.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=20)
    _os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    # Force the api worker to read as alive so the launch_stalled lane runs.
    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: True)
    # Conclusive-stale probe: has a timestamp (not inconclusive) but stale past
    # the grace period (not fresh), so _log_is_stalled_at_shim returns True.
    stale_source = ActivitySource(
        name="test",
        timestamp=old_time,
        staleness_seconds=20 * 60,
        error=None,
    )
    monkeypatch.setattr(
        "charlie_work.worker.real_activity_probe_for",
        lambda w, cfg, now: RealActivityProbe(sources=(stale_source,)),
    )

    state_dir_root = runtime_paths(tmp_path, config.runtime.state_dir).root

    detect_drift(gh, state, config, repo_root=tmp_path)

    sessions = ledger_entries(state_dir_root)
    assert len(sessions) == 1, "launch_stalled api session must settle into the ledger"
    entry = sessions[0]
    assert entry.issue == 55
    assert entry.provider == "example"
    assert entry.usd == pytest.approx(6.15)
