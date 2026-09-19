"""Session/worktree drift tests for ``reconcile.detect_drift`` and ``reconcile.apply_fixes``.

Split out of ``tests/test_reconcile.py`` (issue #1559, Track-1):
dead/stalled/live session detection via sidecars and ``repo_root``,
worktree salvage, provider-throttle classification, api-worker
budget-ledger settlement at the reap sites, and the matching
session_failed / provider_throttle / salvage fix lanes.
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
    _init_bare_remote_and_clone,
    _issue,
    _pr,
    _setup_completed_worktree,
)
from _sessions_db_fixtures import make_sessions_db
from _worktree_fixtures import _git
from charlie_work.config import (
    OrchestratorConfig,
    PostMortemConfig,
)
from charlie_work.devin_shell import SessionRecord
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
from charlie_work.worktree import create_worktree


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


def test_detect_drift_session_failed_relabeled_no_open_pr(tmp_path: Path) -> None:
    """Issue #118: dead session with no open PR should trigger label reconciliation."""
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

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

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect both provider throttle and session_failed_relabeled
    throttle_drift = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert len(throttle_drift) == 1
    assert throttle_drift[0].issue_number == 42

    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) >= 1  # May be multiple if both adapters read the same issue
    assert all(d.issue_number == 42 for d in relabel_drift)


def test_detect_drift_session_failed_with_open_pr_no_relabel(tmp_path: Path) -> None:
    """Issue #118: dead session with OPEN PR should NOT trigger label reconciliation."""
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "OPEN", head_ref="agent/issue-42-x")],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

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

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect provider throttle but NOT session_failed_relabeled
    throttle_drift = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert len(throttle_drift) == 1
    assert throttle_drift[0].issue_number == 42

    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) == 0


def test_detect_drift_session_failed_with_closed_pr_still_relabeled(tmp_path: Path) -> None:
    """Issue #118: dead session with CLOSED PR should still trigger label reconciliation.

    The guard only counts OPEN PRs, not CLOSED/MERGED. A prior closed PR should
    not permanently suppress the relabel.
    """
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[_pr(1, "CLOSED", head_ref="agent/issue-42-x")],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

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

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect both provider throttle and session_failed_relabeled
    throttle_drift = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert len(throttle_drift) == 1
    assert throttle_drift[0].issue_number == 42

    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) >= 1  # May be multiple if both adapters read the same issue
    assert all(d.issue_number == 42 for d in relabel_drift)
    assert config.labels.in_progress in relabel_drift[0].remove_labels


def test_apply_fixes_session_failed_relabeled(tmp_path: Path) -> None:
    """Issue #118: apply_fixes should remove active labels and add ready label."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    # Create a session_failed_relabeled drift item with structured add_labels
    drift = [
        DriftItem(
            kind="session_failed_relabeled",
            issue_number=42,
            pr_number=None,
            detail="issue #42 session died with rate_limited, no open PR",
            fix_actions=(
                f"remove label '{config.labels.in_progress}' from issue #42",
                f"add label '{config.labels.ready}' to issue #42",
            ),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.ready,),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    # Verify labels were removed and added
    assert (42, config.labels.in_progress) in gh.labels_removed
    assert (42, config.labels.ready) in gh.labels_added

    # Verify event was emitted
    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["kind"] == "session_failed_relabeled"
    assert reconcile_events[0]["payload"]["issue_number"] == 42


def test_apply_fixes_session_failed_relabeled_idempotent(tmp_path: Path) -> None:
    """Issue #118: re-running reconcile on already-relabeled issue should be idempotent."""
    config = OrchestratorConfig()
    # Issue already has ready label and no active labels (already relabeled)
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.ready])],
    )
    state = empty_state()

    # Create a session_failed_relabeled drift item (simulating re-run)
    drift = [
        DriftItem(
            kind="session_failed_relabeled",
            issue_number=42,
            pr_number=None,
            detail="issue #42 session died with rate_limited, no open PR",
            fix_actions=(
                f"remove label '{config.labels.in_progress}' from issue #42",
                f"add label '{config.labels.ready}' to issue #42",
            ),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.ready,),
        )
    ]

    # Should not error even though issue doesn't have in_progress label
    new_state = apply_fixes(gh, state, drift, config)

    # Verify the operation completed without error
    assert (42, config.labels.in_progress) in gh.labels_removed
    assert (42, config.labels.ready) in gh.labels_added

    # Verify event was emitted
    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1


def test_apply_fixes_session_failed_relabeled_carries_structured_reason(
    tmp_path: Path,
) -> None:
    """Issue #978: the ``reconcile`` event for a ``session_failed_relabeled``
    drift item must carry the machine-readable ``reason``/``failure_kind`` as
    structured payload fields, not only buried in the free-text ``detail``
    string. A query on ``json_extract(payload, '$.reason')`` must not return
    NULL the way it did when the "why" lived only in English prose."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    drift = [
        DriftItem(
            kind="session_failed_relabeled",
            issue_number=42,
            pr_number=None,
            reason="dead_session_no_open_pr",
            failure_kind="rate_limited",
            detail="issue #42 session died with rate_limited, no open PR",
            fix_actions=(
                f"remove label '{config.labels.in_progress}' from issue #42",
                f"add label '{config.labels.ready}' to issue #42",
            ),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.ready,),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    payload = reconcile_events[0]["payload"]
    assert payload["kind"] == "session_failed_relabeled"
    assert payload["reason"] == "dead_session_no_open_pr"
    assert payload["failure_kind"] == "rate_limited"


def test_apply_fixes_session_failed_relabeled_reason_absent_when_unset(
    tmp_path: Path,
) -> None:
    """Issue #978: a drift item that does not carry ``reason``/``failure_kind``
    must not produce a reconcile payload with those keys present-as-None. The
    payload shape for items without structured fields is unchanged."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    drift = [
        DriftItem(
            kind="session_failed_relabeled",
            issue_number=42,
            pr_number=None,
            detail="issue #42 session died, no open PR",
            fix_actions=(f"remove label '{config.labels.in_progress}' from issue #42",),
            remove_labels=(config.labels.in_progress,),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    payload = reconcile_events[0]["payload"]
    assert "reason" not in payload
    assert "failure_kind" not in payload


def test_detect_drift_session_failed_worker_blocked_escalates_instead_of_relabel(
    tmp_path: Path,
) -> None:
    """Issue #261 F5: a dead session whose post-mortem shows worker_blocked
    (killed by a push-gate hook) must NOT be relabeled to ready/redispatched
    like an ordinary dead session — that would hot-redispatch it straight
    back into the same hook and, per attempt_refs.py's motivation, destroy
    its unpushed local commits on the next branch reset. It must escalate
    (session_failed_escalated) instead, mirroring workflow.py's
    "redispatch_escalated" edge for the same signal."""
    worktree_path = str(tmp_path / "worktree")
    now = datetime.now(UTC)

    db_path = tmp_path / "sessions.db"
    make_sessions_db(
        db_path,
        session_id="sess-1",
        working_directory=worktree_path,
        created_at=now.isoformat(),
        rows=[
            {
                "role": "tool",
                "content": (
                    'Tool blocked: {"decision": "block", "reason": "push-gate hook rejected"}'
                ),
                "created_at": now.isoformat(),
            }
        ],
    )

    config = OrchestratorConfig(post_mortem=PostMortemConfig(db_path=str(db_path)))
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("some work then silence\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-42.json"
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path=worktree_path,
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,  # Dead session
        started_at=now.isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # No hot relabel-to-ready for this issue.
    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert relabel_drift == []

    escalated_drift = [d for d in drift if d.kind == "session_failed_escalated"]
    assert len(escalated_drift) == 1
    assert escalated_drift[0].issue_number == 42
    assert "worker_blocked" in escalated_drift[0].detail
    # Issue #807: detect_drift must carry failure_kind on the drift item so
    # apply_fixes can derive reason_class (judgment vs mechanical) instead of
    # hardcoding "mechanical".
    assert escalated_drift[0].failure_kind == "worker_blocked"

    # detect_drift is read-only regardless of the worker_blocked branch.
    assert gh.labels_added == []
    assert gh.labels_removed == []


def test_apply_fixes_session_failed_escalated_transitions_labels(tmp_path: Path) -> None:
    """Issue #261 F5: apply_fixes must transition session_failed_escalated
    via the 'redispatch_escalated' label edge (adds operator_queue, removes
    the other workflow labels) rather than removing active labels /
    re-adding ready like session_failed_relabeled does.

    Issue #1266: this DriftItem only ever fires for a deterministic
    failure_kind (see detect_drift), which workflow.py's own equivalent
    dead-session sweeps always treat as reason_class="mechanical" -- so the
    edge resolves to operator_queued, landing agent:operator-queue rather
    than agent:human-needed."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

    drift = [
        DriftItem(
            kind="session_failed_escalated",
            issue_number=42,
            pr_number=None,
            detail=(
                "issue #42 session died blocked by a push-gate hook (worker_blocked), "
                "no open PR; suppressing relabel-to-ready, escalating instead"
            ),
            fix_actions=("transition issue #42 labels via 'redispatch_escalated' event",),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    assert (42, config.labels.operator_queue) in gh.labels_added
    assert (42, config.labels.ready) not in gh.labels_added
    # ready must never be added for an escalated worker_blocked session.

    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["kind"] == "session_failed_escalated"
    assert reconcile_events[0]["payload"]["issue_number"] == 42


def test_apply_fixes_session_failed_escalated_judgment_lands_human_needed(
    tmp_path: Path,
) -> None:
    """Issue #807: a dead worker whose failure_kind is
    ``worktree_unsafe_local_commits`` (a deterministic *judgment* failure, not
    mechanical) must escalate via the ``redispatch_escalated`` edge
    (reason_class="judgment"), landing ``agent:human-needed`` -- NOT the
    mechanical ``redispatch_operator_queued`` edge that lands
    ``agent:operator-queue``.

    This is the reconcile.py apply-path regression for the #807 bug: before
    the fix, apply_fixes hardcoded reason_class="mechanical" for every
    session_failed_escalated drift item, so a genuine-local-commits death
    detected by reconcile's drift pass (before any workflow.py sweep) landed
    on operator_queue and became auto-clearable, reproducing the exact
    blanket-mechanical misclassification #807 split the failure kind to
    prevent."""
    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(43, [config.labels.in_progress])],
    )
    state = empty_state()

    drift = [
        DriftItem(
            kind="session_failed_escalated",
            issue_number=43,
            pr_number=None,
            detail=(
                "issue #43 session died with deterministic failure "
                "(worktree_unsafe_local_commits), no open PR; "
                "suppressing relabel-to-ready, escalating instead"
            ),
            fix_actions=("transition issue #43 labels via 'redispatch_escalated' event",),
            failure_kind="worktree_unsafe_local_commits",
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    # Judgment escalation lands human_needed, not operator_queue.
    assert (43, config.labels.human_needed) in gh.labels_added
    assert (43, config.labels.operator_queue) not in gh.labels_added
    assert (43, config.labels.ready) not in gh.labels_added

    reconcile_events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["kind"] == "session_failed_escalated"
    assert reconcile_events[0]["payload"]["issue_number"] == 43
    assert reconcile_events[0]["payload"]["failure_kind"] == "worktree_unsafe_local_commits"


def test_detect_drift_session_failed_already_has_ready_label(tmp_path: Path) -> None:
    """Issue #118: if issue already has ready label, don't add it again."""
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.ready, config.labels.in_progress])],
    )
    state = empty_state()

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

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect session_failed_relabeled but NOT add ready label action
    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) >= 1  # May be multiple if both adapters read the same issue
    assert all(d.issue_number == 42 for d in relabel_drift)
    assert config.labels.in_progress in relabel_drift[0].remove_labels
    # Should not have add ready label in structured field since it's already present
    assert relabel_drift[0].add_labels == ()


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


def test_detect_drift_session_failed_no_pr_mutually_exclusive_with_issue_active_no_pr(
    tmp_path: Path,
) -> None:
    """Issue #118: dead-session-with-no-PR-ever should emit only session_failed_relabeled.

    This test ensures that for a dead session with no PR ever created, we get exactly
    ONE drift item (session_failed_relabeled), not both session_failed_relabeled and
    issue_active_label_no_open_pr. The kinds must be mutually exclusive for a given issue.
    """
    from charlie_work.devin_shell import SessionRecord
    from datetime import UTC, datetime

    config = OrchestratorConfig()
    # Issue with active label, no PRs at all (not even closed)
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    state = empty_state()

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

    # Ensure no claude-code session exists (to avoid double-reading)
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    if claude_sidecar.exists():
        claude_sidecar.unlink()

    # Run detect_drift with repo_root to enable session checking
    drift = detect_drift(gh, state, config, repo_root=tmp_path)

    # Should detect session_failed_relabeled but NOT issue_active_label_no_open_pr
    relabel_drift = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel_drift) == 1, (
        f"Expected exactly 1 session_failed_relabeled, got {len(relabel_drift)}"
    )
    assert all(d.issue_number == 42 for d in relabel_drift)

    issue_active_drift = [d for d in drift if d.kind == "issue_active_label_no_open_pr"]
    assert len(issue_active_drift) == 0, (
        "Should not emit issue_active_label_no_open_pr when session_failed_relabeled handles it"
    )

    # Verify apply_fixes removes the label exactly once
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(42, [config.labels.in_progress])],
    )
    _ = apply_fixes(gh, state, drift, config)

    # Should have exactly one remove call for in_progress
    assert gh.labels_removed.count((42, config.labels.in_progress)) == 1


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
    (sessions_dir / f"issue-{issue_number}.claude.json").unlink(missing_ok=True)


# Adapter-kind -> sidecar filename suffix. Mirrors claude_code._ADAPTER_SIDECAR_SUFFIXES
# without importing it (keeps the test's failure surface independent of the adapter).
_ADAPTER_SIDECAR_SUFFIX = {"devin": "", "claude-code": ".claude", "api": ".api"}


def _write_dead_session_sidecar_for_adapter(
    sessions_dir: Path,
    issue_number: int,
    branch: str,
    worktree_path: Path,
    adapter_kind: str,
    log_text: str,
) -> Path:
    """Write a dead-session sidecar for any adapter kind, plus its log file.

    Unlike ``_write_dead_session_sidecar`` (devin-only, no log content), this
    also writes the log file with ``log_text`` so log-tail classification has
    real bytes to match against -- required for issue #656 regression coverage
    where the log must carry a throttle marker that *would* reclassify a
    non-completed session.
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"issue-{issue_number}.log"
    log_path.write_text(log_text, encoding="utf-8")
    suffix = _ADAPTER_SIDECAR_SUFFIX[adapter_kind]
    sidecar_path = sessions_dir / f"issue-{issue_number}{suffix}.json"
    if adapter_kind == "devin":
        record = SessionRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree_path),
            prompt_path="/tmp/prompt.md",
            command=("devin", "--prompt-file", "/tmp/prompt.md"),
            pid=None,
            started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            log_path=str(log_path),
            error=None,
        )
        sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    else:
        # claude-code / api share the ClaudeWorkerRecord on-disk shape; the
        # ``adapter_kind`` field disambiguates them (worker._from_claude_record
        # honors it so api sidecars surface as adapter_kind=="api").
        sidecar_path.write_text(
            json.dumps(
                {
                    "issue_number": issue_number,
                    "branch": branch,
                    "worktree_path": str(worktree_path),
                    "prompt_path": "/tmp/prompt.md",
                    "command": ["claude", "-p"],
                    "pid": None,
                    "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "log_path": str(log_path),
                    "error": None,
                    "adapter_kind": adapter_kind,
                }
            ),
            encoding="utf-8",
        )
    return sidecar_path


@pytest.mark.parametrize("adapter_kind", ["devin", "claude-code", "api"])
def test_detect_drift_completed_worktree_skips_log_tail_throttle_classification(
    tmp_path: Path, adapter_kind: str
) -> None:
    """Issue #656 regression: a completed worktree's log-tail throttle markers
    must NOT emit ``provider_throttle_detected`` drift.

    This guards the three ``session_completed=True`` call sites in
    ``reconcile.detect_drift`` (one per adapter kind). The log file is seeded
    with ``"usage limit"`` -- a ``match_quota_tail`` / ``quota_error_markers``
    substring that, if log-tail classification ran, would return
    ``quota_exhausted`` plus a 24h ``throttled_until`` and emit a
    ``provider_throttle_detected`` drift item.
    The worktree inspection is ground truth the session completed, so
    ``session_completed=True`` must skip log-tail matching entirely.

    If ``session_completed=True`` is silently dropped from any of the three
    call sites, this test fails: a ``provider_throttle_detected`` drift item
    appears and the salvage drift (which proves the is_completed lane was
    taken) is shadowed by the throttle.
    """
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    issue_number = 656
    worktree_path, branch = _setup_completed_worktree(repo_root, issue_number)

    sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
    # Log tail that quotes a throttle marker in legitimate completion prose --
    # the exact false-positive shape observed live 2026-07-27.
    _write_dead_session_sidecar_for_adapter(
        sessions_dir,
        issue_number,
        branch,
        worktree_path,
        adapter_kind,
        log_text=(
            '## Summary\n\nFixed generic substrings ("rate limit", "usage limit") '
            "that legitimately appear in this codebase's rate-limit/quota domain.\n"
        ),
    )

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(issue_number, [config.labels.in_progress])],
        repo_root=repo_root,
    )
    state = empty_state()

    drift = detect_drift(gh, state, config, repo_root=repo_root)

    # The is_completed lane was taken: salvage drift is emitted.
    salvage = [d for d in drift if d.kind == "session_unpublished_work_salvaged"]
    assert len(salvage) == 1
    assert salvage[0].issue_number == issue_number

    # The throttle must NOT fire despite the "usage limit" marker in the log --
    # session_completed=True skipped log-tail classification entirely.
    throttle = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert not throttle, (
        f"completed {adapter_kind} session was reclassified from log tail despite "
        f"session_completed=True (issue #656 regression): {throttle}"
    )


def test_detect_drift_completed_unpublished_work_salvaged(tmp_path: Path) -> None:
    """Issue #252: dead session with clean, ahead worktree emits salvage drift."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 252)

    sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_dead_session_sidecar(sessions_dir, 252, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(252, [config.labels.in_progress])],
        repo_root=repo_root,
    )
    state = empty_state()

    drift = detect_drift(gh, state, config, repo_root=repo_root)

    salvage = [d for d in drift if d.kind == "session_unpublished_work_salvaged"]
    assert len(salvage) == 1
    assert salvage[0].issue_number == 252
    assert salvage[0].branch == branch
    assert salvage[0].base_branch == "main"
    assert salvage[0].remove_labels == (config.labels.in_progress,)
    assert salvage[0].add_labels == (config.labels.pr_open,)

    # No relabel-to-ready drift should be emitted
    relabel = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert not relabel


def test_detect_drift_dirty_worktree_with_commits_salvaged(tmp_path: Path) -> None:
    """Issue #1130: dead session with a dirty worktree that has commits ahead
    of base emits salvage drift, not relabel-to-ready. The committed work is
    salvageable regardless of working-tree dirt (shim/scaffolding artifacts)."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 253, dirty=True)

    sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_dead_session_sidecar(sessions_dir, 253, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(253, [config.labels.in_progress])],
        repo_root=repo_root,
    )
    state = empty_state()

    drift = detect_drift(gh, state, config, repo_root=repo_root)

    salvage = [d for d in drift if d.kind == "session_unpublished_work_salvaged"]
    assert len(salvage) == 1
    assert salvage[0].issue_number == 253
    relabel = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert not relabel


def test_detect_drift_no_commits_relabels(tmp_path: Path) -> None:
    """Issue #252: dead session with no commits still relabels to ready."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    branch = "agent/issue-254"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    worktree_path = info.path

    sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_dead_session_sidecar(sessions_dir, 254, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(254, [config.labels.in_progress])],
        repo_root=repo_root,
    )
    state = empty_state()

    drift = detect_drift(gh, state, config, repo_root=repo_root)

    salvage = [d for d in drift if d.kind == "session_unpublished_work_salvaged"]
    assert not salvage
    relabel = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel) == 1
    assert relabel[0].issue_number == 254


def test_apply_fixes_salvage_success_creates_pr_and_labels(tmp_path: Path) -> None:
    """Issue #252: apply_fixes pushes, creates a PR, and moves labels to pr_open."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 255)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(255, [config.labels.in_progress])],
        repo_root=repo_root,
        pr_create_return=101,
    )

    drift = [
        DriftItem(
            kind="session_unpublished_work_salvaged",
            issue_number=255,
            pr_number=None,
            detail="salvage",
            fix_actions=("push", "pr_create"),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.pr_open,),
            branch=branch,
            base_branch="main",
        )
    ]

    new_state = apply_fixes(gh, empty_state(), drift, config)

    # PR created
    assert len(gh.prs_created) == 1
    assert gh.prs_created[0]["head"] == branch
    assert gh.prs_created[0]["base"] == "main"

    # Branch pushed to remote
    remote_refs = _git(remote, "show-ref")
    assert "agent/issue-255" in remote_refs.stdout

    # Labels moved
    assert (255, config.labels.in_progress) in gh.labels_removed
    assert (255, config.labels.pr_open) in gh.labels_added

    # Event recorded as salvage
    events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert events[-1]["payload"]["kind"] == "session_unpublished_work_salvaged"


def test_apply_fixes_salvage_push_failure_fallback(tmp_path: Path) -> None:
    """Issue #252: a failed salvage push falls back to relabel-to-ready."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 256)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(256, [config.labels.in_progress])],
        repo_root=repo_root,
        pr_create_return=102,
    )

    drift = [
        DriftItem(
            kind="session_unpublished_work_salvaged",
            issue_number=256,
            pr_number=None,
            detail="salvage",
            fix_actions=("push", "pr_create"),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.pr_open,),
            branch=branch,
            base_branch="main",
        )
    ]

    # Force push to fail
    import charlie_work.reconcile

    original_push_branch = charlie_work.reconcile.push_branch
    charlie_work.reconcile.push_branch = lambda repo, br, worktree_path=None: (
        False,
        "simulated push failure",
    )
    try:
        new_state = apply_fixes(gh, empty_state(), drift, config)
    finally:
        charlie_work.reconcile.push_branch = original_push_branch

    # No PR created, active label removed, ready label added
    assert not gh.prs_created
    assert (256, config.labels.in_progress) in gh.labels_removed
    assert (256, config.labels.ready) in gh.labels_added

    # Event recorded as failed relabel
    events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert events[-1]["payload"]["kind"] == "session_failed_relabeled"
    assert any("salvage_failed" in action for action in events[-1]["payload"]["fix_actions"])


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
