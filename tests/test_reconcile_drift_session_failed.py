"""``session_failed`` relabel and escalation tests for ``reconcile``.

Split out of ``tests/test_reconcile.py`` (issue #1559, Track-1): the
``detect_drift`` ``session_failed`` drift lane and the matching
``apply_fixes`` relabel / escalation transitions.
"""

from __future__ import annotations

import json
from datetime import (
    UTC,
    datetime,
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
from charlie_work.devin_shell import SessionRecord
from charlie_work.reconcile import (
    DriftItem,
    apply_fixes,
    detect_drift,
)
from charlie_work.state import empty_state


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
