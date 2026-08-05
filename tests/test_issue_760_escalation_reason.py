"""Tests for issue #760: escalation via orphaned-worker path must record a reason.

Issue #760: an issue can reach ``status == "escalated"`` without an
``escalation_reason`` when the dead/orphaned-worker route trips the redispatch
cap. The fix introduces a single state-transition helper
(``set_status_escalated``) that requires a reason, and gives the dead-worker
route its own reason value (``orphaned_worker_unrecoverable``) distinct from the
generic ``redispatch_cap_exceeded``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from charlie_work.config import DevinConfig, OrchestratorConfig
from charlie_work.devin_shell import SessionRecord
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    set_status_escalated,
    state_lock,
)
from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

from test_charlie_work import FakeGitHub


def test_set_status_escalated_sets_status_reason_and_class() -> None:
    """``set_status_escalated`` is the single-point escalation transition and
    requires both ``escalation_reason`` and ``reason_class``.
    """
    entry: dict[str, object] = {"number": 123}

    result = set_status_escalated(
        entry,
        reason="orphaned_worker_unrecoverable",
        reason_class="mechanical",
    )
    assert result is entry
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "orphaned_worker_unrecoverable"
    assert entry["reason_class"] == "mechanical"

    with pytest.raises(ValueError):
        set_status_escalated({}, reason="x", reason_class="invalid")


def test_dead_rework_session_cap_uses_orphaned_worker_reason(
    tmp_path: Path,
) -> None:
    """Issue #760: a dead/launch-failed rework worker that exhausts the
    redispatch cap must escalate with ``orphaned_worker_unrecoverable`` so the
    parked issue is triageable without replaying its event history.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    fake_gh = FakeGitHub()
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]

    now = datetime.now(UTC)
    recent_redispatches = [
        (now - timedelta(minutes=m)).isoformat().replace("+00:00", "Z") for m in (6, 4, 2)
    ]

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "dispatched",
            "worker_pid": 99999,
            "worker_process_start_time": 1234567890.0,
            "branch_name": "agent/issue-123-fix-search",
            "redispatch_at": recent_redispatches,
        }
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "decision": "request_changes",
            "reviewed_head_sha": "sha-abc123",
        }
        save_state(paths.state_file, state)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-123.log"
    log_path.write_text(
        "Reached overall message rate limit. Your limit will reset in 0 minutes.\n",
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
        error="devin launch failed: rate limit",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config
    )

    state = load_state(paths.state_file)
    entry = state["issues"]["123"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "orphaned_worker_unrecoverable"
    assert entry["reason_class"] == "mechanical"
    assert (123, config.labels.human_needed) in fake_gh.labels_added
