"""Dead-session classification escalation and sidecar retention: worker-blocked escalation, log-tail fallback, inconclusive-probe retention, conclusive-stale sidecar reap, and one-pass reclaim.

Split out of ``tests/test_charlie_work.py`` (issue #1551, Track-1
wave 5/8).
"""

from __future__ import annotations

import json
import sys
from datetime import (
    UTC,
    datetime,
)
from pathlib import Path
from unittest.mock import patch
import pytest
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from _sessions_db_fixtures import make_sessions_db
from charlie_work.config import (
    AutoMergeConfig,
    DevinConfig,
    OrchestratorConfig,
    PostMortemConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.devin_shell import SessionRecord


def test_classify_dead_sessions_worker_blocked_escalates_and_suppresses_redispatch(
    tmp_path: Path,
) -> None:
    """Issue #261 F5: a dead session whose post-mortem shows worker_blocked
    (killed by a push-gate hook) must escalate to human review instead of
    hot-relabeling to ready — a hot relabel would redispatch straight back
    into the same push-gate hook and, per attempt_refs.py's motivation,
    destroy the worker's unpushed commits on the next branch reset.

    This is the workflow-level counterpart to
    reconcile.py's test_detect_drift_session_failed_worker_blocked_escalates_instead_of_relabel
    and mirrors test_classify_dead_sessions_with_open_pr_suppresses_relabel's
    style, but the suppressing signal is a worker_blocked post-mortem verdict
    (no open PR at all) rather than an open PR.

    Mutation gate: dropping the `worker_blocked or` clause from the escalation
    condition at workflow.py's `_classify_dead_sessions_and_update_throttle_state`
    (the `if (worker_blocked or len(redispatch_at) > ...)` check) fails this test.
    """
    now = datetime.now(UTC)
    worktree_path = str(tmp_path / "worktree")

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

    # Use command adapter to avoid needing a real devin binary.
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        post_mortem=PostMortemConfig(db_path=str(db_path)),
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
    fake_gh.prs = []  # No open PR at all — the ordinary relabel path would fire here.

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

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
        error=None,  # No launch error - exited normally
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # (a) No hot relabel-to-ready — the escalation path must never grant the
    # `ready` label. Since dispatch() selects candidates via
    # gh.issue_list(config.labels.ready), an issue that never receives this
    # label can never be selected by a subsequent dispatch pass — this is a
    # structural (not merely incidental) proof that redispatch cannot fire.
    assert (42, config.labels.ready) not in fake_gh.labels_added

    # The escalation transition (redispatch_escalated) must have actually run:
    # operator_queue added, in_progress removed — proving escalation took the
    # GitHub-mutating path rather than silently no-oping. Issue #1266:
    # worker_blocked is mechanical, so it lands agent:operator-queue, not
    # agent:human-needed.
    assert (42, config.labels.operator_queue) in fake_gh.labels_added
    assert (42, config.labels.in_progress) in fake_gh.labels_removed

    # (b) escalation_reason recorded as worker_blocked, not the generic cap.
    state = load_state(paths.state_file)
    issue_entry = state["issues"]["42"]
    assert issue_entry["status"] == "escalated"
    assert issue_entry["escalation_reason"] == "worker_blocked"

    # No session_failed_relabeled event was appended for this issue — only
    # session_failed_escalated.
    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 42]
    assert "session_failed_relabeled" not in event_kinds
    assert "session_failed_escalated" in event_kinds


def test_classify_dead_sessions_worker_blocked_log_tail_fallback_escalates_and_suppresses_redispatch(
    tmp_path: Path,
) -> None:
    """Issue #260 (corrected premise): the same escalate/suppress-redispatch
    contract as test_classify_dead_sessions_worker_blocked_escalates_and_suppresses_redispatch
    above, but the worker_blocked signal comes from the log-tail fallback
    (post_mortem.classify_and_record's _classify_worker_blocked_from_log_tail)
    rather than a sessions.db match -- exercised here by pointing db_path at
    a location with no database at all, so DB-based extraction degrades to
    matched=False and the log tail ("Error: A tool was rejected by the
    user.", the Devin CLI's own PreToolUse hook-block surfacing) is the only
    signal available. This is the corrected #260 ask: this exact string was
    originally misclassified as a provider throttle signature (rate_limited,
    hot-redispatched after a cooldown) -- it must instead escalate on first
    occurrence, identically to a DB-detected "Tool blocked:" verdict.

    Mutation gate: this test is the log-tail-fallback counterpart of the
    DB-based test above and is covered by the same
    `if (worker_blocked or len(redispatch_at) > ...)` mutation at
    workflow.py's _classify_dead_sessions_and_update_throttle_state; it also
    independently covers post_mortem.classify_and_record's log-tail fallback
    branch (see test_post_mortem_log_tail_fallback.py's
    test_classify_and_record_log_tail_fallback_detects_worker_blocked_when_db_unavailable
    for that mutation gate's verbatim transcript).
    """
    import json
    from datetime import UTC, datetime

    from charlie_work.config import AutoMergeConfig, DevinConfig, PostMortemConfig
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.state import load_state

    now = datetime.now(UTC)
    worktree_path = str(tmp_path / "worktree")

    # No sessions.db at all -- DB-based extraction must degrade to
    # matched=False (extraction_error set), leaving the log tail as the
    # only signal.
    missing_db_path = tmp_path / "does-not-exist" / "sessions.db"

    # Use command adapter to avoid needing a real devin binary.
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit")
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        post_mortem=PostMortemConfig(db_path=str(missing_db_path)),
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
    fake_gh.prs = []  # No open PR at all — the ordinary relabel path would fire here.

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text("Error: A tool was rejected by the user.\n", encoding="utf-8")

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
        error=None,  # No launch error - exited normally
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
    )

    # No hot relabel-to-ready — same structural proof as the DB-based test.
    assert (42, config.labels.ready) not in fake_gh.labels_added

    # Escalation transition actually ran. Issue #1266: worker_blocked is
    # mechanical, so it lands agent:operator-queue, not agent:human-needed.
    assert (42, config.labels.operator_queue) in fake_gh.labels_added
    assert (42, config.labels.in_progress) in fake_gh.labels_removed

    # escalation_reason recorded as worker_blocked, not the generic cap or
    # (crucially, per the corrected premise) rate_limited.
    state = load_state(paths.state_file)
    issue_entry = state["issues"]["42"]
    assert issue_entry["status"] == "escalated"
    assert issue_entry["escalation_reason"] == "worker_blocked"

    # No throttle cooldown was set — a worker_blocked verdict must never
    # carry rate-limit retry semantics.
    assert state.get("throttled_until") is None

    event_kinds = [e["kind"] for e in state["events"] if e["payload"].get("issue_number") == 42]
    assert "session_failed_relabeled" not in event_kinds
    assert "session_failed_escalated" in event_kinds


@pytest.mark.real_activity_probe_live
def test_classify_dead_sessions_retains_sidecar_on_inconclusive_probe(
    tmp_path: Path,
) -> None:
    """Issue #343: a not-alive-looking pid with an inconclusive real-activity
    probe must have its sidecar RETAINED (deferred), not reaped, on this pass.

    Before this fix, ``_classify_dead_sessions_and_update_throttle_state``
    treated ``not w.is_alive()`` as sufficient grounds to relabel the issue
    and delete the sidecar unconditionally -- bypassing the same
    corroboration + inconclusive-probe deferral cap that
    ``classify_worker_health`` already enforces for the sibling stall/kill
    lane. That let a fail-open reap remove the sidecar of a worker whose
    liveness signal was merely ambiguous, leaving the underlying process
    invisible to the concurrency governor (issue #343's concrete production
    instance: pid 23440 verified alive via ``Get-Process``, yet its sidecar
    was removed after a ``matched: false`` post-mortem).

    Marked ``real_activity_probe_live`` so the autouse
    ``_stub_real_activity_probe_for_stalled_tests`` fixture (issue #307)
    leaves ``real_activity_probe_for`` unstubbed -- that stub always returns
    a 30-minute-stale-but-non-erroring probe, which is never "inconclusive"
    (``_real_activity_is_inconclusive`` requires every source to error), so
    it would defeat this test's premise. With the real probe, pointing
    ``post_mortem.db_path`` at a nonexistent path makes every source error
    deterministically regardless of what happens to be on the test host.

    MUTATION GATE: reverting the ``if health is not WorkerHealth.DEAD:
    continue`` gate in ``_classify_dead_sessions_and_update_throttle_state``
    (src/charlie_work/workflow.py) makes this test fail -- the sidecar would
    be reaped and the issue relabeled on this single pass.
    """
    from charlie_work.devin_shell import SessionRecord
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig(
        # Point post-mortem's sessions.db at a path that can never exist, so
        # the real-activity probe is deterministically inconclusive (every
        # source errors) regardless of what happens to be on the test host.
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "no-such-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 343,
            "title": "Ghost sidecar issue",
            "url": "https://example.test/issues/343",
            "body": "x",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    fake_gh.prs = []

    sessions_dir = paths.root / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-343.log"
    log_path.write_text("Working...\n", encoding="utf-8")

    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path

    sidecar_path = devin_sidecar_path(sessions_dir, 343)
    record = SessionRecord(
        issue_number=343,
        branch="agent/issue-343-x",
        worktree_path=str(tmp_path / "worktree-343"),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=54321,  # A pid our liveness check will report as gone (mocked below)
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        process_start_time=1_700_000_000.0,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    with patch("charlie_work.worker.is_session_alive", return_value=False):
        _classify_dead_sessions_and_update_throttle_state(
            sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
        )

    assert sidecar_path.exists(), "sidecar must be RETAINED when the probe is inconclusive"
    assert (343, config.labels.in_progress) not in fake_gh.labels_removed
    assert (343, config.labels.ready) not in fake_gh.labels_added

    # The Signal-1 deferral counter must have advanced so the escalation cap
    # (max_inconclusive_probe_deferrals) is still reachable over later passes.
    persisted = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert persisted.get("inconclusive_probe_deferred_count") == 1


def test_classify_dead_sessions_reaps_sidecar_when_probe_conclusively_stale(
    tmp_path: Path,
) -> None:
    """Regression pin: a genuinely dead pid whose corroboration probe is
    conclusively stale (not fresh, not inconclusive) is still reaped
    immediately -- the issue #343 fix must not invert into "never reap".
    """
    from datetime import timedelta

    from charlie_work.devin_shell import SessionRecord
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = [
        {
            "number": 344,
            "title": "Genuinely dead worker",
            "url": "https://example.test/issues/344",
            "body": "x",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    fake_gh.prs = []

    sessions_dir = paths.root / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-344.log"
    log_path.write_text("Working...\n", encoding="utf-8")

    from charlie_work.devin_shell import _sidecar_path as devin_sidecar_path

    sidecar_path = devin_sidecar_path(sessions_dir, 344)
    record = SessionRecord(
        issue_number=344,
        branch="agent/issue-344-x",
        worktree_path=str(tmp_path / "worktree-344"),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=54322,
        started_at=(datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        process_start_time=1_700_000_000.0,
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    from charlie_work.post_mortem import ActivitySource, RealActivityProbe

    stale_probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=datetime.now(UTC) - timedelta(hours=2),
                staleness_seconds=7200.0,
                error=None,
            ),
        )
    )

    with (
        patch("charlie_work.worker.is_session_alive", return_value=False),
        patch("charlie_work.worker.real_activity_probe_for", return_value=stale_probe),
    ):
        _classify_dead_sessions_and_update_throttle_state(
            sessions_dir, paths.state_file, fake_gh, config, write_gate=_wg(paths.state_file)
        )

    assert not sidecar_path.exists(), "a conclusively-stale probe must still allow reaping"
    assert (344, config.labels.in_progress) in fake_gh.labels_removed
    assert (344, config.labels.ready) in fake_gh.labels_added


def test_classify_dead_sessions_no_open_pr_happy_path_reclaims_in_one_pass(
    tmp_path: Path,
) -> None:
    """Issue #417: the fully-clean happy path (no interruption, no API
    failure) must still fully reclaim a dead session -- with no open PR -- in
    a single pass of the sidecar-based reap lane, and now records
    label_write_ok=True so a genuine future failure is distinguishable from
    success.
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
            "number": 99,
            "title": "Fix thing",
            "url": "https://example.test/issues/99",
            "body": "Broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]

    paths.state_file.parent.mkdir(parents=True, exist_ok=True)

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / "issue-99.log"
    log_path.write_text("Some work done, then the process died.\n", encoding="utf-8")

    sidecar_path = sessions_dir / "issue-99.json"
    record = SessionRecord(
        issue_number=99,
        branch="agent/issue-99-x",
        worktree_path="/tmp/worktree-99",
        prompt_path="/tmp/prompt-99.md",
        command=("devin", "--prompt-file", "/tmp/prompt-99.md"),
        pid=None,  # Dead session
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

    assert (99, config.labels.in_progress) in fake_gh.labels_removed
    assert (99, config.labels.ready) in fake_gh.labels_added

    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "session_failed_relabeled"]
    assert len(events) == 1
    assert events[0]["payload"]["label_write_ok"] is True
    assert events[0]["payload"]["added_ready"] is True
    assert (
        state["issues"]["99"]["redispatch_at"] and len(state["issues"]["99"]["redispatch_at"]) == 1
    )

    # The sidecar must be reaped once the reclaim fully succeeds.
    assert not sidecar_path.exists()
