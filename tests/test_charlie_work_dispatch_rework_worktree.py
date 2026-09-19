"""Rework-dispatch worktree safety: foreign writers and unsafe worktrees.

Split out of ``tests/test_charlie_work.py`` (issue #1547, Track-1 wave 1/8):
the ``test_dispatch_rework_*`` seam's worktree half -- conflicted/unsafe
worktrees escalate as judgment (preserving rework attempts), foreign-writer
redispatch counters, blocked-environment reaps, and the pre-escalation
safety nets that reap foreign writers before escalating. Shared fakes and
helpers in ``tests/_rework_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
import sys
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
import pytest
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import (
    _blocked_env_timestamps,
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
    state_lock,
)
from charlie_work.workflow import OrchestratorApp


def test_dispatch_rework_worktree_unsafe_local_commits_escalates_as_judgment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #807: a rework-dispatch attempt whose failure_kind is
    ``worktree_unsafe_local_commits`` (genuine unpushed local commits on the
    worktree branch) must escalate immediately on the FIRST attempt — like a
    deterministic mechanical failure — but with ``reason_class="judgment"`` so
    the label lands ``agent:human-needed`` (not ``agent:operator-queue``) and
    the de-escalation sweep never auto-clears it.

    Mirrors ``test_dispatch_worktree_unsafe_local_commits_escalates_as_judgment``
    (the fresh-dispatch site) but covers ``_dispatch_rework_impl``'s escalation
    branch — the rework-dispatch counterpart, which carries the same
    reason_class-derivation logic.

    Mutation gate: dropping the ``deterministic_judgment`` half of
    ``immediate_escalation`` in ``_dispatch_rework_impl``'s escalation branch
    makes this test fail (the issue takes the redispatch-cap path and stays
    ``rework_requested`` instead of escalating). Reverting the
    ``_escalation_edge("redispatch_escalated", reason_class)`` to the hardcoded
    ``"mechanical"`` also fails it (the label lands ``agent:operator-queue``
    while state carries ``reason_class="judgment"``).
    """
    from charlie_work.adapters import SessionDispatchResult

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=False,
                error="launch failed: worktree contains local commits",
                failure_kind="worktree_unsafe_local_commits",
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    # A deterministic judgment failure escalates on the FIRST attempt, not
    # after burning max_auto_redispatch.
    result = app.dispatch_rework()
    assert result.ok is False
    assert result.data["failed_count"] == 1
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "worktree_unsafe_local_commits"
    assert state["issues"]["123"]["reason_class"] == "judgment"
    # Escalated on the FIRST failure — not after burning max_auto_redispatch.
    assert len(state["issues"]["123"]["redispatch_at"]) == 1
    # Issue #807: judgment -> agent:human-needed, not agent:operator-queue.
    assert (123, config.labels.human_needed) in fake_gh.labels_added
    assert (123, config.labels.operator_queue) not in fake_gh.labels_added
    assert (123, config.labels.needs_rework) in fake_gh.labels_removed


def test_dispatch_rework_worktree_unsafe_preserves_conflict_rework_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #777(d): the conflict-rework attempt must be counted (via
    _route_janitor_gate_failure_to_rework's conflict_rework_attempts write,
    which merge_ready's dispatch trigger now goes through) BEFORE a later,
    deterministic worktree_unsafe failure at actual worker-launch time can
    escalate the issue through dispatch_rework's separate failure_kind lane.

    Real corpus: PR #679/issue #602, escalation_reason="worktree_unsafe".
    dispatch_rework's deterministic-failure branch only ever writes to
    state["issues"][...] -- it must never zero or otherwise touch the PR
    record's conflict_rework_attempts counter.
    """
    from charlie_work.adapters import SessionDispatchResult
    from charlie_work.config import AutoMergeConfig, DevinConfig, WatchdogConfig

    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed", "Lint & Format", "Pre-commit"),
            update_open_prs="next",
            failed_attempt_alarm=1,
        ),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=3, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "baseRefName": "main",
            "headRefOid": "sha-abc123",
            "mergeStateStatus": "DIRTY",
            "mergeable": "CONFLICTING",
            "body": "Closes #123\n\nTests: regression coverage added.",
            "labels": [],
            "isCrossRepository": False,
        },
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    dispatch_result = app.merge_ready(456, merge=False)
    assert dispatch_result.ok is True
    assert dispatch_result.data["merge_conflict"] is True

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1
    prompt_path = paths.prs / "pr-456" / "rework-prompt.md"
    assert prompt_path.exists()

    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=False,
                error="worktree is unsafe to reuse",
                failure_kind="worktree_unsafe_shim_dirt",
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    result = app.dispatch_rework()
    assert result.ok is False

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "worktree_unsafe_shim_dirt"
    # Issue #1266: this deterministic-failure-kind escalation is mechanical,
    # so it lands agent:operator-queue, not agent:human-needed.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added

    # The attempt counted by the (now-unified) conflict-rework dispatch must
    # survive this SEPARATE escalation lane untouched -- it lives on the PR
    # record, and dispatch_rework's deterministic-failure branch only ever
    # writes to the issue record.
    assert state["prs"]["456"]["conflict_rework_attempts"] == 1


def test_dispatch_rework_worktree_foreign_writer_does_not_increment_redispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1393: a rework dispatch that fails at launch with
    worktree_foreign_writer must NOT increment the redispatch counter.

    A pre-launch environment block (the worktree is a foreign checkout the
    orchestrator did not create) never started a worker session, so it is
    not a "worker produced nothing" signal.  Counting it against the
    redispatch cap converts an operator hygiene problem into a fake
    "worker quality" escalation (redispatch_cap_exceeded).  Instead, the
    failure is recorded in a separate blocked_environment_at list and a
    distinct rework_dispatch_blocked_environment event is emitted.
    """
    from charlie_work.adapters import SessionDispatchResult

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    blocking_error = (
        "worktree C:\\repos\\jc-wt-886 is a foreign checkout the "
        "orchestrator did not create; refusing to adopt it"
    )

    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=False,
                error=blocking_error,
                failure_kind="worktree_foreign_writer",
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    # First blocked launch: blocked_environment_at grows, redispatch_at stays empty.
    result = app.dispatch_rework()
    assert result.ok is False
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["issues"]["123"].get("redispatch_at") is None
    assert len(state["issues"]["123"].get("blocked_environment_at", [])) == 1

    # Second blocked launch: still under the cap, still rework_requested.
    result = app.dispatch_rework()
    assert result.ok is False
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["issues"]["123"].get("redispatch_at") is None
    assert len(state["issues"]["123"].get("blocked_environment_at", [])) == 2

    # Third pass: the candidate-filtering safety net sees
    # blocked_environment_at == 2 >= max_auto_redispatch (2) and escalates
    # with dispatch_blocked_environment BEFORE attempting another launch.
    # This is the correct behavior: the environment conflict is
    # deterministic, so there is no point dispatching again.
    result = app.dispatch_rework()
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "dispatch_blocked_environment"
    assert state["issues"]["123"].get("redispatch_at") is None
    assert len(state["issues"]["123"]["blocked_environment_at"]) == 2
    # Mechanical -> operator-queue, not human-needed.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


def test_dispatch_rework_worktree_foreign_writer_redispatch_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1393: after blocked-environment failures are resolved, a
    subsequent genuine redispatch failure still counts against the
    redispatch cap correctly — the blocked_environment_at list did not
    silently inflate it.
    """
    from charlie_work.adapters import SessionDispatchResult

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(max_auto_redispatch=2, redispatch_window_minutes=240),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    rework_prompt = pr_dir / "rework-prompt.md"
    rework_prompt.write_text("Fix the issues", encoding="utf-8")

    # First: one blocked-environment failure (does NOT touch redispatch_at).
    def fake_blocked(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=False,
                error="foreign checkout",
                failure_kind="worktree_foreign_writer",
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_blocked)
    app.dispatch_rework()
    state = load_state(paths.state_file)
    assert state["issues"]["123"].get("redispatch_at") is None
    assert len(state["issues"]["123"].get("blocked_environment_at", [])) == 1

    # Now: a generic (non-blocked) failure.  redispatch_at should start
    # from 0 (the blocked failure did not inflate it), so the cap is
    # NOT exceeded on the first generic failure.
    def fake_generic(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=False,
                error="git worktree add failed",
                failure_kind=None,
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_generic)
    result = app.dispatch_rework()
    assert result.ok is False
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert len(state["issues"]["123"]["redispatch_at"]) == 1


def test_dispatch_rework_blocked_environment_reap_resets_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1423: at the rework-dispatch blocked-environment cap exhaustion
    (Site 3 — the post-dispatch safety net), a successful foreign-writer reap
    resets ``blocked_environment_at`` and records the reap in
    ``foreign_writer_reaps`` instead of escalating.

    Site 3 is defense-in-depth: the pre-filter (Site 2) uses ``>=`` and Site 3
    uses ``>``, so in normal flow Site 2 always catches an at-cap issue before
    Site 3 can fire. Site 3 is reachable when Site 2's reap succeeds (resetting
    the counter) and ``max_auto_redispatch=0`` so the post-dispatch
    ``len([now]) > 0`` fires Site 3. This test uses that path: Site 2 reaps
    (recording reap #1), the dispatch fails, and Site 3 reaps again (recording
    reap #2). Two reaps confirm Site 3 fired — without it, only reap #1 would
    be recorded and ``blocked_environment_at`` would hold one entry.
    """
    from charlie_work.adapters import SessionDispatchResult
    from charlie_work.config import WRITER_MARKER_FILENAME
    from charlie_work.worktree import worktree_path_for_branch

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(
            max_auto_redispatch=0, redispatch_window_minutes=240, max_foreign_writer_reaps=2
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")

    # Materialize a marker at the worktree path derived from the PR head ref
    # so Site 2 (pre-filter) reaps and lets the issue proceed to dispatch.
    branch_pre = str(fake_gh.prs[0]["headRefName"])
    wt_path_pre = worktree_path_for_branch(tmp_path, branch_pre, app._layout.worktrees)
    wt_path_pre.mkdir(parents=True, exist_ok=True)
    marker = {
        "pid": 1234,
        "session_id": "foreign-session",
        "started_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        "kind": "worker",
        "process_start_time": 1234567890.0,
    }
    (wt_path_pre / WRITER_MARKER_FILENAME).write_text(json.dumps(marker), encoding="utf-8")

    monkeypatch.setattr("charlie_work.workflow.is_pid_alive", lambda pid, start: True)
    # _reap_idle_foreign_writer is shared by Site 2 (inline) and Site 3 (via
    # _try_reap_blocked_foreign_writer). Return True without removing the
    # marker so Site 3 can read it too.
    monkeypatch.setattr(
        "charlie_work.workflow._reap_idle_foreign_writer",
        lambda worktree_path, marker, _config, _sessions_dir=None, **_kw: True,
    )

    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=False,
                error="worktree has a live foreign writer",
                failure_kind="worktree_foreign_writer",
                pid=1234,
                worktree_path=str(wt_path_pre),
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    result = app.dispatch_rework()
    assert result.ok is False
    state = load_state(paths.state_file)
    # Both Site 2 and Site 3 reaped: two reap timestamps recorded. If Site 3
    # had not fired, only one reap would be recorded and
    # blocked_environment_at would hold one entry (the post-dispatch failure).
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["issues"]["123"]["blocked_environment_at"] == []
    assert len(state["issues"]["123"].get("foreign_writer_reaps", [])) == 2


def test_dispatch_rework_pre_escalation_safety_net_reaps_foreign_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1423: the rework candidate pre-filter loop's blocked-environment
    safety net reaps an idle foreign writer before escalating. The marker is
    read from the worktree path derived from the PR's head ref; on a
    successful reap the issue proceeds as a legitimate candidate instead of
    being escalated."""
    from charlie_work.adapters import SessionDispatchResult
    from charlie_work.config import WRITER_MARKER_FILENAME
    from charlie_work.worktree import worktree_path_for_branch

    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)")),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(
            max_auto_redispatch=2, redispatch_window_minutes=240, max_foreign_writer_reaps=2
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "blocked_environment_at": _blocked_env_timestamps(2),
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")

    # Materialize the worktree dir the safety net derives from the PR head ref,
    # with a live writer marker so the reap path has something to read.
    branch_pre = str(fake_gh.prs[0]["headRefName"])
    wt_path_pre = worktree_path_for_branch(tmp_path, branch_pre, app._layout.worktrees)
    wt_path_pre.mkdir(parents=True, exist_ok=True)
    marker = {
        "pid": 1234,
        "session_id": "foreign-session",
        "started_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        "kind": "worker",
        "process_start_time": 1234567890.0,
    }
    (wt_path_pre / WRITER_MARKER_FILENAME).write_text(json.dumps(marker), encoding="utf-8")

    monkeypatch.setattr("charlie_work.workflow.is_pid_alive", lambda pid, start: True)
    reap_calls: list[int] = []

    def _fake_reap(worktree_path, marker, _config, _sessions_dir=None, **_kw):
        reap_calls.append(marker["pid"])
        return True

    monkeypatch.setattr("charlie_work.workflow._reap_idle_foreign_writer", _fake_reap)

    # The actual dispatch after the reap succeeds so the issue is dispatched.
    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=True,
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)

    app.dispatch_rework()

    # The safety net reaped the foreign writer instead of escalating.
    assert reap_calls == [1234]
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] != "escalated"
    assert state["issues"]["123"]["blocked_environment_at"] == []
    assert len(state["issues"]["123"].get("foreign_writer_reaps", [])) == 1


def test_dispatch_rework_pre_filter_own_live_session_not_reaped_escalated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1443 review: an integration-level regression test exercising
    Site 2 (the rework pre-filter in ``_dispatch_rework_impl``) through the
    REAL, non-mocked ``_reap_idle_foreign_writer`` with a genuine own-live-
    session marker (a sidecar in ``sessions_dir`` matching the marker's
    session id and pid).

    The own-live-session guard must short-circuit the reap so the marker is
    NOT removed and the issue is escalated (``dispatch_blocked_environment``)
    instead of being reaped as ``foreign_writer_reaped``. This pins the guard
    the rework pre-filter previously re-implemented incompletely (stale-PID
    only) and dropped, and would fail if ``sessions_dir`` were later dropped
    or misordered at the Site 2 call site — the exact bug shape #1443 was
    filed to fix. Unlike the caller-level tests that mock the reap away, this
    one asserts on the real reap's behavior end-to-end."""
    from charlie_work.config import WRITER_MARKER_FILENAME
    from charlie_work.post_mortem import RealActivityProbe
    from charlie_work.worktree import (
        read_worktree_marker,
        worktree_path_for_branch,
    )

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)"),
        ),
        worker=WorkerRoleConfig(harness="command"),
        watchdog=WatchdogConfig(
            max_auto_redispatch=2,
            redispatch_window_minutes=240,
            max_foreign_writer_reaps=2,
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": config.labels.needs_rework}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
            "blocked_environment_at": _blocked_env_timestamps(2),
        }
        save_state(paths.state_file, state)

    fake_gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")

    # Materialize a marker at the worktree path derived from the PR head ref.
    # The marker's session id matches a live sidecar in sessions_dir, so the
    # real ``_reap_idle_foreign_writer`` own-live-session guard must refuse to
    # reap it (the stall detector owns reaping our own workers).
    branch_pre = str(fake_gh.prs[0]["headRefName"])
    wt_path_pre = worktree_path_for_branch(tmp_path, branch_pre, app._layout.worktrees)
    wt_path_pre.mkdir(parents=True, exist_ok=True)
    marker = {
        "pid": 1234,
        "session_id": "owned-session",
        "started_at": (datetime.now(UTC) - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        "kind": "worker",
        "process_start_time": 1234567890.0,
    }
    (wt_path_pre / WRITER_MARKER_FILENAME).write_text(json.dumps(marker), encoding="utf-8")

    # Genuine own-live-session sidecar: session id + pid match the marker, and
    # ``is_pid_alive`` is mocked True so ``_own_live_session_pids`` reports the
    # session as live. This is what makes the marker "owned", not foreign.
    sessions_dir = app._layout.sessions_dir
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "issue-123.json").write_text(
        json.dumps({"session_id": "owned-session", "pid": 1234, "process_start_time": 1.0}),
        encoding="utf-8",
    )

    # Mock only the process-liveness, activity-probe, and kill primitives (no
    # real process or filesystem interaction). The reap function itself is NOT
    # mocked — this is the integration assertion. ``is_pid_alive`` True keeps
    # the marker "live" so the reap reaches the own-live-session guard instead
    # of the stale-pid short-circuit. The activity probe is forced not-fresh so
    # that WITHOUT the own-live-session guard the reap would proceed to kill
    # (and the test would fail) — this is what makes the mutation check
    # deterministic: the guard is the only thing standing between the marker
    # and the kill path.
    monkeypatch.setattr("charlie_work.worktree.is_pid_alive", lambda pid, start: True)
    monkeypatch.setattr(
        "charlie_work.worktree.real_activity_for_worker",
        lambda *a, **k: RealActivityProbe(sources=()),
    )
    kill_calls: list[tuple[int, float | None]] = []
    monkeypatch.setattr(
        "charlie_work.worktree.kill_process_tree",
        lambda pid, st: kill_calls.append((pid, st)) or [pid],
    )
    monkeypatch.setattr("charlie_work.worktree.sweep_orphan_processes", lambda wt: [])
    monkeypatch.setattr("charlie_work.worktree.kill_orphan_pid", lambda pid: None)

    # The issue must be escalated pre-dispatch, so dispatch_sessions must not
    # be called. A raising fake makes a silent regression loud.
    def _dispatch_must_not_run(_repo_root, _manifest, _results, _settings, _requests):
        raise AssertionError("dispatch must not run when the pre-filter escalates")

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", _dispatch_must_not_run)

    result = app.dispatch_rework()
    # The issue was filtered out by the pre-filter (escalated, not dispatched),
    # so dispatch_rework reports "no rework candidates found" with ok=True —
    # the same as an empty backlog. The escalation is observable in the result
    # data and in state (asserted below), not in result.ok.
    assert result.ok is True
    assert result.data.get("blocked_environment_escalated") == [123]

    # The own-live-session guard refused to reap: the marker is still present.
    assert read_worktree_marker(wt_path_pre) is not None
    # No kill occurred (the guard short-circuited before the kill path).
    assert kill_calls == []

    state = load_state(paths.state_file)
    # The issue was escalated (NOT reaped and redispatched).
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "dispatch_blocked_environment"
    # No reap was recorded — the own-session guard returned False.
    assert state["issues"]["123"].get("foreign_writer_reaps", []) == []
