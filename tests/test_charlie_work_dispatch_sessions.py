"""Dispatch session launching, stagger timing, and per-pass dispatch events.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the session-launch seam of ``dispatch()`` -- launch staggering, worker prompt
and session-manifest writes, template selection, claim-site checks,
config-cap validation, and dispatch pass events (stale-backlog, attention
digest, label-write failure). Shared fakes and helpers in
``tests/_dispatch_fixtures.py``.
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

from _dispatch_fixtures import (
    _requests,
    _seed_backdated_dispatch_event,
)
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from charlie_work.config import (
    ConfigError,
    DevinConfig,
    DispatchConfig,
    NotifyConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
    load_config,
)
from charlie_work.instrumentation import log_event, query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
)
from charlie_work.workflow import OrchestratorApp


def test_dispatch_sessions_staggers_between_launches(tmp_path: Path, monkeypatch) -> None:
    """Issue: burst dispatch trips the Devin provider message rate limit (3
    sessions launched within 6 seconds all died on "Reached overall message
    rate limit"). dispatch_sessions must sleep launch_stagger_seconds BETWEEN
    consecutive launches -- not before the first, not after the last."""
    from charlie_work import adapters
    from charlie_work.adapters import AdapterSettings, dispatch_sessions
    from charlie_work.claude_code import ClaudeWorkerRecord

    sleep_calls: list[float] = []
    monkeypatch.setattr(adapters.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=1000 + issue_number,
            started_at="2026-07-10T00:00:00Z",
            log_path=str(tmp_path / "log"),
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    settings = AdapterSettings(adapter="claude-code", launch_stagger_seconds=45)

    results = dispatch_sessions(
        repo_root,
        tmp_path / "manifest.json",
        tmp_path / "results.json",
        settings,
        _requests(3, tmp_path),
    )

    assert len(results) == 3
    assert all(r.ok for r in results)
    assert sleep_calls == [45, 45]


def test_dispatch_sessions_single_launch_no_stagger_sleep(tmp_path: Path, monkeypatch) -> None:
    """A single launch has no "between launches" gap to fill -- no sleep."""
    from charlie_work import adapters
    from charlie_work.adapters import AdapterSettings, dispatch_sessions
    from charlie_work.claude_code import ClaudeWorkerRecord

    sleep_calls: list[float] = []
    monkeypatch.setattr(adapters.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=1000 + issue_number,
            started_at="2026-07-10T00:00:00Z",
            log_path=str(tmp_path / "log"),
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    settings = AdapterSettings(adapter="claude-code", launch_stagger_seconds=45)

    results = dispatch_sessions(
        repo_root,
        tmp_path / "manifest.json",
        tmp_path / "results.json",
        settings,
        _requests(1, tmp_path),
    )

    assert len(results) == 1
    assert sleep_calls == []


def test_dispatch_sessions_zero_stagger_disables_sleep(tmp_path: Path, monkeypatch) -> None:
    """launch_stagger_seconds=0 disables the stagger entirely, even with
    multiple launches."""
    from charlie_work import adapters
    from charlie_work.adapters import AdapterSettings, dispatch_sessions
    from charlie_work.claude_code import ClaudeWorkerRecord

    sleep_calls: list[float] = []
    monkeypatch.setattr(adapters.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=1000 + issue_number,
            started_at="2026-07-10T00:00:00Z",
            log_path=str(tmp_path / "log"),
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    settings = AdapterSettings(adapter="claude-code", launch_stagger_seconds=0)

    results = dispatch_sessions(
        repo_root,
        tmp_path / "manifest.json",
        tmp_path / "results.json",
        settings,
        _requests(3, tmp_path),
    )

    assert len(results) == 3
    assert sleep_calls == []


def test_dispatch_launch_stagger_seconds_default_is_45() -> None:
    """Default stagger between worker-session launches within a pass."""
    assert DispatchConfig().launch_stagger_seconds == 45


def test_dispatch_writes_worker_prompt_and_session_manifest(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 1
    prompt_path = tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
    manifest_path = tmp_path / ".var" / "charlie-work" / "dispatches" / "session-manifest.json"
    assert prompt_path.exists()
    assert manifest_path.exists()
    assert "Closes #123" in prompt_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sessions"][0]["branch_name"] == "agent/issue-123-fix-search"
    # Manual adapter honesty: a written manifest means QUEUED — no worker has
    # been independently confirmed, so in-progress must not be applied.
    assert (123, "agent:queued") in fake_gh.labels_added
    assert (123, "agent:in-progress") not in fake_gh.labels_added


def test_dispatch_worker_template_selects_claude_code_variant(tmp_path: Path) -> None:
    config = OrchestratorConfig(dispatch=DispatchConfig(worker_template="worker_claude_code.md"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    app.gh.prs[0]["state"] = "CLOSED"
    app.dispatch(limit=1)

    prompt_path = tmp_path / ".var" / "charlie-work" / "issues" / "issue-123" / "worker-prompt.md"
    text = prompt_path.read_text(encoding="utf-8")
    assert "git switch -c agent/issue-123-fix-search" in text  # Claude Code loop
    assert "/create-branch" not in text  # not the Devin skills loop


def test_dispatch_claim_site_has_no_redundant_ci_status_check() -> None:
    """Issue #1258 (AC6): the janitor gate in ``review()`` must stay the SOLE
    source of truth for "is CI red" -- a second, independent CI-status check
    anywhere near the dispatch-claim/launch site would be the exact
    redundant-gate hazard the issue's binding comment warns against (two
    disagreeing sources of truth for the same question).

    AST-scans the three dispatch-claim-site functions/methods (not a plain
    text grep over the whole file, which would also match unrelated code
    elsewhere) for any of the tokens a CI-status read would need to use:
    ``pr_checks``/``summarize_checks``/``failed_required_checks``/
    ``required_checks``/``is_check_failure``/``checkSuite``/
    ``statusCheckRollup``. Zero hits confirmed by recon before this item's
    branch was added; this pins that finding down so a future PR that adds a
    second check here fails CI instead of silently duplicating the gate.

    issue #1283 Phase A: ``_is_review_dispatchable`` and
    ``_select_review_dispatch_candidates`` moved to
    ``charlie_work/dispatch_selection.py``; only ``dispatch_reviews`` (an
    ``OrchestratorApp`` method) stays in workflow.py. Both files are
    AST-parsed and their ``FunctionDef``/``AsyncFunctionDef`` tables unioned
    before the target check below, so the probe keeps covering the same
    three call sites across the split instead of silently losing two of
    them the moment they became import lines in workflow.py.
    """
    import ast

    workflow_path = Path(__file__).parents[1] / "src" / "charlie_work" / "workflow.py"
    dispatch_selection_path = (
        Path(__file__).parents[1] / "src" / "charlie_work" / "dispatch_selection.py"
    )

    targets = {
        "dispatch_reviews",
        "_is_review_dispatchable",
        "_select_review_dispatch_candidates",
    }
    forbidden = (
        "pr_checks",
        "summarize_checks",
        "failed_required_checks",
        "required_checks",
        "is_check_failure",
        "checkSuite",
        "statusCheckRollup",
    )

    found: dict[str, str] = {}
    for src_path in (workflow_path, dispatch_selection_path):
        source = src_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(src_path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in targets:
                segment = ast.get_source_segment(source, node)
                assert segment is not None, f"could not extract source for {node.name}"
                found[node.name] = segment

    assert found.keys() == targets, (
        f"expected to find {sorted(targets)}, found {sorted(found)} -- "
        "a rename/move at the dispatch-claim site invalidated this probe's anchors"
    )

    violations = {
        name: [token for token in forbidden if token in segment] for name, segment in found.items()
    }
    violations = {name: tokens for name, tokens in violations.items() if tokens}
    assert not violations, (
        "a CI-status token was found at the dispatch-claim site -- this is the "
        "redundant-gate hazard: the janitor gate in review() must remain the "
        f"sole source of truth for 'is CI red': {violations}"
    )


def test_dispatch_with_recovery_passes_record_to_adapter(tmp_path: Path, monkeypatch) -> None:
    """Issue #81: verify recovery record is passed through dispatch() to the adapter.

    This test MUST fail if the ordering fix in workflow.py is reverted (i.e., if
    recovery_record is forced to None by the status overwrite bug).
    """
    from charlie_work.claude_code import ClaudeWorkerRecord

    captured: dict[str, object] = {}

    def _fake_launch(issue_number, branch, prompt_text, **kwargs):
        captured["recovery"] = kwargs.get("recovery")
        return ClaudeWorkerRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "wt" / ".orchestrator-prompt.md"),
            command=("claude", "-p"),
            pid=4242,
            started_at="2026-07-02T00:00:00Z",
            log_path=str(tmp_path / "log"),
        )

    monkeypatch.setattr("charlie_work.claude_code.launch_claude_worker", _fake_launch)
    config = OrchestratorConfig(worker=WorkerRoleConfig(harness="claude-code"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Override pr_list to return empty list (no open PRs, so recovery is allowed)
    fake_gh.pr_list = lambda: []

    # Simulate a prior dispatch that crashed (status: dispatched, same branch)
    seed = load_state(paths.state_file)
    seed["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "branch_name": "agent/issue-123-fix-search",  # Same branch as would be generated
        "title": "Fix search",
        "url": "https://example.test/issues/123",
    }
    save_state(paths.state_file, seed)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    assert result.ok is True
    # The critical assertion: recovery record must be passed to the adapter
    assert captured["recovery"] is not None
    assert captured["recovery"]["status"] == "dispatched"
    assert captured["recovery"]["branch_name"] == "agent/issue-123-fix-search"
    assert (123, "agent:in-progress") in fake_gh.labels_added


def test_dispatch_config_max_open_agent_prs_validation_int(tmp_path: Path) -> None:
    """Issue #1129: max_open_agent_prs must be an int."""

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("dispatch:\n  max_open_agent_prs: true\n")
    with pytest.raises(ConfigError, match="must be an int"):
        load_config(config_file)


def test_dispatch_config_max_open_agent_prs_validation_negative(tmp_path: Path) -> None:
    """Issue #1129: max_open_agent_prs must be >= 0."""

    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text("dispatch:\n  max_open_agent_prs: -1\n")
    with pytest.raises(ConfigError, match="must be >= 0"):
        load_config(config_file)


def test_dispatch_emits_attention_digest_for_live_worker_redispatch_averted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #506: a live-worker redispatch averted outcome surfaces in the digest."""
    from charlie_work.adapters import SessionDispatchResult
    from charlie_work.state import empty_state, load_state, save_state

    digest_path = tmp_path / "digest.jsonl"
    config = OrchestratorConfig(
        dispatch=DispatchConfig(max_concurrent_sessions=1, default_limit=1),
        devin=DevinConfig(),
        notify=NotifyConfig(enabled=True, sink="file", file_path=str(digest_path)),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    state = empty_state()
    state["issues"]["123"] = {
        "number": 123,
        "status": "dispatched",
        "branch_name": "agent/issue-123-fix-search",
    }
    save_state(paths.state_file, state)

    fake_gh = FakeGitHub()
    fake_gh.prs = []
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="manual",
                ok=False,
                error="pid_alive",
                failure_kind="live_worker_redispatch_averted",
                pid=12345,
                process_start_time=1_234_567.0,
            )
            for request in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", fake_dispatch_sessions)
    # Issue #523: the live-worker slot count now verifies the recorded PID is
    # actually alive at the OS level. Stub the probe so the result PID counts.
    monkeypatch.setattr("charlie_work.workflow.is_pid_alive", lambda pid, start: True)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["live_worker_count"] == 1

    digest_lines = digest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(digest_lines) == 1
    digest = json.loads(digest_lines[0])
    transitions = digest["transitions"]
    assert len(transitions) == 1
    assert transitions[0]["issue_number"] == 123
    assert transitions[0]["health"] == "DISPATCH_AVERTED"
    assert transitions[0]["terminal_reason"] == "pid_alive"

    state = load_state(paths.state_file)
    assert state["issues"]["123"]["dispatch_alert"] == "DISPATCH_AVERTED"


def test_dispatch_pass_emits_dispatch_stale_event_when_backlog_is_stuck(
    tmp_path: Path,
) -> None:
    """Issue #946 (L3 wiring gap): ``check_dispatch_staleness`` was only ever
    unit-tested directly with hand-built dicts -- nothing verified a real
    ``dispatch()`` pass actually calls it and records the result. Drives a
    full ``app.dispatch()`` pass against a backlog that never gets dispatched
    (the default fixture's issue 123 has an open tracked PR, so
    ``selected_count`` stays 0 every pass) with a stale baseline already in
    events.db, and asserts the ``dispatch_stale`` warning event actually
    lands.
    """
    config = OrchestratorConfig(dispatch=DispatchConfig(dispatch_staleness_minutes=1))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Issue 123 has an open tracked PR (default fixture), so it is never
    # selected this pass -- the short-circuit in check_dispatch_staleness
    # (recent_issue_numbers) does not fire, and the real age comparison runs.
    assert app.gh.prs[0]["state"] == "OPEN"
    old_ts = (
        (datetime.now(UTC) - timedelta(minutes=50))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    _seed_backdated_dispatch_event(paths.state_file, old_ts, [999])

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    stale_events = query_events(paths.state_file, kind="dispatch_stale")
    assert len(stale_events) == 1, stale_events
    assert stale_events[0]["level"] == "warning"
    payload = stale_events[0]["payload"]
    assert payload["stale"] is True
    assert payload["last_dispatch_at"] == old_ts


def test_dispatch_pass_does_not_emit_dispatch_stale_when_within_threshold(
    tmp_path: Path,
) -> None:
    """Negative counterpart: a healthy pass (a recent non-empty dispatch
    already on record, well within the configured threshold) must not emit
    ``dispatch_stale``, even though nothing is dispatched on this particular
    pass either."""
    config = OrchestratorConfig(dispatch=DispatchConfig(dispatch_staleness_minutes=240))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    assert app.gh.prs[0]["state"] == "OPEN"
    recent_ts = (
        (datetime.now(UTC) - timedelta(minutes=5))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    _seed_backdated_dispatch_event(paths.state_file, recent_ts, [999])

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 0
    stale_events = query_events(paths.state_file, kind="dispatch_stale")
    assert stale_events == []


# ---------------------------------------------------------------------------
# Issue #1769 review finding #3 (L3 wiring gap): the tests above exercise
# check_dispatch_staleness directly against hand-seeded state, and
# tests/test_dispatch_cadence_state.py exercises the state.py primitives
# directly -- nothing previously drove a real `app.dispatch()` pass and
# proved `_dispatch_impl` actually calls `record_non_empty_dispatch` /
# `clear_dispatch_stale_alert` / `backfill_dispatch_baseline`. Deleting any
# of those calls from dispatch_state.py left the whole suite green before
# these tests existed.
# ---------------------------------------------------------------------------


def test_dispatch_pass_persists_baseline_marker_on_successful_dispatch(tmp_path: Path) -> None:
    """A pass that actually launches an issue must persist the durable
    baseline marker -- the single write the whole #1769 design depends on.
    Deleting the ``record_non_empty_dispatch`` call in ``dispatch_state.py``
    would leave this failing (every other test seeds the marker by hand)."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.gh.prs[0]["state"] = "CLOSED"  # issue 123 becomes dispatchable

    # The persisted marker is second-precision (`.replace(microsecond=0)` in
    # dispatch_state.py), so bracket it with a floor/ceiling a whole second
    # apart rather than comparing directly against sub-second `datetime.now()`
    # reads, which could otherwise land on either side of the rounding.
    before = datetime.now(UTC).replace(microsecond=0)
    result = app.dispatch(limit=1)
    after = datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=1)

    assert result.ok is True
    assert result.data["selected_count"] == 1
    state = load_state(paths.state_file)
    marker_ts = state["dispatch_cadence"]["last_non_empty_dispatch_at"]
    assert marker_ts is not None
    marker_time = datetime.fromisoformat(marker_ts.replace("Z", "+00:00"))
    assert before <= marker_time <= after
    assert state["dispatch_cadence"]["last_non_empty_dispatch_issue_numbers"] == [123]


def test_dispatch_pass_emits_dispatch_stale_only_once_within_reminder_window(
    tmp_path: Path,
) -> None:
    """Two consecutive real dispatch passes, both stale, inside one reminder
    window must record exactly one ``dispatch_stale`` event -- not one test
    calling the pure function twice, but two actual ``app.dispatch()``
    passes through the real state-lock/save path."""
    config = OrchestratorConfig(dispatch=DispatchConfig(dispatch_staleness_minutes=60))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    assert app.gh.prs[0]["state"] == "OPEN"
    old_ts = (
        (datetime.now(UTC) - timedelta(minutes=90))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    _seed_backdated_dispatch_event(paths.state_file, old_ts, [999])

    first = app.dispatch(limit=1)
    second = app.dispatch(limit=1)

    assert first.ok is True
    assert second.ok is True
    assert first.data["selected_count"] == 0
    assert second.data["selected_count"] == 0
    stale_events = query_events(paths.state_file, kind="dispatch_stale")
    assert len(stale_events) == 1, stale_events


def test_dispatch_stale_alert_clears_once_pass_actually_dispatches(tmp_path: Path) -> None:
    """After an armed stale alert, a pass that genuinely resolves the stall
    (this pass itself dispatches -- reason ``current_pass_dispatched``) must
    reset ``last_stale_alert_at`` so the next, unrelated stall is a fresh
    edge rather than inheriting this episode's reminder cadence."""
    config = OrchestratorConfig(dispatch=DispatchConfig(dispatch_staleness_minutes=60))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    old_ts = (
        (datetime.now(UTC) - timedelta(minutes=90))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    _seed_backdated_dispatch_event(paths.state_file, old_ts, [999])
    assert app.gh.prs[0]["state"] == "OPEN"

    first = app.dispatch(limit=1)
    assert first.data["selected_count"] == 0
    state = load_state(paths.state_file)
    assert state["dispatch_cadence"]["last_stale_alert_at"] is not None

    app.gh.prs[0]["state"] = "CLOSED"
    second = app.dispatch(limit=1)
    assert second.data["selected_count"] == 1

    state = load_state(paths.state_file)
    assert state["dispatch_cadence"]["last_stale_alert_at"] is None


def test_dispatch_pass_backfills_baseline_from_events_db_history(tmp_path: Path) -> None:
    """Wiring proof for the review BLOCKER fix: ``state.json`` starts with no
    ``dispatch_cadence`` key at all (a repo that already existed, or was
    already mid-stall, before this marker shipped), but ``events.db``
    already carries an aged non-empty ``dispatch`` row from before the
    marker existed. A real ``dispatch()`` pass must recover it and report
    staleness correctly instead of permanently reporting ``no_baseline``."""
    import sqlite3

    config = OrchestratorConfig(dispatch=DispatchConfig(dispatch_staleness_minutes=60))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    assert app.gh.prs[0]["state"] == "OPEN"

    old_ts = (
        (datetime.now(UTC) - timedelta(minutes=90))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    log_event(paths.state_file, "dispatch", {"issue_numbers": [777]})
    conn = sqlite3.connect(paths.state_file.parent / "events.db")
    conn.execute("UPDATE events SET ts = ? WHERE kind = 'dispatch'", (old_ts,))
    conn.commit()
    conn.close()
    assert "dispatch_cadence" not in load_state(paths.state_file)

    result = app.dispatch(limit=1)

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["dispatch_cadence"]["baseline_backfill_attempted"] is True
    assert state["dispatch_cadence"]["last_non_empty_dispatch_at"] == old_ts
    stale_events = query_events(paths.state_file, kind="dispatch_stale")
    assert len(stale_events) == 1, stale_events


def test_dispatch_isolates_label_write_failure(tmp_path: Path, monkeypatch) -> None:
    """Issue #135: PARTIAL_FAILURE during dispatch label transition must be recorded."""
    from charlie_work import devin_shell
    from charlie_work.labels import TransitionOutcome
    from charlie_work.worktree import WorktreeInfo

    wt_path = tmp_path / "worktrees" / "agent-issue-123-fix-search"
    wt_path.mkdir(parents=True, exist_ok=True)

    def _fake_create_worktree(repo_root, branch, **kwargs):
        return WorktreeInfo(path=wt_path, branch=branch, venv_junction=None)

    monkeypatch.setattr(devin_shell, "create_worktree", _fake_create_worktree)

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            # Return False to simulate add failure (error-as-value)
            return False

    config = OrchestratorConfig(
        devin=DevinConfig(shell_command=(sys.executable, "-c", "import sys; sys.exit(0)")),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    app.gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    # Worker launched and recorded even though labeling failed - no crash.
    assert 123 in result.data["label_errors"]
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatched"
    label_error = state["issues"]["123"]["label_error"]
    assert label_error is not None
    assert label_error["edge"] == "dispatched"
    assert label_error["outcome"] == TransitionOutcome.PARTIAL_FAILURE.value


def test_dispatch_label_error_reason_in_event_payload(tmp_path: Path) -> None:
    """Issue #453: dispatch label transition failures must carry a reason in the failures map."""
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

    class LabelFailGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            return False

    fake_gh = LabelFailGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.gh.prs[0]["state"] = "CLOSED"

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert 123 in result.data["label_errors"]
    assert 123 in result.data["failures"]
    reason = result.data["failures"][123]
    assert "label transition" in reason
    assert "dispatched" in reason
    assert "partial_failure" in reason

    state = load_state(paths.state_file)
    dispatch_events = [e for e in state["events"] if e["kind"] == "dispatch"]
    assert dispatch_events
    payload = dispatch_events[-1]["payload"]
    assert "123" in payload["failures"]
    assert payload["failures"]["123"] == reason
