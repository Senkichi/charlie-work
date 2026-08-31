"""Tests for OrchestratorApp._maybe_reclaim_worktrees, carved out of test_charlie_work.py (#1284) -- named for the worktree_reclamation domain the lane wires around (charlie_work.worktree's clean_worktrees is the stubbed dependency the lane exercises, supplying the filename), not a literal 1:1 extraction of worktree.py itself."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from _fakes_github import FakeGitHub
from charlie_work.config import ConfigError, OrchestratorConfig, load_config
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp


def _reclamation_app(
    tmp_path: Path,
    *,
    enabled: bool = True,
    interval_minutes: int = 60,
    dry_run: bool = False,
) -> OrchestratorApp:
    from charlie_work.config import WorktreeReclamationConfig

    config = OrchestratorConfig(
        worktree_reclamation=WorktreeReclamationConfig(
            enabled=enabled, interval_minutes=interval_minutes
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    return OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=dry_run)


def test_worktree_reclamation_config_defaults() -> None:
    from charlie_work.config import WorktreeReclamationConfig

    wr = WorktreeReclamationConfig()
    assert wr.enabled is True
    assert wr.interval_minutes == 60


def test_worktree_reclamation_config_absent_block_defaults_enabled(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("labels:\n  ready: automated-ready\n", encoding="utf-8")

    config = load_config(path)

    assert config.worktree_reclamation.enabled is True
    assert config.worktree_reclamation.interval_minutes == 60


def test_worktree_reclamation_config_parses(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "worktree_reclamation:\n  enabled: true\n  interval_minutes: 30\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.worktree_reclamation.enabled is True
    assert config.worktree_reclamation.interval_minutes == 30


def test_worktree_reclamation_config_rejects_non_int_interval(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("worktree_reclamation:\n  interval_minutes: soon\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="interval_minutes"):
        load_config(path)


def test_worktree_reclamation_config_rejects_zero_interval(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("worktree_reclamation:\n  interval_minutes: 0\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="interval_minutes"):
        load_config(path)


def test_is_worktree_reclamation_due_when_no_schedule() -> None:
    from charlie_work.state import is_worktree_reclamation_due

    # An absent schedule means "never run yet" -> due, so the first fleet pass
    # after startup clears the existing backlog (issue #636).
    assert is_worktree_reclamation_due({}) is True
    assert is_worktree_reclamation_due({"worktree_reclamation": {}}) is True


def test_is_worktree_reclamation_due_false_for_future_schedule() -> None:
    from datetime import UTC, datetime

    from charlie_work.state import is_worktree_reclamation_due, schedule_worktree_reclamation

    future = (
        (datetime.now(UTC) + timedelta(hours=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = schedule_worktree_reclamation({}, future)

    assert is_worktree_reclamation_due(state) is False


def test_is_worktree_reclamation_due_true_for_past_schedule() -> None:
    from datetime import UTC, datetime

    from charlie_work.state import is_worktree_reclamation_due, schedule_worktree_reclamation

    past = (
        (datetime.now(UTC) - timedelta(minutes=5))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    state = schedule_worktree_reclamation({}, past)

    assert is_worktree_reclamation_due(state) is True


def test_is_worktree_reclamation_due_treats_malformed_as_due() -> None:
    from charlie_work.state import is_worktree_reclamation_due, schedule_worktree_reclamation

    state = schedule_worktree_reclamation({}, "not-a-timestamp")

    # A corrupt value must not wedge reclamation off forever.
    assert is_worktree_reclamation_due(state) is True


def test_maybe_reclaim_worktrees_disabled_returns_none(tmp_path: Path) -> None:
    from charlie_work import workflow as workflow_module

    app = _reclamation_app(tmp_path, enabled=False)

    def _fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("disabled reclamation must never call clean_worktrees")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "clean_worktrees", _fail_if_called)
    try:
        assert app._maybe_reclaim_worktrees() is None
    finally:
        monkeypatch.undo()

    state = load_state(app.paths.state_file)
    assert state.get("worktree_reclamation", {}).get("next_run_at") is None


def test_maybe_reclaim_worktrees_not_due_does_not_sweep(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from charlie_work import workflow as workflow_module
    from charlie_work.state import save_state, schedule_worktree_reclamation

    app = _reclamation_app(tmp_path, interval_minutes=60)
    future = (
        (datetime.now(UTC) + timedelta(hours=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    save_state(app.paths.state_file, schedule_worktree_reclamation({}, future))

    def _fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("must not sweep before the scheduled interval elapses")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(workflow_module, "clean_worktrees", _fail_if_called)
    try:
        assert app._maybe_reclaim_worktrees() is None
    finally:
        monkeypatch.undo()

    # Schedule is untouched when not due.
    state = load_state(app.paths.state_file)
    assert state["worktree_reclamation"]["next_run_at"] == future


def test_maybe_reclaim_worktrees_dry_run_threads_and_removes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A --dry-run fleet pass must run the sweep in preview mode, which removes
    nothing (the preview-vs-act class tracked in #614-#619). ``dry_run`` is
    threaded from the app into ``clean_worktrees``."""
    from charlie_work import workflow as workflow_module
    from charlie_work.worktree import WorktreeCleanResult

    app = _reclamation_app(tmp_path, dry_run=True)

    calls: list[dict] = []

    def _fake_clean(*args: object, **kwargs: object) -> WorktreeCleanResult:
        calls.append({"args": args, "kwargs": kwargs})
        return WorktreeCleanResult(
            ok=True,
            message="worktree-clean (dry-run): 2 eligible, 0 skipped, 0 orphan(s)",
            data={
                "planned": [{"worktree": "a"}, {"worktree": "b"}],
                "removed": [],
                "skipped": [],
                "failed": [],
                "orphans": {"planned": [], "removed": [], "failed": []},
                "venv_ok": True,
                "venv_message": "ok",
                "attention_events": [],
            },
        )

    monkeypatch.setattr(workflow_module, "clean_worktrees", _fake_clean)

    summary = app._maybe_reclaim_worktrees()

    assert summary is not None
    assert len(calls) == 1
    # dry_run is threaded honestly into the sweep.
    assert calls[0]["kwargs"]["dry_run"] is True
    # The canonical resolved worktrees root is passed -- not a manual
    # re-derivation -- so this call site can never diverge from dispatch's
    # and `charlie worktree-clean`'s (the create/sweep split documented in
    # layout.py's module docstring, "74-uncollected-worktrees").
    assert calls[0]["args"][1] == app.layout.worktrees
    # A preview removes nothing.
    assert summary["removed"] == 0
    assert summary["planned"] == 2
    assert summary["dry_run"] is True
    # The schedule is advanced even in dry-run: clean_worktrees makes its
    # live `gh pr view` fan-out unconditionally (dry_run only gates the final
    # `git worktree remove`), so the cadence gate's cost is identical in both
    # modes and must be rate-limited in both.
    state = load_state(app.paths.state_file)
    assert state["worktree_reclamation"]["next_run_at"] is not None
    # Under dry_run the worktrees_reclaimed event is suppressed per #1324:
    # _record_event now routes through WriteGate, which produces zero
    # events.db/state.json writes under dry_run -- the exact leak this fix
    # closes. The schedule still advances (see above) because that save_state
    # is a raw call outside _record_event's gate.
    events = [e for e in state["events"] if e["kind"] == "worktrees_reclaimed"]
    assert len(events) == 0


def test_maybe_reclaim_worktrees_runs_and_emits_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live (non-dry-run) pass runs the sweep, records a
    ``worktrees_reclaimed`` event with the counts, and advances the cadence
    schedule (issue #636: a maintenance action that leaves no trace is
    indistinguishable from one that never ran -- lesson from #595/#621)."""
    from datetime import UTC, datetime

    from charlie_work import workflow as workflow_module
    from charlie_work.worktree import WorktreeCleanResult

    app = _reclamation_app(tmp_path, interval_minutes=60)

    def _fake_clean(*_args: object, **_kwargs: object) -> WorktreeCleanResult:
        return WorktreeCleanResult(
            ok=True,
            message="worktree-clean: 3 removed, 1 skipped, 0 failed, 0 orphan(s)",
            data={
                "planned": [],
                "removed": [{"issue_number": 1}, {"issue_number": 2}, {"issue_number": 3}],
                "skipped": [{"issue_number": 4}],
                "failed": [],
                "orphans": {"planned": [], "removed": [], "failed": []},
                "venv_ok": True,
                "venv_message": "ok",
                "attention_events": [],
            },
        )

    monkeypatch.setattr(workflow_module, "clean_worktrees", _fake_clean)

    # frozen_now (issue #828) injected so the schedule assertion below is
    # exact instead of racing the sweep's own duration (or a CI stall
    # between the schedule computation and this assertion). No downstream
    # real-clock dependency follows in this test, so no offset is needed.
    frozen_now = datetime.now(UTC)
    summary = app._maybe_reclaim_worktrees(now=frozen_now)

    assert summary is not None
    assert summary["dry_run"] is False
    assert summary["removed"] == 3
    assert summary["skipped_count"] == 1
    state = load_state(app.paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "worktrees_reclaimed"]
    assert len(events) == 1
    assert events[0]["payload"]["removed"] == 3
    # The schedule was advanced ~interval_minutes into the future, so the very
    # next pass does not re-fire the per-candidate gh fan-out.
    next_run_at = state["worktree_reclamation"]["next_run_at"]
    expected = (
        (frozen_now + timedelta(minutes=60))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    assert next_run_at == expected


def test_maybe_reclaim_worktrees_event_carries_skip_reasons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1012: ``clean_worktrees`` computes a distinct ``reason`` string
    per skipped worktree, but only ``skipped_count`` used to reach the durable
    ``worktrees_reclaimed`` event. The full reason strings, plus the
    out-of-scope/registered counts, must now survive into the persisted
    payload -- an operator reading events.db, not live output, is the actual
    consumer this event exists for."""
    from charlie_work import workflow as workflow_module
    from charlie_work.worktree import WorktreeCleanResult

    app = _reclamation_app(tmp_path, interval_minutes=60)

    skipped_entries = [
        {
            "worktree": "C:/wt/agent-issue-4",
            "branch": "agent/issue-4",
            "issue_number": 4,
            "pr_number": 40,
            "reason": "worktree HEAD (abc12345) is not contained in merged PR "
            "head (def67890); stray post-merge commit(s)",
        },
        {
            "worktree": "C:/wt/agent-issue-5",
            "branch": "agent/issue-5",
            "issue_number": 5,
            "pr_number": 50,
            "reason": "live worker detected: recorded PID 1234 is alive",
        },
    ]

    def _fake_clean(*_args: object, **_kwargs: object) -> WorktreeCleanResult:
        return WorktreeCleanResult(
            ok=True,
            message="worktree-clean: 0 removed, 2 skipped, 0 failed, 0 orphan(s)",
            data={
                "planned": [],
                "removed": [],
                "skipped": skipped_entries,
                "failed": [],
                "orphans": {"planned": [], "removed": [], "failed": []},
                "venv_ok": True,
                "venv_message": "ok",
                "attention_events": [],
                "worktrees_registered": 3,
                "worktrees_out_of_scope": 1,
            },
        )

    monkeypatch.setattr(workflow_module, "clean_worktrees", _fake_clean)

    summary = app._maybe_reclaim_worktrees()

    assert summary is not None
    assert summary["skipped_count"] == 2
    # The exact reason strings -- not just a count -- reach the summary.
    assert summary["skipped_examples"] == skipped_entries
    assert summary["worktrees_registered"] == 3
    assert summary["worktrees_out_of_scope"] == 1

    state = load_state(app.paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "worktrees_reclaimed"]
    assert len(events) == 1
    payload = events[0]["payload"]
    # This is the load-bearing assertion: the durable event -- not just the
    # in-memory summary -- carries the reasons. Before this fix, `payload`
    # had no `skipped_examples` key at all.
    assert payload["skipped_examples"] == skipped_entries
    assert "stray post-merge commit(s)" in payload["skipped_examples"][0]["reason"]
    assert "live worker detected" in payload["skipped_examples"][1]["reason"]
    assert payload["worktrees_out_of_scope"] == 1

    # The issue's actual complaint is that reconstructing "why" from events.db
    # was impossible -- state.json's 200-entry array is a convenience cache,
    # not the durable store an operator queries after the fact. Round-trip
    # through the real SQLite dual-write (append_event -> events.db) via the
    # same query_events() helper an operator would use, not just state.json.
    db_events = query_events(app.paths.state_file, kind="worktrees_reclaimed")
    assert len(db_events) == 1
    db_payload = db_events[0]["payload"]
    assert db_payload["skipped_examples"] == skipped_entries
    assert "stray post-merge commit(s)" in db_payload["skipped_examples"][0]["reason"]
    assert "live worker detected" in db_payload["skipped_examples"][1]["reason"]
    assert db_payload["worktrees_out_of_scope"] == 1
    assert db_payload["worktrees_registered"] == 3


def test_maybe_reclaim_worktrees_skip_examples_truncated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A standing backlog of skipped worktrees must not re-emit every reason
    string into events.db on every cadence interval -- the same idiom as
    ``_MAX_DEFERRED_CONCURRENCY_EXAMPLES`` (issue #1005). ``skipped`` stays
    the exact count; ``skipped_examples`` is capped."""
    from charlie_work import workflow as workflow_module
    from charlie_work.workflow import _MAX_SKIPPED_WORKTREE_EXAMPLES
    from charlie_work.worktree import WorktreeCleanResult

    app = _reclamation_app(tmp_path, interval_minutes=60)

    many_skipped = [
        {
            "worktree": f"C:/wt/agent-issue-{i}",
            "branch": f"agent/issue-{i}",
            "issue_number": i,
            "pr_number": i * 10,
            "reason": "PR not merged",
        }
        for i in range(_MAX_SKIPPED_WORKTREE_EXAMPLES + 7)
    ]

    def _fake_clean(*_args: object, **_kwargs: object) -> WorktreeCleanResult:
        return WorktreeCleanResult(
            ok=True,
            message="worktree-clean: 0 removed, many skipped, 0 failed, 0 orphan(s)",
            data={
                "planned": [],
                "removed": [],
                "skipped": many_skipped,
                "failed": [],
                "orphans": {"planned": [], "removed": [], "failed": []},
                "venv_ok": True,
                "venv_message": "ok",
                "attention_events": [],
                "worktrees_registered": len(many_skipped) + 1,
                "worktrees_out_of_scope": 1,
            },
        )

    monkeypatch.setattr(workflow_module, "clean_worktrees", _fake_clean)

    summary = app._maybe_reclaim_worktrees()

    assert summary is not None
    # The exact count is never truncated.
    assert summary["skipped_count"] == len(many_skipped)
    # The examples list IS truncated.
    assert len(summary["skipped_examples"]) == _MAX_SKIPPED_WORKTREE_EXAMPLES
    assert summary["skipped_examples"] == many_skipped[:_MAX_SKIPPED_WORKTREE_EXAMPLES]

    state = load_state(app.paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "worktrees_reclaimed"]
    assert len(events[0]["payload"]["skipped_examples"]) == _MAX_SKIPPED_WORKTREE_EXAMPLES
    assert events[0]["payload"]["skipped_count"] == len(many_skipped)


def test_maybe_reclaim_worktrees_advances_schedule_before_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The schedule is advanced BEFORE the sweep runs, so a sweep that errors
    or takes longer than one poll interval cannot double-fire on the next
    pass."""
    from charlie_work import workflow as workflow_module
    from charlie_work.worktree import WorktreeCleanResult

    app = _reclamation_app(tmp_path, interval_minutes=60)

    def _exploding_clean(*_args: object, **_kwargs: object) -> WorktreeCleanResult:
        # Simulate a sweep that fails: the schedule must already have been
        # advanced before this point.
        state = load_state(app.paths.state_file)
        assert state["worktree_reclamation"]["next_run_at"] is not None
        return WorktreeCleanResult(
            ok=False,
            message="worktree-clean: 0 removed, 0 skipped, 1 failed",
            data={
                "planned": [],
                "removed": [],
                "skipped": [],
                "failed": [{"worktree": "a"}],
                "orphans": {"planned": [], "removed": [], "failed": []},
                "venv_ok": True,
                "venv_message": "ok",
                "attention_events": [],
            },
        )

    monkeypatch.setattr(workflow_module, "clean_worktrees", _exploding_clean)

    summary = app._maybe_reclaim_worktrees()

    assert summary is not None
    assert summary["ok"] is False
    assert summary["failed"] == 1
    # A failed sweep still records an event (observability) and keeps the
    # advanced schedule (no immediate retry storm).
    state = load_state(app.paths.state_file)
    assert state["worktree_reclamation"]["next_run_at"] is not None
    events = [e for e in state["events"] if e["kind"] == "worktrees_reclaimed"]
    assert len(events) == 1
    assert events[0]["payload"]["failed"] == 1
