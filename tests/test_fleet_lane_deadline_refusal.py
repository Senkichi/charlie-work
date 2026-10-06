"""Deadline-refusal control flow for fleet repo lanes (issue #1948 rework).

``test_fleet_lane_deadline.py`` covers the yield points (where the deadline
is *checked*); this module covers the refusal *contract* (what a refusal
*is*): ``GitHub.run()`` raises ``PassDeadlineExceeded`` -- a
``BaseException`` on the ``asyncio.CancelledError`` precedent -- and the
orchestration boundaries convert it into a ``deadline_deferred`` partial
result without ever letting it masquerade as a GitHub failure:

* Lane level: ``_run_fleet_repo_lane`` returns a ``deadline_deferred``
  CommandResult for both full-loop and ``work_only`` lanes.
* Pass level: a mid-scan refusal leaves ``errors[]`` empty, emits no
  ``github_error`` event, and still marks the pass partial even when the
  trip arrives inside the final guarded step (the stale-latch regression).
* Merge level: a refused call inside ``merge_ready`` propagates out and
  never touches ``consecutive_failed_merge_attempts`` /
  ``merge_attempt_alarm`` / escalation; a spent-pass verdict that somehow
  reaches the counter block as a result is likewise inert.
* Fleet level: a partial lane lands in ``deadline_partial_repo_keys`` and
  is excluded from ``observed_repo_keys`` forwarded to the health digest.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from _deadline_fixtures import _build_app
from _fakes_github import FakeGitHubWithMissingRequired
from _fleet_dispatch_fixtures import (
    _StepClock,
    _per_repo_runtime_paths,
)
from charlie_work.config import (
    AutoMergeConfig,
    OrchestratorConfig,
)
from charlie_work.fleet_dispatch import fleet_loop
from charlie_work.fleet_lanes import _run_fleet_repo_lane
from charlie_work.instrumentation import query_events
from charlie_work.pass_deadline import (
    PassDeadlineExceeded,
    set_pass_deadline_exceeded,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import empty_state, load_state, save_state
from charlie_work.command_result import CommandResult
from charlie_work.workflow import OrchestratorApp


def _lane_lock() -> MagicMock:
    return MagicMock(name="lane_lock")


# ---------------------------------------------------------------------------
# Boundary catches: a refusal is a partial lane, never a lane failure
# ---------------------------------------------------------------------------


def test_fleet_repo_lane_converts_mid_loop_refusal_to_partial() -> None:
    """A PassDeadlineExceeded escaping app.loop() becomes a deferred result.

    ``_loop_impl`` normally converts the refusal first; this lane-level
    catch is the boundary for any escape path, and it is what keeps a
    mid-lane refusal out of ``future.result()``'s error surface in
    ``fleet_loop``.
    """
    app = MagicMock()
    app.loop.side_effect = PassDeadlineExceeded(
        "in-pass deadline exceeded; gh call refused: gh api rate_limit"
    )
    lock = _lane_lock()

    result = _run_fleet_repo_lane(
        "owner/repo1",
        app,
        OrchestratorConfig(),
        lock,
        work_only=False,
        drain=False,
        limit=3,
        merge=True,
        ensure_labels=False,
        deadline_exceeded=lambda: False,
    )

    assert result.ok is True
    assert result.data["deadline_deferred"] is True
    lock.release.assert_called_once()


def test_fleet_repo_lane_converts_work_only_refusal_to_partial() -> None:
    """work_only lanes have no _loop_impl wrapper -- the lane catch is it.

    A dispatch-time refusal must surface as ``deadline_deferred`` (the
    partial-lane shape fleet_loop sorts into deadline_partial_repo_keys),
    not propagate into ``_record_repo_lane_error`` as a hard failure.
    """
    app = MagicMock()
    app.dispatch.side_effect = PassDeadlineExceeded(
        "in-pass deadline exceeded; gh call refused: gh issue list"
    )
    lock = _lane_lock()

    result = _run_fleet_repo_lane(
        "owner/repo1",
        app,
        OrchestratorConfig(),
        lock,
        work_only=True,
        drain=False,
        limit=3,
        merge=None,
        ensure_labels=False,
        deadline_exceeded=lambda: False,
    )

    assert result.ok is True
    assert result.data["deadline_deferred"] is True
    lock.release.assert_called_once()


# ---------------------------------------------------------------------------
# _loop_body: the refusal is control flow -- never an error, never a failure
# ---------------------------------------------------------------------------


def _linked_pr(number: int, issue_number: int) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Fix #{issue_number}: work",
        "url": f"https://example.test/pull/{number}",
        "headRefName": f"agent/issue-{issue_number}-work",
        "baseRefName": "main",
        "headRefOid": f"sha-{number}",
        "mergeStateStatus": "CLEAN",
        "body": f"Closes #{issue_number}",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }


def _open_issue(number: int) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Issue {number}",
        "url": f"https://example.test/issues/{number}",
        "body": "body",
        "labels": [],
        "state": "OPEN",
    }


def test_loop_body_deadline_refusal_in_pr_scan_is_not_a_github_error(
    tmp_path: Path,
) -> None:
    """A mid-item refusal must not land in errors[] or emit github_error.

    ``self.review`` raises ``PassDeadlineExceeded`` for the first linked PR.
    The per-PR handler must break the scan -- leaving errors empty, no
    ``github_error`` event, the second linked PR untouched -- and the pass
    end must still mark the result ``deadline_deferred`` via the latched
    trip (the predicate itself never reports spent).
    """
    app, paths, fake_gh = _build_app(tmp_path / "repo")
    fake_gh.issues = [_open_issue(123), _open_issue(124)]
    fake_gh.prs = [_linked_pr(456, 123), _linked_pr(789, 124)]
    app.review = MagicMock(
        side_effect=PassDeadlineExceeded(
            "in-pass deadline exceeded; gh call refused: gh pr view 456"
        )
    )

    result = app._loop_body(
        limit=0,
        merge=False,
        now=datetime.datetime.now(datetime.UTC),
        deadline_exceeded=lambda: False,
    )

    assert result.ok is True
    assert result.data["errors"] == [], "a refusal is not a GitHub failure"
    assert query_events(paths.state_file, kind="github_error") == []
    assert result.data["deadline_deferred"] is True
    # PR 456 was iterated (tracked count incremented before the try); the
    # break kept PR 789 out of the scan entirely.
    assert result.data["open_tracked_prs"] == 1
    assert not (paths.prs / "pr-789").exists()
    events = query_events(paths.state_file, kind="loop_pass_deadline_deferred")
    assert len(events) == 1


def test_loop_body_deadline_tripped_in_final_phase_marks_partial(tmp_path: Path) -> None:
    """A deadline tripped inside the LAST guarded step still marks the pass.

    Regression for the stale-latch bug the review found: the pass-end
    marker must re-evaluate the predicate, not trust a latch last read
    before the final guarded operation (worktree reclamation here) ran.
    Otherwise the lane reports complete and fleet_loop counts the repo as
    observed despite the cut.
    """
    app, paths, fake_gh = _build_app(tmp_path / "repo")
    tripped = {"hit": False}

    def _reclaim_and_trip(*, now: Any = None) -> None:
        tripped["hit"] = True
        return None

    app._maybe_reclaim_worktrees = _reclaim_and_trip  # type: ignore[method-assign]

    result = app._loop_body(
        limit=0,
        merge=False,
        now=datetime.datetime.now(datetime.UTC),
        deadline_exceeded=lambda: tripped["hit"],
    )

    assert result.ok is True
    assert result.data["deadline_deferred"] is True
    assert "(partial: in-pass deadline reached)" in result.message
    events = query_events(paths.state_file, kind="loop_pass_deadline_deferred")
    assert len(events) == 1


def test_loop_body_deadline_defers_every_maintenance_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Trip during intake: NONE of the seven guarded maintenance steps run.

    Regression coverage for the previously-untested guards: quota probe,
    reconcile drift, superseded-CI reclaim, mechanical de-escalation,
    orphan-worker detection, unauthorized-merge tripwire, and worktree
    reclamation. Removing any one ``deadline.call`` wrapper lets that
    callee run on a spent budget, and this test fails.
    """
    app, paths, fake_gh = _build_app(tmp_path / "repo")

    guarded: dict[str, MagicMock] = {}
    for name in (
        "_maybe_probe_quota_recovery",
        "_maybe_reconcile_drift",
        "_maybe_reclaim_superseded_main_ci",
        "_maybe_deescalate_mechanical",
        "_detect_unauthorized_merges",
        "_maybe_reclaim_worktrees",
    ):
        guarded[name] = MagicMock(name=name)
        setattr(app, name, guarded[name])
    orphan_sweep = MagicMock(name="_detect_and_handle_orphaned_workers")
    monkeypatch.setattr("charlie_work.workflow._detect_and_handle_orphaned_workers", orphan_sweep)

    tripped = {"hit": False}
    real_intake = app.intake

    def _intake_and_trip() -> Any:
        tripped["hit"] = True
        return real_intake()

    app.intake = _intake_and_trip  # type: ignore[method-assign]

    result = app._loop_body(
        limit=0,
        merge=False,
        now=datetime.datetime.now(datetime.UTC),
        deadline_exceeded=lambda: tripped["hit"],
    )

    for _name, mock in guarded.items():
        mock.assert_not_called()
    orphan_sweep.assert_not_called()
    assert result.data["deadline_deferred"] is True


def test_loop_impl_converts_escaped_refusal_to_deferred(tmp_path: Path) -> None:
    """A refusal escaping _loop_body becomes a deferred pass at _loop_impl.

    ``_loop_impl`` is the pass-level boundary: whatever _loop_body misses,
    this catch turns into a ``deadline_deferred`` CommandResult so the
    caller (the lane, then fleet_loop) sees a partial pass instead of an
    exception surfacing through ``future.result()``.
    """
    app, paths, _fake_gh = _build_app(tmp_path / "repo")
    app._loop_body = MagicMock(
        side_effect=PassDeadlineExceeded(
            "in-pass deadline exceeded; gh call refused: gh issue list"
        )
    )

    result = app.loop(0, merge=False, deadline_exceeded=lambda: False)

    assert result.ok is True
    assert result.data["deadline_deferred"] is True
    assert "in-pass deadline reached" in result.message


# ---------------------------------------------------------------------------
# merge_ready: a deadline-spent pass must not move the failure streak
# ---------------------------------------------------------------------------


def test_merge_ready_refusal_propagates_without_touching_counters(
    tmp_path: Path,
) -> None:
    """A refusal inside merge_ready escapes; no counter or alarm is written."""
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed",),
            require_approved_review=True,
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    save_state(paths.state_file, empty_state())
    fake_gh = FakeGitHubWithMissingRequired()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    fake_gh.pr_view = MagicMock(
        side_effect=PassDeadlineExceeded(
            "in-pass deadline exceeded; gh call refused: gh pr view 456"
        )
    )

    with pytest.raises(PassDeadlineExceeded):
        app.merge_ready(456, merge=False)

    state = load_state(paths.state_file)
    assert state["prs"].get("456", {}).get("consecutive_failed_merge_attempts", 0) == 0
    assert state["prs"].get("456", {}).get("merge_attempt_alarm", False) is False
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert alarm_events == []


def test_merge_ready_deadline_spent_pass_preserves_failure_streak(tmp_path: Path) -> None:
    """Belt guard: a spent pass that REACHES the counter block must not move it.

    The primary contract is exception propagation (the test above); this
    pins the secondary invariant at the counter write itself -- if a
    lower-level path ever surfaces a spent-budget pass as a normal result,
    ``pass_deadline_spent`` still keeps the verdict from incrementing the
    streak or firing the alarm. Armed on the fake here precisely because
    the fake does NOT consult it, simulating that result-shaped escape.
    """
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(
            required_checks=("Tests passed",),
            require_approved_review=True,
            failed_attempt_alarm=1,
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    save_state(paths.state_file, empty_state())
    fake_gh = FakeGitHubWithMissingRequired()
    set_pass_deadline_exceeded(fake_gh, lambda: True)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    result = app.merge_ready(456, merge=False)

    assert result.data["consecutive_failed_merge_attempts"] == 0
    assert result.data["merge_attempt_alarm"] is False
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["consecutive_failed_merge_attempts"] == 0
    alarm_events = [e for e in state["events"] if e["kind"] == "merge_failed_attempt_alarm"]
    assert alarm_events == []


# ---------------------------------------------------------------------------
# fleet_loop collection: partial lanes are never "observed"
# ---------------------------------------------------------------------------


@patch("charlie_work.fleet_dispatch._build_fleet_attention_digest")
@patch("charlie_work.fleet_dispatch._fleet_notify_config")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_excludes_partial_repo_from_observed_repo_keys(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_notify_config: MagicMock,
    mock_digest: MagicMock,
    tmp_path: Path,
) -> None:
    """The digest's observed_repo_keys must exclude a deadline-partial repo.

    ``observed_repo_keys`` feeds health-baseline reconciliation -- a repo
    whose lane was cut short did NOT observe its issues, so forwarding it
    would let stale health markers be cleared as though the pass checked
    them. The lane returns a real partial result; the collector sorts it
    out of the observed set.
    """
    registry = {
        "repos": {
            "owner/repo1": {
                "repo_root": str(tmp_path / "repo1"),
                "config_path": "orchestrator.config.yaml",
            },
        }
    }
    mock_load_registry.return_value = registry
    (tmp_path / "repo1").mkdir()
    mock_load_layered_config.return_value = OrchestratorConfig()
    mock_runtime_paths.side_effect = _per_repo_runtime_paths
    mock_notify_config.return_value = MagicMock(enabled=True)

    captured: dict[str, Any] = {}

    def _capture_digest(*args: Any, **kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock(transitions=())

    mock_digest.side_effect = _capture_digest

    mock_app = MagicMock()
    mock_app.loop.return_value = CommandResult(
        True, "loop complete (partial: in-pass deadline reached)", {"deadline_deferred": True}
    )
    mock_app_class.return_value = mock_app
    mock_gh_class.return_value = MagicMock()

    clock = _StepClock(steps=[0.0], after=1.0)
    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=("owner/repo1",),
        work_only=False,
        deadline_seconds=100,
        pass_clock=clock,
    )

    assert result.ok is True
    assert result.data["deadline_partial_repo_keys"] == ["owner/repo1"]
    assert captured["observed_repo_keys"] == frozenset(), (
        "a deadline-partial repo must not be forwarded as observed -- its "
        "lane never checked every tracked issue"
    )
