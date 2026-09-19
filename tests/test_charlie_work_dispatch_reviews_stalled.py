"""Review-dispatch stalled-claim reap, redispatch, and empty-diff handling.

Split out of ``tests/test_charlie_work.py`` (issue #1548, Track-1 wave 2/8):
the recovery half of the ``test_dispatch_reviews_*`` seam -- stalled-claim
reap and redispatch, recorded-verdict throttle suppression, and empty-diff
skip/deferral paths. Shared fakes and helpers in
``tests/_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
import os
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from typing import Any

import pytest

from _dispatch_fixtures import _fail_if_launched
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _helpers import _init_git_repo
from _review_fixtures import (
    _dispatch_reviews_app,
    _fake_claude_worker_record,
    _write_review_packet,
)
from _rework_dispatch_fixtures import _wg
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.instrumentation import query_events
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import _detect_and_handle_stalled_reviews


def test_dispatch_reviews_reaps_stalled_claim_when_disabled(monkeypatch, tmp_path: Path) -> None:
    """Issue #868: the reaper sweeps must run even when review_dispatch is off.

    Before this fix, the four reaper sweeps sat BELOW the ``enabled`` early
    return inside ``dispatch_reviews()`` and were unreachable whenever
    dispatch was disabled -- a dead reviewer's claim (and any other stale
    claim) was never freed while the flag was off, so it stayed stuck even
    after the flag was re-enabled (the reap is what makes it re-dispatchable).
    This seeds the exact same dead-reviewer claim as
    ``test_dispatch_reviews_redispatches_stalled_reviews`` above, but with
    dispatch disabled: the claim must still be reaped to
    ``review_dispatch_failed`` even though nothing gets (re)launched, and the
    disabled no-op result must carry an explicit ``disabled`` marker (issue
    #868 part 3) instead of being a bare ``ok=True`` indistinguishable from
    real progress.
    """
    from datetime import timedelta

    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs, enabled=False)
    _write_review_packet(tmp_path, 100, "sha-100")

    # Seed a stale reviewer sidecar + state claim, identical to the
    # redispatch test above.
    reviews_dir = app._layout.reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    old_started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar = {
        "issue_number": 100,
        "branch": "agent/issue-10-fix",
        "worktree_path": str(tmp_path / "worktrees" / "issue-100"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 99999,
        "started_at": old_started,
        "log_path": str(tmp_path / "log.log"),
        "error": None,
        "process_start_time": 1.0,
    }
    (reviews_dir / "issue-100.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")
    # Give the reviewer a readable log with no throttle marker so it
    # classifies as NOT_THROTTLED (the counted-failure path this test
    # exercises). Without this the log read would fail (the sidecar's
    # log_path names a file that was never created) and classify as
    # UNDETERMINED (issue #1069), whose first-few-deaths handling rolls the
    # claim back to re-dispatchable instead of reaping it to
    # review_dispatch_failed.
    Path(sidecar["log_path"]).write_text("ordinary crash output\n", encoding="utf-8")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"] = {
            "number": 100,
            "issue_number": 10,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old_started,
            "reviewer_pid": 99999,
            "reviewer_process_start_time": 1.0,
        }
        save_state(app.paths.state_file, state)

    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(100, "agent/issue-10-fix")

    def fake_is_worker_alive(record: ClaudeWorkerRecord, **_kwargs: Any) -> bool:
        return False

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", fake_is_worker_alive)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data.get("disabled") is True
    assert result.data["launched_count"] == 0
    assert launched == []  # dispatch stays off -- no relaunch, only the reap
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_failed"


def test_dispatch_reviews_redispatches_stalled_reviews(monkeypatch, tmp_path: Path) -> None:
    """Issue #370: a dead/stale reviewer claim is reaped and the PR re-dispatched."""
    from datetime import timedelta

    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")

    # Seed a stale reviewer sidecar + state claim.
    reviews_dir = app._layout.reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    old_started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    old_dispatched = old_started
    sidecar = {
        "issue_number": 100,
        "branch": "agent/issue-10-fix",
        "worktree_path": str(tmp_path / "worktrees" / "issue-100"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 99999,
        "started_at": old_started,
        "log_path": str(tmp_path / "log.log"),
        "error": None,
        "process_start_time": 1.0,
    }
    (reviews_dir / "issue-100.claude.json").write_text(json.dumps(sidecar), encoding="utf-8")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["100"] = {
            "number": 100,
            "issue_number": 10,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old_dispatched,
            "reviewer_pid": 99999,
            "reviewer_process_start_time": 1.0,
        }
        save_state(app.paths.state_file, state)

    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(100, "agent/issue-10-fix")

    def fake_is_worker_alive(record: ClaudeWorkerRecord, **_kwargs: Any) -> bool:
        return False

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", fake_is_worker_alive)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["launched_count"] == 1
    assert launched == [100]
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert state["prs"]["100"]["reviewer_pid"] == 12345


def test_dispatch_reviews_recorded_verdict_suppresses_later_stalled_throttle(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #662: a recorded verdict from a dead reviewer clears reviewer_quota
    and stamps ``last_probe_cleared_at``. A later stale throttled reviewer that
    died before that marker must not re-poison ``reviewer_quota``.
    """
    from charlie_work.state import is_reviewer_quota_exhausted

    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    future_throttle = (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    past_probe = (datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["reviewer_quota"] = {
            "throttled_until": future_throttle,
            "probe_after": past_probe,
        }
        state["prs"]["100"] = {
            **state["prs"].get("100", {}),
            "number": 100,
            "issue_number": 10,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": (datetime.now(UTC) - timedelta(minutes=10))
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "reviewer_pid": 0,
            "reviewer_process_start_time": None,
        }
        save_state(app.paths.state_file, state)

    # Issue #663 landed a layout-module refactor (on main, after this branch
    # diverged) that changed ReviewDispatchConfig.reviews_dir's default to ""
    # -- an empty-string sentinel meaning "derive from layout.reviews_dir_default"
    # (see paths.resolved_layout). `tmp_path / app.config.review_dispatch.reviews_dir`
    # now silently resolves to tmp_path itself, not the real reviews directory
    # dispatch_reviews() reads from (self._layout.reviews_dir) -- use the
    # resolved layout path, matching the sibling
    # test_dispatch_reviews_probe_success_clears_reviewer_quota above.
    reviews_dir = app._layout.reviews_dir
    reviews_dir.mkdir(parents=True, exist_ok=True)
    log_path = reviews_dir / "issue-100-review.claude.log"
    log_path.write_text(
        'Review complete.\n```json\n{"decision": "approved", "summary": "LGTM"}\n```',
        encoding="utf-8",
    )
    sidecar_path = reviews_dir / "issue-100.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 100,
                "branch": "agent/issue-10-fix",
                "worktree_path": str(tmp_path / "wt"),
                "prompt_path": str(tmp_path / "prompt"),
                "command": ["claude"],
                "pid": 0,
                "started_at": (datetime.now(UTC) - timedelta(minutes=10))
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "log_path": str(log_path),
                "adapter_kind": "claude-code",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("charlie_work.claude_code.is_worker_alive", lambda *_: False)
    monkeypatch.setattr(
        "charlie_work.workflow.launch_claude_worker",
        lambda *args, **kwargs: _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        ),
    )

    app.dispatch_reviews()
    state = load_state(app.paths.state_file)
    assert not is_reviewer_quota_exhausted(state)
    marker = state["reviewer_quota"]["last_probe_cleared_at"]
    assert marker

    # Now create a second stale throttled reviewer (PR 200) whose death
    # predates the marker. The reap sweep should suppress backoff.
    marker_dt = datetime.fromisoformat(marker.replace("Z", "+00:00"))
    death_dt = marker_dt - timedelta(seconds=1)
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)

    log_path_200 = reviews_dir / "issue-200-review.claude.log"
    log_path_200.write_text(
        "You've hit your session limit · resets 4:40pm (America/Los_Angeles)\n",
        encoding="utf-8",
    )
    os.utime(log_path_200, (death_dt.timestamp(), death_dt.timestamp()))

    old_started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    (reviews_dir / "issue-200.claude.json").write_text(
        json.dumps(
            {
                "issue_number": 200,
                "branch": "agent/issue-20-fix",
                "worktree_path": str(tmp_path / "wt200"),
                "prompt_path": str(tmp_path / "prompt200"),
                "command": ["claude", "-p"],
                "pid": 999999999,
                "started_at": old_started,
                "log_path": str(log_path_200),
                "error": None,
                "process_start_time": 1.0,
            }
        ),
        encoding="utf-8",
    )

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["200"] = {
            "number": 200,
            "issue_number": 20,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old_started,
            "reviewer_pid": 999999999,
            "reviewer_process_start_time": 1.0,
        }
        save_state(app.paths.state_file, state)

    _detect_and_handle_stalled_reviews(
        reviews_dir,
        app.paths.state_file,
        app.config,
        repo_root,
        write_gate=_wg(app.paths.state_file),
    )

    state = load_state(app.paths.state_file)
    assert not is_reviewer_quota_exhausted(state)
    assert state["prs"]["200"].get("review_dispatch_status") is None
    assert any(
        event.get("kind") == "review_dispatch_stalled"
        and event.get("payload", {}).get("pr_number") == 200
        and event.get("payload", {}).get("reason") == "provider_throttled"
        and event.get("payload", {}).get("backoff_suppressed") is True
        for event in state.get("events", [])
    )


def test_dispatch_reviews_skips_empty_diff_pr(monkeypatch, tmp_path: Path) -> None:
    """Issue #1251: a PR whose diff.patch is empty (zero-file diff vs base)
    must not burn a paid reviewer session. The pre-flight gate skips dispatch,
    emits review_dispatch_skipped_empty_diff, and does NOT increment
    review_dispatch_attempt_count (an empty diff is not a review attempt)."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    # Write an EMPTY diff.patch — the signal for a zero-file PR.
    pr_dir = app.paths.prs / "pr-100"
    (pr_dir / "diff.patch").write_text("", encoding="utf-8")

    launched: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append((args, kwargs))
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    # No reviewer must have been launched.
    assert len(launched) == 0
    assert result.data["launched_count"] == 0
    # The PR must appear in the skipped_empty_diff payload.
    assert 100 in result.data["skipped_empty_diff"]
    # review_dispatch_skipped_empty_diff event must have been emitted.
    skip_events = query_events(app.paths.state_file, kind="review_dispatch_skipped_empty_diff")
    assert len(skip_events) == 1
    assert skip_events[0]["pr_number"] == 100
    # review_dispatch_attempt_count must NOT have been incremented.
    state = load_state(app.paths.state_file)
    pr_state = state["prs"].get("100", {})
    assert int(pr_state.get("review_dispatch_attempt_count", 0)) == 0
    # No dispatch claim must have been written.
    assert pr_state.get("review_dispatch_status") != "review_dispatch_pending"


def test_dispatch_reviews_empty_diff_mixed_with_nonempty(monkeypatch, tmp_path: Path) -> None:
    """Issue #1251: when both an empty-diff PR and a non-empty-diff PR are
    queued, only the non-empty PR is dispatched; the empty one is skipped
    without affecting the other's dispatch."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
        {
            "number": 200,
            "title": "Fix #20",
            "url": "https://example.test/pull/200",
            "headRefName": "agent/issue-20-fix",
            "baseRefName": "main",
            "headRefOid": "sha-200",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #20",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    _write_review_packet(tmp_path, 200, "sha-200")
    # PR 100: empty diff. PR 200: non-empty diff.
    (app.paths.prs / "pr-100" / "diff.patch").write_text("", encoding="utf-8")
    (app.paths.prs / "pr-200" / "diff.patch").write_text(
        "diff --git a/bar.py b/bar.py\n+pass\n", encoding="utf-8"
    )

    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    # Only PR 200 must have been launched.
    assert launched == [200]
    assert result.data["launched_count"] == 1
    # PR 100 must be in the skipped list.
    assert 100 in result.data["skipped_empty_diff"]
    assert 200 not in result.data["skipped_empty_diff"]
    # PR 100's attempt count must not have been incremented.
    state = load_state(app.paths.state_file)
    assert int(state["prs"].get("100", {}).get("review_dispatch_attempt_count", 0)) == 0
    # PR 200's attempt count must have been incremented (claimed).
    assert int(state["prs"]["200"].get("review_dispatch_attempt_count", 0)) == 1


def test_dispatch_reviews_empty_diff_skip_is_registered_and_never_launches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1258 (AC5): the #1251 empty-diff pre-flight guard already
    landed on this clone's main (commit 7de2fa9, PR #1278) before this item
    started -- per the task's "integrate with it instead of duplicating"
    instruction, this item does not reimplement it. This test pins the two
    concrete AC5 guarantees against the ALREADY-MERGED gate as an explicit
    W1 regression (not just #1251's own suite): zero reviewer-launch calls
    for a zero-file PR, and its dedicated event kind is genuinely registered
    in instrumentation.py's exhaustive registry (not just a bare string).

    Naming note: the event kind was originally ``review_skipped_empty_diff``
    (#1251, PR #1278) and has been renamed to ``review_dispatch_skipped_empty_diff``
    for issue #1258 (AC5) so it shares the ``review_dispatch_*`` family AC4
    pins for the new CI-red kind. The rename touches only the
    ``_LEVEL_BY_KIND`` registry key, its single emitter call site, and this
    suite's assertions -- the emission site, payload shape, and warning
    level are unchanged, so this is not a behavior change.

    ``_dispatch_reviews_app`` defaults ``review_dispatch.enabled=True``, so
    ``launched == []`` here is a real launch-avoidance assertion, not a
    disabled-dispatch early return. The positive control proving this fixture
    would otherwise launch is the pre-existing sibling
    ``test_dispatch_reviews_proceeds_with_nonempty_diff`` (same app/packet
    helpers, non-empty ``diff.patch``), which gets exactly one launch.
    """
    from charlie_work.instrumentation import _LEVEL_BY_KIND

    assert "review_dispatch_skipped_empty_diff" in _LEVEL_BY_KIND

    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    (app.paths.prs / "pr-100" / "diff.patch").write_text("", encoding="utf-8")

    launched = _fail_if_launched(monkeypatch)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert launched == []
    assert result.data["launched_count"] == 0
    skip_events = query_events(app.paths.state_file, kind="review_dispatch_skipped_empty_diff")
    assert len(skip_events) == 1
    assert skip_events[0]["pr_number"] == 100


def test_dispatch_reviews_dry_run_empty_diff_preview_is_read_only(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #1251: the dry-run preview branch of dispatch_reviews mirrors the
    real path's empty-diff pre-flight gate (added at the dry-run branch's
    dry_selected/dry_skipped_empty_diff loop) but must stay strictly read-only:
    no review_dispatch_skipped_empty_diff event emitted, no state.json mutation, no
    attempt_count change. This is the only test covering that loop -- no other
    dispatch_reviews dry-run test exists in the file."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
        {
            "number": 200,
            "title": "Fix #20",
            "url": "https://example.test/pull/200",
            "headRefName": "agent/issue-20-fix",
            "baseRefName": "main",
            "headRefOid": "sha-200",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #20",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs, dry_run=True)
    _write_review_packet(tmp_path, 100, "sha-100")
    _write_review_packet(tmp_path, 200, "sha-200")
    # PR 100: empty diff. PR 200: non-empty diff.
    (app.paths.prs / "pr-100" / "diff.patch").write_text("", encoding="utf-8")
    (app.paths.prs / "pr-200" / "diff.patch").write_text(
        "diff --git a/bar.py b/bar.py\n+pass\n", encoding="utf-8"
    )

    # A dry-run pass must never launch a worker. Guard against the launch
    # helper being invoked at all -- if it is, the dry-run gate failed.
    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        raise AssertionError("dry-run dispatch_reviews must not launch a worker")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    # Snapshot state.json's raw bytes so a no-op dry-run can be proven by
    # byte-equality, not just by re-reading parsed fields.
    state_path = app.paths.state_file
    state_before = state_path.read_bytes()

    result = app.dispatch_reviews()

    assert result.ok is True
    # The empty-diff PR is excluded from the would-dispatch count; the
    # non-empty PR is the only one that would be dispatched.
    assert result.data["selected_count"] == 1
    assert result.data["attempted_count"] == 1
    assert result.data["launched_count"] == 0
    # deferred_count = all_candidates - dry_selected = 2 - 1 = 1 (the empty
    # PR is counted as deferred in the dry-run preview, mirroring how the
    # real path's deferred_count = candidates - dispatchable treats a
    # pre-flight-skipped PR as not-selected).
    assert result.data["deferred_count"] == 1
    # The empty-diff PR is reported in skipped_empty_diff; the non-empty is not.
    assert result.data["skipped_empty_diff"] == [100]
    # Read-only contract: no review_dispatch_skipped_empty_diff event emitted.
    skip_events = query_events(app.paths.state_file, kind="review_dispatch_skipped_empty_diff")
    assert skip_events == []
    # Read-only contract: state.json bytes unchanged.
    assert state_path.read_bytes() == state_before
    # Read-only contract: no attempt_count mutation for either PR.
    state = load_state(state_path)
    assert "100" not in state["prs"]
    assert "200" not in state["prs"]


def test_dispatch_reviews_proceeds_with_nonempty_diff(monkeypatch, tmp_path: Path) -> None:
    """Issue #1251: a PR with a non-empty diff.patch must proceed through
    normal dispatch unchanged — the empty-diff gate only stops zero-file PRs."""
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        },
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")
    # Write a NON-EMPTY diff.patch.
    pr_dir = app.paths.prs / "pr-100"
    (pr_dir / "diff.patch").write_text(
        "diff --git a/foo.py b/foo.py\n+print('hello')\n", encoding="utf-8"
    )

    launched: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append((args, kwargs))
        return _fake_claude_worker_record(
            kwargs.get("issue_number") or args[0],
            kwargs.get("branch") or args[1],
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()

    assert result.ok is True
    # The reviewer must have been launched.
    assert len(launched) == 1
    assert result.data["launched_count"] == 1
    # No empty-diff skip.
    assert result.data["skipped_empty_diff"] == []
    # No review_dispatch_skipped_empty_diff event.
    skip_events = query_events(app.paths.state_file, kind="review_dispatch_skipped_empty_diff")
    assert len(skip_events) == 0
    # Dispatch must have proceeded — claim upgraded to dispatched.
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
