"""Regression tests for issue #594: a fleet ``loop()`` pass's stalled-review
sweep silently clobbers a concurrent ``charlie unescalate``.

``_detect_and_handle_stalled_reviews`` historically loaded a state snapshot at
pass entry, mutated only the few PR entries it actually reaped, then
``save_state``-d the *whole* snapshot at the end. A concurrent writer that
committed between the sweep's load and save (e.g. ``charlie unescalate --pr N``
resetting an escalated PR back to the passive open state) had every field it
changed restored to the pre-commit value, with no event explaining the
reversal -- a classic read-modify-write lost update across the sweep's
~tens-of-seconds window.

The fix is merge-on-write: at the save boundary the sweep re-loads fresh state
under the lock and applies only the entries/fields it actually computed, the
same pattern ``unescalate`` itself already uses to defend against a concurrent
reconcile pass.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import charlie_work.state as cw_state
import charlie_work.workflow as cw_workflow
from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig, RuntimeConfig
from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state, state_lock
from charlie_work.workflow import _detect_and_handle_stalled_reviews

from _helpers import _init_git_repo

_THROTTLE_LINE = "You've hit your session limit · resets 4:40pm (America/Los_Angeles)\n"


def _write_throttled_reviewer(reviews_dir: Path, pr_number: int, tmp_path: Path) -> Path:
    """Fabricate a dead reviewer sidecar whose log shows a throttle signature."""
    log_path = reviews_dir / f"issue-{pr_number}-review.claude.log"
    log_path.write_text(_THROTTLE_LINE, encoding="utf-8")
    started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sidecar = {
        "issue_number": pr_number,
        "branch": f"agent/issue-{pr_number}-fix",
        "worktree_path": str(tmp_path / "worktrees" / f"issue-{pr_number}"),
        "prompt_path": str(tmp_path / "prompt.md"),
        "command": ["claude", "-p"],
        "pid": 999999999,  # not a real live pid
        "started_at": started,
        "log_path": str(log_path),
        "error": None,
        "process_start_time": 1.0,
    }
    sidecar_path = reviews_dir / f"issue-{pr_number}.claude.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return sidecar_path


def _seed(tmp_path: Path) -> tuple[Path, Path, OrchestratorConfig, Path]:
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    with state_lock(state_file):
        state = load_state(state_file)
        # PR 100: a dispatched reviewer claim that will be reaped by the sweep.
        state["prs"]["100"] = {
            "number": 100,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": started,
            "reviewer_pid": 999999999,
            "reviewer_process_start_time": 1.0,
        }
        # PR 200: an escalated PR the sweep does NOT touch -- this is the
        # `charlie unescalate --pr 200` target. It carries the exact
        # day-old-failure fingerprint from the live incident.
        state["prs"]["200"] = {
            "number": 200,
            "status": "escalated",
            "review_dispatch_attempt_count": 3,
            "review_dispatch_status": "review_dispatch_failed",
            "review_dispatch_failed_at": "2026-07-24T14:21:39Z",
        }
        save_state(state_file, state)
    return repo_root, reviews_dir, config, state_file


def test_stalled_sweep_does_not_clobber_concurrent_unescalate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #594: the sweep must merge its changes onto fresh state at save
    time rather than restoring its stale entry-pass snapshot wholesale.

    Simulates ``charlie unescalate --pr 200`` committing to disk between the
    sweep's initial ``load_state_locked`` and its final ``save_state`` by
    injecting a fresh on-disk write the first time the sweep iterates workers
    (after its snapshot load, before its save). Without merge-on-write the
    sweep's stale ``prs["200"]`` (escalated, attempt_count=3, day-old
    failed_at) overwrites the unescalate and the recovery is silently lost.
    """
    repo_root, reviews_dir, config, state_file = _seed(tmp_path)
    _write_throttled_reviewer(reviews_dir, 100, tmp_path)

    # Inject the concurrent unescalate commit the first time the sweep calls
    # iter_workers (after its snapshot load, before its save). This is the
    # exact window the live incident exploited.
    real_iter_workers = cw_workflow.iter_workers
    injected = {"done": False}

    def _injecting_iter_workers(reviews_dir_arg: Path):
        if not injected["done"]:
            injected["done"] = True
            with state_lock(state_file):
                fresh = load_state(state_file)
                fresh_pr = dict(fresh["prs"]["200"])
                fresh_pr["status"] = PASSIVE_OPEN_STATUS
                fresh_pr["review_dispatch_attempt_count"] = 0
                fresh_pr.pop("review_dispatch_failed_at", None)
                fresh["prs"]["200"] = fresh_pr
                save_state(state_file, fresh)
        return real_iter_workers(reviews_dir_arg)

    monkeypatch.setattr(cw_workflow, "iter_workers", _injecting_iter_workers)

    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    state = load_state(state_file)

    # The sweep's REAL work survives the merge: PR 100's dead claim was rolled
    # back and the global quota backoff was applied.
    assert state["prs"]["100"].get("review_dispatch_status") is None
    assert state.get("reviewer_quota", {}).get("consecutive_probe_failures") == 1

    # The untouched PR 200 retains the concurrent unescalate -- NOT the stale
    # snapshot's escalated value. This is the bug: every field unescalate
    # changed was being restored to its pre-commit value.
    pr200 = state["prs"]["200"]
    assert pr200.get("status") == PASSIVE_OPEN_STATUS, (
        f"concurrent unescalate clobbered: pr.status={pr200.get('status')!r} "
        f"(expected {PASSIVE_OPEN_STATUS!r})"
    )
    assert pr200.get("review_dispatch_attempt_count") == 0, (
        f"concurrent unescalate clobbered: review_dispatch_attempt_count="
        f"{pr200.get('review_dispatch_attempt_count')!r} (expected 0)"
    )
    assert "review_dispatch_failed_at" not in pr200, (
        f"concurrent unescalate clobbered: stale review_dispatch_failed_at="
        f"{pr200.get('review_dispatch_failed_at')!r} restored"
    )


def test_stalled_sweep_events_survive_merge_when_ring_is_at_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #594 follow-up: the merge-on-write save must not silently drop the
    sweep's own events when the bounded ``state.json`` event ring is already at
    its cap.

    ``append_event`` always rebuilds the events list via ``list(old) + [new]``
    and then truncates from the front once the list exceeds ``max_size`` --
    never mutating existing entries in place. A merge implementation that
    identifies "this sweep's new events" via a length-based slice
    (``current[len(snapshot):]``) is wrong at cap: eviction keeps the list the
    same length as the snapshot, so the slice comes back empty and the
    sweep's own events vanish from the merged ring even though the sweep did
    real, correctly-applied work (this test's PR 100 claim is still reaped).
    The fix must recover new events by identity against the snapshot, not by
    length, so it stays correct regardless of eviction.

    ``append_event`` call sites inside the sweep that omit an explicit
    ``max_size`` (the ones firing here) fall back to the module-level
    ``charlie_work.state.EVENT_RING_SIZE``, which ``OrchestratorApp.__init__``
    syncs from ``config.runtime.event_ring_size`` at startup (issue #525).
    This test calls the sweep directly without constructing an
    ``OrchestratorApp``, so it must replicate that sync itself via
    monkeypatch -- otherwise the inline appends use the real default (2000)
    and eviction never triggers, making the test pass vacuously regardless of
    whether the merge logic is correct.
    """
    repo_root, reviews_dir, config, state_file = _seed(tmp_path)
    _write_throttled_reviewer(reviews_dir, 100, tmp_path)

    # Force a small ring cap and pre-fill it to exactly that cap so any
    # further append necessarily evicts the oldest entry -- the steady-state
    # condition in production, where the default cap (2000) is reached
    # quickly and stays full.
    ring_cap = 3
    config = OrchestratorConfig(
        review_dispatch=config.review_dispatch,
        runtime=RuntimeConfig(event_ring_size=ring_cap),
    )
    monkeypatch.setattr(cw_state, "EVENT_RING_SIZE", ring_cap)
    with state_lock(state_file):
        state = load_state(state_file)
        state["events"] = [
            {"at": f"2026-01-01T00:00:0{i}Z", "kind": "pre_existing", "payload": {}}
            for i in range(ring_cap)
        ]
        save_state(state_file, state)

    _detect_and_handle_stalled_reviews(reviews_dir, state_file, config, repo_root)

    state = load_state(state_file)

    # The sweep's real work still happened...
    assert state["prs"]["100"].get("review_dispatch_status") is None
    # ...and its own event must be present in the merged ring, not silently
    # dropped by an eviction-blind length slice.
    kinds = [e.get("kind") for e in state.get("events", [])]
    assert "review_dispatch_stalled" in kinds, (
        f"sweep's own event missing from merged ring at cap: kinds={kinds!r}"
    )
    assert len(state.get("events", [])) <= ring_cap


def test_orphaned_reap_sweep_does_not_clobber_concurrent_unescalate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #594, sibling site: ``_reap_orphaned_review_checkouts`` has the
    exact same load-snapshot / do-slow-work / bare-``save_state`` shape as
    ``_detect_and_handle_stalled_reviews`` -- except its "slow work" is a
    ``gh.pr_view()`` network call per candidate PR (potentially many), an
    unlocked window that is if anything wider than the claim sweep's
    filesystem checks. Both sweeps run back-to-back in the same
    ``dispatch_reviews`` pass. Without merge-on-write here too, a
    ``charlie unescalate --pr 200`` landing between this sweep's snapshot
    load and its save is silently reverted, identically to the original
    #594 incident.
    """
    from _fakes_github import FakeGitHub

    repo_root, reviews_dir, config, state_file = _seed(tmp_path)

    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 100,
            "title": "Fix #1",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-1-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "body": "Closes #1",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        }
    ]

    monkeypatch.setattr("charlie_work.workflow.remove_review_checkout", lambda *a, **k: True)

    # Inject the concurrent unescalate commit the first time the sweep calls
    # gh.pr_view (after its snapshot load, before its save) -- the exact
    # window the live incident exploited, transplanted onto this sweep's own
    # slow step. PR 100 sorts before PR 200 among the candidates, so the
    # first pr_view call is for PR 100.
    real_pr_view = fake_gh.pr_view
    injected = {"done": False}

    def _injecting_pr_view(number: int):
        if not injected["done"]:
            injected["done"] = True
            with state_lock(state_file):
                fresh = load_state(state_file)
                fresh_pr = dict(fresh["prs"]["200"])
                fresh_pr["status"] = PASSIVE_OPEN_STATUS
                fresh_pr["review_dispatch_attempt_count"] = 0
                fresh_pr.pop("review_dispatch_failed_at", None)
                fresh["prs"]["200"] = fresh_pr
                save_state(state_file, fresh)
        return real_pr_view(number)

    monkeypatch.setattr(fake_gh, "pr_view", _injecting_pr_view)

    from charlie_work.workflow import _reap_orphaned_review_checkouts

    reaped = _reap_orphaned_review_checkouts(fake_gh, repo_root, reviews_dir, state_file, config)

    assert reaped == [100]

    state = load_state(state_file)

    # The sweep's REAL work survives the merge: PR 100 was reaped as merged.
    pr100 = state["prs"]["100"]
    assert pr100.get("status") == "merged"
    assert pr100.get("review_dispatch_status") is None

    # The sweep's own event made it into the merged ring via
    # ``_append_sweep_events`` -- the only event path this sweep uses (unlike
    # the stalled-claim sweep, which also appends inline). Confirms the
    # identity-based event diff covers this path too.
    kinds = [e.get("kind") for e in state.get("events", [])]
    assert "review_dispatch_lifecycle_reaped" in kinds, (
        f"sweep's own event missing from merged ring: kinds={kinds!r}"
    )

    # The untouched PR 200 retains the concurrent unescalate -- NOT the stale
    # snapshot's escalated value. This is the bug: every field unescalate
    # changed was being restored to its pre-commit value.
    pr200 = state["prs"]["200"]
    assert pr200.get("status") == PASSIVE_OPEN_STATUS, (
        f"concurrent unescalate clobbered: pr.status={pr200.get('status')!r} "
        f"(expected {PASSIVE_OPEN_STATUS!r})"
    )
    assert pr200.get("review_dispatch_attempt_count") == 0, (
        f"concurrent unescalate clobbered: review_dispatch_attempt_count="
        f"{pr200.get('review_dispatch_attempt_count')!r} (expected 0)"
    )
    assert "review_dispatch_failed_at" not in pr200, (
        f"concurrent unescalate clobbered: stale review_dispatch_failed_at="
        f"{pr200.get('review_dispatch_failed_at')!r} restored"
    )
