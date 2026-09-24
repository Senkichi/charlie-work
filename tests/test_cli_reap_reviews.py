"""``charlie reap-reviews`` / ``OrchestratorApp.reap_reviews`` (issue #1874).

The dead-reviewer reap only runs inside ``dispatch_reviews`` at the top of a
``loop()`` pass. Issue #1874 was the first live proof of the gap: a reviewer
pid died mid-review and no pass ran for 70+ minutes (``newest_loop_started
age=74m``), so the claim sat open past every stale threshold and the
heartbeat could only flag it — there was no sanctioned command to force the
reap. These tests pin the standalone path: the same sweep block must run
even while the repo's supervisor lock is held (the wedged-supervisor shape),
and must re-dispatch in the same invocation when the lock is free.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from _review_fixtures import (
    _dispatch_reviews_app,
    _fake_claude_worker_record,
    _make_dead_review_sidecar,
    _write_review_packet,
)
from charlie_work import cli, layout
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.supervise import try_acquire_supervisor_lock


def _stale_dispatched_claim(
    app: Any,
    pr_number: int,
    *,
    age: timedelta = timedelta(hours=1),
    pid: int = 999999999,
) -> None:
    """Seed state.json with a stale ``review_dispatch_dispatched`` claim."""
    old = (datetime.now(UTC) - age).isoformat().replace("+00:00", "Z")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": 10,
            "status": "reviewing",
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": old,
            "reviewer_pid": pid,
            "reviewer_process_start_time": 1.0,
        }
        save_state(app.paths.state_file, state)


def _open_pr(pr_number: int, issue_number: int = 10) -> dict[str, Any]:
    return {
        "number": pr_number,
        "title": f"Fix #{issue_number}",
        "url": f"https://example.test/pull/{pr_number}",
        "headRefName": f"agent/issue-{issue_number}-fix",
        "baseRefName": "main",
        "headRefOid": "sha-100",
        "mergeStateStatus": "CLEAN",
        "body": f"Closes #{issue_number}",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }


def test_reap_reviews_frees_dead_claim_while_supervisor_lock_held(tmp_path: Path) -> None:
    """Issue #1874 core: the reap must not depend on a loop pass running.

    Holds the repo's supervisor lock to simulate the wedged supervisor from
    the incident (a live fleet supervisor mid-pass in this repo's lane).
    ``reap_reviews`` must still free the dead reviewer's stale claim — the
    sweeps never launch a reviewer, so the lock's double-dispatch window
    cannot open — and report that dispatch was skipped.
    """
    app = _dispatch_reviews_app(tmp_path, prs=[_open_pr(100)])
    reviews_dir = app._layout.reviews_dir
    _stale_dispatched_claim(app, 100)
    sidecar = _make_dead_review_sidecar(reviews_dir, 100, "ordinary crash output\n")

    lock = try_acquire_supervisor_lock(layout.supervisor_lock_path(app.paths.root))
    assert lock is not None, "test must hold the supervisor lock to simulate the wedge"
    try:
        result = app.reap_reviews()
    finally:
        lock.release()

    assert result.ok is True
    assert result.data["mode"] == "reap_only"
    assert result.data["dispatch_skipped"] == "supervisor_lock_held"
    assert any(entry.get("pr") == 100 for entry in result.data["stalled"])

    state = load_state(app.paths.state_file)
    pr_state = state["prs"]["100"]
    assert pr_state["review_dispatch_status"] == "review_dispatch_failed"
    assert pr_state["reviewer_pid"] is None
    assert any(
        event.get("kind") == "review_dispatch_stalled"
        and event.get("payload", {}).get("pr_number") == 100
        for event in state.get("events", [])
    )
    # The dead sidecar is reaped so it cannot resurface as a phantom claim.
    assert not sidecar.exists()


def test_reap_reviews_dispatches_freed_claim_when_lock_free(
    monkeypatch, tmp_path: Path
) -> None:
    """With no supervisor running, reap + re-dispatch happen in one call.

    The claim is reaped by the sweep block at the top of dispatch_reviews,
    then the freed PR is re-dispatched immediately instead of waiting out a
    pass cadence that may not arrive (the #1874/#1863 failure shape).
    """
    app = _dispatch_reviews_app(tmp_path, prs=[_open_pr(100)])
    reviews_dir = app._layout.reviews_dir
    _write_review_packet(tmp_path, 100, "sha-100")
    _stale_dispatched_claim(app, 100)
    _make_dead_review_sidecar(reviews_dir, 100, "ordinary crash output\n")

    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> ClaudeWorkerRecord:
        launched.append(kwargs.get("issue_number") or args[0])
        return _fake_claude_worker_record(100, "agent/issue-10-fix")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.reap_reviews()

    assert result.ok is True
    assert result.data["mode"] == "reap_and_dispatch"
    assert launched == [100]
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert any(
        event.get("kind") == "review_dispatch_stalled"
        and event.get("payload", {}).get("pr_number") == 100
        for event in state.get("events", [])
    )


def test_reap_reviews_dry_run_reports_without_mutating(tmp_path: Path) -> None:
    """``--dry-run`` previews the reap: the stale claim is reported as
    reapable, but state, sidecar, and checkouts are untouched."""
    app = _dispatch_reviews_app(tmp_path, prs=[_open_pr(100)], dry_run=True)
    reviews_dir = app._layout.reviews_dir
    _stale_dispatched_claim(app, 100)
    sidecar = _make_dead_review_sidecar(reviews_dir, 100, "ordinary crash output\n")

    result = app.reap_reviews()

    assert result.ok is True
    assert result.data["dry_run"] is True
    assert result.data["reapable"] == [100]
    claims = {c["pr"]: c for c in result.data["claims"]}
    assert claims[100]["status"] == "dispatched"
    assert claims[100]["pid_alive"] is False
    assert claims[100]["stale"] is True
    assert claims[100]["would_reap"] is True

    # Nothing was reaped: claim intact, sidecar intact, no sweep events.
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert sidecar.exists()
    assert not any(
        event.get("kind") == "review_dispatch_stalled" for event in state.get("events", [])
    )


def test_reap_reviews_skips_fresh_dead_claim(tmp_path: Path) -> None:
    """A dead reviewer inside the 5-minute stale window is not reaped —
    the timeout exists so a very recently dead reviewer is not immediately
    re-dispatched into a flaky launch path."""
    app = _dispatch_reviews_app(tmp_path, prs=[_open_pr(100)])
    reviews_dir = app._layout.reviews_dir
    _stale_dispatched_claim(app, 100, age=timedelta(seconds=30))
    sidecar = _make_dead_review_sidecar(
        reviews_dir,
        100,
        "ordinary crash output\n",
        started_at=(datetime.now(UTC) - timedelta(seconds=30))
        .isoformat()
        .replace("+00:00", "Z"),
    )

    lock = try_acquire_supervisor_lock(layout.supervisor_lock_path(app.paths.root))
    assert lock is not None
    try:
        result = app.reap_reviews()
    finally:
        lock.release()

    assert result.ok is True
    assert result.data["mode"] == "reap_only"
    assert result.data["stalled"] == []
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert sidecar.exists()


def test_reap_reviews_skips_live_reviewer(monkeypatch, tmp_path: Path) -> None:
    """A stale-timestamped claim whose reviewer pid is still alive is left
    alone: liveness, not the timestamp alone, decides the reap."""
    app = _dispatch_reviews_app(tmp_path, prs=[_open_pr(100)])
    reviews_dir = app._layout.reviews_dir
    _stale_dispatched_claim(app, 100)
    sidecar = _make_dead_review_sidecar(reviews_dir, 100, "ordinary crash output\n")

    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: True)
    monkeypatch.setattr("charlie_work.stalled_review_reap.is_pid_alive", lambda *a: True)

    lock = try_acquire_supervisor_lock(layout.supervisor_lock_path(app.paths.root))
    assert lock is not None
    try:
        result = app.reap_reviews()
    finally:
        lock.release()

    assert result.ok is True
    assert result.data["stalled"] == []
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert sidecar.exists()


def test_reap_reviews_reaps_when_dispatch_disabled(tmp_path: Path) -> None:
    """Issue #868 through the new command: ``review_dispatch.enabled=false``
    gates launches, never the cleanup a previously-dispatched claim needs."""
    app = _dispatch_reviews_app(tmp_path, prs=[_open_pr(100)], enabled=False)
    reviews_dir = app._layout.reviews_dir
    _stale_dispatched_claim(app, 100)
    _make_dead_review_sidecar(reviews_dir, 100, "ordinary crash output\n")

    # Lock free -> reap_reviews delegates to dispatch_reviews, which runs the
    # sweeps above the enabled gate and then returns the disabled result.
    result = app.reap_reviews()

    assert result.ok is True
    assert result.data["mode"] == "reap_and_dispatch"
    assert result.data["disabled"] is True
    state = load_state(app.paths.state_file)
    assert state["prs"]["100"]["review_dispatch_status"] == "review_dispatch_failed"


def test_reap_reviews_no_open_claims_is_clean_noop(tmp_path: Path) -> None:
    """Nothing to reap: a healthy repo returns ok with an empty sweep."""
    app = _dispatch_reviews_app(tmp_path, prs=[])

    lock = try_acquire_supervisor_lock(layout.supervisor_lock_path(app.paths.root))
    assert lock is not None
    try:
        result = app.reap_reviews()
    finally:
        lock.release()

    assert result.ok is True
    assert result.data["mode"] == "reap_only"
    assert result.data["stalled"] == []


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def test_reap_reviews_subparser_parses_flags() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["reap-reviews"])
    assert args.command == "reap-reviews"
    assert args.limit is None
    assert args.dry_run is False

    args = parser.parse_args(["reap-reviews", "--limit", "3", "--dry-run"])
    assert args.limit == 3
    assert args.dry_run is True

    # Global --dry-run works in either position (the _add_dry_run SUPPRESS
    # convention, guarded repo-wide by test_cli_dry_run.py).
    args = parser.parse_args(["--dry-run", "reap-reviews"])
    assert args.dry_run is True


def test_reap_reviews_is_state_affecting_for_sibling_clone_guard() -> None:
    """The command mutates state.json, so it must be in
    ``_STATE_AFFECTING_COMMANDS`` — otherwise a wrong-cwd invocation writes a
    reap record into a sibling clone's phantom state dir (issue #1376)."""
    assert "reap-reviews" in cli._STATE_AFFECTING_COMMANDS
