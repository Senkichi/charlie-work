"""Regression tests for issue #1078: synchronous cross-family review inflates
the shared fleet pass interval.

The fix splits the blocking ``run_cross_family_review`` call across two fleet
passes via an async ``launch_cross_family_review`` (Popen, non-blocking) +
``reap_cross_family_review`` (poll, collect) pair. These tests verify:

1. ``launch_cross_family_review`` returns immediately with ``pending=True``
   even when the subprocess would take a long time — this is the property
   that prevents one repo's reviewer latency from blocking the other repo's
   lane in the shared sequential fleet pass.

2. ``reap_cross_family_review`` correctly distinguishes pending (still
   running), completed (ok), and failed (timeout / empty output) states.

3. The ``cross_family_pending`` guard in ``review()`` / ``_loop_body`` /
   ``_route_rework_candidate_to_review`` correctly skips merge_ready and
   the rework-status flip when the cross-family review is in flight — the
   deferred-packet/skip-merge path that the async split introduces.

The ``fleet_lane_completed`` fleet-level event added by this PR is tested in
``tests/test_fleet_dispatch.py`` alongside the other ``fleet_loop`` event
tests, not here.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from charlie_work.config import CrossFamilyConfig, DevinConfig, OrchestratorConfig
from charlie_work.cross_family import (
    CrossFamilyResult,
    launch_cross_family_review,
    reap_cross_family_review,
    _pending_marker_path,
    _stdout_tmp_path,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp

from test_charlie_work import FakeGitHub


# A body with a real severity marker — passes report_body_is_valid.
_REAL_BODY = "**BLOCKER**\nsomething needs fixing\n\nVerdict: request changes"


class _FakePopen:
    """Fake ``subprocess.Popen`` that records its args and simulates a process.

    ``pid`` is a fixed value that ``is_pid_alive`` will treat as dead by
    default (PID 0 is always dead). Tests that need the "alive" path patch
    ``is_pid_alive`` directly.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.pid = 0
        self._args = args
        self._kwargs = kwargs
        # Capture the stdout/stderr file objects so the test can write to them
        self._stdout_file = kwargs.get("stdout")
        self._stderr_file = kwargs.get("stderr")


def _make_marker(
    report_path: Path,
    *,
    pid: int = 0,
    started_at: float | None = None,
    timeout_seconds: int = 600,
    model: str = "codex",
    head_ref_oid: str | None = "sha-abc123",
    stdout_content: str = "",
    expected_start_time: float | None = None,
) -> dict[str, Any]:
    """Write a pending marker + stdout file and return the marker dict."""
    marker = {
        "pid": pid,
        "started_at": started_at if started_at is not None else time.time(),
        "timeout_seconds": timeout_seconds,
        "model": model,
        "report_path": str(report_path),
        "stdout_path": str(_stdout_tmp_path(report_path)),
        "stderr_path": str(report_path.with_suffix(".stderr.tmp")),
        "head_ref_oid": head_ref_oid,
        "expected_start_time": expected_start_time,
    }
    marker_path = _pending_marker_path(report_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    stdout_path = _stdout_tmp_path(report_path)
    stdout_path.write_text(stdout_content, encoding="utf-8")
    return marker


# ---------------------------------------------------------------------------
# launch_cross_family_review — non-blocking property
# ---------------------------------------------------------------------------


def test_launch_returns_immediately_with_pending(tmp_path: Path) -> None:
    """``launch_cross_family_review`` returns ``pending=True`` without waiting
    for the subprocess to complete. This is the core property that prevents
    one repo's reviewer latency from inflating the shared fleet pass interval.
    """
    report_path = tmp_path / "cross-family-review.md"
    prompt_path = tmp_path / "cross-family-prompt.md"

    result = launch_cross_family_review(
        model="codex",
        command=["echo", "hello"],
        repo_root=tmp_path,
        prompt_text="review this",
        prompt_path=prompt_path,
        report_path=report_path,
        timeout_seconds=600,
        popen=_FakePopen,
    )

    assert result.pending is True
    assert result.ok is False
    assert result.report_path == str(report_path)
    # The pending marker must exist so a later reap can collect the result.
    assert _pending_marker_path(report_path).exists()
    # The prompt must have been written.
    assert prompt_path.read_text(encoding="utf-8") == "review this"
    # The report file must NOT have been written — the review is in flight.
    assert not report_path.exists()


def test_launch_dry_run_returns_not_pending(tmp_path: Path) -> None:
    """Dry-run mode returns a synthetic failure without launching a process."""
    report_path = tmp_path / "cross-family-review.md"
    prompt_path = tmp_path / "cross-family-prompt.md"

    result = launch_cross_family_review(
        model="codex",
        command=["echo", "hello"],
        repo_root=tmp_path,
        prompt_text="review this",
        prompt_path=prompt_path,
        report_path=report_path,
        timeout_seconds=600,
        dry_run=True,
        popen=_FakePopen,
    )

    assert result.pending is False
    assert result.ok is False
    assert "DRY-RUN" in (result.error or "")
    assert not _pending_marker_path(report_path).exists()


def test_launch_oserror_returns_failure_not_pending(tmp_path: Path) -> None:
    """If Popen raises OSError, the result is a failure (not pending) with a
    stub written to report_path — matching run_cross_family_review's contract."""

    def _failing_popen(*args: Any, **kwargs: Any) -> subprocess.Popen:
        raise OSError("command not found")

    report_path = tmp_path / "cross-family-review.md"
    prompt_path = tmp_path / "cross-family-prompt.md"

    result = launch_cross_family_review(
        model="codex",
        command=["nonexistent-binary"],
        repo_root=tmp_path,
        prompt_text="review this",
        prompt_path=prompt_path,
        report_path=report_path,
        timeout_seconds=600,
        popen=_failing_popen,
    )

    assert result.pending is False
    assert result.ok is False
    assert "failed to start" in (result.error or "")
    # The failure stub must be written.
    assert report_path.exists()
    assert "(UNAVAILABLE)" in report_path.read_text(encoding="utf-8")
    # No pending marker.
    assert not _pending_marker_path(report_path).exists()


# ---------------------------------------------------------------------------
# reap_cross_family_review — state transitions
# ---------------------------------------------------------------------------


@patch("charlie_work.cross_family.is_pid_alive", return_value=True)
def test_reap_returns_pending_when_process_still_running(
    mock_alive: MagicMock, tmp_path: Path
) -> None:
    """When the process is alive and within the timeout, reap returns pending."""
    report_path = tmp_path / "cross-family-review.md"
    started = time.time()
    _make_marker(report_path, pid=12345, started_at=started, timeout_seconds=600)

    result = reap_cross_family_review(report_path=report_path)

    assert result is not None
    assert result.pending is True
    assert result.ok is False
    # Marker must still exist (not cleaned up while pending).
    assert _pending_marker_path(report_path).exists()


@patch("charlie_work.cross_family.is_pid_alive", return_value=True)
def test_reap_kills_and_reports_timeout(mock_alive: MagicMock, tmp_path: Path) -> None:
    """When the process is alive but the timeout has elapsed, reap kills it
    and writes a failure stub."""
    report_path = tmp_path / "cross-family-review.md"
    # started_at far in the past so elapsed > timeout. expected_start_time is
    # a distinct, non-None value so the assertion below proves it is actually
    # threaded through from the marker to kill_process_tree, not merely
    # defaulting to None on both sides.
    _make_marker(
        report_path,
        pid=12345,
        started_at=time.time() - 1000,
        timeout_seconds=600,
        stdout_content="partial output",
        expected_start_time=1111111111.5,
    )

    with patch("charlie_work.cross_family.kill_process_tree") as mock_kill:
        result = reap_cross_family_review(report_path=report_path)

    assert result is not None
    assert result.pending is False
    assert result.ok is False
    assert "timed out" in (result.error or "")
    # expected_start_time must be forwarded so kill_process_tree can re-check
    # process identity before killing — otherwise a pid recycled by an
    # unrelated process after the reviewer subprocess exited would be killed
    # in its place (PID-reuse safety).
    mock_kill.assert_called_once_with(12345, 1111111111.5)
    # Marker and temp files must be cleaned up.
    assert not _pending_marker_path(report_path).exists()
    assert not _stdout_tmp_path(report_path).exists()
    # Failure stub must be written.
    assert report_path.exists()
    assert "(UNAVAILABLE)" in report_path.read_text(encoding="utf-8")


@patch("charlie_work.cross_family.is_pid_alive", return_value=False)
def test_reap_collects_successful_result(mock_alive: MagicMock, tmp_path: Path) -> None:
    """When the process has exited and stdout is valid, reap writes the report
    and returns ok=True."""
    report_path = tmp_path / "cross-family-review.md"
    _make_marker(
        report_path,
        pid=12345,
        started_at=time.time() - 100,
        timeout_seconds=600,
        stdout_content=_REAL_BODY,
        head_ref_oid="sha-abc123",
    )

    result = reap_cross_family_review(report_path=report_path)

    assert result is not None
    assert result.pending is False
    assert result.ok is True
    assert result.returncode == 0
    # Report must be written with the header + body.
    text = report_path.read_text(encoding="utf-8")
    assert "Cross-family adversarial review" in text
    assert "BLOCKER" in text
    assert "<!-- PR head SHA: sha-abc123 -->" in text
    # Marker and temp files must be cleaned up.
    assert not _pending_marker_path(report_path).exists()
    assert not _stdout_tmp_path(report_path).exists()


@patch("charlie_work.cross_family.is_pid_alive", return_value=False)
def test_reap_writes_failure_stub_for_empty_output(mock_alive: MagicMock, tmp_path: Path) -> None:
    """When the process has exited but stdout is empty/invalid, reap writes a
    failure stub."""
    report_path = tmp_path / "cross-family-review.md"
    _make_marker(
        report_path,
        pid=12345,
        started_at=time.time() - 100,
        timeout_seconds=600,
        stdout_content="",
    )

    result = reap_cross_family_review(report_path=report_path)

    assert result is not None
    assert result.pending is False
    assert result.ok is False
    assert "empty or blocked" in (result.error or "")
    assert report_path.exists()
    assert "(UNAVAILABLE)" in report_path.read_text(encoding="utf-8")
    assert not _pending_marker_path(report_path).exists()


def test_reap_returns_none_when_no_marker(tmp_path: Path) -> None:
    """When no pending marker exists, reap returns None (no pending review)."""
    report_path = tmp_path / "cross-family-review.md"
    result = reap_cross_family_review(report_path=report_path)
    assert result is None


def test_reap_corrupted_marker_cleans_up(tmp_path: Path) -> None:
    """A corrupted marker file is cleaned up and treated as no pending review."""
    report_path = tmp_path / "cross-family-review.md"
    marker_path = _pending_marker_path(report_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("not valid json {{{", encoding="utf-8")

    result = reap_cross_family_review(report_path=report_path)

    assert result is None
    assert not marker_path.exists()


# ---------------------------------------------------------------------------
# Non-blocking property — the core regression test
# ---------------------------------------------------------------------------


def test_launch_does_not_block_on_slow_subprocess(tmp_path: Path) -> None:
    """The core regression test for issue #1078: ``launch_cross_family_review``
    returns in negligible time even when the subprocess would take a long time.

    Against current ``main`` (which uses ``run_cross_family_review``
    synchronously), this test cannot exist — the function does not exist.
    The mutation check reverts ``_cross_family_for_pr`` to call
    ``run_cross_family_review`` synchronously, which would block for the
    full ``timeout_seconds`` on every call.
    """
    report_path = tmp_path / "cross-family-review.md"
    prompt_path = tmp_path / "cross-family-prompt.md"

    class _SlowPopen:
        """Simulates a Popen that starts a long-running process."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.pid = 999999  # unlikely to match a real process

    start = time.monotonic()
    result = launch_cross_family_review(
        model="codex",
        command=["sleep", "600"],
        repo_root=tmp_path,
        prompt_text="review this",
        prompt_path=prompt_path,
        report_path=report_path,
        timeout_seconds=600,
        popen=_SlowPopen,
    )
    elapsed = time.monotonic() - start

    assert result.pending is True
    # The launch must return in well under a second — the whole point is that
    # it does NOT wait for the subprocess. A 5s ceiling is generous and still
    # proves the blocking is gone (the old synchronous path would take 600s).
    assert elapsed < 5.0, f"launch took {elapsed:.1f}s — expected non-blocking"


# ---------------------------------------------------------------------------
# Integration: cross_family_pending guard in _loop_body / review() /
# _route_rework_candidate_to_review — the deferred-packet/skip-merge path
# ---------------------------------------------------------------------------


def _pending_cf_result(report_path: str) -> CrossFamilyResult:
    """A CrossFamilyResult that simulates a launched-but-not-yet-reaped review."""
    return CrossFamilyResult(
        ok=False,
        report_path=report_path,
        model="codex",
        pending=True,
        error="cross-family review launched, pending",
    )


def test_cross_family_pending_skips_merge_ready_in_loop_body(tmp_path: Path) -> None:
    """When ``launch_cross_family_review`` returns ``pending=True``, ``review()``
    returns ``cross_family_pending=True`` and ``_loop_body``'s already_approved
    branch must skip ``merge_ready`` — the old head's "approved" decision must
    NOT trigger a merge for the new head while the cross-family review is in
    flight. This is the core deferred-packet/skip-merge path introduced by the
    async split (#1078).
    """
    config = OrchestratorConfig(
        cross_family=CrossFamilyConfig(enabled=True, command=["echo", "{model}"]),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Record an approved decision at the current head "sha-abc123". This sets
    # state["prs"]["456"]["decision"]="approved", ["status"]="approved", and
    # ["reviewed_head_sha"]="sha-abc123" — the already_approved fast path's
    # preconditions.
    app.record_review(456, "approved", summary="lgtm", verdict_provenance="fresh_llm_review")

    # Advance the PR head so already_approved's head_matches is False, forcing
    # the branch that calls review() → _cross_family_for_pr → launch.
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.pr_head_shas[456] = "sha-new-head"

    report_path = str(paths.prs / "pr-456" / "cross-family-review.md")

    # Spy on merge_ready: any call is a regression — the pending guard must
    # `continue` before reaching the merge check.
    merge_ready_calls: list[int] = []

    def _spy_merge_ready(pr_number: int, **kwargs: Any) -> Any:
        merge_ready_calls.append(pr_number)
        raise AssertionError(
            f"merge_ready must not be called when cross_family_pending (PR {pr_number})"
        )

    with (
        patch(
            "charlie_work.workflow.launch_cross_family_review",
            return_value=_pending_cf_result(report_path),
        ),
        patch("charlie_work.workflow.reap_cross_family_review", return_value=None),
    ):
        app.merge_ready = _spy_merge_ready  # type: ignore[method-assign]
        app.loop(limit=0, merge=False)

    assert merge_ready_calls == [], (
        "merge_ready must not be called when cross_family_pending is True"
    )


def test_cross_family_pending_skips_rework_status_flip(tmp_path: Path) -> None:
    """When ``launch_cross_family_review`` returns ``pending=True``, ``review()``
    returns ``cross_family_pending=True`` and ``_route_rework_candidate_to_review``
    must NOT flip the issue from ``rework_requested`` to ``reviewing`` — no
    packet was written, so flipping would desync state.json from GitHub reality
    (labels still say needs-rework, no packet exists) with no automated
    recovery path. The issue stays ``rework_requested`` for the next pass.
    """
    from charlie_work.janitor import _calculate_patch_id

    config = OrchestratorConfig(
        cross_family=CrossFamilyConfig(enabled=True, command=["echo", "{model}"]),
        devin=DevinConfig(adapter="command", dispatch_command="exit 0"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Record a request_changes baseline: reviewed_head_sha pins the pre-rework
    # head, reviewed_patch_id pins the pre-rework patch content. This also
    # sets issue #123 status to "rework_requested" and adds the needs_rework
    # label.
    reviewed_diff = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+first"
    )
    fake_gh.diffs[456] = reviewed_diff
    app.record_review(
        456, "request_changes", summary="fix A", verdict_provenance="fresh_llm_review"
    )

    # Head advances AND the diff content genuinely changes (different patch-id,
    # not just a sync-merge) — the condition that makes dispatch_rework route
    # to _route_rework_candidate_to_review.
    live_diff = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+second"
    assert _calculate_patch_id(live_diff) != _calculate_patch_id(reviewed_diff), (
        "fixture must reproduce a genuine content change (distinct patch-ids)"
    )
    fake_gh.prs[0]["headRefOid"] = "sha-new-head"
    fake_gh.pr_head_shas[456] = "sha-new-head"
    fake_gh.diffs[456] = live_diff

    # A rework prompt must exist so the candidate is dispatch-eligible.
    pr_dir = paths.prs / "pr-456"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "rework-prompt.md").write_text("Fix the issues", encoding="utf-8")

    report_path = str(pr_dir / "cross-family-review.md")

    with (
        patch(
            "charlie_work.workflow.launch_cross_family_review",
            return_value=_pending_cf_result(report_path),
        ),
        patch("charlie_work.workflow.reap_cross_family_review", return_value=None),
    ):
        result = app.dispatch_rework()

    # The issue must stay rework_requested — NOT flipped to "reviewing".
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested", (
        "issue must stay rework_requested when cross_family_pending — "
        "flipping to reviewing would desync state from GitHub labels"
    )
    # The routing must not have confirmed it as routed to review (no packet).
    assert 123 not in result.data.get("routed_to_review", []), (
        "dispatch_rework must not report the issue as routed to review "
        "when cross_family_pending prevented a packet write"
    )
    # The reviewing label must not have been applied.
    assert (123, app.config.labels.reviewing) not in fake_gh.labels_added, (
        "reviewing label must not be applied when cross_family_pending"
    )


# ---------------------------------------------------------------------------
# _cross_family_for_pr — reap-success staleness re-validation (#1212 round-3)
# ---------------------------------------------------------------------------


def test_cross_family_for_pr_does_not_serve_stale_reaped_report(tmp_path: Path) -> None:
    """A review reaped as ``ok=True`` was launched against whatever head was
    live at launch time. If the PR's head has since advanced, the reaped
    report is for a diff that is no longer current — ``_cross_family_for_pr``
    must not silently serve it as this pass's cross-family section. It must
    re-validate the reaped report's head against the live ``pr["headRefOid"]``
    (via ``report_is_reusable``, the same predicate the adjacent report-reuse
    branch a few lines below already applies) and, on mismatch, fall through
    to a fresh relaunch instead.
    """
    config = OrchestratorConfig(
        cross_family=CrossFamilyConfig(enabled=True, command=["echo", "{model}"]),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_number = 456
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    report_path = pr_dir / "cross-family-review.md"

    # A review was launched against the OLD head and has now completed.
    # pid=0 is always dead (is_pid_alive returns False for pid <= 0), so
    # reap_cross_family_review takes the "process exited" branch for real,
    # with no is_pid_alive mocking needed.
    _make_marker(
        report_path,
        pid=0,
        started_at=time.time() - 5,
        timeout_seconds=600,
        stdout_content=_REAL_BODY,
        head_ref_oid="sha-old-head",
    )

    # The live PR has since advanced to a NEW head.
    live_pr = {**fake_gh.prs[0], "headRefOid": "sha-new-head"}

    with patch(
        "charlie_work.workflow.launch_cross_family_review",
        return_value=_pending_cf_result(str(report_path)),
    ) as mock_launch:
        section, result = app._cross_family_for_pr(
            pr=live_pr,
            issue=fake_gh.issues[0],
            pr_dir=pr_dir,
            pr_number=pr_number,
            issue_number=123,
            diff_path=pr_dir / "diff.patch",
            enabled=True,
        )

    # The reaped report was generated against sha-old-head, not the live
    # sha-new-head — it must not be silently served.
    assert "BLOCKER" not in section, "stale cross-family section was served"
    assert section == ""
    # It must have fallen through to a fresh relaunch rather than returning
    # the stale reaped result directly.
    mock_launch.assert_called_once()
    assert result is not None
    assert result.pending is True


# ---------------------------------------------------------------------------
# _loop_body's non-already_approved (same-head packet skip) branch — two
# further cross_family_pending guard sites (#1212 round-3 finding 3)
# ---------------------------------------------------------------------------


def test_cross_family_pending_guards_non_already_approved_branch(tmp_path: Path) -> None:
    """The ``else`` branch of ``_loop_body`` (the PR has no recorded
    "approved" decision, so the same-head packet-skip logic runs instead of
    the already_approved fast path) has its own two ``cross_family_pending``
    guard sites, in source order:

    1. The not-reached-charge exclusion: ``if not cross_family_current and
       not review.data.get("cross_family_pending"):
       self._charge_cross_family_regen_not_reached(...)``. This runs BEFORE
       the continue below — confirmed by reading workflow.py directly — so a
       pending cross-family review must not be charged as a "regenerator not
       reached" pass; it WAS reached, it just has not completed yet.
    2. The merge-skip continue: ``if review.data.get("cross_family_pending"):
       continue`` — the old decision (if any) must not drive a merge while
       the report is still in flight.

    Neither of these two sites was exercised by any existing test: the
    existing cross_family_pending regression tests cover the
    already_approved branch (test_cross_family_pending_skips_merge_ready_in_loop_body)
    and _route_rework_candidate_to_review
    (test_cross_family_pending_skips_rework_status_flip), but not this
    same-head-packet-skip branch, despite the PR body's claim that all five
    guard call sites are "covered by construction."
    """
    config = OrchestratorConfig(
        cross_family=CrossFamilyConfig(enabled=True, command=["echo", "{model}"]),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    pr_number = 456

    # No app.record_review call, so state.json's pr_state has no "decision"
    # key and already_approved is False -- _loop_body's else branch runs.
    # No review packet exists yet either, so head_current is False and the
    # branch proceeds straight into review() rather than the same-head skip.
    #
    # But an "approved" verdict IS on disk, written directly to
    # review-decision.json the same way an operator's out-of-band write
    # would appear (see the comment at the "packet is current" skip a few
    # lines above this branch in workflow.py) -- deliberately bypassing
    # app.record_review, which would also set state["prs"]["456"]["decision"]
    # = "approved" and route into the already_approved branch instead of the
    # else branch this test targets. _review_decision reads this file
    # directly (workflow.py's OrchestratorApp._review_decision), so without
    # the "if review.data.get('cross_family_pending'): continue" guard at
    # the merge-skip site, the stale approval would drive a merge_ready call
    # for a PR whose packet was never regenerated for the new head. review()
    # returns early on cf_result.pending (workflow.py, before it ever
    # touches review-decision.json), so this file survives the review() call
    # below untouched.
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "summary": "lgtm"}), encoding="utf-8"
    )

    merge_ready_calls: list[int] = []

    def _spy_merge_ready(pr_number: int, **kwargs: Any) -> Any:
        merge_ready_calls.append(pr_number)
        raise AssertionError(
            f"merge_ready must not be called when cross_family_pending (PR {pr_number})"
        )

    report_path = str(paths.prs / f"pr-{pr_number}" / "cross-family-review.md")

    with (
        patch(
            "charlie_work.workflow.launch_cross_family_review",
            return_value=_pending_cf_result(report_path),
        ),
        patch("charlie_work.workflow.reap_cross_family_review", return_value=None),
    ):
        app.merge_ready = _spy_merge_ready  # type: ignore[method-assign]
        app.loop(limit=0, merge=False)

    # Guard site 2 (merge-skip continue): merge_ready must never be reached.
    assert merge_ready_calls == [], (
        "merge_ready must not be called when cross_family_pending is True "
        "in the non-already_approved branch"
    )

    # Guard site 1 (not-reached-charge exclusion): the "regenerator not
    # reached" counter must remain at 0. If the exclusion were missing (or
    # the continue ran before the charge, reordering the guards), this pass
    # -- which DID reach and launch the regenerator -- would have been
    # miscounted as never having reached it.
    record = app._cross_family_regen_record(pr_number=pr_number, head_sha="sha-abc123")
    assert record.get("not_reached", 0) == 0, (
        "cross_family_pending must exclude the pass from the "
        "not-reached-charge — the regenerator WAS reached, it just has not "
        "completed yet"
    )
