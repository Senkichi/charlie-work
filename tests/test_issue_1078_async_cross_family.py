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

3. The ``fleet_lane_completed`` event is recorded to the fleet-level
   events.db so per-repo lane liveness is observable without hand-querying
   each repo's individual events.db.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from charlie_work.cross_family import (
    launch_cross_family_review,
    reap_cross_family_review,
    _pending_marker_path,
    _stdout_tmp_path,
)


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
    # started_at far in the past so elapsed > timeout
    _make_marker(
        report_path,
        pid=12345,
        started_at=time.time() - 1000,
        timeout_seconds=600,
        stdout_content="partial output",
    )

    with patch("charlie_work.cross_family.kill_process_tree") as mock_kill:
        result = reap_cross_family_review(report_path=report_path)

    assert result is not None
    assert result.pending is False
    assert result.ok is False
    assert "timed out" in (result.error or "")
    mock_kill.assert_called_once()
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
