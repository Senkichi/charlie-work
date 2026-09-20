"""``_log_is_stalled_at_shim`` staleness-predicate tests for ``charlie_work.worker``.

Split out of ``tests/test_worker_stalled_sessions.py`` (issue #1574, Track-1
shoulder, second-stage seam split): the launch-stall predicate tests, including
the issue #280 fresh-real-activity override, the issue #307 inconclusive-probe
deferrals, and the issue #353 worktree-mtime threshold cases. All test bodies
moved verbatim; no renames, no fixture hoists into ``conftest.py``.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work.config import PostMortemConfig, WatchdogConfig
from charlie_work.post_mortem import (
    ActivitySource,
    RealActivityProbe,
    real_activity_for_worker,
)
from charlie_work.worker import _log_is_stalled_at_shim


def test_log_is_stalled_at_shim_with_marker(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns True when log has shim marker and is stale."""
    log_path = tmp_path / "issue-1.log"
    # Write a log with the shim marker (typical frozen log size ~424-425 bytes)
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    # Set mtime to 10 minutes ago (past the default 5-minute grace period)
    old_time = datetime.now(UTC) - timedelta(minutes=10)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    assert _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_without_marker(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns False when log lacks shim marker."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("Some other log content\n", encoding="utf-8")

    # Set mtime to 10 minutes ago
    old_time = datetime.now(UTC) - timedelta(minutes=10)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    assert not _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_within_grace_period(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns False when log is within grace period."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    # Set mtime to 2 minutes ago (within the 5-minute grace period)
    old_time = datetime.now(UTC) - timedelta(minutes=2)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    assert not _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_large_log(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns False when log is large (>1KB)."""
    log_path = tmp_path / "issue-1.log"
    # Write a large log with the shim marker
    large_content = "[shim] .devin infra materialized\n" + "x" * 2000
    log_path.write_text(large_content, encoding="utf-8")

    # Set mtime to 10 minutes ago
    old_time = datetime.now(UTC) - timedelta(minutes=10)
    import os

    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    assert not _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_nonexistent_log(tmp_path: Path) -> None:
    """_log_is_stalled_at_shim returns False when log file doesn't exist."""
    log_path = tmp_path / "issue-1.log"
    now = datetime.now(UTC)
    assert not _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now)


def test_log_is_stalled_at_shim_with_fresh_real_activity(tmp_path: Path) -> None:
    """Issue #280: frozen sidecar log is ignored when real-session activity is fresh."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    fresh_timestamp = now - timedelta(minutes=1)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=fresh_timestamp,
                staleness_seconds=(now - fresh_timestamp).total_seconds(),
                error=None,
            ),
        )
    )

    assert not _log_is_stalled_at_shim(
        log_path, grace_minutes=5, now=now, real_activity_probe=probe
    )


def test_log_is_stalled_at_shim_with_stale_real_activity(tmp_path: Path) -> None:
    """Issue #280: launch stall is still detected when real activity is also stale."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    stale_timestamp = now - timedelta(minutes=10)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=stale_timestamp,
                staleness_seconds=(now - stale_timestamp).total_seconds(),
                error=None,
            ),
        )
    )

    assert _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now, real_activity_probe=probe)


def test_log_is_stalled_at_shim_with_all_errored_probe_deferred(tmp_path: Path) -> None:
    """Issue #307 scope-extension: _log_is_stalled_at_shim must not fail open on an
    all-errored probe.

    This is the site the reviewer reproduced directly: reconcile.py:264 reaches
    this function only for a CONFIRMED-ALIVE worker, and a True return here
    drives an immediate kill_process_tree (reconcile.py:288). An all-errored
    probe is insufficient evidence of a stall.
    """
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=None,
                staleness_seconds=None,
                error="message_nodes query failed (schema drift?): no such column: id",
            ),
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error="no per-PID log found",
            ),
        )
    )

    assert not _log_is_stalled_at_shim(
        log_path, grace_minutes=5, now=now, real_activity_probe=probe
    )


def test_log_is_stalled_at_shim_with_no_match_yet_probe_deferred(tmp_path: Path) -> None:
    """Issue #307 scope-extension: the second inconclusive shape at the shim site.

    Distinct from test_log_is_stalled_at_shim_with_all_errored_probe_deferred:
    here every source is error-free but returned no timestamp match at all
    (e.g. a young session within launch_stall_grace_minutes whose sessions.db
    row hasn't landed yet). This must also defer rather than report a stall.
    """
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    old_time = datetime.now(UTC) - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    now = datetime.now(UTC)
    probe = RealActivityProbe(
        sources=(
            ActivitySource(
                name="sessions.db",
                timestamp=None,
                staleness_seconds=None,
                error=None,
            ),
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error=None,
            ),
        )
    )

    assert not _log_is_stalled_at_shim(
        log_path, grace_minutes=5, now=now, real_activity_probe=probe
    )


def test_log_is_stalled_at_shim_worktree_files_mtime_fresh_beyond_grace(
    tmp_path: Path,
) -> None:
    """Issue #353: worktree mtime freshness uses its own generous threshold."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    now = datetime.now(UTC)
    old_time = now - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    source_file = worktree_path / "foo.py"
    source_file.write_text("# change", encoding="utf-8")
    worktree_mtime = now - timedelta(minutes=8)
    os.utime(source_file, (worktree_mtime.timestamp(), worktree_mtime.timestamp()))

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
    )
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        (now - timedelta(minutes=10)).isoformat(),
        None,
        now,
        watchdog_config=watchdog,
    )

    # 8 minutes is past the 5-minute grace, but within the 45-minute worktree threshold.
    assert not _log_is_stalled_at_shim(
        log_path, grace_minutes=5, now=now, real_activity_probe=probe
    )


def test_log_is_stalled_at_shim_worktree_files_mtime_checkout_noise_stalls(
    tmp_path: Path,
) -> None:
    """Issue #353: checkout-time mtimes do not mask a launch stall."""
    log_path = tmp_path / "issue-1.log"
    log_path.write_text("[shim] .devin infra materialized\n", encoding="utf-8")

    now = datetime.now(UTC)
    old_time = now - timedelta(minutes=10)
    os.utime(log_path, (old_time.timestamp(), old_time.timestamp()))

    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    source_file = worktree_path / "foo.py"
    source_file.write_text("# change", encoding="utf-8")
    started_at = now - timedelta(minutes=30)
    os.utime(source_file, (started_at.timestamp(), started_at.timestamp()))

    watchdog = WatchdogConfig(
        worktree_mtime_enabled=True,
        worktree_mtime_threshold_minutes=45,
    )
    probe = real_activity_for_worker(
        PostMortemConfig(),
        str(worktree_path),
        started_at.isoformat(),
        None,
        now,
        watchdog_config=watchdog,
    )

    assert _log_is_stalled_at_shim(log_path, grace_minutes=5, now=now, real_activity_probe=probe)
