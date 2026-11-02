from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work.process_utils import (
    find_worker_terminal_status,
    start_terminal_status_watcher,
    worker_terminal_status_path,
    write_worker_terminal_status,
)

# --- issue #773: durable worker terminal-status record -----------------------


def test_worker_terminal_status_path_naming(tmp_path: Path) -> None:
    """The terminal-status path follows issue-<n>.<suffix>.terminal.json."""
    path = worker_terminal_status_path(tmp_path, 207, "claude")
    assert path == tmp_path / "issue-207.claude.terminal.json"


def test_write_worker_terminal_status_round_trips_and_is_atomic(tmp_path: Path) -> None:
    """Writing a terminal-status record is atomic (no leftover .tmp) and round-trips."""
    path = tmp_path / "issue-1.claude.terminal.json"
    write_worker_terminal_status(
        path,
        pid=4242,
        exit_code=0,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:05:00Z",
        duration_seconds=300.0,
    )

    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "pid": 4242,
        "exit_code": 0,
        "started_at": "2026-07-30T00:00:00Z",
        "ended_at": "2026-07-30T00:05:00Z",
        "duration_seconds": 300.0,
    }


def test_write_worker_terminal_status_round_trips_worker_outcome(tmp_path: Path) -> None:
    """Issue #935: terminal status preserves the worker's push/PR-failure signal."""
    path = tmp_path / "issue-1.claude.terminal.json"
    worker_outcome = {"push_succeeded": True, "pr_created": False, "error": "gh unauthenticated"}
    write_worker_terminal_status(
        path,
        pid=4242,
        exit_code=0,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:05:00Z",
        duration_seconds=300.0,
        worker_outcome=worker_outcome,
    )

    record = find_worker_terminal_status(tmp_path, 1)
    assert record is not None
    assert record["worker_outcome"] == worker_outcome


def test_write_worker_terminal_status_round_trips_worker_outcome_written_at(
    tmp_path: Path,
) -> None:
    """B5 (wf-review-opus.md) / design doc §3, Group A step A1: the outcome
    file's own mtime is additive alongside ``worker_outcome`` and ``ended_at``
    -- consumers use it as the outcome's freshness anchor instead of
    ``ended_at`` (the watcher's own exit-observation time, not when the
    outcome file was actually written).
    """
    path = tmp_path / "issue-1.claude.terminal.json"
    write_worker_terminal_status(
        path,
        pid=4242,
        exit_code=0,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:05:00Z",
        duration_seconds=300.0,
        worker_outcome={"push_succeeded": True, "pr_created": False},
        worker_outcome_written_at="2026-07-29T23:50:00Z",
    )

    record = find_worker_terminal_status(tmp_path, 1)
    assert record is not None
    assert record["worker_outcome_written_at"] == "2026-07-29T23:50:00Z"
    # Distinct from ended_at -- proving the field is not just a copy.
    assert record["worker_outcome_written_at"] != record["ended_at"]


def test_write_worker_terminal_status_omits_worker_outcome_written_at_when_absent(
    tmp_path: Path,
) -> None:
    """Rollout-window compatibility (design doc §3): a caller that does not
    pass ``worker_outcome_written_at`` gets a record without the key at all,
    not a ``null`` -- matching ``worker_outcome``'s own omit-when-absent
    convention so an old record and a "mtime unavailable" record are
    indistinguishable from a "no signal" record, exactly as intended.
    """
    path = tmp_path / "issue-1.claude.terminal.json"
    write_worker_terminal_status(
        path,
        pid=4242,
        exit_code=0,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:05:00Z",
        duration_seconds=300.0,
        worker_outcome={"push_succeeded": True, "pr_created": False},
    )

    record = find_worker_terminal_status(tmp_path, 1)
    assert record is not None
    assert "worker_outcome_written_at" not in record


def test_write_worker_terminal_status_creates_missing_parent(tmp_path: Path) -> None:
    """The sessions directory does not need to pre-exist."""
    path = tmp_path / "nested" / "issue-1.claude.terminal.json"
    write_worker_terminal_status(
        path,
        pid=1,
        exit_code=1,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:00:01Z",
        duration_seconds=1.0,
    )
    assert path.exists()


def test_find_worker_terminal_status_returns_none_when_absent(tmp_path: Path) -> None:
    """No terminal record for the issue -> None (legacy/fallback path)."""
    assert find_worker_terminal_status(tmp_path, 999) is None


def test_find_worker_terminal_status_returns_none_when_dir_missing(tmp_path: Path) -> None:
    """A sessions dir that does not exist at all is handled without raising."""
    assert find_worker_terminal_status(tmp_path / "does-not-exist", 1) is None


def test_find_worker_terminal_status_reads_written_record(tmp_path: Path) -> None:
    """A record written via write_worker_terminal_status is found by issue number."""
    path = worker_terminal_status_path(tmp_path, 55, "claude")
    write_worker_terminal_status(
        path,
        pid=100,
        exit_code=0,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:01:00Z",
        duration_seconds=60.0,
    )

    record = find_worker_terminal_status(tmp_path, 55)
    assert record is not None
    assert record["pid"] == 100
    assert record["exit_code"] == 0
    assert record["duration_seconds"] == 60.0


def test_find_worker_terminal_status_picks_most_recent_by_mtime(tmp_path: Path) -> None:
    """When multiple adapter suffixes wrote a record, the newest mtime wins."""
    older = worker_terminal_status_path(tmp_path, 7, "claude")
    newer = worker_terminal_status_path(tmp_path, 7, "api")

    write_worker_terminal_status(
        older,
        pid=1,
        exit_code=1,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:00:01Z",
        duration_seconds=1.0,
    )
    write_worker_terminal_status(
        newer,
        pid=2,
        exit_code=0,
        started_at="2026-07-30T00:01:00Z",
        ended_at="2026-07-30T00:01:02Z",
        duration_seconds=2.0,
    )
    # Force an unambiguous mtime ordering regardless of filesystem timestamp
    # resolution (some Windows filesystems only resolve mtime to ~2 seconds).
    old_time = (datetime.now(UTC) - timedelta(minutes=10)).timestamp()
    os.utime(older, (old_time, old_time))

    record = find_worker_terminal_status(tmp_path, 7)
    assert record is not None
    assert record["pid"] == 2
    assert record["exit_code"] == 0


def test_find_worker_terminal_status_skips_corrupt_json(tmp_path: Path) -> None:
    """A corrupt record for one adapter suffix does not hide a valid one."""
    corrupt = worker_terminal_status_path(tmp_path, 8, "claude")
    valid = worker_terminal_status_path(tmp_path, 8, "api")

    corrupt.write_text("{not valid json", encoding="utf-8")
    write_worker_terminal_status(
        valid,
        pid=3,
        exit_code=0,
        started_at="2026-07-30T00:00:00Z",
        ended_at="2026-07-30T00:00:05Z",
        duration_seconds=5.0,
    )

    record = find_worker_terminal_status(tmp_path, 8)
    assert record is not None
    assert record["pid"] == 3


def test_start_terminal_status_watcher_records_exit_code_without_blocking(
    tmp_path: Path,
) -> None:
    """The watcher thread captures the real exit code and does not block the caller."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(1); raise SystemExit(7)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )
    path = tmp_path / "issue-1.claude.terminal.json"

    import time as time_module

    before = time_module.monotonic()
    thread = start_terminal_status_watcher(proc, path)
    call_duration = time_module.monotonic() - before

    # start_terminal_status_watcher must return immediately -- the subprocess
    # sleeps for 1 second before exiting, so a blocking implementation would
    # make this call take >= 1 second.
    assert call_duration < 0.5

    thread.join(timeout=5)
    assert not thread.is_alive()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pid"] == proc.pid
    assert payload["exit_code"] == 7
    assert payload["duration_seconds"] >= 1.0


def test_start_terminal_status_watcher_records_outcome_files_own_mtime(
    tmp_path: Path,
) -> None:
    """B5 (wf-review-opus.md) / design doc §3, Group A step A1: the watcher
    must capture the outcome file's own mtime as ``worker_outcome_written_at``,
    not stamp it with the watcher's own exit-observation time (``ended_at``).
    A leftover outcome file from an earlier dispatch sitting in a reused
    worktree has a mtime from well before this process even started; if the
    watcher stamped it with ``ended_at`` instead, that leftover would read as
    freshly written on every subsequent dispatch that reuses the worktree.
    """
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome_path = worktree_path / ".worker-outcome.json"
    outcome_path.write_text(
        json.dumps({"outcome": "blocked", "reason": "leftover from an earlier dispatch"}),
        encoding="utf-8",
    )
    # Pin the outcome file's mtime well before the process below even starts,
    # simulating a leftover from an earlier dispatch of the same worktree.
    old_mtime = datetime(2020, 1, 1, tzinfo=UTC).timestamp()
    os.utime(outcome_path, (old_mtime, old_mtime))

    proc = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(1)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )
    path = tmp_path / "issue-1.claude.terminal.json"
    thread = start_terminal_status_watcher(proc, path, worktree_path=worktree_path)
    thread.join(timeout=5)
    assert not thread.is_alive()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["worker_outcome"]["outcome"] == "blocked"
    # The recorded written_at reflects the FILE's mtime (2020), not the
    # watcher's own exit-observation time (recorded as ended_at, which is
    # necessarily >= this test's own start time, decades later).
    recorded = datetime.fromisoformat(payload["worker_outcome_written_at"].replace("Z", "+00:00"))
    assert recorded.year == 2020
    assert payload["worker_outcome_written_at"] != payload["ended_at"]
