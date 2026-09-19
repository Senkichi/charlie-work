"""Shared helpers for the wave-2 dispatch/governor test split (issue #1548).

Hoisted out of ``tests/test_charlie_work.py`` so both that module and the new
``test_charlie_work_dispatch_*`` / ``test_charlie_work_*_governor.py`` siblings
can import them:

- ``_stub_real_activity_probe_for_stalled_tests`` -- the autouse fixture that
  stubs ``charlie_work.worker.real_activity_probe_for`` (issue #307). It was
  module-level autouse in ``test_charlie_work.py``; each consumer imports it
  so the same stub keeps applying to every relocated test. Tests that need a
  live probe opt out via the ``real_activity_probe_live`` marker, unchanged.
- ``_requests`` / ``_seed_backdated_dispatch_event`` / ``_fail_if_launched`` /
  ``_make_stalled_sidecar`` / ``_cross_repo_issue_body`` -- plain helpers used
  by the moved tests (some also by tests that stayed behind).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from charlie_work.instrumentation import log_event


@pytest.fixture(autouse=True)
def _stub_real_activity_probe_for_stalled_tests(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Issue #307: stall-detection tests need a stale real-activity probe.

    Without this fixture, ``real_activity_probe_for`` reaches the host's real
    ``sessions.db`` and per-PID logs and returns an all-errored probe. Issue #307
    makes an all-errored probe fail-open (defer), so stall tests that are not
    about real-activity corroboration would otherwise return HEALTHY. Tests that
    intentionally exercise a live (unstubbed) probe opt out via the
    ``real_activity_probe_live`` marker instead of a rename-fragile name match.
    """
    if request.node.get_closest_marker("real_activity_probe_live") is not None:
        return

    from datetime import datetime, timedelta, UTC
    from charlie_work.post_mortem import ActivitySource, RealActivityProbe

    def _stale_probe(*_args: object, **_kwargs: object) -> RealActivityProbe:
        now = datetime.now(UTC)
        timestamp = now - timedelta(minutes=30)
        return RealActivityProbe(
            sources=(
                ActivitySource(
                    name="devin_per_pid_log",
                    timestamp=timestamp,
                    staleness_seconds=(now - timestamp).total_seconds(),
                    error=None,
                ),
            )
        )

    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", _stale_probe)


def _requests(count: int, tmp_path: Path) -> list:
    from charlie_work.adapters import SessionRequest

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt", encoding="utf-8")
    return [
        SessionRequest(
            issue_number=i,
            issue_title=f"Test {i}",
            prompt_path=prompt_path,
            branch_name=f"agent/issue-{i}",
        )
        for i in range(1, count + 1)
    ]


def _seed_backdated_dispatch_event(state_path: Path, ts: str, issue_numbers: list[int]) -> None:
    """Write one ``dispatch`` event to events.db with a caller-chosen ``ts``.

    ``log_event`` always stamps real wall-clock time, so this inserts
    normally and then backdates the row -- the same pattern
    ``tests/test_dispatch_staleness.py`` uses to build a staleness baseline.
    """
    log_event(state_path, "dispatch", {"issue_numbers": issue_numbers})
    db_path = state_path.parent / "events.db"
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.execute("SELECT MAX(id) FROM events WHERE kind = 'dispatch'")
        row = cursor.fetchone()
        if row and row[0]:
            conn.execute("UPDATE events SET ts = ? WHERE id = ?", (ts, row[0]))
            conn.commit()
    finally:
        conn.close()


def _fail_if_launched(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Fail the test loudly if a reviewer is launched -- reused for every
    fixture below that must never reach a paid reviewer session (issue
    #1258's co-occurring-failure gate). Mirrors the identical pattern in
    ``test_fix_escalated_dispatch_gate.py`` for the analogous escalated-PR
    gate."""
    launched: list[Any] = []

    def fake_launch(*args: Any, **kwargs: Any) -> Any:
        launched.append((args, kwargs))
        raise AssertionError("launch_claude_worker must not be called on red CI")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    return launched


def _make_stalled_sidecar(
    sessions_dir: Path, issue_number: int, *, log_text: str
) -> tuple[Path, Path]:
    """Write a fake devin-shell sidecar + log with an old mtime (stalled by time).

    Shared setup for issue #246 regression tests: the stall watchdog must
    classify the log tail (rate-limit/quota signatures) before falling back
    to failure_kind "stalled".
    """
    from datetime import UTC, datetime, timedelta
    import os as _os

    log_file = sessions_dir / f"issue-{issue_number}.log"
    log_file.write_text(log_text, encoding="utf-8")
    old_time = datetime.now(UTC) - timedelta(minutes=25)
    timestamp = old_time.timestamp()
    _os.utime(log_file, (timestamp, timestamp))

    sidecar = sessions_dir / f"issue-{issue_number}.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": issue_number,
                "branch": f"agent/issue-{issue_number}",
                "worktree_path": "/fake/path",
                "prompt_path": "/fake/prompt",
                "command": ["devin", "--print"],
                "pid": 99999,
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_file),
                "error": None,
                "process_start_time": 1234567890.0,
            }
        ),
        encoding="utf-8",
    )
    return sidecar, log_file


def _cross_repo_issue_body() -> str:
    """Issue body whose every referenced file path is absent from the target repo.

    Mirrors the #1010/#953 scenario: the subject code lives in a sibling repo,
    so every path the issue references is missing from the repo the worker is
    dispatched against.
    """
    return (
        "But **#953's code does not live in this repo.** `suite_coverage.py` is at "
        "`C:/Users/operator/repos/ci_runners/src/ci_fleet/suite_coverage.py`; there is no "
        "`src/charlie_work/suite_coverage.py`. The worker, handed an isolated checkout "
        "of a repo that does not contain the file it was asked to change, went to "
        "`C:\\Users\\operator\\repos\\ci_runners` — the **shared main checkout** — and worked there."
    )
