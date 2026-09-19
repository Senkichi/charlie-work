"""Event/corroboration surfaces reported by ``run_doctor``.

Covers recent lane-failure surfacing and in-progress worker
corroboration. Split out of ``tests/test_doctor.py`` (issue #1563,
Track 1 shoulder) -- bodies are verbatim relocations; shared helpers
live in ``tests/_doctor_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from charlie_work.config import (
    AutoMergeConfig,
    DevinConfig,
    WorkerRoleConfig,
)
from charlie_work.doctor import run_doctor
from charlie_work.instrumentation import log_event
from charlie_work.paths import runtime_paths
from charlie_work.subprocess_runner import RunResult
from _doctor_fixtures import (
    FakeDoctorGitHub,
    _config,
    _write_sidecar,
)


def test_check_recent_lane_failures_surfaces_past_event(tmp_path: Path) -> None:
    """#6-G / G-AC3 + G-AC5: a past fleet_pass_config_error event recorded to
    this repo's events.db (by fleet_dispatch._record_lane_failure_event, when
    a lane failed to start on a prior pass) surfaces as a doctor finding.

    Severity is a warning, not a hard error, per doctor.py's own convention
    for "this recently happened, may already be fixed" reports (e.g.
    _check_runner_allocation's staleness checks) — it must not, by itself,
    flip run_doctor()'s overall ok to False the way a currently-broken
    required check does.
    """
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    log_event(
        paths.state_file,
        "fleet_pass_config_error",
        {
            "repo_key": "owner/repo",
            "error": "ConfigError: unknown key(s) in config section 'cross_family': auto_verdict",
        },
        repo="owner/repo",
    )

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    check = by_name["recent lane failures"]
    assert check.ok is False
    assert check.severity == "warning"
    assert "cross_family" in check.detail
    # A warning-severity finding must not by itself block the overall result.
    assert ok is True


def test_check_recent_lane_failures_silent_when_no_events(tmp_path: Path) -> None:
    """No fleet_pass_config_error events -> no "recent lane failures" finding
    at all, so a healthy repo's doctor output stays unchanged (no new noise)."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    assert "recent lane failures" not in by_name


def test_doctor_surfaces_in_progress_corroboration_alive_but_polling(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #1346: doctor's "in-progress worker corroboration" check must
    visibly distinguish an alive-but-polling worker (stale sidecar log mtime,
    fresh events.jsonl corroboration) from a genuinely stalled one (stale log
    AND stale corroboration).

    Both workers share the same stale sidecar log mtime -- the only signal a
    log-mtime monitor sees -- so pre-#1346 they were indistinguishable to an
    operator running `charlie doctor`. After #1346 the check reports the
    watchdog's corroboration verdict (same ``real_activity_probe_for`` +
    ``classify_worker_health`` code path) and buckets the alive-but-polling
    worker under "alive-but-polling" (ok=True, informational) while the
    stalled worker lands under "stalled/dead" (ok=False).
    """
    import os
    import time
    from datetime import UTC, datetime, timedelta
    from unittest.mock import patch

    from charlie_work.config import PostMortemConfig, WatchdogConfig

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    old_time = datetime.now(UTC) - timedelta(minutes=30)
    fresh_time = datetime.now(UTC) - timedelta(minutes=1)

    def _plant_claude_worker(issue_number: int, *, fresh_corroboration: bool) -> None:
        log_path = sessions_dir / f"issue-{issue_number}.claude.log"
        log_path.write_text("working\n", encoding="utf-8")
        # Stale sidecar log mtime for BOTH workers -- the signal log-mtime
        # monitors cannot disambiguate.
        os.utime(log_path, (time.time(), old_time.timestamp()))

        events_path = sessions_dir / f"issue-{issue_number}.events.jsonl"
        ts = fresh_time if fresh_corroboration else old_time
        events_path.write_text(
            f'{{"type": "tool_call", "timestamp": "{ts.isoformat()}"}}\n',
            encoding="utf-8",
        )
        os.utime(events_path, (time.time(), ts.timestamp()))

        _write_sidecar(
            sessions_dir,
            f"issue-{issue_number}.claude.json",
            {
                "issue_number": issue_number,
                "branch": f"agent/issue-{issue_number}",
                "worktree_path": str(tmp_path / f"wt-{issue_number}"),
                "prompt_path": "p.md",
                "command": ["claude", "p.md"],
                "pid": 80000 + issue_number,
                "started_at": (datetime.now(UTC) - timedelta(minutes=10)).isoformat(),
                "log_path": str(log_path),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
            },
        )

    # Worker 1346: alive-but-polling (stale log, fresh corroboration).
    _plant_claude_worker(1346, fresh_corroboration=True)
    # Worker 1347: genuinely stalled (stale log, stale corroboration).
    _plant_claude_worker(1347, fresh_corroboration=False)

    monkeypatch.setattr(
        "charlie_work.devin_shell.probe_devin",
        lambda repo_root, **kwargs: RunResult(returncode=0, stdout="devin 1.2.3", stderr=""),
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        ok, checks = run_doctor(
            tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True
        )

    by_name = {c.name: c for c in checks}
    assert "in-progress worker corroboration" in by_name
    check = by_name["in-progress worker corroboration"]

    # The alive-but-polling worker is healthy per the watchdog; the stalled
    # worker is not. The check fails only on the genuinely stalled one.
    assert check.ok is False
    assert check.severity == "warning"
    assert "alive-but-polling (1)" in check.detail
    assert "issue #1346" in check.detail
    assert "fresh=True" in check.detail
    assert "stalled/dead (1)" in check.detail
    assert "issue #1347" in check.detail
    assert "fresh=False" in check.detail
    # A warning-severity finding must not by itself block the overall result.
    assert ok is True


def test_doctor_in_progress_corroboration_silent_when_no_workers(
    tmp_path: Path, monkeypatch
) -> None:
    """Issue #1346: with no in-progress workers, the corroboration check
    reports an empty detail and ok=True (no noise for a healthy repo)."""
    monkeypatch.setattr(
        "charlie_work.devin_shell.probe_devin",
        lambda repo_root, **kwargs: RunResult(returncode=0, stdout="devin 1.2.3", stderr=""),
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(sessions_dir="sessions"),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    # sessions dir exists but is empty.
    (tmp_path / "sessions").mkdir()

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)

    by_name = {c.name: c for c in checks}
    check = by_name["in-progress worker corroboration"]
    assert check.ok is True
    assert "no in-progress workers" in check.detail
