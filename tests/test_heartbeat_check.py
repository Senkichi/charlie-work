from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _script_loader import load_script_module


def _load_heartbeat_check() -> ModuleType:
    """Load scripts/heartbeat_check.py as a module without adding scripts to sys.path."""
    path = Path(__file__).parent.parent / "scripts" / "heartbeat_check.py"
    return load_script_module(path, "heartbeat_check")


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


class FakePsutil:
    """Stub psutil surface used by heartbeat_check._reviewer_pid_alive."""

    class Error(Exception):
        pass

    class NoSuchProcess(Error):
        pass

    class AccessDenied(Error):
        pass

    def __init__(self, mapping: dict[int, tuple[bool, float | None]]) -> None:
        self.mapping = mapping

    def pid_exists(self, pid: int) -> bool:
        return self.mapping.get(pid, (False, None))[0]

    def Process(self, pid: int) -> "FakeProcess":
        exists, create_time = self.mapping.get(pid, (False, None))
        if not exists:
            raise self.NoSuchProcess(pid)
        return FakeProcess(create_time, self.NoSuchProcess, pid)


class FakeProcess:
    def __init__(
        self,
        create_time_value: float | None,
        no_such_process_cls: type[Exception],
        pid: int,
    ) -> None:
        self._create_time_value = create_time_value
        self._no_such_process_cls = no_such_process_cls
        self.pid = pid

    def create_time(self) -> float:
        if self._create_time_value is None:
            raise self._no_such_process_cls(self.pid)
        return self._create_time_value


def _iso(minutes_ago: float = 0.0, *, base: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp `minutes_ago` before `base`.

    `base` defaults to the real wall clock, sampled here, for the many
    wide-margin callers below. Tests with a tight margin against a rounded
    or exact-value assertion (issue #828) should pass a frozen `base` so the
    fixture and the production `now` it is compared against derive from the
    same instant instead of racing an unbounded CI stall.
    """
    reference = base if base is not None else datetime.now(timezone.utc)
    ts = reference - timedelta(minutes=minutes_ago)
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_repo(hb: ModuleType, tmp_path: Path) -> Any:
    return hb.RepoInfo(
        slug="owner/repo",
        repo_root=tmp_path,
        state_dir=tmp_path / "state",
        config_path=tmp_path / "orchestrator.config.yaml",
    )


def _write_events_db(
    state_dir: Path,
    rows: list[tuple[str, str] | tuple[str, str, str]] | None = None,
) -> Path:
    """Create an events.db next to state.json with the production `events` schema.

    `rows` is a list of either (ts, kind) pairs (defaulting `level` to
    `'info'`, matching the production schema's default) or (ts, kind, level)
    triples for tests that need to seed error/warning-level rows. Mirrors
    `charlie_work.instrumentation`'s `events` table by hand rather than
    importing the package, since heartbeat_check.py deliberately avoids that
    import (see `fleet_dir`'s docstring) and this check must be tested the
    same way.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    db_path = state_dir / "events.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT    NOT NULL,
                kind            TEXT    NOT NULL,
                payload         TEXT    NOT NULL,
                repo            TEXT,
                correlation_id  TEXT,
                pr_number       INTEGER,
                issue_number    INTEGER,
                level           TEXT DEFAULT 'info'
            )
            """
        )
        for row in rows or []:
            ts, kind = row[0], row[1]
            level = row[2] if len(row) > 2 else "info"
            conn.execute(
                "INSERT INTO events (ts, kind, payload, level) VALUES (?, ?, '{}', ?)",
                (ts, kind, level),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _write_state(state_dir: Path, pr_number: int, pr_state: dict[str, Any]) -> None:
    state = {"version": 1, "prs": {str(pr_number): pr_state}}
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")


def _make_pr_dirs(state_dir: Path, pr_number: int, *, pr_mtime: float | None = None) -> Path:
    pr_dir = state_dir / "prs" / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text("{}", encoding="utf-8")
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "pending"}), encoding="utf-8"
    )
    if pr_mtime is not None:
        os.utime(pr_dir, (pr_mtime, pr_mtime))
    return pr_dir


def _patch_gh(monkeypatch: Any, hb: ModuleType, numbers: list[int]) -> None:
    def fake_run_gh_json(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
        return True, [{"number": n} for n in numbers], ""

    monkeypatch.setattr(hb, "run_gh_json", fake_run_gh_json)


def test_review_claim_timestamp_dispatched(hb: ModuleType) -> None:
    pr_state = {
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-20T22:00:00Z",
        "review_dispatch_pending_at": "2026-07-20T21:00:00Z",
        "review_dispatch_failed_at": "2026-07-20T20:00:00Z",
    }
    assert hb._review_claim_timestamp(pr_state) == "2026-07-20T22:00:00Z"


def test_review_claim_timestamp_pending(hb: ModuleType) -> None:
    pr_state = {
        "review_dispatch_status": "review_dispatch_pending",
        "review_dispatch_pending_at": "2026-07-20T21:05:00Z",
    }
    assert hb._review_claim_timestamp(pr_state) == "2026-07-20T21:05:00Z"


def test_review_claim_timestamp_failed(hb: ModuleType) -> None:
    pr_state = {
        "review_dispatch_status": "review_dispatch_failed",
        "review_dispatch_failed_at": "2026-07-20T20:15:00Z",
    }
    assert hb._review_claim_timestamp(pr_state) == "2026-07-20T20:15:00Z"


def test_review_claim_timestamp_unknown_uses_newest(hb: ModuleType) -> None:
    pr_state = {
        "review_dispatch_status": None,
        "review_dispatched_at": "2026-07-20T22:00:00Z",
        "review_dispatch_pending_at": "2026-07-20T23:00:00Z",
    }
    assert hb._review_claim_timestamp(pr_state) == "2026-07-20T23:00:00Z"


def test_reviewer_pid_alive_none_without_pid(hb: ModuleType, monkeypatch: Any) -> None:
    monkeypatch.setattr(hb, "psutil", FakePsutil({}))
    assert hb._reviewer_pid_alive({}) is None


def test_reviewer_pid_alive_false_for_dead_pid(hb: ModuleType, monkeypatch: Any) -> None:
    monkeypatch.setattr(hb, "psutil", FakePsutil({12345: (False, None)}))
    assert hb._reviewer_pid_alive({"reviewer_pid": 12345}) is False


def test_reviewer_pid_alive_true_when_alive(hb: ModuleType, monkeypatch: Any) -> None:
    monkeypatch.setattr(hb, "psutil", FakePsutil({12345: (True, 1000.0)}))
    assert (
        hb._reviewer_pid_alive({"reviewer_pid": 12345, "reviewer_process_start_time": 1000.0})
        is True
    )


def test_reviewer_pid_alive_false_when_recycled(hb: ModuleType, monkeypatch: Any) -> None:
    monkeypatch.setattr(hb, "psutil", FakePsutil({12345: (True, 2000.0)}))
    assert (
        hb._reviewer_pid_alive({"reviewer_pid": 12345, "reviewer_process_start_time": 1000.0})
        is False
    )


def test_reviewer_pid_alive_true_on_indeterminate_start_time(
    hb: ModuleType, monkeypatch: Any
) -> None:
    monkeypatch.setattr(hb, "psutil", FakePsutil({12345: (True, None)}))
    assert (
        hb._reviewer_pid_alive({"reviewer_pid": 12345, "reviewer_process_start_time": 1000.0})
        is True
    )


def test_check_review_liveness_uses_state_timestamp_not_packet_mtime(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Regression for issue #517.

    The packet directory mtime is ancient, but state.json carries a recent
    review_dispatched_at for a live PID.  The heartbeat must report the age of
    the current dispatch attempt, not the monotonically growing packet age.
    """
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [513])
    _make_pr_dirs(
        repo.state_dir,
        513,
        pr_mtime=(datetime(2020, 1, 1, tzinfo=timezone.utc)).timestamp(),
    )
    _write_state(
        repo.state_dir,
        513,
        {
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": _iso(10),
            "reviewer_pid": 12345,
            "reviewer_process_start_time": 1000.0,
        },
    )
    monkeypatch.setattr(hb, "psutil", FakePsutil({12345: (True, 1000.0)}))

    report = hb.Report()
    hb.check_review_liveness(report, repo)

    assert not report.anomaly
    assert report.lines and "review-liveness" in report.lines[0]
    assert "pid=12345 alive" in report.lines[0]
    assert "open_claims=1" in report.lines[0]


def test_check_review_liveness_flags_dead_pid_past_threshold(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [516])
    _make_pr_dirs(repo.state_dir, 516)
    _write_state(
        repo.state_dir,
        516,
        {
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": _iso(60),
            "reviewer_pid": 24616,
            "reviewer_process_start_time": 1000.0,
        },
    )
    monkeypatch.setattr(hb, "psutil", FakePsutil({24616: (False, None)}))

    report = hb.Report()
    hb.check_review_liveness(report, repo)

    assert report.anomaly
    assert "pid=24616 dead" in report.lines[0]
    assert "threshold=45m" in report.lines[0]


def test_check_review_liveness_ok_for_dead_pid_inside_grace_window(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A dead reviewer inside the 30-min grace window should not yet anomaly."""
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [516])
    _make_pr_dirs(repo.state_dir, 516)
    _write_state(
        repo.state_dir,
        516,
        {
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": _iso(20),
            "reviewer_pid": 24616,
            "reviewer_process_start_time": 1000.0,
        },
    )
    monkeypatch.setattr(hb, "psutil", FakePsutil({24616: (False, None)}))

    report = hb.Report()
    hb.check_review_liveness(report, repo)

    assert not report.anomaly
    assert "pid=24616 dead" in report.lines[0]


def test_check_review_liveness_uses_pending_timestamp(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Regression for issue #828 (originally #822's class): exact rounded-minute
    match against an injected clock, not two independently-sampled `now()`s.

    Production formats `oldest_min={round(age_min)}` where `age_min` is
    computed from `now - claim_time`. With two independent samples this flips
    from 5 to 6 (round(5.5) == 6) once ~30s passes between the fixture write
    and the production check -- comfortably within an observed CI stall. `now`
    is frozen and passed to both the fixture and the check so `age_min` is
    exactly 5.0 regardless of how long the process stalls in between.
    """
    frozen_now = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [100])
    _make_pr_dirs(repo.state_dir, 100)
    _write_state(
        repo.state_dir,
        100,
        {
            "review_dispatch_status": "review_dispatch_pending",
            "review_dispatch_pending_at": _iso(5, base=frozen_now),
        },
    )

    report = hb.Report()
    hb.check_review_liveness(report, repo, now=frozen_now)

    assert not report.anomaly
    assert "pid=None" in report.lines[0]
    assert "oldest_min=5" in report.lines[0]


def test_check_review_liveness_falls_back_to_packet_mtime_when_state_timestamp_missing(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """If state.json has no dispatch timestamp, the packet mtime is the only clock left."""
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [200])
    _make_pr_dirs(
        repo.state_dir,
        200,
        pr_mtime=(datetime(2020, 1, 1, tzinfo=timezone.utc)).timestamp(),
    )
    _write_state(repo.state_dir, 200, {"review_dispatch_status": None})

    report = hb.Report()
    hb.check_review_liveness(report, repo)

    assert report.anomaly
    assert "pr-200" in report.lines[0]


# ---------------------------------------------------------------------------
# Smoke tests for the remaining checks (review-liveness is covered above).
# Each test exercises one check's OK and/or anomaly path with stubbed I/O.
# ---------------------------------------------------------------------------


def _gh_dispatch(monkeypatch: Any, hb: ModuleType, handler: Any) -> None:
    """Install a fake run_gh_json that dispatches to ``handler(args, cwd)``."""

    def fake_run_gh_json(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
        return handler(args, cwd)

    monkeypatch.setattr(hb, "run_gh_json", fake_run_gh_json)


def test_state_file_respects_env_override(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """CHARLIE_WORK_HEARTBEAT_STATE overrides the derived fleet_dir path."""
    custom = tmp_path / "custom-state.json"
    monkeypatch.setenv("CHARLIE_WORK_HEARTBEAT_STATE", str(custom))
    assert hb.state_file() == custom


def test_state_file_derives_from_fleet_dir(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Without an explicit override, state_file() follows CHARLIE_WORK_FLEET_DIR."""
    monkeypatch.delenv("CHARLIE_WORK_HEARTBEAT_STATE", raising=False)
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path))
    assert hb.state_file() == tmp_path / "heartbeat-state.json"


def test_save_and_load_state_round_trip(hb: ModuleType, monkeypatch: Any, tmp_path: Path) -> None:
    """save_state writes atomically and load_state reads it back."""
    monkeypatch.setenv("CHARLIE_WORK_HEARTBEAT_STATE", str(tmp_path / "hb.json"))
    payload = {"last_beat_at": "2026-07-22T00:00:00Z", "repos": {}}
    hb.save_state(payload)
    assert hb.load_state() == payload


def test_check_dispatch_throttle_ok_when_no_state(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    report = hb.Report()
    hb.check_dispatch_throttle(report, repo)
    assert not report.anomaly
    assert "none (no state.json)" in report.lines[0]


def test_check_dispatch_throttle_ok_when_not_throttled(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    (repo.state_dir / "state.json").write_text(
        json.dumps({"throttled_until": None}), encoding="utf-8"
    )
    report = hb.Report()
    hb.check_dispatch_throttle(report, repo)
    assert not report.anomaly
    assert "none" in report.lines[0]


def test_check_dispatch_throttle_ok_within_threshold(hb: ModuleType, tmp_path: Path) -> None:
    """Regression for issue #828: the fixture write and the check call are back-to-back
    statements, but a substring assertion against text that flips at the 10-minute
    remaining-time boundary is still exposed on a sufficiently long stall. Freeze `now`
    so `until` and `resolved_now` never drift apart regardless of scheduling delay.
    """
    frozen_now = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    # _iso(-10, base=frozen_now) = 10 minutes ahead of frozen_now (throttle still active)
    (repo.state_dir / "state.json").write_text(
        json.dumps({"throttled_until": _iso(-10, base=frozen_now)}), encoding="utf-8"
    )
    report = hb.Report()
    hb.check_dispatch_throttle(report, repo, now=frozen_now)
    assert not report.anomaly
    assert "throttled until" in report.lines[0]


def test_check_dispatch_throttle_anomaly_when_exceeds_threshold(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Regression for issue #828: same class as the "within threshold" test above,
    frozen against the 30-minute anomaly boundary instead of the 10-minute one.
    """
    frozen_now = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    # _iso(-60, base=frozen_now) = 60 minutes ahead of frozen_now, beyond the 30-min threshold
    (repo.state_dir / "state.json").write_text(
        json.dumps({"throttled_until": _iso(-60, base=frozen_now)}), encoding="utf-8"
    )
    report = hb.Report()
    hb.check_dispatch_throttle(report, repo, now=frozen_now)
    assert report.anomaly
    assert "cooldown exceeds threshold" in report.lines[0]


def test_check_dispatch_coverage_ok_when_no_dispatchable(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (True, [], ""),
    )
    report = hb.Report()
    hb.check_dispatch_coverage(
        report, repo, {}, {}, skip_delta=False, blocked_numbers=None, blocked_err=""
    )
    assert not report.anomaly
    assert "dispatch-coverage" in report.lines[0]


def test_check_dispatch_coverage_anomaly_when_persisting(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    issues = [
        {"number": 42, "labels": [], "updatedAt": _iso(1)},
    ]
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (True, issues, ""),
    )
    prev = {"dispatchable_issues": [42]}
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_dispatch_coverage(
        report, repo, prev, new, skip_delta=False, blocked_numbers=None, blocked_err=""
    )
    assert report.anomaly
    assert "dispatchable across 2 consecutive beats" in report.lines[0]


def test_check_in_progress_staleness_anomaly_when_unchanged_across_beats(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    updated = _iso(5)
    prev = {"in_progress": {"99": updated}}
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_in_progress_staleness(report, repo, [(99, updated)], prev, new, skip_delta=False)
    assert report.anomaly
    assert "99" in report.lines[0]


def test_check_dispatch_failures_ok_when_no_dir(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    report = hb.Report()
    hb.check_dispatch_failures(report, repo, datetime.now(timezone.utc))
    assert not report.anomaly
    assert "scanned=0" in report.lines[0]


def test_check_dispatch_failures_anomaly_for_new_failure(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    dispatches = repo.state_dir / "dispatches"
    dispatches.mkdir(parents=True, exist_ok=True)
    (dispatches / "bad.json").write_text(json.dumps({"error": "boom"}), encoding="utf-8")
    baseline = datetime.now(timezone.utc) - timedelta(hours=1)
    report = hb.Report()
    hb.check_dispatch_failures(report, repo, baseline)
    assert report.anomaly
    assert "bad.json" in report.lines[0]


def test_check_dispatch_failures_ok_when_failure_before_baseline(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    dispatches = repo.state_dir / "dispatches"
    dispatches.mkdir(parents=True, exist_ok=True)
    old_path = dispatches / "old.json"
    old_path.write_text(json.dumps({"error": "boom"}), encoding="utf-8")
    old_time = (datetime(2020, 1, 1, tzinfo=timezone.utc)).timestamp()
    os.utime(old_path, (old_time, old_time))
    baseline = datetime.now(timezone.utc)
    report = hb.Report()
    hb.check_dispatch_failures(report, repo, baseline)
    assert not report.anomaly


def test_check_log_freshness_ok_when_fresh(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    (repo.state_dir / "run.log").write_text("hi", encoding="utf-8")
    report = hb.Report()
    hb.check_log_freshness(report, repo)
    assert not report.anomaly
    assert "run.log" in report.lines[0]


def test_check_log_freshness_anomaly_when_no_files(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    # state_dir exists but contains no log/state/checkpoint files
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    report = hb.Report()
    hb.check_log_freshness(report, repo)
    assert report.anomaly
    assert "no log/state/checkpoint files" in report.lines[0]


def test_check_log_freshness_anomaly_when_stale(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    stale_path = repo.state_dir / "run.log"
    stale_path.write_text("hi", encoding="utf-8")
    old_time = (datetime(2020, 1, 1, tzinfo=timezone.utc)).timestamp()
    os.utime(stale_path, (old_time, old_time))
    report = hb.Report()
    hb.check_log_freshness(report, repo)
    assert report.anomaly
    assert "older than threshold" in report.lines[0]


def test_load_orchestrator_config_ok_when_no_config_path_registered(
    hb: ModuleType, tmp_path: Path
) -> None:
    # load_repos() represents "no config registered for this repo" as
    # Path("") -- not a real file, must not be treated as cwd (Path("")
    # stringifies to "." and Path(".").exists() is True).
    config, error = hb.load_orchestrator_config(Path(""))
    assert config == {}
    assert error is None


def test_load_orchestrator_config_ok_when_file_absent(hb: ModuleType, tmp_path: Path) -> None:
    config, error = hb.load_orchestrator_config(tmp_path / "does-not-exist.yaml")
    assert config == {}
    assert error is None


def test_load_orchestrator_config_ok_when_valid(hb: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "orchestrator.config.yaml"
    path.write_text("dispatch:\n  max_concurrent_sessions: 3\n", encoding="utf-8")
    config, error = hb.load_orchestrator_config(path)
    assert config == {"dispatch": {"max_concurrent_sessions": 3}}
    assert error is None


def test_load_orchestrator_config_error_on_invalid_yaml(hb: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "orchestrator.config.yaml"
    path.write_text("dispatch: [unterminated\n", encoding="utf-8")
    config, error = hb.load_orchestrator_config(path)
    assert config == {}
    assert error is not None
    assert str(path) in error


def test_load_orchestrator_config_error_on_non_utf8_bytes(hb: ModuleType, tmp_path: Path) -> None:
    # A concurrent partial write can leave non-UTF-8 bytes on disk. The
    # original implementation only caught (OSError, yaml.YAMLError) --
    # UnicodeDecodeError is a ValueError subclass, so this used to raise
    # straight out of read_text() instead of degrading. Must not raise.
    path = tmp_path / "orchestrator.config.yaml"
    path.write_bytes(b"\xff\xfe\x00garbage")
    config, error = hb.load_orchestrator_config(path)
    assert config == {}
    assert error is not None
    assert str(path) in error


def test_load_orchestrator_config_error_on_non_mapping_top_level(
    hb: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "orchestrator.config.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    config, error = hb.load_orchestrator_config(path)
    assert config == {}
    assert error is not None
    assert "list" in error


def test_get_mergequeue_label_and_dispatch_cap_default_quietly_on_broken_config(
    hb: ModuleType, tmp_path: Path
) -> None:
    # get_mergequeue_label/get_dispatch_cap must keep degrading to None on a
    # broken config rather than raising or propagating the error -- callers
    # of these two functions are not where issue #703's signal should
    # surface; check_orchestrator_config is.
    path = tmp_path / "orchestrator.config.yaml"
    path.write_text("dispatch: [unterminated\n", encoding="utf-8")
    assert hb.get_mergequeue_label(path) is None
    assert hb.get_dispatch_cap(path) is None


def test_check_orchestrator_config_ok_when_not_registered(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo = replace(repo, config_path=Path(""))
    report = hb.Report()
    hb.check_orchestrator_config(report, repo)
    assert not report.anomaly
    assert "no config_path registered" in report.lines[0]


def test_check_orchestrator_config_ok_when_absent(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)  # default config_path does not exist
    report = hb.Report()
    hb.check_orchestrator_config(report, repo)
    assert not report.anomaly
    assert "not present" in report.lines[0]


def test_check_orchestrator_config_ok_when_valid(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_text("dispatch:\n  max_concurrent_sessions: 3\n", encoding="utf-8")
    report = hb.Report()
    hb.check_orchestrator_config(report, repo)
    assert not report.anomaly
    assert "readable" in report.lines[0]


def test_check_orchestrator_config_anomaly_when_invalid_yaml(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_text("dispatch: [unterminated\n", encoding="utf-8")
    report = hb.Report()
    hb.check_orchestrator_config(report, repo)
    assert report.anomaly
    assert str(repo.config_path) in report.lines[0]


def test_check_orchestrator_config_anomaly_when_non_utf8(hb: ModuleType, tmp_path: Path) -> None:
    # Behavioral red case: pre-fix, this raised UnicodeDecodeError out of
    # main()'s per-repo loop instead of degrading -- the strongest evidence
    # that the fix, not just its return-type signature, changed behavior.
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_bytes(b"\xff\xfe\x00garbage")
    report = hb.Report()
    hb.check_orchestrator_config(report, repo)
    assert report.anomaly
    assert str(repo.config_path) in report.lines[0]


def test_check_merge_flow_ok_when_no_mergequeue_label(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (True, [], ""),
    )
    report = hb.Report()
    hb.check_merge_flow(report, repo, {}, {}, skip_delta=False)
    assert not report.anomaly
    assert "merge-flow" in report.lines[0]


def test_check_merge_flow_anomaly_when_mergequeue_stalled(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    # Write a config with a mergequeue label so the check counts it.
    repo.config_path.parent.mkdir(parents=True, exist_ok=True)
    repo.config_path.write_text("auto_merge:\n  mergequeue_label: mergequeue\n", encoding="utf-8")

    merged_at = "2020-01-01T00:00:00Z"

    def handler(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
        if "--state" in args and "merged" in args[args.index("--state") + 1]:
            return True, [{"number": 9, "mergedAt": merged_at}], ""
        # open PRs: one carrying the mergequeue label
        return True, [{"number": 1, "labels": [{"name": "mergequeue"}]}], ""

    _gh_dispatch(monkeypatch, hb, handler)
    prev = {
        "mergequeue_count": 1,
        "mergequeue_unchanged_streak": 1,
        "last_merged_at": merged_at,
    }
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_merge_flow(report, repo, prev, new, skip_delta=False)
    assert report.anomaly
    assert "mergequeue count stuck" in report.lines[0]


def test_check_github_rate_ok(hb: ModuleType, monkeypatch: Any, tmp_path: Path) -> None:
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (True, {"resources": {"graphql": {"remaining": 5000}}}, ""),
    )
    report = hb.Report()
    hb.check_github_rate(report, tmp_path)
    assert not report.anomaly
    assert "graphql_remaining=5000" in report.lines[0]


def test_check_github_rate_anomaly_when_low(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (True, {"resources": {"graphql": {"remaining": 100}}}, ""),
    )
    report = hb.Report()
    hb.check_github_rate(report, tmp_path)
    assert report.anomaly
    assert "below threshold" in report.lines[0]


def test_check_github_rate_anomaly_on_gh_failure(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (False, None, "gh exploded"),
    )
    report = hb.Report()
    hb.check_github_rate(report, tmp_path)
    assert report.anomaly


def test_check_runners_skipped_off_windows(hb: ModuleType, monkeypatch: Any) -> None:
    monkeypatch.setattr(hb.sys, "platform", "linux")
    report = hb.Report()
    hb.check_runners(report)
    assert not report.anomaly
    assert "skipped on linux" in report.lines[0]


def test_check_runners_ok_on_windows_with_good_result(hb: ModuleType, monkeypatch: Any) -> None:
    monkeypatch.setattr(hb.sys, "platform", "win32")

    class FakeProc:
        returncode = 0
        stdout = "Last Result: 0\n"
        stderr = ""

    monkeypatch.setattr(hb.subprocess, "run", lambda *a, **k: FakeProc())
    report = hb.Report()
    hb.check_runners(report)
    assert not report.anomaly
    assert "last_result=0" in report.lines[0]


def test_check_runners_anomaly_on_windows_with_bad_result(
    hb: ModuleType, monkeypatch: Any
) -> None:
    monkeypatch.setattr(hb.sys, "platform", "win32")

    class FakeProc:
        returncode = 0
        stdout = "Last Result: 1\n"
        stderr = ""

    monkeypatch.setattr(hb.subprocess, "run", lambda *a, **k: FakeProc())
    report = hb.Report()
    hb.check_runners(report)
    assert report.anomaly
    assert "last run result 1" in report.lines[0]


# ---------------------------------------------------------------------------
# Supervisor heartbeat freshness (issue #627)
# ---------------------------------------------------------------------------


def _set_fleet_dir(hb: ModuleType, monkeypatch: Any, tmp_path: Path) -> Path:
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path))
    return tmp_path


def _write_heartbeat(
    hb: ModuleType,
    fleet_dir: Path,
    *,
    last_beat_at: str,
    exited_at: str | None = None,
    pid: int = 12345,
    full_pass_interval_seconds: int = 300,
    max_pass_runtime_seconds: int = 0,
) -> None:
    payload = {
        "pid": pid,
        # Use the beat time as the start time; check_supervisor_heartbeat only
        # reads last_beat_at/exited_at, so this keeps the fixture date-free.
        "started_at": last_beat_at,
        "last_beat_at": last_beat_at,
        "pass_number": 5,
        "full_pass_interval_seconds": full_pass_interval_seconds,
        "max_pass_runtime_seconds": max_pass_runtime_seconds,
        "exited_at": exited_at,
        "exit_code": 0 if exited_at else None,
    }
    (fleet_dir / hb.SUPERVISOR_HEARTBEAT_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_check_supervisor_heartbeat_anomaly_when_absent(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    _set_fleet_dir(hb, monkeypatch, tmp_path)
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert report.anomaly
    assert "no supervisor-heartbeat.json found" in report.lines[0]


def test_check_supervisor_heartbeat_ok_when_fresh(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    last_beat = (now - timedelta(seconds=60)).isoformat().replace("+00:00", "Z")
    _write_heartbeat(hb, fleet_dir, last_beat_at=last_beat, full_pass_interval_seconds=300)
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert not report.anomaly
    assert "supervisor-heartbeat" in report.lines[0]


def test_check_supervisor_heartbeat_anomaly_when_stale_no_exit(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A stale heartbeat with no exited_at means the supervisor was likely killed."""
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    # 30 minutes old, threshold is 2*300s = 10 minutes.
    now = datetime.now(timezone.utc)
    last_beat = (now - timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    _write_heartbeat(
        hb, fleet_dir, last_beat_at=last_beat, exited_at=None, full_pass_interval_seconds=300
    )
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert report.anomaly
    assert "likely killed or hung" in report.lines[0]


def test_check_supervisor_heartbeat_anomaly_when_stale_with_clean_exit(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A stale heartbeat with exited_at set means the watchdog did not restart it."""
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    last_beat = (now - timedelta(minutes=30)).isoformat().replace("+00:00", "Z")
    _write_heartbeat(
        hb,
        fleet_dir,
        last_beat_at=last_beat,
        exited_at=last_beat,
        full_pass_interval_seconds=300,
    )
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert report.anomaly
    assert "watchdog may be disabled" in report.lines[0]


def test_check_supervisor_heartbeat_anomaly_on_corrupt_file(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    (fleet_dir / hb.SUPERVISOR_HEARTBEAT_FILENAME).write_text("{not json", encoding="utf-8")
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert report.anomaly
    assert "unreadable" in report.lines[0]


def test_check_supervisor_heartbeat_threshold_derives_from_max_pass_runtime(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A longer max_pass_runtime_seconds raises the stale threshold."""
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    # 20 minutes old. With max_pass_runtime=600s, threshold = 2*600/60 = 20 min, so OK.
    last_beat = (now - timedelta(minutes=19)).isoformat().replace("+00:00", "Z")
    _write_heartbeat(
        hb,
        fleet_dir,
        last_beat_at=last_beat,
        full_pass_interval_seconds=300,
        max_pass_runtime_seconds=600,
    )
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert not report.anomaly


def test_check_supervisor_heartbeat_uses_max_pass_runtime_not_full_pass_interval(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """The stale threshold is keyed on max_pass_runtime_seconds, not full_pass_interval_seconds.

    This is the M1 fix: a short full_pass_interval_seconds (the fallback pass
    cadence) does not bound a pass's wall-clock runtime, so a long-running but
    live supervisor must not be flagged as killed.
    """
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    # 15 minutes old. full_pass=300s would give a 10-minute threshold and flag
    # this as stale, but max_pass_runtime=1800s gives a 60-minute threshold.
    last_beat = (now - timedelta(minutes=15)).isoformat().replace("+00:00", "Z")
    _write_heartbeat(
        hb,
        fleet_dir,
        last_beat_at=last_beat,
        full_pass_interval_seconds=300,
        max_pass_runtime_seconds=1800,
    )
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert not report.anomaly


def test_check_loop_pass_freshness_anomaly_when_stale_but_log_fresh(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Regression for the log-fresh-but-loop-stale failure shape (#851/#854).

    The #851/#854 outage made the supervisor exit immediately every ~5min
    for a "watchdog restart" while doing ZERO repo passes; state.json kept
    getting touched every beat even though no real loop pass ran, so
    `check_log_freshness` read healthy the entire time. That outage was
    only ~45 minutes -- shorter than charlie-work's own measured healthy
    worst-case gap between loop passes (53.9m), so `LOOP_PASS_STALE_MINUTES`
    is deliberately set to 90 and this check no longer catches that specific
    45-minute magnitude (PR #865 / issue #855 does, via consecutive
    zero-repo-pass cycles instead of elapsed time). What this test still
    pins down is the mechanism: a gap comfortably past the 90m threshold,
    with a fresh log, must still trip this check -- proving
    `check_loop_pass_freshness` is not itself fooled by the fresh-log
    artifact that fooled `check_log_freshness` during the real incident.
    """
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    (repo.state_dir / "state.json").write_text("{}", encoding="utf-8")
    _write_events_db(repo.state_dir, [(_iso(120), "loop_started")])

    log_report = hb.Report()
    hb.check_log_freshness(log_report, repo)
    assert not log_report.anomaly, "log freshness must read healthy, matching the real outage"

    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)
    assert report.anomaly
    assert "loop_started" in report.lines[0]


def test_check_loop_pass_freshness_ok_when_recent(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(5), "loop_started")])
    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)
    assert not report.anomaly
    assert "newest_loop_started=" in report.lines[0]


def test_check_loop_pass_freshness_ok_at_measured_healthy_worst_case(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Pins the false-alarm fix: 54m must NOT trip the 90m threshold.

    Measured production `loop_started` gaps (charlie-work, 39 intervals):
    max=53.9m. At the original LOOP_PASS_STALE_MINUTES=30 this magnitude of
    gap fired on a healthy fleet (~3-4 false alarms/day); at 90 it must not.
    If this threshold is ever lowered back toward 30-45 without addressing
    the repo-ordering cause described on LOOP_PASS_STALE_MINUTES, this test
    goes red before the false alarms return to production.
    """
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(54), "loop_started")])
    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)
    assert not report.anomaly, report.lines


def test_check_loop_pass_freshness_ok_when_db_missing(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)
    assert not report.anomaly
    assert "no events.db" in report.lines[0]


def test_check_loop_pass_freshness_ok_when_table_missing(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    db_path = repo.state_dir / "events.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE unrelated (id INTEGER)")
        conn.commit()
    finally:
        conn.close()

    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)
    assert not report.anomaly
    assert "no events table" in report.lines[0]


def test_check_loop_pass_freshness_ok_when_zero_loop_started_rows(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    # events table exists and has rows, but none of kind loop_started --
    # must be distinguishable from both "no table" and "stale".
    _write_events_db(repo.state_dir, [(_iso(1), "dispatch")])
    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)
    assert not report.anomaly
    assert "no loop_started rows" in report.lines[0]


def test_check_loop_pass_freshness_recent_iso_row_not_misjudged_stale(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Positive control for the ISO-vs-SQLite string-comparison trap.

    `ts` values are `...THH:MM:SSZ`. SQLite's `datetime('now','-90
    minutes')` returns a space-separated, non-`Z` string like
    `2026-07-31 22:25:04`. A predicate such as
    `WHERE ts < datetime('now','-90 minutes')` string-compares these, and
    `'T'` (0x54) sorting after `' '` (0x20) makes the comparison
    unreliable in either direction. This row is genuinely 2 minutes old and
    must read as fresh; if the SQL-based comparison is ever reintroduced in
    place of the Python-side `parse_iso` + timedelta comparison, this test
    must go red.
    """
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(2), "loop_started")])
    report = hb.Report()
    hb.check_loop_pass_freshness(report, repo)
    assert not report.anomaly, report.lines


def test_check_error_events_surfaces_seeded_self_deploy_alarm(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Regression for issue #866: error-level events are emitted, classified,
    documented, and tested -- and had NO consumer anywhere in the codebase.
    A human had to manually open events.db and know which `kind` to search
    for. `self_deploy_alarm` is a real production example of this (emitted
    from `supervise.py`, classified error-level by
    `instrumentation._classify_level` via `_ERROR_KINDS`).

    This test must fail with an AttributeError before `check_error_events`
    exists -- if it doesn't fail first, it isn't testing the gap.
    """
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(2), "self_deploy_alarm", "error")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_error_events(report, repo, baseline)
    assert report.anomaly
    assert "self_deploy_alarm" in report.lines[-1]


def test_check_error_events_ok_when_only_info_and_warning_events(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    _write_events_db(
        repo.state_dir,
        [
            (_iso(1), "dispatch_started", "info"),
            (_iso(1), "review_claim_stale", "warning"),
        ],
    )
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_error_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "error_rows=0" in report.lines[0]


def test_check_error_events_covers_synthetic_kind_not_hardcoded(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Coverage must be derived from the persisted `level` column, never a
    hardcoded list of `kind` strings in heartbeat_check.py (issue #866,
    acceptance criterion: coverage derived from `_ERROR_KINDS`, "or asserts
    the check has no literal kind list"). A `kind` that doesn't exist in
    `_ERROR_KINDS` today -- and never has -- must still be caught purely
    because its row was persisted with `level='error'`. This is also what
    makes PR #865's new `supervisor_zero_pass_alarm` kind land covered "for
    free": nothing in this check needs to change when a new alarm kind is
    added to `_ERROR_KINDS`, because it never enumerated kinds in the first
    place.
    """
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(1), "totally_novel_alarm_kind_xyz", "error")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_error_events(report, repo, baseline)
    assert report.anomaly
    assert "totally_novel_alarm_kind_xyz" in report.lines[-1]


def test_check_error_events_excludes_old_row_despite_sql_trap_shape(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Positive control for the ISO-vs-SQLite string-comparison trap, in the
    direction that matters for THIS check's query shape (`ts > baseline`,
    selecting NEW rows -- the opposite predicate from
    `check_loop_pass_freshness`'s `MAX(ts)`).

    `ts` values are `...THH:MM:SSZ`. If baseline were bound into a SQL
    predicate via a naive `str(datetime)` (space-separated, no `Z`, e.g.
    `2026-07-31 22:25:04+00:00`), a genuinely OLD row's `T`-formatted `ts`
    would still sort as "greater than" that space-formatted cutoff --
    `'T'` (0x54) sorts after `' '` (0x20) -- producing a false alarm on an
    old, already-seen event. This row is 60 minutes older than baseline and
    must be excluded; if SQL-based comparison is ever substituted for the
    Python-side `parse_iso` + `datetime` comparison, this test must go red.
    """
    repo = _make_repo(hb, tmp_path)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=5)
    _write_events_db(repo.state_dir, [(_iso(60), "self_deploy_alarm", "error")])
    report = hb.Report()
    hb.check_error_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "error_rows=1" in report.lines[0]
    assert "new_since_last_beat=0" in report.lines[0]


def test_check_error_events_excludes_row_older_than_cold_start_fallback(
    hb: ModuleType, tmp_path: Path
) -> None:
    """On a cold start (no prior heartbeat-state.json), `main()` falls
    `baseline` back to `now - LOG_FRESHNESS_STALE_MINUTES` (30m). An alarm
    older than that fallback window is deliberately out of scope on the
    very first run -- a bounded, intentional blind spot rather than an
    oversight. Pins that boundary: a 40m-old alarm against the 30m
    fallback baseline must NOT be reported.
    """
    repo = _make_repo(hb, tmp_path)
    now = datetime.now(timezone.utc)
    fallback_baseline = now - timedelta(minutes=hb.LOG_FRESHNESS_STALE_MINUTES)
    _write_events_db(repo.state_dir, [(_iso(40, base=now), "self_deploy_alarm", "error")])
    report = hb.Report()
    hb.check_error_events(report, repo, fallback_baseline)
    assert not report.anomaly, report.lines


def test_check_error_events_anomaly_when_db_missing(hb: ModuleType, tmp_path: Path) -> None:
    """Unlike `check_loop_pass_freshness` (missing db = OK, "no history
    yet"), a missing events.db here is an ANOMALY: this check's entire job
    is "did any alarm fire," and a registered repo with no events.db is a
    repo this check cannot vouch for -- reporting OK would be a silent
    false negative in exactly the direction issue #866 exists to close.
    """
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_error_events(report, repo, baseline)
    assert report.anomaly
    assert "no events.db" in report.lines[-1]


def test_check_error_events_anomaly_when_table_missing(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    db_path = repo.state_dir / "events.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE unrelated (id INTEGER)")
        conn.commit()
    finally:
        conn.close()

    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_error_events(report, repo, baseline)
    assert report.anomaly
    assert "no events table" in report.lines[-1]


def test_check_error_events_anomaly_when_db_unreadable(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    (repo.state_dir / "events.db").write_bytes(b"not a sqlite database at all")

    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_error_events(report, repo, baseline)
    assert report.anomaly


# ---------------------------------------------------------------------------
# check_warning_events (issue #946)
#
# Mirrors the check_error_events tests above one level down the `level`
# column, plus the one deliberate behavioral difference: a found warning
# must surface (via `report.warn`) without setting `report.anomaly`.
# ---------------------------------------------------------------------------


def test_check_warning_events_surfaces_seeded_dispatch_stale(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Regression for issue #946: warning-level events (dispatch_stale and
    ~6 pre-existing kinds) are emitted, classified, documented, and unit
    tested -- and had NO consumer anywhere in the codebase before this
    check. `dispatch_stale` is a real production example (emitted from
    `workflow.check_dispatch_staleness`, classified warning-level by
    `instrumentation._classify_level` via `_WARNING_KINDS`).

    This test must fail with an AttributeError before `check_warning_events`
    exists -- if it doesn't fail first, it isn't testing the gap.
    """
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(2), "dispatch_stale", "warning")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_warning_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "dispatch_stale" in report.lines[-1]
    assert report.lines[-1].startswith("WARN ")


def test_check_warning_events_ok_when_only_info_and_error_events(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    _write_events_db(
        repo.state_dir,
        [
            (_iso(1), "dispatch_started", "info"),
            (_iso(1), "self_deploy_alarm", "error"),
        ],
    )
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_warning_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "warning_rows=0" in report.lines[0]


def test_check_warning_events_covers_synthetic_kind_not_hardcoded(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Coverage must be derived from the persisted `level` column, never a
    hardcoded list of `kind` strings, matching
    `test_check_error_events_covers_synthetic_kind_not_hardcoded` above."""
    repo = _make_repo(hb, tmp_path)
    _write_events_db(repo.state_dir, [(_iso(1), "totally_novel_warning_kind_xyz", "warning")])
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_warning_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "totally_novel_warning_kind_xyz" in report.lines[-1]


def test_check_warning_events_excludes_old_row(hb: ModuleType, tmp_path: Path) -> None:
    """Positive control mirroring
    `test_check_error_events_excludes_old_row_despite_sql_trap_shape`: a row
    60 minutes older than baseline must be excluded from `new_since_last_beat`."""
    repo = _make_repo(hb, tmp_path)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=5)
    _write_events_db(repo.state_dir, [(_iso(60), "dispatch_stale", "warning")])
    report = hb.Report()
    hb.check_warning_events(report, repo, baseline)
    assert not report.anomaly, report.lines
    assert "warning_rows=1" in report.lines[0]
    assert "new_since_last_beat=0" in report.lines[0]


def test_check_warning_events_anomaly_when_db_missing(hb: ModuleType, tmp_path: Path) -> None:
    """Unlike a found warning (non-fatal), the check's OWN inability to read
    events.db is still a genuine anomaly, matching
    `test_check_error_events_anomaly_when_db_missing`: this check cannot
    vouch for the repo at all without a readable database."""
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    baseline = datetime.now(timezone.utc) - timedelta(minutes=10)
    report = hb.Report()
    hb.check_warning_events(report, repo, baseline)
    assert report.anomaly
    assert "no events.db" in report.lines[-1]


def test_report_warn_does_not_set_anomaly(hb: ModuleType) -> None:
    """Direct unit test of the `Report.warn` primitive itself: it must append
    a line but never flip `anomaly`, unlike `Report.anom`."""
    report = hb.Report()
    report.warn("some-check", "some non-fatal detail")
    assert report.anomaly is False
    assert report.lines == ["WARN some-check: some non-fatal detail"]


# ---------------------------------------------------------------------------
# check_stale_open_issue_mentions (issue #902)
#
# Real captured payload text, not paraphrased: PR #824's body starts with
# this exact prose (fetched via `gh pr view 824 --json body`, 2026-08-04).
# PR #824's branch (`fix/817-fleet-health-latch`) has no `agent/issue`
# prefix and its `closingIssuesReferences` came back `[]`, so this sentence
# was the only place the PR declared its target -- and it isn't a closing
# keyword, so `linked_issue_number` never binds on it either.
# ---------------------------------------------------------------------------

_REAL_PR824_BODY_EXCERPT = (
    "## Summary\n\nFor issue #817: `_filter_fleet_health_transitions` "
    "(`src/charlie_work/fleet_dispatch.py`) is a correct edge-detector for "
    "the fleet health digest's dedup baseline, but its producers only ever "
    "constructed `AttentionEntry` objects for *unhealthy* observations."
)


def test_mentioned_issue_numbers_extracts_817_from_real_pr824_body(hb: ModuleType) -> None:
    """Extraction stage, run against real (not synthetic) payload text.

    Proves the regex fires on GitHub's actual body shape, not just on data
    this test's author modeled after it -- the failure mode a purely
    synthetic fixture cannot rule out.
    """
    assert hb._mentioned_issue_numbers(_REAL_PR824_BODY_EXCERPT) == {817}


def test_branch_issue_number_extracts_817_from_real_pr824_branch(hb: ModuleType) -> None:
    assert hb._branch_issue_number("fix/817-fleet-health-latch") == 817


def test_branch_issue_number_ignores_agent_issue_prefixed_branch(hb: ModuleType) -> None:
    """No digit immediately after the slash -- already covered by the normal
    branch-prefix bind path, so a miss here is not a gap for this check."""
    assert (
        hb._branch_issue_number("agent/issue-414-feat-review-line-content-carry-forward") is None
    )


def test_mentioned_issue_numbers_suppresses_negated_reference(hb: ModuleType) -> None:
    """Issue #902 criterion 6 (negation half).

    The 32-char lookback window (matching `charlie_work.github`'s own
    documented tradeoff for the same negation guard) deliberately biases
    toward over-suppression: a match is only exempt from the negation check
    when no negation word appears anywhere in the preceding 32 characters,
    even across a clause boundary. That is a known, accepted cost -- a
    missed report here is safe (this check is advisory-only and re-scans
    every beat), not a correctness bug.
    """
    assert hb._mentioned_issue_numbers("This does not fix #817.") == set()
    # A reference with no negation word anywhere nearby is unaffected.
    assert hb._mentioned_issue_numbers("A completely unrelated change fixes #900.") == {900}


def test_mentioned_issue_numbers_suppresses_quoted_reference(hb: ModuleType) -> None:
    """Issue #902 criterion 6 (quoting half) -- #790's exact incident shape:
    a literal, quoted example inside prose must not be treated as live."""
    text = 'The bug looked the same as "Fixes #649" from before.'
    assert hb._mentioned_issue_numbers(text) == set()


def test_mentioned_issue_numbers_strips_fenced_code_blocks(hb: ModuleType) -> None:
    text = "See below:\n```\nraise ValueError('#404')\n```\nBut really see #817."
    assert hb._mentioned_issue_numbers(text) == {817}


def _init_test_git_repo(repo: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)


def _commit(repo: Path, filename: str, message: str) -> None:
    (repo / filename).write_text(filename, encoding="utf-8")
    subprocess.run(["git", "add", filename], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)


def test_get_merged_commit_messages_parses_real_local_history(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Builds a real (throwaway) git repo and reads it back with the real
    `git log` plumbing -- not a mocked subprocess -- for the parsing stage.

    Self-contained by construction: the repo is created fresh under
    `tmp_path` and never reads this checkout's own history, so unlike the
    #866-shaped test below, it has no dependency on CI's clone depth
    (verified 2026-08-04 after that test failed in CI on a shallow clone).
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_test_git_repo(repo)
    _commit(repo, "a.txt", "first commit\n\nrefs #123")
    _commit(repo, "b.txt", "second commit, unrelated")

    ok, commits, err = hb.get_merged_commit_messages(repo, limit=10)
    assert ok, err
    assert len(commits) == 2
    messages = [message for _, message in commits]
    assert any("refs #123" in message for message in messages)
    assert any("second commit, unrelated" in message for message in messages)


def test_get_merged_commit_messages_error_on_non_git_dir(hb: ModuleType, tmp_path: Path) -> None:
    ok, commits, err = hb.get_merged_commit_messages(tmp_path, limit=10)
    assert not ok
    assert commits == []
    assert err


# Real (not paraphrased) excerpt of PR #864's squash-merge commit body on
# `origin/main` (commit 740484f), captured verbatim via
# `git show -s --format=%B 740484f` (2026-08-04). #864 was filed and merged
# for a different issue (loop-pass staleness); this final bullet -- one of
# three sub-commits GitHub squashed into 740484f -- is the ONLY place
# anywhere that #866 is referenced. Neither "issue" nor a closing keyword
# precedes the reference, so only a bare `#N` scan over commit messages
# (never a PR title/body scan) can find it.
_REAL_PR864_SQUASH_COMMIT_EXCERPT = """\
* feat(heartbeat): surface error-level events with no consumer (refs #866)

self_deploy_alarm and every other member of instrumentation._ERROR_KINDS
(PR #865's supervisor_zero_pass_alarm included) are emitted, classified
error-level, documented, and unit-tested -- but nothing in the codebase
ever reads them back. A human had to manually open events.db and know
which kind string to search for.

check_error_events(report, repo, baseline) closes that gap by filtering
each repo's events.db on the level column persisted at write time by
_classify_level, so coverage is derived rather than a restated kind list
-- new alarm kinds are picked up with zero changes here. Missing/unreadable
events.db degrades to a reported ANOMALY (never an exception), which is
the deliberate point of disagreement with check_loop_pass_freshness's
"missing db = OK, no history yet" convention: this check's whole job is
"did any alarm fire," so an unreadable db is a repo it cannot vouch for.
Timestamps are compared in Python against baseline, never in SQL, per the
same ISO T/Z-vs-SQLite-space-format trap check_loop_pass_freshness already
guards against."""


def test_mentioned_issue_numbers_extracts_866_from_real_pr864_commit_excerpt(
    hb: ModuleType,
) -> None:
    """Extraction stage against the real PR #864 squash-commit text (see
    `_REAL_PR864_SQUASH_COMMIT_EXCERPT` above for provenance)."""
    assert 866 in hb._mentioned_issue_numbers(_REAL_PR864_SQUASH_COMMIT_EXCERPT)


def test_get_merged_commit_messages_extracts_866_from_real_squash_commit_shape(
    hb: ModuleType, tmp_path: Path
) -> None:
    """Issue #902's #866 reproduction, through the REAL `git log` pipeline,
    against a throwaway repo this test owns.

    An earlier version of this test asserted against THIS checkout's own
    git history (commit 740484f, 19 commits back from `origin/main` HEAD at
    filing time) and passed locally. It failed in CI: `.github/workflows/ci.yml`'s
    `actions/checkout@v4` step sets no explicit `fetch-depth`, which defaults
    to a depth-1 (single-commit) shallow clone there, so commit 740484f was
    never fetched and unreachable from `git log` on the runner -- confirmed
    by grepping the workflow file for `fetch-depth` (absent) rather than
    assumed. Seeding an owned throwaway repo with the real commit-message
    text removes the ambient-history dependency while still exercising the
    real `git log` subprocess call end-to-end and the real payload shape --
    the actual failure mode this check exists to catch (#866's fix is only
    traceable via a commit message, never a PR title/body) requires running
    the true plumbing, not mocking it.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_test_git_repo(repo)
    _commit(repo, "unrelated.txt", "unrelated setup commit, PR #1")
    _commit(repo, "heartbeat.py", _REAL_PR864_SQUASH_COMMIT_EXCERPT)

    ok, commits, err = hb.get_merged_commit_messages(repo, limit=10)
    assert ok, err
    matches = [sha for sha, message in commits if 866 in hb._mentioned_issue_numbers(message)]
    assert matches, "expected the seeded commit to reference #866"


def _stale_mention_gh_dispatch(
    monkeypatch: Any,
    hb: ModuleType,
    *,
    open_numbers: list[int],
    merged_prs: list[dict[str, Any]],
    captured: list[list[str]] | None = None,
) -> None:
    def handler(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
        if captured is not None:
            captured.append(list(args))
        if args[:2] == ["issue", "list"]:
            return True, [{"number": n} for n in open_numbers], ""
        if args[:2] == ["pr", "list"]:
            return True, merged_prs, ""
        raise AssertionError(f"unexpected gh call in check_stale_open_issue_mentions: {args}")

    _gh_dispatch(monkeypatch, hb, handler)


def test_check_stale_open_issue_mentions_catches_817_824_reproduction(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 1: the #817/#824 reproduction, verbatim shape."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[817],
        merged_prs=[
            {
                "number": 824,
                "headRefName": "fix/817-fleet-health-latch",
                "title": "fix: feed recovery observations into fleet health digest",
                "body": _REAL_PR824_BODY_EXCERPT,
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-31T17:51:40Z",
            }
        ],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly
    assert "#817" in report.lines[-1]
    assert "PR #824" in report.lines[-1]


def test_check_stale_open_issue_mentions_catches_866_864_reproduction_via_commit(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 2: #866's fix rode inside PR #864, a PR for a
    different issue, with no reference anywhere in #864's own title/body --
    only in one of its commit messages."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[866],
        merged_prs=[
            {
                "number": 864,
                "headRefName": "fix/heartbeat-loop-pass-staleness",
                "title": "feat(heartbeat): detect fleet loop-pass stall that log freshness cannot see",
                "body": "Adds check_loop_pass_freshness. No mention of any other issue here.",
                "closingIssuesReferences": [],
                "mergedAt": "2026-08-01T02:13:00Z",
            }
        ],
    )
    monkeypatch.setattr(
        hb,
        "get_merged_commit_messages",
        lambda root, limit: (
            True,
            [
                ("740484f", "feat(heartbeat): loop-pass stall (#864)"),
                (
                    "e93fe13",
                    "feat(heartbeat): surface error-level events with no consumer (refs #866)",
                ),
            ],
            "",
        ),
    )

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly
    assert "#866" in report.lines[-1]
    assert "commit e93fe13" in report.lines[-1]


def test_check_stale_open_issue_mentions_no_label_filter_or_state_json(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 3 (corrected framing, 2026-08-04): an issue with
    NO LABELS AT ALL must still be reported. Enforced structurally: the
    candidate-set query must carry no `--label` filter, and the check must
    never read `state.json` -- both #817 and #866 have zero labels, and
    `workflow.py`'s `_merged_pr_referenced_issue_numbers` deliberately only
    ever considers the `ready`-labelled set, by design (see that function's
    docstring). This check's whole reason to exist is to not share that
    constraint.
    """
    repo = _make_repo(hb, tmp_path)
    captured: list[list[str]] = []
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[817],
        merged_prs=[],
        captured=captured,
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    def _fail_if_called() -> None:
        raise AssertionError("check_stale_open_issue_mentions must never read state.json")

    monkeypatch.setattr(hb, "load_state", _fail_if_called)

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)  # would raise if load_state were called

    issue_list_calls = [args for args in captured if args[:2] == ["issue", "list"]]
    assert issue_list_calls, "expected an `issue list` call"
    assert "--label" not in issue_list_calls[0]
    assert "automated-ready" not in issue_list_calls[0]


def test_check_stale_open_issue_mentions_ok_when_issue_closed(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 4 (closed half): a closed issue never appears in
    the open-issue candidate set, so a PR mentioning it produces no report
    even though the text-matching machinery would otherwise fire."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[],  # 817 is closed -- absent from the open-issue query
        merged_prs=[
            {
                "number": 824,
                "headRefName": "fix/817-fleet-health-latch",
                "title": "t",
                "body": _REAL_PR824_BODY_EXCERPT,
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-31T17:51:40Z",
            }
        ],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert not report.anomaly
    assert "stale_mentions=0" in report.lines[-1]


def test_check_stale_open_issue_mentions_ok_when_no_reference(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 4 (unreferenced half): an open issue with no
    merged-PR or commit reference anywhere produces no report."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[900],
        merged_prs=[
            {
                "number": 1,
                "headRefName": "fix/unrelated-cleanup",
                "title": "unrelated",
                "body": "nothing to see here",
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-01T00:00:00Z",
            }
        ],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert not report.anomaly


def test_check_stale_open_issue_mentions_never_issues_mutating_gh_calls(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 5: never auto-closes an issue or mutates a
    label. Enforced structurally -- every captured `gh` invocation this
    check makes must be a `list` (read) subcommand."""
    repo = _make_repo(hb, tmp_path)
    captured: list[list[str]] = []
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[817],
        merged_prs=[
            {
                "number": 824,
                "headRefName": "fix/817-fleet-health-latch",
                "title": "t",
                "body": _REAL_PR824_BODY_EXCERPT,
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-31T17:51:40Z",
            }
        ],
        captured=captured,
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly  # sanity: a real finding did occur
    assert captured, "expected at least one gh call"
    for args in captured:
        assert args[1] == "list", f"non-read gh subcommand invoked: {args}"


def test_check_stale_open_issue_mentions_negated_reference_not_reported(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 6 (negation)."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[817],
        merged_prs=[
            {
                "number": 900,
                "headRefName": "fix/900-something-else",
                "title": "t",
                "body": "This change does not fix #817, it is unrelated.",
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-31T17:51:40Z",
            }
        ],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert not report.anomaly


def test_check_stale_open_issue_mentions_quoted_reference_not_reported(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #902 criterion 6 (quoting) -- #790's exact incident shape."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[649],
        merged_prs=[
            {
                "number": 788,
                "headRefName": "fix/negated-phrase-bug",
                "title": "t",
                "body": 'Demonstrates the negated-phrase bug: the same as "Fixes #649" from before.',
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-31T17:51:40Z",
            }
        ],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert not report.anomaly


def test_check_stale_open_issue_mentions_output_bounded(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Guard against unbounded output: many matches are capped and summarized."""
    repo = _make_repo(hb, tmp_path)
    numbers = list(range(1, 26))  # 25 distinct matches, cap is 20
    body = " ".join(f"#{n}" for n in numbers)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=numbers,
        merged_prs=[
            {
                "number": 1,
                "headRefName": "fix/many-refs",
                "title": "t",
                "body": body,
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-31T17:51:40Z",
            }
        ],
    )
    monkeypatch.setattr(hb, "get_merged_commit_messages", lambda root, limit: (True, [], ""))

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly
    line = report.lines[-1]
    detail = line.split("closure path: ", 1)[1].split(" (open=", 1)[0]
    parts = detail.split("; ")
    # 20 shown findings plus one "+N more" summary entry -- never all 25.
    assert len(parts) == hb.STALE_MENTION_REPORT_CAP + 1
    assert parts[-1] == "+5 more"


def test_check_stale_open_issue_mentions_anomaly_on_open_issue_list_failure(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)

    def handler(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
        return False, None, "gh exploded"

    _gh_dispatch(monkeypatch, hb, handler)
    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)
    assert report.anomaly
    assert "gh exploded" in report.lines[-1]


def test_check_stale_open_issue_mentions_anomaly_on_merged_pr_list_failure(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)

    def handler(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
        if args[:2] == ["issue", "list"]:
            return True, [], ""
        return False, None, "merged pr list exploded"

    _gh_dispatch(monkeypatch, hb, handler)
    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)
    assert report.anomaly
    assert "merged pr list exploded" in report.lines[-1]


def test_check_stale_open_issue_mentions_degrades_gracefully_when_git_log_fails(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A local `git log` failure is noted, not fatal -- the gh-sourced
    (branch/title/body) findings still get reported."""
    repo = _make_repo(hb, tmp_path)
    _stale_mention_gh_dispatch(
        monkeypatch,
        hb,
        open_numbers=[817],
        merged_prs=[
            {
                "number": 824,
                "headRefName": "fix/817-fleet-health-latch",
                "title": "t",
                "body": _REAL_PR824_BODY_EXCERPT,
                "closingIssuesReferences": [],
                "mergedAt": "2026-07-31T17:51:40Z",
            }
        ],
    )
    monkeypatch.setattr(
        hb, "get_merged_commit_messages", lambda root, limit: (False, [], "git log failed")
    )

    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)

    assert report.anomaly
    assert "#817" in report.lines[-1]
    assert "commit-message scan degraded" in report.lines[-1]
