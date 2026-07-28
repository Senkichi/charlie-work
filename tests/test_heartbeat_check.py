from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _load_heartbeat_check() -> ModuleType:
    """Load scripts/heartbeat_check.py as a module without adding scripts to sys.path."""
    path = Path(__file__).parent.parent / "scripts" / "heartbeat_check.py"
    spec = importlib.util.spec_from_file_location("heartbeat_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["heartbeat_check"] = module
    spec.loader.exec_module(module)
    return module


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


def _iso(minutes_ago: float = 0.0) -> str:
    """Return an ISO-8601 UTC timestamp relative to right now."""
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_repo(hb: ModuleType, tmp_path: Path) -> Any:
    return hb.RepoInfo(
        slug="owner/repo",
        repo_root=tmp_path,
        state_dir=tmp_path / "state",
        config_path=tmp_path / "orchestrator.config.yaml",
    )


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
    repo = _make_repo(hb, tmp_path)
    _patch_gh(monkeypatch, hb, [100])
    _make_pr_dirs(repo.state_dir, 100)
    _write_state(
        repo.state_dir,
        100,
        {
            "review_dispatch_status": "review_dispatch_pending",
            "review_dispatch_pending_at": _iso(5),
        },
    )

    report = hb.Report()
    hb.check_review_liveness(report, repo)

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
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    # _iso(-10) = 10 minutes in the future (throttle still active)
    (repo.state_dir / "state.json").write_text(
        json.dumps({"throttled_until": _iso(-10)}), encoding="utf-8"
    )
    report = hb.Report()
    hb.check_dispatch_throttle(report, repo)
    assert not report.anomaly
    assert "throttled until" in report.lines[0]


def test_check_dispatch_throttle_anomaly_when_exceeds_threshold(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.state_dir.mkdir(parents=True, exist_ok=True)
    # _iso(-60) = 60 minutes in the future, beyond the 30-min threshold
    (repo.state_dir / "state.json").write_text(
        json.dumps({"throttled_until": _iso(-60)}), encoding="utf-8"
    )
    report = hb.Report()
    hb.check_dispatch_throttle(report, repo)
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
) -> None:
    payload = {
        "pid": pid,
        "started_at": "2026-07-25T17:00:00Z",
        "last_beat_at": last_beat_at,
        "pass_number": 5,
        "full_pass_interval_seconds": full_pass_interval_seconds,
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


def test_check_supervisor_heartbeat_threshold_derives_from_full_pass_interval(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A longer full_pass_interval raises the stale threshold."""
    fleet_dir = _set_fleet_dir(hb, monkeypatch, tmp_path)
    now = datetime.now(timezone.utc)
    # 20 minutes old. With full_pass=600s, threshold = 2*600/60 = 20 min, so OK.
    last_beat = (now - timedelta(minutes=19)).isoformat().replace("+00:00", "Z")
    _write_heartbeat(hb, fleet_dir, last_beat_at=last_beat, full_pass_interval_seconds=600)
    report = hb.Report()
    hb.check_supervisor_heartbeat(report)
    assert not report.anomaly
