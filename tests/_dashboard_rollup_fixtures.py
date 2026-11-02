"""Fixtures for the dashboard rollup tests (real writers; see test_dashboard_rollup).

Fixtures are built through ``touch_repo`` (registry) and ``instrumentation.log_event`` /
``record_loop_pass`` (events DBs). Payloads are copied from the events recon examples;
the event clock is pinned per call.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from charlie_work import instrumentation
from charlie_work.dashboard import rollup, sources
from charlie_work.dashboard.rollup_schema import SOURCE_SCOPED_TABLES
from charlie_work.fleet_registry import touch_repo
from charlie_work.github import GitHub
from charlie_work.paths import runtime_paths

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
ALPHA, BETA = "owner/alpha", "local/beta"

GOV = {
    "available_slots": 4,
    "clamped": False,
    "concurrency_limit": 5,
    "dispatch_limit": 1,
    "fleet_concurrency_limit": 4,
    "fleet_live_session_count": 3,
    "live_session_count": 1,
}
BACKLOG = {
    "open_total": 47,
    "dispatchable": 5,
    "active_label": 3,
    "missing_ready": 18,
    "parked_unready": 7,
    "terminal_label": 15,
    "blocked_by_open_dependency": 0,
    "operator_claimed": 0,
}
RUNNER_ALLOC = {
    "budget": 8,
    "targets": [
        {
            "capacity": 1,
            "demand": 0,
            "oldest_queued_seconds": 0,
            "repo": "Senkichi/fresh-eyes",
            "running": 1,
            "target": 1,
        },
        {
            "capacity": 5,
            "demand": 2,
            "oldest_queued_seconds": 30,
            "repo": "Senkichi/swole",
            "running": 2,
            "target": 3,
        },
    ],
}


def _job(job_id: str, status: str, qw: float | None) -> dict:
    measured = {"kind": "measured", "seconds": qw, "source": "github"}
    unmeasured = {"kind": "unmeasured", "reason": "job has not finished"}
    done = status == "completed"
    return {
        "job_id": job_id,
        "name": "Tests",
        "status": status,
        "durations": {
            "queue_wait": {"kind": "measured", "seconds": 2.0, "source": "github"},
            "execution": measured if done else unmeasured,
            "wall": {"kind": "measured", "seconds": 40.0} if done else unmeasured,
        },
    }


def _register(fleet_dir: Path, repo_root: Path, name: str) -> Path:
    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return name

    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")
    touch_repo(str(fleet_dir), repo_root, paths, FakeGitHub(repo_root=repo_root))
    return paths.root


class Fleet:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.conns: list[sqlite3.Connection] = []
        self.dir = tmp_path / "fleet"
        self.alpha = _register(self.dir, tmp_path / "alpha", ALPHA)
        self.beta = _register(self.dir, tmp_path / "beta", BETA)
        self.fleet_state = self.dir / sources.HEARTBEAT_FILENAME

    def emit(self, state: Path, ts: str, kind: str, payload: dict) -> None:
        self.monkeypatch.setattr(instrumentation, "_now_iso", lambda: ts)
        # state dirs hold state.json; the fleet dir is addressed via the heartbeat file
        path = state if state == self.fleet_state else state / "state.json"
        instrumentation.log_event(path, kind, payload, repo="WRONG/column")

    def close(self) -> None:
        self.release()
        for p in (self.alpha / "state.json", self.beta / "state.json", self.fleet_state):
            instrumentation.close_db(p)

    def sources(self) -> rollup.RollupSources:
        return rollup.rollup_sources(str(self.dir))

    def db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sources().db_path)
        self.conns.append(conn)
        return conn

    def release(self) -> None:
        """Close every dashboard.db connection (Windows cannot unlink an open file)."""
        for conn in self.conns:
            conn.close()
        self.conns.clear()


@pytest.fixture
def fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    f = Fleet(tmp_path, monkeypatch)
    a, b, g = f.alpha, f.beta, f.fleet_state
    e = f.emit
    e(
        a,
        "2026-10-01T08:00:00Z",
        "dispatch",
        {
            "concurrency_governor": GOV,
            "backlog_reachability": BACKLOG,
            "issue_numbers": [2226, 2227],
            "deferred_by_concurrency_count": 4,
        },
    )
    e(a, "2026-10-01T08:01:00Z", "dispatch_rework", {"issue_numbers": [2195], "pr_number": 2197})
    e(
        a,
        "2026-10-01T08:02:00Z",
        "worker_handoff_pr_opened",
        {"issue_number": 2185, "pr_number": 2188, "reason": "worker_handoff_clean_exit"},
    )
    e(
        a,
        "2026-10-01T08:03:00Z",
        "orphaned_worker_opened_pr",
        {"issue_number": 1939, "pr_number": 1940},
    )
    e(a, "2026-10-01T08:04:00Z", "review_dispatch_claim", {"count": 2, "pr_numbers": [2214, 2215]})
    e(
        a,
        "2026-10-01T08:05:00Z",
        "review_dispatch",
        {
            "failed": [{"pr": 1}],
            "fleet_available_review_slots": 6,
            "fleet_live_review_count": 0,
            "fleet_review_concurrency_limit": 6,
            "launched": [2214, 2215],
            "quota_hit": False,
        },
    )
    e(
        a,
        "2026-10-01T08:06:00Z",
        "record_review",
        {"decision": "approved", "escalated": False, "issue_number": 2199, "pr_number": 2208},
    )
    e(
        a,
        "2026-10-01T08:07:00Z",
        "record_review",
        {
            "decision": "request_changes",
            "escalated": True,
            "issue_number": 2200,
            "pr_number": 2209,
        },
    )
    e(
        a,
        "2026-10-01T08:08:00Z",
        "reconcile",
        {"kind": "merged_outside_orchestrator", "issue_number": 2199, "pr_number": 2208},
    )
    # noise: terminal_state_stale repeats, unauthorized sync is excluded outright
    e(a, "2026-10-01T08:09:00Z", "reconcile", {"kind": "terminal_state_stale", "issue_number": 5})
    e(a, "2026-10-01T08:09:01Z", "unauthorized_merge_queue_sync_covered", {"pr_number": 7})
    e(a, "2026-10-01T08:09:02Z", "unauthorized_merge_queue_sync_covered", {"pr_number": 8})
    e(
        a,
        "2026-10-01T08:10:00Z",
        "session_exited",
        {"failure_kind": "stalled", "issue_number": 2195, "worker_health": "DEAD"},
    )
    e(
        a,
        "2026-10-01T08:11:00Z",
        "session_failed_escalated",
        {"issue_number": 2060, "reason": "no_op_rework_cap_exceeded"},
    )
    e(
        a,
        "2026-10-01T08:12:00Z",
        "unescalate",
        {"issue_number": 1808, "cleared_escalation_reason": "dead_dispatched_worker_reap"},
    )
    e(
        a,
        "2026-10-01T08:13:00Z",
        "review_verdict_missed",
        {
            "cause": {"exit_code": 0},
            "issue_number": 2081,
            "pr_number": 2087,
            "reason": "launch_failed",
            "turn_count": 0,
            "tool_call_count": 0,
        },
    )
    e(
        a,
        "2026-10-01T08:14:00Z",
        "review_verdict_missed",
        {"issue_number": 2082, "pr_number": 2088, "reason": "PR #2088 is MERGED on GitHub"},
    )
    e(
        a,
        "2026-10-01T08:15:00Z",
        "review_verdict_missed",
        {"issue_number": 2083, "pr_number": 2089, "reason": "PR #2089 is MERGED on GitHub"},
    )
    # Issue #2476 shape: a stable token reason, the free-text message in
    # detail, and a cause object whose api_error_status wins the cause label.
    e(
        a,
        "2026-10-01T08:15:30Z",
        "review_verdict_missed",
        {
            "cause": {"cause": "died_mid_session", "api_error_status": 429, "exit_code": 1},
            "detail": "reviewer exited before writing a verdict (API error 429)",
            "issue_number": 2084,
            "pr_number": 2090,
            "reason": "died_mid_session",
        },
    )
    e(
        a,
        "2026-10-01T08:16:00Z",
        "self_deploy_succeeded",
        {"changed": True, "error": None, "from_sha": "32e8", "to_sha": "d6c3", "ok": True},
    )
    e(
        a,
        "2026-10-01T08:17:00Z",
        "review_quota_exhausted",
        {"throttled_until": "2026-10-01T13:00:00Z", "source": "stalled_review_sweep"},
    )
    e(
        a,
        "2026-10-01T08:18:00Z",
        "dispatch_backpressure",
        {"clamped_by": "host_load", "clamped_limit": 0, "requested_limit": 3},
    )
    # per-repo copies of global-authoritative kinds, AFTER the global DB first wrote them: skipped
    e(a, "2026-10-01T10:05:00Z", "runner_allocation", RUNNER_ALLOC)
    e(
        a,
        "2026-10-01T10:06:00Z",
        "fleet_job_observations",
        {"jobs": [_job("j1", "completed", 2.0)]},
    )
    e(
        b,
        "2026-10-01T09:00:00Z",
        "dispatch",
        {"concurrency_governor": GOV, "backlog_reachability": BACKLOG, "issue_numbers": []},
    )
    e(
        b,
        "2026-10-01T09:01:00Z",
        "session_exited",
        {"failure_kind": None, "issue_number": 77, "worker_health": "DEAD"},
    )
    e(g, "2026-10-01T10:00:00Z", "runner_allocation", RUNNER_ALLOC)
    e(
        g,
        "2026-10-01T10:01:00Z",
        "fleet_job_observations",
        {"jobs": [_job("j1", "in_progress", None), _job("j2", "completed", 2.0)]},
    )
    e(
        g,
        "2026-10-01T10:02:00Z",
        "fleet_job_observations",
        {"jobs": [_job("j1", "completed", 2.0)]},
    )
    e(
        g,
        "2026-10-01T10:03:00Z",
        "runner_capacity_starved",
        {"capacity": 5, "demand": 7, "running": 5},
    )
    e(g, "2026-10-01T10:04:00Z", "supervisor_started", {})  # unhandled kind: coverage only
    instrumentation.record_loop_pass(a / "state.json", "ed62ee91e2e5", "2026-10-01T08:00:00Z")
    instrumentation.record_loop_pass(
        a / "state.json",
        "ed62ee91e2e5",
        "2026-10-01T08:00:00Z",
        "2026-10-01T08:01:42Z",
        ok=True,
        elapsed_seconds=102.17,
        merge_count=2,
        review_count=3,
        sink_population=9,
    )
    yield f
    f.close()


def _all(db: sqlite3.Connection, sql: str, *args) -> list[tuple]:
    return db.execute(sql, args).fetchall()


def _facts(db: sqlite3.Connection) -> dict[str, list[tuple]]:
    return {t: _all(db, f"SELECT * FROM {t} ORDER BY 1, 2, 3") for t in SOURCE_SCOPED_TABLES}
