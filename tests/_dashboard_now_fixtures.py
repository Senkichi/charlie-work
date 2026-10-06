"""Fixtures for the dashboard Now-model tests (real writers; see test_dashboard_now_model).

Fixtures go through the real writers: ``touch_repo`` (registry),
``status_snapshot.write_status_snapshot`` (snapshot envelope; the clock stamp is
pinned) and ``instrumentation.log_event`` (global ``runner_allocation`` event).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from charlie_work import instrumentation, status_snapshot
from charlie_work.config import LabelConfig
from charlie_work.dashboard import sources
from charlie_work.dashboard.now_types import RepoRead, SourcesRead
from charlie_work.fleet_registry import touch_repo
from charlie_work.github import GitHub
from charlie_work.paths import runtime_paths
from charlie_work.command_result import CommandResult

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
L = LabelConfig()


@dataclass(frozen=True)
class Finding:
    check: str
    repo: str
    severity: str
    detail: str
    facts: dict[str, Any] = field(default_factory=dict)


def _register(fleet_dir: Path, repo_root: Path, name: str) -> Path:
    class FakeGitHub(GitHub):
        def name_with_owner(self) -> str:
            return name

    repo_root.mkdir(parents=True, exist_ok=True)
    paths = runtime_paths(repo_root, ".var/charlie-work")
    touch_repo(str(fleet_dir), repo_root, paths, FakeGitHub(repo_root=repo_root))
    return paths.root


def _write_snapshot(monkeypatch, state_dir: Path, written_at: str, data: dict) -> None:
    """Write via the real ``write_status_snapshot`` with a stub app and a pinned stamp."""
    app = SimpleNamespace(
        paths=SimpleNamespace(root=state_dir),
        repo_root=state_dir,
        status=lambda use_cache=False: CommandResult(True, "ok", data),
    )
    monkeypatch.setattr(status_snapshot, "utc_now", lambda: written_at)
    status_snapshot.write_status_snapshot(app)


def _issue(number: int, *labels: str, dispatchable: bool = False) -> dict:
    return {
        "number": number,
        "title": f"t{number}",
        "url": f"u{number}",
        "labels": [L.ready, *labels],
        "dispatchable": dispatchable,
        "dependencies": {"declared": [], "open": []},
    }


def _worker(repo: str, issue: int) -> dict:
    return {"repo": repo, "issue": issue, "adapter": "devin", "health": "healthy"}


ALPHA = {
    "ready_issue_count": 7,
    "available_issue_count": 2,
    "active_issue_count": 3,
    "open_linked_pr_count": 2,
    "unlinked_pr_count": 1,
    "issues": [
        _issue(1, L.queued),
        _issue(2, L.in_progress),
        _issue(3, L.pr_open),
        _issue(4, L.operator_queue),
        _issue(5, L.human_needed),
        _issue(6, dispatchable=True),
        _issue(7, dispatchable=True),
    ],
    "prs": [
        {"number": 30, "issue_number": 3, "is_draft": False, "reviewDecision": ""},
        {"number": 50, "issue_number": 5, "is_draft": False, "reviewDecision": ""},
    ],
    "unlinked_prs": [{"pr_number": 77}],
    "workers": [_worker("owner/alpha", 2), _worker("owner/alpha", 1)],
    "backlog_reachability": {
        "observed": True,
        "dispatchable": 2,
        "missing_ready": 1,
        # Issue #2314: parked_unready is not Ready either, so it must stay out
        # of the "Ready but not dispatchable" breakdown alongside missing_ready.
        "parked_unready": 3,
        "terminal_label": 2,
        "active_label": 3,
        "operator_claimed": 1,
        "mention_covered_awaiting_operator": 0,
        "blocked_by_open_dependency": 0,
        "unidentified": 0,
        "unreachable_examples": {"terminal_label": [4, 5], "operator_claimed": [9]},
    },
}
BETA = {
    "ready_issue_count": 2,
    "active_issue_count": 2,
    "open_linked_pr_count": 1,
    "unlinked_pr_count": 0,
    "issues": [_issue(11, L.reviewing), _issue(12, L.needs_rework)],
    "prs": [{"number": 60, "issue_number": 11, "is_draft": False, "reviewDecision": ""}],
    "unlinked_prs": [],
    "workers": [_worker("owner/beta", 11)],
    # observed false = unknown, so Dispatchable falls back to the per-issue flags.
    "backlog_reachability": {"observed": False, "dispatchable": 0, "terminal_label": 0},
}
ALLOCATION = {
    "targets": [
        {"repo": "owner/beta", "capacity": 1, "demand": 0, "running": 1, "target": 1},
        {"repo": "owner/alpha", "capacity": 3, "demand": 5, "running": 2, "target": 2},
    ]
}


@pytest.fixture
def fleet(tmp_path: Path, monkeypatch):
    fleet_dir = tmp_path / "fleet"
    state_a = _register(fleet_dir, tmp_path / "alpha", "owner/alpha")
    state_b = _register(fleet_dir, tmp_path / "beta", "owner/beta")
    _write_snapshot(monkeypatch, state_a, "2026-10-01T11:57:30Z", ALPHA)
    _write_snapshot(monkeypatch, state_b, "2026-10-01T11:00:00Z", BETA)
    instrumentation.log_event(
        sources.fleet_sources(str(fleet_dir)).supervisor_heartbeat,
        "runner_allocation",
        ALLOCATION,
    )
    yield fleet_dir
    for state in (state_a, state_b, fleet_dir):
        instrumentation.close_db(state / "state.json")
    instrumentation.close_db(fleet_dir / sources.HEARTBEAT_FILENAME)


def _read(fleet_dir: Path, now: datetime = NOW, **overrides: Any) -> SourcesRead:
    fs = sources.fleet_sources(str(fleet_dir))
    conn, err = sources.open_events_ro(fs.events_db)
    assert err is None and conn is not None
    event = sources.latest_event(conn, "runner_allocation")
    conn.close()
    assert event is not None
    # The real writer stamps wall-clock; pin the stamp relative to ``now``.
    event = {**event, "ts": (now - timedelta(seconds=120)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    since = {"owner/alpha": ((4, now - timedelta(hours=1)), (5, now - timedelta(hours=2)))}
    extra = {
        "owner/alpha": {"reviewers_live": 1, "worker_cap": 2, "review_cap": 4},
        "owner/beta": {"reviewers_live": 0, "worker_cap": 0, "review_cap": 4},
    }
    repos = tuple(
        RepoRead(
            key=r.key,
            repo_root=str(r.repo_root),
            snapshot=sources.read_snapshot(r, now),
            escalated_since=since.get(r.key, ()),
            **extra[r.key],
        )
        for r in sources.enumerate_repos(str(fleet_dir))
    )
    base: dict[str, Any] = {
        "repos": repos,
        "global_worker_cap": 3,
        "global_review_cap": 6,
        "runner_allocation": event,
        "runner_busy": {"owner/alpha": 1},
        "done_24h": 9,
    }
    return SourcesRead(**{**base, **overrides})


def _root(fleet_dir: Path, key: str) -> str:
    return next(str(r.repo_root) for r in sources.enumerate_repos(str(fleet_dir)) if r.key == key)
