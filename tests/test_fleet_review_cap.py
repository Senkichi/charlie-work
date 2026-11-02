"""Issue #2084: fleet-wide reviewer concurrency cap (``fleet.global_max_concurrent_reviews``).

The review lane's twin of the worker cap (``fleet.global_max_concurrent_sessions``,
``tests/test_charlie_work_fleet_governor.py`` / ``tests/test_fleet_launch_lock_scope.py``).
Covered: config parse/validation, ``count_fleet_live_reviews``, the cross-repo
clamp in ``dispatch_reviews`` (live reviewers in repo A limit dispatch in repo
B), cap ``0`` = unlimited (and never reads the fleet), the fleet lock held ->
``fleet_lock_held`` deferral + ``dispatch_deferred`` / ``dispatch_starved``
events, the per-pass knob read, the dry-run preview, and the no-remote local
review lane.

Fixture helpers are reinlined per file -- the zero cross-test-import guard
forbids ``from test_* import``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from _local_lane_fixtures import (
    _git,
    _init_repo,
    _local_config,
    _make_branch,
    _parked_issue,
)
from _review_fixtures import _fake_claude_worker_record, _write_review_packet
from charlie_work import layout
from charlie_work.config import (
    ConfigError,
    FleetConfig,
    OrchestratorConfig,
    ReviewDispatchConfig,
    build_config_from_data,
)
from charlie_work.dispatch_deferral import DISPATCH_DEFERRED_KIND, DISPATCH_STARVED_KIND
from charlie_work.fleet_registry import count_fleet_live_reviews, try_acquire_fleet_lock
from charlie_work.instrumentation import query_events
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state_locked
from charlie_work.workflow import OrchestratorApp

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def test_config_default_is_disabled() -> None:
    assert FleetConfig().global_max_concurrent_reviews == 0
    assert build_config_from_data({}).fleet.global_max_concurrent_reviews == 0


def test_config_parses_valid_value() -> None:
    config = build_config_from_data({"fleet": {"global_max_concurrent_reviews": 4}})
    assert config.fleet.global_max_concurrent_reviews == 4
    # The worker cap is an independent budget.
    assert config.fleet.global_max_concurrent_sessions == 0


def test_config_accepts_explicit_zero() -> None:
    config = build_config_from_data({"fleet": {"global_max_concurrent_reviews": 0}})
    assert config.fleet.global_max_concurrent_reviews == 0


def test_config_rejects_negative() -> None:
    with pytest.raises(
        ConfigError, match=r"^fleet\.global_max_concurrent_reviews: expected >= 0, got -1$"
    ):
        build_config_from_data({"fleet": {"global_max_concurrent_reviews": -1}})


@pytest.mark.parametrize("bad", ["3", 1.5, True, [2]])
def test_config_rejects_non_int(bad: Any) -> None:
    with pytest.raises(
        ConfigError, match=r"^fleet\.global_max_concurrent_reviews: expected int, got "
    ):
        build_config_from_data({"fleet": {"global_max_concurrent_reviews": bad}})


# --------------------------------------------------------------------------
# count_fleet_live_reviews
# --------------------------------------------------------------------------

_OWN_PID = os.getpid()


def _register_repo(fleet_dir: Path, tmp_path: Path, name: str, *, live_reviewers: int = 0) -> Path:
    """Register ``owner/<name>`` in the fleet registry; seed live "ghost" reviewers.

    A ghost is a ``review_dispatch_dispatched`` state record whose
    ``reviewer_pid`` is alive with no sidecar -- the same state.json
    corroboration the live-review counter applies, here pointed at this test
    process's own pid so it is deterministically alive.
    """
    repo_root = tmp_path / name
    (repo_root / ".git").mkdir(parents=True)
    state_dir = repo_root / ".var" / "charlie-work"
    reviews_dir = layout.reviews_dir_default(state_dir)
    reviews_dir.mkdir(parents=True)
    prs = {
        str(900 + i): {
            "number": 900 + i,
            "review_dispatch_status": "review_dispatch_dispatched",
            "reviewer_pid": _OWN_PID,
            "reviewer_process_start_time": None,
        }
        for i in range(live_reviewers)
    }
    (state_dir / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": prs, "events": []}), encoding="utf-8"
    )
    fleet_json = fleet_dir / "fleet.json"
    registry = (
        json.loads(fleet_json.read_text(encoding="utf-8"))
        if fleet_json.exists()
        else {"version": 1, "repos": {}}
    )
    registry["repos"][f"owner/{name}"] = {
        "repo_root": str(repo_root),
        "name_with_owner": f"owner/{name}",
        "config_path": str(repo_root / "orchestrator.config.yaml"),
        "state_dir": str(state_dir),
        "first_seen": "2024-01-01T00:00:00Z",
        "last_seen": "2024-01-01T00:00:00Z",
    }
    fleet_dir.mkdir(parents=True, exist_ok=True)
    fleet_json.write_text(json.dumps(registry), encoding="utf-8")
    return repo_root


def test_count_fleet_live_reviews_sums_across_repos(tmp_path: Path) -> None:
    fleet_dir = tmp_path / "fleet"
    _register_repo(fleet_dir, tmp_path, "a", live_reviewers=2)
    _register_repo(fleet_dir, tmp_path, "b", live_reviewers=1)
    _register_repo(fleet_dir, tmp_path, "c", live_reviewers=0)

    count, skipped = count_fleet_live_reviews(str(fleet_dir))

    assert count == 3
    assert skipped == []


def test_count_fleet_live_reviews_skips_vanished_repo_and_tolerates_missing_reviews_dir(
    tmp_path: Path,
) -> None:
    fleet_dir = tmp_path / "fleet"
    _register_repo(fleet_dir, tmp_path, "a", live_reviewers=1)
    no_reviews = _register_repo(fleet_dir, tmp_path, "b", live_reviewers=3)
    # A repo that never launched a reviewer has no reviews_dir: zero live, NOT skipped.
    for child in layout.reviews_dir_default(no_reviews / ".var" / "charlie-work").iterdir():
        child.unlink()
    layout.reviews_dir_default(no_reviews / ".var" / "charlie-work").rmdir()
    registry = json.loads((fleet_dir / "fleet.json").read_text(encoding="utf-8"))
    registry["repos"]["owner/vanished"] = {
        "repo_root": str(tmp_path / "vanished"),
        "state_dir": str(tmp_path / "vanished" / ".var" / "charlie-work"),
    }
    (fleet_dir / "fleet.json").write_text(json.dumps(registry), encoding="utf-8")

    count, skipped = count_fleet_live_reviews(str(fleet_dir))

    assert count == 1
    assert skipped == ["owner/vanished"]


# --------------------------------------------------------------------------
# dispatch_reviews: cross-repo clamp, cap 0, lock
# --------------------------------------------------------------------------

_PRS = [
    {
        "number": i,
        "title": f"Fix #{i}",
        "url": f"https://example.test/pull/{i}",
        "headRefName": f"agent/issue-{i}-fix",
        "baseRefName": "main",
        "headRefOid": f"sha-{i}",
        "mergeStateStatus": "CLEAN",
        "body": f"Closes #{i}",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }
    for i in range(300, 303)
]


def _review_app(
    tmp_path: Path,
    *,
    fleet_dir: Path,
    fleet_cap: int,
    per_repo_cap: int = 0,
    lock_wait: float = 0.0,
    dry_run: bool = False,
) -> OrchestratorApp:
    """Repo ``b`` with three reviewable PRs, registered in ``fleet_dir``."""
    repo_root = _register_repo(fleet_dir, tmp_path, "b")
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(
            enabled=True, max_local_review_processes=0, max_concurrent_reviews=per_repo_cap
        ),
        fleet=FleetConfig(
            global_max_concurrent_reviews=fleet_cap, launch_lock_wait_seconds=lock_wait
        ),
    )
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = _PRS
    for pr in _PRS:
        _write_review_packet(repo_root, pr["number"], pr["headRefOid"])
    return OrchestratorApp(
        repo_root, paths, config, fake_gh, dry_run=dry_run, fleet_dir_override=str(fleet_dir)
    )


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    launched: list[int] = []

    def fake_launch(*args: Any, **kwargs: Any) -> Any:
        number = kwargs.get("issue_number") or args[0]
        launched.append(number)
        return _fake_claude_worker_record(number, kwargs.get("branch") or args[1])

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)
    return launched


def test_live_reviews_in_repo_a_clamp_dispatch_in_repo_b(
    tmp_path: Path, launches: list[int]
) -> None:
    fleet_dir = tmp_path / "fleet"
    _register_repo(fleet_dir, tmp_path, "a", live_reviewers=2)
    app = _review_app(tmp_path, fleet_dir=fleet_dir, fleet_cap=3)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["launched_count"] == 1
    assert len(launches) == 1
    assert result.data["fleet_review_concurrency_limit"] == 3
    assert result.data["fleet_live_review_count"] == 2
    assert result.data["fleet_available_review_slots"] == 1
    assert result.data["clamped_by"] == "fleet_max"
    # The governor fields ride the claim and dispatch events too.
    events = query_events(app.paths.state_file, kind="review_dispatch_claim")
    assert events and events[-1]["payload"]["fleet_live_review_count"] == 2
    assert events[-1]["payload"]["clamped_by"] == "fleet_max"
    dispatched = query_events(app.paths.state_file, kind="review_dispatch")
    assert dispatched and dispatched[-1]["payload"]["fleet_review_concurrency_limit"] == 3


def test_fleet_at_cap_launches_nothing(tmp_path: Path, launches: list[int]) -> None:
    fleet_dir = tmp_path / "fleet"
    _register_repo(fleet_dir, tmp_path, "a", live_reviewers=3)
    app = _review_app(tmp_path, fleet_dir=fleet_dir, fleet_cap=3)

    result = app.dispatch_reviews()

    assert result.data["launched_count"] == 0
    assert launches == []
    assert result.data["fleet_available_review_slots"] == 0
    assert result.data["clamped_by"] == "fleet_max"


def test_tighter_per_repo_cap_wins_and_is_not_blamed_on_fleet(
    tmp_path: Path, launches: list[int]
) -> None:
    fleet_dir = tmp_path / "fleet"
    _register_repo(fleet_dir, tmp_path, "a", live_reviewers=1)
    app = _review_app(tmp_path, fleet_dir=fleet_dir, fleet_cap=10, per_repo_cap=2)

    result = app.dispatch_reviews()

    assert result.data["launched_count"] == 2
    assert result.data["fleet_live_review_count"] == 1
    assert "clamped_by" not in result.data


def test_cap_zero_is_unlimited_and_never_reads_the_fleet(
    tmp_path: Path, launches: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    fleet_dir = tmp_path / "fleet"
    _register_repo(fleet_dir, tmp_path, "a", live_reviewers=50)
    app = _review_app(tmp_path, fleet_dir=fleet_dir, fleet_cap=0)

    def _boom(_override: str | None) -> tuple[int, list[str]]:
        raise AssertionError("fleet review count read while the cap is disabled")

    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_reviews", _boom)
    # Hold the fleet lock: a disabled cap must not contend for it either.
    held = try_acquire_fleet_lock(str(fleet_dir))
    assert held is not None
    try:
        result = app.dispatch_reviews()
    finally:
        held.release()

    assert result.data["launched_count"] == 3
    assert "deferred_reason" not in result.data
    assert not any(key.startswith("fleet_") for key in result.data)


def test_knob_is_read_per_pass_without_restart(tmp_path: Path, launches: list[int]) -> None:
    """Each pass reads ``self.config``; a fleet that reloads config per pass sees the flip."""
    fleet_dir = tmp_path / "fleet"
    _register_repo(fleet_dir, tmp_path, "a", live_reviewers=3)
    capped = _review_app(tmp_path, fleet_dir=fleet_dir, fleet_cap=3)
    assert capped.dispatch_reviews().data["launched_count"] == 0

    uncapped = OrchestratorApp(
        capped.repo_root,
        capped.paths,
        OrchestratorConfig(
            review_dispatch=capped.config.review_dispatch,
            fleet=FleetConfig(global_max_concurrent_reviews=0),
        ),
        capped.gh,
        fleet_dir_override=str(fleet_dir),
    )
    assert uncapped.dispatch_reviews().data["launched_count"] == 3


def test_lock_held_defers_with_event_and_launches_nothing(
    tmp_path: Path, launches: list[int]
) -> None:
    fleet_dir = tmp_path / "fleet"
    app = _review_app(tmp_path, fleet_dir=fleet_dir, fleet_cap=5)
    held = try_acquire_fleet_lock(str(fleet_dir))
    assert held is not None
    try:
        result = app.dispatch_reviews()
    finally:
        held.release()

    assert result.ok is True
    assert result.data["deferred_reason"] == "fleet_lock_held"
    assert result.data["launched_count"] == 0
    assert result.data["lock_wait_seconds"] == 0.0
    assert launches == []
    deferred = query_events(app.paths.state_file, kind=DISPATCH_DEFERRED_KIND)
    assert len(deferred) == 1
    assert deferred[0]["payload"]["lane"] == "dispatch_reviews"
    assert deferred[0]["payload"]["deferred_reason"] == "fleet_lock_held"
    # Nothing was claimed: the PRs are still dispatchable next pass.
    assert load_state_locked(app.paths.state_file)["prs"] == {}


def test_repeated_lock_deferrals_emit_dispatch_starved(
    tmp_path: Path, launches: list[int]
) -> None:
    fleet_dir = tmp_path / "fleet"
    app = _review_app(tmp_path, fleet_dir=fleet_dir, fleet_cap=5)
    held = try_acquire_fleet_lock(str(fleet_dir))
    assert held is not None
    try:
        for _ in range(3):
            app.dispatch_reviews()
    finally:
        held.release()

    starved = query_events(app.paths.state_file, kind=DISPATCH_STARVED_KIND)
    assert len(starved) == 1
    assert starved[0]["payload"]["lane"] == "dispatch_reviews"

    # A pass that gets the lock resets the streak.
    assert app.dispatch_reviews().data["launched_count"] == 3
    held = try_acquire_fleet_lock(str(fleet_dir))
    assert held is not None  # the pass released the lock on exit
    held.release()


def test_lock_is_released_when_the_pass_raises(tmp_path: Path, monkeypatch: Any) -> None:
    fleet_dir = tmp_path / "fleet"
    app = _review_app(tmp_path, fleet_dir=fleet_dir, fleet_cap=5)

    def _explode(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", _explode)
    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_reviews",
        lambda _o: (_ for _ in ()).throw(OSError()),
    )
    with pytest.raises(OSError):
        app.dispatch_reviews()

    held = try_acquire_fleet_lock(str(fleet_dir))
    assert held is not None
    held.release()


def test_dry_run_previews_the_clamp_without_the_lock(tmp_path: Path, launches: list[int]) -> None:
    fleet_dir = tmp_path / "fleet"
    _register_repo(fleet_dir, tmp_path, "a", live_reviewers=2)
    app = _review_app(tmp_path, fleet_dir=fleet_dir, fleet_cap=3, dry_run=True)
    held = try_acquire_fleet_lock(str(fleet_dir))
    assert held is not None
    try:
        result = app.dispatch_reviews()
    finally:
        held.release()

    assert result.data["selected_count"] == 1
    assert result.data["fleet_live_review_count"] == 2
    assert result.data["clamped_by"] == "fleet_max"
    assert "deferred_reason" not in result.data
    assert launches == []


# --------------------------------------------------------------------------
# Local (no-remote) review lane
# --------------------------------------------------------------------------


def _local_review_app(tmp_path: Path, *, fleet_cap: int) -> OrchestratorApp:
    repo = tmp_path / "localrepo"
    repo.mkdir()
    _init_repo(repo)
    issues_dir = repo / "docs" / "issues"
    config = _local_config(
        repo,
        issues_dir,
        fleet={"global_max_concurrent_reviews": fleet_cap, "launch_lock_wait_seconds": 0},
        review_dispatch={"max_local_review_processes": 0},
    )
    gh = LocalFileGitHub(repo_root=repo, issues_dir=issues_dir)
    paths = runtime_paths(repo, config.runtime.state_dir)
    app = OrchestratorApp(repo, paths, config, gh, fleet_dir_override=str(tmp_path / "fleet"))
    for number in (7, 8):
        _make_branch(repo, f"agent/issue-{number}-x", f"f{number}.py", f"x = {number}\n")
        _git(repo, "checkout", "main")
        _parked_issue(app, issues_dir, number, f"agent/issue-{number}-x")
    app._local_review_packets()
    return app


def test_local_review_lane_is_clamped_by_the_fleet_cap(
    tmp_path: Path, launches: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _local_review_app(tmp_path, fleet_cap=3)
    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_reviews", lambda _o: (2, []))

    result = app._local_dispatch_reviewers()

    assert len(result["launched"]) == 1
    assert len(result["skipped"]) == 1
    assert result["fleet_review_concurrency_limit"] == 3
    assert result["fleet_live_review_count"] == 2
    assert result["clamped_by"] == "fleet_max"


def test_local_review_lane_defers_when_the_fleet_lock_is_held(
    tmp_path: Path, launches: list[int]
) -> None:
    app = _local_review_app(tmp_path, fleet_cap=3)
    held = try_acquire_fleet_lock(str(tmp_path / "fleet"))
    assert held is not None
    try:
        result = app._local_dispatch_reviewers()
    finally:
        held.release()

    assert result["deferred_reason"] == "fleet_lock_held"
    assert result["launched"] == []
    assert launches == []
    deferred = query_events(app.paths.state_file, kind=DISPATCH_DEFERRED_KIND)
    assert deferred and deferred[0]["payload"]["lane"] == "dispatch_reviews"


def test_local_review_lane_cap_zero_is_unlimited(
    tmp_path: Path, launches: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _local_review_app(tmp_path, fleet_cap=0)

    def _boom(_override: str | None) -> tuple[int, list[str]]:
        raise AssertionError("fleet review count read while the cap is disabled")

    monkeypatch.setattr("charlie_work.fleet_registry.count_fleet_live_reviews", _boom)

    result = app._local_dispatch_reviewers()

    assert len(result["launched"]) == 2
    assert "clamped_by" not in result
