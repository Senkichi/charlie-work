"""``collect_sources_read`` on a synthetic fleet dir (not through the ``_read()`` fixture).

The Now-model tests inject every ``SourcesRead`` field by hand, which hid that the real
collector never filled caps, busy listeners, ``done_24h``, reviewers or ages. These tests
build a fleet with the real writers (``touch_repo``, ``log_event``, ``record_loop_pass``,
``run_rollup``) and the layered config files, then collect for real.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from _dashboard_now_fixtures import NOW, _register

from charlie_work import instrumentation
from charlie_work.dashboard import now_cadence, now_model, rollup, sources
from charlie_work.dashboard.now_collect import collect_sources_read

REPO_ROOT = Path(__file__).resolve().parent.parent
DEAD_PID = 4_000_000
DISPATCHED = "review_dispatch_dispatched"


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


class Fleet:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.dir = tmp_path / "fleet"
        self.roots = {"owner/alpha": tmp_path / "alpha", "owner/beta": tmp_path / "beta"}
        self.states = {k: _register(self.dir, r, k) for k, r in self.roots.items()}
        self.hb = self.dir / sources.HEARTBEAT_FILENAME

    def emit(self, state: Path, ts: datetime, kind: str, payload: dict) -> None:
        self.monkeypatch.setattr(instrumentation, "_now_iso", lambda: _iso(ts))
        path = state if state == self.hb else state / "state.json"
        instrumentation.log_event(path, kind, payload)

    def global_config(self, text: str) -> None:
        (self.dir / "config.yaml").write_text(text, encoding="utf-8")

    def repo_config(self, key: str, text: str) -> None:
        (self.roots[key] / "orchestrator.config.yaml").write_text(text, encoding="utf-8")

    def state_json(self, key: str, text: str) -> None:
        self.states[key].mkdir(parents=True, exist_ok=True)
        (self.states[key] / "state.json").write_text(text, encoding="utf-8")

    def collect(self, now: datetime = NOW):
        return collect_sources_read(now, str(self.dir))

    def close(self) -> None:
        for path in (*(s / "state.json" for s in self.states.values()), self.hb):
            instrumentation.close_db(path)


@pytest.fixture
def fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    f = Fleet(tmp_path, monkeypatch)
    yield f
    f.close()


def _repo(read, key: str):
    return next(r for r in read.repos if r.key == key)


def test_caps_come_from_the_layered_config(fleet) -> None:
    fleet.global_config(
        "fleet:\n  global_max_concurrent_sessions: 4\n  global_max_concurrent_reviews: 6\n"
        "dispatch:\n  max_concurrent_sessions: 7\n"
    )
    fleet.repo_config(
        "owner/alpha",
        "dispatch:\n  max_concurrent_sessions: 5\nreview_dispatch:\n  max_concurrent_reviews: 4\n",
    )

    read = fleet.collect()

    assert (read.global_worker_cap, read.global_review_cap) == (4, 6)
    alpha, beta = _repo(read, "owner/alpha"), _repo(read, "owner/beta")
    # alpha's own file wins; beta (no file) inherits the GLOBAL layer's 7, which a
    # single-file read of the repo config would miss.
    assert (alpha.worker_cap, alpha.review_cap) == (5, 4)
    assert beta.worker_cap == 7 and beta.review_cap is not None


def test_unreadable_repo_config_is_unknown_not_a_default_cap(fleet) -> None:
    fleet.global_config("fleet:\n  global_max_concurrent_sessions: 4\n")
    fleet.repo_config("owner/beta", "dispatch: [not, a, mapping\n")

    read = fleet.collect()

    beta = _repo(read, "owner/beta")
    assert (beta.worker_cap, beta.review_cap) == (None, None)
    assert _repo(read, "owner/alpha").worker_cap is not None  # control: alpha loads
    assert read.global_worker_cap == 4  # the readable repo speaks for the fleet layer


def test_empty_fleet_leaves_everything_unknown(tmp_path: Path) -> None:
    read = collect_sources_read(NOW, str(tmp_path / "empty-fleet"))
    assert read.repos == () and read.global_worker_cap is None and read.global_review_cap is None
    assert read.done_24h is None and read.runner_busy == {}


def test_escalation_ages_from_state_json(fleet) -> None:
    issues = {
        "4": {"terminal_since": "2026-10-01T10:00:00Z"},
        "5": {"merged_pr_mention_flagged_at": "2026-10-01T09:00:00Z"},
        "6": {"terminal_since": "garbage"},
        "7": {},
        "x": {"terminal_since": "2026-10-01T10:00:00Z"},
    }
    fleet.state_json("owner/alpha", json.dumps({"issues": issues}))

    read = fleet.collect()

    assert dict(_repo(read, "owner/alpha").escalated_since) == {
        4: NOW - timedelta(hours=2),
        5: NOW - timedelta(hours=3),
    }
    assert _repo(read, "owner/beta").escalated_since == ()  # no state.json


def test_reviewers_live_unknown_is_none_and_measured_idle_is_zero(fleet) -> None:
    prs = {
        "30": {"review_dispatch_status": DISPATCHED, "reviewer_pid": os.getpid()},
        "31": {"review_dispatch_status": DISPATCHED, "reviewer_pid": DEAD_PID},
        "32": {"review_dispatch_status": "pending"},
    }
    fleet.state_json("owner/alpha", json.dumps({"prs": prs}))
    fleet.state_json("owner/beta", json.dumps({"prs": {}}))

    read = fleet.collect()

    assert _repo(read, "owner/alpha").reviewers_live == 1  # only the live pid
    assert _repo(read, "owner/beta").reviewers_live == 0  # measured: nobody reviewing
    fleet.state_json("owner/beta", "{torn")
    assert _repo(fleet.collect(), "owner/beta").reviewers_live is None  # unreadable != 0


def _runners(root: Path, names: dict[str, str]) -> None:
    for name, repo in names.items():
        d = root / name
        d.mkdir(parents=True)
        # The runner writes .runner as UTF-8 with a BOM.
        (d / ".runner").write_text(
            json.dumps({"agentName": name, "gitHubUrl": f"https://github.com/{repo}"}),
            encoding="utf-8-sig",
        )


def _health(fleet: Fleet, ts: datetime) -> None:
    def total(name: str, procs: int) -> dict:
        return {"runner": name, "process_count": procs, "oldest_worker_age_s": None}

    totals = [total("a-1", 19), total("a-2", 0), total("b-1", 0), total("zz-9", 3)]
    fleet.emit(fleet.hb, ts, "runner_health", {"totals": totals})


def _managed(fleet: Fleet, root: Path) -> None:
    fleet.global_config(f"runner_allocation:\n  managed_root: '{root.as_posix()}'\n")


def test_runner_busy_from_runner_health_and_runner_files(fleet, tmp_path: Path) -> None:
    root = tmp_path / "runners"
    _runners(root, {"a-1": "owner/alpha", "a-2": "owner/alpha", "b-1": "owner/beta"})
    _managed(fleet, root)
    _health(fleet, NOW - timedelta(seconds=60))

    # zz-9 has no .runner file under the root, so it is attributed to no repo.
    assert fleet.collect().runner_busy == {"owner/alpha": 1, "owner/beta": 0}


def test_stale_runner_health_is_unknown_not_idle(fleet, tmp_path: Path) -> None:
    root = tmp_path / "runners"
    _runners(root, {"a-1": "owner/alpha"})
    _managed(fleet, root)
    _health(fleet, NOW - timedelta(seconds=60))
    assert fleet.collect().runner_busy == {"owner/alpha": 1}  # control: fresh is read
    assert fleet.collect(NOW + timedelta(hours=3)).runner_busy == {}


def test_runner_discovery_ignores_entries_resolving_outside_the_root(
    fleet, tmp_path: Path
) -> None:
    root, outside = tmp_path / "runners", tmp_path / "unrelated"
    _runners(root, {"a-1": "owner/alpha"})
    _runners(outside, {"b-1": "owner/beta"})
    link, target = root / "linked", outside / "b-1"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:  # no symlink privilege: a junction (the real Windows hazard) needs none
        made = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False
        )
        if made.returncode != 0:
            pytest.skip("neither symlinks nor junctions available")
    _managed(fleet, root)
    _health(fleet, NOW - timedelta(seconds=60))

    # Control: a-1 (inside) is attributed; b-1 is only reachable through the escaping link.
    assert fleet.collect().runner_busy == {"owner/alpha": 1}


def _merge(fleet: Fleet, ts: datetime, kind: str, issue: int, pr: int) -> None:
    payload = {"issue_number": issue, "pr_number": pr}
    fleet.emit(fleet.states["owner/alpha"], ts, kind, payload)


def _rollup(fleet: Fleet) -> None:
    # The rollup requires every registered source DB, the global one included.
    fleet.emit(fleet.hb, NOW - timedelta(minutes=1), "runner_allocation", {"targets": []})
    fleet.emit(fleet.states["owner/beta"], NOW - timedelta(minutes=1), "noop", {})
    assert rollup.run_rollup(rollup.rollup_sources(str(fleet.dir)), NOW).errors == ()


def test_done_24h_counts_distinct_merges_from_dashboard_db(fleet) -> None:
    recent = NOW - timedelta(minutes=5)
    _merge(fleet, NOW - timedelta(hours=30), "merge_succeeded", 1, 10)  # outside the window
    _merge(fleet, NOW - timedelta(hours=2), "merge_succeeded", 2, 20)
    _merge(fleet, recent, "merge_succeeded", 3, 30)
    _merge(fleet, recent, "finalize_externally_merged", 3, 30)  # same merge, second kind
    _rollup(fleet)

    assert fleet.collect().done_24h == 2


def test_done_24h_unknown_without_or_behind_dashboard_db(fleet) -> None:
    assert fleet.collect().done_24h is None  # no dashboard.db yet: unknown, not 0
    _merge(fleet, NOW - timedelta(minutes=5), "merge_succeeded", 3, 30)
    _rollup(fleet)
    assert fleet.collect().done_24h == 1
    # A rollup that has not run for hours would under-count silently: unknown instead.
    assert fleet.collect(NOW + timedelta(hours=3)).done_24h is None


def _passes(fleet: Fleet, gaps: list[int]) -> None:
    state = fleet.states["owner/alpha"] / "state.json"
    at = NOW - timedelta(hours=2)
    for i, gap in enumerate([0, *gaps]):
        at += timedelta(seconds=gap)
        started = _iso(at - timedelta(seconds=5))
        instrumentation.record_loop_pass(state, f"c{i}", started)
        instrumentation.record_loop_pass(state, f"c{i}", started, _iso(at), ok=True)


def test_observed_pass_cadence_drives_the_stale_threshold(fleet) -> None:
    _passes(fleet, [500] * 17 + [1100] * 3)  # nearest-rank p90 of 20 gaps = 1100

    read = fleet.collect()

    assert read.snapshot_gap_p90_seconds == 1100.0
    assert now_model.build_now_model(read, NOW).stale_threshold_seconds == 2200.0


def test_too_few_passes_is_no_observation(fleet) -> None:
    _passes(fleet, [500] * 5)
    assert fleet.collect().snapshot_gap_p90_seconds is None


def test_p90_is_nearest_rank() -> None:
    assert now_cadence.p90([float(n) for n in range(1, 11)]) == 9.0
    assert now_cadence.p90([1.0] * 9) is None


def test_supervisor_rule_matches_heartbeat_check_script() -> None:
    """One rule, two implementations until PR #2239's alarm leaf lands: pin them together."""
    spec = importlib.util.spec_from_file_location(
        "heartbeat_check_pin", REPO_ROOT / "scripts" / "heartbeat_check.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the script binds itself via sys.modules[__name__]
    try:
        spec.loader.exec_module(module)
    except ImportError:
        pytest.skip("heartbeat_check dependencies unavailable")
    finally:
        sys.modules.pop(spec.name, None)
    assert (
        now_cadence.SUPERVISOR_HEARTBEAT_STALE_MULTIPLIER
        == module.SUPERVISOR_HEARTBEAT_STALE_MULTIPLIER
    )
    assert (
        now_cadence.SUPERVISOR_HEARTBEAT_DEFAULT_PASS_TIMEOUT_SECONDS
        == module.SUPERVISOR_HEARTBEAT_DEFAULT_PASS_TIMEOUT_SECONDS
    )
    assert (
        now_cadence.supervisor_beat_threshold_seconds({"max_pass_runtime_seconds": 1800}) == 3600
    )
    assert (
        now_cadence.supervisor_beat_threshold_seconds({"full_pass_interval_seconds": 300}) == 600
    )
    assert now_cadence.supervisor_beat_threshold_seconds({}) == 3600
