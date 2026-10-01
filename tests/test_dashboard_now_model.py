# ruff: noqa: F811  (the imported ``fleet`` fixture is re-bound as a test parameter)
"""Tests for the dashboard Now model (``dashboard/now_model.py``).

Fixtures (``_dashboard_now_fixtures``) go through the real writers: ``touch_repo``
(registry), ``status_snapshot.write_status_snapshot`` (snapshot envelope; clock stamp
pinned) and ``instrumentation.log_event`` (global ``runner_allocation`` event).
"""

from __future__ import annotations

import argparse
import json
import shlex
import socket
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from charlie_work import cli, fleet_status
from charlie_work.dashboard import now_model, sources

from _dashboard_now_fixtures import (  # noqa: F401  (fleet is a pytest fixture)
    ALPHA,
    BETA,
    NOW,
    L,
    Finding,
    _issue,
    _read,
    _root,
    _write_snapshot,
    fleet,
)


def test_freshness_and_threshold(fleet) -> None:
    model = now_model.build_now_model(_read(fleet), NOW)

    # No heartbeat: 2 passes of 300s + one 30s collector tick.
    assert model.stale_threshold_seconds == 630.0
    alpha, beta = model.freshness
    assert (alpha.repo, alpha.age_seconds, alpha.stale) == ("owner/alpha", 150.0, False)
    assert (beta.repo, beta.age_seconds, beta.stale) == ("owner/beta", 3600.0, True)


def test_threshold_follows_heartbeat_pass_interval(fleet, tmp_path: Path) -> None:
    hb = tmp_path / "hb.json"
    hb.write_text(
        json.dumps({"full_pass_interval_seconds": 60, "last_beat_at": "2026-10-01T11:59:50Z"}),
        encoding="utf-8",
    )
    read = _read(
        fleet, supervisor_heartbeat=sources.read_json_file(hb), collector_interval_seconds=10
    )

    assert now_model.build_now_model(read, NOW).stale_threshold_seconds == 130.0


def test_needs_me_exact_rows_and_order(fleet) -> None:
    findings = [
        Finding("merge-flow", "owner/alpha", "anomaly", "no merges 6h"),
        Finding("github-rate", "owner/beta", "warn", "rate low"),
        Finding("loop", "owner/alpha", "ok", "fine"),
    ]
    items = now_model.build_now_model(_read(fleet), NOW, findings).needs_me
    a = shlex.quote(_root(fleet, "owner/alpha"))

    got = [
        (i.kind, i.severity, i.repo, i.age_seconds, i.reason, i.command, i.as_of_snapshot)
        for i in items
    ]
    assert got == [
        ("alarm", "anomaly", "owner/alpha", None, "merge-flow: no merges 6h", None, False),
        (
            "human_needed",
            "action",
            "owner/alpha",
            7200.0,
            "Human needed: PR #50 (issue #5) awaits an operator verdict",
            f"charlie --repo {a} unescalate --pr 50",
            True,
        ),
        (
            "operator_queue",
            "action",
            "owner/alpha",
            3600.0,
            "Operator queue: #4 t4",
            f"charlie --repo {a} unescalate --issue 4",
            True,
        ),
        (
            "stale_source",
            "warn",
            "owner/beta",
            3600.0,
            "snapshot is 3600s old (stale after 630s)",
            None,
            False,
        ),
        ("alarm", "warn", "owner/beta", None, "github-rate: rate low", None, False),
    ]


def test_every_command_is_a_real_cli_invocation(fleet) -> None:
    pause = {"paused": True, "paused_at": "2026-10-01T11:00:00Z", "reason": "ops"}
    items = now_model.build_now_model(_read(fleet, pause=pause), NOW).needs_me
    commands = [i.command for i in items if i.command]
    assert len(commands) == 3
    parser = cli.build_parser()
    for command in commands:
        parts = shlex.split(command)
        assert parts[0] == "charlie"
        assert isinstance(parser.parse_args(parts[1:]), argparse.Namespace)
    paused = next(i for i in items if i.kind == "paused")
    assert (paused.severity, paused.age_seconds, paused.command) == (
        "anomaly",
        3600.0,
        "charlie fleet resume",
    )
    assert paused.reason == "Fleet paused: ops"


def test_supervisor_and_stale_runner_rows(fleet, tmp_path: Path) -> None:
    hb = tmp_path / "hb.json"
    hb.write_text(
        json.dumps(
            {
                "full_pass_interval_seconds": 300,
                "last_beat_at": "2026-10-01T11:00:00Z",
                "exited_at": None,
            }
        ),
        encoding="utf-8",
    )
    read = _read(fleet, supervisor_heartbeat=sources.read_json_file(hb))
    stale_event = {**read.runner_allocation, "ts": "2026-10-01T10:00:00Z"}
    read = replace(read, runner_allocation=stale_event)

    model = now_model.build_now_model(read, NOW)

    rows = [(i.kind, i.severity, i.age_seconds) for i in model.needs_me if i.repo == "fleet"]
    assert rows == [("supervisor", "anomaly", 3600.0), ("stale_source", "warn", 7200.0)]
    assert model.capacity.runners_stale is True


def test_unreadable_snapshot_is_stale_with_error(fleet) -> None:
    sources.enumerate_repos(str(fleet))[0].snapshot_path.write_text("{not json", encoding="utf-8")
    model = now_model.build_now_model(_read(fleet), NOW)

    alpha = model.freshness[0]
    assert alpha.stale is True and alpha.age_seconds is None
    assert alpha.error is not None and alpha.error.startswith("unreadable:")
    row = next(i for i in model.needs_me if i.kind == "stale_source" and i.repo == "owner/alpha")
    assert row.reason == f"snapshot unreadable: {alpha.error}"


def test_flow_stages_use_label_config_and_breakdown(fleet) -> None:
    flow = now_model.build_now_model(_read(fleet), NOW).flow

    assert [(s.name, s.label, s.count) for s in flow.stages] == [
        ("Dispatchable", None, 2),
        ("Queued", L.queued, 1),
        ("In progress", L.in_progress, 1),
        ("PR open", L.pr_open, 1),
        ("Reviewing", L.reviewing, 1),
        ("Needs rework", L.needs_rework, 1),
    ]
    assert all(s.as_of_snapshot for s in flow.stages)
    assert flow.done_24h == 9
    # missing_ready (4) is not Ready, so it is not "Ready but not dispatchable".
    assert [(r.reason, r.count, r.examples) for r in flow.not_dispatchable] == [
        ("terminal_label", 2, ("owner/alpha#4", "owner/alpha#5")),
        ("active_label", 3, ()),
        ("operator_claimed", 1, ("owner/alpha#9",)),
    ]


def test_flow_dispatchable_falls_back_to_issue_flags_when_unobserved(fleet) -> None:
    read = _read(fleet)
    beta = read.repos[1]
    data = {
        **beta.snapshot.data,
        "issues": [_issue(11, dispatchable=True), _issue(12, dispatchable=True)],
    }
    snap = replace(beta.snapshot, data=data)
    repos = (read.repos[0], replace(beta, snapshot=snap))

    flow = now_model.build_now_model(replace(read, repos=repos), NOW).flow

    assert flow.stages[0].count == 2 + 2


def test_capacity_workers_reviewers_runners_and_capped_demand(fleet) -> None:
    cap = now_model.build_now_model(_read(fleet), NOW).capacity

    assert (cap.workers_live, cap.workers_cap) == (3, 3)
    assert [(w.repo, w.live, w.cap) for w in cap.workers_by_repo] == [
        ("owner/alpha", 2, 2),
        ("owner/beta", 1, None),
    ]
    assert (cap.reviewers_live, cap.reviewers_cap) == (1, 6)
    assert [(w.repo, w.live, w.cap) for w in cap.reviewers_by_repo] == [
        ("owner/alpha", 1, 4),
        ("owner/beta", 0, 4),
    ]
    assert [(r.repo, r.capacity, r.online, r.busy, r.parked, r.demand) for r in cap.runners] == [
        ("owner/alpha", 3, 2, 1, 1, 5),
        ("owner/beta", 1, 1, None, 0, 0),
    ]
    assert (cap.runners_age_seconds, cap.runners_stale) == (120.0, False)
    assert cap.capped_demand_now is True
    assert cap.capped_repos == ("owner/alpha",)


def test_capped_demand_follows_repo_cap_then_clears(fleet) -> None:
    read = _read(fleet, global_worker_cap=10)
    cap = now_model.build_now_model(read, NOW).capacity
    # Fleet budget has room, but alpha's own cap (2) is full with Dispatchable work waiting.
    assert (cap.capped_demand_now, cap.capped_repos) == (True, ("owner/alpha",))

    repos = tuple(replace(r, worker_cap=0) for r in read.repos)
    cap = now_model.build_now_model(replace(read, repos=repos), NOW).capacity
    assert (cap.capped_demand_now, cap.capped_repos) == (False, ())


def test_totals_sum_snapshot_counters(fleet) -> None:
    totals = now_model.build_now_model(_read(fleet), NOW).totals
    assert (
        totals.ready_issues,
        totals.active_issues,
        totals.open_linked_prs,
        totals.unlinked_prs,
        totals.live_workers,
    ) == (9, 5, 3, 1, 3)


def test_counts_match_fleet_status_aggregation(fleet, monkeypatch) -> None:
    """Model counts equal ``run_fleet_status`` over the same cached snapshots.

    ``run_fleet_status`` serves ``status(use_cache=True)`` from a fresh snapshot
    without GitHub; the GitHub client factory and sockets are blocked here so a cache
    miss would fail the test rather than reach the network. Snapshots are rewritten
    with the current clock so they sit inside the cache TTL.
    """
    now = datetime.now(UTC)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    for repo, data in zip(sources.enumerate_repos(str(fleet)), (ALPHA, BETA), strict=True):
        _write_snapshot(monkeypatch, repo.state_dir, stamp, data)

    def _blocked(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(fleet_status, "github_client_for", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    result = fleet_status.run_fleet_status(
        argparse.Namespace(fleet_dir=str(fleet), no_cache=False)
    )
    assert result.data["errors"] == []
    assert sorted(result.data["repos"]) == ["owner/alpha", "owner/beta"]

    model = now_model.build_now_model(_read(fleet, now), now)

    per = list(result.data["repos"].values())
    assert model.totals.ready_issues == sum(d["ready_issue_count"] for d in per) == 9
    assert model.totals.active_issues == sum(d["active_issue_count"] for d in per) == 5
    assert model.totals.open_linked_prs == sum(d["open_linked_pr_count"] for d in per) == 3
    assert model.totals.unlinked_prs == sum(d["unlinked_pr_count"] for d in per) == 1
    assert model.capacity.workers_live == sum(len(d["workers"]) for d in per) == 3
    # Stage counts reconcile with the aggregate's own active counter (no double label here).
    stage = {s.name: s.count for s in model.flow.stages}
    active = ("Queued", "In progress", "PR open", "Reviewing", "Needs rework")
    assert sum(stage[name] for name in active) == sum(d["active_issue_count"] for d in per)
