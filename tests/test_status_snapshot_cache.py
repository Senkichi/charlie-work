"""Regression tests for the status-snapshot cache (issue #1463).

``fleet status --json`` was taking ~50s on an idle host because every
invocation re-walked the full GitHub API path (issue_list, pr_list, blocker
prefetch, backlog reachability, runner pool). The fix: the loop pass writes
a ``status-snapshot.json`` at the end of every pass, and ``status()`` serves
from that snapshot when it is fresher than ``runtime.status_snapshot_ttl_seconds``
(default 900s). This file exercises the cache read/write paths and the
fall-back-to-live semantics.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work.config import OrchestratorConfig
from charlie_work.layout import status_snapshot_path
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp


class _FakeGitHub:
    """Minimal stub sufficient for OrchestratorApp.status().

    Counts ``issue_list`` calls so tests can assert whether the cache was
    served (zero calls) or the live path ran (one or more calls).
    """

    def __init__(self) -> None:
        self.issues = [
            {
                "number": 123,
                "title": "Fix search",
                "url": "https://example.test/issues/123",
                "body": "Search is broken",
                "labels": [{"name": "automated-ready"}],
                "state": "OPEN",
            }
        ]
        self.prs: list[dict] = []
        self.issue_list_calls = 0

    def issue_list(self, labels=None, state=None):
        self.issue_list_calls += 1
        if isinstance(labels, str):
            return [
                issue
                for issue in self.issues
                if labels in {label["name"] for label in issue.get("labels", [])}
            ]
        elif labels:
            return [
                issue
                for issue in self.issues
                if any(
                    label in {label_obj["name"] for label_obj in issue.get("labels", [])}
                    for label in labels
                )
            ]
        return self.issues

    def pr_list(self):
        return [pr for pr in self.prs if pr.get("state", "OPEN").upper() == "OPEN"]

    def check_graphql_rate_limit(self, threshold: int) -> tuple[bool, int, int | None]:
        return (True, 10000, 0)

    def run(self, args, *, json_output=False, allow_failure=False):
        return [] if json_output else ""


def _make_app(tmp_path: Path, *, ttl: int = 900) -> tuple[OrchestratorApp, _FakeGitHub]:
    config = OrchestratorConfig()
    if ttl != config.runtime.status_snapshot_ttl_seconds:
        from dataclasses import replace

        config = replace(config, runtime=replace(config.runtime, status_snapshot_ttl_seconds=ttl))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = _FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, gh)
    return app, gh


def _write_snapshot(tmp_path: Path, data: dict, *, age_seconds: float = 0.0) -> None:
    """Write a status-snapshot envelope with the given data and age."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    snapshot_path = status_snapshot_path(paths.root)
    written_at = datetime.now(UTC) - timedelta(seconds=age_seconds)
    envelope = {"snapshot_written_at": written_at.isoformat(), "data": data}
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    tmp.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(snapshot_path)


# ---------------------------------------------------------------------------
# Cache-hit tests
# ---------------------------------------------------------------------------


def test_status_serves_from_fresh_snapshot(tmp_path: Path) -> None:
    """A fresh snapshot is returned without any GitHub API calls."""
    app, gh = _make_app(tmp_path)
    cached_data = {"ready_issue_count": 42, "available_issue_count": 7, "issues": []}
    _write_snapshot(tmp_path, cached_data, age_seconds=10)

    result = app.status()

    assert result.ok is True
    assert result.data == cached_data
    # Zero GitHub API calls — the cache was served, not the live path.
    assert gh.issue_list_calls == 0


def test_status_cache_hit_message(tmp_path: Path) -> None:
    """The cached result's message distinguishes it from a live computation."""
    app, _gh = _make_app(tmp_path)
    _write_snapshot(tmp_path, {"ready_issue_count": 1}, age_seconds=5)

    result = app.status()

    assert "cached" in result.message


# ---------------------------------------------------------------------------
# Cache-miss / fall-back tests
# ---------------------------------------------------------------------------


def test_status_falls_back_when_snapshot_missing(tmp_path: Path) -> None:
    """No snapshot file → live computation runs."""
    app, gh = _make_app(tmp_path)

    result = app.status()

    assert result.ok is True
    assert gh.issue_list_calls > 0
    assert result.data["ready_issue_count"] == 1


def test_status_falls_back_when_snapshot_stale(tmp_path: Path) -> None:
    """A snapshot older than the TTL → live computation runs."""
    app, gh = _make_app(tmp_path, ttl=60)
    _write_snapshot(tmp_path, {"ready_issue_count": 99}, age_seconds=120)

    result = app.status()

    assert result.ok is True
    assert gh.issue_list_calls > 0
    # The live data should reflect the actual issue count (1), not the stale 99.
    assert result.data["ready_issue_count"] == 1


def test_status_falls_back_when_snapshot_corrupt(tmp_path: Path) -> None:
    """A corrupt snapshot file → live computation runs, no crash."""
    app, gh = _make_app(tmp_path)
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    snapshot_path = status_snapshot_path(paths.root)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text("{not valid json", encoding="utf-8")

    result = app.status()

    assert result.ok is True
    assert gh.issue_list_calls > 0


def test_status_ttl_zero_disables_cache(tmp_path: Path) -> None:
    """``status_snapshot_ttl_seconds=0`` → always compute live, even with a
    fresh snapshot file present."""
    app, gh = _make_app(tmp_path, ttl=0)
    _write_snapshot(tmp_path, {"ready_issue_count": 99}, age_seconds=0)

    result = app.status()

    assert result.ok is True
    assert gh.issue_list_calls > 0
    assert result.data["ready_issue_count"] == 1


def test_status_use_cache_false_bypasses_cache(tmp_path: Path) -> None:
    """``use_cache=False`` → always compute live, even with a fresh snapshot."""
    app, gh = _make_app(tmp_path)
    _write_snapshot(tmp_path, {"ready_issue_count": 99}, age_seconds=0)

    result = app.status(use_cache=False)

    assert result.ok is True
    assert gh.issue_list_calls > 0
    assert result.data["ready_issue_count"] == 1


# ---------------------------------------------------------------------------
# Snapshot-write tests
# ---------------------------------------------------------------------------


def test_write_status_snapshot_creates_valid_file(tmp_path: Path) -> None:
    """``_write_status_snapshot`` writes a parseable envelope with the live
    status data."""
    app, _gh = _make_app(tmp_path)

    app._write_status_snapshot()

    snapshot_path = app._status_snapshot_file
    assert snapshot_path.exists()
    envelope = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert "snapshot_written_at" in envelope
    assert isinstance(envelope["snapshot_written_at"], str)
    assert isinstance(envelope["data"], dict)
    assert envelope["data"]["ready_issue_count"] == 1


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    """A snapshot written by ``_write_status_snapshot`` is served by a
    subsequent ``status()`` call with zero GitHub API calls."""
    app, gh = _make_app(tmp_path)

    # First call: live computation (no snapshot yet).
    result_live = app.status()
    assert gh.issue_list_calls > 0

    # Write the snapshot (also calls status(use_cache=False) internally).
    app._write_status_snapshot()
    calls_after_write = gh.issue_list_calls

    # Second call: should serve from cache (zero new API calls).
    result_cached = app.status()
    assert gh.issue_list_calls == calls_after_write
    assert result_cached.data == result_live.data


def test_write_status_snapshot_does_not_recurse(tmp_path: Path) -> None:
    """``_write_status_snapshot`` calls ``status(use_cache=False)`` — it must
    not serve from a pre-existing stale snapshot (which would write stale
    data instead of a fresh live computation)."""
    app, gh = _make_app(tmp_path)
    # Plant a stale snapshot with bogus data.
    _write_snapshot(tmp_path, {"ready_issue_count": 999}, age_seconds=999)

    app._write_status_snapshot()

    # The written snapshot should contain the LIVE data (1 issue), not the
    # stale bogus data (999).
    snapshot_path = app._status_snapshot_file
    envelope = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert envelope["data"]["ready_issue_count"] == 1
    # And the live computation must have run (GitHub API calls were made).
    assert gh.issue_list_calls > 0
