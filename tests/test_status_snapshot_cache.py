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
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from charlie_work import status_snapshot
from charlie_work.cli import build_parser, run_fleet_status
from charlie_work.config import (
    ConfigError,
    DeescalationConfig,
    MainCiReclaimConfig,
    OrchestratorConfig,
    ReconcilePassConfig,
    WorktreeReclamationConfig,
    build_config_from_data,
)
from charlie_work.layout import status_snapshot_path
from charlie_work.paths import runtime_paths
from charlie_work.state import empty_state, save_state
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

    def merged_pr_list(self):
        # Issue #1337: status() fetches merged PRs for the reachability
        # classifier's mention-coverage map. No merged PRs in these tests.
        return [pr for pr in self.prs if pr.get("state", "OPEN").upper() == "MERGED"]

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
    # The cached payload's domain fields are preserved verbatim; the
    # freshness fields (snapshot_written_at / cache_age_seconds) are injected
    # by read_status_snapshot and are asserted separately below.
    assert result.data["ready_issue_count"] == 42
    assert result.data["available_issue_count"] == 7
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
    """``write_status_snapshot`` writes a parseable envelope with the live
    status data."""
    app, _gh = _make_app(tmp_path)

    status_snapshot.write_status_snapshot(app)

    snapshot_path = status_snapshot.snapshot_path(app.paths.root)
    assert snapshot_path.exists()
    envelope = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert "snapshot_written_at" in envelope
    assert isinstance(envelope["snapshot_written_at"], str)
    assert isinstance(envelope["data"], dict)
    assert envelope["data"]["ready_issue_count"] == 1


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    """A snapshot written by ``write_status_snapshot`` is served by a
    subsequent ``status()`` call with zero GitHub API calls."""
    app, gh = _make_app(tmp_path)

    # First call: live computation (no snapshot yet).
    result_live = app.status()
    assert gh.issue_list_calls > 0

    # Write the snapshot (also calls status(use_cache=False) internally).
    status_snapshot.write_status_snapshot(app)
    calls_after_write = gh.issue_list_calls

    # Second call: should serve from cache (zero new API calls).
    result_cached = app.status()
    assert gh.issue_list_calls == calls_after_write
    # Domain fields match; freshness fields differ by design (live has None,
    # cached has the snapshot timestamp + age).
    assert result_cached.data["ready_issue_count"] == result_live.data["ready_issue_count"]
    assert result_cached.data["snapshot_written_at"] is not None
    assert result_cached.data["cache_age_seconds"] is not None
    assert result_live.data["snapshot_written_at"] is None
    assert result_live.data["cache_age_seconds"] is None


def test_write_status_snapshot_does_not_recurse(tmp_path: Path) -> None:
    """``write_status_snapshot`` calls ``status(use_cache=False)`` — it must
    not serve from a pre-existing stale snapshot (which would write stale
    data instead of a fresh live computation)."""
    app, gh = _make_app(tmp_path)
    # Plant a stale snapshot with bogus data.
    _write_snapshot(tmp_path, {"ready_issue_count": 999}, age_seconds=999)

    status_snapshot.write_status_snapshot(app)

    # The written snapshot should contain the LIVE data (1 issue), not the
    # stale bogus data (999).
    snapshot_path = status_snapshot.snapshot_path(app.paths.root)
    envelope = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert envelope["data"]["ready_issue_count"] == 1
    # And the live computation must have run (GitHub API calls were made).
    assert gh.issue_list_calls > 0


# ---------------------------------------------------------------------------
# Issue #1463 review: cache freshness surfaced into the data dict
# ---------------------------------------------------------------------------


def test_status_cached_data_includes_freshness_fields(tmp_path: Path) -> None:
    """A cached response includes ``snapshot_written_at`` and
    ``cache_age_seconds`` in the ``data`` dict so consumers (heartbeat_check,
    humans) can distinguish a cached response from a live one without
    inspecting ``CommandResult.message`` (which ``run_fleet_status``
    discards)."""
    app, _gh = _make_app(tmp_path)
    _write_snapshot(tmp_path, {"ready_issue_count": 5}, age_seconds=30)

    result = app.status()

    assert result.ok is True
    assert result.data["snapshot_written_at"] is not None
    # cache_age_seconds should be approximately 30 (within a tolerance).
    assert 25 <= result.data["cache_age_seconds"] <= 60


def test_status_live_data_includes_null_freshness_fields(tmp_path: Path) -> None:
    """A live (non-cached) response includes ``snapshot_written_at=None`` and
    ``cache_age_seconds=None`` so consumers can positively identify a fresh
    computation, not merely infer it from the absence of a cache."""
    app, _gh = _make_app(tmp_path)

    result = app.status()

    assert result.ok is True
    assert result.data["snapshot_written_at"] is None
    assert result.data["cache_age_seconds"] is None


# ---------------------------------------------------------------------------
# Issue #1463 review: integration tests driving a full app.loop() pass
# ---------------------------------------------------------------------------


def _build_loop_app(root: Path, *, dry_run: bool) -> OrchestratorApp:
    """Build an OrchestratorApp wired for a minimal ``loop()`` pass.

    Mirrors ``test_write_gate_dry_run_loop.py``'s ``_build_app``: disables the
    four cadence-gated lanes that emit events unconditionally on a due pass
    (deescalation, worktree_reclamation, main_ci_reclaim, reconcile_pass) so
    the snapshot-write is the only filesystem artifact this test needs to
    reason about. Uses the shared ``FakeGitHub`` from ``_fakes_github``.
    """
    from _fakes_github import FakeGitHub

    config = OrchestratorConfig(
        deescalation=DeescalationConfig(enabled=False),
        worktree_reclamation=WorktreeReclamationConfig(enabled=False),
        main_ci_reclaim=MainCiReclaimConfig(enabled=False),
        reconcile_pass=ReconcilePassConfig(enabled=False),
    )
    paths = runtime_paths(root, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    save_state(paths.state_file, empty_state())

    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = []

    app = OrchestratorApp(root, paths, config, fake_gh, dry_run=dry_run)
    # The sessions dir must exist so iter_workers does not error.
    sessions_dir = root / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return app


def test_loop_pass_writes_status_snapshot(tmp_path: Path) -> None:
    """A full ``app.loop()`` pass (non-dry-run) writes
    ``status-snapshot.json`` as a side effect — the real production wiring
    (``_loop_impl`` -> ``status_snapshot.write_status_snapshot``), not a
    direct unit call."""
    frozen_now = datetime.now(UTC) + timedelta(hours=1)
    app = _build_loop_app(tmp_path / "live", dry_run=False)
    snapshot_path = status_snapshot.snapshot_path(app.paths.root)

    assert not snapshot_path.exists(), "precondition: no snapshot yet"

    result = app.loop(limit=0, now=frozen_now)

    assert result.ok, f"loop pass must succeed for the snapshot write to run: {result.message}"
    assert snapshot_path.exists(), (
        "a non-dry-run loop pass must write status-snapshot.json so "
        "`fleet status --json` can serve from the cache"
    )
    envelope = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert "snapshot_written_at" in envelope
    assert isinstance(envelope["data"], dict)


def test_loop_pass_dry_run_does_not_write_status_snapshot(tmp_path: Path) -> None:
    """A ``dry_run=True`` ``app.loop()`` pass must NOT write
    ``status-snapshot.json`` — the dry-run invariant (cf. #1412, #1413) that
    gates every other loop-pass filesystem write. A dry-run pass writing the
    snapshot would poison the cache with preview-only data that a subsequent
    ``fleet status --json`` would serve as if it were real."""
    frozen_now = datetime.now(UTC) + timedelta(hours=1)
    app = _build_loop_app(tmp_path / "dry", dry_run=True)
    snapshot_path = status_snapshot.snapshot_path(app.paths.root)

    assert not snapshot_path.exists(), "precondition: no snapshot yet"

    result = app.loop(limit=0, now=frozen_now)

    assert result.ok, f"loop pass must succeed even in dry-run: {result.message}"
    assert not snapshot_path.exists(), (
        "a dry_run=True loop pass must not write status-snapshot.json — "
        "the dry-run invariant (cf. #1412, #1413) gates every loop-pass "
        "filesystem write, including the status-snapshot cache"
    )


# ---------------------------------------------------------------------------
# Issue #1463 review: config-validation tests for status_snapshot_ttl_seconds
# ---------------------------------------------------------------------------


def test_status_snapshot_ttl_seconds_rejects_non_int() -> None:
    """``runtime.status_snapshot_ttl_seconds`` must be an int; a string value
    must raise ConfigError."""
    with pytest.raises(ConfigError, match="status_snapshot_ttl_seconds.*must be an int"):
        build_config_from_data({"runtime": {"status_snapshot_ttl_seconds": "900"}})


def test_status_snapshot_ttl_seconds_rejects_bool() -> None:
    """``bool`` is a subclass of ``int`` in Python but must be rejected —
    ``True`` would silently mean a 1-second TTL."""
    with pytest.raises(ConfigError, match="status_snapshot_ttl_seconds.*must be an int"):
        build_config_from_data({"runtime": {"status_snapshot_ttl_seconds": True}})


def test_status_snapshot_ttl_seconds_rejects_negative() -> None:
    """A negative TTL is nonsensical; must raise ConfigError."""
    with pytest.raises(ConfigError, match="status_snapshot_ttl_seconds.*must be >= 0"):
        build_config_from_data({"runtime": {"status_snapshot_ttl_seconds": -1}})


# ---------------------------------------------------------------------------
# Issue #1463 review: --no-cache CLI flag threads through to status()
# ---------------------------------------------------------------------------


def _write_fleet_registry_with_one_repo(tmp_path: Path) -> Path:
    """Write a fleet.json (under the conftest-redirected fleet dir) with one
    valid repo entry whose ``repo_root`` exists, so ``run_fleet_status`` reaches
    the ``OrchestratorApp.status()`` call instead of skipping it as stale.
    """
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir(parents=True, exist_ok=True)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    now = datetime.now(UTC)
    entries = {
        "owner/repo": {
            "repo_root": str(repo_root),
            "name_with_owner": "owner/repo",
            "config_path": str(repo_root / "orchestrator.config.yaml"),
            "state_dir": str(repo_root / ".var" / "charlie-work"),
            "first_seen": now.isoformat().replace("+00:00", "Z"),
            "last_seen": now.isoformat().replace("+00:00", "Z"),
        },
    }
    (fleet_dir / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": entries}), encoding="utf-8"
    )
    return repo_root


@contextmanager
def _mock_fleet_status_deps():
    """Patch the cli-module deps ``run_fleet_status`` uses so it never touches
    the filesystem or GitHub, and yield the mock ``OrchestratorApp`` instance
    so the caller can inspect ``status.call_args``.

    Mirrors the patch surface in ``test_issue_1372_fleet_registry_stale.py``'s
    ``run_fleet_status`` test: ``load_layered_config``, ``runtime_paths``,
    ``GitHub``, ``OrchestratorApp``, ``compute_api_worker_fleet_report``.
    """
    with (
        patch("charlie_work.cli.load_layered_config") as mock_cfg,
        patch("charlie_work.cli.runtime_paths") as mock_paths,
        patch("charlie_work.cli.GitHub") as mock_gh,
        patch("charlie_work.cli.OrchestratorApp") as mock_app,
        patch("charlie_work.cli.compute_api_worker_fleet_report") as mock_report,
    ):
        mock_cfg.return_value = OrchestratorConfig()
        mock_p = MagicMock()
        mock_p.root = Path(".")
        mock_paths.return_value = mock_p
        mock_gh.return_value = MagicMock()
        mock_a = MagicMock()
        mock_a.status.return_value = MagicMock(ok=True, data={"ready_issue_count": 0})
        mock_app.return_value = mock_a
        mock_report.return_value = None
        yield mock_a


@pytest.mark.parametrize(
    ("argv", "expected_use_cache"),
    [
        (["fleet", "status"], True),
        (["fleet", "status", "--no-cache"], False),
    ],
    ids=["default-threads-use_cache-True", "no-cache-threads-use_cache-False"],
)
def test_no_cache_cli_flag_threads_use_cache_into_status(
    tmp_path: Path, argv: list[str], expected_use_cache: bool
) -> None:
    """Issue #1463 review: the ``--no-cache`` CLI flag is the operator-facing
    mechanism this issue calls for. This test drives it end-to-end through
    ``build_parser()`` / ``run_fleet_status`` (not just ``app.status(...)``
    directly), confirming:

    * ``fleet status`` (no flag) threads ``use_cache=True`` into
      ``OrchestratorApp.status()`` — the cache-serving default.
    * ``fleet status --no-cache`` threads ``use_cache=False`` — the live
      bypass.

    Both branches are exercised by parametrization so a regression in either
    direction (flag dropped, flag inverted, default flipped) is caught.
    """
    _write_fleet_registry_with_one_repo(tmp_path)
    args = build_parser().parse_args(argv)

    with _mock_fleet_status_deps() as mock_app:
        result = run_fleet_status(args)

    assert result.ok is True
    mock_app.status.assert_called_once()
    assert mock_app.status.call_args.kwargs.get("use_cache") is expected_use_cache, (
        f"argv={argv} should thread use_cache={expected_use_cache} into "
        f"OrchestratorApp.status(); got call_args={mock_app.status.call_args}"
    )


# ---------------------------------------------------------------------------
# Issue #1463 round-3 review: roll-call --no-cache parity + freshness consumers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected_use_cache"),
    [
        (["roll-call"], True),
        (["roll-call", "--no-cache"], False),
    ],
    ids=["roll-call-default-True", "roll-call-no-cache-False"],
)
def test_roll_call_no_cache_flag_threads_use_cache(
    argv: list[str], expected_use_cache: bool
) -> None:
    """Round-3 review: ``roll-call`` silently gained cache-serving behavior
    when ``status()`` grew its ``use_cache`` default. This test confirms
    ``roll-call`` now has ``--no-cache`` parity with ``fleet status``: the
    default threads ``use_cache=True``, and ``--no-cache`` threads ``False``.
    """
    from charlie_work.cli import run_command

    args = build_parser().parse_args(argv)
    mock_app = MagicMock()
    mock_app.status.return_value = MagicMock(ok=True, data={})

    run_command(mock_app, args)

    mock_app.status.assert_called_once()
    assert mock_app.status.call_args.kwargs.get("use_cache") is expected_use_cache, (
        f"argv={argv} should thread use_cache={expected_use_cache} into "
        f"OrchestratorApp.status(); got call_args={mock_app.status.call_args}"
    )


def test_fleet_status_human_readable_shows_cache_age_when_cached(
    monkeypatch, capsys, tmp_path
) -> None:
    """Round-3 review: the human-readable (non ``--json``) fleet status
    renderer must print cache freshness so an operator can distinguish a
    cached response from a live one — the ``snapshot_written_at`` /
    ``cache_age_seconds`` fields round-1 required need a real consumer."""
    from charlie_work import cli
    from charlie_work.workflow import CommandResult

    fake_result = CommandResult(
        True,
        "fleet status: 1 repo(s), 0 error(s), 0 stale(s)",
        {
            "repos": {
                "owner/repo": {
                    "ready_issue_count": 5,
                    "active_issue_count": 2,
                    "blocked": [],
                    "stalled": [],
                    "cache_age_seconds": 42.0,
                }
            },
            "errors": [],
            "stale": [],
            "api_worker_report": None,
        },
    )
    monkeypatch.setattr(cli, "run_fleet_status", lambda args: fake_result)

    rc = cli.main(["fleet", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "(cached, 42s old)" in out, (
        f"human-readable fleet status must show cache age for cached repos; got:\n{out}"
    )


def test_fleet_status_human_readable_omits_cache_age_when_live(
    monkeypatch, capsys, tmp_path
) -> None:
    """A live (non-cached) response has ``cache_age_seconds=None``; the
    renderer must NOT print a cache-age line, so a live response is
    distinguishable from a cached one by the absence of the line."""
    from charlie_work import cli
    from charlie_work.workflow import CommandResult

    fake_result = CommandResult(
        True,
        "fleet status: 1 repo(s), 0 error(s), 0 stale(s)",
        {
            "repos": {
                "owner/repo": {
                    "ready_issue_count": 5,
                    "active_issue_count": 2,
                    "blocked": [],
                    "stalled": [],
                    "cache_age_seconds": None,
                }
            },
            "errors": [],
            "stale": [],
            "api_worker_report": None,
        },
    )
    monkeypatch.setattr(cli, "run_fleet_status", lambda args: fake_result)

    rc = cli.main(["fleet", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cached" not in out.lower(), (
        f"human-readable fleet status must not show cache age for live repos; got:\n{out}"
    )
