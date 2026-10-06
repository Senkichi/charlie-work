"""Regression tests for issue #1941: ``run_fleet_status`` per-repo fan-out.

Before the fix, ``run_fleet_status`` walked the fleet registry serially and
called ``app.status()`` per repo with no per-repo bound, so a fleet-wide
status-snapshot TTL miss summed every repo's live recompute (measured at
~45-100s each) against the heartbeat's 120s ``charlie fleet status --json``
subprocess cap. These tests pin the two halves of the fix:

* concurrent execution — wall time is max(per-repo cost), not the sum; and
* a per-repo budget — a wedged repo lands in ``stale`` (not ``errors``,
  so it does not flip the exit code) and does not starve repos that
  already finished.

Every fake ``status()`` below returns within a bounded time even though the
assertions exercise the timeout, so the tests still terminate (and fail on
assertion) against the pre-fix serial implementation instead of hanging.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from charlie_work import cli, fleet_status
from charlie_work.config import OrchestratorConfig
from charlie_work.github import GitHubError
from charlie_work.workflow import CommandResult


def _fleet_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_by_repo: dict[str, Callable[[], CommandResult]],
) -> None:
    """Register repos in a fleet dir and stub the fleet_status-module seams
    ``run_fleet_status`` uses so each repo's ``app.status()`` runs the
    supplied callable.

    ``status_by_repo`` maps fleet repo_key -> zero-arg callable returning a
    ``CommandResult`` (or raising). ``OrchestratorApp`` is patched with a
    factory that dispatches on the ``repo_root`` it is constructed with.
    """
    fleet_dir = tmp_path / "fleet"
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    entries: dict[str, dict] = {}
    for repo_key in status_by_repo:
        repo_root = tmp_path / repo_key.replace("/", "-")
        repo_root.mkdir()
        entries[repo_key] = {
            "repo_root": str(repo_root),
            "name_with_owner": repo_key,
            "config_path": str(repo_root / "orchestrator.config.yaml"),
            "state_dir": str(repo_root / ".var" / "charlie-work"),
            "first_seen": "2026-09-26T00:00:00Z",
            "last_seen": "2026-09-26T00:00:00Z",
        }
    fleet_dir.mkdir(parents=True, exist_ok=True)
    (fleet_dir / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": entries}, indent=2), encoding="utf-8"
    )

    key_by_root = {entry["repo_root"]: key for key, entry in entries.items()}

    monkeypatch.setattr(fleet_status, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    monkeypatch.setattr(fleet_status, "runtime_paths", lambda *a, **k: MagicMock())
    monkeypatch.setattr(fleet_status, "github_client_for", lambda *a, **k: MagicMock())

    def _app(repo_root: Path, *_args: object, **_kwargs: object) -> MagicMock:
        app = MagicMock()
        behavior = status_by_repo[key_by_root[str(repo_root)]]
        app.status.side_effect = lambda **_kw: behavior()
        return app

    monkeypatch.setattr(fleet_status, "OrchestratorApp", _app)
    monkeypatch.setattr(fleet_status, "compute_api_worker_fleet_report", lambda *a, **k: None)


def _args() -> argparse.Namespace:
    return cli.build_parser().parse_args(["fleet", "status"])


def test_fleet_status_runs_repo_status_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two repos whose status() each waits for the other to start only both
    finish if the calls run concurrently; a serial walk eats the full wait
    on the first repo alone.
    """
    both_entered = threading.Event()
    entered: list[str] = []

    def _gated(repo_key: str):
        def _status() -> CommandResult:
            entered.append(repo_key)
            if len(entered) == 2:
                both_entered.set()
            # Bounded so the serial implementation still returns (after ~20s)
            # rather than deadlocking the test run.
            both_entered.wait(timeout=10)
            return CommandResult(True, "ok", {"repo": repo_key})

        return _status

    _fleet_env(
        tmp_path,
        monkeypatch,
        {
            "owner/a": _gated("owner/a"),
            "owner/b": _gated("owner/b"),
        },
    )

    start = time.monotonic()
    result = cli.run_fleet_status(_args())
    elapsed = time.monotonic() - start

    assert result.ok is True
    assert set(result.data["repos"]) == {"owner/a", "owner/b"}
    # Concurrent: ~instant (event gates both). Serial: ~20s (two 10s waits).
    assert elapsed < 8.0, (
        f"fleet status took {elapsed:.1f}s — per-repo status() calls are not "
        "running concurrently (serial walk would take ~20s here)"
    )


def test_fleet_status_repo_timeout_lands_in_stale_not_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo whose status() exceeds the per-repo budget is reported in
    ``stale`` (not ``errors``), does not flip ok/exit-code, and does not
    block the aggregate past the budget.
    """

    release = threading.Event()

    def _wedged() -> CommandResult:
        release.wait(6)  # bounded: a serial implementation still returns
        return CommandResult(True, "ok", {"repo": "owner/wedged"})

    _fleet_env(
        tmp_path,
        monkeypatch,
        {
            "owner/fast": lambda: CommandResult(True, "ok", {"repo": "owner/fast"}),
            "owner/wedged": _wedged,
        },
    )
    monkeypatch.setattr(fleet_status, "FLEET_STATUS_REPO_TIMEOUT_SECONDS", 0.5)

    start = time.monotonic()
    try:
        result = cli.run_fleet_status(_args())
    finally:
        release.set()  # the abandoned worker thread exits now, not 6 s later
    elapsed = time.monotonic() - start

    assert result.ok is True
    assert elapsed < 5.0, (
        f"fleet status took {elapsed:.1f}s — the wedged repo blocked the "
        "aggregate instead of being cut at the per-repo budget"
    )
    assert "owner/fast" in result.data["repos"]
    assert "owner/wedged" not in result.data["repos"]
    stale_by_key = {s["repo_key"]: s for s in result.data["stale"]}
    assert stale_by_key["owner/wedged"]["reason"] == "status_timeout"
    assert stale_by_key["owner/wedged"]["repo_root"]
    assert result.data["errors"] == []
    assert "1 stale(s)" in result.message


def test_fleet_status_timeout_does_not_starve_completed_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo that finished while a slower earlier-keyed repo was still
    running is still collected — the shared deadline cuts only the repo
    that actually blew its budget.
    """

    release = threading.Event()

    def _wedged() -> CommandResult:
        release.wait(6)
        return CommandResult(True, "ok", {"repo": "owner/a-wedged"})

    _fleet_env(
        tmp_path,
        monkeypatch,
        {
            "owner/a-wedged": _wedged,
            "owner/z-fast": lambda: CommandResult(True, "ok", {"repo": "owner/z-fast"}),
        },
    )
    monkeypatch.setattr(fleet_status, "FLEET_STATUS_REPO_TIMEOUT_SECONDS", 0.5)

    try:
        result = cli.run_fleet_status(_args())
    finally:
        release.set()

    assert result.ok is True
    assert "owner/z-fast" in result.data["repos"]
    assert "owner/a-wedged" not in result.data["repos"]
    assert {s["repo_key"] for s in result.data["stale"]} == {"owner/a-wedged"}


def test_fleet_status_worker_exception_routes_to_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exceptions raised inside a worker are delivered through its future:
    the repo lands in ``errors`` (flipping ok, same as the serial loop) while
    other repos still aggregate.
    """

    def _boom() -> CommandResult:
        raise GitHubError("gh exploded")

    _fleet_env(
        tmp_path,
        monkeypatch,
        {
            "owner/bad": _boom,
            "owner/good": lambda: CommandResult(True, "ok", {"repo": "owner/good"}),
        },
    )

    result = cli.run_fleet_status(_args())

    assert result.ok is False
    assert "owner/good" in result.data["repos"]
    errors_by_key = {e["repo_key"]: e for e in result.data["errors"]}
    assert errors_by_key["owner/bad"]["error"] == "gh exploded"
    assert result.data["stale"] == []


def test_fleet_status_worker_timeout_error_routes_to_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker exception that is itself TimeoutError-shaped must land in
    ``errors`` (flipping ok), not be misread as the collection deadline
    expiring into ``stale``.

    ``Future.result(timeout=...)`` raises ``TimeoutError`` for BOTH an
    expired wait and a worker whose stored exception is ``TimeoutError``
    (an ``OSError`` — the type a wedged ``gh`` subprocess surfaces), so the
    collector decides budget expiry from ``wait()``'s return, not from
    exception identity. A budget is patched in anyway so that a regression
    to exception-identity classification could not hide behind the two
    branches agreeing.
    """

    def _timeout() -> CommandResult:
        raise TimeoutError("gh timed out")

    _fleet_env(
        tmp_path,
        monkeypatch,
        {
            "owner/bad": _timeout,
            "owner/good": lambda: CommandResult(True, "ok", {"repo": "owner/good"}),
        },
    )
    monkeypatch.setattr(fleet_status, "FLEET_STATUS_REPO_TIMEOUT_SECONDS", 0.5)

    result = cli.run_fleet_status(_args())

    assert result.ok is False
    assert "owner/good" in result.data["repos"]
    errors_by_key = {e["repo_key"]: e for e in result.data["errors"]}
    assert errors_by_key["owner/bad"]["error"] == "gh timed out"
    assert result.data["stale"] == []


def test_fleet_status_reexported_from_cli() -> None:
    """cli.py stays the command surface: ``run_fleet_status`` and the
    per-repo budget constant are re-exported from the ``fleet_status``
    domain module (the ``run_fleet_stop`` precedent)."""
    assert cli.run_fleet_status is fleet_status.run_fleet_status
    assert cli.FLEET_STATUS_REPO_TIMEOUT_SECONDS is (
        fleet_status.FLEET_STATUS_REPO_TIMEOUT_SECONDS
    )
