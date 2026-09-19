"""``_pull_ci_fleet_sibling`` tests (issue #552 deploy-clone half).

Split out of ``tests/test_supervise.py`` (issue #1562, Track 1) --
bodies are verbatim relocations; shared helpers live in
``tests/_supervise_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from _supervise_fixtures import (
    _make_fake_runner,
    no_fleet_live_sessions as no_fleet_live_sessions,
)
from charlie_work.subprocess_runner import RunResult
from charlie_work.supervise import _pull_ci_fleet_sibling, self_deploy


@pytest.fixture
def captured_log_events(monkeypatch: Any) -> list[tuple[Path, str, dict[str, Any]]]:
    """Capture (state_path, kind, payload) tuples instead of writing to events.db."""
    calls: list[tuple[Path, str, dict[str, Any]]] = []

    def _fake_log_event(
        state_path: Path, kind: str, payload: dict[str, Any], **_kwargs: Any
    ) -> None:
        calls.append((state_path, kind, payload))

    monkeypatch.setattr("charlie_work.supervise.log_event", _fake_log_event)
    return calls


def _fake_declared_root(monkeypatch: Any, sibling_src: Path | None) -> None:
    """Patch declared_ci_fleet_root at its source module.

    ``_pull_ci_fleet_sibling`` imports it lazily inside its body
    (``from .ci_fleet_anchor import declared_ci_fleet_root``), so it must be
    patched at ``charlie_work.ci_fleet_anchor``, not at ``charlie_work.supervise``.
    """
    monkeypatch.setattr("charlie_work.ci_fleet_anchor.declared_ci_fleet_root", lambda: sibling_src)


def test_pull_ci_fleet_sibling_happy_path(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """Clean main sibling, sha moves -> one ok/changed events.db entry."""
    sibling = tmp_path / "ci-fleet"
    _fake_declared_root(monkeypatch, sibling / "src")

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "main\n", ""),  # branch
            RunResult(0, "", ""),  # status --porcelain (clean)
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "def456\n", ""),  # after HEAD
        ]
    )

    outcome = _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    assert outcome is None
    assert len(captured_log_events) == 1
    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is True
    assert payload["changed"] is True
    assert payload["from_sha"] == "abc123"
    assert payload["to_sha"] == "def456"
    assert payload["from_sha"] != payload["to_sha"]
    assert len(calls) == 5
    assert all(c[1] == sibling for c in calls)


def test_pull_ci_fleet_sibling_unchanged_pull(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """Pull succeeds but the sha does not move -> ok True, changed False."""
    sibling = tmp_path / "ci-fleet"
    _fake_declared_root(monkeypatch, sibling / "src")

    runner, _ = _make_fake_runner(
        [
            RunResult(0, "main\n", ""),
            RunResult(0, "", ""),
            RunResult(0, "abc123\n", ""),
            RunResult(0, "Already up to date.\n", ""),
            RunResult(0, "abc123\n", ""),
        ]
    )

    _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is True
    assert payload["changed"] is False
    assert payload["from_sha"] == payload["to_sha"] == "abc123"


def test_pull_ci_fleet_sibling_skips_non_main_branch(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """A sibling not on main is skipped -- pull is never issued.

    This pins the fail-safe precondition: only two canned responses (branch,
    status) are supplied, so if the branch check were deleted the function
    would try to consume a third response and the fake runner's ``pop(0)``
    would raise ``IndexError``, failing the test.
    """
    sibling = tmp_path / "ci-fleet"
    _fake_declared_root(monkeypatch, sibling / "src")

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "feature-branch\n", ""),  # branch
            RunResult(0, "", ""),  # status --porcelain
        ]
    )

    _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is False
    assert "feature-branch" in payload["skipped_reason"]
    assert all(c[0] != ["git", "pull", "--ff-only", "origin", "main"] for c in calls)
    assert len(calls) == 2


def test_pull_ci_fleet_sibling_skips_dirty_tree(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """A dirty sibling tree is skipped -- pull is never issued.

    Only two canned responses are supplied, pinning the precondition the same
    way as the non-main-branch case above.
    """
    sibling = tmp_path / "ci-fleet"
    _fake_declared_root(monkeypatch, sibling / "src")

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "main\n", ""),
            RunResult(0, " M some_file.py\n", ""),  # dirty
        ]
    )

    _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is False
    assert payload["skipped_reason"] == "sibling tree is dirty"
    assert all(c[0] != ["git", "pull", "--ff-only", "origin", "main"] for c in calls)
    assert len(calls) == 2


def test_pull_ci_fleet_sibling_skips_when_no_declared_root(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """No declared ci-fleet source -- skipped, no git commands issued at all."""
    _fake_declared_root(monkeypatch, None)

    runner, calls = _make_fake_runner([])

    _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is False
    assert "no declared" in payload["skipped_reason"]
    assert calls == []


def test_pull_ci_fleet_sibling_pull_failure_is_non_fatal(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """A failed pull is recorded with an error, event still emitted, no raise."""
    sibling = tmp_path / "ci-fleet"
    _fake_declared_root(monkeypatch, sibling / "src")

    runner, _ = _make_fake_runner(
        [
            RunResult(0, "main\n", ""),
            RunResult(0, "", ""),
            RunResult(0, "abc123\n", ""),
            RunResult(1, "", "fatal: could not read from remote repository."),
            RunResult(0, "abc123\n", ""),
        ]
    )

    outcome = _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    assert outcome is None
    assert len(captured_log_events) == 1
    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is False
    assert payload.get("error")


def test_pull_ci_fleet_sibling_declared_root_raises_is_caught(
    tmp_path: Path,
    monkeypatch: Any,
    captured_log_events: list[tuple[Path, str, dict[str, Any]]],
) -> None:
    """An exception from declared_ci_fleet_root is caught, logged, never propagates."""

    def _boom() -> Path | None:
        raise RuntimeError("boom")

    monkeypatch.setattr("charlie_work.ci_fleet_anchor.declared_ci_fleet_root", _boom)

    runner, calls = _make_fake_runner([])

    outcome = _pull_ci_fleet_sibling(tmp_path, run_command=runner, timeout=60)

    assert outcome is None
    _, kind, payload = captured_log_events[0]
    assert kind == "self_deploy_ci_fleet_pull"
    assert payload["ok"] is False
    assert "RuntimeError" in payload["error"]
    assert calls == []


def test_self_deploy_does_not_pull_ci_fleet_sibling_by_default(
    tmp_path: Path, monkeypatch: Any, no_fleet_live_sessions: None
) -> None:
    """self_deploy's default pull_ci_fleet=False never touches the sibling machinery.

    Patches ``_pull_ci_fleet_sibling`` with a fail-if-called stub and drives
    ``self_deploy`` (no ``pull_ci_fleet`` kwarg -> default False) with a
    successful, code-only pull -- confirming the sibling path is gated even
    when the orchestrator's own pull succeeds.
    """

    def _fail_if_called(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("_pull_ci_fleet_sibling must not be called when pull_ci_fleet=False")

    monkeypatch.setattr("charlie_work.supervise._pull_ci_fleet_sibling", _fail_if_called)

    runner, _ = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "def456\n", ""),  # after HEAD
            RunResult(0, "src/foo.py\n", ""),  # diff (code-only)
        ]
    )

    result = self_deploy(tmp_path, run_command=runner)

    assert result.ok is True
    assert result.pulled is True
    assert result.changed is True
