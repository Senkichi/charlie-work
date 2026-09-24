"""Notify-digest staleness dead-man's-switch tests (issue #1859).

Covers ``check_notify_digest_freshness`` (once-per-fleet-pass probe that
emits at most one ``notify_digest_stale`` warning event per stale episode),
its ``fleet_loop`` wiring gate, and the emit_digest isolation added to
``_patch_self_deploy_for_fleet_tests`` so suite runs can no longer append
synthetic entries to a checkout's real digest.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from _fleet_dispatch_fixtures import (
    ISOLATED_NOTIFY_DIGEST_REL,
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)
from charlie_work import notify_freshness
from charlie_work.config import NotifyConfig
from charlie_work.fleet_dispatch import fleet_loop
from charlie_work.notify_freshness import check_notify_digest_freshness
from charlie_work.notify import AttentionDigest, AttentionEntry


@pytest.fixture(autouse=True)
def _clear_stale_latch() -> Any:
    """The episode latch is module state; keep tests independent of order."""
    notify_freshness._stale_episodes.clear()
    yield
    notify_freshness._stale_episodes.clear()


def _config(tmp_path: Path, **kwargs: Any) -> NotifyConfig:
    defaults: dict[str, Any] = dict(
        enabled=True, sink="file", file_path=str(tmp_path / "digest.jsonl")
    )
    defaults.update(kwargs)
    return NotifyConfig(**defaults)


def _patch_log_event(monkeypatch: Any) -> MagicMock:
    log_event = MagicMock(name="log_event")
    monkeypatch.setattr("charlie_work.notify_freshness.log_event", log_event)
    return log_event


def test_stale_check_emits_event_when_digest_missing(tmp_path: Path, monkeypatch: Any) -> None:
    log_event = _patch_log_event(monkeypatch)
    check_notify_digest_freshness(_config(tmp_path), tmp_path / "state.json", now=1000.0)
    log_event.assert_called_once()
    args = log_event.call_args[0]
    assert args[0] == tmp_path / "state.json"
    assert args[1] == "notify_digest_stale"
    assert args[2]["exists"] is False


def test_stale_check_emits_event_when_digest_older_than_bound(
    tmp_path: Path, monkeypatch: Any
) -> None:
    digest = tmp_path / "digest.jsonl"
    digest.write_text("{}\n", encoding="utf-8")
    stale_instant = digest.stat().st_mtime + notify_freshness.NOTIFY_DIGEST_STALE_SECONDS + 1
    log_event = _patch_log_event(monkeypatch)
    check_notify_digest_freshness(_config(tmp_path), tmp_path / "state.json", now=stale_instant)
    log_event.assert_called_once()
    assert log_event.call_args[0][1] == "notify_digest_stale"


def test_stale_check_silent_when_digest_fresh(tmp_path: Path, monkeypatch: Any) -> None:
    digest = tmp_path / "digest.jsonl"
    digest.write_text("{}\n", encoding="utf-8")
    log_event = _patch_log_event(monkeypatch)
    check_notify_digest_freshness(
        _config(tmp_path), tmp_path / "state.json", now=digest.stat().st_mtime + 60
    )
    log_event.assert_not_called()


def test_stale_check_emits_once_per_episode(tmp_path: Path, monkeypatch: Any) -> None:
    """A persistently-stale digest fires one event per ~day, not one per pass."""
    log_event = _patch_log_event(monkeypatch)
    cfg = _config(tmp_path)
    check_notify_digest_freshness(cfg, tmp_path / "state.json", now=1000.0)
    check_notify_digest_freshness(cfg, tmp_path / "state.json", now=2000.0)
    log_event.assert_called_once()


def test_stale_check_refires_after_recovery(tmp_path: Path, monkeypatch: Any) -> None:
    """A fresh digest clears the episode so a *new* outage re-fires."""
    digest = tmp_path / "digest.jsonl"
    log_event = _patch_log_event(monkeypatch)
    cfg = _config(tmp_path)
    check_notify_digest_freshness(cfg, tmp_path / "state.json", now=1000.0)
    digest.write_text("{}\n", encoding="utf-8")
    fresh_instant = digest.stat().st_mtime + 60
    check_notify_digest_freshness(cfg, tmp_path / "state.json", now=fresh_instant)
    assert log_event.call_count == 1
    digest.unlink()
    check_notify_digest_freshness(cfg, tmp_path / "state.json", now=fresh_instant + 60)
    assert log_event.call_count == 2


def test_stale_check_noop_for_non_file_sink(tmp_path: Path, monkeypatch: Any) -> None:
    log_event = _patch_log_event(monkeypatch)
    check_notify_digest_freshness(
        _config(tmp_path, sink="webhook", webhook_url="https://example.test/hook"),
        tmp_path / "state.json",
        now=1000.0,
    )
    log_event.assert_not_called()


def test_stale_check_noop_when_file_path_unset(tmp_path: Path, monkeypatch: Any) -> None:
    """enabled/file/empty-path is reported at supervisor startup instead."""
    log_event = _patch_log_event(monkeypatch)
    check_notify_digest_freshness(
        _config(tmp_path, file_path=""), tmp_path / "state.json", now=1000.0
    )
    log_event.assert_not_called()


def test_stale_check_swallows_probe_errors(tmp_path: Path, monkeypatch: Any) -> None:
    """The check itself must never take down the pass it rides on."""
    monkeypatch.setattr(
        "charlie_work.notify_freshness.digest_file_status",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    check_notify_digest_freshness(_config(tmp_path), tmp_path / "state.json", now=1000.0)


def test_stale_check_swallows_log_event_errors(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "charlie_work.notify_freshness.log_event",
        MagicMock(side_effect=RuntimeError("db gone")),
    )
    check_notify_digest_freshness(_config(tmp_path), tmp_path / "state.json", now=1000.0)


# ---------------------------------------------------------------------------
# fleet_loop wiring gate
# ---------------------------------------------------------------------------


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_runs_stale_check_when_notify_enabled(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    mock_load_registry.return_value = {"repos": {}}
    check = MagicMock(name="check_notify_digest_freshness")
    monkeypatch.setattr("charlie_work.fleet_dispatch.check_notify_digest_freshness", check)
    notify = NotifyConfig(enabled=True, sink="file", file_path="digest.jsonl")
    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=SimpleNamespace(notify=notify),
        repos=None,
        limit=1,
        merge=None,
        dry_run=True,
        work_only=True,
    )
    check.assert_called_once()
    assert check.call_args[0][0] is notify


@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_skips_stale_check_when_notify_disabled(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    mock_load_registry.return_value = {"repos": {}}
    check = MagicMock(name="check_notify_digest_freshness")
    monkeypatch.setattr("charlie_work.fleet_dispatch.check_notify_digest_freshness", check)
    fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=SimpleNamespace(notify=NotifyConfig()),
        repos=None,
        limit=1,
        merge=None,
        dry_run=True,
        work_only=True,
    )
    check.assert_not_called()


# ---------------------------------------------------------------------------
# Fixture isolation: the "test no-op" leak into the real digest (issue #1859)
# ---------------------------------------------------------------------------


def test_autouse_fixture_redirects_file_sink_to_tmp_path(tmp_path: Path) -> None:
    """Regression: fleet emits under the suite land in the test's tmp dir, never
    at the config's file_path -- the leak that wrote ``"test no-op"`` entries
    into the development checkout's real digest for a month."""
    # Import inside the test body so the name binds the autouse-patched
    # wrapper, not the real emit_digest captured at collection time.
    from charlie_work.fleet_dispatch import emit_digest

    real_path = tmp_path / "real-checkout" / "digest.jsonl"
    config = NotifyConfig(enabled=True, sink="file", file_path=str(real_path))
    digest = AttentionDigest(
        generated_at="2026-09-18T00:00:00Z",
        repo="fleet",
        transitions=(
            AttentionEntry(
                issue_number=1,
                adapter_kind="owner/repo",
                health="STALLED",
                previous_health="RUNNING",
                last_log_line="test no-op",
                pid=None,
            ),
        ),
    )
    result = emit_digest(config, digest)
    assert result.ok is True
    assert not real_path.exists()
    isolated = tmp_path / ISOLATED_NOTIFY_DIGEST_REL
    assert isolated.exists()
    assert "test no-op" in isolated.read_text(encoding="utf-8")


def test_autouse_fixture_noops_non_file_sinks(tmp_path: Path) -> None:
    """Shell/desktop/webhook sinks are real side effects too; the fixture
    suppresses them rather than redirecting."""
    from charlie_work.fleet_dispatch import emit_digest

    digest = AttentionDigest(
        generated_at="2026-09-18T00:00:00Z",
        repo="fleet",
        transitions=(),
    )
    config = NotifyConfig(enabled=True, sink="shell", shell_command=("false",))
    result = emit_digest(config, digest)
    assert result.ok is True


def test_autouse_fixture_preserves_empty_file_path_error(tmp_path: Path) -> None:
    """enabled/file/empty-path still returns the real 'file_path is empty'
    error -- the redirect must not mask the misconfiguration."""
    from charlie_work.fleet_dispatch import emit_digest

    digest = AttentionDigest(
        generated_at="2026-09-18T00:00:00Z",
        repo="fleet",
        transitions=(),
    )
    config = NotifyConfig(enabled=True, sink="file")
    result = emit_digest(config, digest)
    assert result.ok is False
    assert "file_path is empty" in (result.error or "")
