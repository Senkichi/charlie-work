"""Reap-path settlement wiring: WorkerView.reap_sidecar for adapter_kind == "api" (issue #480).

Split out of ``tests/test_api_budget.py`` (issue #1571, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_api_budget_unit_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work import api_budget
from charlie_work.api_budget import SessionEntry, ledger_path, load_ledger

from _api_budget_unit_fixtures import (
    _api_worker_view,
    _provider,
    _result_event,
    _write_api_sidecar,
    _write_events,
)


def test_reap_sidecar_settles_api_session_into_ledger(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    _write_api_sidecar(sessions_dir, 42, "example")
    _write_events(
        sessions_dir,
        42,
        [_result_event(input_tokens=1_000_000, output_tokens=200_000, cache_read=500_000)],
    )
    provider = _provider()
    api_config = type("C", (), {"providers": {"example": provider}})()
    view = _api_worker_view(sessions_dir, 42, "example")

    view.reap_sidecar(sessions_dir, api_config=api_config, state_dir=state_dir)

    ledger = load_ledger(ledger_path(state_dir))
    assert len(ledger.sessions) == 1
    entry = ledger.sessions[0]
    assert entry.issue == 42
    assert entry.provider == "example"
    assert entry.model == "example-model"
    assert entry.input_tokens == 1_000_000
    assert entry.output_tokens == 200_000
    assert entry.cached_tokens == 500_000
    # 1M*3 + 0.2M*15 + 0.5M*0.30 = 3 + 3 + 0.15 = 6.15
    assert entry.usd == pytest.approx(6.15)
    assert ledger.lifetime_usd == pytest.approx(6.15)
    # The sidecar was unlinked (reap still happens after settlement).
    from charlie_work.claude_code import _sidecar_path

    assert not _sidecar_path(sessions_dir, 42, "api").exists()


def test_reap_sidecar_settlement_is_idempotent_across_reaps(tmp_path: Path) -> None:
    """Settling the same session on two reaps does not double-count (idempotence
    via the ledger's (issue, started_at, session_id) key)."""
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    # First reap settles + unlinks the sidecar.
    _write_api_sidecar(sessions_dir, 42, "example")
    _write_events(sessions_dir, 42, [_result_event(input_tokens=1_000_000, output_tokens=0)])
    provider = _provider()
    api_config = type("C", (), {"providers": {"example": provider}})()
    view = _api_worker_view(sessions_dir, 42, "example")
    view.reap_sidecar(sessions_dir, api_config=api_config, state_dir=state_dir)
    # Re-create the sidecar (e.g. a second reap cycle) and reap again.
    _write_api_sidecar(sessions_dir, 42, "example")
    view.reap_sidecar(sessions_dir, api_config=api_config, state_dir=state_dir)

    ledger = load_ledger(ledger_path(state_dir))
    assert len(ledger.sessions) == 1
    assert ledger.lifetime_usd == pytest.approx(3.0)  # 1M * 3, counted once


def test_reap_sidecar_without_api_config_skips_settlement(tmp_path: Path) -> None:
    """Legacy callers (no api_config/state_dir) still reap; no ledger written."""
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    _write_api_sidecar(sessions_dir, 42, "example")
    _write_events(sessions_dir, 42, [_result_event(input_tokens=1_000_000, output_tokens=0)])
    view = _api_worker_view(sessions_dir, 42, "example")

    view.reap_sidecar(sessions_dir)  # no kwargs

    assert not ledger_path(state_dir).exists()
    from charlie_work.claude_code import _sidecar_path

    assert not _sidecar_path(sessions_dir, 42, "api").exists()


def test_reap_sidecar_unknown_provider_skips_settlement(tmp_path: Path) -> None:
    """A provider not in the registry → no pricing → skip settlement, still reap."""
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    _write_api_sidecar(sessions_dir, 42, "ghost")
    _write_events(sessions_dir, 42, [_result_event(input_tokens=1_000_000, output_tokens=0)])
    provider = _provider()
    api_config = type("C", (), {"providers": {"example": provider}})()
    view = _api_worker_view(sessions_dir, 42, "ghost")

    view.reap_sidecar(sessions_dir, api_config=api_config, state_dir=state_dir)

    assert not ledger_path(state_dir).exists()
    from charlie_work.claude_code import _sidecar_path

    assert not _sidecar_path(sessions_dir, 42, "api").exists()


def test_reap_sidecar_settlement_failure_does_not_break_reap(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If settlement raises, the sidecar is still unlinked (best-effort accounting)."""
    sessions_dir = tmp_path / "sessions"
    _write_api_sidecar(sessions_dir, 42, "example")
    _write_events(sessions_dir, 42, [_result_event(input_tokens=1_000_000, output_tokens=0)])
    provider = _provider()
    api_config = type("C", (), {"providers": {"example": provider}})()
    view = _api_worker_view(sessions_dir, 42, "example")
    # Make state_dir non-writable-ish by pointing it at a file path so save_ledger
    # raises (parent is a file, not a directory).
    state_dir_bad = tmp_path / "blocker"
    state_dir_bad.write_text("i am a file, not a dir", encoding="utf-8")

    with caplog.at_level("WARNING", logger="charlie_work.worker"):
        view.reap_sidecar(sessions_dir, api_config=api_config, state_dir=state_dir_bad)

    from charlie_work.claude_code import _sidecar_path

    assert not _sidecar_path(sessions_dir, 42, "api").exists()
    assert any("settlement failed" in rec.message for rec in caplog.records)


def test_reap_sidecar_settlement_uses_locked_settle_to_disk(tmp_path: Path, monkeypatch) -> None:
    """The reap wiring routes the on-disk RMW through settle_session_to_disk
    (locked), not the unlocked load_ledger/settle_session/save_ledger trio.
    Patches settle_session_to_disk and asserts it is called with the ledger path.
    """
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    _write_api_sidecar(sessions_dir, 42, "example")
    _write_events(sessions_dir, 42, [_result_event(input_tokens=1_000_000, output_tokens=0)])
    provider = _provider()
    api_config = type("C", (), {"providers": {"example": provider}})()
    view = _api_worker_view(sessions_dir, 42, "example")

    captured: list = []
    real = api_budget.settle_session_to_disk

    def spy(path: Path, entry: SessionEntry) -> bool:
        captured.append((path, entry))
        return real(path, entry)

    monkeypatch.setattr(api_budget, "settle_session_to_disk", spy)

    view.reap_sidecar(sessions_dir, api_config=api_config, state_dir=state_dir)

    assert len(captured) == 1
    assert captured[0][0] == ledger_path(state_dir)
    assert captured[0][1].issue == 42
