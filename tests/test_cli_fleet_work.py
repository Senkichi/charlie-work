"""``charlie fleet work`` layered-config loud/fallback paths.

Split out of ``tests/test_cli.py`` (issue #1561, Track 1) -- bodies are
verbatim relocations; shared fakes live in ``tests/_cli_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from charlie_work import cli
from charlie_work.workflow import CommandResult


def test_run_fleet_work_loud_on_absent_global_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An absent global layer must be loud in `charlie fleet work`, not silent.

    Mirrors test_run_fleet_supervise_loud_on_absent_global_layer: drives the
    REAL ``load_layered_config`` (not a mock) against an empty fleet dir so the
    ``require_global=True`` wiring in ``run_fleet_work`` is exercised
    end-to-end. Every other ``run_fleet_work`` / ``charlie fleet work`` test
    mocks ``load_layered_config`` away (``lambda *a, **k: None``), so without
    this test a future silent drop of ``require_global=True`` would pass CI
    undetected -- fleet-wide knobs (notify, runner prologues) would revert to
    dataclass defaults with no error raised, the #623 failure shape.

    ``fleet_loop`` is mocked so the pass does not touch the real fleet; only the
    config-load path is left real.
    """
    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "pass ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    # Deliberately NOT mocking cli.load_layered_config.

    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(tmp_path), "fleet", "work", "--limit", "1"]
    )
    result = cli.run_fleet_work(args)

    # The require_global=True ConfigError is caught and printed loudly. If
    # require_global=True were silently dropped, load_layered_config would
    # return an OrchestratorConfig (not raise), "config load failed" would
    # never be printed.
    out = capsys.readouterr().out
    assert "config load failed" in out, (
        "an absent global layer must be printed, not silently defaulted"
    )
    assert str(tmp_path / "config.yaml") in out, (
        "the expected global config path must appear in the failure message"
    )
    assert "absent" in out, "an absent layer must read as absent in the message"

    # The pass continued (the command must not crash). The fallback reloads
    # with require_global=False so the per-repo config is NOT discarded with
    # the global layer -- regressing to global_config=None would reproduce the
    # #623 silent-disable failure (every per-repo knob reverting to defaults).
    # Here cwd has no per-repo config, so the reload yields pristine defaults,
    # but the point is it is a real OrchestratorConfig, not the None sentinel
    # that would skip the runner prologues and silence notify unconditionally.
    assert result.ok is True
    assert fleet_loop_mock.call_count == 1
    assert fleet_loop_mock.call_args.kwargs.get("global_config") is not None, (
        "fleet_loop must NOT receive global_config=None when the global layer "
        "is absent -- the per-repo config must survive the fallback, not be "
        "discarded with the global layer (#623 silent-disable regression)"
    )


def test_run_fleet_work_fallback_preserves_per_repo_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The require_global fallback must keep the per-repo config, not discard it.

    The review on PR #630 found that the fallback set ``global_config=None``
    when ``require_global=True`` raised, discarding the per-repo config too --
    not just the global layer. That regressed a previously-valid no-op state
    into the exact #623 silent-disable failure: every per-repo knob (notify,
    labels) reverted to its dataclass default while passes kept reporting
    success. The fallback now reloads with ``require_global=False`` so per-repo
    settings survive.

    cwd becomes a repo with a per-repo config that turns notify ON; the fleet
    dir is a separate empty dir so the global layer is absent. Drives the REAL
    ``load_layered_config`` (not a mock) so the fallback path is exercised
    end-to-end.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "orchestrator.config.yaml").write_text(
        "notify:\n  enabled: true\n", encoding="utf-8"
    )
    monkeypatch.chdir(repo_dir)
    fleet_empty = tmp_path / "fleet"
    fleet_empty.mkdir()

    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "pass ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    # Deliberately NOT mocking cli.load_layered_config.

    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(fleet_empty), "fleet", "work", "--limit", "1"]
    )
    result = cli.run_fleet_work(args)

    # The absent global layer is still loud...
    out = capsys.readouterr().out
    assert "config load failed" in out, (
        "the absent global layer must still be printed even when per-repo config is preserved"
    )

    # ...but the per-repo notify setting survives the fallback. Before the fix
    # global_config was None and notify was effectively off; now it is the
    # per-repo OrchestratorConfig with notify.enabled=True.
    assert result.ok is True
    passed_config = fleet_loop_mock.call_args.kwargs.get("global_config")
    assert passed_config is not None, (
        "the per-repo config must survive the require_global fallback, not be "
        "discarded with the absent global layer"
    )
    assert getattr(passed_config.notify, "enabled", False) is True, (
        "the per-repo notify.enabled=True must survive the require_global "
        "fallback, not revert to the dataclass default False (#623 regression)"
    )
