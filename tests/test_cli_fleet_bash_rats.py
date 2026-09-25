"""``charlie fleet bash-rats`` self-deploy, digest, and loud-on-absent-config paths.

Split out of ``tests/test_cli.py`` (issue #1561, Track 1) -- bodies are
verbatim relocations; shared fakes live in ``tests/_cli_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from charlie_work import cli, layout
from charlie_work.config import NotifyConfig
from charlie_work.notify import NotifyResult
from charlie_work.supervise import SelfDeployResult, supervisor_runtime_paths
from charlie_work.workflow import CommandResult


def test_run_fleet_bash_rats_self_deploys_before_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`charlie fleet bash-rats` calls self_deploy on the orchestrator root first."""
    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=True,
            from_sha="abc123",
            to_sha="def456",
            message="updated: def456",
        )
    )
    monkeypatch.setattr(cli, "self_deploy", deploy_mock)

    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: None)

    args = cli.build_parser().parse_args(
        ["--fleet-dir", "custom-fleet", "fleet", "bash-rats", "--limit", "2"]
    )
    result = cli.run_fleet_bash_rats(args)

    assert result.ok is True
    deploy_mock.assert_called_once()
    orchestrator_root = deploy_mock.call_args[0][0]
    assert isinstance(orchestrator_root, Path)
    assert (orchestrator_root / "pyproject.toml").exists()
    assert deploy_mock.call_args.kwargs.get("fleet_dir_override") == "custom-fleet"
    assert fleet_loop_mock.called is True
    assert fleet_loop_mock.call_args.kwargs.get("fleet_dir_override") == "custom-fleet"

    out = capsys.readouterr().out
    assert "self-deploy: updated: def456" in out


def test_run_fleet_bash_rats_self_deploy_failure_is_non_fatal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed self_deploy does not abort a `charlie fleet bash-rats` pass."""
    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            error="diverged or dirty tree",
        )
    )
    monkeypatch.setattr(cli, "self_deploy", deploy_mock)

    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "pass ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    monkeypatch.setattr(cli, "load_layered_config", lambda *a, **k: None)

    args = cli.build_parser().parse_args(["fleet", "bash-rats"])
    result = cli.run_fleet_bash_rats(args)

    assert result.ok is True
    assert fleet_loop_mock.called is True
    out = capsys.readouterr().out
    assert "self-deploy skipped: diverged or dirty tree" in out


def test_run_fleet_bash_rats_drains_pass_when_sync_starved(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Issue #1855: a starved deferred sync runs the bash-rats pass in drain
    mode -- ``drain=True`` suppresses new dispatch for this pass, matching
    the supervisor's posture -- and the configured bound is plumbed through.
    """
    from charlie_work.config import OrchestratorConfig, SupervisorConfig

    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=True,
            changed=True,
            synced=False,
            head_changed=False,
            from_sha="abc123",
            to_sha="def456",
            message="sync deferred: 2 runners active",
            deferred=True,
            starved=True,
        )
    )
    monkeypatch.setattr(cli, "self_deploy", deploy_mock)

    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "pass ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    monkeypatch.setattr(
        cli,
        "load_layered_config",
        lambda *_a, **_k: OrchestratorConfig(
            supervisor=SupervisorConfig(dependency_sync_starvation_seconds=600)
        ),
    )

    args = cli.build_parser().parse_args(["fleet", "bash-rats"])
    result = cli.run_fleet_bash_rats(args)

    assert result.ok is True
    assert deploy_mock.call_args.kwargs["starvation_seconds"] == 600
    assert fleet_loop_mock.call_args.kwargs["drain"] is True
    out = capsys.readouterr().out
    assert "starvation bound reached" in out


def test_run_fleet_bash_rats_emits_attention_digest_on_repair_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A self_deploy repair failure emits an attention digest when notify is enabled."""
    digest_path = tmp_path / "digest.jsonl"
    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            error="venv pth repair failed: Access is denied",
        )
    )
    monkeypatch.setattr(cli, "self_deploy", deploy_mock)

    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "pass ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)

    # Provide a config whose notify sink writes to a temp file.
    from charlie_work.config import OrchestratorConfig

    notify_config = NotifyConfig(enabled=True, sink="file", file_path=str(digest_path))
    monkeypatch.setattr(
        cli,
        "load_layered_config",
        lambda *_a, **_k: OrchestratorConfig(notify=notify_config),
    )

    args = cli.build_parser().parse_args(["fleet", "bash-rats"])
    result = cli.run_fleet_bash_rats(args)

    assert result.ok is True
    assert fleet_loop_mock.called is True
    assert digest_path.exists()
    digest_line = digest_path.read_text(encoding="utf-8").strip()
    digest = json.loads(digest_line)
    assert digest["repo"] == "fleet"
    assert len(digest["transitions"]) == 1
    assert digest["transitions"][0]["adapter_kind"] == "self-deploy"
    assert digest["transitions"][0]["health"] == "ERROR"
    assert "Access is denied" in digest["transitions"][0]["last_log_line"]


def test_run_fleet_bash_rats_emits_attention_digest_on_venv_repaired(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A successful self_deploy venv repair emits an attention digest when notify is enabled."""
    digest_path = tmp_path / "digest.jsonl"
    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=False,
            changed=False,
            synced=False,
            venv_repaired=True,
            message="venv editable target repaired: shared venv editable .pth targets all resolve to configured checkouts",
        )
    )
    monkeypatch.setattr(cli, "self_deploy", deploy_mock)

    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "pass ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)

    # Provide a config whose notify sink writes to a temp file.
    from charlie_work.config import OrchestratorConfig

    notify_config = NotifyConfig(enabled=True, sink="file", file_path=str(digest_path))
    monkeypatch.setattr(
        cli,
        "load_layered_config",
        lambda *_a, **_k: OrchestratorConfig(notify=notify_config),
    )

    args = cli.build_parser().parse_args(["fleet", "bash-rats"])
    result = cli.run_fleet_bash_rats(args)

    assert result.ok is True
    assert fleet_loop_mock.called is True
    assert digest_path.exists()
    digest_line = digest_path.read_text(encoding="utf-8").strip()
    digest = json.loads(digest_line)
    assert digest["repo"] == "fleet"
    assert len(digest["transitions"]) == 1
    assert digest["transitions"][0]["adapter_kind"] == "self-deploy"
    assert digest["transitions"][0]["health"] == "REPAIRED"


def test_run_fleet_bash_rats_resolves_notify_sentinel_for_emit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Issue #1899: the self-deploy emit gets the sentinel-resolved notify
    config -- the raw ``file_path=""`` default would fail every emit with
    'file_path is empty'."""
    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=False,
            pulled=False,
            changed=False,
            synced=False,
            error="diverged or dirty tree",
        )
    )
    monkeypatch.setattr(cli, "self_deploy", deploy_mock)
    monkeypatch.setattr(
        cli,
        "fleet_loop",
        MagicMock(return_value=CommandResult(True, "ok", {"repos": {}})),
    )
    emit = MagicMock(name="emit_digest", return_value=NotifyResult(ok=True))
    monkeypatch.setattr(cli, "emit_digest", emit)

    from charlie_work.config import OrchestratorConfig

    monkeypatch.setattr(
        cli,
        "load_layered_config",
        lambda *_a, **_k: OrchestratorConfig(notify=NotifyConfig(enabled=True, sink="file")),
    )

    args = cli.build_parser().parse_args(["fleet", "bash-rats"])
    result = cli.run_fleet_bash_rats(args)

    assert result.ok is True
    emit.assert_called_once()
    emitted_config = emit.call_args[0][0]
    assert emitted_config.file_path == str(
        layout.notify_digest_default(
            supervisor_runtime_paths(OrchestratorConfig().runtime.state_dir).root
        )
    )


def test_run_fleet_bash_rats_loud_on_absent_global_layer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An absent global layer must be loud in `charlie fleet bash-rats`, not silent.

    Mirrors test_run_fleet_supervise_loud_on_absent_global_layer: drives the
    REAL ``load_layered_config`` (not a mock) against an empty fleet dir so the
    ``require_global=True`` wiring in ``run_fleet_bash_rats`` is exercised
    end-to-end. Every other ``run_fleet_bash_rats`` test mocks
    ``load_layered_config`` away (``lambda *a, **k: None``), so without this
    test a future silent drop of ``require_global=True`` would pass CI
    undetected -- fleet-wide knobs (notify, runner prologues) would revert to
    dataclass defaults with no error raised, the #623 failure shape.

    ``self_deploy`` and ``fleet_loop`` are mocked so the pass does not touch the
    network or the real fleet; only the config-load path is left real.
    """
    deploy_mock = MagicMock(
        return_value=SelfDeployResult(
            ok=True,
            pulled=False,
            changed=False,
            synced=False,
            message="up to date",
        )
    )
    monkeypatch.setattr(cli, "self_deploy", deploy_mock)
    fleet_loop_mock = MagicMock(return_value=CommandResult(True, "pass ok", {"repos": {}}))
    monkeypatch.setattr(cli, "fleet_loop", fleet_loop_mock)
    # Deliberately NOT mocking cli.load_layered_config.

    args = cli.build_parser().parse_args(
        ["--fleet-dir", str(tmp_path), "fleet", "bash-rats", "--limit", "1"]
    )
    result = cli.run_fleet_bash_rats(args)

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
