"""Binary/quota probes: ``probe_claude`` resolution and
``run_quota_probe`` green/nonzero/throttle-marker outcomes.

Split out of ``tests/test_claude_code_adapter.py`` (issue #1560,
Track 1) -- bodies are verbatim relocations; shared helpers live in
``tests/_claude_adapter_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest

from charlie_work import claude_code
from charlie_work.config import (
    OrchestratorConfig,
    QuotaProbeConfig,
    RuntimeConfig,
)
from charlie_work.claude_code import (
    probe_claude,
    run_quota_probe,
)
from charlie_work.subprocess_runner import RunResult


def test_probe_claude_resolves_argv0_through_resolve_cli_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ``doctor --adapter-probe`` path (``probe_claude``) must resolve
    the same way ``launch_claude_worker`` does, or `charlie doctor` would
    report a healthy `claude` install failing to probe with WinError 2 on a
    machine using the claude-code adapter (issue #487)."""
    captured: dict[str, Any] = {}

    def fake_resolve_cli_binary(name: str) -> str:
        assert name == "claude"
        return "C:\\resolved\\claude.exe"

    def fake_run_captured(command, *, cwd, timeout_seconds):
        captured["command"] = command
        return RunResult(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(claude_code, "resolve_cli_binary", fake_resolve_cli_binary)
    monkeypatch.setattr(claude_code, "run_captured", fake_run_captured)

    result = probe_claude(tmp_path)

    assert result.ok
    assert captured["command"] == ["C:\\resolved\\claude.exe", "--version"]


def test_probe_claude_missing_binary_never_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Force resolve_cli_binary's lookup to fail regardless of whether this
    # test machine happens to have a real `claude` on PATH (as this repo's
    # own dev box does) -- the point of this test is the genuinely-missing
    # case, not this machine's install state.
    monkeypatch.setattr(
        claude_code,
        "resolve_cli_binary",
        lambda name: "this-binary-does-not-exist-xyz",
    )

    result = probe_claude(tmp_path)

    assert result.ok is False
    assert result.error is not None


def test_probe_claude_uses_run_captured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple] = []

    def fake_run_captured(command, *, cwd, timeout_seconds, shell=False):
        calls.append((command, cwd, timeout_seconds))
        from charlie_work.subprocess_runner import RunResult

        return RunResult(returncode=0, stdout="1.0.0", stderr="")

    # Identity resolution: this test is about the run_captured plumbing, not
    # binary resolution (covered separately by
    # test_probe_claude_resolves_argv0_through_resolve_cli_binary).
    monkeypatch.setattr(claude_code, "resolve_cli_binary", lambda name: name)
    monkeypatch.setattr(claude_code, "run_captured", fake_run_captured)

    result = probe_claude(tmp_path)

    assert result.ok
    assert calls[0][0] == ["claude", "--version"]
    assert calls[0][1] == tmp_path


def _quota_probe_config(**overrides: Any) -> OrchestratorConfig:
    return OrchestratorConfig(quota_probe=QuotaProbeConfig(**overrides))


def test_run_quota_probe_green_returns_true_and_pins_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple] = []

    def fake_run_captured(command, *, cwd, timeout_seconds, shell=False, stdin=None):
        calls.append((command, cwd, timeout_seconds, stdin))
        return RunResult(returncode=0, stdout="OK", stderr="")

    monkeypatch.setattr(claude_code, "resolve_cli_binary", lambda name: name)
    monkeypatch.setattr(claude_code, "run_captured", fake_run_captured)

    config = _quota_probe_config(model="claude-haiku-4-5", timeout_seconds=42, prompt="say OK")

    result = run_quota_probe(repo_root=tmp_path, config=config)

    assert result is True
    command, cwd, timeout_seconds, stdin = calls[0]
    assert "--model" in command
    assert command[command.index("--model") + 1] == "claude-haiku-4-5"
    assert "--max-turns" in command
    assert command[command.index("--max-turns") + 1] == "1"
    assert cwd == tmp_path
    assert timeout_seconds == 42
    assert stdin == "say OK"


def test_run_quota_probe_nonzero_exit_returns_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(claude_code, "resolve_cli_binary", lambda name: name)
    monkeypatch.setattr(
        claude_code,
        "run_captured",
        lambda *a, **k: RunResult(returncode=1, stdout="", stderr="boom"),
    )

    result = run_quota_probe(repo_root=tmp_path, config=_quota_probe_config())

    assert result is False


def test_run_quota_probe_quota_exhausted_output_returns_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Issue: exit code 0 with an in-band throttle message must not be
    # misread as a green probe (mirrors _classify_session_failure's log-tail
    # handling for a real worker).
    monkeypatch.setattr(claude_code, "resolve_cli_binary", lambda name: name)
    monkeypatch.setattr(
        claude_code,
        "run_captured",
        lambda *a, **k: RunResult(
            returncode=0, stdout="daily usage quota has been exhausted", stderr=""
        ),
    )

    result = run_quota_probe(repo_root=tmp_path, config=_quota_probe_config())

    assert result is False


def test_run_quota_probe_provider_auth_output_returns_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(claude_code, "resolve_cli_binary", lambda name: name)
    monkeypatch.setattr(
        claude_code,
        "run_captured",
        lambda *a, **k: RunResult(returncode=0, stdout="", stderr="401 unauthorized"),
    )

    result = run_quota_probe(repo_root=tmp_path, config=_quota_probe_config())

    assert result is False


def test_run_quota_probe_custom_throttle_marker_returns_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(claude_code, "resolve_cli_binary", lambda name: name)
    monkeypatch.setattr(
        claude_code,
        "run_captured",
        lambda *a, **k: RunResult(returncode=0, stdout="custom-throttle-marker seen", stderr=""),
    )

    config = OrchestratorConfig(
        quota_probe=QuotaProbeConfig(),
        runtime=RuntimeConfig(throttle_error_markers=("custom-throttle-marker",)),
    )

    result = run_quota_probe(repo_root=tmp_path, config=config)

    assert result is False
