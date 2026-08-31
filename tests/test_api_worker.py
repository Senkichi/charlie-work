"""Tests for the api worker adapter (issue #478).

Mirrors the structure of tests/test_claude_code_adapter.py: launch failures,
env construction, delegation, sidecar shape. Plus the no-key-material invariant
(the API key value must never appear in a sidecar, record dict, log, or argv).
"""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _worker_marker_wait import read_worker_marker

from charlie_work import api_worker, claude_code, instrumentation
from charlie_work.api_budget import DayBucket, Ledger, ledger_path, save_ledger
from charlie_work.api_worker import launch_api_worker
from charlie_work.claude_code import ClaudeWorkerRecord, read_worker_records
from charlie_work.config import (
    ApiBudgetConfig,
    ApiProviderConfig,
    ApiWorkerConfig,
    ClaudeCodeConfig,
    OrchestratorConfig,
    RuntimeConfig,
    WorkerRoleConfig,
)
from charlie_work.worktree import WorktreeInfo


def _fake_worktree(tmp_path: Path, branch: str) -> WorktreeInfo:
    worktree_path = tmp_path / "worktrees" / branch.replace("/", "-")
    worktree_path.mkdir(parents=True, exist_ok=True)
    return WorktreeInfo(path=worktree_path, branch=branch, venv_junction=None)


def _install_fake_create_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    calls: list[dict] | None = None,
) -> None:
    def fake_create_worktree(
        repo_root,
        branch,
        *,
        base_ref="HEAD",
        worktrees_dir=None,
        venv_source=None,
        materialize_dirs=(),
        rework=False,
        recovery=None,
        issue_number=None,
        config=None,
        sessions_dir=None,
    ):
        if calls is not None:
            calls.append(
                {
                    "repo_root": repo_root,
                    "branch": branch,
                    "base_ref": base_ref,
                    "worktrees_dir": worktrees_dir,
                    "venv_source": venv_source,
                    "materialize_dirs": materialize_dirs,
                    "rework": rework,
                    "recovery": recovery,
                    "issue_number": issue_number,
                    "config": config,
                    "sessions_dir": sessions_dir,
                }
            )
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)


def _fake_claude_script(tmp_path: Path) -> tuple[str, ...]:
    """A Python script standing in for the `claude` binary: reads stdin (the
    prompt), writes a marker file next to cwd, and exits 0."""
    script_path = tmp_path / "fake_claude_api.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path

            data = sys.stdin.read()
            Path("worker-ran.txt").write_text(data, encoding="utf-8")
            print("ok")
            """
        ),
        encoding="utf-8",
    )
    return (sys.executable, str(script_path))


def _provider_config(
    *,
    api_key_env: str = "MOONSHOT_API_KEY",
    model: str = "kimi-k3",
    base_url: str = "https://api.moonshot.ai/anthropic",
) -> ApiProviderConfig:
    return ApiProviderConfig(
        base_url=base_url,
        api_key_env=api_key_env,
        model=model,
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        cached_input_usd_per_mtok=0.30,
    )


def _api_worker_config(
    *,
    provider_name: str = "kimi-k3",
    provider: ApiProviderConfig | None = None,
    enabled: bool = True,
) -> ApiWorkerConfig:
    """Build an enabled ApiWorkerConfig with one provider registered."""
    provider = provider or _provider_config()
    return ApiWorkerConfig(
        enabled=enabled,
        provider=provider_name,
        max_concurrent_sessions=1,
        providers={provider_name: provider},
        budget=ApiBudgetConfig(),
        worker_template="worker_claude_code.md",
        rework_template="rework.md",
    )


def test_launch_api_worker_writes_prompt_and_api_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A successful launch writes issue-<n>.api.json (not .claude.json) and
    records adapter_kind='api' + the provider name."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    record = launch_api_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=_fake_claude_script(tmp_path),
    )

    assert record.ok
    assert record.error is None
    assert record.issue_number == 42
    assert record.branch == "agent/issue-42-fix"
    assert record.pid is not None
    assert record.adapter_kind == "api"
    assert record.provider == "kimi-k3"

    # Sidecar lands as issue-<n>.api.json, NOT issue-<n>.claude.json
    api_sidecar = sessions_dir / "issue-42.api.json"
    claude_sidecar = sessions_dir / "issue-42.claude.json"
    assert api_sidecar.exists()
    assert not claude_sidecar.exists()
    payload = json.loads(api_sidecar.read_text(encoding="utf-8"))
    assert payload["issue_number"] == 42
    assert payload["adapter_kind"] == "api"
    assert payload["provider"] == "kimi-k3"
    assert payload["error"] is None

    # Prompt was written into the worktree.
    worktree_path = Path(record.worktree_path)
    prompt_path = worktree_path / ".orchestrator-prompt.md"
    assert prompt_path.read_text(encoding="utf-8") == "Do the thing."


def test_launch_api_worker_force_enables_tee_stream_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """tee_stream_json is force-enabled for api sessions (the budget ledger
    depends on events.jsonl) regardless of the caller's preference."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    captured: dict[str, Any] = {}

    real_launch = claude_code.launch_claude_worker

    def capturing_launch(*args, **kwargs):
        captured["tee_stream_json"] = kwargs.get("tee_stream_json")
        captured["adapter_kind"] = kwargs.get("adapter_kind")
        captured["provider"] = kwargs.get("provider")
        return real_launch(*args, **kwargs)

    # Patch the name as bound in api_worker (it imports launch_claude_worker
    # at module load, so patching claude_code.launch_claude_worker would not
    # reach the call site).
    monkeypatch.setattr(api_worker, "launch_claude_worker", capturing_launch)

    record = launch_api_worker(
        7,
        "agent/issue-7-x",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=_fake_claude_script(tmp_path),
    )

    assert record.ok
    assert captured["tee_stream_json"] is True
    assert captured["adapter_kind"] == "api"
    assert captured["provider"] == "kimi-k3"

    # tee_stream_json=True opens issue-<n>.events.jsonl at launch time, so it
    # must exist on disk (independent of whether the fake worker has written
    # any stream-json events yet).
    assert (sessions_dir / "issue-7.events.jsonl").exists()


def test_launch_api_worker_injects_provider_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The child process receives ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN,
    ANTHROPIC_MODEL, and the small/fast-model overrides pointing at the
    configured provider model."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-probe.txt").write_text(
                "|".join(
                    [
                        os.environ.get("ANTHROPIC_BASE_URL", "<unset>"),
                        os.environ.get("ANTHROPIC_AUTH_TOKEN", "<unset>"),
                        os.environ.get("ANTHROPIC_MODEL", "<unset>"),
                        os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL", "<unset>"),
                        os.environ.get("ANTHROPIC_SMALL_FAST_MODEL", "<unset>"),
                    ]
                ),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_api_worker(
        99,
        "agent/issue-99-env",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=(sys.executable, str(script_path)),
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-probe.txt"
    # Wait for the whole probe, not just for the path to appear: a partially
    # written probe splits into the wrong number of fields (see
    # _worker_marker_wait).
    base_url, auth_token, model, haiku, small_fast = read_worker_marker(
        probe_path,
        expected="|".join(
            [
                "https://api.moonshot.ai/anthropic",
                "sk-test-key-value-1234",
                "kimi-k3",
                "kimi-k3",
                "kimi-k3",
            ]
        ),
    ).split("|")
    assert base_url == "https://api.moonshot.ai/anthropic"
    assert auth_token == "sk-test-key-value-1234"
    assert model == "kimi-k3"
    # Both small/fast-model env vars pinned to the same configured model so no
    # auxiliary call routes to Anthropic's Haiku against the custom endpoint.
    assert haiku == "kimi-k3"
    assert small_fast == "kimi-k3"


def test_launch_api_worker_pins_provider_model_not_worker_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1245: the api worker's ``--model`` flag must equal the provider
    registry's model, not ``worker.model``. The Claude Code CLI gives the
    ``--model`` flag precedence over the ``ANTHROPIC_MODEL`` env var, so
    without pinning the provider's model argv-side the provider's model
    selection is dead code (Moonshot served whatever it maps the
    claude-sonnet-5 alias to, not kimi-k3).

    Asserts over the recorded sidecar ``command`` (which captures argv) and
    ``record.command``. Uses a config whose ``worker.model`` deliberately
    differs from the provider's model so a regression that re-pins
    ``worker.model`` is caught. (Renamed from ``..._not_claude_code_model``
    -- role-config Phase 2 Track E deleted ``ClaudeCodeConfig.model``; the
    model this test pins against now lives on ``WorkerRoleConfig.model``.)"""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    # worker.model deliberately differs from the provider model (kimi-k3)
    # so the two are distinguishable in the pinned --model value.
    config = OrchestratorConfig(worker=WorkerRoleConfig(model="claude-sonnet-5"))

    record = launch_api_worker(
        1245,
        "agent/issue-1245-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=_fake_claude_script(tmp_path),
        config=config,
    )

    assert record.ok
    # The provider registry's model is kimi-k3 (from _provider_config default).
    provider_model = "kimi-k3"
    worker_model = "claude-sonnet-5"
    # The two must differ, otherwise this test cannot discriminate the fix.
    assert provider_model != worker_model

    # record.command carries the pinned argv.
    assert "--model" in record.command
    idx = record.command.index("--model")
    assert record.command[idx + 1] == provider_model
    assert record.command[idx + 1] != worker_model
    # Exactly one --model pin (dedup behavior of _apply_model_pin unchanged).
    assert record.command.count("--model") == 1

    # The sidecar's recorded command must agree (it captures argv too).
    sidecar_path = sessions_dir / "issue-1245.api.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar_command = tuple(payload["command"])
    assert "--model" in sidecar_command
    sidx = sidecar_command.index("--model")
    assert sidecar_command[sidx + 1] == provider_model
    assert sidecar_command.count("--model") == 1


def test_launch_api_worker_worker_env_merged_under_provider_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Operator worker_env values that do not collide with provider routing
    still apply, and provider routing vars win over any colliding worker_env."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    script_path = tmp_path / "env_probe2.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-probe2.txt").write_text(
                "|".join(
                    [
                        os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS", "<unset>"),
                        os.environ.get("ANTHROPIC_MODEL", "<unset>"),
                    ]
                ),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_api_worker(
        140,
        "agent/issue-140-env",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=(sys.executable, str(script_path)),
        # worker_env sets a non-colliding var AND tries to override the model;
        # the provider model must win.
        worker_env={
            "PYTEST_XDIST_AUTO_NUM_WORKERS": "2",
            "ANTHROPIC_MODEL": "operator-override-model",
        },
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-probe2.txt"
    xdist, model = read_worker_marker(probe_path, expected="2|kimi-k3").split("|")
    assert xdist == "2"
    # Provider routing var wins over worker_env.
    assert model == "kimi-k3"


def test_launch_api_worker_missing_key_env_returns_error_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing api_key_env value yields an error record (errors as values,
    never raises) and writes an issue-<n>.api.json sidecar."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    record = launch_api_worker(
        13,
        "agent/issue-13-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=_fake_claude_script(tmp_path),
    )

    assert not record.ok
    assert record.error is not None
    assert "MOONSHOT_API_KEY" in record.error
    assert record.pid is None
    assert record.adapter_kind == "api"
    # The provider was resolved before the key-env lookup failed, so the
    # error record must carry the provider name (not "") for triage.
    assert record.provider == "kimi-k3"

    sidecar_path = sessions_dir / "issue-13.api.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["error"] == record.error
    assert payload["provider"] == "kimi-k3"
    # No key material leaked into the error message.
    assert "sk-test-key-value-1234" not in json.dumps(payload)


def test_launch_api_worker_disabled_config_returns_error_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An disabled ApiWorkerConfig yields an error record (defensive re-check)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    record = launch_api_worker(
        14,
        "agent/issue-14-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(enabled=False),
        command_template=_fake_claude_script(tmp_path),
    )

    assert not record.ok
    assert record.error is not None
    assert "enabled" in record.error
    assert record.pid is None
    # The disabled-config path returns before the provider name is read, so
    # the error record carries provider="" (no provider was resolved).
    assert record.provider == ""


def test_launch_api_worker_unknown_provider_returns_error_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A provider name not in the registry yields an error record (defensive
    re-check; config load already validates this but a directly-constructed
    config must not raise)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    # Build a config whose provider key does not match any registered provider.
    # Bypass __post_init__ validation by constructing enabled=False then using
    # object.__setattr__ to flip enabled True with a stale provider name.
    cfg = _api_worker_config()
    object.__setattr__(cfg, "provider", "nonexistent-provider")

    record = launch_api_worker(
        15,
        "agent/issue-15-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=cfg,
        command_template=_fake_claude_script(tmp_path),
    )

    assert not record.ok
    assert record.error is not None
    assert "nonexistent-provider" in record.error
    assert record.pid is None
    # The configured provider name is known even though it was not in the
    # registry; the error record carries it for triage.
    assert record.provider == "nonexistent-provider"


def test_launch_api_worker_launch_exception_returns_error_record_with_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If launch_claude_worker raises (it never should, but the except clause
    guards against regressions in this module's own plumbing), the exception is
    caught and an error record is returned with the resolved provider name
    carried through (errors as values; never raises)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    def raising_launch(*args, **kwargs):
        raise RuntimeError("plumbing exploded")

    monkeypatch.setattr(api_worker, "launch_claude_worker", raising_launch)

    record = launch_api_worker(
        16,
        "agent/issue-16-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=_fake_claude_script(tmp_path),
    )

    assert not record.ok
    assert record.error is not None
    assert "plumbing exploded" in record.error
    assert record.pid is None
    assert record.adapter_kind == "api"
    # The provider was resolved before the launch call, so the error record
    # carries it for triage.
    assert record.provider == "kimi-k3"
    # No key material leaked into the error record/sidecar.
    sidecar_path = sessions_dir / "issue-16.api.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert "sk-test-key-value-1234" not in json.dumps(payload)


def test_launch_api_worker_delegates_to_claude_code_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """launch_api_worker delegates to claude_code.launch_claude_worker with
    adapter_kind='api' and the resolved provider name, mapping the returned
    record through unchanged."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    captured: dict[str, Any] = {}

    def fake_launch(*args, **kwargs):
        captured.update(kwargs)
        captured["args"] = args
        return ClaudeWorkerRecord(
            issue_number=args[0],
            branch=args[1],
            worktree_path=str(tmp_path / "wt"),
            prompt_path=str(tmp_path / "prompt.md"),
            command=("claude",),
            pid=4242,
            started_at="2026-07-22T00:00:00Z",
            log_path=str(sessions_dir / "issue-42.claude.log"),
            error=None,
            adapter_kind="api",
            provider="kimi-k3",
        )

    monkeypatch.setattr(api_worker, "launch_claude_worker", fake_launch)

    record = launch_api_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=("claude",),
        worktrees_dir=tmp_path / "worktrees",
        materialize_dirs=(".devin",),
        rework=True,
        base_ref="origin/main",
    )

    assert record.ok
    assert record.adapter_kind == "api"
    assert record.provider == "kimi-k3"
    assert captured["adapter_kind"] == "api"
    assert captured["provider"] == "kimi-k3"
    assert captured["tee_stream_json"] is True
    assert captured["worktrees_dir"] == tmp_path / "worktrees"
    assert captured["materialize_dirs"] == (".devin",)
    assert captured["rework"] is True
    assert captured["base_ref"] == "origin/main"
    # Provider env was merged into the env dict passed through.
    env = captured["env"]
    assert env["ANTHROPIC_BASE_URL"] == "https://api.moonshot.ai/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-test-key-value-1234"
    assert env["ANTHROPIC_MODEL"] == "kimi-k3"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "kimi-k3"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "kimi-k3"


def test_launch_api_worker_never_blocks_on_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Launch is non-blocking: launch_claude_worker uses Popen and returns
    immediately (CLAUDE.md invariant). Verified by asserting the returned
    record has a pid but the fake process has not been waited on."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    waited: list[bool] = []

    real_popen = claude_code.subprocess.Popen

    class TrackingPopen(real_popen):  # type: ignore[misc, valid-type]
        def wait(self, *args, **kwargs):
            waited.append(True)
            return super().wait(*args, **kwargs)

        def communicate(self, *args, **kwargs):
            waited.append(True)
            return super().communicate(*args, **kwargs)

    monkeypatch.setattr(claude_code.subprocess, "Popen", TrackingPopen)

    record = launch_api_worker(
        21,
        "agent/issue-21-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=_fake_claude_script(tmp_path),
    )

    assert record.ok
    assert record.pid is not None
    # No wait/communicate was called during launch.
    assert waited == []


def test_no_key_material_in_sidecar_or_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The API key value must never appear in the sidecar JSON, the record
    dict, the prompt, or the command argv — it travels only in the child
    process env."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    key_value = "sk-super-secret-key-value-XYZ-9999"
    monkeypatch.setenv("MOONSHOT_API_KEY", key_value)

    record = launch_api_worker(
        77,
        "agent/issue-77-secret",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=_fake_claude_script(tmp_path),
    )

    assert record.ok

    sidecar_path = sessions_dir / "issue-77.api.json"
    assert sidecar_path.exists()
    sidecar_text = sidecar_path.read_text(encoding="utf-8")
    assert key_value not in sidecar_text

    record_dict = record.to_dict()
    assert key_value not in json.dumps(record_dict)

    # The prompt file in the worktree must not contain the key.
    prompt_path = Path(record.worktree_path) / ".orchestrator-prompt.md"
    assert key_value not in prompt_path.read_text(encoding="utf-8")

    # The recorded command argv must not contain the key.
    assert key_value not in " ".join(record.command)


def test_no_key_material_in_error_sidecar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Even on a launch failure, the error sidecar must not contain key
    material — the error message is built from the env var NAME, not value."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    key_value = "sk-super-secret-key-value-XYZ-9999"
    # Set a DIFFERENT env var to the key; the configured api_key_env is absent,
    # so the error references the NAME, never the value from the other var.
    monkeypatch.setenv("OTHER_API_KEY", key_value)
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)

    record = launch_api_worker(
        78,
        "agent/issue-78-secret",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=_fake_claude_script(tmp_path),
    )

    assert not record.ok
    sidecar_path = sessions_dir / "issue-78.api.json"
    assert sidecar_path.exists()
    sidecar_text = sidecar_path.read_text(encoding="utf-8")
    assert key_value not in sidecar_text
    assert "MOONSHOT_API_KEY" in sidecar_text


def test_read_worker_records_reads_api_sidecars(tmp_path: Path) -> None:
    """read_worker_records(adapter_kind='api') reads issue-<n>.api.json
    sidecars and surfaces them with adapter_kind='api'."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    api_sidecar = sessions_dir / "issue-5.api.json"
    api_sidecar.write_text(
        json.dumps(
            {
                "issue_number": 5,
                "branch": "agent/issue-5",
                "worktree_path": "/tmp/wt-5",
                "prompt_path": "/tmp/prompt-5.md",
                "command": ["claude"],
                "pid": 11111,
                "started_at": "2026-07-22T00:00:00Z",
                "log_path": str(sessions_dir / "issue-5.claude.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1710000000.0,
                "reclaimed": None,
                "adapter_kind": "api",
                "provider": "kimi-k3",
            }
        ),
        encoding="utf-8",
    )

    records = read_worker_records(sessions_dir, adapter_kind="api")
    assert len(records) == 1
    assert records[0].adapter_kind == "api"
    assert records[0].provider == "kimi-k3"
    assert records[0].issue_number == 5

    # Default (claude-code) read must NOT pick up api sidecars.
    assert read_worker_records(sessions_dir) == []


def test_dispatch_sessions_api_adapter_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With devin.adapter='api' set explicitly, dispatch_sessions launches api
    workers end-to-end (fake Popen), sidecars land as issue-<n>.api.json, and
    iter_workers surfaces them with adapter_kind='api'."""
    from charlie_work.adapters import (
        AdapterSettings,
        SessionRequest,
        dispatch_sessions,
    )
    from charlie_work.worker import iter_workers

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Do the thing.", encoding="utf-8")

    settings = AdapterSettings(
        adapter="api",
        sessions_dir=sessions_dir,
        claude_command=_fake_claude_script(tmp_path),
        api_worker_config=_api_worker_config(),
        config=OrchestratorConfig(claude_code=ClaudeCodeConfig()),
    )
    requests = [
        SessionRequest(
            issue_number=42,
            issue_title="test issue",
            prompt_path=prompt_path,
            branch_name="agent/issue-42-fix",
        )
    ]

    results = dispatch_sessions(
        repo_root,
        tmp_path / "manifest.json",
        tmp_path / "results.json",
        settings,
        requests,
    )

    assert len(results) == 1
    assert results[0].ok
    assert results[0].adapter == "api"
    assert results[0].pid is not None

    api_sidecar = sessions_dir / "issue-42.api.json"
    assert api_sidecar.exists()

    workers = iter_workers(sessions_dir)
    api_workers = [w for w in workers if w.adapter_kind == "api"]
    assert len(api_workers) == 1
    assert api_workers[0].issue_number == 42
    assert api_workers[0].adapter_kind == "api"


def test_dispatch_sessions_api_adapter_without_config_returns_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Selecting the api adapter without an api_worker_config yields an error
    result (never raises)."""
    from charlie_work.adapters import (
        AdapterSettings,
        SessionRequest,
        dispatch_sessions,
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Do the thing.", encoding="utf-8")

    settings = AdapterSettings(
        adapter="api",
        sessions_dir=sessions_dir,
        # api_worker_config intentionally omitted (None).
        config=OrchestratorConfig(),
    )
    requests = [
        SessionRequest(
            issue_number=43,
            issue_title="test issue",
            prompt_path=prompt_path,
            branch_name="agent/issue-43-fix",
        )
    ]

    results = dispatch_sessions(
        repo_root,
        tmp_path / "manifest.json",
        tmp_path / "results.json",
        settings,
        requests,
    )

    assert len(results) == 1
    assert not results[0].ok
    assert results[0].adapter == "api"
    assert results[0].error is not None
    assert "api_worker_config" in results[0].error


# ---------------------------------------------------------------------------
# Issue #1514: budget-exhausted preflight in launch_api_worker.
#
# After routing.py (and its sole budget refusal gate, routing._api_preflight)
# was deleted in Phase 2 Track B (PR #1517), the daily/lifetime caps in
# api_budget.budget_status were still computed and displayed (doctor.py,
# fleet_dispatch.py) but nothing refused a launch when they ran out. The
# refusal gate now lives in launch_api_worker itself, mirroring the existing
# enabled/provider/auth-token checks. These tests drive the cap to exhausted
# and assert no launch (no Popen) happens.
# ---------------------------------------------------------------------------


def _today_utc() -> str:
    """Today's UTC date key, derived from the real clock so it can never drift
    stale against budget_status's own ``datetime.now(UTC)`` lookup."""
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _config_with_state_dir(state_dir: Path) -> OrchestratorConfig:
    """An OrchestratorConfig whose runtime.state_dir points at ``state_dir``
    (the directory holding the api-budget.json ledger)."""
    return OrchestratorConfig(runtime=RuntimeConfig(state_dir=str(state_dir)))


def _plant_ledger(state_dir: Path, ledger: Ledger) -> None:
    """Write ``ledger`` to the canonical ledger path under ``state_dir``."""
    state_dir.mkdir(parents=True, exist_ok=True)
    save_ledger(ledger_path(state_dir), ledger)


def test_launch_api_worker_daily_budget_exhausted_refuses_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When today's spend is at the daily cap, launch_api_worker returns an
    error record and never reaches launch_claude_worker (no Popen). Emits an
    api_budget_refused event."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    # Default ApiBudgetConfig: max_usd_per_day=5.0, preflight_reserve_usd=1.0.
    # spend 5.0 + reserve 1.0 <= 5.0 is False -> daily exhausted.
    _plant_ledger(
        state_dir,
        Ledger(days={_today_utc(): DayBucket(usd=5.0)}, lifetime_usd=0.0),
    )

    # Sentinel: if launch_claude_worker is reached, the preflight failed to
    # refuse. Record the call so the assertion can prove it never ran.
    launch_calls: list[dict[str, Any]] = []

    def must_not_launch(*args, **kwargs):
        launch_calls.append({"args": args, "kwargs": kwargs})
        return ClaudeWorkerRecord(
            issue_number=args[0],
            branch=args[1],
            worktree_path="",
            prompt_path="",
            command=(),
            pid=999,
            started_at="",
            log_path="",
            error=None,
            adapter_kind="api",
            provider="kimi-k3",
        )

    monkeypatch.setattr(api_worker, "launch_claude_worker", must_not_launch)

    # Capture the best-effort event emission (api_worker imports log_event
    # locally from .instrumentation at call time, so patching the module
    # attribute reaches it).
    event_calls: list[dict[str, Any]] = []

    def fake_log_event(state_path, kind, payload, *, repo=None, correlation_id=None, level=None):
        event_calls.append(
            {
                "state_path": state_path,
                "kind": kind,
                "payload": payload,
                "level": level,
            }
        )

    monkeypatch.setattr(instrumentation, "log_event", fake_log_event)

    record = launch_api_worker(
        1514,
        "agent/issue-1514-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        config=_config_with_state_dir(state_dir),
    )

    # No launch happened (no Popen).
    assert launch_calls == []
    assert not record.ok
    assert record.error is not None
    assert "budget exhausted" in record.error
    assert "daily" in record.error
    assert record.pid is None
    assert record.adapter_kind == "api"
    assert record.provider == "kimi-k3"

    # An api_budget_refused warning event was emitted.
    assert len(event_calls) == 1
    assert event_calls[0]["kind"] == "api_budget_refused"
    assert event_calls[0]["level"] == "warning"
    assert event_calls[0]["payload"]["issue_number"] == 1514
    assert event_calls[0]["payload"]["provider"] == "kimi-k3"
    assert "daily" in event_calls[0]["payload"]["error"]

    # The error is durable in the sidecar (no key material).
    sidecar_path = sessions_dir / "issue-1514.api.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["error"] == record.error
    assert payload["provider"] == "kimi-k3"
    assert "sk-test-key-value-1234" not in json.dumps(payload)


def test_launch_api_worker_lifetime_budget_exhausted_refuses_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When lifetime spend is at the lifetime cap, launch_api_worker refuses
    and never reaches launch_claude_worker."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    # Default ApiBudgetConfig: lifetime_usd=15.0. lifetime_spent 15.0 < 15.0
    # is False -> lifetime exhausted. Today's spend is 0, so daily has headroom.
    _plant_ledger(state_dir, Ledger(lifetime_usd=15.0))

    launch_calls: list[dict[str, Any]] = []

    def must_not_launch(*args, **kwargs):
        launch_calls.append({"args": args, "kwargs": kwargs})
        return ClaudeWorkerRecord(
            issue_number=args[0],
            branch=args[1],
            worktree_path="",
            prompt_path="",
            command=(),
            pid=999,
            started_at="",
            log_path="",
            error=None,
            adapter_kind="api",
            provider="kimi-k3",
        )

    monkeypatch.setattr(api_worker, "launch_claude_worker", must_not_launch)
    monkeypatch.setattr(instrumentation, "log_event", lambda *a, **k: None)

    record = launch_api_worker(
        1514,
        "agent/issue-1514-lifetime",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        config=_config_with_state_dir(state_dir),
    )

    assert launch_calls == []
    assert not record.ok
    assert record.error is not None
    assert "budget exhausted" in record.error
    assert "lifetime" in record.error
    # Daily had headroom, so only the lifetime reason is present.
    assert "daily" not in record.error
    assert record.pid is None
    assert record.provider == "kimi-k3"


def test_launch_api_worker_budget_headroom_allows_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When spend is under both caps, the preflight allows the launch to
    proceed (regression guard: the preflight must not refuse healthy launches)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    # Spend well under both caps: today 1.0 / 5.0, lifetime 1.0 / 15.0.
    _plant_ledger(
        state_dir,
        Ledger(days={_today_utc(): DayBucket(usd=1.0)}, lifetime_usd=1.0),
    )

    record = launch_api_worker(
        1514,
        "agent/issue-1514-ok",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=_fake_claude_script(tmp_path),
        config=_config_with_state_dir(state_dir),
    )

    assert record.ok
    assert record.error is None
    assert record.pid is not None


def test_launch_api_worker_budget_preflight_skipped_without_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When config is None the budget preflight is skipped (the ledger path
    cannot be located without a runtime.state_dir). A directly-constructed
    config without a state_dir is the same defensive posture as the
    enabled/provider/auth-token re-checks. The launch proceeds even though a
    ledger under the default state root would be exhausted."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    record = launch_api_worker(
        1514,
        "agent/issue-1514-noconfig",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=_fake_claude_script(tmp_path),
        # config intentionally omitted (None) -> preflight skipped.
    )

    assert record.ok
    assert record.error is None
    assert record.pid is not None


def test_launch_api_worker_budget_preflight_missing_ledger_allows_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing ledger (no spend yet) is treated as empty and never refuses a
    launch — only real recorded spend against the caps does."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    state_dir = tmp_path / "state"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-key-value-1234")

    # No ledger planted; state_dir exists but is empty.
    state_dir.mkdir(parents=True, exist_ok=True)

    record = launch_api_worker(
        1514,
        "agent/issue-1514-fresh",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=_api_worker_config(),
        command_template=_fake_claude_script(tmp_path),
        config=_config_with_state_dir(state_dir),
    )

    assert record.ok
    assert record.error is None
    assert record.pid is not None
