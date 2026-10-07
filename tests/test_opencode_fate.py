"""opencode fate/classification wiring: the provider-error digest, quota
classification through the fate profile, early-kill health, data-dir lifecycle,
requires_model config rule, provider-qualified model family, orphan sweep."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from _rework_dispatch_fixtures import _wg

from charlie_work import claude_code, worker_fate
from charlie_work.config import (
    ConfigError,
    OrchestratorConfig,
    RuntimeConfig,
    WorkerRoleConfig,
    build_config_from_data,
)
from charlie_work.devin_failure_classification import get_rate_limit_defer_until
from charlie_work.harnesses import HARNESS_REGISTRY
from charlie_work.opencode_log import opencode_data_dir, provider_error_digest
from charlie_work.opencode_worker import DEFAULT_COMMAND_TEMPLATE
from charlie_work.role_chain import RoleEntry, model_family
from charlie_work.worker import WorkerHealth, WorkerView, classify_worker_health, iter_workers

FINAL_ERROR = (
    '{"type":"error","error":{"name":"APIError","data":{"message":"Usage limit reached",'
    '"statusCode":429,"isRetryable":true,"responseBody":"{\\"type\\":\\"error\\",'
    '\\"error\\":{\\"type\\":\\"GoUsageLimitError\\"}}"}}}'
)
PRINT_LOG_ERROR = (
    "timestamp=2026-10-06T12:00:00.000Z level=ERROR "
    'message="stream error" error.error="AI_APICallError: Usage limit reached"'
)
# "usage limit" appears only inside a tool_use event (the worker read throttle code).
TOOL_OUTPUT_ONLY = (
    '{"type":"step_start"}\n'
    '{"type":"tool_use","part":{"tool":"read","state":{"output":'
    '"if \\"usage limit\\" in tail: rate limit exceeded; quota exhausted"}}}\n'
)


def _log(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "issue-1.claude.log"
    path.write_text(text, encoding="utf-8")
    return path


# --- opencode_log -----------------------------------------------------------


def test_default_command_template_is_json_stream_with_print_logs() -> None:
    assert DEFAULT_COMMAND_TEMPLATE[:2] == ("opencode", "run")
    assert "--print-logs" in DEFAULT_COMMAND_TEMPLATE
    assert "--format" in DEFAULT_COMMAND_TEMPLATE


def test_opencode_data_dir_layout(tmp_path: Path) -> None:
    assert opencode_data_dir(tmp_path, 12) == tmp_path / "opencode-data" / "issue-12"


def test_digest_keeps_final_error_event_with_status_and_body() -> None:
    digest = provider_error_digest('{"type":"step_start"}\n' + FINAL_ERROR + "\n")
    assert "Usage limit reached" in digest
    assert "status=429" in digest
    assert "GoUsageLimitError" in digest


def test_digest_keeps_print_logs_error_lines() -> None:
    digest = provider_error_digest(PRINT_LOG_ERROR + "\n")
    assert "Usage limit reached" in digest


def test_digest_drops_tool_output_and_forged_level_error_in_tool_json() -> None:
    forged = (
        '{"type":"tool_use","part":{"state":{"output":"timestamp=x level=ERROR '
        'error.error=\\"usage limit\\""}}}\n'
    )
    assert provider_error_digest(TOOL_OUTPUT_ONLY) == ""
    assert provider_error_digest(forged) == ""


def test_digest_resets_on_progress_event() -> None:
    """A 429 opencode retried through must not classify a later unrelated death."""
    text = PRINT_LOG_ERROR + '\n{"type":"text","part":{"text":"recovered"}}\n'
    assert provider_error_digest(text) == ""
    text = FINAL_ERROR + '\n{"type":"step_start"}\n' + PRINT_LOG_ERROR + "\n"
    assert provider_error_digest(text).count("Usage limit reached") == 1


# --- classification through the fate profile ---------------------------------


def test_default_quota_markers_include_opencode_go_and_free_limits() -> None:
    markers = RuntimeConfig().quota_error_markers
    assert "GoUsageLimitError" in markers
    assert "FreeUsageLimitError" in markers


@pytest.mark.parametrize("line", [FINAL_ERROR, PRINT_LOG_ERROR], ids=["error-event", "print-logs"])
def test_classify_for_opencode_quota_exhausted(tmp_path: Path, line: str) -> None:
    log = _log(tmp_path, '{"type":"step_start"}\n' + line + "\n")
    kind, until = worker_fate.classify_for(
        "opencode", log, quota_error_markers=RuntimeConfig().quota_error_markers
    )
    assert kind == "quota_exhausted"
    assert until is not None


def test_classify_for_opencode_ignores_usage_limit_in_tool_output(tmp_path: Path) -> None:
    log = _log(tmp_path, TOOL_OUTPUT_ONLY)
    markers = RuntimeConfig().quota_error_markers
    assert worker_fate.classify_for("opencode", log, quota_error_markers=markers) == (None, None)
    # Positive control: the same raw tail DOES classify for a harness with no digest.
    kind, _ = worker_fate.classify_for("claude-code", log, quota_error_markers=markers)
    assert kind is not None


def test_get_rate_limit_defer_until_applies_log_digest(tmp_path: Path) -> None:
    text = TOOL_OUTPUT_ONLY + "\nrate limit exceeded, resets in 5 minutes\n"
    log = _log(tmp_path, text)
    assert get_rate_limit_defer_until(log, 5) is not None  # control: raw text matches
    assert get_rate_limit_defer_until(log, 5, log_digest=provider_error_digest) is None


# --- classify_worker_health early kill ---------------------------------------


def _view(log: Path) -> WorkerView:
    return WorkerView(
        adapter_kind="opencode",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        process_start_time=1710000000.0,
        log_path=str(log),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
        provider="opencode-go",
    )


def _health(view: WorkerView) -> WorkerHealth:
    with patch("charlie_work.worker_fate.is_alive", return_value=True):
        return classify_worker_health(view, OrchestratorConfig(), datetime.now(UTC))


def test_health_opencode_quota_record_is_dead_immediately(tmp_path: Path) -> None:
    assert _health(_view(_log(tmp_path, '{"type":"step_start"}\n' + PRINT_LOG_ERROR + "\n"))) == (
        WorkerHealth.DEAD
    )


def test_health_opencode_quota_text_in_tool_output_is_not_dead(tmp_path: Path) -> None:
    assert _health(_view(_log(tmp_path, TOOL_OUTPUT_ONLY))) == WorkerHealth.HEALTHY


# --- data dir lifecycle ------------------------------------------------------


def test_reap_sidecar_removes_opencode_data_dir(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    sidecar = sessions / "issue-5.opencode.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 5,
                "branch": "b",
                "worktree_path": "",
                "prompt_path": "",
                "command": [],
                "pid": 999_999_937,
                "started_at": "2026-10-06T00:00:00Z",
                "log_path": str(sessions / "issue-5.claude.log"),
                "error": None,
                "adapter_kind": "opencode",
            }
        ),
        encoding="utf-8",
    )
    data = opencode_data_dir(sessions, 5)
    (data / "sub").mkdir(parents=True)
    (data / "sub" / "opencode.db").write_text("x", encoding="utf-8")
    other = opencode_data_dir(sessions, 6)
    other.mkdir(parents=True)

    [view] = iter_workers(sessions)
    view.reap_sidecar(sessions)

    assert not sidecar.exists()
    assert not data.exists()
    assert other.exists()  # another worker's data dir is untouched


def test_launch_wipes_stale_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from charlie_work.config import OpenCodeConfig
    from charlie_work.opencode_worker import launch_opencode_worker
    from charlie_work.worktree import WorktreeInfo

    def fake_create_worktree(repo_root, branch, **_kw):
        path = tmp_path / "wt"
        path.mkdir(exist_ok=True)
        return WorktreeInfo(path=path, branch=branch, venv_junction=None)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "host"))
    sessions = tmp_path / "sessions"
    stale = opencode_data_dir(sessions, 9) / "opencode.db"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")
    config = OrchestratorConfig(
        worker=WorkerRoleConfig(harness="opencode", model="glm-5.3-flash"),
        opencode=OpenCodeConfig(),
    )
    record = launch_opencode_worker(
        9,
        "agent/issue-9",
        "p",
        repo_root=tmp_path,
        sessions_dir=sessions,
        command_template=(os.sys.executable, "-c", "pass"),
        config=config,
    )
    assert record.ok, record.error
    assert not stale.exists()


# --- requires_model / model_family -------------------------------------------


def test_requires_model_declared_for_opencode_only() -> None:
    assert HARNESS_REGISTRY["opencode"].requires_model is True
    assert [n for n, c in HARNESS_REGISTRY.items() if c.requires_model] == ["opencode"]


def test_empty_model_rejected_for_opencode_primary_without_fallbacks() -> None:
    with pytest.raises(ConfigError, match=r"worker\.model"):
        build_config_from_data({"worker": {"harness": "opencode", "model": ""}})


def test_empty_model_rejected_for_opencode_fallback_entry() -> None:
    data = {
        "worker": {
            "harness": "claude-code",
            "model": "claude-sonnet-5-5",
            "fallbacks": [{"harness": "opencode", "model": ""}],
        }
    }
    with pytest.raises(ConfigError, match=r"fallbacks"):
        build_config_from_data(data)


def test_opencode_with_model_and_claude_code_without_model_load() -> None:
    cfg = build_config_from_data({"worker": {"harness": "opencode", "model": "glm-5.3-flash"}})
    assert cfg.worker.harness == "opencode"
    build_config_from_data({"worker": {"harness": "claude-code", "model": ""}})


@pytest.mark.parametrize(
    ("entry", "family"),
    [
        (RoleEntry("opencode", "opencode-go/glm-5.3-flash"), "zhipu"),
        (RoleEntry("opencode", "glm-5.3-flash"), "zhipu"),
        (RoleEntry("opencode", "anthropic/claude-sonnet-5-5"), "anthropic"),
        (RoleEntry("opencode", "opencode-go/unknown-model"), None),
    ],
)
def test_model_family_strips_provider_prefix(entry: RoleEntry, family: str | None) -> None:
    assert model_family(entry) == family


# --- orphan sweep -------------------------------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="orphan sweep is Windows-only")
def test_orphan_sweep_covers_opencode_sidecars(tmp_path: Path) -> None:
    from charlie_work.dead_worker_sweep.effects_sessions import (
        _sweep_orphan_processes_for_dead_sessions,
    )
    from charlie_work.paths import runtime_paths

    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "issue-3.opencode.json").write_text(
        json.dumps(
            {
                "issue_number": 3,
                "branch": "b",
                "worktree_path": "/dead/opencode-wt",
                "prompt_path": "",
                "command": [],
                "pid": 999_999_937,
                "started_at": "2026-10-06T00:00:00Z",
                "log_path": "",
                "error": None,
                "adapter_kind": "opencode",
            }
        ),
        encoding="utf-8",
    )
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    swept: list[str] = []

    def fake_sweep(worktree_path: str):
        swept.append(worktree_path)
        return []

    with (
        patch("charlie_work.worker_fate.is_alive", return_value=False),
        patch(
            "charlie_work.dead_worker_sweep.effects_sessions.sweep_orphan_processes",
            side_effect=fake_sweep,
        ),
    ):
        _sweep_orphan_processes_for_dead_sessions(
            sessions, paths.state_file, config, write_gate=_wg(paths.state_file)
        )

    assert swept == ["/dead/opencode-wt"]
