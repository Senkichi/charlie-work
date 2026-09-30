"""Issue #2010: headless claude-code worker permissions."""

from __future__ import annotations

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

from charlie_work import claude_code, worker_fate
from charlie_work.claude_code import (
    _WORKER_COMMAND_TEMPLATE,
    PROMPTING_PERMISSION_MODES,
    _sidecar_path,
    worker_permission_denied,
)
from charlie_work.config import ClaudeCodeConfig, OrchestratorConfig, WorkerRoleConfig
from charlie_work.doctor_worker_permissions import (
    _check_claude_worker_permissions,
    effective_permission_mode,
)

DENIAL = "I could not run the tests. If you approve command execution, I can finish these steps."

_classify_session_failure = partial(worker_fate.classify_for, "claude-code")


def _run_check(repo: Path, config: OrchestratorConfig) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []
    _check_claude_worker_permissions(lambda n, ok, d, **_: out.append((n, ok, d)), repo, config)
    return out


def _cc_config(command: tuple[str, ...] = ()) -> OrchestratorConfig:
    return replace(
        OrchestratorConfig(),
        worker=replace(WorkerRoleConfig(), harness="claude-code"),
        claude_code=ClaudeCodeConfig(command=command),
    )


def test_default_worker_template_is_not_a_prompting_mode() -> None:
    mode = effective_permission_mode(_WORKER_COMMAND_TEMPLATE)
    assert mode == "bypassPermissions"
    assert mode not in PROMPTING_PERMISSION_MODES
    assert claude_code._REVIEW_COMMAND_TEMPLATE[-1] == "plan"


def test_doctor_flags_prompting_mode_without_bash_allow(tmp_path: Path) -> None:
    cfg = _cc_config(("claude", "-p", "--permission-mode", "acceptEdits"))
    ((name, ok, detail),) = _run_check(tmp_path, cfg)
    assert not ok and "acceptEdits" in detail


def test_doctor_flags_equals_form_and_missing_flag(tmp_path: Path) -> None:
    assert not _run_check(tmp_path, _cc_config(("claude", "-p", "--permission-mode=plan")))[0][1]
    assert not _run_check(tmp_path, _cc_config(("claude", "-p")))[0][1]


def test_doctor_passes_default_bypass_and_bash_allow(tmp_path: Path) -> None:
    assert _run_check(tmp_path, _cc_config())[0][1]
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"allow": ["Bash(git status:*)"]}}), encoding="utf-8"
    )
    cfg = _cc_config(("claude", "-p", "--permission-mode", "acceptEdits"))
    assert _run_check(tmp_path, cfg)[0][1]


def test_doctor_skips_non_claude_code_harness(tmp_path: Path) -> None:
    assert _run_check(tmp_path, OrchestratorConfig()) == []


def test_denial_tail_classified_permission_denied_not_blocked(tmp_path: Path) -> None:
    log = tmp_path / "w.log"
    log.write_text("edits done\n" + DENIAL, encoding="utf-8")
    assert _classify_session_failure(log) == ("permission_denied", None)


def _seed_sidecar(sessions: Path, log: Path) -> None:
    path = _sidecar_path(sessions, 7, "claude-code")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"log_path": str(log)}), encoding="utf-8")


def test_worker_permission_denied_from_log_or_detail(tmp_path: Path) -> None:
    sessions = tmp_path / "s"
    log = tmp_path / "w.log"
    log.write_text(DENIAL, encoding="utf-8")
    _seed_sidecar(sessions, log)
    assert worker_permission_denied(sessions, 7)
    assert worker_permission_denied(tmp_path / "none", 8, DENIAL)
    assert not worker_permission_denied(tmp_path / "none", 8, "cross-repo scope")
    log.write_text("all good", encoding="utf-8")
    assert not worker_permission_denied(sessions, 7)
