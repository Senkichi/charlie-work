"""Wiring for the stalled lane's pipe-holder release (#2794): every seam the decision crosses.

``test_dws_stalled_release.py`` settles the decision and ``test_pipe_holder_release.py``
the kill; these pin the joints between them -- which harness may release, how the
shell serves the request, where the one-shot stamp is stored and read back, and the
config kill switch -- so a decision cannot be correct yet unreachable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from charlie_work import adapter_fate_profile
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.claude_code import _sidecar_path as claude_sidecar_path
from charlie_work.config import OrchestratorConfig, build_config_from_data
from charlie_work.config_validation import ConfigError
from charlie_work.dead_worker_sweep import apply_stalled
from charlie_work.dead_worker_sweep.stalled_model import (
    STALLED_REQUEST_TYPES,
    AdapterFacts,
    ReadAdapterProfile,
    ReleaseBlockedTool,
)
from charlie_work.worker import WorkerView, _from_claude_record, update_worker_log_stat
from charlie_work.write_gate import WriteGate

ISSUE = 5
NOW = datetime.now(UTC)


def _view(adapter_kind: str = "opencode", **kw: Any) -> WorkerView:
    base: dict[str, Any] = {
        "adapter_kind": adapter_kind,
        "issue_number": ISSUE,
        "repo_key": "",
        "pid": 5151,
        "started_at": NOW.isoformat(),
        "process_start_time": 1234.5,
        "log_path": "",
        "worktree_path": "",
        "error": None,
        "failure_kind": None,
        "reclaimed": None,
    }
    return WorkerView(**{**base, **kw})


def _serve(request: Any, w: WorkerView, gate: WriteGate, tmp_path: Path) -> Any:
    return apply_stalled._serve(
        request,
        w,
        sessions_dir=tmp_path,
        state_file=tmp_path / "state.json",
        config=OrchestratorConfig(),
        now=NOW,
        write_gate=gate,
    )


def _gate(tmp_path: Path, *, dry_run: bool) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=tmp_path / "state.json", repo="r")


def test_only_the_opencode_profile_may_release_a_blocked_tool() -> None:
    """The mechanism was reproduced on opencode alone; no other harness inherits it."""
    profiles = adapter_fate_profile._build_profiles()

    assert {kind for kind, p in profiles.items() if p.releases_blocked_tool} == {"opencode"}


@pytest.mark.parametrize(("kind", "expected"), [("opencode", True), ("devin", False)])
def test_the_shell_reports_the_profile_capability_to_the_decision(
    kind: str, expected: bool, tmp_path: Path
) -> None:
    facts = _serve(
        ReadAdapterProfile(ISSUE), _view(kind), _gate(tmp_path, dry_run=False), tmp_path
    )

    assert isinstance(facts, AdapterFacts)
    assert facts.can_release_blocked_tool is expected


def test_the_shell_serves_a_release_and_honours_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, int | None, float | None]] = []

    def fake(worktree: Path, pid: int | None, start: float | None) -> tuple[int, ...]:
        calls.append((worktree, pid, start))
        return (77,)

    monkeypatch.setattr(apply_stalled, "release_pipe_holders", fake)
    worktree = tmp_path / "wt"
    w = _view(worktree_path=str(worktree))

    live = _serve(ReleaseBlockedTool(ISSUE), w, _gate(tmp_path, dry_run=False), tmp_path)
    dry = _serve(ReleaseBlockedTool(ISSUE), w, _gate(tmp_path, dry_run=True), tmp_path)

    assert live == (77,)
    assert calls == [(worktree, 5151, 1234.5)], "dry-run must not reach the kill primitive"
    assert dry == ()


def test_the_release_request_is_a_member_of_the_stalled_request_set() -> None:
    """``run_flow`` rejects a request outside the set; an unlisted one would abort the pass."""
    assert ReleaseBlockedTool in STALLED_REQUEST_TYPES


def _opencode_sidecar(sessions_dir: Path, log: Path, **extra: Any) -> Path:
    path = claude_sidecar_path(sessions_dir, ISSUE, "opencode")
    path.write_text(
        json.dumps(
            {
                "issue_number": ISSUE,
                "branch": "agent/issue-5",
                "worktree_path": str(sessions_dir),
                "prompt_path": str(sessions_dir / "prompt.md"),
                "command": ["opencode"],
                "pid": 5151,
                "started_at": NOW.isoformat().replace("+00:00", "Z"),
                "log_path": str(log),
                "error": None,
                "failure_kind": None,
                "process_start_time": 1234.5,
                "reclaimed": None,
                "adapter_kind": "opencode",
                **extra,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_the_release_stamp_round_trips_from_the_stall_stamp_to_the_decision_view(
    tmp_path: Path,
) -> None:
    """Written by ``StampSidecar(tool_released_at=...)``, read back as ``WorkerView``."""
    log = tmp_path / "w.log"
    log.write_text("x\n", encoding="utf-8")
    sidecar = _opencode_sidecar(tmp_path, log)
    stamp = NOW.isoformat()

    update_worker_log_stat(tmp_path, _view(log_path=str(log)), tool_released_at=stamp)

    record = ClaudeWorkerRecord.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))
    assert record.tool_released_at == stamp
    assert _from_claude_record(record, "").tool_released_at == stamp


def test_a_later_stat_refresh_does_not_clear_the_release_stamp(tmp_path: Path) -> None:
    """The lane stamps the sidecar every pass; the one-shot marker must survive that."""
    log = tmp_path / "w.log"
    log.write_text("x\n", encoding="utf-8")
    sidecar = _opencode_sidecar(tmp_path, log, tool_released_at=NOW.isoformat())

    update_worker_log_stat(tmp_path, _view(log_path=str(log)))

    assert json.loads(sidecar.read_text(encoding="utf-8"))["tool_released_at"] == NOW.isoformat()


def test_a_sidecar_that_never_released_reads_as_not_released(tmp_path: Path) -> None:
    log = tmp_path / "w.log"
    sidecar = _opencode_sidecar(tmp_path, log)

    record = ClaudeWorkerRecord.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))

    assert record.tool_released_at is None


def test_the_kill_switch_defaults_on_and_is_a_typed_boolean() -> None:
    assert build_config_from_data({}).opencode.release_blocked_tool is True
    off = build_config_from_data({"opencode": {"release_blocked_tool": False}})
    assert off.opencode.release_blocked_tool is False
    with pytest.raises(ConfigError, match="release_blocked_tool"):
        build_config_from_data({"opencode": {"release_blocked_tool": "yes"}})
