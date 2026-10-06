"""D7: the dead-worker sweep reads time through the clock host port."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg

from charlie_work.config import (
    AutoMergeConfig,
    DevinConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.devin_shell import SessionRecord
from charlie_work.host.fakes import FakeClock
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state


def test_dead_session_reclaim_stamps_redispatch_from_clock_port(
    tmp_path: Path, fake_host: Callable[..., Any]
) -> None:
    # Distinct from the real clock, so only the port can produce this stamp.
    frozen = datetime(2031, 3, 4, 5, 6, 7, tzinfo=UTC)
    fake_host(clock=FakeClock(frozen))
    config = OrchestratorConfig(
        auto_merge=AutoMergeConfig(required_checks=("Tests passed",)),
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "print('ok')")),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub()
    gh.issues = [
        {
            "number": 99,
            "title": "Fix thing",
            "url": "https://example.test/issues/99",
            "body": "Broken",
            "labels": [{"name": config.labels.in_progress}],
        }
    ]
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-99.log"
    log_path.write_text("work, then the process died.\n", encoding="utf-8")
    record = SessionRecord(
        issue_number=99,
        branch="agent/issue-99-x",
        worktree_path="/tmp/worktree-99",
        prompt_path="/tmp/prompt-99.md",
        command=("devin", "--prompt-file", "/tmp/prompt-99.md"),
        pid=None,
        started_at=(frozen - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
    )
    (sessions_dir / "issue-99.json").write_text(json.dumps(record.to_dict()), encoding="utf-8")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, paths.state_file, gh, config, write_gate=_wg(paths.state_file)
    )

    (stamp,) = load_state(paths.state_file)["issues"]["99"]["redispatch_at"]
    assert datetime.fromisoformat(stamp.replace("Z", "+00:00")) == frozen


def test_read_clock_samples_the_port(tmp_path: Path, fake_host: Callable[..., Any]) -> None:
    # Issue #2233: the sweep's ReadClock step must stamp SweepContext.now
    # from host.current().clock so a fake clock freezes every downstream
    # sweep decision -- under the unfixed code it read datetime.now(UTC).
    from charlie_work.dead_worker_sweep.apply_context import SweepContext
    from charlie_work.dead_worker_sweep.apply_requests_pre import read_clock
    from charlie_work.dead_worker_sweep.model import ReadClock
    from charlie_work.dead_worker_sweep.ports import ports_from_workflow

    frozen = datetime(2031, 3, 4, 5, 6, 7, tzinfo=UTC)
    fake_host(clock=FakeClock(frozen))
    state_file = tmp_path / "state.json"
    ctx = SweepContext(
        sessions_dir=tmp_path,
        state_file=state_file,
        config=OrchestratorConfig(),
        gh=None,
        write_gate=_wg(state_file),
        ports=ports_from_workflow(),
        review_callback=None,
        record_review_callback=None,
        enrich_checks_callback=None,
        fleet_dir_override=None,
        repo_root=None,
        worktrees_dir=None,
        state={},
        now=datetime(2000, 1, 1, tzinfo=UTC),
    )

    assert read_clock(ctx, ReadClock()) == frozen
    assert ctx.now == frozen
