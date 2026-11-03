"""The stalled lane's release, end to end through ``_handle_worker`` (issue #2794).

``test_dws_stalled_release.py`` settles the pure decision against canned answers and
``test_tool_release_wiring.py`` each seam on its own. Neither would notice a seam that
is correct alone and unreachable in combination -- the stamp the apply shell writes
not being the stamp the next pass reads, an event kind the shell cannot record, a
request the shell does not serve. This drives the real ``decide_stalled``, the real
``_serve`` and ``_apply``, and an opencode sidecar on disk, one pass at a time, with
each pass's ``WorkerView`` re-read from that sidecar as ``run_stalled_sweep`` does.

Only ``release_pipe_holders`` is patched (killing an orphan is covered against real
processes in ``test_pipe_holder_release.py``). The worker is a real sleeping process,
so "left running" and "reaped" are observed on the process, not inferred from a call.
(Named ``test_dws_*`` so the dormant-module guard does not read this as a
one-module-one-test file.)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psutil
import pytest
from charlie_work import worker
from charlie_work.claude_code import _sidecar_path as claude_sidecar_path
from charlie_work.config import OrchestratorConfig
from charlie_work.dead_worker_sweep import apply_stalled
from charlie_work.dead_worker_sweep.decide_stalled import RELEASE_RESUME_GRACE
from charlie_work.process_utils import get_process_start_time
from charlie_work.state import load_state
from charlie_work.write_gate import WriteGate

ISSUE = 9
RELEASED = (7701, 7702)
# The base interpreter is the process itself; a venv ``python.exe`` on Windows is a
# launcher whose pid is not the sleeper (see ``test_pipe_holder_release``).
_PYTHON = getattr(sys, "_base_executable", sys.executable)


@dataclass(frozen=True)
class Stalled:
    """One stalled opencode dispatch: its directories, its real process, its clock."""

    sessions_dir: Path
    state_file: Path
    worktree: Path
    sidecar: Path
    proc: subprocess.Popen[bytes]
    t0: datetime

    def sidecar_field(self, name: str) -> Any:
        return json.loads(self.sidecar.read_text(encoding="utf-8")).get(name)


@pytest.fixture
def stalled(tmp_path: Path) -> Iterator[Stalled]:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()
    t0 = datetime.now(UTC)
    long_ago = (t0 - timedelta(hours=3)).timestamp()
    # A frozen worktree and a frozen log: the stall signature of #2794.
    frozen = worktree / "frozen.txt"
    frozen.write_text("x\n", encoding="utf-8")
    log = sessions_dir / "worker.log"
    log.write_text("running\n", encoding="utf-8")
    for path in (frozen, log):
        os.utime(path, (long_ago, long_ago))

    proc = subprocess.Popen([_PYTHON, "-c", "import time; time.sleep(120)"], cwd=worktree)
    started = get_process_start_time(proc.pid)
    assert started is not None
    sidecar = claude_sidecar_path(sessions_dir, ISSUE, "opencode")
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": ISSUE,
                "branch": f"agent/issue-{ISSUE}",
                "worktree_path": str(worktree),
                "prompt_path": str(sessions_dir / "prompt.md"),
                "command": ["opencode", "run"],
                "pid": proc.pid,
                "started_at": (t0 - timedelta(hours=3)).isoformat().replace("+00:00", "Z"),
                "log_path": str(log),
                "error": None,
                "failure_kind": None,
                "process_start_time": started,
                "reclaimed": None,
                "adapter_kind": "opencode",
            }
        ),
        encoding="utf-8",
    )
    try:
        yield Stalled(sessions_dir, tmp_path / "state.json", worktree, sidecar, proc, t0)
    finally:
        for live in _tree(proc.pid):
            try:
                live.kill()
            except psutil.Error:
                pass
        proc.wait(timeout=10)


def _tree(pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except psutil.Error:
        return []


class _ReleaseSpy:
    """Stands in for ``release_pipe_holders`` and records how it was called."""

    def __init__(self, result: tuple[int, ...]) -> None:
        self.result = result
        self.calls: list[tuple[Path, int | None, float | None]] = []

    def __call__(self, worktree: Path, pid: int | None, start: float | None) -> tuple[int, ...]:
        self.calls.append((worktree, pid, start))
        return self.result


def _pass(s: Stalled, now: datetime, *, dry_run: bool = False) -> dict[str, int] | None:
    """One lane pass over the dispatch: the view is re-read from the sidecar on disk."""
    (view,) = worker.iter_workers(s.sessions_dir)
    return apply_stalled._handle_worker(
        view,
        sessions_dir=s.sessions_dir,
        state_file=s.state_file,
        config=OrchestratorConfig(),
        now=now,
        write_gate=WriteGate(dry_run=dry_run, state_path=s.state_file, repo="r"),
    )


def _event_kinds(s: Stalled) -> list[str]:
    return [e["kind"] for e in load_state(s.state_file).get("events", [])]


def test_release_then_grace_then_the_ordinary_reap(
    stalled: Stalled, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _ReleaseSpy(RELEASED)
    monkeypatch.setattr(apply_stalled, "release_pipe_holders", spy)
    s = stalled
    pid = s.proc.pid

    # Pass 1: the stalled worker's holders are released, the worker is left running.
    assert _pass(s, s.t0) is None
    assert spy.calls == [(s.worktree, pid, s.sidecar_field("process_start_time"))]
    assert s.sidecar_field("tool_released_at") == s.t0.isoformat()
    assert _event_kinds(s) == ["session_tool_released"]
    released_event = load_state(s.state_file)["events"][0]["payload"]
    assert released_event["released_pids"] == list(RELEASED)
    assert released_event["pid"] == pid and released_event["issue_number"] == ISSUE
    assert s.proc.poll() is None, "pass 1 must not reap the worker"

    # Pass 2, inside the resume grace: left alone, no second release, nothing new recorded.
    assert _pass(s, s.t0 + timedelta(minutes=2)) is None
    assert len(spy.calls) == 1
    assert s.sidecar_field("tool_released_at") == s.t0.isoformat()
    assert _event_kinds(s) == ["session_tool_released"]
    assert s.proc.poll() is None, "pass 2 must not reap the worker"

    # Pass 3, past the grace and still stalled: the pre-#2794 reap, without a second release.
    entry = _pass(s, s.t0 + RELEASE_RESUME_GRACE + timedelta(minutes=1))
    assert entry == {"issue": ISSUE, "pid": pid}
    assert len(spy.calls) == 1
    assert s.proc.wait(timeout=15) is not None, "pass 3 must reap the worker"
    assert _event_kinds(s) == ["session_tool_released", "session_stalled"]


def test_nothing_to_release_is_reaped_at_once_and_leaves_no_release_stamp(
    stalled: Stalled, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _ReleaseSpy(())
    monkeypatch.setattr(apply_stalled, "release_pipe_holders", spy)

    entry = _pass(stalled, stalled.t0)

    assert entry == {"issue": ISSUE, "pid": stalled.proc.pid}
    assert len(spy.calls) == 1
    assert stalled.sidecar_field("tool_released_at") is None
    assert stalled.proc.wait(timeout=15) is not None
    assert _event_kinds(stalled) == ["session_stalled"]


def test_dry_run_kills_nothing_stamps_nothing_and_never_reaches_the_release(
    stalled: Stalled, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The outcome production actually reaches under dry-run: a reap preview, no release."""
    spy = _ReleaseSpy(RELEASED)
    monkeypatch.setattr(apply_stalled, "release_pipe_holders", spy)

    entry = _pass(stalled, stalled.t0, dry_run=True)

    assert entry == {"issue": ISSUE, "pid": stalled.proc.pid}, "the reap is previewed"
    assert spy.calls == []
    assert stalled.sidecar_field("tool_released_at") is None
    assert stalled.proc.poll() is None, "dry-run must not kill the worker"
    assert "session_tool_released" not in _event_kinds(stalled)
