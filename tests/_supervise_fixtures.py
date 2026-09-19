"""Shared fakes/helpers for the supervise test modules.

Hoisted verbatim out of ``tests/test_supervise.py`` (issue #1562,
Track 1) when that module was split into seam-named siblings -- the
``tests/_*.py`` hoisted-fixture convention is the sanctioned import target
for shared test helpers (see ``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest

from charlie_work.config import OrchestratorConfig, SupervisorConfig
from charlie_work.paths import resolved_layout
from charlie_work.subprocess_runner import RunResult
from charlie_work.workflow import CommandResult


class _FakePaths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.prs = root / "prs"
        self.state_file = root / "state.json"


class FakeApp:
    """Minimal OrchestratorApp stand-in for run_supervised tests.

    ``results`` is a list of CommandResult objects returned by successive
    ``loop()`` calls (cycles).
    """

    def __init__(
        self,
        tmp_path: Path,
        results: list[CommandResult],
        *,
        supervisor_cfg: SupervisorConfig | None = None,
    ) -> None:
        cfg_supervisor = supervisor_cfg if supervisor_cfg is not None else SupervisorConfig()
        self.config = OrchestratorConfig(supervisor=cfg_supervisor)
        self.paths = _FakePaths(tmp_path / ".var" / "charlie-work")
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self._sessions_dir = tmp_path / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        # Public layout contract mirroring OrchestratorApp.layout. Override
        # sessions_dir back onto the stub's own `_sessions_dir` (rather than
        # whatever resolved_layout derives from config.devin.sessions_dir) so
        # fixtures that write session files into that exact directory keep
        # working -- the rest of the resolved layout is unused by
        # run_supervised today but kept real (not stubbed) so this stays a
        # faithful stand-in.
        self.layout = replace(
            resolved_layout(self.config, tmp_path), sessions_dir=self._sessions_dir
        )
        self._results = list(results)
        self._call_count = 0
        # Issue #1339: records ensure_labels() invocations so a test can assert
        # the supervisor runs the startup label ensure exactly once.
        self.ensure_labels_calls = 0

    def _resolve(self, path_str: str) -> Path:
        """Resolve a config path string — returns sessions_dir for any input."""
        return self._sessions_dir

    def ensure_labels(self) -> CommandResult:
        self.ensure_labels_calls += 1
        return CommandResult(True, "labels ensured", {"labels": [], "missing": []})

    def loop(self, limit: Any = None, *, merge: Any = None) -> CommandResult:
        if self._call_count < len(self._results):
            result = self._results[self._call_count]
        else:
            # Default drained result when list exhausted
            result = _drained_result()
        self._call_count += 1
        return result


def _drained_result() -> CommandResult:
    """A fully drained pass result (nothing dispatched/merged, no open PRs)."""
    return CommandResult(
        True,
        "loop complete",
        {
            "dispatch": {"selected_count": 0},
            "dispatch_rework": {"selected_count": 0},
            "merges": [],
            "reviews": [],
            "errors": [],
            "open_tracked_prs": 0,
            "skipped_reviews": 0,
        },
    )


def _active_result(
    *,
    dispatched: int = 0,
    rework: int = 0,
    merged: int = 0,
    merge_failed: int = 0,
    open_prs: int = 0,
    warnings: list[str] | None = None,
) -> CommandResult:
    """A pass result with some activity.

    ``merged`` produces that many successful merge entries ("merged": True).
    ``merge_failed`` produces that many failed merge ATTEMPT entries
    ("merged": False) -- mirrors merge_ready() appending one entry per
    approved PR regardless of outcome (workflow.py merge_ready). Both land in
    the same "merges" list.
    """
    merges = [{"pr": i, "merged": True} for i in range(merged)]
    merges += [{"pr": 1000 + i, "merged": False} for i in range(merge_failed)]
    return CommandResult(
        True,
        "loop complete",
        {
            "dispatch": {"selected_count": dispatched},
            "dispatch_rework": {"selected_count": rework},
            "merges": merges,
            "reviews": [],
            "errors": [],
            "warnings": warnings if warnings is not None else [],
            "open_tracked_prs": open_prs,
            "skipped_reviews": 0,
        },
    )


class FakeClock:
    """Monotonically advancing fake clock.

    Advances by ``auto_advance`` on each ``sleep()`` call.
    """

    def __init__(self, start: float = 0.0, auto_advance: float = 0.0) -> None:
        self._now = start
        self._auto_advance = auto_advance
        self.sleep_calls: list[float] = []

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self._now += self._auto_advance if self._auto_advance else seconds


def _make_fake_runner(
    responses: list[RunResult],
) -> tuple[Callable[..., RunResult], list[tuple[list[str], Path, int]]]:
    """Return a callable that consumes ``responses`` and records its calls."""
    calls: list[tuple[list[str], Path, int]] = []

    def runner(command: list[str], *, cwd: Path, timeout_seconds: int) -> RunResult:
        calls.append((command, cwd, timeout_seconds))
        return responses.pop(0)

    return runner, calls


@pytest.fixture
def no_fleet_live_sessions(monkeypatch: Any) -> None:
    """Patch fleet live-session counting to zero so self_deploy tests stay hermetic."""
    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (0, []),
    )
