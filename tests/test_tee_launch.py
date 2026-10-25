"""Issue #2814 rework round 4: the tee wiring extracted into
``charlie_work.tee_launch`` — unit tests for the extracted module's contract
that the end-to-end launch tests in
``tests/test_claude_code_adapter_launch_process.py`` cannot see.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from charlie_work import claude_code
from charlie_work.tee_launch import TEE_SPLITTER_PATH, TeeLaunchError, TeeLaunched, launch_tee


class _FakeStdin:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakePopen:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()


def _fake_popen_factory(record: list[tuple[tuple[str, ...], dict]], fail_on: int):
    """A ``popen_worker`` double: records ``(argv, kwargs)`` per call, returns
    a fake Popen, and raises ``OSError`` on call number ``fail_on`` (0=never)."""

    def fake_popen(argv: tuple[str, ...], **popen_kwargs: Any) -> Any:
        record.append((argv, popen_kwargs))
        if fail_on and len(record) == fail_on:
            raise OSError("mock spawn failure")
        return _FakePopen()

    return fake_popen


def test_tee_launch_devnull_splitter_output_and_stdin_handoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Round 4: the splitter child's stdout/stderr are DEVNULL (it must not
    inherit and hold the supervisor's output pipes for a worker's lifetime),
    the worker's stdout is the splitter's stdin, and the launcher releases its
    duplicate of the splitter stdin after the handoff."""
    log = tmp_path / "w.log"
    events = tmp_path / "w.events.jsonl"
    calls: list[tuple[tuple[str, ...], dict]] = []
    monkeypatch.setattr(claude_code, "popen_worker", _fake_popen_factory(calls, fail_on=0))

    result = launch_tee(
        command=("claude", "--print"),
        cwd=str(tmp_path),
        prompt_path=tmp_path / "prompt.md",
        feed_stdin=False,
        log_path=log,
        events_path=events,
        worker_env={"PATH": "x"},
    )

    assert isinstance(result, TeeLaunched)
    splitter_argv, splitter_kwargs = calls[0]
    assert splitter_argv == (sys.executable, str(TEE_SPLITTER_PATH), str(log), str(events))
    assert splitter_kwargs["stdin"] is subprocess.PIPE
    assert splitter_kwargs["stdout"] is subprocess.DEVNULL
    assert splitter_kwargs["stderr"] is subprocess.DEVNULL

    _worker_argv, worker_kwargs = calls[1]
    assert worker_kwargs["stdout"] is result.splitter.stdin
    # The launcher released its duplicate of the splitter's stdin handle.
    assert result.splitter.stdin.closed


def test_tee_launch_unopenable_destination_is_tee_launch_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A destination that cannot be pre-created (the events path is inside a
    missing directory) returns a TeeLaunchError with the same record message
    the old inline block produced, and never spawns a process."""
    calls: list[tuple[tuple[str, ...], dict]] = []
    monkeypatch.setattr(claude_code, "popen_worker", _fake_popen_factory(calls, fail_on=0))

    result = launch_tee(
        command=("claude",),
        cwd=str(tmp_path),
        prompt_path=tmp_path / "prompt.md",
        feed_stdin=False,
        log_path=tmp_path / "w.log",
        events_path=tmp_path / "missing-dir" / "w.events.jsonl",
        worker_env={"PATH": "x"},
    )

    assert isinstance(result, TeeLaunchError)
    assert "failed to create tee log files" in result.error
    assert not calls


def test_tee_launch_splitter_spawn_failure_is_tee_launch_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An OSError from the SPLITTER spawn returns a TeeLaunchError naming the
    splitter — exactly the message the old inline block recorded."""
    calls: list[tuple[tuple[str, ...], dict]] = []
    monkeypatch.setattr(claude_code, "popen_worker", _fake_popen_factory(calls, fail_on=1))

    result = launch_tee(
        command=("claude",),
        cwd=str(tmp_path),
        prompt_path=tmp_path / "prompt.md",
        feed_stdin=False,
        log_path=tmp_path / "w.log",
        events_path=tmp_path / "w.events.jsonl",
        worker_env={"PATH": "x"},
    )

    assert isinstance(result, TeeLaunchError)
    assert "failed to launch tee splitter" in result.error


def test_tee_launch_worker_spawn_failure_releases_splitter_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An OSError from the WORKER spawn closes the splitter's stdin (EOF ->
    the splitter exits) and re-raises; the caller's handler then produces the
    'failed to launch claude' record."""
    calls: list[tuple[tuple[str, ...], dict]] = []
    monkeypatch.setattr(claude_code, "popen_worker", _fake_popen_factory(calls, fail_on=2))

    with pytest.raises(OSError, match="mock spawn failure"):
        launch_tee(
            command=("claude",),
            cwd=str(tmp_path),
            prompt_path=tmp_path / "prompt.md",
            feed_stdin=False,
            log_path=tmp_path / "w.log",
            events_path=tmp_path / "w.events.jsonl",
            worker_env={"PATH": "x"},
        )
    assert calls[1][0][0] == "claude"
