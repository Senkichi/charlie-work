"""``run_captured``'s timeout bounds the whole process tree (issue #2139).

``subprocess.run(timeout=...)`` kills only the direct child on timeout and then
calls ``communicate()`` with no timeout; on Windows that blocks for as long as
any surviving grandchild holds the inherited stdout/stderr handles. These are
real-subprocess tests of that exact shape: a grandchild that inherits the pipes
and sleeps far longer than the timeout.

Each call runs in a daemon thread joined with a hard bound, so a regression
fails the test instead of hanging the suite; every spawned sleeper is killed in
``finally`` so a failing test never leaves a 120s process behind.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from charlie_work.process_utils import is_pid_alive
from charlie_work.subprocess_runner import RunResult, run_captured

# Hard bound on one ``run_captured`` call: the 2s timeout, the tree kill
# (Windows enumerates children via PowerShell), and the 5s bounded drain.
_CALL_BOUND_SECONDS = 15.0
_SLEEPER_SECONDS = 120


def _grandchild_script(*, then_exit: bool) -> str:
    """Spawn a sleeper grandchild that inherits stdout/stderr, record pids."""
    tail = "sys.exit(0)" if then_exit else f"time.sleep({_SLEEPER_SECONDS})"
    return (
        "import subprocess, sys, time\n"
        "gc = subprocess.Popen([sys.executable, '-c', "
        f"'import time; time.sleep({_SLEEPER_SECONDS})'])\n"
        "tmp = sys.argv[1] + '.tmp'\n"
        "with open(tmp, 'w') as fh:\n"
        "    fh.write(f'{gc.pid} {__import__(\"os\").getpid()}')\n"
        "__import__('os').replace(tmp, sys.argv[1])\n"
        "sys.stdout.write('started\\n'); sys.stdout.flush()\n"
        f"{tail}\n"
    )


def _run_bounded(command: list[str], cwd: Path, **kwargs: object) -> tuple[RunResult, float]:
    box: dict[str, RunResult] = {}

    def _target() -> None:
        box["result"] = run_captured(command, cwd=cwd, **kwargs)  # type: ignore[arg-type]

    start = time.monotonic()
    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(_CALL_BOUND_SECONDS + 15)
    elapsed = time.monotonic() - start
    assert "result" in box, f"run_captured did not return within {elapsed:.1f}s (hang)"
    return box["result"], elapsed


def _read_pids(pid_file: Path) -> list[int]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if pid_file.exists():
            return [int(part) for part in pid_file.read_text().split()]
        time.sleep(0.05)
    return []


def _force_kill(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False
        )
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _wait_dead(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not is_pid_alive(pid):
            return True
        time.sleep(0.1)
    return not is_pid_alive(pid)


@pytest.mark.timeout(90)
def test_timeout_kills_grandchild_holding_pipes(tmp_path: Path) -> None:
    pid_file = tmp_path / "pids.txt"
    command = [sys.executable, "-c", _grandchild_script(then_exit=False), str(pid_file)]
    pids: list[int] = []
    try:
        result, elapsed = _run_bounded(command, tmp_path, timeout_seconds=2)
        pids = _read_pids(pid_file)

        assert result.timed_out is True
        assert result.returncode is None
        assert result.error == "command timed out after 2s"
        assert elapsed < _CALL_BOUND_SECONDS, f"took {elapsed:.1f}s"
        assert len(pids) == 2, "child never recorded its grandchild pid"
        grandchild_pid = pids[0]
        assert _wait_dead(grandchild_pid), "grandchild survived the timeout"
    finally:
        for pid in pids or _read_pids(pid_file):
            _force_kill(pid)


@pytest.mark.timeout(90)
def test_child_exits_but_grandchild_holds_pipes_returns_bounded(tmp_path: Path) -> None:
    """The production shape: the launcher is gone, a descendant keeps the pipes."""
    pid_file = tmp_path / "pids.txt"
    command = [sys.executable, "-c", _grandchild_script(then_exit=True), str(pid_file)]
    pids: list[int] = []
    try:
        result, elapsed = _run_bounded(command, tmp_path, timeout_seconds=2)
        pids = _read_pids(pid_file)

        assert elapsed < _CALL_BOUND_SECONDS, f"took {elapsed:.1f}s"
        assert result.timed_out is True
        assert result.error == "command timed out after 2s"
    finally:
        for pid in pids or _read_pids(pid_file):
            _force_kill(pid)


@pytest.mark.timeout(60)
def test_stdin_none_gives_child_eof(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import sys; print(len(sys.stdin.read()))"]
    result, elapsed = _run_bounded(command, tmp_path, timeout_seconds=20)

    assert result.ok, result
    assert result.stdout.strip() == "0"
    assert elapsed < 15


_STDIN_HARNESS = (
    "import sys\n"
    "from charlie_work.subprocess_runner import run_captured\n"
    "r = run_captured([sys.executable, '-c', 'import sys; print(len(sys.stdin.read()))'],"
    " cwd='.', timeout_seconds=10)\n"
    "print(r.timed_out, r.stdout.strip())\n"
)


@pytest.mark.timeout(90)
def test_stdin_none_does_not_inherit_open_parent_stdin(tmp_path: Path) -> None:
    """Discriminating form: the *caller's* stdin is an open, never-written pipe.

    Under pytest the in-process test above passes either way (pytest's own
    stdin is not an open interactive handle), so this runs ``run_captured`` in
    a harness process whose stdin is held open -- the supervisor shape, where
    an inherited stdin let git's y/n prompt wait.
    """
    src_root = Path(run_captured.__code__.co_filename).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(src_root)}
    out_file = tmp_path / "harness.out"
    with out_file.open("w") as out:
        harness = subprocess.Popen(
            [sys.executable, "-c", _STDIN_HARNESS],
            cwd=tmp_path,
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            harness.wait(timeout=60)
        finally:
            if harness.poll() is None:
                _force_kill(harness.pid)
            if harness.stdin is not None:
                harness.stdin.close()

    assert out_file.read_text().split() == ["False", "0"], out_file.read_text()


def test_stdin_text_still_reaches_child(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"]
    result = run_captured(command, cwd=tmp_path, timeout_seconds=20, stdin="hello")

    assert result.ok, result
    assert result.stdout.strip() == "HELLO"


_PRINT_GIT_ENV = (
    "import os; print(os.environ.get('GIT_ASK_YESNO'), os.environ.get('GIT_TERMINAL_PROMPT'))"
)


def test_child_env_disables_git_prompts(tmp_path: Path) -> None:
    result = run_captured([sys.executable, "-c", _PRINT_GIT_ENV], cwd=tmp_path, timeout_seconds=20)

    assert result.ok, result
    assert result.stdout.split() == ["false", "0"]


def test_extra_env_overrides_git_prompt_defaults(tmp_path: Path) -> None:
    result = run_captured(
        [sys.executable, "-c", _PRINT_GIT_ENV],
        cwd=tmp_path,
        timeout_seconds=20,
        extra_env={"GIT_ASK_YESNO": "custom", "GIT_TERMINAL_PROMPT": "1"},
    )

    assert result.ok, result
    assert result.stdout.split() == ["custom", "1"]


def test_child_env_keeps_parent_environment(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import os; print('PATH' in os.environ)"]
    result = run_captured(command, cwd=tmp_path, timeout_seconds=20)

    assert result.ok, result
    assert result.stdout.strip() == "True"


def test_missing_binary_comes_back_as_value(tmp_path: Path) -> None:
    result = run_captured(["definitely-not-a-real-binary-2139"], cwd=tmp_path, timeout_seconds=5)

    assert result.ok is False
    assert result.returncode is None
    assert result.error


def test_cwd_none_inherits_current_directory() -> None:
    """``cwd=None`` means inherit -- ``str(None)`` ("None") must never reach
    ``Popen`` as a directory name (issue #2450)."""
    result = run_captured(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        cwd=None,
        timeout_seconds=20,
    )

    assert result.ok, result
    assert os.path.samefile(result.stdout.strip(), os.getcwd())
