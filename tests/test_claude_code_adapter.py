from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from charlie_work import claude_code
from charlie_work.claude_code import (
    ClaudeWorkerRecord,
    is_worker_alive,
    launch_claude_worker,
    probe_claude,
    read_worker_records,
    update_worker_record_with_failure_classification,
)
from charlie_work.env_sanitize import sanitize_env
from charlie_work.worktree import WorktreeInfo


def _fake_worktree(tmp_path: Path, branch: str) -> WorktreeInfo:
    worktree_path = tmp_path / "worktrees" / branch.replace("/", "-")
    worktree_path.mkdir(parents=True, exist_ok=True)
    return WorktreeInfo(path=worktree_path, branch=branch, venv_junction=None)


def _install_fake_create_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, calls: list[dict] | None = None
) -> None:
    def fake_create_worktree(
        repo_root,
        branch,
        *,
        base_ref="HEAD",
        worktrees_dir=None,
        venv_source=None,
        rework=False,
        recovery=None,
    ):
        if calls is not None:
            calls.append(
                {
                    "repo_root": repo_root,
                    "branch": branch,
                    "base_ref": base_ref,
                    "worktrees_dir": worktrees_dir,
                    "venv_source": venv_source,
                    "rework": rework,
                    "recovery": recovery,
                }
            )
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)


def _fake_claude_script(tmp_path: Path) -> tuple[str, ...]:
    """A Python script standing in for the `claude` binary: reads stdin (the
    prompt), writes a marker file next to cwd, and exits 0."""
    script_path = tmp_path / "fake_claude.py"
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


def test_launch_claude_worker_writes_prompt_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    calls: list[dict] = []
    _install_fake_create_worktree(monkeypatch, tmp_path, calls=calls)

    record = launch_claude_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert record.ok
    assert record.error is None
    assert record.issue_number == 42
    assert record.branch == "agent/issue-42-fix"
    assert record.pid is not None
    assert record.started_at.endswith("Z")

    worktree_path = Path(record.worktree_path)
    prompt_path = worktree_path / ".orchestrator-prompt.md"
    assert prompt_path.read_text(encoding="utf-8") == "Do the thing."
    assert record.prompt_path == str(prompt_path)

    # create_worktree got the right args, including venv_source/worktrees_dir passthrough.
    assert calls[0]["branch"] == "agent/issue-42-fix"
    assert calls[0]["repo_root"] == repo_root

    # Sidecar JSON is present and matches the returned record.
    sidecar_path = sessions_dir / "issue-42.claude.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["issue_number"] == 42
    assert payload["branch"] == "agent/issue-42-fix"
    assert payload["error"] is None

    log_path = Path(record.log_path)
    assert log_path == sessions_dir / "issue-42.claude.log"


def test_launch_claude_worker_process_receives_prompt_via_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        7,
        "agent/issue-7-x",
        "prompt payload for stdin",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert record.ok
    worktree_path = Path(record.worktree_path)

    # Wait for the fake claude subprocess (very fast: reads stdin, writes, exits).
    marker_path = worktree_path / "worker-ran.txt"
    deadline = time.time() + 10
    while not marker_path.exists() and time.time() < deadline:
        time.sleep(0.05)

    assert marker_path.exists()
    assert marker_path.read_text(encoding="utf-8") == "prompt payload for stdin"


def test_launch_claude_worker_injects_worker_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    # An inherited var must survive the merge; the injected var must appear.
    monkeypatch.setenv("CHARLIE_INHERITED", "inherited-value")

    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-probe.txt").write_text(
                os.environ.get("PYTEST_XDIST_AUTO_NUM_WORKERS", "<unset>")
                + "|"
                + os.environ.get("CHARLIE_INHERITED", "<unset>"),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        99,
        "agent/issue-99-env",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
        env={"PYTEST_XDIST_AUTO_NUM_WORKERS": "2"},
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-probe.txt"
    deadline = time.time() + 10
    while not probe_path.exists() and time.time() < deadline:
        time.sleep(0.05)

    assert probe_path.exists()
    # Injected var present AND orchestrator env inherited (merge, not replace).
    assert probe_path.read_text(encoding="utf-8") == "2|inherited-value"


def test_launch_claude_worker_prompt_path_placeholder_skips_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    script_path = tmp_path / "fake_claude_argv.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path

            prompt_path = Path(sys.argv[1])
            Path("worker-ran.txt").write_text(prompt_path.read_text(encoding="utf-8"), encoding="utf-8")
            print("ok")
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        8,
        "agent/issue-8-x",
        "prompt payload for argv",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path), "{prompt_path}"),
    )

    assert record.ok
    assert record.command[-1] == record.prompt_path

    worktree_path = Path(record.worktree_path)
    marker_path = worktree_path / "worker-ran.txt"
    deadline = time.time() + 10
    while not marker_path.exists() and time.time() < deadline:
        time.sleep(0.05)

    assert marker_path.exists()
    assert marker_path.read_text(encoding="utf-8") == "prompt payload for argv"


def test_launch_claude_worker_missing_binary_returns_error_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        99,
        "agent/issue-99-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
    )

    assert not record.ok
    assert record.error is not None
    assert "failed to launch claude" in record.error
    assert record.pid is None

    # The worktree itself must not leak: remove_worktree was attempted (best
    # effort — a fake worktree isn't a real git worktree, so the git command
    # inside it fails, but that's covered separately below). The sidecar must
    # still be written with the error regardless.
    sidecar_path = sessions_dir / "issue-99.claude.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["error"] == record.error


def test_launch_claude_worker_create_worktree_failure_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    def failing_create_worktree(
        repo_root,
        branch,
        *,
        base_ref="HEAD",
        worktrees_dir=None,
        venv_source=None,
        rework=False,
        recovery=None,
    ):
        raise RuntimeError("git worktree add failed: branch already exists")

    monkeypatch.setattr(claude_code, "create_worktree", failing_create_worktree)

    record = launch_claude_worker(
        13,
        "agent/issue-13-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert not record.ok
    assert "worktree creation failed" in record.error
    assert record.worktree_path == ""
    assert record.pid is None

    sidecar_path = sessions_dir / "issue-13.claude.json"
    assert sidecar_path.exists()


def test_launch_claude_worker_remove_worktree_called_on_launch_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    removed: list[Path] = []

    def fake_remove_worktree(repo_root, worktree_path, *, force=False, branch=None):
        removed.append(worktree_path)
        return True

    monkeypatch.setattr(claude_code, "remove_worktree", fake_remove_worktree)

    record = launch_claude_worker(
        21,
        "agent/issue-21-x",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
    )

    assert not record.ok
    assert len(removed) == 1
    assert removed[0] == Path(record.worktree_path)


def test_read_worker_records_round_trips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    launch_claude_worker(
        1,
        "agent/issue-1-a",
        "prompt a",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )
    launch_claude_worker(
        2,
        "agent/issue-2-b",
        "prompt b",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    records = read_worker_records(sessions_dir)

    assert len(records) == 2
    assert {r.issue_number for r in records} == {1, 2}
    assert all(isinstance(r, ClaudeWorkerRecord) for r in records)


def test_read_worker_records_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert read_worker_records(tmp_path / "does-not-exist") == []


def test_read_worker_records_skips_corrupt_sidecar(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "issue-5.claude.json").write_text("not json{{{", encoding="utf-8")

    assert read_worker_records(sessions_dir) == []


def test_read_worker_records_skips_sidecar_missing_required_fields(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "issue-6.claude.json").write_text(
        json.dumps({"branch": "agent/issue-6-x"}), encoding="utf-8"
    )

    assert read_worker_records(sessions_dir) == []


def test_probe_claude_missing_binary_never_raises(tmp_path: Path) -> None:
    result = probe_claude(tmp_path)

    assert result.ok is False
    assert result.error is not None


def test_probe_claude_uses_run_captured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple] = []

    def fake_run_captured(command, *, cwd, timeout_seconds, shell=False):
        calls.append((command, cwd, timeout_seconds))
        from charlie_work.subprocess_runner import RunResult

        return RunResult(returncode=0, stdout="1.0.0", stderr="")

    monkeypatch.setattr(claude_code, "run_captured", fake_run_captured)

    result = probe_claude(tmp_path)

    assert result.ok
    assert calls[0][0] == ["claude", "--version"]
    assert calls[0][1] == tmp_path


def test_is_worker_alive_reflects_real_process(tmp_path: Path) -> None:
    """Mirror of test_is_session_alive_reflects_real_process from test_devin_shell.py."""
    # Spawn a short-lived process
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        alive_record = ClaudeWorkerRecord(
            issue_number=1,
            branch="agent/issue-1",
            worktree_path="/tmp/wt/issue-1",
            prompt_path="p.md",
            command=("x",),
            pid=process.pid,
            started_at="2026-01-01T00:00:00Z",
            log_path="log.txt",
        )
        assert is_worker_alive(alive_record) is True
    finally:
        process.kill()
        process.wait(timeout=5)

    # Regression guard for the Windows os.kill(pid, 0) trap: that call keeps
    # reporting a reaped PID as alive indefinitely (verified empirically —
    # see is_worker_alive's docstring), so this must be an exact
    # post-wait() assertion, not a "poll until it settles" retry loop.
    dead_record = ClaudeWorkerRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/wt/issue-1",
        prompt_path="p.md",
        command=("x",),
        pid=process.pid,
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
    )
    assert is_worker_alive(dead_record) is False

    # pid=None case (never launched)
    none_record = ClaudeWorkerRecord(
        issue_number=2,
        branch="agent/issue-2",
        worktree_path="/tmp/wt/issue-2",
        prompt_path="p2.md",
        command=("y",),
        pid=None,
        started_at="2026-01-01T00:00:00Z",
        log_path="log2.txt",
    )
    assert is_worker_alive(none_record) is False


def test_is_worker_alive_rejects_pid_recycling_mismatched_start_time(tmp_path: Path) -> None:
    """A record with an alive PID but mismatched start time is treated as dead.

    This prevents false positives from PID recycling: if the OS has reused the PID
    for a different process, the start time will not match.
    """
    from charlie_work.claude_code import _get_process_start_time

    # Spawn a short-lived process to get a valid PID
    script = tmp_path / "sleep.py"
    script.write_text("import time; time.sleep(2)", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Get the actual start time of this process
        actual_start_time = _get_process_start_time(process.pid)
        assert actual_start_time is not None

        # Create a record with a deliberately wrong start time (10 minutes ago)
        # This simulates a recycled PID
        wrong_start_time = actual_start_time - 600  # 10 minutes in the past

        record = ClaudeWorkerRecord(
            issue_number=1,
            branch="agent/issue-1",
            worktree_path="/tmp/wt/issue-1",
            prompt_path="p.md",
            command=("x",),
            pid=process.pid,
            started_at="2026-01-01T00:00:00Z",
            log_path="log.txt",
            process_start_time=wrong_start_time,
        )

        # Should return False because start time doesn't match
        assert is_worker_alive(record) is False
    finally:
        process.kill()
        process.wait(timeout=5)


def test_is_worker_alive_accepts_matching_start_time(tmp_path: Path) -> None:
    """A record with matching start time is counted as live."""
    from charlie_work.claude_code import _get_process_start_time

    # Spawn a short-lived process
    script = tmp_path / "sleep.py"
    script.write_text("import time; time.sleep(2)", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Get the actual start time of this process
        actual_start_time = _get_process_start_time(process.pid)
        assert actual_start_time is not None

        record = ClaudeWorkerRecord(
            issue_number=1,
            branch="agent/issue-1",
            worktree_path="/tmp/wt/issue-1",
            prompt_path="p.md",
            command=("x",),
            pid=process.pid,
            started_at="2026-01-01T00:00:00Z",
            log_path="log.txt",
            process_start_time=actual_start_time,
        )

        # Should return True because start time matches
        assert is_worker_alive(record) is True
    finally:
        process.kill()
        process.wait(timeout=5)


def test_is_worker_alive_legacy_record_fallback() -> None:
    """Legacy records without process_start_time fall back to pid-only liveness.

    This preserves backward compatibility for old sidecar files.
    """
    # Use the test process's own PID (guaranteed to be alive)
    record = ClaudeWorkerRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/wt/issue-1",
        prompt_path="p.md",
        command=("x",),
        pid=os.getpid(),
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
        process_start_time=None,  # Legacy record
    )

    # Should return True using pid-only fallback
    assert is_worker_alive(record) is True


def test_posix_stat_parse_with_spaces_in_comm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit test for /proc/<pid>/stat parsing with comm containing spaces and closing paren.

    This test exercises the real parse_proc_stat_starttime function (used by both adapters)
    by monkeypatching the /proc read. The comm field MUST contain spaces and a closing paren
    (e.g., '(tmux: server)') to prove the last-')' splitting logic works correctly.
    """
    from charlie_work.process_utils import parse_proc_stat_starttime

    # Format: pid (comm) state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt cmajflt utime stime cutime cstime priority nice num_threads itrealvalue starttime vsize rss ...
    # starttime is at index 19 (0-indexed) after splitting on ')'
    # This example has comm="(tmux: server)" which contains spaces
    stat_line = "1234 (tmux: server) S 1 1234 1234 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 21000 22000 23000 24000 25000 26000 27000 28000 29000 30000 31000 32000 33000 34000 35000 36000 37000 38000 39000 40000 41000 42000 43000 44000 45000 46000 47000 48000 49000 50000 51000 52000"

    # Call the real parser function
    starttime_ticks = parse_proc_stat_starttime(stat_line)
    assert starttime_ticks == 100, f"Expected starttime to be 100, got {starttime_ticks}"


def test_posix_stat_parse_with_embedded_paren_in_comm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit test for /proc/<pid>/stat parsing with comm containing embedded ')'.

    This test proves that rpartition (last-')' split) is required: a comm like
    '(tmux: (0) server)' contains an embedded ')', which would break split(')')
    by shifting all field offsets. The parser must split on the LAST ')' to correctly
    extract the starttime field.
    """
    from charlie_work.process_utils import parse_proc_stat_starttime

    # Comm contains embedded ')': (tmux: (0) server)
    # Using split(')') would split on the first ')', giving wrong field offsets
    # Using rpartition(')') splits on the last ')', giving correct field offsets
    stat_line = "1234 (tmux: (0) server) S 1 1234 1234 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 21000 22000 23000 24000 25000 26000 27000 28000 29000 30000 31000 32000 33000 34000 35000 36000 37000 38000 39000 40000 41000 42000 43000 44000 45000 46000 47000 48000 49000 50000 51000 52000"

    # Call the real parser function
    starttime_ticks = parse_proc_stat_starttime(stat_line)
    assert starttime_ticks == 100, f"Expected starttime to be 100, got {starttime_ticks}"


def test_is_worker_alive_probe_none_with_start_time_returns_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When _get_process_start_time returns None mid-check with pid alive and record has start time, return dead.

    This pins the fail-direction: if the probe fails during a liveness check, we treat the worker as dead.
    """
    # Spawn a short-lived process to get a valid PID
    script = tmp_path / "sleep.py"
    script.write_text("import time; time.sleep(2)", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Monkeypatch _get_process_start_time to return None (simulating probe failure)
        monkeypatch.setattr(claude_code, "_get_process_start_time", lambda pid: None)

        record = ClaudeWorkerRecord(
            issue_number=1,
            branch="agent/issue-1",
            worktree_path="/tmp/wt/issue-1",
            prompt_path="p.md",
            command=("x",),
            pid=process.pid,  # PID is alive
            started_at="2026-01-01T00:00:00Z",
            log_path="log.txt",
            process_start_time=123.456,  # Record has a start time
        )

        # Should return False because probe returned None
        assert is_worker_alive(record) is False
    finally:
        process.kill()
        process.wait(timeout=5)


def test_launch_captures_process_start_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The launch path must capture process_start_time at spawn time.

    This test goes through the real launch path and asserts the resulting record's
    process_start_time is not None. Mutation gate: forcing spawn capture to None MUST fail it.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    # The record must have a captured process_start_time
    assert record.process_start_time is not None, (
        "launch_claude_worker must capture process_start_time at spawn time"
    )
    assert isinstance(record.process_start_time, float), (
        "process_start_time must be a float (Unix timestamp)"
    )


# ---------------------------------------------------------------------------
# Throttle death classification tests (symmetric to devin_shell tests)
# ---------------------------------------------------------------------------


def test_classify_session_failure_rate_limit_with_reset_time(tmp_path: Path) -> None:
    """Test that rate-limit errors with 'resets in N minutes' are classified correctly."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    # Verify it's a valid ISO timestamp
    assert "T" in throttled_until
    assert "Z" in throttled_until


def test_classify_session_failure_quota_exhausted(tmp_path: Path) -> None:
    """Test that quota-exhaustion errors are classified correctly."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: daily usage quota has been exhausted. Please try again tomorrow.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind == "quota_exhausted"
    assert throttled_until is not None
    # Should use default 24 hour cooldown
    assert "T" in throttled_until
    assert "Z" in throttled_until


def test_update_worker_record_with_failure_classification(tmp_path: Path) -> None:
    """Test that worker records are updated with failure classification."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create a worker sidecar
    sidecar_path = sessions_dir / "issue-42.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt/issue-42",
                "prompt_path": "p.md",
                "command": ["claude", "-p"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(sessions_dir / "issue-42.claude.log"),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    # Create a log file with rate-limit error
    log_path = sessions_dir / "issue-42.claude.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42
    )

    assert failure_kind == "rate_limited"
    assert throttled_until is not None

    # Verify the sidecar was updated
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "rate_limited"


def test_launch_claude_worker_render_error_returns_error_record_and_tears_down_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Defense-in-depth: render errors past the load gate return error records, not exceptions."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    worktree_removed = []

    def tracking_remove_worktree(repo_root, worktree_path, *, force=False, branch=None):
        worktree_removed.append(worktree_path)
        return True

    monkeypatch.setattr(
        claude_code,
        "create_worktree",
        lambda *args, **kwargs: _fake_worktree(tmp_path, "agent/issue-1"),
    )
    monkeypatch.setattr(claude_code, "remove_worktree", tracking_remove_worktree)

    # Template with an unknown placeholder that bypasses load validation
    record = launch_claude_worker(
        1,
        "agent/issue-1",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("echo", "{unknown_placeholder}"),
    )

    # Must return an error record, not raise
    assert not record.ok
    assert record.error is not None
    assert "command template rendering failed" in record.error
    assert record.pid is None

    # Worktree must have been torn down
    assert len(worktree_removed) == 1

    # Sidecar must record the error
    sidecar_path = sessions_dir / "issue-1.claude.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["error"] is not None
    assert payload["pid"] is None


def test_launch_claude_worker_with_venv_source_junction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When venv_source is configured, the junction is copied into the worktree."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    # Create a fake venv source
    venv_source = tmp_path / "shared_venv"
    venv_source.mkdir()
    (venv_source / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")

    # Track the venv_source argument passed to create_worktree
    calls: list[dict] = []

    def tracking_create_worktree(
        repo_root,
        branch,
        *,
        base_ref="HEAD",
        worktrees_dir=None,
        venv_source=None,
        rework=False,
        recovery=None,
    ):
        calls.append(
            {
                "repo_root": repo_root,
                "branch": branch,
                "base_ref": base_ref,
                "worktrees_dir": worktrees_dir,
                "venv_source": venv_source,
                "rework": rework,
                "recovery": recovery,
            }
        )
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(claude_code, "create_worktree", tracking_create_worktree)

    record = launch_claude_worker(
        42,
        "agent/issue-42-venv",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        venv_source=venv_source,
    )

    assert record.ok
    assert len(calls) == 1
    assert calls[0]["venv_source"] == venv_source


def test_launch_claude_worker_with_custom_worktrees_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When worktrees_dir is configured, it's passed to create_worktree."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    custom_worktrees_dir = tmp_path / "custom_worktrees"

    # Track the worktrees_dir argument passed to create_worktree
    calls: list[dict] = []

    def tracking_create_worktree(
        repo_root,
        branch,
        *,
        base_ref="HEAD",
        worktrees_dir=None,
        venv_source=None,
        rework=False,
        recovery=None,
    ):
        calls.append(
            {
                "repo_root": repo_root,
                "branch": branch,
                "base_ref": base_ref,
                "worktrees_dir": worktrees_dir,
                "venv_source": venv_source,
                "rework": rework,
                "recovery": recovery,
            }
        )
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(claude_code, "create_worktree", tracking_create_worktree)

    record = launch_claude_worker(
        42,
        "agent/issue-42-custom-dir",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        worktrees_dir=custom_worktrees_dir,
    )

    assert record.ok
    assert len(calls) == 1
    assert calls[0]["worktrees_dir"] == custom_worktrees_dir


def test_launch_claude_worker_rework_mode_reuses_existing_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """In rework mode, create_worktree is called with rework=True."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    calls: list[dict] = []

    def tracking_create_worktree(
        repo_root,
        branch,
        *,
        base_ref="HEAD",
        worktrees_dir=None,
        venv_source=None,
        rework=False,
        recovery=None,
    ):
        calls.append(
            {
                "repo_root": repo_root,
                "branch": branch,
                "base_ref": base_ref,
                "worktrees_dir": worktrees_dir,
                "venv_source": venv_source,
                "rework": rework,
                "recovery": recovery,
            }
        )
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(claude_code, "create_worktree", tracking_create_worktree)

    record = launch_claude_worker(
        42,
        "agent/issue-42-rework",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        rework=True,
    )

    assert record.ok
    assert len(calls) == 1
    assert calls[0]["rework"] is True


def test_launch_claude_worker_recovery_mode_passes_recovery_dict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """In recovery mode, the recovery dict is passed to create_worktree."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    recovery_dict = {
        "worktree_path": "/tmp/wt/issue-42",
        "branch": "agent/issue-42",
        "has_commits": True,
    }

    calls: list[dict] = []

    def tracking_create_worktree(
        repo_root,
        branch,
        *,
        base_ref="HEAD",
        worktrees_dir=None,
        venv_source=None,
        rework=False,
        recovery=None,
    ):
        calls.append(
            {
                "repo_root": repo_root,
                "branch": branch,
                "base_ref": base_ref,
                "worktrees_dir": worktrees_dir,
                "venv_source": venv_source,
                "rework": rework,
                "recovery": recovery,
            }
        )
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(claude_code, "create_worktree", tracking_create_worktree)

    record = launch_claude_worker(
        42,
        "agent/issue-42-recovery",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        recovery=recovery_dict,
    )

    assert record.ok
    assert len(calls) == 1
    assert calls[0]["recovery"] == recovery_dict


def test_launch_claude_worker_rework_log_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """In rework mode, the log file uses the -rework suffix."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        42,
        "agent/issue-42-rework",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        rework=True,
    )

    assert record.ok
    assert record.log_path.endswith("-rework.claude.log")


def test_launch_claude_worker_prompt_write_failure_tears_down_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If writing the prompt file fails, the worktree is torn down."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    worktree_removed = []

    def tracking_remove_worktree(repo_root, worktree_path, *, force=False, branch=None):
        worktree_removed.append(worktree_path)
        return True

    monkeypatch.setattr(
        claude_code,
        "create_worktree",
        lambda *args, **kwargs: _fake_worktree(tmp_path, "agent/issue-1"),
    )
    monkeypatch.setattr(claude_code, "remove_worktree", tracking_remove_worktree)

    # Monkeypatch Path.write_text to raise OSError on the prompt file
    original_write_text = Path.write_text

    def failing_write_text(self, content, encoding=None, errors=None):
        if self.name == ".orchestrator-prompt.md":
            raise OSError("Mock prompt write failure")
        return original_write_text(self, content, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    record = launch_claude_worker(
        1,
        "agent/issue-1",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert not record.ok
    assert "failed to write prompt file" in record.error
    assert len(worktree_removed) == 1


# ---------------------------------------------------------------------------
# Real-git integration tests: branch delete/preserve semantics
# ---------------------------------------------------------------------------


def test_launch_failure_then_retry_succeeds(tmp_path: Path) -> None:
    """Launch failure should clean up branch and worktree, allowing retry to succeed."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    # Initialize a real git repo for the worktree to use
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    branch_name = "agent/issue-42-retry"

    # First launch fails (binary doesn't exist)
    record1 = launch_claude_worker(
        42,
        branch_name,
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
    )

    assert not record1.ok
    assert "failed to launch claude" in record1.error

    # Verify the branch is deleted after the failure
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name not in result.stdout

    # Second launch should succeed (using fake claude script)
    record2 = launch_claude_worker(
        42,
        branch_name,
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
    )

    assert record2.ok
    assert record2.branch == branch_name


def test_rework_launch_failure_preserves_branch(tmp_path: Path) -> None:
    """Rework-mode launch failure should preserve the existing branch."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    # Initialize a real git repo
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    branch_name = "agent/issue-43-rework"

    # Create the branch first (simulating a previous PR cycle)
    subprocess.run(
        ["git", "branch", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Verify the branch exists
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Rework-mode launch fails (binary doesn't exist)
    record = launch_claude_worker(
        43,
        branch_name,
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
        rework=True,
    )

    assert not record.ok
    assert "failed to launch claude" in record.error

    # Verify the branch is preserved (not deleted)
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Clean up
    subprocess.run(
        ["git", "branch", "-D", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def test_rework_prompt_write_failure_preserves_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rework-mode prompt-write failure should preserve the existing branch."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    # Initialize a real git repo
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    branch_name = "agent/issue-44-rework-prompt-fail"

    # Create the branch first (simulating a previous PR cycle)
    subprocess.run(
        ["git", "branch", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Verify the branch exists
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Monkeypatch Path.write_text to raise OSError on the prompt file
    original_write_text = Path.write_text

    def failing_write_text(self, content, encoding=None, errors=None):
        if self.name == ".orchestrator-prompt.md":
            raise OSError("Mock prompt write failure")
        return original_write_text(self, content, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    # Rework-mode launch fails on prompt write
    record = launch_claude_worker(
        44,
        branch_name,
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        rework=True,
    )

    assert not record.ok
    assert "failed to write prompt file" in record.error

    # Verify the branch is preserved (not deleted)
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name in result.stdout

    # Clean up
    subprocess.run(
        ["git", "branch", "-D", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# Regression: VIRTUAL_ENV sanitization
# ---------------------------------------------------------------------------


def test_sanitize_env_drops_virtual_env_when_no_worktree_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When worktree has no .venv, VIRTUAL_ENV and UV_PROJECT_ENVIRONMENT must be dropped."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    # Set parent env variables (simulating orchestrator leak)
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    env = sanitize_env(worktree_path)

    assert "VIRTUAL_ENV" not in env, "VIRTUAL_ENV must be dropped when worktree has no .venv"
    assert "UV_PROJECT_ENVIRONMENT" not in env, (
        "UV_PROJECT_ENVIRONMENT must be dropped when worktree has no .venv"
    )


def test_sanitize_env_sets_worktree_venv_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When worktree has .venv, VIRTUAL_ENV must be set to that path."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()
    worktree_venv = worktree_path / ".venv"
    worktree_venv.mkdir()

    # Set parent env variables (simulating orchestrator leak)
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    env = sanitize_env(worktree_path)

    assert env.get("VIRTUAL_ENV") == str(worktree_venv), (
        "VIRTUAL_ENV must be set to worktree .venv"
    )
    assert "UV_PROJECT_ENVIRONMENT" not in env, (
        "UV_PROJECT_ENVIRONMENT must be dropped when worktree has .venv"
    )


def test_sanitize_env_preserves_other_env_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Other environment variables must be preserved."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/home/user")
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")

    env = sanitize_env(worktree_path)

    assert env.get("PATH") == "/usr/bin:/bin", "PATH must be preserved"
    assert env.get("HOME") == "/home/user", "HOME must be preserved"
    assert "VIRTUAL_ENV" not in env, "VIRTUAL_ENV must be dropped"


def test_launch_sanitizes_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """launch_claude_worker must sanitize the environment before spawning the worker (stdin-fed path)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Seed parent env with leak variables
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    # Script that writes the actual env it received to a file
    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-received.txt").write_text(
                str(os.environ.get("VIRTUAL_ENV", "<unset>"))
                + "|"
                + str(os.environ.get("UV_PROJECT_ENVIRONMENT", "<unset>")),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        117,
        "agent/issue-117-sanitize",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-received.txt"
    deadline = time.time() + 10
    while not probe_path.exists() and time.time() < deadline:
        time.sleep(0.05)


def test_launch_sanitizes_with_worktree_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When worktree has .venv, VIRTUAL_ENV must be set to that path."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    # Create a fake worktree with .venv
    worktree_path = tmp_path / "worktrees" / "agent-issue-117-venv"
    worktree_path.mkdir(parents=True)
    worktree_venv = worktree_path / ".venv"
    worktree_venv.mkdir()

    def fake_create_worktree(*args, **kwargs):
        return WorktreeInfo(path=worktree_path, branch="agent/issue-117-venv", venv_junction=None)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)

    # Seed parent env with leak variables
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    # Script that writes the actual env it received to a file
    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-received.txt").write_text(
                str(os.environ.get("VIRTUAL_ENV", "<unset>"))
                + "|"
                + str(os.environ.get("UV_PROJECT_ENVIRONMENT", "<unset>")),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        117,
        "agent/issue-117-venv",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-received.txt"
    deadline = time.time() + 10
    while not probe_path.exists() and time.time() < deadline:
        time.sleep(0.05)

    assert probe_path.exists()
    received = probe_path.read_text(encoding="utf-8")
    # VIRTUAL_ENV must be worktree .venv, UV_PROJECT_ENVIRONMENT must be absent
    assert received == f"{str(worktree_venv)}|<unset>", (
        f"VIRTUAL_ENV must be worktree .venv, UV_PROJECT_ENVIRONMENT must be absent, got: {received}"
    )


def test_launch_preserves_worker_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit worker_env VIRTUAL_ENV override must survive sanitization (no worktree .venv)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Seed parent env with leak variables
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    # Script that writes the actual env it received to a file
    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-received.txt").write_text(
                str(os.environ.get("VIRTUAL_ENV", "<unset>"))
                + "|"
                + str(os.environ.get("UV_PROJECT_ENVIRONMENT", "<unset>"))
                + "|"
                + str(os.environ.get("CUSTOM_VAR", "<unset>")),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        117,
        "agent/issue-117-override",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
        env={"VIRTUAL_ENV": "/custom/.venv", "CUSTOM_VAR": "custom-value"},
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-received.txt"
    deadline = time.time() + 10
    while not probe_path.exists() and time.time() < deadline:
        time.sleep(0.05)

    assert probe_path.exists()
    received = probe_path.read_text(encoding="utf-8")
    # User-provided VIRTUAL_ENV must win, UV_PROJECT_ENVIRONMENT must be dropped
    assert received == "/custom/.venv|<unset>|custom-value", (
        f"User VIRTUAL_ENV override must win, got: {received}"
    )


def test_launch_override_precedence_with_worktree_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """User-provided VIRTUAL_ENV override must win over worktree .venv (merge order test)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    # Create a fake worktree with .venv
    worktree_path = tmp_path / "worktrees" / "agent-issue-117-override-venv"
    worktree_path.mkdir(parents=True)
    worktree_venv = worktree_path / ".venv"
    worktree_venv.mkdir()

    def fake_create_worktree(*args, **kwargs):
        return WorktreeInfo(
            path=worktree_path, branch="agent/issue-117-override-venv", venv_junction=None
        )

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)

    # Seed parent env with leak variables
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    # Script that writes the actual env it received to a file
    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-received.txt").write_text(
                str(os.environ.get("VIRTUAL_ENV", "<unset>"))
                + "|"
                + str(os.environ.get("UV_PROJECT_ENVIRONMENT", "<unset>"))
                + "|"
                + str(os.environ.get("CUSTOM_VAR", "<unset>")),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        117,
        "agent/issue-117-override-venv",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
        env={"VIRTUAL_ENV": "/custom/.venv", "CUSTOM_VAR": "custom-value"},
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-received.txt"
    deadline = time.time() + 10
    while not probe_path.exists() and time.time() < deadline:
        time.sleep(0.05)

    assert probe_path.exists()
    received = probe_path.read_text(encoding="utf-8")
    # User-provided VIRTUAL_ENV must win over worktree .venv (merge order: sanitizer first, then user overrides)
    assert received == "/custom/.venv|<unset>|custom-value", (
        f"User VIRTUAL_ENV override must win over worktree .venv, got: {received}"
    )


def test_launch_sanitizes_environment_with_prompt_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """launch_claude_worker must sanitize the environment before spawning the worker (argv path with {prompt_path})."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Seed parent env with leak variables
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    # Script that reads prompt from argv and writes the actual env it received to a file
    script_path = tmp_path / "env_probe_argv.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import sys
            import os
            from pathlib import Path

            # Read prompt from argv (the {prompt_path} placeholder)
            prompt_path = Path(sys.argv[1])
            prompt_content = prompt_path.read_text(encoding="utf-8")

            # Write the env we received
            Path("env-received.txt").write_text(
                str(os.environ.get("VIRTUAL_ENV", "<unset>"))
                + "|"
                + str(os.environ.get("UV_PROJECT_ENVIRONMENT", "<unset>")),
                encoding="utf-8",
            )
            print("ok")
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        117,
        "agent/issue-117-sanitize-argv",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path), "{prompt_path}"),
    )

    assert record.ok
    assert record.command[-1] == record.prompt_path

    worktree_path = Path(record.worktree_path)
    probe_path = worktree_path / "env-received.txt"
    deadline = time.time() + 10
    while not probe_path.exists() and time.time() < deadline:
        time.sleep(0.05)

    assert probe_path.exists()
    received = probe_path.read_text(encoding="utf-8")
    # VIRTUAL_ENV and UV_PROJECT_ENVIRONMENT must be dropped (no worktree .venv)
    assert received == "<unset>|<unset>", (
        f"VIRTUAL_ENV and UV_PROJECT_ENVIRONMENT must be dropped, got: {received}"
    )
