from __future__ import annotations

import json
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
