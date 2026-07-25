from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from charlie_work import claude_code
from charlie_work.config import (
    ClaudeCodeConfig,
    OrchestratorConfig,
    ReviewDispatchConfig,
    RuntimeConfig,
)
from charlie_work.claude_code import (
    ClaudeProgress,
    ClaudeWorkerRecord,
    is_worker_alive,
    launch_claude_worker,
    parse_claude_events,
    probe_claude,
    read_worker_records,
    resolve_review_effort,
    update_worker_record_with_failure_classification,
    _apply_model_pin,
    _apply_effort_pin,
    _review_effort_arm,
    _sanitize_review_command_template,
    _sidecar_path,
)
from charlie_work.env_sanitize import sanitize_env
from charlie_work.subprocess_runner import RunResult
from charlie_work.worktree import WorktreeInfo


def _fake_worktree(tmp_path: Path, branch: str) -> WorktreeInfo:
    worktree_path = tmp_path / "worktrees" / branch.replace("/", "-")
    worktree_path.mkdir(parents=True, exist_ok=True)
    return WorktreeInfo(path=worktree_path, branch=branch, venv_junction=None)


def _fake_worktree_with_venv(tmp_path: Path, branch: str) -> WorktreeInfo:
    """Create a fake worktree with a .venv directory.

    This makes sanitize_env actively SET VIRTUAL_ENV (instead of POP-ing it),
    which makes the merge order testable: if worker_env is merged first,
    sanitize_env will clobber the override.
    """
    worktree_path = tmp_path / "worktrees" / branch.replace("/", "-")
    worktree_path.mkdir(parents=True, exist_ok=True)
    (worktree_path / ".venv").mkdir()
    return WorktreeInfo(path=worktree_path, branch=branch, venv_junction=None)


def _install_fake_create_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    calls: list[dict] | None = None,
    with_venv: bool = False,
) -> None:
    def fake_create_worktree(
        repo_root,
        branch,
        *,
        base_ref="HEAD",
        worktrees_dir=None,
        venv_source=None,
        materialize_dirs=(),
        rework=False,
        recovery=None,
        issue_number=None,
        config=None,
        sessions_dir=None,
    ):
        if calls is not None:
            calls.append(
                {
                    "repo_root": repo_root,
                    "branch": branch,
                    "base_ref": base_ref,
                    "worktrees_dir": worktrees_dir,
                    "venv_source": venv_source,
                    "materialize_dirs": materialize_dirs,
                    "rework": rework,
                    "recovery": recovery,
                    "issue_number": issue_number,
                    "config": config,
                    "sessions_dir": sessions_dir,
                }
            )
        if with_venv:
            return _fake_worktree_with_venv(tmp_path, branch)
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


def test_launch_claude_worker_worker_env_overrides_sanitize_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """worker_env overrides sanitize_env: operator-provided VIRTUAL_ENV wins.

    This is a mutation gate for the merge order in launch_claude_worker:
    the current order is {**sanitize_env(...), **worker_env}, so worker_env
    clobbers sanitized keys. If the order is inverted (worker_env first,
    sanitize_env clobbering it), this test fails.

    The fixture uses with_venv=True so sanitize_env actively SETS VIRTUAL_ENV
    (instead of POP-ing it), making the merge order sensitive.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path, with_venv=True)

    # Set a VIRTUAL_ENV in the orchestrator's environment (which sanitize_env
    # would normally strip). Then provide an explicit override via worker_env.
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/venv")

    script_path = tmp_path / "env_probe.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import os
            from pathlib import Path

            Path("env-probe.txt").write_text(
                os.environ.get("VIRTUAL_ENV", "<unset>"),
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        140,
        "agent/issue-140-env-override",
        "prompt",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
        env={"VIRTUAL_ENV": "/custom/override/venv"},
    )

    assert record.ok
    probe_path = Path(record.worktree_path) / "env-probe.txt"
    deadline = time.time() + 10
    while not probe_path.exists() and time.time() < deadline:
        time.sleep(0.05)

    assert probe_path.exists()
    # worker_env VIRTUAL_ENV override wins over sanitize_env's stripping
    assert probe_path.read_text(encoding="utf-8") == "/custom/override/venv"


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
    assert record.prompt_path in record.command

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


def test_launch_claude_worker_resolves_argv0_through_resolve_cli_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #487: a bare ``"claude"`` on Windows is an npm ``.CMD`` shim that
    ``Popen(shell=False)`` cannot find (WinError 2). ``launch_claude_worker``
    must resolve argv[0] through ``resolve_cli_binary`` before spawning —
    exercised here by stubbing the resolver to swap in a real Python script
    standing in for the resolved ``.exe``, and asserting the recorded
    ``command`` reflects the resolved binary, not the original template
    token."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    fake_script = _fake_claude_script(tmp_path)
    resolved_calls: list[str] = []

    def fake_resolve_cli_binary(name: str) -> str:
        resolved_calls.append(name)
        assert name == "claude"
        return fake_script[0]

    monkeypatch.setattr(claude_code, "resolve_cli_binary", fake_resolve_cli_binary)

    record = launch_claude_worker(
        55,
        "agent/issue-55-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("claude", str(fake_script[1])),
    )

    assert record.ok
    assert resolved_calls == ["claude"]
    assert record.command[0] == fake_script[0]
    assert record.command[0] != "claude"


def test_probe_claude_resolves_argv0_through_resolve_cli_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ``doctor --adapter-probe`` path (``probe_claude``) must resolve
    the same way ``launch_claude_worker`` does, or `charlie doctor` would
    report a healthy `claude` install failing to probe with WinError 2 on a
    machine using the claude-code adapter (issue #487)."""
    captured: dict[str, Any] = {}

    def fake_resolve_cli_binary(name: str) -> str:
        assert name == "claude"
        return "C:\\resolved\\claude.exe"

    def fake_run_captured(command, *, cwd, timeout_seconds):
        captured["command"] = command
        return RunResult(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(claude_code, "resolve_cli_binary", fake_resolve_cli_binary)
    monkeypatch.setattr(claude_code, "run_captured", fake_run_captured)

    result = probe_claude(tmp_path)

    assert result.ok
    assert captured["command"] == ["C:\\resolved\\claude.exe", "--version"]


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
        materialize_dirs=(),
        rework=False,
        recovery=None,
        issue_number=None,
        config=None,
        sessions_dir=None,
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


def test_launch_claude_worker_fetch_failure_yields_error_record_not_exception(
    tmp_path: Path,
) -> None:
    """End-to-end test: real fetch failure inside create_worktree must return error record.

    This test forces a real git fetch failure by creating a repo with a broken origin URL,
    then calling launch_claude_worker with base_ref that triggers a fetch. The adapter must
    catch the RuntimeError from create_worktree and return an error record, not raise.

    This is a mutation gate: if RuntimeError is removed from the except tuple in
    launch_claude_worker, this test will fail with an uncaught exception.
    """
    # Create a real git repo with an origin remote
    remote_repo = tmp_path / "remote"
    remote_repo.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    (remote_repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=remote_repo,
        check=True,
        capture_output=True,
    )

    # Clone the remote repo to create a local repo with origin configured
    repo_root = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", str(remote_repo), str(repo_root)],
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

    sessions_dir = tmp_path / "sessions"

    # Break the origin remote to simulate a fetch failure
    subprocess.run(
        ["git", "remote", "set-url", "origin", "file:///nonexistent/path"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Call launch_claude_worker with base_ref that triggers a fetch
    # Empty string resolves to origin/main, which will trigger the fetch in create_worktree
    record = launch_claude_worker(
        142,
        "agent/issue-142-fetch-failure",
        "do the thing",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        base_ref="",  # Triggers fetch of origin/main
    )

    # The adapter must catch the RuntimeError and return an error record
    assert not record.ok
    assert record.error is not None
    assert "worktree creation failed" in record.error
    assert record.worktree_path == ""
    assert record.pid is None

    # Verify the sidecar was written with the error
    sidecar_path = sessions_dir / "issue-142.claude.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["error"] is not None
    assert "worktree creation failed" in payload["error"]


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


def test_probe_claude_missing_binary_never_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Force resolve_cli_binary's lookup to fail regardless of whether this
    # test machine happens to have a real `claude` on PATH (as this repo's
    # own dev box does) -- the point of this test is the genuinely-missing
    # case, not this machine's install state.
    monkeypatch.setattr(
        claude_code,
        "resolve_cli_binary",
        lambda name: "this-binary-does-not-exist-xyz",
    )

    result = probe_claude(tmp_path)

    assert result.ok is False
    assert result.error is not None


def test_probe_claude_uses_run_captured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple] = []

    def fake_run_captured(command, *, cwd, timeout_seconds, shell=False):
        calls.append((command, cwd, timeout_seconds))
        from charlie_work.subprocess_runner import RunResult

        return RunResult(returncode=0, stdout="1.0.0", stderr="")

    # Identity resolution: this test is about the run_captured plumbing, not
    # binary resolution (covered separately by
    # test_probe_claude_resolves_argv0_through_resolve_cli_binary).
    monkeypatch.setattr(claude_code, "resolve_cli_binary", lambda name: name)
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


def test_is_worker_alive_probe_none_treats_indeterminate_as_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #360 criterion #1: a start-time probe failure is not a definitive dead signal.

    When ``get_process_start_time`` returns ``None`` for a live PID, ``is_worker_alive``
    treats the liveness signal as indeterminate and returns ``True`` rather than
    reaping a potentially-live worker.
    """
    import charlie_work.process_utils as process_utils

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
        # Simulate a start-time probe failure while the process is still alive.
        monkeypatch.setattr(process_utils, "get_process_start_time", lambda pid: None)

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

        # Should return True because the probe was indeterminate, not definitive dead.
        assert is_worker_alive(record) is True
    finally:
        process.kill()
        process.wait(timeout=5)


def test_sidecar_path_returns_correct_path(tmp_path: Path) -> None:
    """_sidecar_path returns the expected path for a given issue number."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    path = _sidecar_path(sessions_dir, 123)
    assert path == sessions_dir / "issue-123.claude.json"


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


def test_classify_session_failure_includes_resume_margin(tmp_path: Path) -> None:
    """Issue #499: killed-worker rate-limit classification must include the resume margin."""
    from datetime import UTC, datetime, timedelta

    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Your limit will reset in 3 minutes.\n",
        encoding="utf-8",
    )

    now = datetime.now(UTC)
    failure_kind, throttled_until = _classify_session_failure(log_path, resume_margin_seconds=90)

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    parsed = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    expected = now + timedelta(minutes=3, seconds=90)
    assert abs((parsed - expected).total_seconds()) < 1


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


def test_update_worker_record_with_failure_classification_includes_resume_margin(
    tmp_path: Path,
) -> None:
    """Issue #499: update wrapper applies config.runtime.throttle_resume_margin_s."""
    from datetime import UTC, datetime, timedelta

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

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

    log_path = sessions_dir / "issue-42.claude.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Your limit will reset in 5 minutes.\n",
        encoding="utf-8",
    )

    config = OrchestratorConfig(runtime=RuntimeConfig(throttle_resume_margin_s=90))
    failure_kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42, config=config
    )

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    now = datetime.now(UTC)
    parsed = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    expected = now + timedelta(minutes=5, seconds=90)
    assert abs((parsed - expected).total_seconds()) < 1


def _make_worker_sidecar(sessions_dir: Path, issue_number: int, log_path: Path) -> Path:
    """Write a minimal worker sidecar for failure-classification tests."""
    sidecar_path = sessions_dir / f"issue-{issue_number}.claude.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": issue_number,
                "branch": f"agent/issue-{issue_number}",
                "worktree_path": "/tmp/wt",
                "prompt_path": "p.md",
                "command": ["claude", "-p"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(log_path),
                "error": None,
            }
        ),
        encoding="utf-8",
    )
    return sidecar_path


def test_classify_session_failure_tool_rejected_is_not_throttle(tmp_path: Path) -> None:
    """Issue #260, corrected premise: 'A tool was rejected by the user' is the
    Devin CLI's own surfacing of a PreToolUse hook block, not a provider
    throttle condition — it must NOT classify as rate_limited (no retry
    semantics, no throttled_until). See test_devin_shell.py's mirror test
    and test_post_mortem.py for the worker_blocked log-tail fallback that
    now owns this signature instead."""
    from charlie_work.claude_code import _classify_session_failure

    log_path = tmp_path / "session.claude.log"
    log_path.write_text(
        "Error: A tool was rejected by the user.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind is None
    assert throttled_until is None


def test_update_worker_record_tool_rejected_is_not_rate_limited(tmp_path: Path) -> None:
    """Issue #260, corrected premise: a tool-rejected sidecar log must not be
    classified rate_limited by the adapter's own log-tail classifier."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    log_path = sessions_dir / "issue-42.claude.log"
    log_path.write_text(
        "Error: A tool was rejected by the user.\n",
        encoding="utf-8",
    )
    sidecar_path = _make_worker_sidecar(sessions_dir, 42, log_path)

    failure_kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42, fallback_kind="stalled"
    )

    assert failure_kind == "stalled"
    assert throttled_until is None
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "stalled"


def test_update_worker_record_unknown_tail_falls_back_to_stalled(tmp_path: Path) -> None:
    """Unknown log tail should fall back to the provided fallback_kind."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    log_path = sessions_dir / "issue-42.claude.log"
    log_path.write_text(
        "Error: something completely unrelated went wrong\n",
        encoding="utf-8",
    )
    sidecar_path = _make_worker_sidecar(sessions_dir, 42, log_path)

    failure_kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42, fallback_kind="stalled"
    )

    assert failure_kind == "stalled"
    assert throttled_until is None
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "stalled"


def test_update_worker_record_custom_throttle_markers(tmp_path: Path) -> None:
    """RuntimeConfig.throttle_error_markers is configurable without code changes."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    log_path = sessions_dir / "issue-42.claude.log"
    log_path.write_text(
        "Error: provider-specific frobnicate limit exceeded\n",
        encoding="utf-8",
    )
    sidecar_path = _make_worker_sidecar(sessions_dir, 42, log_path)

    config = OrchestratorConfig(
        runtime=RuntimeConfig(throttle_error_markers=("frobnicate limit exceeded",))
    )
    failure_kind, throttled_until = update_worker_record_with_failure_classification(
        sessions_dir, 42, config=config
    )

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
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
        materialize_dirs=(),
        rework=False,
        recovery=None,
        issue_number=None,
        config=None,
        sessions_dir=None,
    ):
        calls.append(
            {
                "repo_root": repo_root,
                "branch": branch,
                "base_ref": base_ref,
                "worktrees_dir": worktrees_dir,
                "venv_source": venv_source,
                "materialize_dirs": materialize_dirs,
                "rework": rework,
                "recovery": recovery,
                "issue_number": issue_number,
                "config": config,
                "sessions_dir": sessions_dir,
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
        materialize_dirs=(),
        rework=False,
        recovery=None,
        issue_number=None,
        config=None,
        sessions_dir=None,
    ):
        calls.append(
            {
                "repo_root": repo_root,
                "branch": branch,
                "base_ref": base_ref,
                "worktrees_dir": worktrees_dir,
                "venv_source": venv_source,
                "materialize_dirs": materialize_dirs,
                "rework": rework,
                "recovery": recovery,
                "issue_number": issue_number,
                "config": config,
                "sessions_dir": sessions_dir,
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
        materialize_dirs=(),
        rework=False,
        recovery=None,
        issue_number=None,
        config=None,
        sessions_dir=None,
    ):
        calls.append(
            {
                "repo_root": repo_root,
                "branch": branch,
                "base_ref": base_ref,
                "worktrees_dir": worktrees_dir,
                "venv_source": venv_source,
                "materialize_dirs": materialize_dirs,
                "rework": rework,
                "recovery": recovery,
                "issue_number": issue_number,
                "config": config,
                "sessions_dir": sessions_dir,
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
        materialize_dirs=(),
        rework=False,
        recovery=None,
        issue_number=None,
        config=None,
        sessions_dir=None,
    ):
        calls.append(
            {
                "repo_root": repo_root,
                "branch": branch,
                "base_ref": base_ref,
                "worktrees_dir": worktrees_dir,
                "venv_source": venv_source,
                "materialize_dirs": materialize_dirs,
                "rework": rework,
                "recovery": recovery,
                "issue_number": issue_number,
                "config": config,
                "sessions_dir": sessions_dir,
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
    assert calls[0]["recovery"] == {
        **recovery_dict,
        "inconclusive_probe_deferred_count": 0,
    }


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
    assert record.prompt_path in record.command

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


def test_launch_claude_worker_includes_start_new_session_on_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """launch_claude_worker should include start_new_session=True on POSIX systems."""
    from unittest.mock import patch

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Capture the kwargs passed to subprocess.Popen
    popen_kwargs: dict = {}
    original_popen = subprocess.Popen

    def capture_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        # Return a fake process that exits immediately
        return original_popen([sys.executable, "-c", "pass"], **kwargs)

    with patch("subprocess.Popen", side_effect=capture_popen):
        launch_claude_worker(
            999,
            "agent/issue-999-start-new-session",
            "prompt",
            repo_root=repo_root,
            sessions_dir=sessions_dir,
            command_template=(sys.executable, "-c", "pass"),
        )

    # Worker spawns use hidden_console_kwargs: CREATE_NEW_CONSOLE plus a
    # STARTUPINFO with wShowWindow=SW_HIDE so descendants inherit a hidden
    # console. Policy A survival flags (DETACHED_PROCESS, CREATE_BREAKAWAY_FROM_JOB)
    # are out of scope for issue #360.
    if os.name != "nt":
        assert popen_kwargs.get("start_new_session") is True
        assert "creationflags" not in popen_kwargs
        assert "startupinfo" not in popen_kwargs
    else:
        assert popen_kwargs.get("start_new_session") is False
        flags = popen_kwargs.get("creationflags", 0)
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
        assert flags & subprocess.CREATE_NEW_CONSOLE
        assert not (flags & subprocess.CREATE_NO_WINDOW)
        assert not (flags & subprocess.DETACHED_PROCESS)
        assert not (flags & subprocess.CREATE_BREAKAWAY_FROM_JOB)
        startupinfo = popen_kwargs.get("startupinfo")
        assert startupinfo is not None
        assert startupinfo.wShowWindow == subprocess.SW_HIDE
        assert startupinfo.dwFlags & subprocess.STARTF_USESHOWWINDOW


def test_launch_claude_worker_routes_creationflags_through_hidden_console_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """launch_claude_worker must obtain its Popen kwargs via
    ``hidden_console_kwargs`` (issue #459) so worker descendants inherit a
    hidden console instead of each allocating their own visible window.

    Note: patching ``subprocess.Popen`` globally also intercepts the
    internal ``Popen`` calls that ``subprocess.run`` makes under the hood
    (e.g. from any incidental git cleanup on the error path), so we record
    kwargs *per call* and assert on the first one -- the actual worker
    launch -- rather than a merged/overwritten dict.
    """
    from unittest.mock import MagicMock, patch

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    popen_calls: list[dict] = []

    def capture_popen(*args, **kwargs):
        popen_calls.append(kwargs)
        return MagicMock(pid=12345)

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    sentinel_kwargs = {
        "creationflags": subprocess.CREATE_NEW_CONSOLE,
        "startupinfo": startupinfo,
    }
    with (
        patch("subprocess.Popen", side_effect=capture_popen),
        patch(
            "charlie_work.process_utils.hidden_console_kwargs",
            return_value=sentinel_kwargs,
        ) as mock_helper,
    ):
        launch_claude_worker(
            998,
            "agent/issue-998-hidden-console",
            "prompt",
            repo_root=repo_root,
            sessions_dir=sessions_dir,
            command_template=(sys.executable, "-c", "pass"),
        )

    expected_group_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    mock_helper.assert_called_once_with(expected_group_flag)
    assert popen_calls, "expected at least one Popen call from the worker launch"
    assert popen_calls[0].get("creationflags") == subprocess.CREATE_NEW_CONSOLE
    assert popen_calls[0].get("startupinfo") is startupinfo


# ---------------------------------------------------------------------------
# Tests for parse_claude_events (issue #160)
# ---------------------------------------------------------------------------


def test_parse_claude_events_file_not_exists(tmp_path: Path) -> None:
    """parse_claude_events returns None when the events file doesn't exist."""
    events_path = tmp_path / "issue-1.events.jsonl"
    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is None


def test_parse_claude_events_empty_file(tmp_path: Path) -> None:
    """parse_claude_events returns None when the events file is empty."""
    events_path = tmp_path / "issue-1.events.jsonl"
    events_path.write_text("", encoding="utf-8")
    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is None


def test_parse_claude_events_wellformed(tmp_path: Path) -> None:
    """parse_claude_events correctly accumulates counts and usage from well-formed JSONL."""
    events_path = tmp_path / "issue-1.events.jsonl"
    events = [
        '{"type": "user_message"}',
        '{"type": "assistant_message"}',
        '{"type": "tool_call"}',
        '{"type": "tool_call"}',
        '{"type": "assistant_message", "tokens": 1000, "cost_usd": 0.01}',
        '{"type": "tool_call"}',
        '{"type": "assistant_message", "tokens": 1500, "cost_usd": 0.015}',
    ]
    events_path.write_text("\n".join(events), encoding="utf-8")

    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is not None
    assert result.tool_call_count == 3
    assert result.turn_count == 4  # 4 user/assistant messages total
    assert result.tokens == 1500  # Last-seen value
    assert result.cost_usd == 0.015  # Last-seen value


def test_parse_claude_events_truncated_final_line(tmp_path: Path) -> None:
    """parse_claude_events tolerates a truncated final line (live-appending file)."""
    events_path = tmp_path / "issue-1.events.jsonl"
    events = [
        '{"type": "user_message"}',
        '{"type": "tool_call"}',
        '{"type": "assistant_message", "tokens": 500',  # Truncated JSON
    ]
    events_path.write_text("\n".join(events), encoding="utf-8")

    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is not None
    assert result.tool_call_count == 1
    assert result.turn_count == 1
    assert result.tokens is None  # Truncated line is skipped, so no tokens field


def test_parse_claude_events_malformed_lines_skipped(tmp_path: Path) -> None:
    """parse_claude_events skips malformed JSON lines without raising."""
    events_path = tmp_path / "issue-1.events.jsonl"
    events = [
        '{"type": "user_message"}',
        "not valid json",
        '{"type": "tool_call"}',
        '{"type": "assistant_message"}',
        "also not json",
        '{"type": "tool_call"}',
    ]
    events_path.write_text("\n".join(events), encoding="utf-8")

    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is not None
    assert result.tool_call_count == 2
    assert result.turn_count == 2


def test_parse_claude_events_only_malformed_returns_none(tmp_path: Path) -> None:
    """parse_claude_events returns None when the file contains only malformed JSON."""
    events_path = tmp_path / "issue-1.events.jsonl"
    events_path.write_text("not json\nalso not json", encoding="utf-8")

    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is None


def test_parse_claude_events_non_dict_events_skipped(tmp_path: Path) -> None:
    """parse_claude_events skips non-dict JSON values (arrays, strings, etc)."""
    events_path = tmp_path / "issue-1.events.jsonl"
    events = [
        '{"type": "user_message"}',
        '["array", "value"]',
        '{"type": "tool_call"}',
        '"string value"',
        '{"type": "assistant_message"}',
    ]
    events_path.write_text("\n".join(events), encoding="utf-8")

    result: ClaudeProgress | None = parse_claude_events(events_path)
    assert result is not None
    assert result.tool_call_count == 1
    assert result.turn_count == 2


# Critical regression test for issue #160: tee thread file handle closure bug
# ---------------------------------------------------------------------------
def test_launch_claude_worker_tee_stream_json_writes_to_both_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test for issue #160: tee thread must write to both log and events files.

    This test verifies that when tee_stream_json=True, the background tee thread
    successfully writes output to both the plaintext log file and the events.jsonl file.
    The original bug closed file handles immediately after starting the thread,
    causing all writes to fail silently and leaving both files empty.

    This test would fail against the buggy code (empty files) and pass after the fix.
    """
    from charlie_work.process_utils import is_session_stalled
    from charlie_work.claude_code import _classify_session_failure

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Create a fake claude script that outputs multiple lines
    script_path = tmp_path / "fake_claude_tee.py"
    script_path.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path

            data = sys.stdin.read()
            Path("worker-ran.txt").write_text(data, encoding="utf-8")
            # Output multiple lines to verify tee writes all of them
            print("line 1")
            print("line 2")
            print("line 3")
            """
        ),
        encoding="utf-8",
    )

    record = launch_claude_worker(
        160,
        "agent/issue-160-tee-test",
        "test prompt for tee",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script_path)),
        tee_stream_json=True,  # Enable the tee feature
    )

    assert record.ok
    worktree_path = Path(record.worktree_path)

    # Wait for the worker to complete
    marker_path = worktree_path / "worker-ran.txt"
    deadline = time.time() + 10
    while not marker_path.exists() and time.time() < deadline:
        time.sleep(0.05)

    assert marker_path.exists()

    # Give the tee thread a moment to finish writing
    time.sleep(0.2)

    # Verify both files exist and have content
    log_path = Path(record.log_path)
    events_path = sessions_dir / "issue-160.events.jsonl"

    assert log_path.exists(), "Log file should exist"
    assert events_path.exists(), "Events file should exist"

    log_content = log_path.read_text(encoding="utf-8")
    events_content = events_path.read_text(encoding="utf-8")

    # Both files should have content (the bug would leave them empty)
    assert len(log_content) > 0, "Log file should have content (bug would leave it empty)"
    assert len(events_content) > 0, "Events file should have content (bug would leave it empty)"

    # Verify the content matches what we expect
    assert "line 1" in log_content
    assert "line 2" in log_content
    assert "line 3" in log_content

    # Events file should have the same content (it's a tee)
    assert "line 1" in events_content
    assert "line 2" in events_content
    assert "line 3" in events_content

    # Verify is_session_stalled works correctly on the tee'd log
    # The log should not be stalled (it was just written)
    is_stalled, last_line = is_session_stalled(log_path, stall_threshold_minutes=20)
    assert is_stalled is False, "Freshly written log should not be stalled"
    assert last_line is not None, "Should be able to read last line from log"

    # Verify _classify_session_failure works correctly on the tee'd log
    # Should return None (no failure) for a successful run
    failure_kind, throttled_until = _classify_session_failure(log_path)
    assert failure_kind is None, "Successful run should not be classified as a failure"
    assert throttled_until is None


def test_launch_claude_worker_tee_stream_json_popen_failure_closes_handles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test: in the tee_stream_json branch, log_handle/events_handle
    are opened without a `with` block so the background tee thread can own their
    lifecycle (closing them itself once the process exits). But if
    subprocess.Popen raises before the thread ever starts, nobody closes them.
    Popen failure must close both handles before the OSError propagates.
    """
    from unittest.mock import patch

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    opened_handles: list = []
    original_open = Path.open

    def tracking_open(self, *args, **kwargs):
        handle = original_open(self, *args, **kwargs)
        if self.name.endswith((".claude.log", ".events.jsonl")):
            opened_handles.append(handle)
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)

    with patch("subprocess.Popen", side_effect=OSError("mock spawn failure")):
        record = launch_claude_worker(
            161,
            "agent/issue-161-tee-popen-failure",
            "prompt text",
            repo_root=repo_root,
            sessions_dir=sessions_dir,
            command_template=("claude",),
            tee_stream_json=True,
        )

    assert not record.ok
    assert record.error is not None
    assert "failed to launch claude" in record.error

    assert len(opened_handles) == 2, "expected exactly log_handle and events_handle to be opened"
    assert all(handle.closed for handle in opened_handles), (
        "log_handle/events_handle must be closed when Popen fails, not leaked"
    )


# ---------------------------------------------------------------------------
# Review-dispatch isolation (issue #370/#397): launch_claude_worker(review=True)
# must route through create_review_checkout (a PR-keyed, detached-HEAD
# checkout), never create_worktree (the worker's branch-slug worktree). These
# tests exercise the REAL worktree.create_review_checkout/create_worktree
# code, not a monkeypatched stand-in, so a regression that re-routes review
# through the shared worker worktree would fail them.
# ---------------------------------------------------------------------------


def _init_real_repo(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
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
    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"], cwd=repo_root, check=True, capture_output=True
    )


def _repo_head_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_launch_claude_worker_review_uses_isolated_checkout_not_worker_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A review=True launch must never call create_worktree — only
    create_review_checkout — and must land in a directory distinct from a
    live worker's worktree for the very same branch.
    """
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    branch = "agent/issue-500-fix"
    head_sha = _repo_head_sha(repo_root)

    # Simulate a live worker worktree for this branch, using the real
    # (non-monkeypatched) create_worktree, sitting under a separate dir.
    from charlie_work.worktree import create_worktree

    worker_info = create_worktree(
        repo_root, branch, base_ref="HEAD", worktrees_dir=tmp_path / "worktrees"
    )
    worker_marker = worker_info.path / "worker-in-progress.txt"
    worker_marker.write_text("do not touch\n", encoding="utf-8")

    def _forbid_create_worktree(*_args, **_kwargs):
        raise AssertionError(
            "review=True must never call create_worktree — it must use "
            "create_review_checkout instead"
        )

    monkeypatch.setattr(claude_code, "create_worktree", _forbid_create_worktree)

    record = launch_claude_worker(
        500,
        branch,
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        review=True,
        head_sha=head_sha,
    )

    assert record.ok, record.error
    review_path = Path(record.worktree_path)
    assert review_path != worker_info.path
    assert review_path.parent == sessions_dir

    # The worker's worktree is completely untouched by the review launch.
    assert worker_info.path.exists()
    assert worker_marker.exists()
    assert worker_marker.read_text(encoding="utf-8") == "do not touch\n"


def test_launch_claude_worker_review_defaults_to_read_only_permission_mode(
    tmp_path: Path,
) -> None:
    """Reviewer sessions default to --permission-mode plan (read-only), not
    the worker default of acceptEdits."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)

    record = launch_claude_worker(
        501,
        "agent/issue-501-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
    )

    assert "--permission-mode" in record.command
    mode_index = record.command.index("--permission-mode")
    assert record.command[mode_index + 1] == "plan"
    assert "acceptEdits" not in record.command


def test_launch_claude_worker_review_ignores_caller_command_template_override(
    tmp_path: Path,
) -> None:
    """Round-2 review (PR #397): a reviewer's read-only posture must not be
    defeatable by an operator's worker-tuning `claude_code.command` override.

    workflow.dispatch_reviews forwards `command_template` from
    ClaudeCodeConfig.command only when non-empty (see workflow.py); this test
    simulates the exact defeat scenario the round-2 verdict describes — an
    operator who has uncommented the example config's acceptEdits override
    for worker-tuning reasons — by passing a non-empty, acceptEdits-bearing
    command_template straight into launch_claude_worker(review=True, ...).
    The reviewer must still launch with plan mode: the adapter hard-pins the
    review command template and ignores any caller-supplied override.
    """
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)

    operator_worker_override = ("claude", "-p", "--permission-mode", "acceptEdits")

    record = launch_claude_worker(
        504,
        "agent/issue-504-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
        command_template=operator_worker_override,
    )

    # Matches the sibling default-permission-mode test's convention: assert
    # on the rendered argv, not on launch success (the `claude` binary need
    # not be spawnable in every test environment — the command is rendered
    # and recorded before the process-launch step either way).
    assert "--permission-mode" in record.command
    mode_index = record.command.index("--permission-mode")
    assert record.command[mode_index + 1] == "plan"
    assert "acceptEdits" not in record.command


def test_launch_claude_worker_pins_configured_model_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #530: a worker launch with no explicit config must still pin
    ClaudeCodeConfig's default model — never fall back to ambient global CLI
    state (the 2026-07-22 outage: every reviewer launch silently inherited an
    interactive session's premium `/model` choice and hit a credits wall)."""
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
    )

    assert "--model" in record.command
    idx = record.command.index("--model")
    assert record.command[idx + 1] == ClaudeCodeConfig().model


def test_launch_claude_worker_honors_configured_model_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    config = OrchestratorConfig(claude_code=ClaudeCodeConfig(model="claude-opus-4-8"))

    record = launch_claude_worker(
        42,
        "agent/issue-42-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        config=config,
    )

    assert record.command.count("--model") == 1
    idx = record.command.index("--model")
    assert record.command[idx + 1] == "claude-opus-4-8"


def test_launch_claude_worker_review_pins_configured_model_by_default(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)

    record = launch_claude_worker(
        502,
        "agent/issue-502-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
    )

    assert "--model" in record.command
    idx = record.command.index("--model")
    assert record.command[idx + 1] == ClaudeCodeConfig().model


def test_launch_claude_worker_review_uses_review_effort_when_set(tmp_path: Path) -> None:
    """A reviewer session must pin review_dispatch.review_effort over
    claude_code.effort when review_effort is explicitly set."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)
    config = OrchestratorConfig(
        claude_code=ClaudeCodeConfig(effort="low"),
        review_dispatch=ReviewDispatchConfig(review_effort="high"),
    )

    record = launch_claude_worker(
        503,
        "agent/issue-503-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
        config=config,
    )

    assert record.command.count("--effort") == 1
    idx = record.command.index("--effort")
    assert record.command[idx + 1] == "high"


def test_launch_claude_worker_review_falls_back_to_claude_code_effort_when_unset(
    tmp_path: Path,
) -> None:
    """review_effort empty (the default) must fall back to claude_code.effort,
    same as any other reviewer launch before this config knob existed."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)
    config = OrchestratorConfig(
        claude_code=ClaudeCodeConfig(effort="medium"),
        review_dispatch=ReviewDispatchConfig(review_effort=""),
    )

    record = launch_claude_worker(
        504,
        "agent/issue-504-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
        config=config,
    )

    assert record.command.count("--effort") == 1
    idx = record.command.index("--effort")
    assert record.command[idx + 1] == "medium"


def test_launch_claude_worker_worker_never_uses_review_effort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A worker (non-review) launch must never pick up review_effort, even
    when it's set and differs from claude_code.effort."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)
    config = OrchestratorConfig(
        claude_code=ClaudeCodeConfig(effort="low"),
        review_dispatch=ReviewDispatchConfig(review_effort="high"),
    )

    record = launch_claude_worker(
        43,
        "agent/issue-43-fix",
        "Do the thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        config=config,
    )

    assert record.command.count("--effort") == 1
    idx = record.command.index("--effort")
    assert record.command[idx + 1] == "low"


def test_review_effort_arm_is_deterministic() -> None:
    """Same inputs must always yield the same arm (stable across re-dispatches)."""
    for pr_number in (1, 42, 999, 123456):
        first = _review_effort_arm(pr_number, 0.5, "salt")
        second = _review_effort_arm(pr_number, 0.5, "salt")
        assert first == second


def test_review_effort_arm_fraction_zero_never_treatment() -> None:
    for pr_number in range(1, 200):
        assert _review_effort_arm(pr_number, 0.0, "") is False


def test_review_effort_arm_fraction_one_always_treatment() -> None:
    for pr_number in range(1, 200):
        assert _review_effort_arm(pr_number, 1.0, "") is True


def test_review_effort_arm_salt_change_flips_some_assignments() -> None:
    """Changing the salt re-randomizes arm assignment for a new epoch."""
    prs = range(1, 500)
    arms_a = {pr: _review_effort_arm(pr, 0.5, "epoch-1") for pr in prs}
    arms_b = {pr: _review_effort_arm(pr, 0.5, "epoch-2") for pr in prs}
    flipped = sum(1 for pr in prs if arms_a[pr] != arms_b[pr])
    assert flipped > 0


def test_review_effort_arm_distribution_sanity() -> None:
    """Over many sequential PR numbers, treatment share should land near
    the configured fraction (loose band to avoid test flakiness)."""
    fraction = 0.5
    prs = range(1, 1001)
    treatment_count = sum(1 for pr in prs if _review_effort_arm(pr, fraction, "sanity-salt"))
    share = treatment_count / len(prs)
    assert 0.4 < share < 0.6


def test_resolve_review_effort_disabled_uses_review_effort_unconditionally() -> None:
    """fraction<=0.0 (default): review_effort, if set, applies to every PR ---
    exactly the pre-experiment behavior. arm is None (experiment not running)."""
    review_dispatch = ReviewDispatchConfig(review_effort="high")
    claude_code_cfg = ClaudeCodeConfig(effort="low")
    for pr_number in (1, 2, 3, 4, 5):
        effort, arm = resolve_review_effort(pr_number, review_dispatch, claude_code_cfg)
        assert effort == "high"
        assert arm is None


def test_resolve_review_effort_disabled_falls_back_when_review_effort_unset() -> None:
    review_dispatch = ReviewDispatchConfig(review_effort="")
    claude_code_cfg = ClaudeCodeConfig(effort="medium")
    effort, arm = resolve_review_effort(101, review_dispatch, claude_code_cfg)
    assert effort == "medium"
    assert arm is None


def test_resolve_review_effort_enabled_splits_treatment_and_control() -> None:
    """fraction=1.0: every PR is treatment and gets review_effort. fraction=0.0-adjacent
    control case is exercised via a PR known to hash to False for a tiny fraction."""
    review_dispatch = ReviewDispatchConfig(
        review_effort="high", review_effort_experiment_fraction=1.0
    )
    claude_code_cfg = ClaudeCodeConfig(effort="low")
    effort, arm = resolve_review_effort(777, review_dispatch, claude_code_cfg)
    assert (effort, arm) == ("high", "treatment")

    # A vanishingly small fraction (but > 0.0, so the experiment IS enabled)
    # makes control the overwhelmingly likely outcome for an arbitrary PR;
    # assert against the deterministic arm function directly instead of
    # relying on probability for a single PR.
    tiny_fraction = 1e-9
    salt = ""
    is_treatment = _review_effort_arm(777, tiny_fraction, salt)
    review_dispatch_tiny = ReviewDispatchConfig(
        review_effort="high", review_effort_experiment_fraction=tiny_fraction
    )
    effort, arm = resolve_review_effort(777, review_dispatch_tiny, claude_code_cfg)
    if is_treatment:
        assert (effort, arm) == ("high", "treatment")
    else:
        assert (effort, arm) == ("low", "control")


def test_launch_claude_worker_review_experiment_treatment_pins_review_effort(
    tmp_path: Path,
) -> None:
    """Experiment enabled (fraction=1.0, so every PR is treatment) --> the
    reviewer session pins review_effort, same as the always-on case."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)
    config = OrchestratorConfig(
        claude_code=ClaudeCodeConfig(effort="low"),
        review_dispatch=ReviewDispatchConfig(
            review_effort="high", review_effort_experiment_fraction=1.0
        ),
    )

    record = launch_claude_worker(
        601,
        "agent/issue-601-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
        config=config,
    )

    assert record.command.count("--effort") == 1
    idx = record.command.index("--effort")
    assert record.command[idx + 1] == "high"


def test_launch_claude_worker_review_experiment_control_falls_back(tmp_path: Path) -> None:
    """Experiment enabled with a vanishingly small fraction --> an arbitrary
    PR is (deterministically) assigned control and falls back to
    claude_code.effort instead of review_effort."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)
    pr_number = 602
    tiny_fraction = 1e-9
    assert _review_effort_arm(pr_number, tiny_fraction, "") is False
    config = OrchestratorConfig(
        claude_code=ClaudeCodeConfig(effort="low"),
        review_dispatch=ReviewDispatchConfig(
            review_effort="high", review_effort_experiment_fraction=tiny_fraction
        ),
    )

    record = launch_claude_worker(
        pr_number,
        "agent/issue-602-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
        config=config,
    )

    assert record.command.count("--effort") == 1
    idx = record.command.index("--effort")
    assert record.command[idx + 1] == "low"


def test_launch_claude_worker_review_uses_resolved_review_effort_passthrough(
    tmp_path: Path,
) -> None:
    """When the caller (dispatch_reviews) already resolved the review_effort
    experiment arm at claim time and passes it via resolved_review_effort,
    launch_claude_worker must use that value directly rather than
    re-resolving from config -- this is the single-computation-site
    invariant: the claim-time resolution is authoritative, not a preview."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)
    # Config alone would resolve to "high" (fraction=1.0, review_effort=high),
    # but the passed-through resolved_review_effort deliberately differs so
    # the test can distinguish "used the passthrough" from "recomputed".
    config = OrchestratorConfig(
        claude_code=ClaudeCodeConfig(effort="low"),
        review_dispatch=ReviewDispatchConfig(
            review_effort="high", review_effort_experiment_fraction=1.0
        ),
    )

    record = launch_claude_worker(
        603,
        "agent/issue-603-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=head_sha,
        config=config,
        resolved_review_effort="medium",
    )

    assert record.command.count("--effort") == 1
    idx = record.command.index("--effort")
    assert record.command[idx + 1] == "medium"


def test_sanitize_review_command_template_strips_duplicate_space_form_flags() -> None:
    """Round-3 review (PR #397): a template with duplicate space-form
    `--permission-mode` flags must not let the trailing occurrence survive —
    CLI parsers apply last-flag-wins semantics, so a naive first-match fix
    would still launch in acceptEdits mode."""
    template = (
        "claude",
        "-p",
        "--permission-mode",
        "plan",
        "--permission-mode",
        "acceptEdits",
    )

    result = _sanitize_review_command_template(template)

    assert result.count("--permission-mode") == 1
    idx = result.index("--permission-mode")
    assert result[idx + 1] == "plan"
    assert idx == len(result) - 2  # positioned last


def test_sanitize_review_command_template_strips_equals_joined_flag() -> None:
    """An equals-joined `--permission-mode=acceptEdits` token must be removed
    entirely, not merely left in place because the append-based happy path
    currently makes it look safe by accident."""
    template = ("claude", "-p", "--permission-mode=acceptEdits")

    result = _sanitize_review_command_template(template)

    assert not any(tok.startswith("--permission-mode=") for tok in result)
    assert result[-2:] == ("--permission-mode", "plan")


def test_sanitize_review_command_template_strips_mixed_forms() -> None:
    """Mixed equals-joined and space-form occurrences are all stripped,
    leaving a single trailing `--permission-mode plan`."""
    template = (
        "claude",
        "-p",
        "--permission-mode=acceptEdits",
        "--permission-mode",
        "acceptEdits",
    )

    result = _sanitize_review_command_template(template)

    assert result.count("--permission-mode") == 1
    assert not any(tok.startswith("--permission-mode=") for tok in result)
    assert result[-2:] == ("--permission-mode", "plan")


def test_sanitize_review_command_template_handles_bare_trailing_flag() -> None:
    """A malformed trailing `--permission-mode` with no value token must not
    raise (e.g. IndexError) — it is stripped like any other occurrence and
    the authoritative flag is appended."""
    template = ("claude", "-p", "--permission-mode")

    result = _sanitize_review_command_template(template)

    assert result == ("claude", "-p", "--permission-mode", "plan")


def test_sanitize_review_command_template_preserves_lookalike_token() -> None:
    """A token like `--permission-modex` must not be matched as the flag —
    only an exact `--permission-mode` token or exact `--permission-mode=`
    prefix count."""
    template = ("claude", "-p", "--permission-modex", "plan")

    result = _sanitize_review_command_template(template)

    assert "--permission-modex" in result
    assert result == ("claude", "-p", "--permission-modex", "plan", "--permission-mode", "plan")


def test_apply_model_pin_appends_to_template_without_model() -> None:
    """Issue #530: a bare template (no --model) must get the configured
    model pinned so the subprocess never falls back to ambient global CLI
    state (e.g. an interactive session's last `/model` choice)."""
    template = ("claude", "-p", "--permission-mode", "plan")

    result = _apply_model_pin(template, "claude-sonnet-5")

    assert result == ("claude", "-p", "--permission-mode", "plan", "--model", "claude-sonnet-5")


def test_apply_model_pin_strips_existing_space_form_flag() -> None:
    template = ("claude", "-p", "--model", "claude-opus-4-8", "--permission-mode", "plan")

    result = _apply_model_pin(template, "claude-sonnet-5")

    assert result.count("--model") == 1
    idx = result.index("--model")
    assert result[idx + 1] == "claude-sonnet-5"
    assert idx == len(result) - 2  # positioned last, last-flag-wins


def test_apply_model_pin_strips_equals_joined_flag() -> None:
    template = ("claude", "-p", "--model=claude-opus-4-8")

    result = _apply_model_pin(template, "claude-sonnet-5")

    assert not any(tok.startswith("--model=") for tok in result)
    assert result[-2:] == ("--model", "claude-sonnet-5")


def test_apply_model_pin_handles_bare_trailing_flag() -> None:
    template = ("claude", "-p", "--model")

    result = _apply_model_pin(template, "claude-sonnet-5")

    assert result == ("claude", "-p", "--model", "claude-sonnet-5")


def test_apply_model_pin_preserves_lookalike_token() -> None:
    template = ("claude", "-p", "--modelx", "plan")

    result = _apply_model_pin(template, "claude-sonnet-5")

    assert "--modelx" in result
    assert result == ("claude", "-p", "--modelx", "plan", "--model", "claude-sonnet-5")


def test_apply_model_pin_handles_empty_template() -> None:
    assert _apply_model_pin((), "claude-sonnet-5") == ("--model", "claude-sonnet-5")


def test_apply_effort_pin_appends_to_template_without_effort() -> None:
    template = ("claude", "-p", "--permission-mode", "plan")

    result = _apply_effort_pin(template, "medium")

    assert result == ("claude", "-p", "--permission-mode", "plan", "--effort", "medium")


def test_apply_effort_pin_strips_existing_space_form_flag() -> None:
    template = ("claude", "-p", "--effort", "high", "--permission-mode", "plan")

    result = _apply_effort_pin(template, "medium")

    assert result.count("--effort") == 1
    idx = result.index("--effort")
    assert result[idx + 1] == "medium"
    assert idx == len(result) - 2


def test_apply_effort_pin_strips_equals_joined_flag() -> None:
    template = ("claude", "-p", "--effort=high")

    result = _apply_effort_pin(template, "medium")

    assert not any(tok.startswith("--effort=") for tok in result)
    assert result[-2:] == ("--effort", "medium")


def test_apply_effort_pin_empty_effort_is_noop() -> None:
    template = ("claude", "-p", "--permission-mode", "plan")

    result = _apply_effort_pin(template, "")

    assert result == template


def test_apply_effort_pin_handles_bare_trailing_flag() -> None:
    template = ("claude", "-p", "--effort")

    result = _apply_effort_pin(template, "medium")

    assert result == ("claude", "-p", "--effort", "medium")


def test_apply_effort_pin_preserves_lookalike_token() -> None:
    template = ("claude", "-p", "--effortx", "plan")

    result = _apply_effort_pin(template, "medium")

    assert "--effortx" in result
    assert result == ("claude", "-p", "--effortx", "plan", "--effort", "medium")


def test_apply_effort_pin_handles_empty_template() -> None:
    assert _apply_effort_pin((), "medium") == ("--effort", "medium")


def test_launch_claude_worker_worker_defaults_to_accept_edits_permission_mode(
    tmp_path: Path,
) -> None:
    """Non-review (worker) launches keep the pre-existing acceptEdits default."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "sessions"

    record = launch_claude_worker(
        502,
        "agent/issue-502-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
    )

    assert "--permission-mode" in record.command
    mode_index = record.command.index("--permission-mode")
    assert record.command[mode_index + 1] == "acceptEdits"


def test_launch_claude_worker_review_missing_head_sha_returns_error_record(
    tmp_path: Path,
) -> None:
    """review=True without head_sha is a caller error (ValueError), surfaced
    as an error record — launch_claude_worker must never raise."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"

    record = launch_claude_worker(
        503,
        "agent/issue-503-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        review=True,
        head_sha=None,
    )

    assert not record.ok
    assert record.pid is None
    assert "head_sha" in record.error


def test_launch_claude_worker_review_prompt_write_failure_tears_down_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If writing the prompt file fails in review mode, the isolated review
    checkout — not a worker worktree — is torn down (via
    remove_review_checkout, keyed by PR number, not remove_worktree)."""
    repo_root = tmp_path / "repo"
    _init_real_repo(repo_root)
    sessions_dir = tmp_path / "reviews"
    head_sha = _repo_head_sha(repo_root)

    original_write_text = Path.write_text

    def failing_write_text(self, content, encoding=None, errors=None):
        if self.name == ".orchestrator-prompt.md":
            raise OSError("Mock prompt write failure")
        return original_write_text(self, content, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    record = launch_claude_worker(
        504,
        "agent/issue-504-fix",
        "prompt text",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        review=True,
        head_sha=head_sha,
    )

    assert not record.ok
    assert "failed to write prompt file" in record.error

    checkout_path = sessions_dir / "pr-504"
    assert not checkout_path.exists()
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert str(checkout_path) not in result.stdout


def test_launch_claude_worker_api_kind_sidecar_naming(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An api-kind launch writes ``issue-<n>.api.json`` and records the kind/provider."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_claude_worker(
        42,
        "agent/issue-42-api",
        "Do the api thing.",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=_fake_claude_script(tmp_path),
        adapter_kind="api",
        provider="openai",
    )

    assert record.ok
    assert record.adapter_kind == "api"
    assert record.provider == "openai"

    sidecar_path = sessions_dir / "issue-42.api.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["adapter_kind"] == "api"
    assert payload["provider"] == "openai"

    # Default read_worker_records filters to claude-code and must not return api records.
    assert read_worker_records(sessions_dir) == []

    # Reading with the api kind filter returns the record and preserves identity.
    api_records = read_worker_records(sessions_dir, adapter_kind="api")
    assert len(api_records) == 1
    assert api_records[0].adapter_kind == "api"
    assert api_records[0].provider == "openai"


def test_claude_worker_record_round_trips_new_fields(tmp_path: Path) -> None:
    """``to_dict`` / ``from_dict`` carry ``adapter_kind`` and ``provider``."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    record = ClaudeWorkerRecord(
        issue_number=7,
        branch="agent/issue-7",
        worktree_path=str(tmp_path / "wt"),
        prompt_path="p.md",
        command=("claude", "-p", "p.md"),
        pid=123,
        started_at="2026-07-20T00:00:00Z",
        log_path="log.txt",
        adapter_kind="api",
        provider="openai",
    )

    sidecar_path = sessions_dir / "issue-7.api.json"
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    [restored] = read_worker_records(sessions_dir, adapter_kind=None)
    assert restored == record


def test_claude_worker_record_from_dict_defaults_legacy_keys() -> None:
    """Sidecars written before the new fields default them on read."""
    payload = {
        "issue_number": 9,
        "branch": "agent/issue-9",
        "worktree_path": "/wt",
        "prompt_path": "p.md",
        "command": ["claude", "-p", "p.md"],
        "pid": 999,
        "started_at": "2026-07-20T00:00:00Z",
        "log_path": "log.txt",
    }

    record = ClaudeWorkerRecord.from_dict(payload)

    assert record.adapter_kind == "claude-code"
    assert record.provider == ""
