from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from charlie_work import devin_shell
from charlie_work.devin_shell import (
    DEFAULT_COMMAND_TEMPLATE,
    SessionRecord,
    is_session_alive,
    launch_devin_session,
    probe_devin,
    read_session_records,
)
from charlie_work.worktree import WorktreeInfo

# A tiny fake "devin" CLI: writes its argv to stdout and exits 0. Launched via
# sys.executable to dodge PATH entirely (mirrors the sys.executable fake-binary
# pattern in tests/test_charlie_work.py).
_FAKE_DEVIN_SLEEP = """
import sys, time
sys.stdout.write("fake-devin argv=" + " ".join(sys.argv[1:]) + "\\n")
sys.stdout.flush()
time.sleep(0.2)
"""

_FAKE_DEVIN_VERSION = """
import sys
print("devin-fake 0.0.1")
sys.exit(0)
"""


def _write_fake_devin(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake_devin.py"
    script.write_text(body, encoding="utf-8")
    return script


def _fake_worktree(tmp_path: Path, branch: str) -> WorktreeInfo:
    worktree_path = tmp_path / "worktrees" / branch.replace("/", "-")
    worktree_path.mkdir(parents=True, exist_ok=True)
    return WorktreeInfo(path=worktree_path, branch=branch, venv_junction=None)


def _install_fake_create_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, calls: list[dict] | None = None
) -> None:
    def fake_create_worktree(
        repo_root, branch, *, base_ref="HEAD", worktrees_dir=None, venv_source=None
    ):
        if calls is not None:
            calls.append(
                {
                    "repo_root": repo_root,
                    "branch": branch,
                    "worktrees_dir": worktrees_dir,
                }
            )
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(devin_shell, "create_worktree", fake_create_worktree)


# ---------------------------------------------------------------------------
# Regression: DEFAULT_COMMAND_TEMPLATE must include --permission-mode dangerous
# ---------------------------------------------------------------------------


def test_default_command_template_contains_permission_mode_dangerous() -> None:
    """Devin CLI defaults to --permission-mode auto (read-only); headless
    workers stall the moment they need git/uv/gh. The default template must
    explicitly pass --permission-mode dangerous."""
    template = DEFAULT_COMMAND_TEMPLATE
    template_str = " ".join(template)
    assert "--permission-mode" in template_str, (
        "DEFAULT_COMMAND_TEMPLATE must contain '--permission-mode'"
    )
    assert "dangerous" in template_str, (
        "DEFAULT_COMMAND_TEMPLATE must set --permission-mode dangerous"
    )


def test_default_command_template_permission_mode_flag_is_adjacent() -> None:
    """--permission-mode and dangerous must be consecutive argv tokens."""
    tpl = list(DEFAULT_COMMAND_TEMPLATE)
    idx = tpl.index("--permission-mode")
    assert tpl[idx + 1] == "dangerous", (
        f"Expected 'dangerous' after '--permission-mode', got {tpl[idx + 1]!r}"
    )


# ---------------------------------------------------------------------------
# Regression: launch cwd must be the worktree, not repo_root
# ---------------------------------------------------------------------------


def test_launch_cwd_is_worktree_not_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workers must run inside an isolated per-issue worktree, not in repo_root.

    Concurrent workers sharing repo_root fight over one checkout (competing
    `git checkout -b`, index mutations, test artifacts)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")

    worktree_calls: list[dict] = []
    _install_fake_create_worktree(monkeypatch, tmp_path, calls=worktree_calls)

    # Script writes its cwd to stdout so we can verify it.
    cwd_script = tmp_path / "echo_cwd.py"
    cwd_script.write_text(
        "import os, sys\nsys.stdout.write(os.getcwd() + '\\n')\nsys.stdout.flush()\n",
        encoding="utf-8",
    )

    record = launch_devin_session(
        55,
        "agent/issue-55-test",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(cwd_script)),
    )

    assert record.error is None
    assert record.pid is not None

    # worktree creation must have been called
    assert len(worktree_calls) == 1
    assert worktree_calls[0]["branch"] == "agent/issue-55-test"
    assert worktree_calls[0]["repo_root"] == repo_root

    # worktree_path in record must not be repo_root
    assert record.worktree_path != str(repo_root), (
        "launch cwd should be the worktree, not repo_root"
    )
    assert record.worktree_path  # non-empty

    # The sidecar must also record worktree_path
    sidecar = json.loads((sessions_dir / "issue-55.json").read_text(encoding="utf-8"))
    assert sidecar["worktree_path"] == record.worktree_path

    # Give the subprocess a moment, then verify it actually ran in the worktree
    deadline = time.time() + 5
    while time.time() < deadline:
        log_text = Path(record.log_path).read_text(encoding="utf-8")
        if log_text.strip():
            break
        time.sleep(0.05)
    log_text = Path(record.log_path).read_text(encoding="utf-8").strip()
    assert Path(log_text).resolve() == Path(record.worktree_path).resolve(), (
        f"Process cwd was {log_text!r}, expected worktree {record.worktree_path!r}"
    )


# ---------------------------------------------------------------------------
# Core launch behaviour
# ---------------------------------------------------------------------------


def test_launch_writes_sidecar_json_with_expected_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("do the thing", encoding="utf-8")
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_SLEEP)

    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_devin_session(
        123,
        "agent/issue-123-fix",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script), "{prompt_path}", "{issue_number}"),
    )

    assert record.issue_number == 123
    assert record.branch == "agent/issue-123-fix"
    assert record.prompt_path == str(prompt_path)
    assert record.command == (sys.executable, str(script), str(prompt_path), "123")
    assert record.error is None
    assert record.pid is not None
    assert record.started_at  # non-empty ISO-ish timestamp
    assert record.log_path == str(sessions_dir / "issue-123.log")
    assert record.worktree_path  # non-empty

    sidecar_path = sessions_dir / "issue-123.json"
    assert sidecar_path.is_file()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["issue_number"] == 123
    assert payload["branch"] == "agent/issue-123-fix"
    assert payload["prompt_path"] == str(prompt_path)
    assert payload["pid"] == record.pid
    assert payload["log_path"] == str(sessions_dir / "issue-123.log")
    assert payload["error"] is None
    assert payload["worktree_path"] == record.worktree_path

    # Non-blocking: launch_devin_session must not have waited for the
    # subprocess (which sleeps 0.2s) — give it a moment then check the log.
    deadline = time.time() + 5
    while time.time() < deadline and not Path(record.log_path).read_text(encoding="utf-8"):
        time.sleep(0.05)
    log_text = Path(record.log_path).read_text(encoding="utf-8")
    assert "fake-devin argv=" in log_text


def test_launch_with_missing_binary_yields_error_record_not_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_devin_session(
        7,
        "agent/issue-7-x",
        tmp_path / "prompt.md",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(
            "definitely-not-a-real-devin-binary-xyz",
            "--prompt-file",
            "{prompt_path}",
        ),
    )

    assert record.pid is None
    assert record.error is not None
    assert record.issue_number == 7

    sidecar_path = sessions_dir / "issue-7.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["pid"] is None
    assert payload["error"] is not None


def test_launch_worktree_creation_failure_yields_error_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If worktree creation fails, launch must return an error record, not raise."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    def failing_create_worktree(*args, **kwargs):
        raise RuntimeError("git worktree add failed: branch already exists")

    monkeypatch.setattr(devin_shell, "create_worktree", failing_create_worktree)

    record = launch_devin_session(
        8,
        "agent/issue-8-conflict",
        tmp_path / "prompt.md",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
    )

    assert record.pid is None
    assert record.error is not None
    assert "worktree creation failed" in record.error
    assert record.worktree_path == ""

    payload = json.loads((sessions_dir / "issue-8.json").read_text(encoding="utf-8"))
    assert payload["pid"] is None
    assert payload["error"] is not None


def test_read_session_records_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_SLEEP)
    prompt_a = tmp_path / "a.md"
    prompt_b = tmp_path / "b.md"
    prompt_a.write_text("a", encoding="utf-8")
    prompt_b.write_text("b", encoding="utf-8")

    _install_fake_create_worktree(monkeypatch, tmp_path)

    launch_devin_session(
        1,
        "agent/issue-1",
        prompt_a,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script), "{prompt_path}"),
    )
    launch_devin_session(
        2,
        "agent/issue-2",
        prompt_b,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script), "{prompt_path}"),
    )

    records = read_session_records(sessions_dir)

    assert len(records) == 2
    by_issue = {record.issue_number: record for record in records}
    assert set(by_issue) == {1, 2}
    assert by_issue[1].branch == "agent/issue-1"
    assert by_issue[2].branch == "agent/issue-2"
    assert all(isinstance(record, SessionRecord) for record in records)


def test_read_session_records_empty_dir_returns_empty_list(tmp_path: Path) -> None:
    assert read_session_records(tmp_path / "does-not-exist") == []


def test_read_session_records_skips_malformed_sidecar(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "issue-99.json").write_text("{not valid json", encoding="utf-8")

    assert read_session_records(sessions_dir) == []


def test_read_session_records_skips_claude_code_sidecars(tmp_path: Path) -> None:
    # Both adapters share one sessions_dir. The devin glob `issue-*.json` also
    # matches the claude-code adapter's `issue-N.claude.json` sidecars, so
    # read_session_records must skip them (otherwise doctor double-counts every
    # Claude worker and tries to parse a foreign schema).
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    devin_payload = {
        "issue_number": 5,
        "branch": "agent/issue-5",
        "worktree_path": "/tmp/wt/issue-5",
        "prompt_path": "p.md",
        "command": ["devin", "--print"],
        "pid": 1234,
        "started_at": "2026-01-01T00:00:00Z",
        "log_path": "issue-5.log",
        "error": None,
    }
    (sessions_dir / "issue-5.json").write_text(json.dumps(devin_payload), encoding="utf-8")
    # A claude-code sidecar with a deliberately foreign schema: if the exclusion
    # regressed, from_dict would choke on the missing devin-shaped keys.
    (sessions_dir / "issue-6.claude.json").write_text(
        json.dumps({"issue_number": 6, "worktree": "wt", "pid": 5678}), encoding="utf-8"
    )

    records = read_session_records(sessions_dir)

    assert [record.issue_number for record in records] == [5]


def test_probe_devin_ok_for_zero_exit_fake(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_VERSION)

    result = probe_devin(repo_root, command=(sys.executable, str(script)))

    assert result.ok is True
    assert result.returncode == 0
    assert "devin-fake" in result.stdout


def test_probe_devin_not_ok_for_missing_binary(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    result = probe_devin(repo_root, command=("definitely-not-a-real-devin-binary-xyz",))

    assert result.ok is False
    assert result.error is not None


def test_is_session_alive_reflects_real_process(tmp_path: Path) -> None:
    script = _write_fake_devin(tmp_path, "import time; time.sleep(2)")
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        alive_record = SessionRecord(
            issue_number=1,
            branch="agent/issue-1",
            worktree_path="/tmp/wt/issue-1",
            prompt_path="p.md",
            command=("x",),
            pid=process.pid,
            started_at="2026-01-01T00:00:00Z",
            log_path="log.txt",
        )
        assert is_session_alive(alive_record) is True
    finally:
        process.kill()
        process.wait(timeout=5)

    # Regression guard for the Windows os.kill(pid, 0) trap: that call keeps
    # reporting a reaped PID as alive indefinitely (verified empirically —
    # see is_session_alive's docstring), so this must be an exact
    # post-wait() assertion, not a "poll until it settles" retry loop.
    dead_record = SessionRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/wt/issue-1",
        prompt_path="p.md",
        command=("x",),
        pid=process.pid,
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
    )
    assert is_session_alive(dead_record) is False


def test_is_session_alive_false_for_none_pid() -> None:
    record = SessionRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="",
        prompt_path="p.md",
        command=("x",),
        pid=None,
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
        error="devin not found",
    )

    assert is_session_alive(record) is False


def test_is_session_alive_false_for_implausible_pid() -> None:
    # A PID this large was never handed out by Popen; OpenProcess should
    # simply fail to find it (and on POSIX os.kill raises ProcessLookupError).
    # Confirms the "not found" path returns False without raising.
    record = SessionRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="",
        prompt_path="p.md",
        command=("x",),
        pid=999_999_999,
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
    )

    assert is_session_alive(record) is False


def test_command_template_renders_issue_and_branch_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_SLEEP)

    _install_fake_create_worktree(monkeypatch, tmp_path)

    record = launch_devin_session(
        42,
        "agent/issue-42-widgets",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(
            sys.executable,
            str(script),
            "--issue",
            "{issue_number}",
            "--branch",
            "{branch}",
            "--prompt-file",
            "{prompt_path}",
        ),
    )

    assert record.command == (
        sys.executable,
        str(script),
        "--issue",
        "42",
        "--branch",
        "agent/issue-42-widgets",
        "--prompt-file",
        str(prompt_path),
    )
