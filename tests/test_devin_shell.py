from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from devin_orchestrator.devin_shell import (
    SessionRecord,
    is_session_alive,
    launch_devin_session,
    probe_devin,
    read_session_records,
)

# A tiny fake "devin" CLI: writes its argv to stdout and exits 0. Launched via
# sys.executable to dodge PATH entirely (mirrors the sys.executable fake-binary
# pattern in tests/test_devin_orchestrator.py).
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


def test_launch_writes_sidecar_json_with_expected_fields(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("do the thing", encoding="utf-8")
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_SLEEP)

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

    sidecar_path = sessions_dir / "issue-123.json"
    assert sidecar_path.is_file()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["issue_number"] == 123
    assert payload["branch"] == "agent/issue-123-fix"
    assert payload["prompt_path"] == str(prompt_path)
    assert payload["pid"] == record.pid
    assert payload["log_path"] == str(sessions_dir / "issue-123.log")
    assert payload["error"] is None

    # Non-blocking: launch_devin_session must not have waited for the
    # subprocess (which sleeps 0.2s) — give it a moment then check the log.
    deadline = time.time() + 5
    while time.time() < deadline and not Path(record.log_path).read_text(encoding="utf-8"):
        time.sleep(0.05)
    log_text = Path(record.log_path).read_text(encoding="utf-8")
    assert "fake-devin argv=" in log_text


def test_launch_with_missing_binary_yields_error_record_not_exception(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

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


def test_read_session_records_round_trips(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_SLEEP)
    prompt_a = tmp_path / "a.md"
    prompt_b = tmp_path / "b.md"
    prompt_a.write_text("a", encoding="utf-8")
    prompt_b.write_text("b", encoding="utf-8")

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
        prompt_path="p.md",
        command=("x",),
        pid=999_999_999,
        started_at="2026-01-01T00:00:00Z",
        log_path="log.txt",
    )

    assert is_session_alive(record) is False


def test_command_template_renders_issue_and_branch_placeholders(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_SLEEP)

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
