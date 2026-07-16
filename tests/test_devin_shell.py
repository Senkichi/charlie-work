from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from charlie_work import devin_shell
from charlie_work.config import DevinConfig, OrchestratorConfig, RuntimeConfig
from charlie_work.devin_shell import (
    DEFAULT_COMMAND_TEMPLATE,
    SessionRecord,
    get_rate_limit_defer_until,
    is_session_alive,
    launch_devin_session,
    probe_devin,
    read_session_records,
    update_session_record_with_failure_classification,
    _sidecar_path,
    _write_json,
)
from charlie_work.env_sanitize import sanitize_env
from charlie_work.worktree import WorktreeInfo, create_worktree, is_junction, remove_worktree

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
    ):
        if calls is not None:
            calls.append(
                {
                    "repo_root": repo_root,
                    "branch": branch,
                    "worktrees_dir": worktrees_dir,
                    "venv_source": venv_source,
                    "materialize_dirs": materialize_dirs,
                    "rework": rework,
                    "recovery": recovery,
                    "issue_number": issue_number,
                    "config": config,
                }
            )
        if with_venv:
            return _fake_worktree_with_venv(tmp_path, branch)
        return _fake_worktree(tmp_path, branch)

    monkeypatch.setattr(devin_shell, "create_worktree", fake_create_worktree)


def _init_repo(repo_root: Path) -> None:
    """Create a minimal git repo at ``repo_root`` for worktree tests."""
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=main"], cwd=repo_root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.test"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=repo_root, check=True, capture_output=True
    )
    (repo_root / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"], cwd=repo_root, check=True, capture_output=True
    )


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


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
    assert "{model_args}" in template_str, (
        "DEFAULT_COMMAND_TEMPLATE must contain '{model_args}' placeholder for config-driven model selection"
    )


def test_default_command_template_permission_mode_flag_is_adjacent() -> None:
    """--permission-mode and dangerous must be consecutive argv tokens."""
    # After rendering with an empty worker_model, {model_args} becomes an empty
    # string and is filtered out, so --permission-mode and dangerous are adjacent.
    from charlie_work.devin_shell import _render_command

    rendered = _render_command(
        DEFAULT_COMMAND_TEMPLATE,
        issue_number=1,
        branch="x",
        prompt_path=Path("p.md"),
        worker_model="",
    )
    idx = rendered.index("--permission-mode")
    assert rendered[idx + 1] == "dangerous", (
        f"Expected 'dangerous' after '--permission-mode', got {rendered[idx + 1]!r}"
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


def test_launch_devin_session_worker_env_overrides_sanitize_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """worker_env overrides sanitize_env: operator-provided VIRTUAL_ENV wins.

    This is a mutation gate for the merge order in launch_devin_session:
    the current order is {**sanitize_env(...), **worker_env}, so worker_env
    clobbers sanitized keys. If the order is inverted (worker_env first,
    sanitize_env clobbering it), this test fails.

    The fixture uses with_venv=True so sanitize_env actively SETS VIRTUAL_ENV
    (instead of POP-ing it), making the merge order sensitive.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("do the thing", encoding="utf-8")

    _install_fake_create_worktree(monkeypatch, tmp_path, with_venv=True)

    # Set a VIRTUAL_ENV in the orchestrator's environment (which sanitize_env
    # would normally strip). Then provide an explicit override via worker_env.
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/venv")

    # Script writes VIRTUAL_ENV to stdout so we can verify it
    env_probe_script = tmp_path / "env_probe.py"
    env_probe_script.write_text(
        "import os, sys\nsys.stdout.write(os.environ.get('VIRTUAL_ENV', '<unset>') + '\\n')\nsys.stdout.flush()\n",
        encoding="utf-8",
    )

    record = launch_devin_session(
        140,
        "agent/issue-140-env-override",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(env_probe_script)),
        worker_env={"VIRTUAL_ENV": "/custom/override/venv"},
    )

    assert record.error is None
    assert record.pid is not None

    # Give the subprocess a moment, then verify VIRTUAL_ENV in the log
    deadline = time.time() + 5
    while time.time() < deadline:
        log_text = Path(record.log_path).read_text(encoding="utf-8")
        if log_text.strip():
            break
        time.sleep(0.05)
    log_text = Path(record.log_path).read_text(encoding="utf-8").strip()
    # worker_env VIRTUAL_ENV override wins over sanitize_env's stripping
    assert log_text == "/custom/override/venv"


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
    # Regression guard: empty strings should not appear in the rendered command
    assert "" not in record.command

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


def test_launch_devin_session_passes_rework_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rework flag should be passed through to create_worktree."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"

    rework_calls = []

    def tracking_create_worktree(*args, **kwargs):
        rework_calls.append(kwargs.get("rework", False))
        # Return a fake WorktreeInfo to avoid actual git operations
        from charlie_work.worktree import WorktreeInfo

        return WorktreeInfo(path=tmp_path / "fake-wt", branch="test", venv_junction=None)

    monkeypatch.setattr(devin_shell, "create_worktree", tracking_create_worktree)

    # Test with rework=False (default)
    launch_devin_session(
        1,
        "agent/issue-1",
        tmp_path / "prompt.md",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        rework=False,
    )

    # Test with rework=True
    launch_devin_session(
        2,
        "agent/issue-2",
        tmp_path / "prompt.md",
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        rework=True,
    )

    assert rework_calls == [False, True]


def test_launch_devin_session_fetch_failure_yields_error_record_not_exception(
    tmp_path: Path,
) -> None:
    """End-to-end test: real fetch failure inside create_worktree must return error record.

    This test forces a real git fetch failure by creating a repo with a broken origin URL,
    then calling launch_devin_session with base_ref that triggers a fetch. The adapter must
    catch the RuntimeError from create_worktree and return an error record, not raise.

    This is a mutation gate: if RuntimeError is removed from the except tuple in
    launch_devin_session, this test will fail with an uncaught exception.
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
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("do the thing", encoding="utf-8")

    # Break the origin remote to simulate a fetch failure
    subprocess.run(
        ["git", "remote", "set-url", "origin", "file:///nonexistent/path"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Call launch_devin_session with base_ref that triggers a fetch
    # Empty string resolves to origin/main, which will trigger the fetch in create_worktree
    record = launch_devin_session(
        142,
        "agent/issue-142-fetch-failure",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        base_ref="",  # Triggers fetch of origin/main
    )

    # The adapter must catch the RuntimeError and return an error record
    assert record.pid is None
    assert record.error is not None
    assert "worktree creation failed" in record.error
    assert record.worktree_path == ""

    # Verify the sidecar was written with the error
    sidecar_path = sessions_dir / "issue-142.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["pid"] is None
    assert payload["error"] is not None
    assert "worktree creation failed" in payload["error"]


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


def test_read_session_records_skips_post_mortem_sidecars(tmp_path: Path) -> None:
    """Issue #343 Finding 1: the devin glob `issue-*.json` also matches
    post_mortem's `issue-N.post-mortem.json` sidecars (both live in the same
    sessions_dir). SessionRecord.from_dict only strictly requires
    `issue_number` (present in a PostMortemRecord payload too), so an
    unfiltered post-mortem file parses into a bogus SessionRecord(pid=None,
    log_path=""). That phantom bypasses `if w.pid is not None:`
    corroboration downstream in the dead-session reaper and reaches
    `reap_sidecar`, which resolves to the SAME path as the real
    `issue-N.json` sidecar and deletes it -- silently reaping a worker whose
    liveness was never actually re-verified.

    read_session_records must skip `issue-N.post-mortem.json` (and, for
    completeness, the pre-existing `issue-N.claude.json` exclusion) and
    return exactly the one real devin sidecar.
    """
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    devin_payload = {
        "issue_number": 343,
        "branch": "agent/issue-343-x",
        "worktree_path": "/tmp/wt/issue-343",
        "prompt_path": "p.md",
        "command": ["devin", "--print"],
        "pid": 4242,
        "started_at": "2026-01-01T00:00:00Z",
        "log_path": "issue-343.log",
        "error": None,
    }
    (sessions_dir / "issue-343.json").write_text(json.dumps(devin_payload), encoding="utf-8")
    (sessions_dir / "issue-343.log").write_text("Working...\n", encoding="utf-8")
    # A post-mortem sidecar with a foreign schema (issue_number is the only
    # key it shares with SessionRecord) -- if the exclusion regressed, this
    # would parse into a bogus pid=None SessionRecord.
    post_mortem_payload = {
        "issue_number": 343,
        "generated_at": "2026-01-01T00:05:00Z",
        "db_path": "C:/fake/sessions.db",
        "matched": False,
        "extraction_error": "no session found matching working_directory",
    }
    (sessions_dir / "issue-343.post-mortem.json").write_text(
        json.dumps(post_mortem_payload), encoding="utf-8"
    )
    # The claude-code adapter's sidecar, included for completeness alongside
    # the pre-existing exclusion this test also guards.
    (sessions_dir / "issue-343.claude.json").write_text(
        json.dumps({"issue_number": 343, "worktree": "wt", "pid": 9999}), encoding="utf-8"
    )

    records = read_session_records(sessions_dir)

    assert len(records) == 1
    assert records[0].issue_number == 343
    assert records[0].pid == 4242


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


def test_is_session_alive_rejects_pid_recycling_mismatched_start_time(tmp_path: Path) -> None:
    """A record with an alive PID but mismatched start time is treated as dead.

    This prevents false positives from PID recycling: if the OS has reused the PID
    for a different process, the start time will not match.
    """
    from charlie_work.devin_shell import _get_process_start_time

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

        record = SessionRecord(
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
        assert is_session_alive(record) is False
    finally:
        process.kill()
        process.wait(timeout=5)


def test_is_session_alive_accepts_matching_start_time(tmp_path: Path) -> None:
    """A record with matching start time is counted as live."""
    from charlie_work.devin_shell import _get_process_start_time

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

        record = SessionRecord(
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
        assert is_session_alive(record) is True
    finally:
        process.kill()
        process.wait(timeout=5)


def test_is_session_alive_legacy_record_fallback() -> None:
    """Legacy records without process_start_time fall back to pid-only liveness.

    This preserves backward compatibility for old sidecar files.
    """
    # Use the test process's own PID (guaranteed to be alive)
    record = SessionRecord(
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
    assert is_session_alive(record) is True


def test_sidecar_path_returns_correct_path(tmp_path: Path) -> None:
    """_sidecar_path returns the expected path for a given issue number."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    path = _sidecar_path(sessions_dir, 123)
    assert path == sessions_dir / "issue-123.json"


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


def test_is_session_alive_probe_none_treats_indeterminate_as_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #360 criterion #1: a start-time probe failure is not a definitive dead signal.

    When ``get_process_start_time`` returns ``None`` for a live PID, ``is_session_alive``
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

        record = SessionRecord(
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
        assert is_session_alive(record) is True
    finally:
        process.kill()
        process.wait(timeout=5)


def test_launch_captures_process_start_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launch path must capture process_start_time at spawn time.

    This test goes through the real launch path and asserts the resulting record's
    process_start_time is not None. Mutation gate: forcing spawn capture to None MUST fail it.
    """
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

    # The record must have a captured process_start_time
    assert record.process_start_time is not None, (
        "launch_devin_session must capture process_start_time at spawn time"
    )
    assert isinstance(record.process_start_time, float), (
        "process_start_time must be a float (Unix timestamp)"
    )


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
    # Regression guard: empty strings should not appear in the rendered command
    assert "" not in record.command


def test_command_template_injects_model_when_worker_model_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When worker_model is set, the rendered command must include --model <value>."""
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
        command_template=(sys.executable, str(script), "{model_args}", "{prompt_path}"),
        worker_model="claude-sonnet-4-5",
    )

    # The rendered command must include --model claude-sonnet-4-5 as separate tokens
    assert "--model" in record.command
    model_idx = record.command.index("--model")
    assert record.command[model_idx + 1] == "claude-sonnet-4-5"
    # Verify the custom template structure
    assert record.command[0] == sys.executable
    assert str(script) in record.command
    # Regression guard: empty strings should not appear in the rendered command
    assert "" not in record.command


def test_command_template_omits_model_when_worker_model_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When worker_model is empty, the rendered command must omit --model."""
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
        command_template=(sys.executable, str(script), "{model_args}", "{prompt_path}"),
        worker_model="",
    )

    # The rendered command must NOT include --model
    assert "--model" not in record.command
    # Verify the custom template structure
    assert record.command[0] == sys.executable
    assert str(script) in record.command
    # Regression guard: empty strings should not appear in the rendered command
    assert "" not in record.command


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
    """launch_devin_session must sanitize the environment before spawning the worker."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")

    # Script that writes VIRTUAL_ENV to stdout so we can verify it's sanitized
    env_script = tmp_path / "echo_env.py"
    env_script.write_text(
        "import os, sys\nsys.stdout.write(os.environ.get('VIRTUAL_ENV', 'UNSET') + '\\n')\nsys.stdout.flush()\n",
        encoding="utf-8",
    )

    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Set parent VIRTUAL_ENV (simulating orchestrator leak)
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")

    record = launch_devin_session(
        55,
        "agent/issue-55-test",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(env_script)),
    )

    assert record.error is None
    assert record.pid is not None

    # Give the subprocess a moment, then verify it didn't inherit VIRTUAL_ENV
    deadline = time.time() + 5
    while time.time() < deadline:
        log_text = Path(record.log_path).read_text(encoding="utf-8")
        if log_text.strip():
            break
        time.sleep(0.05)
    log_text = Path(record.log_path).read_text(encoding="utf-8").strip()

    assert log_text == "UNSET", (
        f"Worker inherited VIRTUAL_ENV={log_text!r}, expected UNSET (sanitized)"
    )


# ---------------------------------------------------------------------------
# Throttle death classification tests
# ---------------------------------------------------------------------------


def test_classify_session_failure_rate_limit_with_reset_time(tmp_path: Path) -> None:
    """Test that rate-limit errors with 'resets in N minutes' are classified correctly."""
    from charlie_work.devin_shell import _classify_session_failure
    from datetime import UTC, datetime, timedelta

    log_path = tmp_path / "session.log"
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
    # Verify the cooldown reflects the parsed 10 minutes
    throttle_time = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
    expected_time = datetime.now(UTC) + timedelta(minutes=10)
    # Allow 1 second tolerance for test execution time
    assert abs((throttle_time - expected_time).total_seconds()) < 1


def test_classify_session_failure_rate_limit_without_reset_time(tmp_path: Path) -> None:
    """Test that rate-limit errors without reset time use default cooldown."""
    from charlie_work.devin_shell import _classify_session_failure

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Some work done...\nError: Reached overall message rate limit. Please try again later.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    # Should use default 15 minute cooldown
    assert "T" in throttled_until
    assert "Z" in throttled_until


def test_classify_session_failure_quota_exhausted(tmp_path: Path) -> None:
    """Test that quota-exhaustion errors are classified correctly."""
    from charlie_work.devin_shell import _classify_session_failure

    log_path = tmp_path / "session.log"
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


def test_classify_session_failure_no_throttle(tmp_path: Path) -> None:
    """Test that non-throttle errors return None."""
    from charlie_work.devin_shell import _classify_session_failure

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Some work done...\nError: something went wrong with the task\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind is None
    assert throttled_until is None


def test_classify_session_failure_missing_log(tmp_path: Path) -> None:
    """Test that missing log files return None."""
    from charlie_work.devin_shell import _classify_session_failure

    log_path = tmp_path / "nonexistent.log"

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind is None
    assert throttled_until is None


def test_get_rate_limit_defer_until_with_reset_time(tmp_path: Path) -> None:
    """Test that get_rate_limit_defer_until returns a deadline offset by the parsed reset time plus slack."""
    from datetime import UTC, datetime, timedelta

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    now = datetime.now(UTC)
    defer_until = get_rate_limit_defer_until(log_path, slack_minutes=2, now=now)

    assert defer_until is not None
    assert "T" in defer_until
    assert "Z" in defer_until
    expected = now + timedelta(minutes=10 + 2)
    parsed = datetime.fromisoformat(defer_until.replace("Z", "+00:00"))
    assert abs((parsed - expected).total_seconds()) < 1


def test_get_rate_limit_defer_until_without_reset_time(tmp_path: Path) -> None:
    """Test that get_rate_limit_defer_until uses the default cooldown when no reset time is present."""
    from datetime import UTC, datetime, timedelta

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Please try again later.\n",
        encoding="utf-8",
    )

    now = datetime.now(UTC)
    defer_until = get_rate_limit_defer_until(log_path, slack_minutes=2, now=now)

    assert defer_until is not None
    expected = now + timedelta(minutes=15 + 2)
    parsed = datetime.fromisoformat(defer_until.replace("Z", "+00:00"))
    assert abs((parsed - expected).total_seconds()) < 1


def test_get_rate_limit_defer_until_no_match(tmp_path: Path) -> None:
    """Test that get_rate_limit_defer_until returns None for non-rate-limit logs."""
    log_path = tmp_path / "session.log"
    log_path.write_text("Working on task...\n", encoding="utf-8")

    assert get_rate_limit_defer_until(log_path, slack_minutes=2) is None


def test_update_session_record_with_failure_classification(tmp_path: Path) -> None:
    """Test that session records are updated with failure classification."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create a session sidecar
    sidecar_path = sessions_dir / "issue-42.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt/issue-42",
                "prompt_path": "p.md",
                "command": ["devin", "--print"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(sessions_dir / "issue-42.log"),
                "error": None,
            }
        ),
        encoding="utf-8",
    )

    # Create a log file with rate-limit error
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = update_session_record_with_failure_classification(
        sessions_dir, 42
    )

    assert failure_kind == "rate_limited"
    assert throttled_until is not None

    # Verify the sidecar was updated
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "rate_limited"


def test_update_session_record_skips_already_classified(tmp_path: Path) -> None:
    """Test that already-classified records are not re-classified."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create a session sidecar with existing classification
    sidecar_path = sessions_dir / "issue-42.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt/issue-42",
                "prompt_path": "p.md",
                "command": ["devin", "--print"],
                "pid": 1234,
                "started_at": "2026-01-01T00:00:00Z",
                "log_path": str(sessions_dir / "issue-42.log"),
                "error": None,
                "failure_kind": "rate_limited",  # Already classified
            }
        ),
        encoding="utf-8",
    )

    # Create a log file with a different error (should be ignored)
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Error: daily usage quota has been exhausted.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = update_session_record_with_failure_classification(
        sessions_dir, 42
    )

    # Should return the existing classification, not re-classify
    assert failure_kind == "rate_limited"
    assert throttled_until is None  # No new throttled_until

    # Verify the sidecar was not changed
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "rate_limited"


def _make_session_sidecar(sessions_dir: Path, issue_number: int, log_path: Path) -> Path:
    """Write a minimal session sidecar for failure-classification tests."""
    sidecar_path = sessions_dir / f"issue-{issue_number}.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "issue_number": issue_number,
                "branch": f"agent/issue-{issue_number}",
                "worktree_path": "/tmp/wt",
                "prompt_path": "p.md",
                "command": ["devin", "--print"],
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
    semantics, no throttled_until). The original PR #263 premise treated
    this string as a throttle signature; a correction comment on issue #260
    established it is a hard failure that must instead route through
    post_mortem.classify_and_record's worker_blocked log-tail fallback (see
    test_post_mortem.py), which composes with escalation, not cooldown."""
    from charlie_work.devin_shell import _classify_session_failure

    log_path = tmp_path / "session.log"
    log_path.write_text(
        "Error: A tool was rejected by the user.\n",
        encoding="utf-8",
    )

    failure_kind, throttled_until = _classify_session_failure(log_path)

    assert failure_kind is None
    assert throttled_until is None


def test_update_session_record_tool_rejected_is_not_rate_limited(tmp_path: Path) -> None:
    """Issue #260, corrected premise: a tool-rejected sidecar log must not be
    classified rate_limited by the adapter's own log-tail classifier — see
    test_classify_session_failure_tool_rejected_is_not_throttle."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Error: A tool was rejected by the user.\n",
        encoding="utf-8",
    )
    sidecar_path = _make_session_sidecar(sessions_dir, 42, log_path)

    failure_kind, throttled_until = update_session_record_with_failure_classification(
        sessions_dir, 42, fallback_kind="stalled"
    )

    assert failure_kind == "stalled"
    assert throttled_until is None
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "stalled"


def test_update_session_record_unknown_tail_falls_back_to_stalled(tmp_path: Path) -> None:
    """Unknown log tail should fall back to the provided fallback_kind."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Error: something completely unrelated went wrong\n",
        encoding="utf-8",
    )
    sidecar_path = _make_session_sidecar(sessions_dir, 42, log_path)

    failure_kind, throttled_until = update_session_record_with_failure_classification(
        sessions_dir, 42, fallback_kind="stalled"
    )

    assert failure_kind == "stalled"
    assert throttled_until is None
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "stalled"


def test_update_session_record_custom_throttle_markers(tmp_path: Path) -> None:
    """RuntimeConfig.throttle_error_markers is configurable without code changes."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Error: provider-specific frobnicate limit exceeded\n",
        encoding="utf-8",
    )
    sidecar_path = _make_session_sidecar(sessions_dir, 42, log_path)

    config = OrchestratorConfig(
        runtime=RuntimeConfig(throttle_error_markers=("frobnicate limit exceeded",))
    )
    failure_kind, throttled_until = update_session_record_with_failure_classification(
        sessions_dir, 42, config=config
    )

    assert failure_kind == "rate_limited"
    assert throttled_until is not None
    updated_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert updated_sidecar["failure_kind"] == "rate_limited"


def test_launch_render_error_returns_error_record_and_tears_down_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defense-in-depth: render errors past the load gate return error records, not exceptions."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")

    worktree_removed = []

    def tracking_remove_worktree(repo_root, worktree_path, *, force=False, branch=None):
        worktree_removed.append(worktree_path)

    monkeypatch.setattr(
        devin_shell,
        "create_worktree",
        lambda *args, **kwargs: _fake_worktree(tmp_path, "agent/issue-1"),
    )
    monkeypatch.setattr(devin_shell, "remove_worktree", tracking_remove_worktree)

    # Template with an unknown placeholder that bypasses load validation
    record = launch_devin_session(
        1,
        "agent/issue-1",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("echo", "{unknown_placeholder}"),
    )

    # Must return an error record, not raise
    assert record.error is not None
    assert "command template rendering failed" in record.error
    assert record.pid is None

    # Worktree must have been torn down
    assert len(worktree_removed) == 1

    # Sidecar must record the error
    sidecar_path = sessions_dir / "issue-1.json"
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["error"] is not None
    assert payload["pid"] is None


def test_launch_failure_then_retry_succeeds(tmp_path: Path) -> None:
    """Launch failure should clean up branch and worktree, allowing retry to succeed."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt text\n", encoding="utf-8")

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
    record1 = launch_devin_session(
        42,
        branch_name,
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
    )

    assert record1.error is not None
    assert "failed to launch devin" in record1.error

    # Verify the branch is deleted after the failure
    result = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert branch_name not in result.stdout

    # Second launch should succeed (using fake devin script)
    script = _write_fake_devin(tmp_path, _FAKE_DEVIN_SLEEP)
    record2 = launch_devin_session(
        42,
        branch_name,
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script), "{prompt_path}"),
    )

    assert record2.error is None
    assert record2.branch == branch_name


def test_rework_launch_failure_preserves_branch(tmp_path: Path) -> None:
    """Rework-mode launch failure should preserve the existing branch."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt text\n", encoding="utf-8")

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

    branch_name = "agent/issue-44-rework"

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
    record = launch_devin_session(
        44,
        branch_name,
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=("this-binary-does-not-exist-xyz",),
        rework=True,
    )

    assert record.error is not None
    assert "failed to launch devin" in record.error

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
# Tests for venv_source and worker_env parity with claude-code
# ---------------------------------------------------------------------------


def test_launch_devin_session_passes_venv_source_to_create_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """venv_source should be passed through to create_worktree."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")

    worktree_calls: list[dict] = []
    _install_fake_create_worktree(monkeypatch, tmp_path, calls=worktree_calls)

    venv_source = tmp_path / "shared-venv"
    venv_source.mkdir()

    # Hermetic: use sys.executable instead of real devin binary
    launch_devin_session(
        123,
        "agent/issue-123-venv",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        venv_source=venv_source,
        command_template=(sys.executable, "-c", "pass"),
    )

    assert len(worktree_calls) == 1
    assert worktree_calls[0]["venv_source"] == venv_source


def test_devin_config_default_venv_source_is_none() -> None:
    """Issue #112: devin-shell must default to isolated per-worktree venvs."""
    assert DevinConfig().venv_source is None
    assert OrchestratorConfig().devin.venv_source is None


def test_devin_shell_reuse_unlinks_shared_venv_junction_and_isolates_imports(
    tmp_path: Path,
) -> None:
    """Issue #112: devin-shell reuse with venv_source=None must unlink a pre-existing
    .venv junction so a raw `python -c "import <pkg>"` resolves inside the worktree's
    own isolated venv, not the shared venv's stale .pth target.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    shared_venv = tmp_path / "shared-venv"
    shared_venv.mkdir()
    other_worktree = tmp_path / "other-worktree"
    other_worktree.mkdir()
    other_src = other_worktree / "src"
    other_src.mkdir(parents=True)
    (other_src / "job_finder.py").write_text("# other\n", encoding="utf-8")

    # Pre-seed the shared venv with an editable .pth pointing at another worktree,
    # simulating the last `uv sync` having been run from worktree A.
    pth = shared_venv / "Lib" / "site-packages" / "_editable_impl_job_finder.pth"
    pth.parent.mkdir(parents=True)
    pth.write_text(str(other_src) + "\n", encoding="utf-8")

    branch_name = "agent/issue-112-isolated"

    # Pre-PR default: create the worktree with a shared-venv junction.
    info1 = create_worktree(
        repo_root,
        branch_name,
        base_ref="HEAD",
        venv_source=shared_venv,
    )
    assert info1.venv_junction == info1.path / ".venv"
    assert is_junction(info1.path / ".venv")

    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")

    # Fake devin CLI: create a local venv in the worktree, install an editable .pth
    # pointing at the worktree's own src, and run `python -c "import job_finder"`.
    probe_script = tmp_path / "probe_devin.py"
    probe_script_content = "\n".join(
        [
            "import os",
            "import stat",
            "import subprocess",
            "import sys",
            "from pathlib import Path",
            "",
            "worktree = Path(os.getcwd())",
            "venv = worktree / '.venv'",
            "",
            "# If the junction was not unlinked, the venv would be created in the shared",
            "# venv target and poison every other worktree. Fail loudly in that case.",
            "if os.name == 'nt':",
            "    if venv.exists() and (venv.stat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):",
            "        raise SystemExit('BUG: .venv is still a junction')",
            "else:",
            "    if venv.is_symlink():",
            "        raise SystemExit('BUG: .venv is still a symlink')",
            "",
            "subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(venv)], check=True)",
            "",
            "py_version = f'{sys.version_info.major}.{sys.version_info.minor}'",
            "if os.name == 'nt':",
            "    site_packages = venv / 'Lib/site-packages'",
            "else:",
            "    site_packages = venv / f'lib/python{py_version}/site-packages'",
            "site_packages.mkdir(parents=True, exist_ok=True)",
            "",
            "src = worktree / 'src'",
            "src.mkdir(exist_ok=True)",
            "(src / 'job_finder.py').write_text('__file__ = __file__\\n', encoding='utf-8')",
            "pth = site_packages / '_editable_impl_job_finder.pth'",
            "pth.write_text(str(src) + '\\n', encoding='utf-8')",
            "",
            "py = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')",
            "cmd = 'import job_finder, inspect; print(inspect.getfile(job_finder))'",
            "result = subprocess.run([str(py), '-c', cmd], capture_output=True, text=True)",
            "sys.stdout.write(result.stdout)",
            "sys.stderr.write(result.stderr)",
            "raise SystemExit(result.returncode)",
        ]
    )
    probe_script.write_text(probe_script_content, encoding="utf-8")

    # Re-dispatch with the new default (venv_source=None) via the devin-shell code path.
    record = launch_devin_session(
        112,
        branch_name,
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        venv_source=None,
        command_template=(sys.executable, str(probe_script)),
        rework=True,
    )

    assert record.error is None, record.error
    assert record.pid is not None
    assert record.worktree_path == str(info1.path)

    # Wait for the probe to finish and assert the import resolved inside the worktree.
    deadline = time.time() + 30
    log_path = Path(record.log_path)
    while time.time() < deadline:
        log_text = log_path.read_text(encoding="utf-8")
        if log_text.strip() and not is_session_alive(record):
            break
        time.sleep(0.05)

    log_text = log_path.read_text(encoding="utf-8").strip()
    assert log_text, "probe produced no output"
    assert not log_text.startswith(str(other_src)), (
        f"import resolved from the shared-venv target: {log_text!r}"
    )
    assert Path(log_text).resolve().is_relative_to(Path(info1.path).resolve()), (
        f"import did not resolve inside the worktree: {log_text!r}"
    )

    # The shared venv's stale .pth must remain untouched and still point to A.
    assert pth.read_text(encoding="utf-8").strip() == str(other_src)

    # Cleanup
    remove_worktree(repo_root, info1.path, force=True, branch=branch_name)


def test_launch_devin_session_injects_worker_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """worker_env should be merged into the process environment."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")

    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Script that writes an env var to a file
    env_script = tmp_path / "env_probe.py"
    env_script.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "Path('env-probe.txt').write_text(\n"
        "    os.environ.get('TEST_VAR', '<unset>')\n"
        ")\n",
        encoding="utf-8",
    )

    record = launch_devin_session(
        99,
        "agent/issue-99-env",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(env_script)),
        worker_env={"TEST_VAR": "test-value"},
    )

    assert record.error is None
    assert record.pid is not None

    # Wait for the subprocess to complete
    deadline = time.time() + 10
    probe_path = Path(record.worktree_path) / "env-probe.txt"
    while not probe_path.exists() and time.time() < deadline:
        time.sleep(0.05)

    assert probe_path.exists()
    assert probe_path.read_text(encoding="utf-8") == "test-value"


def test_launch_devin_session_passes_materialize_dirs_to_create_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """materialize_dirs should be passed through to create_worktree."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")

    worktree_calls: list[dict] = []
    _install_fake_create_worktree(monkeypatch, tmp_path, calls=worktree_calls)

    # Hermetic: use sys.executable instead of real devin binary
    launch_devin_session(
        456,
        "agent/issue-456-materialize",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        materialize_dirs=(".devin", ".config"),
        command_template=(sys.executable, "-c", "pass"),
    )

    assert len(worktree_calls) == 1
    assert worktree_calls[0]["materialize_dirs"] == (".devin", ".config")


def test_launch_devin_session_includes_start_new_session_on_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """launch_devin_session should include start_new_session=True on POSIX systems."""
    from unittest.mock import patch

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("x", encoding="utf-8")

    _install_fake_create_worktree(monkeypatch, tmp_path)

    # Capture the kwargs passed to subprocess.Popen
    popen_kwargs: dict = {}
    original_popen = subprocess.Popen

    def capture_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        # Return a fake process that exits immediately
        return original_popen([sys.executable, "-c", "pass"], **kwargs)

    with patch("subprocess.Popen", side_effect=capture_popen):
        launch_devin_session(
            789,
            "agent/issue-789-start-new-session",
            prompt_path,
            repo_root=repo_root,
            sessions_dir=sessions_dir,
            command_template=(sys.executable, "-c", "pass"),
        )

    # Detachment is enforced by no_console_window_kwargs + CREATE_NEW_PROCESS_GROUP
    # directly; Policy A survival flags (DETACHED_PROCESS, CREATE_BREAKAWAY_FROM_JOB)
    # are out of scope for issue #360.
    if os.name != "nt":
        assert popen_kwargs.get("start_new_session") is True
        assert "creationflags" not in popen_kwargs
    else:
        assert "start_new_session" not in popen_kwargs
        flags = popen_kwargs.get("creationflags", 0)
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
        assert flags & subprocess.CREATE_NO_WINDOW
        assert not (flags & subprocess.DETACHED_PROCESS)
        assert not (flags & subprocess.CREATE_BREAKAWAY_FROM_JOB)


# ---------------------------------------------------------------------------
# Tests for log-stat enrichment fields (issue #160)
# ---------------------------------------------------------------------------


def test_session_record_log_stat_fields_roundtrip(tmp_path: Path) -> None:
    """SessionRecord with last_activity_at and log_bytes fields round-trips through to_dict/from_dict."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    original = SessionRecord(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path="/tmp/worktree-1",
        prompt_path="/tmp/prompt-1.md",
        command=("devin", "prompt.md"),
        pid=12345,
        started_at="2026-07-06T00:00:00Z",
        log_path="/tmp/issue-1.log",
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        last_activity_at="2026-07-06T01:30:45Z",
        log_bytes=2048,
    )

    payload = original.to_dict()
    assert "last_activity_at" in payload
    assert "log_bytes" in payload
    assert payload["last_activity_at"] == "2026-07-06T01:30:45Z"
    assert payload["log_bytes"] == 2048

    reconstructed = SessionRecord.from_dict(payload)
    assert reconstructed.last_activity_at == "2026-07-06T01:30:45Z"
    assert reconstructed.log_bytes == 2048


def test_session_record_from_dict_missing_log_stat_fields(tmp_path: Path) -> None:
    """A payload missing last_activity_at and log_bytes still constructs with None defaults."""
    payload = {
        "issue_number": 1,
        "branch": "agent/issue-1",
        "worktree_path": "/tmp/worktree-1",
        "prompt_path": "/tmp/prompt-1.md",
        "command": ["devin", "prompt.md"],
        "pid": 12345,
        "started_at": "2026-07-06T00:00:00Z",
        "log_path": "/tmp/issue-1.log",
        # error, failure_kind, process_start_time, reclaimed, last_activity_at, log_bytes omitted
    }

    record = SessionRecord.from_dict(payload)
    assert record.last_activity_at is None
    assert record.log_bytes is None


def test_session_record_log_stat_fields_persist_to_sidecar(tmp_path: Path) -> None:
    """SessionRecord with log stat fields can be written to and read from a sidecar file."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42",
        worktree_path="/tmp/worktree-42",
        prompt_path="/tmp/prompt-42.md",
        command=("devin", "prompt.md"),
        pid=54321,
        started_at="2026-07-06T00:00:00Z",
        log_path="/tmp/issue-42.log",
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        last_activity_at="2026-07-06T02:15:30Z",
        log_bytes=4096,
    )

    sidecar_path = _sidecar_path(sessions_dir, 42)
    _write_json(sidecar_path, record.to_dict())

    # Read back through read_session_records
    records = read_session_records(sessions_dir)
    assert len(records) == 1
    restored = records[0]
    assert restored.last_activity_at == "2026-07-06T02:15:30Z"
    assert restored.log_bytes == 4096
