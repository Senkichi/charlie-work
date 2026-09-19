"""Launch-path tests for ``devin_shell.launch_devin_session``.

Split out of ``tests/test_devin_shell.py`` (issue #1542, Track-1 pilot):
worktree cwd, env merging and caps, sidecar writes, the rework flag,
process start-time capture, the shared-venv junction reuse default, and
the venv_source / worker_env / materialize_dirs / start_new_session
parity surface with claude-code. The launch failure/retry error-record
paths live in ``tests/test_devin_shell_launch_failures.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from _devin_shell_fixtures import (
    _FAKE_DEVIN_SLEEP,
    _init_repo,
    _install_fake_create_worktree,
    _write_fake_devin,
)
from _worker_marker_wait import read_worker_marker

from charlie_work import devin_shell
from charlie_work.config import DevinConfig, OrchestratorConfig
from charlie_work.devin_shell import (
    is_session_alive,
    launch_devin_session,
)
from charlie_work.worktree import (
    create_worktree,
    is_junction,
    remove_worktree,
)


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


def test_launch_devin_session_records_default_xdist_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #646: with no ambient/config cap, the sidecar records the safe default."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("do the thing", encoding="utf-8")
    _install_fake_create_worktree(monkeypatch, tmp_path)
    monkeypatch.delenv("PYTEST_XDIST_AUTO_NUM_WORKERS", raising=False)
    monkeypatch.delenv("UV_NO_SYNC", raising=False)

    exit_script = _write_fake_devin(tmp_path, "import sys\nsys.exit(0)\n")
    record = launch_devin_session(
        141,
        "agent/issue-141-default-cap",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(exit_script)),
    )

    assert record.error is None
    assert record.xdist_cap == "2"
    assert record.uv_no_sync is None  # no .venv in this fake worktree

    payload = json.loads((sessions_dir / "issue-141.json").read_text(encoding="utf-8"))
    assert payload["xdist_cap"] == "2"
    assert payload["uv_no_sync"] is None


def test_launch_devin_session_worker_env_pytest_cap_override_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #646: an explicit worker_env cap must win over the ambient env and default."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("do the thing", encoding="utf-8")
    _install_fake_create_worktree(monkeypatch, tmp_path, with_venv=True)
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "6")

    exit_script = _write_fake_devin(tmp_path, "import sys\nsys.exit(0)\n")
    record = launch_devin_session(
        142,
        "agent/issue-142-cap-override",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(exit_script)),
        worker_env={"PYTEST_XDIST_AUTO_NUM_WORKERS": "1", "UV_NO_SYNC": "0"},
    )

    assert record.error is None
    assert record.xdist_cap == "1"
    assert record.uv_no_sync == "0"


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

    # Wait for the subprocess to finish WRITING, not merely to create the file.
    probe_path = Path(record.worktree_path) / "env-probe.txt"
    read_worker_marker(probe_path, expected="test-value")


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

    # Worker spawns use hidden_console_kwargs: CREATE_NEW_CONSOLE plus a
    # STARTUPINFO with wShowWindow=SW_HIDE so descendants inherit a hidden
    # console. Policy A survival flags (DETACHED_PROCESS, CREATE_BREAKAWAY_FROM_JOB)
    # are out of scope for issue #360.
    if os.name != "nt":
        assert popen_kwargs.get("start_new_session") is True
        assert "creationflags" not in popen_kwargs
        assert "startupinfo" not in popen_kwargs
    else:
        assert "start_new_session" not in popen_kwargs
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
