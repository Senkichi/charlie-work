"""Launch contracts for ``scripts/hook-python.sh`` and ``.claude/hooks/session-start.sh``.

The launcher is what every tracked ``.claude/settings.json`` hook now goes
through: it picks ``.venv/Scripts/python.exe`` (Windows host) before
``.venv/bin/python`` (Linux/macOS, Claude Code on the web) and fails OPEN
with a stderr note when neither exists. The SessionStart hook runs
``uv sync --all-extras`` only when ``CLAUDE_CODE_REMOTE=true``; that gate is
the one thing keeping it away from the live orchestrator venv on the host.

Everything runs under bash, the way the harness runs hook commands. The
PreToolUse cwd-drift tests mirror ``test_stop_hook_command_survives_cwd_drift``
in ``test_worker_stop_gate_hook_command.py``.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_LAUNCHER = REPO_ROOT / "scripts" / "hook-python.sh"
_SESSION_START = REPO_ROOT / ".claude" / "hooks" / "session-start.sh"
_SETTINGS_PATH = REPO_ROOT / ".claude" / "settings.json"
_SKIP_NOTE = "hook skipped (fail-open)"
_VENV_INTERPRETERS = (
    REPO_ROOT / ".venv" / "Scripts" / "python.exe",
    REPO_ROOT / ".venv" / "bin" / "python",
)


def _bash_or_skip() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH; Claude Code runs hooks under bash")
    return bash


def _require_repo_venv() -> None:
    if not any(p.exists() for p in _VENV_INTERPRETERS):
        pytest.skip("no .venv interpreter in this checkout; nothing for the launcher to find")


def _run_bash(
    script: str, *, cwd: Path, env: dict[str, str], stdin: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_bash_or_skip(), "-c", script],
        cwd=cwd,
        env=env,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _copy_launcher(root: Path) -> Path:
    """Place a copy of the launcher at ``root/scripts/`` so it resolves ``root/.venv``."""
    dest = root / "scripts" / _LAUNCHER.name
    dest.parent.mkdir(parents=True)
    shutil.copyfile(_LAUNCHER, dest)
    return dest


def _fake_interpreter(path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/usr/bin/env bash\necho "{label} $*"\n', encoding="utf-8", newline="\n")
    path.chmod(0o755)


# ---------------------------------------------------------------------------
# scripts/hook-python.sh
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fake interpreters are shell scripts; Windows can't exec one named python.exe "
    "(the real-venv launch test below covers the Windows path)",
)
@pytest.mark.parametrize(
    ("present", "expected"),
    [
        (("Scripts/python.exe", "bin/python"), "Scripts/python.exe"),
        (("bin/python",), "bin/python"),
        (("Scripts/python.exe",), "Scripts/python.exe"),
    ],
)
def test_launcher_prefers_windows_layout_then_posix(tmp_path, present, expected):
    root = tmp_path / "project"
    launcher = _copy_launcher(root)
    for rel in present:
        _fake_interpreter(root / ".venv" / rel, rel)

    proc = _run_bash(
        f"bash {shlex.quote(launcher.as_posix())} -m some.module --flag",
        cwd=tmp_path,
        env=dict(os.environ),
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"{expected} -m some.module --flag"
    assert _SKIP_NOTE not in proc.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="relies on POSIX exec permission bits")
def test_launcher_skips_a_non_executable_candidate(tmp_path):
    root = tmp_path / "project"
    launcher = _copy_launcher(root)
    _fake_interpreter(root / ".venv" / "Scripts" / "python.exe", "Scripts/python.exe")
    (root / ".venv" / "Scripts" / "python.exe").chmod(0o644)
    _fake_interpreter(root / ".venv" / "bin" / "python", "bin/python")

    proc = _run_bash(
        f"bash {shlex.quote(launcher.as_posix())}", cwd=tmp_path, env=dict(os.environ)
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "bin/python"


def test_launcher_fails_open_without_a_venv(tmp_path):
    root = tmp_path / "project"
    launcher = _copy_launcher(root)

    proc = _run_bash(
        f"bash {shlex.quote(launcher.as_posix())} -c 'raise SystemExit(7)'",
        cwd=tmp_path,
        env=dict(os.environ),
    )

    assert proc.returncode == 0
    assert proc.stdout == ""
    assert _SKIP_NOTE in proc.stderr


def test_launcher_runs_this_checkouts_venv(tmp_path):
    """Real launch, on whatever platform the suite runs on: the interpreter the
    launcher execs is this checkout's venv, whichever layout it has."""
    _require_repo_venv()
    proc = _run_bash(
        f"bash {shlex.quote(_LAUNCHER.as_posix())} -c "
        "'import sys, pathlib; print(pathlib.Path(sys.prefix).resolve())'",
        cwd=tmp_path,
        env=dict(os.environ),
    )

    assert proc.returncode == 0, proc.stderr
    assert _SKIP_NOTE not in proc.stderr
    assert Path(proc.stdout.strip()) == (REPO_ROOT / ".venv").resolve()


# ---------------------------------------------------------------------------
# .claude/hooks/session-start.sh
# ---------------------------------------------------------------------------


def _run_session_start(
    tmp_path: Path, remote: str | None
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run a copy of the hook against a throwaway root with ``uv`` stubbed as a
    bash function (takes precedence over PATH on every platform), so no real
    sync ever runs. Returns the process and the stub's marker file."""
    root = tmp_path / "project"
    hook = root / ".claude" / "hooks" / _SESSION_START.name
    hook.parent.mkdir(parents=True)
    shutil.copyfile(_SESSION_START, hook)
    marker = root / "uv-invoked"

    env = {k: v for k, v in os.environ.items() if k != "CLAUDE_CODE_REMOTE"}
    if remote is not None:
        env["CLAUDE_CODE_REMOTE"] = remote
    script = (
        "uv() { printf '%s\\n' \"$*\" > uv-invoked; }; export -f uv; "
        f"exec bash {shlex.quote(hook.as_posix())}"
    )
    return _run_bash(script, cwd=tmp_path, env=env), marker


@pytest.mark.parametrize("remote", [None, "", "false", "1", "TRUE"])
def test_session_start_is_a_noop_outside_cloud_sessions(tmp_path, remote):
    proc, marker = _run_session_start(tmp_path, remote)

    assert proc.returncode == 0, proc.stderr
    assert not marker.exists(), "uv ran outside a cloud session -- the host venv guard is broken"


def test_session_start_syncs_all_extras_from_the_repo_root_in_cloud(tmp_path):
    proc, marker = _run_session_start(tmp_path, "true")

    assert proc.returncode == 0, proc.stderr
    # The stub writes relative to its cwd, so the marker's location proves the
    # hook cd'd to the repo root (two levels above .claude/hooks/).
    assert marker.exists(), "uv was not invoked with CLAUDE_CODE_REMOTE=true"
    assert marker.read_text(encoding="utf-8").strip() == "sync --all-extras"


def test_settings_json_session_start_runs_this_hook_anchored():
    settings = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    commands = [
        hook["command"]
        for entry in settings["hooks"]["SessionStart"]
        for hook in entry["hooks"]
        if hook.get("type") == "command"
    ]
    assert commands == ['bash "$CLAUDE_PROJECT_DIR/.claude/hooks/session-start.sh"']


# ---------------------------------------------------------------------------
# PreToolUse hooks launch from a drifted cwd
# ---------------------------------------------------------------------------


def _pre_tool_use_commands() -> list[tuple[str, str]]:
    settings = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    return [
        (entry["matcher"], hook["command"])
        for entry in settings["hooks"]["PreToolUse"]
        for hook in entry["hooks"]
        if hook.get("type") == "command"
    ]


def _strip_fail_open(command: str) -> str:
    """Drop the ``|| true`` crash net so a launch failure is visible, not masked."""
    stripped = command.strip()
    assert stripped.endswith("|| true"), command
    return stripped.removesuffix("|| true").strip()


@pytest.mark.parametrize(
    ("matcher", "command"),
    _pre_tool_use_commands(),
    ids=[f"{m}:{c.split()[-3]}" for m, c in _pre_tool_use_commands()],
)
def test_pre_tool_use_hook_launches_from_drifted_cwd(tmp_path, matcher, command):
    """A cwd that is not the project dir and has no ``.venv`` -- the drifted
    worktree shape. With the crash net removed the hook must still launch
    through the anchored launcher and allow a harmless call; the cwd-relative
    form is run first as a positive control that the drift is real."""
    _require_repo_venv()
    drifted_cwd = tmp_path / "drifted-worktree"
    drifted_cwd.mkdir()
    env = {**os.environ, "CLAUDE_PROJECT_DIR": REPO_ROOT.as_posix()}
    bare = _strip_fail_open(command)
    tool_name = "Bash" if matcher == "Bash" else matcher
    tool_input = {"command": "ls"} if matcher == "Bash" else {}
    payload = json.dumps(
        {"tool_name": tool_name, "tool_input": tool_input, "cwd": str(drifted_cwd)}
    )

    relative = bare.replace('"$CLAUDE_PROJECT_DIR/', '"')
    control = _run_bash(relative, cwd=drifted_cwd, env=env, stdin=payload)
    assert control.returncode == 127, (
        f"positive control failed: cwd-relative form launched from a foreign cwd "
        f"(exit {control.returncode}): {control.stderr!r}"
    )

    proc = _run_bash(bare, cwd=drifted_cwd, env=env, stdin=payload)
    assert "No such file or directory" not in proc.stderr, proc.stderr
    assert _SKIP_NOTE not in proc.stderr, "launcher found no interpreter from a drifted cwd"
    assert proc.returncode == 0, f"exit {proc.returncode}: {proc.stderr!r}"
    assert '"deny"' not in proc.stdout, proc.stdout
