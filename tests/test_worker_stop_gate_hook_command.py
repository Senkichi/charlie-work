"""Hook-command contract for the tracked ``.claude/settings.json`` Stop hook.

Split out of ``test_worker_stop_gate.py`` (already over the per-module line
cap; the file-size ratchet, issue #1442, never lets an over-cap file grow).
These tests do not exercise the gate module's logic -- they pin the *shape*
of the command line that launches it, and launch it for real once.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "worker_stop_gate.py"


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A real, throwaway git repo with one committed file -- the gate is
    pointed at it via the payload's ``cwd`` so it takes the no-diff fast path."""
    root = tmp_path / "repo"
    root.mkdir()
    _run_git(["init"], cwd=root)
    _run_git(["config", "user.email", "test@example.com"], cwd=root)
    _run_git(["config", "user.name", "Test"], cwd=root)
    (root / "README.md").write_text("placeholder\n", encoding="utf-8")
    _run_git(["add", "README.md"], cwd=root)
    _run_git(["commit", "-m", "init"], cwd=root)
    return root


# ---------------------------------------------------------------------------
# cw<->jc hook-parity check (cw#1259; this is the origin repo for this gate --
# a downstream port carries a back-adapted copy of this same check):
# this repo's own tracked .claude/settings.json must declare a Stop hook
# that points at the module under test. Derived entirely from the JSON
# itself -- no hardcoded script name -- and self-contained: it never reads
# a peer repository's checkout (a same-commit-opposite-verdict failure
# class in this environment; see cw#1259's recon note 4 and the in-repo
# memory on cross-tree test premises).
#
# The command is modelled the way the harness actually runs it: Claude Code
# hands the string to bash with ``CLAUDE_PROJECT_DIR`` exported (the project
# root, forward-slash form on Windows) and the *session's current directory*
# as cwd. That cwd drifts whenever a compound command ``cd``s into a worktree,
# so a cwd-relative token like ``.venv/Scripts/python.exe`` resolves against
# the wrong tree and the hook dies with "No such file or directory" on every
# subsequent turn (2026-09-03, observed in a downstream repo that ports this
# gate). The invariant these tests pin: every path token in the command
# is anchored to ``$CLAUDE_PROJECT_DIR`` -- never cwd-relative, and never a
# machine-specific absolute path baked into a tracked file.
# ---------------------------------------------------------------------------

_SETTINGS_PATH = _SCRIPT_PATH.parent.parent / ".claude" / "settings.json"
_PROJECT_DIR_REF = re.compile(r"^\$(\{CLAUDE_PROJECT_DIR\}|CLAUDE_PROJECT_DIR)/")


def _stop_hook_commands() -> list[str]:
    settings = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    commands = [
        hook["command"]
        for entry in settings["hooks"]["Stop"]
        for hook in entry["hooks"]
        if hook.get("type") == "command"
    ]
    assert commands, "no Stop hook command entries found in .claude/settings.json"
    return commands


def _hook_argv(command: str, project_dir: Path) -> list[str]:
    """Expand the command the way bash will: ``$CLAUDE_PROJECT_DIR`` substituted,
    then shell-split so quoted tokens survive spaces in the path."""
    expanded = command.replace("${CLAUDE_PROJECT_DIR}", project_dir.as_posix()).replace(
        "$CLAUDE_PROJECT_DIR", project_dir.as_posix()
    )
    return shlex.split(expanded, posix=True)


def test_settings_json_stop_hook_references_this_script():
    repo_root = _SETTINGS_PATH.parent.parent
    referenced_scripts = [
        Path(token).resolve()
        for command in _stop_hook_commands()
        for token in _hook_argv(command, repo_root)
        if token.endswith(_SCRIPT_PATH.name)
    ]
    assert referenced_scripts, (
        f"no {_SCRIPT_PATH.name} reference found in Stop hook commands: {_stop_hook_commands()}"
    )
    for script_path in referenced_scripts:
        assert script_path == _SCRIPT_PATH.resolve(), (
            f"Stop hook references {script_path}, not the module under test {_SCRIPT_PATH}"
        )
        assert script_path.exists(), f"Stop hook references {script_path}, which does not exist"


def test_settings_json_stop_hook_paths_are_anchored_to_project_dir():
    """Every path token must derive from ``$CLAUDE_PROJECT_DIR``: a relative
    token breaks the moment the session cwd drifts; a literal absolute path
    would pin a tracked file to one machine."""
    for command in _stop_hook_commands():
        raw_tokens = shlex.split(command, posix=True)
        path_tokens = [tok for tok in raw_tokens if "/" in tok or "\\" in tok]
        assert path_tokens, f"Stop hook command has no path tokens at all: {command!r}"
        for tok in path_tokens:
            assert _PROJECT_DIR_REF.match(tok), (
                f"Stop hook path token {tok!r} is not anchored to $CLAUDE_PROJECT_DIR "
                f"(cwd-relative or machine-specific) in command {command!r}"
            )
        # And once expanded, every path token is absolute -- the property the
        # anchoring exists to guarantee.
        for tok in _hook_argv(command, _SETTINGS_PATH.parent.parent):
            if "/" in tok:
                assert Path(tok).is_absolute(), f"expanded token {tok!r} is still relative"


def _bash_or_skip() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not on PATH; Claude Code runs hooks under bash")
    return bash


def _run_hook_from(cwd: Path, command: str, project_dir: Path, payload: dict, tmp: Path):
    """Run ``command`` exactly as the harness would -- bash, stdin JSON payload --
    but from ``cwd`` rather than the project dir, with the gate's own temp
    state redirected under ``tmp`` so the real per-session counters are
    untouched."""
    env = {
        **os.environ,
        "CLAUDE_PROJECT_DIR": project_dir.as_posix(),
        "TMPDIR": str(tmp),
        "TEMP": str(tmp),
        "TMP": str(tmp),
    }
    return subprocess.run(
        [_bash_or_skip(), "-c", command],
        cwd=cwd,
        env=env,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_stop_hook_command_survives_cwd_drift(repo, tmp_path):
    """L4: the tracked command launches from a cwd that is *not* the project
    dir and has no ``.venv`` of its own -- the exact shape of the drifted
    worktree failure. The gate is pointed (via the payload's ``cwd``) at a
    clean throwaway repo so it takes the no-diff fast path; the assertion is
    about launching, not about gate verdicts.

    The pre-fix relative form is run from the same cwd as a positive control:
    if it did not fail there, this test could not distinguish the two."""
    project_dir = _SETTINGS_PATH.parent.parent
    commands = _stop_hook_commands()
    for command in commands:
        for tok in _hook_argv(command, project_dir):
            if "/" in tok and not Path(tok).exists():
                pytest.skip(f"hook path {tok} does not exist in this environment (no venv?)")
    drifted_cwd = tmp_path / "drifted-worktree"
    drifted_cwd.mkdir()
    payload = {"session_id": "cwd-drift-test", "cwd": str(repo)}

    relative_form = " ".join(
        Path(tok).relative_to(project_dir).as_posix() if "/" in tok else tok
        for tok in _hook_argv(commands[0], project_dir)
    )
    control = _run_hook_from(drifted_cwd, relative_form, project_dir, payload, tmp_path)
    assert control.returncode == 127, (
        "positive control failed: the cwd-relative form should not launch from a "
        f"foreign cwd, got exit {control.returncode}: {control.stderr!r}"
    )

    for command in commands:
        proc = _run_hook_from(drifted_cwd, command, project_dir, payload, tmp_path)
        assert "No such file or directory" not in proc.stderr, proc.stderr
        assert proc.returncode == 0, (
            f"anchored command failed from drifted cwd (exit {proc.returncode}): "
            f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
        )
