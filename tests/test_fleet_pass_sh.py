"""``scripts/fleet-pass.sh``: the POSIX launcher has ``fleet-pass.ps1``'s semantics.

The script is copied into a temporary repo root (it derives its root from its own
location), and ``uv`` is a shim on PATH, so no test touches this checkout's log or
starts a real supervisor. Bash is Git Bash on Windows, never the WSL launcher.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from _ci_step_runner import _bash, _posix
from charlie_work import layout

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "fleet-pass.sh"

FAKE_UV = """#!/usr/bin/env bash
printf '%s\\n' "$*" > "$FAKE_UV_ARGS"
printf '%s\\n' "$PYTHONIOENCODING $PYTHONUTF8 $PYTHONUNBUFFERED" > "$FAKE_UV_ENV"
exit "${FAKE_UV_RC:-0}"
"""


def _run(tmp_path: Path, env: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], str]:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copyfile(SCRIPT, root / "scripts" / "fleet-pass.sh")
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "uv").write_text(FAKE_UV, encoding="utf-8", newline="\n")
    (shim / "uv").chmod(0o755)
    base = {
        k: v
        for k, v in os.environ.items()
        if k not in ("CHARLIE_WORK_FLEET_DIR", "XDG_STATE_HOME")
    }
    full_env = {
        **base,
        "HOME": _posix(tmp_path / "home"),
        "FAKE_UV_ARGS": _posix(tmp_path / "uv.args"),
        "FAKE_UV_ENV": _posix(tmp_path / "uv.env"),
        **env,
    }
    proc = subprocess.run(  # noqa: S603 - fixed argv
        [
            _bash(),
            "-c",
            f'PATH="{_posix(shim)}:$PATH" exec bash "{_posix(root / "scripts" / "fleet-pass.sh")}"',
        ],
        env=full_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )

    log = root / ".var" / "charlie-work" / "logs" / "fleet-pass.log"
    return proc, log.read_text(encoding="utf-8") if log.exists() else ""


def test_script_pins_the_layout_pause_filename() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    assert f'/{layout.FLEET_PAUSE_FILENAME}"' in script


def test_refuses_to_start_while_paused_via_fleet_dir_env(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    fleet.mkdir()
    (fleet / layout.FLEET_PAUSE_FILENAME).write_text("{}", encoding="utf-8")
    proc, log = _run(tmp_path, {"CHARLIE_WORK_FLEET_DIR": _posix(fleet)})
    assert proc.returncode == 0, proc.stderr
    assert "NOT started: fleet paused" in log
    assert "fleet supervise-loop start" not in log
    assert not (tmp_path / "uv.args").exists()


def test_default_fleet_dir_is_xdg_state_then_home(tmp_path: Path) -> None:
    flag = tmp_path / "home" / ".local" / "state" / "charlie-work" / layout.FLEET_PAUSE_FILENAME
    flag.parent.mkdir(parents=True)
    flag.write_text("{}", encoding="utf-8")
    proc, log = _run(tmp_path, {})
    assert proc.returncode == 0, proc.stderr
    assert "NOT started: fleet paused" in log


def test_xdg_state_home_wins_over_home(tmp_path: Path) -> None:
    flag = tmp_path / "xdg" / "charlie-work" / layout.FLEET_PAUSE_FILENAME
    flag.parent.mkdir(parents=True)
    flag.write_text("{}", encoding="utf-8")
    proc, log = _run(tmp_path, {"XDG_STATE_HOME": _posix(tmp_path / "xdg")})
    assert proc.returncode == 0, proc.stderr
    assert "NOT started: fleet paused" in log


def test_unpaused_launches_supervise_loop_and_brackets_the_run(tmp_path: Path) -> None:
    proc, log = _run(
        tmp_path, {"CHARLIE_WORK_FLEET_DIR": _posix(tmp_path / "fleet"), "FAKE_UV_RC": "7"}
    )
    assert proc.returncode == 0, proc.stderr
    args = (tmp_path / "uv.args").read_text(encoding="utf-8")
    assert args.startswith("run --no-sync --project ")
    assert args.rstrip().endswith("python -m charlie_work fleet supervise-loop -- --max-runtime 0")
    assert (tmp_path / "uv.env").read_text(encoding="utf-8").split() == [
        "utf-8:surrogateescape",
        "1",
        "1",
    ]
    start = log.index("--- fleet supervise-loop start ")
    assert log.index("--- fleet supervise-loop exit=7 ") > start
