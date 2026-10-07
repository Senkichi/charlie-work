"""Run a ``ci.yml`` step's bash body locally with a fake ``gh`` (tests only).

GitHub Actions YAML cannot be unit-tested by running it, so the structural
tests extract a step's ``run:`` body from the parsed workflow and execute it
with bash: Git Bash on Windows (``C:\\Program Files\\Git\\bin\\bash.exe``, the
same shell ``shell: bash`` uses on windows-latest), else ``bash`` on PATH.
The System32 ``bash.exe`` (WSL launcher) is never used. No bash -> the test
fails loudly rather than skipping.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CI_YML = REPO / ".github" / "workflows" / "ci.yml"
_GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")

FAKE_GH = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_GH_LOG"
case "$2" in
  */actions/artifacts*) [ "${FAKE_ARTIFACTS_RC:-0}" = 0 ] || exit "$FAKE_ARTIFACTS_RC"; printf '%s' "${FAKE_ARTIFACTS_OUT:-}";;
  */commits/*/pulls) [ "${FAKE_PULLS_RC:-0}" = 0 ] || exit "$FAKE_PULLS_RC"; printf '%s' "${FAKE_PULLS_OUT:-}";;
  */pulls/*) [ "${FAKE_PR_RC:-0}" = 0 ] || exit "$FAKE_PR_RC"; printf '%s' "${FAKE_PR_OUT:-}";;
  *) exit 99;;
esac
"""


@dataclass(frozen=True)
class StepRun:
    returncode: int
    outputs: dict[str, str]
    gh_calls: list[str]
    stdout: str
    stderr: str


def load_workflow(path: Path = CI_YML) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def find_step(workflow: dict, job_id: str, name: str) -> dict:
    for step in workflow["jobs"][job_id]["steps"]:
        if step.get("name") == name:
            return step
    raise AssertionError(f"job {job_id!r} has no step named {name!r}")


def _bash() -> str:
    if os.name == "nt" and _GIT_BASH.exists():
        return str(_GIT_BASH)
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    pytest.fail("bash not found: install Git for Windows (Git Bash) or put bash on PATH")


def _posix(path: Path) -> str:
    text = str(path.resolve())
    if os.name == "nt" and len(text) > 1 and text[1] == ":":
        return "/" + text[0].lower() + text[2:].replace("\\", "/")
    return text


def run_step(step: dict, tmp_path: Path, env: dict[str, str]) -> StepRun:
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir(exist_ok=True)
    gh = shim_dir / "gh"
    gh.write_text(FAKE_GH, encoding="utf-8", newline="\n")
    gh.chmod(0o755)
    output = tmp_path / "github_output"
    output.write_text("", encoding="utf-8")
    log = tmp_path / "gh.log"
    log.write_text("", encoding="utf-8")
    script = tmp_path / "step.sh"
    script.write_text(
        f'export PATH="{_posix(shim_dir)}:$PATH"\n{step["run"]}',
        encoding="utf-8",
        newline="\n",
    )
    step_env = {
        str(k): ("" if "${{" in str(v) else str(v)) for k, v in (step.get("env") or {}).items()
    }
    full_env = {
        **os.environ,
        **step_env,
        **env,
        "GITHUB_OUTPUT": _posix(output),
        "FAKE_GH_LOG": _posix(log),
    }
    proc = subprocess.run(
        [_bash(), _posix(script)],
        env=full_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    outputs: dict[str, str] = {}
    for line in output.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            outputs[key] = value
    calls = [line for line in log.read_text(encoding="utf-8").splitlines() if line]
    return StepRun(proc.returncode, outputs, calls, proc.stdout, proc.stderr)
