"""Structural guards on the hidden VBS launchers under ``scripts/`` (issue #2753).

``wscript.exe`` is a GUI-subsystem host: Task Scheduler's ``LastTaskResult`` is
whatever the script passes to ``WScript.Quit``. A launcher that calls
``shell.Run(..., 0, True)`` but never quits with the returned code reports
success on every tick, even when the wrapped ``.ps1`` failed to start.

The guards scan every ``.vbs`` under ``scripts/`` (by glob, never a fixed list)
so a new launcher cannot reintroduce the discard silently.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

_RUN_CALL = re.compile(r"(?P<object>[A-Za-z_]\w*)\s*\.\s*Run\b", re.IGNORECASE)
_QUIT_CALL = re.compile(r"WScript\.Quit\s*\(?\s*(?P<operand>[A-Za-z_]\w*)\b", re.IGNORECASE)
_ASSIGNMENT = re.compile(r"(?P<name>[A-Za-z_]\w*)\s*=")


def _strip_comment(line: str) -> str:
    """Drop a VBS apostrophe comment, leaving quoted string content intact."""
    in_string = False
    for index, char in enumerate(line):
        if char == '"':
            in_string = not in_string
        elif char == "'" and not in_string:
            return line[:index]
    return line


def _logical_lines(source: str) -> list[str]:
    """Comment-stripped lines with ``_`` continuations joined."""
    lines: list[str] = []
    pending = ""
    for raw in source.splitlines():
        line = _strip_comment(raw).strip()
        if line.endswith("_"):
            pending += line[:-1] + " "
            continue
        line = (pending + line).strip()
        pending = ""
        if line:
            lines.append(line)
    if pending.strip():
        lines.append(pending.strip())
    return lines


def _arguments(call: str) -> list[str]:
    """Split a call's argument list on top-level commas."""
    arguments: list[str] = []
    current: list[str] = []
    depth = 0
    in_string = False
    for char in call:
        if char == '"':
            in_string = not in_string
        if not in_string:
            if char in "([":
                depth += 1
            elif char in ")]":
                depth -= 1
            elif char == "," and depth == 0:
                arguments.append("".join(current).strip())
                current = []
                continue
        current.append(char)
    arguments.append("".join(current).strip())
    return arguments


def _call_arguments(line: str, open_paren: int) -> str | None:
    """Return the text between ``open_paren`` and its matching close paren."""
    depth = 0
    in_string = False
    for index in range(open_paren, len(line)):
        char = line[index]
        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return line[open_paren + 1 : index]
    return None


def _waited_run_calls(source: str) -> list[tuple[str, str]]:
    """Every ``shell.Run`` call whose wait argument is the literal ``True``.

    Returns ``(prefix, line)`` pairs, where ``prefix`` is the text before the
    called object -- enough to tell ``WScript.Quit shell.Run(...)`` and
    ``exitCode = shell.Run(...)`` from a bare statement that discards the code.
    """
    waited: list[tuple[str, str]] = []
    for line in _logical_lines(source):
        for match in _RUN_CALL.finditer(line):
            rest = line[match.end() :]
            stripped = rest.lstrip()
            if stripped.startswith("("):
                open_paren = match.end() + len(rest) - len(stripped)
                arguments = _call_arguments(line, open_paren)
            else:
                arguments = stripped
            if arguments is None:
                continue
            parts = _arguments(arguments)
            if not parts or parts[-1].strip().lower() != "true":
                continue
            waited.append((line[: match.start()], line))
    return waited


def _discarded_exit_codes(source: str) -> list[str]:
    """Return the ``shell.Run(..., True)`` lines whose result never reaches Quit."""
    code = "\n".join(_logical_lines(source))
    quit_operands = {match.group("operand").lower() for match in _QUIT_CALL.finditer(code)}
    discarded: list[str] = []
    for prefix, line in _waited_run_calls(source):
        normalized = prefix.strip().lower().rstrip("(").strip()
        if normalized == "wscript.quit":
            continue
        assignment = _ASSIGNMENT.fullmatch(prefix.strip())
        if assignment is not None and assignment.group("name").lower() in quit_operands:
            continue
        discarded.append(line)
    return discarded


def _vbs_scripts() -> list[Path]:
    """Every VBS launcher under ``scripts/``, derived by globbing."""
    if not SCRIPTS_DIR.is_dir():
        return []
    return sorted(SCRIPTS_DIR.rglob("*.vbs"))


def test_vbs_launchers_are_discovered() -> None:
    """Positive control for the scan: an empty glob must not make it vacuous."""
    assert _vbs_scripts(), "no scripts/**/*.vbs launchers were found; update this guard"


@pytest.mark.parametrize("script", _vbs_scripts(), ids=lambda path: path.name)
def test_waited_run_propagates_the_child_exit_code(script: Path) -> None:
    """A launcher that waits for its child must pass the exit code to Quit."""
    source = script.read_text(encoding="utf-8", errors="replace")
    discarded = _discarded_exit_codes(source)
    assert not discarded, (
        f"{script.relative_to(REPO_ROOT)} discards the exit code of a waited "
        "child:\n"
        + "\n".join(f"    {line}" for line in discarded)
        + "\n\nTask Scheduler reads LastTaskResult from wscript.exe, which stays "
        "0 unless the launcher ends with `WScript.Quit shell.Run(...)` (or "
        "`exitCode = shell.Run(...)` followed by `WScript.Quit exitCode`). "
        "See issue #2753."
    )


def test_scan_flags_a_planted_launcher_that_discards_the_exit_code(
    tmp_path: Path,
) -> None:
    """Positive control: a planted discarding launcher must be flagged."""
    planted = tmp_path / "planted-hidden.vbs"
    planted.write_text(
        'Set shell = CreateObject("WScript.Shell")\n'
        'shell.Run "powershell.exe -File stub.ps1", 0, True\n',
        encoding="utf-8",
    )

    discarded = _discarded_exit_codes(planted.read_text(encoding="utf-8"))

    assert discarded == ['shell.Run "powershell.exe -File stub.ps1", 0, True']


@pytest.mark.skipif(
    shutil.which("cscript") is None,
    reason="cscript.exe is not available on this runner",
)
def test_fleet_pass_launcher_returns_the_stub_exit_code(tmp_path: Path) -> None:
    """Run the real launcher against a stub that exits 7 and expect 7 back."""
    launcher = tmp_path / "fleet-pass-hidden.vbs"
    launcher.write_text(
        (SCRIPTS_DIR / "fleet-pass-hidden.vbs").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "fleet-pass.ps1").write_text("exit 7\n", encoding="utf-8-sig")

    completed = subprocess.run(
        ["cscript", "//nologo", str(launcher)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 7, (
        "fleet-pass-hidden.vbs did not propagate the stub's exit code; "
        f"cscript exited {completed.returncode}\n"
        f"stdout: {completed.stdout}\n"
        f"stderr: {completed.stderr}"
    )
