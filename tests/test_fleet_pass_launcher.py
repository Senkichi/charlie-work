"""Guards on how the fleet supervisor process is launched (issue #854).

The supervisor runs an IN-PROCESS self-deploy: `self_deploy()` shells out to
`uv sync` whenever a pulled commit touched `pyproject.toml` / `uv.lock`. `uv sync`
reinstalls the editable project every run, and its uninstall half must delete
`.venv/Scripts/charlie.exe`.

Windows locks running executables. So if the supervisor is launched through the
`charlie` console script, the process holds an exclusive handle on the exact file
its own self-deploy has to replace, and every sync fails with:

    error: failed to remove file `...\\.venv\\...\\../../Scripts/charlie.exe`:
           Access is denied. (os error 5)

That is structural rather than a race -- the process invoking the sync *is* the
lock holder -- so no retry, backoff, or timeout can ever make it succeed. The
observed consequence was a permanently-pinned `pending-sync.json` marker and
dependency updates that silently never reached the live fleet.

Entering through `python -m charlie_work` makes the locked image `python.exe`,
which `uv sync` never replaces, leaving `charlie.exe` free to be rewritten.

These tests exist because the console-script form is the *obvious* way to write
the launcher and reads as strictly simpler. Without a guard it gets restored by
anyone tidying the script, and the failure it reintroduces is silent for as long
as no dependency changes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

# The supervisor entry point that must never be reached via the console script.
_SUPERVISE_INVOCATION = re.compile(r"\bfleet\s+supervise\b")

# `charlie` / `charlie-work` as a bare command word (both are console scripts
# declared in pyproject's [project.scripts], and both produce the same .exe
# lock). Anchored on a word boundary so `python -m charlie_work` does not match.
_CONSOLE_SCRIPT_CALL = re.compile(r"(?<![\w./\\-])charlie(?:-work)?\s+fleet\s+supervise\b")

_MODULE_CALL = re.compile(r"python\s+-m\s+charlie_work\s+fleet\s+supervise\b")


def _launcher_scripts() -> list[Path]:
    """Every PowerShell script in scripts/ that invokes `fleet supervise`.

    Derived by scanning rather than hardcoded so that a second launcher added
    later is covered automatically -- a hardcoded path would let a new script
    reintroduce the lock silently.
    """
    if not SCRIPTS_DIR.is_dir():
        return []
    return [
        path
        for path in sorted(SCRIPTS_DIR.glob("*.ps1"))
        if _SUPERVISE_INVOCATION.search(_command_lines(path))
    ]


def _command_lines(path: Path) -> str:
    """Return the script's non-comment lines.

    The rationale comment in `fleet-pass.ps1` deliberately quotes both the bad
    invocation and the error text, so scanning raw file content would match the
    documentation and defeat the guard. Only executable lines are considered.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def test_at_least_one_launcher_is_discovered() -> None:
    """Positive control for the scan itself.

    Without this, deleting or renaming the launcher would make every other test
    in this module vacuously pass over an empty list.
    """
    assert _launcher_scripts(), (
        "no scripts/*.ps1 invoking `fleet supervise` was found; if the launcher "
        "moved, update this module rather than leaving the guards inert"
    )


@pytest.mark.parametrize("script", _launcher_scripts(), ids=lambda p: p.name)
def test_launcher_does_not_use_console_script(script: Path) -> None:
    """The supervisor must not be launched via `charlie`/`charlie-work` (#854)."""
    offending = [
        line for line in _command_lines(script).splitlines() if _CONSOLE_SCRIPT_CALL.search(line)
    ]
    assert not offending, (
        f"{script.name} launches the supervisor through a console script:\n"
        + "\n".join(f"    {line.strip()}" for line in offending)
        + "\n\nThis re-creates the issue #854 deadlock: the running charlie.exe is "
        "the file `uv sync` must delete, so self-deploy fails permanently with "
        "os error 5. Use `python -m charlie_work fleet supervise` instead."
    )


@pytest.mark.parametrize("script", _launcher_scripts(), ids=lambda p: p.name)
def test_launcher_uses_module_entry_point(script: Path) -> None:
    """Assert the fix positively, not just the absence of the old form.

    A launcher rewritten to some third invocation would pass the negative test
    above while still holding the lock.
    """
    assert _MODULE_CALL.search(_command_lines(script)), (
        f"{script.name} does not invoke `python -m charlie_work fleet supervise`; "
        "see issue #854 for why the module entry point is required."
    )


def test_module_entry_point_exists() -> None:
    """`python -m charlie_work` must actually be dispatchable.

    The launcher guard is worthless if the module entry point it mandates is
    missing -- that would swap a failing sync for a fleet that does not start.
    """
    main_module = REPO_ROOT / "src" / "charlie_work" / "__main__.py"
    assert main_module.is_file(), "src/charlie_work/__main__.py is required by the launcher"
    source = main_module.read_text(encoding="utf-8")
    assert "main" in source, "__main__.py must dispatch to the CLI entry point"
