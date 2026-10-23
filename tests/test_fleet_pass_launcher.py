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


# Structural invariant for the issue #2795 secretenv injection, matched as ONE
# multi-line regex over the script's non-comment lines so the individual
# fragments cannot drift apart. It ties the whole guarded launch together:
#   1. `$secretenv` resolves to the per-user secretenv.exe (not PATH),
#   2. `$secretenvOk` is initialised to $false,
#   3. the if-condition carries BOTH guards — the key-absence (no double-wrap)
#      test AND `Test-Path -LiteralPath $secretenv`,
#   4. inside that same guard block, `secretenv verify ... *> $null` runs and
#      `$secretenvOk` is assigned from ($LASTEXITCODE -eq 0) right after it,
#   5. the launch is `if ($secretenvOk) { & $secretenv run ... -- cmd /c $cmdLine
#      } else { & cmd /c $cmdLine }` — wrap in the if-branch, plain launch
#      (a missing secret never strands the fleet) in the else-branch.
# Each of these can only be checked together: the substring form previously in
# this module passed even when the gate was inverted (`if (-not $secretenvOk)`)
# or the branches swapped, because every fragment still existed individually.
_SECRETENV_LAUNCH_RE = re.compile(
    r"\$secretenv\s*=\s*Join-Path\s+\$env:USERPROFILE\s+'\.local\\bin\\secretenv\.exe'"
    r"[^{}]*?\$secretenvOk\s*=\s*\$false\b"
    r"[^{}]*?if\s*\([^\n\(]*-not\s+\[Environment\]::GetEnvironmentVariable\('EMPERICUS_ANTHROPIC_API_KEY'\)"
    r"[^{}]*?Test-Path\s+-LiteralPath\s+\$secretenv"
    r"[^{}]*?\)[^\n{]*\{\s*\n\s*"
    r"&\s*\$secretenv\s+verify\s+EMPERICUS_ANTHROPIC_API_KEY\s+\*>\s*\$null\s*\n\s*"
    r"\$secretenvOk\s*=\s*\(\s*\n?\s*\$LASTEXITCODE\s+-eq\s+0\s*\)\s*\n"
    r"\}\s*\n"
    r"if\s*\(\s*\$secretenvOk\s*\)\s*\{\s*\n\s*"
    r"&\s*\$secretenv\s+run\s+EMPERICUS_ANTHROPIC_API_KEY\s+--\s+cmd\s+/c\s+\$cmdLine\s*\n"
    r"\}\s*else\s*\{\s*\n\s*"
    r"&\s*cmd\s+/c\s+\$cmdLine\s*\n?"
    r"\}",
)


def test_fleet_pass_launch_structure_ties_secretenv_wrap_to_its_guards() -> None:
    """Issue #2795: the secretenv wrap and its guards are one invariant.

    The `api` adapter resolves the provider key from the supervisor's own env
    (`os.environ.get(provider.api_key_env)`), and the key lives only in the
    DPAPI secretenv store -- so the scheduled-task-spawned supervisor cannot
    run `harness: api` launches unless the launcher wraps the supervisor in
    `secretenv run`. The wrap must be guarded (never double-wrap an env that
    already has the key, never trust a secretenv.exe that disappears between
    the exists-check and the spawn, never allow a missing/undecryptable secret
    to strand the whole fleet) and must never emit the secret value. Smaller
    mutants than the full block (inverted gate, `if ($true)`, dropped
    Test-Path guard, swapped branches, `$secretenvOk = $true`) all fail this
    single structural match -- see the E-field in the mutation matrix this
    test's PR documents.
    """
    script = SCRIPTS_DIR / "fleet-pass.ps1"
    code = _command_lines(script)
    assert _SECRETENV_LAUNCH_RE.search(code), (
        "fleet-pass.ps1's secretenv launch must keep the full guarded structure: "
        "$secretenv per-user path, $secretenvOk=$false, an if-condition with BOTH "
        "the key-absence and Test-Path guards, $secretenvOk=($LASTEXITCODE -eq 0) "
        "right after the verify call, and the wrap/`& cmd /c $cmdLine` split as "
        "if/else (issue #2795)"
    )


def test_fleet_pass_keeps_plain_launch_fallback() -> None:
    """Issue #2795: a missing secret must never strand the whole fleet.

    The structural regex above pins the if/else split itself; this test keeps
    the trailing `exit=$LASTEXITCODE` log marker, which must stay OUTSIDE both
    branches so the marker records the same child exit code regardless of
    whether the launch was wrapped.
    """
    script = SCRIPTS_DIR / "fleet-pass.ps1"
    assert "exit=$LASTEXITCODE" in script.read_text(encoding="utf-8"), (
        "fleet-pass.ps1 must keep the trailing `exit=$LASTEXITCODE` log marker "
        "(secretenv propagates the child exit code)"
    )
