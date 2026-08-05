"""``python -m charlie_work.cli`` must actually run, and must propagate its code.

Issue #959. ``cli.py`` declares ``main()`` but had no ``if __name__ ==
"__main__":`` guard, so the module form imported the module, executed nothing,
printed nothing, and exited 0. Every subcommand was a silent no-op that could not
be told apart from success -- observed live when ``python -m charlie_work.cli
tripwire ack 951 --reason ...`` reported nothing at exit 0 while leaving the
finding in ``state.json``.

**Why a subprocess.** The defect is about what happens when the module is run as
``__main__``. Importing :mod:`charlie_work.cli` in-process and asserting anything
about ``main`` passes identically with and without the guard -- the function
exists either way. Only a real ``-m`` invocation observes it.

**Why PYTHONPATH is set explicitly.** These tests must exercise *this* checkout.
Relying on the installed distribution makes the subprocess resolve whatever the
venv's editable install points at, which in a git worktree is the main checkout's
``src`` -- so a worktree run would test code the branch does not contain and
report green for the wrong tree.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _run(*args: str) -> "subprocess.CompletedProcess[str]":
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(_SRC) + (os.pathsep + existing if existing else "")
    return subprocess.run(
        [sys.executable, "-m", "charlie_work.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_SRC.parent),
    )


def test_module_form_actually_executes() -> None:
    """``-m ... --help`` prints usage instead of silently doing nothing.

    Without the guard this returns 0 with completely empty stdout, which is the
    exact shape that made the bug invisible: exit 0 and no output reads as
    "nothing to report" when it in fact means "nothing ran".
    """
    result = _run("--help")

    assert result.returncode == 0, (
        f"--help exited {result.returncode}; stderr={result.stderr[:400]}"
    )
    # The load-bearing assertion. An empty stdout here IS the regression.
    assert result.stdout.strip(), (
        "module form produced no stdout -- cli.py is not executing as __main__ "
        f"(stderr={result.stderr[:400]})"
    )
    assert "usage" in result.stdout.lower(), (
        f"expected argparse usage text, got: {result.stdout[:300]!r}"
    )


def test_module_form_propagates_a_nonzero_exit_code() -> None:
    """A failing invocation must not come back as 0.

    This is the half a bare ``main()`` call would leave broken: the module would
    execute and print, but always exit 0. That matters beyond tidiness --
    ``main()`` returns ``EXIT_RESTART_REQUESTED`` (3) as a cross-version wire
    contract read by the supervise-loop wrapper, and an invocation that always
    exited 0 would make a restart request read as a clean exit, which is the
    failure mode of #862.

    An argparse usage error is used as the failing case because it is decided
    before any handler runs, so this test touches no repository state.
    """
    result = _run("definitely-not-a-real-subcommand")

    assert result.returncode != 0, (
        "a bogus subcommand exited 0 -- the module form is swallowing the exit "
        f"code (stdout={result.stdout[:200]!r})"
    )


def test_console_script_and_module_form_agree() -> None:
    """Control: the two entry points are the same artifact.

    ``[project.scripts]`` maps both ``charlie`` and ``charlie-work`` to
    ``charlie_work.cli:main``. If this test ever fails while the two above pass,
    the module form is running *something*, just not the same thing operators
    get -- which would be a worse bug than the original silence.
    """
    module_help = _run("--help")
    direct = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, r'''%s'''); "
            "from charlie_work.cli import build_parser; "
            "print(build_parser().format_help())" % str(_SRC),
        ],
        capture_output=True,
        text=True,
    )

    assert direct.returncode == 0, f"control failed to run: {direct.stderr[:300]}"
    assert direct.stdout.strip(), "control produced no output -- test is inert"
    # Compare the command list rather than exact text: argparse wraps usage to
    # terminal width, which differs between the two invocations.
    assert module_help.stdout.split() == direct.stdout.split(), (
        "module form's help differs from build_parser()'s own help"
    )
