"""PowerShell quoting for copied commands: unit cases plus a real PowerShell round-trip."""

from __future__ import annotations

import base64
import shutil
import subprocess

import pytest

from charlie_work.dashboard.now_shell import join_command, ps_quote

HOSTILE = [
    r"C:\Users\me\my repo",
    r"C:\it's here",
    "a$(Get-Date)b",
    "$env:USERNAME",
    "back`tick",
    "semi;colon & amp | pipe",
    'dq"uote',
    "smart\u2018quote\u2019s",
    "x,y",
    "@splat",
    "<approved|blocked>",
    "line1\nline2",
    "",
]


def test_safe_tokens_stay_bare() -> None:
    assert join_command(["charlie", "--repo", "C:/a/b-c_d.e", "--issue", "7"]) == (
        "charlie --repo C:/a/b-c_d.e --issue 7"
    )


def test_quoting_doubles_every_single_quote_form() -> None:
    assert ps_quote("it's") == "'it''s'"
    assert ps_quote("\u2018") == "'\u2018\u2018'"
    assert ps_quote("") == "''"
    assert ps_quote("a b") == "'a b'"
    assert '"' not in ps_quote("it's")


def _powershell() -> str | None:
    return shutil.which("powershell.exe") or shutil.which("powershell")


@pytest.mark.skipif(_powershell() is None, reason="powershell.exe not available")
def test_round_trip_through_real_powershell() -> None:
    script = (
        "function f { foreach ($a in $args) { "
        "'' + (($a.ToCharArray() | ForEach-Object { [int]$_ }) -join ',') } }\n"
        "f " + " ".join(ps_quote(a) for a in HOSTILE)
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    run = subprocess.run(
        [_powershell(), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    got = [tuple(int(n) for n in ln.split(",")) if ln else () for ln in lines]
    # One line per argument, including the empty-string argument.
    assert got == [tuple(ord(c) for c in a) for a in HOSTILE]
