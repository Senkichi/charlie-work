"""Regression coverage for issue #1071.

Issue #1071: CI runs on ``windows-latest`` (self-hosted Windows runner) where
Python resolves its text layer from the ANSI codepage (cp1252), not UTF-8.
``ci.yml`` set neither ``PYTHONUTF8`` nor ``PYTHONIOENCODING``, so any test that
touches a non-ASCII path, fixture, event payload, or subprocess pipe was decoded
with the wrong codec in CI -- asymmetric with the dev box, which sets both
machine-wide.

The fix adds a top-level ``env:`` block to ``.github/workflows/ci.yml`` setting
both variables. These tests guard that fix:

1. ``test_ci_yml_sets_utf8_env_block`` -- a static guard that parses the
   workflow YAML and asserts the env block exists with the exact values. This
   is the mutation-checkable test: reverting ``ci.yml`` to its merge-base
   version (no env block) makes it fail.

2. ``test_default_open_round_trips_non_ascii`` -- the positive control the
   issue asks for: writes and reads back a non-ASCII string (em-dash, CJK)
   through ``open()`` *without* an explicit ``encoding=`` argument, which is
   the path ``PYTHONUTF8=1`` fixes and ``PYTHONIOENCODING`` does not. This
   test passes in CI (env block sets ``PYTHONUTF8=1``) and on the dev box
   (machine-wide), but would fail on ``windows-latest`` without the env block
   because ``open()`` defaults to cp1252. It cannot be mutation-checked
   locally (the dev box sets ``PYTHONUTF8=1`` machine-wide), so the mutation
   check claim in the PR body covers only test 1.
"""

from __future__ import annotations

import locale
from pathlib import Path

import pytest
import yaml

CI_YML = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def test_ci_yml_sets_utf8_env_block() -> None:
    """The top-level ``env:`` block in ``ci.yml`` must set both
    ``PYTHONUTF8=1`` and ``PYTHONIOENCODING=utf-8:surrogateescape``.

    A top-level (workflow-level) ``env:`` applies to every job in the workflow,
    which is the required scope: both the Tests and Lint jobs spawn Python
    processes. Job-level or step-level ``env:`` would silently miss any future
    job that forgets to copy it.

    The ``:surrogateescape`` suffix is load-bearing -- a bare ``utf-8`` overrides
    the surrogateescape handlers that UTF-8 mode installs and resets
    stdin/stdout to ``strict``, turning graceful degradation into a hard crash.
    See ``scripts/fleet-pass.ps1:53-60`` for the full warning.
    """
    assert CI_YML.exists(), f"ci.yml not found at {CI_YML}"
    workflow = yaml.safe_load(CI_YML.read_text(encoding="utf-8"))

    env = workflow.get("env")
    assert env is not None, (
        "ci.yml has no top-level 'env:' block -- PYTHONUTF8/PYTHONIOENCODING "
        "are not set for CI jobs (issue #1071)"
    )
    assert env.get("PYTHONUTF8") == "1", (
        f"ci.yml env.PYTHONUTF8 is {env.get('PYTHONUTF8')!r}, expected '1'"
    )
    assert env.get("PYTHONIOENCODING") == "utf-8:surrogateescape", (
        f"ci.yml env.PYTHONIOENCODING is {env.get('PYTHONIOENCODING')!r}, "
        "expected 'utf-8:surrogateescape' -- a bare 'utf-8' is actively harmful "
        "(see fleet-pass.ps1:53-60)"
    )


@pytest.mark.parametrize(
    "text",
    [
        "em-dash: \u2014 and en-dash: \u2013",
        "CJK: \u4e2d\u6587\u6d4b\u8bd5",
        "mixed: caf\u00e9 \u2014 \u4e2d\u6587 \u00fc\u00f1",
    ],
    ids=["em-dash", "cjk", "mixed"],
)
def test_default_open_round_trips_non_ascii(text: str, tmp_path: Path) -> None:
    """Positive control: a non-ASCII string written and read back through
    ``open()`` *without* an explicit ``encoding=`` argument must survive the
    round trip.

    ``PYTHONUTF8=1`` (set by the ci.yml env block) fixes the ``open()``
    default encoding to UTF-8. Without it on Windows, ``open()`` defaults to
    the ANSI codepage (cp1252), which cannot encode CJK and mangles em-dashes
    into mojibake. ``PYTHONIOENCODING`` does not touch the ``open()`` default,
    so the two variables are not redundant -- this test specifically exercises
    the path only ``PYTHONUTF8`` fixes.

    This test passes under the ci.yml env block and on the dev box (machine-wide
    ``PYTHONUTF8=1``). It would fail on ``windows-latest`` without the env
    block. It is not mutation-checkable locally because the dev box sets the
    variable machine-wide; the mutation check in the PR body covers
    ``test_ci_yml_sets_utf8_env_block`` only.
    """
    # Sanity: the preferred encoding must be UTF-8 for the default open() to
    # round-trip non-ASCII. This is what PYTHONUTF8=1 guarantees.
    preferred = locale.getpreferredencoding(False)
    assert preferred.lower() in ("utf-8", "utf8"), (
        f"locale.getpreferredencoding is {preferred!r}, expected utf-8 -- "
        "PYTHONUTF8 is not active in this process"
    )

    path = tmp_path / "non_ascii.txt"
    # Deliberately NO encoding= argument: this is the path PYTHONUTF8 fixes.
    with open(path, "w") as f:
        f.write(text)
    with open(path, "r") as f:
        round_tripped = f.read()

    assert round_tripped == text, (
        f"non-ASCII round-trip failed: wrote {text!r}, read {round_tripped!r}"
    )
