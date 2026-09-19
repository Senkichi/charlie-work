"""Global ``--dry-run`` flag semantics for ``charlie`` commands.

Split out of ``tests/test_cli.py`` (issue #1561, Track 1) -- bodies are
verbatim relocations; shared fakes live in ``tests/_cli_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from _cli_fixtures import (
    _FakeGitHub,
    _make_repo,
)
from charlie_work import cli


def test_cli_dry_run_does_not_write_fleet_registry_through_build_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #618: ``--dry-run`` through ``build_app`` must not create or mutate
    the fleet registry.

    The ``verdict`` command dispatches through ``build_app``, which calls
    ``touch_repo(..., dry_run=args.dry_run)``.  Main's #1157 refactor once
    silently dropped the ``dry_run=`` kwarg from that call site; this test
    ensures a future refactor cannot do the same without a test failing; the
    registry must not be created or bumped when ``--dry-run`` is set.
    """
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    summary = repo / "summary.md"
    summary.write_text("lgtm", encoding="utf-8")

    # The autouse _isolate_fleet_registry fixture points
    # CHARLIE_WORK_FLEET_DIR at tmp_path / "fleet"; resolve it explicitly so
    # the assertion is against the same path touch_repo would write to.
    fleet_json = Path(tmp_path / "fleet" / "fleet.json")

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "--dry-run",
            "verdict",
            "--pr",
            "1",
            "--decision",
            "approved",
            "--summary-file",
            str(summary),
        ]
    )

    assert rc == 0
    assert not fleet_json.exists(), (
        "fleet.json must not be created in dry-run mode (build_app/touch_repo "
        "dropped the dry_run= kwarg)"
    )


def test_cli_dry_run_does_not_write_fleet_registry_through_doctor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #618: ``--dry-run`` through ``run_doctor_command`` must not create
    or mutate the fleet registry.

    The ``doctor`` command dispatches through ``run_doctor_command``, which
    calls ``touch_repo(..., dry_run=args.dry_run)`` — the second call site in
    cli.py.  This test ensures that call site is also gated, so a refactor
    that drops the kwarg from either site is caught.
    """
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)

    # Avoid running the real doctor checks — we only care about whether
    # touch_repo wrote the registry.
    monkeypatch.setattr(cli, "run_doctor", lambda *a, **k: (True, []))

    fleet_json = Path(tmp_path / "fleet" / "fleet.json")

    rc = cli.main(
        [
            "--repo",
            str(repo),
            "--dry-run",
            "doctor",
        ]
    )

    assert rc == 0
    assert not fleet_json.exists(), (
        "fleet.json must not be created in dry-run mode (doctor/touch_repo "
        "dropped the dry_run= kwarg)"
    )


# --------------------------------------------------------------------------
# Global --dry-run must survive subcommand parsing
# --------------------------------------------------------------------------


def _iter_subparsers(parser: Any, prefix: tuple[str, ...] = ()) -> Any:
    """Yield (argv_prefix, parser) for every subparser, recursively."""
    import argparse

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield (*prefix, name), sub
                yield from _iter_subparsers(sub, (*prefix, name))


def _dry_run_action(parser: Any) -> Any:
    for action in parser._actions:
        if "--dry-run" in getattr(action, "option_strings", ()):
            return action
    return None


def test_every_subcommand_dry_run_flag_defers_to_the_global_one() -> None:
    """A subcommand-level --dry-run must not clobber the top-level one.

    ``--dry-run`` exists on the top-level parser, so an operator may write it
    before or after the subcommand and both must work. argparse applies a
    subparser's own default *after* the global flag was parsed, so a plain
    ``action="store_true"`` on a subparser silently overwrites True with False.
    Discovered on the live host: ``charlie --dry-run runners allocate`` launched a
    real runner listener and reported its PID.

    Derived from the parser itself rather than a hand-maintained list, so a new
    subcommand that repeats the plain idiom fails this test.
    """
    import argparse

    parser = cli.build_parser()
    offenders = []
    for path, sub in _iter_subparsers(parser):
        action = _dry_run_action(sub)
        if action is not None and action.default is not argparse.SUPPRESS:
            offenders.append(" ".join(path))
        elif "dry_run" in sub._defaults:
            # argparse seeds a namespace from two places -- action defaults and
            # the parser's own ``_defaults`` (what ``set_defaults`` writes) -- and
            # a subparser parses into a *fresh* namespace, then copies every key
            # onto the parent's. So ``set_defaults(dry_run=False)`` overwrites the
            # already-parsed global True exactly like an action default does,
            # while leaving ``action.default is SUPPRESS`` and this test green.
            # Checking both is what makes this guard complete rather than a patch
            # over the one mechanism that happened to bite us.
            offenders.append(" ".join(path) + " (via set_defaults)")

    assert offenders == [], (
        "these subcommands clobber the global --dry-run; route them through "
        f"cli._add_dry_run: {offenders}"
    )

    # The top-level flag must keep a real default so args.dry_run always exists.
    assert _dry_run_action(parser).default is False


def test_set_defaults_clobbers_a_global_flag_too() -> None:
    """Pins the argparse behaviour that the guard above's second check exists for.

    Stdlib-only, no repo coupling: a subparser that never declares ``--dry-run``
    but calls ``set_defaults(dry_run=False)`` still overwrites an already-parsed
    global ``--dry-run``. That is the blind spot in checking ``action.default``
    alone -- there is no action here to inspect.

    If a future Python stops copying subparser defaults over values the parent
    already parsed, this test fails and the guard's ``_defaults`` branch can go.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    leaf = parser.add_subparsers(dest="command").add_parser("leaf")
    leaf.set_defaults(dry_run=False)

    assert _dry_run_action(leaf) is None, "no action to inspect -- that is the point"
    assert parser.parse_args(["--dry-run", "leaf"]).dry_run is False


def test_global_dry_run_reaches_runners_allocate_in_either_position() -> None:
    parser = cli.build_parser()

    assert parser.parse_args(["--dry-run", "runners", "allocate"]).dry_run is True
    assert parser.parse_args(["runners", "allocate", "--dry-run"]).dry_run is True
    assert parser.parse_args(["runners", "allocate"]).dry_run is False
