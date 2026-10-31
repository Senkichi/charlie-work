"""``--fleet-dir`` must reach the fleet-registry read in every fleet subcommand (issue #2807).

Regression guard for a defect where ``fleet status``, ``fleet review-queue``
and ``fleet operator-queue`` layered each repo's config from ``--fleet-dir``
but loaded ``fleet.json`` from the *default* fleet dir (they called
``layout.fleet_registry_path()`` with no ``override``), silently enumerating
the wrong repo set whenever the two registries diverged.

The subcommand list is derived from the argparse ``fleet`` subparser's
choices rather than hand-listed; subcommands that do not read the registry
are skipped. Every registry-reading subcommand is driven both ways:

* with ``CHARLIE_WORK_FLEET_DIR`` (decoy, conftest-isolated) registering
  repo A and ``--fleet-dir`` registering repo B, the command must enumerate
  B and not A;
* a planted positive control with no ``--fleet-dir`` must enumerate A —
  proving the decoy registry is the one actually read.
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from charlie_work import cli
from charlie_work.config import OrchestratorConfig

# The fleet subcommands that read the fleet registry through
# ``layout.fleet_registry_path``. Subcommands are NOT hand-listed for the
# parametrization itself — the choices come from ``build_parser()`` — but a
# command outside this set cannot be asserted against a registry and is
# skipped: it is shape-different (launches workers, mutates state) and has
# its own coverage.
_REGISTRY_READERS = frozenset({"status", "review-queue", "operator-queue"})


def _fleet_subcommand_choices() -> list[str]:
    """Every ``fleet`` subcommand name as argparse actually sees it."""
    parser = cli.build_parser()
    top_sub = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    fleet_parser = top_sub.choices["fleet"]
    fleet_sub = next(
        action
        for action in fleet_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return sorted(fleet_sub.choices)


def _registry_entry(repo_root: Path) -> dict[str, str]:
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return {
        "repo_root": str(repo_root),
        "name_with_owner": "owner/x",
        "config_path": str(repo_root / "orchestrator.config.yaml"),
        "state_dir": str(repo_root / ".var" / "charlie-work"),
        "first_seen": now,
        "last_seen": now,
    }


def _write_registry(fleet_dir: Path, repo_key: str, repo_root: Path) -> None:
    fleet_dir.mkdir(parents=True, exist_ok=True)
    repo_root.mkdir(parents=True, exist_ok=True)
    (fleet_dir / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": {repo_key: _registry_entry(repo_root)}}),
        encoding="utf-8",
    )


def _run(fleet_command: str, fleet_dir_override: str | None):
    """Run one ``fleet`` subcommand through the real parser and runner,
    stubbing each per-repo status to a fixed result.
    """
    if fleet_command == "status":
        module, name = "charlie_work.fleet_status", "run_fleet_status"
    elif fleet_command == "review-queue":
        module, name = "charlie_work.cli", "run_fleet_review_queue"
    elif fleet_command == "operator-queue":
        module, name = "charlie_work.cli", "run_fleet_operator_queue"
    else:  # pragma: no cover - guarded by _REGISTRY_READERS
        pytest.fail(f"unhandled registry-reading fleet subcommand: {fleet_command}")
    runner = getattr(__import__(module, fromlist=[name]), name)

    argv = ["fleet", fleet_command]
    if fleet_dir_override is not None:
        argv = ["--fleet-dir", fleet_dir_override, *argv]
    args = cli.build_parser().parse_args(argv)

    app_cls = MagicMock()
    app_cls.return_value = MagicMock(
        status=MagicMock(return_value=MagicMock(ok=True, data={"ready_issue_count": 0})),
        review_queue=MagicMock(return_value=MagicMock(ok=True, data={"entries": []})),
        operator_queue=MagicMock(return_value=MagicMock(ok=True, data={"entries": []})),
    )

    with ExitStack() as stack:
        for patcher in (
            patch(f"{module}.load_layered_config", return_value=OrchestratorConfig()),
            patch(f"{module}.runtime_paths", return_value=MagicMock()),
            patch(f"{module}.github_client_for", return_value=MagicMock()),
            patch(f"{module}.GitHub"),
            patch(f"{module}.OrchestratorApp", app_cls),
        ):
            stack.enter_context(patcher)
        if fleet_command == "status":
            patch_fleet_report = patch(
                "charlie_work.fleet_status.compute_api_worker_fleet_report",
                return_value=None,
            )
            stack.enter_context(patch_fleet_report)
        result = runner(args)
    return result


@pytest.fixture()
def decoy_and_override(tmp_path: Path) -> tuple[Path, Path]:
    """Repo A registered in the env-directed decoy dir, repo B in a
    ``--fleet-dir``-directed dir. The conftest autouse fixture
    ``_isolate_fleet_registry`` already points ``CHARLIE_WORK_FLEET_DIR``
    at ``tmp_path / "fleet"`` — that dir is the decoy.
    """
    decoy = tmp_path / "fleet"
    override = tmp_path / "override-fleet"
    _write_registry(decoy, "owner/repo-a", tmp_path / "repo-a")
    _write_registry(override, "owner/repo-b", tmp_path / "repo-b")
    return decoy, override


@pytest.mark.parametrize(
    "fleet_command",
    sorted(_REGISTRY_READERS),
    ids=sorted(_REGISTRY_READERS),
)
def test_fleet_dir_flag_reaches_registry(decoy_and_override, fleet_command: str) -> None:
    """``--fleet-dir`` selects the registry, not the default fleet dir."""
    choices = _fleet_subcommand_choices()
    assert fleet_command in choices, (
        f"argparse fleet subparser lost the '{fleet_command}' choice; "
        "this test's parametrization is no longer covering it"
    )
    _decoy, override = decoy_and_override

    result = _run(fleet_command, str(override))

    assert result.ok is True, result.data
    assert "owner/repo-b" in result.data["repos"]
    assert "owner/repo-a" not in result.data["repos"]
    # Nothing mis-reported as stale/failed: the whole difference between the
    # two registries must be invisible noise-wise.
    assert result.data.get("stale", []) == []


@pytest.mark.parametrize(
    "fleet_command",
    sorted(_REGISTRY_READERS),
    ids=sorted(_REGISTRY_READERS),
)
def test_no_flag_reads_decoy_registry(decoy_and_override, fleet_command: str) -> None:
    """Positive control: with no ``--fleet-dir``, the decoy (env) registry —
    repo A — is the one read. Without this, a fix that always pointed the
    registry read somewhere else could pass the first test while still
    ignoring the flag.
    """
    assert fleet_command in _fleet_subcommand_choices()
    decoy, _override = decoy_and_override

    result = _run(fleet_command, None)

    assert result.ok is True, result.data
    assert "owner/repo-a" in result.data["repos"]
    assert "owner/repo-b" not in result.data["repos"]
    assert result.data.get("stale", []) == []


def test_parametrization_covers_every_argparse_fleet_choice() -> None:
    """Every non-registry-reading choice in argparse's ``fleet`` subparser is
    explicitly accounted for here, so a NEW registry reader (a future
    ``fleet <x>`` that loads fleet.json) cannot silently miss coverage: it
    joins this non-registry set by default, which makes this test fail until
    the author either writes its registry test or proves it never reads the
    registry.
    """
    non_registry = set(_fleet_subcommand_choices()) - _REGISTRY_READERS
    # Names actually known not to read the registry (each launches a worker /
    # mutates fleet-dir state / is purely local). Additions here must cite the
    # new subcommand in the test docstring.
    #
    # ``retire`` / ``unretire`` (issue #2754) edit ``fleet.retired`` in the
    # repo's own ``orchestrator.config.yaml`` and deliberately leave the
    # fleet registry alone: ``fleet_retire.py`` never imports or calls
    # ``fleet_registry``, and the registry's ``retired`` field is derived on
    # the next ``touch_repo``. They are covered by
    # ``tests/test_fleet_retire_render.py``.
    ACCOUNTED = {
        "work",
        "bash-rats",
        "supervise",
        "supervise-loop",
        "stop",
        "pause",
        "resume",
        "retire",
        "unretire",
    }
    assert non_registry <= ACCOUNTED, (
        f"new fleet subcommand(s) {sorted(non_registry - ACCOUNTED)} were added to "
        "the argparse fleet subparser but are accounted nowhere: if they read "
        "the fleet registry, add them to _REGISTRY_READERS and cover them; "
        "otherwise add them to ACCOUNTED with a citation."
    )
