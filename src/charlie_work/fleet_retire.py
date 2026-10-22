"""``charlie fleet retire`` / ``unretire`` (issue #2754).

The command edits ``fleet.retired`` in the repo's **own**
``orchestrator.config.yaml`` -- never the fleet-wide layer -- so retirement
is a deliberate per-repo decision. It leaves the fleet registry entry alone:
``fleet_registry.touch_repo`` derives the registry's ``retired`` field from
this file on the next touch/pass.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from .atomic_write import write_text_atomic
from .config import DEFAULT_CONFIG_FILENAME, find_config_path
from .paths import find_repo_root
from .command_result import CommandResult

logger = logging.getLogger(__name__)


def _config_path(args: argparse.Namespace, repo_root: Path) -> Path:
    """The repo's own config file, whether or not it exists yet."""
    resolved = find_config_path(repo_root, getattr(args, "config", None))
    return resolved if resolved is not None else repo_root / DEFAULT_CONFIG_FILENAME


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: unreadable config: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def _set_retired(args: argparse.Namespace, *, retired: bool) -> CommandResult:
    repo_root = find_repo_root(
        getattr(args, "repo", None), explicit=getattr(args, "repo", None) is not None
    )
    path = _config_path(args, repo_root)
    try:
        data = _read_config(path)
    except ValueError as exc:
        return CommandResult(False, f"fleet {'retire' if retired else 'unretire'}: {exc}", {})

    fleet = data.get("fleet")
    if fleet is None:
        fleet = {}
    if not isinstance(fleet, dict):
        return CommandResult(
            False,
            f"fleet {'retire' if retired else 'unretire'}: {path}: fleet must be a mapping",
            {},
        )
    before = fleet.get("retired", False)
    fleet["retired"] = retired
    data["fleet"] = fleet

    verb = "retire" if retired else "unretire"
    if bool(before) == retired:
        return CommandResult(
            True,
            f"fleet {verb}: {path} already has fleet.retired: {str(retired).lower()}",
            {"path": str(path), "changed": False, "retired": retired},
        )

    dry_run = bool(getattr(args, "dry_run", False))
    if not dry_run:
        rendered = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        write_text_atomic(path, rendered)
    return CommandResult(
        True,
        (
            f"fleet {verb}: set fleet.retired: {str(retired).lower()} in {path}"
            + (" (dry run)" if dry_run else "")
        ),
        {
            "path": str(path),
            "changed": True,
            "previous": bool(before),
            "retired": retired,
            "dry_run": dry_run,
        },
    )


def run_fleet_retire(args: argparse.Namespace) -> CommandResult:
    """Set ``fleet.retired: true`` in the repo's own config."""
    return _set_retired(args, retired=True)


def run_fleet_unretire(args: argparse.Namespace) -> CommandResult:
    """Set ``fleet.retired: false`` in the repo's own config."""
    return _set_retired(args, retired=False)
