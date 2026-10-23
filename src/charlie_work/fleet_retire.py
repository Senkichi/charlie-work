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
import re
from pathlib import Path
from typing import Any

import yaml

from .atomic_write import write_text_atomic
from .config import DEFAULT_CONFIG_FILENAME, find_config_path
from .paths import find_repo_root
from .command_result import CommandResult

logger = logging.getLogger(__name__)

_FLEET_LINE = re.compile(r"^fleet\s*:\s*(?P<rest>.*)$")
_RETIRED_LINE = re.compile(
    r"^(?P<indent>[ \t]+)retired\s*:\s*(?P<value>[^#]*?)(?P<comment>\s*#.*)?$"
)


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


def _render_retired(text: str, retired: bool) -> str:
    """Return *text* with only ``fleet.retired`` changed.

    A full ``yaml.safe_dump`` round-trip would delete every comment and
    reflow the operator's file. This edits the one line in place (or inserts
    the one line into an existing ``fleet:`` block) and leaves every other
    byte untouched.
    """
    value = "true" if retired else "false"
    lines = text.splitlines()
    fleet_index: int | None = None
    for index, line in enumerate(lines):
        match = _FLEET_LINE.match(line)
        if match is None:
            continue
        fleet_index = index
        rest = match.group("rest").strip()
        # ``fleet: {retired: true}`` / ``fleet: null`` cannot be edited line
        # in place; replace the flow/null value with a block mapping. A plain
        # inline comment stays as written.
        if rest and not rest.startswith("#"):
            lines[index] = "fleet:"
        break

    if fleet_index is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(["fleet:", f"  retired: {value}"])
        return "\n".join(lines) + "\n"

    block_end = fleet_index + 1
    while block_end < len(lines):
        candidate = lines[block_end]
        if candidate.strip() and not candidate.startswith((" ", "\t")):
            break
        block_end += 1
    insert_at = block_end
    while insert_at > fleet_index + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1

    indent = "  "
    for line in lines[fleet_index + 1 : block_end]:
        if line.strip():
            indent = line[: len(line) - len(line.lstrip(" \t"))]
            break

    for index in range(fleet_index + 1, block_end):
        match = _RETIRED_LINE.match(lines[index])
        if match is None:
            continue
        comment = match.group("comment") or ""
        lines[index] = f"{match.group('indent')}retired: {value}{comment}"
        return "\n".join(lines) + "\n"

    lines.insert(insert_at, f"{indent}retired: {value}")
    return "\n".join(lines) + "\n"


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

    verb = "retire" if retired else "unretire"
    if bool(before) == retired:
        return CommandResult(
            True,
            f"fleet {verb}: {path} already has fleet.retired: {str(retired).lower()}",
            {"path": str(path), "changed": False, "retired": retired},
        )

    dry_run = bool(getattr(args, "dry_run", False))
    if not dry_run:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        write_text_atomic(path, _render_retired(text, retired))
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
