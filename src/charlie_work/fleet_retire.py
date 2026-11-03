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

from .atomic_write import write_bytes_atomic
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


def _read_text(path: Path) -> str:
    """Read *path* verbatim, preserving its newline style (CRLF stays CRLF)."""
    if not path.exists():
        return ""
    try:
        # ``newline=""`` disables universal-newline translation; the text is
        # handed back exactly as written so a render can preserve CRLF.
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except OSError as exc:
        raise ValueError(f"{path}: unreadable config: {exc}") from exc


def _parse_config(text: str, path: Path) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: unreadable config: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def _flow_mapping_end(
    lines: list[str], fleet_index: int, value_start: int
) -> tuple[int, int] | None:
    """``(line, column)`` of the ``}`` closing the flow mapping at *value_start*.

    *value_start* is the column on ``fleet_index`` where the value text begins.
    Quote-aware so a brace inside a quoted scalar does not unbalance the scan;
    ``#`` starts a comment only at the start of a line or after whitespace.
    """
    depth = 0
    started = False
    in_single = False
    in_double = False
    for line_index in range(fleet_index, len(lines)):
        line = lines[line_index]
        column = value_start if line_index == fleet_index else 0
        while column < len(line):
            char = line[column]
            following = line[column + 1] if column + 1 < len(line) else ""
            if in_double:
                if char == "\\":
                    column += 2
                    continue
                if char == '"':
                    in_double = False
            elif in_single:
                if char == "'":
                    if following == "'":
                        column += 2
                        continue
                    in_single = False
            elif char == '"':
                in_double = True
            elif char == "'":
                in_single = True
            elif char == "#" and (column == 0 or line[column - 1].isspace()):
                break
            elif char == "{":
                depth += 1
                started = True
            elif char == "}":
                depth -= 1
                if started and depth == 0:
                    return line_index, column
            column += 1
    return None


def _render_flow_fleet(
    lines: list[str],
    fleet_index: int,
    match: re.Match[str],
    retired: bool,
) -> list[str] | None:
    """Rewrite a flow-style ``fleet: {...}`` as a block mapping, keeping keys.

    Returns ``None`` when the flow text is not a mapping (``_set_retired``
    rejects that before rendering). The sibling keys and their order are
    preserved; the closing line's trailing comment moves to the ``fleet:``
    line.
    """
    value_start = match.start("rest") + (
        len(match.group("rest")) - len(match.group("rest").lstrip())
    )
    end = _flow_mapping_end(lines, fleet_index, value_start)
    if end is None:
        return None
    end_line, end_column = end
    chunks: list[str] = []
    for line_index in range(fleet_index, end_line + 1):
        start = value_start if line_index == fleet_index else 0
        stop = end_column + 1 if line_index == end_line else len(lines[line_index])
        chunks.append(lines[line_index][start:stop])
    try:
        mapping = yaml.safe_load("\n".join(chunks))
    except yaml.YAMLError:
        return None
    if not isinstance(mapping, dict):
        return None
    mapping["retired"] = retired
    dumped = yaml.safe_dump(mapping, sort_keys=False, default_flow_style=False)
    block = ["fleet:"] + [f"  {line}" for line in dumped.rstrip("\n").splitlines()]
    trailing = lines[end_line][end_column + 1 :].strip()
    if trailing.startswith("#"):
        block[0] = f"fleet:  {trailing}"
    lines[fleet_index : end_line + 1] = block
    return lines


def _render_retired(text: str, retired: bool) -> str:
    """Return *text* with only ``fleet.retired`` changed.

    A full ``yaml.safe_dump`` round-trip would delete every comment and
    reflow the operator's file. This edits the one line in place (or inserts
    the one line into an existing ``fleet:`` block) and leaves every other
    byte untouched. A flow-style mapping is rewritten as a block so sibling
    keys survive, and the file's newline style is preserved.

    ``_set_retired`` re-parses the result and refuses to write unless it
    yields ``{**data, "fleet": {**fleet, "retired": retired}}``, so a render
    that cannot be expressed surgically fails closed instead of corrupting
    the operator's file.
    """
    value = "true" if retired else "false"
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    fleet_index: int | None = None
    for index, line in enumerate(lines):
        match = _FLEET_LINE.match(line)
        if match is None:
            continue
        fleet_index = index
        rest = match.group("rest").strip()
        if rest.startswith("{") and _render_flow_fleet(lines, index, match, retired) is not None:
            return newline.join(lines) + newline
        # ``fleet: null`` / any other scalar cannot be edited in place;
        # replace the value with a block mapping. A plain inline comment
        # stays as written.
        if rest and not rest.startswith("#"):
            lines[index] = "fleet:"
        break

    if fleet_index is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(["fleet:", f"  retired: {value}"])
        return newline.join(lines) + newline

    block_end = fleet_index + 1
    while block_end < len(lines):
        candidate = lines[block_end]
        stripped = candidate.lstrip()
        if (
            candidate.strip()
            and not candidate.startswith((" ", "\t"))
            and not stripped.startswith("#")
        ):
            break
        # A comment at column 0 inside the block is still part of the block
        # for our purposes: comments carry no indentation requirement.
        block_end += 1
    insert_at = block_end
    while insert_at > fleet_index + 1 and (
        not lines[insert_at - 1].strip() or lines[insert_at - 1].lstrip().startswith("#")
    ):
        insert_at -= 1

    indent = "  "
    for line in lines[fleet_index + 1 : block_end]:
        if line.strip() and not line.lstrip().startswith("#"):
            indent = line[: len(line) - len(line.lstrip(" \t"))]
            break

    for index in range(fleet_index + 1, block_end):
        match = _RETIRED_LINE.match(lines[index])
        if match is None:
            continue
        comment = match.group("comment") or ""
        lines[index] = f"{match.group('indent')}retired: {value}{comment}"
        return newline.join(lines) + newline

    lines.insert(insert_at, f"{indent}retired: {value}")
    return newline.join(lines) + newline


def _set_retired(args: argparse.Namespace, *, retired: bool) -> CommandResult:
    repo_root = find_repo_root(
        getattr(args, "repo", None), explicit=getattr(args, "repo", None) is not None
    )
    path = _config_path(args, repo_root)
    try:
        text = _read_text(path)
        data = _parse_config(text, path)
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
    rendered = _render_retired(text, retired)
    # The single write point is also the single verification point (issue
    # #2754 rework): the surgical renderer must reproduce the loaded document
    # with exactly this one leaf changed. A duplicate ``retired:`` key, an
    # unhandled flow shape, or any other render that would not round-trip
    # fails closed -- we report the error and write nothing.
    expected = {**data, "fleet": {**fleet, "retired": retired}}
    try:
        reparsed = yaml.safe_load(rendered)
    except yaml.YAMLError as exc:
        return CommandResult(
            False,
            f"fleet {verb}: refusing to write {path}: rendered config is not valid YAML: {exc}",
            {},
        )
    if reparsed != expected:
        return CommandResult(
            False,
            (
                f"fleet {verb}: refusing to write {path}: editing fleet.retired "
                "would not preserve the rest of the config; fix the file by hand"
            ),
            {},
        )
    if not dry_run:
        # Bytes, not text: ``write_text_atomic`` would translate ``\n`` to the
        # platform newline and turn a rendered CRLF file into CRCRLF.
        write_bytes_atomic(path, rendered.encode("utf-8"))
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
