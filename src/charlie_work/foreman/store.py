"""Foreman discovery and scaffolding.

``list_foremen`` is the read every other subsystem starts from: it returns
the directories under ``foremen/`` that pass the name check and the
containment check, sorted. ``scaffold_foreman`` is what ``charlie gang new``
calls — it creates the one directory a foreman needs to exist
(``charter.yaml``, ``instructions.md``, ``memory/``, ``turns/``, a paused
``state.json``), either from built-in minimal documents or copied from a
``--from`` directory.

The fleet registry is read straight from ``fleet.json`` — a foreman's
``repos`` entry must name a registered key, so the keys are all this layer
needs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import layout
from ..atomic_write import write_text_atomic
from . import Result
from .paths import (
    CHARTER_FILENAME,
    FOREMEN_DIRNAME,
    INSTRUCTIONS_FILENAME,
    TURNS_DIRNAME,
    foreman_dir,
    foremen_root,
    list_foreman_dirs,
    memory_dir,
)
from .state import ForemanState, save_state

_CHARTER_TEMPLATE = """\
# The charter for foreman "{name}". Only a human edits this file.
name: {name}
outcome: >-
  TODO: the single outcome this foreman owns, in one sentence.
repos: ["<owner>/<repo>"]   # exactly one fleet registry key
standing: false
arming: auto                # auto | propose
bounds:
  max_open_issues: 5
  max_issues_per_turn: 3
  min_turn_interval_minutes: 30
  max_quiet_hours: 24
  max_turns_per_day: 12
  max_unanswered_days: 7
# done_when:
#   check: ["uv", "run", "--extra", "dev", "pytest", "-q"]
# cadence:
#   report_at: "09:00"      # local HH:MM
"""

_INSTRUCTIONS_TEMPLATE = """\
# Instructions for {name}

Standing instructions for this foreman's model call. Shared partials under
`foremen/_shared/*.md` can be included here as `$$section_<stem>`.
"""


def registered_repo_keys(fleet_dir: Path) -> tuple[str, ...]:
    """The repo keys of the fleet registry (``fleet.json``), sorted.

    A missing or corrupt registry yields no keys — charter validation then
    reports the charter's repo as unregistered rather than crashing.
    """
    path = Path(fleet_dir) / layout.FLEET_REGISTRY_FILENAME
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    repos = data.get("repos") if isinstance(data, dict) else None
    if not isinstance(repos, dict):
        return ()
    return tuple(sorted(str(key) for key in repos))


def list_foremen(fleet_dir: Path) -> tuple[tuple[str, Path], ...]:
    """Every foreman as ``(name, resolved_dir)``, sorted by name."""
    return tuple((path.name, path) for path in list_foreman_dirs(fleet_dir))


def resolve_foreman(fleet_dir: Path, name: str) -> Result[Path]:
    """Resolve ``foremen/<name>``, requiring it to exist as a directory."""
    try:
        path = foreman_dir(fleet_dir, name)
    except ValueError as exc:
        return Result(error=str(exc))
    if not path.is_dir():
        return Result(error=f"foreman {name!r}: not found at {path}")
    return Result(value=path)


def scaffold_foreman(
    fleet_dir: Path,
    name: str,
    *,
    from_dir: Path | None = None,
) -> Result[Path]:
    """Create ``foremen/<name>/`` with a charter, instructions, and paused state.

    ``from_dir``, when given, supplies ``charter.yaml`` and
    ``instructions.md`` to copy verbatim; otherwise built-in minimal
    documents are written. Refuses an existing foreman or an invalid name.
    """
    try:
        root = foreman_dir(fleet_dir, name)
    except ValueError as exc:
        return Result(error=str(exc))
    if root.exists():
        return Result(error=f"foreman {name!r}: already exists at {root}")
    # Read and decode --from sources BEFORE any mkdir: a seed that cannot be
    # read or decoded is an error value, never a half-scaffolded charterless
    # foreman that blocks retry (a UnicodeDecodeError out of read_text is not
    # an OSError — it must be named here).
    copied: dict[str, str] | None = None
    if from_dir is not None:
        copied = {}
        for filename in (CHARTER_FILENAME, INSTRUCTIONS_FILENAME):
            source = from_dir / filename
            if not source.is_file():
                return Result(error=f"--from {from_dir}: {filename} not found")
            try:
                copied[filename] = source.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                return Result(error=f"--from {from_dir}: {filename}: unreadable: {exc}")
    try:
        foremen_root(fleet_dir).mkdir(parents=True, exist_ok=True)
        root.mkdir()
        memory_dir(root).mkdir()
        (root / TURNS_DIRNAME).mkdir()
    except OSError as exc:
        return Result(error=f"foreman {name!r}: cannot create directories: {exc}")
    try:
        if copied is not None:
            for filename in (CHARTER_FILENAME, INSTRUCTIONS_FILENAME):
                write_text_atomic(root / filename, copied[filename])
        else:
            write_text_atomic(root / CHARTER_FILENAME, _CHARTER_TEMPLATE.format(name=name))
            write_text_atomic(
                root / INSTRUCTIONS_FILENAME, _INSTRUCTIONS_TEMPLATE.format(name=name)
            )
        # The foreman-state store is not the orchestrator's gated mutator layer —
        # write-gate-exempt(issue=3117): name-matched only, no write_gate receiver here.
        saved = save_state(root, ForemanState())
        if not saved.ok:
            return Result(error=saved.error)
    except OSError as exc:
        return Result(error=f"foreman {name!r}: scaffold failed: {exc}")
    return Result(value=root)


__all__ = [
    "FOREMEN_DIRNAME",
    "list_foremen",
    "registered_repo_keys",
    "resolve_foreman",
    "scaffold_foreman",
]
