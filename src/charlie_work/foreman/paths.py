"""Path layout for the foreman store.

Every name under ``fleet_dir()/foremen/`` is spelled exactly once, here. Any
consumer that needs a foreman path asks this module; it validates the foreman
name, resolves both sides, and containment-checks the result through
``safe_path`` so a symlink or junction can never land the caller outside the
foreman store.

Layout (design §4)::

    foremen/
      _shared/            # prompt partials any foreman's instructions.md may include
        issue-style.md
        operator-facts.md
      <name>/
        charter.yaml      # the contract; only a human edits it
        instructions.md   # standing instructions for the model call
        memory/           # markdown notes the foreman keeps for itself
        state.json        # runtime cache; the file lock lives beside it
        journal.jsonl     # one dated line per Turn, append-only
        inbox.jsonl       # operator messages, append-only
        signals.jsonl     # external signals, append-only
        items.json        # typed decision/permission/report items
        turns/            # one directory per Turn
        tree/             # detached worktree of the registered repo
"""

from __future__ import annotations

import re
from pathlib import Path

from .. import layout
from ..safe_path import contains, require_contained

FOREMEN_DIRNAME = "foremen"
SHARED_DIRNAME = "_shared"

CHARTER_FILENAME = "charter.yaml"
INSTRUCTIONS_FILENAME = "instructions.md"
ITEMS_FILENAME = "items.json"
JOURNAL_FILENAME = "journal.jsonl"
INBOX_FILENAME = "inbox.jsonl"
SIGNALS_FILENAME = "signals.jsonl"
MEMORY_DIRNAME = "memory"
TURNS_DIRNAME = "turns"
TREE_DIRNAME = "tree"

#: A foreman name: lowercase alphanumeric, hyphens allowed after the first
#: character. The pattern deliberately excludes ``_``, so ``_shared`` can
#: never be mistaken for a foreman directory.
FOREMAN_NAME_RE = re.compile(r"[a-z0-9][a-z0-9-]*")


def is_foreman_name(name: str) -> bool:
    """True when ``name`` is a valid foreman name (full regex match)."""
    return FOREMAN_NAME_RE.fullmatch(name) is not None


def foremen_root(fleet_dir: Path) -> Path:
    """Return ``<fleet_dir>/foremen`` — the root every foreman lives under."""
    return Path(fleet_dir) / FOREMEN_DIRNAME


def shared_dir(fleet_dir: Path) -> Path:
    """Return ``<fleet_dir>/foremen/_shared`` — the shared-partials directory."""
    return foremen_root(fleet_dir) / SHARED_DIRNAME


def foreman_dir(fleet_dir: Path, name: str) -> Path:
    """Resolve the directory for foreman ``name``, containment-checked.

    Raises ``ValueError`` when the name fails the regex or the resolved path
    escapes ``foremen/`` (e.g. ``foremen/`` itself is a reparse point).
    Callers convert that to an error value at their own boundary.
    """
    if not is_foreman_name(name):
        raise ValueError(f"invalid foreman name {name!r}: must match {FOREMAN_NAME_RE.pattern!r}")
    return require_contained(
        foremen_root(fleet_dir), foremen_root(fleet_dir) / name, context="foreman_dir"
    )


def list_foreman_dirs(fleet_dir: Path) -> list[Path]:
    """Return the resolved directories of every foreman, sorted by name.

    Entries that fail the name regex (including ``_shared`` and stray files)
    are skipped; entries whose resolved path escapes ``foremen/`` are skipped
    too — they are silently foreign, not foremen.
    """
    root = foremen_root(fleet_dir)
    if not root.is_dir():
        return []
    found: list[Path] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir() or not is_foreman_name(entry.name):
            continue
        if not contains(root, entry):
            continue
        found.append(entry.resolve())
    return found


def _sub(foreman_root: Path, name: str) -> Path:
    return require_contained(foreman_root, foreman_root / name, context=f"foreman/{name}")


def memory_dir(foreman_root: Path) -> Path:
    return _sub(foreman_root, MEMORY_DIRNAME)


def memory_file(foreman_root: Path, name: str) -> Path:
    """Resolve a ``memory/<stem>.md`` path, containment-checked against ``memory/``.

    ``name`` may be given with or without the ``.md`` suffix. Anything that
    resolves outside ``memory/`` — a ``..`` segment, an absolute path, or a
    symlink/junction landing elsewhere — raises ``ValueError``.
    """
    stem = name[: -len(".md")] if name.endswith(".md") else name
    if not stem or stem in (".", "..") or "/" in stem or "\\" in stem:
        raise ValueError(f"invalid memory name {name!r}: a bare file stem, no separators")
    root = memory_dir(foreman_root)
    return require_contained(root, root / f"{stem}.md", context="memory_file")


def turn_dir(foreman_root: Path, turn_id: str) -> Path:
    """Resolve ``turns/<turn_id>``, containment-checked."""
    root = _sub(foreman_root, TURNS_DIRNAME)
    return require_contained(root, root / turn_id, context="turn_dir")


def charter_path(foreman_root: Path) -> Path:
    return _sub(foreman_root, CHARTER_FILENAME)


def instructions_path(foreman_root: Path) -> Path:
    return _sub(foreman_root, INSTRUCTIONS_FILENAME)


def state_path(foreman_root: Path) -> Path:
    return _sub(foreman_root, layout.STATE_FILENAME)


def items_path(foreman_root: Path) -> Path:
    return _sub(foreman_root, ITEMS_FILENAME)


def journal_path(foreman_root: Path) -> Path:
    return _sub(foreman_root, JOURNAL_FILENAME)


def inbox_path(foreman_root: Path) -> Path:
    return _sub(foreman_root, INBOX_FILENAME)


def signals_path(foreman_root: Path) -> Path:
    return _sub(foreman_root, SIGNALS_FILENAME)


def tree_path(foreman_root: Path) -> Path:
    return _sub(foreman_root, TREE_DIRNAME)
