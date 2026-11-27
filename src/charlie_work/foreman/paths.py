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
        check/            # transient done_when.check worktree (created and removed per run)
"""

from __future__ import annotations

import re
from pathlib import Path

from .. import layout
from ..safe_path import contains, require_contained

#: ``<fleet_dir>/foremen`` — owned by ``layout.py`` (the hub's fleet-dir
#: surface needs it to gate the ``charlie scheme`` spawn without importing
#: this package); aliased here so the package's own path helpers read the
#: same source of truth.
FOREMEN_DIRNAME = layout.FOREMEN_DIRNAME
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
CHECK_DIRNAME = "check"

ACTIONS_FILENAME = "actions.json"
APPLIED_FILENAME = "applied.jsonl"
BRIEFING_FILENAME = "briefing.md"
#: The Turn's bookkeeping sidecar (cursors, fingerprint, decide usage) — the
#: resume source when ``actions.json`` outlives its pass (issue #3122).
TURN_META_FILENAME = "turn.json"
#: Written only after a Turn's bookkeeping lands: journal line, cursor/fingerprint
#: state update. A ``turn-*`` dir holding ``actions.json`` without it is a
#: resume candidate.
COMPLETE_FILENAME = "complete.json"
#: Fleet-wide pass lock (``foremen/pass.lock``): at most one ``charlie scheme``
#: pass — and so at most one Decide — at a time.
PASS_LOCK_FILENAME = "pass.lock"
#: Per-foreman Turn lock (``foremen/<name>/turn.lock``): one foreman's Turns
#: never overlap, even across pass processes.
TURN_LOCK_FILENAME = "turn.lock"
#: Turn directory name prefixes — ``turn-`` for real Turns, ``dry-`` for
#: ``--dry-run`` runs so a dry plan is never a resume candidate.
TURN_DIR_PREFIX = "turn-"
DRY_TURN_DIR_PREFIX = "dry-"

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


def turns_root(foreman_root: Path) -> Path:
    """Resolve the ``turns/`` directory itself, containment-checked."""
    return _sub(foreman_root, TURNS_DIRNAME)


def turn_dir(foreman_root: Path, turn_id: str) -> Path:
    """Resolve ``turns/<turn_id>``, containment-checked."""
    root = _sub(foreman_root, TURNS_DIRNAME)
    return require_contained(root, root / turn_id, context="turn_dir")


def actions_path(foreman_root: Path, turn_id: str) -> Path:
    """Resolve ``turns/<turn_id>/actions.json`` — Decide's output, Apply's input."""
    return _sub(turn_dir(foreman_root, turn_id), ACTIONS_FILENAME)


def applied_path(foreman_root: Path, turn_id: str) -> Path:
    """Resolve ``turns/<turn_id>/applied.jsonl`` — Apply's per-action record."""
    return _sub(turn_dir(foreman_root, turn_id), APPLIED_FILENAME)


def briefing_path(foreman_root: Path, turn_id: str) -> Path:
    """Resolve ``turns/<turn_id>/briefing.md`` — Collect's document."""
    return _sub(turn_dir(foreman_root, turn_id), BRIEFING_FILENAME)


def turn_meta_path(foreman_root: Path, turn_id: str) -> Path:
    """Resolve ``turns/<turn_id>/turn.json`` — the resume/bookkeeping sidecar."""
    return _sub(turn_dir(foreman_root, turn_id), TURN_META_FILENAME)


def complete_path(foreman_root: Path, turn_id: str) -> Path:
    """Resolve ``turns/<turn_id>/complete.json`` — the Turn-finished marker."""
    return _sub(turn_dir(foreman_root, turn_id), COMPLETE_FILENAME)


def pass_lock_path(fleet_dir: Path) -> Path:
    """Resolve ``foremen/pass.lock`` — the fleet-wide foreman-pass lock file."""
    return foremen_root(fleet_dir) / PASS_LOCK_FILENAME


def turn_lock_path(foreman_root: Path) -> Path:
    """Resolve ``foremen/<name>/turn.lock`` — the per-foreman Turn lock file."""
    return _sub(foreman_root, TURN_LOCK_FILENAME)


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


def done_check_path(foreman_root: Path) -> Path:
    """Resolve ``check/`` — the transient worktree ``done_when.check`` runs in."""
    return _sub(foreman_root, CHECK_DIRNAME)
