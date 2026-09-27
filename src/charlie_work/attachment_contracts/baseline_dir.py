"""On-disk baseline store: the per-entry ``.attachment-budgets/`` directory.

Issue #1839: the single ``.attachment-budgets.json`` document was a shared
append point -- two PRs touching DIFFERENT attachment points still edited
the same JSON ``entries`` array, so they conflicted on merge, and a
conflicting PR gets no ``pull_request`` CI run at all (the #1801 silent-
stall class; same defect as #1802's file-size/private-slug registries).

Layout (mirrors ``charlie_work.ratchet_baseline``'s per-entry-directory
precedent -- reimplemented rather than imported because this package ships
as a standalone wheel that must not import from outside
``charlie_work.attachment_contracts``, per issue #1544):

- ``meta.json`` -- every top-level document key EXCEPT ``entries``
  (``version``/``generated_by``/``generated_at``/``floor``, plus optional
  ``kind_stats`` and ``mode``). Carrying ``entries`` here would recreate
  the shared registry inside the directory, so a meta.json that has the
  key fails closed.
- ``entries/<file>/<kind>--<leaf>.json`` -- one self-describing entry
  object per baselined attachment point. ``<file>`` is the entry's host
  file path mirrored verbatim, so diffs group under the file they guard;
  ``<leaf>`` is the identity with its redundant ``<file>::`` prefix
  stripped (test-module identities read ``<file>::module``) and the
  remainder percent-escaped so no Windows-unsafe or ``/``-bearing
  character can reach the filesystem. The filename is DERIVED from the
  entry's ``(kind, file, identity)`` key -- a file whose content names a
  different key than its path fails closed.

Two PRs that touch different entries therefore touch different files and
merge cleanly; two PRs editing the SAME attachment point still meet in the
same file, which is the required control direction.

Reads fail closed on anything structurally off (``TamperError``): a missing
``meta.json``, a stray file under ``entries/`` that is not a ``.json``
entry, malformed JSON, a non-object entry, a path/content key mismatch, or
a duplicate ``(kind, file, identity)`` key. Writes are atomic per file
(temp + ``replace()``), skip byte-identical rewrites, delete entries absent
from the document, and prune emptied subdirectories (git cannot track
them).

Legacy fallback: pre-#1839 checkouts carry ``.attachment-budgets.json``.
``find_baseline`` prefers the directory and falls back to the file, and
``load``/``dump`` dispatch on the name so both layouts flow through the
same entry points; the single-file codec itself stays in ``baseline.py``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import quote

from charlie_work.attachment_contracts import baseline
from charlie_work.attachment_contracts.baseline import (
    BASELINE_DIRNAME,
    BASELINE_FILENAME,
    TamperError,
)
from charlie_work.attachment_contracts.model import BaselineEntry

META_FILENAME = "meta.json"
ENTRIES_DIRNAME = "entries"
ENTRY_SUFFIX = ".json"


def _escape_leaf(identity: str) -> str:
    """Percent-encode an identity leaf segment for the filesystem.

    ``quote(safe="")`` leaves only unreserved characters (``A-Z a-z 0-9
    _ . - ~``) -- every ``/``, ``\\``, ``:``, ``*``, ``?``, ``<``, ``>``,
    ``|``, ``"``, and ``%`` is encoded, so the result is a single safe
    filename component on any filesystem and the mapping is injective
    (distinct identities can never collide on one leaf name).
    """
    return quote(identity, safe="")


def _validate_entry_file(file: str) -> str:
    """Return ``file`` unchanged, or raise if it could escape ``entries/``.

    Entry ``file`` fields become real path segments under the baseline
    directory. Anything absolute, backslashed, colon-bearing, or containing
    ``.``/``..``/empty segments would let a crafted document write outside
    it (same key-safety contract as ``ratchet_baseline._validate_key``).
    """
    parts = file.split("/")
    if (
        not file
        or file.startswith("/")
        or "\\" in file
        or ":" in file
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise TamperError(
            f"invalid baseline entry file {file!r}: must be a relative POSIX "
            "path with no '.', '..', empty, backslash, or colon segments"
        )
    return file


def entry_relpath(entry: BaselineEntry) -> str:
    """Baseline-relative POSIX path of the file carrying *entry*.

    ``entries/<file>/<kind>--<leaf>.json`` where ``<leaf>`` is the identity
    minus a redundant ``<file>::`` prefix, percent-escaped.
    """
    _validate_entry_file(entry.file)
    identity = entry.identity
    prefix = entry.file + "::"
    leaf_identity = identity[len(prefix) :] if identity.startswith(prefix) else identity
    leaf = f"{entry.kind}--{_escape_leaf(leaf_identity)}{ENTRY_SUFFIX}"
    return f"{ENTRIES_DIRNAME}/{entry.file}/{leaf}"


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, indent=1, sort_keys=True) + "\n"


def _parse_json(text: str, *, name: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise TamperError(f"baseline file {name!r} is not valid JSON: {exc}") from exc


def document_files(document: dict[str, object]) -> dict[str, str]:
    """The exact ``{relative-path: text}`` payload a document writes to disk.

    ``meta.json`` carries every top-level key except ``entries``; each entry
    lands at its derived ``entries/...`` path serialized verbatim (unknown
    keys on an entry survive a round-trip, matching ``dumps()``'s passthrough
    of the raw entry dicts). Every payload is serialized BEFORE any caller
    writes, so a non-JSON-serializable document fails without a partial
    directory. No filesystem I/O -- used by ``dump_dir`` and available to
    tests/review code that need the file map without touching disk.
    """
    meta = {key: value for key, value in document.items() if key != "entries"}
    files: dict[str, str] = {META_FILENAME: _canonical_json(meta)}
    entries_raw = document.get("entries", [])
    if not isinstance(entries_raw, list):
        raise TamperError("baseline 'entries' must be a list")
    for raw in entries_raw:
        if not isinstance(raw, dict):
            raise TamperError(f"baseline entry must be an object, got {type(raw)!r}")
        rel = entry_relpath(baseline._entry_from_dict(raw))
        if rel in files:
            raise TamperError(f"duplicate baseline entry path {rel!r}")
        files[rel] = _canonical_json(raw)
    return files


def _write_entry_file(path: Path, payload: str) -> None:
    """Atomic temp-file + replace write; skips byte-identical rewrites.

    Leaving an unchanged entry's bytes (and mtime) alone keeps a routine
    ``--ratchet`` diff minimal -- only the entries that actually changed
    appear in the commit.
    """
    if path.is_file():
        try:
            if path.read_text(encoding="utf-8") == payload:
                return
        except (OSError, UnicodeDecodeError):
            pass  # unreadable or non-UTF-8 -- rewrite it below
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def _prune_empty_dirs(directory: Path) -> None:
    """Remove now-empty subdirectories under *directory* (deepest first).

    Mirrors ``ratchet_baseline._prune_empty_dirs``: git does not track empty
    directories, so a sync that deletes the last entry in a mirrored subdir
    must remove the husk or the tree accumulates untracked empties.
    """
    for path in sorted(
        (p for p in directory.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            path.rmdir()
        except OSError:
            pass


def dump_dir(document: dict[str, object], directory: Path) -> None:
    """Sync *directory* to exactly *document* -- create, update, delete.

    ``meta.json`` and each ``entries/`` file are written atomically; files
    under ``entries/`` absent from the document are deleted and emptied
    directories pruned. Top-level files that are not entries (e.g. a
    README.md) are left alone -- ``load_files`` ignores them, and
    ``load_dir`` only fails on files *inside* ``entries/``.
    """
    wanted = document_files(document)
    for rel in sorted(wanted):
        _write_entry_file(directory / Path(rel), wanted[rel])
    entries_root = directory / ENTRIES_DIRNAME
    if entries_root.is_dir():
        for path in sorted(p for p in entries_root.rglob("*") if p.is_file()):
            rel = path.relative_to(directory).as_posix()
            if rel not in wanted:
                path.unlink()
        _prune_empty_dirs(entries_root)
        if entries_root.is_dir() and not any(entries_root.iterdir()):
            entries_root.rmdir()


def read_files(directory: Path) -> dict[str, str]:
    """Every regular file under *directory* as ``{relative-path: text}``.

    The raw map the review packet feeds ``load_files`` after reconstructing
    PR-head content from a diff. Fails closed (``TamperError``) on a missing
    directory or an unreadable/non-UTF-8 member.
    """
    if not directory.is_dir():
        raise TamperError(f"baseline directory not found: {directory}")
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(directory).as_posix()
        try:
            files[rel] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise TamperError(f"unreadable baseline file {rel!r} in {directory}: {exc}") from exc
    return files


def load_files(files: Mapping[str, str]) -> dict[str, object]:
    """Reassemble and validate a baseline document from a file map.

    ``files`` maps paths relative to the baseline directory to file text
    (``read_files`` output, or the same shape reconstructed from a diff).
    ``meta.json`` supplies every top-level key; each ``entries/**/*.json``
    file contributes one entry. Anything structurally off -- missing meta,
    a stray non-``.json`` file under ``entries/``, malformed JSON, a
    non-object entry, a path whose content names a different key, or a
    duplicate entry -- raises ``TamperError`` rather than silently passing.
    """
    meta_text = files.get(META_FILENAME)
    if meta_text is None:
        raise TamperError(f"baseline directory is missing {META_FILENAME}")
    meta = _parse_json(meta_text, name=META_FILENAME)
    if not isinstance(meta, dict):
        raise TamperError(f"{META_FILENAME} must be a JSON object, got {type(meta)!r}")
    if "entries" in meta:
        raise TamperError(
            f"{META_FILENAME} must not carry an 'entries' key -- entries live in "
            f"per-entry files under {ENTRIES_DIRNAME}/"
        )
    entries_prefix = ENTRIES_DIRNAME + "/"
    raw_entries: list[dict[str, object]] = []
    for rel in sorted(files):
        if not rel.startswith(entries_prefix):
            continue
        if not rel.endswith(ENTRY_SUFFIX):
            raise TamperError(
                f"unexpected file {rel!r} in baseline entries: "
                f"entries must end with {ENTRY_SUFFIX}"
            )
        raw = _parse_json(files[rel], name=rel)
        if not isinstance(raw, dict):
            raise TamperError(
                f"baseline entry file {rel!r} must contain a JSON object, got {type(raw)!r}"
            )
        try:
            entry = baseline._entry_from_dict(raw)
        except TamperError as exc:
            raise TamperError(f"{rel}: {exc}") from exc
        expected = entry_relpath(entry)
        if rel != expected:
            raise TamperError(
                f"entry file {rel!r} does not match its own key -- expected "
                f"{expected!r} for (kind, file, identity) = "
                f"({entry.kind!r}, {entry.file!r}, {entry.identity!r})"
            )
        raw_entries.append(raw)
    return baseline._validate_document({**meta, "entries": raw_entries})


def load_dir(directory: Path) -> dict[str, object]:
    """Load a ``.attachment-budgets/`` directory into a baseline document."""
    return load_files(read_files(directory))


def find_baseline(root: Path) -> Path | None:
    """Locate a checkout's committed baseline: the directory, else the file.

    The ``.attachment-budgets/`` directory wins when both exist (the
    directory is the current layout; a leftover ``.attachment-budgets.json``
    beside it is stale). ``None`` when neither is present.
    """
    directory = root / BASELINE_DIRNAME
    if directory.is_dir():
        return directory
    legacy = root / BASELINE_FILENAME
    if legacy.is_file():
        return legacy
    return None


def load(path: Path) -> dict[str, object]:
    """Load a baseline from either on-disk layout, dispatched by name.

    ``<root>/.attachment-budgets`` -> the per-entry directory;
    anything else (i.e. ``<root>/.attachment-budgets.json``) -> the legacy
    single-file codec in ``baseline.py``.
    """
    if path.name == BASELINE_DIRNAME:
        return load_dir(path)
    return baseline.load(path)


def dump(document: dict[str, object], path: Path) -> None:
    """Write a baseline to either on-disk layout, dispatched by name.

    Same rule as ``load``: a path named ``.attachment-budgets`` is written
    as the directory layout, anything else as the single JSON file. Callers
    pass ``find_baseline()``'s result (preserving whatever layout exists) or
    ``root / BASELINE_DIRNAME`` for a fresh write.
    """
    if path.name == BASELINE_DIRNAME:
        dump_dir(document, path)
    else:
        baseline.dump(document, path)
