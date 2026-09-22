"""Per-entry ratchet baseline directories (issue #1802).

A ratchet baseline is a *directory of small per-entry files* rather than one
shared JSON document. The single-file shape was a shared append point: two
concurrent PRs that each touched a DIFFERENT entry still edited the same file,
so they conflicted on merge (and a CONFLICTING PR gets no pull_request CI run
at all -- it reads as "CI is stuck"). With one file per entry, distinct-entry
PRs add distinct files and merge cleanly; two PRs raising the SAME entry to
different values still edit the same file and conflict (or fail the guard),
which is the required control direction.

Two entry kinds share the shape:

* **Counted entries** -- ``<key>.count`` files. The first line is a
  non-negative decimal integer (the recorded mark/count); every remaining
  line is blank or a ``#`` comment, which is where the per-entry audit trail
  the monolithic baselines carried as inline comments now lives.
* **Set entries** -- marker files under a set directory. Presence means
  membership; the file's content is freeform rationale text the loader
  ignores.

Entry keys mirror the guarded path's own repo-relative layout
(``files/src/charlie_work/cli.py.count`` records the count for
``src/charlie_work/cli.py``), so diffs read naturally and no name-encoding
scheme is needed. The ``.count`` suffix also keeps entry files out of every
``*.py``/``test_*.py`` consumer (``git ls-files '*.py'``, ruff, pytest
collection) even though the key itself ends in ``.py``.

Guards fail closed: a missing baseline directory, a file in a counted
baseline that does not end in ``.count``, or an entry whose first line is
not a bare integer all raise :class:`BaselineFormatError` rather than
silently passing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path

COUNT_SUFFIX = ".count"

_COUNT_LINE = re.compile(r"[0-9]+")


class BaselineFormatError(ValueError):
    """A baseline directory or one of its entries is missing or malformed.

    Raised instead of returning a partial/degenerate baseline so a corrupt
    baseline fails the guard that consumes it rather than silently passing.
    """


def _validate_key(key: str, *, owner: str) -> str:
    """Return *key* unchanged, or raise if it could escape the baseline dir.

    Keys are POSIX-style relative paths ("a/b.py"). Anything absolute,
    backslashed, or containing ``.``/``..``/empty segments would let a
    crafted mapping write outside the baseline directory.
    """
    parts = key.split("/")
    if (
        not key
        or key.startswith("/")
        or "\\" in key
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise BaselineFormatError(
            f"invalid baseline entry key {key!r} for {owner}: keys are "
            "relative POSIX paths with no '.', '..', empty, or backslash segments"
        )
    return key


def _entry_files(directory: Path) -> list[Path]:
    """Every regular file under *directory*, sorted. Raises when missing."""
    if not directory.is_dir():
        raise BaselineFormatError(f"baseline directory not found: {directory}")
    return sorted(p for p in directory.rglob("*") if p.is_file())


def _prune_empty_dirs(directory: Path) -> None:
    """Remove now-empty subdirectories under *directory* (deepest first).

    Git does not track empty directories, so a sync that deletes the last
    entry in a mirrored subdir must also remove the husk or the directory
    accumulates untracked empties.
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


def parse_count_text(text: str, *, name: str) -> int:
    """Parse one ``.count`` entry body: integer first line, ``#`` comments after.

    *name* is the entry's relative path, used only in error messages.
    """
    lines = text.splitlines()
    if not lines or not _COUNT_LINE.fullmatch(lines[0]):
        raise BaselineFormatError(
            f"malformed count entry {name!r}: first line must be a non-negative integer"
        )
    for line in lines[1:]:
        if line and not line.startswith("#"):
            raise BaselineFormatError(
                f"malformed count entry {name!r}: line {line!r} is neither blank nor a '#' comment"
            )
    return int(lines[0])


def load_count_baseline(directory: Path) -> dict[str, int]:
    """Load a counted baseline directory into ``{key: count}``.

    Every regular file under *directory* must be a ``.count`` entry; a stray
    file of any other name fails closed. The key is the file's POSIX
    relative path with the suffix stripped.
    """
    counts: dict[str, int] = {}
    for entry in _entry_files(directory):
        rel = entry.relative_to(directory).as_posix()
        if not rel.endswith(COUNT_SUFFIX):
            raise BaselineFormatError(
                f"unexpected file {rel!r} in count baseline {directory}: "
                f"entries must end with {COUNT_SUFFIX}"
            )
        try:
            text = entry.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise BaselineFormatError(
                f"unreadable count entry {rel!r} in {directory}: {exc}"
            ) from exc
        counts[rel[: -len(COUNT_SUFFIX)]] = parse_count_text(text, name=rel)
    return counts


def load_set_baseline(directory: Path) -> frozenset[str]:
    """Load a set baseline directory into the frozenset of member names.

    Every regular file under *directory* is a member whose name is its POSIX
    relative path; file contents are freeform rationale and ignored.
    """
    return frozenset(entry.relative_to(directory).as_posix() for entry in _entry_files(directory))


def _write_entry_file(path: Path, payload: str) -> None:
    """Atomic temp-file + replace write, per the repo's state-write invariant."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def write_count_baseline(directory: Path, counts: Mapping[str, int]) -> None:
    """Sync *directory* to exactly *counts* -- create, update, and delete.

    An entry whose recorded value already equals the target is left
    byte-identical (its comment lines and its mtime are preserved). On a
    value change the first line is rewritten and any ``#`` comment lines
    carry over, so hand-added audit notes survive a scripted refresh.
    Entries absent from *counts* are deleted and emptied parent directories
    pruned. Non-entry files (not ending in ``.count``) are left alone --
    ``load_count_baseline`` fails on them, so they cannot hide.
    """
    wanted = {_validate_key(k, owner=str(directory)): int(v) for k, v in counts.items()}
    for key, value in wanted.items():
        path = directory / (key + COUNT_SUFFIX)
        comments: list[str] = []
        if path.is_file():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                lines = []
            if lines and lines[0] == str(value):
                continue
            comments = [line for line in lines[1:] if not line or line.startswith("#")]
        payload = f"{value}\n" + ("\n".join(comments) + "\n" if comments else "")
        _write_entry_file(path, payload)

    if directory.is_dir():
        for entry in _entry_files(directory):
            rel = entry.relative_to(directory).as_posix()
            if rel.endswith(COUNT_SUFFIX) and rel[: -len(COUNT_SUFFIX)] not in wanted:
                entry.unlink()
        _prune_empty_dirs(directory)


def write_set_baseline(directory: Path, members: Iterable[str]) -> None:
    """Sync *directory* to exactly *members* -- create missing markers, drop stale.

    An existing member file is left untouched (its rationale content is
    preserved); only absent members get a new empty marker file.
    """
    wanted = {_validate_key(m, owner=str(directory)) for m in members}
    for member in wanted:
        path = directory / member
        if not path.exists():
            _write_entry_file(path, "")
    if directory.is_dir():
        for entry in _entry_files(directory):
            rel = entry.relative_to(directory).as_posix()
            if rel not in wanted:
                entry.unlink()
        _prune_empty_dirs(directory)


def count_delta_in_diff(diff_text: str, directory_prefix: str) -> int:
    """Net change of ``.count`` entries under *directory_prefix* in a diff.

    Parses *diff_text* (``git diff`` output) and sums ``+N`` minus ``-N``
    over lines that are a bare non-negative integer in files under the
    repo-relative *directory_prefix* whose name ends in ``.count``. Because
    a well-formed count entry carries its integer on line 1 and only
    ``#`` comments after it, a bare-integer added/removed line IS the
    count change: a comment-only edit contributes 0, a ``10 -> 11`` bump
    contributes ``+1``, a new entry contributes its value, a deleted entry
    subtracts it. Callers enforce well-formedness at the guard side by
    loading the head baseline strictly; a malformed entry at the *base*
    side simply contributes its own (mis)parseable delta, which fails the
    bump check rather than hiding.
    """
    prefix = directory_prefix.rstrip("/") + "/"
    old_path = ""
    new_path = ""
    delta = 0

    def _under(path: str) -> bool:
        return path.startswith(prefix) and path.endswith(COUNT_SUFFIX)

    for line in diff_text.splitlines():
        if line.startswith("--- "):
            rest = line[4:]
            if rest == "/dev/null":
                old_path = ""
            elif rest.startswith("a/"):
                old_path = rest[2:]
            else:
                old_path = rest
            continue
        if line.startswith("+++ "):
            rest = line[4:]
            if rest == "/dev/null":
                new_path = ""
            elif rest.startswith("b/"):
                new_path = rest[2:]
            else:
                new_path = rest
            continue
        if line.startswith("+"):
            if _under(new_path) and _COUNT_LINE.fullmatch(line[1:] or ""):
                delta += int(line[1:])
            continue
        if line.startswith("-"):
            if _under(old_path) and _COUNT_LINE.fullmatch(line[1:] or ""):
                delta -= int(line[1:])
            continue
    return delta
