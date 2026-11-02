"""Atomic file writes: unique temp file + rename, shared by every writer.

Single point of enforcement for the repo's atomic-write invariant (CLAUDE.md:
"All JSON state writes are atomic"). Every call site used to hand-roll

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, ...)
    tmp.replace(path)

which gives every writer of a destination ONE shared temp name. Two processes
saving the same file at once -- a CLI call beside a loop pass, two repo
lanes, or any writer that does not share the readers' lock -- then open,
truncate, and replace the same tmp file, and on Windows ``replace()`` raises
``PermissionError`` [WinError 5] while another process holds the destination
or the tmp file open without ``FILE_SHARE_DELETE`` (issue #2265; observed on
``http-etag-cache.json``, where a fleet loop pass and a ``charlie verdict``
call lost memoized ETags to it).

Every write here gets:

* a UNIQUE temp name -- ``<stem>.<random><suffix>.tmp`` inside the
  destination's own directory -- so concurrent writers never share a tmp
  file. The name still ends ``<suffix>.tmp`` so the existing
  ``*.json.tmp`` orphan sweep (``adapters.cleanup_stale_session_tmp_files``)
  and ``*.md.tmp`` read exclusions keep matching a crash-stranded tmp;
* a bounded ``PermissionError`` retry on the rename -- a lock-free reader
  (``load_state``, a dashboard render) or an antivirus/indexer holding the
  destination open is a transient condition measured in tens of ms. Mirrors
  the budget ``state.save_state`` has used since issue #1062;
* unconditional tmp cleanup on failure, so failed writes do not pile up
  orphans. A leftover can now only come from process death mid-write --
  exactly what the orphan sweeps exist for.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import IO, Any, Callable

# Bounded retry for transient ``PermissionError`` on the atomic rename (see
# module docstring). ``state.save_state`` aliases ``REPLACE_ATTEMPTS`` for its
# operator-facing error message -- keep the values in sync through it rather
# than redeclaring a number here.
REPLACE_ATTEMPTS = 3
REPLACE_DELAY_SECONDS = 0.1


def _replace_with_retry(tmp_path: Path, path: Path) -> None:
    """Rename ``tmp_path`` onto ``path``, retrying transient PermissionError."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            tmp_path.replace(path)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_DELAY_SECONDS)


def _write_atomic(
    path: Path,
    mode: str,
    write: Callable[[IO[Any]], None],
    *,
    encoding: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem + ".", suffix=path.suffix + ".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        try:
            if "b" in mode:
                handle = os.fdopen(fd, mode)
            else:
                handle = os.fdopen(fd, mode, encoding=encoding)
        except OSError:
            os.close(fd)
            raise
        with handle:
            write(handle)
        _replace_with_retry(tmp_path, path)
    finally:
        # A consumed (replaced) tmp is already gone; anything still at
        # tmp_path is a failed write's orphan -- remove it rather than pile
        # up (issue #2265).
        try:
            tmp_path.unlink()
        except OSError:
            pass


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (unique temp file + rename).

    Binary mode -- no newline translation on Windows.
    """
    _write_atomic(path, "wb", lambda handle: handle.write(data))


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically (unique temp file + rename)."""
    _write_atomic(path, "w", lambda handle: handle.write(text), encoding=encoding)


def write_json_atomic(path: Path, value: Any) -> None:
    """Serialize ``value`` (``indent=2``, ``sort_keys``, trailing newline) atomically."""

    def _dump(handle: IO[Any]) -> None:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _write_atomic(path, "w", _dump, encoding="utf-8")


__all__ = ["write_bytes_atomic", "write_json_atomic", "write_text_atomic"]
