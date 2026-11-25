"""Bounded on-disk ETag cache for the pooled HTTP transport (issue #1834).

Every REST GET the HTTP transport sends carries `If-None-Match` when a prior
response for the exact same path was cached, so an unchanged resource comes
back as a cheap `304 Not Modified` instead of re-transferring the full body
-- the same conditional-GET behavior `gh` itself does not do (it never
persists ETags across invocations), so this is a genuine improvement over
the `gh` subprocess path rather than parity work.

On-disk layout (issue #3138): the cost of one operation is proportional to
one entry, never to the whole cache.

* The **index** is a small JSON file next to `state.json` (`"version": 2`).
  An index entry holds `etag`, `status`, `stored_at`, `link` and the name and
  byte length of its body file -- never the body.
* Each **body** is one file in a sibling directory (`<index name>.d/`), named
  by a hash of the request path *and* the ETag. The name is therefore unique
  to one version of one resource: a reader that holds an older index still
  finds the body that belongs to the ETag it read, or finds nothing.

List endpoints return bodies of 1-2.5 MB per page. Version 1 kept every body
inline in the one JSON file, parsed the whole file on every GET and rewrote
the whole file on every fresh 200; a few dozen list pages made that a 76 MB
parse per request and a 76 MB rewrite per changed page.

Reads are lock-free. `get_cached` reads the index -- memoised per process,
keyed by the index file's `(st_mtime_ns, st_size)` -- and exactly one body
file. Writes are a locked read-modify-write (`state.advisory_file_lock` on
the index path): write the new body file, then the index, then remove body
files the index no longer names. Every write goes through `atomic_write`
(CLAUDE.md's atomic-write invariant), so a concurrent reader never sees a
torn file.

Bounded two ways: `record_response` evicts least-recently-stored entries
until at most `_MAX_ENTRIES` remain and their bodies total at most
`_MAX_BODY_BYTES`.

Fail-closed like every cache in this repo: a missing, unreadable, or
structurally-corrupt index, a version 1 file, or a missing or short body
file all degrade to "no entry cached" (an unconditional GET), never to a
wrong or stale body. A write failure is best-effort -- logged and swallowed
-- since a lost cache entry only costs one avoidable full transfer next
call, never a correctness problem.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..atomic_write import write_bytes_atomic, write_json_atomic
from ..state import StateLockBusy, advisory_file_lock, utc_now

logger = logging.getLogger(__name__)

_INDEX_VERSION = 2

# Bounds the index to a small, predictable size regardless of how many
# distinct REST paths a long-running `charlie fleet supervise` process
# accumulates over its lifetime.
_MAX_ENTRIES = 500

# Bounds the body directory. Sized well above one busy repo's live set of
# list pages (about 80 MB observed) so the pages a pass re-reads every time
# are not evicted between passes.
_MAX_BODY_BYTES = 256 * 1024 * 1024

_BODY_SUFFIX = ".body"
# Bodies are GitHub JSON, valid UTF-8 in practice; surrogatepass keeps the
# round trip exact for anything the decoder upstream let through.
_BODY_ERRORS = "surrogatepass"


@dataclass(frozen=True)
class CachedResponse:
    """One memoized REST GET response, keyed by request path.

    ``link`` is the response's ``Link`` header when it carried one: a 304
    need not repeat it (RFC 9110 15.4.5), and ``paginate_rest`` reads the
    served response's headers to keep following pages.
    """

    etag: str
    status: int
    body: str
    stored_at: str
    link: str | None = None


@dataclass(frozen=True)
class _IndexEntry:
    """One index row: everything about a cached response except its body."""

    etag: str
    status: int
    stored_at: str
    body_file: str
    body_bytes: int
    link: str | None = None


_Index = Mapping[str, _IndexEntry]

# Parsed index per cache path, valid while the file's (st_mtime_ns, st_size)
# is unchanged. A run of GETs in one pass parses the index once.
_INDEX_MEMO: dict[Path, tuple[tuple[int, int], _Index]] = {}
_INDEX_MEMO_LOCK = threading.Lock()


def _cache_key(path: str) -> str:
    return path


def _bodies_dir(cache_path: Path) -> Path:
    """Directory holding one file per cached body, sibling of the index."""
    return cache_path.with_name(cache_path.name + ".d")


def _body_file_name(key: str, etag: str) -> str:
    digest = hashlib.sha256(f"{key}\0{etag}".encode("utf-8", _BODY_ERRORS)).hexdigest()
    return digest[:40] + _BODY_SUFFIX


def _parse_index(cache_path: Path) -> _Index:
    """Parse the index file. `{}` on anything but a well-formed version 2 file."""
    try:
        with cache_path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, LookupError, ValueError):
        logger.warning(
            "HTTP ETag cache %s unreadable; treating as empty (every request "
            "this pass sends an unconditional GET)",
            cache_path,
        )
        return {}
    if not isinstance(data, dict) or data.get("version") != _INDEX_VERSION:
        # A version 1 file (bodies inline) is not migrated: it reads as empty
        # and the next record replaces it.
        return {}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        return {}
    result: dict[str, _IndexEntry] = {}
    for key, value in entries.items():
        if not isinstance(value, dict):
            continue
        etag = value.get("etag")
        status = value.get("status")
        stored_at = value.get("stored_at")
        body_file = value.get("body_file")
        body_bytes = value.get("body_bytes")
        link = value.get("link")
        if not isinstance(etag, str) or not isinstance(status, int):
            continue
        if not isinstance(stored_at, str) or not isinstance(body_bytes, int):
            continue
        # The name is written by this module; anything that is not a bare
        # file name of the expected shape is corruption, never a path to follow.
        if not isinstance(body_file, str) or not _is_body_file_name(body_file):
            continue
        result[str(key)] = _IndexEntry(
            etag=etag,
            status=status,
            stored_at=stored_at,
            body_file=body_file,
            body_bytes=body_bytes,
            link=link if isinstance(link, str) else None,
        )
    return result


def _is_body_file_name(name: str) -> bool:
    stem = name.removesuffix(_BODY_SUFFIX)
    return name.endswith(_BODY_SUFFIX) and len(stem) == 40 and stem.isalnum()


def _load_index(cache_path: Path) -> _Index:
    """Return the index, re-parsing only when the file changed on disk."""
    try:
        stat = cache_path.stat()
    except OSError:
        return {}
    signature = (stat.st_mtime_ns, stat.st_size)
    with _INDEX_MEMO_LOCK:
        memo = _INDEX_MEMO.get(cache_path)
    if memo is not None and memo[0] == signature:
        return memo[1]
    index = _parse_index(cache_path)
    with _INDEX_MEMO_LOCK:
        _INDEX_MEMO[cache_path] = (signature, index)
    return index


def _read_body(cache_path: Path, entry: _IndexEntry) -> str | None:
    """Read one entry's body, or None when its file is missing or not intact."""
    try:
        raw = (_bodies_dir(cache_path) / entry.body_file).read_bytes()
    except OSError:
        return None
    if len(raw) != entry.body_bytes:
        return None
    try:
        return raw.decode("utf-8", _BODY_ERRORS)
    except UnicodeDecodeError:
        return None


def _to_response(entry: _IndexEntry, body: str) -> CachedResponse:
    return CachedResponse(
        etag=entry.etag,
        status=entry.status,
        body=body,
        stored_at=entry.stored_at,
        link=entry.link,
    )


def load_cache(cache_path: Path) -> dict[str, CachedResponse]:
    """Load every cached response, bodies included.

    Returns `{}` (every path a miss) on any missing/unreadable/wrong-shaped
    index -- see module docstring. Reads every body file, so it is for
    inspection and tests; the transport's per-request path is `get_cached`.
    """
    result: dict[str, CachedResponse] = {}
    for key, entry in _load_index(cache_path).items():
        body = _read_body(cache_path, entry)
        if body is not None:
            result[key] = _to_response(entry, body)
    return result


def _save_index(cache_path: Path, index: _Index) -> None:
    """Atomically persist the index (``atomic_write.write_json_atomic``).

    Unique per-writer temp name + bounded ``PermissionError`` retry on the
    rename (issue #2265): a fixed shared ``.tmp`` name let concurrent saves
    collide, and a lock-free reader holding the destination open makes the
    rename transiently fail on Windows.
    """
    payload = {
        "version": _INDEX_VERSION,
        "entries": {
            key: {
                "etag": entry.etag,
                "status": entry.status,
                "stored_at": entry.stored_at,
                "link": entry.link,
                "body_file": entry.body_file,
                "body_bytes": entry.body_bytes,
            }
            for key, entry in index.items()
        },
    }
    write_json_atomic(cache_path, payload)


def _evict(index: _Index, keep: str) -> _Index:
    """Drop least-recently-stored entries until both bounds hold.

    ``keep`` (the entry just recorded) is never evicted, so one body larger
    than the whole byte budget is still cached on its own.
    """
    survivors = dict(index)
    total_bytes = sum(entry.body_bytes for entry in survivors.values())
    oldest_first = sorted(
        (key for key in survivors if key != keep), key=lambda k: survivors[k].stored_at
    )
    for key in oldest_first:
        if len(survivors) <= _MAX_ENTRIES and total_bytes <= _MAX_BODY_BYTES:
            break
        total_bytes -= survivors.pop(key).body_bytes
    return survivors


def _remove_unreferenced_bodies(cache_path: Path, index: _Index) -> None:
    """Delete body files the index no longer names (best-effort).

    Covers evicted entries, a superseded version of a re-recorded path, and
    temp files a crashed writer left behind. Only called under the write
    lock, so no writer has a body in flight. A leftover file is harmless; a
    reader that still has one open keeps it (the unlink fails on Windows and
    the next record retries).
    """
    referenced = {entry.body_file for entry in index.values()}
    try:
        present = list(_bodies_dir(cache_path).iterdir())
    except OSError:
        return
    for path in present:
        if path.name in referenced:
            continue
        try:
            path.unlink()
        except OSError:
            continue


def get_cached(cache_path: Path, path: str) -> CachedResponse | None:
    """Return the memoized response for `path`, or None on a cache miss.

    Opens the index (memoised) and this path's body file, nothing else.
    """
    entry = _load_index(cache_path).get(_cache_key(path))
    if entry is None:
        return None
    body = _read_body(cache_path, entry)
    if body is None:
        return None
    return _to_response(entry, body)


def record_response(
    cache_path: Path,
    path: str,
    *,
    etag: str,
    status: int,
    body: str,
    link: str | None = None,
) -> None:
    """Durably record a fresh `(etag, status, body)` for `path`.

    Locked read-modify-write, mirroring `queue_sync_coverage_cache.record_covered`:
    concurrent passes recording different paths cannot lose one another's
    entry. Writes this path's body file and the index; no other path's body
    file is touched. Best-effort in the fail-safe direction -- `StateLockBusy`
    and `OSError` are caught, logged, and swallowed; a skipped write only
    costs one avoidable full transfer on the next call to this path.
    """
    key = _cache_key(path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with advisory_file_lock(cache_path):
            raw = body.encode("utf-8", _BODY_ERRORS)
            body_file = _body_file_name(key, etag)
            write_bytes_atomic(_bodies_dir(cache_path) / body_file, raw)
            recorded = _IndexEntry(
                etag=etag,
                status=status,
                stored_at=utc_now(),
                body_file=body_file,
                body_bytes=len(raw),
                link=link,
            )
            index = _evict({**_load_index(cache_path), key: recorded}, keep=key)
            _save_index(cache_path, index)
            _remove_unreferenced_bodies(cache_path, index)
    except (StateLockBusy, OSError) as exc:
        logger.warning(
            "HTTP ETag cache write failed at %s; skipping memoization of %s "
            "(next call to this path sends an unconditional GET) (%s)",
            cache_path,
            path,
            exc,
        )


__all__ = ["CachedResponse", "get_cached", "load_cache", "record_response"]


class FileEtagCache:
    """The transport's ``EtagCache`` port over the on-disk conditional-GET file."""

    def __init__(self, cache_path: Path) -> None:
        self._cache_path = cache_path

    def get(self, path: str) -> CachedResponse | None:
        return get_cached(self._cache_path, path)

    def record(
        self, path: str, *, etag: str, status: int, body: str, link: str | None = None
    ) -> None:
        record_response(self._cache_path, path, etag=etag, status=status, body=body, link=link)
