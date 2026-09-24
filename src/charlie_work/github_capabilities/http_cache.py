"""Bounded on-disk ETag cache for the pooled HTTP transport (issue #1834).

Every REST GET the HTTP transport sends carries `If-None-Match` when a prior
response for the exact same path was cached, so an unchanged resource comes
back as a cheap `304 Not Modified` instead of re-transferring the full body
-- the same conditional-GET behavior `gh` itself does not do (it never
persists ETags across invocations), so this is a genuine improvement over
the `gh` subprocess path rather than parity work.

Modeled directly on `queue_sync_coverage_cache.py`'s shape: a small JSON file
next to `state.json`, written via temp-file + `replace()` (CLAUDE.md's
atomic-write invariant), with the read-modify-write on the write path
serialized by `state.advisory_file_lock`. Reads are lock-free -- an atomic
`replace()` never exposes a torn file to a concurrent reader.

Bounded (unlike the queue-sync cache, which never evicts): a fleet host
issues an effectively unbounded variety of REST paths over the life of a
repo (per-PR/per-issue/per-commit endpoints), so an unbounded cache would
grow forever. `record_response` evicts the least-recently-stored entry once
the map exceeds `_MAX_ENTRIES` on write, keeping the file small and bounding
lock hold time.

Fail-closed like every cache in this repo: a missing, unreadable, or
structurally-corrupt file degrades to "no entry cached" (an unconditional
GET), never to a wrong or stale ETag. A write failure is best-effort --
logged and swallowed -- since a lost cache entry only costs one avoidable
full transfer next call, never a correctness problem.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..state import StateLockBusy, advisory_file_lock, utc_now

logger = logging.getLogger(__name__)

# Bounds the cache file to a small, predictable size regardless of how many
# distinct REST paths a long-running `charlie fleet supervise` process
# accumulates over its lifetime.
_MAX_ENTRIES = 500


@dataclass(frozen=True)
class CachedResponse:
    """One memoized REST GET response, keyed by request path."""

    etag: str
    status: int
    body: str
    stored_at: str


def _cache_key(path: str) -> str:
    return path


def load_cache(cache_path: Path) -> dict[str, CachedResponse]:
    """Load the ETag cache. Returns `{}` (every path a miss) on any
    missing/unreadable/wrong-shaped file -- see module docstring.
    """
    if not cache_path.exists():
        return {}
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
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return {}
    result: dict[str, CachedResponse] = {}
    for key, value in entries.items():
        if not isinstance(value, dict):
            continue
        etag = value.get("etag")
        status = value.get("status")
        body = value.get("body")
        stored_at = value.get("stored_at")
        if not isinstance(etag, str) or not isinstance(status, int):
            continue
        if not isinstance(body, str) or not isinstance(stored_at, str):
            continue
        result[str(key)] = CachedResponse(etag=etag, status=status, body=body, stored_at=stored_at)
    return result


def _save_cache(cache_path: Path, entries: dict[str, CachedResponse]) -> None:
    """Atomically persist the cache (temp-file + `replace()`)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "entries": {
            key: {
                "etag": entry.etag,
                "status": entry.status,
                "body": entry.body,
                "stored_at": entry.stored_at,
            }
            for key, entry in entries.items()
        },
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(cache_path)


def get_cached(cache_path: Path, path: str) -> CachedResponse | None:
    """Return the memoized response for `path`, or None on a cache miss."""
    return load_cache(cache_path).get(_cache_key(path))


def record_response(cache_path: Path, path: str, *, etag: str, status: int, body: str) -> None:
    """Durably record a fresh `(etag, status, body)` for `path`.

    Locked read-modify-write, mirroring `queue_sync_coverage_cache.record_covered`:
    concurrent passes recording different paths cannot lose one another's
    entry. Best-effort in the fail-safe direction -- `StateLockBusy` and
    `OSError` are caught, logged, and swallowed; a skipped write only costs
    one avoidable full transfer on the next call to this path.
    """
    key = _cache_key(path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with advisory_file_lock(cache_path):
            entries = load_cache(cache_path)
            entries[key] = CachedResponse(etag=etag, status=status, body=body, stored_at=utc_now())
            if len(entries) > _MAX_ENTRIES:
                oldest_key = min(entries, key=lambda k: entries[k].stored_at)
                if oldest_key != key:
                    del entries[oldest_key]
            _save_cache(cache_path, entries)
    except (StateLockBusy, OSError) as exc:
        logger.warning(
            "HTTP ETag cache write failed at %s; skipping memoization of %s "
            "(next call to this path sends an unconditional GET) (%s)",
            cache_path,
            path,
            exc,
        )


__all__ = ["CachedResponse", "get_cached", "load_cache", "record_response"]
