from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

STATE_VERSION = 1

# Cross-process lock timeout (seconds) — best-effort to prevent wedging
_LOCK_TIMEOUT_SECONDS = 30

# Retry for transient read errors (e.g. Windows sharing violations) before
# treating the file as unrecoverable.
_LOAD_RETRY_ATTEMPTS = 3
_LOAD_RETRY_DELAY_SECONDS = 0.1

# Stale claim timeout (minutes) — claims older than this are re-dispatchable
# to prevent crashed phase-2 from wedging issues
_STALE_CLAIM_TIMEOUT_MINUTES = 30

logger = logging.getLogger(__name__)


class StateLockBusy(RuntimeError):
    """Raised when the advisory state lock cannot be acquired within its budget.

    A state writer that cannot acquire the lock must fail that unit of work as
    a value (skip + event log), never write unlocked.
    """


# Intra-process serialization for state_lock.
#
# The file lock below (msvcrt.locking / fcntl.flock) serializes across
# PROCESSES, but byte-range file locks are owned by the process, not the
# thread — two threads in the SAME process are not serialized by it and both
# enter the read-modify-write section concurrently. On Windows their atomic
# ``tmp.replace(state.json)`` calls then collide (destination held open by the
# other thread's read) and raise ``PermissionError``; on any platform the
# concurrent load→save races lose updates (issue #16).
#
# A per-path threading.Lock, acquired before the file lock, restores
# deterministic intra-process serialization. Keyed by normalized absolute path
# so distinct Path objects for the same file share one lock. The registry
# itself is guarded by a plain lock; entries are created on demand and never
# removed (one small Lock per distinct state path for the process lifetime).
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(path: Path) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(path))
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _THREAD_LOCKS[key] = lock
        return lock


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _to_float(value: Any) -> float | None:
    """Coerce a JSON-deserialized value to a float, or return None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _canonical_started_at(started_at: Any, process_start_time: Any | None = None) -> str:
    """Coerce ``started_at`` to a canonical ISO-8601 UTC string (Z, no microseconds).

    Accepts ISO-8601 strings (with or without timezone, with ``Z`` or ``+HH:MM``)
    and numeric Unix timestamps. If ``started_at`` is missing or unparseable, falls
    back to ``process_start_time`` (a Unix timestamp). Raises ``ValueError`` if no
    usable timestamp can be produced.
    """
    if started_at is None:
        started_at_str = ""
    else:
        started_at_str = str(started_at).strip()
    if started_at_str in {"", "None", "null"}:
        started_at_str = ""

    if started_at_str:
        try:
            parsed = datetime.fromisoformat(started_at_str)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            parsed = parsed.astimezone(UTC)
            return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except (ValueError, TypeError):
            pass

    # Fall back to the process start time, or a numeric started_at string.
    fallback_ts = _to_float(process_start_time)
    if fallback_ts is None and started_at_str:
        fallback_ts = _to_float(started_at_str)
    if fallback_ts is not None:
        try:
            parsed = datetime.fromtimestamp(fallback_ts, tz=UTC)
            return parsed.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except (ValueError, OSError, OverflowError):
            pass

    raise ValueError(
        f"started_at must be a valid ISO-8601 timestamp or numeric Unix timestamp; "
        f"got {started_at!r}"
    )


def is_claim_stale(claim_timestamp: str | None) -> bool:
    """Check if a dispatch_pending claim is stale and should be re-dispatchable.

    A claim is stale if it's older than _STALE_CLAIM_TIMEOUT_MINUTES.
    This prevents crashed phase-2 processes from wedging issues permanently.
    """
    if not claim_timestamp:
        return False
    try:
        claim_time = datetime.fromisoformat(claim_timestamp.replace("Z", "+00:00"))
        age = datetime.now(UTC) - claim_time
        return age > timedelta(minutes=_STALE_CLAIM_TIMEOUT_MINUTES)
    except (ValueError, TypeError):
        # Malformed timestamp — treat as stale to be safe
        return True


@contextmanager
def state_lock(state_path: Path):
    """Cross-process advisory lock for state.json read-modify-write cycles.

    Uses platform-specific file locking (Windows: msvcrt.locking, POSIX: fcntl.flock)
    on a lockfile alongside state.json. The lock is advisory and time-bounded to
    prevent wedging on stale locks.

    Deterministic: if the lock cannot be acquired within ``_LOCK_TIMEOUT_SECONDS``,
    the context manager raises ``StateLockBusy``. A state writer that cannot acquire
    the lock must fail that unit of work as a value (skip + event log), never write
    unlocked.

    A per-path threading.Lock is held around the whole critical section so that
    concurrent THREADS in this process are serialized too (the file lock only
    serializes across processes — see ``_thread_lock_for``).
    """
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_file = None
    acquired = False

    thread_lock = _thread_lock_for(state_path)
    thread_lock.acquire()
    try:
        # Create the lock file if needed. touch() leaves it at 0 bytes, which
        # is fine: msvcrt.locking(..., LK_NBLCK, 1) succeeds on a 0-byte file
        # on the deployed runtime (Python 3.13.5, Windows 11) — probed in
        # #324/#328, which removed the same write-1-byte guards from
        # file_lock.py as dead code.
        lock_path.touch()

        if sys.platform == "win32":
            import msvcrt

            lock_file = lock_path.open("r+b", encoding=None)
            # msvcrt.locking mode: 0 = lock, 1 = unlock
            # LK_NBLCK = non-blocking lock, LK_LOCK = blocking lock
            # We use a retry loop with timeout for bounded waiting
            import time

            start = time.time()
            while time.time() - start < _LOCK_TIMEOUT_SECONDS:
                try:
                    # Try non-blocking lock first
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except OSError:
                    # Lock held, wait and retry
                    time.sleep(0.1)
            else:
                # Timeout — the lock is held by another writer. Fail this unit
                # of work as a value rather than degrading integrity.
                logger.warning(
                    f"Failed to acquire lock on {lock_path} after {_LOCK_TIMEOUT_SECONDS}s"
                )
                raise StateLockBusy(
                    f"Could not acquire state lock at {lock_path} within {_LOCK_TIMEOUT_SECONDS}s"
                )
        else:
            import fcntl
            import time

            lock_file = lock_path.open("r+b", encoding=None)
            start = time.time()
            while time.time() - start < _LOCK_TIMEOUT_SECONDS:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (IOError, BlockingIOError):
                    # Lock held, wait and retry
                    time.sleep(0.1)
            else:
                # Timeout — the lock is held by another writer. Fail this unit
                # of work as a value rather than degrading integrity.
                logger.warning(
                    f"Failed to acquire lock on {lock_path} after {_LOCK_TIMEOUT_SECONDS}s"
                )
                raise StateLockBusy(
                    f"Could not acquire state lock at {lock_path} within {_LOCK_TIMEOUT_SECONDS}s"
                )

        yield
    finally:
        # Close whenever the handle was opened, regardless of whether the
        # lock was acquired — on the timeout path the handle was still opened
        # above and must not leak. Unlock only when acquired=True (nothing to
        # unlock otherwise). The two operations are independent failure modes:
        # an unlock raising OSError must not skip the close.
        if lock_file is not None:
            if acquired:
                try:
                    if sys.platform == "win32":
                        import msvcrt

                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    # Best-effort unlock — ignore failures
                    pass
            try:
                lock_file.close()
            except OSError:
                # Best-effort close — ignore failures
                pass
        # Release the intra-process thread lock last, after the file handle is
        # closed, so the next thread never observes a half-released critical
        # section. Always paired with the acquire() above the try.
        thread_lock.release()


def empty_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "generated_at": utc_now(),
        "issues": {},
        "prs": {},
        "events": [],
        "throttled_until": None,  # ISO timestamp when provider throttle cooldown ends
    }


def _quarantine_state(path: Path, exc: Exception) -> None:
    """Rename an unparseable state file for forensics and log a loud signal.

    The dispatch/loop path calls ``load_state`` frequently; emitting a
    top-level error here makes a silent state wipe visible in logs.
    """
    quarantine = path.with_name(f"{path.name}.corrupt-{utc_now().replace(':', '')}")
    logger.error(
        "State file %s is unrecoverable (%s: %s); quarantining to %s",
        path,
        type(exc).__name__,
        exc,
        quarantine,
    )
    try:
        path.replace(quarantine)
    except OSError as move_err:
        logger.error("Failed to quarantine state file %s: %s", path, move_err)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_state()

    data: Any = None
    for attempt in range(_LOAD_RETRY_ATTEMPTS):
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            # Genuine JSON corruption (truncated files, etc.) — quarantine.
            _quarantine_state(path, exc)
            return empty_state()
        except (LookupError, ValueError) as exc:
            # Decoding-level corruption (e.g. UTF-16LE+BOM, unknown encoding).
            # A wrong-encoding state file is not a transient read error.
            _quarantine_state(path, exc)
            return empty_state()
        except OSError as exc:
            # Sharing/permission violations on Windows are often transient.
            # Retry before falling back to quarantine.
            if attempt < _LOAD_RETRY_ATTEMPTS - 1:
                time.sleep(_LOAD_RETRY_DELAY_SECONDS)
                continue
            _quarantine_state(path, exc)
            return empty_state()
        else:
            break

    if not isinstance(data, dict):
        return empty_state()
    data.setdefault("version", STATE_VERSION)
    data.setdefault("generated_at", utc_now())
    data.setdefault("issues", {})
    data.setdefault("prs", {})
    data.setdefault("events", [])
    data.setdefault("throttled_until", None)  # Backward compatibility
    return data


def save_state(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Persist a fresh copy of ``data`` without mutating the caller's dict."""
    to_save = {**data, "generated_at": utc_now()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(to_save, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)
    return to_save


def load_state_locked(path: Path) -> dict[str, Any]:
    """Load a state snapshot while holding the advisory lock.

    This is the single point of enforcement for read-only ``load_state`` calls
    outside an explicit ``state_lock`` context. Callers receive a fresh snapshot
    and must not mutate it without re-acquiring the lock and saving explicitly.
    """
    with state_lock(path):
        return load_state(path)


def append_event(data: dict[str, Any], kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return a new state dict with the event appended; do not mutate ``data``."""
    events = list(data.get("events", []))
    events.append({"at": utc_now(), "kind": kind, "payload": payload})
    if len(events) > 200:
        events = events[-200:]
    return {**data, "events": events}


def is_throttled(data: dict[str, Any]) -> bool:
    """Check if the orchestrator is currently in a provider throttle cooldown window.

    Returns True if now < throttled_until, False otherwise.
    """
    throttled_until = data.get("throttled_until")
    if not throttled_until:
        return False
    try:
        throttle_time = datetime.fromisoformat(throttled_until.replace("Z", "+00:00"))
        return datetime.now(UTC) < throttle_time
    except (ValueError, TypeError):
        # Malformed timestamp — treat as not throttled to be safe
        return False


def set_throttled_until(data: dict[str, Any], throttled_until: str) -> dict[str, Any]:
    """Set the provider throttle cooldown window.

    Returns a new state dict with throttled_until set; does not mutate ``data``.
    """
    return {**data, "throttled_until": throttled_until}
