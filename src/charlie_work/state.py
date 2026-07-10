from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

STATE_VERSION = 1

# Cross-process lock timeout (seconds) — best-effort to prevent wedging
_LOCK_TIMEOUT_SECONDS = 30

# Stale claim timeout (minutes) — claims older than this are re-dispatchable
# to prevent crashed phase-2 from wedging issues
_STALE_CLAIM_TIMEOUT_MINUTES = 30

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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

    Best-effort: if locking fails, the context manager still yields — the orchestrator
    prefers forward progress over perfect serialization, and atomic writes already
    prevent torn files.
    """
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_file = None
    acquired = False

    try:
        # Create lock file if it doesn't exist.
        # Write 1 byte so msvcrt.locking(... 1) has a byte-range to lock:
        # msvcrt locks specific byte ranges and raises EACCES on a 0-byte file.
        # (Same gap as supervisor.lock — both use LK_NBLCK with nbytes=1.)
        if not lock_path.exists():
            lock_path.write_bytes(b"\x00")

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
                # Timeout — proceed anyway (best-effort)
                logger.warning(
                    f"Failed to acquire lock on {lock_path} after {_LOCK_TIMEOUT_SECONDS}s "
                    f"— proceeding without lock"
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
                # Timeout — proceed anyway (best-effort)
                logger.warning(
                    f"Failed to acquire lock on {lock_path} after {_LOCK_TIMEOUT_SECONDS}s "
                    f"— proceeding without lock"
                )

        yield
    finally:
        if lock_file is not None and acquired:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except OSError:
                # Best-effort unlock — ignore failures
                pass


def empty_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "generated_at": utc_now(),
        "issues": {},
        "prs": {},
        "events": [],
        "throttled_until": None,  # ISO timestamp when provider throttle cooldown ends
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_state()
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        # Never crash the orchestrator on a truncated/corrupt state file, and
        # never silently discard it either — quarantine it for forensics.
        quarantine = path.with_name(f"{path.name}.corrupt-{utc_now().replace(':', '')}")
        try:
            path.replace(quarantine)
        except OSError:
            pass
        return empty_state()
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
