from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATE_VERSION = 1

# Cross-process lock timeout (seconds) — best-effort to prevent wedging
_LOCK_TIMEOUT_SECONDS = 30

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
        # Create lock file if it doesn't exist
        lock_path.touch(exist_ok=True)

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
