"""Host test-slot pytest plugin (issue #2124): fixed-width OS file-lock slots.

Standalone and stdlib-only on purpose. It is loaded into *another repo's*
pytest through ``PYTEST_PLUGINS=test_slot_plugin`` + ``PYTHONPATH=<this dir>``,
so it cannot import ``charlie_work`` (the worker's interpreter may not have it)
and this directory holds nothing else -- putting it on ``PYTHONPATH`` must
expose exactly one module. It lives outside ``src/`` so a worktree pytest can
never resolve it through the main checkout's editable install.

Model: ``count`` slots of ``width`` cores each. A slot is an OS byte-range lock
on ``<dir>/slot-K.lock`` held for the life of the suite; a crashed holder
releases it automatically (no lease, heartbeat, reaper or daemon). Slot 0 is
taken only by the merge gate (``ROLE=gate``); agent suites compete for slots
``1..count-1``, so the gate never queues behind workers.

Only a *wide* run takes a slot: acquisition happens after collection and only
when the collected item count is at least ``MIN_ITEMS``. Targeted runs never wait.
The waiting is bounded and visible: a status line every 30s, then exit code
``HOST_BUSY_EXIT_CODE`` with ``HOST_BUSY_MESSAGE`` once ``TIMEOUT`` seconds pass.
A silent wait would push a ``claude -p`` worker into backgrounding the suite (#2096).

Inert unless ``CHARLIE_TEST_SLOT_COUNT`` is set (the orchestrator sets it; the
``test_slots.enabled`` kill switch simply stops setting it), and inert on xdist
workers (``PYTEST_XDIST_WORKER``) -- only the controller holds a slot.

Under xdist the controller never collects, so ``session.items`` is empty at
``pytest_collection_finish``; the item count arrives from the first worker via
``pytest_xdist_node_collection_finished`` instead, and the slot is taken there.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import time
from pathlib import Path

import pytest

ENV_DIR = "CHARLIE_TEST_SLOT_DIR"
ENV_COUNT = "CHARLIE_TEST_SLOT_COUNT"
ENV_MIN_ITEMS = "CHARLIE_TEST_SLOT_MIN_ITEMS"
ENV_TIMEOUT = "CHARLIE_TEST_SLOT_WAIT_TIMEOUT_SECONDS"
ENV_ROLE = "CHARLIE_TEST_SLOT_ROLE"
# Test seams: how often to probe and how often to print the wait line.
ENV_POLL = "CHARLIE_TEST_SLOT_POLL_SECONDS"
ENV_LOG_INTERVAL = "CHARLIE_TEST_SLOT_LOG_INTERVAL_SECONDS"

ROLE_AGENT = "agent"
ROLE_GATE = "gate"
GATE_SLOT = 0

HOST_BUSY_EXIT_CODE = 75  # EX_TEMPFAIL; distinct from pytest's own 0-5
HOST_BUSY_MESSAGE = "test-slot: host busy; retry or run targeted tests"

TIMEOUT_RECORD_DIR = "timeouts"
_MAX_WAITERS = 64


def _try_lock(path: Path):
    """Return an open handle holding the lock on ``path``, or None if held/unavailable."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        handle = path.open("r+b")
    except OSError:
        return None
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:  # BlockingIOError is an OSError
        handle.close()
        return None
    return handle


def _unlock(handle) -> None:
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        handle.close()
    except OSError:
        pass


def _env_number(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _write_timeout_record(slot_dir: Path, payload: dict) -> None:
    """One atomic file per timeout; the orchestrator drains them into events.db."""
    try:
        out_dir = slot_dir / TIMEOUT_RECORD_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        final = out_dir / f"{time.time_ns()}-{os.getpid()}.json"
        tmp = final.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(final)
    except OSError:
        pass  # observability must never mask the exit path


class SlotPool:
    """Acquire/hold one slot. Pure of pytest so it can be driven directly."""

    def __init__(self, slot_dir: Path, count: int, role: str) -> None:
        self.slot_dir = slot_dir
        self.count = count
        self.role = role
        self._held = None
        self.slot: int | None = None

    @property
    def candidates(self) -> range:
        return range(GATE_SLOT, GATE_SLOT + 1) if self.role == ROLE_GATE else range(1, self.count)

    def _slot_path(self, k: int) -> Path:
        return self.slot_dir / f"slot-{k}.lock"

    def try_acquire(self) -> tuple[bool, int]:
        """One non-blocking pass. Returns ``(acquired, held_count)``."""
        held = 0
        for k in self.candidates:
            handle = _try_lock(self._slot_path(k))
            if handle is not None:
                self._held, self.slot = handle, k
                return True, held
            held += 1
        return False, held

    def register_waiter(self):
        """Take a waiter lock so peers can count the queue (same fixed-width trick)."""
        for k in range(_MAX_WAITERS):
            handle = _try_lock(self.slot_dir / f"waiter-{k}.lock")
            if handle is not None:
                return handle
        return None

    def queued(self) -> int:
        n = 0
        for k in range(_MAX_WAITERS):
            path = self.slot_dir / f"waiter-{k}.lock"
            if not path.exists():
                break
            handle = _try_lock(path)
            if handle is None:
                n += 1
            else:
                _unlock(handle)
        return n

    def release(self) -> None:
        handle, self._held, self.slot = self._held, None, None
        if handle is not None:
            _unlock(handle)


class _Controller:
    def __init__(self, config) -> None:
        self.config = config
        self.pool = SlotPool(
            Path(os.environ.get(ENV_DIR) or _default_dir()),
            int(_env_number(ENV_COUNT, 0)),
            os.environ.get(ENV_ROLE, ROLE_AGENT),
        )
        self.min_items = int(_env_number(ENV_MIN_ITEMS, 300))
        self.timeout = _env_number(ENV_TIMEOUT, 480.0)
        self.poll = max(0.01, _env_number(ENV_POLL, 1.0))
        self.log_interval = _env_number(ENV_LOG_INTERVAL, 30.0)
        self._decided = False

    def _say(self, text: str) -> None:
        capman = self.config.pluginmanager.get_plugin("capturemanager")
        if capman is not None:
            with capman.global_and_fixture_disabled():
                print(text, file=sys.stderr, flush=True)
        else:
            print(text, file=sys.stderr, flush=True)

    def maybe_acquire(self, n_items: int) -> None:
        if self._decided:
            return
        self._decided = True
        if self.config.option.collectonly or n_items < self.min_items:
            return
        acquired, held = self.pool.try_acquire()
        if acquired:
            return
        waiter = self.pool.register_waiter()
        started = time.monotonic()
        next_log = started
        try:
            while True:
                now = time.monotonic()
                if now - started >= self.timeout:
                    self._timeout(n_items, held, now - started)
                if now >= next_log:
                    self._say(
                        f"test-slot: waiting ({held} held, {max(1, self.pool.queued())} queued)"
                    )
                    next_log = now + self.log_interval
                time.sleep(min(self.poll, max(0.0, self.timeout - (now - started))))
                acquired, held = self.pool.try_acquire()
                if acquired:
                    return
        finally:
            if waiter is not None:
                _unlock(waiter)

    def _timeout(self, n_items: int, held: int, waited: float) -> None:
        _write_timeout_record(
            self.pool.slot_dir,
            {
                "role": self.pool.role,
                "waited_seconds": round(waited, 1),
                "held": held,
                "slots": self.pool.count,
                "items": n_items,
                "cwd": os.getcwd(),
                "pid": os.getpid(),
            },
        )
        self._say(HOST_BUSY_MESSAGE)
        pytest.exit(HOST_BUSY_MESSAGE, returncode=HOST_BUSY_EXIT_CODE)

    # -- hooks (registered only on the controller) --------------------------
    def pytest_collection_finish(self, session) -> None:
        if session.items:  # empty under xdist: the controller does not collect
            self.maybe_acquire(len(session.items))

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids) -> None:
        self.maybe_acquire(len(ids))

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session) -> None:
        self.pool.release()


def _default_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(
        os.path.expanduser("~"), ".local", "state"
    )
    return os.path.join(base, "charlie-work", "test-slots")


def pytest_configure(config) -> None:
    if not os.environ.get(ENV_COUNT):
        return
    if os.environ.get("PYTEST_XDIST_WORKER") or hasattr(config, "workerinput"):
        return
    controller = _Controller(config)
    config.pluginmanager.register(controller, "charlie-test-slot-controller")
    atexit.register(controller.pool.release)
