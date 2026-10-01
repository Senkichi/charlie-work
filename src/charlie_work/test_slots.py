"""Host test-slot pool: config, worker/gate env, and the timeout-event drain (issue #2124).

Dispatch admission (``dispatch.host_load_max_pytest_*``) only bounds how many
sessions start; once admitted, each decides when to run a suite and how wide.
This module is the orchestrator side of the execution-time bound: ``S =
cpu_count // width`` fixed-width slots, each an OS file lock held by one wide
suite. The enforcement lives in ``plugins/test_slot/test_slot_plugin.py`` -- a
standalone, stdlib-only pytest plugin (it must load into any repo's venv, so it
cannot import this package). What belongs here is everything the plugin cannot
know on its own: the config section, the env that arms the plugin for a worker
or for the merge gate, and the consumer for its timeout records.

``slot_dir()`` is computed once, here, and handed to every consumer through
``CHARLIE_TEST_SLOT_DIR``. Two interpreters resolving ``%LOCALAPPDATA%``
independently can land in different MSIX-virtualized copies and silently run two
pools; passing the resolved path makes that impossible.

The ``test_slots`` section is host-wide: one machine has one pool.

This module must not import ``config`` (``config`` re-exports its dataclass).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from .config_validation import AtLeastOne, NonNeg, Positive, Typed
from .fleet_paths import fleet_dir

PLUGIN_MODULE = "test_slot_plugin"
PLUGIN_DIR_NAME = "plugins/test_slot"

# Shared with the plugin by value (it cannot import this module);
# tests/test_test_slots.py asserts the two sets stay identical.
ENV_DIR = "CHARLIE_TEST_SLOT_DIR"
ENV_COUNT = "CHARLIE_TEST_SLOT_COUNT"
ENV_MIN_ITEMS = "CHARLIE_TEST_SLOT_MIN_ITEMS"
ENV_TIMEOUT = "CHARLIE_TEST_SLOT_WAIT_TIMEOUT_SECONDS"
ENV_ROLE = "CHARLIE_TEST_SLOT_ROLE"
ROLE_AGENT = "agent"
ROLE_GATE = "gate"
TIMEOUT_RECORD_DIR = "timeouts"

_XDIST_WIDTH_VAR = "PYTEST_XDIST_AUTO_NUM_WORKERS"


@dataclass(frozen=True)
class SlotPoolConfig:
    """Fixed-width test-slot pool (``test_slots:``).

    ``enabled`` is the kill switch (default on): off stops arming the plugin, so
    suites run ungoverned exactly as before. ``width`` is cores per slot and the
    ``-n`` each wide suite runs at. ``min_items`` is the collected-item count
    below which a run is "targeted" and never takes a slot.
    ``wait_timeout_seconds`` bounds the wait for a slot; keep it under the Claude
    Bash tool's 10-minute ceiling (default 480s) or a worker is killed mid-wait
    instead of seeing the "host busy" message.
    """

    enabled: Annotated[bool, Typed] = True
    width: Annotated[int, Typed, AtLeastOne] = 4
    min_items: Annotated[int, Typed, NonNeg] = 300
    wait_timeout_seconds: Annotated[int, Typed, Positive] = 480


def slot_count(width: int, cpu_count: int | None = None) -> int:
    """Total slots ``S = cpu_count // width``, floored at 2.

    Slot 0 is reserved for the merge gate, so fewer than 2 would leave agent
    suites with no slot at all and make every wide run time out.
    """
    cpus = os.cpu_count() if cpu_count is None else cpu_count
    return max(2, (cpus or 0) // max(1, width))


def slot_dir() -> Path:
    """``%LOCALAPPDATA%\\charlie-work\\test-slots`` (POSIX: the XDG state equivalent)."""
    return fleet_dir() / "test-slots"


def plugin_dir() -> Path:
    """The plugin's own directory, resolved from this module's checkout."""
    return Path(__file__).resolve().parents[2] / PLUGIN_DIR_NAME


def _prepend(value: str, existing: str | None) -> str:
    return os.pathsep.join([value, existing]) if existing else value


def arm_env(
    cfg: SlotPoolConfig,
    *,
    role: str,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Env that arms the plugin for one process tree; ``{}`` when disabled.

    ``PYTHONPATH`` / ``PYTEST_PLUGINS`` are prepended to the ambient values in
    ``base_env`` (default ``os.environ``) rather than replacing them -- a worker
    env entry is a full replacement, so clobbering would break any repo that
    already uses either variable. ``PYTEST_XDIST_AUTO_NUM_WORKERS`` carries the
    slot width through the existing plumbing.
    """
    if not cfg.enabled:
        return {}
    env = os.environ if base_env is None else base_env
    return {
        "PYTHONPATH": _prepend(str(plugin_dir()), env.get("PYTHONPATH")),
        "PYTEST_PLUGINS": ",".join(filter(None, [PLUGIN_MODULE, env.get("PYTEST_PLUGINS")])),
        _XDIST_WIDTH_VAR: str(cfg.width),
        ENV_DIR: str(slot_dir()),
        ENV_COUNT: str(slot_count(cfg.width)),
        ENV_MIN_ITEMS: str(cfg.min_items),
        ENV_TIMEOUT: str(cfg.wait_timeout_seconds),
        ENV_ROLE: role,
    }


def drain_wait_timeouts(
    directory: Path,
    emit: Callable[[dict[str, Any]], None],
) -> int:
    """Hand each timeout record to ``emit``; returns how many were emitted.

    The caller logs the ``test_slot_wait_timeout`` event (a literal kind, so the
    event-kind registry scan can resolve it).

    The plugin runs in a foreign pytest and cannot reach ``events.db``, so it
    leaves one atomic JSON file per timeout. A record is claimed by renaming it
    (atomic, first drainer wins), so concurrent repo loops never double-emit.
    A malformed record is dropped with its claim -- it was already unreadable.
    """
    records = directory / TIMEOUT_RECORD_DIR
    try:
        pending = sorted(records.glob("*.json"))
    except OSError:
        return 0
    emitted = 0
    for path in pending:
        claimed = path.with_suffix(".claimed")
        try:
            path.replace(claimed)
        except OSError:
            continue  # another loop claimed it
        try:
            payload = json.loads(claimed.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        finally:
            claimed.unlink(missing_ok=True)
        if isinstance(payload, dict):
            emit(payload)
            emitted += 1
    return emitted
