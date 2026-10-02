"""Fleet-scoped quota ledger keyed by ``(harness, model)`` (issue #2086).

A provider quota or rate limit is account-wide, so a restriction learned from
one repo's session death applies to every repo. This ledger is a small JSON
file in the fleet directory (next to ``fleet.json`` / ``fleet.lock``):

.. code-block:: json

    {"version": 1, "restrictions": {"devin-shell|swe-2-high": {
        "harness": "devin-shell", "model": "swe-2-high",
        "until": "2026-09-30T08:00:00Z", "reason": "rate_limited",
        "source": "dead_worker_classification", "role": "worker",
        "updated_at": "2026-09-30T06:00:00Z"}}}

Writes are atomic (tmp + ``replace()``), serialized across processes by a
byte-range lock with a bounded wait, and **monotonic**: a new ``until`` never
shortens a still-active one (the #2043 ``set_throttled_until`` rule). Entries
are never cleared early; they simply expire.

Keys come from the **session**, never from config: every launch stamps the
chosen role-chain entry onto the session sidecar (:func:`stamp_session`), and
the quota/rate-limit classification sites read that stamp back
(:func:`record_from_sidecar_payload`). A sidecar without the stamp (launched
before this module shipped) records nothing -- the per-repo throttle still
applies to it exactly as before.

The ledger file lives at ``fleet_dir()`` (``CHARLIE_WORK_FLEET_DIR`` or the
platform default). It deliberately ignores the per-invocation ``--fleet-dir``
override: the classification sites that write it run deep inside reap lanes
that never receive that override, and reader and writer must agree.

Every function here is best-effort and never raises: a ledger I/O problem
degrades to "no restriction recorded", which is the pre-#2086 behavior.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic_write import write_json_atomic
from .file_lock import try_acquire_byte_range_lock
from .fleet_paths import fleet_dir
from .harnesses import HARNESS_REGISTRY
from .iso_timestamp import parse_iso_timestamp

logger = logging.getLogger(__name__)

LEDGER_FILENAME = "role_quota_ledger.json"
LEDGER_LOCK_FILENAME = "role_quota_ledger.lock"
LEDGER_VERSION = 1

# Sidecar key carrying the role-chain entry a session was launched with.
SESSION_ROLE_KEY = "role_entry"

# Worker failure kinds that restrict a model. ``provider_auth`` is excluded:
# a dead credential is not a quota window, and it keeps its own per-repo
# handling (``clear_quota_throttles`` treats it specially).
RESTRICTING_FAILURE_KINDS = frozenset({"rate_limited", "quota_exhausted"})

_LOCK_WAIT_SECONDS = 2.0


def ledger_path() -> Path:
    return fleet_dir() / LEDGER_FILENAME


def _format_z(moment: datetime) -> str:
    aware = moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
    return aware.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _key(harness: str, model: str) -> str:
    return f"{harness}|{model}"


def _read_raw(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict) or not isinstance(raw.get("restrictions"), dict):
        return {}
    return raw


def load_restrictions() -> dict[tuple[str, str], datetime]:
    """Every recorded ``(harness, model) -> until``, expired or not. Never raises."""
    result: dict[tuple[str, str], datetime] = {}
    for record in _read_raw(ledger_path()).get("restrictions", {}).values():
        if not isinstance(record, dict):
            continue
        harness, model = record.get("harness"), record.get("model")
        until = parse_iso_timestamp(record.get("until"))
        if isinstance(harness, str) and isinstance(model, str) and until is not None:
            result[(harness, model)] = until
    return result


def _acquire_lock(path: Path) -> Any:
    deadline = time.monotonic() + _LOCK_WAIT_SECONDS
    lock = try_acquire_byte_range_lock(path)
    while lock is None and time.monotonic() < deadline:
        time.sleep(random.uniform(0.02, 0.08))
        lock = try_acquire_byte_range_lock(path)
    return lock


def record_restriction(
    harness: str,
    model: str,
    until: str | datetime,
    *,
    reason: str | None,
    source: str,
    role: str = "",
    now: datetime | None = None,
) -> bool:
    """Restrict ``(harness, model)`` fleet-wide until ``until``; monotonic.

    Returns True when the stored window changed. A shorter window than a
    still-active stored one is ignored. Never raises.
    """
    until_dt = parse_iso_timestamp(until)
    if until_dt is None or not isinstance(harness, str) or not isinstance(model, str):
        return False
    resolved_now = now if now is not None else datetime.now(UTC)
    path = ledger_path()
    lock = _acquire_lock(path.with_name(LEDGER_LOCK_FILENAME))
    if lock is None:
        # Bounded wait exhausted: still write (the read-merge below keeps it
        # monotonic against whatever is on disk); a lost race costs at most
        # one concurrent extension, never a torn file.
        logger.warning("role quota ledger lock busy; writing without it")
    try:
        raw = _read_raw(path)
        restrictions = dict(raw.get("restrictions", {}))
        existing = restrictions.get(_key(harness, model))
        existing_until = (
            parse_iso_timestamp(existing.get("until")) if isinstance(existing, dict) else None
        )
        if existing_until is not None and existing_until >= until_dt:
            return False
        restrictions[_key(harness, model)] = {
            "harness": harness,
            "model": model,
            "until": _format_z(until_dt),
            "reason": reason,
            "source": source,
            "role": role,
            "updated_at": _format_z(resolved_now),
        }
        payload = {"version": LEDGER_VERSION, "restrictions": restrictions}
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, payload)
        return True
    except OSError as exc:
        logger.warning("role quota ledger write failed: %s", exc)
        return False
    finally:
        if lock is not None:
            lock.release()


def session_stamp(role: str, harness: str, model: str, chain_index: int) -> dict[str, Any]:
    return {"role": role, "harness": harness, "model": model, "chain_index": chain_index}


def sidecar_path_for(sessions_dir: Path, harness: str, number: int) -> Path | None:
    """The sidecar a ``harness`` launch writes for issue/PR ``number``."""
    capabilities = HARNESS_REGISTRY.get(harness)
    if capabilities is None:
        return None
    if capabilities.adapter_kind == "devin":
        from . import devin_shell

        return devin_shell._sidecar_path(sessions_dir, number)
    if capabilities.adapter_kind in ("claude-code", "api"):
        from . import claude_code

        return claude_code._sidecar_path(sessions_dir, number, capabilities.adapter_kind)
    return None


def _view_kind_to_harness(adapter_kind: str) -> str | None:
    for name, capabilities in HARNESS_REGISTRY.items():
        if capabilities.adapter_kind == adapter_kind:
            return name
    return None


def stamp_session(sidecar: Path | None, stamp: Mapping[str, Any]) -> bool:
    """Record the launch's role-chain entry on its sidecar. Never raises.

    A missing sidecar (dry-run, a failed launch, a test fake) is a no-op.
    """
    if sidecar is None or not sidecar.exists():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False
        payload[SESSION_ROLE_KEY] = dict(stamp)
        write_json_atomic(sidecar, payload)
        return True
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("role stamp on %s failed: %s", sidecar, exc)
        return False


def record_from_sidecar_payload(
    payload: Mapping[str, Any],
    until: str | datetime | None,
    *,
    reason: str | None,
    source: str,
) -> bool:
    """Restrict the stamped ``(harness, model)`` of a classified session.

    No stamp (a pre-#2086 sidecar) or no ``until`` records nothing.
    """
    stamp = payload.get(SESSION_ROLE_KEY) if isinstance(payload, Mapping) else None
    if not isinstance(stamp, Mapping) or until is None:
        return False
    harness, model = stamp.get("harness"), stamp.get("model")
    if not isinstance(harness, str) or not isinstance(model, str):
        return False
    return record_restriction(
        harness, model, until, reason=reason, source=source, role=str(stamp.get("role") or "")
    )


def record_classified_death(
    payload: Mapping[str, Any], failure_kind: str | None, until: str | None, *, source: str
) -> bool:
    """The classification-site hook: record only a quota/rate-limit death."""
    if failure_kind not in RESTRICTING_FAILURE_KINDS:
        return False
    return record_from_sidecar_payload(payload, until, reason=failure_kind, source=source)


def record_for_session(
    sessions_dir: Path,
    adapter_kind: str,
    number: int,
    until: str | datetime | None,
    *,
    reason: str | None,
    source: str,
) -> bool:
    """Read a session's sidecar (by ``WorkerView.adapter_kind``) and record it."""
    harness = _view_kind_to_harness(adapter_kind)
    sidecar = sidecar_path_for(sessions_dir, harness, number) if harness else None
    if sidecar is None or until is None:
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return record_from_sidecar_payload(payload, until, reason=reason, source=source)


def record_view(
    sessions_dir: Path,
    view: Any,
    until: str | datetime | None,
    source: str,
    write_gate: Any = None,
    reason: str = "quota_exhausted",
) -> bool:
    """:func:`record_for_session` for a ``WorkerView``; a no-op under ``--dry-run``."""
    if write_gate is not None and getattr(write_gate, "dry_run", False):
        return False
    return record_for_session(
        sessions_dir, view.adapter_kind, view.issue_number, until, reason=reason, source=source
    )
