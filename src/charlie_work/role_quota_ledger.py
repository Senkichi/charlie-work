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

Harness-wide records (issue #2810): the same file also holds a
``"<harness>|*"`` record -- ``model`` is the :data:`HARNESS_WIDE_MODEL`
sentinel -- written when a harness CLI itself cannot authenticate (every launch
on that harness fails, whatever the model). It restricts every chain entry on
the harness (:func:`effective_restriction`) and carries the episode fields the
alert-once and resume logic need (:class:`HarnessBlock`). Unlike a quota
window it is cleared by proof, not by the clock alone:
:func:`mark_harness_resumed` is stamped when a probe launch outlives the
survival period.

Every function here is best-effort and never raises: a ledger I/O problem
degrades to "no restriction recorded", which is the pre-#2086 behavior.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .atomic_write import write_json_atomic
from .file_lock import try_acquire_byte_range_lock
from .fleet_paths import fleet_dir
from .harnesses import HARNESS_REGISTRY
from .iso_timestamp import parse_iso_timestamp
from .throttle_signatures import HARNESS_AUTH_FAILED_KIND

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

# Consecutive-hit backoff (issue #2767): a (harness, model) that fails again
# after its window expired gets an escalating restriction, doubling the base
# cooldown per consecutive hit. The count resets once a restriction has been
# expired for longer than the reset window without a new hit.
CONSECUTIVE_HIT_RESET_WINDOW = timedelta(hours=2)
MAX_ESCALATED_BACKOFF = timedelta(hours=4)
# Only short base windows escalate -- a base already past this is a provider
# reset window (5h Go window, 24h quota, weekly/monthly), not a short
# unstated-reset cooldown (15m rate limit or 1h Zen cooldown + resume margin).
_ESCALATABLE_BASE_MAX = timedelta(hours=2)

_LOCK_WAIT_SECONDS = 2.0

# Issue #2810: the ``model`` of a harness-wide record. Not a real model name, so
# a ``(harness, "*")`` key can never collide with a chain entry's own key.
HARNESS_WIDE_MODEL = "*"
HARNESS_AUTH_REASON = HARNESS_AUTH_FAILED_KIND
# Base window of a harness-wide block. Short on purpose: the resume check
# (``harness_auth_block``) is a cheap credential probe, so a short window keeps
# resume latency low; consecutive launch deaths escalate it by the #2767 backoff.
HARNESS_AUTH_BASE_WINDOW = timedelta(minutes=15)


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


def _iter_restrictions(*, wide: bool | None) -> dict[tuple[str, str], datetime]:
    """Parsed ledger records; ``wide`` True = harness-wide only, False = per-model only."""
    result: dict[tuple[str, str], datetime] = {}
    for record in _read_raw(ledger_path()).get("restrictions", {}).values():
        if not isinstance(record, dict):
            continue
        harness, model = record.get("harness"), record.get("model")
        if wide is not None and (model == HARNESS_WIDE_MODEL) != wide:
            continue
        until = parse_iso_timestamp(record.get("until"))
        if isinstance(harness, str) and isinstance(model, str) and until is not None:
            result[(harness, model)] = until
    return result


def load_restrictions(*, harness_auth: bool = True) -> dict[tuple[str, str], datetime]:
    """Every recorded ``(harness, model) -> until``, expired or not. Never raises.

    ``harness_auth=False`` (the ``fleet.harness_auth_block_enabled`` kill switch,
    issue #2810) drops the harness-wide records, so a disabled feature reads the
    ledger exactly as it did before they existed -- even a block recorded
    earlier stops being honored the moment the switch flips.
    """
    return _iter_restrictions(wide=None if harness_auth else False)


def load_harness_restrictions() -> dict[tuple[str, str], datetime]:
    """Only the harness-wide ``(harness, "*") -> until`` records. Never raises.

    What a length-1 role chain reads: with no ``fallbacks:`` it has no per-model
    routing to do, but a dead harness credential still has to stop the launch.
    """
    return _iter_restrictions(wide=True)


def list_harness_blocks() -> list[HarnessBlock]:
    """Every harness-wide block in the ledger (any episode state). Never raises."""
    blocks = [
        _parse_block(record)
        for record in _read_raw(ledger_path()).get("restrictions", {}).values()
        if isinstance(record, dict) and record.get("model") == HARNESS_WIDE_MODEL
    ]
    return [block for block in blocks if block is not None]


def effective_restriction(
    ledger: Mapping[tuple[str, str], datetime], harness: str, model: str, now: datetime
) -> tuple[datetime | None, bool]:
    """``(until, harness_wide)``: how long ``(harness, model)`` is restricted at ``now``.

    The later of the entry's own window and its harness's wide block (issue
    #2810), counting only windows still live at ``now``; ``(None, False)`` when
    neither is. ``harness_wide`` is True when the harness block is what binds
    (it ends at or after the entry's own window), so callers can tell a dead
    credential from a quota wall. Pure.
    """
    own = ledger.get((harness, model))
    wide = ledger.get((harness, HARNESS_WIDE_MODEL))
    own = own if own is not None and own > now else None
    wide = wide if wide is not None and wide > now else None
    if wide is not None and (own is None or wide >= own):
        return wide, True
    return own, False


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
    resolved_now = (now if now is not None else datetime.now(UTC)).replace(microsecond=0)
    until_dt = until_dt.replace(microsecond=0)
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
        prev_hits = existing.get("consecutive_hits", 1) if isinstance(existing, dict) else 1
        if not (isinstance(prev_hits, int) and prev_hits >= 1):
            prev_hits = 1
        if (
            existing_until is not None
            and existing_until <= resolved_now
            and resolved_now - existing_until <= CONSECUTIVE_HIT_RESET_WINDOW
        ):
            # A post-expiry hit within the reset window: a subsequent launch
            # failed again, so the short base cooldown escalates. A hit while
            # the window was still live is the same batch dying, not a new
            # launch, and never reaches this branch.
            consecutive_hits = prev_hits + 1
            base_duration = until_dt - resolved_now
            if timedelta(0) < base_duration <= _ESCALATABLE_BASE_MAX:
                until_dt = resolved_now + min(
                    base_duration * (2 ** min(consecutive_hits - 1, 3)),
                    MAX_ESCALATED_BACKOFF,
                )
        elif existing_until is not None and existing_until > resolved_now:
            consecutive_hits = prev_hits
        else:
            consecutive_hits = 1
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
            "consecutive_hits": consecutive_hits,
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


@dataclass(frozen=True)
class HarnessBlock:
    """A harness-wide block (issue #2810): the harness CLI cannot authenticate.

    One *episode* runs from the first death that arms the block to the probe
    that proves the harness healthy again (``resumed_at``). ``alerted_at`` is
    stamped once per episode (the alert-once claim); ``stage_a_green_at`` says
    the cheap credential check passed for the current window, so one real probe
    launch may be admitted. ``affected`` is every ``(repo, issue)`` whose death
    armed or re-armed the block this episode.
    """

    harness: str
    until: datetime
    evidence: str
    consecutive_hits: int
    episode_started_at: datetime | None
    alerted_at: datetime | None
    stage_a_green_at: datetime | None
    resumed_at: datetime | None
    affected: tuple[tuple[str, int], ...] = ()

    @property
    def episode_open(self) -> bool:
        return self.resumed_at is None


@dataclass(frozen=True)
class ArmOutcome:
    """Result of :func:`arm_harness_block`: the stored block and whether it began an episode."""

    block: HarnessBlock | None
    new_episode: bool = False


def _backoff_window(base: timedelta, hits: int) -> timedelta:
    """The #2767 escalation: the base doubles per consecutive hit, capped at 4h."""
    return min(base * (2 ** min(max(hits, 1) - 1, 3)), MAX_ESCALATED_BACKOFF)


def _parse_block(record: Any) -> HarnessBlock | None:
    if not isinstance(record, dict):
        return None
    harness = record.get("harness")
    until = parse_iso_timestamp(record.get("until"))
    if not isinstance(harness, str) or until is None:
        return None
    hits = record.get("consecutive_hits", 1)
    affected: list[tuple[str, int]] = []
    raw_affected = record.get("affected")
    if isinstance(raw_affected, dict):
        for repo, issues in raw_affected.items():
            if isinstance(repo, str) and isinstance(issues, list):
                affected.extend((repo, n) for n in issues if isinstance(n, int))
    return HarnessBlock(
        harness=harness,
        until=until,
        evidence=str(record.get("evidence") or ""),
        consecutive_hits=hits if isinstance(hits, int) and hits >= 1 else 1,
        episode_started_at=parse_iso_timestamp(record.get("episode_started_at")),
        alerted_at=parse_iso_timestamp(record.get("alerted_at")),
        stage_a_green_at=parse_iso_timestamp(record.get("stage_a_green_at")),
        resumed_at=parse_iso_timestamp(record.get("resumed_at")),
        affected=tuple(sorted(set(affected))),
    )


def read_harness_block(harness: str) -> HarnessBlock | None:
    """The harness-wide record for ``harness`` (live, lapsed or resumed), else ``None``."""
    record = (
        _read_raw(ledger_path()).get("restrictions", {}).get(_key(harness, HARNESS_WIDE_MODEL))
    )
    return _parse_block(record)


def _update_harness_record(
    harness: str, mutate: Callable[[dict[str, Any] | None], dict[str, Any] | None]
) -> HarnessBlock | None:
    """Locked read-modify-write of the harness-wide record; returns the stored block.

    ``mutate`` gets the current record (``None`` when absent) and returns the
    record to store, or ``None`` for "no change" -- in which case the current
    record is returned unparsed-as-is. Never raises (an I/O problem is "no
    change").
    """
    path = ledger_path()
    lock = _acquire_lock(path.with_name(LEDGER_LOCK_FILENAME))
    if lock is None:
        logger.warning("role quota ledger lock busy; writing without it")
    try:
        raw = _read_raw(path)
        restrictions = dict(raw.get("restrictions", {}))
        key = _key(harness, HARNESS_WIDE_MODEL)
        existing = restrictions.get(key)
        updated = mutate(existing if isinstance(existing, dict) else None)
        if updated is None:
            return _parse_block(existing)
        restrictions[key] = updated
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, {"version": LEDGER_VERSION, "restrictions": restrictions})
        return _parse_block(updated)
    except OSError as exc:
        logger.warning("role quota ledger write failed: %s", exc)
        return None
    finally:
        if lock is not None:
            lock.release()


def arm_harness_block(
    harness: str,
    *,
    base_window: timedelta,
    evidence: str | None,
    source: str,
    now: datetime | None = None,
    escalate: bool = True,
    affected: tuple[str, int] | None = None,
) -> ArmOutcome:
    """Arm (or re-arm) the harness-wide block for ``harness``. Never raises.

    * No record, or one whose episode has ended (``resumed_at``): a NEW episode
      -- ``base_window`` from ``now``, one hit, alert not yet sent.
    * A record whose window is still live: the same batch dying; ``until`` and
      the hit count are untouched (only evidence and ``affected`` merge).
    * A lapsed record: ``escalate=True`` (a launch died of the signature) is
      another consecutive hit and the window grows by the #2767 backoff;
      ``escalate=False`` (the cheap credential check was still red -- no launch
      happened) just re-arms ``base_window`` so resume latency stays low.

    Re-arming clears ``stage_a_green_at``. Arming never touches ``alerted_at``:
    the once-per-episode alert is claimed separately by :func:`claim_harness_alert`.
    """
    resolved_now = (now if now is not None else datetime.now(UTC)).replace(microsecond=0)
    state: dict[str, bool] = {"new_episode": False}

    def mutate(existing: dict[str, Any] | None) -> dict[str, Any] | None:
        block = _parse_block(existing)
        new_episode = block is None or not block.episode_open
        if block is None or new_episode:
            hits, until = 1, resolved_now + base_window
            started, alerted = resolved_now, None
            issues_by_repo: dict[str, list[int]] = {}
            stage_a = None
        else:
            hits, until = block.consecutive_hits, block.until
            started, alerted = block.episode_started_at or resolved_now, block.alerted_at
            issues_by_repo = {}
            for repo, issue in block.affected:
                issues_by_repo.setdefault(repo, []).append(issue)
            stage_a = block.stage_a_green_at
            if block.until <= resolved_now:
                stage_a = None
                if escalate:
                    hits += 1
                    until = resolved_now + _backoff_window(base_window, hits)
                else:
                    until = resolved_now + base_window
        if affected is not None:
            repo, issue = affected
            bucket = issues_by_repo.setdefault(repo, [])
            if issue not in bucket:
                bucket.append(issue)
        state["new_episode"] = new_episode
        record: dict[str, Any] = {
            "harness": harness,
            "model": HARNESS_WIDE_MODEL,
            "until": _format_z(until),
            "reason": HARNESS_AUTH_REASON,
            "source": source,
            "role": "",
            "updated_at": _format_z(resolved_now),
            "consecutive_hits": hits,
            "evidence": evidence or (block.evidence if block is not None else ""),
            "episode_started_at": _format_z(started),
            "alerted_at": _format_z(alerted) if alerted is not None else None,
            "stage_a_green_at": _format_z(stage_a) if stage_a is not None else None,
            "resumed_at": None,
            "affected": {repo: sorted(issues) for repo, issues in sorted(issues_by_repo.items())},
        }
        return record

    block = _update_harness_record(harness, mutate)
    if block is None:
        return ArmOutcome(None)
    return ArmOutcome(block, new_episode=state["new_episode"])


def claim_harness_alert(harness: str, *, now: datetime | None = None) -> HarnessBlock | None:
    """Take the once-per-episode alert claim. Never raises.

    Returns the block for exactly ONE caller per open episode -- the claim is a
    locked read-modify-write, so racing processes cannot both win -- and
    ``None`` for everyone else (already alerted, no block, episode over).
    """
    resolved_now = (now if now is not None else datetime.now(UTC)).replace(microsecond=0)
    state = {"claimed": False}

    def mutate(existing: dict[str, Any] | None) -> dict[str, Any] | None:
        block = _parse_block(existing)
        if existing is None or block is None or not block.episode_open or block.alerted_at:
            return None
        state["claimed"] = True
        return {**existing, "alerted_at": _format_z(resolved_now)}

    block = _update_harness_record(harness, mutate)
    return block if state["claimed"] else None


def mark_stage_a_green(harness: str, *, now: datetime | None = None) -> bool:
    """Record that the cheap credential check passed for the current window.

    True when the stamp landed on an open episode. Never raises.
    """
    resolved_now = (now if now is not None else datetime.now(UTC)).replace(microsecond=0)

    def mutate(existing: dict[str, Any] | None) -> dict[str, Any] | None:
        block = _parse_block(existing)
        if existing is None or block is None or not block.episode_open:
            return None
        return {**existing, "stage_a_green_at": _format_z(resolved_now)}

    block = _update_harness_record(harness, mutate)
    return block is not None and block.episode_open and block.stage_a_green_at is not None


def mark_harness_resumed(harness: str, *, now: datetime | None = None) -> HarnessBlock | None:
    """End the episode: a probe outlived the survival period. Never raises.

    Returns the final block when THIS call ended an open episode (the caller
    emits ``harness_auth_resumed`` exactly once), else ``None``.
    """
    resolved_now = (now if now is not None else datetime.now(UTC)).replace(microsecond=0)
    state = {"ended": False}

    def mutate(existing: dict[str, Any] | None) -> dict[str, Any] | None:
        block = _parse_block(existing)
        if existing is None or block is None or not block.episode_open:
            return None
        state["ended"] = True
        return {**existing, "resumed_at": _format_z(resolved_now)}

    block = _update_harness_record(harness, mutate)
    return block if state["ended"] else None


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
    from . import claude_code

    if capabilities.adapter_kind in claude_code.CLAUDE_RECORD_KINDS:
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


def _session_stamp_for(
    sessions_dir: Path, adapter_kind: str, number: int
) -> Mapping[str, Any] | None:
    """The ``role_entry`` stamp on the session's sidecar, else ``None``.

    A missing/unreadable sidecar or an unstamped session (pre-#2086
    launches) yields ``None``.
    """
    harness = _view_kind_to_harness(adapter_kind)
    sidecar = sidecar_path_for(sessions_dir, harness, number) if harness else None
    if sidecar is None:
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    stamp = payload.get(SESSION_ROLE_KEY) if isinstance(payload, dict) else None
    return stamp if isinstance(stamp, Mapping) else None


def _record_from_stamp(
    stamp: Mapping[str, Any], until: str | datetime, *, reason: str | None, source: str
) -> bool:
    harness, model = stamp.get("harness"), stamp.get("model")
    if not isinstance(harness, str) or not isinstance(model, str):
        return False
    return record_restriction(
        harness, model, until, reason=reason, source=source, role=str(stamp.get("role") or "")
    )


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
    return _record_from_stamp(stamp, until, reason=reason, source=source)


def record_classified_death(
    payload: Mapping[str, Any],
    failure_kind: str | None,
    until: str | None,
    *,
    source: str,
    harness: str | None = None,
    evidence: str | None = None,
    repo: str | None = None,
) -> bool:
    """The classification-site hook: record a quota/rate-limit or harness-auth death.

    A ``rate_limited`` / ``quota_exhausted`` death restricts the session's
    stamped ``(harness, model)`` until ``until``. A ``harness_auth_failed``
    death (issue #2810) instead arms the harness-wide block for ``harness``
    (default: the session stamp's harness) -- the stamp's *model* is irrelevant
    because every launch on a harness with a dead credential fails alike, and an
    unstamped (pre-#2086) sidecar still blocks when the caller names the
    ``harness``. ``evidence`` is the log line the classification fired on;
    ``repo`` and the payload's ``issue_number`` record who was affected. Any
    other kind records nothing.
    """
    if failure_kind == HARNESS_AUTH_REASON:
        stamp = payload.get(SESSION_ROLE_KEY) if isinstance(payload, Mapping) else None
        resolved = harness or (stamp.get("harness") if isinstance(stamp, Mapping) else None)
        if not isinstance(resolved, str) or not resolved:
            return False
        issue = payload.get("issue_number") if isinstance(payload, Mapping) else None
        return (
            arm_harness_block(
                resolved,
                base_window=HARNESS_AUTH_BASE_WINDOW,
                evidence=evidence,
                source=source,
                affected=(repo, issue)
                if isinstance(repo, str) and isinstance(issue, int)
                else None,
            ).block
            is not None
        )
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
    if until is None:
        return False
    stamp = _session_stamp_for(sessions_dir, adapter_kind, number)
    if stamp is None:
        return False
    return _record_from_stamp(stamp, until, reason=reason, source=source)


def role_key_for_session(
    sessions_dir: Path, adapter_kind: str, number: int
) -> tuple[str, str] | None:
    """The ``(harness, model)`` the session's role stamp names, else ``None``.

    Read half of :func:`stamp_session`: the per-repo throttle writers stamp
    the window with the chain entry whose session produced it (issue #2279)
    so ``role_selection.window_covered`` can tell a fallback's quota window
    from the selected entry's own death. A missing/unreadable sidecar or an
    unstamped session (pre-#2086 launches) yields ``None`` -- the window then
    keeps its untagged, still-blocking provenance.
    """
    stamp = _session_stamp_for(sessions_dir, adapter_kind, number)
    if stamp is None:
        return None
    stamped_harness, stamped_model = stamp.get("harness"), stamp.get("model")
    if not isinstance(stamped_harness, str) or not isinstance(stamped_model, str):
        return None
    return (stamped_harness, stamped_model)


def role_key_for_view(sessions_dir: Path, view: Any) -> tuple[str | None, str | None]:
    """The ``(harness, model)`` pair of the session's role stamp, unpacked.

    The single unwrapping site for the #2279 window writers: every arming
    site (``worker_fate`` evidence, the stalled/dead/reconcile throttle
    lanes, ``stalled_review_reap``'s quota record) needs the stamp as a
    ``harness``/``model`` kwarg pair, and ``(None, None)`` -- never the
    ``| None`` key itself -- is the unstamped answer each one writes.
    """
    return role_key_for_session(sessions_dir, view.adapter_kind, view.issue_number) or (None, None)


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
