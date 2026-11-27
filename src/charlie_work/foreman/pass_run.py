"""The foreman pass — one deterministic iteration over every foreman (#3122).

``charlie scheme`` drives this module. A pass is design §7: for each foreman
in ``active`` or ``degraded`` state, evaluate the wake predicate, apply the
deferral bounds, and run a Turn (Collect → Decide → Apply → bookkeeping)
where the predicate holds. The fleet's supervise loop spawns ``charlie
scheme`` as a child process every ``foreman.pass_interval_minutes`` and never
waits on it — the pass exits after one iteration.

Locking is two byte-range file locks (``file_lock.py``, non-blocking,
released on process death): ``foremen/pass.lock`` makes at most one pass —
and so at most one Decide — run at a time, and ``foremen/<name>/turn.lock``
is held for a foreman's whole Turn so its Turns never overlap. A pass that
cannot take a foreman's lock moves on to the next foreman.

A Turn's disk record under ``turns/<turn_id>/`` is the resume journal:
``briefing.md`` and ``turn.json`` (cursors, fingerprint, signal flag, decide
usage) land at Collect, ``actions.json`` before Apply, and ``complete.json``
only after the journal/state bookkeeping lands. A ``turn-*`` directory with
``actions.json`` but no ``complete.json`` is a crash survivor — the next pass
resumes it at Apply and never asks the model again. ``dry-*`` directories are
``--dry-run`` artifacts and never resume.

Failure accounting (design §10): a failed Collect/Decide/Apply increments
``consecutive_failures`` and writes ``foreman_turn_failed``; three in a row
move the foreman to ``faulted`` and raise a report item. A deferred Decide
(provider throttle/quota, chain exhausted) is a ``foreman_wake_deferred``
instead — nothing increments. A successful Turn resets the counter.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import layout
from ..atomic_write import write_json_atomic, write_text_atomic
from ..config_validation import ConfigError
from ..file_lock import try_acquire_byte_range_lock
from ..fleet_pause import fleet_pause_pending
from ..foreman_config import ForemanConfig
from ..github import GitHub
from ..global_config import load_layered_config
from ..instrumentation import log_event
from ..local_issues import github_client_for
from ..notify_freshness import resolve_fleet_notify
from . import Result
from . import items as fitems
from . import journal as fjournal
from . import limits as flimits
from . import paths as fpaths
from . import state as fstate
from . import store
from .actions import TURN_OUTPUT_SCHEMA
from .apply import apply_turn
from .apply_context import OUTCOME_APPLIED, OUTCOME_REFUSED, OUTCOME_SKIPPED
from .collect import Foreman, build_briefing, load_foreman
from .decide import DecideForeman, decide
from .journal import utc_now
from .notify import notify_unnotified_items
from .wake import (  # noqa: F401 -- deliberate re-export (moved under the file-size ratchet; the pass's public surface stays put)
    DEFER_DECIDE,
    DEFER_FLEET_PAUSED,
    DEFER_MAX_TURNS,
    DEFER_MIN_INTERVAL,
    DEFER_STALE_SNAPSHOT,
    EVENT_WAKE_DEFERRED,
    WAKE_INBOX,
    WAKE_ITEM_ANSWER,
    WAKE_LIFECYCLE,
    WAKE_QUIET,
    WAKE_REPORT_DUE,
    WAKE_RESUME,
    WAKE_SIGNAL,
    _deferral_reason,
    _emit_deferred,
    _today,
    wake_reason,
)

EVENT_TURN_COMPLETED = "foreman_turn_completed"
EVENT_TURN_FAILED = "foreman_turn_failed"
#: The fleet-level "a foreman pass ran" heartbeat signal (design §10's
#: "foreman pass not seen for twice its interval" alarm); written to the
#: fleet events.db by ``run_pass`` — a literal, not an alias, so the
#: event-kind scanner resolves the emit site.
EVENT_PASS_COMPLETED = "foreman_pass_completed"

#: Three consecutive failed Turns move a foreman to ``faulted`` (design §10).
FAULT_LIMIT = 3

#: Turn directories beyond the newest N are pruned after each Turn (§6).
TURN_RETENTION = 50


@dataclass(frozen=True)
class ForemanPassOutcome:
    """What one foreman did this pass — a value, never an exception."""

    name: str
    outcome: str  # "turn" | "idle" | "deferred" | "skipped" | "failed" | "dry_run"
    reason: str | None = None  # deferral reason or skip reason
    turn_id: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class PassResult:
    """The whole pass: per-foreman outcomes plus pass-level short-circuits."""

    outcomes: tuple[ForemanPassOutcome, ...] = ()
    locked: bool = False  # another pass holds foremen/pass.lock
    disabled: bool = False  # foreman.enabled is false — the pass is a no-op


def _stamp(now: datetime) -> str:
    """The ``journal.utc_now`` timestamp shape for ``now``."""
    return now.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --- Turn directory management ------------------------------------------------


def _new_turn_id(foreman_root: Path, now: datetime, *, dry_run: bool) -> str:
    """Mint ``turn-<UTC seconds>`` (or ``dry-`` for --dry-run), suffixed on collision."""
    prefix = fpaths.DRY_TURN_DIR_PREFIX if dry_run else fpaths.TURN_DIR_PREFIX
    base = f"{prefix}{now.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    turn_id = base
    suffix = 2
    while True:
        try:
            if not fpaths.turn_dir(foreman_root, turn_id).exists():
                return turn_id
        except ValueError:
            return turn_id
        turn_id = f"{base}-n{suffix}"
        suffix += 1


def _resumable_turn(foreman_root: Path) -> str | None:
    """The newest ``turn-*`` dir holding ``actions.json`` without ``complete.json``.

    The set is at most one under the Turn lock, but a stray second candidate
    resolves newest-first so an abandoned older dir can never starve it.
    """
    try:
        root = fpaths.turns_root(foreman_root)
    except ValueError:
        return None
    if not root.is_dir():
        return None
    candidates: list[str] = []
    for entry in root.iterdir():
        if not entry.is_dir() or not entry.name.startswith(fpaths.TURN_DIR_PREFIX):
            continue
        if entry.name.startswith(fpaths.DRY_TURN_DIR_PREFIX):
            continue
        try:
            actions = fpaths.actions_path(foreman_root, entry.name)
            complete = fpaths.complete_path(foreman_root, entry.name)
        except ValueError:
            continue
        if actions.is_file() and not complete.is_file():
            candidates.append(entry.name)
    return max(candidates) if candidates else None


def _read_turn_meta(foreman_root: Path, turn_id: str) -> dict[str, Any] | None:
    """The ``turn.json`` sidecar for a resumable Turn, or ``None``."""
    try:
        path = fpaths.turn_meta_path(foreman_root, turn_id)
    except ValueError:
        return None
    try:
        meta: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def _write_complete(foreman_root: Path, turn_id: str, outcomes: tuple) -> None:
    """Write ``complete.json`` — the marker that closes a Turn to resume."""
    try:
        write_json_atomic(
            fpaths.complete_path(foreman_root, turn_id),
            {
                "finished_at": utc_now(),
                "applied": sum(1 for o in outcomes if o.outcome == OUTCOME_APPLIED),
                "refused": sum(1 for o in outcomes if o.outcome == OUTCOME_REFUSED),
                "skipped": sum(1 for o in outcomes if o.outcome == OUTCOME_SKIPPED),
            },
        )
    except (OSError, ValueError):
        pass  # the marker is best-effort; a missing one only means a resume


def _prune_turns(foreman_root: Path, *, keep: int = TURN_RETENTION) -> None:
    """Keep the newest ``keep`` Turn directories; remove the rest (design §6).

    Name order is the age order — ids embed the UTC start timestamp. Foreign
    entries (not ours, or resolving outside ``turns/``) are never touched.
    """
    try:
        root = fpaths.turns_root(foreman_root)
    except ValueError:
        return
    if not root.is_dir():
        return
    ours = [
        entry
        for entry in root.iterdir()
        if entry.is_dir()
        and not entry.is_symlink()
        and (
            entry.name.startswith(fpaths.TURN_DIR_PREFIX)
            or entry.name.startswith(fpaths.DRY_TURN_DIR_PREFIX)
        )
    ]
    for doomed in sorted(ours, key=lambda p: p.name, reverse=True)[keep:]:
        try:
            resolved = doomed.resolve()
        except OSError:
            continue
        try:
            root_resolved = root.resolve()
        except OSError:
            return
        if root_resolved not in resolved.parents:
            continue  # a junction landing outside turns/ is foreign — never delete
        shutil.rmtree(doomed, ignore_errors=True)


# --- failure accounting ---------------------------------------------------------


def _record_failure(
    foreman: Foreman, *, stage: str, error: str, turn_id: str | None, now: datetime
) -> str:
    """Increment ``consecutive_failures``; ``faulted`` + report item at the limit.

    Returns the detail text for the pass outcome. Best-effort: a state save
    that fails still reports the original error.
    """
    root = foreman.root
    state_now = fstate.read_state_snapshot(root)
    failures = state_now.consecutive_failures + 1
    faulted = failures >= FAULT_LIMIT
    updates: dict[str, Any] = {"consecutive_failures": failures}
    if faulted:
        updates["state"] = "faulted"
    # write-gate-exempt(issue=3122): no write_gate receiver on the pass — the failure counter is the foreman's own state.json.
    saved = fstate.save_state(root, fstate.with_updates(state_now, **updates))
    if turn_id is not None:
        fjournal.append_journal(root, f"Turn {turn_id}: {stage} failed: {error}", turn_id=turn_id)
    detail = error
    if faulted:
        fitems.create_item(
            root,
            kind=fitems.ITEM_REPORT,
            text=(
                f"Foreman {foreman.name} faulted: {failures} consecutive Turn "
                f"failures (last: {error}). No further Turns until the operator "
                "resumes it."
            ),
            command=f"charlie gang resume {foreman.name}",
        )
        detail = f"{error} — foreman faulted after {failures} consecutive failures"
    if not saved.ok:
        detail = f"{detail} (state save failed: {saved.error})"
    # write-gate-exempt(issue=3122): same shape as apply._emit — no write_gate receiver on the pass.
    log_event(  # event-consumer: audit-only — consecutive_failures is the record Apply reads
        fpaths.state_path(root),
        EVENT_TURN_FAILED,
        {
            "foreman": foreman.name,
            "turn_id": turn_id,
            "stage": stage,
            "error": error,
            "consecutive_failures": failures,
            "faulted": faulted,
        },
        repo=foreman.repo_key,
    )
    # Issue #3123: notify runs after EVERY Turn, not only successful ones — a
    # faulted foreman's report item above is exactly the notification that
    # must not wait for a Turn that will never come.
    notify_unnotified_items(root, resolve_fleet_notify(foreman.config.notify, foreman.state_root))
    return detail


# --- the Turn -----------------------------------------------------------------


def _apply_gh(foreman: Foreman, *, dry_run: bool) -> Any:
    """The issue client for Apply — the repo's configured backend."""
    return github_client_for(foreman.repo_root, foreman.config, github=GitHub, dry_run=dry_run)


def _finish_turn(
    foreman: Foreman,
    turn_id: str,
    meta: dict[str, Any],
    applied: Any,
    now: datetime,
) -> ForemanPassOutcome:
    """Journal line, cursor/fingerprint persist, marker, notify, usage event.

    Idempotent on ``state.last_turn_id``: a pass resuming a Turn whose
    bookkeeping already landed (crash between state save and marker write)
    only repairs the marker — it must not double-count turns or re-append
    the journal line.
    """
    root = foreman.root
    wake = str(meta.get("wake") or "")
    state_now = fstate.read_state_snapshot(root)
    if state_now.last_turn_id != turn_id:
        if not any(e.turn_id == turn_id for e in fjournal.read_journal(root)):
            fjournal.append_journal(
                root,
                f"Turn {turn_id} ({wake}): {applied.turn_summary}",
                turn_id=turn_id,
            )
        register = meta.get("issues")
        issues = {
            int(number): fstate.AuthoredIssue(
                number=int(number), body_hash=str(entry.get("body_hash", ""))
            )
            for number, entry in (register or {}).items()
            if isinstance(entry, dict)
        }
        scheduled = wake == WAKE_REPORT_DUE
        # write-gate-exempt(issue=3122): no write_gate receiver on the pass — this save IS the Turn's bookkeeping landing.
        saved = fstate.save_state(
            root,
            fstate.with_updates(
                state_now,
                event_cursor=int(meta.get("event_cursor") or 0),
                inbox_cursor=int(meta.get("inbox_cursor") or 0),
                signal_cursor=int(meta.get("signal_cursor") or 0),
                lifecycle_fingerprint=str(meta.get("lifecycle_fingerprint") or ""),
                issues=issues,
                turns_total=state_now.turns_total + 1,
                turns_today=(
                    state_now.turns_today + 1 if state_now.turns_today_date == _today(now) else 1
                ),
                turns_today_date=_today(now),
                consecutive_failures=0,
                last_turn_at=_stamp(now),
                last_scheduled_turn_at=(
                    _stamp(now) if scheduled else state_now.last_scheduled_turn_at
                ),
                last_turn_id=turn_id,
            ),
        )
        if not saved.ok:
            detail = _record_failure(
                foreman,
                stage="bookkeeping",
                error=f"state.json: {saved.error}",
                turn_id=turn_id,
                now=now,
            )
            return ForemanPassOutcome(foreman.name, "failed", turn_id=turn_id, detail=detail)
        _write_complete(root, turn_id, applied.outcomes)
        decide_meta = meta.get("decide") if isinstance(meta.get("decide"), dict) else {}
        # write-gate-exempt(issue=3122): no write_gate receiver on the pass — the usage event is §10's ledger record for the Turn.
        log_event(  # event-consumer: audit-only — turns_total/state.json is the usage record
            fpaths.state_path(root),
            EVENT_TURN_COMPLETED,
            {
                "foreman": foreman.name,
                "turn_id": turn_id,
                "wake": wake,
                "harness": decide_meta.get("harness"),
                "model": decide_meta.get("model"),
                "input_tokens": int(decide_meta.get("input_tokens") or 0),
                "output_tokens": int(decide_meta.get("output_tokens") or 0),
                "num_turns": int(decide_meta.get("num_turns") or 0),
                "cost_usd": float(decide_meta.get("cost_usd") or 0.0),
                "duration_seconds": float(decide_meta.get("duration_seconds") or 0.0),
                "applied": sum(1 for o in applied.outcomes if o.outcome == OUTCOME_APPLIED),
                "refused": sum(1 for o in applied.outcomes if o.outcome == OUTCOME_REFUSED),
                "skipped": sum(1 for o in applied.outcomes if o.outcome == OUTCOME_SKIPPED),
            },
            repo=foreman.repo_key,
        )
    else:
        _write_complete(root, turn_id, applied.outcomes)
    # Items raised by this Apply (or stamped-but-unsent earlier) go to the
    # notify sink here — design §8.3's "item store, then notify" tail. The
    # file_path sentinel resolves against the foreman's own state root.
    notify_unnotified_items(root, resolve_fleet_notify(foreman.config.notify, foreman.state_root))
    _prune_turns(root)
    counts = (
        sum(1 for o in applied.outcomes if o.outcome == OUTCOME_APPLIED),
        sum(1 for o in applied.outcomes if o.outcome == OUTCOME_REFUSED),
        sum(1 for o in applied.outcomes if o.outcome == OUTCOME_SKIPPED),
    )
    return ForemanPassOutcome(
        foreman.name,
        "turn",
        turn_id=turn_id,
        detail=f"{applied.turn_summary} ({counts[0]} applied, {counts[1]} refused, {counts[2]} skipped)",
    )


def _run_turn(
    foreman: Foreman,
    *,
    resume_id: str | None,
    wake: str,
    dry_run: bool,
    no_decide: bool,
    now: datetime,
) -> ForemanPassOutcome:
    """One Turn: Collect → Decide → Apply → bookkeeping, or a resume at Apply."""
    root = foreman.root
    if resume_id is not None:
        meta = _read_turn_meta(root, resume_id)
        if meta is None:
            detail = _record_failure(
                foreman,
                stage="resume",
                error=f"{fpaths.TURN_META_FILENAME}: missing or unreadable",
                turn_id=resume_id,
                now=now,
            )
            return ForemanPassOutcome(foreman.name, "failed", turn_id=resume_id, detail=detail)
        meta.setdefault("wake", wake)
        applied = apply_turn(
            root,
            resume_id,
            charter=foreman.charter,
            gh=_apply_gh(foreman, dry_run=False),
            labels=foreman.labels,
            signal_marked=bool(meta.get("signal_marked")),
        )
        if not applied.ok:
            detail = _record_failure(
                foreman, stage="apply", error=applied.error or "", turn_id=resume_id, now=now
            )
            return ForemanPassOutcome(foreman.name, "failed", turn_id=resume_id, detail=detail)
        return _finish_turn(foreman, resume_id, meta, applied.value, now)

    turn_id = _new_turn_id(root, now, dry_run=dry_run)
    try:
        turn_dir = fpaths.turn_dir(root, turn_id)
        turn_dir.mkdir(parents=True)
    except (OSError, ValueError) as exc:
        return ForemanPassOutcome(
            foreman.name, "failed", turn_id=turn_id, detail=f"turn dir: {exc}"
        )

    briefing = build_briefing(foreman, now)
    if not briefing.ok:
        error = briefing.error or ""
        if "status snapshot" in error:
            _emit_deferred(foreman, reason=DEFER_STALE_SNAPSHOT, wake=wake, detail=error)
            return ForemanPassOutcome(
                foreman.name, "deferred", reason=DEFER_STALE_SNAPSHOT, detail=error
            )
        detail = _record_failure(foreman, stage="collect", error=error, turn_id=turn_id, now=now)
        return ForemanPassOutcome(foreman.name, "failed", turn_id=turn_id, detail=detail)
    brief = briefing.value
    briefing_text = brief.render()
    try:
        write_text_atomic(fpaths.briefing_path(root, turn_id), briefing_text)
    except (OSError, ValueError) as exc:
        detail = _record_failure(
            foreman, stage="collect", error=f"briefing.md: {exc}", turn_id=turn_id, now=now
        )
        return ForemanPassOutcome(foreman.name, "failed", turn_id=turn_id, detail=detail)

    meta: dict[str, Any] = {
        "turn_id": turn_id,
        "wake": wake,
        "started_at": _stamp(now),
        "dry_run": dry_run,
        "signal_marked": brief.signal_marked,
        "event_cursor": brief.event_cursor,
        "inbox_cursor": brief.inbox_cursor,
        "signal_cursor": brief.signal_cursor,
        "lifecycle_fingerprint": brief.lifecycle_fingerprint,
        "issues": {
            str(entry.number): {"body_hash": entry.body_hash} for entry in brief.issue_register
        },
    }
    try:
        write_json_atomic(fpaths.turn_meta_path(root, turn_id), meta)
    except (OSError, ValueError) as exc:
        detail = _record_failure(
            foreman, stage="collect", error=f"turn.json: {exc}", turn_id=turn_id, now=now
        )
        return ForemanPassOutcome(foreman.name, "failed", turn_id=turn_id, detail=detail)

    if no_decide:
        _write_complete(root, turn_id, ())
        return ForemanPassOutcome(
            foreman.name,
            "dry_run",
            reason="no_decide",
            turn_id=turn_id,
            detail=f"briefing written to {fpaths.briefing_path(root, turn_id)}",
        )

    decided = decide(
        DecideForeman(root=root, config=foreman.config),
        briefing_text,
        TURN_OUTPUT_SCHEMA,
        turn_dir,
    )
    if decided.deferred:
        error = decided.error or "decide deferred"
        _emit_deferred(foreman, reason=DEFER_DECIDE, wake=wake, detail=error)
        return ForemanPassOutcome(
            foreman.name, "deferred", reason=DEFER_DECIDE, turn_id=turn_id, detail=error
        )
    if not decided.ok:
        detail = _record_failure(
            foreman, stage="decide", error=decided.error or "", turn_id=turn_id, now=now
        )
        return ForemanPassOutcome(foreman.name, "failed", turn_id=turn_id, detail=detail)

    entry = decided.entry
    meta["decide"] = {
        "harness": entry.harness if entry is not None else None,
        "model": entry.model if entry is not None else None,
        "input_tokens": decided.input_tokens,
        "output_tokens": decided.output_tokens,
        "num_turns": decided.num_turns,
        "cost_usd": decided.cost_usd,
        "duration_seconds": decided.duration_seconds,
    }
    try:
        write_json_atomic(fpaths.turn_meta_path(root, turn_id), meta)
        write_json_atomic(fpaths.actions_path(root, turn_id), decided.output)
    except (OSError, ValueError) as exc:
        detail = _record_failure(
            foreman, stage="decide", error=f"turn record: {exc}", turn_id=turn_id, now=now
        )
        return ForemanPassOutcome(foreman.name, "failed", turn_id=turn_id, detail=detail)

    applied = apply_turn(
        root,
        turn_id,
        charter=foreman.charter,
        gh=_apply_gh(foreman, dry_run=dry_run),
        labels=foreman.labels,
        signal_marked=brief.signal_marked,
        dry_run=dry_run,
    )
    if not applied.ok:
        detail = _record_failure(
            foreman, stage="apply", error=applied.error or "", turn_id=turn_id, now=now
        )
        return ForemanPassOutcome(foreman.name, "failed", turn_id=turn_id, detail=detail)
    result = applied.value
    if dry_run:
        _write_complete(root, turn_id, result.outcomes)
        plan = [
            {
                "type": outcome.type,
                "key": outcome.key,
                "outcome": outcome.outcome,
                "reason": outcome.reason,
            }
            for outcome in result.outcomes
        ]
        return ForemanPassOutcome(
            foreman.name,
            "dry_run",
            turn_id=turn_id,
            detail=json.dumps({"turn_summary": result.turn_summary, "plan": plan}),
        )
    return _finish_turn(foreman, turn_id, meta, result, now)


# --- the pass -----------------------------------------------------------------


def _pass_one(
    fleet_dir: Path,
    name: str,
    *,
    fleet_paused: bool,
    dry_run: bool,
    no_decide: bool,
    now: datetime,
    foreman_config: ForemanConfig,
) -> ForemanPassOutcome:
    """Wake-evaluate and (maybe) run one foreman's Turn this pass."""
    loaded = load_foreman(fleet_dir, name)
    if not loaded.ok:
        return ForemanPassOutcome(name, "failed", detail=loaded.error)
    foreman = loaded.value
    if foreman.state.state not in ("active", "degraded"):
        return ForemanPassOutcome(name, "skipped", reason=f"state:{foreman.state.state}")

    # Structural limits first (design §10, issue #3123): breaker, unattended
    # limit, and the once-per-foreman verdict are evaluated at the start of
    # each pass so a transition applies to the Turn about to run. A
    # just-degraded foreman still runs Turns — degraded gates arming, not
    # planning — so the rest of the pass continues with the updated state.
    # --dry-run skips the evaluation entirely: like the pass-completed event
    # and Apply itself, a dry pass must not persist degraded state, raise
    # items, or set forced_arming on a live foreman.
    if not dry_run:
        flimits.evaluate_limits(
            foreman, fleet_dir=fleet_dir, breaker=foreman_config.breaker, now=now
        )

    # A crash survivor is outstanding work the pass owes the foreman — it
    # wakes the foreman even when no predicate fires (design §8.3).
    resume_id = None if dry_run else _resumable_turn(foreman.root)
    reason = wake_reason(foreman, now=now)
    if resume_id is None and reason is None:
        return ForemanPassOutcome(name, "idle")
    wake = reason or WAKE_RESUME

    deferral = _deferral_reason(foreman, fleet_paused=fleet_paused, now=now)
    if deferral is not None:
        _emit_deferred(foreman, reason=deferral, wake=wake, detail=None)
        return ForemanPassOutcome(name, "deferred", reason=deferral)

    lock = try_acquire_byte_range_lock(fpaths.turn_lock_path(foreman.root))
    if lock is None:
        return ForemanPassOutcome(name, "skipped", reason="turn lock held")
    try:
        return _run_turn(
            foreman,
            resume_id=resume_id,
            wake=wake,
            dry_run=dry_run,
            no_decide=no_decide,
            now=now,
        )
    finally:
        lock.release()


def run_pass(
    fleet_dir: Path,
    *,
    only: str | None = None,
    dry_run: bool = False,
    no_decide: bool = False,
    now: datetime | None = None,
) -> Result[PassResult]:
    """Run one foreman pass over ``fleet_dir``; every failure is a value.

    ``only`` restricts the pass to one named foreman. ``dry_run`` runs
    Collect + Decide and reports what Apply would do with no effect;
    ``no_decide`` (``--dry-run`` only) stops after writing ``briefing.md``.
    """
    if no_decide and not dry_run:
        return Result(error="--no-decide requires --dry-run")
    fleet_dir = Path(fleet_dir)
    now = now or datetime.now(UTC)

    # The repo slot resolves to ``<fleet_dir>/orchestrator.config.yaml``, which
    # never exists — the pass reads only the global layer, which is also the
    # only layer allowed to hold the host-wide ``foreman:`` section.
    try:
        config = load_layered_config(fleet_dir, None, fleet_dir_override=str(fleet_dir))
    except (ConfigError, OSError) as exc:
        return Result(error=f"config failed to load: {exc}")
    foreman_config = getattr(config, "foreman", None)
    if foreman_config is None:
        foreman_config = ForemanConfig()
    if not getattr(foreman_config, "enabled", True):
        return Result(value=PassResult(disabled=True))

    if only is not None:
        resolved = store.resolve_foreman(fleet_dir, only)
        if not resolved.ok:
            return Result(error=resolved.error)
        names = [only]
    else:
        names = [name for name, _root in store.list_foremen(fleet_dir)]
    if not names:
        return Result(value=PassResult())

    pass_lock = try_acquire_byte_range_lock(fpaths.pass_lock_path(fleet_dir))
    if pass_lock is None:
        return Result(value=PassResult(locked=True))
    try:
        paused = fleet_pause_pending(str(fleet_dir))
        outcomes = tuple(
            _pass_one(
                fleet_dir,
                name,
                fleet_paused=paused,
                dry_run=dry_run,
                no_decide=no_decide,
                now=now,
                foreman_config=foreman_config,
            )
            for name in names
        )
        result = PassResult(outcomes=outcomes)
        if not dry_run:
            # The fleet-level "a pass ran" signal the heartbeat's
            # pass-freshness alarm reads (design §10, issue #3123); siblings
            # of supervisor-heartbeat.json land in the fleet events.db.
            # write-gate-exempt(issue=3123): fleet-level event, no write_gate receiver — mirrors the completed pass into the fleet events.db like foreman_pass_start_failed.
            log_event(  # event-consumer: audit-only — the alarm recomputes from the event ts
                layout.state_file_path(fleet_dir),
                EVENT_PASS_COMPLETED,
                {
                    "foremen": len(outcomes),
                    "outcomes": {
                        o.outcome: sum(1 for x in outcomes if x.outcome == o.outcome)
                        for o in outcomes
                    },
                },
            )
        return Result(value=result)
    finally:
        pass_lock.release()


__all__ = [
    "DEFER_DECIDE",
    "DEFER_FLEET_PAUSED",
    "DEFER_MAX_TURNS",
    "DEFER_MIN_INTERVAL",
    "DEFER_STALE_SNAPSHOT",
    "EVENT_PASS_COMPLETED",
    "EVENT_TURN_COMPLETED",
    "EVENT_TURN_FAILED",
    "EVENT_WAKE_DEFERRED",
    "FAULT_LIMIT",
    "ForemanPassOutcome",
    "PassResult",
    "TURN_RETENTION",
    "WAKE_INBOX",
    "WAKE_ITEM_ANSWER",
    "WAKE_LIFECYCLE",
    "WAKE_QUIET",
    "WAKE_REPORT_DUE",
    "WAKE_RESUME",
    "WAKE_SIGNAL",
    "run_pass",
    "wake_reason",
]
