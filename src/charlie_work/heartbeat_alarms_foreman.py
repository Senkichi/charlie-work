"""Foreman alarm evaluators (issue #3123, design §10).

The five foreman heartbeat checks share one definition here — a third
stdlib-only leaf beside ``heartbeat_alarms``/``heartbeat_alarms_fleet``, so
``scripts/heartbeat_check.py`` and the dashboard evaluate the identical
verdicts: rows/state values already read by the caller, ``now`` injected,
``Finding`` out. Reads nothing and never touches the clock.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

try:
    from charlie_work.heartbeat_alarms import Finding, anomaly_finding, ok_finding, parse_iso
except ImportError:  # pragma: no cover - exercised by the file-path-load subprocess test
    # Loaded by file path (scripts/heartbeat_*.py with ``charlie_work`` not
    # importable): resolve the sibling leaf the same way, sharing the module
    # object the loader already registered so ``Finding`` has one identity.
    import importlib.util as _ilu
    import sys as _sys

    _LEAF = "_cw_heartbeat_alarms"
    _mod = _sys.modules.get(_LEAF)
    if _mod is None:
        _spec = _ilu.spec_from_file_location(
            _LEAF, Path(__file__).with_name("heartbeat_alarms.py")
        )
        _mod = _ilu.module_from_spec(_spec)
        _sys.modules[_LEAF] = _mod
        _spec.loader.exec_module(_mod)
    Finding, anomaly_finding, ok_finding, parse_iso = (
        _mod.Finding,
        _mod.anomaly_finding,
        _mod.ok_finding,
        _mod.parse_iso,
    )

#: A Turn's tokens spike past this multiple of the trailing median.
FOREMAN_TURN_SPIKE_MULTIPLIER = 3
#: How many Turns (before the newest) form the trailing-median baseline.
FOREMAN_TURN_MEDIAN_WINDOW = 20
#: Stale pass = this multiple of the pass interval without a completed pass.
FOREMAN_PASS_STALE_MULTIPLIER = 2
#: The ``state.degraded_reasons`` member the escalation breaker writes —
#: mirrored from ``foreman.limits.DEGRADED_BREAKER`` (ADR-0010 forbids the
#: import; the string is the on-disk contract).
FOREMAN_BREAKER_REASON = "breaker"


def eval_foreman_turn_tokens(
    name: str,
    turn_tokens: Sequence[int],
    spike_multiplier: int = FOREMAN_TURN_SPIKE_MULTIPLIER,
    window: int = FOREMAN_TURN_MEDIAN_WINDOW,
) -> Finding:
    """Alarm: a Turn's tokens exceed ``spike_multiplier``x the trailing median.

    ``turn_tokens`` = ``input_tokens + output_tokens`` of every
    ``foreman_turn_completed`` event in the foreman's own events.db, oldest
    first. The median window is the ``window`` Turns *preceding* the newest,
    so the spike itself cannot dilute its own baseline. A zero median means
    the prior Turns recorded no usage (pre-instrumentation or free turns) —
    no ratio exists, so the check is quiet rather than alarmed.
    """
    check = f"foreman-turn-tokens {name}"
    if len(turn_tokens) < 2:
        return ok_finding(check, name, f"turns={len(turn_tokens)} (need 2+ for a trailing median)")
    prior = list(turn_tokens[-(window + 1) : -1])
    median = statistics.median(prior)
    latest = turn_tokens[-1]
    facts = f"latest={latest} median_last{len(prior)}={median} threshold={spike_multiplier}x"
    if median <= 0:
        return ok_finding(check, name, f"median is zero -- no baseline to spike against ({facts})")
    if latest > spike_multiplier * median:
        return Finding(
            check,
            name,
            "anomaly",
            f"Turn token spike: latest Turn used {latest} tokens, "
            f"> {spike_multiplier}x the trailing median {median} of the "
            f"foreman's last {len(prior)} Turns ({facts})",
            facts,
        )
    return ok_finding(check, name, facts)


def eval_foreman_turns_per_day(
    name: str,
    turns_today: int,
    turns_today_date: str,
    bound: int | None,
    now: datetime,
) -> Finding:
    """Alarm: the charter's ``max_turns_per_day`` bound was reached today.

    ``turns_today``/``turns_today_date`` come from the foreman's state.json
    (the counter a Turn's bookkeeping advances); ``bound`` is
    ``charter.bounds.max_turns_per_day`` decoded by the caller, or ``None``
    when the charter is unreadable (the check then only reports the counter).
    """
    check = f"foreman-turns-per-day {name}"
    today = now.astimezone(timezone.utc).strftime("%Y-%m-%d")
    facts = f"turns_today={turns_today} date={turns_today_date or 'unset'} bound={bound}"
    if turns_today_date != today:
        return ok_finding(check, name, f"counter is not today's ({facts})")
    if bound is None:
        return ok_finding(check, name, f"no charter bound to compare ({facts})")
    if turns_today >= bound:
        return Finding(
            check,
            name,
            "anomaly",
            f"max_turns_per_day reached: {turns_today} Turns today against "
            f"bound {bound} -- the foreman can take no further Turn until "
            f"tomorrow ({facts})",
            facts,
        )
    return ok_finding(check, name, facts)


def eval_foreman_breaker(name: str, degraded_reasons: Sequence[str]) -> Finding:
    """Alarm: the escalation breaker has the foreman degraded.

    ``degraded_reasons`` = ``state.degraded_reasons`` from the foreman's
    state.json; ``"breaker"`` in it means the escalation-rate trip is active
    (new issues stay unarmed until the item is resolved or the rate drops).
    """
    check = f"foreman-breaker {name}"
    if FOREMAN_BREAKER_REASON in degraded_reasons:
        return Finding(
            check,
            name,
            "anomaly",
            f"escalation breaker tripped on foreman {name} -- degraded, new "
            "issues stay unarmed until the breaker's decision item is "
            "resolved or the rate falls back under the threshold "
            f"(degraded_reasons={sorted(degraded_reasons)})",
            f"degraded_reasons={sorted(degraded_reasons)}",
        )
    return ok_finding(check, name, f"degraded_reasons={sorted(degraded_reasons)}")


def eval_foreman_state(name: str, state: str, degraded_reasons: Sequence[str]) -> Finding:
    """Alarm: ``faulted`` is an anomaly; a non-breaker ``degraded`` is a warn.

    The unattended limit degrades a foreman until its stale decision item is
    resolved — operator action is needed but it is the limit working, not a
    fault, so it reports WARN rather than ANOMALY.
    """
    check = f"foreman-state {name}"
    if state == "faulted":
        return Finding(
            check,
            name,
            "anomaly",
            f"foreman {name} is faulted (3 consecutive failed Turns) -- "
            f"no further Turns until `charlie gang resume {name}`",
            f"state={state}",
        )
    if state == "degraded" and FOREMAN_BREAKER_REASON not in degraded_reasons:
        reasons = sorted(degraded_reasons) or ["unrecorded"]
        return Finding(
            check,
            name,
            "warn",
            f"foreman {name} is degraded ({', '.join(reasons)}) -- an open "
            f"decision item is past its max_unanswered_days bound; see "
            f"`charlie gang interventions {name}`",
            f"state={state} degraded_reasons={reasons}",
        )
    return ok_finding(check, name, f"state={state}")


def eval_foreman_pass_freshness(
    newest_ts: str | None,
    interval_minutes: float,
    now: datetime,
    stale_multiplier: float = FOREMAN_PASS_STALE_MULTIPLIER,
) -> Finding:
    """Alarm: no foreman pass seen for ``stale_multiplier``x its interval.

    ``newest_ts`` = ``MAX(ts)`` of the fleet-level ``foreman_pass_completed``
    events (None = the pass has never completed while foremen exist);
    ``interval_minutes`` = ``foreman.pass_interval_minutes`` from the fleet
    config. The caller skips this check entirely when no foremen exist, the
    foreman layer is disabled, or the fleet is paused — in all three the
    supervise loop legitimately never spawns the pass.
    """
    check = "foreman-pass-freshness"
    stale_min = stale_multiplier * interval_minutes
    if newest_ts is None:
        return anomaly_finding(
            check,
            None,
            "no foreman_pass_completed event recorded -- foremen exist but "
            "the pass has never run (the supervise loop's `charlie scheme` "
            "spawn may be failing; see foreman_pass_start_failed events)",
        )
    newest_dt = parse_iso(newest_ts)
    if newest_dt is None:
        return anomaly_finding(
            check, None, f"newest foreman_pass_completed ts unparseable: {newest_ts!r}"
        )
    age_min = (now - newest_dt).total_seconds() / 60.0
    facts = f"newest_pass={newest_ts} age={round(age_min)}m interval={interval_minutes}m threshold={round(stale_min)}m"
    if age_min > stale_min:
        return Finding(
            check,
            None,
            "anomaly",
            f"no foreman pass in {round(age_min)}m (newest "
            f"foreman_pass_completed {newest_ts}), "
            f"threshold={round(stale_min)}m = {stale_multiplier}x "
            f"interval -- the supervise loop stopped spawning `charlie "
            f"scheme` ({facts})",
            facts,
        )
    return ok_finding(check, None, facts)
