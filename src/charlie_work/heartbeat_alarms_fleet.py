"""Host/fleet-level pure alarm evaluators (companion to ``heartbeat_alarms``).

Loop/log freshness, supervisor heartbeat, wedge-kill loop and notify digest:
same contract as ``heartbeat_alarms`` -- rows/values already read by the caller,
``now`` injected, ``Finding`` out, stdlib only.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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

# Healthy worst-case gap between loop passes measured 53.9m; see
# heartbeat_check.py's history for the data behind the 90m coarse backstop.
LOOP_PASS_STALE_MINUTES = 90
LOG_FRESHNESS_STALE_MINUTES = 30
# Stale supervisor heartbeat = this multiple of the pass timeout the heartbeat
# itself records (max_pass_runtime_seconds, else full_pass_interval_seconds).
SUPERVISOR_HEARTBEAT_STALE_MULTIPLIER = 2
SUPERVISOR_HEARTBEAT_DEFAULT_PASS_TIMEOUT_SECONDS = 1800
SUPERVISOR_WEDGE_LOOP_LOOKBACK_HOURS = 24
NOTIFY_DIGEST_STALE_HOURS = 72


def eval_loop_pass_freshness(
    slug: str,
    newest_ts: str | None,
    now: datetime,
    stale_min: int = LOOP_PASS_STALE_MINUTES,
    *,
    marker_hint: str = "",
) -> Finding:
    """``newest_ts`` = ``MAX(ts)`` of ``loop_started`` (None = no rows yet)."""
    check = f"loop-pass-freshness {slug}"
    if newest_ts is None:
        return ok_finding(check, slug, "no loop_started rows recorded yet")
    newest_dt = parse_iso(newest_ts)
    if newest_dt is None:
        return anomaly_finding(check, slug, f"newest loop_started ts unparseable: {newest_ts!r}")
    age_min = (now - newest_dt).total_seconds() / 60
    facts = f"newest_loop_started={newest_ts} age={round(age_min)}m"
    if age_min > stale_min:
        return Finding(
            check,
            slug,
            "anomaly",
            f"no loop pass in {slug} for {round(age_min)}m "
            f"(newest loop_started {newest_ts}), threshold={stale_min}m -- "
            f"at or beyond the observed healthy worst case, so the supervisor "
            f"may be dead or wedged. Cause is open-ended at this duration; "
            f"{marker_hint} is one thing worth checking, not the only one. "
            f"({facts})",
            facts,
        )
    return ok_finding(check, slug, facts)


def eval_log_freshness(
    slug: str,
    newest_mtime_epoch: float | None,
    name: str,
    now: datetime,
    stale_min: int = LOG_FRESHNESS_STALE_MINUTES,
) -> Finding:
    """Freshest of the state dir's log/state/checkpoint files (None = none exist)."""
    check = f"log-freshness {slug}"
    if newest_mtime_epoch is None:
        return anomaly_finding(check, slug, "no log/state/checkpoint files found under state dir")
    mtime = datetime.fromtimestamp(newest_mtime_epoch, tz=timezone.utc)
    age_min = (now - mtime).total_seconds() / 60
    facts = f"freshest={name} age={round(age_min)}m"
    if age_min > stale_min:
        return Finding(
            check,
            slug,
            "anomaly",
            f"freshest file older than threshold={stale_min}m ({facts})",
            facts,
        )
    return ok_finding(check, slug, facts)


def _pass_timeout_seconds(data: dict[str, Any]) -> int:
    default = SUPERVISOR_HEARTBEAT_DEFAULT_PASS_TIMEOUT_SECONDS
    try:
        raw = data.get("max_pass_runtime_seconds")
        timeout = int(raw) if raw is not None else None
    except (TypeError, ValueError):
        timeout = None
    if timeout is not None and timeout > 0:
        return timeout
    try:
        raw = data.get("full_pass_interval_seconds")
        return int(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def eval_supervisor_heartbeat(data: Any, now: datetime) -> Finding:
    """``data`` = parsed ``supervisor-heartbeat.json`` (None = file absent)."""
    check = "supervisor-heartbeat"
    fname = "supervisor-heartbeat.json"
    if data is None:
        return anomaly_finding(
            check,
            None,
            f"no {fname} found (supervisor has never started, or the heartbeat was wiped)",
        )
    if not isinstance(data, dict):
        return anomaly_finding(check, None, f"{fname} malformed (not a JSON object)")
    last_beat = parse_iso(data.get("last_beat_at"))
    if last_beat is None:
        return anomaly_finding(check, None, f"{fname} has no parseable last_beat_at")

    age_min = (now - last_beat).total_seconds() / 60.0
    exited_at = data.get("exited_at")
    pass_timeout = _pass_timeout_seconds(data)
    stale_min = (SUPERVISOR_HEARTBEAT_STALE_MULTIPLIER * pass_timeout) / 60.0
    facts = (
        f"last_beat={round(age_min)}m ago pid={data.get('pid')} exited_at={exited_at} "
        f"pass_timeout={pass_timeout}s"
    )
    if age_min <= stale_min:
        return ok_finding(check, None, facts)
    if exited_at is not None:
        detail = (
            f"supervisor exited cleanly at {exited_at} but has not restarted in "
            f"{round(age_min)}m (threshold={round(stale_min)}m) — the "
            f"watchdog may be disabled ({facts})"
        )
    else:
        detail = (
            f"supervisor heartbeat stale: last beat {round(age_min)}m ago with no "
            f"clean exit (threshold={round(stale_min)}m) — likely killed "
            f"or hung ({facts})"
        )
    return Finding(check, None, "anomaly", detail, facts)


def eval_wedge_kill_loop(
    ts_list: Sequence[str],
    now: datetime,
    lookback_h: int = SUPERVISOR_WEDGE_LOOP_LOOKBACK_HOURS,
) -> Finding:
    """``ts_list`` = every fleet-level ``supervisor_wedge_loop`` ts."""
    check = "supervisor-wedge-kill-loop"
    cutoff = now - timedelta(hours=lookback_h)
    recent = [ts for ts in ts_list if ((dt := parse_iso(ts)) is None or dt >= cutoff)]
    facts = f"total_events={len(ts_list)} recent={len(recent)} lookback_hours={lookback_h}"
    if not recent:
        return ok_finding(check, None, facts)
    return Finding(
        check,
        None,
        "anomaly",
        f"supervisor_wedge_loop fired {len(recent)} time(s) in the last "
        f"{lookback_h}h (most recent {max(recent)}) -- "
        f"the wedge-kill backstop is looping instead of recovering ({facts})",
        facts,
    )


def eval_notify_digest(
    resolution_row: tuple[Any, Any] | None,
    stale_ts_list: Sequence[str],
    now: datetime,
    stale_hours: int,
    probe_digest: Callable[[Path], Any],
) -> Finding:
    """Verdict on the digest writer from fleet events.

    ``resolution_row`` = latest ``notify_resolution`` ``(ts, payload)`` (None =
    none yet); ``stale_ts_list`` = ``notify_digest_stale`` ts values.
    ``probe_digest`` is the caller's read-only file probe returning an object
    with ``exists``/``age_hours``/``age_source``/``error`` (``DigestProbe``);
    injected so this stays I/O-free. Cannot-tell outcomes are WARN.
    """
    check = "notify-digest"
    cutoff = now - timedelta(hours=stale_hours)
    recent_stale = [ts for ts in stale_ts_list if ((dt := parse_iso(ts)) is None or dt >= cutoff)]
    stale_fact = f"stale_events_{stale_hours}h={len(recent_stale)}"

    def warn(detail: str) -> Finding:
        return Finding(check, None, "warn", detail)

    if resolution_row is None:
        return warn(
            "no notify_resolution event yet -- the supervisor predates this "
            f"instrumentation or died before its startup report ({stale_fact})"
        )
    res_ts_raw, res_payload_raw = resolution_row
    try:
        resolution = json.loads(res_payload_raw) if isinstance(res_payload_raw, str) else None
    except json.JSONDecodeError:
        resolution = None
    if not isinstance(resolution, dict):
        return warn(
            f"latest notify_resolution payload is not a JSON object: {str(res_payload_raw)[:120]!r}"
        )
    res_ts = parse_iso(res_ts_raw if isinstance(res_ts_raw, str) else None)
    res_fact = f"resolved at {(res_ts.isoformat() if res_ts else res_ts_raw)}"

    if not resolution.get("enabled"):
        return warn(
            "supervisor resolved notify enabled=false -- the digest writer "
            "is off. If this fleet opted in to notifications, its notify: "
            f"block was lost ({res_fact}; {stale_fact})"
        )
    sink = str(resolution.get("sink") or "file").lower()
    if sink != "file":
        return ok_finding(
            check, None, f"enabled with sink={sink} (no digest file to tail; {res_fact})"
        )
    resolved_raw = resolution.get("resolved_file_path")
    if resolution.get("file_path_empty") or not isinstance(resolved_raw, str) or not resolved_raw:
        return anomaly_finding(
            check,
            None,
            "enabled with sink=file but file_path is unset -- every emit "
            "fails 'file_path is empty' (per the supervisor's own "
            f"notify_resolution event; {res_fact})",
        )
    digest_path = Path(resolved_raw)
    probe = probe_digest(digest_path)
    if probe.error is not None:
        return anomaly_finding(
            check, None, f"{digest_path} unreadable: {probe.error} ({stale_fact})"
        )
    if not probe.exists:
        return anomaly_finding(
            check,
            None,
            f"notify enabled but {digest_path} does not exist -- the writer "
            f"has never landed a line ({res_fact}; {stale_fact})",
        )
    age_hours = probe.age_hours if probe.age_hours is not None else 0.0
    facts = (
        f"last entry {age_hours:.1f}h old ({probe.age_source}); "
        f"threshold={stale_hours}h path={digest_path}; {stale_fact}"
    )
    if age_hours > stale_hours:
        return Finding(
            check,
            None,
            "anomaly",
            f"notify digest writer looks dead: {facts} -- the enabled file "
            "sink has produced nothing past the staleness bound",
            facts,
        )
    return ok_finding(check, None, facts)
