"""Adapter-agnostic worker abstraction for unified fleet supervision.

This module provides a unified view of worker sessions across all adapters
(devin-shell, claude-code, and future adapters). It collapses the duplicated
adapter-iteration loops in workflow.py into a single abstraction point.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from json import JSONDecodeError
from os import stat_result
from pathlib import Path
from typing import Any

from .claude_code import (
    ClaudeWorkerRecord,
    _sidecar_path as claude_sidecar_path,
    is_worker_alive,
    read_worker_records,
)
from .config import WRITER_MARKER_FILENAME, OrchestratorConfig
from .devin_shell import (
    SessionRecord,
    _sidecar_path as devin_sidecar_path,
    is_session_alive,
    read_session_records,
)
from .post_mortem import (
    RealActivityProbe,
    _events_path_from_log,
    real_activity_for_worker,
)

logger = logging.getLogger(__name__)


# Shim marker that indicates successful infra materialization (issue #221)
# Workers that hang at this line are detected as launch_stalled
_SHIM_MARKER = "[shim] .devin infra materialized"


def _log_has_shim_marker(log_path: Path) -> bool:
    """Check if a log file contains the shim marker line.

    Returns True if the marker is found in the log, False otherwise.
    This is used to detect whether a worker has progressed past the
    initial shim materialization phase (issue #221).

    Args:
        log_path: Path to the worker's log file

    Returns:
        True if the shim marker is found, False otherwise
    """
    if not log_path.exists():
        return False

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        return _SHIM_MARKER in log_text
    except OSError:
        return False


def _log_is_stalled_at_shim(
    log_path: Path,
    grace_minutes: int,
    now: datetime,
    *,
    real_activity_probe: RealActivityProbe | None = None,
) -> bool:
    """Check if a log is stalled at the shim marker (issue #221).

    A log is considered stalled at shim if:
    1. The shim marker is present in the log
    2. The log file has not been modified within the grace period
    3. The log is small (<= 1KB) - indicating no real progress
    4. Independent real-session activity (sessions.db, per-PID Devin log, or
       Claude Code events.jsonl) is also quiet past the grace period, and the
       probe was conclusive (see below)

    This detects workers that hang immediately after shim materialization
    with error:null, alive PID, and frozen logs, while avoiding false
    positives for shim-wrapped workers whose real activity lives outside
    the sidecar log (issue #280).

    Args:
        log_path: Path to the worker's log file
        grace_minutes: Grace period in minutes to allow for shim materialization
        now: Current datetime for staleness calculation
        real_activity_probe: Optional corroboration probe from real-session
            activity sources. When fresh, the sidecar stall is ignored. When
            inconclusive (``_real_activity_is_inconclusive`` — every source
            errored, or every source is error-free but no-match-yet), the
            stall verdict is also deferred rather than failing open (issue
            #307): this call site is reached only for a CONFIRMED-ALIVE
            worker and its True return drives an immediate process kill.

    Returns:
        True if the log is stalled at shim and real activity is quiet and
        conclusive, False otherwise
    """
    from datetime import UTC

    if not log_path.exists():
        return False

    try:
        log_stat = log_path.stat()
        log_mtime = datetime.fromtimestamp(log_stat.st_mtime, tz=UTC)
        log_size = log_stat.st_size

        # Check if log is small (frozen at shim marker, typically ~424-425 bytes)
        if log_size > 1024:  # More than 1KB means likely made progress
            return False

        # Check if log has shim marker
        if not _log_has_shim_marker(log_path):
            return False

        # Check if log is stale (no modification within grace period)
        age = now - log_mtime
        if age <= timedelta(minutes=grace_minutes):
            return False

        # Corroborate against real-session activity before declaring a stall
        if real_activity_probe is not None and real_activity_probe.is_fresh(grace_minutes):
            return False
        elif _real_activity_is_inconclusive(real_activity_probe):
            # Probe was consulted but produced no timestamp evidence either way
            # (issue #307: same single enforcement point as classify_worker_health's
            # Signal 3). That is insufficient evidence to kill a CONFIRMED-ALIVE
            # worker; defer rather than fail open to a stall verdict.
            return False

        return True
    except OSError:
        return False


class WorkerHealth(Enum):
    """Health status of a worker session.

    This enum provides a closed set of health states that the supervisor can use
    to classify worker sessions. It unifies liveness, staleness, and terminal-marker
    signals into a single classification point.

    Signal → verdict, first-to-fire-wins order:
    1. liveness → DEAD (unless a fresh probe vetoes, or an inconclusive probe
       defers up to the configured cap)
    2. terminal marker → DEAD (unconditional; overrides a fresh probe)
    3. progress staleness → STALLED (unless a fresh or inconclusive probe vetoes it)
    4. wall-clock deadline → SLOW (or RUNAWAY if wall_clock_kill=True)
    5. loop/no-progress (Claude Code only) → SLOW (or RUNAWAY if loop_kill=True, capped at SLOW for Devin)
    6. cost/token budget (Claude Code only) → SLOW (or RUNAWAY if cost_budget_action="kill")
    7. (none of the above) → HEALTHY

    SLOW, RUNAWAY, and ORPHANED are reserved for future issues (#162, #163, B6a).
    """

    HEALTHY = "healthy"
    SLOW = "slow"  # Reserved for #162 (wall-clock/loop tripwires) and #163 (warn mode)
    STALLED = "stalled"
    RUNAWAY = "runaway"  # Reserved for #162/#163 (cost/token tripwires)
    DEAD = "dead"
    ORPHANED = "orphaned"  # Reserved for B6a (sidecar dead/non-live but process still references worktree)


@dataclass(frozen=True)
class UsageSnapshot:
    """Cumulative usage metrics parsed from Claude Code's events.jsonl.

    This frozen dataclass holds the latest cumulative tokens and cost_usd
    values extracted from the events.jsonl file.
    """

    tokens: int | None = None
    cost_usd: float | None = None


def parse_cumulative_usage(events_path: Path) -> UsageSnapshot | None:
    """Read issue-<n>.events.jsonl and return cumulative tokens/cost.

    Returns None if the file doesn't exist (Devin sessions, or a Claude
    session that hasn't emitted a usage event yet) — absence is NOT an
    error and must never be treated as unhealthy.

    Malformed or partial trailing JSON lines (a worker killed mid-write)
    are skipped, not raised — same defensive posture as state.load_state's
    corrupt-file handling.

    Args:
        events_path: Path to the events.jsonl file

    Returns:
        UsageSnapshot with cumulative tokens/cost, or None if file doesn't exist
    """
    if not events_path.exists():
        return None

    tokens = None
    cost_usd = None

    try:
        with events_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed lines (including partial trailing lines)
                    continue

                if not isinstance(event, dict):
                    continue

                # Extract cumulative usage fields (take last-seen value)
                if "tokens" in event and isinstance(event["tokens"], int):
                    tokens = event["tokens"]
                if "cost_usd" in event and isinstance(event["cost_usd"], (int, float)):
                    cost_usd = float(event["cost_usd"])

    except OSError:
        # File read error - treat as no usage data
        return None

    # If we parsed no usage data, return None
    if tokens is None and cost_usd is None:
        return None

    return UsageSnapshot(tokens=tokens, cost_usd=cost_usd)


@dataclass(frozen=True)
class WorkerView:
    """Adapter-agnostic view of a worker session.

    This frozen dataclass provides a unified shape for worker records across
    all adapters, enabling single-loop iteration over the entire fleet without
    duplicating devin-shell/claude-code-specific code.
    """

    adapter_kind: str  # "devin" | "claude-code"
    issue_number: int
    repo_key: (
        str  # required — cross-repo disambiguation (fleet work); "" for single-repo callers today
    )
    pid: int | None
    started_at: str
    process_start_time: float | None
    log_path: str
    worktree_path: str
    error: str | None
    failure_kind: str | None
    reclaimed: str | None
    branch: str = ""
    last_activity_at: str | None = None  # ISO timestamp from log_path.stat().st_mtime
    log_bytes: int | None = None  # log_path.stat().st_size
    rate_limit_defer_until: str | None = (
        None  # ISO timestamp when the stall kill is deferred (issue #247)
    )
    inconclusive_probe_deferred_count: int = 0  # Signal-1 deferral counter (issue #338)
    session_id: str | None = None  # unique session id for worktree writer marker (issue #400)

    def is_alive(self) -> bool:
        """Check whether the process behind this worker is still running.

        Dispatches to the adapter-specific liveness probe based on adapter_kind.
        Preserves the existing PID + process_start_time recycling-safe check.
        """
        if self.adapter_kind == "devin":
            # Reconstruct a minimal SessionRecord for the liveness probe
            record = SessionRecord(
                issue_number=self.issue_number,
                branch="",  # Not used by is_session_alive
                worktree_path=self.worktree_path,
                prompt_path="",  # Not used by is_session_alive
                command=(),  # Not used by is_session_alive
                pid=self.pid,
                started_at=self.started_at,
                log_path=self.log_path,
                error=self.error,
                failure_kind=self.failure_kind,
                process_start_time=self.process_start_time,
                reclaimed=self.reclaimed,
                last_activity_at=self.last_activity_at,
                log_bytes=self.log_bytes,
                rate_limit_defer_until=self.rate_limit_defer_until,
                inconclusive_probe_deferred_count=self.inconclusive_probe_deferred_count,
                session_id=self.session_id,
            )
            return is_session_alive(record)
        elif self.adapter_kind == "claude-code":
            # Reconstruct a minimal ClaudeWorkerRecord for the liveness probe
            record = ClaudeWorkerRecord(
                issue_number=self.issue_number,
                branch="",  # Not used by is_worker_alive
                worktree_path=self.worktree_path,
                prompt_path="",  # Not used by is_worker_alive
                command=(),  # Not used by is_worker_alive
                pid=self.pid,
                started_at=self.started_at,
                log_path=self.log_path,
                error=self.error,
                failure_kind=self.failure_kind,
                process_start_time=self.process_start_time,
                reclaimed=self.reclaimed,
                last_activity_at=self.last_activity_at,
                log_bytes=self.log_bytes,
                rate_limit_defer_until=self.rate_limit_defer_until,
                inconclusive_probe_deferred_count=self.inconclusive_probe_deferred_count,
                session_id=self.session_id,
            )
            return is_worker_alive(record)
        elif self.adapter_kind == "api":
            # api workers are Claude Code CLI processes with provider env
            # injected; their sidecar/record shape is identical to claude-code
            # (issue-<n>.api.json via launch_claude_worker adapter_kind="api"),
            # so liveness delegates to the same is_worker_alive probe.
            record = ClaudeWorkerRecord(
                issue_number=self.issue_number,
                branch="",
                worktree_path=self.worktree_path,
                prompt_path="",
                command=(),
                pid=self.pid,
                started_at=self.started_at,
                log_path=self.log_path,
                error=self.error,
                failure_kind=self.failure_kind,
                process_start_time=self.process_start_time,
                reclaimed=self.reclaimed,
                last_activity_at=self.last_activity_at,
                log_bytes=self.log_bytes,
                rate_limit_defer_until=self.rate_limit_defer_until,
                inconclusive_probe_deferred_count=self.inconclusive_probe_deferred_count,
                session_id=self.session_id,
                adapter_kind="api",
            )
            return is_worker_alive(record)
        else:
            # Unknown adapter kind - conservatively treat as dead
            return False

    def log_stat(self) -> stat_result | None:
        """Stat() the log_path, swallow OSError -> None.

        Returns the os.stat_result for the log file if it exists and is accessible,
        None otherwise. This is useful for checking log file mtime for stall detection.
        """
        try:
            return Path(self.log_path).stat()
        except OSError:
            return None

    def _remove_writer_marker(self) -> None:
        """Remove this worker's in-worktree writer marker when the sidecar is reaped.

        Uses ``session_id`` to avoid wiping an unrelated operator-claim marker.
        """
        if not self.worktree_path or not self.session_id:
            return
        marker_path = Path(self.worktree_path) / WRITER_MARKER_FILENAME
        try:
            if not marker_path.exists():
                return
            with marker_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and data.get("session_id") == self.session_id:
                marker_path.unlink()
        except (OSError, json.JSONDecodeError):
            pass

    def reap_sidecar(
        self,
        sessions_dir: Path,
        *,
        api_config: Any = None,
        state_dir: Path | str | None = None,
    ) -> None:
        """Delete the sidecar file for this worker to prevent phantom sessions.

        Dispatches to the adapter-specific sidecar path function based on adapter_kind.
        Best-effort cleanup: OSError is swallowed to avoid failing the entire dead-session
        classification loop if a single unlink fails (the sidecar will be reaped on the
        next cycle). Also removes the in-worktree writer marker when it matches this
        worker's session id.

        This is called after a session is detected as dead and classified to prevent
        phantom sessions from PID recycling (issue #113).

        For ``adapter_kind == "api"`` (issue #480), when ``api_config`` and
        ``state_dir`` are supplied, the session's spend is settled into the
        api-budget ledger BEFORE the sidecar is unlinked (so the sidecar's
        provider field can still be read). Settlement is best-effort accounting:
        any failure is logged and swallowed so it can never break the reap.
        Callers without api context (legacy callers, tests, review-lane reaps)
        omit the kwargs and skip settlement — the unlink still happens.
        """
        if self.adapter_kind == "devin":
            sidecar_path = devin_sidecar_path(sessions_dir, self.issue_number)
        elif self.adapter_kind == "claude-code":
            sidecar_path = claude_sidecar_path(sessions_dir, self.issue_number)
        elif self.adapter_kind == "api":
            # api sidecars share the claude-code sidecar-path derivation,
            # routed through the adapter_kind-aware _sidecar_path helper so the
            # .api.json suffix is selected.
            sidecar_path = claude_sidecar_path(sessions_dir, self.issue_number, "api")
            # Best-effort spend settlement before the sidecar is unlinked (issue #480).
            if api_config is not None and state_dir is not None:
                self._settle_api_budget(sidecar_path, api_config, state_dir)
        else:
            # Unknown adapter kind - nothing to reap
            return

        try:
            sidecar_path.unlink(missing_ok=True)
        except OSError:
            # Best-effort cleanup - don't fail if unlink fails
            pass

        self._remove_writer_marker()

    def _settle_api_budget(
        self,
        sidecar_path: Path,
        api_config: Any,
        state_dir: Path | str,
    ) -> None:
        """Best-effort spend settlement for an api worker session (issue #480).

        Reads the sidecar for the provider name, parses the session's
        events.jsonl for token usage (via the shared ``iter_claude_events``
        primitive), computes USD from the provider's configured pricing, and
        atomically settles the entry into the api-budget ledger. Never raises
        — settlement is accounting, not enforcement, and must not break the
        sidecar reap that prevents phantom sessions.
        """
        from .api_budget import (
            SessionEntry,
            cost_usd,
            ledger_path,
            load_ledger,
            save_ledger,
            settle_session,
            usage_from_events,
        )
        from .claude_code import iter_claude_events

        try:
            provider_name = ""
            try:
                with sidecar_path.open("r", encoding="utf-8") as handle:
                    sidecar = json.load(handle)
                if isinstance(sidecar, dict):
                    provider_name = str(sidecar.get("provider") or "")
            except (OSError, json.JSONDecodeError):
                pass

            provider_cfg = api_config.providers.get(provider_name) if provider_name else None
            if provider_cfg is None:
                # No pricing for the resolved provider → cannot compute cost.
                # Skip settlement rather than recording a zero-cost entry that
                # would understate spend.
                return

            events_path = _events_path_from_log(Path(self.log_path))
            usage = usage_from_events(iter_claude_events(events_path))
            usd = cost_usd(usage, provider_cfg)
            ended_at = (
                datetime.now(timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
            entry = SessionEntry(
                issue=self.issue_number,
                session_id=self.session_id or "",
                provider=provider_name,
                model=provider_cfg.model,
                started_at=self.started_at,
                ended_at=ended_at,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_tokens=usage.cached_tokens,
                usd=usd,
                duration_s=self.runtime_seconds(),
                outcome=self.failure_kind or "reaped",
            )
            path = ledger_path(state_dir)
            ledger = load_ledger(path)
            ledger = settle_session(ledger, entry)
            save_ledger(path, ledger)
        except Exception:
            logger.warning(
                "api budget settlement failed for issue %s; reap continues",
                self.issue_number,
                exc_info=True,
            )

    def runtime_seconds(self) -> float:
        """Calculate runtime in seconds from started_at to now.

        Returns the elapsed time since the worker started. If started_at is invalid
        or cannot be parsed, returns 0.0.
        """
        from datetime import UTC

        try:
            started_at = datetime.fromisoformat(self.started_at)
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=UTC)
            now = datetime.now(UTC)
            return (now - started_at).total_seconds()
        except (ValueError, TypeError):
            return 0.0


def _tail_last_tool_call_timestamp(events_path: Path) -> datetime | None:
    """Extract the timestamp of the last tool_call event from an events.jsonl file.

    Reads the events.jsonl file line by line and returns the timestamp of the most
    recent event with type="tool_call". Returns None if the file doesn't exist,
    can't be read, or contains no tool_call events.

    Args:
        events_path: Path to the events.jsonl file

    Returns:
        datetime of the last tool_call event, or None if not found
    """
    from datetime import UTC

    if not events_path.exists():
        return None

    try:
        last_tool_call_at: datetime | None = None
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if isinstance(event, dict) and event.get("type") == "tool_call":
                        timestamp_str = event.get("timestamp")
                        if timestamp_str:
                            try:
                                timestamp = datetime.fromisoformat(timestamp_str)
                                if timestamp.tzinfo is None:
                                    timestamp = timestamp.replace(tzinfo=UTC)
                                last_tool_call_at = timestamp
                            except (ValueError, TypeError):
                                continue
                except (JSONDecodeError, TypeError):
                    continue
        return last_tool_call_at
    except OSError:
        return None


def _parse_started_at(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (ValueError, TypeError):
        return None


def _real_activity_is_fresh(probe: RealActivityProbe | None, stall_minutes: int) -> bool:
    """Return True when the probe shows real-session activity within the stall window.

    A fresh real-activity signal from any source (sessions.db, per-PID Devin log,
    Claude Code events.jsonl, or worktree file mtimes) is enough to veto an
    immediate DEAD/STALLED verdict. Sources with their own threshold (e.g.
    worktree file mtimes) override the default ``stall_minutes``.
    """
    if probe is None:
        return False
    return probe.is_fresh(stall_minutes)


def _real_activity_is_inconclusive(probe: RealActivityProbe | None) -> bool:
    """Return True when the probe produced no timestamp evidence either way.

    This is the single enforcement point for "insufficient evidence to kill",
    shared by both classify_worker_health's Signal 3 and
    ``_log_is_stalled_at_shim`` (issue #307). It covers both inconclusive
    shapes by keying off ``probe.latest_timestamp`` (computed once in
    ``RealActivityProbe.__post_init__``):

    1. every source errored (e.g. sessions.db schema drift, no per-PID log)
    2. every source returned no error but also no timestamp match yet (e.g.
       a young devin-shell session within the launch-stall grace period
       whose sessions.db row hasn't landed)

    Neither shape is evidence the worker is actually dead/stalled, so callers
    should defer the verdict rather than fail open to DEAD/STALLED.
    """
    if probe is None or not probe.sources:
        return False
    return probe.latest_timestamp is None


def _next_inconclusive_probe_deferred_count(
    view: WorkerView,
    probe: RealActivityProbe | None,
    health: WorkerHealth,
) -> int:
    """Compute the next sidecar value for Signal-1's inconclusive-probe deferral counter.

    The counter advances only when a dead worker is classified as HEALTHY because
    the real-activity probe was inconclusive and the escalation cap had not yet
    been reached. It is reset to 0 whenever the worker is alive, the probe becomes
    conclusive, or the cap trips and the worker is reaped as DEAD.
    """
    if view.is_alive() or health is WorkerHealth.DEAD:
        return 0
    if _real_activity_is_inconclusive(probe) and health is WorkerHealth.HEALTHY:
        return view.inconclusive_probe_deferred_count + 1
    return 0


def classify_worker_health(
    view: WorkerView,
    config: OrchestratorConfig,
    now: datetime,
    real_activity_probe: RealActivityProbe | None = None,
) -> WorkerHealth:
    """Classify a worker's health based on liveness, staleness, and terminal markers.

    This is a pure function that takes a pre-fetched WorkerView and config and returns
    a WorkerHealth enum. It performs no I/O beyond what WorkerView.log_stat() already
    captured this pass, and has no side effects.

    Signal → verdict, first-to-fire-wins order:
    1. liveness → DEAD (unless a fresh real-session activity signal vetoes it,
       or an inconclusive probe defers up to ``max_inconclusive_probe_deferrals``)
    2. terminal marker → DEAD (unconditional; overrides a fresh probe)
    3. progress staleness → STALLED (unless a fresh or inconclusive probe vetoes it)
    4. wall-clock deadline → SLOW (or RUNAWAY if wall_clock_kill=True)
    5. loop/no-progress (Claude Code only) → SLOW (or RUNAWAY if loop_kill=True, capped at SLOW for Devin)
    6. cost/token budget (Claude Code only) → SLOW (or RUNAWAY if cost_budget_action="kill")
    7. (none of the above) → HEALTHY

    Signals 1 and 3 are corroborated against ``real_activity_probe`` (issues #280,
    #301, #307, #338). If the tracked process is gone or the sidecar log is stale,
    but any real-session activity source (sessions.db, per-PID Devin log, or Claude
    Code events.jsonl) is fresh, the worker is not classified as DEAD/STALLED this
    pass. If the probe was consulted but every source errored, it is treated as
    inconclusive and the verdict is deferred rather than failing open to STALLED.
    Signal 1 uses the same single enforcement point as Signal 3 but also applies an
    escalation cap so a genuinely-dead worker with a permanently-broken probe is
    still reaped after ``max_inconclusive_probe_deferrals`` deferred passes (issue
    #338). Signal 2 (terminal error markers) bypasses corroboration and still
    returns DEAD immediately.

    Args:
        view: WorkerView with pre-fetched worker state (pid, process_start_time, log_path, ...)
        config: OrchestratorConfig containing watchdog settings
        now: Current datetime for staleness calculation
        real_activity_probe: Optional pre-fetched real-session activity probe.
            When fresh, it overrides a stale sidecar log mtime or a missing process.

    Returns:
        WorkerHealth enum member indicating the worker's health status
    """
    from datetime import UTC, timedelta

    started_at = _parse_started_at(view.started_at)

    # Signal 1: liveness
    if not view.is_alive():
        # Issue #307: a process that just exited normally (e.g., after publishing a PR)
        # can still have a fresh real-session activity signal. Defer the DEAD verdict
        # for one pass instead of reaping it as a stall.
        if _real_activity_is_fresh(real_activity_probe, config.watchdog.stall_minutes):
            pass
        elif _real_activity_is_inconclusive(real_activity_probe):
            # Issue #338: an inconclusive probe is not evidence the worker is dead.
            # Defer the DEAD verdict, but only up to the configured cap so a
            # genuinely-dead worker with a permanently-broken probe is still reaped.
            if (
                view.inconclusive_probe_deferred_count
                < config.watchdog.max_inconclusive_probe_deferrals
            ):
                pass
            else:
                return WorkerHealth.DEAD
        else:
            return WorkerHealth.DEAD

    # Signal 2: terminal marker
    log_path = Path(view.log_path)
    terminal_error_markers = config.watchdog.terminal_error_markers

    # Check for terminal error markers in the log
    has_terminal_error = False
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        lines = log_text.splitlines()
        if lines:
            last_log_line = lines[-1].strip()
            for pattern in terminal_error_markers:
                if pattern in last_log_line:
                    has_terminal_error = True
                    break
    except OSError:
        pass

    if has_terminal_error:
        return WorkerHealth.DEAD

    # Signal 3: progress staleness
    log_stat = view.log_stat()
    if log_stat is not None:
        log_mtime = datetime.fromtimestamp(log_stat.st_mtime, tz=UTC)
        age = now - log_mtime
        is_stalled_by_mtime = age > timedelta(minutes=config.watchdog.stall_minutes)

        if is_stalled_by_mtime:
            # Corroborate against real-session activity before killing (issues #280, #307)
            if _real_activity_is_fresh(real_activity_probe, config.watchdog.stall_minutes):
                # Sidecar log is frozen but the real session is still moving
                pass
            elif _real_activity_is_inconclusive(real_activity_probe):
                # Probe was consulted but produced no timestamp evidence either
                # way (all sources errored, or all sources are error-free but
                # no-match-yet). That is insufficient evidence to kill; defer
                # rather than fail open to STALLED.
                pass
            else:
                return WorkerHealth.STALLED

    # Signal 4: wall-clock deadline (both adapters)
    if started_at is not None:
        wall_clock_age = now - started_at
        if wall_clock_age > timedelta(minutes=config.watchdog.wall_clock_minutes):
            return WorkerHealth.RUNAWAY if config.watchdog.wall_clock_kill else WorkerHealth.SLOW

    # Signal 5: loop/no-progress detection (Claude Code only)
    if view.adapter_kind == "claude-code":
        # Check for events.jsonl file (sibling to log_path)
        events_path = _events_path_from_log(log_path)
        if events_path.exists():
            last_tool_call_at = _tail_last_tool_call_timestamp(events_path)
            log_stat = view.log_stat()
            if log_stat is not None:
                log_mtime = datetime.fromtimestamp(log_stat.st_mtime, tz=UTC)
                log_still_advancing = (now - log_mtime) < timedelta(
                    minutes=config.watchdog.stall_minutes
                )

                # Calculate time since last tool call (or since start if never seen)
                if last_tool_call_at is not None or started_at is not None:
                    no_new_tool_call_for = now - (last_tool_call_at or started_at)

                    # Trip if log is advancing but no tool calls for 2 * stall_minutes
                    if log_still_advancing and no_new_tool_call_for > timedelta(
                        minutes=config.watchdog.loop_stall_multiplier
                        * config.watchdog.stall_minutes
                    ):
                        return (
                            WorkerHealth.RUNAWAY
                            if config.watchdog.loop_kill
                            else WorkerHealth.SLOW
                        )
    elif view.adapter_kind == "devin":
        # Devin has no structured event stream - this tripwire caps at SLOW
        # regardless of config, to avoid killing chatty-but-healthy patterns
        # (this is the single-point-of-enforcement for the Devin hard cap)
        pass

    # Signal 6: cost/token budget tripwire (issue #163)
    # Only applies to Claude Code workers (has events.jsonl file)
    # Devin sessions have no structured usage stream, so absence is never unhealthy
    if view.adapter_kind == "claude-code":
        events_path = _events_path_from_log(Path(view.log_path))
        usage = parse_cumulative_usage(events_path)

        if usage is not None:
            # Check cost budget
            cost_budget = config.watchdog.cost_budget_usd
            if cost_budget is not None and cost_budget > 0:
                if usage.cost_usd is not None and usage.cost_usd > cost_budget:
                    # Budget exceeded - return SLOW (warn) or RUNAWAY (kill)
                    if config.watchdog.cost_budget_action == "kill":
                        return WorkerHealth.RUNAWAY
                    else:
                        return WorkerHealth.SLOW

            # Check token budget
            token_budget = config.watchdog.token_budget
            if token_budget is not None and token_budget > 0:
                if usage.tokens is not None and usage.tokens > token_budget:
                    # Budget exceeded - return SLOW (warn) or RUNAWAY (kill)
                    if config.watchdog.cost_budget_action == "kill":
                        return WorkerHealth.RUNAWAY
                    else:
                        return WorkerHealth.SLOW

    # Signal 7: (none of the above)
    return WorkerHealth.HEALTHY


def _from_session_record(record: SessionRecord, repo_key: str) -> WorkerView:
    """Convert a SessionRecord to a WorkerView."""
    return WorkerView(
        adapter_kind="devin",
        issue_number=record.issue_number,
        repo_key=repo_key,
        pid=record.pid,
        started_at=record.started_at,
        process_start_time=record.process_start_time,
        log_path=record.log_path,
        worktree_path=record.worktree_path,
        branch=record.branch,
        error=record.error,
        failure_kind=record.failure_kind,
        reclaimed=record.reclaimed,
        last_activity_at=record.last_activity_at,
        log_bytes=record.log_bytes,
        rate_limit_defer_until=record.rate_limit_defer_until,
        inconclusive_probe_deferred_count=record.inconclusive_probe_deferred_count,
        session_id=record.session_id,
    )


def _from_claude_record(record: ClaudeWorkerRecord, repo_key: str) -> WorkerView:
    """Convert a ClaudeWorkerRecord to a WorkerView.

    The record's own ``adapter_kind`` is honored so ``issue-<n>.api.json``
    sidecars (written by the api adapter, which delegates to
    ``launch_claude_worker`` with ``adapter_kind="api"``) surface as
    ``adapter_kind == "api"`` rather than being mis-tagged ``claude-code``.
    Plain claude-code records carry ``adapter_kind == "claude-code"``.
    """
    return WorkerView(
        adapter_kind=record.adapter_kind,
        issue_number=record.issue_number,
        repo_key=repo_key,
        pid=record.pid,
        started_at=record.started_at,
        process_start_time=record.process_start_time,
        log_path=record.log_path,
        worktree_path=record.worktree_path,
        branch=record.branch,
        error=record.error,
        failure_kind=record.failure_kind,
        reclaimed=record.reclaimed,
        last_activity_at=record.last_activity_at,
        log_bytes=record.log_bytes,
        rate_limit_defer_until=record.rate_limit_defer_until,
        inconclusive_probe_deferred_count=record.inconclusive_probe_deferred_count,
        session_id=record.session_id,
    )


def iter_workers(sessions_dir: Path, *, repo_key: str = "") -> list[WorkerView]:
    """Read every devin-shell + claude-code sidecar in sessions_dir and return
    a unified, adapter-tagged list of WorkerView.

    Malformed sidecars are skipped (matches the existing read_session_records/
    read_worker_records contract) — never raises.

    Args:
        sessions_dir: Directory containing session sidecar files
        repo_key: Cross-repo disambiguation key (empty string for single-repo)

    Returns:
        List of WorkerView objects, one per valid sidecar file
    """
    workers: list[WorkerView] = []

    # Read devin-shell sidecars
    for record in read_session_records(sessions_dir):
        workers.append(_from_session_record(record, repo_key))

    # Read claude-code sidecars
    for record in read_worker_records(sessions_dir):
        workers.append(_from_claude_record(record, repo_key))

    # Read api-worker sidecars (issue-<n>.api.json). The api adapter delegates
    # to launch_claude_worker with adapter_kind="api", so the records are
    # ClaudeWorkerRecord instances surfaced via the #476 record reader.
    for record in read_worker_records(sessions_dir, adapter_kind="api"):
        workers.append(_from_claude_record(record, repo_key))

    return workers


def _alive_review_worker_issue_numbers(sessions_dir: Path) -> set[int]:
    """Return the set of PR/issue numbers whose reviewer sidecar is still alive.

    This is the single source of truth for liveness-exempt reap decisions
    across workflow.py's review-dispatch sweeps and reconcile.py's drift
    fixers. A PR whose reviewer process has not exited is deferred until a
    later pass, so its isolated review checkout is not torn down mid-session.
    """
    return {w.issue_number for w in iter_workers(sessions_dir) if w.is_alive()}


def _log_activity_advanced(
    log_stat_result: stat_result | None,
    previous_activity_at: str | None,
) -> bool:
    """Return True if the log has advanced since the previous stored stat.

    A log advances when the log file is accessible and its mtime is newer than
    the previous stored mtime. The first successful stat after launch
    (previous_activity_at is None) is treated as the baseline, not an advance.
    """
    if log_stat_result is None:
        return False
    if previous_activity_at is None:
        return False
    previous_mtime = _iso_to_timestamp(previous_activity_at)
    if previous_mtime is None:
        return False
    return log_stat_result.st_mtime > previous_mtime


def _iso_to_timestamp(value: str) -> float | None:
    """Parse an ISO timestamp (Z or +HH:MM) into a Unix timestamp."""
    from datetime import UTC

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    except (ValueError, TypeError):
        return None


def update_worker_log_stat(
    sessions_dir: Path,
    worker: WorkerView,
    *,
    rate_limit_defer_until: str | None = None,
    inconclusive_probe_deferred_count: int | None = None,
) -> None:
    """Update last_activity_at and log_bytes fields on a worker's sidecar.

    This reads the current sidecar, updates the log stat fields from a fresh
    stat() of the log file, and writes back atomically. This is called during
    passes over live workers to keep progress signals fresh.

    If the log has advanced (new mtime or larger size) and no new
    rate_limit_defer_until is being set, any previously stored defer deadline
    is cleared because the worker has resumed activity.

    Args:
        sessions_dir: Directory containing session sidecar files
        worker: WorkerView to update (must have valid log_path)
        rate_limit_defer_until: Optional new defer deadline to persist on the
            sidecar. When provided, the defer field is set and not cleared.
        inconclusive_probe_deferred_count: Optional new Signal-1 deferral counter
            to persist on the sidecar (issue #338).
    """
    import json

    if worker.adapter_kind == "devin":
        sidecar_path = devin_sidecar_path(sessions_dir, worker.issue_number)
    elif worker.adapter_kind == "claude-code":
        sidecar_path = claude_sidecar_path(sessions_dir, worker.issue_number)
    elif worker.adapter_kind == "api":
        sidecar_path = claude_sidecar_path(sessions_dir, worker.issue_number, "api")
    else:
        # Unknown adapter kind - nothing to update
        return

    if not sidecar_path.exists():
        return

    try:
        with sidecar_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, JSONDecodeError):
        return

    if not isinstance(payload, dict):
        return

    # Preserve the previously stored values so we can detect new log activity.
    previous_activity_at = payload.get("last_activity_at")

    # Stat the log file
    log_stat_result = worker.log_stat()
    if log_stat_result is None:
        # Log file doesn't exist or is inaccessible - clear the fields
        payload["last_activity_at"] = None
        payload["log_bytes"] = None
    else:
        # Update with fresh stat data
        payload["last_activity_at"] = datetime.fromtimestamp(
            log_stat_result.st_mtime, tz=timezone.utc
        ).isoformat()
        payload["log_bytes"] = log_stat_result.st_size

    if rate_limit_defer_until is not None:
        # Caller is explicitly setting a new defer deadline.
        payload["rate_limit_defer_until"] = rate_limit_defer_until
    else:
        # If the log has advanced, the worker is no longer stalled; clear the
        # defer deadline. Otherwise keep it level-triggered for the next pass.
        activity_advanced = _log_activity_advanced(log_stat_result, previous_activity_at)
        if activity_advanced:
            payload["rate_limit_defer_until"] = None

    if inconclusive_probe_deferred_count is not None:
        payload["inconclusive_probe_deferred_count"] = inconclusive_probe_deferred_count

    # Write back atomically using the adapter-specific helper
    if worker.adapter_kind == "devin":
        from .devin_shell import _write_json

        _write_json(sidecar_path, payload)
    elif worker.adapter_kind in ("claude-code", "api"):
        # api sidecars share the claude-code atomic-write helper (same on-disk
        # record shape, just a different filename suffix).
        from .claude_code import _write_json_atomic

        _write_json_atomic(sidecar_path, payload)


def real_activity_probe_for(
    view: WorkerView, config: OrchestratorConfig, now: datetime
) -> RealActivityProbe:
    """Build a real-session activity probe for ``view``.

    This is the convenience wrapper that the watchdog callers use; it delegates
    to ``post_mortem.real_activity_for_worker`` with the fields ``view`` already
    carries (worktree_path, started_at, pid, log_path) so the rest of the
    codebase does not repeat that plumbing.
    """
    return real_activity_for_worker(
        config.post_mortem,
        view.worktree_path,
        view.started_at,
        view.pid,
        now,
        view.log_path,
        config.watchdog,
    )


__all__ = [
    "WorkerHealth",
    "UsageSnapshot",
    "WorkerView",
    "classify_worker_health",
    "iter_workers",
    "parse_cumulative_usage",
    "real_activity_probe_for",
    "update_worker_log_stat",
    "_log_is_stalled_at_shim",
]
