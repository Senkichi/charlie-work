"""Fleet-scoped provider throttle view and staggered resume (issue #1993).

A provider rate limit is per *account*, but ``state["throttled_until"]`` is
recorded per repo, by whichever repo happened to lose a worker to it. Every
other repo kept launching into the same limit (observed 2026-09-29: six repos,
three waves, each death spending a redispatch or rework attempt).

Point 1 -- fleet-wide view. Rather than adding a second writer next to each
``set_throttled_until`` call (four sites, one of them operator-facing, all of
which would have to remember it), the *reader* is made fleet-wide: the gate
takes the max ``throttled_until`` across this repo's and every registered
repo's ``state.json`` for the launching adapter. Any writer -- the reaper, the
reconcile fix, or an operator -- is therefore covered by construction, and a
deadline is never shortened because nothing is copied. Only ``rate_limited`` /
``quota_exhausted`` windows scope fleet-wide; ``provider_auth`` is a dead
credential, not an account budget.

Point 3 -- staggered resume. When the fleet window has expired, the whole
fleet used to reopen in the same minute and re-hit the limit together. Now
the first launch after expiry is a *probe*: a marker in the fleet dir stamps
its launch time, further launches for that adapter are deferred until the
probe has lived ``RESUME_SURVIVAL_SECONDS``, and each pass admits at most one
launch while no probe has survived. A probe that dies of the limit re-arms a
newer window (its ``throttled_until`` is later than the probe's stamp), which
puts the adapter straight back into probe mode.

That window can lag the death by many minutes (the reaper only records it once
it classifies the dead worker's log), so "no newer window" is *not* survival.
The stamp therefore records the probe's pid(s); the fleet reopens only while a
probe process is provably still alive after the survival period, and a probe
that is dead with no window recorded is re-probed (``admit_one``), never
treated as having survived. Survival is persisted (``survived_at``) so a probe
that later finishes normally does not put the fleet back into probe mode.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from . import host as _host
from .layout import fleet_dir
from .state import advisory_file_lock, save_state
from .command_result import CommandResult

logger = logging.getLogger(__name__)

#: ``throttle_reason`` values that are an account-wide budget (fleet-scoped).
FLEET_SCOPED_THROTTLE_REASONS: frozenset[str] = frozenset({"rate_limited", "quota_exhausted"})

#: How long the resume probe must stay alive before the adapter fully reopens.
RESUME_SURVIVAL_SECONDS = 300

RESUME_PROBE_FILENAME = "provider_resume_probe.json"

#: ``throttle_adapter_kind`` recorded as ``None`` means claude-code-shaped
#: (see ``state.set_throttled_until``).
_DEFAULT_ADAPTER_KIND = "claude-code"


ResumeAction = Literal["open", "defer_throttled", "defer_probe", "admit_one"]


@dataclass(frozen=True)
class ResumeDecision:
    """Outcome of the fleet gate for one adapter on one launch pass."""

    action: ResumeAction
    throttled_until: datetime | None = None
    #: True when ``open`` was granted because a probe proved alive past the
    #: survival period and that fact is not yet persisted.
    probe_survived: bool = False
    #: The adapter the gate was evaluated for (the launch's selected role-chain
    #: entry); the probe stamp reuses it so gate and stamp can never disagree.
    adapter_kind: str | None = None

    @property
    def deferred(self) -> bool:
        return self.action in ("defer_throttled", "defer_probe")

    @property
    def deferred_reason(self) -> str | None:
        """Stable ``deferred_reason`` for a deferred decision, else ``None``."""
        if self.action == "defer_throttled":
            return "provider_throttled_fleet"
        if self.action == "defer_probe":
            return "provider_resume_staggered"
        return None

    def deferral_data(self) -> dict[str, Any]:
        """``deferred_reason`` / ``throttled_until`` fields for a deferred result."""
        return {
            "deferred_reason": self.deferred_reason,
            "throttled_until": self.throttled_until.isoformat() if self.throttled_until else None,
        }

    def deferred_result(self, what: str, data: dict[str, Any]) -> CommandResult:
        """The not-ok ``CommandResult`` a lane returns when this decision defers it."""
        return CommandResult(
            False,
            f"{what} deferred: fleet provider throttle ({self.action})",
            {**data, **self.deferral_data()},
        )

    def cap_limit(self, limit: int) -> int:
        """A probe pass admits at most one launch; otherwise ``limit`` unchanged."""
        return min(limit, 1) if self.action == "admit_one" else limit


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def latest_fleet_throttle(
    state_files: Iterable[Path], adapter_kind: str, *, selection: Any = None
) -> datetime | None:
    """Return the latest fleet-scoped throttle deadline for ``adapter_kind``.

    Scans each ``state.json`` (missing/corrupt files are skipped -- additive
    evidence, never load-bearing) and returns the max ``throttled_until`` over
    windows whose reason is account-wide and whose recorded adapter matches.
    The deadline may already be in the past; callers decide what expiry means.

    When ``selection`` is the launch's role-chain selection, a window the
    fleet-wide quota ledger explains (:func:`role_selection.window_covered` --
    a ``throttle_harness``/``throttle_model`` stamp attributable to a
    restricted entry other than the selected one, issue #2279) is a per-entry
    hold, not an account hold, and does not count.
    """
    from . import role_selection

    latest: datetime | None = None
    for path in state_files:
        data = _read_json(path)
        reason = data.get("throttle_reason")
        if reason not in FLEET_SCOPED_THROTTLE_REASONS:
            continue
        window_adapter = data.get("throttle_adapter_kind") or _DEFAULT_ADAPTER_KIND
        if window_adapter != adapter_kind:
            continue
        until = _parse_iso(data.get("throttled_until"))
        if until is None:
            continue
        if selection is not None and role_selection.window_covered(
            data.get("throttled_until"),
            selection,
            reason=reason,
            adapter_kind=window_adapter,
            harness=data.get("throttle_harness"),
            model=data.get("throttle_model"),
        ):
            continue
        if latest is None or until > latest:
            latest = until
    return latest


def resume_probe_path(fleet_dir_override: str | None = None) -> Path:
    return fleet_dir(override=fleet_dir_override) / RESUME_PROBE_FILENAME


def decide_launch(
    state_files: Iterable[Path],
    adapter_kind: str,
    *,
    selection: Any = None,
    fleet_dir_override: str | None = None,
    now: datetime | None = None,
) -> ResumeDecision:
    """Decide whether a launch for ``adapter_kind`` may proceed fleet-wide."""
    resolved_now = now if now is not None else datetime.now(UTC)
    latest = latest_fleet_throttle(state_files, adapter_kind, selection=selection)
    if latest is None:
        return ResumeDecision("open")
    if latest > resolved_now:
        return ResumeDecision("defer_throttled", throttled_until=latest)
    probe = _read_json(resume_probe_path(fleet_dir_override)).get(adapter_kind)
    if not isinstance(probe, dict):
        return ResumeDecision("admit_one", throttled_until=latest)
    probe_at = _parse_iso(probe.get("launched_at"))
    if probe_at is None or probe_at < latest:
        return ResumeDecision("admit_one", throttled_until=latest)
    survived_at = _parse_iso(probe.get("survived_at"))
    if survived_at is not None and survived_at >= latest:
        return ResumeDecision("open", throttled_until=latest)
    if not _any_probe_alive(probe.get("probes")):
        # Probe died (or was never provably running) and no window says why:
        # the fleet must not reopen on the strength of silence.
        return ResumeDecision("admit_one", throttled_until=latest)
    if resolved_now - probe_at < timedelta(seconds=RESUME_SURVIVAL_SECONDS):
        return ResumeDecision("defer_probe", throttled_until=latest)
    return ResumeDecision("open", throttled_until=latest, probe_survived=True)


def _any_probe_alive(probes: Any) -> bool:
    if not isinstance(probes, list):
        return False
    for entry in probes:
        if not isinstance(entry, dict) or not isinstance(entry.get("pid"), int):
            continue
        start = entry.get("process_start_time")
        if _host.current().probe.is_alive(
            entry["pid"], start if isinstance(start, (int, float)) else None
        ):
            return True
    return False


def _write_probe_entry(
    adapter_kind: str, fleet_dir_override: str | None, entry: dict[str, Any]
) -> None:
    path = resume_probe_path(fleet_dir_override)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with advisory_file_lock(path):
            data = _read_json(path)
            data[adapter_kind] = entry
            # Fleet-dir probe sidecar, not the repo state file a WriteGate binds.
            # write-gate-exempt(issue=1993): module-level helper, no write_gate receiver.
            save_state(path, data)
    except (OSError, RuntimeError) as exc:
        # Best effort: a lost stamp only means the next pass admits another
        # probe, which is the pre-#1993 behaviour, not a new failure mode.
        logger.warning("could not record resume probe for %s: %s", adapter_kind, exc)


def note_probe_launch(
    adapter_kind: str,
    probes: Sequence[tuple[int, float | None]],
    *,
    fleet_dir_override: str | None = None,
    now: datetime | None = None,
) -> None:
    """Stamp a resume probe for ``adapter_kind`` with its ``(pid, start_time)`` list."""
    resolved_now = now if now is not None else datetime.now(UTC)
    _write_probe_entry(
        adapter_kind,
        fleet_dir_override,
        {
            "launched_at": _iso(resolved_now),
            "probes": [{"pid": pid, "process_start_time": start} for pid, start in probes],
        },
    )


def note_probe_survived(
    adapter_kind: str,
    *,
    fleet_dir_override: str | None = None,
    now: datetime | None = None,
) -> None:
    """Persist that the current probe outlived the survival period."""
    resolved_now = now if now is not None else datetime.now(UTC)
    entry = _read_json(resume_probe_path(fleet_dir_override)).get(adapter_kind)
    if isinstance(entry, dict):
        _write_probe_entry(
            adapter_kind, fleet_dir_override, {**entry, "survived_at": _iso(resolved_now)}
        )


def note_probe_from_results(app: Any, resume: ResumeDecision, results: Iterable[Any]) -> None:
    """Stamp the probe after a launch pass admitted under ``admit_one``.

    Only launches that succeeded and report a pid count: a failed launch is not
    a probe, and a probe without a pid cannot prove survival (the next pass
    simply admits another probe -- one launch, never a fleet reopening).
    """
    if resume.action != "admit_one":
        return
    probes = [(r.pid, r.process_start_time) for r in results if r.ok and r.pid is not None]
    if probes:
        note_probe_launch(
            resume.adapter_kind or worker_adapter_kind(app.config.worker.harness),
            probes,
            fleet_dir_override=app.fleet_dir_override,
        )


def worker_adapter_kind(harness: str) -> str:
    """Map a ``worker.harness`` name to its session ``adapter_kind``."""
    from .harnesses import HARNESS_REGISTRY

    cap = HARNESS_REGISTRY.get(harness)
    return cap.adapter_kind if cap is not None else harness


def decide_for_app(
    app: Any, *, selection: Any = None, now: datetime | None = None
) -> ResumeDecision:
    """Fleet gate for the worker adapter a launch will run on (dispatch and rework lanes).

    ``selection`` is the permit's ``role_selection``: with ``worker.fallbacks``
    configured the launch runs on the selected chain entry, which may not be the
    primary, so the gate (and the decision's probe stamp) key on *that* entry's
    adapter. Without a selection the configured primary is used.

    Scans this repo's own ``state.json`` plus every registered repo's.
    Reviewers are not gated here: they launch through their own harness and
    quota (``reviewer_quota``), so a Devin window never defers claude-code.
    """
    from .fleet_registry import registered_state_dirs
    from .layout import state_file_path

    override = app.fleet_dir_override
    state_files = [
        app.paths.state_file,
        *(state_file_path(d) for d in registered_state_dirs(override)),
    ]
    adapter_kind = None
    if selection is not None:
        from .role_selection import selection_adapter_kind

        adapter_kind = selection_adapter_kind(selection)
    if adapter_kind is None:
        adapter_kind = worker_adapter_kind(app.config.worker.harness)
    decision = replace(
        decide_launch(
            state_files,
            adapter_kind,
            selection=selection,
            fleet_dir_override=override,
            now=now,
        ),
        adapter_kind=adapter_kind,
    )
    if decision.probe_survived:
        note_probe_survived(adapter_kind, fleet_dir_override=override, now=now)
    return decision
