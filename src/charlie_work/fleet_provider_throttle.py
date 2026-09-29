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
puts the adapter straight back into probe mode -- no liveness tracking needed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .layout import fleet_dir
from .state import advisory_file_lock, save_state

logger = logging.getLogger(__name__)

#: ``throttle_reason`` values that are an account-wide budget (fleet-scoped).
FLEET_SCOPED_THROTTLE_REASONS: frozenset[str] = frozenset({"rate_limited", "quota_exhausted"})

#: How long the resume probe must stay alive before the adapter fully reopens.
RESUME_SURVIVAL_SECONDS = 300

RESUME_PROBE_FILENAME = "provider_resume_probe.json"

#: ``throttle_adapter_kind`` recorded as ``None`` means claude-code-shaped
#: (see ``state.set_throttled_until``).
_DEFAULT_ADAPTER_KIND = "claude-code"


@dataclass(frozen=True)
class ResumeDecision:
    """Outcome of the fleet gate for one adapter on one launch pass."""

    action: str  # "open" | "defer_throttled" | "defer_probe" | "admit_one"
    throttled_until: datetime | None = None

    @property
    def deferred(self) -> bool:
        return self.action in ("defer_throttled", "defer_probe")


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


def latest_fleet_throttle(state_files: Iterable[Path], adapter_kind: str) -> datetime | None:
    """Return the latest fleet-scoped throttle deadline for ``adapter_kind``.

    Scans each ``state.json`` (missing/corrupt files are skipped -- additive
    evidence, never load-bearing) and returns the max ``throttled_until`` over
    windows whose reason is account-wide and whose recorded adapter matches.
    The deadline may already be in the past; callers decide what expiry means.
    """
    latest: datetime | None = None
    for path in state_files:
        data = _read_json(path)
        if data.get("throttle_reason") not in FLEET_SCOPED_THROTTLE_REASONS:
            continue
        if (data.get("throttle_adapter_kind") or _DEFAULT_ADAPTER_KIND) != adapter_kind:
            continue
        until = _parse_iso(data.get("throttled_until"))
        if until is not None and (latest is None or until > latest):
            latest = until
    return latest


def resume_probe_path(fleet_dir_override: str | None = None) -> Path:
    return fleet_dir(override=fleet_dir_override) / RESUME_PROBE_FILENAME


def decide_launch(
    state_files: Iterable[Path],
    adapter_kind: str,
    *,
    fleet_dir_override: str | None = None,
    now: datetime | None = None,
) -> ResumeDecision:
    """Decide whether a launch for ``adapter_kind`` may proceed fleet-wide."""
    resolved_now = now if now is not None else datetime.now(UTC)
    latest = latest_fleet_throttle(state_files, adapter_kind)
    if latest is None:
        return ResumeDecision("open")
    if latest > resolved_now:
        return ResumeDecision("defer_throttled", throttled_until=latest)
    probe = _read_json(resume_probe_path(fleet_dir_override)).get(adapter_kind)
    probe_at = _parse_iso(probe.get("launched_at")) if isinstance(probe, dict) else None
    if probe_at is None or probe_at < latest:
        return ResumeDecision("admit_one", throttled_until=latest)
    if resolved_now - probe_at < timedelta(seconds=RESUME_SURVIVAL_SECONDS):
        return ResumeDecision("defer_probe", throttled_until=latest)
    return ResumeDecision("open", throttled_until=latest)


def note_probe_launch(
    adapter_kind: str,
    *,
    fleet_dir_override: str | None = None,
    now: datetime | None = None,
) -> None:
    """Stamp a resume probe launch for ``adapter_kind`` (atomic, fleet-locked)."""
    resolved_now = now if now is not None else datetime.now(UTC)
    path = resume_probe_path(fleet_dir_override)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with advisory_file_lock(path):
            data = _read_json(path)
            data[adapter_kind] = {"launched_at": _iso(resolved_now)}
            save_state(path, data)
    except (OSError, RuntimeError) as exc:
        # Best effort: a lost stamp only means the next pass admits another
        # probe, which is the pre-#1993 behaviour, not a new failure mode.
        logger.warning("could not record resume probe for %s: %s", adapter_kind, exc)


def worker_adapter_kind(harness: str) -> str:
    """Map a ``worker.harness`` name to its session ``adapter_kind``."""
    from .harnesses import HARNESS_REGISTRY

    cap = HARNESS_REGISTRY.get(harness)
    return cap.adapter_kind if cap is not None else harness


def decide_for_app(app: Any, *, now: datetime | None = None) -> ResumeDecision:
    """Fleet gate for ``app``'s worker adapter (dispatch and rework lanes).

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
    return decide_launch(
        state_files,
        worker_adapter_kind(app.config.worker.harness),
        fleet_dir_override=override,
        now=now,
    )
