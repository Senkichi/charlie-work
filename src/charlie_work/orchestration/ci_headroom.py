"""CI-capacity headroom for the dispatch concurrency governor (issue #1770).

Step 1 of 2 (see the issue): this module is the pure-ish computation the
concurrency governor's future fourth clamp term will call. It is not wired
into ``_apply_concurrency_governor`` yet -- that wiring, and the
``dispatch.ci_capacity_headroom_ratio`` config knob it reads its ratio from,
land separately.

Why events.db and not a fresh measurement
------------------------------------------
The design doc this implements (``runner-starvation.md``, sections 3/4)
originally sketched ``ci_headroom_available(repo, config, gh)`` re-querying
GitHub Actions runs directly, mirroring ``_detect_ci_run_never_created``'s
``self.gh.workflow_runs_for_head``-style call a few lines above the governor.
That would add a GitHub API call to every dispatch pass. It is unnecessary:
``ci_fleet.runner_allocation_pass`` already measures exactly this data --
registered-runner ceiling and live queued+in_progress demand, per repo --
once per allocation pass, and unconditionally logs it as a ``"runner_allocation"``
event (payload = ``ci_fleet.runner_allocation.plan_summary(plan)``) to this
fleet's ``events.db``, via the ``ci_fleet.observability`` event sink
``charlie_work.instrumentation`` installs at import time. Reading that event
back costs zero new GitHub calls and zero new hardcoded thresholds -- it
reuses state the fleet already collects, per CLAUDE.md's "Runner slots move by
start/park" invariant (this repo goes through ``ci_fleet.charlie_work_adapter``
for fleet data; this module does not import any ``ci_fleet`` name at all,
since the only thing it reads is this repo's own event store).

Caller contract
----------------
``fleet_state_path`` must be the **fleet-level** ``state.json`` path (i.e.
``layout.state_file_path(fleet_paths.fleet_dir(override=...))``), never a
per-repo one -- ``runner_allocation_pass`` is invoked once per host against
the fleet-wide event store, not per repo (see the
``events-db-two-locations-and-kind-prefixes`` trap: multiple ``events.db``
files exist, disjoint by scope). Passing a per-repo state path here will
silently and permanently return ``None`` (no matching event), which fails
open -- safe, but worth getting right so the clamp actually engages.

``repo`` must be the ``"owner/name"`` slug ``ci_fleet`` derives from each
runner's ``.runner`` file (``repo_slug_from_github_url``), not a bare
directory name -- that is the key every ``targets`` entry in the event
payload is keyed by.

Fail-open discipline
---------------------
``ci_headroom_available`` returns ``None`` -- "do not clamp" -- whenever the
data cannot be trusted, and always records why via a ``ci_headroom_unavailable``
event so "0 dispatched" stays diagnosable from events.db instead of reading as
an idle fleet (same discipline as ``dispatch_backpressure``). The cases:

* No ``runner_allocation`` event has ever been recorded (``"no_data"``).
* The freshest one is older than ``max_data_age_minutes`` (``"stale"``) --
  the allocation pass writes this event on every cycle, so an old one means
  the pass itself has stopped running, not that nothing changed.
* The event's payload cannot be parsed as the expected shape (``"malformed"``).
* The repo has no entry in the latest plan's ``targets`` (``"unconfigured"``)
  -- matches ``ci_fleet.demand.Measured``'s sum-type discipline: an
  unconfigured repo is unmeasured, never zero-capacity, so it must never be
  treated as "no headroom" (which would wrongly hard-stop dispatch for every
  repo this feature has not been turned on for).
* The repo's demand reading is ``pinned`` (``"pinned"``) -- ``ci_fleet``
  marks a repo pinned when its busy-runner listing could not be fetched that
  pass, and reports ``demand=0`` as a bookkeeping placeholder in that case
  (``ci_fleet.planner._slots_wanted`` returns 0 for anything that is not a
  ``Measured`` reading). Trusting that zero would silently read "wide open"
  for a repo whose true demand is simply unknown -- the same wrong-default-to-0
  shape the ``"unconfigured"`` case guards against, just on the demand side
  instead of the capacity side.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from charlie_work.instrumentation import log_event, query_events

logger = logging.getLogger(__name__)

# ci_fleet.runner_allocation_pass logs this event, unconditionally, on every
# completed allocation pass (i.e. whether or not anything started/parked) --
# see its module docstring and the `log_event(state_path, "runner_allocation",
# summary)` call after `plan_summary(plan)`. Distinct from the
# edge-triggered `runner_capacity_starved`/`_recovered` pair (#799), which
# only fires when a repo's starved condition flips, and from
# `fleet_dispatch.py`'s own locally-built "runner_allocation" digest dict
# (never written to events.db, and only populated when something moved) --
# this is ci_fleet's own write, carrying every repo's numbers every pass.
ALLOCATION_EVENT_KIND = "runner_allocation"

UNAVAILABLE_EVENT_KIND = "ci_headroom_unavailable"

# ci_fleet's fleet pass runs at `fleet.full_pass_interval_seconds`
# (config.py, default 300s/5min). Six missed passes is long enough to
# absorb an ordinary supervisor respawn without a false "stale" verdict, but
# short enough to catch a genuinely dead allocation pass well inside an
# hour. A parameter default rather than a config read, so this function
# stays a pure computation over its arguments -- the governor wiring is free
# to thread a config value through instead once one exists.
DEFAULT_MAX_DATA_AGE_MINUTES = 30


def ci_headroom_available(
    repo: str,
    *,
    headroom_ratio: float,
    fleet_state_path: Path,
    max_data_age_minutes: int = DEFAULT_MAX_DATA_AGE_MINUTES,
    now: datetime | None = None,
) -> int | None:
    """Remaining CI dispatch headroom for ``repo``, or ``None`` if unknown.

    Computes ``max(0, floor(capacity * headroom_ratio) - demand)`` from the
    freshest ``runner_allocation`` event's ``targets`` entry for ``repo``,
    where ``capacity`` is its registered-runner ceiling and ``demand`` is its
    live queued+in_progress GitHub Actions run count (both already measured
    by ``ci_fleet``'s allocation pass -- see the module docstring). The
    result is the number of additional in-flight PRs this repo's CI capacity
    can still absorb before it starts queuing work faster than runners can
    drain it.

    Returns ``None`` -- meaning "the caller must not clamp on this" -- when
    the data is missing, stale, malformed, or the repo is unconfigured or its
    demand reading is pinned/unmeasurable. Every ``None`` path logs a
    ``ci_headroom_unavailable`` event carrying the reason (see the module
    docstring's "Fail-open discipline" section), except when a database
    error itself prevents that write (best-effort, like every other
    ``log_event`` call).

    Never raises: a malformed or unreadable event is a data problem, not a
    caller bug, so it is folded into the ``None``/logged-reason path along
    with every other "cannot trust this" case.
    """
    resolved_now = now if now is not None else datetime.now(UTC)

    events = query_events(fleet_state_path, kind=ALLOCATION_EVENT_KIND, limit=1)
    if not events:
        _log_unavailable(fleet_state_path, repo, "no_data", "no runner_allocation event recorded")
        return None
    event = events[0]

    event_time = _parse_event_ts(event.get("ts"))
    if event_time is None:
        _log_unavailable(
            fleet_state_path,
            repo,
            "malformed",
            f"unreadable timestamp on runner_allocation event: {event.get('ts')!r}",
        )
        return None

    age = resolved_now - event_time
    max_age = timedelta(minutes=max_data_age_minutes)
    if age > max_age:
        _log_unavailable(
            fleet_state_path,
            repo,
            "stale",
            f"freshest runner_allocation event is {age} old (max {max_age})",
        )
        return None

    payload = event.get("payload")
    if not isinstance(payload, dict):
        _log_unavailable(
            fleet_state_path, repo, "malformed", "runner_allocation event payload is not a dict"
        )
        return None

    target = _target_for_repo(payload, repo)
    if target is None:
        _log_unavailable(
            fleet_state_path,
            repo,
            "unconfigured",
            f"{repo} has no entry in the latest runner_allocation plan's targets",
        )
        return None

    if target.get("pinned"):
        _log_unavailable(
            fleet_state_path,
            repo,
            "pinned",
            f"{repo}'s demand is pinned (unmeasurable) this pass; its reported "
            "demand is a bookkeeping placeholder, not a real reading",
        )
        return None

    capacity = target.get("capacity")
    demand = target.get("demand")
    if not _is_plain_int(capacity) or not _is_plain_int(demand):
        _log_unavailable(
            fleet_state_path,
            repo,
            "malformed",
            f"{repo}'s allocation target has non-integer capacity/demand: {target!r}",
        )
        return None

    return max(0, math.floor(capacity * headroom_ratio) - demand)


def _is_plain_int(value: Any) -> bool:
    """``True`` for a JSON-decoded integer, ``False`` for a bool or anything else.

    ``json.loads`` never produces Python ``bool`` for a numeric field here,
    but ``isinstance(True, int)`` is ``True`` in Python, so a bare
    ``isinstance(value, int)`` check would silently accept one if a future
    payload shape regressed. Explicit rather than defensive-for-its-own-sake:
    ``capacity``/``demand`` feed directly into arithmetic below.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _target_for_repo(payload: dict[str, Any], repo: str) -> dict[str, Any] | None:
    """The ``targets`` entry for ``repo``, or ``None`` if absent/malformed."""
    targets = payload.get("targets")
    if not isinstance(targets, list):
        return None
    for entry in targets:
        if isinstance(entry, dict) and entry.get("repo") == repo:
            return entry
    return None


def _parse_event_ts(raw: Any) -> datetime | None:
    """Parse an events.db ``ts`` string (``instrumentation._now_iso``'s format).

    Mirrors ``capacity_starvation_escalation``'s timestamp parsing: UTC ISO-8601
    with a ``Z`` suffix, which ``datetime.fromisoformat`` accepts directly on
    this project's Python floor (3.11+).
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _log_unavailable(fleet_state_path: Path, repo: str, reason: str, detail: str) -> None:
    logger.info("ci_headroom_available(%s): unavailable (%s): %s", repo, reason, detail)
    log_event(
        fleet_state_path,
        UNAVAILABLE_EVENT_KIND,
        {"reason": reason, "detail": detail},
        repo=repo,
        level="warning",
    )
