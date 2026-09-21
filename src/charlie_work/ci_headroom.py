"""CI-capacity headroom for the dispatch concurrency governor (issue #1770).

This module is the pure-ish computation ``_apply_concurrency_governor``'s
fourth clamp term calls (``orchestration/reap_dispatch.py``, fresh-issue
dispatch only -- rework/recovery/loop callers leave ``ci_capacity_headroom_
ratio`` unread, same exemption ``max_open_agent_prs`` uses). The
``dispatch.ci_capacity_headroom_ratio`` config knob (``config.py``) supplies
the ratio.

What the result actually measures
----------------------------------
``capacity`` is a registered-*runner* ceiling and ``demand`` is a live
queued+in_progress *job* count (both from ``ci_fleet``'s allocation pass --
see below); the difference this module returns is therefore a
**runner-slot-utilization headroom**, not a count of PRs. One PR can fan out
to many CI jobs (a matrix workflow), and that fan-out factor is repo-specific
and not represented anywhere in this computation. Two consequences worth
knowing before choosing a ratio for a repo:

* A ratio of ``1.0`` reads as "PRs may add jobs until every runner is busy",
  which is a permanent dispatch stop for a repo whose runners are simply
  keeping up with legitimate load (``demand == capacity``, zero queue, is
  *healthy* utilization, not saturation) -- there is no queuing to relieve.
  Repos with a multi-job fan-out need a ratio meaningfully above ``1.0`` to
  leave room for a job's siblings, not just the job itself.
* A ratio that comfortably covers one repo's fan-out may be far too
  permissive for a repo with a wider matrix -- there is no single safe
  default across repos, which is why this ships at ``0.0`` (off) rather than
  with a suggested nonzero value.

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

``diagnostic_state_path``/``diagnostic_repo`` are separate from
``fleet_state_path``/``repo`` on purpose (issue #1770 review finding 8): the
``runner_allocation`` event this function *reads* only ever lives in the
fleet-wide store keyed by the ``owner/name`` slug (above), but the
``ci_headroom_unavailable`` diagnostic it *writes* is a per-dispatch-pass fact
about one repo, exactly like the ``dispatch_backpressure`` event the caller
writes right next to it. Left at their defaults (``None``), both diagnostic
writes fall back to ``fleet_state_path``/``repo`` -- the original, single-
store behavior every existing caller and test relies on. The caller in
``reap_dispatch.py`` passes its own per-repo ``state_file`` and the
directory-name ``repo`` spelling ``dispatch_backpressure`` already uses, so
both halves of one clamp decision land in the same database under the same
``repo`` key -- an operator grepping one repo's ``events.db`` for "why did
this pass dispatch 0" finds both events, instead of the fail-open half being
invisible in a different database under a different key spelling (the
``events-db-two-locations-and-kind-prefixes`` trap).

Fail-open discipline
---------------------
``ci_headroom_available`` returns ``None`` -- "do not clamp" -- whenever the
data cannot be trusted, and records why via a ``ci_headroom_unavailable``
event so "0 dispatched" stays diagnosable from events.db instead of reading as
an idle fleet (same discipline as ``dispatch_backpressure``). The write is
edge-triggered and rate-limited (issue #1770 review finding 2): it fires the
first time a repo goes unavailable, again immediately if the *reason*
changes, and otherwise at most once per ``max_data_age_minutes`` while the
same reason persists -- never once per dispatch pass unconditionally. A dead
allocation pass, a repo ci_fleet never lists, or a target stuck ``pinned``
each stay true for a long time, and this feature's own dispatch cadence is
far tighter than that -- an unconditional write would turn one stuck
condition into an unbounded, ever-growing warning stream in the very store
``charlie doctor``/digest counts use as a baseline, degrading the real signal
("a repeating burst here means that channel itself needs attention" -- the
comment ``instrumentation.py`` puts on this kind's level registration) into
noise before anyone could see a burst in it. The dedup check reads the
freshest prior ``ci_headroom_unavailable`` event back from
``diagnostic_state_path`` (or ``fleet_state_path`` when not given) --
events.db is the durable record for this, not a new in-memory or on-disk
counter, matching how every other cross-pass fact in this codebase is kept.
The cases:

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
    min_in_flight_demand: int = 0,
    diagnostic_state_path: Path | None = None,
    diagnostic_repo: str | None = None,
) -> int | None:
    """Remaining CI slot-utilization headroom for ``repo``, or ``None`` if unknown.

    Computes ``max(0, floor(capacity * headroom_ratio) - max(demand,
    min_in_flight_demand))`` from the freshest ``runner_allocation`` event's
    ``targets`` entry for ``repo``, where ``capacity`` is its
    registered-runner ceiling and ``demand`` is its live queued+in_progress
    GitHub Actions *job* count (both already measured by ``ci_fleet``'s
    allocation pass -- see the module docstring). This is runner-slot
    headroom, not a PR count (see the module docstring's "What the result
    actually measures" section) -- a repo whose workflows fan out to several
    jobs per PR can absorb fewer additional PRs than this number.

    ``min_in_flight_demand`` is the caller's own floor on demand: local work
    already committed to becoming CI load that the freshest
    ``runner_allocation`` reading cannot see yet (issue #1770 review finding
    3). ``demand`` is a snapshot from the *last* allocation pass; a worker
    dispatched since then has not opened a PR yet, so it contributes zero
    measured demand for several minutes even though it fully intends to.
    Without this floor, every pass between allocation refreshes re-grants the
    same full headroom against unchanged ``demand``, letting fresh dispatch
    compound far past the intended ceiling before the first PR's CI jobs
    finally show up and demand catches up (the failure mode the clamp exists
    to prevent). The caller passes its own live worker-session count plus
    already-open agent-PR count for this repo -- work it already knows is
    in flight regardless of what the last allocation pass measured. Left at
    the default ``0``, behavior is unchanged (``max(demand, 0) == demand``).

    ``diagnostic_state_path``/``diagnostic_repo`` default to
    ``fleet_state_path``/``repo`` (see the module docstring's "Caller
    contract" section) -- pass them to route the ``ci_headroom_unavailable``
    diagnostic to a different store/key than the ``runner_allocation`` event
    this function reads.

    Returns ``None`` -- meaning "the caller must not clamp on this" -- when
    the data is missing, stale, malformed, or the repo is unconfigured or its
    demand reading is pinned/unmeasurable. Every ``None`` path logs a
    ``ci_headroom_unavailable`` event carrying the reason, rate-limited (see
    the module docstring's "Fail-open discipline" section), except when a
    database error itself prevents that write (best-effort, like every other
    ``log_event`` call).

    Never raises: a malformed or unreadable event is a data problem, not a
    caller bug, so it is folded into the ``None``/logged-reason path along
    with every other "cannot trust this" case.
    """
    resolved_now = now if now is not None else datetime.now(UTC)
    diag_state_path = (
        diagnostic_state_path if diagnostic_state_path is not None else fleet_state_path
    )
    diag_repo = diagnostic_repo if diagnostic_repo is not None else repo

    def log_unavailable(reason: str, detail: str) -> None:
        _log_unavailable(
            diag_state_path,
            diag_repo,
            reason,
            detail,
            now=resolved_now,
            min_interval_minutes=max_data_age_minutes,
        )

    events = query_events(fleet_state_path, kind=ALLOCATION_EVENT_KIND, limit=1)
    if not events:
        log_unavailable("no_data", "no runner_allocation event recorded")
        return None
    event = events[0]

    event_time = _parse_event_ts(event.get("ts"))
    if event_time is None:
        log_unavailable(
            "malformed",
            f"unreadable timestamp on runner_allocation event: {event.get('ts')!r}",
        )
        return None

    age = resolved_now - event_time
    max_age = timedelta(minutes=max_data_age_minutes)
    if age > max_age:
        log_unavailable(
            "stale",
            f"freshest runner_allocation event is {age} old (max {max_age})",
        )
        return None

    payload = event.get("payload")
    if not isinstance(payload, dict):
        log_unavailable("malformed", "runner_allocation event payload is not a dict")
        return None

    target = _target_for_repo(payload, repo)
    if target is None:
        log_unavailable(
            "unconfigured",
            f"{repo} has no entry in the latest runner_allocation plan's targets",
        )
        return None

    if target.get("pinned"):
        log_unavailable(
            "pinned",
            f"{repo}'s demand is pinned (unmeasurable) this pass; its reported "
            "demand is a bookkeeping placeholder, not a real reading",
        )
        return None

    capacity = target.get("capacity")
    demand = target.get("demand")
    if not _is_plain_int(capacity) or not _is_plain_int(demand):
        log_unavailable(
            "malformed",
            f"{repo}'s allocation target has non-integer capacity/demand: {target!r}",
        )
        return None

    effective_demand = max(demand, min_in_flight_demand)
    return max(0, math.floor(capacity * headroom_ratio) - effective_demand)


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


def _log_unavailable(
    state_path: Path,
    repo: str,
    reason: str,
    detail: str,
    *,
    now: datetime,
    min_interval_minutes: int,
) -> None:
    """Write a ``ci_headroom_unavailable`` event, edge-triggered and rate-limited.

    Issue #1770 review finding 2: an unconditional write here turns every
    fail-open dispatch pass into a warning-level events.db row -- for a repo
    stuck ``stale``/``pinned``/``unconfigured`` (all long-lived conditions),
    that is one row per pass forever, degrading the digest's warning-count
    baseline until a real burst is indistinguishable from the constant noise.
    Skips the write when the freshest prior ``ci_headroom_unavailable`` event
    for this ``repo`` has the *same* reason and is younger than
    ``min_interval_minutes`` -- otherwise logs (first-ever, a reason
    transition, or the interval elapsed). events.db is the source of truth
    for "when did we last say this", not a new counter: nothing else in this
    module keeps state across calls.
    """
    if not _unavailable_should_emit(
        state_path, repo, reason, now=now, min_interval_minutes=min_interval_minutes
    ):
        return
    logger.info("ci_headroom_available(%s): unavailable (%s): %s", repo, reason, detail)
    log_event(
        state_path,
        UNAVAILABLE_EVENT_KIND,
        {"reason": reason, "detail": detail},
        repo=repo,
        level="warning",
    )


def _unavailable_should_emit(
    state_path: Path,
    repo: str,
    reason: str,
    *,
    now: datetime,
    min_interval_minutes: int,
) -> bool:
    """``True`` when a fresh ``ci_headroom_unavailable`` write is warranted.

    Reads the single freshest prior event for this exact ``repo`` back from
    ``state_path`` -- cheap (one indexed, ``LIMIT 1`` query) and correct even
    across process restarts, since it is derived from the same durable store
    the write itself lands in rather than any in-memory tracking. A prior
    event whose timestamp cannot be parsed fails toward emitting (the same
    "cannot trust this data" posture ``ci_headroom_available`` itself takes),
    never toward silently suppressing forever.
    """
    previous = query_events(state_path, kind=UNAVAILABLE_EVENT_KIND, repo=repo, limit=1)
    if not previous:
        return True
    prior_payload = previous[0].get("payload")
    prior_reason = prior_payload.get("reason") if isinstance(prior_payload, dict) else None
    if prior_reason != reason:
        return True
    prior_time = _parse_event_ts(previous[0].get("ts"))
    if prior_time is None:
        return True
    return now - prior_time >= timedelta(minutes=min_interval_minutes)
