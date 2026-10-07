"""Fleet-supervisor-scoped config section (issue #1978, step 1 of #1964).

``SupervisorConfig`` mixed knobs for the per-repo supervised loop
(``charlie bash-rats`` -- ``supervise.run_supervised``) with knobs only the
cross-repo fleet supervisor daemon reads (``charlie fleet supervise`` --
``fleet_dispatch.run_fleet_supervise`` / ``fleet_loop`` and
``cli.run_fleet_bash_rats``). Per CONTEXT.md the Supervisor and the Fleet
are separate layers, so the fleet-only knobs now live on
``FleetSupervisorConfig`` under the ``fleet_supervisor:`` section.

Issue #1979 removed the migration window's legacy ``supervisor.<key>``
fallback (and with it ``resolve_fleet_supervisor_layer``'s per-layer fold):
a moved key still written under ``supervisor:`` is now an ordinary
unknown-key ``ConfigError``.

``fleet_supervisor:`` is a host-wide-only section (same treatment as
``runner_allocation``/``runner_capacity_escalation``): every knob on it
belongs to the one fleet supervisor daemon, so
``global_config.load_layered_config`` rejects the section in a per-repo
config.

The dataclass lives here rather than in ``config.py`` so the relocation
does not grow that over-cap monolith (file-size ratchet, issue
#1442) -- the same arrangement as ``capacity_starvation_escalation.py`` and
``deescalation_config.py``. ``config.py`` re-exports the dataclass; the
generic section loop in ``build_config_from_data`` validates it like every
other section. This module must therefore not import ``config`` at module
level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from .config_validation import BoolTolerant, Typed

#: The top-level YAML section name.
FLEET_SUPERVISOR_SECTION = "fleet_supervisor"


@dataclass(frozen=True)
class FleetSupervisorConfig:
    """Knobs only the cross-repo fleet supervisor reads.

    Every field relocated verbatim out of ``supervisor:`` (issue #1978);
    defaults are unchanged. The loop-cadence knobs shared with the per-repo
    ``charlie bash-rats`` loop (``poll_interval_seconds``,
    ``full_pass_interval_seconds``, ``active_cooldown_seconds``,
    ``max_runtime_minutes``) stay on ``SupervisorConfig``.

    ``max_pass_runtime_seconds``: upper bound on a single fleet pass's
    wall-clock duration. The supervisor heartbeat freshness check uses this
    bound so a long-running pass is not mistaken for a dead supervisor
    (default 1800 s / 30 min).
    ``self_deploy_failure_alarm``: consecutive ``self_deploy`` failures before
    a ``self_deploy_alarm`` events.db entry fires (default 3, mirrors
    ``AutoMergeConfig.failed_attempt_alarm``). 0 disables the alarm.
    ``zero_pass_alarm``: consecutive fleet-supervisor cycles that complete
    with zero repo passes, despite at least one repo being configured,
    before a ``supervisor_zero_pass_alarm`` events.db entry fires (default 3,
    mirrors ``self_deploy_failure_alarm``). 0 disables the alarm. A cycle
    with zero repos configured never counts toward this streak in either
    direction -- that is a configuration state, not an incident (issue #855).
    ``self_deploy_pull_ci_fleet``: when true, ``self_deploy`` also FF-pulls
    ``origin/main`` in the declared ``ci-fleet`` sibling checkout after a
    successful orchestrator pull, but only when that sibling is clean and on
    ``main``. Default false: in a development layout the sibling is a working
    repo whose HEAD must never be moved out from under a session. Enable it
    only for a dedicated deploy clone (issue #552), where the sibling exists
    solely to be deployed to -- without it the daemon's editable ``ci_fleet``
    is a silent version freeze, since ``self_deploy`` otherwise only ever
    pulls the orchestrator checkout.
    ``wedge_kill_loop_alarm``: consecutive ``supervisor_wedged_killed`` events
    with no ``fleet_pass_completed`` event in between (i.e. the wedge-kill
    backstop firing without the supervisor ever completing a pass again)
    before a ``supervisor_wedge_loop`` events.db entry fires (default 3,
    mirrors ``self_deploy_failure_alarm``/``zero_pass_alarm``). 0 disables
    the alarm (issue #1832).
    ``dependency_sync_starvation_seconds``: upper bound on how long a deferred
    self-deploy ``uv sync`` may stay pending while fleet workers are live
    before the supervisor stops admitting new dispatches (drain posture:
    workers in flight are untouched) so the live-worker count can reach zero
    and the sync can land (issue #1855). Measured wall-clock from the
    pending-sync marker's ``written_at`` -- the first deferral of the
    episode -- so it is robust to supervisor restarts. Default 14400 s (4 h):
    comfortably above observed worker session durations, and far below the
    multi-hour continuous deferral observed under sustained fleet load.
    <= 0 disables the bound.
    ``fleet_lane_concurrency``: maximum number of per-repo lanes one
    ``fleet_loop`` pass runs concurrently (issue #1934). Per-repo lane work is
    I/O-bound and repo-isolated (own config, GitHub client, supervisor lock,
    state files), so lanes run on a bounded thread pool: a pass's wall-clock
    approximates the slowest lane instead of the sum of every lane (~35-42
    min observed across 6 repos for a configured 5-minute cadence). When the
    cap meets or exceeds the registered repo count, a repo's lane-to-lane gap
    is bounded by its own lane duration plus the supervisor's pass cadence --
    decoupled from sibling lanes' workloads. <= 0 falls back to the built-in
    default at the call site; 1 restores the pre-#1934 strict-serial order.
    ``reap_sweep_interval_seconds``: cadence for the fleet supervisor's
    out-of-band review-claim reap scheduler (issue #1934). The
    dead-reviewer-claim sweep set (``OrchestratorApp._run_review_reap_sweeps``
    -- the identical block ``dispatch_reviews`` and ``reap_reviews`` run)
    executes once per registered repo on this interval from a dedicated
    thread, independent of whether a fleet pass is due or in flight, so a
    dead claim is freed on a ~5-minute cadence instead of once per fleet-wide
    round. The sweeps launch nothing and are ``state_lock``-serialized /
    merge-on-write safe against a concurrent lane (issue #1874's design), so
    they run even while a repo's supervisor lock is held. <= 0 disables the
    scheduler.
    """

    max_pass_runtime_seconds: Annotated[int, Typed, BoolTolerant] = 1800
    self_deploy_failure_alarm: Annotated[int, Typed, BoolTolerant] = 3
    self_deploy_pull_ci_fleet: Annotated[bool, Typed] = False
    zero_pass_alarm: Annotated[int, Typed, BoolTolerant] = 3
    wedge_kill_loop_alarm: Annotated[int, Typed, BoolTolerant] = 3
    dependency_sync_starvation_seconds: Annotated[int, Typed, BoolTolerant] = 14400
    fleet_lane_concurrency: Annotated[int, Typed, BoolTolerant] = 8
    reap_sweep_interval_seconds: Annotated[int, Typed, BoolTolerant] = 300
