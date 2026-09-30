"""Fleet-supervisor-scoped config section (issue #1978, step 1 of #1964).

``SupervisorConfig`` mixed knobs for the per-repo supervised loop
(``charlie bash-rats`` -- ``supervise.run_supervised``) with knobs only the
cross-repo fleet supervisor daemon reads (``charlie fleet supervise`` --
``fleet_dispatch.run_fleet_supervise`` / ``fleet_loop`` and
``cli.run_fleet_bash_rats``). Per CONTEXT.md the Supervisor and the Fleet
are separate layers, so the fleet-only knobs now live on
``FleetSupervisorConfig`` under the ``fleet_supervisor:`` section.

Precedence during the migration window: ``fleet_supervisor.<key>`` wins,
then the legacy ``supervisor.<key>`` location (still honored -- every moved
key is registered in ``config_deprecations.DEPRECATED_CONFIG_KEYS`` so a
legacy read emits ``config_key_deprecated_read`` and the fleet retirement
sweep arms removal issue #1979 once the old keys are absent from every
layer), then the dataclass default. Both locations set to *different*
values in the *same file* is a ``ConfigError`` naming both.

``fleet_supervisor:`` is a host-wide-only section (same treatment as
``runner_allocation``/``runner_capacity_escalation``): every knob on it
belongs to the one fleet supervisor daemon, so
``global_config.load_layered_config`` rejects the section in a per-repo
config and resolves each layer's new>legacy precedence *before* merging
(``resolve_fleet_supervisor_layer``). A repo layer's leftover
``supervisor.<key>`` therefore simply wins its key in the ordinary
repo-over-global merge -- deterministic, and still visible through the
deprecation event -- instead of surfacing as a merged-view conflict that
would discard the entire global layer.

The dataclass and its parser live here rather than in ``config.py`` so the
relocation does not grow that over-cap monolith (file-size ratchet, issue
#1442) -- the same arrangement as ``capacity_starvation_escalation.py`` and
``deescalation_config.py``. ``config.py`` re-exports the dataclass and calls
``parse_fleet_supervisor`` from ``build_config_from_data``; this module must
therefore not import ``config`` at module level (it imports only ``config_validation``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Annotated, Any, Mapping

from .config_validation import BoolTolerant, FieldError, Typed, validate_section

#: The top-level YAML section name. Referenced by ``config_deprecations``'s
#: ``replacement`` strings and the parser below so the name is declared once.
FLEET_SUPERVISOR_SECTION = "fleet_supervisor"

#: The legacy section the moved keys still parse from during the migration
#: window (removal tracked by issue #1979).
LEGACY_SUPERVISOR_SECTION = "supervisor"


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


def parse_fleet_supervisor(data: dict[str, Any]) -> FleetSupervisorConfig:
    """Validate and build the ``fleet_supervisor`` config section.

    Also honors the legacy ``supervisor.<key>`` locations during the
    migration window: a moved key found under ``supervisor:`` is removed from
    that section's mapping *in place* (so ``build_config_from_data``'s later
    ``SupervisorConfig`` build does not reject it as unknown) and adopted
    here when ``fleet_supervisor`` does not set it. ``data`` is
    ``build_config_from_data``'s private deepcopy, so the in-place removal
    never touches the caller's dict.

    A key set in both locations to different values is a ``ConfigError``
    naming both -- silently picking one would make an operator's real intent
    unresolvable. A legacy ``null`` is ignored, and a ``fleet_supervisor`` ``null``
    yields to a non-null legacy value; with no legacy value the ``null`` is kept
    (``None``, not the default -- the same null semantics as every other section).

    Each location is validated on its own (the moved keys of the legacy
    section are projected through the same ``FleetSupervisorConfig`` rules) so
    the error names the section the operator actually wrote; the merged
    result is then built once.
    """
    new_section = data.get(FLEET_SUPERVISOR_SECTION)
    if not isinstance(new_section, dict):
        new_section = {}
    legacy_section = data.get(LEGACY_SUPERVISOR_SECTION)
    if not isinstance(legacy_section, dict):
        legacy_section = {}

    moved = {f.name for f in fields(FleetSupervisorConfig)}
    validate_section(FleetSupervisorConfig, new_section, path=FLEET_SUPERVISOR_SECTION)
    validate_section(
        FleetSupervisorConfig,
        {k: v for k, v in legacy_section.items() if k in moved},
        path=LEGACY_SUPERVISOR_SECTION,
    )
    merged = _adopt_legacy_keys(new_section, legacy_section)
    return validate_section(FleetSupervisorConfig, merged, path=FLEET_SUPERVISOR_SECTION)


def _adopt_legacy_keys(
    new_section: Mapping[str, Any], legacy_section: dict[str, Any]
) -> dict[str, Any]:
    """Resolve one source's ``fleet_supervisor``/``supervisor`` pair.

    Moved keys found in ``legacy_section`` are popped out of it *in place*
    (so a later ``SupervisorConfig`` build never sees them as unknown) and
    adopted into the returned mapping when ``new_section`` does not set the
    key. A key in both locations with different values is a ``ConfigError``
    naming both; ``null`` counts as unset on either side. The conflict
    wording is shared verbatim by ``parse_fleet_supervisor`` (single file)
    and ``resolve_fleet_supervisor_layer`` (one merge layer) so the two
    paths cannot drift on what "both locations disagree" means.
    """
    merged = dict(new_section)
    for f in fields(FleetSupervisorConfig):
        key = f.name
        if key not in legacy_section:
            continue
        legacy_value = legacy_section.pop(key)
        if legacy_value is None:
            continue
        new_value = merged.get(key)
        if new_value is not None and new_value != legacy_value:
            raise FieldError(
                f"{FLEET_SUPERVISOR_SECTION}.{key}",
                f"one value across '{FLEET_SUPERVISOR_SECTION}' and legacy "
                f"'{LEGACY_SUPERVISOR_SECTION}'",
                f"{new_value!r} vs {legacy_value!r} (removal tracked by #1979); "
                f"delete '{LEGACY_SUPERVISOR_SECTION}.{key}'",
                "raw",
            )
        if new_value is None:
            merged[key] = legacy_value
    return merged


def resolve_fleet_supervisor_layer(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return one raw config-*layer* dict with legacy keys folded in.

    ``global_config.load_layered_config`` runs this on each layer BEFORE the
    repo-over-global merge so the moved-key adoption sees one file at a
    time: a ``fleet_supervisor.<key>``/``supervisor.<key>`` disagreement
    within one file still raises the ``ConfigError`` above, while a
    repo-layer ``supervisor.<key>`` that disagrees with the global layer's
    ``fleet_supervisor.<key>`` is just an ordinary per-key override -- the
    repo layer wins it, like every other repo-layer key. Without this, the
    cross-layer disagreement only surfaced post-merge as a conflict that
    triggered ``load_layered_config``'s #665 rescue and discarded the
    *entire* global layer over one disputed key.

    The input mapping is never mutated: the layer dict and both section
    dicts are copied, so callers keep the raw layer for deprecation-event
    attribution (``emit_deprecated_key_reads`` must still see the legacy
    spelling in the file it was read from). No type validation happens here
    -- ``parse_fleet_supervisor`` re-checks the merged ``fleet_supervisor``
    values, and a bad legacy value in a repo layer resurfaces with its own
    section name when the #665 rescue re-parses that file standalone.
    """
    out = dict(data)
    new_section = data.get(FLEET_SUPERVISOR_SECTION)
    new = dict(new_section) if isinstance(new_section, dict) else {}
    legacy_section = data.get(LEGACY_SUPERVISOR_SECTION)
    legacy = dict(legacy_section) if isinstance(legacy_section, dict) else {}

    effective = _adopt_legacy_keys(new, legacy)
    out[FLEET_SUPERVISOR_SECTION] = effective
    out[LEGACY_SUPERVISOR_SECTION] = legacy
    return out
