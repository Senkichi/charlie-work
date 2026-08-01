"""Bounded relaunch wrapper for the fleet supervisor (#862).

The supervisor exits deliberately when the code on disk changes underneath it
(self-deploy pulled a new commit, or HEAD moved out-of-band). That exit is
correct and stays as-is -- Python does not hot-reload modules, so only a fresh
interpreter runs the new commit. What was wrong is *when* the replacement
arrives: the only thing that relaunched was the 5-minute scheduled-task tick, so
every self-deploy bought a fleet-wide gap of up to a full watchdog interval, and
`LastTaskResult` read 0 throughout because a restart-requesting exit was
indistinguishable from a clean timeout.

**Why a wrapper process rather than an in-process retry loop.** The stale module
objects are the problem; re-entering `run_fleet_supervise` in the same
interpreter would re-run exactly the code the exit was trying to escape. The
replacement has to be a new process.

**Why this wrapper being stale itself is acceptable.** It imports charlie_work
too, so a self-deploy leaves it running pre-deploy code as well. That is
tolerable only because its job is small and stable (spawn, wait, count) and
because it *exits* at the cap -- at which point the scheduled task's tick
launches a fresh wrapper on the new commit. A wrapper that retried forever would
pin stale code in memory indefinitely and, worse, take over the restart
authority the tick currently holds. The bound is what keeps the tick meaningful,
which is why it is not merely a safety valve against runaway loops.

**Why not the alternatives** (both raised and rejected in #862): shortening the
watchdog interval multiplies wakeups for every repo to fix a rare event, and
`os.execv` re-exec keeps the same PID, so the scheduled task's
`MultipleInstancesPolicy=IgnoreNew` would keep suppressing ticks against a
process that is no longer the one it started.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# The supervisor's exit code meaning "I stopped because the code changed; launch
# a replacement now". Distinct from 0 (deliberate stop: max-runtime, max-passes,
# operator interrupt) and 1 (aborted). Deliberately consumed only inside Python:
# the launcher script never compares against the literal, so the number lives in
# exactly one place.
EXIT_RESTART_REQUESTED = 3

# Chosen to be generous relative to the real event (a self-deploy chain is
# normally one restart) while still bounding a pathological loop to well under
# the 5-minute tick it must not starve.
DEFAULT_MAX_RELAUNCHES = 10


@dataclass(frozen=True)
class SuperviseLoopResult:
    """Outcome of a wrapper run.

    ``relaunches`` counts replacements only, so ``launches == relaunches + 1``.
    ``cap_reached`` is True when the final child asked for a restart that the
    bound refused -- the case that must log loudly rather than spin.
    """

    launches: int
    relaunches: int
    last_exit_code: int
    cap_reached: bool


def run_supervise_relaunch_loop(
    spawn: Callable[[int], int],
    *,
    max_relaunches: int = DEFAULT_MAX_RELAUNCHES,
    log: Callable[[str], None] = print,
    on_cap_reached: Callable[[SuperviseLoopResult], None] | None = None,
) -> SuperviseLoopResult:
    """Run the supervisor, relaunching while it requests a restart.

    ``spawn`` takes the 1-based launch number and returns the child's exit code;
    it is injected so the bound can be tested without spawning processes, and so
    the subprocess details stay in the CLI layer.

    ``on_cap_reached`` fires exactly once, only when the bound actually refuses a
    restart. Injected rather than calling ``log_event`` directly to keep this
    loop free of I/O -- the caller wires it to the fleet event sink.
    """
    if max_relaunches < 0:
        raise ValueError(f"max_relaunches must be >= 0, got {max_relaunches}")

    launches = 0
    relaunches = 0
    while True:
        launches += 1
        exit_code = spawn(launches)
        if exit_code != EXIT_RESTART_REQUESTED:
            # Includes 0 (max-runtime / max-passes / operator stop) and 1
            # (aborted). AC4: a normal exit must not relaunch.
            return SuperviseLoopResult(
                launches=launches,
                relaunches=relaunches,
                last_exit_code=exit_code,
                cap_reached=False,
            )

        if relaunches >= max_relaunches:
            result = SuperviseLoopResult(
                launches=launches,
                relaunches=relaunches,
                last_exit_code=exit_code,
                cap_reached=True,
            )
            log(
                f"supervise-loop: relaunch cap reached ({relaunches}/{max_relaunches}) "
                f"after {launches} launch(es); the supervisor is still requesting a "
                f"restart. Exiting so the scheduled task's next tick relaunches on "
                f"current code -- repeated caps mean self-deploy is not converging."
            )
            if on_cap_reached is not None:
                on_cap_reached(result)
            return result

        relaunches += 1
        log(
            f"supervise-loop: supervisor requested restart (exit "
            f"{EXIT_RESTART_REQUESTED}); relaunching immediately "
            f"({relaunches}/{max_relaunches})"
        )
