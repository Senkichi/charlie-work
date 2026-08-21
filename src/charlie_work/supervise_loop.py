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

import time
from collections.abc import Callable
from dataclasses import dataclass

# The supervisor's exit code meaning "I stopped because the code changed; launch
# a replacement now". Distinct from 0 (deliberate stop: max-runtime, max-passes,
# operator interrupt) and 1 (aborted). Deliberately consumed only inside Python:
# the launcher script never compares against the literal, so the number lives in
# exactly one place *in the source tree*.
#
# NEVER CHANGE THIS VALUE. It is a cross-version wire contract between two
# processes running different commits, not merely a module constant. The wrapper
# imports it at startup and holds it in memory; the supervisor child it spawns
# loads it fresh from disk. A self-deploy is precisely the moment those two
# disagree -- stale wrapper, new child. If a commit changed 3 to anything else,
# the wrapper would compare the child's new code against its old value, read the
# restart request as a normal exit, and not relaunch: #862, reintroduced for one
# watchdog interval, on the deploy that changed it.
#
# The recovery is bounded (the wrapper exits, the 5-minute tick starts a fresh
# one), which is why this is a documented invariant rather than a runtime
# handshake. Same discipline CLAUDE.md applies to label strings.
EXIT_RESTART_REQUESTED = 3

# The supervisor's exit code meaning "a fatal preflight check failed at
# startup; I never entered the pass loop" (issue #1363). Distinct from 0
# (deliberate clean stop), 1 (aborted/uncaught exception), and 3
# (EXIT_RESTART_REQUESTED, above). Deliberately a NEW value rather than
# reusing 1: exit 1 is folded into "an ordinary crash" by every consumer of
# this wrapper's result, while a preflight refusal is a distinct, actionable,
# named condition (see `preflight.py`) an operator should be able to tell
# apart from a stack trace at a glance -- `supervise_loop.py`'s log line and
# the fleet pass log both print the check name and detail already.
#
# This wrapper's own relaunch logic needs no special case for this value: any
# exit code other than EXIT_RESTART_REQUESTED already falls through to "do
# not relaunch" (see the `if exit_code != EXIT_RESTART_REQUESTED` branch
# below), so a preflight refusal at startup correctly does not spin the
# wrapper -- it surfaces once, exactly like exit 1 does today, and the
# 5-minute scheduled-task tick retries on its own cadence. A test proves this
# explicitly (`test_preflight_refusal_exit_code_does_not_relaunch`) rather
# than relying on the fallthrough being self-evident.
#
# NEVER reuse or renumber this to collide with EXIT_RESTART_REQUESTED -- same
# cross-version wire-contract discipline as that constant, since a stale
# wrapper and a freshly-deployed child could disagree on its meaning across a
# self-deploy boundary exactly as documented above.
PREFLIGHT_REFUSAL_EXIT_CODE = 4

# Chosen to be generous relative to the real event (a self-deploy chain is
# normally one restart) while still bounding a pathological loop to well under
# the 5-minute tick it must not starve.
DEFAULT_MAX_RELAUNCHES = 10

# How long a child must run before its restart request counts as "it did real
# work, then deployed" rather than "it died on startup again" (#903).
#
# This does NOT gate the cap -- the cap is never reset, because exiting at it is
# how this wrapper retires itself and hands restart authority back to the
# scheduled tick (see the module docstring). It gates only the *diagnosis*
# reported when the cap is reached, which is the one thing an operator reads at
# that moment.
#
# Both shapes are observable in fleet-pass.log: a drift-exit at startup
# completes in ~2s ("1 pass(es) in 0:00:02"), while a healthy child self-deploys
# after minutes of work ("4 pass(es) in 0:22:03"). 60s sits well above the
# former and well below the ~5-minute pass interval that bounds the latter.
SUSTAINED_RUN_SECONDS = 60.0

# ``cap_cause`` values. Both mean "the bound refused a restart"; they differ in
# what the operator should do about it.
CAP_CAUSE_NON_CONVERGENCE = "non_convergence"
CAP_CAUSE_RETIREMENT = "retirement"


@dataclass(frozen=True)
class SuperviseLoopResult:
    """Outcome of a wrapper run.

    ``relaunches`` counts replacements only, so ``launches == relaunches + 1``.
    ``cap_reached`` is True when the final child asked for a restart that the
    bound refused -- the case that must log loudly rather than spin.

    ``cap_cause`` distinguishes the two situations that reach the cap (#903):
    ``retirement`` (children ran and worked between restarts -- this wrapper is
    simply old, and stepping aside is the designed behavior) versus
    ``non_convergence`` (every restart came from a child that died almost
    immediately). It is ``None`` when the cap was not reached. Added with a
    default so existing positional construction keeps working, and so a record
    written before this field existed is not made to assert a cause it never
    observed.
    """

    launches: int
    relaunches: int
    last_exit_code: int
    cap_reached: bool
    cap_cause: str | None = None


def run_supervise_relaunch_loop(
    spawn: Callable[[int], int],
    *,
    max_relaunches: int = DEFAULT_MAX_RELAUNCHES,
    log: Callable[[str], None] = print,
    on_cap_reached: Callable[[SuperviseLoopResult], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sustained_run_seconds: float = SUSTAINED_RUN_SECONDS,
) -> SuperviseLoopResult:
    """Run the supervisor, relaunching while it requests a restart.

    ``spawn`` takes the 1-based launch number and returns the child's exit code;
    it is injected so the bound can be tested without spawning processes, and so
    the subprocess details stay in the CLI layer.

    ``on_cap_reached`` fires exactly once, only when the bound actually refuses a
    restart. Injected rather than calling ``log_event`` directly to keep this
    loop free of I/O -- the caller wires it to the fleet event sink.

    ``monotonic`` is injected for the same reason ``spawn`` is: it keeps the loop
    testable without real elapsed time. It must be a monotonic source, not a wall
    clock -- a clock step during a long-lived wrapper run would otherwise
    misclassify the cause reported at the cap.
    """
    if max_relaunches < 0:
        raise ValueError(f"max_relaunches must be >= 0, got {max_relaunches}")

    launches = 0
    relaunches = 0
    # Counts only restarts whose child died before it could do any work. Reset
    # by any sustained run, so it measures a *consecutive* fast-restart streak.
    # Deliberately separate from ``relaunches``: that one is the bound, and
    # resetting it would make this wrapper immortal and pin stale wrapper code
    # in memory -- the failure mode the module docstring rules out.
    consecutive_fast = 0
    while True:
        launches += 1
        started_at = monotonic()
        exit_code = spawn(launches)
        child_ran_for = monotonic() - started_at
        if exit_code != EXIT_RESTART_REQUESTED:
            # Includes 0 (max-runtime / max-passes / operator stop) and 1
            # (aborted). AC4: a normal exit must not relaunch.
            return SuperviseLoopResult(
                launches=launches,
                relaunches=relaunches,
                last_exit_code=exit_code,
                cap_reached=False,
            )

        sustained = child_ran_for >= sustained_run_seconds
        consecutive_fast = 0 if sustained else consecutive_fast + 1

        if relaunches >= max_relaunches:
            # Both causes exit here; only the diagnosis differs. Non-convergence
            # requires that *every* child this wrapper ever spawned died fast --
            # one sustained run anywhere is proof the child can get up and do
            # work, which is the opposite of a restart loop. Stated as an
            # equality against ``launches`` rather than a threshold on
            # ``consecutive_fast`` so it stays obviously correct at the
            # boundaries, including ``max_relaunches=0``.
            non_converging = consecutive_fast == launches
            result = SuperviseLoopResult(
                launches=launches,
                relaunches=relaunches,
                last_exit_code=exit_code,
                cap_reached=True,
                cap_cause=(CAP_CAUSE_NON_CONVERGENCE if non_converging else CAP_CAUSE_RETIREMENT),
            )
            if non_converging:
                log(
                    f"supervise-loop: relaunch cap reached ({relaunches}/{max_relaunches}) "
                    f"after {launches} launch(es); every child exited within "
                    f"{sustained_run_seconds:.0f}s without completing work. Exiting so "
                    f"the scheduled task's next tick relaunches on current code -- "
                    f"this means self-deploy is not converging."
                )
            else:
                log(
                    f"supervise-loop: relaunch cap reached ({relaunches}/{max_relaunches}) "
                    f"after {launches} launch(es), with children running normally in "
                    f"between. Retiring this wrapper so the scheduled task's next tick "
                    f"starts one on current code -- expected after sustained uptime, "
                    f"not a self-deploy fault."
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
