"""Event-``kind`` constants shared across a package/no-package boundary.

This module exists for exactly one reason: ``scripts/heartbeat_check.py`` is
deliberately stdlib-only (plus ``psutil``/``yaml``) so that a broken
``charlie_work``/``ci_fleet`` install can never break the check meant to
detect that (see ``scripts/README.md``). ``charlie_work.instrumentation`` --
the natural home for anything describing event ``kind`` strings -- imports
``ci_fleet.observability`` and ``ci_fleet.provenance`` at module load, so it
is unsafe for ``heartbeat_check.py`` to import from directly: a broken
``ci_fleet`` install would turn an intended ANOMALY line into an unhandled
``ImportError`` on exactly the failure class the check exists to report.

Keep this module a genuine leaf: stdlib-only, and no imports of any other
``charlie_work`` or ``ci_fleet`` module, ever -- that is the entire point of
its existence. If a constant declared here ever needs something heavier,
move it to ``instrumentation.py`` (or wherever fits) instead of adding an
import here.
"""

from __future__ import annotations

# Issue #1271: warning-level kinds that are normal-operation signals, not
# faults -- ``scripts/heartbeat_check.py``'s own ``check_warning_events``
# docstring already named these as such (``session_exited``: process
# liveness proving the worker is gone, not that it failed, per #873;
# ``dispatch_stale``: a paused fleet with a non-empty backlog;
# ``runner_capacity_starved`` / ``draft_pr_ready_held``: self-pacing, not
# damage). They fire at high volume relative to genuinely rare warnings, so
# ``check_warning_events`` buckets them into a summarized count instead of
# interleaving them with the flat detailed listing every other warning kind
# still gets.
#
# This is the single declared source of truth for that bucket:
# ``heartbeat_check.py`` imports it directly from here (never from
# ``charlie_work.instrumentation``, and never re-declared or hardcoded), so
# adding a member needs no change on that side and never risks pulling
# ``ci_fleet`` into the stdlib-only script. ``charlie_work.instrumentation``
# re-exports the same object for in-package consumers -- both spellings
# refer to the identical frozenset.
#
# Every member must be registered in ``instrumentation._LEVEL_BY_KIND`` at
# ``"warning"`` -- bucketing only makes sense for warnings -- which
# ``test_expected_operational_kinds_are_all_registered_warnings`` enforces.
EXPECTED_OPERATIONAL_KINDS: frozenset[str] = frozenset(
    {
        "session_exited",
        "dispatch_stale",
        "runner_capacity_starved",
        "draft_pr_ready_held",
        # Issue #1314 item 3: the operator-queue depth gauge fires every pass
        # the gauge is due when the queue is chronically deep (depth exceeds
        # ``operator_queue_depth_threshold``). That is exactly the
        # high-volume-relative-to-genuinely-rare-warnings shape this bucket
        # exists for: a fleet with a stuck queue emits this on every pass,
        # and interleaving it flat with rare warnings would drown them out.
        # The consumer (``heartbeat_check.py``'s ``check_warning_events``)
        # already buckets every member of this set into a summarized count,
        # so adding the kind here IS the consumer wiring the
        # signal-without-a-consumer rule requires to land in the same PR.
        "operator_queue_depth",
    }
)

# Issue #1363: `preflight_warning` (clock_sanity) and `preflight_config_stale`
# (config_freshness) are deliberately NOT added to EXPECTED_OPERATIONAL_KINDS
# above, even though both are non-fatal tripwires. That bucket exists for
# kinds that "routinely dominate warning volume" (see its own docstring) --
# summarizing them into a count is what keeps genuinely rare warnings from
# being drowned out. Both preflight tripwires are the opposite: clock_sanity
# should almost never fire on a healthy host, and config_freshness's own spec
# (AC5) requires it to fire exactly once per config-file edit, not
# routinely. AC7's rationale for requiring them classified at all is "they
# are exactly the kinds an operator must see" -- bucketing them WITH the
# high-volume kinds would misfile them under the wrong reason. They still
# get their required `warning`-level classification via
# `instrumentation._LEVEL_BY_KIND`, which is what makes them visible to
# `check_warning_events` at all; they simply keep the flat, unsummarized
# `kind@ts` presentation like every other rare warning kind.
