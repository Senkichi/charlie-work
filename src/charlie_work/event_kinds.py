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
    }
)
