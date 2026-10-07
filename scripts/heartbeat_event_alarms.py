"""Per-repo ``events.db`` kind/level anomaly checks for ``heartbeat_check.py`` (issue #1895).

The five checks here share one signature
(``report, repo, baseline -> None``), one db-availability posture (a missing
or unreadable ``events.db`` is ``report.anom`` -- a registered repo this
check cannot read is a repo it cannot vouch for), and one timestamp
convention (ISO ``ts`` strings parsed with the leaf's ``heartbeat_alarms.parse_iso`` and compared
against ``baseline`` in Python, never in SQL -- see
``check_loop_pass_freshness``'s docstring in ``heartbeat_check.py`` for the
ISO-``T``/``Z``-vs-SQLite-space-format trap).

Loaded from ``heartbeat_check.py`` via ``importlib`` from the sibling script
path, never a bare ``import`` -- matching how #1879 loads
``heartbeat_local_repo.py`` and how ``git_push_lint_hook.py`` loads
``worker_stop_gate.py``: ``scripts/`` is not a package and is deliberately
kept off ``sys.path`` by the test harness (``tests/_script_loader.py``).
``heartbeat_check`` re-exports all five names, so ``hb.check_*`` attribute
references in tests keep resolving unchanged. This module is never run
standalone.

Extracted verbatim out of ``heartbeat_check.py`` purely to create file-size
ratchet headroom (``file_size_ratchet_baseline/scripts/heartbeat_check.py.count``)
-- no behavior change; the move unblocks #1861 / PR #1879, whose merge of a
grown ``origin/main`` could not fit under the 3200 mark.

Stdlib-only, same constraint as ``heartbeat_check.py`` itself
(``scripts/README.md``): no ``charlie_work`` or third-party imports beyond
the guarded ``charlie_work.event_kinds`` leaf below, and never an import
back into ``heartbeat_check`` -- that would cycle through its loader block,
which is also why ``Report``/``RepoInfo`` are ``TYPE_CHECKING``-only names
and the shared ``parse_iso`` comes from the ``charlie_work.heartbeat_alarms`` leaf.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

    # ``heartbeat_check`` is not importable as a module at runtime
    # (``scripts/`` is not a package and stays off ``sys.path``); these names
    # exist so the moved functions keep their original annotations
    # byte-identically. A runtime import would cycle through
    # ``heartbeat_check``'s own loader block.
    from heartbeat_check import RepoInfo, Report

# Issue #1271: the single declared source of truth for which warning-level
# kinds are normal-operation signals (see the frozenset's own docstring).
# Imported, never re-declared or hardcoded here, so check_warning_events'
# bucketing stays correct as that set changes without this file needing an
# edit. Unlike the rest of the heartbeat script (see `heartbeat_check.py`'s
# stale-open-issue-mention section docstring for why it otherwise avoids
# importing charlie_work), this one string-literal set is worth importing
# directly: duplicating it here would be exactly the kind of hardcoded list
# that drifts from the registry it is supposed to mirror.
#
# Imported from `charlie_work.event_kinds` specifically, NEVER from
# `charlie_work.instrumentation` -- that module imports `ci_fleet` at
# module load, and this script family must stay importable even when
# `ci_fleet` isn't (that's the entire reason it is stdlib-only; see
# scripts/README.md). `event_kinds` is a genuine leaf: no charlie_work or
# ci_fleet imports of its own, so importing it can never reach ci_fleet
# transitively.
#
# Guarded with try/except, not a bare import: heartbeat_check.py is routinely
# run via `uv run --active`, which resolves against whatever venv happens to
# be active rather than this project's own -- a documented failure mode in
# this fleet (see the `uv-worktree-virtualenv-shadowing` project memory)
# where `charlie_work` itself is not importable at all, not merely
# `ci_fleet`. `scripts/README.md`'s invariant is "a broken package install
# can never break the check that would detect it" -- unconditional, not
# scoped to ci_fleet -- so a missing `charlie_work` degrades
# check_warning_events to the exact pre-#1271 behavior (no bucketing; every
# warning kind goes to the flat detailed list) instead of crashing on
# import. `heartbeat_check` re-exports this name
# (`EXPECTED_OPERATIONAL_KINDS = _event_alarms.EXPECTED_OPERATIONAL_KINDS`)
# so the empty-frozenset degradation the ci_fleet-isolation tests assert on
# still resolves on the loaded heartbeat_check module.
try:
    from charlie_work.event_kinds import EXPECTED_OPERATIONAL_KINDS
except ImportError:
    EXPECTED_OPERATIONAL_KINDS: frozenset[str] = frozenset()

# The verdict logic for every check below lives in the stdlib-only leaf
# ``charlie_work.heartbeat_alarms`` (shared with the fleet dashboard); this
# module only reads events.db and adapts the returned ``Finding`` onto the
# heartbeat ``Report``. Resolution order: the installed package first; if
# ``charlie_work`` is not importable (wrong venv -- see above) the leaf files
# are loaded BY PATH from ``<repo>/src/charlie_work`` (they are stdlib-only,
# so that works with no package at all). Only when both fail does a check
# report a loud ANOMALY it cannot evaluate -- never silently green.
_LEAF_DIR = Path(__file__).resolve().parent.parent / "src" / "charlie_work"
# Shared sys.modules name: ``heartbeat_alarms_fleet``'s file-path fallback
# reuses this entry so ``Finding`` has one identity across both leaves.
_HA_LEAF_NAME = "_cw_heartbeat_alarms"


def _load_leaf_by_path(module_name: str, filename: str) -> Any:
    """Load ``<repo>/src/charlie_work/<filename>`` by path; ``None`` on any failure."""
    import importlib.util
    import sys

    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    path = _LEAF_DIR / filename
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001 - absent/corrupt leaf => caller reports ANOMALY
        sys.modules.pop(module_name, None)
        return None


def load_alarm_leaves() -> tuple[Any, Any]:
    """``(heartbeat_alarms, heartbeat_alarms_fleet)``, each ``None`` if unloadable."""
    try:
        from charlie_work import heartbeat_alarms as ha
    except ImportError:
        ha = _load_leaf_by_path(_HA_LEAF_NAME, "heartbeat_alarms.py")
    try:
        from charlie_work import heartbeat_alarms_fleet as haf
    except ImportError:
        haf = _load_leaf_by_path("_cw_heartbeat_alarms_fleet", "heartbeat_alarms_fleet.py")
    return ha, haf


_ha, _ = load_alarm_leaves()


def _read_rows(
    report: Report, check: str, db_path: Path, prefix: str, queries: list[tuple[str, tuple]]
) -> list[list[tuple]] | None:
    """Run each ``(sql, params)`` against ``db_path``; ``None`` after an ANOMALY.

    A missing/unreadable events.db or absent ``events`` table is ``report.anom``
    (``prefix`` is the per-check message lead): a repo this check cannot read is
    a repo it cannot vouch for. Timestamps are never compared in SQL.
    """
    if not db_path.exists():
        report.anom(check, f"{prefix}: no events.db at {db_path}")
        return None
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        report.anom(check, f"{prefix}: events.db unreadable: {exc}")
        return None
    try:
        try:
            table_row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            if table_row is None:
                report.anom(check, f"{prefix}: events.db has no events table")
                return None
            return [conn.execute(sql, params).fetchall() for sql, params in queries]
        except sqlite3.Error as exc:
            report.anom(check, f"{prefix}: events.db unreadable: {exc}")
            return None
    finally:
        conn.close()


def _run(
    report: Report,
    repo: RepoInfo,
    base: str,
    prefix: str,
    queries: list[tuple[str, tuple]],
    evaluate: Callable[..., Any],
) -> None:
    check = f"{base} {repo.slug}"
    if _ha is None:
        report.anom(check, "cannot evaluate: charlie_work.heartbeat_alarms not importable")
        return
    results = _read_rows(report, check, repo.state_dir / "events.db", prefix, queries)
    if results is None:
        return
    out = evaluate(*results)
    for finding in out if isinstance(out, list) else [out]:
        _ha.emit(report, finding)


def check_error_events(report: Report, repo: RepoInfo, baseline: datetime) -> None:
    """Surface error-level events that fire but have no consumer (issue #866).

    Coverage is DERIVED from the persisted per-row ``level = 'error'`` (ground
    truth for what the classifier assigned at write time), never a kind list.
    A missing/unreadable events.db is an ANOMALY here -- this check's whole job
    is "did any alarm fire." Only rows newer than ``baseline`` (previous beat)
    are reported. Verdict: ``heartbeat_alarms.eval_error_events``.
    """
    _run(
        report,
        repo,
        "error-events",
        "cannot check for alarms",
        [("SELECT ts, kind FROM events WHERE level = 'error'", ())],
        lambda rows: _ha.eval_error_events(repo.slug, rows, baseline),
    )


def check_warning_events(report: Report, repo: RepoInfo, baseline: datetime) -> None:
    """Surface warning-level events that fire but have no consumer (issue #946).

    Mirrors ``check_error_events`` one level down, but new rows are WARN, never
    ANOMALY (several warning kinds are normal operation). Kinds in
    ``EXPECTED_OPERATIONAL_KINDS`` (issue #1271) are bucketed into a one-line
    sorted count. Verdict: ``heartbeat_alarms.eval_warning_events``.
    """
    _run(
        report,
        repo,
        "warning-events",
        "cannot check for warnings",
        [("SELECT ts, kind FROM events WHERE level = 'warning'", ())],
        lambda rows: _ha.eval_warning_events(
            repo.slug, rows, baseline, EXPECTED_OPERATIONAL_KINDS
        ),
    )


def check_infra_blocked_events(report: Report, repo: RepoInfo, baseline: datetime) -> None:
    """Surface ``check_infra_blocked`` events and their persisted escalation
    (issue #1383, AC4): escalated is an ANOMALY, bare blocks a WARN.
    Verdict: ``heartbeat_alarms.eval_infra_blocked``."""
    _run(
        report,
        repo,
        "infra-blocked-events",
        "cannot check",
        [
            ("SELECT ts, kind FROM events WHERE kind = ?", ("check_infra_blocked",)),
            ("SELECT ts FROM events WHERE kind = ?", ("infra_blocked_escalated",)),
        ],
        lambda blocked, escalated: _ha.eval_infra_blocked(repo.slug, blocked, escalated, baseline),
    )


def check_draft_pr_blocked_events(report: Report, repo: RepoInfo, baseline: datetime) -> None:
    """Surface ``draft_pr_blocked`` events periodically (issue #1366) as WARN --
    a state to surface, not a fleet emergency. Verdict:
    ``heartbeat_alarms.eval_draft_pr_blocked``."""
    _run(
        report,
        repo,
        "draft-pr-blocked-events",
        "cannot check",
        [("SELECT ts FROM events WHERE kind = ?", ("draft_pr_blocked",))],
        lambda rows: _ha.eval_draft_pr_blocked(repo.slug, rows, baseline),
    )


def check_ci_headroom_unavailable(report: Report, repo: RepoInfo, baseline: datetime) -> None:
    """Surface ``ci_headroom_unavailable`` events periodically (issue #1770) as
    WARN, with payload ``reason`` bucketing. Verdict:
    ``heartbeat_alarms.eval_ci_headroom_unavailable``."""
    _run(
        report,
        repo,
        "ci-headroom-unavailable-events",
        "cannot check",
        [("SELECT ts, payload FROM events WHERE kind = ?", ("ci_headroom_unavailable",))],
        lambda rows: _ha.eval_ci_headroom_unavailable(repo.slug, rows, baseline),
    )


# Issue #1968: emitted once per local-lane pass per disabled
# ``review_dispatch.enabled``/``auto_merge.enabled`` switch while a
# review-ready issue sits parked past ``local_lane.kill_switch_stall_hours``
# on a ``local_issues`` repo. The literal is declared here (not imported):
# the kind only exists in the newest charlie_work, and this script must run
# against any installed package version.
LOCAL_LANE_KILL_SWITCH_STALLED = "local_lane_kill_switch_stalled"

# Issue #2441: emitted once per episode of a PR sitting in the merge queue past
# the stall threshold. Literal declared here for the same reason as above.
MERGEQUEUE_STALLED = "mergequeue_stalled"


def check_local_lane_kill_switch_stalled(
    report: Report, repo: RepoInfo, baseline: datetime
) -> None:
    """Surface ``local_lane_kill_switch_stalled`` events as ANOMALY (#1968): a
    stranded review/merge lane means finished work is going nowhere. Forwards
    the event detail only; it does not re-evaluate config. Verdict:
    ``heartbeat_alarms.eval_local_lane_stalled``."""
    _run(
        report,
        repo,
        "local_lane_kill_switch_stalled",
        "cannot check",
        [("SELECT ts, payload FROM events WHERE kind = ?", (LOCAL_LANE_KILL_SWITCH_STALLED,))],
        lambda rows: _ha.eval_local_lane_stalled(repo.slug, rows, baseline),
    )


def check_mergequeue_stalled(report: Report, repo: RepoInfo, baseline: datetime) -> None:
    """Surface ``mergequeue_stalled`` events as WARN (#2441): a PR has carried the
    merge-queue label past the stall threshold without merging. Forwards the
    event detail only. Verdict: ``heartbeat_alarms.eval_mergequeue_stalled``."""
    _run(
        report,
        repo,
        "mergequeue_stalled",
        "cannot check",
        [("SELECT ts, payload FROM events WHERE kind = ?", (MERGEQUEUE_STALLED,))],
        lambda rows: _ha.eval_mergequeue_stalled(repo.slug, rows, baseline),
    )
