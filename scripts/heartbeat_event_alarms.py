"""Per-repo ``events.db`` kind/level anomaly checks for ``heartbeat_check.py`` (issue #1895).

The five checks here share one signature
(``report, repo, baseline -> None``), one db-availability posture (a missing
or unreadable ``events.db`` is ``report.anom`` -- a registered repo this
check cannot read is a repo it cannot vouch for), and one timestamp
convention (ISO ``ts`` strings parsed with ``parse_iso`` and compared
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
and ``parse_iso`` is mirrored below rather than shared.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


# Mirrors ``heartbeat_check.parse_iso`` verbatim -- sibling scripts cannot
# import each other (``scripts/`` is not a package), and a back-import would
# cycle through ``heartbeat_check``'s loader block. Keep it byte-identical:
# the timestamp comparison convention above depends on the two copies
# agreeing.
def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def check_error_events(report: Report, repo: RepoInfo, baseline: datetime) -> None:
    """Surface error-level events that fire but have no consumer (issue #866).

    `self_deploy_alarm` and every other member of `instrumentation._ERROR_KINDS`
    (e.g. PR #865's `supervisor_zero_pass_alarm`) are emitted, classified
    error-level, documented, and unit-tested -- but before this check,
    nothing in the codebase ever read them. A human had to manually open
    `events.db` and know which `kind` string to search for. This check
    closes that detection-to-delivery gap; `check_loop_pass_freshness` above
    is a separate, coarser backstop (defense in depth), not a substitute --
    that one answers "did the loop run recently," this one answers "did
    anything already flag itself as an error."

    Coverage is DERIVED, never a hardcoded `kind` list: `level` is computed
    once and persisted per-row at write time by
    `instrumentation._classify_level` (checked against `_ERROR_KINDS`/
    `_WARNING_KINDS` there), so filtering on the persisted `level = 'error'`
    column here picks up every current and future error kind without this
    script importing `charlie_work` or restating its kind list. This is more
    correct than importing `_ERROR_KINDS` directly would be, too:
    `_ERROR_KINDS` reflects the currently-installed code, while a row's
    `level` reflects what the classifier actually assigned when that row was
    written -- the two can disagree across a deploy boundary, and the
    persisted column is ground truth for "what actually happened."

    Unlike `check_loop_pass_freshness` (missing db/table = OK, "no history
    yet" -- a fresh install is not a failure), a missing or unreadable
    events.db HERE is an ANOMALY: this check's entire job is "did any alarm
    fire," and a registered repo this check cannot read is a repo it cannot
    vouch for, not one it can call clean.

    Only rows with `ts` strictly after `baseline` (the previous heartbeat
    beat -- same mechanism as `check_dispatch_failures`) are reported, so an
    already-seen alarm is not re-flagged forever. On a cold start (no prior
    `heartbeat-state.json`), `main()` falls `baseline` back to
    `now - LOG_FRESHNESS_STALE_MINUTES`, so alarms older than that fallback
    window are silently out of scope on the very first run -- a deliberate,
    bounded blind spot, not an oversight.

    CRITICAL: timestamps are compared in Python, never in SQL -- the same
    ISO-`T`/`Z`-vs-SQLite-space-format trap documented on
    `check_loop_pass_freshness`. All `level='error'` rows are pulled
    unfiltered by time and each `ts` is parsed with `parse_iso` and compared
    against the `baseline` `datetime` in Python.
    """
    check = f"error-events {repo.slug}"
    db_path = repo.state_dir / "events.db"
    if not db_path.exists():
        report.anom(check, f"cannot check for alarms: no events.db at {db_path}")
        return

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        report.anom(check, f"cannot check for alarms: events.db unreadable: {exc}")
        return

    try:
        try:
            table_row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            if table_row is None:
                report.anom(check, "cannot check for alarms: events.db has no events table")
                return
            rows = conn.execute("SELECT ts, kind FROM events WHERE level = 'error'").fetchall()
        except sqlite3.Error as exc:
            report.anom(check, f"cannot check for alarms: events.db unreadable: {exc}")
            return
    finally:
        conn.close()

    new_alarms: list[str] = []
    for ts, kind in rows:
        ts_dt = parse_iso(ts)
        # An unparseable ts fails toward visibility (reported), not silence.
        if ts_dt is None or ts_dt > baseline:
            new_alarms.append(f"{kind}@{ts}")

    facts = f"error_rows={len(rows)} new_since_last_beat={len(new_alarms)}"
    if new_alarms:
        report.anom(check, f"new error-level event(s) since last beat: {new_alarms} ({facts})")
    else:
        report.ok(check, facts)


def check_warning_events(report: Report, repo: RepoInfo, baseline: datetime) -> None:
    """Surface warning-level events that fire but have no consumer (issue #946).

    Mirrors `check_error_events` above one level down the `level` column:
    every member of `instrumentation._WARNING_KINDS` -- issue #946's
    motivating kind plus roughly a dozen other pre-existing ones -- is
    emitted, classified, documented, and unit tested, but before this check
    nothing in the codebase ever read a warning-level row. This gives all of
    them their first reader at once, the same detection-to-delivery gap
    `check_error_events` closed for `level = 'error'`.

    Coverage is DERIVED, never a hardcoded `kind` list, for the identical
    reason as `check_error_events`: `level` is computed once and persisted
    per-row at write time by `instrumentation._classify_level` (checked
    against `_WARNING_KINDS` there), so filtering on the persisted
    `level = 'warning'` column here picks up every current and future
    warning kind without restating the kind list. This one check does import
    `charlie_work.event_kinds` (see the module-level import's own comment,
    and NOT `charlie_work.instrumentation` -- that module reaches `ci_fleet`
    at import time, which this stdlib-only script must never depend on) --
    but only for `EXPECTED_OPERATIONAL_KINDS`, the presentation bucketing
    below, which is a distinct question from coverage.

    Deliberately different from `check_error_events` in exactly one place:
    a new warning-level event is reported via `report.warn`, not
    `report.anom`. Several `_WARNING_KINDS` members are normal-operation
    events, not faults -- and a deliberately paused fleet with a non-empty
    backlog is not a crash either (see `EXPECTED_OPERATIONAL_KINDS` for
    exactly which kinds). Flipping the heartbeat to failure on every one of
    those would make this check permanently red and get ignored within a
    day; visibility is the goal, not a new alarm. The db-availability guards
    below stay `report.anom`, matching `check_error_events`: an unreadable
    events.db means this check cannot vouch for the repo at all, which is a
    genuine anomaly independent of whether any warning fired.

    See `check_error_events`'s docstring for the missing-db-is-an-anomaly
    rationale and the ISO-vs-SQLite string-comparison trap this avoids by
    comparing `ts` in Python against `baseline`, never in SQL.

    Issue #1271: the kinds in `EXPECTED_OPERATIONAL_KINDS` routinely
    dominate warning volume (a live 7-day window measured them as the
    majority of 676 total warnings) and drowned the rare genuine warning
    kinds in a flat listing. New rows whose `kind` is a member are bucketed
    into a one-line summarized count instead of the detailed listing; every
    other kind keeps the original flat `kind@ts` format unchanged. Both are
    still reported via `report.warn`, never `report.anom` -- bucketing
    changes presentation, not severity. Kind counts within the summary are
    ordered by sorted kind name (never dict/insertion order) so two runs
    over the same fixture produce byte-identical report lines.
    """
    check = f"warning-events {repo.slug}"
    db_path = repo.state_dir / "events.db"
    if not db_path.exists():
        report.anom(check, f"cannot check for warnings: no events.db at {db_path}")
        return

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        report.anom(check, f"cannot check for warnings: events.db unreadable: {exc}")
        return

    try:
        try:
            table_row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            if table_row is None:
                report.anom(check, "cannot check for warnings: events.db has no events table")
                return
            rows = conn.execute("SELECT ts, kind FROM events WHERE level = 'warning'").fetchall()
        except sqlite3.Error as exc:
            report.anom(check, f"cannot check for warnings: events.db unreadable: {exc}")
            return
    finally:
        conn.close()

    new_warnings_detail: list[str] = []
    expected_operational_counts: dict[str, int] = {}
    for ts, kind in rows:
        ts_dt = parse_iso(ts)
        # An unparseable ts fails toward visibility (reported), not silence.
        if ts_dt is None or ts_dt > baseline:
            if kind in EXPECTED_OPERATIONAL_KINDS:
                expected_operational_counts[kind] = expected_operational_counts.get(kind, 0) + 1
            else:
                new_warnings_detail.append(f"{kind}@{ts}")

    total_new = len(new_warnings_detail) + sum(expected_operational_counts.values())
    facts = f"warning_rows={len(rows)} new_since_last_beat={total_new}"

    if new_warnings_detail:
        report.warn(
            check, f"new warning-level event(s) since last beat: {new_warnings_detail} ({facts})"
        )
    if expected_operational_counts:
        # Sorted by kind name -- never dict/insertion order -- for
        # deterministic, byte-identical output across repeated runs.
        counts_str = ", ".join(
            f"{kind}={expected_operational_counts[kind]}"
            for kind in sorted(expected_operational_counts)
        )
        report.warn(
            check,
            f"{sum(expected_operational_counts.values())} routine operational warnings "
            f"({counts_str}) ({facts})",
        )
    if not new_warnings_detail and not expected_operational_counts:
        report.ok(check, facts)


def check_infra_blocked_events(report: Report, repo: RepoInfo, baseline: datetime) -> None:
    """Surface ``check_infra_blocked`` events and their persisted escalation
    (issue #1383, AC4).

    ``check_infra_blocked`` is a warning-level event emitted per affected PR
    when a required check fails due to a fleet-wide infrastructure condition
    (Actions budget/runner outage) rather than the PR's code. The
    operator-facing ``infra_blocked_escalated`` error event is emitted at
    most once per configured window when the condition persists across N
    passes. Both kinds already appear in the generic
    ``check_warning_events`` / ``check_error_events`` listings, but those
    are flat ``kind@ts`` lines with no correlation to the affected PRs or
    the persistence state. This dedicated check gives the operator a
    structured view: how many PRs are currently infra-blocked, which
    checks, and whether the persistence escalation has fired.

    Same db-availability posture as ``check_error_events``: a missing or
    unreadable events.db is an anomaly (this check cannot vouch for a repo
    it cannot read), not a silent OK. Timestamps are compared in Python
    against ``baseline`` for the same ISO-vs-SQLite reason documented on
    ``check_loop_pass_freshness``.
    """
    check = f"infra-blocked-events {repo.slug}"
    db_path = repo.state_dir / "events.db"
    if not db_path.exists():
        report.anom(check, f"cannot check: no events.db at {db_path}")
        return

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        report.anom(check, f"cannot check: events.db unreadable: {exc}")
        return

    try:
        try:
            table_row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            if table_row is None:
                report.anom(check, "cannot check: events.db has no events table")
                return
            blocked_rows = conn.execute(
                "SELECT ts, kind FROM events WHERE kind = ?",
                ("check_infra_blocked",),
            ).fetchall()
            escalated_rows = conn.execute(
                "SELECT ts FROM events WHERE kind = ?",
                ("infra_blocked_escalated",),
            ).fetchall()
        except sqlite3.Error as exc:
            report.anom(check, f"cannot check: events.db unreadable: {exc}")
            return
    finally:
        conn.close()

    new_blocked: list[str] = []
    for ts, _kind in blocked_rows:
        ts_dt = parse_iso(ts)
        if ts_dt is None or ts_dt > baseline:
            new_blocked.append(ts)

    new_escalated: list[str] = []
    for (ts,) in escalated_rows:
        ts_dt = parse_iso(ts)
        if ts_dt is None or ts_dt > baseline:
            new_escalated.append(ts)

    facts = f"blocked_rows={len(blocked_rows)} escalated_rows={len(escalated_rows)}"
    if new_escalated:
        report.anom(
            check,
            f"infra_blocked_escalated since last beat: {new_escalated} ({facts})",
        )
    elif new_blocked:
        report.warn(
            check,
            f"check_infra_blocked since last beat: {len(new_blocked)} event(s) ({facts})",
        )
    else:
        report.ok(check, facts)


def check_draft_pr_blocked_events(report: Report, repo: RepoInfo, baseline: datetime) -> None:
    """Surface ``draft_pr_blocked`` events periodically (issue #1366).

    A PR parked as draft-only-blocked (``workflow.py``'s ``failures_changed``
    arm) emits one ``draft_pr_blocked`` warning event per park. The event is
    the durable record that the park happened; this check is its first
    automated reader, surfacing new parks since the last beat so an operator
    sees draft-blocked PRs alongside the other stuck-state detectors in the
    same heartbeat output rather than only via a manual
    ``query_events(kind="draft_pr_blocked")`` grep.

    Deliberately a ``report.warn`` (passive, periodic surfacing), not
    ``report.anom``: a draft-blocked PR is a state to surface periodically,
    not a fleet emergency -- the operator comment on issue #1366 placed this
    kind on the periodic-surfacing path rather than the real-alert path
    reserved for ``venv_editable_anchor_violation`` (see
    ``check_supervisor_venv_refusal``). The db-availability guards stay
    ``report.anom``, matching ``check_infra_blocked_events``: an unreadable
    events.db means this check cannot vouch for the repo at all, which is a
    genuine anomaly independent of whether any park fired.

    Same db-availability posture and ISO-vs-SQLite comparison convention as
    ``check_infra_blocked_events``: timestamps are compared in Python against
    ``baseline`` (never in SQL) so an unparseable ``ts`` fails toward
    visibility, not silence.
    """
    check = f"draft-pr-blocked-events {repo.slug}"
    db_path = repo.state_dir / "events.db"
    if not db_path.exists():
        report.anom(check, f"cannot check: no events.db at {db_path}")
        return

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        report.anom(check, f"cannot check: events.db unreadable: {exc}")
        return

    try:
        try:
            table_row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            if table_row is None:
                report.anom(check, "cannot check: events.db has no events table")
                return
            rows = conn.execute(
                "SELECT ts FROM events WHERE kind = ?",
                ("draft_pr_blocked",),
            ).fetchall()
        except sqlite3.Error as exc:
            report.anom(check, f"cannot check: events.db unreadable: {exc}")
            return
    finally:
        conn.close()

    new_blocked: list[str] = []
    for (ts,) in rows:
        ts_dt = parse_iso(ts)
        if ts_dt is None or ts_dt > baseline:
            new_blocked.append(ts)

    facts = f"blocked_rows={len(rows)} new_since_last_beat={len(new_blocked)}"
    if new_blocked:
        report.warn(
            check,
            f"draft_pr_blocked since last beat: {len(new_blocked)} event(s) ({facts})",
        )
    else:
        report.ok(check, facts)


def check_ci_headroom_unavailable(report: Report, repo: RepoInfo, baseline: datetime) -> None:
    """Surface ``ci_headroom_unavailable`` events periodically (issue #1770).

    ``ci_headroom_available()`` (``charlie_work/ci_headroom.py``) records
    this warning-level event whenever it cannot trust the freshest
    ``runner_allocation`` reading for a repo that has opted into the
    CI-headroom clamp (``dispatch.ci_capacity_headroom_ratio > 0``) -- the
    emission site's own comment on the event's level registration
    (``instrumentation.py``) says "a repeating burst here means that channel
    itself needs attention", but before this check nothing read the kind
    back: it had only a test-only consumer (review finding 6), the
    ``signal-without-consumer`` shape this codebase's review history flags
    as its most recurrent defect class. Mirrors
    ``check_draft_pr_blocked_events`` exactly -- same db-availability
    posture, same periodic ``report.warn`` (not ``report.anom``: a fail-open
    reading is a diagnosability gap, not by itself a fleet emergency) --
    since the emitter (issue #1770 review finding 2) is itself now
    edge-triggered/rate-limited, so a burst here is meaningful rather than
    an artifact of unconditional per-pass writes.

    Same ISO-vs-SQLite comparison convention as ``check_draft_pr_blocked_
    events``: timestamps are compared in Python against ``baseline``, never
    in SQL, so an unparseable ``ts`` fails toward visibility, not silence.
    """
    check = f"ci-headroom-unavailable-events {repo.slug}"
    db_path = repo.state_dir / "events.db"
    if not db_path.exists():
        report.anom(check, f"cannot check: no events.db at {db_path}")
        return

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        report.anom(check, f"cannot check: events.db unreadable: {exc}")
        return

    try:
        try:
            table_row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            if table_row is None:
                report.anom(check, "cannot check: events.db has no events table")
                return
            rows = conn.execute(
                "SELECT ts, payload FROM events WHERE kind = ?",
                ("ci_headroom_unavailable",),
            ).fetchall()
        except sqlite3.Error as exc:
            report.anom(check, f"cannot check: events.db unreadable: {exc}")
            return
    finally:
        conn.close()

    new_unavailable: list[str] = []
    reasons: dict[str, int] = {}
    for ts, payload_json in rows:
        ts_dt = parse_iso(ts)
        if ts_dt is not None and ts_dt <= baseline:
            continue
        new_unavailable.append(ts)
        reason = "unknown"
        try:
            parsed_payload = json.loads(payload_json)
            if isinstance(parsed_payload, dict):
                reason = str(parsed_payload.get("reason", "unknown"))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        reasons[reason] = reasons.get(reason, 0) + 1

    facts = f"unavailable_rows={len(rows)} new_since_last_beat={len(new_unavailable)}"
    if new_unavailable:
        report.warn(
            check,
            f"ci_headroom_unavailable since last beat: {len(new_unavailable)} event(s), "
            f"reasons={reasons} ({facts})",
        )
    else:
        report.ok(check, facts)
