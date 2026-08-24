"""Deterministic fleet-heartbeat check.

Replaces LLM-driven heartbeat judgment (which misread log freshness from file
HEAD lines, parsed a PR number as a count, and false-alarmed a merge-queue
stall) with plain data collection + threshold comparison. An LLM only ever
sees this script's stdout.

Run via:
    cd /c/Users/senki/repos/charlie-work
    env -u VIRTUAL_ENV uv run --active --no-sync python scripts/heartbeat_check.py

Output contract (stdout): one line per check, either
    OK <check>: <compact facts>
    ANOMALY <check>: <what tripped, with numbers and the threshold>
    SUPPRESSED <check>: [#<issue> until <date>] <what tripped> (issue #1361 --
        a registry-matched, non-expired anomaly: visible but does not flip
        the exit code)
Exit code 0 if no anomalies (including suppressed ones), 1 if any anomaly is
unsuppressed or a suppression itself has expired.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psutil
import yaml

# Issue #1271: the single declared source of truth for which warning-level
# kinds are normal-operation signals (see the frozenset's own docstring).
# Imported, never re-declared or hardcoded here, so check_warning_events'
# bucketing stays correct as that set changes without this file needing an
# edit. Unlike the rest of this module (see the stale-open-issue-mention
# section docstring below for why it otherwise avoids importing
# charlie_work), this one string-literal set is worth importing directly:
# duplicating it here would be exactly the kind of hardcoded list that
# drifts from the registry it is supposed to mirror.
#
# Imported from `charlie_work.event_kinds` specifically, NEVER from
# `charlie_work.instrumentation` -- that module imports `ci_fleet` at
# module load, and this script must stay importable even when `ci_fleet`
# isn't (that's the entire reason it is stdlib-only; see scripts/README.md).
# `event_kinds` is a genuine leaf: no charlie_work or ci_fleet imports of
# its own, so importing it can never reach ci_fleet transitively.
#
# Guarded with try/except, not a bare import: this script is routinely run
# via `uv run --active`, which resolves against whatever venv happens to be
# active rather than this project's own -- a documented failure mode in this
# fleet (see the `uv-worktree-virtualenv-shadowing` project memory) where
# `charlie_work` itself is not importable at all, not merely `ci_fleet`.
# `scripts/README.md`'s invariant is "a broken package install can never
# break the check that would detect it" -- unconditional, not scoped to
# ci_fleet -- so a missing `charlie_work` degrades this script to the exact
# pre-#1271 behavior (no bucketing; every warning kind goes to the flat
# detailed list) instead of crashing on import.
try:
    from charlie_work.event_kinds import EXPECTED_OPERATIONAL_KINDS
except ImportError:
    EXPECTED_OPERATIONAL_KINDS: frozenset[str] = frozenset()

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

GH_TIMEOUT_SECONDS = 30
ISSUE_LIST_LIMIT = 200
MERGED_PR_LOOKBACK_LIMIT = 5

QUEUED_STALE_MINUTES = 20

# armable-backlog (2026-08-23): "plenty to work on" = one full wave of armed,
# unclaimed issues (dispatch.max_concurrent_sessions); fallback when the cap
# is unreadable. Gating labels mark an open issue as *triaged but deliberately
# not armed*, so it leaves the un-triaged "armable" pool; keep this set in step
# with the label taxonomy both repos share (`needs-design`, `human-action`,
# `blocked`) plus GitHub's default terminal labels.
ARMABLE_RUNWAY_FLOOR_DEFAULT = 3
ARMED_LABEL = "automated-ready"
ARMABLE_GATING_LABELS: frozenset[str] = frozenset(
    {"blocked", "needs-design", "human-action", "question", "wontfix", "duplicate", "invalid"}
)
ARMABLE_PREVIEW_LIMIT = 8
REVIEW_CLAIM_STALE_MINUTES = 45
LOG_FRESHNESS_STALE_MINUTES = 30
# Measured production cadence (charlie-work `loop_started` gaps, last 39
# intervals, 2026-07-31): min=5.5m median=10.4m p90=20.2m max=53.9m.
# `loop_started` is logged per repo (workflow.py's `_loop_impl`, into that
# repo's own events.db), and the supervisor processes repos sequentially in
# one pass, so a single repo's gap is gated by how long its SIBLING repos
# take, not by supervisor health -- job-cannon's reconcile alone walks
# ~690 issues / ~877 PRs and can push charlie-work's gap past 50 minutes on
# a perfectly healthy fleet.
#
# Set to 90: comfortably above the observed healthy maximum (53.9m) so a
# slow-but-alive fleet cannot false-alarm (at 30m this fired on 1 of 39
# healthy intervals, ~3-4 false alarms/day). This deliberately means it
# will NOT catch a sub-90-minute stall -- the outage that motivated this
# check (issues #851/#854) was only ~45 minutes, shorter than charlie-work's
# own legitimate worst-case gap, so no per-repo threshold can separate a
# real stall of that length from a healthy-but-slow pass. This check is a
# coarse backstop for prolonged, total fleet death (fresh log, zero passes,
# for well over an hour) -- not a detector for the #851/#854 class. That
# class is caught by PR #865 (issue #855), which escalates consecutive
# zero-repo-pass supervisor cycles: edge-triggered on the actual failure
# mode, needs no cadence-based threshold, and can't be confused with a
# merely slow loop. Do not lower this value to "catch" that outage faster --
# it will just reintroduce the false-alarm noise measured above; extend
# PR #865's check instead.
LOOP_PASS_STALE_MINUTES = 90
MERGEQUEUE_STALL_BEATS = 2
GRAPHQL_RATE_LIMIT_MIN_REMAINING = 500
DISPATCH_THROTTLE_MAX_MINUTES = 30
MIN_BEAT_INTERVAL_MINUTES = 10
CHARLIE_STATUS_TIMEOUT_SECONDS = 60

# in-progress-stale worktree mtime threshold (issue #1379). The events-based
# check flags an issue when its GitHub updatedAt hasn't moved across 2 beats,
# but long-running workers routinely emit no events for 40-60+ minutes while
# actively working (events fire at dispatch/PR/exit boundaries, not during
# implementation). Before flagging, the check also looks at the newest file
# mtime under the issue's worker worktree: a healthy worker's worktree shows
# file activity (edits, pytest cache, compiled bytecode) far more frequently
# than events fire.
#
# The threshold separates the two cases observed on 2026-08-21: the false
# positives (#1372) had worktree mtimes 4 seconds to 17 minutes old (alive),
# while the true positive (#1744) had a worktree mtime ~48 minutes old (dead).
# 30 minutes sits between them with margin on both sides (~13m below the dead
# case, ~13m above the oldest alive case). Do not lower this without revisiting
# those data points -- too-low reintroduces the alert fatigue the issue was
# filed to fix; too-high lets a genuinely dead worker run longer before
# surfacing.
IN_PROGRESS_STALE_WORKTREE_MINUTES = 30

# Bound on files scanned per worktree in _newest_worktree_mtime (issue #1379
# acceptance: "scan cost bounded"). The scan short-circuits as soon as a file
# newer than the stale window is found, so this cap only bounds the worst case
# (a genuinely dead worktree with many stale files). 5000 comfortably covers a
# typical worktree's non-.git file count; a worktree with >5000 files all older
# than the window is overwhelmingly likely to be dead, and the cap fails toward
# flagging (conservative), never toward green.
_WORKTREE_MTIME_SCAN_FILE_CAP = 5000

# Supervisor heartbeat freshness (issue #627). The supervisor writes
# supervisor-heartbeat.json at the top of every loop iteration. On a live
# supervisor ``last_beat_at`` is at most one ``max_pass_runtime_seconds``
# (plus the post-pass sleep) old; a stale
# heartbeat means the supervisor is down — killed (no ``exited_at``) or
# cleanly stopped but not restarted by the watchdog (``exited_at`` set).
# The stale threshold is a multiplier on ``max_pass_runtime_seconds``
# recorded in the heartbeat itself, so it derives from the config knob that
# actually bounds a single pass's wall-clock runtime. The multiplier covers
# a full pass duration plus the post-pass cooldown/poll sleep with margin.
# Older heartbeats that lack ``max_pass_runtime_seconds`` fall back to
# ``full_pass_interval_seconds`` (the pre-fix behavior) for transition safety.
SUPERVISOR_HEARTBEAT_FILENAME = "supervisor-heartbeat.json"
SUPERVISOR_HEARTBEAT_STALE_MULTIPLIER = 2
SUPERVISOR_HEARTBEAT_DEFAULT_PASS_TIMEOUT_SECONDS = 1800

# Disk-free thresholds (issue #1359): the 2026-08-19 outage drained C: to 0
# bytes free over ~3.5 days at ~4 MB/s while every fleet pass failed with
# `OSError: [Errno 28] No space left on device` and state.json went stale in
# both lanes -- with zero early warning, because this script had no disk-space
# check. A coarse threshold would have surfaced this days before ENOSPC.
#
# Defaults: WARN below 100 GB or 5% free; ANOMALY below 20 GB or 1% free. The
# ANOMALY flips the exit code exactly like other anomalies; the WARN goes
# through `report.warn` (routine-operational, never flips the exit code) so a
# low-but-not-critical volume surfaces without making the check permanently
# red. This script has no per-check config file -- thresholds are constants
# here, matching every other threshold in this block (LOOP_PASS_STALE_MINUTES,
# SUPERVISOR_HEARTBEAT_STALE_MULTIPLIER, ...); override by editing this file.
DISK_FREE_WARN_BYTES = 100 * 1024**3  # 100 GB
DISK_FREE_WARN_RATIO = 0.05  # 5%
DISK_FREE_ANOMALY_BYTES = 20 * 1024**3  # 20 GB
DISK_FREE_ANOMALY_RATIO = 0.01  # 1%

# check_stale_open_issue_mentions (issue #902): two bulk API sources plus one
# free local one, per the issue's "API economy matters" constraint -- never a
# gh call per candidate issue. STALE_MENTION_PR_LOOKBACK_LIMIT bounds the
# `gh pr list --state merged` call; 300 comfortably covers the "last 60
# merged PRs" sample #902 was scoped from with headroom for a slower week.
# STALE_MENTION_COMMIT_LOOKBACK bounds the local `git log` scan (issue #866's
# reproduction: PR #864's squashed merge commit sits 19 commits back from
# HEAD at filing time) -- purely a perf/output cap, not an API cost, since
# `git log` never touches the network.
STALE_MENTION_PR_LOOKBACK_LIMIT = 300
STALE_MENTION_COMMIT_LOOKBACK = 500
STALE_MENTION_REPORT_CAP = 20

DELTA_SKIP_SUFFIX = " (delta skipped: last beat <10m ago)"

FLEET_TASK_NAME = "charlie-fleet-pass"
# schtasks "Last Result" codes that mean "not actually a failure":
# 0 = success, 267009 = task currently running, 267011 = task has not yet run,
# -2147020576 = 0x800710E0, "the operator or administrator has refused the
# request" -- what Task Scheduler records when a repetition trigger fires while
# a previous instance is still running and the task is configured
# MultipleInstances: IgnoreNew. Fleet passes routinely exceed the 5-minute
# repetition interval, so this is the documented, intended behaviour of that
# setting rather than a failure (issue #587).
SCHTASKS_OK_RESULT_CODES = {0, 267009, 267011, -2147020576}

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class RepoInfo:
    slug: str
    repo_root: Path
    state_dir: Path
    config_path: Path


@dataclass(frozen=True)
class SuppressionEntry:
    """One entry from `scripts/heartbeat-suppressions.yaml` (issue #1361).

    `check` is the *base* check name as emitted, with no repo suffix (e.g.
    ``"stale-open-issue-mentions"``, never ``"stale-open-issue-mentions
    Senkichi/charlie-work"``). Per-repo checks build their emitted check
    string as ``f"{base} {repo.slug}"`` (the convention already used by every
    per-repo check in this file); `Report._match` reconstructs that same
    convention to test a candidate entry against an emitted check string, so
    this dataclass itself never needs to know which checks are per-repo.
    """

    check: str
    issue: int
    expires: str  # ISO date (YYYY-MM-DD), UTC
    repo: str | None = None
    match: str = ""
    note: str = ""

    def is_expired(self, now: datetime) -> bool:
        """An entry expiring today counts as expired (issue #1361 AC7)."""
        try:
            expires_date = datetime.strptime(self.expires, "%Y-%m-%d").date()
        except ValueError:
            return True  # malformed dates are caught at load time; fail closed regardless
        return now.date() >= expires_date


SUPPRESSION_REGISTRY_FILENAME = "heartbeat-suppressions.yaml"


def suppression_registry_path() -> Path:
    """Resolve the suppression registry path, next to this script by default.

    ``CHARLIE_WORK_HEARTBEAT_SUPPRESSIONS`` overrides it -- tests must always
    set this (or pass an explicit path to `load_suppression_registry`
    directly) rather than relying on the default, since after this file ships
    the default path resolves to the real seeded registry.
    """
    override = os.environ.get("CHARLIE_WORK_HEARTBEAT_SUPPRESSIONS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / SUPPRESSION_REGISTRY_FILENAME


def _is_iso_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def load_suppression_registry(path: Path) -> tuple[list[SuppressionEntry], str | None]:
    """Load and validate the suppression registry.

    Returns ``(entries, error)``. A missing file is not an error: it means
    zero suppressions, exactly today's (pre-#1361) behavior. A malformed
    file or entry returns ``([], error)`` -- fail closed: the entries list
    comes back empty so a bad edit can never silently suppress anything
    (only ever ADD an anomaly, this error, plus every previously-suppressed
    condition resurfacing as a raw, unsuppressed ANOMALY -- the safe
    direction to fail in).
    """
    if not path.exists():
        return [], None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], f"{path}: unreadable: {exc}"
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return [], f"{path}: YAML parse error: {exc}"
    if data is None:
        return [], None
    if not isinstance(data, list):
        return [], f"{path}: expected a YAML list at the top level, got {type(data).__name__}"

    entries: list[SuppressionEntry] = []
    for idx, raw_entry in enumerate(data):
        if not isinstance(raw_entry, dict):
            return [], f"{path}: entry {idx} is not a mapping"
        check = raw_entry.get("check")
        if not isinstance(check, str) or not check:
            return [], f"{path}: entry {idx} missing required string field 'check'"
        issue = raw_entry.get("issue")
        if not isinstance(issue, int) or isinstance(issue, bool):
            return [], f"{path}: entry {idx} missing required integer field 'issue'"
        expires = raw_entry.get("expires")
        if not isinstance(expires, str) or not _is_iso_date(expires):
            return [], f"{path}: entry {idx} missing/invalid required ISO date field 'expires'"
        repo = raw_entry.get("repo")
        if repo is not None and not isinstance(repo, str):
            return [], f"{path}: entry {idx} field 'repo' must be a string"
        match = raw_entry.get("match", "")
        if not isinstance(match, str):
            return [], f"{path}: entry {idx} field 'match' must be a string"
        note = raw_entry.get("note", "")
        if not isinstance(note, str):
            return [], f"{path}: entry {idx} field 'note' must be a string"
        entries.append(
            SuppressionEntry(
                check=check, issue=issue, expires=expires, repo=repo, match=match, note=note
            )
        )
    return entries, None


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)
    anomaly: bool = False
    suppressions: list[SuppressionEntry] = field(default_factory=list)
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _matched_indices: set[int] = field(default_factory=set, repr=False, compare=False)

    def ok(self, check: str, facts: str) -> None:
        self.lines.append(f"OK {check}: {facts}")

    def _find_suppression(self, check: str, detail: str) -> tuple[int, SuppressionEntry] | None:
        """Match an emitted check string against the registry.

        A registry entry's `check` is the base name; per-repo checks emit
        ``f"{base} {repo}"`` (see `SuppressionEntry`'s docstring), so an
        entry with a `repo` matches only that exact combined string, and an
        entry with no `repo` matches the base name as a whole leading
        component (never a bare substring -- ``"dispatch"`` must not match
        ``"dispatch-coverage ..."``). `match`, when set, must additionally
        appear as a substring of `detail` -- never of the whole line and
        never a hash of it, since detail contains counts that change every
        beat (issue #1361 constraint).
        """
        for idx, entry in enumerate(self.suppressions):
            if entry.repo is not None:
                if check != f"{entry.check} {entry.repo}":
                    continue
            else:
                if check != entry.check and not check.startswith(f"{entry.check} "):
                    continue
            if entry.match and entry.match not in detail:
                continue
            return idx, entry
        return None

    def anom(self, check: str, detail: str) -> None:
        found = self._find_suppression(check, detail)
        if found is not None:
            idx, entry = found
            self._matched_indices.add(idx)
            if entry.is_expired(self.now):
                self.lines.append(
                    f"ANOMALY {check}: [suppression #{entry.issue} EXPIRED {entry.expires}] {detail}"
                )
                self.anomaly = True
            else:
                self.lines.append(
                    f"SUPPRESSED {check}: [#{entry.issue} until {entry.expires}] {detail}"
                )
            return
        self.lines.append(f"ANOMALY {check}: {detail}")
        self.anomaly = True

    def warn(self, check: str, detail: str) -> None:
        """Surface a non-fatal finding.

        Unlike `anom`, this does not set `self.anomaly`, so it never flips
        `main()`'s exit code. Issue #946: warning-level events are worth
        surfacing but several existing `_WARNING_KINDS` members are
        normal-operation events, not faults (see `EXPECTED_OPERATIONAL_KINDS`,
        issue #1271, for exactly which ones) -- alarming on them would make
        this check permanently red and get ignored within a day.

        Never suppressed: issue #1361 deliberately scopes the suppression
        registry to ANOMALY lines only -- WARN already does not affect the
        exit code, so suppressing it would add complexity for no behavior
        change.
        """
        self.lines.append(f"WARN {check}: {detail}")

    def suppression_summary(self) -> str | None:
        """`OK suppressions: active=N expired=M unmatched=K`, or None.

        Returns None when the registry is empty (missing file or a malformed
        one, which loads as zero entries) -- AC5 says the summary appears
        "whenever the registry is non-empty", and with zero entries there is
        nothing to summarize. `active`/`expired` classify registry entries by
        their own expiry date, independent of whether anything matched this
        run; `unmatched` is the orthogonal count of entries that matched zero
        `anom()` calls this run -- issue #1361's signal that a condition has
        cleared and the entry is a candidate for deletion (surfaced, never
        auto-deleted).
        """
        if not self.suppressions:
            return None
        active = sum(1 for e in self.suppressions if not e.is_expired(self.now))
        expired = len(self.suppressions) - active
        unmatched = sum(
            1 for idx in range(len(self.suppressions)) if idx not in self._matched_indices
        )
        return f"active={active} expired={expired} unmatched={unmatched}"


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------


def fleet_dir() -> Path:
    """Mirror charlie_work.fleet_paths.fleet_dir() without importing the package."""
    override = os.environ.get("CHARLIE_WORK_FLEET_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "charlie-work"


def state_file() -> Path:
    """Resolve the heartbeat state file path.

    Derives from :func:`fleet_dir` (so it follows the same platform-aware base
    and ``CHARLIE_WORK_FLEET_DIR`` override) unless an explicit
    ``CHARLIE_WORK_HEARTBEAT_STATE`` env var points at a specific file.  Never
    hard-coded to a developer-machine path.
    """
    override = os.environ.get("CHARLIE_WORK_HEARTBEAT_STATE")
    if override:
        return Path(override)
    return fleet_dir() / "heartbeat-state.json"


def load_repos() -> tuple[list[RepoInfo], str | None]:
    """Load registered repos from fleet.json. Returns (repos, error)."""
    fleet_json = fleet_dir() / "fleet.json"
    if not fleet_json.exists():
        return [], f"fleet.json not found at {fleet_json}"
    try:
        data = json.loads(fleet_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], f"fleet.json unreadable: {exc}"
    repos: list[RepoInfo] = []
    for slug, entry in data.get("repos", {}).items():
        try:
            repos.append(
                RepoInfo(
                    slug=slug,
                    repo_root=Path(entry["repo_root"]),
                    state_dir=Path(entry["state_dir"]),
                    config_path=Path(entry.get("config_path", "")),
                )
            )
        except KeyError:
            continue
    return repos, None


def run_gh_json(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
    """Run `gh <args>` and parse stdout as JSON. Never raises."""
    try:
        proc = subprocess.run(
            ["gh", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GH_TIMEOUT_SECONDS,
            creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, None, f"gh {' '.join(args)} failed to run: {exc}"
    if proc.returncode != 0:
        stderr = proc.stderr.strip().replace("\n", " ")[:200]
        return False, None, f"gh {' '.join(args)} exited {proc.returncode}: {stderr}"
    try:
        return True, json.loads(proc.stdout), ""
    except json.JSONDecodeError as exc:
        return False, None, f"gh {' '.join(args)} produced invalid JSON: {exc}"


def get_blocked_issue_numbers(any_repo_root: Path) -> tuple[dict[str, set[int]], str]:
    """Fetch per-repo blocked-issue numbers via `charlie fleet status --json`.

    Reuses the orchestrator's own dependency-gate logic (`_filter_blocked_issues`)
    as the single point of enforcement, rather than reimplementing blocker
    parsing (body-text + GitHub native dependencies) in this script. Returns
    ({} , error_message) on any failure so callers can degrade gracefully.
    """
    try:
        proc = subprocess.run(
            ["charlie", "fleet", "status", "--json"],
            cwd=str(any_repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CHARLIE_STATUS_TIMEOUT_SECONDS,
            creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {}, f"charlie fleet status --json failed to run: {exc}"
    if proc.returncode != 0:
        stderr = proc.stderr.strip().replace("\n", " ")[:200]
        return {}, f"charlie fleet status --json exited {proc.returncode}: {stderr}"
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {}, f"charlie fleet status --json produced invalid JSON: {exc}"

    try:
        repos = payload["data"]["repos"]
        blocked_by_repo = {
            slug: {int(entry["issue"]) for entry in repo_data.get("blocked", [])}
            for slug, repo_data in repos.items()
        }
    except (KeyError, TypeError) as exc:
        return {}, f"charlie fleet status --json unexpected payload shape: {exc}"
    return blocked_by_repo, ""


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_state() -> dict[str, Any]:
    path = state_file()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict[str, Any]) -> None:
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------
# Worker worktree mtime signal (issue #1379)
# --------------------------------------------------------------------------
#
# These helpers are stdlib-only reimplementations of layout/worktree helpers
# this script cannot import (see scripts/README.md's "stdlib-only" invariant).
# They mirror:
#   - charlie_work.layout.worktrees_dir(state_root) -> state_root / "worktrees"
#   - charlie_work.worktree._slugify(branch)
#   - charlie_work.worktree.worktree_path_for_branch(root, branch, worktrees_dir)
# If those ever diverge, tests/test_heartbeat_check.py's worktree-mtime tests
# will catch it (they build the worktree dir the same way the orchestrator
# does, via the same slugify, so a slug mismatch surfaces as a missing dir).


def _slugify_branch(branch: str) -> str:
    """Mirror ``charlie_work.worktree._slugify`` (stdlib-only reimplementation).

    The production function lives in ``charlie_work.worktree``; this script
    cannot import it (stdlib-only invariant, scripts/README.md). The two must
    agree so the worktree path derived here matches the one the orchestrator
    created. ``tests/test_heartbeat_check.py`` exercises the same derivation
    against real branch names, so a drift surfaces as a missing-dir test
    failure rather than a silent false ANOMALY.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", branch).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:80].rstrip("-") or "worktree"


def _resolved_worktrees_dir(repo: RepoInfo) -> Path:
    """Resolve the worktrees root for ``repo``, honouring ``claude_code.worktrees_dir``.

    Mirrors ``charlie_work.paths.resolved_layout``'s worktrees resolution
    (issue #1379 review): ``claude_code.worktrees_dir`` is a sentinel-style
    override -- ``None``/empty means "derive from ``runtime.state_dir``"
    (``<state_dir>/worktrees``), a non-empty value is an explicit path
    (absolute returned as-is, relative joined to ``repo_root``). This script
    cannot import ``charlie_work.config``/``paths`` (stdlib-only invariant,
    scripts/README), so the resolution is reimplemented locally against the
    config dict ``load_orchestrator_config`` already returns -- the same
    reimplement-locally treatment ``fleet_dir`` and the stale-open-issue-mention
    primitives use. A broken/unreadable config degrades to the default
    (fail-toward-flagging: a missing worktree dir reads as ANOMALY, not OK).
    """
    default = repo.state_dir / "worktrees"
    config, _error = load_orchestrator_config(repo.config_path)
    raw = config.get("claude_code", {}).get("worktrees_dir")
    if not raw or not isinstance(raw, str):
        return default
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else repo.repo_root / candidate


def _worktree_path_for_branch(
    repo: RepoInfo, branch: str, worktrees_dir: Path | None = None
) -> Path:
    """Return the worktree dir for ``branch`` under ``repo``'s worktrees root.

    Mirrors ``charlie_work.worktree.worktree_path_for_branch``. The worktrees
    root defaults to ``_resolved_worktrees_dir(repo)`` (which honours
    ``claude_code.worktrees_dir``); pass ``worktrees_dir`` to override it
    once (e.g. a caller that resolves it once for many branches). ``repo.state_dir``
    is the state root (the directory holding ``state.json``, as registered in
    fleet.json).
    """
    root = worktrees_dir if worktrees_dir is not None else _resolved_worktrees_dir(repo)
    return root / _slugify_branch(branch)


def _load_state_issues(repo: RepoInfo) -> dict[str, Any]:
    """Load the ``issues`` map from ``repo``'s ``state.json``.

    Returns ``{}`` on any read/parse failure or missing file -- the caller
    (``check_in_progress_staleness``) degrades to events-only behavior when no
    ``branch_name`` is found, which is the correct (fail-toward-flagging)
    direction for a corrupt state file.
    """
    state_json = repo.state_dir / "state.json"
    if not state_json.exists():
        return {}
    try:
        data = json.loads(state_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    issues = data.get("issues")
    if not isinstance(issues, dict):
        return {}
    return issues


def _is_reparse_point(path: str) -> bool:
    """True if ``path`` is a reparse point (Windows junction or any symlink).

    ``os.path.islink`` does NOT detect Windows directory junctions (verified
    empirically on this host: ``islink`` returns ``False`` for a junction
    while the reparse-point file attribute is set), and ``os.walk`` with
    ``followlinks=False`` recurses straight through a junction regardless --
    so a ``dirs[:]`` filter built on ``islink`` alone lets the scan walk a
    ``.venv`` junction into a shared venv whose mtimes reflect *other*
    worktrees' test runs, not this worker's activity. That can mask a
    genuinely dead worker (issue #1379 review). This check is what keeps the
    scan out of such a junction: ``islink`` catches POSIX symlinks (and
    Windows symlinks), and the reparse-point attribute catches Windows
    junctions that ``islink`` misses.
    """
    if os.path.islink(path):
        return True
    if sys.platform == "win32":
        try:
            attrs = os.lstat(path).st_file_attributes
        except (OSError, AttributeError):
            return False
        return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return False


def _newest_worktree_mtime(
    worktree: Path,
    *,
    threshold: datetime,
    file_cap: int = _WORKTREE_MTIME_SCAN_FILE_CAP,
) -> datetime | None:
    """Newest file mtime under ``worktree``, excluding ``.git/``. Bounded scan.

    Returns ``None`` when the directory does not exist or contains no scannable
    files. Short-circuits as soon as a file at or after ``threshold`` is found
    (the caller only needs to know whether ANY file is fresher than the stale
    window), so the ``file_cap`` only bounds the worst case -- a dead worktree
    whose every file is older than the window. The cap fails toward flagging
    (returns the newest-so-far, which is stale), never toward green.

    Windows notes (issue #1379): uses the newest *file* mtime, not directory
    mtimes (dir mtimes do not propagate on Windows). Excludes ``.git/``
    (background git ops are not worker activity). Does not recurse into
    junctions/symlinks via ``_is_reparse_point`` (a ``.venv`` junction can
    point at a shared venv whose mtimes reflect other worktrees' test runs,
    not this worker's activity); ``os.path.islink`` alone is insufficient
    because it does not detect Windows junctions.
    """
    if not worktree.is_dir():
        return None
    newest: datetime | None = None
    scanned = 0
    for root, dirs, files in os.walk(worktree, followlinks=False):
        # Exclude .git (background git ops) and do not recurse into
        # junctions/symlinks. _is_reparse_point catches Windows junctions
        # that os.path.islink misses (issue #1379 review).
        dirs[:] = [d for d in dirs if d != ".git" and not _is_reparse_point(os.path.join(root, d))]
        for fname in files:
            scanned += 1
            if scanned > file_cap:
                return newest
            fpath = os.path.join(root, fname)
            try:
                mtime = os.path.getmtime(fpath)
            except OSError:
                continue
            mt = datetime.fromtimestamp(mtime, tz=timezone.utc)
            if newest is None or mt > newest:
                newest = mt
                if mt >= threshold:
                    return newest
    return newest


def load_orchestrator_config(config_path: Path) -> tuple[dict[str, Any], str | None]:
    """Load an orchestrator.config.yaml.

    Returns (config, error). error is None when the config is legitimately
    absent -- including an unset config_path, which load_repos() represents
    as Path("") (== Path("."), the "no config registered for this repo"
    sentinel -- deliberately not treated as cwd-relative) -- or when the file
    parses cleanly to a mapping. error is a message when config_path is set
    and points at something that exists but fails to read, isn't valid UTF-8,
    fails to parse as YAML, or parses to something other than a mapping.
    That "present but broken" case (issue #703) must not be treated the same
    as "absent" -- callers that need it surfaced use check_orchestrator_config
    below rather than reading the error here.
    """
    if config_path == Path(""):
        return {}, None
    try:
        if not config_path.exists():
            return {}, None
        raw = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {}, f"{config_path}: {exc}"
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        return {}, f"{config_path}: invalid YAML: {exc}"
    if not isinstance(data, dict):
        return {}, f"{config_path}: expected a mapping at top level, got {type(data).__name__}"
    return data, None


def get_mergequeue_label(config_path: Path) -> str | None:
    config, _error = load_orchestrator_config(config_path)
    return config.get("auto_merge", {}).get("mergequeue_label")


def get_dispatch_cap(config_path: Path) -> int | None:
    """Read dispatch.max_concurrent_sessions (per-repo concurrency cap), or None."""
    config, _error = load_orchestrator_config(config_path)
    cap = config.get("dispatch", {}).get("max_concurrent_sessions")
    return cap if isinstance(cap, int) else None


def check_orchestrator_config(report: Report, repo: RepoInfo) -> None:
    """Surface a present-but-broken orchestrator.config.yaml as a loud anomaly.

    get_mergequeue_label/get_dispatch_cap deliberately keep defaulting to
    None/absent on a broken config so the checks that consume them (dispatch
    cap, mergequeue label) degrade gracefully instead of raising. But that
    means the read error itself would otherwise never reach any
    operator-visible surface -- a signal without a consumer. This check is
    the single dedicated reader of that error: it turns "config exists but
    is corrupt/unreadable/malformed" into a loud report.anom (this script's
    stdout/exit-code contract is the only channel an operator actually sees),
    while "no config registered" and "config absent/valid" both stay quiet.
    """
    check = f"orchestrator-config {repo.slug}"
    if repo.config_path == Path(""):
        report.ok(check, "no config_path registered")
        return
    _config, error = load_orchestrator_config(repo.config_path)
    if error:
        report.anom(check, error)
    elif repo.config_path.exists():
        report.ok(check, f"{repo.config_path} readable")
    else:
        report.ok(check, f"{repo.config_path} not present (defaults apply)")


# --------------------------------------------------------------------------
# Stale-open-issue-mention scanning primitives (issue #902)
#
# charlie_work.github already has `issue_numbers_mentioned_by_pr` (a same-repo
# PR title/body scanner) and `iter_unnegated_closing_keyword_matches` (a
# negation-aware `#N` scanner used by `closing_keyword_gate.py`). This script
# deliberately does NOT import charlie_work.github, or any other
# ci_fleet-reachable charlie_work module (see the module docstring, `fleet_dir`,
# and the `charlie_work.event_kinds` import's own comment for the one narrow,
# stdlib-only exception), so the small negation/quote-stripping heuristics
# below are a minimal, self-contained reimplementation for this one check
# rather than a reuse of those functions. Two differences from `issue_numbers_mentioned_by_pr`
# are intentional, not drift:
#
# 1. Bare `#N` is matched, not just `issue #N` / closing-keyword `#N`. Issue
#    #866's only trace anywhere is its fix's commit message, "refs #866" --
#    neither "issue" nor a closing keyword precedes it, so the narrower
#    pattern used by dispatch's mention detector would miss the exact
#    reproduction this check exists to catch.
# 2. It also scans commit messages (via local `git log`), not just PR
#    title/body -- again, the #866 shape.
#
# Quote/negation suppression exists for the same reason #790 forced it onto
# `iter_unnegated_closing_keyword_matches`: a literal, quoted example like
# `"Fixes #649"` inside prose is not an intentional reference and must not
# be surfaced (issue #902 acceptance criterion 6).
# --------------------------------------------------------------------------

_ISSUE_REF_RE = re.compile(r"#(\d+)\b")
# The repo's own branch-naming convention for non-agent-dispatched work:
# `<type>/<issueNumber>-<slug>` (e.g. `fix/817-fleet-health-latch`). Matched
# separately from `_ISSUE_REF_RE` because there is no `#` in a branch name.
# Deliberately does NOT match `agent/issue-N-...` branches (digit is not
# immediately after the slash there) -- those are already covered by the
# normal branch-prefix binding path (`linked_issue_number`), so a miss here
# is not a gap, just redundant with machinery this check exists to backstop.
_BRANCH_ISSUE_NUMBER_RE = re.compile(r"^[A-Za-z][\w.]*/(\d+)(?=[-_/]|$)")
_FENCED_CODE_BLOCK_RE = re.compile(r"```.*?```", flags=re.DOTALL)
_NEGATION_WORDS = ("not", "never", "without", "cannot")
_NEGATION_CONTRACTION_SUFFIX = "n't"
_NEGATION_RE = re.compile(
    r"\b(?:" + "|".join(_NEGATION_WORDS) + r")\b|" + re.escape(_NEGATION_CONTRACTION_SUFFIX),
    flags=re.IGNORECASE,
)
_NEGATION_LOOKBEHIND_CHARS = 32
_QUOTE_CHARS = "\"'`"
_QUOTE_LOOKAROUND_CHARS = 40


def _has_preceding_negation(text: str, match_start: int) -> bool:
    """True if a negation word/contraction appears shortly before match_start.

    Same 32-char lookback window as `charlie_work.github._has_preceding_negation`
    (kept in sync by convention, not import -- see the section docstring above).
    """
    window_start = max(0, match_start - _NEGATION_LOOKBEHIND_CHARS)
    return bool(_NEGATION_RE.search(text, window_start, match_start))


def _is_quoted(text: str, match_start: int, match_end: int) -> bool:
    """True if the match sits inside a quoted span on the same line.

    A bare `#N` match (unlike a `<keyword> #N` closing-keyword match) can sit
    arbitrarily far from the quote character that wraps the whole phrase --
    #790's incident was the literal text `"Fixes #649"`, where the opening
    quote is 7 characters before the `#`. So this looks for a quote character
    (`"`, `'`, or a backtick) within `_QUOTE_LOOKAROUND_CHARS` before the
    match AND a matching quote character within the same distance after it,
    both bounded to the current line so a quote on an unrelated line can
    never suppress a real reference.
    """
    line_start = text.rfind("\n", 0, match_start) + 1
    line_end = text.find("\n", match_end)
    if line_end == -1:
        line_end = len(text)
    before = text[max(line_start, match_start - _QUOTE_LOOKAROUND_CHARS) : match_start]
    after = text[match_end : min(line_end, match_end + _QUOTE_LOOKAROUND_CHARS)]
    return any(q in before and q in after for q in _QUOTE_CHARS)


def _mentioned_issue_numbers(text: str) -> set[int]:
    """Return every bare `#N` reference in `text`, minus quoted/negated ones.

    Fenced code blocks are stripped first (a code sample containing the
    literal text `#123` is not a reference), mirroring
    `charlie_work.github`'s same defense for its own mention scanner.
    """
    stripped = _FENCED_CODE_BLOCK_RE.sub("", text)
    numbers: set[int] = set()
    for match in _ISSUE_REF_RE.finditer(stripped):
        if _has_preceding_negation(stripped, match.start()):
            continue
        if _is_quoted(stripped, match.start(), match.end()):
            continue
        numbers.add(int(match.group(1)))
    return numbers


def _branch_issue_number(branch: str) -> int | None:
    match = _BRANCH_ISSUE_NUMBER_RE.match(branch)
    return int(match.group(1)) if match else None


_GIT_LOG_RECORD_SEP = "\x1e"
_GIT_LOG_FIELD_SEP = "\x1f"


def get_merged_commit_messages(
    repo_root: Path, limit: int
) -> tuple[bool, list[tuple[str, str]], str]:
    """Return (ok, [(short_sha, full_message), ...], err) for the local checkout's history.

    Reads `git log` on the already-checked-out branch -- every commit on it is
    by definition already merged into that branch, so this needs no `--merged`
    flag and, crucially, no `gh` call at all (issue #902's "API economy"
    constraint: this is the free local source, not one of the two bulk `gh`
    calls). This is what catches issue #866's reproduction: its fix rode in
    as a commit inside PR #864, a PR *for a different issue*, so no scan of
    PR title/body/branch name (for #864 or any other PR) could ever find it --
    only a scan of #864's own commit messages can.
    """
    try:
        proc = subprocess.run(
            [
                "git",
                "log",
                f"-n{limit}",
                f"--pretty=format:%h{_GIT_LOG_FIELD_SEP}%B{_GIT_LOG_RECORD_SEP}",
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GH_TIMEOUT_SECONDS,
            creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, [], f"git log failed to run: {exc}"
    if proc.returncode != 0:
        stderr = proc.stderr.strip().replace("\n", " ")[:200]
        return False, [], f"git log exited {proc.returncode}: {stderr}"

    commits: list[tuple[str, str]] = []
    for record in proc.stdout.split(_GIT_LOG_RECORD_SEP):
        record = record.strip("\n")
        if not record:
            continue
        sha, _, message = record.partition(_GIT_LOG_FIELD_SEP)
        commits.append((sha, message))
    return True, commits, ""


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_dispatch_throttle(
    report: Report, repo: RepoInfo, *, now: datetime | None = None
) -> None:
    """Report the provider throttle cooldown (state.json's throttled_until).

    Being throttled is normal self-protection (OK-level), but the line must
    always print so a zero-dispatch beat is instantly explainable. Only an
    unusually long cooldown (beyond DISPATCH_THROTTLE_MAX_MINUTES) is an anomaly.

    ``now`` is the injectable clock (issue #828): defaults to
    ``datetime.now(timezone.utc)`` when not supplied, so production behavior
    is byte-identical. Callers running multiple checks in one pass (see
    ``main``) should sample ``now`` once and pass the same value to every
    check instead of letting each check independently race the wall clock.
    """
    check = f"dispatch-throttle {repo.slug}"
    state_json = repo.state_dir / "state.json"
    if not state_json.exists():
        report.ok(check, "none (no state.json)")
        return
    try:
        data = json.loads(state_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.anom(check, f"state.json unreadable: {exc}")
        return

    throttled_until_raw = data.get("throttled_until") if isinstance(data, dict) else None
    until = parse_iso(throttled_until_raw)
    resolved_now = now if now is not None else datetime.now(timezone.utc)
    if until is None or until <= resolved_now:
        report.ok(check, "none")
        return

    remaining_min = round((until - resolved_now).total_seconds() / 60)
    facts = f"throttled until {throttled_until_raw} ({remaining_min} min remaining)"
    if remaining_min > DISPATCH_THROTTLE_MAX_MINUTES:
        report.anom(
            check, f"cooldown exceeds threshold={DISPATCH_THROTTLE_MAX_MINUTES}m ({facts})"
        )
    else:
        report.ok(check, facts)


def check_dispatch_coverage(
    report: Report,
    repo: RepoInfo,
    prev_repo_state: dict[str, Any],
    new_repo_state: dict[str, Any],
    skip_delta: bool,
    blocked_numbers: set[int] | None,
    blocked_err: str,
    *,
    now: datetime | None = None,
) -> None:
    """``now`` is the injectable clock (issue #828); see ``check_dispatch_throttle``."""
    check = f"dispatch-coverage {repo.slug}"
    args = [
        "issue",
        "list",
        "-R",
        repo.slug,
        "--label",
        "automated-ready",
        "--state",
        "open",
        "--json",
        "number,labels,updatedAt",
        "--limit",
        str(ISSUE_LIST_LIMIT),
    ]
    ok, data, err = run_gh_json(args, repo.repo_root)
    if not ok:
        report.anom(check, err)
        return

    dispatchable: list[int] = []
    queued: list[tuple[int, datetime | None]] = []
    in_progress: list[tuple[int, str | None]] = []
    for issue in data:
        names = {label["name"] for label in issue.get("labels", [])}
        agent_labels = {n for n in names if n.startswith("agent:")}
        number = issue["number"]
        is_blocked = blocked_numbers is not None and number in blocked_numbers
        if not agent_labels and not is_blocked:
            dispatchable.append(number)
        if "agent:queued" in agent_labels:
            queued.append((number, parse_iso(issue.get("updatedAt"))))
        if "agent:in-progress" in agent_labels:
            in_progress.append((number, issue.get("updatedAt")))

    reasons: list[str] = []
    cap = get_dispatch_cap(repo.config_path) if repo.config_path else None

    drain_note: str | None = None
    if skip_delta:
        # Carry forward the last real beat's snapshot untouched so the next
        # real-interval beat still compares against it, not against this
        # manual/test sample.
        new_repo_state["dispatchable_issues"] = prev_repo_state.get("dispatchable_issues", [])
    else:
        prev_dispatchable = set(prev_repo_state.get("dispatchable_issues", []))
        cur_dispatchable = set(dispatchable)
        persisting = sorted(prev_dispatchable & cur_dispatchable)
        if persisting:
            if cap is not None and len(in_progress) >= cap:
                # Backlog exceeding the drain rate while every dispatch slot is
                # occupied is designed behavior, not a dispatch failure.
                drain_note = (
                    f"backlog={len(persisting)} draining at cap "
                    f"(in_progress={len(in_progress)}/cap={cap})"
                )
            else:
                reasons.append(
                    f"issue(s) {persisting} dispatchable across 2 consecutive beats "
                    "(threshold: must clear within 1 beat)"
                )
        new_repo_state["dispatchable_issues"] = sorted(cur_dispatchable)

    resolved_now = now if now is not None else datetime.now(timezone.utc)
    stale_queued = []
    for number, updated in queued:
        if updated is None:
            continue
        age_min = (resolved_now - updated).total_seconds() / 60
        if age_min > QUEUED_STALE_MINUTES:
            stale_queued.append((number, round(age_min)))
    if stale_queued:
        reasons.append(
            f"agent:queued stuck {stale_queued} minutes (threshold={QUEUED_STALE_MINUTES}m)"
        )

    facts = f"dispatchable={len(dispatchable)} queued={len(queued)} in_progress={len(in_progress)}"
    if cap is not None:
        facts += f" cap={cap}"
    if drain_note:
        facts = f"{drain_note}; {facts}"
    if skip_delta:
        facts += DELTA_SKIP_SUFFIX

    if reasons:
        if blocked_err:
            # The blocked set is unavailable, so a "dispatchable" issue may
            # actually be blocked. Surface the degraded lookup as a caveat so
            # the anomaly is not read as a confirmed dispatch failure.
            detail = (
                f"possibly-spurious due to blocked-issue lookup degraded: "
                f"{blocked_err}; {'; '.join(reasons)}"
            )
        else:
            detail = "; ".join(reasons)
        report.anom(check, f"{detail} ({facts})")
    else:
        if blocked_err:
            # Degradation can only *inflate* the dispatchable set, so an empty
            # reasons list with a degraded lookup is still a sound OK.
            facts += f" (blocked-issue lookup degraded: {blocked_err}; result is sound)"
        report.ok(check, facts)

    check_dispatch_throttle(report, repo, now=resolved_now)
    check_in_progress_staleness(
        report, repo, in_progress, prev_repo_state, new_repo_state, skip_delta, now=now
    )


def check_armable_backlog(
    report: Report,
    repo: RepoInfo,
    blocked_numbers: set[int] | None,
    blocked_err: str,
) -> None:
    """Is the armed runway thin while un-triaged, armable issues sit idle?

    ``dispatch-coverage`` asks "did the fleet pick up what is armed?"; this
    check asks the question upstream of it: "is there enough armed work for
    the fleet to pick up, and if not, is that because the backlog is
    genuinely empty or because nobody has triaged it?" (2026-08-23: both
    lanes were about to idle with 12 + 39 open issues carrying no label at
    all -- neither ``automated-ready`` nor any gating label -- so the fleet
    starved with work available.)

    Three buckets over the open issues:

    * ``runway``  -- ``automated-ready``, no ``agent:*`` label, not blocked:
      what dispatch can take next. Healthy when ``>= floor``.
    * ``active``  -- carries an ``agent:*`` label (in flight / terminal).
    * ``armable`` -- no ``agent:*`` label, not ``automated-ready``, and no
      *gating* label (``ARMABLE_GATING_LABELS``) or blocked-by-dependency
      entry. This is the un-triaged pool: every issue here is either a
      missed arm or a missed gate, and a triage pass drives it to zero.

    Verdict:

    * runway ``>= floor``                      -> OK (plenty to work on)
    * runway ``< floor`` and armable is empty  -> OK (genuinely empty)
    * runway ``< floor`` and armable non-empty -> ANOMALY: triage needed

    ``floor`` is the repo's ``dispatch.max_concurrent_sessions`` cap (one
    full wave of work), falling back to ``ARMABLE_RUNWAY_FLOOR_DEFAULT``.

    Degraded blocked-issue lookup (``blocked_err``) can only *inflate* both
    ``runway`` and ``armable``: an inflated runway can turn an anomaly into
    a false OK, an inflated armable can turn an OK into a false anomaly. The
    caveat is surfaced on whichever verdict is emitted rather than guessed
    around.
    """
    check = f"armable-backlog {repo.slug}"
    args = [
        "issue",
        "list",
        "-R",
        repo.slug,
        "--state",
        "open",
        "--json",
        "number,labels",
        "--limit",
        str(ISSUE_LIST_LIMIT),
    ]
    ok, data, err = run_gh_json(args, repo.repo_root)
    if not ok:
        report.anom(check, err)
        return

    runway: list[int] = []
    active = 0
    gated = 0
    armable: list[int] = []
    for issue in data:
        number = issue["number"]
        names = {label["name"] for label in issue.get("labels", [])}
        if any(n.startswith("agent:") for n in names):
            active += 1
            continue
        is_blocked = blocked_numbers is not None and number in blocked_numbers
        if is_blocked or names & ARMABLE_GATING_LABELS:
            gated += 1
            continue
        if ARMED_LABEL in names:
            runway.append(number)
        else:
            armable.append(number)

    cap = get_dispatch_cap(repo.config_path) if repo.config_path else None
    floor = cap if cap is not None else ARMABLE_RUNWAY_FLOOR_DEFAULT
    facts = (
        f"runway={len(runway)} floor={floor} armable={len(armable)} "
        f"active={active} gated={gated} open={len(data)}"
    )
    caveat = f" (blocked-issue lookup degraded: {blocked_err})" if blocked_err else ""

    if len(runway) >= floor:
        report.ok(check, f"plenty armed; {facts}{caveat}")
    elif not armable:
        report.ok(
            check, f"runway thin but backlog genuinely empty of armable issues; {facts}{caveat}"
        )
    else:
        preview = sorted(armable)[:ARMABLE_PREVIEW_LIMIT]
        more = len(armable) - len(preview)
        suffix = f" (+{more} more)" if more > 0 else ""
        report.anom(
            check,
            f"runway thin ({len(runway)} < floor {floor}) while {len(armable)} "
            f"un-triaged armable issue(s) sit idle: {preview}{suffix} -- triage: "
            f"label each `{ARMED_LABEL}` or one of {sorted(ARMABLE_GATING_LABELS)}"
            f"; {facts}{caveat}",
        )


def check_in_progress_staleness(
    report: Report,
    repo: RepoInfo,
    in_progress: list[tuple[int, str | None]],
    prev_repo_state: dict[str, Any],
    new_repo_state: dict[str, Any],
    skip_delta: bool,
    *,
    now: datetime | None = None,
) -> None:
    """Flag agent:in-progress issues whose updatedAt hasn't moved across 2 beats.

    Persists {issue_number: updatedAt} per repo in the state file so the next
    beat can compare. Entries for issues no longer in-progress are pruned
    automatically since cur_map is rebuilt fresh from this beat's data.

    Issue #1379: events-stale does not mean the worker is dead. Long-running
    workers emit events only at dispatch/PR/exit boundaries, not during
    implementation, so a healthy worker routinely shows zero new events across
    2 beats. Before flagging, the check also looks at the newest file mtime
    under the issue's worker worktree (the state record carries ``branch_name``;
    the worktree dir is derived from it). Worktree mtime cleanly separates a
    healthy worker (files modified seconds-to-minutes ago) from a dead one (no
    file activity for tens of minutes).

    Decision matrix:
    - events fresh (updatedAt moved) -> OK (not in the stale set at all).
    - events stale AND worktree mtime fresh -> OK, naming the mtime signal.
    - events stale AND worktree mtime stale -> ANOMALY, with BOTH ages in the
      line ("no events across 2 beats; worktree idle Nm") so the true-dead case
      reads unambiguously.
    - events stale AND worktree dir missing -> ANOMALY (events-only, today's
      behavior), with "no worktree found" in the line. Absence of the directory
      must not read as activity (fail toward flagging, not toward green).
    """
    check = f"in-progress-stale {repo.slug}"
    prev_map: dict[str, str] = prev_repo_state.get("in_progress", {})

    if skip_delta:
        # Leave the last real beat's snapshot untouched; this is not a real
        # comparison interval.
        new_repo_state["in_progress"] = prev_map
        report.ok(check, f"tracked={len(prev_map)} stale=0{DELTA_SKIP_SUFFIX}")
        return

    resolved_now = now if now is not None else datetime.now(timezone.utc)

    cur_map: dict[str, str] = {}
    stale: list[int] = []
    for number, updated_at in in_progress:
        if updated_at is None:
            continue
        cur_map[str(number)] = updated_at
        if prev_map.get(str(number)) == updated_at:
            stale.append(number)

    new_repo_state["in_progress"] = cur_map

    if not stale:
        report.ok(check, f"tracked={len(cur_map)} stale=0")
        return

    # Issue #1379: before flagging events-stale issues, check the worker's
    # worktree for recent file activity as a second liveness signal.
    state_issues = _load_state_issues(repo)
    threshold = resolved_now - timedelta(minutes=IN_PROGRESS_STALE_WORKTREE_MINUTES)
    # Resolve the worktrees root once (honours claude_code.worktrees_dir, issue
    # #1379 review) instead of re-reading the config per stale issue.
    worktrees_root = _resolved_worktrees_dir(repo)
    truly_stale_details: list[str] = []
    worktree_fresh: list[str] = []

    for number in sorted(stale):
        entry = state_issues.get(str(number))
        branch = entry.get("branch_name") if isinstance(entry, dict) else None
        if not branch:
            # No branch recorded in state: cannot locate a worktree. Keep
            # events-only behavior (fail toward flagging, not toward green).
            truly_stale_details.append(f"#{number}: no events across 2 beats; no worktree found")
            continue
        worktree = _worktree_path_for_branch(repo, branch, worktrees_dir=worktrees_root)
        if not worktree.is_dir():
            # Worktree missing entirely: absence must not read as activity.
            truly_stale_details.append(f"#{number}: no events across 2 beats; no worktree found")
            continue
        newest = _newest_worktree_mtime(worktree, threshold=threshold)
        if newest is not None and newest >= threshold:
            age_min = (resolved_now - newest).total_seconds() / 60
            worktree_fresh.append(f"#{number} worktree mtime {round(age_min)}m")
        else:
            wt_age = (
                round((resolved_now - newest).total_seconds() / 60) if newest is not None else 0
            )
            truly_stale_details.append(
                f"#{number}: no events across 2 beats; worktree idle {wt_age}m"
            )

    if truly_stale_details:
        detail = "; ".join(truly_stale_details) + " (threshold: 2 beats)"
        report.anom(check, detail)
    else:
        facts = (
            f"tracked={len(cur_map)} events-stale={len(stale)} "
            f"worktree-fresh={len(worktree_fresh)}"
        )
        if worktree_fresh:
            facts += "; " + ", ".join(worktree_fresh)
        report.ok(check, facts)


def _read_review_decision_payload(decision_path: Path) -> dict[str, Any] | None:
    """Read and parse a packet's ``review-decision.json``.

    Returns the parsed dict, or ``None`` when the file is missing, unreadable,
    or not a JSON object. A missing/unreadable file is treated as an open
    claim by :func:`_claim_is_open` (the placeholder has not been overwritten
    with a terminal verdict), so ``None`` here means "open, but no payload to
    inspect" rather than "definitely closed".
    """
    if not decision_path.exists():
        return None
    try:
        data = json.loads(decision_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _claim_is_open(decision_path: Path) -> bool:
    """A review claim is OPEN until a non-pending decision is recorded.

    Claims write a placeholder review-decision.json with decision="pending"
    at claim time, then overwrite it once the review actually completes. A
    missing file, a still-"pending" file, or an unparseable file all mean the
    claim has not been resolved yet.
    """
    data = _read_review_decision_payload(decision_path)
    if data is None:
        return True
    return data.get("decision") == "pending"


def _reviewer_pid_alive(entry: dict[str, Any]) -> bool | None:
    """Check if a reviewer PID recorded in state.json is alive.

    Companion to ``charlie_work.workflow._reviewer_pid_alive`` /
    ``charlie_work.process_utils.is_pid_alive``, adapted for the heartbeat
    script's reporting needs.  Uses ``psutil`` (already a project dependency)
    for a cross-platform PID + start-time probe rather than the
    platform-specific ctypes/``os.kill`` code in ``process_utils``.  Returns
    three-valued (``None``/``True``/``False``) so the heartbeat can distinguish
    "no PID recorded" from "alive" / "dead": ``None`` when no PID is recorded,
    ``True`` when the process is alive or its state is indeterminate, and
    ``False`` only when we can prove the PID is dead or has been recycled.
    """
    reviewer_pid = entry.get("reviewer_pid")
    if reviewer_pid is None:
        return None
    try:
        pid = int(reviewer_pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return False

    if not psutil.pid_exists(pid):
        return False

    expected_start_time = entry.get("reviewer_process_start_time")
    if expected_start_time is None:
        return True

    try:
        current_start_time = psutil.Process(pid).create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
        # Indeterminate probe: treat as alive rather than falsely flagging a
        # live worker as dead (issue #360/#343 criterion).
        return True

    try:
        expected = float(expected_start_time)
    except (TypeError, ValueError):
        return True

    return abs(current_start_time - expected) <= 1.0


def _review_claim_timestamp(
    pr_state: dict[str, Any],
    *,
    pr_dir: Path | None = None,
    decision: dict[str, Any] | None = None,
) -> str | None:
    """Return the most relevant claim timestamp for an open review.

    The orchestrator's ground truth is ``state.json``:

    * ``review_dispatch_dispatched`` -> ``review_dispatched_at``
    * ``review_dispatch_pending``    -> ``review_dispatch_pending_at``
    * ``review_dispatch_failed``     -> ``review_dispatch_failed_at``

    For unknown/missing status, fall back to the newest present timestamp.  This
    replaces the packet-directory ``st_mtime`` that never updates across redispatch
    retries (issue #517).

    Issue #1403: ``review_dispatch_completed`` is a special case.  After a review
    cycle finishes, ``record_review`` stamps ``review_dispatch_status`` to
    ``review_dispatch_completed`` and never touches ``review_dispatched_at``.
    When the rework cycle rebuilds the packet for a new head, the on-disk
    ``review-decision.json`` is back to ``pending`` (so the claim is open again)
    but state.json still carries the PRIOR cycle's ``review_dispatched_at`` --
    ``dispatch_reviews()`` only refreshes it when it actually launches the next
    reviewer, which may be waiting on a CI check.  The newest-timestamp fallback
    below would date the claim by that stale prior dispatch and overcount the
    age by the full inter-cycle gap (a false 138m ANOMALY observed on pr-1395).

    Detect this case structurally: the on-disk pending decision is head-stamped
    (``record_decision`` at packet build writes ``reviewed_head_sha`` = the new
    PR head), while state.json's ``reviewed_head_sha`` is the prior cycle's
    reviewed head.  A difference means the packet was rebuilt for a head that
    has not been reviewed yet.  Anchor on the packet-rebuild evidence -- the
    ``review-prompt.md`` mtime, rewritten on every ``review()`` packet build --
    instead of the stale state.json dispatch timestamp.  ``pr_dir`` and
    ``decision`` are optional so direct unit callers (and the pre-fix tests)
    keep the original state.json-only behavior.
    """
    status = pr_state.get("review_dispatch_status")
    if status == "review_dispatch_dispatched":
        return pr_state.get("review_dispatched_at")
    if status == "review_dispatch_pending":
        return pr_state.get("review_dispatch_pending_at")
    if status == "review_dispatch_failed":
        return pr_state.get("review_dispatch_failed_at")

    # Issue #1403: completed prior cycle whose packet was rebuilt for a newer
    # head.  See the docstring for why state.json's dispatch timestamps are
    # stale here.  The on-disk pending decision's ``reviewed_head_sha`` is the
    # new packet head; state.json's is the prior cycle's reviewed head.  They
    # differ exactly when the packet was rebuilt for a not-yet-reviewed head.
    if (
        status == "review_dispatch_completed"
        and pr_dir is not None
        and isinstance(decision, dict)
        and decision.get("decision") == "pending"
        and decision.get("reviewed_head_sha")
        and decision.get("reviewed_head_sha") != pr_state.get("reviewed_head_sha")
    ):
        prompt_path = pr_dir / "review-prompt.md"
        if prompt_path.exists():
            return (
                datetime.fromtimestamp(prompt_path.stat().st_mtime, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )

    newest: str | None = None
    newest_dt: datetime | None = None
    for key in (
        "review_dispatched_at",
        "review_dispatch_pending_at",
        "review_dispatch_failed_at",
    ):
        raw = pr_state.get(key)
        if not raw:
            continue
        parsed = parse_iso(raw)
        if parsed is None:
            continue
        if newest_dt is None or parsed > newest_dt:
            newest_dt = parsed
            newest = raw
    return newest


def check_review_liveness(report: Report, repo: RepoInfo, *, now: datetime | None = None) -> None:
    """``now`` is the injectable clock (issue #828); see ``check_dispatch_throttle``."""
    check = f"review-liveness {repo.slug}"
    prs_dir = repo.state_dir / "prs"
    if not prs_dir.exists():
        report.ok(check, "open_claims=0 (no prs dir)")
        return

    ok, open_data, err = run_gh_json(
        ["pr", "list", "-R", repo.slug, "--state", "open", "--json", "number"],
        repo.repo_root,
    )
    if not ok:
        report.anom(check, err)
        return
    open_pr_numbers = {pr["number"] for pr in open_data}

    state_json = repo.state_dir / "state.json"
    state_data: dict[str, Any] = {}
    state_read_error: str | None = None
    if state_json.exists():
        try:
            raw_state = json.loads(state_json.read_text(encoding="utf-8"))
            if isinstance(raw_state, dict):
                state_data = raw_state
        except (OSError, json.JSONDecodeError) as exc:
            state_read_error = str(exc)
    prs_state = state_data.get("prs", {}) if isinstance(state_data, dict) else {}
    if not isinstance(prs_state, dict):
        prs_state = {}

    resolved_now = now if now is not None else datetime.now(timezone.utc)
    open_claims = 0
    escalated_claims = 0
    stale: list[str] = []
    claims: list[tuple[int, int, str]] = []
    for entry in sorted(prs_dir.iterdir()):
        if not entry.is_dir():
            continue
        pr_json = entry / "pr.json"
        if not pr_json.exists():
            continue
        try:
            pr_number = int(entry.name.removeprefix("pr-"))
        except ValueError:
            continue
        if pr_number not in open_pr_numbers:
            # PR already resolved (merged/closed); claim dir is stale history
            # from before reap, not evidence of a live stuck review.
            continue
        # Read the on-disk decision once (issue #1403): _claim_is_open and the
        # completed-cycle-rebuilt-packet detection in _review_claim_timestamp
        # both need it, and re-reading races with a concurrent record_review.
        decision_payload = _read_review_decision_payload(entry / "review-decision.json")
        if decision_payload is None:
            # Missing/unreadable file: open claim, no payload to inspect.
            is_open = True
        else:
            is_open = decision_payload.get("decision") == "pending"
        if not is_open:
            continue

        pr_state = prs_state.get(str(pr_number), {}) if isinstance(prs_state, dict) else {}
        if not isinstance(pr_state, dict):
            pr_state = {}

        # Issue #1357: an escalated PR (``status == "escalated"`` in state.json,
        # ``agent:human-needed`` on the issue) never completes a review -- the
        # escalation gate stops further dispatch, so the placeholder
        # ``decision="pending"`` file written at packet-build time is accurate
        # history, not a liveness signal. Counting it as an open claim trips
        # ANOMALY on every heartbeat indefinitely. Skip the open-claim/stale
        # accounting for escalated entries and surface them separately in the
        # facts string instead. The packet and its pending decision file are
        # reused on unescalate (same-head packet semantics, #1351/#1352), so
        # the scoping belongs in this liveness check -- forging a terminal
        # decision or deleting the packet would corrupt review state to quiet
        # a monitor. The ``status`` field read here is the same one
        # ``charlie_work.escalation._escalation_flags`` keys on, so the
        # definition of "escalated" stays single-sourced.
        if pr_state.get("status") == "escalated":
            escalated_claims += 1
            continue

        open_claims += 1

        timestamp = _review_claim_timestamp(pr_state, pr_dir=entry, decision=decision_payload)
        claim_time = parse_iso(timestamp)
        if claim_time is None:
            # Last resort: the packet directory's mtime.  This is a fallback for
            # state.json entries that predate the dispatch-status fields, not the
            # primary clock (issue #517).
            claim_time = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)

        age_min = (resolved_now - claim_time).total_seconds() / 60
        age_rounded = round(age_min)

        pid_alive = _reviewer_pid_alive(pr_state)
        reviewer_pid = pr_state.get("reviewer_pid")
        if pid_alive is None:
            pid_label = "pid=None"
        elif reviewer_pid is None:
            pid_label = "pid=None"
        elif pid_alive:
            pid_label = f"pid={reviewer_pid} alive"
        else:
            pid_label = f"pid={reviewer_pid} dead"

        claims.append((pr_number, age_rounded, pid_label))
        if age_min > REVIEW_CLAIM_STALE_MINUTES:
            stale.append(f"{entry.name}: {age_rounded}m {pid_label}")

    facts = f"open_claims={open_claims}"
    if escalated_claims:
        facts += f" escalated={escalated_claims}"
    if open_claims:
        oldest = max(claims, key=lambda c: c[1])
        facts += f" oldest_min={oldest[1]} oldest={oldest[2]}"
    if state_read_error:
        facts += f" (state.json unreadable: {state_read_error})"
    if stale:
        report.anom(
            check,
            f"claim dir(s) {'; '.join(stale)} minutes old "
            f"(threshold={REVIEW_CLAIM_STALE_MINUTES}m) ({facts})",
        )
    else:
        report.ok(check, facts)


def check_dispatch_failures(report: Report, repo: RepoInfo, baseline: datetime) -> None:
    check = f"dispatch-failures {repo.slug}"
    dispatches_dir = repo.state_dir / "dispatches"
    if not dispatches_dir.exists():
        report.ok(check, "scanned=0")
        return

    candidates = list(dispatches_dir.glob("reviews/*.json")) + list(dispatches_dir.glob("*.json"))
    flagged: list[str] = []
    scanned = 0
    for path in candidates:
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        scanned += 1
        error = data.get("error")
        if error is not None and mtime > baseline:
            flagged.append(f"{path.name}: {str(error)[:80]}")

    facts = f"scanned={scanned}"
    if flagged:
        report.anom(check, f"new failures since last beat: {flagged} ({facts})")
    else:
        report.ok(check, facts)


def check_log_freshness(report: Report, repo: RepoInfo, *, now: datetime | None = None) -> None:
    """``now`` is the injectable clock (issue #828); see ``check_dispatch_throttle``."""
    check = f"log-freshness {repo.slug}"
    candidates = list(repo.state_dir.glob("*.log"))
    state_json = repo.state_dir / "state.json"
    if state_json.exists():
        candidates.append(state_json)
    candidates.extend(repo.state_dir.glob("*checkpoint*"))
    candidates = [c for c in candidates if c.is_file()]

    if not candidates:
        report.anom(check, "no log/state/checkpoint files found under state dir")
        return

    freshest = max(candidates, key=lambda p: p.stat().st_mtime)
    resolved_now = now if now is not None else datetime.now(timezone.utc)
    mtime = datetime.fromtimestamp(freshest.stat().st_mtime, tz=timezone.utc)
    age_min = (resolved_now - mtime).total_seconds() / 60

    facts = f"freshest={freshest.name} age={round(age_min)}m"
    if age_min > LOG_FRESHNESS_STALE_MINUTES:
        report.anom(
            check, f"freshest file older than threshold={LOG_FRESHNESS_STALE_MINUTES}m ({facts})"
        )
    else:
        report.ok(check, facts)


def check_loop_pass_freshness(
    report: Report,
    repo: RepoInfo,
    *,
    now: datetime | None = None,
    stale_minutes: int = LOOP_PASS_STALE_MINUTES,
) -> None:
    """Coarse backstop for prolonged, total fleet death -- NOT a detector
    for the #851/#854 outage class specifically (see ``LOOP_PASS_STALE_MINUTES``
    for the measured cadence data and why: that outage was shorter than this
    repo's own legitimate worst-case gap between loop passes, so no per-repo
    threshold can separate the two; PR #865 / issue #855 catches that class
    instead, by watching for consecutive zero-repo-pass cycles rather than
    elapsed time).

    What this still catches: the process exited 0, the scheduled task
    reported success, and the state dir kept getting touched
    (``self_deploy_succeeded`` fires every beat), so ``check_log_freshness``
    reads healthy indefinitely even though the loop body itself
    (``workflow.py``'s ``_loop_impl``, which is the only place that logs
    ``loop_started``) has not run in any of this repo's passes for well
    over an hour. The only ground truth for "is the loop actually running"
    is the ABSENCE of ``loop_started`` rows in ``events.db``.

    ``now`` is the injectable clock (issue #828); see ``check_dispatch_throttle``.

    Missing DB, missing table, and zero ``loop_started`` rows are each
    reported OK with a distinct message -- a fresh install/state dir with no
    history yet is not the same failure as a fleet that stopped mid-flight.

    CRITICAL: the freshness comparison is done in Python on parsed
    ``datetime`` objects, never in SQL. ``ts`` values are ISO strings like
    ``2026-07-31T22:25:04Z`` (``T``/``Z``); SQLite's
    ``datetime('now','-90 minutes')`` returns a space-separated,
    non-``Z`` string like ``2026-07-31 22:25:04``. A predicate such as
    ``WHERE ts < datetime('now','-90 minutes')`` compares them as strings,
    where ``'T'`` (0x54) sorts after ``' '`` (0x20) -- this silently
    misclassifies rows in both directions instead of raising, so the bug
    doesn't fail loudly, it just returns the wrong answer.
    """
    check = f"loop-pass-freshness {repo.slug}"
    db_path = repo.state_dir / "events.db"
    if not db_path.exists():
        report.ok(check, "no events.db (fresh install or pre-instrumentation state dir)")
        return

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        report.anom(check, f"events.db unreadable: {exc}")
        return

    try:
        try:
            table_row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            if table_row is None:
                report.ok(check, "events.db has no events table yet")
                return
            newest_ts = conn.execute(
                "SELECT MAX(ts) FROM events WHERE kind = 'loop_started'"
            ).fetchone()[0]
        except sqlite3.Error as exc:
            report.anom(check, f"events.db unreadable: {exc}")
            return
    finally:
        conn.close()

    if newest_ts is None:
        report.ok(check, "no loop_started rows recorded yet")
        return

    newest_dt = parse_iso(newest_ts)
    if newest_dt is None:
        report.anom(check, f"newest loop_started ts unparseable: {newest_ts!r}")
        return

    resolved_now = now if now is not None else datetime.now(timezone.utc)
    age_min = (resolved_now - newest_dt).total_seconds() / 60
    facts = f"newest_loop_started={newest_ts} age={round(age_min)}m"

    if age_min > stale_minutes:
        marker_path = repo.state_dir / "pending-sync.json"
        report.anom(
            check,
            f"no loop pass in {repo.slug} for {round(age_min)}m "
            f"(newest loop_started {newest_ts}), threshold={stale_minutes}m -- "
            f"at or beyond the observed healthy worst case, so the supervisor "
            f"may be dead or wedged. Cause is open-ended at this duration; "
            f"{marker_path} is one thing worth checking, not the only one. "
            f"({facts})",
        )
    else:
        report.ok(check, facts)


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


def check_merge_flow(
    report: Report,
    repo: RepoInfo,
    prev_repo_state: dict[str, Any],
    new_repo_state: dict[str, Any],
    skip_delta: bool,
) -> None:
    check = f"merge-flow {repo.slug}"
    ok_open, open_data, err_open = run_gh_json(
        ["pr", "list", "-R", repo.slug, "--state", "open", "--json", "number,labels"],
        repo.repo_root,
    )
    if not ok_open:
        report.anom(check, err_open)
        return

    mergequeue_label = get_mergequeue_label(repo.config_path) if repo.config_path else None
    if mergequeue_label:
        mergequeue_count = sum(
            1
            for pr in open_data
            if mergequeue_label in {label["name"] for label in pr.get("labels", [])}
        )
    else:
        mergequeue_count = 0

    merged_args = [
        "pr",
        "list",
        "-R",
        repo.slug,
        "--state",
        "merged",
        "--limit",
        str(MERGED_PR_LOOKBACK_LIMIT),
        "--json",
        "number,mergedAt",
    ]
    ok_merged, merged_data, err_merged = run_gh_json(merged_args, repo.repo_root)
    latest_merged_at: str | None = None
    if ok_merged and merged_data:
        merged_times = [pr["mergedAt"] for pr in merged_data if pr.get("mergedAt")]
        if merged_times:
            latest_merged_at = max(merged_times)

    prev_count = prev_repo_state.get("mergequeue_count")
    prev_streak = prev_repo_state.get("mergequeue_unchanged_streak", 0)
    prev_last_merged = prev_repo_state.get("last_merged_at")

    if skip_delta:
        # Carry forward the last real beat's delta state untouched.
        new_repo_state["mergequeue_count"] = prev_count
        new_repo_state["mergequeue_unchanged_streak"] = prev_streak
        new_repo_state["last_merged_at"] = prev_last_merged
        facts = (
            f"open={len(open_data)} mergequeue={mergequeue_count} "
            f"unchanged_streak={prev_streak}{DELTA_SKIP_SUFFIX}"
        )
        if not ok_merged:
            facts += f" (merged-pr lookup degraded: {err_merged})"
        report.ok(check, facts)
        return

    unchanged = prev_count is not None and prev_count == mergequeue_count
    merged_since_last_beat = latest_merged_at is not None and (
        prev_last_merged is None or latest_merged_at > prev_last_merged
    )
    new_streak = prev_streak + 1 if unchanged else 0

    new_repo_state["mergequeue_count"] = mergequeue_count
    new_repo_state["mergequeue_unchanged_streak"] = new_streak
    new_repo_state["last_merged_at"] = latest_merged_at or prev_last_merged

    facts = f"open={len(open_data)} mergequeue={mergequeue_count} unchanged_streak={new_streak}"
    if not ok_merged:
        facts += f" (merged-pr lookup degraded: {err_merged})"

    if (
        mergequeue_count > 0
        and new_streak >= MERGEQUEUE_STALL_BEATS
        and not merged_since_last_beat
    ):
        report.anom(
            check,
            f"mergequeue count stuck for {new_streak} beats "
            f"(threshold={MERGEQUEUE_STALL_BEATS}) with no merge since last beat ({facts})",
        )
    else:
        report.ok(check, facts)


def check_stale_open_issue_mentions(report: Report, repo: RepoInfo) -> None:
    """Surface open issues referenced by already-merged work with no closure path (issue #902).

    The gap: `workflow.py`'s finalization path (`_merged_pr_referenced_issue_numbers`)
    intersects a merged PR's mentioned issue numbers against the *currently
    `ready`-labelled* issue set before ever surfacing anything, by design --
    its docstring is explicit that this exists so "a stray mention of an
    issue not in the dispatch queue does not get actioned." That intersect is
    correct and this check does not touch it: a bare mention must never
    *authorize* a lifecycle transition (see #781/#790's false-close
    incident). But the same intersect means an issue with **zero labels**
    (never dispatched, never triaged) can be fully fixed and merged and the
    finalization path will never even consider it, because it was never a
    candidate to begin with. #817 and #866 are exactly this: both carried no
    labels at all and both stayed open after their fixing PRs merged.

    This check is a separate, read-only roll-call living entirely outside
    the dispatch/finalization lane -- #203's originally-proposed option 3,
    never implemented. Its candidate set is `gh issue list --state open`
    with **no label filter and no `state.json` read**: that is the one
    property that makes it able to see what the dispatch-lane check
    structurally cannot. It never labels, comments, or closes anything --
    only ever calls `report.anom`/`report.ok`.

    Three sources, matching the issue's "API economy" constraint (this runs
    unattended alongside nine other checks, so no `gh` call may scale with
    the number of open issues or merged PRs):

    1. One `gh issue list --state open --json number` call for the
       candidate set.
    2. One `gh pr list --state merged --json number,headRefName,title,body,
       closingIssuesReferences,mergedAt` call (bounded by
       `STALE_MENTION_PR_LOOKBACK_LIMIT`), scanned for a branch-name issue
       number (`_branch_issue_number`) or a bare `#N` mention
       (`_mentioned_issue_numbers`) in title/body. This is what would catch
       #817: PR #824's branch is `fix/817-fleet-health-latch` (no
       `agent/issue` prefix, so `linked_issue_number` never trusts it) and
       its body reads "For issue #817:" (a reference, but not a closing
       keyword, so `closingIssuesReferences` came back empty on the PR
       itself).
    3. Local `git log` on the already-checked-out branch (`get_merged_commit_messages`,
       bounded by `STALE_MENTION_COMMIT_LOOKBACK`) -- zero API cost, and the
       only source that can catch #866: its fix landed as a commit inside PR
       #864, a PR *for a different issue*, so no scan of any PR's own
       title/body/branch name -- #864's or otherwise -- could ever find it.

    `closingIssuesReferences` is fetched (per the issue's specified command
    shape) but not used to gate reporting: since the candidate set is
    already restricted to *currently open* issues, any issue GitHub's native
    auto-close already resolved via a real closing-keyword match is
    definitionally no longer in that set. No extra filtering on that field
    can change which issues get reported here.

    Quoted or negated mentions are excluded (`_mentioned_issue_numbers`),
    consistent with #781/#790: a mention is evidence for a human to check,
    never grounds to act automatically, and a quoted/negated one is not even
    that. Output is capped at `STALE_MENTION_REPORT_CAP` issues (with a
    "+K more" suffix) so a large true positive count cannot flood the beat.
    """
    check = f"stale-open-issue-mentions {repo.slug}"

    ok_open, open_data, err_open = run_gh_json(
        [
            "issue",
            "list",
            "-R",
            repo.slug,
            "--state",
            "open",
            "--json",
            "number",
            "--limit",
            str(ISSUE_LIST_LIMIT),
        ],
        repo.repo_root,
    )
    if not ok_open:
        report.anom(check, err_open)
        return
    open_numbers = {issue["number"] for issue in open_data}

    ok_merged, merged_data, err_merged = run_gh_json(
        [
            "pr",
            "list",
            "-R",
            repo.slug,
            "--state",
            "merged",
            "--limit",
            str(STALE_MENTION_PR_LOOKBACK_LIMIT),
            "--json",
            "number,headRefName,title,body,closingIssuesReferences,mergedAt",
        ],
        repo.repo_root,
    )
    if not ok_merged:
        report.anom(check, err_merged)
        return

    ok_commits, commits, err_commits = get_merged_commit_messages(
        repo.repo_root, STALE_MENTION_COMMIT_LOOKBACK
    )

    mentions: dict[int, list[str]] = {}

    def record(number: int, evidence: str) -> None:
        if number in open_numbers:
            mentions.setdefault(number, []).append(evidence)

    for pr in merged_data:
        pr_number = pr.get("number")
        branch = str(pr.get("headRefName") or "")
        branch_issue = _branch_issue_number(branch)
        if branch_issue is not None:
            record(branch_issue, f"PR #{pr_number} branch {branch!r}")
        text = f"{pr.get('title') or ''}\n{pr.get('body') or ''}"
        for number in _mentioned_issue_numbers(text):
            record(number, f"PR #{pr_number} title/body")

    for sha, message in commits:
        for number in _mentioned_issue_numbers(message):
            record(number, f"commit {sha}")

    facts = (
        f"open={len(open_numbers)} merged_prs_scanned={len(merged_data)} "
        f"commits_scanned={len(commits)}"
    )
    if not ok_commits:
        facts += f" (commit-message scan degraded: {err_commits})"

    if not mentions:
        report.ok(check, f"stale_mentions=0 ({facts})")
        return

    matched_numbers = sorted(mentions)
    shown = matched_numbers[:STALE_MENTION_REPORT_CAP]
    detail_parts = [f"#{n} ({mentions[n][0]})" for n in shown]
    if len(matched_numbers) > STALE_MENTION_REPORT_CAP:
        detail_parts.append(f"+{len(matched_numbers) - STALE_MENTION_REPORT_CAP} more")

    report.anom(
        check,
        f"{len(matched_numbers)} open issue(s) referenced by merged work with no closure "
        f"path: {'; '.join(detail_parts)} ({facts})",
    )


def check_github_rate(report: Report, any_repo_root: Path) -> None:
    check = "github-rate"
    ok, data, err = run_gh_json(["api", "rate_limit"], any_repo_root)
    if not ok:
        report.anom(check, err)
        return
    try:
        remaining = data["resources"]["graphql"]["remaining"]
    except (KeyError, TypeError):
        report.anom(check, f"unexpected rate_limit payload shape: {str(data)[:150]}")
        return

    facts = f"graphql_remaining={remaining}"
    if remaining < GRAPHQL_RATE_LIMIT_MIN_REMAINING:
        report.anom(
            check,
            f"graphql remaining below threshold={GRAPHQL_RATE_LIMIT_MIN_REMAINING} ({facts})",
        )
    else:
        report.ok(check, facts)


def _volume_label(anchor: str) -> str:
    """Compact display label for a volume anchor (e.g. ``C:\\`` -> ``C:``).

    Strips trailing path separators so the per-volume line reads
    ``OK disk-space C: free=...`` rather than ``... C:\\ ...``. A root-only
    anchor (POSIX ``/``) is returned unchanged so it does not collapse to an
    empty label.
    """
    stripped = anchor.rstrip("\\/")
    return stripped if stripped else anchor


def check_disk_space(report: Report, repos: list[RepoInfo]) -> None:
    """Flag low free disk space on any volume hosting a monitored state root.

    Issue #1359: the 2026-08-19 disk-full outage drained the host volume to 0
    bytes free while every fleet pass failed with
    ``OSError: [Errno 28] No space left on device`` and state.json went stale
    in both lanes. The first heartbeat signal was error-events firing AFTER
    writes were already failing fleet-wide; free space had been draining for
    ~3.5 days with zero early warning. This check surfaces the drain at a
    coarse threshold days before impact.

    The volume set is DERIVED from configuration the script already knows --
    each registered repo's ``state_dir`` (which holds ``state.json`` and
    ``events.db`` -- the same paths the rest of this script monitors) plus the
    fleet dir (which holds ``heartbeat-state.json`` and
    ``supervisor-heartbeat.json``) -- never a hardcoded drive letter. Volumes
    are deduplicated by drive anchor (``Path.anchor``: ``C:\\`` on Windows,
    the mount-root ``/`` on POSIX), so a single volume hosting several repos'
    state dirs -- the common one-drive-host case -- reports once, not N times.
    ``events.db`` lives at ``state_dir / "events.db"`` and therefore shares
    ``state_dir``'s volume, so it needs no separate entry.

    Uses ``shutil.disk_usage`` (stdlib, no new deps, consistent with this
    script's stdlib-only invariant in ``scripts/README.md``). Below the hard
    threshold (``DISK_FREE_ANOMALY_BYTES`` or ``DISK_FREE_ANOMALY_RATIO``) ->
    ``report.anom`` (flips the exit code, exactly like other anomalies);
    between soft and hard -> ``report.warn`` (routine-operational, never flips
    the exit code, matching ``check_warning_events``'s treatment of
    normal-operation warnings); otherwise ``report.ok``.
    """
    # Collect candidate paths whose hosting volumes matter, then deduplicate
    # by drive anchor so each volume reports exactly once.
    candidates: list[Path] = [repo.state_dir for repo in repos]
    candidates.append(fleet_dir())
    volumes: dict[str, Path] = {}
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            # An unresolvable path (e.g. a registered state_dir on a drive
            # that no longer exists) still has an anchor worth probing; fall
            # back to the literal path so disk_usage can surface the failure
            # rather than silently dropping the volume from the report.
            resolved = path
        anchor = resolved.anchor or str(resolved)
        volumes.setdefault(anchor, resolved)

    for anchor, sample_path in volumes.items():
        check = f"disk-space {_volume_label(anchor)}"
        try:
            usage = shutil.disk_usage(str(sample_path))
        except OSError as exc:
            report.anom(check, f"cannot stat volume {anchor}: {exc}")
            continue
        free = usage.free
        total = usage.total
        ratio = free / total if total > 0 else 0.0
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        facts = f"free={free_gb:.1f}GB ({ratio * 100:.1f}%) total={total_gb:.1f}GB"
        if free < DISK_FREE_ANOMALY_BYTES or ratio < DISK_FREE_ANOMALY_RATIO:
            report.anom(
                check,
                f"free space below hard threshold (free={free_gb:.1f}GB/"
                f"{ratio * 100:.1f}%, anomaly below "
                f"{DISK_FREE_ANOMALY_BYTES / 1024**3:.0f}GB or "
                f"{DISK_FREE_ANOMALY_RATIO * 100:.0f}%) ({facts})",
            )
        elif free < DISK_FREE_WARN_BYTES or ratio < DISK_FREE_WARN_RATIO:
            report.warn(
                check,
                f"free space below soft threshold (free={free_gb:.1f}GB/"
                f"{ratio * 100:.1f}%, warn below "
                f"{DISK_FREE_WARN_BYTES / 1024**3:.0f}GB or "
                f"{DISK_FREE_WARN_RATIO * 100:.0f}%) ({facts})",
            )
        else:
            report.ok(check, facts)


def check_runners(report: Report) -> None:
    check = "runners"
    if sys.platform != "win32":
        # schtasks is Windows-only; on other platforms there is no equivalent
        # scheduled-task probe, so report OK with an explicit note rather than
        # a false anomaly from the OSError catch.
        report.ok(check, f"skipped on {sys.platform} (schtasks is Windows-only)")
        return
    try:
        proc = subprocess.run(
            ["schtasks", "/query", "/tn", FLEET_TASK_NAME, "/fo", "LIST", "/v"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GH_TIMEOUT_SECONDS,
            creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        report.anom(check, f"schtasks failed to run: {exc}")
        return

    if proc.returncode != 0:
        stderr = proc.stderr.strip().replace("\n", " ")[:150]
        report.anom(
            check, f"scheduled task {FLEET_TASK_NAME!r} not found or query failed: {stderr}"
        )
        return

    last_result: int | None = None
    for line in proc.stdout.splitlines():
        if line.strip().startswith("Last Result:"):
            raw = line.split(":", 1)[1].strip()
            try:
                last_result = int(raw)
            except ValueError:
                last_result = None
            break

    if last_result is None:
        report.anom(check, "could not parse 'Last Result' from schtasks output")
        return

    facts = f"task={FLEET_TASK_NAME} last_result={last_result}"
    if last_result not in SCHTASKS_OK_RESULT_CODES:
        report.anom(
            check,
            f"last run result {last_result} not in OK set {sorted(SCHTASKS_OK_RESULT_CODES)} ({facts})",
        )
    else:
        report.ok(check, facts)


def check_supervisor_heartbeat(report: Report) -> None:
    """Flag a stale or absent fleet supervisor heartbeat (issue #627).

    The supervisor writes ``supervisor-heartbeat.json`` in the fleet dir every
    loop iteration. A stale ``last_beat_at`` means the supervisor is not making
    progress — either killed (``exited_at`` null, no clean exit recorded) or
    cleanly stopped but not restarted by the watchdog (``exited_at`` set, the
    2026-07-25 18:24 UTC outage shape where the watchdog task was disabled).

    This is the independent detector that catches both shapes: a killed
    supervisor leaves the heartbeat stale with no ``exited_at``, and a
    supervisor whose launcher was also killed leaves no marker at all — the
    heartbeat file's age is the only remaining signal. The stale threshold
    derives from ``max_pass_runtime_seconds`` recorded in the heartbeat
    itself (the config knob that bounds a single pass's wall-clock runtime),
    falling back to ``full_pass_interval_seconds`` for older heartbeats.
    """
    check = "supervisor-heartbeat"
    path = fleet_dir() / SUPERVISOR_HEARTBEAT_FILENAME
    if not path.exists():
        report.anom(
            check,
            f"no {SUPERVISOR_HEARTBEAT_FILENAME} found under {fleet_dir()} "
            "(supervisor has never started, or the heartbeat was wiped)",
        )
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.anom(check, f"{SUPERVISOR_HEARTBEAT_FILENAME} unreadable: {exc}")
        return
    if not isinstance(data, dict):
        report.anom(check, f"{SUPERVISOR_HEARTBEAT_FILENAME} malformed (not a JSON object)")
        return

    last_beat = parse_iso(data.get("last_beat_at"))
    if last_beat is None:
        report.anom(check, f"{SUPERVISOR_HEARTBEAT_FILENAME} has no parseable last_beat_at")
        return

    now = datetime.now(timezone.utc)
    age_min = (now - last_beat).total_seconds() / 60.0
    exited_at = data.get("exited_at")
    try:
        raw_timeout = data.get("max_pass_runtime_seconds")
        pass_timeout = int(raw_timeout) if raw_timeout is not None else None
    except (TypeError, ValueError):
        pass_timeout = None
    if pass_timeout is None or pass_timeout <= 0:
        try:
            raw_interval = data.get("full_pass_interval_seconds")
            pass_timeout = (
                int(raw_interval)
                if raw_interval is not None
                else SUPERVISOR_HEARTBEAT_DEFAULT_PASS_TIMEOUT_SECONDS
            )
        except (TypeError, ValueError):
            pass_timeout = SUPERVISOR_HEARTBEAT_DEFAULT_PASS_TIMEOUT_SECONDS
    stale_threshold_min = (SUPERVISOR_HEARTBEAT_STALE_MULTIPLIER * pass_timeout) / 60.0

    pid = data.get("pid")
    facts = (
        f"last_beat={round(age_min)}m ago pid={pid} exited_at={exited_at} "
        f"pass_timeout={pass_timeout}s"
    )

    if age_min <= stale_threshold_min:
        report.ok(check, facts)
        return

    if exited_at is not None:
        report.anom(
            check,
            f"supervisor exited cleanly at {exited_at} but has not restarted in "
            f"{round(age_min)}m (threshold={round(stale_threshold_min)}m) — the "
            f"watchdog may be disabled ({facts})",
        )
    else:
        report.anom(
            check,
            f"supervisor heartbeat stale: last beat {round(age_min)}m ago with no "
            f"clean exit (threshold={round(stale_threshold_min)}m) — likely killed "
            f"or hung ({facts})",
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    # Sampled once for this entire beat (issue #828) and threaded into every
    # sub-check below instead of each one independently racing the wall
    # clock -- keeps all checks in one run reporting against a single
    # consistent instant, and makes the run's own last_beat_at exact. Issue
    # #1361 reuses this same instant for suppression-registry expiry checks
    # rather than sampling the clock a second time.
    now = datetime.now(timezone.utc)

    suppressions, suppression_err = load_suppression_registry(suppression_registry_path())
    report = Report(suppressions=suppressions, now=now)
    if suppression_err:
        # Fail closed (issue #1361): the registry loaded as empty, so no
        # suppression applies this run -- this ANOMALY is additive, not a
        # replacement for whatever previously-suppressed conditions now
        # resurface as raw ANOMALY lines below.
        report.anom("suppression-registry", suppression_err)

    repos, load_err = load_repos()
    if load_err:
        report.anom("fleet-registry", load_err)
        print("\n".join(report.lines))
        return 1

    prev_state = load_state()
    prev_last_beat_at = parse_iso(prev_state.get("last_beat_at"))
    baseline = prev_last_beat_at or (now - timedelta(minutes=LOG_FRESHNESS_STALE_MINUTES))
    skip_delta = prev_last_beat_at is not None and (now - prev_last_beat_at) < timedelta(
        minutes=MIN_BEAT_INTERVAL_MINUTES
    )

    new_state: dict[str, Any] = {
        "last_beat_at": now.isoformat(),
        "repos": {},
    }

    blocked_by_repo: dict[str, set[int]] = {}
    blocked_err = ""
    if repos:
        blocked_by_repo, blocked_err = get_blocked_issue_numbers(repos[0].repo_root)

    for repo in repos:
        prev_repo_state = prev_state.get("repos", {}).get(repo.slug, {})
        new_repo_state: dict[str, Any] = {}
        check_orchestrator_config(report, repo)
        check_dispatch_coverage(
            report,
            repo,
            prev_repo_state,
            new_repo_state,
            skip_delta,
            blocked_by_repo.get(repo.slug),
            blocked_err,
            now=now,
        )
        check_armable_backlog(report, repo, blocked_by_repo.get(repo.slug), blocked_err)
        check_review_liveness(report, repo, now=now)
        check_dispatch_failures(report, repo, baseline)
        check_error_events(report, repo, baseline)
        check_warning_events(report, repo, baseline)
        check_infra_blocked_events(report, repo, baseline)
        check_log_freshness(report, repo, now=now)
        check_loop_pass_freshness(report, repo, now=now)
        check_merge_flow(report, repo, prev_repo_state, new_repo_state, skip_delta)
        check_stale_open_issue_mentions(report, repo)
        new_state["repos"][repo.slug] = new_repo_state

    if repos:
        check_github_rate(report, repos[0].repo_root)
    else:
        report.anom("github-rate", "no repos registered, cannot resolve a cwd for gh")

    check_disk_space(report, repos)
    check_runners(report)
    check_supervisor_heartbeat(report)

    save_state(new_state)

    summary = report.suppression_summary()
    if summary is not None:
        report.ok("suppressions", summary)

    print("\n".join(report.lines))
    return 1 if report.anomaly else 0


if __name__ == "__main__":
    sys.exit(main())
