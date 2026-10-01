"""Deterministic fleet-heartbeat check.

Replaces LLM-driven heartbeat judgment (which misread log freshness from file
HEAD lines, parsed a PR number as a count, and false-alarmed a merge-queue
stall) with plain data collection + threshold comparison. An LLM only ever
sees this script's stdout.

Run via:
    cd /path/to/charlie-work
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

import importlib.util
import json
import os
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

# Issue #1895: the events.db kind/level anomaly checks
# (check_error_events, check_warning_events, check_infra_blocked_events,
# check_draft_pr_blocked_events, check_ci_headroom_unavailable, and --
# since #1968 -- check_local_lane_kill_switch_stalled) live in the
# sibling module scripts/heartbeat_event_alarms.py -- extracted for
# file-size-ratchet headroom. Loaded via importlib
# from the sibling script path (never a bare `import`: scripts/ is not a
# package and tests/_script_loader.py deliberately keeps it off sys.path) --
# the same pattern #1879 uses for heartbeat_local_repo.py and
# git_push_lint_hook.py uses for worker_stop_gate.py. The names are
# re-exported below so `hb.check_*` attribute references in tests keep
# resolving unchanged -- as does EXPECTED_OPERATIONAL_KINDS, whose guarded
# `charlie_work.event_kinds` import moved to the sibling with
# check_warning_events, its only consumer (see that module for the full
# import-guard rationale); the subprocess guards in
# tests/test_heartbeat_check_ci_fleet_isolation.py still exercise the
# empty-frozenset degradation through this re-export.
_alarms_path = Path(__file__).resolve().parent / "heartbeat_event_alarms.py"
_alarms_spec: Any = importlib.util.spec_from_file_location("_heartbeat_event_alarms", _alarms_path)
_event_alarms = importlib.util.module_from_spec(_alarms_spec)
sys.modules[_alarms_spec.name] = _event_alarms
_alarms_spec.loader.exec_module(_event_alarms)

EXPECTED_OPERATIONAL_KINDS = _event_alarms.EXPECTED_OPERATIONAL_KINDS
check_error_events = _event_alarms.check_error_events
check_warning_events = _event_alarms.check_warning_events
check_infra_blocked_events = _event_alarms.check_infra_blocked_events
check_draft_pr_blocked_events = _event_alarms.check_draft_pr_blocked_events
check_ci_headroom_unavailable = _event_alarms.check_ci_headroom_unavailable
check_local_lane_kill_switch_stalled = _event_alarms.check_local_lane_kill_switch_stalled

# Issue #1476: the config + worktree-path resolution helpers
# (``load_orchestrator_config``, ``_slugify_branch``,
# ``_resolved_worktrees_dir``, ``_worktree_path_for_branch``) live in the
# sibling module scripts/heartbeat_worktree.py -- extraction for file-size
# ratchet headroom, same importlib-loader pattern as heartbeat_event_alarms
# above. Re-exported here so existing call sites and tests
# (``hb._worktree_path_for_branch``, ``hb._slugify_branch``) keep resolving.
_worktree_path = Path(__file__).resolve().parent / "heartbeat_worktree.py"
_worktree_spec: Any = importlib.util.spec_from_file_location("_heartbeat_worktree", _worktree_path)
_worktree_helpers = importlib.util.module_from_spec(_worktree_spec)
sys.modules[_worktree_spec.name] = _worktree_helpers
_worktree_spec.loader.exec_module(_worktree_helpers)

load_orchestrator_config = _worktree_helpers.load_orchestrator_config
_slugify_branch = _worktree_helpers._slugify_branch
_resolved_worktrees_dir = _worktree_helpers._resolved_worktrees_dir
_worktree_path_for_branch = _worktree_helpers._worktree_path_for_branch

# Issue #1861: local_issues gating lives in the sibling module
# scripts/heartbeat_local_repo.py -- same importlib-loader pattern as the
# extractions above. Re-exported here so call sites and tests
# (``hb.LOCAL_ONLY_SKIP_DETAIL``, ``hb.FLEET_GLOBAL_CONFIG_FILENAME``) keep
# resolving.
_lr_path = Path(__file__).resolve().parent / "heartbeat_local_repo.py"
_lr_spec: Any = importlib.util.spec_from_file_location("_heartbeat_local_repo", _lr_path)
_local_repo = importlib.util.module_from_spec(_lr_spec)
sys.modules[_lr_spec.name] = _local_repo
_lr_spec.loader.exec_module(_local_repo)
FLEET_GLOBAL_CONFIG_FILENAME = _local_repo.FLEET_GLOBAL_CONFIG_FILENAME
LOCAL_ONLY_SKIP_DETAIL = _local_repo.LOCAL_ONLY_SKIP_DETAIL
_local_issues_enabled = _local_repo.local_issues_enabled

# Issue #2004: check_armable_backlog and its dependency/sweep gates (the
# guarded parse_blockers/DEPRECATED_CONFIG_KEYS imports, the self-repo
# anchor, and the ARMABLE_* constants) live in the sibling module
# scripts/heartbeat_armable_gate.py -- extraction for file-size ratchet
# headroom, same importlib-loader pattern as the siblings above. The sibling
# binds this module via ``bind`` and reads ``run_gh_json`` /
# ``get_dispatch_cap`` / ``LOCAL_ONLY_SKIP_DETAIL`` / ``ISSUE_LIST_LIMIT``
# off the handle at call time, so ``hb.*`` monkeypatches in tests keep
# working. Names are re-exported below so ``hb.check_armable_backlog`` and
# ``hb.ARMABLE_*`` attribute references keep resolving unchanged.
_gate_path = Path(__file__).resolve().parent / "heartbeat_armable_gate.py"
_gate_spec: Any = importlib.util.spec_from_file_location("_heartbeat_armable_gate", _gate_path)
_armable_gate = importlib.util.module_from_spec(_gate_spec)
sys.modules[_gate_spec.name] = _armable_gate
_gate_spec.loader.exec_module(_armable_gate)
_armable_gate.bind(sys.modules[__name__])

ARMABLE_RUNWAY_FLOOR_DEFAULT = _armable_gate.ARMABLE_RUNWAY_FLOOR_DEFAULT
ARMED_LABEL = _armable_gate.ARMED_LABEL
ARMABLE_GATING_LABELS = _armable_gate.ARMABLE_GATING_LABELS
ARMABLE_PREVIEW_LIMIT = _armable_gate.ARMABLE_PREVIEW_LIMIT
check_armable_backlog = _armable_gate.check_armable_backlog

# Issue #2048: the stale-open-issue-mention scanning seam (the markdown
# block-structure scan, the ``#N`` mention primitives, the local ``git log``
# reader, and the #2048 exclusion classifiers) lives in the sibling module
# scripts/heartbeat_stale_mentions.py -- extraction for file-size-ratchet
# headroom, same importlib-loader pattern as heartbeat_event_alarms and
# heartbeat_worktree above. Re-exported here so existing call sites and
# tests (``hb._mentioned_issue_numbers``, ``hb._scan_markdown_structure``,
# ``hb.get_merged_commit_messages`` ...) keep resolving unchanged.
_sm_path = Path(__file__).resolve().parent / "heartbeat_stale_mentions.py"
_sm_spec: Any = importlib.util.spec_from_file_location("_heartbeat_stale_mentions", _sm_path)
_stale_mentions = importlib.util.module_from_spec(_sm_spec)
sys.modules[_sm_spec.name] = _stale_mentions
_sm_spec.loader.exec_module(_stale_mentions)

_MarkdownStructure = _stale_mentions._MarkdownStructure
_md_split_lines = _stale_mentions._md_split_lines
_md_strip_marker_indent = _stale_mentions._md_strip_marker_indent
_md_find_fence_close = _stale_mentions._md_find_fence_close
_scan_markdown_structure = _stale_mentions._scan_markdown_structure
_strip_fenced_code_blocks = _stale_mentions._strip_fenced_code_blocks
_has_preceding_negation = _stale_mentions._has_preceding_negation
_is_quoted = _stale_mentions._is_quoted
_mentioned_issue_numbers = _stale_mentions._mentioned_issue_numbers
_branch_issue_number = _stale_mentions._branch_issue_number
_is_non_closing_context = _stale_mentions._is_non_closing_context
issue_mention_occurrences = _stale_mentions.issue_mention_occurrences
STALE_MENTION_PARKED_LABELS_DEFAULT = _stale_mentions.STALE_MENTION_PARKED_LABELS_DEFAULT
parked_label_names = _stale_mentions.parked_label_names
BOT_AUTHOR_LOGINS = _stale_mentions.BOT_AUTHOR_LOGINS
pr_author_is_bot = _stale_mentions.pr_author_is_bot
lifecycle_label_names = _stale_mentions.lifecycle_label_names
mention_exempt_by_label = _stale_mentions.mention_exempt_by_label
get_merged_commit_messages = _stale_mentions.get_merged_commit_messages

# Pure alarm verdicts shared with the fleet dashboard live in the stdlib-only
# leaves charlie_work.heartbeat_alarms / heartbeat_alarms_fleet; the thresholds
# below are re-exported from them (single source). Loaded via the installed
# package, else BY FILE PATH from <repo>/src (see
# heartbeat_event_alarms.load_alarm_leaves) so a wrong-venv run keeps real
# verdicts; only if both fail do the checks report a loud ANOMALY.
_ha, _haf = _event_alarms.load_alarm_leaves()
_ALARMS_UNAVAILABLE = (
    "cannot evaluate: heartbeat_alarms leaves not importable and not loadable from src/"
)

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

GH_TIMEOUT_SECONDS = 30
ISSUE_LIST_LIMIT = 200
MERGED_PR_LOOKBACK_LIMIT = 5

QUEUED_STALE_MINUTES = 20

REVIEW_CLAIM_STALE_MINUTES = 45
LOG_FRESHNESS_STALE_MINUTES = _haf.LOG_FRESHNESS_STALE_MINUTES if _haf else 30
# Measured production cadence (charlie-work `loop_started` gaps, last 39
# intervals, 2026-07-31): min=5.5m median=10.4m p90=20.2m max=53.9m.
# `loop_started` is logged per repo (workflow.py's `_loop_impl`, into that
# repo's own events.db), and the supervisor processes repos sequentially in
# one pass, so a single repo's gap is gated by how long its SIBLING repos
# take, not by supervisor health -- a sibling registered repo's reconcile alone walks
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
LOOP_PASS_STALE_MINUTES = _haf.LOOP_PASS_STALE_MINUTES if _haf else 90
MERGEQUEUE_STALL_BEATS = 2
GRAPHQL_RATE_LIMIT_MIN_REMAINING = 500
DISPATCH_THROTTLE_MAX_MINUTES = 30
MIN_BEAT_INTERVAL_MINUTES = 10
# Issue #1438: `charlie fleet status --json` wall time is ~60s on this host
# (dominated by remote API calls). A timeout equal to the typical runtime is a
# coin flip, not a bound -- the lookup degraded on any beat that landed a hair
# past the line (reproduced at 59.955s against a 60s timeout). 120s makes the
# bound a real outlier detector (2x the median) instead of the median itself.
# The result is fetched ONCE per heartbeat run (in main(), before the per-repo
# loop) and threaded into every consumer, so this timeout is paid at most once
# per beat, not once per consumer per repo.
CHARLIE_STATUS_TIMEOUT_SECONDS = 120
# Issue #1463: ``fleet status --json`` serves from the status-snapshot cache
# by default. A cache older than this threshold means the blocked set may not
# reflect the current state — surface a staleness warning so downstream checks
# annotate their output as degraded rather than silently trusting stale data.
#
# The threshold must sit ABOVE the producer's serving ceiling, not inside it
# (issue #1886). ``read_status_snapshot`` only serves a snapshot younger than
# ``runtime.status_snapshot_ttl_seconds`` (default 900s), so every served
# ``cache_age_seconds`` is already <= that TTL — while the measured healthy
# per-repo refresh gap (the same series LOOP_PASS_STALE_MINUTES cites: median
# ~10.4m, p90 ~20m) routinely lands served ages in the upper TTL band. The
# original 600s therefore flagged normal operation on every repo, every tick
# (observed 881s fleet-wide, 2026-09-24). 1800s = 2x the default TTL: under
# default config this can only fire when a deployment widened the TTL past
# 30 minutes and is genuinely serving blocked data that old — the degraded
# case this warning exists for.
STATUS_CACHE_STALE_SECONDS = 1800

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
SUPERVISOR_HEARTBEAT_STALE_MULTIPLIER = _haf.SUPERVISOR_HEARTBEAT_STALE_MULTIPLIER if _haf else 2
SUPERVISOR_HEARTBEAT_DEFAULT_PASS_TIMEOUT_SECONDS = (
    _haf.SUPERVISOR_HEARTBEAT_DEFAULT_PASS_TIMEOUT_SECONDS if _haf else 1800
)

# Issue #1832: a ``supervisor_wedge_loop`` event means the wedge-kill
# backstop (issue #728) fired N consecutive times with no fleet pass
# recovering in between -- the relaunch loop itself is stuck, which is
# exactly the condition a stale-heartbeat check alone cannot distinguish
# from "still degraded but eventually recovering." Only events within this
# lookback window count, so a single old occurrence that has since resolved
# does not alarm forever.
SUPERVISOR_WEDGE_LOOP_EVENT_KIND = "supervisor_wedge_loop"
SUPERVISOR_WEDGE_LOOP_LOOKBACK_HOURS = _haf.SUPERVISOR_WEDGE_LOOP_LOOKBACK_HOURS if _haf else 24

# Issue #1859: notify_digest_check is the stdlib-only file-probe leaf;
# notify_digest_heartbeat holds the events.db-read-plus-verdict logic (see
# its docstring), both behind the same guarded import as event_kinds.
# NOTIFY_DIGEST_STALE_HOURS is re-exported for tests.
try:
    from charlie_work import notify_digest_check as _ndc
    from charlie_work import notify_digest_heartbeat as _ndh
except ImportError:
    _ndc = None
    _ndh = None

NOTIFY_DIGEST_STALE_HOURS = _ndc.NOTIFY_DIGEST_STALE_HOURS if _ndc else 72
NOTIFY_RESOLUTION_EVENT_KIND = "notify_resolution"
NOTIFY_DIGEST_STALE_EVENT_KIND = "notify_digest_stale"

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
    # Resolved by load_repos via heartbeat_local_repo (#1861); False = GitHub-backed.
    local_issues_enabled: bool = False


@dataclass(frozen=True)
class SuppressionEntry:
    """One entry from `scripts/heartbeat-suppressions.yaml` (issue #1361).

    `check` is the *base* check name as emitted, with no repo suffix (e.g.
    ``"stale-open-issue-mentions"``, never ``"stale-open-issue-mentions
    owner/charlie-work"``). Per-repo checks build their emitted check
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
    # local_issues.enabled may live in the fleet layer, not the repo's own config (#1861).
    fleet_config, _fleet_err = load_orchestrator_config(fleet_dir() / FLEET_GLOBAL_CONFIG_FILENAME)
    repos: list[RepoInfo] = []
    for slug, entry in data.get("repos", {}).items():
        try:
            config_path = Path(entry.get("config_path", ""))
            repo_config, _config_err = load_orchestrator_config(config_path)
            repos.append(
                RepoInfo(
                    slug=slug,
                    repo_root=Path(entry["repo_root"]),
                    state_dir=Path(entry["state_dir"]),
                    config_path=config_path,
                    local_issues_enabled=_local_issues_enabled(repo_config, fleet_config),
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
    # Issue #1463: surface stale-cache degradation. ``fleet status --json``
    # serves from the status-snapshot cache by default; a very stale cache
    # means the blocked set may not reflect the current state. The blocked
    # data is still returned (it is the best available), but a staleness
    # warning is returned so downstream checks annotate their output.
    ages = {
        slug: a
        for slug, r in repos.items()
        if isinstance(a := r.get("cache_age_seconds"), (int, float))
    }
    stale = {slug: a for slug, a in ages.items() if a > STATUS_CACHE_STALE_SECONDS}
    if stale:
        # blocked_err is one fleet-wide string threaded into every repo's
        # check lines; name the worst offender so the caveat identifies which
        # repo's snapshot actually drove it.
        slug, age = max(stale.items(), key=lambda item: item[1])
        return blocked_by_repo, (
            f"status snapshot cache {age:.0f}s old for {slug} "
            f"(>{STATUS_CACHE_STALE_SECONDS}s); blocked set may be stale"
        )
    return blocked_by_repo, ""


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
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
# The worktree-path helpers themselves moved to the sibling module
# scripts/heartbeat_worktree.py for file-size-ratchet headroom and are
# re-exported above. They are stdlib-only reimplementations of
# layout/worktree helpers this script cannot import (see
# scripts/README.md's "stdlib-only" invariant). They mirror:
#   - charlie_work.layout.worktrees_dir(state_root) -> state_root / "worktrees"
#   - charlie_work.worktree._slugify(branch)
#   - charlie_work.worktree.worktree_path_for_branch(root, branch, worktrees_dir)
# If those ever diverge, tests/test_heartbeat_check_in_progress_stale.py's
# worktree-mtime tests will catch it (they build the worktree dir the same
# way the orchestrator does, via the same slugify, so a slug mismatch
# surfaces as a missing dir).


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


def get_mergequeue_label(config_path: Path) -> str | None:
    config, _error = load_orchestrator_config(config_path)
    return config.get("auto_merge", {}).get("mergequeue_label")


def get_dispatch_cap(config_path: Path) -> int | None:
    """Read dispatch.max_concurrent_sessions (per-repo concurrency cap), or None."""
    config, _error = load_orchestrator_config(config_path)
    cap = config.get("dispatch", {}).get("max_concurrent_sessions")
    return cap if isinstance(cap, int) else None


def _last_dispatch_drain_signal(state_dir: Path) -> dict[str, Any] | None:
    """Read the last ``dispatch`` event's concurrency governor from events.db.

    Returns the parsed payload dict of the most recent ``dispatch`` event in
    ``state_dir/events.db`` (which carries ``concurrency_governor`` and
    ``deferred_by_concurrency_count``), or ``None`` when no events.db exists,
    the events table is absent, no ``dispatch`` row is recorded, or the
    payload is unparseable. Mirrors the stdlib-only events.db access pattern
    in ``check_loop_pass_freshness`` / ``check_error_events`` rather than
    importing ``charlie_work.instrumentation`` -- this script stays importable
    when ``ci_fleet`` is not installed (see the module header).

    The ``concurrency_governor`` sub-payload is written by
    ``workflow.py``'s dispatch path and carries ``dispatch_limit``,
    ``fleet_concurrency_limit`` / ``fleet_live_session_count`` (present only
    when the fleet governor is enabled), and the per-repo
    ``concurrency_limit`` / ``live_session_count`` / ``available_slots``.
    ``deferred_by_concurrency_count`` is a sibling top-level field counting
    every ordered candidate the cap turned away this pass.
    """
    db_path = state_dir / "events.db"
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error:
        return None
    try:
        try:
            table_row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            if table_row is None:
                return None
            # ORDER BY id DESC: id is AUTOINCREMENT, so the highest id is the
            # most recent insert. This avoids the ISO-T/Z-vs-SQLite-space
            # timestamp-format trap documented on check_loop_pass_freshness
            # (lexicographic ts ordering happens to work for ISO-8601, but id
            # is the insertion order and is unambiguously monotonic).
            row = conn.execute(
                "SELECT payload FROM events WHERE kind = 'dispatch' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        except sqlite3.Error:
            return None
    finally:
        conn.close()
    if row is None:
        return None
    try:
        payload = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _governor_drain_note(
    persisting_count: int,
    governor: dict[str, Any],
    deferred_by_concurrency_count: int | None,
) -> str | None:
    """Build the ``draining at cap`` note from a dispatch event's governor.

    Returns a human-readable drain note when the governor shows the backlog
    is draining at the allowed rate (fleet-wide cap saturated, or the
    effective dispatch limit was at or below the deferred backlog), or
    ``None`` when the governor shows free slots -- the caller keeps the
    anomaly in that case (issue #1424: the #1398 head-of-line case is real).

    The condition mirrors the one stated in issue #1424:
    ``fleet_live_session_count >= fleet_concurrency_limit`` (when the fleet
    governor is enabled) OR ``dispatch_limit <= deferred_by_concurrency_count``.
    """
    cg = governor.get("concurrency_governor") if isinstance(governor, dict) else None
    if not isinstance(cg, dict):
        return None
    fleet_cap = cg.get("fleet_concurrency_limit")
    fleet_live = cg.get("fleet_live_session_count")
    dispatch_limit = cg.get("dispatch_limit")

    fleet_at_cap = (
        isinstance(fleet_cap, int) and isinstance(fleet_live, int) and fleet_live >= fleet_cap
    )
    deferred_at_cap = (
        isinstance(dispatch_limit, int)
        and isinstance(deferred_by_concurrency_count, int)
        and dispatch_limit <= deferred_by_concurrency_count
    )

    if not (fleet_at_cap or deferred_at_cap):
        return None

    parts = [f"backlog={persisting_count} draining at cap"]
    if fleet_at_cap:
        parts.append(f"fleet_live={fleet_live}/fleet_cap={fleet_cap}")
    if deferred_at_cap and isinstance(dispatch_limit, int):
        parts.append(f"dispatch_limit={dispatch_limit}")
    return " ".join(parts)


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
    if repo.local_issues_enabled:
        _local_repo.skip_dispatch_coverage(report, check, prev_repo_state, new_repo_state)
        # Sub-checks below are state-file, not gh -- throttle always prints; staleness self-skips.
        check_dispatch_throttle(report, repo, now=now)
        check_in_progress_staleness(
            report, repo, [], prev_repo_state, new_repo_state, skip_delta, now=now
        )
        return
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
            # Issue #1424: the governor that actually bounds dispatch is
            # fleet-wide (fleet_live_session_count vs fleet_concurrency_limit
            # across all repos), not the per-repo cap. A repo at 2/5 with the
            # sibling repo holding the other slots reads as a dispatch failure
            # under the per-repo denominator when it is designed behaviour.
            # Derive the drain condition from the last dispatch event's
            # concurrency_governor payload in this repo's events.db; fall back
            # to the per-repo cap check only when no governor data is
            # available (fresh install with no dispatch events yet).
            last_dispatch = _last_dispatch_drain_signal(repo.state_dir)
            if last_dispatch is not None:
                deferred_count = last_dispatch.get("deferred_by_concurrency_count")
                drain_note = _governor_drain_note(len(persisting), last_dispatch, deferred_count)
                if drain_note is None:
                    # The governor shows free slots and the issues still
                    # persist -- the #1398 head-of-line case, which is real.
                    reasons.append(
                        f"issue(s) {persisting} dispatchable across 2 consecutive beats "
                        "(threshold: must clear within 1 beat)"
                    )
            elif cap is not None and len(in_progress) >= cap:
                # Fallback: no dispatch event in events.db yet (fresh install).
                # Backlog exceeding the drain rate while every dispatch slot
                # is occupied is designed behavior, not a dispatch failure.
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

    if repo.local_issues_enabled:
        _local_repo.skip_in_progress_staleness(report, check, prev_map, new_repo_state)
        return

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

    Detect this case structurally: a pending on-disk decision while status is
    ``review_dispatch_completed`` is ALREADY the contradiction that proves a
    packet rebuild (``record_review`` writes a terminal decision; only a packet
    build resets it to ``pending``).  Anchor on the packet-rebuild evidence --
    the ``review-prompt.md`` mtime, rewritten on every ``review()`` packet
    build -- instead of the stale state.json dispatch timestamp.  ``pr_dir``
    and ``decision`` are optional so direct unit callers (and the pre-fix
    tests) keep the original state.json-only behavior.

    Issue #1436: the #1403 fix gated the rebuild anchor on the on-disk
    decision's ``reviewed_head_sha`` DIFFERING from state.json's.  That misses
    a packet rebuilt for the SAME head (conflict-path re-review,
    verdict-missed retry, manual re-request): the SHA equality lets the guard
    fall through and the newest-timestamp fallback dates the claim by the
    prior cycle's ``review_dispatched_at`` (false 467m ANOMALY on pr-1432).
    The SHA comparison is now only a tiebreak for the missing-prompt-file
    case; the primary gate is ``status == completed and decision == pending``.

    The prompt mtime is guarded against being OLDER than the prior cycle's
    ``review_dispatched_at`` (clock skew / a packet that was not actually
    rebuilt): the newer of the two is used, so this can only shrink a false
    age and never hide a genuinely stale claim from before the completed
    cycle.
    """
    status = pr_state.get("review_dispatch_status")
    if status == "review_dispatch_dispatched":
        return pr_state.get("review_dispatched_at")
    if status == "review_dispatch_pending":
        return pr_state.get("review_dispatch_pending_at")
    if status == "review_dispatch_failed":
        return pr_state.get("review_dispatch_failed_at")

    # Issue #1403/#1436: completed prior cycle whose packet was rebuilt.  See
    # the docstring for why state.json's dispatch timestamps are stale here.
    # A pending on-disk decision while status is completed is the structural
    # contradiction that proves a rebuild -- regardless of whether the head
    # advanced (#1403, differing SHA) or stayed put (#1436, same SHA).
    if (
        status == "review_dispatch_completed"
        and pr_dir is not None
        and isinstance(decision, dict)
        and decision.get("decision") == "pending"
    ):
        prompt_path = pr_dir / "review-prompt.md"
        if prompt_path.exists():
            prompt_dt = datetime.fromtimestamp(prompt_path.stat().st_mtime, tz=timezone.utc)
            # Guard against the prompt mtime being OLDER than the prior
            # cycle's dispatch (clock skew / unrebuilt packet): use the newer
            # of the two, so this can only shrink a false age and never hide a
            # genuinely stale claim from before the completed cycle.
            dispatch_raw = pr_state.get("review_dispatched_at")
            dispatch_dt = parse_iso(dispatch_raw) if dispatch_raw else None
            if dispatch_dt is not None and dispatch_dt > prompt_dt:
                return dispatch_raw
            return prompt_dt.isoformat().replace("+00:00", "Z")
        # Prompt file missing: the SHA comparison survives only as a tiebreak
        # here.  A differing SHA still indicates a rebuilt packet, but without
        # the prompt mtime we have no rebuild timestamp; fall back to current
        # behavior (the newest-timestamp fallback below) either way.

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

    if repo.local_issues_enabled:
        _local_repo.skip_review_liveness(report, check)
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
    """``now`` is the injectable clock (issue #828); see ``check_dispatch_throttle``.

    Reads the freshest log/state/checkpoint mtime under the state dir; the
    verdict is ``heartbeat_alarms_fleet.eval_log_freshness``.
    """
    check = f"log-freshness {repo.slug}"
    if _haf is None:
        report.anom(check, _ALARMS_UNAVAILABLE)
        return
    candidates = list(repo.state_dir.glob("*.log"))
    state_json = repo.state_dir / "state.json"
    if state_json.exists():
        candidates.append(state_json)
    candidates.extend(repo.state_dir.glob("*checkpoint*"))
    candidates = [c for c in candidates if c.is_file()]
    freshest = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    _ha.emit(
        report,
        _haf.eval_log_freshness(
            repo.slug,
            freshest.stat().st_mtime if freshest else None,
            freshest.name if freshest else "",
            now if now is not None else datetime.now(timezone.utc),
            LOG_FRESHNESS_STALE_MINUTES,
        ),
    )


def check_loop_pass_freshness(
    report: Report,
    repo: RepoInfo,
    *,
    now: datetime | None = None,
    stale_minutes: int = LOOP_PASS_STALE_MINUTES,
) -> None:
    """Coarse backstop for prolonged, total fleet death -- NOT a detector
    for the #851/#854 outage class specifically (see ``LOOP_PASS_STALE_MINUTES``
    for the measured cadence data; PR #865 / issue #855 catches that class).

    What this still catches: the loop body (``workflow.py``'s ``_loop_impl``,
    the only place that logs ``loop_started``) has not run for well over an
    hour while the state dir keeps getting touched, so ``check_log_freshness``
    reads healthy. Ground truth is the ABSENCE of ``loop_started`` rows.

    ``now`` is the injectable clock (issue #828). Missing DB, missing table, and
    zero rows are each OK with a distinct message (fresh install is not a dead
    fleet). ``ts`` is compared in Python, never SQL (ISO ``T``/``Z`` vs SQLite
    space-format mis-compares silently). Verdict:
    ``heartbeat_alarms_fleet.eval_loop_pass_freshness``.
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

    if _haf is None:
        report.anom(check, _ALARMS_UNAVAILABLE)
        return
    _ha.emit(
        report,
        _haf.eval_loop_pass_freshness(
            repo.slug,
            newest_ts,
            now if now is not None else datetime.now(timezone.utc),
            stale_minutes,
            marker_hint=str(repo.state_dir / "pending-sync.json"),
        ),
    )


def check_supervisor_venv_refusal(
    report: Report, repos: list[RepoInfo], baseline: datetime
) -> None:
    """Surface ``venv_editable_anchor_violation`` as a real alert (issue #1366).

    The supervisor's startup guard (issue #974) refuses to run when an
    editable ``.pth`` in the interpreter's venv points outside the
    interpreter-derived checkout -- everything below that refusal would run
    unreviewed code. The refusal returns before the supervisor writes its
    heartbeat, so ``check_supervisor_heartbeat``'s freshness check only
    catches the *indirect* symptom (a stale or absent heartbeat); the
    violation event is the only direct evidence of *why* the supervisor is
    not running. Before this check that event had no automated reader beyond
    the log line written at the refusal -- a silent refusal could stall the
    whole supervisor with nothing but a log line as evidence.

    This is the real-alert path the operator comment on issue #1366 reserved
    for this kind: a new ``venv_editable_anchor_violation`` event since the
    last beat is an ANOMALY (flips the exit code), not a passive WARN -- the
    cost of an unnoticed supervisor refusal is much higher than the cost of
    an unnoticed draft PR. It runs alongside ``check_supervisor_heartbeat``
    in the same supervisor-health section of the heartbeat output.

    The event is logged to a per-repo ``events.db`` (``run_fleet_supervise``
    logs to the orchestrator repo's own state dir; ``run_supervised`` logs to
    the supervised repo's), so this check scans every registered repo's
    ``events.db`` and aggregates the findings into one fleet-level alert. A
    repo with no ``events.db`` (or one with no ``events`` table) is skipped,
    not flagged: this check looks for an actual refusal event, and a repo
    that has never been supervised has none to find -- the
    ``check_supervisor_heartbeat`` freshness check separately vouches for
    supervisor liveness.
    """
    check = "supervisor-venv-refusal"
    offending: list[str] = []
    for repo in repos:
        db_path = repo.state_dir / "events.db"
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
        except sqlite3.Error:
            continue
        try:
            rows = conn.execute(
                "SELECT ts FROM events WHERE kind = ?",
                ("venv_editable_anchor_violation",),
            ).fetchall()
        except sqlite3.Error:
            conn.close()
            continue
        conn.close()
        for (ts,) in rows:
            ts_dt = parse_iso(ts)
            if ts_dt is None or ts_dt > baseline:
                offending.append(f"{repo.slug}@{ts}")

    if offending:
        report.anom(
            check,
            f"venv_editable_anchor_violation since last beat: {offending} "
            "(supervisor refused to start -- editable .pth anchor violated, "
            "issue #974)",
        )
    else:
        report.ok(check, f"refusals_since_last_beat=0 repos_scanned={len(repos)}")


def check_merge_flow(
    report: Report,
    repo: RepoInfo,
    prev_repo_state: dict[str, Any],
    new_repo_state: dict[str, Any],
    skip_delta: bool,
) -> None:
    check = f"merge-flow {repo.slug}"
    if repo.local_issues_enabled:
        _local_repo.skip_merge_flow(report, check, prev_repo_state, new_repo_state)
        return

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

    1. One `gh issue list --state open --json number,labels` call for the
       candidate set. `labels` is data on the returned records, not a query
       filter -- the candidate set is still every open issue; the per-issue
       label set only feeds exclusion (2) below.
    2. One `gh pr list --state merged --json number,headRefName,title,body,
       closingIssuesReferences,mergedAt,author` call (bounded by
       `STALE_MENTION_PR_LOOKBACK_LIMIT`), scanned for a branch-name issue
       number (`_branch_issue_number`) or a bare `#N` mention
       (`issue_mention_occurrences`) in title/body. This is what would catch
       #817: PR #824's branch is `fix/817-fleet-health-latch` (no
       `agent/issue` prefix, so `linked_issue_number` never trusts it) and
       its body reads "For issue #817:" (a reference, but not a closing
       keyword, so `closingIssuesReferences` came back empty on the PR
       itself).
    3. Local `git log` on the already-checked-out branch (`get_merged_commit_messages`,
       bounded by `STALE_MENTION_COMMIT_LOOKBACK`) -- zero API cost, and the
       only source that can catch a #866-shape fix: work that landed as a
       commit inside a PR *for a different issue*, so no scan of any PR's
       own title/body/branch name could ever find it.

    `closingIssuesReferences` is fetched (per the issue's specified command
    shape) but not used to gate reporting: since the candidate set is
    already restricted to *currently open* issues, any issue GitHub's native
    auto-close already resolved via a real closing-keyword match is
    definitionally no longer in that set. No extra filtering on that field
    can change which issues get reported here.

    Quoted or negated mentions are excluded (`issue_mention_occurrences`),
    consistent with #781/#790: a mention is evidence for a human to check,
    never grounds to act automatically, and a quoted/negated one is not even
    that.

    Issue #2048 exclusions -- a mention stops counting as closure evidence
    when any of these holds (the check measured ~0% precision, 0/52
    closable, on 2026-09-30; see the issue for the recurring false-positive
    shapes):

    1. The mention sits in a non-closing context -- `Refs`/`Ref`/`Related`/
       `See`/`follow-up`/`deferred`/`part of`/`filed as` in the clause
       preceding it (`_is_non_closing_context`, per occurrence).
    2. The issue carries a parked or active label: a parked label from the
       repo's `heartbeat: stale_mention_parked_labels` config knob
       (`HeartbeatConfig`, mirrored for this stdlib-only script as
       `STALE_MENTION_PARKED_LABELS_DEFAULT` -- `parked_label_names`
       resolves configured-vs-default), or a fleet lifecycle label (the
       `agent:` prefix namespace, plus the repo's `labels:` config section
       values minus `ready` -- `lifecycle_label_names` derives that set
       from LabelConfig's serialized form; configured renames off the
       `agent:` prefix still count). The default parked set deliberately
       matches `ARMABLE_GATING_LABELS`' triage taxonomy plus the
       tracker/umbrella/epic names the issue enumerates.
    3. The mentioning PR is bot-authored (`pr_author_is_bot` -- dependabot/
       renovate and every other `[bot]` account).

    Every suppression is counted and surfaced in the facts line
    (`excluded_refs`/`excluded_labels`/`excluded_bots`) so a silent zero is
    auditable, never invisible. A malformed `orchestrator.config.yaml`
    (`load_orchestrator_config` error) is likewise surfaced in the facts
    line as `config load degraded` rather than swallowed -- the exclusion
    machinery then runs on the mirrored default parked set only. Output is
    capped at `STALE_MENTION_REPORT_CAP` issues (with a "+K more" suffix)
    so a large true positive count cannot flood the beat.
    """
    check = f"stale-open-issue-mentions {repo.slug}"
    if repo.local_issues_enabled:
        report.ok(check, LOCAL_ONLY_SKIP_DETAIL)
        return

    ok_open, open_data, err_open = run_gh_json(
        [
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
        ],
        repo.repo_root,
    )
    if not ok_open:
        report.anom(check, err_open)
        return
    open_labels: dict[int, set[str]] = {
        issue["number"]: {label["name"] for label in issue.get("labels", [])}
        for issue in open_data
    }

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
            "number,headRefName,title,body,closingIssuesReferences,mergedAt,author",
        ],
        repo.repo_root,
    )
    if not ok_merged:
        report.anom(check, err_merged)
        return

    ok_commits, commits, err_commits = get_merged_commit_messages(
        repo.repo_root, STALE_MENTION_COMMIT_LOOKBACK
    )

    config, config_err = load_orchestrator_config(repo.config_path)
    managed_labels = parked_label_names(config.get("heartbeat")) | lifecycle_label_names(
        config.get("labels")
    )

    mentions: dict[int, list[str]] = {}
    # Excluded-mention tallies, surfaced in the facts line below so a silent
    # zero stays auditable (issue #2048): refs = non-closing-context
    # occurrences, bots = occurrences dropped because the PR is bot-authored.
    excluded_counts = {"refs": 0, "bots": 0}

    def record(number: int, evidence: str, excluded: str | None = None) -> None:
        if number not in open_labels:
            return
        if excluded is not None:
            excluded_counts[excluded] += 1
            return
        mentions.setdefault(number, []).append(evidence)

    for pr in merged_data:
        pr_number = pr.get("number")
        bot = pr_author_is_bot(pr.get("author"))
        branch = str(pr.get("headRefName") or "")
        branch_issue = _branch_issue_number(branch)
        if branch_issue is not None:
            record(
                branch_issue,
                f"PR #{pr_number} branch {branch!r}",
                "bots" if bot else None,
            )
        text = f"{pr.get('title') or ''}\n{pr.get('body') or ''}"
        for number, non_closing in issue_mention_occurrences(text):
            record(
                number,
                f"PR #{pr_number} title/body",
                "bots" if bot else ("refs" if non_closing else None),
            )

    for sha, message in commits:
        for number, non_closing in issue_mention_occurrences(message):
            record(number, f"commit {sha}", "refs" if non_closing else None)

    label_exempt = {
        number
        for number in mentions
        if mention_exempt_by_label(open_labels[number], managed_labels)
    }
    for number in label_exempt:
        del mentions[number]

    facts = (
        f"open={len(open_labels)} merged_prs_scanned={len(merged_data)} "
        f"commits_scanned={len(commits)} excluded_refs={excluded_counts['refs']} "
        f"excluded_labels={len(label_exempt)} excluded_bots={excluded_counts['bots']}"
    )
    if not ok_commits:
        facts += f" (commit-message scan degraded: {err_commits})"
    if config_err is not None:
        # A malformed config must not be silently swallowed: the check still
        # runs (parked_label_names falls back to the mirrored default and
        # lifecycle_label_names contributes nothing on a non-mapping), but the
        # facts line records that the configured knob was not consulted.
        facts += f" (config load degraded: {config_err})"

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
    loop iteration; a stale ``last_beat_at`` means it is killed (``exited_at``
    null) or cleanly stopped but not restarted (``exited_at`` set -- the
    2026-07-25 watchdog-disabled shape). The file's age is the only signal
    left when the launcher died too. This reads the file; the verdict (stale
    threshold derived from the heartbeat's own ``max_pass_runtime_seconds``) is
    ``heartbeat_alarms_fleet.eval_supervisor_heartbeat``.
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
    if _haf is None:
        report.anom(check, _ALARMS_UNAVAILABLE)
        return
    _ha.emit(report, _haf.eval_supervisor_heartbeat(data, datetime.now(timezone.utc)))


def check_wedge_kill_loop(report: Report) -> None:
    """Surface a ``supervisor_wedge_loop`` event -- the wedge-kill backstop looping (issue #1832).

    ``supervisor_lifecycle.detect_wedge_kill_loop`` records this error-level
    event in the FLEET-level ``events.db`` (a sibling of
    ``supervisor-heartbeat.json``), which ``check_error_events`` -- scanning
    only registered repos' state dirs -- never sees. This is a thin read; the
    streak logic stays in that one implementation, and the verdict (events
    within the lookback window) is ``heartbeat_alarms_fleet.eval_wedge_kill_loop``.

    A missing fleet ``events.db`` is OK, not an anomaly: the supervisor may never
    have started, and ``check_supervisor_heartbeat`` owns that failure.
    """
    check = "supervisor-wedge-kill-loop"
    db_path = fleet_dir() / "events.db"
    if not db_path.exists():
        report.ok(check, "no fleet events.db yet (supervisor has never logged an event)")
        return

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        report.anom(check, f"cannot check for wedge-kill loop: fleet events.db unreadable: {exc}")
        return

    try:
        try:
            table_row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
            if table_row is None:
                report.ok(check, "fleet events.db has no events table yet")
                return
            rows = conn.execute(
                "SELECT ts FROM events WHERE kind = ?",
                (SUPERVISOR_WEDGE_LOOP_EVENT_KIND,),
            ).fetchall()
        except sqlite3.Error as exc:
            report.anom(
                check, f"cannot check for wedge-kill loop: fleet events.db unreadable: {exc}"
            )
            return
    finally:
        conn.close()

    if _haf is None:
        report.anom(check, _ALARMS_UNAVAILABLE)
        return
    _ha.emit(
        report,
        _haf.eval_wedge_kill_loop(
            [ts for (ts,) in rows],
            datetime.now(timezone.utc),
            SUPERVISOR_WEDGE_LOOP_LOOKBACK_HOURS,
        ),
    )


def check_notify_digest_freshness(report: Report, *, now: datetime | None = None) -> None:
    """Issue #1859 consumer: flag an enabled file sink whose digest is dead.

    Thin wrapper around ``notify_digest_heartbeat.check_notify_digest_freshness``
    (see its docstring); only job here is the guarded-leaf contract -- WARN,
    never raise, when the delegate modules aren't importable or the call raises.
    """
    if _ndc is None or _ndh is None:
        report.warn(
            "notify-digest", "check unavailable: charlie_work.notify_digest_check not importable"
        )
        return
    try:
        _ndh.check_notify_digest_freshness(
            report,
            now=now if now is not None else datetime.now(timezone.utc),
            ndc=_ndc,
            fleet_dir=fleet_dir(),
            parse_iso=parse_iso,
            stale_hours=NOTIFY_DIGEST_STALE_HOURS,
        )
    except Exception as exc:
        report.warn("notify-digest", f"check failed unexpectedly: {exc!r}")


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
        check_draft_pr_blocked_events(report, repo, baseline)
        check_ci_headroom_unavailable(report, repo, baseline)
        check_local_lane_kill_switch_stalled(report, repo, baseline)
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
    check_supervisor_venv_refusal(report, repos, baseline)
    check_supervisor_heartbeat(report)
    check_wedge_kill_loop(report)
    check_notify_digest_freshness(report, now=now)

    save_state(new_state)

    summary = report.suppression_summary()
    if summary is not None:
        report.ok("suppressions", summary)

    print("\n".join(report.lines))
    return 1 if report.anomaly else 0


if __name__ == "__main__":
    sys.exit(main())
