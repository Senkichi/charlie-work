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
Exit code 0 if no anomalies, 1 if any.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
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


@dataclass
class Report:
    lines: list[str] = field(default_factory=list)
    anomaly: bool = False

    def ok(self, check: str, facts: str) -> None:
        self.lines.append(f"OK {check}: {facts}")

    def anom(self, check: str, detail: str) -> None:
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
        """
        self.lines.append(f"WARN {check}: {detail}")


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
        report, repo, in_progress, prev_repo_state, new_repo_state, skip_delta
    )


def check_in_progress_staleness(
    report: Report,
    repo: RepoInfo,
    in_progress: list[tuple[int, str | None]],
    prev_repo_state: dict[str, Any],
    new_repo_state: dict[str, Any],
    skip_delta: bool,
) -> None:
    """Flag agent:in-progress issues whose updatedAt hasn't moved across 2 beats.

    Persists {issue_number: updatedAt} per repo in the state file so the next
    beat can compare. Entries for issues no longer in-progress are pruned
    automatically since cur_map is rebuilt fresh from this beat's data.
    """
    check = f"in-progress-stale {repo.slug}"
    prev_map: dict[str, str] = prev_repo_state.get("in_progress", {})

    if skip_delta:
        # Leave the last real beat's snapshot untouched; this is not a real
        # comparison interval.
        new_repo_state["in_progress"] = prev_map
        report.ok(check, f"tracked={len(prev_map)} stale=0{DELTA_SKIP_SUFFIX}")
        return

    cur_map: dict[str, str] = {}
    stale: list[int] = []
    for number, updated_at in in_progress:
        if updated_at is None:
            continue
        cur_map[str(number)] = updated_at
        if prev_map.get(str(number)) == updated_at:
            stale.append(number)

    new_repo_state["in_progress"] = cur_map

    if stale:
        report.anom(check, f"issue(s) {sorted(stale)} no activity across 2 beats (threshold: 2)")
    else:
        report.ok(check, f"tracked={len(cur_map)} stale=0")


def _claim_is_open(decision_path: Path) -> bool:
    """A review claim is OPEN until a non-pending decision is recorded.

    Claims write a placeholder review-decision.json with decision="pending"
    at claim time, then overwrite it once the review actually completes. A
    missing file, a still-"pending" file, or an unparseable file all mean the
    claim has not been resolved yet.
    """
    if not decision_path.exists():
        return True
    try:
        data = json.loads(decision_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if not isinstance(data, dict):
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


def _review_claim_timestamp(pr_state: dict[str, Any]) -> str | None:
    """Return the most relevant claim timestamp for an open review.

    The orchestrator's ground truth is ``state.json``:

    * ``review_dispatch_dispatched`` -> ``review_dispatched_at``
    * ``review_dispatch_pending``    -> ``review_dispatch_pending_at``
    * ``review_dispatch_failed``     -> ``review_dispatch_failed_at``

    For unknown/missing status, fall back to the newest present timestamp.  This
    replaces the packet-directory ``st_mtime`` that never updates across redispatch
    retries (issue #517).
    """
    status = pr_state.get("review_dispatch_status")
    if status == "review_dispatch_dispatched":
        return pr_state.get("review_dispatched_at")
    if status == "review_dispatch_pending":
        return pr_state.get("review_dispatch_pending_at")
    if status == "review_dispatch_failed":
        return pr_state.get("review_dispatch_failed_at")

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
        if not _claim_is_open(entry / "review-decision.json"):
            continue

        open_claims += 1

        pr_state = prs_state.get(str(pr_number), {}) if isinstance(prs_state, dict) else {}
        if not isinstance(pr_state, dict):
            pr_state = {}

        timestamp = _review_claim_timestamp(pr_state)
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
    report = Report()

    repos, load_err = load_repos()
    if load_err:
        report.anom("fleet-registry", load_err)
        print("\n".join(report.lines))
        return 1

    prev_state = load_state()
    prev_last_beat_at = parse_iso(prev_state.get("last_beat_at"))
    # Sampled once for this entire beat (issue #828) and threaded into every
    # sub-check below instead of each one independently racing the wall
    # clock -- keeps all checks in one run reporting against a single
    # consistent instant, and makes the run's own last_beat_at exact.
    now = datetime.now(timezone.utc)
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
        check_review_liveness(report, repo, now=now)
        check_dispatch_failures(report, repo, baseline)
        check_error_events(report, repo, baseline)
        check_warning_events(report, repo, baseline)
        check_log_freshness(report, repo, now=now)
        check_loop_pass_freshness(report, repo, now=now)
        check_merge_flow(report, repo, prev_repo_state, new_repo_state, skip_delta)
        check_stale_open_issue_mentions(report, repo)
        new_state["repos"][repo.slug] = new_repo_state

    if repos:
        check_github_rate(report, repos[0].repo_root)
    else:
        report.anom("github-rate", "no repos registered, cannot resolve a cwd for gh")

    check_runners(report)
    check_supervisor_heartbeat(report)

    save_state(new_state)

    print("\n".join(report.lines))
    return 1 if report.anomaly else 0


if __name__ == "__main__":
    sys.exit(main())
