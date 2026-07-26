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
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psutil
import yaml

# --------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------

GH_TIMEOUT_SECONDS = 30
ISSUE_LIST_LIMIT = 200
MERGED_PR_LOOKBACK_LIMIT = 5

QUEUED_STALE_MINUTES = 20
REVIEW_CLAIM_STALE_MINUTES = 45
LOG_FRESHNESS_STALE_MINUTES = 30
MERGEQUEUE_STALL_BEATS = 2
GRAPHQL_RATE_LIMIT_MIN_REMAINING = 500
DISPATCH_THROTTLE_MAX_MINUTES = 30
MIN_BEAT_INTERVAL_MINUTES = 10
CHARLIE_STATUS_TIMEOUT_SECONDS = 60

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


def load_orchestrator_config(config_path: Path) -> dict[str, Any]:
    """Load an orchestrator.config.yaml. Returns {} on any read/parse failure."""
    try:
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def get_mergequeue_label(config_path: Path) -> str | None:
    return load_orchestrator_config(config_path).get("auto_merge", {}).get("mergequeue_label")


def get_dispatch_cap(config_path: Path) -> int | None:
    """Read dispatch.max_concurrent_sessions (per-repo concurrency cap), or None."""
    cap = load_orchestrator_config(config_path).get("dispatch", {}).get("max_concurrent_sessions")
    return cap if isinstance(cap, int) else None


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def check_dispatch_throttle(report: Report, repo: RepoInfo) -> None:
    """Report the provider throttle cooldown (state.json's throttled_until).

    Being throttled is normal self-protection (OK-level), but the line must
    always print so a zero-dispatch beat is instantly explainable. Only an
    unusually long cooldown (beyond DISPATCH_THROTTLE_MAX_MINUTES) is an anomaly.
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
    now = datetime.now(timezone.utc)
    if until is None or until <= now:
        report.ok(check, "none")
        return

    remaining_min = round((until - now).total_seconds() / 60)
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
) -> None:
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

    now = datetime.now(timezone.utc)
    stale_queued = []
    for number, updated in queued:
        if updated is None:
            continue
        age_min = (now - updated).total_seconds() / 60
        if age_min > QUEUED_STALE_MINUTES:
            stale_queued.append((number, round(age_min)))
    if stale_queued:
        reasons.append(
            f"agent:queued stuck {stale_queued} minutes (threshold={QUEUED_STALE_MINUTES}m)"
        )

    facts = f"dispatchable={len(dispatchable)} queued={len(queued)} in_progress={len(in_progress)}"
    if cap is not None:
        facts += f" cap={cap}"
    if blocked_err:
        facts += f" (blocked-issue lookup degraded: {blocked_err})"
    if drain_note:
        facts = f"{drain_note}; {facts}"
    if skip_delta:
        facts += DELTA_SKIP_SUFFIX

    if reasons:
        report.anom(check, f"{'; '.join(reasons)} ({facts})")
    else:
        report.ok(check, facts)

    check_dispatch_throttle(report, repo)
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


def check_review_liveness(report: Report, repo: RepoInfo) -> None:
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

    now = datetime.now(timezone.utc)
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

        age_min = (now - claim_time).total_seconds() / 60
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


def check_log_freshness(report: Report, repo: RepoInfo) -> None:
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
    now = datetime.now(timezone.utc)
    mtime = datetime.fromtimestamp(freshest.stat().st_mtime, tz=timezone.utc)
    age_min = (now - mtime).total_seconds() / 60

    facts = f"freshest={freshest.name} age={round(age_min)}m"
    if age_min > LOG_FRESHNESS_STALE_MINUTES:
        report.anom(
            check, f"freshest file older than threshold={LOG_FRESHNESS_STALE_MINUTES}m ({facts})"
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
        check_dispatch_coverage(
            report,
            repo,
            prev_repo_state,
            new_repo_state,
            skip_delta,
            blocked_by_repo.get(repo.slug),
            blocked_err,
        )
        check_review_liveness(report, repo)
        check_dispatch_failures(report, repo, baseline)
        check_log_freshness(report, repo)
        check_merge_flow(report, repo, prev_repo_state, new_repo_state, skip_delta)
        new_state["repos"][repo.slug] = new_repo_state

    if repos:
        check_github_rate(report, repos[0].repo_root)
    else:
        report.anom("github-rate", "no repos registered, cannot resolve a cwd for gh")

    check_runners(report)

    save_state(new_state)

    print("\n".join(report.lines))
    return 1 if report.anomaly else 0


if __name__ == "__main__":
    sys.exit(main())
