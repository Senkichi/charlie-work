"""Host and GitHub side of runner-slot allocation: discovery, demand, actuation.

The policy that decides *how many* listeners each repo should run lives in
``runner_allocation.py``. This module is everything that touches the world:
finding the configured runners on this host, measuring each repo's live Actions
demand, starting and parking listeners, and persisting the slack history the
hysteresis rule reads.

Two safety properties are enforced here, not in policy:

- **A running job is never interrupted.** ``park_runner_slot`` re-checks for a
  live ``Runner.Worker`` child immediately before stopping a listener, because
  a job can be handed out between planning and actuation and GitHub's ``busy``
  flag is eventually consistent.
- **Foreign runners are unreachable.** ``discover_runner_instances`` walks
  exactly the configured managed root, non-recursively, and only accepts
  directories holding this platform's launch script. A runner installed
  anywhere else on the host — for example an unrelated service install at
  ``C:\\actions-runner`` — is outside the traversal, not merely filtered out of
  it.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Mapping

from .github import GitHub, GitHubError
from .runner_allocation import (
    AllocationPlan,
    RepoDemand,
    RunnerInstance,
    SlotAction,
    SlotChangeResult,
    repo_slug_from_github_url,
)
from .runners import (
    get_runner_launcher_process,
    get_runner_listener_process,
    is_runner_launched,
    launch_runner_listener,
)


logger = logging.getLogger(__name__)

# Persistent per-repo slack tracking for demotion hysteresis.
ALLOCATION_STATE_FILENAME = "runner-allocation.json"

# Which code path persisted an allocation pass. Both paths write the same
# host-wide state file, so the timestamp alone cannot distinguish "the daemon is
# rebalancing on its own" from "a human just ran `charlie runners allocate`" —
# and only the first answers issue #590. Recording the writer keeps that
# distinction in the artifact instead of leaving it to be guessed from age.
AllocationSource = Literal["prologue", "cli"]

# The two writers, named once here so that what the daemon stamps, what the CLI
# stamps, and what the doctor probe tests for cannot drift apart. Every writer and
# every reader spells them this way; a bare string literal at a call site would
# reintroduce exactly the drift the ``Literal`` type is meant to prevent, since
# ``Literal`` constrains the value but nothing keeps three separate copies of it
# in agreement.
UNATTENDED_ALLOCATION_SOURCE: AllocationSource = "prologue"
CLI_ALLOCATION_SOURCE: AllocationSource = "cli"

# Label that marks a job as targeting this host's runners rather than
# GitHub-hosted ones. This is GitHub's own reserved label, not a local
# convention, so it needs no per-repo configuration.
SELF_HOSTED_LABEL = "self-hosted"

_ACTIVE_JOB_STATUSES = ("queued", "in_progress", "pending", "waiting")

# psutil's process name for the per-job child the listener spawns.
_WORKER_PROCESS_HINT = "runner.worker"


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def launch_script_name(platform: str) -> str:
    """Name of the runner launch script for a ``sys.platform`` value."""
    return "run.cmd" if platform == "win32" else "run.sh"


def discover_runner_instances(
    managed_root: Path,
    *,
    platform: str | None = None,
) -> tuple[list[RunnerInstance], list[str]]:
    """Find every configured runner directly under ``managed_root``.

    A directory qualifies when it holds this platform's launch script and a
    parseable ``.runner``. Repo ownership is read from that file rather than
    inferred from directory-name conventions, so adding a repo to the host
    needs no code or config change.

    Returns:
        Tuple of (instances, notes). Notes describe skipped directories so an
        unreadable ``.runner`` is visible rather than silently absent.
    """
    script_name = launch_script_name(platform if platform is not None else sys.platform)

    if not managed_root.exists():
        return [], [f"managed_root does not exist: {managed_root}"]

    instances: list[RunnerInstance] = []
    notes: list[str] = []

    resolved_root = managed_root.resolve()

    for entry in sorted(managed_root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        # ``is_dir()`` follows junctions and symlinks, so the traversal's shape
        # (one level, non-recursive) bounds how *deep* discovery reaches but not
        # *which directory* an entry names. A junction under managed_root would
        # hand back a tree somewhere else entirely — including the unrelated
        # runner service at C:\actions-runner — and make it start/park-eligible.
        # Enforce containment on the resolved path rather than trusting shape.
        if not entry.resolve().is_relative_to(resolved_root):
            notes.append(f"{entry.name}: resolves outside managed_root, skipped")
            continue
        if not (entry / script_name).exists():
            continue

        runner_file = entry / ".runner"
        if not runner_file.exists():
            notes.append(f"{entry.name}: no .runner file, skipped")
            continue

        try:
            # The runner writes .runner as UTF-8 *with* a BOM; a plain utf-8
            # read raises on the leading marker.
            data = json.loads(runner_file.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            notes.append(f"{entry.name}: unreadable .runner ({exc}), skipped")
            continue

        repo = repo_slug_from_github_url(str(data.get("gitHubUrl", "")))
        if repo is None:
            notes.append(f"{entry.name}: no repo in .runner gitHubUrl, skipped")
            continue

        instances.append(
            RunnerInstance(
                path=entry,
                name=str(data.get("agentName") or entry.name),
                repo=repo,
                # Mid-startup counts as running: a runner whose launch script is
                # still bringing .NET up has no listener process yet, and
                # reporting it parked would have the planner start it again.
                running=is_runner_launched(entry),
            )
        )

    return instances, notes


def has_active_job(runner_path: Path) -> bool:
    """Check locally whether a runner is executing a job right now.

    The listener spawns a ``Runner.Worker`` child for the duration of a job, so
    a live child is direct evidence of work in flight. This backs up GitHub's
    ``busy`` flag, which can report not-busy for a job just handed out.

    Fails closed: if the process tree cannot be inspected, reports True so the
    runner is left alone rather than killed mid-job.
    """
    try:
        import psutil
    except ImportError:
        return True

    process = get_runner_listener_process(runner_path)
    if process is None:
        return False

    try:
        for child in psutil.Process(process.pid).children(recursive=True):
            try:
                if _WORKER_PROCESS_HINT in (child.name() or "").lower():
                    return True
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                # A child that exited between enumeration and inspection cannot be
                # a worker with a job in flight; keep scanning its siblings.
                continue
            except psutil.AccessDenied:
                # Unreadable name means this child cannot be *ruled out* as the
                # worker, and the docstring's fail-closed contract has to hold on
                # this path too — the outer handler already fails closed for the
                # same reason. Treating it as "not a worker" is what would let
                # park_runner_slot terminate a listener mid-job and abort CI.
                return True
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False
    except psutil.AccessDenied:
        return True

    return False


# --------------------------------------------------------------------------
# GitHub observation
# --------------------------------------------------------------------------


def fetch_busy_runner_names(gh: GitHub, repo: str) -> tuple[set[str], str | None]:
    """Names of runners GitHub currently reports as busy for ``repo``.

    The repo is addressed by explicit slug rather than gh's ``{owner}/{repo}``
    placeholders, so one client can query every repo with runners on this host
    without needing a local checkout of each.
    """
    try:
        data = gh.run(["api", f"repos/{repo}/actions/runners?per_page=100"], json_output=True)
    except GitHubError as exc:
        return set(), str(exc)
    runners = data.get("runners", []) if isinstance(data, dict) else []
    return {str(r.get("name")) for r in runners if r.get("busy") is True}, None


def measure_repo_demand(gh: GitHub, repo: str, max_runs_scanned: int) -> RepoDemand:
    """Count self-hosted Actions jobs that want or hold a slot in ``repo``.

    Demand is measured in *jobs*, not runs: one workflow run can hold several
    runner slots, and it is jobs that map one-to-one onto slots. Only jobs
    carrying GitHub's ``self-hosted`` label count — hosted jobs consume none of
    this host's capacity.

    Runs are queried by ``status`` rather than filtered client-side from a
    single page, so a repo with a long completed-run history cannot push its
    own in-flight work off the end of the page. Note that this counts runs on
    *every* branch: agent PR branches are where the CI load actually is, so a
    default-branch-only query would read near-zero demand while the queue is
    deep.

    Never raises. A failed measurement comes back as ``ok=False``, which the
    allocator treats as "leave this repo alone".
    """
    queued = 0
    in_progress = 0
    truncated = False
    seen_runs = 0

    for status in ("queued", "in_progress"):
        try:
            data = gh.run(
                [
                    "api",
                    f"repos/{repo}/actions/runs?status={status}&per_page={max_runs_scanned}",
                ],
                json_output=True,
            )
        except GitHubError as exc:
            return RepoDemand(repo=repo, ok=False, error=f"runs?status={status}: {exc}")

        runs = data.get("workflow_runs", []) if isinstance(data, dict) else []
        for run in runs:
            run_id = run.get("id")
            if run_id is None:
                continue
            if seen_runs >= max_runs_scanned:
                truncated = True
                break
            seen_runs += 1

            try:
                jobs_data = gh.run(
                    ["api", f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"],
                    json_output=True,
                )
            except GitHubError as exc:
                return RepoDemand(repo=repo, ok=False, error=f"run {run_id} jobs: {exc}")

            jobs = jobs_data.get("jobs", []) if isinstance(jobs_data, dict) else []
            for job in jobs:
                job_status = job.get("status")
                if job_status not in _ACTIVE_JOB_STATUSES:
                    continue
                labels = [str(label) for label in (job.get("labels") or [])]
                if SELF_HOSTED_LABEL not in labels:
                    continue
                if job_status == "in_progress":
                    in_progress += 1
                else:
                    queued += 1

    return RepoDemand(
        repo=repo,
        queued_jobs=queued,
        in_progress_jobs=in_progress,
        ok=True,
        truncated=truncated,
    )


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def load_idle_streaks(fleet_dir: Path) -> dict[str, int]:
    """Read persisted slack streaks. Any problem degrades to "no history"."""
    path = fleet_dir / ALLOCATION_STATE_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    repos = data.get("repos") if isinstance(data, dict) else None
    if not isinstance(repos, dict):
        return {}
    streaks: dict[str, int] = {}
    for repo, entry in repos.items():
        if isinstance(entry, dict) and isinstance(entry.get("idle_streak"), int):
            streaks[str(repo)] = entry["idle_streak"]
    return streaks


@dataclass(frozen=True)
class AllocationStamp:
    """Provenance of the last persisted allocation pass.

    ``source`` is ``None`` for a file written before provenance was recorded,
    which is reported as unknown rather than assumed to be either path.
    """

    updated_at: datetime | None
    source: str | None


def load_allocation_stamp(fleet_dir: Path) -> AllocationStamp | None:
    """Read when the last allocation pass ran and which path wrote it.

    Returns ``None`` when the state file is absent — the caller distinguishes
    "never ran" from "ran but wrote something unreadable", which is why an
    unparseable file yields a stamp with empty fields instead of ``None``.

    This is the only reader of the state file's top-level shape outside
    ``save_idle_streaks``; keeping it here means a change to the payload does not
    have to be mirrored into the doctor probe.
    """
    path = fleet_dir / ALLOCATION_STATE_FILENAME
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return AllocationStamp(updated_at=None, source=None)
    if not isinstance(raw, dict):
        return AllocationStamp(updated_at=None, source=None)

    updated_at: datetime | None = None
    stamp = raw.get("updated_at")
    if isinstance(stamp, str):
        try:
            updated_at = datetime.fromisoformat(stamp)
        except ValueError:
            updated_at = None
    if updated_at is not None and updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)

    source = raw.get("source")
    return AllocationStamp(
        updated_at=updated_at,
        source=source if isinstance(source, str) else None,
    )


def save_idle_streaks(
    fleet_dir: Path, streaks: Mapping[str, int], *, source: AllocationSource
) -> None:
    """Persist slack streaks using the project's atomic temp-file + replace.

    ``source`` is required rather than defaulted: every writer has to say which
    path it is, because a caller that silently inherited the unattended default
    would make the state file assert that the daemon ran when it did not.
    """
    fleet_dir.mkdir(parents=True, exist_ok=True)
    path = fleet_dir / ALLOCATION_STATE_FILENAME
    payload = {
        "version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "source": source,
        "repos": {repo: {"idle_streak": streak} for repo, streak in sorted(streaks.items())},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------
# Actuation
# --------------------------------------------------------------------------


def park_runner_slot(
    instance: RunnerInstance,
    *,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Stop a runner's listener while leaving its registration in place.

    Re-checks for a live ``Runner.Worker`` immediately before stopping, since a
    job can be handed out between planning and actuation. This guard is what
    makes parking safe: the listener is terminated outright, which would abort
    an in-flight job.

    The runner goes ``offline`` on GitHub and keeps its credentials, so
    bringing it back is a plain process launch — no registration token, no
    GitHub write.
    """
    if has_active_job(instance.path):
        return False, f"{instance.name} picked up a job since planning; left running"

    process = get_runner_listener_process(instance.path)
    if process is None:
        # A live launch script with no listener yet means the runner is mid
        # startup. Killing the wrapper would not park it — ``cmd /c`` leaves its
        # child behind, so the listener would come online orphaned. Decline and
        # let the next pass park it properly; the plan is idempotent.
        if get_runner_launcher_process(instance.path) is not None:
            return False, f"{instance.name} is still starting up; will park on a later pass"
        return True, f"{instance.name} is already parked"

    if dry_run:
        return True, f"Would park {instance.name} (PID {process.pid})"

    try:
        import psutil
    except ImportError:
        return False, "psutil unavailable; cannot park safely"

    try:
        proc = psutil.Process(process.pid)
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except psutil.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except psutil.NoSuchProcess:
        return True, f"{instance.name} exited before it could be parked"
    except Exception as exc:
        return False, f"Failed to park {instance.name}: {exc}"

    return True, f"Parked {instance.name} (registration retained)"


def apply_allocation(
    plan: AllocationPlan,
    *,
    dry_run: bool = False,
) -> list[SlotChangeResult]:
    """Execute a plan's changes. Every failure is reported, never raised."""
    results: list[SlotChangeResult] = []

    for change in plan.changes:
        if change.action is SlotAction.START:
            # Re-check liveness for the same reason park does: the plan was
            # built from an earlier snapshot, and launching a second listener
            # into a directory that already has one is a config conflict.
            # This must count a runner that is still *starting* as running —
            # otherwise two controllers acting seconds apart (a fleet pass and
            # an operator running `runners allocate`) both launch it, and the
            # two listeners race for one registration.
            if is_runner_launched(change.path):
                ok, message = True, f"{change.runner_name} was already running"
            else:
                ok, message = launch_runner_listener(change.path, dry_run=dry_run)
        else:
            ok, message = park_runner_slot(
                RunnerInstance(
                    path=change.path,
                    name=change.runner_name,
                    repo=change.repo,
                    running=True,
                ),
                dry_run=dry_run,
            )
        results.append(SlotChangeResult(change=change, ok=ok, message=message))
        logger.info(
            "runner-allocation %s %s (%s): %s",
            change.action.value,
            change.runner_name,
            change.repo,
            message,
        )

    return results
