"""Fleet-wide ``status`` aggregation fan-out (issue #1941).

``charlie fleet status`` walks the fleet registry (``fleet.json``) and calls
``OrchestratorApp.status()`` per registered repo. Before #1941 that walk was
serial and unbounded: a fleet-wide status-snapshot TTL miss summed every
repo's live recompute (~45-100s each) against the heartbeat's 120s
``charlie fleet status --json`` subprocess cap
(``CHARLIE_STATUS_TIMEOUT_SECONDS`` in ``scripts/heartbeat_check.py``).

This module owns the fix: each live repo's status runs on a daemon worker
thread bounded by ``FLEET_STATUS_REPO_TIMEOUT_SECONDS``, collected against a
single deadline so the aggregate cost is max(per-repo), not sum. A repo that
blows the budget lands in ``stale`` beside the #1372 dead-root entries — a
slow lane, like a dead one, must not pin the aggregate nor flip the exit
code — while a real per-repo failure still lands in ``errors`` and flips
``ok`` exactly as the serial loop did.

Daemon threads rather than ``ThreadPoolExecutor`` because a straggler must
not hold the process past the heartbeat's cap:
``concurrent.futures.thread._python_exit`` joins every pool worker at
interpreter teardown, so a TPE straggler would reintroduce the unbounded wait
this change removes (verified on CPython 3.13: a 30s work item held process
exit ~32s even after ``shutdown(wait=False)``). A ``Future`` per repo keeps
the prescribed ``future.result(timeout=...)`` collection semantics.

The collection loop decides "the budget expired" from ``wait()``'s return
value, never from exception identity: ``Future.result(timeout=...)`` raises
``TimeoutError`` for BOTH an expired wait and a worker whose own exception is
``TimeoutError``-shaped (``TimeoutError`` is an ``OSError`` — a wedged ``gh``
subprocess surfaces exactly that type), so an ``except TimeoutError`` around
``result()`` cannot tell a cut lane from a real failure and would silently
misclassify the latter into ``stale``.

``cli`` re-exports ``run_fleet_status`` (the ``fleet_stop`` /
``run_fleet_stop`` precedent): cli.py is an over-cap monolith under the
file-size ratchet (#1442), so the fan-out lives here and cli.py keeps a
one-line import.
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import Future, wait
from pathlib import Path
from typing import Any

from . import layout
from .config import ConfigError
from .fleet_dispatch import compute_api_worker_fleet_report
from .fleet_registry import _load_registry
from .github import GitHub, GitHubError
from .global_config import load_layered_config
from .local_issues import github_client_for
from .paths import RepoNotFoundError, runtime_paths
from .workflow import CommandResult, OrchestratorApp

# Issue #1941: per-repo budget for the ``fleet status`` fan-out. Sized to
# leave headroom under the heartbeat's 120s subprocess cap
# (CHARLIE_STATUS_TIMEOUT_SECONDS in scripts/heartbeat_check.py) while
# covering the measured per-repo live-recompute cost (~45-100s under a
# fleet-wide snapshot-TTL miss) so a merely-slow repo still contributes data
# and only a genuinely wedged one is cut.
FLEET_STATUS_REPO_TIMEOUT_SECONDS = 90.0


def _fleet_status_repo(
    fut: Future[CommandResult],
    repo_root: Path,
    fleet_dir_override: str | None,
    use_cache: bool,
) -> None:
    """Resolve one registered repo's ``status()`` result into ``fut`` (issue #1941).

    Runs on a daemon thread spawned by ``run_fleet_status``. Every exception
    is delivered through the future, so the collecting thread sees the same
    failure surface the serial loop used to produce.
    """
    try:
        config = load_layered_config(repo_root, None, fleet_dir_override=fleet_dir_override)
        paths = runtime_paths(repo_root, config.runtime.state_dir)
        gh = github_client_for(repo_root, config, github=GitHub, dry_run=True)
        app = OrchestratorApp(repo_root, paths, config, gh, dry_run=True)
        fut.set_result(app.status(use_cache=use_cache))
    except BaseException as exc:
        fut.set_exception(exc)


def run_fleet_status(args: argparse.Namespace) -> CommandResult:
    """Run fleet status aggregation across all registered repos.

    This is a read-only command that:
    - Loads the fleet registry from fleet.json
    - For each registered repo, calls OrchestratorApp.status() with dry_run=True
      on a daemon worker thread, bounded by FLEET_STATUS_REPO_TIMEOUT_SECONDS
    - Aggregates results keyed by repo_key (nameWithOwner)
    - Isolates per-repo errors (missing/broken/slow repos) without aborting the
      whole aggregation

    Note: This command does not include per-worker health fields yet. That will be added
    in a follow-up issue (#167) once the worker health abstraction lands.
    """
    fleet_json_path = layout.fleet_registry_path()
    registry = _load_registry(fleet_json_path)
    per_repo: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    # Issue #1372: stale entries (repo_root no longer exists) are reported in a
    # separate "stale" list that does NOT flip ok/exit-code, so one corpse
    # cannot degrade fleet-wide tooling (e.g. the heartbeat's blocked-issue
    # enrichment that treats any nonzero exit as degraded). Issue #1941: a
    # repo whose status computation blows FLEET_STATUS_REPO_TIMEOUT_SECONDS
    # joins the same list -- a slow lane, like a dead one, must not pin the
    # aggregate against the heartbeat's 120s subprocess cap.
    stale: list[dict[str, str]] = []

    live: dict[str, Path] = {}
    for repo_key, entry in sorted(registry.get("repos", {}).items()):
        repo_root = Path(entry.get("repo_root") or "")
        if not repo_root.exists():
            # Issue #1372: a stale entry is not a live failing lane —
            # report it separately so it does not affect the exit code.
            stale.append({"repo_key": repo_key, "repo_root": str(repo_root)})
            continue
        live[repo_key] = repo_root

    # Issue #1941: fan the per-repo status calls out on daemon threads --
    # the work is I/O-bound (gh subprocesses / HTTP), each repo gets its own
    # OrchestratorApp/GitHub instance and its own state-lock path, so there
    # is no shared mutable state to guard. Daemon threads rather than
    # ThreadPoolExecutor because a repo that exceeds the timeout must not be
    # able to hold the process past the heartbeat's cap:
    # concurrent.futures.thread._python_exit joins every pool worker at
    # interpreter teardown, so a TPE straggler would reintroduce the exact
    # unbounded wait this change removes (verified on CPython 3.13: a 30s
    # work item held process exit ~32s even after shutdown(wait=False)).
    # A Future per repo keeps the prescribed future.result(timeout=...)
    # collection semantics.
    use_cache = not getattr(args, "no_cache", False)
    futures: dict[str, Future[CommandResult]] = {}
    for repo_key, repo_root in live.items():
        fut: Future[CommandResult] = Future()
        futures[repo_key] = fut
        threading.Thread(
            target=_fleet_status_repo,
            args=(fut, repo_root, args.fleet_dir, use_cache),
            name=f"fleet-status-{repo_key}",
            daemon=True,
        ).start()

    # Collect against a single deadline measured from fan-out start, so the
    # aggregate is bounded by the per-repo budget (max), not a stacked
    # per-future timeout (sum). Iterating in sorted repo order preserves the
    # previous aggregation order; a repo that finished while a slower
    # earlier-keyed repo was still running still lands, because result()
    # returns instantly for an already-complete future even at timeout=0.
    deadline = time.monotonic() + FLEET_STATUS_REPO_TIMEOUT_SECONDS
    for repo_key, fut in futures.items():
        # wait() decides the budget by its return value, not by exception
        # identity: fut.result(timeout=...) raises TimeoutError both for an
        # expired wait AND for a worker whose stored exception is
        # TimeoutError-shaped, and an ``except TimeoutError`` cannot
        # distinguish the two -- a real failure (TimeoutError is an OSError,
        # e.g. a wedged gh subprocess) would silently land in stale instead
        # of errors. Once the future is done, the no-timeout result() below
        # returns instantly and only ever raises the worker's own exception.
        done, _pending = wait([fut], timeout=max(deadline - time.monotonic(), 0.0))
        if not done:
            stale.append(
                {
                    "repo_key": repo_key,
                    "repo_root": str(live[repo_key]),
                    "reason": "status_timeout",
                }
            )
            continue
        try:
            result = fut.result()
        except (RepoNotFoundError, ConfigError, GitHubError, OSError) as exc:
            errors.append({"repo_key": repo_key, "error": str(exc)})
        else:
            per_repo[repo_key] = result.data

    # api-worker fleet report line (issue #483): read-only, never raises.
    api_worker_report = compute_api_worker_fleet_report(fleet_dir_override=args.fleet_dir)

    return CommandResult(
        ok=not errors,
        message=f"fleet status: {len(per_repo)} repo(s), {len(errors)} error(s), {len(stale)} stale(s)",
        data={
            "repos": per_repo,
            "errors": errors,
            "stale": stale,
            "api_worker_report": api_worker_report.to_dict()
            if api_worker_report is not None
            else None,
        },
    )
