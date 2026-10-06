"""Per-repo fleet lane execution and the out-of-band reap scheduler.

Extracted from ``fleet_dispatch`` (issue #1934, under the file-size
ratchet lineage of #1442): the lane body submitted to the pass's bounded
thread pool, the fleet-wide dead-review-claim reap sweep, and the
scheduler thread that fires it independently of pass cadence. The names
are re-exported through ``fleet_dispatch`` so existing callers and test
patch seams keep resolving on the ``fleet_dispatch.<name>`` facade.
"""

from __future__ import annotations

import datetime
import logging
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import layout, markdown_guard
from .config import OrchestratorConfig
from .fleet_paths import fleet_dir
from .fleet_registry import _load_registry, _select_repos
from .github import GitHub
from .github_transport.budget_pass import emit_github_budget_pass
from .global_config import load_layered_config
from .instrumentation import log_event
from .local_issues import github_client_for
from .pass_deadline import PassDeadlineExceeded, set_pass_deadline_exceeded
from .paths import runtime_paths
from .command_result import CommandResult
from .workflow import OrchestratorApp

logger = logging.getLogger(__name__)

# Issue #1934: per-repo lane work inside one fleet pass is I/O-bound (gh
# calls, state file I/O, worker/reviewer launches that return immediately)
# and repo-isolated (each lane builds its own config, GitHub client,
# OrchestratorApp, and supervisor lock against disjoint state dirs). The
# pre-#1934 serial ``for`` loop made one pass cost the SUM of every repo's
# lane (~35-42 min across 6 repos against a configured 5-minute cadence),
# starving every lane-embedded sweep that assumes a per-pass cadence. Lanes
# now run on a bounded pool: the pass is bounded by the slowest lane, and --
# when the cap meets the registered repo count -- a repo's lane-to-lane gap
# depends only on its own lane duration plus the supervisor's pass cadence.
_DEFAULT_FLEET_LANE_CONCURRENCY = 8


def _resolve_fleet_lane_concurrency(global_config: Any) -> int:
    """Effective per-pass lane cap: ``fleet_supervisor.fleet_lane_concurrency``
    or the built-in default when absent/misconfigured (including a None
    ``global_config`` on direct CLI call sites)."""
    fleet_supervisor_cfg = getattr(global_config, "fleet_supervisor", None)
    value = getattr(fleet_supervisor_cfg, "fleet_lane_concurrency", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return _DEFAULT_FLEET_LANE_CONCURRENCY
    return value


def _run_fleet_repo_lane(
    repo_key: str,
    app: OrchestratorApp,
    config: OrchestratorConfig,
    lock: Any,
    *,
    work_only: bool,
    drain: bool,
    limit: int | None,
    merge: bool | None,
    ensure_labels: bool,
    deadline_exceeded: Callable[[], bool] | None = None,
    fleet_state_path: Path | None = None,
) -> CommandResult:
    """One repo's lane body, executed on a pool thread (issue #1934).

    Everything this touches is per-repo: ``app`` was constructed on the
    submitting thread for this repo alone and no lane shares an
    OrchestratorApp/GitHub client/config instance. The supervisor ``lock``
    is held from submission until this lane returns (released here in
    ``finally``), preserving the pre-#1934 "one lane owns the repo for the
    pass" invariant. A raised exception propagates to the collector future,
    which renders it into the same per-repo error result the serial loop
    produced.

    ``deadline_exceeded`` (issue #1948) is ``fleet_loop``'s cooperative
    in-pass deadline predicate. Before #1948 it was only consulted between
    lanes, so a single lane could overrun the pass budget by tens of
    minutes of sequential gh timeouts and retry backoffs. The lane arms it
    on its own ``GitHub`` client (``run()`` then refuses new calls/aborts
    retry chains the moment the budget is spent), checks it at the lane's
    own yield points, and threads it into ``app.loop()`` for the
    sub-phase-boundary checks there. A lane cut short returns a
    ``CommandResult`` carrying ``data["deadline_deferred"] = True`` instead
    of finishing every queued phase.
    """
    # This lane's own ambient sink for markdown_guard_disagreement events: the
    # pool thread has no binding until it sets one, and `app` was built on the
    # submitting thread, so the last-constructed repo must not receive it.
    sink_token = None
    try:
        sink_token = markdown_guard.bind_sink(app.paths.state_file, app.repo_root.name)
        # Arm the deadline hook on this lane's own client (issue #1948).
        # isinstance, not getattr duck-typing: the suite patches
        # ``charlie_work.fleet_dispatch.GitHub`` with a MagicMock, so
        # ``app.gh`` on a mocked app must be skipped here; a real app built
        # on LocalFileGitHub (local_issues backend) has no run() loop to
        # bound and is skipped the same way.
        if deadline_exceeded is not None and isinstance(app.gh, GitHub):
            set_pass_deadline_exceeded(app.gh, deadline_exceeded)

        # Issue #1339: ensure every LabelConfig-derived label exists on the
        # repo before the lane runs. Idempotent and best-effort:
        # ``ensure_labels`` records failures as events and never raises, so
        # a missing-label drift self-heals on the supervisor's first pass
        # without blocking the lane. The supervisor passes
        # ``ensure_labels=True`` on its first pass only (see
        # run_fleet_supervise), so this is once per startup per repo, not
        # once per pass.
        if ensure_labels:
            try:
                app.ensure_labels()
            except Exception as exc:  # noqa: BLE001 — never block a lane
                logger.warning("fleet label ensure failed for %s: %s", repo_key, exc)

        # Issue #1948: label-ensure can consume the remainder of the pass
        # budget after fleet_loop's post-slot-wait re-check ran. Bail before
        # the lane's first real phase rather than starting work on a dead
        # budget -- the unfinished work defers to the next pass.
        if deadline_exceeded is not None and deadline_exceeded():
            return CommandResult(
                True,
                "in-pass deadline reached before lane work; deferred to next pass",
                {"deadline_deferred": True},
            )

        if work_only:
            # Dispatch-only path (worker dispatch + optional review dispatch)
            result = app.dispatch(0 if drain else limit)
            if config.review_dispatch.enabled:
                # Issue #1948: yield-point check between the work lane's two
                # sequential GitHub phases -- skip review dispatch once the
                # pass budget is spent instead of accumulating timeouts.
                if deadline_exceeded is not None and deadline_exceeded():
                    combined_data = dict(result.data)
                    combined_data["deadline_deferred"] = True
                    combined_data["dispatch_reviews"] = {"deadline_deferred": True}
                    return CommandResult(
                        result.ok,
                        f"{result.message}; review dispatch deferred: in-pass deadline",
                        combined_data,
                    )
                review_dispatch_result = app.dispatch_reviews(0 if drain else limit)
                ok = result.ok and review_dispatch_result.ok
                message = (
                    "work-only dispatch: "
                    f"workers={result.data.get('selected_count', 0)}, "
                    f"reviews={review_dispatch_result.data.get('selected_count', 0)}"
                )
                combined_data = dict(result.data)
                combined_data["dispatch_reviews"] = review_dispatch_result.data
                result = CommandResult(ok, message, combined_data)
            return result
        # Full loop (intake -> dispatch -> review -> merge). A
        # drain pass forces limit=0: dispatch_rework/dispatch
        # slice candidates[:0] (0 is not None, so it is never
        # replaced by default_limit) while the reap, review, and
        # merge lanes below them run normally.
        # ``deadline_exceeded`` (issue #1948) lets _loop_body's
        # sub-phase-boundary checks stop the pass mid-lane.
        return app.loop(0 if drain else limit, merge=merge, deadline_exceeded=deadline_exceeded)
    except PassDeadlineExceeded:
        # Issue #1948: a refusal escaped the lane body -- the pass budget
        # was spent mid-flight on a path the pass's own deadline markers
        # didn't cover (a work_only dispatch has no _loop_impl wrapper).
        # Report the lane as a deadline partial -- a distinct shape from
        # both a hard lane failure and an observed repo, so fleet_loop's
        # collector sorts it into deadline_partial_repo_keys.
        return CommandResult(
            True,
            "in-pass deadline reached mid-lane; deferred to next pass",
            {"deadline_deferred": True},
        )
    finally:
        markdown_guard.unbind_sink(sink_token)
        if fleet_state_path is not None:
            # Issue #2439: this lane's GitHub spend, by capability, in the one
            # fleet-level events.db so the hourly ranking is a single query.
            emit_github_budget_pass(app.gh, fleet_state_path, pass_kind="lane", repo_key=repo_key)
        lock.release()


def _run_fleet_reap_sweep(
    *,
    fleet_dir_override: str | None = None,
    repos: tuple[str, ...] | None = None,
    dry_run: bool = False,
    now: datetime.datetime | None = None,
) -> dict[str, Any]:
    """Run the dead-review-claim reap sweep set once per selected repo (issue #1934).

    Each repo's sweep is ``OrchestratorApp._run_review_reap_sweeps`` -- the
    identical block ``dispatch_reviews`` runs at the top of every lane pass
    and the standalone ``reap_reviews`` command runs out-of-band (issue
    #1874). No supervisor lock is taken: the sweeps never launch a reviewer,
    so the double-dispatch window the lock exists to close cannot open, and
    every write inside them is ``state_lock``-serialized / merge-on-write
    safe against a concurrent lane pass (issue #594, #1874). That makes the
    reap scheduler immune to the exact failure this issue exists to fix --
    a fleet lane currently holding the lock does not postpone this repo's
    reap, and a reap in flight never causes a lane's lock probe to skip.

    Per-repo isolation mirrors the lane loop's: a stale entry is skipped, and
    a failure in one repo's sweep is logged/recorded and does not abort the
    round. Every repo's outcome lands in the fleet-level events.db as one
    ``fleet_reap_sweep`` event (warning level on failure) so the cadence is
    observable from one query. ``dry_run`` propagates into the app, where
    ``_run_review_reap_sweeps`` returns the empty summary read-only.
    """
    fleet_json_path = layout.fleet_registry_path(override=fleet_dir_override)
    registry = _load_registry(fleet_json_path)
    selected = _select_repos(registry, repos)
    fleet_state_path = layout.state_file_path(fleet_dir(override=fleet_dir_override))
    resolved_now = now if now is not None else datetime.datetime.now(datetime.UTC)
    results: dict[str, Any] = {}
    for repo_key, entry in selected:
        repo_root = Path(entry.get("repo_root") or "")
        if not repo_root.is_dir():
            # Stale-entry bookkeeping lives in the lane path (issue #1372);
            # the sweep just has nothing to do here.
            continue
        try:
            explicit_cfg = entry.get("config_path")
            config = load_layered_config(
                repo_root,
                Path(explicit_cfg) if explicit_cfg else None,
                fleet_dir_override=fleet_dir_override,
            )
            # Fleet mode owns notification emission; keep the sweep's per-repo
            # side effects quiet the same way the lane does.
            config = replace(config, notify=replace(config.notify, enabled=False))
            paths = runtime_paths(repo_root, config.runtime.state_dir)
            gh = github_client_for(repo_root, config, github=GitHub, dry_run=dry_run)
            app = OrchestratorApp(
                repo_root,
                paths,
                config,
                gh,
                dry_run=dry_run,
                fleet_dir_override=fleet_dir_override,
            )
            sweep = app._run_review_reap_sweeps(resolved_now)
            emit_github_budget_pass(gh, fleet_state_path, pass_kind="reap", repo_key=repo_key)
            verdict_result = sweep.get("verdict_result") or {}
            payload = {
                "repo_key": repo_key,
                "stalled_reaped": len(sweep.get("stalled") or []),
                "verdicts_recorded": len(verdict_result.get("recorded") or []),
                "verdicts_missed": len(verdict_result.get("missed") or []),
                "reconciled_verdicts": len(sweep.get("reconciled_verdicts") or []),
                "reaped_checkouts": len(sweep.get("reaped_checkouts") or []),
                "orphaned_checkouts": len(sweep.get("orphaned_checkouts") or []),
                "dry_run": dry_run,
            }
            try:
                # write-gate-exempt(issue=1934): no write_gate param; sibling raw calls remain
                log_event(fleet_state_path, "fleet_reap_sweep", payload, repo=repo_key)
            except Exception:
                logger.debug("Failed to record fleet_reap_sweep for %s", repo_key)
            results[repo_key] = payload
        except Exception as exc:  # noqa: BLE001 — one repo must not starve the round
            error_message = f"{type(exc).__name__}: {exc}"
            logger.warning("fleet reap sweep failed for %s: %s", repo_key, error_message)
            try:
                # write-gate-exempt(issue=1934): no write_gate param; sibling raw calls remain
                log_event(
                    fleet_state_path,
                    "fleet_reap_sweep",
                    {"repo_key": repo_key, "error": error_message},
                    repo=repo_key,
                    level="warning",
                )
            except Exception:
                logger.debug("Failed to record fleet_reap_sweep error for %s", repo_key)
            results[repo_key] = {"repo_key": repo_key, "error": error_message}
    return results


def _fleet_reap_sweep_loop(
    stop_event: threading.Event,
    interval_seconds: float,
    *,
    fleet_dir_override: str | None,
    repos: tuple[str, ...] | None,
    dry_run: bool,
) -> None:
    """Out-of-band reap scheduler loop (issue #1934).

    Fires ``_run_fleet_reap_sweep`` every ``interval_seconds`` until
    ``stop_event`` is set. ``Event.wait`` doubles as the sleep so shutdown is
    immediate rather than waiting out a pending interval. A failing round is
    logged and the loop continues -- a transient registry/config error must
    not permanently disarm the only reaper that runs while lanes are slow.
    """
    while not stop_event.wait(interval_seconds):
        try:
            _run_fleet_reap_sweep(
                fleet_dir_override=fleet_dir_override,
                repos=repos,
                dry_run=dry_run,
            )
        except Exception:  # noqa: BLE001 — the scheduler must survive a bad round
            logger.exception("fleet reap sweep round failed")


def _start_fleet_reap_scheduler(
    *,
    interval_seconds: int,
    fleet_dir_override: str | None,
    repos: tuple[str, ...] | None,
    dry_run: bool,
) -> tuple[threading.Thread, threading.Event]:
    """Start the daemon reap-sweep thread; return ``(thread, stop_event)``."""
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_fleet_reap_sweep_loop,
        args=(stop_event, interval_seconds),
        kwargs={
            "fleet_dir_override": fleet_dir_override,
            "repos": repos,
            "dry_run": dry_run,
        },
        name="fleet-reap-sweeps",
        daemon=True,
    )
    thread.start()
    return thread, stop_event


def _run_fleet_config_retirement_sweep(
    fleet_dir_override: str | None,
    global_config: Any,
    dry_run: bool,
    now: datetime.datetime | None,
) -> list[dict[str, Any]]:
    """Run the deprecated-config-key retirement sweep once per fleet pass
    (issue #1976).

    Checks every key in ``config_deprecations.DEPRECATED_CONFIG_KEYS``
    against every config layer of every registered repo and marks the key's
    ``removal_issue`` Ready once it has been absent everywhere for the
    ``runtime.config_retirement_quiet_days`` quiet window. Removal issues
    live on the orchestrator's own repo, so the client is built against
    ``orchestrator_root()`` rather than a fleet member's root; a ``None`` or
    non-config ``global_config`` falls back to dataclass defaults (direct
    CLI call sites pass no GlobalConfig).

    Returns digest attention entries; the sweep itself never raises.
    """
    from .config_retirement_sweep import run_config_retirement_sweep

    config = (
        global_config if isinstance(global_config, OrchestratorConfig) else OrchestratorConfig()
    )
    try:
        # ``github=`` rather than a pre-built ``gh`` so the client
        # construction (``github_client_for`` against ``orchestrator_root()``)
        # happens inside the sweep's own never-raises boundary -- a client
        # that cannot be built must degrade to an attention entry, not break
        # the fleet pass at this call site.
        return run_config_retirement_sweep(
            fleet_dir_override=fleet_dir_override,
            config=config,
            github=GitHub,
            dry_run=dry_run,
            now=now,
        )["attention"]
    except Exception:  # noqa: BLE001 — the sweep is observability+scheduling; the pass must survive it
        logger.exception("config-retirement sweep failed")
        return [
            {
                "repo_key": "fleet",
                "type": "config_retirement_error",
                "reason": "config-retirement sweep failed; see fleet log",
            }
        ]
