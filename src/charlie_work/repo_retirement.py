"""First-class repo retirement (issue #2754).

A repo is retired when its **own** ``orchestrator.config.yaml`` sets
``fleet.retired: true``. The key is deliberately read from the repo layer
only: a fleet-wide ``<fleet_dir>/config.yaml`` value cannot retire every
registered repo, so retirement stays a per-repo operator decision.

The registry keeps the repo's entry. ``fleet_registry.touch_repo`` refreshes
the derived ``retired`` field on every touch so external readers (the
heartbeat script, ``fleet status``) see it without maintaining a list by
hand, and this module is the one place that derives it from config.

Retirement is enforced at the three per-repo pass-context builders named in
the issue: ``_select_repos``, ``_iter_registered_repo_contexts`` and
``fleet_loop``. A retired repo that still has live work runs in wind-down --
reap and verdict ingestion only -- and is skipped entirely once quiescent.
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import host as _host
from . import layout
from .config import find_config_path
from .live_session_count import REVIEW_LANE, WORKER_LANE, count_live_sessions
from .local_suite_runner import RUNNER_STATUS_FILENAME, _GATE_DIR_NAME

logger = logging.getLogger(__name__)

RETIRED_FIELD = "retired"


@dataclass(frozen=True)
class LiveWork:
    """The live work a retired repo must finish before it is quiescent."""

    workers: int = 0
    reviewers: int = 0
    gate_suites: int = 0

    @property
    def total(self) -> int:
        return self.workers + self.reviewers + self.gate_suites

    @property
    def quiescent(self) -> bool:
        return self.total == 0


def _config_path(repo_root: Path, explicit: Path | None) -> Path | None:
    """Resolve the repo's own config slot, mapping a relative explicit path."""
    if explicit is not None and not explicit.is_absolute():
        explicit = repo_root / explicit
    return find_config_path(repo_root, explicit)


def repo_retirement_state(repo_root: Path, explicit: Path | None = None) -> bool | None:
    """``fleet.retired`` from the repo's own config, or ``None`` if unreadable.

    Only the repo layer is consulted -- never ``<fleet_dir>/config.yaml``.
    The file is read as raw YAML rather than through ``load_config``: this
    runs in read-only fleet enumerations (``_select_repos``,
    ``managed_repo_names``/``managed_repo_roots``,
    ``_take_fleet_snapshot``) on every pass, so it must not re-validate the
    whole config, emit ``config_key_deprecated_read`` events, or raise on a
    config that a full load would reject.

    ``None`` lets callers preserve a previously recorded retirement rather
    than clearing it on a transient read failure (issue #2754's "never
    clears retirement" contract). Every read is contained: an unreadable or
    malformed file is a warning, never an exception escaping a fleet walk.
    """
    path: Path | None = None
    try:
        path = _config_path(repo_root, explicit)
        if path is None or not path.exists():
            return None
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - total by contract (see docstring)
        logger.warning(
            "repo retirement: could not read %s (%s)",
            path if path is not None else (explicit or repo_root),
            exc,
        )
        return None
    if raw is None:
        return False
    if not isinstance(raw, dict):
        logger.warning(
            "repo retirement: expected a YAML mapping in %s, got %s",
            path,
            type(raw).__name__,
        )
        return None
    fleet = raw.get("fleet")
    if fleet is None:
        return False
    if not isinstance(fleet, dict):
        logger.warning("repo retirement: %s: fleet must be a mapping", path)
        return None
    return bool(fleet.get(RETIRED_FIELD, False))


def entry_is_retired(entry: Mapping[str, Any]) -> bool:
    """Derive a registry entry's ``retired`` flag from its repo config.

    ``config_path`` may be absolute (what ``touch_repo`` writes) or relative
    to ``repo_root`` (what older test fixtures and hand-written registries
    carry), so both shapes resolve to the same file.
    """
    repo_root_str = entry.get("repo_root")
    if not repo_root_str:
        return False
    repo_root = Path(str(repo_root_str))
    config_path = entry.get("config_path")
    explicit = Path(str(config_path)) if config_path else None
    state = repo_retirement_state(repo_root, explicit)
    if state is not None:
        return state
    # Unreadable or absent config: preserve the last recorded flag rather
    # than clearing retirement on evidence we could not read.
    return bool(entry.get(RETIRED_FIELD, False))


def _running_gate_suites(state_dir: Path) -> int:
    """Count local merge-gate suites whose wrapper process is still alive."""
    gate_root = layout.dispatches_dir(state_dir) / _GATE_DIR_NAME
    if not gate_root.is_dir():
        return 0
    running = 0
    for pid_file in gate_root.glob(f"pr-*/{RUNNER_STATUS_FILENAME}"):
        try:
            identity = json.loads(pid_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(identity, dict):
            continue
        pid = identity.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int):
            continue
        if _host.current().probe.is_alive(pid, identity.get("process_start_time")):
            running += 1
    return running


def _pending_session_sidecars(directory: Path) -> int:
    """Count every session sidecar under *directory*, live or dead.

    A dead sidecar is unfinished wind-down bookkeeping: the retired repo must
    still run its reap lane so the sidecar can be removed. Counting only
    alive sessions would make a repo whose last worker just died look
    quiescent and skip the very lane that reaps it.
    """
    from .worker import iter_workers

    return len(iter_workers(directory)) if directory.exists() else 0


def repo_live_work(
    repo_root: Path,
    config: Any,
    state_dir: Path,
) -> LiveWork:
    """Count work a retired repo must wind down before it is quiescent.

    ``workers``/``reviewers`` use the same ``count_live_sessions`` rule the
    per-repo caps use, but take the larger of that and the number of session
    sidecars present: a dead sidecar is pending reap work, not quiescence.
    A non-``Path`` state dir (a test double) reads as no live work rather
    than raising.
    """
    if not isinstance(state_dir, Path):
        return LiveWork()
    sessions_dir = layout.resolve_state_child(
        config.devin.sessions_dir,
        repo_root=repo_root,
        default=layout.sessions_dir_default(state_dir),
    )
    reviews_dir = layout.resolve_state_child(
        config.review_dispatch.reviews_dir,
        repo_root=repo_root,
        default=layout.reviews_dir_default(state_dir),
    )
    state_file = layout.state_file_path(state_dir)
    workers = max(
        count_live_sessions(sessions_dir, state_file, WORKER_LANE),
        _pending_session_sidecars(sessions_dir),
    )
    reviewers = max(
        count_live_sessions(reviews_dir, state_file, REVIEW_LANE),
        _pending_session_sidecars(reviews_dir),
    )
    return LiveWork(
        workers=workers, reviewers=reviewers, gate_suites=_running_gate_suites(state_dir)
    )


def run_wind_down(app: Any, now: datetime.datetime | None = None) -> dict[str, Any]:
    """Reap dead/stalled work and ingest verdicts for a retired repo.

    This is the only work a retired repo may still do. It launches no worker,
    rework, rescue, reviewer or merge-gate suite and makes no tracker commit:
    every called helper is a reap/ingest lane.
    """
    import charlie_work.workflow as _wf

    resolved_now = now if now is not None else datetime.datetime.now(datetime.UTC)
    sessions_dir = app._layout.sessions_dir
    stalled = _wf._detect_and_handle_stalled_sessions(
        sessions_dir,
        app.paths.state_file,
        app.config,
        write_gate=app.write_gate,
        now=now,
    )
    reaped = _wf._classify_dead_sessions_and_update_throttle_state(
        sessions_dir,
        app.paths.state_file,
        app.gh,
        app.config,
        write_gate=app.write_gate,
        persist_inconclusive_probe_counter=False,
        now=now,
        fleet_dir_override=app.fleet_dir_override,
    )
    _wf._detect_and_handle_orphaned_workers(
        sessions_dir,
        app.paths.state_file,
        app.config,
        app.gh,
        write_gate=app.write_gate,
        fleet_dir_override=app.fleet_dir_override,
    )
    review_sweep = app._run_review_reap_sweeps(resolved_now)
    return {
        "retired": True,
        "wind_down": True,
        # Not ``stalled``: fleet_loop's attention extractor reads that key as
        # a list of stalled-session dicts and a count here would crash it.
        "stalled_reaped": len(stalled or []),
        "reaped": len(reaped or []),
        "review_sweep": {
            "stalled_reaped": len(review_sweep.get("stalled") or []),
            "recorded": len((review_sweep.get("verdict_result") or {}).get("recorded") or []),
        },
    }


def config_is_retired(config: Any) -> bool:
    """Whether a loaded config's ``fleet.retired`` is on."""
    fleet = getattr(config, "fleet", None)
    return bool(getattr(fleet, RETIRED_FIELD, False))


def mark_selection_retired(selected: list[tuple[str, dict[str, Any]]]) -> None:
    """Annotate a ``_select_repos`` result with the derived ``retired`` field."""
    for _key, entry in selected:
        if isinstance(entry, dict):
            entry[RETIRED_FIELD] = entry_is_retired(entry)


def touched_retired_flag(entry: Mapping[str, Any]) -> bool:
    """Derive the flag ``touch_repo`` persists, preserving an unreadable entry."""
    repo_root = Path(str(entry["repo_root"]))
    candidate = entry.get("config_path") or str(repo_root / "orchestrator.config.yaml")
    config_path = Path(str(candidate))
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    state = repo_retirement_state(repo_root, config_path)
    return state if state is not None else bool(entry.get(RETIRED_FIELD, False))


def retired_pass_disposition(
    repo_key: str,
    repo_root: Path,
    config: Any,
    state_dir: Path,
    retired_by_key: dict[str, bool],
) -> bool:
    """Record this pass's retirement; return True when the repo is skipped."""
    retired = config_is_retired(config)
    retired_by_key[repo_key] = retired
    return retired and repo_live_work(repo_root, config, state_dir).quiescent


def active_registry_repos(repos: Mapping[str, Any]) -> dict[str, Any]:
    """Registry entries minus retired repos (issue #2754 enumerations)."""
    return {
        key: entry
        for key, entry in repos.items()
        if not (isinstance(entry, dict) and entry_is_retired(entry))
    }
