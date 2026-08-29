"""Preflight gate: refuse a loop pass or supervisor startup when a host
precondition is unmet, instead of failing midway with a generic error.

Why this exists (issue #1363): every recent whole-fleet outage was a host
precondition failure that the software invariants survived but could not
prevent or name -- a 2026-08-19 disk-full outage burned 8 noisy aborted pass
attempts where one named refusal would have sufficed; a cached-config trap
makes operator edits silently inert; a wrong-venv/wrong-checkout class once
synced a live venv against the wrong lockfile; clock skew silently inverts
staleness math. Per the four-gate taxonomy this is a **pre-flight gate**:
block entry on unmet preconditions, create no partial work.

This module implements the checks themselves (issue #1363 PART 1) as pure,
injectable-probe functions so they are unit-testable without touching a real
disk, clock, or interpreter. Wiring these into
``OrchestratorApp._loop_impl`` and the supervisor startup path is a
separate, later step -- this module does not import ``workflow`` or
``supervise_loop``.

Four checks, run in order, each independently classified fatal/non-fatal via
``config.PreflightConfig`` (never hardcoded at a call site):

1. ``disk_floor`` (fatal by default) -- refuse before a pass half-writes
   anything.
2. ``clock_sanity`` (non-fatal by default) -- a tripwire, not NTP.
3. ``venv_identity`` (fatal by default) -- catches the wrong-venv/
   wrong-checkout class before it dispatches work computed from the wrong
   code.
4. ``config_freshness`` (non-fatal by default) -- makes the silent-inert-
   edit trap loud; does not hot-reload.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import charlie_work

from . import layout
from .config import PreflightConfig
from .instrumentation import log_event

UTC = timezone.utc

#: Injectable signature for a disk-usage probe: takes a path-like anchor,
#: returns an object with a ``.free`` attribute (matches
#: ``shutil.disk_usage``'s return value).
DiskUsageFn = Callable[[str], object]
#: Injectable signature for an mtime probe: takes a path, returns the path's
#: ``os.stat_result`` (only ``.st_mtime`` is read).
StatFn = Callable[[Path], object]


@dataclass(frozen=True)
class PreflightCheck:
    """Result of a single preflight check."""

    name: str
    ok: bool
    detail: str
    fatal: bool


@dataclass(frozen=True)
class PreflightResult:
    """Outcome of a full ``run_preflight`` call: all four checks."""

    checks: tuple[PreflightCheck, ...]

    @property
    def ok(self) -> bool:
        """True unless a fatal check failed. Non-fatal failures do not
        affect this -- the caller emits ``preflight_warning`` for those and
        proceeds with the pass."""
        return not any(check.fatal and not check.ok for check in self.checks)

    @property
    def fatal_failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(check for check in self.checks if check.fatal and not check.ok)

    @property
    def non_fatal_failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(check for check in self.checks if not check.fatal and not check.ok)


@dataclass(frozen=True)
class PreflightPaths:
    """Filesystem anchors the checks are evaluated against.

    Every field is derived from paths the app already holds (repo root,
    state dir) -- never a hardcoded drive letter or absolute path, per issue
    #1363 AC1.
    """

    repo_root: Path
    state_dir: Path
    #: Directory the venv is expected to live in. Defaults to
    #: ``package_root / ".venv"`` (the standard layout) when not given
    #: explicitly.
    expected_venv_dir: Path | None = None
    #: The orchestrator's OWN source-tree root -- i.e. ``supervise.
    #: orchestrator_root()``, the checkout ``charlie_work`` is installed
    #: from and the target of self-deploy's ``git pull``/``uv sync``.
    #:
    #: Deliberately distinct from ``repo_root``: ``repo_root`` is the
    #: *target* repo a given pass is processing (charlie-work, another
    #: sibling repo, whatever is registered in the fleet), which varies per ``OrchestratorApp``
    #: instance even though every one of those instances runs from the same
    #: single orchestrator install. venv_identity's whole job is asking "is
    #: THIS PROCESS running the correct orchestrator code and venv" -- a
    #: question about the process, not about whichever repo it happens to
    #: be processing this pass -- so it must anchor on the orchestrator's
    #: own checkout, never the target repo's.
    #:
    #: Defaults to ``repo_root`` when not given, which keeps single-repo
    #: callers (every existing test, and any caller where the orchestrator
    #: manages only itself) working unchanged.
    orchestrator_root: Path | None = None

    @property
    def package_root(self) -> Path:
        return self.orchestrator_root if self.orchestrator_root is not None else self.repo_root

    @property
    def venv_dir(self) -> Path:
        if self.expected_venv_dir is not None:
            return self.expected_venv_dir
        return self.package_root / ".venv"


def _drive_anchor(path: Path) -> str:
    """Return the filesystem root component of *path* (``C:\\`` on Windows,
    ``/`` on POSIX) -- never a hardcoded drive letter, always derived from
    the path itself."""
    anchor = path.anchor
    return anchor if anchor else str(path)


def _check_disk_floor(
    paths: PreflightPaths,
    cfg: PreflightConfig,
    disk_usage: DiskUsageFn,
) -> PreflightCheck:
    name = "disk_floor"
    floor_bytes = cfg.disk_floor_gb * (1024**3)

    # Dedupe by drive anchor: repo_root and state_dir are usually the same
    # volume, but never assume it -- check every distinct anchor among the
    # paths the app already holds.
    anchors: dict[str, list[str]] = {}
    for label, raw in (("repo_root", paths.repo_root), ("state_dir", paths.state_dir)):
        try:
            resolved = Path(raw).resolve()
        except OSError:
            resolved = Path(raw)
        anchor = _drive_anchor(resolved)
        anchors.setdefault(anchor, []).append(label)

    worst_anchor: str | None = None
    worst_free: int | None = None
    errors: list[str] = []
    for anchor in anchors:
        try:
            usage = disk_usage(anchor)
        except OSError as exc:
            errors.append(f"{anchor}: {exc}")
            continue
        free = int(usage.free)  # type: ignore[attr-defined]
        if worst_free is None or free < worst_free:
            worst_free = free
            worst_anchor = anchor

    if worst_free is None:
        detail = f"disk_usage failed for every anchor: {'; '.join(errors)}"
        return PreflightCheck(name, ok=False, detail=detail, fatal=cfg.disk_floor_fatal)

    ok = worst_free >= floor_bytes
    labels = ", ".join(sorted({lbl for lbls in anchors.values() for lbl in lbls}))
    detail = (
        f"{worst_free / (1024**3):.2f} GB free on {worst_anchor!r} "
        f"(floor {cfg.disk_floor_gb} GB; covers: {labels})"
    )
    if errors:
        detail += f"; probe errors ignored (other anchor had a value): {'; '.join(errors)}"
    return PreflightCheck(name, ok=ok, detail=detail, fatal=cfg.disk_floor_fatal)


def _check_clock_sanity(
    paths: PreflightPaths,
    cfg: PreflightConfig,
    now: datetime,
    stat_fn: StatFn,
) -> PreflightCheck:
    name = "clock_sanity"
    state_file = paths.state_dir / layout.STATE_FILENAME
    try:
        stat_result = stat_fn(state_file)
    except OSError as exc:
        # No state.json yet (first-ever run) -- nothing to compare the clock
        # against. Not a failure: report ok so a fresh checkout doesn't spuriously
        # trip a tripwire that has nothing to measure.
        return PreflightCheck(
            name,
            ok=True,
            detail=f"state file not found ({state_file}): {exc}; clock check skipped",
            fatal=cfg.clock_sanity_fatal,
        )

    mtime = datetime.fromtimestamp(stat_result.st_mtime, tz=UTC)  # type: ignore[attr-defined]
    age_seconds = (now - mtime).total_seconds()
    max_age_seconds = cfg.clock_max_skew_hours * 3600

    # A negative age (state.json mtime in the future relative to `now`) is
    # always suspicious regardless of the configured bound -- it means the
    # clock moved backward, or state was written by a device with a
    # different/faster clock.
    ok = 0 <= age_seconds <= max_age_seconds
    detail = (
        f"state.json age {age_seconds:.1f}s relative to now={now.isoformat()} "
        f"(bounds: 0s..{max_age_seconds:.0f}s / {cfg.clock_max_skew_hours}h)"
    )
    return PreflightCheck(name, ok=ok, detail=detail, fatal=cfg.clock_sanity_fatal)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except AttributeError:  # pragma: no cover -- Path.is_relative_to is 3.9+
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


def _check_venv_identity(
    paths: PreflightPaths,
    cfg: PreflightConfig,
    *,
    sys_executable: str,
    package_file: str,
) -> PreflightCheck:
    name = "venv_identity"
    expected_venv = paths.venv_dir.resolve()
    expected_repo_root = paths.package_root.resolve()
    exec_path = Path(sys_executable).resolve()
    pkg_path = Path(package_file).resolve()

    exec_ok = _is_relative_to(exec_path, expected_venv)
    pkg_ok = _is_relative_to(pkg_path, expected_repo_root)
    ok = exec_ok and pkg_ok
    detail = (
        f"sys.executable={exec_path} expected under {expected_venv} (ok={exec_ok}); "
        f"charlie_work.__file__={pkg_path} expected under {expected_repo_root} (ok={pkg_ok})"
    )
    return PreflightCheck(name, ok=ok, detail=detail, fatal=cfg.venv_identity_fatal)


def _check_config_freshness(
    config_sources: Sequence[str],
    known_mtimes: dict[str, float],
    cfg: PreflightConfig,
    stat_fn: StatFn,
) -> PreflightCheck:
    name = "config_freshness"
    if not config_sources:
        return PreflightCheck(
            name,
            ok=True,
            detail="no config file sources recorded; nothing to check",
            fatal=cfg.config_freshness_fatal,
        )

    changed: list[str] = []
    for source in config_sources:
        path = Path(source)
        try:
            current_mtime = stat_fn(path).st_mtime  # type: ignore[attr-defined]
        except OSError:
            # File vanished since load -- not this check's job to flag a
            # missing config file; skip it rather than false-alarming.
            continue
        previous_mtime = known_mtimes.get(source)
        if previous_mtime is not None and current_mtime != previous_mtime:
            changed.append(f"{source}: mtime {previous_mtime} -> {current_mtime}")
        # Update the cache regardless -- this is what makes the event fire
        # exactly once per change: the next call's comparison is against the
        # NEW mtime, so a stable file reads ok again immediately after.
        known_mtimes[source] = current_mtime

    if changed:
        return PreflightCheck(
            name, ok=False, detail="; ".join(changed), fatal=cfg.config_freshness_fatal
        )
    return PreflightCheck(
        name,
        ok=True,
        detail="config file mtimes unchanged since last check",
        fatal=cfg.config_freshness_fatal,
    )


def run_preflight(
    paths: PreflightPaths,
    config: PreflightConfig,
    *,
    now: datetime | None = None,
    disk_usage: DiskUsageFn = shutil.disk_usage,
    stat_fn: StatFn = lambda p: Path(p).stat(),
    sys_executable: str | None = None,
    package_file: str | None = None,
    config_sources: Sequence[str] = (),
    known_config_mtimes: dict[str, float] | None = None,
) -> PreflightResult:
    """Run all four preflight checks, in order, and return their results.

    Every probe is injectable so this can be tested without touching a real
    disk, clock, or interpreter:

    - ``disk_usage`` defaults to ``shutil.disk_usage``.
    - ``stat_fn`` defaults to a real ``Path.stat()``, used for both the
      clock_sanity and config_freshness checks.
    - ``sys_executable``/``package_file`` default to the real
      ``sys.executable`` / ``charlie_work.__file__``.
    - ``known_config_mtimes``, if given, is a caller-owned dict mutated in
      place: the caller (e.g. ``OrchestratorApp``, in-memory for the life of
      the supervisor process) must hold onto it across passes so
      config_freshness's "exactly once per change" semantics work. A fresh
      dict (the default) means every pass looks like a first-ever
      observation -- harmless for a single-call test, but the real wiring
      must persist the same dict across passes.
    """
    if now is None:
        now = datetime.now(UTC)
    if sys_executable is None:
        sys_executable = sys.executable
    if package_file is None:
        package_file = charlie_work.__file__
    if known_config_mtimes is None:
        known_config_mtimes = {}

    checks = (
        _check_disk_floor(paths, config, disk_usage),
        _check_clock_sanity(paths, config, now, stat_fn),
        _check_venv_identity(
            paths, config, sys_executable=sys_executable, package_file=package_file
        ),
        _check_config_freshness(config_sources, known_config_mtimes, config, stat_fn),
    )
    return PreflightResult(checks=checks)


def _default_db_available(state_path: Path) -> bool:
    """Best-effort check that the events.db backing *state_path* can be
    opened. Reuses ``instrumentation``'s own connection cache/open logic
    (a disk-full host is exactly the case where this returns False) rather
    than re-implementing sqlite error handling here."""
    from . import instrumentation as _instrumentation

    return _instrumentation._get_db(state_path) is not None  # noqa: SLF001


def emit_preflight_refusal(
    state_path: Path,
    check: PreflightCheck,
    *,
    repo: str | None = None,
    log_event_fn: Callable[..., None] = log_event,
    db_available_fn: Callable[[Path], bool] | None = _default_db_available,
    stderr: TextIO = sys.stderr,
) -> None:
    """Emit ``loop_refused_preflight`` for a fatal check failure, best-effort.

    Disk-full is exactly the condition ``disk_floor`` describes, so the
    event write itself may fail (sqlite needs to allocate pages). This must
    never silence the refusal: when the event database is unreachable, or
    the event write itself raises, the refusal is printed to *stderr*
    instead -- which the ``fleet supervise`` wrapper already captures into
    the fleet pass log.

    ``db_available_fn`` and ``log_event_fn`` are both injectable so this can
    be unit-tested without a real disk-full condition.
    """
    payload = {"check": check.name, "detail": check.detail, "fatal": check.fatal}

    available = True
    if db_available_fn is not None:
        try:
            available = db_available_fn(state_path)
        except Exception:  # noqa: BLE001 -- must never raise
            available = False

    if available:
        try:
            log_event_fn(state_path, "loop_refused_preflight", payload, repo=repo, level="error")
            return
        except Exception as exc:  # noqa: BLE001 -- must never raise
            print(
                f"PREFLIGHT REFUSAL (event write failed: {exc}): {check.name}: {check.detail}",
                file=stderr,
            )
            return

    print(
        f"PREFLIGHT REFUSAL (event log unavailable): {check.name}: {check.detail}",
        file=stderr,
    )
