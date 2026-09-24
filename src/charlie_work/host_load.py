"""Host-load backpressure for the dispatch concurrency governor (issue #1843).

Motivation
----------
On a self-hosted fleet box, CI runs on the same machine the fleet dispatches
local workers to. Each dispatched worker spawns its own ``pytest`` suite, and
each suite can fan out to xdist workers, so dispatch that ignores host load
compounds contention until unrelated suites stall. Issue #1843's incident: a
``Tests`` job attempt was cancelled at its 22-minute timeout while ~18 other
pytest processes were live on the host (CI runs on sibling runners plus
several fleet-worker worktree suites under ``.var/.../worktrees/*``); the
rerun on an unloaded host finished in under 15 minutes. Nothing hung -- the
box was simply oversubscribed, and each wasted attempt cost a full Tests run
plus a rerun.

What it measures
----------------
``pytest_tree_load`` reduces one process snapshot to a ``HostLoad``: the
number of distinct live *pytest trees* and the total number of processes
inside them. A *tree root* is any process whose command line names a pytest
invocation (``pytest``, ``pytest.exe``, ``python -m pytest``, ``uv run pytest``,
...); every descendant of a root belongs to its tree, which is how xdist
worker processes (``python -u -c ...``, no ``pytest`` token of their own) get
counted. Nested pytest invocations inside an outer tree count once -- the
outer tree is the suite.

The issue offered either signal -- host CPU saturation or live pytest
process-tree count. This implements the process-tree count: it measures
exactly the load class dispatch is about to add to (test suites), it is the
same quantity the incident report counted, and unlike an instantaneous CPU%
sample it cannot be fooled by a brief compile spike into deferring a launch
the box could have absorbed. A CPU term can be added beside it later if
non-test host load ever needs to gate dispatch too.

Fail-open discipline
---------------------
``measure_host_load`` returns ``None`` -- "do not clamp" -- whenever the
process snapshot cannot be taken, and records why via a
``host_load_unavailable`` event so a permanently broken probe reads as a
diagnosable warning rather than a silently missing clamp (same discipline as
``ci_headroom``'s fail-open event, issue #1770). The write is edge-triggered
and rate-limited: it fires the first time the probe fails, and otherwise at
most once per ``min_interval_minutes`` while the failure persists -- never
once per dispatch pass unconditionally. The probe runs inside the
``_apply_concurrency_governor`` call every dispatch pass makes, so an
unconditional write would turn one broken PowerShell into an unbounded
warning stream in the very store the digest's warning counts read.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work import quiesce
from charlie_work.instrumentation import log_event, query_events

logger = logging.getLogger(__name__)

UNAVAILABLE_EVENT_KIND = "host_load_unavailable"

# Same cadence class as ci_headroom's diagnostic dedup: the governor runs once
# per dispatch pass (default pass interval 5 min), so a stuck probe produces a
# warning roughly six times an hour -- often enough to stay diagnosable, rare
# enough not to drown the digest's warning baseline.
DEFAULT_UNAVAILABLE_INTERVAL_MINUTES = 30

# Matches a pytest invocation token in a command line: ``pytest`` /
# ``pytest.exe`` at a word-ish boundary (start, whitespace, quote, or path
# separator) followed by whitespace, quote, or end. Deliberately rejects
# ``pytest.ini``/``pytest-xdist`` (followed by ``.``/``-``) and
# ``test_pytest.py`` (preceded by a word char). ``python -m pytest``,
# ``uv run pytest``, ``bash -c "... pytest ..."`` wrappers, and Windows
# ``Scripts\pytest.exe`` paths all match; xdist's ``python -u -c ...``
# workers deliberately do NOT (they are counted as tree descendants, not
# roots, so a suite and its workers form exactly one tree).
_PYTEST_INVOCATION_RE = re.compile(
    r'(?:^|[\s"\'/\\])pytest(?:\.exe)?(?=[\s"\']|$)',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HostLoad:
    """One host-wide measurement of live pytest process-tree load.

    ``pytest_tree_count`` is the number of distinct live pytest trees (i.e.
    how many separate test suites are running); ``pytest_process_count`` is
    the total number of processes inside those trees (controllers plus xdist
    workers plus any descendants), which is the number the dispatch governor
    threshold compares against.
    """

    pytest_tree_count: int
    pytest_process_count: int


def pytest_tree_load(
    processes: Iterable[quiesce.ProcessInfo],
    *,
    self_pid: int | None = None,
) -> HostLoad:
    """Reduce a process snapshot to the live pytest-tree load on the host.

    Pure and side-effect free so it is directly testable against a fabricated
    ``processes`` list. The tree containing ``self_pid`` (default: this
    process) is excluded so a measurement taken from inside a pytest suite --
    including this project's own test suite calling the function -- never
    counts itself as the load it is guarding against.
    """
    resolved_self_pid = self_pid if self_pid is not None else os.getpid()
    ppid_by_pid: dict[int, int] = {}
    children_by_ppid: dict[int, list[int]] = {}
    root_pids: set[int] = set()
    for proc in processes:
        ppid_by_pid[proc.pid] = proc.ppid
        children_by_ppid.setdefault(proc.ppid, []).append(proc.pid)
        if _PYTEST_INVOCATION_RE.search(proc.command_line or ""):
            root_pids.add(proc.pid)

    excluded = _self_tree(children_by_ppid, ppid_by_pid, root_pids, resolved_self_pid)

    seen: set[int] = set()
    trees = 0
    for pid in root_pids:
        if pid in seen or pid in excluded:
            continue
        trees += 1
        seen |= _subtree_pids(children_by_ppid, pid)
    return HostLoad(pytest_tree_count=trees, pytest_process_count=len(seen))


def _self_tree(
    children_by_ppid: dict[int, list[int]],
    ppid_by_pid: dict[int, int],
    root_pids: set[int],
    self_pid: int,
) -> set[int]:
    """Subtree of the *topmost* pytest root on ``self_pid``'s ancestor chain.

    Excluding the outermost containing tree -- rather than the nearest one --
    folds nested invocations (a suite that itself shells out to pytest) into
    the one suite this measurement is part of, so only *other* trees count as
    external load. Empty when the caller is not running under pytest at all.
    """
    node: int | None = self_pid
    chain: set[int] = set()
    top: int | None = None
    while node is not None and node not in chain:
        chain.add(node)
        if node in root_pids:
            top = node
        node = ppid_by_pid.get(node) or None  # ppid 0 / absent: end of chain
    if top is None:
        return set()
    return _subtree_pids(children_by_ppid, top)


def _subtree_pids(children_by_ppid: dict[int, list[int]], root: int) -> set[int]:
    """``root`` plus every descendant PID, over the in-memory snapshot."""
    out: set[int] = set()
    stack = [root]
    while stack:
        pid = stack.pop()
        if pid in out:
            continue
        out.add(pid)
        stack.extend(children_by_ppid.get(pid, ()))
    return out


def list_host_processes() -> tuple[Sequence[quiesce.ProcessInfo], str | None]:
    """Snapshot ``(pid, ppid, name, command_line)`` for every host process.

    Windows uses the ``Win32_Process`` CIM enumeration in
    ``quiesce.list_processes`` (PowerShell, no psutil dependency); POSIX
    platforms read ``/proc``. Returns ``(processes, error)`` rather than
    raising -- the same "external process errors come back as values"
    invariant the rest of the repo follows, so ``measure_host_load`` can
    fail open on an error instead of crashing a dispatch pass.
    """
    if sys.platform == "win32":
        return quiesce.list_processes()
    return _list_processes_posix()


def _list_processes_posix(
    proc_root: Path = Path("/proc"),
) -> tuple[Sequence[quiesce.ProcessInfo], str | None]:
    """``/proc``-backed process snapshot for non-Windows hosts.

    ``proc_root`` is a parameter (not a constant read at call time) so tests
    can point it at a fabricated procfs tree without touching the host's.
    Processes that exit or become unreadable mid-scan are skipped -- a
    snapshot can never be perfectly atomic, and a vanished entry is not an
    error worth failing the whole measurement over.
    """
    if not proc_root.is_dir():
        return (), f"{proc_root} does not exist on this platform"
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        return (), f"cannot list {proc_root}: {exc}"
    procs: list[quiesce.ProcessInfo] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (
                (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            ).strip()
            stat_tail = (entry / "stat").read_text().rpartition(")")[2]
            ppid = int(stat_tail.split()[1])
        except (OSError, ValueError, IndexError):
            continue
        procs.append(
            quiesce.ProcessInfo(
                pid=int(entry.name),
                ppid=ppid,
                name="",
                command_line=cmdline,
            )
        )
    return tuple(procs), None


def measure_host_load(
    *,
    lister: quiesce.ProcessLister | None = None,
    self_pid: int | None = None,
    diagnostic_state_path: Path | None = None,
    diagnostic_repo: str | None = None,
    now: datetime | None = None,
    min_interval_minutes: int = DEFAULT_UNAVAILABLE_INTERVAL_MINUTES,
) -> HostLoad | None:
    """Measure host pytest-tree load, or ``None`` when it cannot be trusted.

    ``lister`` defaults to `list_host_processes` (resolved through this
    module's globals at call time, so tests and the conftest autouse stub can
    substitute a fake with zero subprocess use). ``self_pid`` defaults to
    ``os.getpid()``; see `pytest_tree_load` for the self-exclusion rule.

    ``diagnostic_state_path``/``diagnostic_repo`` route the rate-limited
    ``host_load_unavailable`` event written on every fail-open path -- the
    governor passes its own per-repo ``state_file``/repo name so the
    diagnostic lands next to the ``dispatch_backpressure`` events for the
    same repo. When ``diagnostic_state_path`` is ``None`` the failure is only
    logged, not persisted.
    """
    resolved_lister: quiesce.ProcessLister = lister if lister is not None else list_host_processes
    processes, error = resolved_lister()
    if error is not None:
        _log_unavailable(
            diagnostic_state_path,
            diagnostic_repo,
            "measurement_failed",
            f"process listing failed: {error}",
            now=now if now is not None else datetime.now(UTC),
            min_interval_minutes=min_interval_minutes,
        )
        return None
    return pytest_tree_load(processes, self_pid=self_pid)


def _log_unavailable(
    state_path: Path | None,
    repo: str | None,
    reason: str,
    detail: str,
    *,
    now: datetime,
    min_interval_minutes: int,
) -> None:
    """Write a ``host_load_unavailable`` event, edge-triggered and rate-limited.

    Skips the write when the freshest prior event for this ``repo`` has the
    *same* reason and is younger than ``min_interval_minutes`` -- otherwise
    logs (first-ever, a reason transition, or the interval elapsed).
    events.db is the source of truth for "when did we last say this", not a
    new in-memory counter, matching ``ci_headroom``'s dedup discipline.
    """
    if state_path is None:
        logger.info("measure_host_load: unavailable (%s): %s", reason, detail)
        return
    if not _unavailable_should_emit(
        state_path, repo, reason, now=now, min_interval_minutes=min_interval_minutes
    ):
        return
    logger.info("measure_host_load(%s): unavailable (%s): %s", repo, reason, detail)
    log_event(
        state_path,
        UNAVAILABLE_EVENT_KIND,
        {"reason": reason, "detail": detail},
        repo=repo,
        level="warning",
    )


def _unavailable_should_emit(
    state_path: Path,
    repo: str | None,
    reason: str,
    *,
    now: datetime,
    min_interval_minutes: int,
) -> bool:
    """``True`` when a fresh ``host_load_unavailable`` write is warranted.

    Reads the single freshest prior event for this ``repo`` back from
    ``state_path`` -- cheap (one indexed, ``LIMIT 1`` query) and correct even
    across process restarts, since it derives from the same durable store the
    write lands in. A prior event whose timestamp cannot be parsed fails
    toward emitting, never toward silently suppressing forever.
    """
    previous = query_events(state_path, kind=UNAVAILABLE_EVENT_KIND, repo=repo, limit=1)
    if not previous:
        return True
    prior_payload = previous[0].get("payload")
    prior_reason = prior_payload.get("reason") if isinstance(prior_payload, dict) else None
    if prior_reason != reason:
        return True
    prior_time = _parse_event_ts(previous[0].get("ts"))
    if prior_time is None:
        return True
    return now - prior_time >= timedelta(minutes=min_interval_minutes)


def _parse_event_ts(raw: object) -> datetime | None:
    """Parse an events.db ``ts`` string (``instrumentation._now_iso``'s format).

    Mirrors ``ci_headroom._parse_event_ts``: UTC ISO-8601 with a ``Z``
    suffix, which ``datetime.fromisoformat`` accepts directly on this
    project's Python floor (3.11+).
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
