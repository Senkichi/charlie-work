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

Scope (issue #1943)
-------------------
Both counts are scoped to *orchestrator-attributable* trees. A merged tree
counts only when some member's -- or some member-ancestor's -- command line
references an orchestrator state path: the ``.var/charlie-work`` convention
marker (``layout.DEFAULT_STATE_DIR``; every default-layout repo on the host
keeps its worktrees, dispatch prompt files, and session records under it),
plus any caller-supplied ``scope_paths`` (the measuring repo's resolved
worktrees/state roots and every fleet-registered ``state_dir``, which keep
``runtime.state_dir``/``claude_code.worktrees_dir`` overrides attributable).

This is the #1943 fix: previously the count was host-wide, so three swole
self-hosted CI runners mid-suite (~50 xdist processes under
``C:\\actions-runners\\*\\_work``) tripped the 48-process brake and
hard-clamped *every* repo's dispatch to 0 for 13+ hours while host CPU sat
under 40%. CI fan-out is bounded by its own allocation ceiling (ci_fleet),
not by this governor -- so the brake's scope now matches the workload it
exists to bound: orchestrator-dispatched test suites. Attribution failures
degrade toward *not* counting -- the same fail-open direction the probe
itself uses.

The governor consumes both numbers (issue #1903): the tree count feeds the
primary clamp (``dispatch.host_load_max_pytest_trees``, applied as
remaining-suite headroom, since one launch ≈ one new suite), while the
process count is the fan-out brake
(``dispatch.host_load_max_pytest_processes`` -- catches a single suite run
at pathological ``-n`` width that a tree count cannot see).

The issue offered either signal -- host CPU saturation or live pytest
process-tree count. This implements the process-tree count: it measures
exactly the load class dispatch is about to add to (test suites), it is the
same quantity the incident report counted, and unlike an instantaneous CPU%
sample it cannot be fooled by a brief compile spike into deferring a launch
the box could have absorbed. (#1903 later confirmed the tree count was the
right signal: the original raw-process threshold was the part that
miscalibrated.) A CPU term can be added beside it later if non-test host
load ever needs to gate dispatch too.

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

from charlie_work import layout, process_chain, quiesce
from charlie_work.instrumentation import log_event, query_events

logger = logging.getLogger(__name__)

UNAVAILABLE_EVENT_KIND = "host_load_unavailable"

# The orchestrator-attribution marker (issue #1943): the state-dir convention
# every managed repo shares -- ``<repo>/.var/charlie-work``. Matched at
# state-dir granularity rather than ``.../worktrees`` so the ancestor walk in
# ``_tree_references_scope`` also attributes launches whose only
# path-carrying ancestor names a state-dir sibling (a devin worker's
# ``--prompt-file <state_dir>/dispatches/...``, a session record path).
# Derived from layout.DEFAULT_STATE_DIR -- never re-spelled (enforced by
# tests/test_no_path_literals.py rule 2).
_ORCHESTRATOR_SCOPE_MARKER = layout.DEFAULT_STATE_DIR

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
    """One measurement of live pytest process-tree load.

    ``pytest_tree_count`` is the number of distinct live pytest trees (i.e.
    how many separate test suites are running) -- the count the dispatch
    governor's suite-headroom clamp compares against;
    ``pytest_process_count`` is the total number of processes inside those
    trees (controllers plus xdist workers plus any descendants), which feeds
    the governor's fan-out brake. When the measurement was scoped
    (``pytest_tree_load``'s ``scope_paths`` -- the production behavior via
    ``measure_host_load``, issue #1943), both counts cover only
    orchestrator-attributable trees.
    """

    pytest_tree_count: int
    pytest_process_count: int


def pytest_tree_load(
    processes: Iterable[quiesce.ProcessInfo],
    *,
    self_pid: int | None = None,
    scope_paths: Iterable[str | Path] | None = None,
) -> HostLoad:
    """Reduce a process snapshot to the live pytest-tree load on the host.

    Pure and side-effect free so it is directly testable against a fabricated
    ``processes`` list. The tree containing ``self_pid`` (default: this
    process) is excluded so a measurement taken from inside a pytest suite --
    including this project's own test suite calling the function -- never
    counts itself as the load it is guarding against.

    ``scope_paths`` (issue #1943): when not ``None``, only
    *orchestrator-attributable* trees count -- a merged tree qualifies when
    any member's or member-ancestor's command line references one of the
    given path markers (see ``_tree_references_scope``). ``None`` measures
    every pytest tree on the host (the pre-#1943 behavior, kept for
    diagnostics and tests); ``measure_host_load`` always scopes.
    """
    resolved_self_pid = self_pid if self_pid is not None else os.getpid()
    snapshot = tuple(processes)
    rows = process_chain.proc_rows(snapshot)
    proc_by_pid: dict[int, quiesce.ProcessInfo] = {}
    children_by_ppid: dict[int, list[int]] = {}
    root_pids: set[int] = set()
    for proc in snapshot:
        proc_by_pid[proc.pid] = proc
        children_by_ppid.setdefault(proc.ppid, []).append(proc.pid)
        if _PYTEST_INVOCATION_RE.search(proc.command_line or ""):
            root_pids.add(proc.pid)

    excluded = _self_tree(children_by_ppid, rows, root_pids, resolved_self_pid)

    # A nested root's subtree lies inside its outer root's, so overlapping
    # subtrees are one suite. Merging by overlap -- rather than skipping roots
    # already ``seen`` -- makes the count independent of visit order (#1918:
    # a nested root with a lower PID than its outer root was counted twice)
    # and still counts a ppid cycle joining two roots as one tree.
    trees: list[set[int]] = []
    for pid in root_pids:
        if pid in excluded:
            continue
        merged = _subtree_pids(children_by_ppid, pid)
        disjoint = []
        for tree in trees:
            if tree & merged:
                merged |= tree
            else:
                disjoint.append(tree)
        trees = [*disjoint, merged]

    if scope_paths is not None:
        scope_re = _scope_matcher(scope_paths)
        trees = [
            tree
            for tree in trees
            if scope_re is not None and _tree_references_scope(tree, proc_by_pid, scope_re)
        ]

    seen = set().union(*trees) if trees else set()
    return HostLoad(pytest_tree_count=len(trees), pytest_process_count=len(seen))


def _normalize_scope_text(text: str) -> str:
    """Fold a command line or path marker to the scope-comparison form.

    ``os.path.normcase`` lowercases on Windows only (matching that platform's
    case-insensitive filesystem) and is identity elsewhere; the separator
    fold lets one needle match ``\\``, ``\\\\`` (escaped/JSON-embedded), and
    ``/`` spellings of the same path in a command line.
    """
    return re.sub(r"[/\\]+", "/", os.path.normcase(text))


def _scope_matcher(scope_paths: Iterable[str | Path]) -> re.Pattern[str] | None:
    """Compile the boundary-anchored path-marker matcher for ``scope_paths``.

    A needle matches only at path-token boundaries on both sides:
    ``.var/charlie-work`` must not match ``.var/charlie-work-legacy``
    (trailing) or ``repo.var/charlie-work`` (leading), and an absolute needle
    must not match its own tail inside a longer path. Returns ``None`` when
    no usable needle survives normalization -- an empty scope attributes
    nothing, which is what ``scope_paths=()`` must mean.
    """
    needles = [
        needle
        for raw in scope_paths
        if (needle := _normalize_scope_text(str(raw)).strip().rstrip("/"))
    ]
    if not needles:
        return None
    body = "|".join(re.escape(needle) for needle in needles)
    return re.compile(r"(?:^|[\s/\"'=;(])(?:" + body + r")(?=[/\s\"';)]|$)")


def _tree_references_scope(
    tree_pids: set[int],
    proc_by_pid: dict[int, quiesce.ProcessInfo],
    scope_re: re.Pattern[str],
) -> bool:
    """``True`` when the tree is attributable to an orchestrator-managed path.

    Checks every member's own command line plus its ancestor chain (bounded
    by the snapshot: already-visited pids are skipped, so a ppid cycle cannot
    spin). The ancestor half is what attributes the common pathless-root
    shapes -- ``uv run pytest``, ``python -m pytest`` on a PATH interpreter,
    ``bash -c "cd <wt> && pytest"`` -- whose own command line carries no
    managed path but whose launcher (the ``uv`` wrapper's venv child, a
    ``cd``-into-worktree shell, a worker harness's ``--prompt-file`` or
    session-dir argument) names one.
    """
    seen = set(tree_pids)
    stack = list(tree_pids)
    while stack:
        proc = proc_by_pid.get(stack.pop())
        if proc is None:
            continue
        if scope_re.search(_normalize_scope_text(proc.command_line or "")):
            return True
        if proc.ppid and proc.ppid not in seen:
            seen.add(proc.ppid)
            stack.append(proc.ppid)
    return False


def _self_tree(
    children_by_ppid: dict[int, list[int]],
    proc_rows: dict[int, process_chain.ProcRow],
    root_pids: set[int],
    self_pid: int,
) -> set[int]:
    """Subtree of the *topmost* pytest root on ``self_pid``'s ancestor chain.

    Excluding the outermost containing tree -- rather than the nearest one --
    folds nested invocations (a suite that itself shells out to pytest) into
    the one suite this measurement is part of, so only *other* trees count as
    external load. Empty when the caller is not running under pytest at all.

    The upward walk is ``process_chain.ancestor_chain_pids`` (issue #2058):
    it stops at a parent created after its child, so a pid recycled since the
    recorded parent exited cannot pull an unrelated process -- or the pytest
    tree above it -- into the exclusion set.
    """
    top: int | None = None
    for pid in process_chain.ancestor_chain_pids(self_pid, proc_rows):
        if pid in root_pids:
            top = pid
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
    scope_paths: Iterable[str | Path] | None = None,
    diagnostic_state_path: Path | None = None,
    diagnostic_repo: str | None = None,
    now: datetime | None = None,
    min_interval_minutes: int = DEFAULT_UNAVAILABLE_INTERVAL_MINUTES,
) -> HostLoad | None:
    """Measure orchestrator-attributable pytest load, or ``None`` on failure.

    ``lister`` defaults to `list_host_processes` (resolved through this
    module's globals at call time, so tests and the conftest autouse stub can
    substitute a fake with zero subprocess use). ``self_pid`` defaults to
    ``os.getpid()``; see `pytest_tree_load` for the self-exclusion rule.

    The measurement is always scoped (issue #1943): the built-in
    ``.var/charlie-work`` state-dir convention marker
    (``layout.DEFAULT_STATE_DIR``) covers every default-layout repo on the
    host, and ``scope_paths`` layers on additional markers -- the caller
    passes the repo's resolved worktrees/state roots plus the
    fleet-registered ``state_dir`` values so ``runtime.state_dir`` /
    ``claude_code.worktrees_dir`` overrides stay attributable. CI-runner
    suites and any other unattributable tree feed neither count.

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
    scope = [_ORCHESTRATOR_SCOPE_MARKER, *(scope_paths or ())]
    return pytest_tree_load(processes, self_pid=self_pid, scope_paths=scope)


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
