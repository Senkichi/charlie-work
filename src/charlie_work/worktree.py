"""Git worktree lifecycle for isolated per-branch worker environments.

Ports the battle-tested job-cannon shell scripts (``setup_worker.sh`` /
``finish_worker.sh``) into library code. The critical invariant this module
exists to enforce: worktrees may share ONE dev+eval virtualenv via a Windows
junction (or a symlink elsewhere) at ``<worktree>/.venv``, and naive removal
(``git worktree remove --force`` / ``rm -rf``) FOLLOWS that reparse point and
recursively deletes the shared venv's contents — corrupting every other live
worktree. ``remove_worktree`` orders teardown so the junction itself is
unlinked (never the target it points at) before the worktree is removed.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

from .attempt_refs import AttemptSnapshot, snapshot_attempt_ref
from .config import OrchestratorConfig, WORKER_OUTCOME_FILENAME, WRITER_MARKER_FILENAME
from . import git_pull_blockers
from .github import GitHubRunResult, PR_VIEW_MERGED_FIELDS, linked_issue_number
from .janitor import _calculate_patch_id
from . import layout
from .paths import runtime_paths
from .post_mortem import real_activity_for_worker
from .process_utils import is_pid_alive
from .safe_path import contains
from .safe_ref import require_valid_ref_name, require_valid_rev, require_valid_sha
from .subprocess_runner import RunResult, run_captured
from . import state as _state

_DEFAULT_TIMEOUT_SECONDS = 60
# Shorter timeout for network-touching git commands (ls-remote, fetch) so a
# stalled remote call cannot consume the entire local dispatch budget.
_REMOTE_TIMEOUT_SECONDS = 20


def _run_remote_captured(command: list[str], cwd: Path) -> RunResult:
    """Run a network-touching git command with a short, retryable timeout."""
    result = run_captured(command, cwd=cwd, timeout_seconds=_REMOTE_TIMEOUT_SECONDS)
    # Retry once on timeout: a transient network stall should not permanently
    # block reclaim of a pristine worktree.
    if result.timed_out:
        result = run_captured(command, cwd=cwd, timeout_seconds=_REMOTE_TIMEOUT_SECONDS)
    return result


# Sentinel values for operator claim markers. The operator marker intentionally
# does not encode the CLI invocation's transient PID; liveness is derived from
# the ``operator_claimed_at`` field in state.json.
OPERATOR_MARKER_SESSION_ID = "operator-claim"
OPERATOR_MARKER_KIND = "operator"


class WorktreeUnsafeError(RuntimeError):
    """Raised when ``create_worktree`` is about to reset a worktree that is
    CONFIRMED to contain local work (uncommitted modifications, or local
    commits not on the remote branch). This is a deterministic finding — the
    launch shim surfaces it as ``failure_kind="worktree_unsafe"``, which sits
    in ``config.DETERMINISTIC_ESCALATION_FAILURE_KINDS`` and escalates to a
    human on first occurrence. Do not raise this for a probe that merely
    *failed to determine* dirtiness — see ``WorktreeProbeFailedError``.
    """


class WorktreeProbeFailedError(RuntimeError):
    """Raised when the safety probe used to decide whether a worktree/branch
    reset would destroy local work itself failed (e.g. ``git status
    --porcelain`` hit an index lock, I/O error, or permissions issue) — NOT
    when the probe ran and confirmed the worktree is dirty.

    This is transient contention that an ordinary redispatch-cap retry would
    plausibly heal, unlike a confirmed-dirty worktree. The launch shim
    surfaces it as ``failure_kind="worktree_probe_failed"``, which must stay
    OFF ``config.DETERMINISTIC_ESCALATION_FAILURE_KINDS`` so it takes the
    normal redispatch-cap path instead of escalating on first occurrence
    (issue #288 follow-up review finding, PR #314).

    The reset is still refused (the caller cannot confirm safety), so this
    is a sibling of ``WorktreeUnsafeError`` rather than a subclass of it —
    keeping the two independent avoids an ``isinstance`` check on the more
    general class silently reclassifying a probe failure as confirmed-dirty.
    """


# Never pushed; charlie-work owns this namespace exclusively (same convention
# as attempt_refs.ATTEMPT_REF_PREFIX). A rescue ref preserves worker-authored
# dirty content immediately before a worktree reset that would destroy it, so
# a correct ``WorktreeUnsafeError`` refusal becomes recoverable instead of
# terminal (issue #849).
RESCUE_REF_PREFIX = "refs/charlie/rescue"


@dataclass(frozen=True)
class RescueCapture:
    """Result of capturing worktree work to a rescue ref before a reset.

    A rescue ref preserves worker-authored dirty content (tracked modifications
    + untracked files, excluding orchestrator scaffolding) so that a correct
    refusal to reset a worktree is recoverable instead of terminal (issue #849).

    ``ref_name``/``commit_sha`` are None when capture failed (``error`` set).
    Callers must treat capture failure as a hard refusal — the reset must NOT
    proceed if the work could not be preserved.
    """

    ref_name: str | None
    commit_sha: str | None
    error: str | None = None


class LiveWorkerRedispatchError(RuntimeError):
    """Raised when a recovery/redispatch path is about to destroy a worktree
    but the recorded worker process is still alive (or sessions.db shows fresh
    activity). Carries the probe result so the orchestrator can restore labels
    and emit a diagnostic event.

    ``inconclusive_probe_deferred_count`` mirrors the Signal-1 deferral counter
    used by the stall/dead lanes (issue #338). It is reset to 0 when a live
    signal is detected and incremented when the probe is inconclusive, so a
    structurally permanent no-match does not block recovery forever (issue #426).
    """

    def __init__(
        self,
        *,
        issue_number: int | None,
        pid: int | None,
        process_start_time: float | None,
        probe_result: str,
        inconclusive_probe_deferred_count: int = 0,
    ) -> None:
        self.issue_number = issue_number
        self.pid = pid
        self.process_start_time = process_start_time
        self.probe_result = probe_result
        self.inconclusive_probe_deferred_count = inconclusive_probe_deferred_count
        super().__init__(probe_result)


class WorktreeForeignWriterError(RuntimeError):
    """Raised when ``create_worktree`` is about to use a worktree that has a
    live writer marker belonging to a session the orchestrator does not own
    (e.g. an operator's editor or an out-of-band agent), OR when the target
    branch is already checked out in a worktree at a foreign path the
    orchestrator did not create (issue #1118). The launch shim surfaces this
    as ``failure_kind="worktree_foreign_writer"`` so the issue stays queued
    and the dispatch event log records the conflict.
    """

    def __init__(
        self,
        *,
        worktree_path: Path,
        pid: int | None,
        session_id: str | None,
    ) -> None:
        self.worktree_path = worktree_path
        self.pid = pid
        self.session_id = session_id
        if pid is None and session_id is None:
            # Issue #1118: the branch is checked out in a worktree at a path
            # the orchestrator did not create — no writer marker to inspect.
            super().__init__(
                f"worktree {worktree_path} is a foreign checkout the "
                f"orchestrator did not create; refusing to adopt it"
            )
        else:
            super().__init__(
                f"worktree {worktree_path} has a live foreign writer "
                f"(pid={pid}, session_id={session_id})"
            )


# How many lines of git's stderr to carry into a pre-merge failure message.
# Enough for git's "untracked working tree files would be overwritten" header
# plus the first few offending paths, without pasting a 200-line list into an
# event payload.
_PRE_MERGE_STDERR_LINES = 6


def _first_lines(text: str | None, *, limit: int) -> str:
    """Collapse the first ``limit`` non-blank lines of ``text`` onto one line."""
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    head = lines[:limit]
    joined = " | ".join(head)
    if len(lines) > limit:
        joined += f" | (+{len(lines) - limit} more lines)"
    return joined


class ReworkBranchConflictError(RuntimeError):
    """Raised when a rework worktree's failed pre-merge cannot even be
    recovered from — ``git merge --abort`` itself failed, or a ``MERGE_HEAD``
    is somehow still present afterward, leaving the worktree mid-merge and
    unusable.

    An ordinary merge conflict (the pre-merge fails, but ``merge --abort``
    cleanly restores the branch to its pre-merge tip) does NOT raise this —
    see ``_merge_update_rework_branch``, which returns a ``ReworkMergeConflict``
    notice instead so the rework worker launches and resolves it. Only the
    genuinely unrecoverable case reaches here. The launch shim surfaces it as
    ``failure_kind="rework_branch_conflict"``, which sits in
    ``config.DETERMINISTIC_ESCALATION_FAILURE_KINDS`` and escalates to a human
    on first occurrence.
    """

    def __init__(
        self,
        *,
        worktree_path: Path,
        branch: str,
        base_ref: str,
        conflicted_paths: tuple[str, ...],
        stderr: str | None = None,
        stage: str = "conflict",
    ) -> None:
        self.worktree_path = worktree_path
        self.branch = branch
        self.base_ref = base_ref
        self.conflicted_paths = conflicted_paths
        self.stderr = stderr
        self.stage = stage
        if stage == "pre_merge":
            # The merge never began, so there is no conflict to describe and
            # `conflicted_paths` is empty by construction. Naming the stage and
            # carrying git's own stderr is the whole diagnostic value here: the
            # message is what reaches the event payload via `str(exc)`, and a
            # bare "conflicts with base ...; conflicted paths: (unknown)" sent
            # a reader looking for a content conflict that does not exist.
            detail = _first_lines(stderr, limit=_PRE_MERGE_STDERR_LINES) or "(no stderr captured)"
            blocked = ", ".join(conflicted_paths) if conflicted_paths else "(none identified)"
            super().__init__(
                f"rework branch {branch!r} could not begin a merge with base "
                f"{base_ref!r} (no MERGE_HEAD — this is a pre-merge failure, "
                f"NOT a content conflict); blocking paths outside declared "
                f"scaffolding: {blocked}; git said: {detail}"
            )
            return
        paths_str = ", ".join(conflicted_paths) if conflicted_paths else "(unknown)"
        super().__init__(
            f"rework branch {branch!r} conflicts with base {base_ref!r}; "
            f"conflicted paths: {paths_str}"
        )


# Known porcelain flag keys that may appear as space-less lines (value=True)
# These are the only keys that map to True in git worktree --porcelain output
KNOWN_FLAG_KEYS = frozenset({"bare", "detached", "locked", "prunable"})


@dataclass(frozen=True)
class ReworkMergeConflict:
    """Structured notice describing a rework pre-merge that hit real conflicts.

    Populated on ``WorktreeInfo.rework_conflict`` when
    ``_merge_update_rework_branch`` could not cleanly merge the base ref into
    the rework branch: the merge was aborted (restoring the branch to its
    pre-merge tip) and the worktree is otherwise usable, so the launch
    proceeds and hands this notice to the worker instead of failing closed.

    ``base_branch`` is bare (no ``origin/`` prefix), so callers can render
    "merge origin/<base_branch>" consistently regardless of whether the
    conflict was found against a local or remote-tracking base ref.
    """

    base_branch: str
    base_sha: str
    conflicted_files: tuple[str, ...]


@dataclass(frozen=True)
class WorktreeInfo:
    path: Path
    branch: str
    venv_junction: Path | None
    # "fetch-fallback" | "pruned" | "salvaged" | "reset-origin:*" | None
    reclaimed: str | None = None
    # Set when a redispatch reset a branch tip that had commits worth
    # preserving (issue #261) — see attempt_refs.snapshot_attempt_ref.
    attempt_snapshot: AttemptSnapshot | None = None
    # Forward-slash-normalized worktree-relative paths written by the materializer
    # (issue #469). Derived from what the materializer actually writes, never a
    # hardcoded list, and used to keep injected orchestrator-owned files from
    # appearing as dirty in git status / CI-parity clean-tree gates.
    materialized_paths: tuple[str, ...] = ()
    # Set when the rework pre-merge step (_merge_update_rework_branch) hit a
    # real, recoverable merge conflict against the base ref. None for every
    # non-rework worktree and for a clean rework pre-merge.
    rework_conflict: ReworkMergeConflict | None = None
    # Set when a worktree reset was permitted only after capturing the dirty
    # working tree to a rescue ref (issue #849). None when no capture was
    # needed (the worktree was safe to reset) or when capture failed (the
    # reset was refused and WorktreeUnsafeError was raised instead).
    rescue_capture: RescueCapture | None = None


class WorktreeState(str, Enum):
    """Classification of a worker worktree after the process dies.

    - completed: branch is ahead of the dispatch base and the working tree is clean.
    - partial: the worktree has uncommitted changes (with or without commits).
    - no_commits: the working tree is clean and has no commits beyond the base.
    - unknown: git probing failed; treat as partial and avoid destructive actions.
    """

    COMPLETED = "completed"
    PARTIAL = "partial"
    NO_COMMITS = "no_commits"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class WorktreeInspection:
    """Result of inspecting a worker worktree after the process dies."""

    state: WorktreeState
    ahead_count: int = 0
    dirty: bool = False
    resolved_base_ref: str | None = None
    error: str | None = None


def _slugify(value: str, *, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:max_length].rstrip("-") or "worktree"


def _default_worktrees_dir(repo_root: Path) -> Path:
    """Return the package-default worktrees root, ignoring any config override.

    Equals the historical (pre-layout-module) path only when ``state_dir`` is
    left at its default AND ``claude_code.worktrees_dir`` is unset. This is
    the final fallback for direct library callers with no config in hand
    (e.g. some unit tests); the production paths -- dispatch and ``charlie
    worktree-clean`` -- always resolve through
    ``paths.resolved_layout(config, repo_root).worktrees`` instead, which
    honours both overrides.
    """
    return layout.worktrees_dir(layout.default_state_root(repo_root))


def worktree_path_for_branch(
    repo_root: Path, branch: str, worktrees_dir: Path | None = None
) -> Path:
    """Return the filesystem path for the worktree that serves ``branch``."""
    target_dir = worktrees_dir or _default_worktrees_dir(repo_root)
    return target_dir / _slugify(branch)


def _write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON atomically using a temp file + rename (issue #400)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def read_worker_outcome(worktree_path: Path) -> dict[str, Any] | None:
    """Read a worker's structured outcome file if present and well-formed.

    Workers write this file (per ``$section_push_pr_outcome``) when they
    successfully push a branch but cannot open a PR because ``gh`` is
    unauthenticated. The orchestrator reads it both from the worktree and,
    when the terminal-status watcher copies it, from durable terminal status.
    """
    path = worktree_path / WORKER_OUTCOME_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def write_worktree_marker(
    worktree_path: Path, pid: int, session_id: str, kind: str = "worker"
) -> None:
    """Write a ``.charlie-writer.json`` marker into the worktree root.

    Records the process id and a session identifier so the orchestrator can
    detect a live foreign writer before dispatching a second one into the
    same worktree. ``kind`` distinguishes long-lived operator claim markers
    (``pid`` is a sentinel) from ordinary worker session markers.
    """
    marker_path = worktree_path / WRITER_MARKER_FILENAME
    marker = {
        "pid": pid,
        "session_id": session_id,
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "kind": kind,
    }
    _write_json_atomic(marker_path, marker)


def read_worktree_marker(worktree_path: Path) -> dict[str, Any] | None:
    """Read the writer marker for ``worktree_path``, if any."""
    marker_path = worktree_path / WRITER_MARKER_FILENAME
    if not marker_path.exists():
        return None
    try:
        with marker_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def remove_worktree_marker(worktree_path: Path, session_id: str | None = None) -> bool:
    """Remove the writer marker for ``worktree_path``.

    If ``session_id`` is provided, only removes the marker when its
    ``session_id`` matches, preventing an operator-claim marker from being
    wiped by a worker reap.
    """
    marker_path = worktree_path / WRITER_MARKER_FILENAME
    if not marker_path.exists():
        return False
    if session_id is not None:
        marker = read_worktree_marker(worktree_path)
        if marker is None or marker.get("session_id") != session_id:
            return False
    try:
        marker_path.unlink()
        return True
    except OSError:
        return False


def _own_live_session_pids(sessions_dir: Path) -> dict[str, int]:
    """Map live recorded session ids to their pids from sidecar files."""
    live: dict[str, int] = {}
    if not sessions_dir.is_dir():
        return live
    for path in sessions_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        session_id = payload.get("session_id")
        pid = payload.get("pid")
        if not session_id or not isinstance(pid, int) or pid <= 0:
            continue
        if is_pid_alive(pid, payload.get("process_start_time")):
            live[str(session_id)] = pid
    return live


def _operator_marker_is_live(
    marker: dict[str, Any], state_file: Path | None, issue_number: int | None
) -> bool:
    """Return True when an operator marker corresponds to a live operator claim.

    Operator markers intentionally do not encode a real process id (the CLI
    that wrote them exits immediately). Liveness is derived from
    ``operator_claimed_at`` in state.json for the matching issue.
    """
    if state_file is None or issue_number is None:
        return False
    try:
        data = _state.load_state(state_file)
        entry = data.get("issues", {}).get(str(issue_number))
        return _state.is_operator_claimed(entry)
    except (OSError, ValueError, TypeError):
        # Treat unreadable or malformed state as "no live claim" so a stale
        # marker does not permanently block dispatch.
        return False


def _check_worktree_writer_marker(
    worktree_path: Path,
    sessions_dir: Path,
    issue_number: int | None = None,
    state_file: Path | None = None,
) -> None:
    """Refuse to enter a worktree that is currently occupied by a foreign writer.

    A marker with a dead pid is cleaned and the worktree is usable. A marker
    whose session id matches a live recorded session is considered an owned
    worker and is ignored here (dispatch/rework state guards prevent double
    dispatch). Any other live marker is treated as a foreign writer.

    Operator claim markers (``kind == "operator"`` or legacy ``operator-*``
    session ids) are live while state.json says the issue is operator-claimed,
    independent of PID.
    """
    marker = read_worktree_marker(worktree_path)
    if marker is None:
        return

    pid = marker.get("pid")
    session_id = marker.get("session_id")
    kind = marker.get("kind")

    if kind == OPERATOR_MARKER_KIND or (
        isinstance(session_id, str) and session_id.startswith("operator-")
    ):
        if _operator_marker_is_live(marker, state_file, issue_number):
            raise WorktreeForeignWriterError(
                worktree_path=worktree_path, pid=pid, session_id=session_id
            )
        # Claim released or state unavailable: clean the marker and proceed.
        remove_worktree_marker(
            worktree_path, session_id=session_id if isinstance(session_id, str) else None
        )
        return

    if not isinstance(pid, int) or pid <= 0 or not is_pid_alive(pid, None):
        # Marker is stale — clean it and proceed.
        remove_worktree_marker(worktree_path)
        return
    own = _own_live_session_pids(sessions_dir)
    if session_id and own.get(session_id) == pid:
        # Marker belongs to a live session we already know about.
        return
    raise WorktreeForeignWriterError(worktree_path=worktree_path, pid=pid, session_id=session_id)


def _read_origin_head_symref(repo_root: Path) -> str | None:
    """Read refs/remotes/origin/HEAD and return "origin/<branch>", or None."""
    result = run_captured(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if result.ok and result.stdout.strip():
        # Output is "refs/remotes/origin/main" -> extract "origin/main"
        ref = result.stdout.strip()
        if ref.startswith("refs/remotes/"):
            return ref[len("refs/remotes/") :]
    return None


def _resolve_default_branch_ref(repo_root: Path) -> str:
    """Resolve the repository's default branch as a remote-tracking ref.

    Returns a string like "origin/main" or "origin/master". Only when the repo
    has no origin remote at all (pure-local repos, e.g. test fixtures) does it
    return "HEAD".

    Uses git symbolic-ref refs/remotes/origin/HEAD, the standard way to read
    the default branch without the GitHub CLI. That ref is only written at
    clone time (or by git remote set-head), so when it is missing, heal it
    with ``git remote set-head origin --auto`` and retry once.

    Raises:
        RuntimeError: an origin remote exists but the default branch cannot be
            determined even after the set-head heal. Falling back to local
            HEAD here is how a stale/dirty operator checkout leaks into fresh
            worker branches (issue #239: two agent PRs shipped the operator's
            unpublished docs commit). The adapter boundary converts this raise
            into a SessionRecord error value, failing the one dispatch loudly.
    """
    if not _has_origin_remote(repo_root):
        # Pure-local repo: local HEAD is the only meaningful base.
        return "HEAD"

    resolved = _read_origin_head_symref(repo_root)
    if resolved is not None:
        return resolved

    # origin/HEAD is unset on this clone; set-head persists it in-repo so both
    # future dispatches and operator git usage benefit. Failure surfaces via
    # the retried symref read below, not an exception (run_captured contract).
    set_head_result = run_captured(
        ["git", "remote", "set-head", "origin", "--auto"],
        cwd=repo_root,
        timeout_seconds=_REMOTE_TIMEOUT_SECONDS,
    )
    resolved = _read_origin_head_symref(repo_root)
    if resolved is not None:
        return resolved

    detail = set_head_result.error or set_head_result.stderr
    raise RuntimeError(
        "Cannot resolve origin's default branch: refs/remotes/origin/HEAD is "
        "unset and 'git remote set-head origin --auto' did not heal it"
        f"{f' ({detail})' if detail else ''}. "
        "Refusing to base a fresh worktree on local HEAD (issue #239)."
    )


def _has_origin_remote(repo_root: Path) -> bool:
    """Check if the repo has an 'origin' remote configured.

    Returns True if 'git remote get-url origin' succeeds (exit code 0),
    False otherwise. This is a deterministic check for remote existence
    before attempting fetch operations.
    """
    result = run_captured(
        ["git", "remote", "get-url", "origin"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    return result.ok


def _remote_branch_exists(repo_root: Path, branch: str) -> bool | None:
    """Check if a branch exists on the origin remote.

    Returns True if the branch exists, False if it does not exist, or None if the probe
    failed (network/auth error). This distinguishes 'remote ref missing' from 'broken remote':
    - exists: exit 0 AND non-empty stdout
    - missing: exit 0 AND empty stdout
    - probe-failed: nonzero exit (e.g., network error, auth failure)

    Uses the shorter network timeout so a stalled probe cannot consume the full
    local dispatch budget.
    """
    result = _run_remote_captured(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=repo_root,
    )
    if not result.ok:
        # Probe failed (network error, auth failure, etc.)
        return None
    if result.stdout.strip():
        # Branch exists
        return True
    # Branch does not exist (exit 0 with empty stdout)
    return False


def _remote_branch_head_sha(repo_root: Path, branch: str) -> str | None:
    """Return the commit SHA for ``origin/{branch}`` if it exists, else None.

    A probe failure is also returned as ``None``; call ``_remote_branch_exists``
    if you need to distinguish a missing ref from a broken remote.
    """
    result = _run_remote_captured(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=repo_root,
    )
    if not result.ok:
        return None
    stdout = result.stdout.strip()
    if not stdout:
        return None
    # ls-remote output: "<sha>\trefs/heads/<name>"
    return stdout.split()[0]


def remote_branch_head_sha(repo_root: Path, branch: str) -> str | None:
    """Public wrapper for ``_remote_branch_head_sha``.

    Returns the commit SHA for ``origin/{branch}`` if it exists, else ``None``.
    Used by the orphan-sweep redispatch cap (issue #1243) to measure branch
    progress across no-open-PR redispatches: an unchanged remote head SHA across
    attempts is evidence the worker pushed nothing new.
    """
    return _remote_branch_head_sha(repo_root, branch)


def worktree_head_sha(worktree_path: Path) -> str | None:
    """Return the HEAD commit SHA of the worktree at ``worktree_path``.

    Returns ``None`` when the worktree does not exist or ``git rev-parse HEAD``
    fails. Used by the orphan-sweep redispatch cap (issue #1243) to detect
    stranded commits -- work the worker completed but never pushed. A moving
    local head with a dead worker is the salvage path's job, not an escalation.
    """
    if not worktree_path.is_dir():
        return None
    result = run_captured(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not result.ok:
        return None
    return result.stdout.strip() or None


def remote_branch_ahead_count(
    repo_root: Path, branch: str, base_ref: str = ""
) -> tuple[int | None, str | None]:
    """Return how many commits ``origin/{branch}`` is ahead of its base branch.

    Returns ``(ahead_count, error)``. ``ahead_count`` is ``None`` when the branch
    does not exist on origin, the base cannot be resolved, or git fails. A
    returned count of ``0`` means the branch points at the base or is behind it
    (no commits unique to the branch).
    """
    head_sha = _remote_branch_head_sha(repo_root, branch)
    if head_sha is None:
        return None, f"remote branch {branch} does not exist or probe failed"

    base_branch = resolve_base_branch_name(repo_root, base_ref)
    base_ref_local = f"origin/{base_branch}"
    base_result = run_captured(
        ["git", "rev-parse", "--verify", base_ref_local],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not base_result.ok:
        # Try a fetch to refresh the tracking ref before giving up.
        fetch_result = run_captured(
            ["git", "fetch", "origin", base_branch],
            cwd=repo_root,
            timeout_seconds=_REMOTE_TIMEOUT_SECONDS,
        )
        if not fetch_result.ok:
            return None, (
                f"base branch {base_branch!r} not resolvable: "
                f"{base_result.error or base_result.stderr}"
            )
        base_result = run_captured(
            ["git", "rev-parse", "--verify", base_ref_local],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not base_result.ok:
            return None, (
                f"base branch {base_branch!r} not resolvable after fetch: "
                f"{base_result.error or base_result.stderr}"
            )
    base_sha = base_result.stdout.strip()

    merge_base_result = run_captured(
        ["git", "merge-base", base_sha, head_sha],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not merge_base_result.ok:
        return None, f"merge-base failed: {merge_base_result.error or merge_base_result.stderr}"
    merge_base = merge_base_result.stdout.strip()

    count_result = run_captured(
        ["git", "rev-list", "--count", f"{merge_base}..{head_sha}"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not count_result.ok:
        return None, f"rev-list failed: {count_result.error or count_result.stderr}"
    try:
        return int(count_result.stdout.strip()), None
    except ValueError:
        return None, f"rev-list returned non-integer: {count_result.stdout!r}"


def worktree_ahead_of_sha(worktree_path: Path, base_sha: str) -> tuple[int | None, str | None]:
    """Return how many commits the worktree HEAD is ahead of ``base_sha``.

    Used by the rework death-loop escalation (issue #1134) to detect
    *stranded* work — commits the worker made but never pushed because it
    died mid-push.  Returns ``(ahead_count, error)``.  ``ahead_count`` is
    ``None`` when the worktree does not exist, ``base_sha`` is not a known
    object, or git fails.  A count of ``0`` means the worktree HEAD is at
    or behind ``base_sha`` (no stranded commits).
    """
    if not worktree_path.is_dir():
        return None, f"worktree does not exist: {worktree_path}"

    head_result = run_captured(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not head_result.ok:
        return None, f"rev-parse HEAD failed: {head_result.error or head_result.stderr}"
    head_sha = head_result.stdout.strip()

    if not _object_exists(worktree_path, base_sha):
        return None, f"base sha not in worktree object store: {base_sha}"

    merge_base_result = run_captured(
        ["git", "merge-base", base_sha, head_sha],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not merge_base_result.ok:
        return None, f"merge-base failed: {merge_base_result.error or merge_base_result.stderr}"
    merge_base = merge_base_result.stdout.strip()

    count_result = run_captured(
        ["git", "rev-list", "--count", f"{merge_base}..{head_sha}"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not count_result.ok:
        return None, f"rev-list failed: {count_result.error or count_result.stderr}"
    try:
        return int(count_result.stdout.strip()), None
    except ValueError:
        return None, f"rev-list returned non-integer: {count_result.stdout!r}"


def salvage_head_on_base(repo_root: Path, head_sha: str, base_ref: str) -> bool | None:
    """Return whether ``head_sha`` is already reachable from ``origin/<base>``.

    Used by the salvage lane's superseded-work check (issue #1241): a dead
    worker's worktree may still hold commits that have *already* merged to the
    base branch via another path (an operator salvage, or the rework lane's
    own auto-salvage). Opening a "Salvaged work for #N" PR for those commits
    produces a duplicate that consumes a review session and leaves a stale
    branch for the operator to clean up.

    ``git merge-base --is-ancestor <head> origin/<base>`` after a fetch is the
    reachability test the issue specifies -- reachability, not branch-existence,
    because the agent branch may already be deleted while its commits live on
    in the merge commit's history.

    Returns ``True`` when ``head_sha`` is an ancestor of the fetched base tip
    (the work is already on the base branch -- salvage would duplicate it),
    ``False`` when it is provably not, and ``None`` when the probe could not
    decide (fetch failed, ref unresolvable, object missing, merge-base
    errored). ``None`` is fail-open: the caller proceeds with salvage as
    today, because a duplicate PR is recoverable while silently dropped work
    is not.
    """
    try:
        safe_head = require_valid_sha(head_sha, context="salvage_head_on_base head_sha")
    except ValueError:
        return None

    base_branch = resolve_base_branch_name(repo_root, base_ref)
    # Always refresh the tracking ref before the ancestry check. The merge
    # that landed the work happened on the remote, so the local
    # ``origin/<base>`` is stale until fetched -- checking ancestry against a
    # stale ref would read "not an ancestor" and open a duplicate PR, the
    # exact defect this probe exists to prevent. The issue specifies the
    # fetch explicitly: "``git merge-base --is-ancestor <head> origin/main``
    # after a fetch".
    fetch_result = run_captured(
        ["git", "fetch", "origin", base_branch],
        cwd=repo_root,
        timeout_seconds=_REMOTE_TIMEOUT_SECONDS,
    )
    if not fetch_result.ok:
        # Fail-open: a fetch failure proceeds with salvage as today. A
        # duplicate PR is recoverable; silently dropped work is not.
        return None
    base_ref_local = f"origin/{base_branch}"
    base_result = run_captured(
        ["git", "rev-parse", "--verify", base_ref_local],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not base_result.ok:
        return None
    base_sha = base_result.stdout.strip()

    # Gate on object existence: ``--is-ancestor`` exits non-zero both for "not
    # an ancestor" and for "unknown object" (see ``_object_exists``). The head
    # lives in the shared object store (worktrees share it), but be defensive
    # against a pruned/missing object and fail open rather than misclassify.
    if not _object_exists(repo_root, safe_head) or not _object_exists(repo_root, base_sha):
        return None

    ancestor_result = run_captured(
        ["git", "merge-base", "--is-ancestor", safe_head, base_sha],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    # exit 0 => ancestor (superseded); exit 1 => not ancestor; 128/other => error.
    if ancestor_result.returncode == 0:
        return True
    if ancestor_result.returncode == 1:
        return False
    return None


@dataclass(frozen=True)
class SalvagePushResult:
    """Outcome of a salvage-push attempt on a dead worker's worktree (#1248).

    Three shapes, discriminated by ``pushed``/``skip_reason``/``error``:
    ``pushed=True`` (work published), ``skip_reason`` set (preconditions not
    met -- nothing was attempted, the worktree is untouched), or ``error``
    set (the push itself was attempted and failed; ``old_remote_sha`` and
    ``commit_count`` still carry what was known at attempt time).
    """

    pushed: bool
    skip_reason: str | None = None
    error: str | None = None
    old_remote_sha: str | None = None
    new_remote_sha: str | None = None
    commit_count: int | None = None


def salvage_push_stranded_commits(
    repo_root: Path,
    branch: str,
    worktree_path: Path,
    *,
    base_ref: str = "",
) -> SalvagePushResult:
    """Fast-forward-push committed-but-unpushed work from a dead worker's worktree.

    A worker that completed its work locally but died before ``git push``
    presents to the orphan sweep exactly like a worker that did nothing: zero
    remote delta. This helper publishes that stranded work so the sweep can
    classify against the real head instead of redispatching (issue #1248).

    Safety properties, in order of enforcement:

    - The caller must have already established the worker is dead (state PID
      records); this helper additionally refuses when the worktree carries a
      live or operator writer marker.
    - Push happens only when the remote tip is an ancestor of the local branch
      tip (a pure fast-forward), or when the branch does not exist on origin
      yet and the worktree has commits beyond the base branch. **Never** a
      force push -- and ``push_branch`` is a plain push, so even a race with a
      concurrent remote write fails closed on the server side.
    - Ancestry is only consulted after ``_object_exists`` confirms the remote
      SHA is present locally: ``merge-base --is-ancestor`` cannot distinguish
      "not an ancestor" from "unknown object", and a remote tip this worktree
      has never seen means someone else pushed -- diverged, skip.

    Never raises; every failure comes back as a value.
    """
    try:
        branch = require_valid_ref_name(branch, context="salvage_push branch")
    except ValueError as exc:
        return SalvagePushResult(pushed=False, skip_reason=f"invalid_branch: {exc}")

    if not worktree_path.is_dir():
        return SalvagePushResult(pushed=False, skip_reason="no_worktree")

    marker = read_worktree_marker(worktree_path)
    if marker is not None:
        if marker.get("kind") == OPERATOR_MARKER_KIND:
            return SalvagePushResult(pushed=False, skip_reason="operator_claimed")
        marker_pid = marker.get("pid")
        if isinstance(marker_pid, int) and marker_pid > 0 and is_pid_alive(marker_pid):
            return SalvagePushResult(pushed=False, skip_reason="live_writer_marker")

    local_result = run_captured(
        ["git", "rev-parse", "--verify", f"refs/heads/{branch}"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not local_result.ok:
        return SalvagePushResult(pushed=False, skip_reason="branch_ref_missing")
    local_sha = local_result.stdout.strip()

    remote_exists = _remote_branch_exists(worktree_path, branch)
    if remote_exists is None:
        return SalvagePushResult(pushed=False, skip_reason="remote_probe_failed")

    old_remote_sha: str | None = None
    if remote_exists:
        old_remote_sha = _remote_branch_head_sha(worktree_path, branch)
        if old_remote_sha is None:
            return SalvagePushResult(pushed=False, skip_reason="remote_probe_failed")
        if old_remote_sha == local_sha:
            return SalvagePushResult(
                pushed=False, skip_reason="up_to_date", old_remote_sha=old_remote_sha
            )
        if not _object_exists(worktree_path, old_remote_sha):
            return SalvagePushResult(
                pushed=False,
                skip_reason="remote_head_not_local",
                old_remote_sha=old_remote_sha,
            )
        if not _is_ancestor(worktree_path, old_remote_sha, local_sha):
            return SalvagePushResult(
                pushed=False, skip_reason="diverged", old_remote_sha=old_remote_sha
            )
        count_result = run_captured(
            ["git", "rev-list", "--count", f"{old_remote_sha}..{local_sha}"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not count_result.ok:
            return SalvagePushResult(
                pushed=False, skip_reason="count_failed", old_remote_sha=old_remote_sha
            )
        try:
            commit_count = int(count_result.stdout.strip())
        except ValueError:
            return SalvagePushResult(
                pushed=False, skip_reason="count_failed", old_remote_sha=old_remote_sha
            )
        if commit_count == 0:
            return SalvagePushResult(
                pushed=False,
                skip_reason="no_stranded_commits",
                old_remote_sha=old_remote_sha,
            )
    else:
        # Branch never made it to origin. Only publish it when the worktree
        # actually carries commits beyond the base branch -- creating an empty
        # remote branch would feed the #935 open-PR lane a PR with no diff.
        base_branch = resolve_base_branch_name(repo_root, base_ref)
        base_result = run_captured(
            ["git", "rev-parse", "--verify", f"origin/{base_branch}"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not base_result.ok:
            return SalvagePushResult(pushed=False, skip_reason="base_unresolvable")
        merge_base_result = run_captured(
            ["git", "merge-base", base_result.stdout.strip(), local_sha],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not merge_base_result.ok:
            return SalvagePushResult(pushed=False, skip_reason="base_unresolvable")
        count_result = run_captured(
            ["git", "rev-list", "--count", f"{merge_base_result.stdout.strip()}..{local_sha}"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not count_result.ok:
            return SalvagePushResult(pushed=False, skip_reason="count_failed")
        try:
            commit_count = int(count_result.stdout.strip())
        except ValueError:
            return SalvagePushResult(pushed=False, skip_reason="count_failed")
        if commit_count == 0:
            return SalvagePushResult(pushed=False, skip_reason="no_commits_beyond_base")

    ok, push_error = push_branch(repo_root, branch, worktree_path)
    if not ok:
        return SalvagePushResult(
            pushed=False,
            error=f"push_failed: {push_error}",
            old_remote_sha=old_remote_sha,
            commit_count=commit_count,
        )
    return SalvagePushResult(
        pushed=True,
        old_remote_sha=old_remote_sha,
        new_remote_sha=local_sha,
        commit_count=commit_count,
    )


def _object_exists(repo_root: Path, sha: str) -> bool:
    """Return True if ``sha`` names an object present in the local object store.

    Callers that ask ancestry questions must gate on this first: ``git
    merge-base --is-ancestor`` exits non-zero both for "not an ancestor" and
    for "I have never heard of that object", and conflating the two produces a
    correct-by-accident decision attached to a false reason string.
    """
    result = run_captured(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    return result.ok


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    """Return True if ``ancestor`` is an ancestor of ``descendant``."""
    result = run_captured(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    return result.ok


def _resolve_merge_base_ref(repo_root: Path, base_ref: str) -> str:
    """Resolve ``base_ref`` to a concrete ref that can be merged inside a worktree.

    Remote-tracking refs (``origin/main``) are returned as-is so callers can
    fetch/merge them. The ``HEAD`` sentinel is resolved to the main worktree's
    current branch name or, when detached, its tip SHA, so ``git merge HEAD``
    inside a child worktree does not accidentally merge the child branch into
    itself.
    """
    if base_ref != "HEAD":
        return base_ref
    sym_result = run_captured(
        ["git", "symbolic-ref", "HEAD"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if sym_result.ok:
        ref = sym_result.stdout.strip()
        if ref.startswith("refs/heads/"):
            return ref[len("refs/heads/") :]
    rev_result = run_captured(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if rev_result.ok and rev_result.stdout.strip():
        return rev_result.stdout.strip()
    return base_ref


def _merge_head_present(worktree_path: Path) -> bool:
    """Return True if ``worktree_path`` is still mid-merge (MERGE_HEAD exists).

    Uses ``git rev-parse --verify`` rather than reaching for the ``.git``
    filesystem entry directly — a linked worktree's ``.git`` is a file
    pointing at the real gitdir elsewhere, so plumbing is the only
    worktree-agnostic way to ask this question.
    """
    result = run_captured(
        ["git", "rev-parse", "--verify", "-q", "MERGE_HEAD"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    return result.ok


def _merge_update_rework_branch(
    repo_root: Path,
    worktree_path: Path,
    branch: str,
    base_ref: str,
    injected_paths: tuple[str, ...] = (),
    materialize_dirs: tuple[str, ...] = (),
) -> ReworkMergeConflict | None:
    """Merge-update a checked-out rework branch onto the current base.

    Before a rework worker starts, merge the resolved base ref into the PR
    branch. If GitHub later tries to build ``refs/pull/N/merge`` it will see a
    branch that already contains the latest base changes, which avoids the
    "DIRTY/no CI" stall for the common case.

    A real conflict does NOT abort worktree creation: launching a worker is
    the only thing that can ever resolve it (nobody rebases a branch whose
    worktree was never created — issue investigation "rework-conflict"). The
    merge is aborted, restoring the branch to its pre-merge tip, and the
    conflict is handed back as a ``ReworkMergeConflict`` so the caller can
    brief the worker to merge and resolve it itself.

    Returns:
        None if the merge completed cleanly (including "already up to date").
        A ``ReworkMergeConflict`` describing the aborted merge otherwise.

    ``injected_paths`` and ``materialize_dirs`` name the orchestrator's own
    scaffolding. They are used only to clear a self-inflicted pre-merge
    collision (see below); nothing outside them is ever removed.

    Raises:
        ReworkBranchConflictError: in two distinct situations, distinguished by
            the exception's ``stage``.

            ``stage="conflict"`` — a merge really was in progress and recovery
            from it failed: ``git merge --abort`` reported an error AND a
            ``MERGE_HEAD`` is still present. The worktree is mid-merge and
            unusable, so this escalates instead of handing a worker a broken
            workspace.

            ``stage="pre_merge"`` — the merge never began (no ``MERGE_HEAD``),
            and the reason could not be remediated. This is NOT a content
            conflict; the branch may merge perfectly cleanly. Two causes are
            remediated, both being the orchestrator's own scaffolding colliding
            with the incoming tree in a reused worktree, and both repaired
            before a single retry: an *untracked* copy of a base-tracked file
            (removed), and a *locally modified tracked* file the merge would
            overwrite (restored from ``HEAD``). Anything else — or any blocker
            outside the declared scaffolding — is left untouched and escalated.
        RuntimeError: if the base ref cannot be fetched or the merge command
            itself cannot be run.
    """
    merge_base_ref = _resolve_merge_base_ref(repo_root, base_ref)

    if merge_base_ref.startswith("origin/") and _has_origin_remote(repo_root):
        remote_branch = merge_base_ref[len("origin/") :]
        fetch_result = run_captured(
            ["git", "fetch", "origin", remote_branch],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not fetch_result.ok:
            raise RuntimeError(
                f"Fetch failed for base ref {merge_base_ref!r}: "
                f"{fetch_result.error or fetch_result.stderr}"
            )

    merge_result = run_captured(
        ["git", "merge", "--no-edit", merge_base_ref],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if merge_result.ok:
        return None

    # A failed `git merge` has three outcomes, not two, and only MERGE_HEAD
    # separates them. Capture it BEFORE aborting, because `merge --abort`
    # destroys the very state being asked about:
    #
    #   (a) merge started, conflicted, aborts cleanly  -> ReworkMergeConflict
    #   (b) merge started, conflicted, abort fails     -> unrecoverable, raise
    #   (c) merge never started at all                 -> not a conflict
    #
    # Case (c) is what an untracked-file collision produces: the merge is
    # refused up front, so there is no MERGE_HEAD, `--diff-filter=U` is empty,
    # and `merge --abort` exits 128 "There is no merge to abort". Keying the
    # raise off `not abort_result.ok` read that failure as evidence of a broken
    # mid-merge worktree when it means the exact opposite — the tree is clean
    # and untouched. That misread laundered a self-inflicted scaffolding
    # collision into `rework_branch_conflict`, which is a deterministic-
    # escalation kind, so 17 branches were parked at `agent:human-needed` on
    # first occurrence with no real conflict between any of them and the base.
    if not _merge_head_present(worktree_path):
        # Case (c). The common cause is orchestrator scaffolding left behind by
        # a previous attempt on a reused worktree: materialization runs later in
        # `create_worktree` than this merge, so on a fresh worktree these files
        # do not exist yet, and on a reused one they are untracked copies of
        # paths the base now tracks. Clear only those and retry once.
        #
        # Scaffolding blocks a merge in two distinct ways and git reports them
        # with two different messages, so both sets are collected before
        # deciding anything. Repairing only the untracked half leaves a worktree
        # that is *also* holding a modified tracked copy still blocked, and the
        # retry then fails for a reason the first repair could never have
        # addressed.
        untracked_blocking = _untracked_paths_shadowing_ref(worktree_path, merge_base_ref)
        modified_blocking = _modified_paths_overwritten_by_ref(worktree_path, merge_base_ref)
        blocking = untracked_blocking + modified_blocking
        undeclared = tuple(
            path
            for path in blocking
            if not _declared_scaffolding_matcher(injected_paths, materialize_dirs)(path)
        )
        # The declared/undeclared verdict is taken over the *union*: a single
        # undeclared blocker in either class means nothing is repaired at all,
        # because a partial repair that still fails the merge would destroy
        # files for no benefit.
        if (
            blocking
            and not undeclared
            and _repair_declared_scaffolding_blockers(
                worktree_path,
                untracked_blocking,
                modified_blocking,
                injected_paths,
                materialize_dirs,
            )
        ):
            retry_result = run_captured(
                ["git", "merge", "--no-edit", merge_base_ref],
                cwd=worktree_path,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            )
            if retry_result.ok:
                return None
            if _merge_head_present(worktree_path):
                # The retry got far enough to conflict for real. Fall through to
                # the conflict path below so the worker is briefed to resolve it.
                merge_result = retry_result
            else:
                raise ReworkBranchConflictError(
                    worktree_path=worktree_path,
                    branch=branch,
                    base_ref=merge_base_ref,
                    conflicted_paths=(),
                    stderr=retry_result.stderr or retry_result.error,
                    stage="pre_merge",
                )
        else:
            # Either nothing identifiable is blocking, or something outside the
            # declared scaffolding is. Escalating is correct — but say what it
            # actually was instead of calling it a content conflict.
            raise ReworkBranchConflictError(
                worktree_path=worktree_path,
                branch=branch,
                base_ref=merge_base_ref,
                conflicted_paths=undeclared,
                stderr=merge_result.stderr or merge_result.error,
                stage="pre_merge",
            )

    # Read the unmerged paths while the merge is still in progress, then abort.
    diff_result = run_captured(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    conflicted_paths: tuple[str, ...] = ()
    if diff_result.ok:
        conflicted_paths = tuple(p.strip() for p in diff_result.stdout.splitlines() if p.strip())

    abort_result = run_captured(
        ["git", "merge", "--abort"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not abort_result.ok or _merge_head_present(worktree_path):
        # Case (b): a merge genuinely was in progress and did not come back.
        # The worktree may still be mid-merge, so escalate rather than handing a
        # worker a broken workspace. Prefer the merge's own stderr — the abort's
        # is about the abort, and the merge's is what explains the failure.
        raise ReworkBranchConflictError(
            worktree_path=worktree_path,
            branch=branch,
            base_ref=merge_base_ref,
            conflicted_paths=conflicted_paths,
            stderr=merge_result.stderr or abort_result.stderr,
        )

    base_sha_result = run_captured(
        ["git", "rev-parse", merge_base_ref],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    base_sha = base_sha_result.stdout.strip() if base_sha_result.ok else ""
    base_branch = (
        merge_base_ref[len("origin/") :]
        if merge_base_ref.startswith("origin/")
        else merge_base_ref
    )
    return ReworkMergeConflict(
        base_branch=base_branch,
        base_sha=base_sha,
        conflicted_files=conflicted_paths,
    )


REWORK_CONFLICT_NOTICE_BEGIN = "<!-- charlie:rework-conflict-notice:begin -->"
REWORK_CONFLICT_NOTICE_END = "<!-- charlie:rework-conflict-notice:end -->"
_REWORK_CONFLICT_NOTICE_RE = re.compile(
    re.escape(REWORK_CONFLICT_NOTICE_BEGIN) + r".*?" + re.escape(REWORK_CONFLICT_NOTICE_END),
    re.DOTALL,
)


def render_rework_conflict_notice(conflict: ReworkMergeConflict) -> str:
    """Render the prompt section appended when a rework worktree carries a
    ``ReworkMergeConflict`` (see ``WorktreeInfo.rework_conflict``).

    Single point of enforcement for this wording: every adapter launch site
    (claude-code, api, devin-shell) appends this same text to its per-session
    prompt file rather than composing its own phrasing. The sentinel pair
    bounding the block exists so ``apply_rework_conflict_notice`` can excise
    a previous attempt's notice from a prompt file that is mutated in place
    across redispatches (devin_shell).
    """
    files = (
        "\n".join(f"- {path}" for path in conflict.conflicted_files)
        if conflict.conflicted_files
        else "- (conflicted paths unavailable)"
    )
    short_sha = conflict.base_sha[:12] if conflict.base_sha else "unknown"
    return (
        "\n\n"
        f"{REWORK_CONFLICT_NOTICE_BEGIN}\n"
        "\n---\n"
        "## ACTION REQUIRED: unresolved merge conflict with base branch\n\n"
        f"This branch has unresolved merge conflicts with "
        f"`{conflict.base_branch}`@`{short_sha}` in the following file(s):\n\n"
        f"{files}\n\n"
        f"FIRST: merge `origin/{conflict.base_branch}` into this branch and "
        "resolve the conflicts faithfully, preserving both this PR's intent "
        "and the base branch's changes. Push only after the conflicts are "
        "fully resolved.\n\n"
        "THEN: address the review feedback below.\n"
        f"{REWORK_CONFLICT_NOTICE_END}\n"
    )


def apply_rework_conflict_notice(prompt_text: str, conflict: ReworkMergeConflict) -> str:
    """Return ``prompt_text`` carrying exactly one, current conflict notice.

    Idempotent by construction: any notice block from a previous dispatch
    attempt (bounded by the sentinel pair) is removed before the fresh one is
    appended. Without this, a redispatch over a caller-supplied prompt file
    that was mutated in place (devin_shell -- the file is never regenerated
    per attempt) stacks one notice per attempt, and the base branch can move
    between attempts, so the stacked notices carry contradictory base SHAs.
    """
    stripped = _REWORK_CONFLICT_NOTICE_RE.sub("", prompt_text)
    # A begin marker without its end (torn write from a crashed attempt) would
    # otherwise survive the regex and leave garbage above the fresh notice.
    dangling = stripped.find(REWORK_CONFLICT_NOTICE_BEGIN)
    if dangling != -1:
        stripped = stripped[:dangling]
    return stripped.rstrip() + render_rework_conflict_notice(conflict)


def _rework_patch_id(repo_root: Path, base_ref: str, head_ref: str) -> str:
    """Return the patch-id of the changes ``head_ref`` introduces over ``base_ref``.

    Uses the three-dot diff ``git diff base_ref...head_ref`` so only changes
    unique to ``head_ref`` (relative to the merge-base) are hashed. Empty or
    failed diffs return an empty string.
    """
    result = run_captured(
        ["git", "diff", f"{base_ref}...{head_ref}"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not result.ok:
        return ""
    return _calculate_patch_id(result.stdout)


def _rework_reset_reason(repo_root: Path, branch: str, base_ref: str) -> str:
    """Decide why a non-FF rework reset is safe (or loud) before trusting origin.

    Compares the patch-id of local ``branch`` with ``origin/{branch}`` relative
    to ``base_ref``. Identical patch-ids mean the local-only commits carry the
    same content as the origin tip (e.g. a rebase-only rewrite), so nothing real
    is lost on reset. Different patch-ids mean the reset discards genuinely
    different local work and must be reported loudly.
    """
    local_patch_id = _rework_patch_id(repo_root, base_ref, branch)
    origin_patch_id = _rework_patch_id(repo_root, base_ref, f"origin/{branch}")
    if local_patch_id and origin_patch_id and local_patch_id == origin_patch_id:
        return "reset-origin:identical-patch-id"
    if local_patch_id or origin_patch_id:
        return "reset-origin:different-patch-id"
    return "reset-origin:indeterminate-patch-id"


def _parse_status_v2_paths(stdout: str) -> list[str]:
    """Parse ``git status --porcelain=v2 -z`` output into a flat list of
    worktree-relative paths representing real changes: ordinary tracked
    changes, renames/copies (current path only), unmerged conflicts, and
    untracked files. Ignored-file records (``!``, never emitted unless
    ``--ignored`` is passed) and any unrecognized record type are skipped
    defensively rather than mis-parsed.

    ``-z`` NUL-terminates every record AND every path field, which is what
    makes this unambiguous: paths can never contain NUL, so ``-z`` never
    quotes or escapes them, and there is no line-based whitespace to
    ``.strip()`` (the v1 fixed-column bug: stripping a leading status-column
    space before slicing shifted the path left by one character and dropped
    its first character, e.g. on tracked-but-unstaged-modified files whose
    path begins with a dot). Each v2 record type has a fixed, documented
    number of space-separated header fields before the path (see
    ``git-status(1)`` "Porcelain Format Version 2"); a rename/copy record
    additionally consumes one extra NUL-delimited field for ``origPath``,
    which is discarded here since dirty-checking only cares about the
    current path.
    """
    paths: list[str] = []
    fields = stdout.split("\0")
    i = 0
    n = len(fields)
    while i < n:
        record = fields[i]
        if not record:
            i += 1
            continue
        tag = record[0]
        if tag == "1":
            # Ordinary changed entry: "1 <XY> <sub> <mH> <mI> <mW> <hH> <hI> <path>"
            parts = record.split(" ", 8)
            if len(parts) == 9:
                paths.append(parts[8])
            i += 1
        elif tag == "2":
            # Renamed/copied entry: "2 <XY> <sub> <mH> <mI> <mW> <hH> <hI> <X><score> <path>"
            # followed by a separate NUL-terminated <origPath> field.
            parts = record.split(" ", 9)
            if len(parts) == 10:
                paths.append(parts[9])
            i += 2  # this record's path field + the trailing origPath field
        elif tag == "u":
            # Unmerged entry: "u <XY> <sub> <m1> <m2> <m3> <mW> <h1> <h2> <h3> <path>"
            parts = record.split(" ", 10)
            if len(parts) == 11:
                paths.append(parts[10])
            i += 1
        elif tag == "?":
            # Untracked entry: "? <path>"
            paths.append(record[2:])
            i += 1
        else:
            # "!" ignored entries (not requested here) or any unrecognized
            # record type: skip rather than guess at a layout we don't know.
            i += 1
    return paths


def _declared_scaffolding_matcher(
    injected_paths: tuple[str, ...] = (),
    materialize_dirs: tuple[str, ...] = (),
) -> Callable[[str], bool]:
    """Build a predicate matching worktree-relative paths the orchestrator
    itself declares it writes (``injected_paths`` + ``materialize_dirs``).

    Extracted so the dirty-check and the pre-merge collision cleanup share one
    definition of "orchestrator scaffolding, not worker product". Two copies of
    this rule that drift apart would let the cleanup delete something the dirty
    check considers worker-authored — the one outcome that must never happen.
    """
    # Normalize the configured side too, so a Windows-style backslash override
    # still matches git's forward-slash path reporting.
    excluded = [
        PurePosixPath(str(p).replace("\\", "/")) for p in (*injected_paths, *materialize_dirs)
    ]

    def _is_declared(raw_path: str) -> bool:
        # Git may emit backslashes on Windows; normalize for comparison.
        path = PurePosixPath(str(raw_path).replace("\\", "/"))
        return any(
            path == excluded_path or excluded_path in path.parents for excluded_path in excluded
        )

    return _is_declared


def _worktree_git_runner(worktree_path: Path) -> git_pull_blockers.GitRunner:
    """Bind :mod:`git_pull_blockers`' runner seam to this worktree."""

    def run_git(argv: list[str]) -> RunResult:
        return run_captured(argv, cwd=worktree_path, timeout_seconds=_DEFAULT_TIMEOUT_SECONDS)

    return run_git


def _untracked_paths_shadowing_ref(worktree_path: Path, ref: str) -> tuple[str, ...]:
    """Worktree-relative untracked paths that ``ref`` also tracks.

    Refusal class (a) — see :mod:`charlie_work.git_pull_blockers`, which owns
    the implementation so the orchestrator's own self-deploy asks the same
    question the same way.
    """
    return git_pull_blockers.untracked_paths_shadowing_ref(
        _worktree_git_runner(worktree_path), ref
    )


def _modified_paths_overwritten_by_ref(worktree_path: Path, ref: str) -> tuple[str, ...]:
    """Worktree-relative tracked paths that are locally modified *and* that
    merging ``ref`` would change.

    Refusal class (b) — the other half of the same refusal
    ``_untracked_paths_shadowing_ref`` covers, and the reason handling only the
    untracked half left the rest escalating. Implementation and the full
    rationale (merge-base basis, ``--diff-filter=M``) live in
    :mod:`charlie_work.git_pull_blockers`.

    Live example this exists for: the devin shim rewrites ``.devin/prompts/*``
    in the worktree, and job-cannon's ``15dacbb6`` *deleted* those paths from
    the base. Merging then wants to remove a locally modified file, which git
    refuses. Every branch forked before that commit hits it.
    """
    return git_pull_blockers.modified_paths_overwritten_by_ref(
        _worktree_git_runner(worktree_path), ref
    )


def _clear_declared_scaffolding_collisions(
    worktree_path: Path,
    blocking_paths: tuple[str, ...],
    injected_paths: tuple[str, ...],
    materialize_dirs: tuple[str, ...],
) -> bool:
    """Remove orchestrator scaffolding that is blocking a pre-merge.

    Returns True only if every blocking path was declared scaffolding AND the
    removal succeeded. If anything outside the declared set is blocking, this
    refuses outright and removes nothing — a partial cleanup that still fails
    the merge would destroy files for no benefit.

    Safe by three independent properties:

    1. Only paths matching ``_declared_scaffolding_matcher`` are eligible —
       the same predicate ``_worker_authored_dirty`` uses to decide a path is
       not worker product. Everything removed here is re-materialized
       unconditionally later in ``create_worktree``.
    2. Removal goes through ``git clean``, which structurally cannot remove a
       tracked file or a modified tracked file. A filesystem ``rmtree`` could.
    3. Each path is containment-checked against the worktree on its *resolved*
       form, and anything at or under ``.venv`` is refused regardless — that
       link is a junction into the SHARED virtualenv on this host, so
       following it would corrupt every worktree at once.
    """
    if not _eligible_for_scaffolding_repair(
        worktree_path, blocking_paths, injected_paths, materialize_dirs
    ):
        return False
    result = run_captured(
        ["git", "clean", "-f", "-d", "--", *blocking_paths],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    return result.ok


def _eligible_for_scaffolding_repair(
    worktree_path: Path,
    blocking_paths: tuple[str, ...],
    injected_paths: tuple[str, ...],
    materialize_dirs: tuple[str, ...],
) -> bool:
    """The safety gate every scaffolding repair must pass before touching disk.

    Shared by the untracked-removal and modified-restore paths for the same
    reason ``_declared_scaffolding_matcher`` is shared with the dirty check: two
    copies of a rule that decides what may be destroyed are two chances for them
    to drift into disagreeing, and only one of those outcomes is recoverable.
    """
    if not blocking_paths:
        return False
    is_declared = _declared_scaffolding_matcher(injected_paths, materialize_dirs)
    if not all(is_declared(path) for path in blocking_paths):
        return False
    for path in blocking_paths:
        if PurePosixPath(path).parts[:1] == (".venv",):
            return False
        if not contains(worktree_path, worktree_path / path):
            return False
    return True


def _restore_declared_scaffolding_modifications(
    worktree_path: Path,
    blocking_paths: tuple[str, ...],
    injected_paths: tuple[str, ...],
    materialize_dirs: tuple[str, ...],
) -> bool:
    """Discard local modifications to orchestrator scaffolding blocking a merge.

    The tracked-file counterpart of ``_clear_declared_scaffolding_collisions``,
    gated by the identical eligibility check. Discarding these edits loses
    nothing: every path here is one the orchestrator itself wrote and
    re-materializes unconditionally later in ``create_worktree``, which is the
    same premise that lets ``_worker_authored_dirty`` ignore them when deciding
    whether a worktree holds real work.

    ``git checkout HEAD --`` rather than ``git checkout --``: the latter
    restores from the index, so a *staged* scaffolding edit would survive and
    the retried merge would be blocked by the same path a second time.
    """
    if not _eligible_for_scaffolding_repair(
        worktree_path, blocking_paths, injected_paths, materialize_dirs
    ):
        return False
    result = run_captured(
        ["git", "checkout", "HEAD", "--", *blocking_paths],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    return result.ok


def _repair_declared_scaffolding_blockers(
    worktree_path: Path,
    untracked_blocking: tuple[str, ...],
    modified_blocking: tuple[str, ...],
    injected_paths: tuple[str, ...],
    materialize_dirs: tuple[str, ...],
) -> bool:
    """Repair both classes of scaffolding blocker; True only if all of them were.

    Each class is skipped when empty rather than treated as a failure — a
    worktree blocked by only one class is the common case, and requiring both to
    be non-empty would refuse to repair either.

    A False return does not promise nothing was written: if the removal succeeds
    and the restore then fails, the removed files stay removed. That is
    deliberate and harmless — every path eligible here is re-materialized
    unconditionally later in ``create_worktree`` — whereas rolling a partial
    repair back would mean re-creating files from content this function never
    had.
    """
    if not untracked_blocking and not modified_blocking:
        return False
    if untracked_blocking and not _clear_declared_scaffolding_collisions(
        worktree_path, untracked_blocking, injected_paths, materialize_dirs
    ):
        return False
    if modified_blocking and not _restore_declared_scaffolding_modifications(
        worktree_path, modified_blocking, injected_paths, materialize_dirs
    ):
        return False
    return True


def _worker_authored_dirty(
    worktree_path: Path,
    injected_paths: tuple[str, ...] = (),
    materialize_dirs: tuple[str, ...] = (),
) -> bool:
    """Return True if the worktree has uncommitted changes that are NOT
    orchestrator scaffolding.

    ``injected_paths`` and ``materialize_dirs`` are worktree-relative paths
    (files or directories) that the orchestrator writes into the worktree
    (e.g. the Claude Code prompt file, per-dispatch prompt templates under
    ``.devin/prompts/...``, or a custom convention the operator has
    configured). They are excluded from the dirty check so completed worker
    work is not stranded by scaffolding noise (issue #381, issue #471).

    Uses ``git status --porcelain=v2 -z --untracked-files=all``:
    ``--untracked-files=all`` disables directory collapsing, so every
    untracked file gets its own record — there is no "does this line
    represent one file or a whole directory" ambiguity left to special-case
    (the v1 rework history: a wholly-untracked directory collapsed to a
    single ``?? dir/`` line could hide a worker-authored sibling file next to
    a configured injected path or directory, and a probe re-scoped to that
    directory was needed to tell the two apart). Every entry — whether it
    matches an excluded file or a whole excluded directory — is now checked
    against the same normalized-path predicate below, with no separate
    collapse-probing code path to keep in sync.

    Raises:
        WorktreeProbeFailedError: if the ``git status`` probe itself fails
            (index lock, corruption, permissions).
    """
    status_result = run_captured(
        [
            "git",
            "-c",
            "core.quotePath=off",
            "status",
            "--porcelain=v2",
            "-z",
            "--untracked-files=all",
        ],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not status_result.ok:
        detail = status_result.error or status_result.stderr.strip() or "unknown error"
        raise WorktreeProbeFailedError(
            f"worktree status probe failed; treating as dirty: {detail}"
        )

    is_declared = _declared_scaffolding_matcher(injected_paths, materialize_dirs)
    for raw_path in _parse_status_v2_paths(status_result.stdout):
        if is_declared(raw_path):
            continue
        return True
    return False


def _capture_worktree_work_to_rescue_ref(
    repo_root: Path,
    worktree_path: Path,
    issue_number: int | None,
    injected_paths: tuple[str, ...] = (),
    materialize_dirs: tuple[str, ...] = (),
) -> RescueCapture:
    """Capture worker-authored dirty content to a rescue ref (issue #849).

    Stages all worktree changes — tracked modifications, deletions, and
    untracked files — except orchestrator scaffolding (``injected_paths``,
    ``materialize_dirs``, and ``.venv``), creates a tree from the resulting
    index, wraps it in a commit with the worktree's current ``HEAD`` as
    parent, and saves it to ``refs/charlie/rescue/issue-<n>-<timestamp>``.

    The index is restored to ``HEAD`` after the tree is captured, so the
    worktree's staging state is unchanged regardless of capture outcome.

    Never raises: every git invocation goes through ``run_captured`` (errors
    as values). A capture failure returns a ``RescueCapture`` with ``error``
    set — the caller must refuse the reset in that case, exactly as today.
    """
    # Build exclusion pathspecs. ``.venv`` is always excluded: it is either a
    # junction into the shared virtualenv (following it would add every other
    # worktree's venv contents) or a local venv that is not worker content.
    exclusions: list[str] = [":(exclude).venv"]
    for p in (*injected_paths, *materialize_dirs):
        normalized = str(p).replace("\\", "/").strip("/")
        if normalized:
            exclusions.append(f":(exclude){normalized}")

    add_result = run_captured(
        ["git", "add", "-A", "--", ".", *exclusions],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )

    tree_sha: str | None = None
    if add_result.ok:
        tree_result = run_captured(
            ["git", "write-tree"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if tree_result.ok and tree_result.stdout.strip():
            tree_sha = tree_result.stdout.strip()

    # Always restore the index to HEAD so the worktree's staging state is
    # unchanged.  This runs whether or not the tree was captured: a failed
    # ``git add`` may have left a partially-staged index, and ``git reset
    # --mixed HEAD`` unstages everything without touching the working tree.
    run_captured(
        ["git", "reset", "--mixed", "HEAD"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )

    if tree_sha is None:
        detail = add_result.error or add_result.stderr or "unknown error"
        return RescueCapture(
            ref_name=None,
            commit_sha=None,
            error=f"capture failed at add/write-tree stage: {detail}",
        )

    head_result = run_captured(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not head_result.ok or not head_result.stdout.strip():
        return RescueCapture(
            ref_name=None,
            commit_sha=None,
            error=f"capture failed: cannot resolve HEAD: "
            f"{head_result.error or head_result.stderr}",
        )
    head_sha = head_result.stdout.strip()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    issue_part = f"issue-{issue_number}" if issue_number is not None else "issue-unknown"
    ref_name = f"{RESCUE_REF_PREFIX}/{issue_part}-{timestamp}"

    commit_result = run_captured(
        ["git", "commit-tree", tree_sha, "-p", head_sha, "-m", f"rescue: {issue_part}"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not commit_result.ok or not commit_result.stdout.strip():
        return RescueCapture(
            ref_name=None,
            commit_sha=None,
            error=f"capture failed at commit-tree: {commit_result.error or commit_result.stderr}",
        )
    commit_sha = commit_result.stdout.strip()

    update_result = run_captured(
        ["git", "update-ref", ref_name, commit_sha],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not update_result.ok:
        return RescueCapture(
            ref_name=None,
            commit_sha=commit_sha,
            error=f"capture failed at update-ref: {update_result.error or update_result.stderr}",
        )

    return RescueCapture(ref_name=ref_name, commit_sha=commit_sha)


def _is_confirmed_missing_ref(result: RunResult) -> bool:
    """True only when ``git rev-parse --verify -q <ref>`` ran to completion and
    definitively reported that ``<ref>`` does not resolve to a single
    revision (unborn ``HEAD`` in an empty repo, or a branch/tag/sha that does
    not exist) -- the one non-``ok`` outcome where "nothing to lose" is a
    sound conclusion.

    This is an allow-list on git's exit code, not a deny-list on failure
    reasons: with ``-q``/``--quiet``, git reserves exit code 1 exclusively
    for "the given ref does not resolve" and suppresses the fatal message
    entirely (confirmed against git 2.45 for an empty repo's unborn ``HEAD``
    and for a missing branch name -- both produce ``returncode=1`` with empty
    stdout/stderr). Any other non-zero outcome -- ``returncode=128`` (not a
    git repository, corrupted refs, permissions error), a git binary missing
    from PATH entirely (``RunResult.error`` set, ``returncode is None``), or
    the probe timing out -- fails this check and is therefore treated as a
    probe failure by the caller, not as a confirmed-absent ref.

    Exit code, not a stderr string match, is the discriminator on purpose:
    git's fatal messages are locale-translatable, so matching on message text
    would silently stop working (fail closed forever, not loudly) on a host
    with a non-English git locale. The exit-code contract for ``--verify -q``
    is part of git's documented plumbing behavior and does not vary with
    locale. This is the safe default either way: we only ever fail OPEN
    (report "nothing to lose") on a positive match, never on the absence of
    one.
    """
    return not result.timed_out and result.returncode == 1


def _worktree_refuse_to_reset_reason(
    repo_root: Path,
    branch: str,
    base_ref: str,
    worktree_path: Path | None = None,
    injected_paths: tuple[str, ...] = (),
    materialize_dirs: tuple[str, ...] = (),
) -> str | None:
    """Return a human-readable reason if resetting the branch/worktree would destroy
    local work, otherwise ``None``.

    Checks for:
      - uncommitted modifications in ``worktree_path`` (if it exists)
      - local commits that are not present on the remote branch

    This is read-only: it never commits, fetches, or resets. It resolves ``base_ref``
    locally and skips the ``git ls-remote`` probe entirely when the local tip is
    already at or behind the base. The remote probe is only consulted when the
    local branch has commits beyond ``base_ref``.

    Raises:
        WorktreeProbeFailedError: if a probe (``git status --porcelain``,
            ``git rev-parse --verify`` on the local tip, or ``git ls-remote``)
            itself fails. The reset is still refused (we cannot confirm the
            worktree is clean), but this is classified distinctly from a
            confirmed-dirty worktree — see the class docstring for why callers
            must not conflate the two under the same ``failure_kind``. The
            local-tip ``rev-parse --verify`` only counts as "nothing to lose"
            (returns ``None``) when it confirms the ref is genuinely absent —
            see ``_is_confirmed_missing_ref``; any other failure (index lock,
            AV-held handle, timeout, missing git binary) refuses instead.
    """
    # Uncommitted modifications are only meaningful when the worktree directory exists.
    if worktree_path is not None and worktree_path.is_dir():
        if _worker_authored_dirty(worktree_path, injected_paths, materialize_dirs):
            return "worktree has uncommitted modifications"
        local_tip_result = run_captured(
            ["git", "rev-parse", "--verify", "-q", "HEAD"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
    else:
        local_tip_result = run_captured(
            ["git", "rev-parse", "--verify", "-q", branch],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )

    if not local_tip_result.ok:
        if _is_confirmed_missing_ref(local_tip_result):
            # Confirmed: no branch or no commit exists to lose.
            return None
        detail = local_tip_result.error or local_tip_result.stderr.strip() or "unknown error"
        raise WorktreeProbeFailedError(
            f"git rev-parse --verify local-tip probe failed (not a confirmed-"
            f"missing ref): {detail}"
        )

    if not local_tip_result.stdout.strip():
        # `rev-parse --verify` succeeded (exit 0) but produced no sha. Not
        # reachable in practice -- a successful verify always prints the
        # resolved revision -- but if it ever happens, git did not error at
        # all, so there is no ambiguity to fail closed on: no sha means
        # nothing to compare against, so nothing to lose.
        return None

    local_sha = local_tip_result.stdout.strip()

    # Resolve the dispatch base ref locally. If the local tip is the base commit
    # or an ancestor of it, the worktree has no commits beyond the base and is
    # safe to reclaim without a network round-trip.
    base = base_ref if base_ref else _resolve_default_branch_ref(repo_root)
    base_sha_result = run_captured(
        ["git", "rev-parse", "--verify", base],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if base_sha_result.ok and base_sha_result.stdout.strip():
        base_sha = base_sha_result.stdout.strip()
        if local_sha == base_sha:
            return None
        ancestor_result = run_captured(
            ["git", "merge-base", "--is-ancestor", local_sha, base_sha],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if ancestor_result.ok:
            return None

    # The local tip has commits beyond the base (or the base is unresolvable),
    # so we must ask the remote whether those commits are already pushed.
    remote_sha: str | None = None
    if _has_origin_remote(repo_root):
        ls_remote_result = _run_remote_captured(
            ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
            cwd=repo_root,
        )
        if ls_remote_result.timed_out:
            raise WorktreeProbeFailedError(
                f"git ls-remote origin refs/heads/{branch} timed out after "
                f"{_REMOTE_TIMEOUT_SECONDS}s"
            )
        if not ls_remote_result.ok:
            raise WorktreeProbeFailedError(
                f"git ls-remote origin refs/heads/{branch} failed: "
                f"{ls_remote_result.error or ls_remote_result.stderr}"
            )
        if ls_remote_result.stdout.strip():
            remote_sha = ls_remote_result.stdout.strip().split()[0]

    if remote_sha is None:
        # Branch does not exist on origin (or the probe failed). Any commits beyond
        # the base ref are unpushed and must not be discarded.
        merge_base_result = run_captured(
            ["git", "merge-base", base, local_sha],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not merge_base_result.ok:
            return "worktree has local commits not on remote branch"
        merge_base = merge_base_result.stdout.strip()
        rev_list_result = run_captured(
            ["git", "rev-list", "--count", f"{merge_base}..{local_sha}"],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if (
            rev_list_result.ok
            and rev_list_result.stdout.strip().isdigit()
            and int(rev_list_result.stdout.strip()) > 0
        ):
            return f"worktree has {rev_list_result.stdout.strip()} local commit(s) not on remote branch"
        return None

    # Remote branch exists. If local tip matches the remote tip, there are no
    # local-only commits.
    if local_sha == remote_sha:
        return None

    # local_sha != remote_sha. If local_sha is an ancestor of remote_sha, the local
    # branch is behind the remote and has no local commits not on remote. Otherwise
    # local is ahead or has diverged, which means local commits not on remote.
    ancestor_result = run_captured(
        ["git", "merge-base", "--is-ancestor", local_sha, remote_sha],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if ancestor_result.ok:
        return None

    return "worktree has local commits not on remote branch"


def _worktree_dirty_reason(
    worktree_path: Path,
    injected_paths: tuple[str, ...] = (),
    materialize_dirs: tuple[str, ...] = (),
) -> str | None:
    """Return a reason string if ``worktree_path`` has uncommitted modifications.

    Used by the merged-PR cleanup path (``clean_worktrees``), which decides
    eligibility by comparing the worktree's local HEAD against the merged PR's
    ``headRefOid`` rather than by comparing against the (possibly already
    deleted) remote branch — see ``clean_worktrees`` for why
    ``_worktree_refuse_to_reset_reason``'s remote-branch comparison does not
    apply once a PR has been squash-merged with its branch deleted.

    Raises:
        WorktreeProbeFailedError: if the ``git status --porcelain`` probe itself
            fails (index lock, corruption, permissions) — mirrors
            ``_worktree_refuse_to_reset_reason``'s handling of the same failure
            mode.
    """
    if not worktree_path.is_dir():
        return None
    if _worker_authored_dirty(worktree_path, injected_paths, materialize_dirs):
        return "worktree has uncommitted modifications"
    return None


def _is_pristine_orchestrator_worktree(
    repo_root: Path,
    worktree_path: Path,
    base_ref: str,
) -> bool:
    """Return True after bringing an existing worktree to the current dispatch base.

    A worktree created by the orchestrator and never handed to a worker is
    pristine when it has no worker-authored uncommitted changes and its HEAD
    is at the dispatch base. Remote-tracking base refs are fetched first so
    the comparison uses the live origin tip rather than a stale local
    remote-tracking ref. If the worktree is behind the fetched base it is
    reset to that tip in-place and still considered reclaimable; if it is
    ahead or diverged it is not reclaimable without a remote probe.

    Such a worktree can be reclaimed directly without the remote
    ``git ls-remote`` probe.
    """
    if not worktree_path.is_dir():
        return False
    # A truly pristine worktree has no uncommitted changes at all, including
    # orchestrator-injected prompt files. Injected-only dirty worktrees are
    # reclaimed by pruning and recreating, not by reuse (issue #381), while
    # still avoiding the remote git probe because the subsequent reset guard
    # runs the same local checks.
    status_result = run_captured(
        ["git", "status", "--porcelain"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not status_result.ok or status_result.stdout.strip():
        return False

    head_result = run_captured(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not head_result.ok or not head_result.stdout.strip():
        return False
    head_sha = head_result.stdout.strip()

    # For remote-tracking base refs, fetch first so we compare against the
    # live origin tip rather than a stale local remote-tracking ref.
    if base_ref.startswith("origin/") and _has_origin_remote(repo_root):
        remote_branch = base_ref[len("origin/") :]
        fetch_result = _run_remote_captured(
            ["git", "fetch", "origin", remote_branch],
            cwd=repo_root,
        )
        if not fetch_result.ok:
            # Cannot verify base freshness; refuse to reuse a potentially stale worktree.
            return False

    base_sha_result = run_captured(
        ["git", "rev-parse", "--verify", base_ref],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not base_sha_result.ok or not base_sha_result.stdout.strip():
        return False
    base_sha = base_sha_result.stdout.strip()

    if head_sha == base_sha:
        return True

    # If the worktree is behind the fetched base, reset it in-place.
    ancestor_result = run_captured(
        ["git", "merge-base", "--is-ancestor", head_sha, base_sha],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if ancestor_result.ok:
        reset_result = run_captured(
            ["git", "reset", "--hard", base_sha],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        return reset_result.ok

    return False


def _salvage_worktree(repo_root: Path, worktree_path: Path, branch: str) -> str | None:
    """Salvage a worktree with uncommitted changes or unpushed commits.

    Commits the current state to a salvage ref, pushes it to origin, and returns
    the salvage ref name. Returns None if the worktree is clean (nothing to salvage).
    Raises RuntimeError if the salvage push fails.
    """
    from .state import utc_now

    # Check if there's anything to salvage
    dirty_result = run_captured(
        ["git", "status", "--porcelain"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    # If the probe fails (index lock, corruption, permissions), treat as dirty to be safe
    has_dirty = not dirty_result.ok or bool(dirty_result.stdout.strip())

    # Check for unpushed commits (only if branch exists on origin)
    has_unpushed = False
    remote_exists = None
    if _has_origin_remote(repo_root):
        remote_exists = _remote_branch_exists(repo_root, branch)
    if remote_exists is True:
        unpushed_result = run_captured(
            ["git", "log", f"origin/{branch}..HEAD", "--oneline"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        has_unpushed = unpushed_result.ok and bool(unpushed_result.stdout.strip())
    else:
        # Branch doesn't exist on origin: any commits are considered "unpushed"
        # (killed before first push scenario)
        # Check if HEAD has any commits beyond the default branch tip
        default_ref = _resolve_default_branch_ref(repo_root)
        if default_ref == "HEAD":
            # No origin remote: there is no authoritative default-branch tip to compare against.
            # Conservatively assume there are unpushed commits to preserve work.
            # This replaces the previous accidental behavior (failed merge-base → same default)
            # with an intentional, documented trade-off.
            has_unpushed = True
        else:
            # Origin exists: compare against the remote-tracking tip directly.
            # Using the remote-tracking ref (e.g., "origin/master") is the correct salvage
            # semantic — we want to know if there are commits beyond what is actually on
            # the shared default branch. No local-name conversion is needed.
            merge_base_result = run_captured(
                ["git", "merge-base", "HEAD", default_ref],
                cwd=worktree_path,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            )
            if merge_base_result.ok:
                merge_base = merge_base_result.stdout.strip()
                rev_list_result = run_captured(
                    ["git", "rev-list", "--count", f"{merge_base}..HEAD"],
                    cwd=worktree_path,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                has_unpushed = rev_list_result.ok and int(rev_list_result.stdout.strip()) > 0
            else:
                # If merge-base fails, assume has commits to be safe
                has_unpushed = True

    if not has_dirty and not has_unpushed:
        return None

    # Create salvage ref name
    timestamp = utc_now().replace(":", "-").replace("+00:00", "Z")
    salvage_ref = f"salvage/{branch.replace('/', '-')}-{timestamp}"

    # Commit dirty changes if any
    if has_dirty:
        run_captured(
            ["git", "add", "-A"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        run_captured(
            ["git", "commit", "-m", f"Salvage before worktree cleanup: {timestamp}"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )

    # Create the salvage ref
    run_captured(
        ["git", "update-ref", f"refs/{salvage_ref}", "HEAD"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )

    # Push the salvage ref if origin exists
    if _has_origin_remote(repo_root):
        push_result = run_captured(
            ["git", "push", "origin", f"refs/{salvage_ref}:refs/{salvage_ref}"],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not push_result.ok:
            raise RuntimeError(
                f"Failed to push salvage ref {salvage_ref!r} to origin: "
                f"{push_result.error or push_result.stderr}"
            )

    return salvage_ref


def is_junction(path: Path) -> bool:
    """Return True if ``path`` is a reparse point (Windows junction/symlink)
    or, on non-Windows platforms, a symlink. ``os.path.islink()`` alone is
    unreliable for junctions on some Windows Python builds, so the Windows
    path checks the reparse-point file attribute directly."""
    if os.name == "nt":
        try:
            result = os.stat(path, follow_symlinks=False)
        except OSError:
            return False
        return bool(result.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    return os.path.islink(path)


def is_live_foreign_worktree(entry: Path, repo_root: Path) -> bool:
    """Return True if ``entry`` is a live git worktree administered by a
    repository other than ``repo_root``.

    Worker launch shims provision sibling checkouts of *other* repos inside
    this repo's worktrees dir (e.g. the ``ci_runners`` worktree the ci-fleet
    editable resolves against). Such a directory is never in this repo's
    ``git worktree list``, so the orphan sweep would classify it as residue
    and delete it out from under running workers (2026-08-09 incident: the
    08:15:43Z reclaim pass removed the sibling minutes after provisioning).

    A linked worktree's ``.git`` is a *file* containing ``gitdir: <admin>``.
    The directory is a live foreign worktree when that admin dir exists and
    is not under this repo's own ``.git``. A dangling gitdir (admin dir gone,
    e.g. after ``git worktree remove`` failed to delete the tree) is residue
    and stays sweepable. Unreadable/unresolvable ``.git`` fails closed
    (treated as foreign): deleting what we cannot classify risks destroying
    live state, while skipping it merely defers cleanup."""
    gitfile = entry / ".git"
    try:
        if not gitfile.is_file():
            return False
        content = gitfile.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    for line in content.splitlines():
        if line.startswith("gitdir:"):
            admin = Path(line.split(":", 1)[1].strip())
            break
    else:
        return True
    if not admin.is_absolute():
        admin = entry / admin
    try:
        admin = admin.resolve()
        if not admin.is_dir():
            return False
        # ``repo_root/.git`` is a directory for a main checkout; resolve()
        # canonicalizes both sides so the containment test defeats junctions
        # and 8.3 short names, same as the ci_fleet containment checks.
        own_git = (repo_root / ".git").resolve()
    except OSError:
        return True
    return not (admin == own_git or admin.is_relative_to(own_git))


def _unlink_reparse_point(path: Path) -> None:
    """Remove a reparse point (Windows junction/symlink) or POSIX symlink.

    On Windows, ``os.unlink`` removes a directory symlink (a name-surrogate
    reparse point) without following into the target.  Some Windows/Python
    combinations reject ``os.unlink`` on a junction with ``EACCES`` or
    ``EISDIR``; ``os.rmdir`` is the fallback because it removes the reparse
    point itself, never the target directory it points at.  On POSIX,
    ``os.unlink`` removes the symlink.  In all cases the target is left
    untouched.
    """
    if os.name == "nt":
        try:
            os.unlink(path)
        except (IsADirectoryError, PermissionError, OSError):
            # Older Windows builds or junctions may reject unlink; rmdir on a
            # reparse point removes the link, not the target.
            os.rmdir(path)
    else:
        os.unlink(path)


def _unlink_worktree_reparse_points(worktree_path: Path) -> None:
    """Unlink all reparse points inside ``worktree_path`` without following them.

    ``os.walk`` with ``followlinks=False`` does not descend into symlinks, but
    it may still list directory-symlink and junction names in ``dirnames``.
    We prune those entries after unlinking them so the walk never treats a
    reparse point as a real directory to recurse into.  Regular files and
    directories are left untouched.
    """
    if not worktree_path.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(
        str(worktree_path), topdown=True, onerror=lambda exc: None
    ):
        parent = Path(dirpath)
        for name in dirnames + filenames:
            child = parent / name
            try:
                if child.is_symlink() or is_junction(child):
                    _unlink_reparse_point(child)
            except OSError:
                # Best-effort: leave the path for the rmtree fallback to report.
                pass
        # Do not descend into directories that are actually reparse points.
        remaining: list[str] = []
        for d in dirnames:
            dpath = parent / d
            try:
                if dpath.is_symlink() or is_junction(dpath):
                    continue
            except OSError:
                pass
            remaining.append(d)
        dirnames[:] = remaining


def _robust_rmtree(path: Path) -> bool:
    """Remove a directory tree, never following reparse points into targets.

    Unlinks all junctions/symlinks first, then deletes the remaining files and
    directories with ``shutil.rmtree``.  Returns True when the path no longer
    exists.
    """
    if not path.exists() and not is_junction(path):
        return True
    _unlink_worktree_reparse_points(path)
    try:
        shutil.rmtree(path)
    except OSError:
        return False
    return not path.exists() and not is_junction(path)


def _create_junction_or_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or is_junction(link_path):
        raise RuntimeError(f"venv link target already exists: {link_path}")
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target_path), str(link_path))
    else:
        try:
            os.symlink(target_path, link_path, target_is_directory=True)
        except FileExistsError as exc:
            raise RuntimeError(f"venv link target already exists: {link_path}") from exc


def _is_git_tracked(repo_root: Path, path: Path) -> bool:
    """Check if a path is tracked by git in the repo.

    Returns True if the path is tracked, False if it's untracked or doesn't exist.
    """
    result = run_captured(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(repo_root))],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    return result.ok


# Path components and suffixes that must never be materialized into a worktree.
# ``.git`` is the repo metadata directory. ``__pycache__`` and ``*.pyc`` are
# compiled Python bytecode — interpreter-version-specific, regenerable, and a
# drift signal when stale mixed-version .pyc files propagate across machines
# or interpreter upgrades (issue #711).
_MATERIALIZED_EXCLUDED_DIR_PARTS: frozenset[str] = frozenset({".git", "__pycache__"})
_MATERIALIZED_EXCLUDED_FILE_SUFFIXES: tuple[str, ...] = (".pyc",)


def _is_materialize_excluded(path: Path) -> bool:
    """Return True if *path* should be skipped during directory materialization.

    Excludes ``.git`` / ``__pycache__`` directory components anywhere in the
    path, and ``*.pyc`` files (compiled bytecode).  This is the single
    enforcement point for the materialization exclusion set — both the file
    copy loop and the empty-subdirectory recreation loop in
    ``_materialize_directory`` route through here.
    """
    if any(part in _MATERIALIZED_EXCLUDED_DIR_PARTS for part in path.parts):
        return True
    return path.suffix in _MATERIALIZED_EXCLUDED_FILE_SUFFIXES


def _materialize_directory(repo_root: Path, worktree_path: Path, dir_path: str) -> tuple[str, ...]:
    """Copy files from ``repo_root / dir_path`` into the worktree.

    Files are copied file-by-file; existing target files are only overwritten
    when the source content differs. All tracked paths under the configured
    materialize surface are marked ``assume-unchanged`` so that orchestrator-
    injected content (e.g. per-dispatch ``.devin/prompts/worker.md``) written by
    an external shim or worker after ``create_worktree`` returns does not
    appear as working-tree dirt in ``git status`` or CI-parity clean-tree
    gates. Empty source subdirectories are recreated in the target (matching
    the behaviour of the previous ``shutil.copytree`` implementation).

    Compiled Python bytecode (``__pycache__/`` directories and ``*.pyc`` files)
    is never materialized — it is interpreter-version-specific, regenerable on
    first import, and propagating stale mixed-version ``.pyc`` files into
    worktrees is a drift signal with no upside (issue #711).

    Returns a tuple of worktree-relative paths (forward-slash normalized) that
    were written, derived from the materializer's own manifest.
    """
    dir_path_posix = Path(dir_path).as_posix()
    source = repo_root / dir_path_posix
    if not source.exists():
        return ()
    if not contains(repo_root, source):
        return ()

    target_root = worktree_path / dir_path_posix
    if source.is_file():
        source_files = [source]
        source_root = source.parent
    else:
        source_files = [p for p in source.rglob("*") if p.is_file()]
        source_root = source

    written: list[str] = []
    for src_file in source_files:
        if _is_materialize_excluded(src_file):
            continue

        if source.is_file():
            rel = Path(".")
        else:
            rel = src_file.relative_to(source_root)
        target_file = (target_root / rel).resolve()
        if target_file.is_dir():
            continue
        if target_file.exists():
            try:
                if target_file.read_bytes() == src_file.read_bytes():
                    continue
            except OSError:
                pass

        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, target_file)
        rel_worktree = (Path(dir_path_posix) / rel).as_posix()
        written.append(rel_worktree)

    # Recreate empty source subdirectories that the file copy loop above would
    # otherwise silently drop.
    if source.is_dir():
        target_root.mkdir(parents=True, exist_ok=True)
        for src_dir in source.rglob("*"):
            if not src_dir.is_dir() or _is_materialize_excluded(src_dir):
                continue
            rel = src_dir.relative_to(source_root)
            (target_root / rel).mkdir(parents=True, exist_ok=True)

    # Mark every tracked path under the configured materialize surface as
    # assume-unchanged, regardless of whether the materializer's own copy loop
    # rewrote it. This is what shields injected files from later writes (e.g.
    # a worker launch shim mutating ``.devin/prompts/*.md`` after
    # ``create_worktree`` returns) from ``git status`` and clean-tree gates.
    ls_result = run_captured(
        ["git", "ls-files", "--", dir_path_posix],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if ls_result.ok:
        tracked_paths = [line for line in ls_result.stdout.splitlines() if line]
        if tracked_paths:
            assume_result = run_captured(
                ["git", "update-index", "--assume-unchanged", "--", *tracked_paths],
                cwd=worktree_path,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            )
            if not assume_result.ok:
                raise RuntimeError(
                    f"Failed to mark materialized paths as assume-unchanged: "
                    f"{assume_result.error or assume_result.stderr}"
                )

    return tuple(written)


_PERMANENT_NO_MATCH_ERROR_PREFIXES = (
    "no session found matching working_directory",
    "no pid",
    "per-PID log directory not found",
    "no per-PID log found",
    "no worktree_path provided",
    "worktree path is not a directory",
    "no eligible worktree files found",
)


def _is_permanent_no_match_error(error: str | None) -> bool:
    """Return True when a probe error indicates the source has no record at all.

    A missing sessions.db row, a missing per-PID log, or a missing worktree are
    structurally permanent for a worker that never registered. Transient errors
    (locked/corrupt DB, schema drift, I/O failures) are *not* permanent and
    remain fail-closed: they do not count toward the deferral cap.
    """
    if not error:
        return False
    return any(error.startswith(prefix) for prefix in _PERMANENT_NO_MATCH_ERROR_PREFIXES)


def _worker_kind_from_recovery(recovery: dict[str, Any], config: OrchestratorConfig) -> str | None:
    """Extract the recorded worker adapter kind from a recovery record.

    Returns the most recent ``adapter_history`` entry's ``kind`` (written by
    ``routing.record_adapter_choice`` when api routing is enabled), falling
    back to ``config.devin.adapter`` when no history is recorded (api routing
    disabled — every worker uses the repo default adapter). Returns ``None``
    only when neither source yields a usable string, which causes
    ``real_activity_for_worker`` to consult all sources as before (issue #639).
    """
    history = recovery.get("adapter_history")
    if isinstance(history, list) and history:
        latest = history[-1]
        if isinstance(latest, dict):
            kind = latest.get("kind")
            if isinstance(kind, str) and kind:
                return kind
    default = config.devin.adapter
    if isinstance(default, str) and default:
        return default
    return None


def _probe_recovery_liveness(
    recovery: dict[str, Any],
    worktree_path: Path,
    config: OrchestratorConfig | None,
    issue_number: int | None,
) -> None:
    """Abort recovery/redispatch if the recorded worker is still alive.

    Checks the recorded PID with start-time fingerprint matching first. If the
    PID is dead/wrong, corroborate with the Devin CLI's real-activity probe
    (``real_activity_for_worker``) for devin-shell sessions — a recent
    ``sessions.db`` message_nodes row OR a fresh per-PID Devin log for this
    worktree means the real leaf session is still moving even if the wrapper
    PID is gone.

    The probe is best-effort and never raises by design: a schema-drifted or
    locked sessions.db, or an unreadable per-PID log, comes back as an
    ``ActivitySource`` with ``error`` set rather than a timestamp. That is an
    INCONCLUSIVE result, not a confirmed-dead one — treating it as "no
    activity" would fail OPEN into the same destructive reset issue #282 exists
    to prevent (this reintroduced that exact shape once already; see the
    review that required this fix). So an errored source aborts recovery unless
    every errored source is a structurally permanent no-match and the same
    ``max_inconclusive_probe_deferrals`` cap used by the stall/dead lanes has
    been reached (issue #426).

    Raises ``LiveWorkerRedispatchError`` when a live signal is detected, or
    when liveness could not be determined and the deferral cap has not yet
    been reached.
    """
    resolved_config = config or OrchestratorConfig()

    raw_deferred_count = recovery.get("inconclusive_probe_deferred_count", 0)
    try:
        current_deferred_count = int(raw_deferred_count) if raw_deferred_count is not None else 0
    except (TypeError, ValueError):
        current_deferred_count = 0

    worker_pid = recovery.get("worker_pid")
    worker_process_start_time = recovery.get("worker_process_start_time")
    if worker_pid is None:
        worker_pid = recovery.get("last_known_worker_pid")
        worker_process_start_time = recovery.get("last_known_worker_process_start_time")

    try:
        worker_pid = int(worker_pid) if worker_pid is not None else None
    except (TypeError, ValueError):
        worker_pid = None

    pid_alive = False
    if worker_pid is not None:
        pid_alive = is_pid_alive(worker_pid, worker_process_start_time)
        if pid_alive:
            raise LiveWorkerRedispatchError(
                issue_number=issue_number,
                pid=worker_pid,
                process_start_time=worker_process_start_time,
                probe_result="pid_alive",
                inconclusive_probe_deferred_count=0,
            )
    # A confirmed-dead PID is stronger evidence than an inconclusive activity
    # probe. Reconcile the two signals at this single point (issue #506).
    confirmed_dead = worker_pid is not None and not pid_alive

    # For devin-shell sessions, the real-activity probe (sessions.db +
    # per-PID Devin log) is the source of truth even when the wrapper PID is
    # gone or has been recycled. The probe's Devin sources are skipped for
    # non-Devin workers (issue #639): a claude-code/api worker has no
    # sessions.db rows or per-PID Devin logs by construction, so consulting
    # them would produce permanent "no session found" / "no pid" errors that
    # block recovery forever — "no Devin subject exists" is not "subject
    # exists but could not be read".
    if resolved_config.devin.adapter == "devin-shell":
        started_at = recovery.get("started_at") or recovery.get("dispatched_at") or ""
        pm_config = resolved_config.post_mortem
        now = datetime.now(UTC)
        worker_kind = _worker_kind_from_recovery(recovery, resolved_config)
        try:
            probe = real_activity_for_worker(
                pm_config,
                str(worktree_path),
                started_at,
                worker_pid,
                now,
                worker_kind=worker_kind,
            )
        except Exception:
            # The probe is best-effort; never let it crash the recovery path.
            return

        # Any errored source means liveness is UNKNOWN for that signal, not
        # confirmed dead — never let an inconclusive probe fall through to a
        # destructive reset (issue #282 rework). However, if every errored
        # source is a structurally permanent absence-of-record, apply the same
        # max_inconclusive_probe_deferrals cap used by the stall/dead lanes so
        # a worker whose probe is permanently broken is not stuck forever
        # (issue #426).
        errored_sources = [source for source in probe.sources if source.error is not None]
        # Bind ``all_permanent`` once before the first ``errored_sources`` branch
        # so the second guarded read below is provably bound to Pyright (issue
        # #640). The default is never observed in a decision: both reads of
        # ``all_permanent`` are guarded by ``if errored_sources:``, and a
        # truthy ``errored_sources`` always rebinds it here first.
        all_permanent = False
        if errored_sources:
            all_permanent = all(
                _is_permanent_no_match_error(source.error) for source in errored_sources
            )
            max_deferrals = resolved_config.watchdog.max_inconclusive_probe_deferrals
            if all_permanent and current_deferred_count >= max_deferrals:
                # Probe has been inconclusive for N consecutive passes and every
                # failure is a permanent absence-of-record. Allow the destructive
                # reset rather than leaving the issue stuck indefinitely.
                return

        # Consult BOTH activity sources via the probe's own freshest-signal
        # aggregation (the same corroboration classify_worker_health uses for
        # the stall watchdog, issue #280) instead of a sessions.db-only check
        # that would ignore fresh devin_per_pid_log activity.
        stall_minutes = resolved_config.watchdog.stall_minutes or 20
        stall_seconds = stall_minutes * 60
        if probe.latest_timestamp is not None:
            staleness_seconds = (now - probe.latest_timestamp).total_seconds()
            if staleness_seconds <= stall_seconds:
                source_label = (probe.latest_source or "real_activity").replace(".", "_")
                raise LiveWorkerRedispatchError(
                    issue_number=issue_number,
                    pid=worker_pid,
                    process_start_time=worker_process_start_time,
                    probe_result=f"{source_label}_activity",
                    inconclusive_probe_deferred_count=0,
                )

        if errored_sources:
            if all_permanent and confirmed_dead:
                # A confirmed-dead PID overrides an inconclusive probe only
                # when every errored source is a structurally permanent absence-
                # of-record. Transient errors (locked/corrupt DB, I/O failures)
                # remain fail-closed (issue #282/#426).
                return

            new_deferred_count = current_deferred_count + 1
            raise LiveWorkerRedispatchError(
                issue_number=issue_number,
                pid=worker_pid,
                process_start_time=worker_process_start_time,
                probe_result="probe_error",
                inconclusive_probe_deferred_count=new_deferred_count,
            )


def create_worktree(
    repo_root: Path,
    branch: str,
    *,
    base_ref: str = "HEAD",
    worktrees_dir: Path | None = None,
    venv_source: Path | None = None,
    materialize_dirs: tuple[str, ...] = (),
    rework: bool = False,
    recovery: dict[str, Any] | None = None,
    issue_number: int | None = None,
    config: OrchestratorConfig | None = None,
    sessions_dir: Path | None = None,
) -> WorktreeInfo:
    """Create a git worktree for ``branch`` (a new branch) off ``base_ref``.

    If ``venv_source`` is given, a Windows junction (symlink elsewhere) is
    created at ``<worktree>/.venv`` pointing at it, so workers share one
    dev+eval virtualenv instead of cold-building their own. Raises
    RuntimeError if that link target already exists in the fresh worktree
    (programmer error / stale state — fail loudly rather than silently
    reusing or overwriting it).

    If ``rework`` is True, the branch is assumed to already exist (from a
    previous PR cycle). In rework mode:
    - If a worktree for the branch already exists, fetch and fast-forward it
      to the origin tip. When the local branch has diverged non-FF from
      origin (e.g. the PR was rebased), hard-reset the worktree and branch
      to ``origin/{branch}`` instead of failing; the old tip is snapshotted
      first if ``issue_number`` is provided.
    - Otherwise, use ``git worktree add <path> <branch>`` (no ``-b``) to
      attach to the existing branch at the origin tip. On non-FF divergence
      the local branch ref is reset to ``origin/{branch}`` before the
      worktree is added.

    For an OPEN PR's agent branch, origin is authoritative. A non-FF reset is
    guarded by a patch-id comparison against ``base_ref``; the resulting
    reason string is surfaced in ``WorktreeInfo.reclaimed``.

    If ``recovery`` is provided (a dict with state file dispatch record),
    this is a dead-worker recovery re-dispatch. The dict must contain
    ``branch_name`` matching the requested ``branch``. Recovery mode:
    - If the leftover worktree/branch has NO commits beyond the merge-base
      with ``base_ref`` (crashed before committing): remove worktree + branch
      and create fresh — a clean restart.
    - If it HAS commits or a dirty tree (crashed mid-work): reuse via the
      existing rework-style attach (fetch/ff if possible), so the relaunched
      worker continues from the partial work.
    - If the branch exists WITHOUT a matching state record (foreign state):
      fail loudly — that protects against clobbering anything that is not ours.

    The ``base_ref`` parameter controls where the new branch bases off:
    - Empty string ("") means auto-resolve to the repository's default branch
      as a remote-tracking ref (e.g., "origin/main"). This is the recommended
      setting for production to ensure fresh worktrees base off the latest
      remote tip instead of a potentially stale local HEAD.
    - A remote-tracking ref like "origin/main" or "origin/master" will trigger
      a git fetch before worktree creation to ensure the ref is up-to-date.
    - Any other ref (e.g., "HEAD", a commit SHA, or a local branch name) is
      used as-is without fetching.

    ``issue_number``, when given, enables attempt-tip preservation (issue
    #261): immediately before any ``git branch -D`` in this function that
    could discard commits a dead worker made but never got to push, the old
    tip is snapshotted to ``refs/charlie/attempts/issue-<n>/attempt-<k>``
    (see ``attempt_refs.snapshot_attempt_ref``) and surfaced on the returned
    ``WorktreeInfo.attempt_snapshot``. Best-effort — a snapshot failure never
    blocks the branch reset. When ``issue_number`` is None, no snapshot is
    attempted (existing callers/tests are unaffected).

    ``sessions_dir``, when given, enables the foreign-writer marker guard
    (issue #400): the worktree is checked for a ``.charlie-writer.json``
    marker. Worker markers with a live pid that does not belong to a recorded
    session are refused; operator markers are live while state.json reports an
    active ``operator_claimed_at`` for ``issue_number``. Pass ``None`` to skip
    the guard (e.g. unit tests that focus on worktree git mechanics).
    """
    # Validate branch/base_ref before they reach any git argv (issue #659).
    # branch originates from GitHub-derived branch names; base_ref may be a
    # ref name, a SHA, or "" (auto-resolve sentinel).
    branch = require_valid_ref_name(branch, context="create_worktree branch")
    if base_ref != "":
        require_valid_rev(base_ref, context="create_worktree base_ref")

    # Resolve base_ref: empty string means auto-resolve to origin/<default>
    resolved_base_ref = base_ref
    if base_ref == "":
        resolved_base_ref = _resolve_default_branch_ref(repo_root)

    target_dir = worktrees_dir or _default_worktrees_dir(repo_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    worktree_path = target_dir / _slugify(branch)

    # Worktree-relative paths that the orchestrator injects (e.g. rendered
    # prompts). These are excluded from "is this worktree dirty?" checks.
    injected_paths = config.dispatch.injected_paths if config is not None else ()

    # Issue #400: refuse to enter a worktree that is currently occupied by a
    # live foreign writer. The marker check is independent of the recovery
    # liveness probe so an operator's claim is honored before any git reset.
    state_file: Path | None = None
    if config is not None:
        state_file = runtime_paths(repo_root, config.runtime.state_dir).state_file
    if sessions_dir is not None:
        _check_worktree_writer_marker(
            worktree_path, sessions_dir, issue_number=issue_number, state_file=state_file
        )

    # Recovery mode: dead-worker re-dispatch with leftover worktree/branch
    reclaimed: str | None = None
    attempt_snapshot: AttemptSnapshot | None = None
    rework_conflict: ReworkMergeConflict | None = None
    rescue_capture: RescueCapture | None = None

    def _snapshot_before_delete(target_branch: str) -> None:
        """Best-effort attempt-tip snapshot immediately before a branch reset.

        Updates the enclosing ``attempt_snapshot`` (last snapshot wins — only
        one branch-delete path fires per call). No-op when ``issue_number``
        was not provided.
        """
        nonlocal attempt_snapshot
        if issue_number is None:
            return
        attempt_snapshot = snapshot_attempt_ref(
            repo_root, target_branch, issue_number, base_ref=resolved_base_ref
        )

    def _emit_rescue_event(capture: RescueCapture, unsafe_reason: str, wt_path: Path) -> None:
        """Best-effort: record a ``worktree_rescue_captured`` event (issue #849).

        Deferred import so worktree.py never hard-depends on instrumentation's
        own import chain (ci_fleet at module bottom). A missing state_file
        (no config) or an instrumentation I/O error is silently skipped —
        the rescue ref itself is the durable artifact, not the event.
        """
        if state_file is None:
            return
        try:
            from .instrumentation import log_event

            log_event(
                state_file,
                "worktree_rescue_captured",
                {
                    "issue_number": issue_number,
                    "rescue_ref": capture.ref_name,
                    "commit_sha": capture.commit_sha,
                    "worktree_path": str(wt_path),
                    "reason": unsafe_reason,
                },
            )
        except Exception:  # noqa: BLE001 — instrumentation is best-effort
            pass

    def _capture_or_raise(
        check_path: Path, unsafe_reason: str, capture_injected: tuple[str, ...]
    ) -> None:
        """Attempt rescue capture before refusing a reset (issue #849).

        If capture succeeds, records it on the enclosing ``rescue_capture``,
        cleans the working tree so the captured dirty content cannot survive
        into a new work session, and returns — the reset is permitted because
        the work is now durable on a ref. If capture fails, raises
        ``WorktreeUnsafeError`` exactly as today — capture failure must never
        downgrade the safety property.

        The working-tree clean is the single enforcement point for the "never
        commit tracked modifications the shim did not itself produce"
        invariant on the capture-succeeds path. Without it, a reuse-in-place
        ``git merge --ff-only`` that doesn't touch the dirty files would leave
        them in place, silently surviving into the next worker's session
        (issue #849 rework finding).
        """
        nonlocal rescue_capture
        capture = _capture_worktree_work_to_rescue_ref(
            repo_root,
            check_path,
            issue_number,
            capture_injected,
            materialize_dirs,
        )
        if capture.error is None and capture.ref_name is not None:
            rescue_capture = capture
            _emit_rescue_event(capture, unsafe_reason, check_path)
            _clean_captured_worktree(check_path, capture_injected)
            return
        raise WorktreeUnsafeError(unsafe_reason)

    def _clean_captured_worktree(wt_path: Path, clean_injected: tuple[str, ...]) -> None:
        """Reset and clean the working tree after a successful rescue capture.

        The work is already durable on a rescue ref, so discarding the
        working-tree copy is safe. Tracked modifications are reset to HEAD;
        untracked files are removed except orchestrator scaffolding
        (``.venv``, ``clean_injected``, ``materialize_dirs``) — the same
        exclusions the capture used, so exactly what was captured is
        discarded and exactly what was excluded is preserved.

        Raises ``RuntimeError`` on failure: a successful capture with a
        still-dirty tree is a hazardous state that must surface, not be
        silently left for the next worker to commit.
        """
        reset_result = run_captured(
            ["git", "reset", "--hard", "HEAD"],
            cwd=wt_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not reset_result.ok:
            raise RuntimeError(
                f"Cannot reset worktree after rescue capture: "
                f"{reset_result.error or reset_result.stderr}"
            )
        exclusions: list[str] = [":(exclude).venv"]
        for p in (*clean_injected, *materialize_dirs):
            normalized = str(p).replace("\\", "/").strip("/")
            if normalized:
                exclusions.append(f":(exclude){normalized}")
        clean_result = run_captured(
            ["git", "clean", "-fd", "--", ".", *exclusions],
            cwd=wt_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not clean_result.ok:
            raise RuntimeError(
                f"Cannot clean worktree after rescue capture: "
                f"{clean_result.error or clean_result.stderr}"
            )

    def _raise_if_unsafe_to_reset(target_path: Path | None = None) -> None:
        """Hard-refuse to reset if the worktree/branch contains local work."""
        check_path = target_path or worktree_path
        reason = _worktree_refuse_to_reset_reason(
            repo_root,
            branch,
            resolved_base_ref,
            check_path,
            injected_paths,
            materialize_dirs,
        )
        if not reason:
            return
        # Issue #1141: dirt in a LIVE writer's tree is normal working state,
        # not residue — a dirt verdict is only meaningful once death is
        # established. The recovery liveness probe upstream can miss a live
        # writer (stale/recycled pid in the recovery record), so re-check the
        # worktree's own writer marker at the point of judgment, the same
        # plan-is-a-snapshot discipline park_runner_slot applies before
        # terminating a listener. A live writer defers the redispatch; it is
        # never escalated as unsafe.
        marker = read_worktree_marker(check_path)
        if marker is not None and marker.get("kind") != OPERATOR_MARKER_KIND:
            marker_pid = marker.get("pid")
            marker_session_id = marker.get("session_id")
            live_pid: int | None = None
            if sessions_dir is not None and isinstance(marker_session_id, str):
                # Prefer the recorded session sidecar: it carries the
                # process-start-time fingerprint, defeating pid recycling.
                live_pid = _own_live_session_pids(sessions_dir).get(marker_session_id)
            if (
                live_pid is None
                and isinstance(marker_pid, int)
                and marker_pid > 0
                and is_pid_alive(marker_pid, None)
            ):
                live_pid = marker_pid
            if live_pid is not None:
                raise LiveWorkerRedispatchError(
                    issue_number=issue_number,
                    pid=live_pid,
                    process_start_time=None,
                    probe_result="live_writer_at_unsafe_evaluation",
                    inconclusive_probe_deferred_count=0,
                )
        # Issue #849: before refusing, attempt to capture the work durably
        # onto a rescue ref. If capture succeeds, the reset is permitted
        # (the work is preserved on a ref, so resetting destroys nothing).
        # If capture fails, the refusal stands — capture failure must never
        # downgrade the safety property.
        _capture_or_raise(check_path, reason, injected_paths)

    if recovery is not None:
        # Validate that the recovery record matches the requested branch
        recovery_branch = recovery.get("branch_name")
        if recovery_branch != branch:
            raise RuntimeError(
                f"Recovery record branch_name {recovery_branch!r} does not match "
                f"requested branch {branch!r}"
            )

        # Issue #282: verify the prior worker is actually dead before any
        # destructive reset. If the recorded PID (or sessions.db real activity)
        # says the worker is still alive, abort the redispatch and let the
        # orchestrator restore the in-progress label.
        _probe_recovery_liveness(recovery, worktree_path, config, issue_number)

        # Issue #110: Check if the branch exists on origin before attempting fetch
        # If the branch doesn't exist on origin (killed before first push), fall through
        # to fresh dispatch instead of failing with fetch error 128
        # If the probe fails (network/auth error), abort dispatch to avoid data loss
        remote_exists = None
        if _has_origin_remote(repo_root):
            remote_exists = _remote_branch_exists(repo_root, branch)
            if remote_exists is None:
                # Probe failed: abort dispatch to avoid triggering fallback on transient error
                raise RuntimeError(
                    f"Failed to probe remote branch {branch!r} for recovery: "
                    f"transient network or auth error. Aborting dispatch to avoid data loss."
                )
        if remote_exists is False:
            # Branch doesn't exist on origin - this is a killed-before-push session
            # Fall through to fresh dispatch (rework=False) after cleaning up local state
            reclaimed = "fetch-fallback"
            # Clean up any local worktree/branch that might exist
            existing_worktrees = list_worktrees(repo_root)
            existing_wt = next(
                (wt for wt in existing_worktrees if Path(wt["worktree"]) == worktree_path),
                None,
            )
            if existing_wt:
                wt_path = Path(existing_wt["worktree"])
                # Check if it's on the correct branch (should be, since it's our recovery record)
                wt_branch = existing_wt.get("branch", "")
                normalized_wt_branch = wt_branch.replace("refs/heads/", "")
                if (
                    normalized_wt_branch == branch
                    or normalized_wt_branch == f"refs/heads/{branch}"
                ):
                    # It's our worktree - refuse to reset it if it holds local work
                    _raise_if_unsafe_to_reset(wt_path)
                    # Remove the worktree (junction-safe)
                    if not remove_worktree(repo_root, wt_path, force=True):
                        raise RuntimeError(
                            f"Failed to remove leftover worktree {wt_path} for recovery"
                        )
            # Delete the local branch if it exists
            branch_result = run_captured(
                ["git", "branch", "--list", branch],
                cwd=repo_root,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            )
            if branch_result.ok and branch_result.stdout.strip():
                # fetch-fallback: refuse to reset if the branch holds local work.
                _raise_if_unsafe_to_reset(worktree_path)
                # The branch never made it to origin, so any commits here are
                # local-only. If the guard allowed the reset, the worktree is
                # clean and the branch has no local commits, so we can safely
                # remove the branch. Snapshot the tip before deleting it, best-
                # effort, as a defensive artifact for post-mortem.
                _snapshot_before_delete(branch)
                branch_delete_result = run_captured(
                    ["git", "branch", "-D", branch],
                    cwd=repo_root,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                if not branch_delete_result.ok:
                    raise RuntimeError(
                        f"Failed to delete branch {branch!r} for recovery: "
                        f"{branch_delete_result.error or branch_delete_result.stderr}"
                    )
            # Fall through to fresh dispatch below (rework=False)
        else:
            # Branch exists on origin or no origin - proceed with normal recovery logic
            # Check if a worktree exists at the expected path (by slug)
            existing_worktrees = list_worktrees(repo_root)
            existing_wt = next(
                (wt for wt in existing_worktrees if Path(wt["worktree"]) == worktree_path),
                None,
            )

            if existing_wt:
                # AC #3: Fail loudly if the worktree at the expected path is on a FOREIGN branch
                # (i.e., a worktree whose branch does not match our recovery record).
                # This protects against clobbering work that is not ours.
                wt_branch = existing_wt.get("branch", "")
                # Normalize branch names for comparison (strip refs/heads/ prefix)
                normalized_wt_branch = wt_branch.replace("refs/heads/", "")
                if (
                    normalized_wt_branch != branch
                    and normalized_wt_branch != f"refs/heads/{branch}"
                ):
                    raise RuntimeError(
                        f"Recovery mode found leftover worktree at {worktree_path} on foreign branch {normalized_wt_branch!r}, "
                        f"but recovery record specifies branch {branch!r}. "
                        f"This is not our crashed worker — refusing to clobber foreign work."
                    )

                # Worktree exists on the correct branch: check if it has commits beyond the merge-base
                wt_path = Path(existing_wt["worktree"])
                # Check for dirty working tree, ignoring orchestrator-injected files.
                try:
                    has_dirty = _worker_authored_dirty(wt_path, injected_paths, materialize_dirs)
                except WorktreeProbeFailedError:
                    # If the probe fails (index lock, corruption, permissions), treat as dirty to be safe
                    has_dirty = True

                # Check for commits beyond merge-base with resolved_base_ref
                merge_base_result = run_captured(
                    ["git", "merge-base", resolved_base_ref, branch],
                    cwd=repo_root,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                if merge_base_result.ok:
                    merge_base = merge_base_result.stdout.strip()
                    # Count commits from merge-base to branch tip
                    rev_list_result = run_captured(
                        ["git", "rev-list", "--count", f"{merge_base}..{branch}"],
                        cwd=repo_root,
                        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                    )
                    has_commits = rev_list_result.ok and int(rev_list_result.stdout.strip()) > 0
                else:
                    # If merge-base fails, assume has commits to be safe
                    has_commits = True

                if has_commits or has_dirty:
                    # Has work: reuse via rework-style attach
                    # Fall through to rework logic below by setting rework=True
                    rework = True
                else:
                    # Clean: remove worktree and branch, then create fresh
                    _raise_if_unsafe_to_reset(wt_path)
                    if not remove_worktree(repo_root, wt_path, force=True):
                        raise RuntimeError(
                            f"Failed to remove leftover worktree {wt_path} for recovery"
                        )
                    # Delete the branch and check the result
                    _raise_if_unsafe_to_reset(worktree_path)
                    _snapshot_before_delete(branch)
                    branch_delete_result = run_captured(
                        ["git", "branch", "-D", branch],
                        cwd=repo_root,
                        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                    )
                    if not branch_delete_result.ok:
                        raise RuntimeError(
                            f"Failed to delete branch {branch!r} for recovery: "
                            f"{branch_delete_result.error or branch_delete_result.stderr}"
                        )
                    reclaimed = "pruned"
                    # Fall through to fresh dispatch below (rework=False)
            else:
                # No worktree exists, but branch might exist
                # Check if branch exists
                branch_result = run_captured(
                    ["git", "branch", "--list", branch],
                    cwd=repo_root,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                if branch_result.ok and branch_result.stdout.strip():
                    # Branch exists without worktree: check commits and reuse or delete
                    merge_base_result = run_captured(
                        ["git", "merge-base", resolved_base_ref, branch],
                        cwd=repo_root,
                        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                    )
                    if merge_base_result.ok:
                        merge_base = merge_base_result.stdout.strip()
                        rev_list_result = run_captured(
                            ["git", "rev-list", "--count", f"{merge_base}..{branch}"],
                            cwd=repo_root,
                            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                        )
                        has_commits = (
                            rev_list_result.ok and int(rev_list_result.stdout.strip()) > 0
                        )
                    else:
                        has_commits = True

                    if has_commits:
                        # Has commits: reuse via rework-style attach
                        rework = True
                    else:
                        # Clean: delete branch and create fresh
                        _raise_if_unsafe_to_reset(worktree_path)
                        _snapshot_before_delete(branch)
                        branch_delete_result = run_captured(
                            ["git", "branch", "-D", branch],
                            cwd=repo_root,
                            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                        )
                        if not branch_delete_result.ok:
                            raise RuntimeError(
                                f"Failed to delete branch {branch!r} for recovery: "
                                f"{branch_delete_result.error or branch_delete_result.stderr}"
                            )
                        reclaimed = "pruned"
                        # Fall through to fresh dispatch below (rework=False)

    if rework:
        # Rework mode: branch already exists, reuse or attach to it
        existing_worktrees = list_worktrees(repo_root)

        # Issue #461: a leftover worktree can occupy the target path DETACHED
        # (crashed attempt, reboot mid-rework). `git worktree list --porcelain`
        # emits no `branch` line for a detached worktree, so the branch-name
        # match below can never see it — and the attach path's
        # `git worktree add` against the still-registered directory then fails
        # with exit 128 on every pass, forever. Reclaim by path first,
        # mirroring the fresh-dispatch reclaim (issue #110) and the recovery
        # path, before falling back to the branch-name lookup.
        stale_at_path = next(
            (wt for wt in existing_worktrees if Path(wt["worktree"]) == worktree_path),
            None,
        )
        if stale_at_path is not None:
            stale_branch = (stale_at_path.get("branch") or "").replace("refs/heads/", "")
            if stale_branch != branch:
                if not worktree_path.exists():
                    # Directory missing but still registered: prune the record.
                    run_captured(
                        ["git", "worktree", "prune"],
                        cwd=repo_root,
                        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                    )
                else:
                    # Refuse to clobber uncommitted or unpushed work — a dirty
                    # leftover needs attention, not silent deletion. The branch
                    # ref itself is never deleted here, only the checkout.
                    _raise_if_unsafe_to_reset(worktree_path)
                    if not remove_worktree(repo_root, worktree_path, force=True):
                        raise RuntimeError(
                            f"Failed to reclaim stale worktree {worktree_path} "
                            f"for rework branch {branch!r}"
                        )
                existing_worktrees = list_worktrees(repo_root)

        # Branch names in git worktree list may have refs/heads/ prefix
        existing_wt = next(
            (
                wt
                for wt in existing_worktrees
                if wt.get("branch", "").endswith(f"/{branch}") or wt.get("branch") == branch
            ),
            None,
        )

        if existing_wt:
            # Issue #1118: refuse to adopt a worktree at a foreign path. The
            # branch-name lookup above spans ALL registered worktrees, so a
            # branch checked out by the operator in a different directory
            # (e.g. .claude/worktrees/<name>) would be silently adopted,
            # committing the operator's uncommitted edits as worker output.
            # Only a worktree at the orchestrator's expected path is one we
            # could have created; anything else is a foreign checkout.
            existing_wt_path = Path(existing_wt["worktree"])
            if existing_wt_path != worktree_path:
                raise WorktreeForeignWriterError(
                    worktree_path=existing_wt_path,
                    pid=None,
                    session_id=None,
                )
            # Reuse existing worktree: fetch and fast-forward to origin tip
            worktree_path = existing_wt_path
            # Issue #1118: a dirty worktree at adoption time is an independent
            # hard stop — never commit tracked modifications the shim did not
            # itself produce. In recovery mode the dirt is a prior (owned)
            # worker's partial work, so the check is skipped there.
            #
            # The ``.venv`` junction is orchestrator scaffolding (a reparse
            # point, not worker content), so it is excluded from the dirty
            # check — a leftover junction from a prior dispatch must not
            # trigger a false positive here. It is unlinked below when
            # ``venv_source`` is None.
            if recovery is None:
                dirty_injected = injected_paths
                venv_link = worktree_path / ".venv"
                if is_junction(venv_link):
                    dirty_injected = injected_paths + (".venv",)
                dirty_reason = _worktree_dirty_reason(
                    worktree_path, dirty_injected, materialize_dirs
                )
                if dirty_reason:
                    _capture_or_raise(worktree_path, dirty_reason, dirty_injected)
            # Only fetch if origin remote exists (deterministic check)
            if _has_origin_remote(repo_root):
                # Fetch the remote-tracking ref only (branch:<branch> fails when branch is checked out)
                fetch_result = _run_remote_captured(
                    ["git", "fetch", "origin", branch],
                    cwd=repo_root,
                )
                # Fast-forward inside the worktree if fetch succeeded
                if fetch_result.ok:
                    ff_result = run_captured(
                        ["git", "merge", "--ff-only", f"origin/{branch}"],
                        cwd=worktree_path,
                        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                    )
                    if not ff_result.ok:
                        # Non-fast-forward: the PR branch was rebased or force-pushed
                        # on origin. For an open PR, origin is authoritative; snapshot
                        # the old tip (best-effort), compare patch-ids, and reset.
                        # Uncommitted worker-authored edits must survive, so refuse
                        # to reset a dirty worktree even when the branch diverged.
                        dirty_reason = _worktree_dirty_reason(
                            worktree_path, injected_paths, materialize_dirs
                        )
                        if dirty_reason:
                            _capture_or_raise(worktree_path, dirty_reason, injected_paths)
                        _snapshot_before_delete(branch)
                        base_branch = (
                            resolved_base_ref[len("origin/") :]
                            if resolved_base_ref.startswith("origin/")
                            else None
                        )
                        if base_branch and base_branch != branch:
                            _run_remote_captured(
                                ["git", "fetch", "origin", base_branch],
                                cwd=repo_root,
                            )
                        reset_reason = _rework_reset_reason(repo_root, branch, resolved_base_ref)
                        reset_result = run_captured(
                            ["git", "reset", "--hard", f"origin/{branch}"],
                            cwd=worktree_path,
                            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                        )
                        if not reset_result.ok:
                            raise RuntimeError(
                                f"Cannot reset rework branch {branch!r} to origin tip: "
                                f"{reset_result.error or reset_result.stderr}"
                            )
                        reclaimed = reset_reason
                # If fetch failed with origin present, raise (real network/error failure)
                if not fetch_result.ok:
                    raise RuntimeError(
                        f"[git fetch origin {branch}] Fetch failed for rework branch {branch!r}: "
                        f"{fetch_result.error or fetch_result.stderr}"
                    )
            if recovery is None:
                rework_conflict = _merge_update_rework_branch(
                    repo_root,
                    worktree_path,
                    branch,
                    resolved_base_ref,
                    injected_paths,
                    materialize_dirs,
                )
            venv_link = worktree_path / ".venv"
            venv_junction: Path | None = None
            if venv_source is not None:
                if venv_link.exists() or is_junction(venv_link):
                    venv_junction = venv_link
                else:
                    try:
                        _create_junction_or_symlink(venv_link, venv_source)
                        venv_junction = venv_link
                    except (OSError, RuntimeError):
                        # Clean up the orphan worktree (but not the branch, which already exists in rework mode)
                        remove_worktree(repo_root, worktree_path, force=True, branch=None)
                        raise
            else:
                # If a pre-existing .venv junction from the previous default era
                # is present, unlink it so the worker's uv sync creates a local
                # .venv instead of writing through the reparse point.
                if is_junction(venv_link):
                    _unlink_reparse_point(venv_link)
            return WorktreeInfo(
                path=worktree_path,
                branch=branch,
                venv_junction=venv_junction,
                reclaimed=reclaimed,
                attempt_snapshot=attempt_snapshot,
                rework_conflict=rework_conflict,
                rescue_capture=rescue_capture,
            )
        else:
            # No existing worktree: attach to existing branch (no -b flag)
            # Fetch first to ensure we materialize at the origin tip, but only if origin exists
            if _has_origin_remote(repo_root):
                fetch_result = _run_remote_captured(
                    ["git", "fetch", "origin", branch],
                    cwd=repo_root,
                )
                # If fetch failed with origin present, raise (real network/error failure)
                if not fetch_result.ok:
                    raise RuntimeError(
                        f"[git fetch origin {branch}] Fetch failed for rework branch {branch!r}: "
                        f"{fetch_result.error or fetch_result.stderr}"
                    )

                # Decide whether the local branch can stay at its current tip.
                # origin/branch is an ancestor of branch -> local is ahead/equal,
                # so keep the local tip (it contains the origin work). Otherwise
                # branch is behind or diverged; trust origin and reset the ref.
                origin_is_ancestor = _is_ancestor(repo_root, f"origin/{branch}", branch)
                branch_is_ancestor = _is_ancestor(repo_root, branch, f"origin/{branch}")
                if not origin_is_ancestor and not branch_is_ancestor:
                    # Diverged: snapshot, guard, and hard-reset the local ref.
                    _snapshot_before_delete(branch)
                    base_branch = (
                        resolved_base_ref[len("origin/") :]
                        if resolved_base_ref.startswith("origin/")
                        else None
                    )
                    if base_branch and base_branch != branch:
                        _run_remote_captured(
                            ["git", "fetch", "origin", base_branch],
                            cwd=repo_root,
                        )
                    reclaimed = _rework_reset_reason(repo_root, branch, resolved_base_ref)
                if not origin_is_ancestor:
                    # Local is behind or diverged: set branch to origin tip.
                    # When diverged this is a hard reset; when behind it is a
                    # fast-forward. Either way origin is authoritative for an open PR.
                    update_result = run_captured(
                        ["git", "branch", "-f", branch, f"origin/{branch}"],
                        cwd=repo_root,
                        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                    )
                    if not update_result.ok:
                        raise RuntimeError(
                            f"Cannot reset rework branch {branch!r} to origin tip: "
                            f"{update_result.error or update_result.stderr}"
                        )

            result = run_captured(
                ["git", "worktree", "add", str(worktree_path), branch],
                cwd=repo_root,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            )
            if not result.ok:
                raise RuntimeError(
                    f"git worktree add failed for rework branch {branch!r}: {result.error or result.stderr}"
                )
            if recovery is None:
                rework_conflict = _merge_update_rework_branch(
                    repo_root,
                    worktree_path,
                    branch,
                    resolved_base_ref,
                    injected_paths,
                    materialize_dirs,
                )
    else:
        # Fresh dispatch: create new branch off base_ref
        # Issue #110: Stale worktree reclamation before git worktree add
        # Issue #461: a pristine orchestrator-created leftover worktree can be
        # reclaimed directly without a remote fetch or ls-remote probe.
        existing_worktrees = list_worktrees(repo_root)
        existing_wt = next(
            (wt for wt in existing_worktrees if Path(wt["worktree"]) == worktree_path),
            None,
        )
        need_worktree_add = True
        if existing_wt:
            wt_path = Path(existing_wt["worktree"])
            if not wt_path.exists():
                # Directory missing but worktree still registered: prune it
                run_captured(
                    ["git", "worktree", "prune"],
                    cwd=repo_root,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                reclaimed = "pruned"
            elif _is_pristine_orchestrator_worktree(repo_root, wt_path, resolved_base_ref):
                # Reuse the pristine leftover worktree directly. It is already
                # at the current dispatch base (remote-tracking bases are
                # fetched and reset in-place by the pristine check), and it has
                # no worker-authored changes.
                worktree_path = wt_path
                reclaimed = "reused"
                need_worktree_add = False
            else:
                # Refuse to reset a worktree that still contains local work.
                # Partial/dirty worktrees are the redispatch case, not the fresh
                # dispatch case (issue #257).
                _raise_if_unsafe_to_reset(wt_path)
                # Directory is clean and has no local commits: remove and recreate
                if not remove_worktree(repo_root, wt_path, force=True):
                    raise RuntimeError(
                        f"git worktree remove failed for stale worktree {wt_path} for fresh dispatch"
                    )
                reclaimed = "pruned"

        if need_worktree_add:
            # Delete the branch if it exists (it might be leftover from a killed session)
            branch_result = run_captured(
                ["git", "branch", "--list", branch],
                cwd=repo_root,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            )
            if branch_result.ok and branch_result.stdout.strip():
                # Refuse to reset a branch that still contains local commits.
                _raise_if_unsafe_to_reset(worktree_path)
                # Leftover branch from a killed session may hold real, unpushed
                # commits (e.g. it died between committing and pushing) — snapshot
                # before reclaiming it, same as the fetch-fallback recovery path.
                _snapshot_before_delete(branch)
                branch_delete_result = run_captured(
                    ["git", "branch", "-D", branch],
                    cwd=repo_root,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                if not branch_delete_result.ok:
                    raise RuntimeError(
                        f"git branch -D failed for branch {branch!r} for fresh dispatch: "
                        f"{branch_delete_result.error or branch_delete_result.stderr}"
                    )

            if resolved_base_ref.startswith("origin/"):
                remote_branch = resolved_base_ref[len("origin/") :]
                fetch_result = _run_remote_captured(
                    ["git", "fetch", "origin", remote_branch],
                    cwd=repo_root,
                )
                if not fetch_result.ok:
                    raise RuntimeError(
                        f"[git fetch origin {remote_branch}] Failed to fetch base ref "
                        f"{resolved_base_ref!r} before worktree creation: "
                        f"{fetch_result.error or fetch_result.stderr}"
                    )

            result = run_captured(
                ["git", "worktree", "add", "-b", branch, str(worktree_path), resolved_base_ref],
                cwd=repo_root,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            )
            if not result.ok:
                raise RuntimeError(
                    f"git worktree add failed for branch {branch!r}: {result.error or result.stderr}"
                )

    venv_junction: Path | None = None
    if venv_source is not None:
        venv_link = worktree_path / ".venv"
        try:
            _create_junction_or_symlink(venv_link, venv_source)
            venv_junction = venv_link
        except (OSError, RuntimeError):
            # Clean up the orphan worktree and branch if junction creation fails
            # (to prevent leaks on stale .venv or bad venv_source)
            # In rework mode, the branch already exists, so don't delete it
            delete_branch = None if rework else branch
            remove_worktree(repo_root, worktree_path, force=True, branch=delete_branch)
            raise

    # Materialize git-excluded directories into the worktree
    materialized_paths: list[str] = []
    for dir_path in materialize_dirs:
        try:
            written = _materialize_directory(repo_root, worktree_path, dir_path)
        except (OSError, RuntimeError) as exc:
            # Clean up the worktree and branch if materialization fails
            delete_branch = None if rework else branch
            remove_worktree(repo_root, worktree_path, force=True, branch=delete_branch)
            raise RuntimeError(
                f"Failed to materialize directory {dir_path} into worktree: {exc}"
            ) from exc
        else:
            materialized_paths.extend(written)

    return WorktreeInfo(
        path=worktree_path,
        branch=branch,
        venv_junction=venv_junction,
        reclaimed=reclaimed,
        attempt_snapshot=attempt_snapshot,
        materialized_paths=tuple(materialized_paths),
        rework_conflict=rework_conflict,
        rescue_capture=rescue_capture,
    )


def remove_worktree(
    repo_root: Path, worktree_path: Path, *, force: bool = False, branch: str | None = None
) -> bool:
    """Remove a worktree, taking care never to follow reparse points into a
    shared virtualenv or other targets.

    Teardown order is mandatory:
      1. If ``<worktree>/.venv`` exists and is a real directory (not a
         junction/symlink), ABORT and return False unless ``force=True`` —
         and even then, the directory is removed as part of the normal tree
         deletion, never a junction target.
      2. If ``.venv`` is a junction/symlink, unlink the reparse point itself.
      3. Unlink any other reparse points found anywhere under the worktree so
         ``git worktree remove``/``shutil.rmtree`` cannot follow them.
      4. ``git worktree remove``.
      5. On failure, ``git worktree prune`` to clear stale metadata, then a
         reparse-point-safe ``shutil.rmtree`` fallback (only when ``force=True``
         so we do not destroy uncommitted work the caller asked us to keep).
      6. Verify the directory is actually gone; if not, report failure.
      7. If ``branch`` is provided, delete the branch with ``git branch -D``.

    Returns False for expected failures (real .venv dir without force, git
    command failure, directory survives); never raises for those. Programmer
    errors surface as False via a failed git command.
    """
    venv_path = worktree_path / ".venv"
    if venv_path.exists() or is_junction(venv_path):
        # Any OS-level failure removing the reparse point / local venv is an
        # "expected failure" per this function's contract (a locked/open file
        # under force=True raises PermissionError/WinError 32) — return False,
        # never raise, so one worktree's teardown can't crash the whole batch.
        try:
            if is_junction(venv_path):
                # Unlink the reparse point itself; never follow into the target.
                _unlink_reparse_point(venv_path)
            elif venv_path.is_dir():
                if not force:
                    return False
                # Real .venv directory: let git remove --force / rmtree delete it.
            else:
                venv_path.unlink()
        except OSError:
            return False

    # Remove any other directory symlinks/junctions in the tree before git or
    # rmtree touch it.  This is the single point of enforcement for reparse
    # point safety during worktree teardown (issue #462).
    _unlink_worktree_reparse_points(worktree_path)

    args = ["git", "worktree", "remove", str(worktree_path)]
    if force:
        args.append("--force")
    result = run_captured(args, cwd=repo_root, timeout_seconds=_DEFAULT_TIMEOUT_SECONDS)
    git_result_ok = result.ok
    if not git_result_ok:
        run_captured(
            ["git", "worktree", "prune"], cwd=repo_root, timeout_seconds=_DEFAULT_TIMEOUT_SECONDS
        )
        # When the caller has explicitly forced removal, fall back to a
        # reparse-point-safe rmtree.  Without force we must not silently
        # delete a worktree that git refused to remove (e.g. uncommitted work).
        if force:
            _robust_rmtree(worktree_path)

    # Post-delete verification: report failure if the directory survived.
    if not git_result_ok and not force:
        worktree_removed = False
    else:
        worktree_removed = not worktree_path.exists() and not is_junction(worktree_path)

    # Delete the branch if provided (to prevent branch leaks on launch failure)
    # Attempt branch deletion independently of worktree-removal success to avoid
    # leaking branches when worktree removal itself fails (e.g., Windows file locks)
    branch_deleted = True
    if branch is not None:
        branch_result = run_captured(
            ["git", "branch", "-D", branch],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        branch_deleted = branch_result.ok

    return worktree_removed and branch_deleted


def _commit_exists_locally(repo_root: Path, sha: str) -> bool:
    """True if ``sha`` resolves to a commit object already present locally."""
    result = run_captured(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    return result.ok


def create_review_checkout(
    repo_root: Path,
    pr_number: int,
    head_sha: str,
    *,
    reviews_dir: Path,
) -> WorktreeInfo:
    """Create an isolated, detached-HEAD checkout of a PR's head SHA for a
    reviewer session.

    This is the reviewer-side sibling of ``create_worktree``, but deliberately
    NOT built on top of it: a reviewer must never share a worker's branch or
    worktree path. Where ``create_worktree`` keys the checkout path by
    branch-slug under ``worktrees_dir`` and (in ``rework``/``review`` mode)
    attaches to the branch's live tip, this function keys the path by PR
    number under a distinct ``reviews_dir`` and always checks out in detached
    HEAD state at the exact ``head_sha``. That guarantees:

    - The reviewer's checkout can never alias a worker's worktree directory,
      even for the same branch (different parent dir, different key scheme).
    - The reviewer never has a branch checked out, so nothing it does (or a
      bug in a permissive command template) can land a commit on the PR's
      real branch — the checkout is read-only in spirit even before the
      caller's ``--permission-mode`` is applied.

    Idempotent: if a checkout already exists at this PR's keyed path (e.g.
    from a previous review round at a different head_sha), it is torn down
    and recreated fresh rather than reused or fast-forwarded in place.

    Raises ``ValueError`` if ``head_sha`` is empty (caller error — never a
    transient condition). Raises ``RuntimeError`` if the fetch or worktree-add
    fails.
    """
    if not head_sha:
        raise ValueError(
            f"create_review_checkout requires a non-empty head_sha for PR #{pr_number}"
        )
    # Validate head_sha format before it reaches git argv (issue #659).
    # head_sha is passed as a plain positional to ``git fetch origin`` and
    # ``git worktree add --detach``, so a flag-like value would be parsed as
    # an option without this guard.
    head_sha = require_valid_sha(head_sha, context="create_review_checkout head_sha")

    target_dir = reviews_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    checkout_path = target_dir / f"pr-{pr_number}"

    # Tear down any stale checkout at this path first (idempotent replace) —
    # never reuse/fast-forward a review checkout in place.
    remove_review_checkout(repo_root, pr_number, reviews_dir=target_dir)

    # Only fetch if an origin remote exists (mirrors create_worktree's own
    # guard). Pure-local repos (test fixtures) have no origin to fetch from —
    # the caller-supplied head_sha must already be reachable locally there.
    # Skip the network round-trip entirely when the commit is already local.
    if _has_origin_remote(repo_root) and not _commit_exists_locally(repo_root, head_sha):
        fetch_result = _run_remote_captured(
            ["git", "fetch", "origin", head_sha],
            cwd=repo_root,
        )
        if not fetch_result.ok:
            # Raw-SHA fetches are refused by the server when the SHA is not
            # currently advertised (e.g. read just before a push updated the
            # branch). The PR head ref is always advertised — fall back to it
            # so a transient race doesn't block the review checkout; the
            # worktree add below still pins the exact caller-supplied head_sha.
            _run_remote_captured(
                ["git", "fetch", "origin", f"refs/pull/{pr_number}/head"],
                cwd=repo_root,
            )
            if not _commit_exists_locally(repo_root, head_sha):
                raise RuntimeError(
                    f"[git fetch origin {head_sha}] Failed to fetch head_sha {head_sha!r} "
                    f"for PR #{pr_number} review checkout: "
                    f"{fetch_result.error or fetch_result.stderr}"
                )

    result = run_captured(
        ["git", "worktree", "add", "--detach", str(checkout_path), head_sha],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not result.ok:
        raise RuntimeError(
            f"git worktree add --detach failed for PR #{pr_number} review checkout at "
            f"{head_sha!r}: {result.error or result.stderr}"
        )

    return WorktreeInfo(path=checkout_path, branch=head_sha, venv_junction=None)


def remove_review_checkout(repo_root: Path, pr_number: int, *, reviews_dir: Path) -> bool:
    """Idempotent teardown of a PR's isolated review checkout.

    Returns True if the checkout was removed or was already absent; never
    raises. Safe to call speculatively (e.g. before creating a fresh checkout,
    or during a stale-claim/completed-verdict sweep that isn't sure whether a
    checkout exists for a given PR).
    """
    checkout_path = reviews_dir / f"pr-{pr_number}"

    existing_worktrees = list_worktrees(repo_root)
    is_registered = any(Path(wt["worktree"]) == checkout_path for wt in existing_worktrees)
    if not is_registered and not checkout_path.exists():
        return True

    return remove_worktree(repo_root, checkout_path, force=True, branch=None)


def inspect_worktree_state(
    worktree_path: Path,
    base_ref: str = "",
    injected_paths: tuple[str, ...] = (),
    materialize_dirs: tuple[str, ...] = (),
) -> WorktreeInspection:
    """Inspect a worker worktree after the process dies.

    Returns a ``WorktreeInspection`` describing whether the worker has:
    - completed work (clean working tree and commits ahead of the dispatch base)
    - partial work (uncommitted changes)
    - no commits (clean and at the base)
    - unknown (git probing failed)

    This is the single enforcement point for worktree inspection; it is used by
    both the workflow dead-session lane and reconcile.py drift detection.
    """
    # Issue #660: api workers set worktree_path="" (they do not use dedicated
    # worktrees). Path("") normalizes to Path("."), which would pass the
    # is_dir() check below and probe the *caller's* cwd with real git
    # merge-base/rev-list calls. If that cwd has local commits ahead of its
    # resolved default branch (a normal state for a live checkout), the
    # inspection returns COMPLETED, which forces the dead-session
    # classification lanes (workflow.py / reconcile.py) to "unpublished_work"
    # and skips the log-tail based provider-auth/quota/crash classification
    # entirely. Short-circuit an empty/unset path to UNKNOWN so those lanes
    # fall through to log-tail analysis instead.
    if worktree_path == Path(""):
        return WorktreeInspection(
            WorktreeState.UNKNOWN,
            error="worktree_path is empty (api workers have no dedicated worktree)",
        )

    if not worktree_path.is_dir():
        return WorktreeInspection(
            WorktreeState.UNKNOWN,
            error=f"worktree path does not exist: {worktree_path}",
        )

    try:
        dirty = _worker_authored_dirty(worktree_path, injected_paths, materialize_dirs)
    except WorktreeProbeFailedError as exc:
        return WorktreeInspection(
            WorktreeState.UNKNOWN,
            error=str(exc),
        )

    if base_ref == "":
        try:
            resolved_base_ref = _resolve_default_branch_ref(worktree_path)
        except RuntimeError as exc:
            return WorktreeInspection(
                WorktreeState.UNKNOWN,
                error=f"failed to resolve base ref: {exc}",
            )
    else:
        resolved_base_ref = base_ref

    merge_base_result = run_captured(
        ["git", "merge-base", resolved_base_ref, "HEAD"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not merge_base_result.ok:
        return WorktreeInspection(
            WorktreeState.UNKNOWN,
            error=merge_base_result.error or merge_base_result.stderr or "merge-base failed",
        )

    merge_base = merge_base_result.stdout.strip()
    rev_list_result = run_captured(
        ["git", "rev-list", "--count", f"{merge_base}..HEAD"],
        cwd=worktree_path,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not rev_list_result.ok:
        return WorktreeInspection(
            WorktreeState.UNKNOWN,
            error=rev_list_result.error or rev_list_result.stderr or "rev-list failed",
        )

    try:
        ahead_count = int(rev_list_result.stdout.strip())
    except ValueError:
        return WorktreeInspection(
            WorktreeState.UNKNOWN,
            error=f"invalid rev-list count: {rev_list_result.stdout!r}",
        )

    if ahead_count > 0 and not dirty:
        return WorktreeInspection(
            WorktreeState.COMPLETED,
            ahead_count=ahead_count,
            dirty=False,
            resolved_base_ref=resolved_base_ref,
        )
    if ahead_count > 0 and dirty:
        return WorktreeInspection(
            WorktreeState.PARTIAL,
            ahead_count=ahead_count,
            dirty=True,
            resolved_base_ref=resolved_base_ref,
        )
    if dirty:
        return WorktreeInspection(
            WorktreeState.PARTIAL,
            ahead_count=0,
            dirty=True,
            resolved_base_ref=resolved_base_ref,
        )
    return WorktreeInspection(
        WorktreeState.NO_COMMITS,
        ahead_count=0,
        dirty=False,
        resolved_base_ref=resolved_base_ref,
    )


def push_branch(
    repo_root: Path, branch: str, worktree_path: Path | None = None
) -> tuple[bool, str | None]:
    """Push ``branch`` to origin and verify via ``git ls-remote``.

    Returns ``(ok, error)``. Pushes can fail silently on some transports, so the
    remote branch tip is explicitly checked and compared to the local branch tip.
    """
    try:
        branch = require_valid_ref_name(branch, context="push_branch branch")
    except ValueError as exc:
        return False, str(exc)

    cwd = worktree_path if worktree_path else repo_root
    push_result = run_captured(
        ["git", "push", "origin", branch],
        cwd=cwd,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not push_result.ok:
        return False, push_result.error or push_result.stderr or "git push failed"

    ls_remote_result = _run_remote_captured(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=cwd,
    )
    if not ls_remote_result.ok:
        return False, ls_remote_result.error or ls_remote_result.stderr or "git ls-remote failed"

    remote_line = ls_remote_result.stdout.strip()
    if not remote_line:
        return False, f"remote branch {branch} not found after push"

    remote_sha = remote_line.split()[0]
    local_sha_result = run_captured(
        ["git", "rev-parse", "--verify", branch],
        cwd=cwd,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not local_sha_result.ok:
        return False, local_sha_result.error or local_sha_result.stderr or "git rev-parse failed"

    if local_sha_result.stdout.strip() == remote_sha:
        return True, None
    return False, f"remote branch {branch} does not match local tip after push"


def resolve_base_branch_name(repo_root: Path, base_ref: str) -> str:
    """Convert a base ref (e.g. ``origin/main`` or ``HEAD``) into a branch name.

    ``gh pr create --base`` expects a simple branch name. Remote-tracking refs
    are stripped to their local branch name; ``HEAD`` falls back to the current
    branch or ``main``.
    """
    if base_ref.startswith("refs/remotes/origin/"):
        return base_ref[len("refs/remotes/origin/") :]
    if base_ref.startswith("refs/heads/"):
        return base_ref[len("refs/heads/") :]
    if base_ref.startswith("origin/"):
        return base_ref[len("origin/") :]
    if base_ref == "HEAD":
        current_branch = run_captured(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if current_branch.ok and current_branch.stdout.strip():
            return current_branch.stdout.strip()
    return "main"


# Cap on commit subjects rendered into a salvage body. A runaway branch should
# not paste hundreds of lines into a PR description; the count is reported so
# the elision is visible rather than silent.
_SALVAGE_LOG_LIMIT = 20


def _resolve_salvage_branch_ref(repo_root: Path, safe_branch: str) -> str | None:
    """Return a ref for ``safe_branch`` that actually resolves in ``repo_root``.

    Tries the local branch first, then the ``origin``-tracking ref, and
    returns ``None`` if neither exists -- there is no evidence to report, and
    fabricating a range against a name git has never heard of is exactly the
    failure ``summarize_branch_work`` exists to avoid.
    """
    local = run_captured(
        ["git", "rev-parse", "--verify", "--quiet", f"{safe_branch}^{{commit}}"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if local.ok:
        return safe_branch

    remote_candidate = f"origin/{safe_branch}"
    remote = run_captured(
        ["git", "rev-parse", "--verify", "--quiet", f"{remote_candidate}^{{commit}}"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if remote.ok:
        return remote_candidate

    return None


def summarize_branch_work(
    repo_root: Path,
    branch: str,
    base_ref: str,
    *,
    test_path_globs: Sequence[str] = (),
) -> str:
    """Render the worker's own evidence about ``branch`` for a salvage PR body.

    A salvage PR is opened by the orchestrator, not by the worker that did the
    work, so there is no author-written body to carry the change's rationale.
    The janitor's body gate (``review.require_tests_or_rationale``) still
    applies to it, so a fixed boilerplate body can never pass: every salvage
    PR would fail a gate on text the orchestrator itself wrote.

    The honest input is the worker's own commit subjects plus the test files
    the branch touched, and this renders those verbatim rather than injecting
    the gate's keywords.

    Be precise about how strong that is, because it is easy to overclaim. The
    only case this still fails is a branch with **no commits** ahead of base:
    that returns ``""``, leaving boilerplate that cannot match, so the gate
    keeps a real failure mode. A branch that *does* have commits will pass,
    including one whose commits are all "wip" and which touched no tests --
    the ``## Tests`` heading alone satisfies the regex. That is accepted
    deliberately, not overlooked:

    - The body gate's job is "does this PR carry the author's rationale", and
      the full commit log is that rationale. For a salvage PR it is the only
      authored text that exists.
    - Whether a change carries *enough* tests is ``test_adequacy``'s job, and
      it already routes product-code-without-tests to rework and has a real
      exemption mechanism (``Test-exempt:``). Re-litigating that here would
      enforce one rule in two places and give the second copy no way to be
      exempted.

    ``test_path_globs`` should come from ``config.test_adequacy.test_path_globs``
    so test-file classification has one definition repo-wide (janitor reuses
    the same globs even when ``test_adequacy`` is disabled).

    Returns a markdown block, or ``""`` when the branch history cannot be read.
    A git failure is reported as *no evidence*, never as fabricated evidence.
    """
    try:
        safe_branch = require_valid_ref_name(branch, context="summarize_branch_work branch")
        safe_base = require_valid_rev(base_ref, context="summarize_branch_work base_ref")
    except ValueError:
        return ""

    # ``require_valid_ref_name`` checks NAME FORMAT only -- it never asks git
    # whether ``safe_branch`` actually resolves in ``repo_root``. The
    # orphaned-branch salvage lane (``_open_pr_for_orphaned_branch``) triggers
    # on evidence that creates no local ref at all -- a durable
    # ``reported_push`` record or an ``ahead_count`` from ``git ls-remote`` --
    # so ``branch`` routinely exists only as ``refs/remotes/origin/<branch>``.
    # Passing that bare name straight into the ranges below makes ``git log``
    # exit 128 and the whole summary come back "", falling through to the
    # boilerplate body this function exists to replace -- the same defect,
    # reached through the other operand. Resolve to whichever ref actually
    # exists before building the ranges, so both use the same resolved value.
    resolved_branch = _resolve_salvage_branch_ref(repo_root, safe_branch)
    if resolved_branch is None:
        return ""

    # Two dots for log, three for diff -- deliberately different operators.
    # ``git log A...B`` is the SYMMETRIC difference and would list commits made
    # on the base branch since it forked, attributing unrelated work to this
    # worker. ``A..B`` is "on B, not on A". ``git diff A...B`` conversely means
    # "changes B introduced since the merge-base", which is the one that
    # ignores base-branch drift.
    log_range = f"{safe_base}..{resolved_branch}"
    diff_range = f"{safe_base}...{resolved_branch}"

    log = run_captured(
        ["git", "log", "--no-merges", "--format=%s", log_range],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not log.ok:
        return ""

    subjects = [line.strip() for line in log.stdout.splitlines() if line.strip()]

    names = run_captured(
        ["git", "diff", "--name-only", diff_range],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    # ``None`` means "the diff never produced an answer", which is NOT the same
    # as "the diff produced an empty answer". Collapsing the two lets a failed
    # ``git diff`` render as "changed no test files (0 file(s) changed in
    # total)" -- a factual claim manufactured from a command that failed, which
    # is precisely the fabricated evidence this function's contract forbids.
    # ``git log`` and ``git diff`` genuinely diverge here: on a branch with no
    # merge base the log succeeds and the diff exits 128.
    changed: list[str] | None = (
        [line.strip() for line in names.stdout.splitlines() if line.strip()] if names.ok else None
    )
    test_files = (
        [name for name in changed if any(fnmatch.fnmatch(name, glob) for glob in test_path_globs)]
        if changed is not None
        else []
    )

    if not subjects:
        # The commit log IS the worker's rationale. With no commits there is
        # nothing honest to report, and any phrasing of "no tests were changed"
        # would still contain the gate's keywords -- passing a body that says
        # nothing. Return nothing instead, so the janitor gate keeps a real
        # failure mode and routes the PR to rework.
        return ""

    shown = subjects[:_SALVAGE_LOG_LIMIT]
    lines = "\n".join(f"- {subject}" for subject in shown)
    if len(subjects) > len(shown):
        lines += f"\n- ... and {len(subjects) - len(shown)} more commit(s)"
    sections = [f"## Worker's commit log\n\n{lines}"]

    if changed is None:
        # Report the absence of evidence as an absence, not as a zero. This
        # deliberately does NOT change whether the gate passes: the ``## Tests``
        # heading already satisfies the regex on its own, which the contract
        # above accepts on purpose for any branch that has commits. The defect
        # being fixed is the false factual claim, not the gate outcome.
        sections.append(
            "## Tests\n\nThe set of changed files could not be determined "
            "(`git diff` failed), so this PR asserts nothing about which files "
            "it touched. `test_adequacy` gates that separately."
        )
    elif test_files:
        listed = "\n".join(f"- `{name}`" for name in test_files)
        sections.append(
            f"## Tests\n\nThe branch changed {len(test_files)} test file(s):\n\n{listed}"
        )
    else:
        sections.append(
            f"## Tests\n\nThe branch changed no test files ({len(changed)} file(s) "
            "changed in total). `test_adequacy` gates that separately."
        )

    return "\n\n".join(sections)


def _list_worktrees_porcelain(repo_root: Path) -> tuple[list[dict], str | None]:
    """Parse ``git worktree list --porcelain`` into one dict per worktree.

    Returns a tuple of (worktrees, error_message). Invalid entries (missing
    required 'worktree' key or unknown flag keys) are dropped entirely - every
    returned dict is guaranteed to have a 'worktree' key with a Path value.
    Git command failures are returned as an error string instead of an empty
    list, so callers can distinguish "no worktrees" from "could not determine".
    """
    result = run_captured(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not result.ok:
        error = result.error or result.stderr or "git worktree list failed"
        return [], error

    worktrees: list[dict] = []
    current: dict = {}
    entry_malformed = False

    for line in result.stdout.splitlines():
        if not line.strip():
            # Entry boundary: flush current entry if valid
            if current and not entry_malformed and "worktree" in current:
                worktrees.append(current)
            current = {}
            entry_malformed = False
            continue

        if " " in line:
            key, _, value = line.partition(" ")
        else:
            # Space-less line: must be a known flag key
            key, value = line, True
            if key not in KNOWN_FLAG_KEYS:
                # Unknown flag key marks this entry as malformed
                entry_malformed = True

        if key == "worktree":
            # Worktree lines must have a path (str), not True
            if isinstance(value, str):
                current[key] = Path(value)
            else:
                # Malformed worktree line (bare "worktree" with no path)
                entry_malformed = True
        elif not entry_malformed:
            # Only add other keys if entry is not already malformed
            current[key] = value

    # Flush final entry if valid
    if current and not entry_malformed and "worktree" in current:
        worktrees.append(current)

    return worktrees, None


def list_worktrees(repo_root: Path) -> list[dict]:
    """Parse ``git worktree list --porcelain`` into one dict per worktree.

    Invalid entries (missing required 'worktree' key or unknown flag keys) are
    dropped entirely - every returned dict is guaranteed to have a 'worktree' key
    with a Path value. This makes all downstream consumers safe by construction.

    Git command failures return an empty list; callers that need to distinguish
    "empty" from "unknown" should use ``_list_worktrees_porcelain``.
    """
    worktrees, _ = _list_worktrees_porcelain(repo_root)
    return worktrees


def _site_packages_dir(venv_path: Path) -> Path | None:
    """Locate the ``site-packages`` directory inside a virtualenv.

    Tries Windows and POSIX layouts, then falls back to a recursive search.
    """
    candidates = [
        venv_path / "Lib" / "site-packages",
        venv_path / "lib" / "site-packages",
    ]
    lib_dir = venv_path / "lib"
    if lib_dir.is_dir():
        candidates.extend(lib_dir.glob("python*/site-packages"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    for found in venv_path.rglob("site-packages"):
        if found.is_dir():
            return found
    return None


def _top_level_package_names(repo_root: Path) -> frozenset[str]:
    """Return the top-level importable package names under ``repo_root/src``.

    Used to identify which ``.pth`` files in the shared venv belong to this
    project without hardcoding the project name.
    """
    src = repo_root / "src"
    if not src.is_dir():
        return frozenset()
    names: set[str] = set()
    for child in src.iterdir():
        if child.is_dir() and (child / "__init__.py").is_file():
            names.add(child.name)
        elif child.is_file() and child.suffix == ".py":
            names.add(child.stem)
    return frozenset(names)


def _venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _resolve_pth_line(site_packages: Path, line: str) -> Path:
    """Resolve a single line from a ``.pth`` file.

    Returns an empty path for executable/comment/empty lines. Relative paths
    are resolved against the ``site-packages`` directory containing the file.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("import"):
        return Path()
    p = Path(stripped)
    if p.is_absolute():
        return p.resolve()
    return (site_packages / p).resolve()


def _verify_shared_venv_by_import(repo_root: Path, venv_path: Path) -> tuple[bool, str]:
    """Fallback import-based verification of the shared venv."""
    python = _venv_python(venv_path)
    if not python.exists():
        return False, f"venv python not found at {python} (hint: uv sync --all-extras)"
    main_src = (repo_root / "src").resolve()
    script = "import charlie_work, pathlib; print(pathlib.Path(charlie_work.__file__).resolve())"
    result = run_captured([str(python), "-c", script], cwd=repo_root, timeout_seconds=60)
    if not result.ok:
        return False, (
            f"shared venv cannot import charlie_work: {result.error or result.stderr} "
            "(hint: uv sync --all-extras --reinstall-package charlie-work)"
        )
    try:
        imported = Path(result.stdout.strip()).resolve()
    except OSError:
        return False, (
            f"shared venv charlie_work __file__ is not a valid path: {result.stdout!r} "
            "(hint: uv sync --all-extras --reinstall-package charlie-work)"
        )
    if not contains(main_src, imported):
        return False, (
            f"shared venv imports charlie_work from {imported}, not main checkout {main_src} "
            "(hint: uv sync --all-extras --reinstall-package charlie-work)"
        )
    return True, "shared venv imports charlie_work from main checkout src"


def verify_shared_venv(repo_root: Path, venv_path: Path) -> tuple[bool, str]:
    """Verify the shared venv's editable ``.pth`` points to the main checkout src.

    Searches ``site-packages`` for ``.pth`` files whose names contain a top-level
    package name from ``repo_root/src``. For each matching ``.pth``, every path
    line must resolve to ``repo_root/src``; a path pointing anywhere else is the
    poisoned-editable-pth case and surfaces the ``--reinstall-package`` recovery
    hint.
    """
    site_packages = _site_packages_dir(venv_path)
    if not site_packages:
        return False, "could not locate site-packages in shared venv"
    main_src = (repo_root / "src").resolve()
    package_names = _top_level_package_names(repo_root)
    project_pth_files: list[Path] = []
    for pth in site_packages.glob("*.pth"):
        if any(name in pth.name for name in package_names):
            project_pth_files.append(pth)
    if not project_pth_files:
        return _verify_shared_venv_by_import(repo_root, venv_path)
    for pth in project_pth_files:
        content = pth.read_text(encoding="utf-8", errors="replace")
        for raw_line in content.splitlines():
            target = _resolve_pth_line(site_packages, raw_line)
            if target == Path() or target == main_src:
                continue
            return False, (
                f"editable .pth {pth.name} points outside main checkout: {target} "
                "(hint: uv sync --all-extras --reinstall-package charlie-work)"
            )
    return True, "shared venv editable .pth points to main checkout src"


def _find_linked_pr_number(
    issue_number: int, issue_state: dict[str, Any], state_prs: dict[str, Any]
) -> int | None:
    """Resolve the PR number linked to ``issue_number`` from state.json."""
    if issue_state.get("pr_number"):
        return int(issue_state["pr_number"])
    for pr_key, pr_entry in state_prs.items():
        if pr_entry.get("issue_number") == issue_number:
            return int(pr_key)
    return None


def _cleanup_live_writer_reason(issue_state: dict[str, Any], worktree_path: Path) -> str | None:
    """Return a reason when a merged-PR worktree is demonstrably still in use.

    The merged-PR cleanup lane asks a strictly narrower question than the
    redispatch lane's ``_probe_recovery_liveness``, and must NOT reuse it.

    By the time this runs, ``clean_worktrees`` has already established three
    things about the worktree: its PR is confirmed ``MERGED`` by a live
    ``gh pr view``, its working tree carries no worker-authored changes, and
    its local HEAD is identical to the merged PR's ``headRefOid``. The tree is
    therefore byte-identical to what already landed on the base branch — there
    is no work left to lose. The residual hazard is only ever *deleting a
    directory out from under a running process*, which is answered by positive
    evidence of a live process and never by the absence of a record.

    ``_probe_recovery_liveness`` answers a different question (may I reset a
    branch and relaunch a worker into this workspace?) and is correctly
    fail-closed on an inconclusive activity probe, because there a reset can
    destroy uncommitted work (issue #282). Reusing it here produced a
    permanent skip for every merged worktree:

      * its escape hatch is a deferral counter that only the recovery lane
        persists to state.json, so ``current_deferred_count`` is always 0 on
        this path and the ``max_inconclusive_probe_deferrals`` cap is
        unreachable by construction; and
      * it probes the Devin CLI's ``sessions.db``, which holds no records at
        all for claude-code/api workers, so a finished worker with no recorded
        pid yields ``devin_per_pid_log: no pid`` — the absence of a subject to
        look up, reported as an errored source and read as "maybe alive".

    Positive liveness signals, in order:
      1. the recorded worker pid is alive (start-time fingerprint matched);
      2. a live operator claim marker (liveness comes from
         ``operator_claimed_at`` in state, not the marker's sentinel pid);
      3. any writer marker whose pid is alive.

    Read-only by design: unlike ``_check_worktree_writer_marker`` it never
    deletes a stale marker, so a ``--dry-run`` preview stays write-free.
    """
    worker_pid = issue_state.get("worker_pid")
    process_start_time = issue_state.get("worker_process_start_time")
    if worker_pid is None:
        worker_pid = issue_state.get("last_known_worker_pid")
        process_start_time = issue_state.get("last_known_worker_process_start_time")
    try:
        worker_pid = int(worker_pid) if worker_pid is not None else None
    except (TypeError, ValueError):
        worker_pid = None
    if worker_pid is not None and is_pid_alive(worker_pid, process_start_time):
        return f"recorded worker pid {worker_pid} is alive"

    marker = read_worktree_marker(worktree_path)
    if marker is None:
        return None
    marker_pid = marker.get("pid")
    session_id = marker.get("session_id")
    if marker.get("kind") == OPERATOR_MARKER_KIND or (
        isinstance(session_id, str) and session_id.startswith("operator-")
    ):
        # Operator markers carry a sentinel pid; state.json is the authority.
        if _state.is_operator_claimed(issue_state):
            return "worktree is operator-claimed"
        return None
    if isinstance(marker_pid, int) and marker_pid > 0 and is_pid_alive(marker_pid, None):
        return f"live writer marker (pid={marker_pid}, session_id={session_id})"
    return None


@dataclass(frozen=True)
class WorktreeCleanResult:
    """Result of a ``clean_worktrees`` run.

    ``data`` carries ``planned``/``removed``/``skipped``/``failed`` (lists of
    per-worktree dicts, each ``skipped`` entry carrying a distinct ``reason``
    string), ``orphans`` (planned/removed/failed orphan dirs),
    ``venv_ok``/``venv_message``, ``attention_events``, and
    ``worktrees_registered``/``worktrees_out_of_scope`` (total worktrees git
    reports vs. how many were outside ``worktrees_dir`` or off the dispatch
    branch prefix and so never became candidates at all -- issue #1012).
    ``worktrees_out_of_scope`` has a floor of 1 in practice: the repo's own
    main checkout is itself a ``git worktree list`` entry and is always
    outside ``worktrees_dir``, so a bare count of 1 does not by itself mean
    an operator-created worktree was found.
    Kept as a dict rather than further nested dataclasses since callers
    (``CommandResult``) consume it as a JSON-able blob for CLI output.
    """

    ok: bool
    message: str
    data: dict[str, Any]


@runtime_checkable
class WorktreeCleanGH(Protocol):
    """Slice of :class:`GitHub` that ``clean_worktrees`` depends on.

    The cleanup lane only reads PR merge state via a single ``gh pr view``
    call, so it needs just ``run`` -- the rest of ``GitHub`` (list caches,
    mutating ops, retry config) is irrelevant here. Narrowing the parameter
    type to this protocol lets test doubles satisfy the contract structurally
    without subclassing the frozen ``GitHub`` dataclass, and documents exactly
    which GitHub surface the cleanup lane relies on (issue #641).
    """

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> Any: ...


def clean_worktrees(
    repo_root: Path,
    worktrees_dir: Path,
    state: dict[str, Any],
    config: OrchestratorConfig,
    gh: WorktreeCleanGH,
    *,
    dry_run: bool = False,
) -> WorktreeCleanResult:
    """Junction-safe cleanup of worker worktrees for merged or closed-unmerged PRs.

    Enumerates worktrees under ``worktrees_dir`` whose linked issue/PR resolves
    to a PR number in ``state.json``, then applies one of two eligibility
    checks per worktree, chosen by the PR's *terminal* state:

    **Merged path** (PR state ``MERGED``):

      1. The PR must be confirmed ``MERGED`` by a *live* ``gh pr view`` call.
         ``state.json`` claiming "merged" is corroboration only -- it is never
         sufficient on its own (state.json reliability history: #285/#309/
         #310), and an unavailable/erroring ``gh`` call fails CLOSED (skip)
         rather than falling back to trusting state.json.
      2. The worktree's working tree is clean (no uncommitted modifications).
      3. The worktree's local HEAD is *contained in* the merged PR's
         ``headRefOid`` (equal to it, or an ancestor of it).
      4. No live worker (see ``_cleanup_live_writer_reason``).

    **Closed-unmerged path** (PR state ``CLOSED`` with ``mergedAt`` null, from
    the same live ``gh pr view`` call -- no extra quota): a closed-and-never-
    merged PR is a decision, not a pending state, so waiting on it to become
    ``MERGED`` waits forever (issue #990). This path substitutes an
    "ordinary" nothing-would-be-lost check for the merged path's containment
    check (there is no merged head to contain against): the working tree is
    clean and the branch has no commits absent from its remote copy, both via
    ``_worktree_refuse_to_reset_reason``. Unlike a merge, closing a PR never
    deletes the remote branch, so this is not the squash-merge special case
    described below. Same liveness gate as the merged path.

    A PR that is neither confirmed ``MERGED`` nor confirmed closed-unmerged
    (open, unknown, or an erroring/ambiguous ``gh`` call) falls through to the
    fail-closed "PR not merged" skip -- still waiting is the correct default
    for a PR that might yet merge.

    The order matters and is load-bearing. Checks 2 and 3 together prove the
    tree is byte-identical to what already landed on the base branch, so
    nothing can be *lost* by removing it; that is what lets check 4 answer the
    narrow "is a process using this directory right now?" question with a
    positive-evidence probe instead of the redispatch lane's fail-closed one.
    Running liveness first (as this did until the reuse of
    ``_probe_recovery_liveness`` was removed) skipped every merged worktree
    forever — see ``_cleanup_live_writer_reason`` for the two independent
    reasons that probe can never clear on this path.

    Check 3 is deliberately a standalone comparison rather than a reuse of
    ``_worktree_refuse_to_reset_reason``'s remote-branch-ahead check. This
    repo's production ``AutoMergeConfig`` defaults to ``strategy="squash"``
    with ``delete_branch=True``, so after a real merge the remote branch is
    gone and the squash commit landed on the base branch is NOT an ancestor of
    the worker branch's commits. Reusing the remote-branch-ahead helper here
    would count every legitimately merged worktree's own work as "local
    commit(s) not on remote branch" and skip it forever (issue #286 rework,
    PR #340 review finding 1) -- local-ahead-of-a-deleted-remote-branch is the
    EXPECTED shape for a squash-merged worktree, not a sign of unpushed work.
    A worktree whose local HEAD does NOT match the merged PR's head (a stray
    post-merge commit) is skipped with a distinct reason instead of being
    silently folded into the same "unsafe" bucket.

    Eligible worktrees are removed with ``remove_worktree`` (junction-safe)
    and the local branch is deleted. After removals, the shared venv is
    checked for a poisoned editable ``.pth``.

    A final orphan sweep removes directories under ``worktrees_dir`` whose
    git admin record is gone but whose tree remains (the residue of
    ``git worktree remove`` failing on a reparse point).  Such leftovers are
    removed with the same reparse-point-safe rmtree and reported through
    ``data["orphans"]`` so they cannot accumulate silently (issue #462).
    """
    state_issues = state.get("issues", {})
    state_prs = state.get("prs", {})
    planned: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    orphans: dict[str, list[dict[str, Any]]] = {"planned": [], "removed": [], "failed": []}
    attention_events: list[dict[str, Any]] = []

    registered_worktrees, list_error = _list_worktrees_porcelain(repo_root)
    worktree_list_failed = list_error is not None
    # Worktrees git knows about that this sweep never even considers: outside
    # `worktrees_dir`, or on a branch that doesn't carry the dispatch prefix
    # (e.g. an operator-created worktree). These never reach `skipped` --
    # counting them separately is what lets a durable payload distinguish
    # "never a candidate" from "considered and skipped" (issue #1012).
    out_of_scope = 0
    for wt in registered_worktrees:
        wt_path = wt.get("worktree")
        if not isinstance(wt_path, Path) or not contains(worktrees_dir, wt_path):
            out_of_scope += 1
            continue
        raw_branch = str(wt.get("branch", ""))
        branch = raw_branch.removeprefix("refs/heads/")
        if not branch.startswith(config.dispatch.branch_prefix):
            out_of_scope += 1
            continue
        issue_number = linked_issue_number(
            {"headRefName": branch},
            is_cross_repository=False,
            branch_prefix=config.dispatch.branch_prefix,
        )
        if issue_number is None:
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "reason": "could not parse issue number from branch",
                }
            )
            continue
        issue_state = state_issues.get(str(issue_number), {})
        pr_number = _find_linked_pr_number(issue_number, issue_state, state_prs)
        if not pr_number:
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "reason": "no linked PR in state.json",
                }
            )
            continue
        pr_state = state_prs.get(str(pr_number), {})
        state_merged = bool(pr_state.get("status") == "merged" or pr_state.get("merged") is True)

        # `allow_failure=True` means this never raises (see GitHub.run): errors
        # come back as GitHubRunResult(ok=False, error=...), not exceptions.
        gh_result = gh.run(
            ["pr", "view", str(pr_number), "--json", PR_VIEW_MERGED_FIELDS],
            json_output=True,
            allow_failure=True,
        )
        gh_ok = isinstance(gh_result, GitHubRunResult) and gh_result.ok
        gh_merged = False
        gh_closed_unmerged = False
        merged_head_sha: str | None = None
        if gh_ok and isinstance(gh_result.value, dict):
            gh_pr_state = gh_result.value.get("state")
            gh_merged = gh_pr_state == "MERGED"
            merged_head_sha = gh_result.value.get("headRefOid")
            # Terminal, not pending: CLOSED-and-never-merged is a decision, not
            # a not-yet-merged state, so it must not collapse into the
            # "PR not merged" wait-forever branch below (issue #990). This is
            # still positive proof from the SAME live `gh pr view` call the
            # merged path already makes -- no extra quota -- and it is exactly
            # as fail-closed: an erroring/ambiguous `gh` call still falls
            # through to the generic "not merged" branch, which skips.
            gh_closed_unmerged = (
                gh_pr_state == "CLOSED" and gh_result.value.get("mergedAt") is None
            )

        if not gh_merged and not gh_closed_unmerged:
            # Fail-closed: only a live `gh pr view` MERGED or CLOSED-unmerged
            # confirmation may authorize destructive removal. state.json is
            # corroboration at best -- never sufficient on its own (state.json
            # reliability history: #285/#309/#310). An unavailable/erroring
            # `gh` call falls into this branch too and is DISTINGUISHED from a
            # confirmed-not-merged PR rather than falling back to trusting
            # state.json.
            if not gh_ok:
                gh_error = gh_result.error if isinstance(gh_result, GitHubRunResult) else "unknown"
                reason = f"gh pr view unavailable; cannot confirm merge status: {gh_error}"
            elif state_merged:
                reason = "state.json says merged but gh pr view did not confirm MERGED"
            else:
                reason = "PR not merged"
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": reason,
                }
            )
            continue

        if gh_merged:
            try:
                dirty_reason = _worktree_dirty_reason(
                    wt_path,
                    config.dispatch.injected_paths,
                    config.dispatch.materialize_dirs,
                )
            except WorktreeProbeFailedError as exc:
                skipped.append(
                    {
                        "worktree": str(wt_path),
                        "branch": branch,
                        "issue_number": issue_number,
                        "pr_number": pr_number,
                        "reason": f"worktree status probe failed: {exc}",
                    }
                )
                continue
            if dirty_reason:
                skipped.append(
                    {
                        "worktree": str(wt_path),
                        "branch": branch,
                        "issue_number": issue_number,
                        "pr_number": pr_number,
                        "reason": dirty_reason,
                    }
                )
                continue
            if not merged_head_sha:
                skipped.append(
                    {
                        "worktree": str(wt_path),
                        "branch": branch,
                        "issue_number": issue_number,
                        "pr_number": pr_number,
                        "reason": "gh pr view did not return headRefOid for the merged PR",
                    }
                )
                continue
            head_result = run_captured(
                ["git", "rev-parse", "--verify", "HEAD"],
                cwd=wt_path,
                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
            )
            if not head_result.ok or not head_result.stdout.strip():
                skipped.append(
                    {
                        "worktree": str(wt_path),
                        "branch": branch,
                        "issue_number": issue_number,
                        "pr_number": pr_number,
                        "reason": "could not resolve worktree HEAD",
                    }
                )
                continue
            local_head_sha = head_result.stdout.strip()
            if local_head_sha != merged_head_sha:
                # Containment, NOT equality. The question this gate exists to ask
                # is "does the worktree hold commits that did not get merged?",
                # and an equality test cannot tell the two directions apart:
                #
                #   local BEHIND merged  -> everything here is reachable from the
                #       merged head; nothing to lose. This is the ORDINARY shape,
                #       because the merge path advances the PR branch after the
                #       worker's last local commit (Aviator merge-queue rebases,
                #       merge-train updates, base-into-branch merges). 46 of 47
                #       mismatching worktrees on this host were this shape and were
                #       all reported as "stray post-merge commit(s)".
                #   local AHEAD/DIVERGED -> real unmerged work; refuse.
                #
                # This is the same class of error the note above records for
                # `_worktree_refuse_to_reset_reason`: a check whose shape counts
                # the expected post-merge topology as danger.
                #
                # Known limitation, deliberately left fail-closed: the containment
                # test needs `merged_head_sha` to still be in the local object
                # store. For a squash-merged PR whose remote branch was deleted,
                # nothing references that SHA once the local branch sits behind it,
                # so a `git gc` can prune it and the object-presence gate below
                # starts refusing. That is the safe direction (refuse, don't
                # remove), and it reports its own distinct reason string — if
                # worktree-clean ever "stops removing things" again, read the
                # reasons before re-deriving anything.
                if not _object_exists(repo_root, merged_head_sha):
                    skipped.append(
                        {
                            "worktree": str(wt_path),
                            "branch": branch,
                            "issue_number": issue_number,
                            "pr_number": pr_number,
                            "reason": (
                                f"merged PR head ({merged_head_sha[:8]}) is not present in the "
                                "local object store; cannot verify the worktree adds nothing "
                                "beyond it"
                            ),
                        }
                    )
                    continue
                if not _is_ancestor(repo_root, local_head_sha, merged_head_sha):
                    skipped.append(
                        {
                            "worktree": str(wt_path),
                            "branch": branch,
                            "issue_number": issue_number,
                            "pr_number": pr_number,
                            "reason": (
                                f"worktree HEAD ({local_head_sha[:8]}) is not contained in "
                                f"merged PR head ({merged_head_sha[:8]}); stray post-merge "
                                "commit(s)"
                            ),
                        }
                    )
                    continue
        elif gh_closed_unmerged:
            # A closed-and-never-merged PR is a terminal decision (issue
            # #990), not a pending state -- nothing will ever advance it to
            # MERGED, so the worktree cannot wait on that. There is no
            # "merged head" to compare against here (the PR never landed), so
            # eligibility is the ORDINARY "would removing this lose anything"
            # question instead of the merged path's special
            # squash-with-deleted-branch containment check: is the working
            # tree clean, and does the branch have any commits that are not
            # also on its remote copy? `_worktree_refuse_to_reset_reason`
            # already answers exactly that (dirty-tree check + remote-ahead
            # check, positive-proof throughout) for the redispatch lane, and a
            # closed-unmerged PR's branch is the ordinary case it was built
            # for -- closing without merging does not *automatically* delete
            # the remote branch the way GitHub's merge-time auto-delete does
            # (an operator or Aviator can still delete it by hand), so this is
            # not the squash-merge special case documented above; the helper's
            # remote-ahead check already refuses when the remote branch is
            # gone and the local tip carries commits beyond base. Reused as-is
            # rather than re-derived.
            try:
                closed_unsafe_reason = _worktree_refuse_to_reset_reason(
                    repo_root,
                    branch,
                    config.dispatch.base_ref,
                    wt_path,
                    config.dispatch.injected_paths,
                    config.dispatch.materialize_dirs,
                )
            except (WorktreeProbeFailedError, RuntimeError) as exc:
                skipped.append(
                    {
                        "worktree": str(wt_path),
                        "branch": branch,
                        "issue_number": issue_number,
                        "pr_number": pr_number,
                        "reason": f"closed-unmerged PR reclaim safety probe failed: {exc}",
                    }
                )
                continue
            if closed_unsafe_reason:
                skipped.append(
                    {
                        "worktree": str(wt_path),
                        "branch": branch,
                        "issue_number": issue_number,
                        "pr_number": pr_number,
                        "reason": f"closed-unmerged PR: {closed_unsafe_reason}",
                    }
                )
                continue
        else:
            # Unreachable today: the `not gh_merged and not gh_closed_unmerged`
            # guard above already continues for every state that is neither.
            # Kept as an explicit fail-closed branch rather than relying on
            # that guard alone, so a future third eligibility state added
            # above this `if` cannot silently fall through into either
            # removal path via a bare `else`.
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": "PR eligibility state not recognized (neither merged nor closed-unmerged)",
                }
            )
            continue
        # Liveness is checked LAST, and deliberately so: the gates above have
        # already proven this tree holds nothing that is not already merged
        # or already pushed, which is what reduces the question to "is a
        # process using this directory right now?" See
        # _cleanup_live_writer_reason for why the redispatch lane's
        # fail-closed probe is the wrong instrument here.
        live_reason = _cleanup_live_writer_reason(issue_state, wt_path)
        if live_reason:
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": f"live worker detected: {live_reason}",
                }
            )
            continue
        if dry_run:
            planned.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                }
            )
            continue
        if not remove_worktree(repo_root, wt_path, force=True, branch=branch):
            failed.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": "remove_worktree failed",
                }
            )
            continue
        removed.append(
            {
                "worktree": str(wt_path),
                "branch": branch,
                "issue_number": issue_number,
                "pr_number": pr_number,
            }
        )

    # Orphan sweep: directories under worktrees_dir with no git admin record.
    # This is the residue left when ``git worktree remove`` unregisters a
    # worktree but cannot delete its tree because of a reparse point.
    if worktree_list_failed:
        # A git worktree list failure must never be read as "zero worktrees".
        # Skip the sweep and surface the failure so live worker state is not
        # silently destroyed by a transient git hiccup.
        attention_events.append(
            {
                "type": "worktree_list_failed",
                "reason": list_error or "git worktree list failed",
            }
        )
    elif worktrees_dir.is_dir():
        registered_paths = {Path(wt["worktree"]) for wt in registered_worktrees}
        for child in worktrees_dir.iterdir():
            if not child.is_dir() or child in registered_paths or is_junction(child):
                continue
            if is_live_foreign_worktree(child, repo_root):
                # e.g. the sibling ci_runners checkout worker shims provision
                # for the ci-fleet editable — another repo's live worktree,
                # not this repo's residue. Never sweep it.
                continue
            if dry_run:
                orphans["planned"].append({"worktree": str(child)})
                continue
            if _robust_rmtree(child):
                orphans["removed"].append({"worktree": str(child)})
            else:
                reason = "orphan directory removal failed"
                orphans["failed"].append({"worktree": str(child), "reason": reason})
                attention_events.append(
                    {
                        "type": "worktree_orphan_removal_failed",
                        "worktree": str(child),
                        "reason": reason,
                    }
                )

    venv_source = config.devin.venv_source or config.claude_code.venv_source
    venv_ok = True
    venv_message = "no shared venv configured; pth verification skipped"
    if venv_source:
        venv_path = repo_root / venv_source
        if venv_path.is_dir():
            venv_ok, venv_message = verify_shared_venv(repo_root, venv_path)
        else:
            venv_ok = False
            venv_message = f"shared venv not found: {venv_path}"

    # Surface regular removal failures as attention events too.
    for failure in failed:
        attention_events.append(
            {
                "type": "worktree_removal_failed",
                "worktree": failure["worktree"],
                "reason": failure.get("reason", "remove_worktree failed"),
            }
        )

    data = {
        "planned": planned,
        "removed": removed,
        "skipped": skipped,
        "failed": failed,
        "orphans": orphans,
        "venv_ok": venv_ok,
        "venv_message": venv_message,
        "attention_events": attention_events,
        "worktrees_registered": len(registered_worktrees),
        "worktrees_out_of_scope": out_of_scope,
    }
    ok = not failed and not orphans["failed"] and venv_ok and not worktree_list_failed
    if dry_run:
        message = (
            f"worktree-clean (dry-run): {len(planned)} eligible, {len(skipped)} skipped, "
            f"{len(orphans['planned'])} orphan(s)"
        )
    else:
        message = (
            f"worktree-clean: {len(removed)} removed, {len(skipped)} skipped, "
            f"{len(failed)} failed, {len(orphans['removed'])} orphan(s)"
        )
        if orphans["failed"]:
            message = f"{message}, {len(orphans['failed'])} orphan removal(s) failed"
    if worktree_list_failed:
        message = f"{message}; could not list worktrees: {list_error}"
    if not venv_ok:
        message = f"{message}; {venv_message}"
    return WorktreeCleanResult(ok=ok, message=message, data=data)
