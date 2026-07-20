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

import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from .attempt_refs import AttemptSnapshot, snapshot_attempt_ref
from .config import OrchestratorConfig, WRITER_MARKER_FILENAME
from .github import GitHub, GitHubRunResult, PR_VIEW_MERGED_FIELDS, linked_issue_number
from .janitor import _calculate_patch_id
from .paths import runtime_paths
from .post_mortem import real_activity_for_worker
from .process_utils import is_pid_alive
from .subprocess_runner import run_captured
from . import state as _state

_DEFAULT_TIMEOUT_SECONDS = 60

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
    (e.g. an operator's editor or an out-of-band agent). The launch shim
    surfaces this as ``failure_kind="worktree_foreign_writer"`` so the issue
    stays queued and the dispatch event log records the conflict.
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
        super().__init__(
            f"worktree {worktree_path} has a live foreign writer "
            f"(pid={pid}, session_id={session_id})"
        )


# Known porcelain flag keys that may appear as space-less lines (value=True)
# These are the only keys that map to True in git worktree --porcelain output
KNOWN_FLAG_KEYS = frozenset({"bare", "detached", "locked", "prunable"})


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
    return repo_root / ".var" / "charlie-work" / "worktrees"


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
    run_captured(
        ["git", "remote", "set-head", "origin", "--auto"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    resolved = _read_origin_head_symref(repo_root)
    if resolved is not None:
        return resolved

    raise RuntimeError(
        "Cannot resolve origin's default branch: refs/remotes/origin/HEAD is "
        "unset and 'git remote set-head origin --auto' did not heal it. "
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
    """
    result = run_captured(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not result.ok:
        # Probe failed (network error, auth failure, etc.)
        return None
    if result.stdout.strip():
        # Branch exists
        return True
    # Branch does not exist (exit 0 with empty stdout)
    return False


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    """Return True if ``ancestor`` is an ancestor of ``descendant``."""
    result = run_captured(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    return result.ok


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


def _worker_authored_dirty(
    worktree_path: Path,
    injected_paths: tuple[str, ...] = (),
) -> bool:
    """Return True if the worktree has uncommitted changes that are NOT
    orchestrator-injected prompt files.

    ``injected_paths`` are worktree-relative paths (files or directories) that
    the orchestrator writes into the worktree (e.g. the Claude Code prompt
    file, or a custom ``.devin/prompts/...`` convention that the operator has
    configured). They are excluded from the dirty check so completed worker
    work is not stranded by prompt-injection noise (issue #381).

    Uses ``git status --porcelain=v2 -z --untracked-files=all``:
    ``--untracked-files=all`` disables directory collapsing, so every
    untracked file gets its own record — there is no "does this line
    represent one file or a whole directory" ambiguity left to special-case
    (the v1 rework history: a wholly-untracked directory collapsed to a
    single ``?? dir/`` line could hide a worker-authored sibling file next to
    a configured injected path or directory, and a probe re-scoped to that
    directory was needed to tell the two apart). Every entry — whether it
    matches an injected file or a whole injected directory — is now checked
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

    # Normalize the configured side too, so a Windows-style backslash override
    # still matches git's forward-slash path reporting.
    injected = [PurePosixPath(str(p).replace("\\", "/")) for p in injected_paths]
    for raw_path in _parse_status_v2_paths(status_result.stdout):
        # Git may emit backslashes on Windows; normalize for comparison.
        path = PurePosixPath(str(raw_path).replace("\\", "/"))
        if any(
            path == injected_path or injected_path in path.parents for injected_path in injected
        ):
            continue
        return True
    return False


def _worktree_refuse_to_reset_reason(
    repo_root: Path,
    branch: str,
    base_ref: str,
    worktree_path: Path | None = None,
    injected_paths: tuple[str, ...] = (),
) -> str | None:
    """Return a human-readable reason if resetting the branch/worktree would destroy
    local work, otherwise ``None``.

    Checks for:
      - uncommitted modifications in ``worktree_path`` (if it exists)
      - local commits that are not present on the remote branch (``git ls-remote``
        comparison, falling back to the merge-base with ``base_ref`` when the remote
        branch does not exist)

    This is read-only: it never commits, fetches, or resets.

    Raises:
        WorktreeProbeFailedError: if the ``git status --porcelain`` probe itself
            fails (index lock, corruption, permissions). The reset is still
            refused (we cannot confirm the worktree is clean), but this is
            classified distinctly from a confirmed-dirty worktree — see the
            class docstring for why callers must not conflate the two under
            the same ``failure_kind``.
    """
    # Uncommitted modifications are only meaningful when the worktree directory exists.
    if worktree_path is not None and worktree_path.is_dir():
        if _worker_authored_dirty(worktree_path, injected_paths):
            return "worktree has uncommitted modifications"
        local_tip_result = run_captured(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=worktree_path,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
    else:
        local_tip_result = run_captured(
            ["git", "rev-parse", "--verify", branch],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )

    if not local_tip_result.ok or not local_tip_result.stdout.strip():
        # No branch or no commit; nothing to lose.
        return None

    local_sha = local_tip_result.stdout.strip()

    # Compare against the remote branch via git ls-remote.
    remote_sha: str | None = None
    if _has_origin_remote(repo_root):
        ls_remote_result = run_captured(
            ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if ls_remote_result.ok and ls_remote_result.stdout.strip():
            remote_sha = ls_remote_result.stdout.strip().split()[0]

    if remote_sha is None:
        # Branch does not exist on origin (or the probe failed). Any commits beyond
        # the base ref are unpushed and must not be discarded.
        base = base_ref if base_ref else _resolve_default_branch_ref(repo_root)
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
    if _worker_authored_dirty(worktree_path, injected_paths):
        return "worktree has uncommitted modifications"
    return None


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


def _materialize_directory(repo_root: Path, worktree_path: Path, dir_path: str) -> tuple[str, ...]:
    """Copy files from ``repo_root / dir_path`` into the worktree.

    Files are copied file-by-file; existing target files are only overwritten
    when the source content differs. Any written path that maps to a tracked
    file in the worktree is marked ``assume-unchanged`` so that orchestrator-
    injected content (e.g. per-dispatch ``.devin/prompts/worker.md``) does not
    appear as working-tree dirt in ``git status`` or CI-parity clean-tree
    gates.

    Returns a tuple of worktree-relative paths (forward-slash normalized) that
    were written, derived from the materializer's own manifest.
    """
    source = repo_root / dir_path
    if not source.exists():
        return ()
    if not source.is_relative_to(repo_root):
        return ()

    if source.is_file():
        source_files = [source]
        source_root = source.parent
    else:
        source_files = [p for p in source.rglob("*") if p.is_file()]
        source_root = source

    written: list[str] = []
    for src_file in source_files:
        if ".git" in src_file.parts:
            continue

        if source.is_file():
            rel = Path(".")
        else:
            rel = src_file.relative_to(source_root)
        target_file = (worktree_path / dir_path / rel).resolve()
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
        rel_worktree = (Path(dir_path) / rel).as_posix()
        written.append(rel_worktree)

    # Mark any tracked target paths as assume-unchanged so that orchestrator-
    # injected per-dispatch content does not surface as tracked-file dirt.
    tracked_written = [p for p in written if _is_git_tracked(worktree_path, worktree_path / p)]
    if tracked_written:
        assume_result = run_captured(
            ["git", "update-index", "--assume-unchanged", "--", *tracked_written],
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

    if worker_pid is not None and is_pid_alive(worker_pid, worker_process_start_time):
        raise LiveWorkerRedispatchError(
            issue_number=issue_number,
            pid=worker_pid,
            process_start_time=worker_process_start_time,
            probe_result="pid_alive",
            inconclusive_probe_deferred_count=0,
        )

    # For devin-shell sessions, the real-activity probe (sessions.db +
    # per-PID Devin log) is the source of truth even when the wrapper PID is
    # gone or has been recycled.
    if resolved_config.devin.adapter == "devin-shell":
        started_at = recovery.get("started_at") or recovery.get("dispatched_at") or ""
        pm_config = resolved_config.post_mortem
        now = datetime.now(UTC)
        try:
            probe = real_activity_for_worker(
                pm_config,
                str(worktree_path),
                started_at,
                worker_pid,
                now,
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

            new_deferred_count = current_deferred_count + 1
            raise LiveWorkerRedispatchError(
                issue_number=issue_number,
                pid=worker_pid,
                process_start_time=worker_process_start_time,
                probe_result="probe_error",
                inconclusive_probe_deferred_count=new_deferred_count,
            )

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
    # Resolve base_ref: empty string means auto-resolve to origin/<default>
    resolved_base_ref = base_ref
    if base_ref == "":
        resolved_base_ref = _resolve_default_branch_ref(repo_root)

    # Fetch if the resolved base_ref is a remote-tracking ref (origin/<branch>)
    # Only do this for fresh dispatch (not rework/recovery) to avoid moving existing tips
    if not rework and recovery is None and resolved_base_ref.startswith("origin/"):
        # Extract the branch name from "origin/<branch>"
        remote_branch = resolved_base_ref[len("origin/") :]
        fetch_result = run_captured(
            ["git", "fetch", "origin", remote_branch],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not fetch_result.ok:
            raise RuntimeError(
                f"Failed to fetch base ref {resolved_base_ref!r} before worktree creation: "
                f"{fetch_result.error or fetch_result.stderr}"
            )

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

    def _raise_if_unsafe_to_reset(target_path: Path | None = None) -> None:
        """Hard-refuse to reset if the worktree/branch contains local work."""
        check_path = target_path or worktree_path
        reason = _worktree_refuse_to_reset_reason(
            repo_root, branch, resolved_base_ref, check_path, injected_paths
        )
        if reason:
            raise WorktreeUnsafeError(reason)

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
                    has_dirty = _worker_authored_dirty(wt_path, injected_paths)
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
            # Reuse existing worktree: fetch and fast-forward to origin tip
            worktree_path = Path(existing_wt["worktree"])
            # Only fetch if origin remote exists (deterministic check)
            if _has_origin_remote(repo_root):
                # Fetch the remote-tracking ref only (branch:<branch> fails when branch is checked out)
                fetch_result = run_captured(
                    ["git", "fetch", "origin", branch],
                    cwd=repo_root,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
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
                        dirty_reason = _worktree_dirty_reason(worktree_path, injected_paths)
                        if dirty_reason:
                            raise WorktreeUnsafeError(dirty_reason)
                        _snapshot_before_delete(branch)
                        base_branch = (
                            resolved_base_ref[len("origin/") :]
                            if resolved_base_ref.startswith("origin/")
                            else None
                        )
                        if base_branch and base_branch != branch:
                            run_captured(
                                ["git", "fetch", "origin", base_branch],
                                cwd=repo_root,
                                timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
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
                        f"Fetch failed for rework branch {branch!r}: "
                        f"{fetch_result.error or fetch_result.stderr}"
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
            )
        else:
            # No existing worktree: attach to existing branch (no -b flag)
            # Fetch first to ensure we materialize at the origin tip, but only if origin exists
            if _has_origin_remote(repo_root):
                fetch_result = run_captured(
                    ["git", "fetch", "origin", branch],
                    cwd=repo_root,
                    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
                )
                # If fetch failed with origin present, raise (real network/error failure)
                if not fetch_result.ok:
                    raise RuntimeError(
                        f"Fetch failed for rework branch {branch!r}: "
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
                        run_captured(
                            ["git", "fetch", "origin", base_branch],
                            cwd=repo_root,
                            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
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
    else:
        # Fresh dispatch: create new branch off base_ref
        # Issue #110: Stale worktree reclamation before git worktree add
        existing_worktrees = list_worktrees(repo_root)
        existing_wt = next(
            (wt for wt in existing_worktrees if Path(wt["worktree"]) == worktree_path),
            None,
        )
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
            elif wt_path.exists():
                # Refuse to reset a worktree that still contains local work.
                # Partial/dirty worktrees are the redispatch case, not the fresh
                # dispatch case (issue #257).
                _raise_if_unsafe_to_reset(wt_path)
                # Directory is clean and has no local commits: remove and recreate
                if not remove_worktree(repo_root, wt_path, force=True):
                    raise RuntimeError(
                        f"Failed to remove stale worktree {wt_path} for fresh dispatch"
                    )
                reclaimed = "pruned"

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
                    f"Failed to delete branch {branch!r} for fresh dispatch: "
                    f"{branch_delete_result.error or branch_delete_result.stderr}"
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


def _default_reviews_dir(repo_root: Path) -> Path:
    return repo_root / ".var" / "charlie-work" / "dispatches" / "reviews"


def create_review_checkout(
    repo_root: Path,
    pr_number: int,
    head_sha: str,
    *,
    reviews_dir: Path | None = None,
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

    target_dir = reviews_dir or _default_reviews_dir(repo_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    checkout_path = target_dir / f"pr-{pr_number}"

    # Tear down any stale checkout at this path first (idempotent replace) —
    # never reuse/fast-forward a review checkout in place.
    remove_review_checkout(repo_root, pr_number, reviews_dir=target_dir)

    # Only fetch if an origin remote exists (mirrors create_worktree's own
    # guard). Pure-local repos (test fixtures) have no origin to fetch from —
    # the caller-supplied head_sha must already be reachable locally there.
    if _has_origin_remote(repo_root):
        fetch_result = run_captured(
            ["git", "fetch", "origin", head_sha],
            cwd=repo_root,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )
        if not fetch_result.ok:
            raise RuntimeError(
                f"Failed to fetch head_sha {head_sha!r} for PR #{pr_number} review checkout: "
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


def remove_review_checkout(
    repo_root: Path, pr_number: int, *, reviews_dir: Path | None = None
) -> bool:
    """Idempotent teardown of a PR's isolated review checkout.

    Returns True if the checkout was removed or was already absent; never
    raises. Safe to call speculatively (e.g. before creating a fresh checkout,
    or during a stale-claim/completed-verdict sweep that isn't sure whether a
    checkout exists for a given PR).
    """
    target_dir = reviews_dir or _default_reviews_dir(repo_root)
    checkout_path = target_dir / f"pr-{pr_number}"

    existing_worktrees = list_worktrees(repo_root)
    is_registered = any(Path(wt["worktree"]) == checkout_path for wt in existing_worktrees)
    if not is_registered and not checkout_path.exists():
        return True

    return remove_worktree(repo_root, checkout_path, force=True, branch=None)


def inspect_worktree_state(
    worktree_path: Path,
    base_ref: str = "",
    injected_paths: tuple[str, ...] = (),
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
    if not worktree_path.is_dir():
        return WorktreeInspection(
            WorktreeState.UNKNOWN,
            error=f"worktree path does not exist: {worktree_path}",
        )

    try:
        dirty = _worker_authored_dirty(worktree_path, injected_paths)
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
    cwd = worktree_path if worktree_path else repo_root
    push_result = run_captured(
        ["git", "push", "origin", branch],
        cwd=cwd,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not push_result.ok:
        return False, push_result.error or push_result.stderr or "git push failed"

    ls_remote_result = run_captured(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=cwd,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
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
    if not imported.is_relative_to(main_src):
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


@dataclass(frozen=True)
class WorktreeCleanResult:
    """Result of a ``clean_worktrees`` run.

    ``data`` carries ``planned``/``removed``/``skipped``/``failed`` (lists of
    per-worktree dicts), ``orphans`` (planned/removed/failed orphan dirs),
    ``venv_ok``/``venv_message``, and ``attention_events``. Kept as a dict
    rather than further nested dataclasses since callers (``CommandResult``)
    consume it as a JSON-able blob for CLI output.
    """

    ok: bool
    message: str
    data: dict[str, Any]


def clean_worktrees(
    repo_root: Path,
    worktrees_dir: Path,
    state: dict[str, Any],
    config: OrchestratorConfig,
    gh: GitHub,
    *,
    dry_run: bool = False,
) -> WorktreeCleanResult:
    """Junction-safe cleanup of worker worktrees for merged PRs.

    Enumerates worktrees under ``worktrees_dir`` whose linked issue/PR resolves
    to a PR number in ``state.json``, then applies a merged-PR-aware
    eligibility check per worktree:

      1. The PR must be confirmed ``MERGED`` by a *live* ``gh pr view`` call.
         ``state.json`` claiming "merged" is corroboration only -- it is never
         sufficient on its own (state.json reliability history: #285/#309/
         #310), and an unavailable/erroring ``gh`` call fails CLOSED (skip)
         rather than falling back to trusting state.json.
      2. No live worker (see ``_probe_recovery_liveness``).
      3. The worktree's working tree is clean (no uncommitted modifications).
      4. The worktree's local HEAD equals the merged PR's ``headRefOid``.

    Check 4 is deliberately a standalone comparison rather than a reuse of
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
    for wt in registered_worktrees:
        wt_path = wt.get("worktree")
        if not isinstance(wt_path, Path) or not wt_path.is_relative_to(worktrees_dir):
            continue
        raw_branch = str(wt.get("branch", ""))
        branch = raw_branch.removeprefix("refs/heads/")
        if not branch.startswith(config.dispatch.branch_prefix):
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
        merged_head_sha: str | None = None
        if gh_ok and isinstance(gh_result.value, dict):
            gh_merged = gh_result.value.get("state") == "MERGED"
            merged_head_sha = gh_result.value.get("headRefOid")

        if not gh_merged:
            # Fail-closed: only a live `gh pr view` MERGED confirmation may
            # authorize destructive removal. state.json is corroboration at
            # best -- never sufficient on its own (state.json reliability
            # history: #285/#309/#310). An unavailable/erroring `gh` call
            # falls into this branch too and is DISTINGUISHED from a
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
        try:
            _probe_recovery_liveness(issue_state, wt_path, config, issue_number)
        except LiveWorkerRedispatchError as exc:
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": f"live worker detected: {exc.probe_result}",
                }
            )
            continue
        try:
            dirty_reason = _worktree_dirty_reason(wt_path, config.dispatch.injected_paths)
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
            skipped.append(
                {
                    "worktree": str(wt_path),
                    "branch": branch,
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "reason": (
                        f"worktree HEAD ({local_head_sha[:8]}) does not match merged PR head "
                        f"({merged_head_sha[:8]}); stray post-merge commit(s)"
                    ),
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
