"""Worker post-mortem extraction from the Devin CLI's local session store.

Motivation (issue #261, extends #260): when the stall/dead-worker reaper in
``workflow.py`` kills a worker, the only diagnostic signal it has today is a
regex grep over the last 2KB of the worker's own log tail
(``devin_shell._classify_session_failure`` / the claude-code twin). That
misses the single most useful failure mode: a worker blocked by a
``.devin`` push-gate hook (``decision: block``) looks identical, from the
log tail alone, to a generic stall — so it gets hot-redispatched straight
into the same hook, and its unpushed local commits get wiped by the very
next ``git branch -D`` (see ``attempt_refs.py``).

This module reads the Devin CLI's own session store
(``%APPDATA%\\devin\\cli\\sessions.db``, a SQLite database — see
``docs/design/extraction-dossier.md`` §4) to recover the worker's terminal
tool call directly, independent of what made it into the log tail. It is a
strictly best-effort, read-only side channel:

- The database is opened read-only (SQLite URI ``mode=ro&immutable=1``); if
  that fails (locked by the live ``devin`` CLI process, missing, or the
  schema has drifted from what this module expects), extraction degrades to
  a recorded ``extraction_error`` and this module never raises — the
  reaper's existing log-tail classification always still runs as the
  fallback, unaffected.
- ``message_nodes`` schema (role/content/created_at columns keyed by
  session_id) is inferred from the block-payload behavior described in the
  issue, not from official Devin CLI documentation (none exists for this
  table — see extraction-dossier.md item 23). Any mismatch is caught by
  ``sqlite3.Error`` and treated as schema drift, not a crash.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import OrchestratorConfig, SignatureRule

if TYPE_CHECKING:
    from .attempt_refs import AttemptSnapshot
    from .worker import WorkerView

# Content prefix marking a role="tool" message_nodes row as a blocked tool
# call (issue #261 spec: "block payload is JSON embedded in a role=tool
# node's content string prefixed with 'Tool blocked:'").
_BLOCK_CONTENT_PREFIX = "Tool blocked:"


def _default_db_path() -> Path:
    """Resolve the Devin CLI's session store default location.

    Windows: ``%APPDATA%\\devin\\cli\\sessions.db``. POSIX:
    ``~/.local/share/devin/cli/sessions.db`` (extraction-dossier.md §4).
    Env-expanded at call time, never hardcoded to a literal user path.
    """
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "devin" / "cli" / "sessions.db"
    return Path.home() / ".local" / "share" / "devin" / "cli" / "sessions.db"


def _resolve_db_path(configured: str) -> Path:
    if configured:
        return Path(os.path.expandvars(os.path.expanduser(configured)))
    return _default_db_path()


@dataclass(frozen=True)
class MessageNode:
    role: str
    content: str
    created_at: str | None = None


@dataclass(frozen=True)
class PostMortemRecord:
    """Result of a post-mortem extraction attempt for one dead worker.

    ``matched``/``extraction_error`` are mutually informative, not mutually
    exclusive with a partial result: a DB that opened fine but had no
    matching session still sets ``matched=False`` with a human-readable
    ``extraction_error`` describing why, so ``doctor`` can distinguish "no
    signal available" from "this worker was healthy."
    """

    issue_number: int
    generated_at: str
    db_path: str
    matched: bool
    session_id: str | None = None
    extraction_error: str | None = None
    terminal_tool: str | None = None
    terminal_reason: str | None = None
    failure_kind: str | None = None  # "worker_blocked" | None
    message_nodes: tuple[MessageNode, ...] = ()
    attempt_ref: str | None = None
    attempt_ahead_of_main: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> PostMortemRecord:
        nodes = payload.get("message_nodes") or []
        return PostMortemRecord(
            issue_number=int(payload["issue_number"]),
            generated_at=str(payload.get("generated_at", "")),
            db_path=str(payload.get("db_path", "")),
            matched=bool(payload.get("matched", False)),
            session_id=payload.get("session_id"),
            extraction_error=payload.get("extraction_error"),
            terminal_tool=payload.get("terminal_tool"),
            terminal_reason=payload.get("terminal_reason"),
            failure_kind=payload.get("failure_kind"),
            message_nodes=tuple(
                MessageNode(
                    role=str(n.get("role", "")),
                    content=str(n.get("content", "")),
                    created_at=n.get("created_at"),
                )
                for n in nodes
                if isinstance(n, dict)
            ),
            attempt_ref=payload.get("attempt_ref"),
            attempt_ahead_of_main=payload.get("attempt_ahead_of_main"),
        )


def _sidecar_path(sessions_dir: Path, issue_number: int) -> Path:
    return sessions_dir / f"issue-{issue_number}.post-mortem.json"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def read_post_mortem(sessions_dir: Path, issue_number: int) -> PostMortemRecord | None:
    """Read a previously-written post-mortem sidecar, or None if absent/unreadable.

    Never raises — a corrupt sidecar must not take down doctor/digest reporting.
    """
    path = _sidecar_path(sessions_dir, issue_number)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return PostMortemRecord.from_dict(payload)
    except (KeyError, ValueError, TypeError):
        return None


def _open_readonly(db_path: Path) -> tuple[sqlite3.Connection | None, Path | None, str | None]:
    """Open ``db_path`` read-only without taking a lock that could contend
    with the live Devin CLI process.

    Returns ``(connection, temp_copy_path_or_None, error)``. On success,
    ``temp_copy_path`` is set only when the copy-to-temp fallback was used
    (caller must clean it up). Never raises.
    """
    if not db_path.exists():
        return None, None, f"sessions.db not found at {db_path}"

    uri = f"file:{db_path.as_posix()}?mode=ro&immutable=1"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        # immutable=1 does not itself force I/O — probe with a trivial query
        # so a genuinely locked/corrupt file fails here, not on first real query.
        conn.execute("SELECT 1").fetchone()
        return conn, None, None
    except sqlite3.Error:
        pass

    # Fallback: the live CLI holds an exclusive/reserved lock even against
    # our read-only URI connection. Copy to a private temp file instead —
    # a stale-by-milliseconds snapshot is an acceptable tradeoff for a
    # best-effort diagnostic that must never block or fail the reaper.
    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="charlie-work-postmortem-"))
        tmp_copy = tmp_dir / "sessions.db"
        shutil.copy2(db_path, tmp_copy)
        conn = sqlite3.connect(f"file:{tmp_copy.as_posix()}?mode=ro", uri=True, timeout=5)
        conn.execute("SELECT 1").fetchone()
        return conn, tmp_copy, None
    except (OSError, sqlite3.Error) as exc:
        return None, None, f"failed to open sessions.db (locked or corrupt): {exc}"


def _cleanup_temp_copy(temp_copy_path: Path | None) -> None:
    if temp_copy_path is None:
        return
    try:
        temp_copy_path.unlink(missing_ok=True)
        temp_copy_path.parent.rmdir()
    except OSError:
        # Best-effort — a leaked temp dir is not worth failing the pass over.
        pass


def _find_matching_session(
    conn: sqlite3.Connection,
    working_directory: str,
    window_start: datetime,
    window_end: datetime,
) -> tuple[str | None, str | None]:
    """Find the session whose working_directory matches this worktree and
    whose created_at falls within [window_start, window_end].

    Returns (session_id, error). Multiple matches pick the most recent
    created_at (a stale prior session in the same directory is possible if
    a worktree was reused across attempts before this window was narrowed).
    """
    try:
        cursor = conn.execute(
            "SELECT id FROM sessions "
            "WHERE working_directory = ? AND created_at >= ? AND created_at <= ? "
            "ORDER BY created_at DESC LIMIT 1",
            (
                working_directory,
                window_start.isoformat(),
                window_end.isoformat(),
            ),
        )
        row = cursor.fetchone()
    except sqlite3.Error as exc:
        return None, f"sessions table query failed (schema drift?): {exc}"
    if row is None:
        return None, "no session found matching working_directory within the time window"
    return str(row[0]), None


def _extract_last_n_nodes(
    conn: sqlite3.Connection, session_id: str, limit: int
) -> tuple[tuple[MessageNode, ...], str | None]:
    """Return the last ``limit`` message_nodes for ``session_id`` in chronological order."""
    try:
        cursor = conn.execute(
            "SELECT role, content, created_at FROM message_nodes "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cursor.fetchall()
    except sqlite3.Error as exc:
        return (), f"message_nodes table query failed (schema drift?): {exc}"
    nodes = tuple(
        MessageNode(role=str(r[0]), content=str(r[1]), created_at=r[2]) for r in reversed(rows)
    )
    return nodes, None


def _terminal_tool_name(content: str) -> str | None:
    """Best-effort extraction of a tool name from a role=tool node's content.

    The content is often JSON (``{"tool": "...", ...}``) but this must never
    raise on non-JSON content — plain-text tool output is equally valid.
    """
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(payload, dict):
        tool = payload.get("tool") or payload.get("tool_name") or payload.get("name")
        if isinstance(tool, str):
            return tool
    return None


def _classify_nodes(
    nodes: tuple[MessageNode, ...], signature_rules: tuple[SignatureRule, ...]
) -> tuple[str | None, str | None, str | None]:
    """Classify the terminal tool call in ``nodes`` against config-driven signatures.

    Returns (failure_kind, reason, terminal_tool). Scans role="tool" nodes
    from most recent to oldest so the terminal (last) tool call wins when
    multiple nodes match a signature. Compiled per-call (list is small,
    O(10) nodes x O(few) rules — not a hot loop) rather than caching, to
    keep this module free of module-level mutable state.
    """
    tool_nodes = [n for n in nodes if n.role == "tool"]
    if not tool_nodes:
        return None, None, None

    terminal_tool = _terminal_tool_name(tool_nodes[-1].content)

    for node in reversed(tool_nodes):
        for rule in signature_rules:
            try:
                if re.search(rule.pattern, node.content, re.IGNORECASE):
                    reason = node.content
                    if node.content.startswith(_BLOCK_CONTENT_PREFIX):
                        reason = node.content[len(_BLOCK_CONTENT_PREFIX) :].strip()
                    return rule.kind, reason, terminal_tool or _terminal_tool_name(node.content)
            except re.error:
                # A malformed pattern would already have been rejected at
                # config-load time; treat any runtime surprise as no-match
                # rather than raising out of the reaper pass.
                continue

    return None, None, terminal_tool


def _write_failure_kind_to_sidecar(
    sessions_dir: Path, worker: WorkerView, failure_kind: str
) -> None:
    """Write ``failure_kind`` into the adapter-specific sidecar for ``worker``.

    This is the hook exploited to make ``update_session_record_with_failure_
    classification`` / ``update_worker_record_with_failure_classification``
    no-op via their existing "skip if already classified" short-circuit —
    see devin_shell.py / claude_code.py. Best-effort: any I/O failure here
    just means the existing log-tail classifier runs normally instead.
    """
    if worker.adapter_kind == "devin":
        from .devin_shell import _sidecar_path as devin_sidecar_path
        from .devin_shell import _write_json

        sidecar_path = devin_sidecar_path(sessions_dir, worker.issue_number)
        writer = _write_json
    elif worker.adapter_kind == "claude-code":
        from .claude_code import _sidecar_path as claude_sidecar_path
        from .claude_code import _write_json_atomic as writer_fn

        sidecar_path = claude_sidecar_path(sessions_dir, worker.issue_number)
        writer = writer_fn
    else:
        return

    if not sidecar_path.exists():
        return
    try:
        with sidecar_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    if payload.get("failure_kind") is not None:
        return
    payload["failure_kind"] = failure_kind
    writer(sidecar_path, payload)


def classify_and_record(
    sessions_dir: Path,
    config: OrchestratorConfig,
    worker: WorkerView,
    *,
    now: datetime | None = None,
) -> str | None:
    """Extract a post-mortem for a just-reaped dead ``worker`` and record it.

    Must be called BEFORE the adapter's own log-tail classification
    (``update_session_record_with_failure_classification`` /
    ``update_worker_record_with_failure_classification``) — when this
    detects a ``worker_blocked`` signature, it writes ``failure_kind``
    directly into the sidecar, which makes those functions' existing
    "skip if already classified" short-circuit take over. This keeps the
    integration additive: no rewrite of either adapter's regex classifier.

    Returns "worker_blocked" when detected (caller should treat this as a
    signal to suppress hot redispatch), otherwise None (existing
    log-tail-then-stalled classification proceeds unaffected). Always
    writes a ``post-mortem.json`` sidecar when the section is enabled and a
    session was matched (or the reason it wasn't), for ``doctor``/digest
    surfacing — but a write failure never propagates as an exception.
    """
    pm_config = config.post_mortem
    if not pm_config.enabled:
        return None

    resolved_now = now or datetime.now(UTC)
    db_path = _resolve_db_path(pm_config.db_path)

    conn, temp_copy_path, open_error = _open_readonly(db_path)
    if conn is None:
        _try_write_record(
            sessions_dir,
            PostMortemRecord(
                issue_number=worker.issue_number,
                generated_at=resolved_now.isoformat(),
                db_path=str(db_path),
                matched=False,
                extraction_error=open_error,
            ),
        )
        return None

    try:
        margin = timedelta(seconds=pm_config.match_window_margin_seconds)
        try:
            started_at = datetime.fromisoformat(worker.started_at)
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            started_at = resolved_now - margin
        window_start = started_at - margin
        window_end = resolved_now + margin

        session_id, match_error = _find_matching_session(
            conn, worker.worktree_path, window_start, window_end
        )
        if session_id is None:
            _try_write_record(
                sessions_dir,
                PostMortemRecord(
                    issue_number=worker.issue_number,
                    generated_at=resolved_now.isoformat(),
                    db_path=str(db_path),
                    matched=False,
                    extraction_error=match_error,
                ),
            )
            return None

        nodes, extract_error = _extract_last_n_nodes(
            conn, session_id, pm_config.message_node_limit
        )
        failure_kind, reason, terminal_tool = _classify_nodes(nodes, pm_config.signature_rules)

        record = PostMortemRecord(
            issue_number=worker.issue_number,
            generated_at=resolved_now.isoformat(),
            db_path=str(db_path),
            matched=True,
            session_id=session_id,
            extraction_error=extract_error,
            terminal_tool=terminal_tool,
            terminal_reason=reason,
            failure_kind=failure_kind,
            message_nodes=nodes,
        )
        _try_write_record(sessions_dir, record)

        if failure_kind == "worker_blocked":
            _write_failure_kind_to_sidecar(sessions_dir, worker, "worker_blocked")
            return "worker_blocked"
        return None
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
        _cleanup_temp_copy(temp_copy_path)


def _try_write_record(sessions_dir: Path, record: PostMortemRecord) -> None:
    try:
        _write_json_atomic(_sidecar_path(sessions_dir, record.issue_number), record.to_dict())
    except OSError:
        # Post-mortem write is a diagnostic aid, never a gate — swallow.
        pass


def merge_attempt_snapshot(
    sessions_dir: Path, issue_number: int, attempt_snapshot: AttemptSnapshot
) -> None:
    """Fold an ``AttemptSnapshot`` (issue #261) into an existing post-mortem sidecar.

    Called by the adapters right after ``create_worktree`` returns from a
    redispatch: if a branch tip was preserved for this issue immediately
    before the redispatch reset it, the ref name and ahead-of-main count
    describe the attempt that just died — the same one this post-mortem
    sidecar (if any) already describes. No-op when there is no existing
    sidecar (nothing to attach the ref to) or when the snapshot itself has
    no ref (nothing was preserved, e.g. a clean branch with no commits).
    """
    if attempt_snapshot.ref_name is None:
        return
    existing = read_post_mortem(sessions_dir, issue_number)
    if existing is None:
        return
    updated = dataclasses.replace(
        existing,
        attempt_ref=attempt_snapshot.ref_name,
        attempt_ahead_of_main=attempt_snapshot.ahead_of_main_count,
    )
    _try_write_record(sessions_dir, updated)


__all__ = [
    "MessageNode",
    "PostMortemRecord",
    "classify_and_record",
    "merge_attempt_snapshot",
    "read_post_mortem",
]
