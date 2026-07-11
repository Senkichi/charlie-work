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

from .config import OrchestratorConfig, PostMortemConfig, SignatureRule

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
class AttemptAttachment:
    """One preserved-branch-tip attempt (issue #261) attached to a post-mortem.

    A single dead-session post-mortem sidecar can outlive more than one
    redispatch attempt before it's next read/rotated — ``PostMortemRecord.
    attempts`` is append-only (see ``merge_attempt_snapshot``) so an earlier
    attempt's ref is never silently overwritten by a later one.
    """

    ref: str
    ahead_of_main: int | None
    recorded_at: str


@dataclass(frozen=True)
class PostMortemRecord:
    """Result of a post-mortem extraction attempt for one dead worker.

    ``matched``/``extraction_error`` are mutually informative, not mutually
    exclusive with a partial result: a DB that opened fine but had no
    matching session still sets ``matched=False`` with a human-readable
    ``extraction_error`` describing why, so ``doctor`` can distinguish "no
    signal available" from "this worker was healthy."

    ``window_start_fallback`` is set only when ``worker.started_at`` failed
    to parse and the match window had to be anchored to "now" (a config
    lookback) instead — surfaced so a resulting non-match is diagnosable
    rather than silently indistinguishable from "genuinely no session."
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
    window_start_fallback: str | None = None
    attempts: tuple[AttemptAttachment, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> PostMortemRecord:
        nodes = payload.get("message_nodes") or []
        attempts = payload.get("attempts") or []
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
            window_start_fallback=payload.get("window_start_fallback"),
            attempts=tuple(
                AttemptAttachment(
                    ref=str(a.get("ref", "")),
                    ahead_of_main=a.get("ahead_of_main"),
                    recorded_at=str(a.get("recorded_at", "")),
                )
                for a in attempts
                if isinstance(a, dict)
            ),
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


def _parse_session_created_at(value: Any) -> datetime | None:
    """Defensively parse a sessions.db ``created_at`` value.

    Observed in the wild storing the same logical column as ISO-8601
    strings (naive or tz-aware) AND as epoch integers/floats (e.g.
    ``1783760135``) — the ``created_at`` column is declared TEXT in the
    schema this module infers, but SQLite's type affinity does not coerce
    a value the writer inserted as INTEGER, so both shapes can appear
    side-by-side across rows. A naive datetime is treated as UTC (this
    matches ``classify_and_record``'s own ``started_at`` handling). Never
    raises — returns None on anything unparseable.
    """
    if value is None:
        return None
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromtimestamp(float(text), tz=UTC)
        except (OverflowError, ValueError):
            pass
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    return None


def _find_matching_session(
    conn: sqlite3.Connection,
    working_directory: str,
    window_start: datetime,
    window_end: datetime,
) -> tuple[str | None, str | None]:
    """Find the session whose working_directory matches this worktree and
    whose created_at falls within [window_start, window_end].

    Fetches every row for ``working_directory`` and does the time-window
    comparison in Python rather than in the SQL query: a SQL-side
    ``created_at >= ?`` bound compares lexicographically, which silently
    produces wrong results whenever an ISO-string bound is compared against
    an epoch-int row (or vice versa) — see ``_parse_session_created_at``.
    Rows whose created_at is unparseable are skipped, not treated as a
    match. Returns (session_id, error). Multiple matches within the window
    pick the most recent by parsed created_at (a stale prior session in the
    same directory is possible if a worktree was reused across attempts).
    """
    try:
        cursor = conn.execute(
            "SELECT id, created_at FROM sessions WHERE working_directory = ?",
            (working_directory,),
        )
        rows = cursor.fetchall()
    except sqlite3.Error as exc:
        return None, f"sessions table query failed (schema drift?): {exc}"

    if not rows:
        return None, "no session found matching working_directory"

    candidates: list[tuple[datetime, str]] = []
    for row in rows:
        parsed = _parse_session_created_at(row[1])
        if parsed is None:
            continue
        if window_start <= parsed <= window_end:
            candidates.append((parsed, str(row[0])))

    if not candidates:
        return None, "no session found matching working_directory within the time window"

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], None


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


def _classify_via_sessions_db(
    sessions_dir: Path,
    pm_config: PostMortemConfig,
    worker: WorkerView,
    resolved_now: datetime,
) -> str | None:
    """DB-based post-mortem extraction — the primary signal.

    Always writes a diagnostic ``PostMortemRecord`` (matched True/False plus
    why) for ``doctor``/digest surfacing; returns ``"worker_blocked"`` only
    when a ``signature_rules`` rule of that kind matched the terminal tool
    call, else None. Extracted from ``classify_and_record`` so the caller can
    compose this with the log-tail fallback (``_classify_worker_blocked_
    from_log_tail`` below) without duplicating the sidecar-diagnostics
    plumbing.
    """
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
        window_start_fallback: str | None = None
        try:
            started_at = datetime.fromisoformat(worker.started_at)
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=UTC)
            window_start = started_at - margin
        except (ValueError, TypeError):
            # worker.started_at itself is unparseable — there is no reliable
            # anchor for the window at all. A narrow now-minus-margin window
            # (the old behavior) missed real sessions that actually started
            # well before "now" (reaping can run long after a worker died).
            # Widen to a config-derived lookback from "now" instead, and
            # record that this fallback fired so a resulting non-match is
            # diagnosable rather than indistinguishable from "no session."
            window_start_fallback = "unparseable_started_at"
            lookback = timedelta(seconds=pm_config.unparseable_started_at_lookback_seconds)
            window_start = resolved_now - lookback
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
                    window_start_fallback=window_start_fallback,
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
            window_start_fallback=window_start_fallback,
        )
        _try_write_record(sessions_dir, record)

        return failure_kind if failure_kind == "worker_blocked" else None
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
        _cleanup_temp_copy(temp_copy_path)


def _classify_worker_blocked_from_log_tail(
    log_path: Path, signature_rules: tuple[SignatureRule, ...]
) -> bool:
    """Fallback ``worker_blocked`` classifier over the worker's own log tail.

    Used when DB-based extraction (``_classify_via_sessions_db`` above)
    misses — ``post_mortem.enabled`` is False, sessions.db is
    locked/missing/schema-drifted, or no session matched this worker's
    working_directory/time window. The log tail is the only signal left in
    that case (issue #260, corrected premise: "A tool was rejected by the
    user" is the Devin CLI's own log/stdout surfacing of a PreToolUse hook
    block, not something that ever reaches sessions.db as a "Tool blocked:"
    message-node row).

    Reuses ``PostMortemConfig.signature_rules`` (filtered to
    ``kind == "worker_blocked"``) rather than adding a parallel config
    surface — see that field's docstring. Matches against the last 2KB of
    the log, mirroring ``devin_shell._classify_session_failure``'s tail
    window.

    A ``worker_blocked`` verdict from this function must never carry
    throttle retry semantics (no ``throttled_until``) — provider-throttle
    tail matching is a separate, adapter-owned concern with its own retry
    semantics (see ``throttle_signatures.match_throttle_tail``); this
    function only ever returns a bare bool, so there is no way for a caller
    to accidentally wire it into that retry path.
    """
    if not log_path.exists():
        return False
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    tail = log_text[-2048:] if len(log_text) > 2048 else log_text
    for rule in signature_rules:
        if rule.kind != "worker_blocked":
            continue
        try:
            if re.search(rule.pattern, tail, re.IGNORECASE):
                return True
        except re.error:
            # A malformed pattern would already have been rejected at
            # config-load time; treat any runtime surprise as no-match
            # rather than raising out of the reaper pass.
            continue
    return False


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

    Tries the sessions.db extraction first (``_classify_via_sessions_db``);
    when that misses, falls back to the worker's own log tail
    (``_classify_worker_blocked_from_log_tail``, issue #260 corrected
    premise) — the only remaining signal when the DB is unavailable or
    didn't contain a matching session.

    Returns "worker_blocked" when detected by either source (caller should
    treat this as a signal to suppress hot redispatch), otherwise None
    (existing log-tail-then-stalled classification proceeds unaffected).
    Always writes a ``post-mortem.json`` sidecar when the section is enabled
    and a session was matched (or the reason it wasn't), for
    ``doctor``/digest surfacing — but a write failure never propagates as an
    exception.
    """
    pm_config = config.post_mortem
    resolved_now = now or datetime.now(UTC)

    if pm_config.enabled:
        if _classify_via_sessions_db(sessions_dir, pm_config, worker, resolved_now) == (
            "worker_blocked"
        ):
            _write_failure_kind_to_sidecar(sessions_dir, worker, "worker_blocked")
            return "worker_blocked"

        if _classify_worker_blocked_from_log_tail(
            Path(worker.log_path), pm_config.signature_rules
        ):
            _write_failure_kind_to_sidecar(sessions_dir, worker, "worker_blocked")
            return "worker_blocked"

    return None


def _try_write_record(sessions_dir: Path, record: PostMortemRecord) -> None:
    try:
        _write_json_atomic(_sidecar_path(sessions_dir, record.issue_number), record.to_dict())
    except OSError:
        # Post-mortem write is a diagnostic aid, never a gate — swallow.
        pass


def merge_attempt_snapshot(
    sessions_dir: Path,
    issue_number: int,
    attempt_snapshot: AttemptSnapshot,
    *,
    now: datetime | None = None,
) -> None:
    """Append an ``AttemptSnapshot`` (issue #261) onto an existing post-mortem sidecar.

    Called by the adapters right after ``create_worktree`` returns from a
    redispatch: if a branch tip was preserved for this issue immediately
    before the redispatch reset it, the ref name and ahead-of-main count
    describe the attempt that just died — the same one this post-mortem
    sidecar (if any) already describes. No-op when there is no existing
    sidecar (nothing to attach the ref to) or when the snapshot itself has
    no ref (nothing was preserved, e.g. a clean branch with no commits).

    Appends to ``PostMortemRecord.attempts`` rather than overwriting a
    singular field: a sidecar can outlive more than one redispatch attempt
    before it is next read or rotated, and an overwrite silently dropped
    every attempt but the most recent one.
    """
    if attempt_snapshot.ref_name is None:
        return
    existing = read_post_mortem(sessions_dir, issue_number)
    if existing is None:
        return
    attachment = AttemptAttachment(
        ref=attempt_snapshot.ref_name,
        ahead_of_main=attempt_snapshot.ahead_of_main_count,
        recorded_at=(now or datetime.now(UTC)).isoformat(),
    )
    updated = dataclasses.replace(existing, attempts=(*existing.attempts, attachment))
    _try_write_record(sessions_dir, updated)


__all__ = [
    "AttemptAttachment",
    "MessageNode",
    "PostMortemRecord",
    "classify_and_record",
    "merge_attempt_snapshot",
    "read_post_mortem",
]
