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
- ``message_nodes`` schema was verified against a live production
  sessions.db (2026-07-12, ~268k rows): ``(row_id INTEGER PRIMARY KEY,
  session_id TEXT, node_id INTEGER, parent_node_id INTEGER, chat_message
  TEXT, created_at INTEGER, metadata TEXT)`` with UNIQUE(session_id,
  node_id). There is no ``role``/``content``/``id`` column — role and
  content live inside the ``chat_message`` JSON blob (see
  ``_parse_chat_message``), per-session ordering is by ``node_id``, and
  ``created_at`` is an epoch integer. There is still no official Devin CLI
  documentation for this table (extraction-dossier.md item 23), so any
  future drift is caught by ``sqlite3.Error`` / defensive JSON parsing and
  treated as schema drift, not a crash.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import OrchestratorConfig, PostMortemConfig, SignatureRule, WatchdogConfig

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


def _devin_logs_dir(db_path: Path) -> Path:
    """Resolve the Devin CLI per-PID log directory alongside sessions.db."""
    return db_path.parent / "logs"


@dataclass(frozen=True)
class ActivitySource:
    """One real-activity source consulted for a live worker session.

    ``threshold_minutes`` is optional and overrides the caller's default
    freshness window for sources (like worktree file mtimes) that need a
    longer grace period than the sidecar stall threshold.
    """

    name: str
    timestamp: datetime | None
    staleness_seconds: float | None
    error: str | None
    threshold_minutes: int | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat() if self.timestamp is not None else None,
            "staleness_seconds": self.staleness_seconds,
            "error": self.error,
            "threshold_minutes": self.threshold_minutes,
        }


@dataclass(frozen=True)
class RealActivityProbe:
    """Corroboration probe for a live worker session.

    Bundles every real-activity source that was consulted (sessions.db last
    message node, per-PID Devin log mtime, worktree file mtimes) and the
    freshest timestamp among them. Used by the stall watchdog to avoid
    false-positive kills when a shim has frozen the sidecar log but the real
    Devin session is still working.
    """

    sources: tuple[ActivitySource, ...]
    latest_timestamp: datetime | None = field(init=False)
    latest_source: str | None = field(init=False)

    def __post_init__(self) -> None:
        latest: datetime | None = None
        latest_source: str | None = None
        for source in self.sources:
            if source.timestamp is not None:
                if latest is None or source.timestamp > latest:
                    latest = source.timestamp
                    latest_source = source.name
        object.__setattr__(self, "latest_timestamp", latest)
        object.__setattr__(self, "latest_source", latest_source)

    def is_fresh(self, default_threshold_minutes: int) -> bool:
        """Return True when any source is fresh within its threshold.

        Sources that declare their own ``threshold_minutes`` (e.g. worktree
        file mtimes) are compared against that value; all other sources use
        the caller's default threshold.
        """
        for source in self.sources:
            if source.timestamp is None or source.staleness_seconds is None:
                continue
            threshold = source.threshold_minutes
            if threshold is None:
                threshold = default_threshold_minutes
            if source.staleness_seconds <= threshold * 60:
                return True
        return False

    def to_payload(self) -> dict[str, Any]:
        return {
            "sources": [source.to_payload() for source in self.sources],
            "latest_timestamp": self.latest_timestamp.isoformat()
            if self.latest_timestamp is not None
            else None,
            "latest_source": self.latest_source,
        }


@dataclass(frozen=True)
class MessageNode:
    """One message_nodes row, mapped out of its ``chat_message`` JSON blob.

    ``tool_name`` is resolved for role="tool" nodes by joining their
    ``tool_call_id`` against the ``tool_calls`` of the assistant nodes in
    the same extraction window (see ``_extract_last_n_nodes``) — None when
    the issuing assistant node fell outside the window or the blob had no
    ``tool_call_id``.
    """

    role: str
    content: str
    created_at: str | None = None
    tool_name: str | None = None


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
                    tool_name=n.get("tool_name"),
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


def _normalize_working_directory(path: str) -> tuple[str, ...]:
    """Return a path representation that is insensitive to case, separator
    style, trailing separators, and leading drive-letter punctuation.

    Two paths that differ only by these formatting details (e.g.
    ``C:\\repo\\.var\\worktrees\\issue-42``,
    ``C:/repo/.var/worktrees/issue-42``,
    ``/c/repo/.var/worktrees/issue-42``,
    or ``c:/repo/.var/worktrees/issue-42/``) collapse to the same tuple of
    segments. This avoids assuming a single canonical string representation
    for the ``working_directory`` value written by the Devin CLI.
    """
    if not isinstance(path, str):
        path = str(path)
    lowered = path.lower().replace("\\", "/")
    parts = [part for part in lowered.split("/") if part]
    if parts and len(parts[0]) == 2 and parts[0][1] == ":" and parts[0][0].isalpha():
        parts[0] = parts[0][0]
    return tuple(parts)


# Number of trailing path segments used by the suffix-match fallback in
# _find_matching_session (issue #343). Fleet worktree paths are built as
# ``<worktrees-dir>/<slug>`` where ``<slug>`` embeds the issue number (see
# worktree.create_worktree's ``target_dir / _slugify(branch)``), so the last
# two segments (parent dir name + issue-numbered leaf) are the most that can
# be matched reliably.
#
# NOT a cross-repo-unique key. ``sessions.db`` is machine-wide
# (``%APPDATA%\devin\cli\sessions.db``, see ``_default_db_path``), shared by
# every fleet repo dispatched from this machine, and this 2-segment suffix
# carries no repo identity at all -- it is only (parent-dir-literal,
# issue-slug). Two different repos in the fleet that both dispatch the same
# issue number with a similar-enough title (so ``_slugify`` produces the same
# slug) inside overlapping dispatch windows (see the ``created_at`` window
# filter in ``_find_matching_session``) can collide here, misattributing one
# repo's session data to another repo's worker.
#
# Widening this to include a repo-identifying segment (e.g. 5 segments to
# reach ``<repo-dir>/.var/charlie-work/worktrees/<slug>``) was considered and
# rejected: production evidence (``test_real_activity_for_worker_matches_
# real_fleet_working_directory_shape`` in tests/test_post_mortem.py, sampled
# from a live sessions.db 2026-07-13) shows the Devin CLI's recorded
# ``working_directory`` is frequently rooted under an unrelated
# ``AppData\Local\Temp\...`` session directory that preserves *none* of the
# worktree path's ``.var/charlie-work/worktrees`` segments -- only the
# trailing ``worktrees/<slug>`` pair survives. A wider suffix would silently
# stop matching that real shape, trading a narrow cross-repo collision risk
# for a total loss of corroboration on the common case this fallback exists
# for. The collision risk above is accepted for now (tracked as a follow-up,
# not solved by widening the suffix).
_WORKING_DIRECTORY_SUFFIX_SEGMENTS = 2


def _working_directory_suffix(path: str) -> tuple[str, ...]:
    """Return the last ``_WORKING_DIRECTORY_SUFFIX_SEGMENTS`` normalized segments.

    Used as a last-resort match key when neither an exact nor a fully
    normalized ``working_directory`` comparison finds a row (issue #343):
    production evidence shows the Devin CLI can record a ``working_directory``
    whose absolute prefix bears no relationship at all to the worktree path
    this process computed (different resolved root / mount point / capture
    point), while the dispatch-unique trailing segments -- the worktrees
    directory name and the issue-numbered slug leaf -- are identical.

    This is a coarse, machine-wide, repo-agnostic key -- see the collision
    caveat on ``_WORKING_DIRECTORY_SUFFIX_SEGMENTS`` above.
    """
    normalized = _normalize_working_directory(path)
    if len(normalized) < _WORKING_DIRECTORY_SUFFIX_SEGMENTS:
        return normalized
    return normalized[-_WORKING_DIRECTORY_SUFFIX_SEGMENTS:]


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

    If the exact SQL match for ``working_directory`` returns no rows, this
    falls back to a tolerant comparison using ``_normalize_working_directory``,
    and then to a suffix-only comparison using ``_working_directory_suffix``
    (issue #343: real fleet dispatches have been observed where the Devin
    CLI's recorded ``working_directory`` shares no absolute-prefix
    relationship at all with the worktree path, only the trailing
    issue-numbered slug) before giving up, and when it does give up it
    surfaces a sample of the distinct ``working_directory`` values actually
    stored in the database for diagnostics.
    """
    try:
        cursor = conn.execute(
            "SELECT id, created_at FROM sessions WHERE working_directory = ?",
            (working_directory,),
        )
        exact_rows = cursor.fetchall()
    except sqlite3.Error as exc:
        return None, f"sessions table query failed (schema drift?): {exc}"

    if exact_rows:
        rows = exact_rows
    else:
        # No byte-for-byte match. Try a tolerant comparison with the whole
        # table (we are already in the rare miss path) before declaring defeat.
        try:
            cursor = conn.execute("SELECT id, working_directory, created_at FROM sessions")
            all_rows = cursor.fetchall()
        except sqlite3.Error as exc:
            return None, f"sessions table query failed (schema drift?): {exc}"

        target_norm = _normalize_working_directory(working_directory)
        norm_rows = [
            (row[0], row[2])
            for row in all_rows
            if _normalize_working_directory(row[1]) == target_norm
        ]
        if norm_rows:
            rows = norm_rows
        else:
            # Last resort: match on the trailing (worktrees-dir, issue-slug)
            # segment pair only, ignoring the absolute prefix entirely.
            target_suffix = _working_directory_suffix(working_directory)
            suffix_rows = [
                (row[0], row[2])
                for row in all_rows
                if target_suffix and _working_directory_suffix(row[1]) == target_suffix
            ]
            if suffix_rows:
                rows = suffix_rows
            else:
                rows = []

        if not rows:
            distinct = sorted({str(row[1]) for row in all_rows if row[1] is not None})[:10]
            if distinct:
                return (
                    None,
                    f"no session found matching working_directory; sample distinct working_directory values in sessions.db: {distinct}",
                )
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


def _parse_chat_message(raw: Any) -> tuple[str, str, dict[str, str], str | None]:
    """Map one ``message_nodes.chat_message`` JSON blob to
    ``(role, content, tool_call_names, tool_call_id)``.

    Verified shape (live sessions.db, 2026-07-12): ``{"message_id": ...,
    "role": "assistant"|"tool"|"system"|"user", "content": str,
    "tool_calls": [{"id": ..., "name": ..., "arguments": ...}, ...],
    "tool_call_id": ..., "thinking": ..., "metadata": ...}`` —
    ``tool_calls`` appears on assistant nodes, ``tool_call_id`` on the
    tool-result nodes answering them. Never raises: a non-JSON or non-dict
    blob degrades to ``("", raw_text, {}, None)`` so signature
    classification still sees the raw content.
    """
    text = raw if isinstance(raw, str) else str(raw)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return "", text, {}, None
    if not isinstance(payload, dict):
        return "", text, {}, None
    role = payload.get("role")
    content = payload.get("content")
    if not isinstance(content, str):
        content = json.dumps(content) if content is not None else ""
    tool_call_names: dict[str, str] = {}
    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if isinstance(call, dict):
                call_id = call.get("id")
                call_name = call.get("name")
                if isinstance(call_id, str) and isinstance(call_name, str):
                    tool_call_names[call_id] = call_name
    tool_call_id = payload.get("tool_call_id")
    return (
        role if isinstance(role, str) else "",
        content,
        tool_call_names,
        tool_call_id if isinstance(tool_call_id, str) else None,
    )


def _extract_last_n_nodes(
    conn: sqlite3.Connection, session_id: str, limit: int
) -> tuple[tuple[MessageNode, ...], str | None]:
    """Return the last ``limit`` message_nodes for ``session_id`` in chronological order.

    Ordered by ``node_id`` — per-session monotonic and backed by the real
    schema's UNIQUE(session_id, node_id) index; the table has no ``id``
    column. Each row's ``chat_message`` JSON blob is mapped through
    ``_parse_chat_message``; tool-result nodes get ``tool_name`` resolved by
    joining their ``tool_call_id`` against the ``tool_calls`` of assistant
    nodes earlier in the window (a single chronological pass suffices — the
    issuing assistant node always precedes its results). ``created_at``
    epoch integers are normalized to ISO-8601 for the JSON sidecar.
    """
    try:
        cursor = conn.execute(
            "SELECT chat_message, created_at FROM message_nodes "
            "WHERE session_id = ? ORDER BY node_id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cursor.fetchall()
    except sqlite3.Error as exc:
        return (), f"message_nodes table query failed (schema drift?): {exc}"

    nodes: list[MessageNode] = []
    tool_call_names: dict[str, str] = {}
    for chat_message, created_raw in reversed(rows):
        role, content, call_names, tool_call_id = _parse_chat_message(chat_message)
        tool_call_names.update(call_names)
        created_dt = _parse_session_created_at(created_raw)
        if created_dt is not None:
            created_at = created_dt.isoformat()
        else:
            created_at = str(created_raw) if created_raw is not None else None
        nodes.append(
            MessageNode(
                role=role,
                content=content,
                created_at=created_at,
                tool_name=tool_call_names.get(tool_call_id) if tool_call_id else None,
            )
        )
    return tuple(nodes), None


def _terminal_tool_name(content: str) -> str | None:
    """Best-effort extraction of a tool name from a role=tool node's content.

    Fallback for when the ``tool_call_id`` join in ``_extract_last_n_nodes``
    did not resolve a ``MessageNode.tool_name`` (issuing assistant node
    outside the window, or a block payload written directly as content per
    the issue #261 spec). The content is sometimes JSON
    (``{"tool": "...", ...}``) but this must never raise on non-JSON
    content — plain-text tool output is equally valid.
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

    terminal_node = tool_nodes[-1]
    terminal_tool = terminal_node.tool_name or _terminal_tool_name(terminal_node.content)

    for node in reversed(tool_nodes):
        for rule in signature_rules:
            try:
                if re.search(rule.pattern, node.content, re.IGNORECASE):
                    reason = node.content
                    if node.content.startswith(_BLOCK_CONTENT_PREFIX):
                        reason = node.content[len(_BLOCK_CONTENT_PREFIX) :].strip()
                    return (
                        rule.kind,
                        reason,
                        terminal_tool or node.tool_name or _terminal_tool_name(node.content),
                    )
            except re.error:
                # A malformed pattern would already have been rejected at
                # config-load time; treat any runtime surprise as no-match
                # rather than raising out of the reaper pass.
                continue

    return None, None, terminal_tool


def _events_path_from_log(log_path: Path) -> Path:
    """Return the events.jsonl sibling path for a worker log path.

    Supports Claude Code's ``issue-<n>.claude.log`` /
    ``issue-<n>-rework.claude.log`` as well as the devin-shell
    ``issue-<n>.log`` shape.
    """
    name = log_path.name
    if name.endswith(".claude.log"):
        return log_path.with_name(name[: -len(".claude.log")] + ".events.jsonl")
    return log_path.with_suffix(".events.jsonl")


def _last_event_timestamp(events_path: Path) -> datetime | None:
    """Return the latest ISO timestamp found in an events.jsonl file.

    Lines are expected to be JSON objects containing a ``timestamp`` field.
    Returns None for a missing file, malformed content, or no timestamps.
    Naive timestamps are treated as UTC.
    """
    if not events_path.exists():
        return None

    latest: datetime | None = None
    try:
        with events_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                timestamp_str = event.get("timestamp")
                if not isinstance(timestamp_str, str):
                    continue
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=UTC)
                    if latest is None or timestamp > latest:
                        latest = timestamp
                except (ValueError, TypeError):
                    continue
    except OSError:
        return None
    return latest


_WORKTREE_MTIME_CHECKOUT_BUFFER_SECONDS = 1


def _worktree_mtime_source(
    worktree_path: str,
    watchdog_config: WatchdogConfig | None,
    now: datetime,
    start_baseline: datetime | None = None,
) -> ActivitySource | None:
    """Build the worktree file mtime ActivitySource for a worker session.

    Walks up to ``watchdog_config.worktree_mtime_max_depth`` directory levels
    below ``worktree_path`` and skips directories whose basename is in
    ``watchdog_config.worktree_mtime_exclude_dirs``. Returns the most recent
    regular file mtime as the activity timestamp, or an errored source when the
    path is missing, empty, or unreadable. Returns None when the source is
    disabled.

    When ``start_baseline`` is provided, only file mtimes strictly newer than
    ``start_baseline + _WORKTREE_MTIME_CHECKOUT_BUFFER_SECONDS`` are considered
    real post-start activity. If no files have been modified after that cutoff,
    the source reports the baseline itself as the timestamp with a threshold of
    0 minutes, so the source is conclusively stale rather than inconclusive.
    """
    name = "worktree_files_mtime"
    if watchdog_config is None or not watchdog_config.worktree_mtime_enabled:
        return None

    if not worktree_path:
        return ActivitySource(
            name=name,
            timestamp=None,
            staleness_seconds=None,
            error="no worktree_path provided",
        )

    path = Path(worktree_path)
    try:
        if not path.is_dir():
            return ActivitySource(
                name=name,
                timestamp=None,
                staleness_seconds=None,
                error=f"worktree path is not a directory: {worktree_path}",
            )
    except OSError as exc:
        return ActivitySource(
            name=name,
            timestamp=None,
            staleness_seconds=None,
            error=f"worktree path stat failed: {exc}",
        )

    exclude = set(watchdog_config.worktree_mtime_exclude_dirs)
    max_depth = watchdog_config.worktree_mtime_max_depth
    base_parts = len(path.parts)
    max_mtime: float | None = None

    if start_baseline is not None and start_baseline.tzinfo is None:
        start_baseline = start_baseline.replace(tzinfo=UTC)

    try:
        for root, dirs, files in os.walk(path, topdown=True):
            root_path = Path(root)
            current_depth = len(root_path.parts) - base_parts
            if current_depth >= max_depth:
                dirs[:] = []
            else:
                dirs[:] = [d for d in dirs if d not in exclude]
            for filename in files:
                file_path = root_path / filename
                try:
                    st = os.lstat(file_path)
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    if start_baseline is not None:
                        cutoff = start_baseline + timedelta(
                            seconds=_WORKTREE_MTIME_CHECKOUT_BUFFER_SECONDS
                        )
                        if st.st_mtime <= cutoff.timestamp():
                            continue
                    if max_mtime is None or st.st_mtime > max_mtime:
                        max_mtime = st.st_mtime
                except OSError:
                    continue
    except OSError as exc:
        return ActivitySource(
            name=name,
            timestamp=None,
            staleness_seconds=None,
            error=f"worktree walk failed: {exc}",
        )

    if max_mtime is not None:
        timestamp = datetime.fromtimestamp(max_mtime, tz=UTC)
        return ActivitySource(
            name=name,
            timestamp=timestamp,
            staleness_seconds=(now - timestamp).total_seconds(),
            error=None,
            threshold_minutes=watchdog_config.worktree_mtime_threshold_minutes,
        )

    if start_baseline is not None:
        return ActivitySource(
            name=name,
            timestamp=start_baseline,
            staleness_seconds=(now - start_baseline).total_seconds(),
            error=None,
            threshold_minutes=0,
        )

    return ActivitySource(
        name=name,
        timestamp=None,
        staleness_seconds=None,
        error="no eligible worktree files found",
    )


def real_activity_for_worker(
    pm_config: PostMortemConfig,
    worktree_path: str,
    started_at: str,
    pid: int | None,
    now: datetime,
    log_path: str | None = None,
    watchdog_config: WatchdogConfig | None = None,
) -> RealActivityProbe:
    """Build a ``RealActivityProbe`` for a live worker session.

    Corroborates the sidecar log's mtime against independent, real-session
    signals that continue to advance even when a shim has frozen the sidecar:

    1. The Devin CLI's sessions.db ``message_nodes`` table, matched by the
       worker's ``worktree_path`` and its ``started_at`` window.
    2. The Devin CLI's own per-PID log file(s) for the worker PID, located
       in the ``logs/`` sibling of sessions.db.
    3. Claude Code's ``events.jsonl`` stream-json sibling to the worker's
       ``log_path``, using either the last event timestamp or the file's mtime.
    4. The worker's ``worktree_path`` file mtimes (bounded depth, excluding
       ``.git`` and ``.venv``), with a generous config-gated threshold.

    Any source with a fresh timestamp is sufficient to show the worker is
    healthy. If every available source is quiet past the threshold, the caller
    should treat the session as genuinely stalled. Any I/O or schema problem
    is recorded as a source ``error`` and does not raise — this is best-effort,
    just like the post-mortem extractor.
    """
    sources: list[ActivitySource] = []
    db_path = _resolve_db_path(pm_config.db_path)
    logs_dir = _devin_logs_dir(db_path)

    start_baseline: datetime | None = None
    try:
        start_baseline = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        if start_baseline.tzinfo is None:
            start_baseline = start_baseline.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        pass

    # --- Source 1: sessions.db last message_nodes row for the matching session
    conn: sqlite3.Connection | None = None
    temp_copy_path: Path | None = None
    db_error: str | None = None
    db_timestamp: datetime | None = None
    try:
        conn, temp_copy_path, db_error = _open_readonly(db_path)
        if conn is None:
            sources.append(
                ActivitySource(
                    name="sessions.db", timestamp=None, staleness_seconds=None, error=db_error
                )
            )
        else:
            margin = timedelta(seconds=pm_config.match_window_margin_seconds)
            if start_baseline is not None:
                window_start = start_baseline - margin
            else:
                lookback = timedelta(seconds=pm_config.unparseable_started_at_lookback_seconds)
                window_start = now - lookback
            window_end = now + margin

            session_id, match_error = _find_matching_session(
                conn, worktree_path, window_start, window_end
            )
            if session_id is None:
                sources.append(
                    ActivitySource(
                        name="sessions.db",
                        timestamp=None,
                        staleness_seconds=None,
                        error=match_error,
                    )
                )
            else:
                try:
                    row = conn.execute(
                        "SELECT created_at FROM message_nodes "
                        "WHERE session_id = ? ORDER BY node_id DESC LIMIT 1",
                        (session_id,),
                    ).fetchone()
                    if row is None:
                        db_error = "no message_nodes for matched session"
                    else:
                        db_timestamp = _parse_session_created_at(row[0])
                        if db_timestamp is None:
                            db_error = f"unparseable message_nodes.created_at {row[0]!r}"
                except sqlite3.Error as exc:
                    db_error = f"message_nodes query failed (schema drift?): {exc}"

                if db_timestamp is None:
                    sources.append(
                        ActivitySource(
                            name="sessions.db",
                            timestamp=None,
                            staleness_seconds=None,
                            error=db_error,
                        )
                    )
                else:
                    sources.append(
                        ActivitySource(
                            name="sessions.db",
                            timestamp=db_timestamp,
                            staleness_seconds=(now - db_timestamp).total_seconds(),
                            error=None,
                        )
                    )
    except Exception as exc:
        # Belt-and-suspenders: _open_readonly is already defensive, but if
        # something unexpected escapes, record it and keep going.
        sources.append(
            ActivitySource(
                name="sessions.db",
                timestamp=None,
                staleness_seconds=None,
                error=f"unexpected: {exc}",
            )
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        _cleanup_temp_copy(temp_copy_path)

    # --- Source 2: per-PID Devin log mtime
    per_pid_timestamp: datetime | None = None
    per_pid_error: str | None = None
    if pid is None:
        per_pid_error = "no pid"
    elif not logs_dir.is_dir():
        per_pid_error = "per-PID log directory not found"
    else:
        try:
            log_paths = list(logs_dir.glob(f"devin_*_{pid}.log"))
            if not log_paths:
                per_pid_error = "no per-PID log found"
            else:
                latest_log = max(log_paths, key=lambda p: p.stat().st_mtime)
                per_pid_timestamp = datetime.fromtimestamp(latest_log.stat().st_mtime, tz=UTC)
        except OSError as exc:
            per_pid_error = f"per-PID log stat failed: {exc}"

    if per_pid_timestamp is None:
        sources.append(
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=None,
                staleness_seconds=None,
                error=per_pid_error,
            )
        )
    else:
        sources.append(
            ActivitySource(
                name="devin_per_pid_log",
                timestamp=per_pid_timestamp,
                staleness_seconds=(now - per_pid_timestamp).total_seconds(),
                error=None,
            )
        )

    # --- Source 3: Claude Code events.jsonl stream-json sibling
    if log_path:
        events_path = _events_path_from_log(Path(log_path))
        if events_path.exists():
            try:
                events_stat = events_path.stat()
                events_mtime = datetime.fromtimestamp(events_stat.st_mtime, tz=UTC)
                last_event_ts = _last_event_timestamp(events_path)
                timestamp = events_mtime
                if last_event_ts is not None and last_event_ts > timestamp:
                    timestamp = last_event_ts
                sources.append(
                    ActivitySource(
                        name="claude_events_jsonl",
                        timestamp=timestamp,
                        staleness_seconds=(now - timestamp).total_seconds(),
                        error=None,
                    )
                )
            except OSError as exc:
                sources.append(
                    ActivitySource(
                        name="claude_events_jsonl",
                        timestamp=None,
                        staleness_seconds=None,
                        error=f"events.jsonl read failed: {exc}",
                    )
                )

    # --- Source 4: worktree file mtimes (issue #353)
    worktree_source = _worktree_mtime_source(worktree_path, watchdog_config, now, start_baseline)
    if worktree_source is not None:
        sources.append(worktree_source)

    return RealActivityProbe(sources=tuple(sources))


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
    "ActivitySource",
    "AttemptAttachment",
    "MessageNode",
    "PostMortemRecord",
    "RealActivityProbe",
    "classify_and_record",
    "merge_attempt_snapshot",
    "read_post_mortem",
    "real_activity_for_worker",
]
