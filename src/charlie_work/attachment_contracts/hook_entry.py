"""PreToolUse stdin protocol.

stdin JSON: `{"tool_name": "Write|Edit|MultiEdit", "tool_input": {"file_path": ...}}`.

PreToolUse fires BEFORE the write lands, so the on-disk file is still the
pre-edit version. `_compute_proposed_content` reconstructs what the file will
read after the edit (full content for Write, old_string/new_string applied
for Edit/MultiEdit) and passes it to `check_file` as a content override, so
the verdict is against the edit that is actually about to happen. When the
proposed content can't be determined (unrecognized tool, malformed payload,
`old_string` not found, or no on-disk base yet) this falls back to the
on-disk file, same as before -- so this is best-effort pre-edit feedback, not
a guarantee; CI's full-tree check remains the real, authoritative gate.

- No `.attachment-budgets.json` found walking up from the target file -> exit 0
  silently (fast no-op outside piloted repos).
- Unattended (`CHARLIE_FLEET_WORKER=1` or `CLAUDE_CODE_UNATTENDED=1`) -> ALWAYS
  advisory, never exit 2: print `{"hookSpecificOutput": {"additionalContext": ...}}`
  to stdout and best-effort append a marker line to
  `.var/attachment-contracts/advisories.jsonl`.
- Interactive + mode=enforce -> exit 2 with the redirect message on stderr.
- Mode source: `ATTACHMENT_CONTRACTS_MODE` env, else the baseline file's `mode`
  key, else `"advise"`.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Mapping

from charlie_work.attachment_contracts.baseline import BASELINE_FILENAME, TamperError
from charlie_work.attachment_contracts.baseline import load as load_baseline
from charlie_work.attachment_contracts.check import check_file
from charlie_work.attachment_contracts.model import AdvisoryRecord, Finding

_ADVISORY_LOG_REL = Path(".var/attachment-contracts/advisories.jsonl")
_ACTIONABLE_SEVERITIES = frozenset({"block", "error"})

# Issue #1466: the stable first-line marker a worker writes on the PR comment
# it publishes to surface triggered advisories. The review-packet builder
# (``OrchestratorApp._build_attachment_budget_section``) scans PR issue-level
# comments for one whose body starts with this marker (prefix test, matching
# ``ORCHESTRATOR_COMMENT_MARKER``'s discipline) and parses the fenced JSON
# block that follows it into ``AdvisoryRecord`` entries via
# ``parse_advisories_comment``. The ``v1`` suffix is a schema version so a
# future shape change can coexist with stale comments from an older worker
# without the builder misparsing either side.
ADVISORY_COMMENT_MARKER = "<!-- attachment-advisories v1 -->"


def _advisory_record_from_raw(raw: object) -> AdvisoryRecord | None:
    """Build a single ``AdvisoryRecord`` from one decoded JSON value.

    Shared by ``read_advisories`` (JSONL log) and ``parse_advisories_comment``
    (PR-comment fenced JSON array) so the two channels cannot drift on
    schema. Returns ``None`` for anything that is not a JSON object or is
    missing a required field (``severity``/``file``/``identity``/
    ``message``) -- the caller skips it rather than raising, since both
    channels feed an advisory-only review section, never a blocking gate.
    ``redirect``/``timestamp`` are optional so old-shape records written
    before issue #1460 still parse.
    """
    if not isinstance(raw, dict):
        return None
    try:
        severity = str(raw["severity"])
        file_ = str(raw["file"])
        identity = str(raw["identity"])
        message = str(raw["message"])
    except KeyError:
        return None
    redirect_raw = raw.get("redirect")
    redirect = str(redirect_raw) if isinstance(redirect_raw, str) else None
    timestamp_raw = raw.get("timestamp")
    timestamp = str(timestamp_raw) if isinstance(timestamp_raw, str) else None
    return AdvisoryRecord(
        severity=severity,  # type: ignore[arg-type]
        file=file_,
        identity=identity,
        message=message,
        redirect=redirect,
        timestamp=timestamp,
    )


def _find_baseline_root(target: Path) -> Path | None:
    """Walk upward from `target` looking for `.attachment-budgets.json`."""
    start = target if target.is_dir() else target.parent
    current = start.resolve()
    while True:
        if (current / BASELINE_FILENAME).is_file():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _is_unattended(env: Mapping[str, str]) -> bool:
    return env.get("CHARLIE_FLEET_WORKER") == "1" or env.get("CLAUDE_CODE_UNATTENDED") == "1"


def _resolve_mode(root: Path, env: Mapping[str, str]) -> str:
    override = env.get("ATTACHMENT_CONTRACTS_MODE")
    if override:
        return override
    try:
        document = load_baseline(root / BASELINE_FILENAME)
    except (TamperError, OSError, ValueError):
        return "advise"
    mode = document.get("mode", "advise")
    return mode if isinstance(mode, str) and mode else "advise"


def _format_message(findings: list[Finding]) -> str:
    lines = ["Attachment-Point Contracts:"]
    for f in findings:
        line = f"[{f.severity}] {f.identity} ({f.file}): {f.message}"
        if f.redirect:
            line += f" -> redirect: {f.redirect}"
        lines.append(line)
    return "\n".join(lines)


def _append_advisory_log(root: Path, findings: list[Finding]) -> None:
    log_path = root / _ADVISORY_LOG_REL
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            for f in findings:
                fh.write(
                    json.dumps(
                        {
                            "severity": f.severity,
                            "file": f.file,
                            "identity": f.identity,
                            "message": f.message,
                            # Issue #1460: structural fields for the review
                            # packet's redirects-not-taken section.
                            # ``redirect`` comes straight from
                            # ``Finding.redirect`` -- never parsed back out of
                            # ``message`` -- and ``timestamp`` is recorded at
                            # write time so the review packet can reason
                            # about advisory recency.
                            "redirect": f.redirect,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    + "\n"
                )
    except OSError:
        pass  # best-effort only; never block on log-write failure


def advisory_log_exists(root: Path) -> bool:
    """True when the advisories log file exists under ``root`` (issue #1460).

    Distinguishes "log unavailable" (file missing -- the review packet
    cannot compute redirects-not-taken and must say so) from "log available
    but empty" (file exists with zero/no matching records -- a legitimate
    clean-pass case) for callers of ``read_advisories``, whose own return
    value is ``()`` in both cases.
    """
    return (root / _ADVISORY_LOG_REL).is_file()


def read_advisories(root: Path) -> tuple[AdvisoryRecord, ...]:
    """Read every advisory record logged under ``root`` (issue #1460).

    Tolerant of old-shape records written before ``redirect``/``timestamp``
    existed (both default to ``None`` on ``AdvisoryRecord``), and best-effort
    on malformed lines: a line that isn't valid JSON, isn't an object, or is
    missing a required field is skipped rather than raising -- this reader
    feeds an advisory-only review section, never a blocking gate. A missing
    log file yields an empty tuple (nothing to report, not an error).
    """
    log_path = root / _ADVISORY_LOG_REL
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()

    records: list[AdvisoryRecord] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        record = _advisory_record_from_raw(raw)
        if record is not None:
            records.append(record)
    return tuple(records)


def parse_advisories_comment(body: str) -> tuple[AdvisoryRecord, ...] | None:
    """Parse the worker-published advisories PR comment (issue #1466).

    The comment body MUST start (after leading whitespace) with
    ``ADVISORY_COMMENT_MARKER``; a body that does not is not an advisories
    comment and this returns ``None`` -- the caller treats that as "no PR-
    comment channel present" and falls back to the local advisories log.

    The marker is followed by a fenced JSON block (``\\`\\`\\`json ... \\`\\`\\`\\``)
    containing a JSON array of records in the same schema
    ``read_advisories`` parses (one object per advisory, fields
    ``severity``/``file``/``identity``/``message``/``redirect``/``timestamp``).
    Returns the parsed ``AdvisoryRecord`` tuple, or ``()`` when the marker is
    present but the fence is missing/empty/malformed -- a present marker with
    no parseable records is still a present channel (distinguished from
    ``None``, which means "no channel at all"), so the caller does NOT fall
    back to the local log and the review packet renders an empty
    redirects-not-taken list rather than the "log not available" NOTE.

    Best-effort on malformed content, mirroring ``read_advisories``: a fence
    whose body is not valid JSON, is not an array, or contains non-object /
    missing-field elements yields the records that DID parse (possibly
    ``()``), never raises -- this feeds an advisory-only review section.
    """
    if not isinstance(body, str) or not body.lstrip().startswith(ADVISORY_COMMENT_MARKER):
        return None

    # Extract the first fenced block after the marker. A fenced block opens
    # with a line of three-or-more backticks (optionally tagged ``json``) and
    # closes with the next line of three-or-more backticks. Anything between
    # is the JSON payload. Tolerant of ``\\r\\n`` line endings.
    lines = body.splitlines()
    fence_open_index: int | None = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fence_open_index = i
            break
    if fence_open_index is None:
        return ()
    fence_body: list[str] = []
    for line in lines[fence_open_index + 1 :]:
        if line.lstrip().startswith("```"):
            break
        fence_body.append(line)
    payload_text = "\n".join(fence_body).strip()
    if not payload_text:
        return ()
    try:
        decoded = json.loads(payload_text)
    except json.JSONDecodeError:
        return ()
    if not isinstance(decoded, list):
        return ()
    records: list[AdvisoryRecord] = []
    for element in decoded:
        record = _advisory_record_from_raw(element)
        if record is not None:
            records.append(record)
    return tuple(records)


def _extract_file_path(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    file_path = tool_input.get("file_path")
    return file_path if isinstance(file_path, str) and file_path else None


def _apply_string_edit(content: str, old: str, new: str, replace_all: bool) -> str:
    if replace_all:
        return content.replace(old, new)
    index = content.find(old)
    if index == -1:
        return content
    return content[:index] + new + content[index + len(old) :]


def _compute_proposed_content(
    tool_name: object, tool_input: object, current_text: str | None
) -> str | None:
    """Reconstruct the file's content as it will read AFTER this PreToolUse
    edit lands, so the hook can evaluate the pending edit instead of the
    stale pre-edit on-disk file (PreToolUse fires before the write happens).

    Returns None when the proposed content cannot be determined (unknown
    tool, malformed payload, an `old_string` that doesn't match the current
    text, or -- for Edit/MultiEdit -- no on-disk base to apply against yet):
    the caller falls back to the current on-disk check in that case, same as
    before this fix, rather than guessing.
    """
    if not isinstance(tool_input, dict):
        return None
    if tool_name == "Write":
        content = tool_input.get("content")
        return content if isinstance(content, str) else None
    if current_text is None:
        return None
    if tool_name == "Edit":
        old = tool_input.get("old_string")
        new = tool_input.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            return None
        return _apply_string_edit(current_text, old, new, bool(tool_input.get("replace_all")))
    if tool_name == "MultiEdit":
        edits = tool_input.get("edits")
        if not isinstance(edits, list):
            return None
        text = current_text
        for edit in edits:
            if not isinstance(edit, dict):
                return None
            old = edit.get("old_string")
            new = edit.get("new_string")
            if not isinstance(old, str) or not isinstance(new, str):
                return None
            text = _apply_string_edit(text, old, new, bool(edit.get("replace_all")))
        return text
    return None


def main(
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    in_stream = stdin if stdin is not None else sys.stdin
    out_stream = stdout if stdout is not None else sys.stdout
    err_stream = stderr if stderr is not None else sys.stderr
    environ = env if env is not None else os.environ

    try:
        payload = json.load(in_stream)
    except (json.JSONDecodeError, ValueError):
        return 0  # malformed input -> fast no-op, never block on our own bug

    file_path = _extract_file_path(payload)
    if file_path is None:
        return 0

    target = Path(file_path)
    root = _find_baseline_root(target)
    if root is None:
        return 0  # no piloted repo above this file -> fast no-op

    try:
        rel_path = target.resolve().relative_to(root).as_posix()
    except ValueError:
        return 0  # target somehow outside root

    current_text: str | None
    try:
        current_text = (root / rel_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        current_text = None  # new file, or unreadable -- fall back gracefully below

    proposed_text = _compute_proposed_content(
        payload.get("tool_name") if isinstance(payload, dict) else None,
        payload.get("tool_input") if isinstance(payload, dict) else None,
        current_text,
    )
    content_overrides = {rel_path: proposed_text} if proposed_text is not None else None

    try:
        findings = check_file(rel_path, root, content_overrides=content_overrides)
    except Exception:
        # Fail-open against ANY unforeseen scan error, not only the decode/OS
        # cases archetypes.py now guards itself: the hook's contract is
        # advisory-only and must never crash into a blocking exit code.
        return 0
    actionable = [f for f in findings if f.severity in _ACTIONABLE_SEVERITIES]
    if not actionable:
        return 0

    unattended = _is_unattended(environ)
    mode = _resolve_mode(root, environ)
    message = _format_message(actionable)

    if unattended or mode != "enforce":
        out_stream.write(json.dumps({"hookSpecificOutput": {"additionalContext": message}}) + "\n")
        _append_advisory_log(root, actionable)
        return 0

    err_stream.write(message + "\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
