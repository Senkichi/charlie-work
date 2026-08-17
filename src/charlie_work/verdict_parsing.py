"""Reviewer-verdict-parsing free-function family (issue #1283 Phase A).

Extracted verbatim from ``workflow.py``: the free-function family that parses
a reviewer session's raw text/JSON output into a structured verdict (fenced
JSON extraction, stream-json event decoding, mtime-gated file fallback
recovery), plus the reviewer-session-summary reconstruction used when no
verdict is found and the frozen dataclass that summary returns.
``workflow.py`` re-exports every symbol here via a facade import block
(mirroring ``config.py``'s ``RunnerAllocationConfig`` re-export pattern and
this repo's own ``dispatch_selection.py``/``escalation.py`` precedents), so
existing import paths and monkeypatch targets keep working unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from .claude_code import extract_event_text, iter_stream_json_events, parse_claude_events
from .config import OrchestratorConfig
from .throttle_signatures import match_throttle_tail

# Language-tag group accepts any tag (not just ``json``), mirroring the fix in
# ``cross_family._VERDICT_FENCE_RE``: a fence opened with an unrecognized tag
# (e.g. ```python) previously failed to match as an opening delimiter at all,
# causing its own closing ``` to be misread as a spurious new opening and
# desynchronizing every fence pair after it. In practice this path is
# protected here because each stream-json event's text is checked in
# isolation (see ``_extract_verdict_from_stream_json``) and the reviewer's
# final verdict fence normally lands in its own turn, separate from any
# earlier code-citation turns -- but the defect is real and latent, so it is
# fixed here too rather than left to fire the day a reviewer's final message
# happens to combine both.
_VERDICT_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)

# Absolute path ending in .md, as reviewers reference their summary files in
# final output (e.g. "Full review written to `C:\...\review.md`"). Colons,
# quotes, and whitespace terminate the match so "path:line" refs don't bleed.
_REVIEW_MD_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/]|~[\\/]|/)[^\s`\"'<>|*?:]+\.md")

_REVIEW_FALLBACK_FILE_MAX_BYTES = 1024 * 1024
_REVIEW_FALLBACK_MTIME_SLACK_S = 120
_REVIEW_FALLBACK_MAX_CANDIDATES = 8


def _validate_review_verdict(data: Any) -> dict[str, Any] | None:
    """Validate one decoded JSON candidate as a review verdict.

    A valid verdict must contain:

    - ``decision`` in ``{"approved", "request_changes", "blocked"}``
    - ``summary`` as a non-empty string, for EVERY decision including
      ``approved`` (issue #597), and not an unfilled ``<...>`` template
      placeholder
    - ``required_changes``, if present, must be a list of strings. This
      function does not require it to be non-empty: an empty or absent
      ``required_changes`` on a ``request_changes``/``blocked`` verdict is
      repaired downstream, at ``record_review`` (issue #792), which derives
      it from ``summary`` or marks it ``findings_channel: "vacuous"`` when
      nothing is derivable -- never by rejecting the verdict here. Rejecting
      an empty ``required_changes`` at this layer would recreate the
      unbounded re-review loop that gate is designed to avoid: see
      ``record_review``'s derivation block and
      ``test_record_review_never_rejects_for_empty_required_changes``.

    ``approved`` used to be exempt from the non-empty-summary rule, on the
    reasoning that ``record_review`` only rejects empty summaries where a
    reason is actionable. That exemption is what let a contentless approval
    through: an approval with no stated reason is indistinguishable from a
    reviewer that never formed an opinion, and approvals are the one decision
    that leads directly to a merge. Requiring a reason costs a reviewer one
    sentence; not requiring it cost ten unreviewed merges. A rejected verdict
    is fail-safe here -- the caller records no verdict and the review is
    retried, rather than merging on a verdict nobody stands behind.

    Returns the normalized verdict dict, or ``None`` if invalid.
    """
    if not isinstance(data, dict):
        return None

    decision = data.get("decision")
    if decision not in {"approved", "request_changes", "blocked"}:
        return None

    summary = data.get("summary")
    if not isinstance(summary, str):
        return None
    stripped_summary = summary.strip()
    if not stripped_summary:
        return None
    # An unfilled template placeholder ("<concise summary of the review>") is
    # prompt boilerplate that leaked into the verdict, never a real summary.
    if stripped_summary.startswith("<") and stripped_summary.endswith(">"):
        return None

    required_changes = data.get("required_changes")
    if required_changes is not None and not isinstance(required_changes, list):
        return None
    if required_changes is not None and not all(
        isinstance(item, str) for item in required_changes
    ):
        return None

    return {
        "decision": decision,
        "summary": summary,
        "required_changes": required_changes if required_changes is not None else [],
    }


def _extract_verdict_from_text(text: str) -> dict[str, Any] | None:
    """Extract the last valid fenced JSON verdict block from plain text.

    Accepts fences with or without a ``json`` language tag, scanning from the
    last fence (the final output) backwards.
    """
    for match in reversed(list(_VERDICT_FENCE_RE.finditer(text))):
        candidate = match.group(1).strip()
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        verdict = _validate_review_verdict(data)
        if verdict is not None:
            return verdict
    return None


def _extract_verdict_from_stream_json(raw_text: str) -> dict[str, Any] | None:
    """Extract a verdict from tee'd stream-json JSONL text.

    With ``tee_stream_json`` enabled the sidecar log is JSONL where every
    fence lives *inside* a JSON string (``\\n`` as escape sequences), so a
    regex over the raw text can never match. Decode the events and accept a
    fence ONLY from the final output: the ``result`` event's text, or —
    absent a usable one (session killed before the result line) — the single
    last assistant text. Never scan further back: a mid-session draft or an
    echo of the review prompt's own few-shot example (which contains a
    literal ``"decision": "approved"`` fence) must not be recorded as the
    session's verdict when the reviewer produced no final one. A fence-less
    final output returns ``None`` so the caller's no-verdict path
    (turn-limit summary + retry) handles it as designed.
    """
    result_text: str | None = None
    last_assistant_text: str | None = None
    for event in iter_stream_json_events(raw_text):
        text = extract_event_text(event)
        if not text:
            continue
        if event.get("type") == "result":
            result_text = text
        else:
            last_assistant_text = text

    for text in (result_text, last_assistant_text):
        if text:
            verdict = _extract_verdict_from_text(text)
            if verdict is not None:
                return verdict
    return None


def _parse_review_verdict_from_log(log_path: Path) -> dict[str, Any] | None:
    """Extract a fenced JSON verdict block from a reviewer's sidecar log.

    Handles both log formats: plaintext logs (fences matched directly) and
    stream-json JSONL logs produced by ``tee_stream_json`` (fences are
    JSON-escaped inside event strings, so events are decoded first). Returns
    the parsed dict on success, or ``None`` if no valid block is found.
    Malformed/truncated logs and 0-byte logs both return ``None``.
    """
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    verdict = _extract_verdict_from_text(log_text)
    if verdict is not None:
        return verdict
    return _extract_verdict_from_stream_json(log_text)


def _parse_review_verdict_from_events(events_path: Path) -> dict[str, Any] | None:
    """Extract a fenced JSON verdict block from a reviewer's events.jsonl.

    Fallback for when ``_parse_review_verdict_from_log`` fails: the log may be
    truncated or the verdict block split across tee buffer boundaries, but the
    structured events.jsonl carries the assistant's text in discrete JSONL
    lines. Decodes real stream-json events (``assistant``/``result``) as well
    as the legacy ``assistant_message`` shape.

    Returns the parsed dict on success, or ``None`` if no valid block is found.
    """
    try:
        raw_text = events_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return _extract_verdict_from_stream_json(raw_text)


def _parse_review_verdict_from_files(
    log_path: Path,
    packet_dir: Path,
    started_at: str | None,
) -> tuple[dict[str, Any], str] | None:
    """Last-resort verdict recovery from files the reviewer wrote (issue #566).

    Reviewers sometimes write their review summary (verdict block included) to
    a Markdown file and merely *reference* it in final output instead of
    re-emitting the fenced JSON. Before counting a completed review as a
    failed attempt, scan ``.md`` paths mentioned in the reviewer's decoded
    output text, newest-mention-first.

    **Nothing inside ``packet_dir`` is ever a candidate (issue #597).** Review
    sessions launch with a hard-pinned ``--permission-mode plan`` (see
    ``claude_code._force_review_permission_mode``), so a reviewer cannot write
    any file, anywhere — every file in the packet directory is authored by the
    orchestrator itself. ``review-prompt.md`` is one of them, and it embeds an
    example verdict block. Globbing ``packet_dir`` for ``*.md`` therefore did
    not recover reviewer verdicts; it parsed the orchestrator's own
    instructions and recorded whatever the example said. Because the example
    read ``"decision": "approved"``, a reviewer that emitted no verdict had an
    approval manufactured for it, which then took the merge label. Ten PRs
    across two repos merged unreviewed that way. ``_extract_verdict_from_stream_json``
    already guarded against this exact echo; this path was added later and
    bypassed that guard.

    ``packet_dir`` is still taken as a parameter because it defines the
    exclusion zone: a reviewer that *mentions* a packet path in its prose must
    not pull the prompt back in through the mention branch either.

    Every candidate is mtime-gated to the reviewer session's ``started_at``
    (minus slack): a stale review file from a previous round must never
    resurrect an old verdict for a new head. Without a parseable
    ``started_at`` there is no safe gate, so no fallback is attempted.

    Returns ``(verdict, source_path)`` or ``None``.
    """
    if not started_at:
        return None
    try:
        cutoff = datetime.fromisoformat(started_at.replace("Z", "+00:00")) - timedelta(
            seconds=_REVIEW_FALLBACK_MTIME_SLACK_S
        )
    except ValueError:
        return None
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=UTC)

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log_text = ""

    candidates: list[Path] = []
    seen: set[str] = set()

    texts = [
        text
        for text in (extract_event_text(event) for event in iter_stream_json_events(log_text))
        if text
    ]
    if not texts and log_text:
        texts = [log_text]
    for text in reversed(texts):
        for match in _REVIEW_MD_PATH_RE.finditer(text):
            raw = match.group(0)
            if raw not in seen:
                seen.add(raw)
                candidates.append(Path(raw).expanduser())

    # Issue #597: the packet directory is orchestrator-authored territory (see
    # this function's docstring). Never read anything inside it -- not via a
    # glob, and not via a path the reviewer happened to mention in its prose.
    try:
        excluded_root = packet_dir.resolve()
    except OSError:
        excluded_root = packet_dir

    def _inside_packet_dir(candidate: Path) -> bool:
        try:
            resolved = candidate.resolve()
        except OSError:
            return False
        return resolved == excluded_root or excluded_root in resolved.parents

    # Stat-filter BEFORE capping: the cap bounds expensive file reads, and
    # spurious path-looking mentions in the reviewer's text (nonexistent,
    # stale, oversized) must not starve genuine candidates out of the read
    # budget.
    readable: list[Path] = []
    for candidate in candidates:
        if _inside_packet_dir(candidate):
            continue
        try:
            stat = candidate.stat()
        except OSError:
            continue
        if not candidate.is_file():
            continue
        if stat.st_size > _REVIEW_FALLBACK_FILE_MAX_BYTES:
            continue
        if datetime.fromtimestamp(stat.st_mtime, tz=UTC) < cutoff:
            continue
        readable.append(candidate)
        if len(readable) >= _REVIEW_FALLBACK_MAX_CANDIDATES:
            break

    for candidate in readable:
        try:
            content = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        verdict = _extract_verdict_from_text(content)
        if verdict is not None:
            return verdict, str(candidate)

    return None


def _reviewer_session_metrics(events_path: Path, verdict_source: str) -> dict[str, Any] | None:
    """Parse reviewer session telemetry for a recorded verdict (perf/spend visibility).

    Returns a dict of ``tokens``/``cost_usd``/``turn_count``/``tool_call_count``/
    ``verdict_source`` for ``record_review`` to fold into the ``record_review``
    event and the PR's state entry, or ``None`` when the events.jsonl sidecar
    is missing or unparseable (devin workers, or a claude-code session launched
    without tee_stream_json). Never raises: ``parse_claude_events`` already
    tolerates a missing/malformed file, and a missing telemetry file must never
    block recording the verdict itself.
    """
    progress = parse_claude_events(events_path)
    if progress is None:
        return None
    return {
        "tokens": progress.tokens,
        "cost_usd": progress.cost_usd,
        "turn_count": progress.turn_count,
        "tool_call_count": progress.tool_call_count,
        "verdict_source": verdict_source,
    }


# Why a reviewer session ended without a structured verdict. These are the
# ``reason`` values on ``review_verdict_missed`` events; they have disjoint
# remediations, so collapsing them into one label points every diagnostic at
# the wrong fix (issue #588).
REVIEW_MISS_TURN_LIMIT = "turn_limit_summary_posted"
REVIEW_MISS_LAUNCH_FAILED = "launch_failed"
REVIEW_MISS_DIED_MID_SESSION = "died_mid_session"

# Default markers for _extract_review_session_summary's session-limit
# reclassification (issue #651/#652). These are the NARROW
# RuntimeConfig.session_limit_markers, NOT the broad throttle_error_markers:
# reviewer launches force tee_stream_json=True (claude_code.py), making
# log_path and events_path byte-identical, so any marker matched against the
# log tail is also present in the parsed assistant text. The generic markers
# in throttle_error_markers ("rate limit", "usage limit") legitimately appear
# in this codebase's rate-limit/quota domain review commentary and would
# false-positive on real review work. session_limit_markers contains only the
# CLI's own specific session-limit death message phrasing, which is safe to
# match against reviewer text. Callers pass their config's list explicitly so
# a new session-limit phrasing only needs a config change.
_DEFAULT_REVIEW_SESSION_LIMIT_MARKERS = OrchestratorConfig().runtime.session_limit_markers

# Tail length for the raw-log session-limit match in _extract_review_session_summary.
# Mirrors the 2048-char tail used by the stalled-session sweep (the
# ``log_text[-2048:]`` slice at the ``Path(w.log_path).read_text(...)`` call in
# _handle_stalled_review_sessions): the CLI prints its session-limit notice at
# the very end of the log, so the tail isolates the death message from the
# multi-turn analysis prose earlier in the log.
_REVIEW_THROTTLE_TAIL_CHARS = 2048


def _log_tail_throttled(log_path: Path, markers: Sequence[str]) -> bool:
    """Return True when the raw process log's tail contains a session-limit marker.

    Reads the last ``_REVIEW_THROTTLE_TAIL_CHARS`` chars of ``log_path`` and
    matches against ``markers`` via ``match_throttle_tail``. This is the same
    raw-log-tail boundary the stalled-session sweep uses (the
    ``log_text[-2048:]`` slice at the ``Path(w.log_path).read_text(...)`` call
    in ``_handle_stalled_review_sessions``). Missing or unreadable logs do not
    match.

    Note (issue #652 review): because reviewer launches force
    ``tee_stream_json=True``, ``log_path`` and ``events_path`` are
    byte-identical — the raw log tail IS the events content, so matching
    against the raw tail does NOT avoid matching the reviewer's own parsed
    assistant text. The false-positive protection comes from the NARROW
    ``session_limit_markers`` list (specific CLI death-message phrasing, not
    generic domain terms), not from the raw-vs-parsed distinction. The
    ``tool_call_count == 0`` guard in the caller is defense-in-depth.
    """
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if not log_text:
        return False
    if len(log_text) > _REVIEW_THROTTLE_TAIL_CHARS:
        tail = log_text[-_REVIEW_THROTTLE_TAIL_CHARS:]
    else:
        tail = log_text
    return match_throttle_tail(tail, markers)[0]


@dataclass(frozen=True)
class ReviewSessionOutcome:
    """A reviewer session that ended without producing a structured verdict.

    ``did_substantial_work`` is the distinction that matters downstream: a
    session that completed turns and then died is a PR-level outcome (the
    review didn't fit its budget), while one that never reached its first turn
    is an environmental failure that says nothing about the PR.
    """

    text: str
    reason: str
    turn_count: int
    tool_call_count: int

    @property
    def did_substantial_work(self) -> bool:
        return self.reason != REVIEW_MISS_LAUNCH_FAILED


def _extract_review_session_summary(
    events_path: Path,
    log_path: Path,
    max_turns: int,
    *,
    session_limit_markers: Sequence[str] | None = None,
) -> ReviewSessionOutcome | None:
    """Summarize and classify a reviewer session that produced no verdict.

    When a reviewer hits the ``--max-turns`` limit (or dies for any other
    reason after doing substantial work), the structured verdict block is
    missing but the events.jsonl contains the assistant's analysis text and
    tool-call metrics. This function reconstructs a human-readable summary
    from those events so the work is not silently lost.

    It also classifies *why* the verdict is missing. A session that never
    reached its first turn did not hit a turn limit -- it never ran -- and the
    text recovered from its log is the process's own error output, not
    analysis. Reporting the two identically hid a 25-hour outage in which 19
    reviewers died on a rejected argv while every signal said "turn limit"
    (issue #588).

    A session whose only output is a provider session-limit notice (e.g.
    Claude Code's "hit your session limit") is also environmental, not a
    PR-level outcome: ``parse_claude_events`` counts that notice as one turn,
    so it fails the ``turn_count == 0`` check and would otherwise fall through
    to ``REVIEW_MISS_DIED_MID_SESSION`` -- the same bucket as a session that
    genuinely did substantial review work. That misclassification silently
    defeats the #583 throttle-rollback guard (``did_substantial_work`` reads
    ``reason != REVIEW_MISS_LAUNCH_FAILED`` and is persisted as
    ``review_turn_limit_summary_posted``), letting a global session-limit
    outage burn a PR's ``review_dispatch_attempt_count`` budget with zero
    actual review work performed (issue #651). When the raw process log's tail
    matches a session-limit marker AND the session made no tool calls, classify
    as ``REVIEW_MISS_LAUNCH_FAILED`` so the rollback guard fires.

    The match is against the raw process log tail (last 2048 chars), mirroring
    the stalled-session sweep pattern at the ``log_text =
    Path(w.log_path).read_text(...)`` call below. Because reviewer launches
    force ``tee_stream_json=True`` (claude_code.py), ``log_path`` and
    ``events_path`` are byte-identical -- the raw log tail IS the events
    content, so matching against the raw tail does NOT avoid matching the
    reviewer's own parsed assistant text (issue #652 review). The
    false-positive protection comes from the NARROW
    ``session_limit_markers`` list (specific CLI death-message phrasing like
    "hit your session limit", NOT generic domain terms like "rate limit" /
    "usage limit" that legitimately appear in this codebase's rate-limit/quota
    review commentary), and the ``tool_call_count == 0`` guard ensures a
    session that made any tool calls (real review actions) is never
    reclassified regardless of what the tail contains.

    ``session_limit_markers`` defaults to
    ``RuntimeConfig.session_limit_markers`` so the marker list stays
    config-driven (single point of enforcement in
    ``throttle_signatures.match_throttle_tail``); callers pass their config's
    list explicitly.

    Returns ``None`` if the events file is missing and the log contains no
    recoverable text (nothing to summarize).
    """
    progress = parse_claude_events(events_path)
    # Also try the plaintext log as a fallback for assistant text.
    assistant_texts: list[str] = []

    if events_path.exists():
        try:
            raw_events = events_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw_events = ""
        for event in iter_stream_json_events(raw_events):
            text = extract_event_text(event)
            if text.strip():
                assistant_texts.append(text.strip())

    # Fallback: if no events.jsonl, try the log. When it is a stream-json
    # tee, decode the events; otherwise keep the remaining prose lines with
    # fenced code blocks stripped (those are verdict attempts, not analysis).
    if not assistant_texts:
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
        for event in iter_stream_json_events(log_text):
            text = extract_event_text(event)
            if text.strip():
                assistant_texts.append(text.strip())
        if not assistant_texts:
            stripped = re.sub(r"```(?:json)?\s*\n.*?```", "", log_text, flags=re.DOTALL)
            for line in stripped.splitlines():
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith((">", "#", "-")):
                    assistant_texts.append(stripped_line)

    if not assistant_texts:
        return None

    turn_count = progress.turn_count if progress else 0
    tool_call_count = progress.tool_call_count if progress else 0
    tokens = progress.tokens if progress else None
    cost_usd = progress.cost_usd if progress else None

    markers = (
        session_limit_markers
        if session_limit_markers is not None
        else _DEFAULT_REVIEW_SESSION_LIMIT_MARKERS
    )

    # A session with no turns and no tool calls never reached its first turn:
    # the process died at launch and whatever text we recovered is its error
    # output, not reviewer analysis.
    if turn_count == 0 and tool_call_count == 0:
        reason = REVIEW_MISS_LAUNCH_FAILED
    elif max_turns > 0 and turn_count >= max_turns:
        reason = REVIEW_MISS_TURN_LIMIT
    elif tool_call_count == 0 and _log_tail_throttled(log_path, markers):
        # A session that made no tool calls but whose raw process log tail
        # contains a provider session-limit notice (e.g. "hit your session
        # limit") died on the notice, not after review work. The notice
        # gets counted as one turn by parse_claude_events, so it fails the
        # turn_count == 0 check above and would fall through to
        # REVIEW_MISS_DIED_MID_SESSION -- the same bucket as a session that
        # genuinely did substantial review work. That misclassification silently
        # defeats the #583 throttle-rollback guard (did_substantial_work reads
        # reason != REVIEW_MISS_LAUNCH_FAILED and is persisted as
        # review_turn_limit_summary_posted), letting a global session-limit
        # outage burn a PR's attempt budget with zero review work performed
        # (issue #651).
        #
        # Two boundaries make this safe (issue #652 review):
        # (1) Match only the NARROW session_limit_markers (specific CLI
        #     death-message phrasing like "hit your session limit"), NOT the
        #     broad throttle_error_markers. Reviewer launches force
        #     tee_stream_json=True, so log_path and events_path are
        #     byte-identical -- the raw log tail IS the events content, so
        #     matching the raw tail does not avoid the reviewer's own parsed
        #     assistant text. Generic markers ("rate limit", "usage limit")
        #     legitimately appear in this codebase's rate-limit/quota review
        #     commentary and would false-positive on real review work. The
        #     specific session-limit phrasing is the CLI's own death message,
        #     not a domain term, so it is safe to match against reviewer text.
        # (2) Guard on ``tool_call_count == 0``: a session that made any tool
        #     calls did real review actions and is a PR-level outcome
        #     (died_mid_session) regardless of what its log tail says -- a
        #     throttle on the final API call after real work is not a launch
        #     failure. The turn-limit branch above already owns sessions that
        #     exhausted their turn budget, so this only intercepts deaths that
        #     occurred before any tool use.
        reason = REVIEW_MISS_LAUNCH_FAILED
    else:
        reason = REVIEW_MISS_DIED_MID_SESSION

    if reason == REVIEW_MISS_LAUNCH_FAILED:
        parts = ["## Reviewer session failed to start\n"]
        parts.append(
            "The automated reviewer exited before running a single turn, so no "
            "review was performed. This is an environmental or launch failure, "
            "not a judgement about this PR.\n"
        )
        parts.append("\n### Error output from the reviewer process:\n")
    else:
        parts = ["## Reviewer session summary (no verdict produced)\n"]
        if reason == REVIEW_MISS_TURN_LIMIT:
            parts.append(
                f"The automated reviewer hit the {max_turns}-turn limit before "
                f"producing a structured verdict.\n"
            )
        else:
            parts.append(
                f"The automated reviewer ran for {turn_count} turns "
                f"({tool_call_count} tool calls) but did not produce a structured verdict.\n"
            )
        # Include the last few assistant messages — earlier turns are usually
        # tool-use planning; the final messages contain the analysis.
        parts.append("\n### Recent analysis from the reviewer:\n")

    recent = assistant_texts[-3:]
    for text in recent:
        if len(text) > 2000:
            text = text[:2000] + "\n... (truncated)"
        parts.append(text)
        parts.append("\n---\n")

    meta_parts: list[str] = []
    if turn_count:
        meta_parts.append(f"turns: {turn_count}")
    if tool_call_count:
        meta_parts.append(f"tool calls: {tool_call_count}")
    if tokens is not None:
        meta_parts.append(f"tokens: {tokens:,}")
    if cost_usd is not None:
        meta_parts.append(f"cost: ${cost_usd:.4f}")
    if meta_parts:
        parts.append(f"\n*{' · '.join(meta_parts)}*")

    return ReviewSessionOutcome(
        text="\n".join(parts),
        reason=reason,
        turn_count=turn_count,
        tool_call_count=tool_call_count,
    )
