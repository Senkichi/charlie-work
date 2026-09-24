"""Post-hoc ``worker_literal_tmp_path`` signal for literal ``/tmp`` misuse (issue #1780).

#1767 retargeted ``TMP``/``TEMP``/``TMPDIR`` at a worktree-local directory,
but a literal ``/tmp/...`` path typed into a shell command still resolves
through MSYS's install-wide cached mount under Git Bash (and to the single
shared temp dir on POSIX) — a directory every concurrent worker session on
the host can read and overwrite, which is how a
``gh pr view ... > /tmp/pr-body.md`` scratch write got corrupted by a
sibling worker. The rendered prompt now forbids the literal path
(``worker_sections/session_scratch_dir.md``, kept present by
``prompts.assert_session_scratch_dir`` at the dispatch boundary); this
module is the other half of the issue — the structural record when a
session did it anyway.

Two surfaces:

* :func:`literal_tmp_shell_commands` scans a session's stream-json
  transcript — via ``claude_code.iter_claude_events``, the shared JSONL
  parsing primitive — for ``tool_use`` blocks carrying a shell ``command``
  with a literal ``/tmp`` token.
* :func:`emit_literal_tmp_path_warning` is the dead-session lane's single
  entry point, called by
  ``dead_worker_reap._classify_dead_sessions_and_update_throttle_state``
  after the sidecar is reaped: it runs the scan for claude-code/api
  workers and emits one warning-level ``worker_literal_tmp_path`` event
  per offending session.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from .claude_code import iter_claude_events
from .state import load_state, state_lock

if TYPE_CHECKING:
    from .worker import WorkerView
    from .write_gate import WriteGate


# Issue #1780: boundary for a literal ``/tmp`` path token inside a shell
# command string. #1767 retargeted TMP/TEMP/TMPDIR at a worktree-local
# directory, but MSYS resolves ``/tmp`` through its install-wide cached
# mount regardless of environment, so a worker that types the path anyway
# lands in a directory shared with every concurrent session on the host
# (the ``gh pr view ... > /tmp/pr-body.md`` incident). The lookbehind
# rejects positions where ``/tmp`` is not a standalone absolute-path
# token -- ``foo/tmp`` (word char), ``./tmp``/``../tmp`` (``.``),
# ``~/tmp`` (``~``), ``${BASE}/tmp`` (``}``), ``$X/tmp`` (``$``), and
# ``//tmp`` (``/``) -- and the lookahead rejects longer names like
# ``/tmpdir``, ``/tmp-x``, and ``/tmp.bak``. Compliant
# ``$TMPDIR``/``${TMPDIR}`` command forms never contain the literal
# substring, so the instructed form cannot trip this; ``${TMPDIR:-/tmp}``
# does match -- correctly, since it hard-codes the shared directory as a
# fallback (a ``-`` lookbehind would exempt exactly that shell idiom).
_LITERAL_TMP_PATH = re.compile(r"(?<![\w$}.~/])/tmp(?![\w.-])")


def literal_tmp_shell_commands(events_source: Path | str) -> list[str]:
    """Shell commands in a Claude Code stream-json transcript that use a literal ``/tmp`` path.

    Post-hoc signal for the residual class issue #1767 could not close:
    ``sanitize_env`` points TMP/TEMP/TMPDIR at a per-session directory,
    but under Git Bash a literal ``/tmp/...`` token resolves through
    MSYS's install-wide cached mount (and on POSIX to the one shared temp
    dir), so a worker that types the path anyway still lands in a
    directory every concurrent worker session on the host can read and
    overwrite. The rendered prompt now forbids the literal path; this
    scan is the structural record when a session did it anyway.

    ``events_source`` is anything :func:`iter_claude_events` can read -- a
    tee'd ``.events.jsonl`` sidecar, or the session's ``.claude.log``,
    which carries the identical stream-json when ``tee_stream_json`` is
    enabled (always true for api sessions). A plain-text log from a
    non-tee session yields no events, so the scan is a silent no-op there;
    the prompt instruction is the primary mitigation for that gap.

    Only ``assistant`` events' ``tool_use`` blocks carrying an
    ``input.command`` string are inspected -- the shell-command surface
    (``Bash`` today, name-agnostic by shape so a renamed or future shell
    tool is still caught). Assistant *text* and ``result`` prose are
    deliberately excluded: a worker explaining why it avoided ``/tmp``
    must not trip the detector.

    Returns the offending command strings in transcript order; an empty
    list when the file is absent or carries no hits (absence is a valid
    state, matching ``iter_claude_events``' contract). Never raises on
    unusable input — a ``Path`` conversion failure is treated like file
    absence — so diagnostic callers can scan unconditionally.
    """
    hits: list[str] = []
    try:
        path = events_source if isinstance(events_source, Path) else Path(events_source)
    except (TypeError, ValueError):
        return hits
    for event in iter_claude_events(path):
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_input = block.get("input")
            command = tool_input.get("command") if isinstance(tool_input, dict) else None
            if isinstance(command, str) and _LITERAL_TMP_PATH.search(command):
                hits.append(command)
    return hits


def emit_literal_tmp_path_warning(
    state_file: Path, worker: "WorkerView", write_gate: "WriteGate"
) -> None:
    """Emit ``worker_literal_tmp_path`` once for a just-reaped session that used ``/tmp``.

    The dead-session lane's single entry point for the issue #1780 post-hoc
    signal. ``worker`` is the ``WorkerView`` whose sidecar was just reaped
    by the caller — the event therefore fires once per session by
    construction, since no later pass can re-see this worker.

    Only ``claude-code``/``api`` sessions are scanned: they are the
    Git-Bash shell-command surface the hazard applies to, and the only
    adapters whose logs carry stream-json. The scan reads the session's
    own log via :func:`iter_claude_events`: api sessions always tee
    stream-json (claude-code sessions when ``tee_stream_json`` is
    configured), exposing the Bash ``tool_use`` commands; a non-tee
    plain-text log yields no tool calls and the scan is a silent no-op
    there — the prompt instruction is the primary mitigation for that
    gap. The scan never raises (unusable input is an empty result), so it
    cannot break the reap pass.

    The event is warning-level, not error: the reaped session's own work
    may be fine — the hazard is cross-session scratch-file corruption on
    a *sibling* lane, so this is a diagnosable record for post-incident
    triage and drift measurement, not a terminal verdict on the worker.
    The consumer is ``heartbeat_check.py``'s ``check_warning_events``,
    which reads every ``level='warning'`` row.
    """
    if worker.adapter_kind not in ("claude-code", "api"):
        return
    tmp_commands = literal_tmp_shell_commands(worker.log_path)
    if not tmp_commands:
        return
    with state_lock(state_file):
        state = load_state(state_file)
        state = write_gate.append_event(
            state,
            "worker_literal_tmp_path",
            {
                "issue_number": worker.issue_number,
                "adapter_kind": worker.adapter_kind,
                "pid": worker.pid,
                "log_path": worker.log_path,
                "command_count": len(tmp_commands),
                "commands": [c[:300] for c in tmp_commands[:3]],
            },
            level="warning",
        )
        write_gate.save_state(state)
