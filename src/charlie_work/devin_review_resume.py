"""Resume a devin-shell review session that ended on a refused exec (issue #2090).

Headless ``devin --print`` auto-rejects any exec outside the read-only
allow-list and the rejection ENDS the session with no verdict. Prompt wording
(#2024, #2032) failed with two model families, so a refusal has to be
recoverable: the reaper relaunches the SAME Devin session with
``--resume <id> --print`` and a nudge, at most
``review_dispatch.review_exec_rejection_max_resumes`` times per dispatch.

Session id source: ``devin list --format json`` run in the review checkout's
cwd. It is Devin's own public CLI, scoped to the cwd (one per-PR checkout), and
returns each session's id and ``last_activity_at``. The per-pid logs under
``%APPDATA%/devin/cli/logs`` are NOT usable: the pid ``Popen`` returns is the
REPL parent, whose log never names the session (the ACP child's log does, under
a different pid).

Invariants: the relaunch is a non-blocking ``Popen`` (no wait/communicate);
every failure comes back as a value (``False`` plus a
``review_exec_rejection_resume_failed`` event), never a raise; the sidecar is
rewritten atomically over the SAME ``issue-<pr>.json`` so the resumed process
occupies the same review slot -- ``_count_live_reviews`` and the fleet reviewer
cap count sidecars/pids, so a resume replaces the dead session and is never a
second live reviewer.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import launch_events
from .atomic_write import write_text_atomic
from .claude_code import _events_path, _rotate_old_log
from .devin_review_mode import (
    _review_exec_commands_text,
    _sanitize_review_command_template,
    _write_review_permissions,
)
from .devin_shell import (
    _get_process_start_time,
    _sidecar_path,
    _write_json,
    read_session_records,
)
from .devin_terminal_record import maybe_start_terminal_status_watcher
from .env_sanitize import sanitize_env
from .process_utils import CpuPriority, popen_worker, worker_terminal_status_path
from .state import utc_now
from .subprocess_runner import run_captured
from .verdict_parsing import CAUSE_REVIEWER_EXEC_REJECTED, _extract_terminating_cause
from .worktree import write_worktree_marker

logger = logging.getLogger(__name__)

_LIST_TIMEOUT_SECONDS = 30
_ACTIVITY_SLACK_SECONDS = 2


def review_exec_nudge_text() -> str:
    """The message sent into the resumed session; allow-list rendered from the single source."""
    return (
        "Your last shell command was REFUSED: this session is headless and cannot "
        "approve commands. Do not retry it.\n\n"
        f"The ONLY shell commands you may run: {_review_exec_commands_text()}. "
        "Run nothing else -- no tests, interpreters, package managers, other `git` "
        "subcommands, `sed`, `awk` or `jq`. Read files with your file-reading and "
        "search tools; the review packet already holds the diff, metadata and CI "
        "status.\n\n"
        "Finish the review now from what you have already read and emit your final "
        "verdict in the fenced JSON format your original instructions require.\n"
    )


def find_devin_session_id(
    devin_bin: str, checkout: Path, *, started_epoch: float | None
) -> str | None:
    """Id of the newest Devin session in ``checkout`` active since ``started_epoch``.

    ``devin list`` is cwd-scoped and review checkouts are per-PR, but a PR's
    checkout path is reused across review rounds, so the activity gate keeps an
    earlier round's session from being picked. Returns ``None`` (never raises)
    when the CLI fails, prints non-JSON, or no session qualifies.
    """
    result = run_captured(
        [devin_bin, "list", "--format", "json"],
        cwd=checkout,
        timeout_seconds=_LIST_TIMEOUT_SECONDS,
    )
    if not result.ok:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    floor = math.floor(started_epoch) - _ACTIVITY_SLACK_SECONDS if started_epoch else None
    best: tuple[int, str] | None = None
    for row in payload:
        if not isinstance(row, dict):
            continue
        session_id = row.get("id")
        activity = row.get("last_activity_at")
        if not isinstance(session_id, str) or not session_id:
            continue
        if isinstance(activity, bool) or not isinstance(activity, int):
            continue
        if floor is not None and activity < floor:
            continue
        if best is None or activity > best[0]:
            best = (activity, session_id)
    return best[1] if best else None


def build_resume_command(
    command: tuple[str, ...], session_id: str, nudge_path: Path
) -> tuple[str, ...]:
    """Recorded launch argv -> ``... --resume <id> --prompt-file <nudge>``.

    Drops the original ``--prompt-file``/``--resume`` and re-sanitizes, so the
    no-``--permission-mode`` review posture is re-pinned on this path too.
    """
    kept: list[str] = []
    skip_next = False
    for token in _sanitize_review_command_template(command):
        if skip_next:
            skip_next = False
            continue
        if token in ("--prompt-file", "--resume", "-r"):
            skip_next = True
            continue
        if token.startswith(("--prompt-file=", "--resume=")):
            continue
        kept.append(token)
    return (*kept, "--resume", session_id, "--prompt-file", str(nudge_path))


def _model_from_command(command: tuple[str, ...] | list[str]) -> str:
    """The ``--model <x>`` / ``--model=<x>`` a recorded launch argv pinned.

    The resumed session must run the same model the original dispatch rendered
    (for a reviewer, the role-chain's reviewer entry) -- read it back from the
    sidecar rather than re-deriving it from the live config, which may have
    changed since dispatch.
    """
    for index, token in enumerate(command):
        if token == "--model" and index + 1 < len(command):
            return str(command[index + 1])
        if token.startswith("--model="):
            return token.split("=", 1)[1]
    return ""


def _commit(app: Any, apply: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    """Load state under the lock, apply ``apply`` (which appends the event), and save."""
    from . import workflow as _wf  # state_lock is patched on the workflow namespace

    with _wf.state_lock(app.paths.state_file):
        state = _wf.load_state(app.paths.state_file)
        app.write_gate.save_state(apply(state))


def resume_exec_rejected_review(app: Any, worker: Any, pr_number: int, reviews_dir: Path) -> bool:
    """Try to resume a dead, verdict-less devin review that ended on an exec rejection.

    Returns True when the same session was relaunched (the caller must skip the
    miss path for this PR this pass); False for every other outcome, in which
    case the caller proceeds exactly as before (``review_verdict_missed``).
    """
    max_resumes = app.config.review_dispatch.review_exec_rejection_max_resumes
    if max_resumes <= 0 or worker.adapter_kind != "devin":
        return False
    log_path = Path(worker.log_path)
    cause = _extract_terminating_cause(_events_path(reviews_dir, pr_number, review=True), log_path)
    if cause.get("cause") != CAUSE_REVIEWER_EXEC_REJECTED:
        return False

    from . import workflow as _wf

    with _wf.state_lock(app.paths.state_file):
        pr_state = _wf.load_state(app.paths.state_file).get("prs", {}).get(str(pr_number), {})
    # The resume budget is per dispatch: keyed to ``review_dispatched_at``, which
    # every fresh launch re-stamps, so a new dispatch starts at zero with no
    # reset hook to forget.
    dispatched_at = pr_state.get("review_dispatched_at")
    same_dispatch = pr_state.get("review_exec_resume_dispatched_at") == dispatched_at
    prior = int(pr_state.get("review_exec_resume_count") or 0) if same_dispatch else 0
    if prior >= max_resumes:
        return False
    attempt = prior + 1
    cached_id = pr_state.get("review_exec_resume_session_id") if same_dispatch else None

    def _fail(reason: str, *, error_class: str | None = None) -> bool:
        _commit(
            app,
            lambda state: app.write_gate.append_event(
                state,
                "review_exec_rejection_resume_failed",
                {"pr_number": pr_number, "attempt": attempt, "reason": reason},
            ),
        )
        # Issue #2246: an error_class marks a resume path that reached the
        # launch attempt (env prep or Popen) and came back errored -- emit one
        # launch_failed at this seam. The precondition declines above
        # (sidecar_unreadable / review_checkout_missing / session_id_unavailable)
        # never attempted a launch and stay off this event. The model is read
        # back out of the sidecar's recorded argv -- the role-chain entry the
        # original dispatch actually rendered (reviewer.model, not worker.model).
        if error_class is not None:
            launch_events.emit_launch_failed(
                app.paths.state_file,
                role="reviewer",
                harness="devin-shell",
                model=_model_from_command(record.command),
                pr_number=pr_number,
                error_class=error_class,
                error=reason,
            )
        return False

    record = next(
        (r for r in read_session_records(reviews_dir) if r.issue_number == pr_number), None
    )
    if record is None or not record.command:
        return _fail("sidecar_unreadable")
    from . import worker_fate

    if worker_fate.is_alive(record.pid, record.process_start_time):
        # The caller's ``worker`` is a snapshot; the freshly-read sidecar names a
        # live process, i.e. a concurrent reaper already resumed this session
        # (issue #2110). Never relaunch over it, and tell the caller to skip the
        # miss path (which would tear down the live session's checkout).
        return True
    checkout = Path(record.worktree_path)
    if not checkout.is_dir():
        return _fail("review_checkout_missing")
    session_id = cached_id or find_devin_session_id(
        record.command[0], checkout, started_epoch=record.process_start_time
    )
    if not session_id:
        return _fail("session_id_unavailable")

    nudge_path = reviews_dir / f"issue-{pr_number}.resume-nudge.md"
    try:
        write_text_atomic(nudge_path, review_exec_nudge_text())
        worker_env = app._adapter_settings(adapter="devin-shell").worker_env
        env = {**sanitize_env(checkout), **{str(k): str(v) for k, v in worker_env.items()}}
    except OSError as exc:
        return _fail(f"prepare_failed: {exc}", error_class=launch_events.LAUNCH_ERR_ENV)

    command = build_resume_command(record.command, session_id, nudge_path)
    _write_review_permissions(checkout)
    _rotate_old_log(log_path)
    worker_terminal_status_path(reviews_dir, pr_number, "devin").unlink(missing_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            process = popen_worker(
                list(command),
                priority=CpuPriority.BELOW_NORMAL,
                cwd=str(checkout),
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
            )
    except OSError as exc:
        # The miss path classifies the cause from this log, so put the original
        # session's log back rather than leave the empty one we just opened.
        rotated = log_path.with_suffix(log_path.suffix + ".1")
        try:
            if rotated.exists():
                log_path.unlink(missing_ok=True)
                rotated.rename(log_path)
        except OSError:
            pass
        return _fail(f"launch_failed: {exc}", error_class=launch_events.LAUNCH_ERR_SPAWN)

    pid = process.pid
    start_time = _get_process_start_time(pid)
    maybe_start_terminal_status_watcher(process, reviews_dir, pr_number, worktree_path=None)
    try:
        write_worktree_marker(
            checkout, pid, record.session_id or str(uuid.uuid4()), process_start_time=start_time
        )
    except OSError:
        pass
    _write_json(
        _sidecar_path(reviews_dir, pr_number),
        replace(
            record,
            command=command,
            pid=pid,
            started_at=utc_now(),
            process_start_time=start_time,
            error=None,
        ).to_dict(),
    )

    relaunched_at = utc_now()

    def _bump(entry: dict[str, Any]) -> dict[str, Any]:
        return {
            **entry,
            # Re-stamp the claim age to the relaunch (issue #2162) so the stall
            # sweep does not judge the resumed session by the original dispatch
            # time. The resume-budget key moves with it, keeping the count.
            "review_dispatched_at": relaunched_at,
            "reviewer_pid": pid,
            "reviewer_process_start_time": start_time,
            "review_exec_resume_dispatched_at": relaunched_at,
            "review_exec_resume_count": attempt,
            "review_exec_resume_session_id": session_id,
        }

    def _apply(state: dict[str, Any]) -> dict[str, Any]:
        prs = dict(state.get("prs") or {})
        prs[str(pr_number)] = _bump(dict(prs.get(str(pr_number)) or {}))
        return app.write_gate.append_event(
            {**state, "prs": prs},
            "review_exec_rejection_resumed",
            {"pr_number": pr_number, "attempt": attempt, "session_id": session_id},
        )

    _commit(app, _apply)
    return True
