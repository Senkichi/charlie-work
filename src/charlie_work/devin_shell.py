"""Headless Devin CLI dispatch — non-blocking session launch with a durable
sidecar so the orchestrator and ``doctor`` can see what is in flight.

There is no Devin session-creation API (per the internal extraction dossier,
"headless"/"--prompt-file"). Production reality is spawning the ``devin`` CLI
in print mode: ``devin --prompt-file <path> --print --permission-mode
dangerous``. Sessions run for many minutes, so dispatch must return immediately
after ``Popen`` — callers must never block on the worker finishing. Each launch
writes a JSON sidecar file (``sessions_dir/issue-<n>.json``) atomically (tmp +
replace, matching ``adapters._write_json``) *before* returning, so a crash of
the orchestrator process itself never loses track of a session that was actually
spawned. A daemon-thread watcher (``process_utils.start_terminal_status_watcher``,
issue #2052) then persists ``issue-<n>.devin.terminal.json`` once the spawned
process exits — exit code, duration, and a copy of the worktree's
``.worker-outcome.json`` — so ``worker_fate`` resolves a dead devin worker from
the same durable terminal record it uses for claude-code.

Each worker is launched in an isolated per-issue git worktree (created via
``worktree.create_worktree()``, mirroring the claude-code adapter) so
concurrent sessions do not contend over the shared checkout.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from charlie_work import process_utils as _process_utils
from charlie_work.process_utils import CpuPriority, popen_worker
from . import launch_events
from .atomic_write import write_json_atomic
from .claude_code import _events_path, _rotate_old_log
from .config import OrchestratorConfig
from .devin_failure_classification import (  # noqa: F401 (deliberate re-export; #1442 extraction keeps devin_shell under its mark)
    get_rate_limit_defer_until,
    update_session_record_with_failure_classification,
)
from .devin_review_mode import (  # noqa: F401 (deliberate re-export; #2069-rework extraction keeps devin_shell under its mark)
    REVIEW_EXEC_SECTION_HEADING,
    _REVIEW_COMMAND_TEMPLATE,
    _REVIEW_EXEC_ALLOWLIST,
    _review_exec_prompt_section,
    _sanitize_review_command_template,
    _write_devin_review_prompt,
    _write_review_permissions,
)
from .devin_terminal_record import maybe_start_terminal_status_watcher
from .env_sanitize import resolve_pytest_cap, resolve_uv_no_sync, sanitize_env
from .post_mortem import merge_attempt_snapshot
from .state import _canonical_started_at, utc_now
from .subprocess_runner import RunResult, run_captured
from .worktree import (
    LiveWorkerRedispatchError,
    ReworkBranchConflictError,
    WorktreeForeignWriterError,
    WorktreeInfo,
    WorktreeProbeFailedError,
    WorktreeUnsafeError,
    create_review_checkout,
    create_worktree,
    remove_review_checkout,
    remove_worktree,
    apply_rework_conflict_notice,
    write_worktree_marker,
)


logger = logging.getLogger(__name__)

# ``--permission-mode dangerous`` is required for headless workers: without it
# the Devin CLI defaults to ``auto`` (read-only tools), stalls on any
# git/uv/gh call, and exits asking the operator to restart with this flag.
# {model_args} is a placeholder for config-driven model selection (e.g.
# "--model claude-sonnet-4-5"). When worker.model is empty, this renders
# to an empty string, preserving CLI default behavior.
DEFAULT_COMMAND_TEMPLATE: tuple[str, ...] = (
    "devin",
    "{model_args}",
    "--prompt-file",
    "{prompt_path}",
    "--print",
    "--permission-mode",
    "dangerous",
    # A 2026-08 Devin CLI update enforces workspace trust in --print mode:
    # non-interactive runs cannot show the trust prompt and fail hard in any
    # untrusted directory. Worker worktrees are created fresh per issue and
    # are never interactively trusted, so every headless launch died at
    # startup (10 rework sessions across both lanes on 2026-08-08). The CLI's
    # documented remedy for exactly this case is passing the flag explicitly.
    "--respect-workspace-trust",
    "false",
)


@dataclass(frozen=True)
class SessionRecord:
    issue_number: int
    branch: str
    worktree_path: str
    prompt_path: str
    command: tuple[str, ...]
    pid: int | None
    started_at: str
    log_path: str
    error: str | None = None
    failure_kind: str | None = None  # "rate_limited" | "quota_exhausted" | ...
    process_start_time: float | None = None  # Unix timestamp in seconds (process creation time)
    reclaimed: str | None = None  # "fetch-fallback" | "pruned" | "salvaged" | None
    last_activity_at: str | None = None  # ISO timestamp from log_path.stat().st_mtime
    log_bytes: int | None = None  # log_path.stat().st_size
    attempt_ref: str | None = None  # refs/charlie/attempts/issue-<n>/attempt-<k> (issue #261)
    attempt_ahead_of_main: int | None = None  # commit count ahead of base_ref at snapshot time
    rate_limit_defer_until: str | None = (
        None  # ISO timestamp when the stall kill is deferred (issue #247)
    )
    inconclusive_probe_deferred_count: int = 0  # Signal-1 deferral counter (issue #338)
    session_id: str | None = None  # unique session id for worktree writer marker (issue #400)
    xdist_cap: str | None = None  # resolved PYTEST_XDIST_AUTO_NUM_WORKERS at launch (issue #646)
    uv_no_sync: str | None = None  # resolved UV_NO_SYNC at launch, or None if no .venv (#646)

    def __post_init__(self) -> None:
        """Enforce a canonical ISO-8601 UTC ``started_at`` at construction time."""
        canonical = _canonical_started_at(self.started_at, self.process_start_time)
        if canonical != self.started_at:
            object.__setattr__(self, "started_at", canonical)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        return payload

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> SessionRecord:
        command = payload.get("command") or []
        return SessionRecord(
            issue_number=int(payload["issue_number"]),
            branch=str(payload.get("branch", "")),
            worktree_path=str(payload.get("worktree_path", "")),
            prompt_path=str(payload.get("prompt_path", "")),
            command=tuple(str(part) for part in command),
            pid=int(payload["pid"]) if payload.get("pid") is not None else None,
            started_at=str(payload.get("started_at", "")),
            log_path=str(payload.get("log_path", "")),
            error=payload.get("error"),
            failure_kind=payload.get("failure_kind"),
            process_start_time=payload.get("process_start_time"),
            reclaimed=payload.get("reclaimed"),
            last_activity_at=payload.get("last_activity_at"),
            log_bytes=payload.get("log_bytes"),
            attempt_ref=payload.get("attempt_ref"),
            attempt_ahead_of_main=payload.get("attempt_ahead_of_main"),
            rate_limit_defer_until=payload.get("rate_limit_defer_until"),
            inconclusive_probe_deferred_count=int(
                payload.get("inconclusive_probe_deferred_count") or 0
            ),
            session_id=payload.get("session_id"),
            xdist_cap=payload.get("xdist_cap"),
            uv_no_sync=payload.get("uv_no_sync"),
        )


def _sidecar_path(sessions_dir: Path, issue_number: int) -> Path:
    return sessions_dir / f"issue-{issue_number}.json"


def _log_path(sessions_dir: Path, issue_number: int, *, rework: bool = False) -> Path:
    suffix = "-rework.log" if rework else ".log"
    return sessions_dir / f"issue-{issue_number}{suffix}"


def _read_sidecar_inconclusive_count(sessions_dir: Path, issue_number: int) -> int:
    """Read the existing sidecar's Signal-1 deferral counter, if any."""
    sidecar_path = _sidecar_path(sessions_dir, issue_number)
    try:
        with sidecar_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    raw = payload.get("inconclusive_probe_deferred_count")
    if raw is None:
        return 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _write_json(path: Path, value: Any) -> None:
    write_json_atomic(path, value)


def _render_command(
    command_template: tuple[str, ...],
    *,
    issue_number: int,
    branch: str,
    prompt_path: Path,
    worker_model: str = "",
) -> tuple[str, ...]:
    model_args = f"--model {worker_model}" if worker_model else ""
    values = {
        "prompt_path": str(prompt_path),
        "issue_number": str(issue_number),
        "branch": branch,
        "model_args": model_args,
    }
    rendered = tuple(part.format(**values) for part in command_template)
    # Filter out empty-string placeholders to avoid spurious empty argv tokens.
    # Also split model_args into separate tokens if it contains --model.
    result: list[str] = []
    for part in rendered:
        if not part:
            continue
        if part.startswith("--model "):
            # Split "--model <value>" into two separate tokens
            result.extend(part.split())
        else:
            result.append(part)
    return tuple(result)


def launch_devin_session(
    issue_number: int,
    branch: str,
    prompt_path: Path,
    *,
    repo_root: Path,
    sessions_dir: Path,
    worktrees_dir: Path | None = None,
    command_template: tuple[str, ...] = DEFAULT_COMMAND_TEMPLATE,
    worker_model: str = "",
    venv_source: Path | None = None,
    worker_env: dict[str, str] | None = None,
    materialize_dirs: tuple[str, ...] = (),
    rework: bool = False,
    recovery: dict[str, Any] | None = None,
    base_ref: str = "",
    config: OrchestratorConfig | None = None,
    review: bool = False,
    head_sha: str = "",
) -> SessionRecord:
    """Launch a headless Devin CLI session for one issue and return immediately.

    Creates an isolated per-issue git worktree (via ``worktree.create_worktree``)
    and launches the Devin CLI inside it, so concurrent workers do not contend
    over a shared checkout. Mirrors the claude-code adapter's worktree lifecycle:
    creation before launch; ``remove_worktree`` (junction-safe) on failure.

    Non-blocking: uses ``Popen`` (never waits for the process). stdout/stderr
    are redirected to a per-session log file since the worker can run for many
    minutes. The sidecar JSON is written atomically before this function
    returns, so any crash after that point still leaves a durable record for
    ``read_session_records``/``doctor`` to find. Never raises — worktree-
    creation failures, a missing ``devin`` binary, or any other ``OSError``
    comes back as a record with ``pid=None`` and ``error`` set.

    If ``rework`` is True, the worktree is created in rework mode (reuse existing
    worktree or attach to existing branch instead of creating a new branch).

    If ``recovery`` is provided (a dict with state file dispatch record), this is
    a dead-worker recovery re-dispatch. The worktree layer will inspect the
    leftover worktree/branch and either clean it (no commits) or reuse it (has
    commits/dirty work).

    If ``review`` is True (issue #1513), this launches a PR-reviewer session
    instead of an issue worker: ``branch``/``worktrees_dir``/``rework``/
    ``recovery``/``base_ref`` are ignored for worktree purposes and
    ``sessions_dir`` is expected to be the caller's ``reviews_dir``; a
    detached-HEAD checkout is created via ``worktree.create_review_checkout``
    (keyed by ``issue_number`` interpreted as the PR number, at ``head_sha``,
    which is required) instead of ``create_worktree``, and torn down via
    ``worktree.remove_review_checkout`` on any launch failure. The command
    template is sanitized via ``_sanitize_review_command_template`` regardless
    of what ``command_template`` the caller passed, so no config combination
    can grant a reviewer write access. ``prompt_path`` is used as-is (devin's
    prompt file always lives outside the worktree, in both modes).
    """
    if review:
        command_template = _sanitize_review_command_template(command_template)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(sessions_dir, issue_number, rework=rework)
    # Issue #2036: the claude-code adapter's per-issue events.jsonl shares this
    # sessions_dir and is read harness-agnostically by the verdict/miss/metrics
    # readers. Only the claude launcher rotated it, so a devin session inherited
    # the previous claude round's transcript as if it were its own. Retire it at
    # launch so every reader sees this session's file or none.
    _rotate_old_log(_events_path(sessions_dir, issue_number, rework=rework, review=review))
    session_id = str(uuid.uuid4())

    # Issue #426: recovery probes carry a Signal-1-style deferral counter. Seed
    # the recovery dict from the existing sidecar (if any) so consecutive
    # recovery attempts observe the same counter the reaper does.
    if recovery is not None:
        recovery = dict(recovery)
        recovery.setdefault(
            "inconclusive_probe_deferred_count",
            _read_sidecar_inconclusive_count(sessions_dir, issue_number),
        )

    # Issue #2246: every error record produced below emits exactly one
    # launch_failed event at the seam. State-dir resolution is a pure path
    # computation -- no event is written on the success path. A reviewer's
    # ``issue_number`` is really the PR number, so it lands on ``pr_number``.
    launch_state_path = launch_events.state_path_for(repo_root, config)

    def _emit_launch_failed(
        error_class: str, error: str, *, failure_kind: str | None = None
    ) -> None:
        launch_events.emit_launch_failed(
            launch_state_path,
            role="reviewer" if review else "worker",
            harness="devin-shell",
            model=worker_model,
            issue_number=None if review else issue_number,
            pr_number=issue_number if review else None,
            error_class=error_class,
            error=error,
            failure_kind=failure_kind,
        )

    # --- worktree creation ---------------------------------------------------
    try:
        if review:
            if not head_sha:
                raise ValueError(
                    f"launch_devin_session(review=True) requires head_sha for PR #{issue_number}"
                )
            worktree: WorktreeInfo = create_review_checkout(
                repo_root,
                issue_number,
                head_sha,
                reviews_dir=sessions_dir,
            )
        else:
            worktree = create_worktree(
                repo_root,
                branch,
                worktrees_dir=worktrees_dir,
                venv_source=venv_source,
                materialize_dirs=materialize_dirs,
                rework=rework,
                recovery=recovery,
                base_ref=base_ref,
                issue_number=issue_number,
                config=config,
                sessions_dir=sessions_dir,
            )
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        if isinstance(exc, WorktreeProbeFailedError):
            # Transient probe contention (e.g. index lock), not a confirmed-dirty
            # worktree. Must stay off DETERMINISTIC_ESCALATION_FAILURE_KINDS so it
            # takes the ordinary redispatch-cap path (issue #288 follow-up, PR #314).
            failure_kind = "worktree_probe_failed"
        elif isinstance(exc, WorktreeUnsafeError):
            # Issue #807: the discriminator (shim dirt vs local commits) is
            # computed at detection time and carried on the exception, so the
            # launch shim emits a distinct failure_kind without classifying
            # after the fact.
            failure_kind = exc.kind
        elif isinstance(exc, ReworkBranchConflictError):
            failure_kind = "rework_branch_conflict"
        elif isinstance(exc, WorktreeForeignWriterError):
            failure_kind = "worktree_foreign_writer"
        elif isinstance(exc, LiveWorkerRedispatchError):
            failure_kind = "live_worker_redispatch_averted"
        else:
            failure_kind = None
        record = SessionRecord(
            issue_number=issue_number,
            branch=branch,
            # str() is load-bearing: the exception stores a Path, and an
            # unserializable field here destroys this whole failure record
            # mid-json.dump, downgrading the diagnosis to a generic launch
            # failure that burns the rework cap (issue #1184).
            worktree_path=str(getattr(exc, "worktree_path", ""))
            if isinstance(exc, WorktreeForeignWriterError)
            else "",
            prompt_path=str(prompt_path),
            command=command_template,
            pid=exc.pid
            if isinstance(exc, LiveWorkerRedispatchError)
            else getattr(exc, "pid", None),
            started_at=utc_now(),
            log_path=str(log_path),
            error=str(exc)
            if isinstance(exc, (LiveWorkerRedispatchError, WorktreeForeignWriterError))
            else f"worktree creation failed: {exc}",
            failure_kind=failure_kind,
            process_start_time=exc.process_start_time
            if isinstance(exc, LiveWorkerRedispatchError)
            else None,
            inconclusive_probe_deferred_count=exc.inconclusive_probe_deferred_count
            if isinstance(exc, LiveWorkerRedispatchError)
            else 0,
        )
        _write_json(_sidecar_path(sessions_dir, issue_number), record.to_dict())
        _emit_launch_failed(
            launch_events.LAUNCH_ERR_WORKTREE,
            record.error or "",
            failure_kind=failure_kind,
        )
        return record

    def _teardown_worktree() -> None:
        if review:
            remove_review_checkout(repo_root, issue_number, reviews_dir=sessions_dir)
        elif worktree.foreign_adopted:
            # Issue #1476: the checkout belongs to whoever created it — a
            # launch failure must never delete a borrowed worktree. No
            # orchestrator artifacts exist in it to clean up — the writer
            # marker is only written after a successful Popen, and the devin
            # adapter's prompt lives outside the worktree.
            return
        else:
            remove_worktree(
                repo_root, worktree.path, force=True, branch=None if rework else branch
            )

    def _fail(error: str, *, error_class: str) -> SessionRecord:
        # Shared by the three post-worktree-creation failure paths below
        # (rework-conflict-notice, command-rendering, env-sanitization): each
        # already called _teardown_worktree() and just needs an identically
        # shaped SessionRecord (pid/failure_kind at their defaults) persisted
        # and returned. Deliberately not reused by the worktree-creation
        # except-block above, which branches on failure_kind/pid per
        # exception type, or the launch except-block below, which does not
        # return immediately.
        record = SessionRecord(
            issue_number=issue_number,
            branch=branch,
            worktree_path=str(worktree.path),
            prompt_path=str(prompt_path),
            command=command_template,
            pid=None,
            started_at=utc_now(),
            log_path=str(log_path),
            error=error,
        )
        _write_json(_sidecar_path(sessions_dir, issue_number), record.to_dict())
        _emit_launch_failed(error_class, error)
        return record

    # A redispatch may have just preserved the prior attempt's branch tip
    # (issue #261) — fold that into whatever post-mortem sidecar already
    # exists for this issue so the ref is discoverable alongside the block
    # diagnosis it belongs to. Best-effort: never blocks or fails dispatch.
    if worktree.attempt_snapshot is not None and worktree.attempt_snapshot.ref_name is not None:
        merge_attempt_snapshot(sessions_dir, issue_number, worktree.attempt_snapshot)

    # The rework pre-merge hit a real conflict (worktree.py's
    # _merge_update_rework_branch): the worktree still launches, but the
    # worker must resolve it before touching the review feedback. Append the
    # notice to prompt_path in place -- it is caller-supplied and lives
    # outside the worktree (never copied in, unlike claude-code), but it is
    # still the exact file the devin CLI reads -- rather than failing closed
    # (see worktree.ReworkMergeConflict).
    if worktree.rework_conflict is not None:
        try:
            existing_prompt = prompt_path.read_text(encoding="utf-8")
            prompt_path.write_text(
                apply_rework_conflict_notice(existing_prompt, worktree.rework_conflict),
                encoding="utf-8",
            )
        except OSError as exc:
            _teardown_worktree()
            return _fail(
                f"failed to append rework conflict notice to prompt file: {exc}",
                error_class=launch_events.LAUNCH_ERR_PROMPT,
            )

    # --- command rendering (prompt_path is caller-supplied, lives outside wt) -
    launch_prompt_path = _write_devin_review_prompt(prompt_path) if review else prompt_path
    try:
        command = _render_command(
            command_template,
            issue_number=issue_number,
            branch=branch,
            prompt_path=launch_prompt_path,
            worker_model=worker_model,
        )
    except (KeyError, IndexError, ValueError) as exc:
        _teardown_worktree()
        return _fail(
            f"command template rendering failed: {exc}",
            error_class=launch_events.LAUNCH_ERR_RENDER,
        )

    # Sanitize environment to prevent VIRTUAL_ENV leaks from the orchestrator,
    # then merge user-provided worker_env overrides on top (e.g. PYTEST_XDIST_AUTO_NUM_WORKERS).
    # sanitize_env drops GH_TOKEN/GITHUB_TOKEN and forces GH_CONFIG_DIR to a
    # worktree-local empty directory so workers do not inherit the orchestrator's
    # admin token or stored gh auth state (issue #502), and points TMP/TEMP/TMPDIR
    # at a worktree-local directory so a concurrent session on this host cannot
    # collide on a shared temp path (issue #1767). To give workers a scoped
    # GitHub token, set worker_env={"GH_TOKEN": "<scoped-PAT>"} in the adapter config.
    try:
        sanitized_env = sanitize_env(worktree.path)
    except OSError as exc:
        _teardown_worktree()
        return _fail(
            f"failed to prepare worker environment: {exc}",
            error_class=launch_events.LAUNCH_ERR_ENV,
        )
    worker_env_dict = {
        **sanitized_env,
        **{str(k): str(v) for k, v in (worker_env or {}).items()},
    }
    # Issue #646: resolve what sanitize_env()+worker_env actually settled on,
    # purely for the launch-time diagnostic log below (does not affect
    # worker_env_dict itself, which already carries the real values).
    xdist_cap, xdist_cap_source = resolve_pytest_cap(sanitized_env, worker_env)
    uv_no_sync, uv_no_sync_source = resolve_uv_no_sync(worktree.path, sanitized_env, worker_env)

    if review:
        _write_review_permissions(worktree.path)

    pid: int | None = None
    error: str | None = None
    process_start_time: float | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = popen_worker(
                list(command),
                priority=CpuPriority.BELOW_NORMAL,
                cwd=str(worktree.path),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=worker_env_dict,
            )
        pid = process.pid
        # Capture process creation time immediately after spawn to verify identity later
        process_start_time = _get_process_start_time(pid)
    except OSError as exc:
        _teardown_worktree()
        error = f"failed to launch devin: {exc}"
        _emit_launch_failed(launch_events.LAUNCH_ERR_SPAWN, error)

    if pid is not None and error is None:
        # Issue #2052: devin-shell joins the terminal-record contract. The
        # profile gate and watcher start live in devin_terminal_record
        # (extracted under the file-size ratchet on PR #2069 rework) -- it
        # leaves ``issue-<n>.devin.terminal.json`` when this process exits,
        # the durable exit evidence ``worker_fate.resolve_fate`` reads back
        # adapter-agnostically. Review launches pass ``worktree_path=None``
        # (claude_code's #1354 parity: a review checkout holds no worker
        # outcome file).
        maybe_start_terminal_status_watcher(
            process,
            sessions_dir,
            issue_number,
            worktree_path=None if review else worktree.path,
        )

        # Write the worktree writer marker so this process is recorded as the
        # legitimate occupant of the worktree (issue #400).
        try:
            write_worktree_marker(
                worktree.path, pid, session_id, process_start_time=process_start_time
            )
        except OSError:
            # Best-effort marker write must not derail a successful launch.
            pass

        # Issue #646: launch-time INFO log so a reader can answer "how many
        # suites were running at <time>, from which worktrees, at what cap"
        # without process forensics. Paired with the exit-side census log in
        # workflow.py (_log_worker_census) — join on session_id/pid/worktree.
        logger.info(
            "worker launch: adapter=devin-shell issue=%s worktree=%s pid=%s session_id=%s "
            "xdist_cap=%s(%s) uv_no_sync=%s(%s) at=%s",
            issue_number,
            worktree.path,
            pid,
            session_id,
            xdist_cap,
            xdist_cap_source,
            uv_no_sync,
            uv_no_sync_source,
            datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )

    record = SessionRecord(
        issue_number=issue_number,
        branch=branch,
        worktree_path=str(worktree.path),
        prompt_path=str(prompt_path),
        command=command,
        pid=pid,
        started_at=utc_now(),
        log_path=str(log_path),
        error=error,
        process_start_time=process_start_time,
        reclaimed=worktree.reclaimed,
        attempt_ref=worktree.attempt_snapshot.ref_name if worktree.attempt_snapshot else None,
        attempt_ahead_of_main=(
            worktree.attempt_snapshot.ahead_of_main_count if worktree.attempt_snapshot else None
        ),
        session_id=session_id,
        xdist_cap=xdist_cap if pid is not None and error is None else None,
        uv_no_sync=uv_no_sync if pid is not None and error is None else None,
    )
    _write_json(_sidecar_path(sessions_dir, issue_number), record.to_dict())
    return record


_DEVIN_SIDECAR_STEM_RE = re.compile(r"^issue-\d+$")


def read_session_records(sessions_dir: Path) -> list[SessionRecord]:
    """Read every sidecar JSON in ``sessions_dir`` back into ``SessionRecord``s.

    Unreadable or malformed sidecars are skipped rather than raising — a
    corrupt file must not take down doctor/status reporting for every other
    in-flight session.
    """
    if not sessions_dir.is_dir():
        return []
    records: list[SessionRecord] = []
    for path in sorted(sessions_dir.glob("issue-*.json")):
        # `issue-*.json` also matches every other adapter's/subsystem's
        # sidecar that happens to share the `issue-<n>.<extra>.json` naming
        # scheme in the same sessions_dir — e.g. the claude-code adapter's
        # `issue-N.claude.json` and post_mortem's `issue-N.post-mortem.json`.
        # A devin session sidecar's stem (path name minus the final `.json`)
        # is exactly `issue-<digits>`; anything with additional dotted
        # segments belongs to a different writer and must be skipped, or its
        # foreign schema gets misread as a bogus SessionRecord (pid=None)
        # that bypasses corroboration downstream (issue #343).
        if not _DEVIN_SIDECAR_STEM_RE.match(path.stem):
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            records.append(SessionRecord.from_dict(payload))
        except (KeyError, TypeError, ValueError):
            continue
    return records


def probe_devin(
    repo_root: Path, *, command: tuple[str, ...] = ("devin", "--version")
) -> RunResult:
    """Run a cheap Devin CLI probe (e.g. ``devin --version``) for
    ``doctor --adapter-probe``. Delegates to ``run_captured``, so a missing
    binary or non-zero exit comes back as a not-ok result, never an exception.
    """
    return run_captured(list(command), cwd=repo_root, timeout_seconds=30)


def _get_process_start_time(pid: int) -> float | None:
    """Process creation time as a Unix timestamp (delegates to ``process_utils``)."""
    return _process_utils.get_process_start_time(pid)


__all__ = [
    "DEFAULT_COMMAND_TEMPLATE",
    "SessionRecord",
    "launch_devin_session",
    "read_session_records",
    "probe_devin",
    "update_session_record_with_failure_classification",
    "get_rate_limit_defer_until",
    "_get_process_start_time",
    "_sidecar_path",
]
