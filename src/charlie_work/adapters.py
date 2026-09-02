from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import layout
from .config import ApiWorkerConfig, OrchestratorConfig
from .harnesses import WORKER_HARNESSES
from .subprocess_runner import run_captured


@dataclass(frozen=True)
class SessionRequest:
    issue_number: int
    issue_title: str
    prompt_path: Path
    branch_name: str
    rework: bool = False
    recovery: dict[str, Any] | None = None


@dataclass(frozen=True)
class AdapterSettings:
    """Everything an adapter needs, resolved by the caller (paths absolute).

    ``adapter`` values: "manual" (manifest for the operator), "command"
    (blocking per-issue dispatch_command), "devin-shell" (non-blocking headless
    devin CLI with sidecar tracking), "claude-code" (worktree-isolated Claude
    Code workers).
    """

    adapter: str = "manual"
    dispatch_command: str | tuple[str, ...] = ""
    command_timeout_seconds: int = 300
    sessions_dir: Path | None = None
    shell_command: tuple[str, ...] = ()
    claude_command: tuple[str, ...] = ()
    worktrees_dir: Path | None = None
    venv_source: Path | None = None
    # Extra env merged over the orchestrator's env in each worker process
    # (claude-code and devin-shell). Primary use: PYTEST_XDIST_AUTO_NUM_WORKERS
    # to bound local test parallelism. Empty means no overrides.
    worker_env: dict[str, str] = field(default_factory=dict)
    # devin-shell worker model; empty string means CLI default.
    worker_model: str = ""
    # Repo-root-relative paths copied into each worktree after creation
    # (e.g. [".devin"]). Copy-not-link (workers may write marker files);
    # skip-if-tracked (tracked paths are already present).
    materialize_dirs: tuple[str, ...] = ()
    # dry_run: if True, adapters return synthetic results without launching
    # real worker processes or mutating worktrees.
    dry_run: bool = False
    # Base ref for fresh worktree creation. Empty string means auto-resolve to
    # origin/<default-branch>. Passed to create_worktree for both devin-shell
    # and claude-code adapters.
    base_ref: str = ""
    # Seconds to sleep between consecutive worker-session launches within a
    # single dispatch pass (devin-shell and claude-code only; those two spawn
    # real out-of-process worker sessions that can trip a provider's message
    # rate limit when launched back-to-back). 0 disables the stagger. Mirrors
    # config.DispatchConfig.launch_stagger_seconds.
    launch_stagger_seconds: int = 0
    # Opt-in: tee Claude Code's --output-format stream-json to a separate events.jsonl file.
    # When enabled, the worker launch command is extended with --output-format stream-json
    # and the structured JSONL output is written to issue-<n>.events.jsonl alongside the
    # plaintext log. This enables downstream parsing of tool_call_count, turn_count, tokens,
    # and cost_usd for tripwires and progress reporting. Default False until #162/#163 land.
    tee_stream_json: bool = False
    # api-worker adapter: the resolved ApiWorkerConfig registry section. Carried
    # through AdapterSettings the same way claude_command/worker_env carry the
    # claude-code settings. None for non-api adapters; required for the "api"
    # adapter branch. The provider registry, budget caps, and active provider
    # name all live on this frozen value object (added in #475).
    api_worker_config: ApiWorkerConfig | None = None
    # Full orchestrator config passed to worktree creation/recovery for liveness probes.
    config: OrchestratorConfig | None = None


@dataclass(frozen=True)
class SessionDispatchResult:
    issue_number: int
    issue_title: str
    prompt_path: str
    branch_name: str
    adapter: str
    ok: bool
    command: str | list[str] | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    reclaimed: str | None = None  # "fetch-fallback" | "pruned" | "salvaged" | None
    pid: int | None = None  # Worker process PID for state-based liveness detection
    process_start_time: float | None = None  # Process creation time for PID recycling protection
    failure_kind: str | None = None  # stable machine-readable classification of a failure
    # Issue #1423: the worktree path the launch resolved to. Carried so the
    # blocked-environment reap path (``_try_reap_blocked_foreign_writer``) can
    # read the writer marker without re-deriving the path from the branch name.
    # Empty for adapters that never create a worktree (command/manual/dry-run).
    worktree_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_number": self.issue_number,
            "issue_title": self.issue_title,
            "prompt_path": self.prompt_path,
            "branch_name": self.branch_name,
            "adapter": self.adapter,
            "ok": self.ok,
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "reclaimed": self.reclaimed,
            "pid": self.pid,
            "process_start_time": self.process_start_time,
            "failure_kind": self.failure_kind,
            "worktree_path": self.worktree_path,
        }


def _dispatch_manual(
    repo_root: Path,
    requests: list[SessionRequest],
    sessions_dir: Path,
    settings: AdapterSettings,
) -> list[SessionDispatchResult]:
    return [_manual_result(request) for request in requests]


def _dispatch_command(
    repo_root: Path,
    requests: list[SessionRequest],
    sessions_dir: Path,
    settings: AdapterSettings,
) -> list[SessionDispatchResult]:
    return [
        _run_command_adapter(
            repo_root, request, settings.dispatch_command, settings.command_timeout_seconds
        )
        for request in requests
    ]


def _dispatch_devin_shell(
    repo_root: Path,
    requests: list[SessionRequest],
    sessions_dir: Path,
    settings: AdapterSettings,
) -> list[SessionDispatchResult]:
    return _launch_staggered(
        requests,
        lambda request: _run_devin_shell_adapter(repo_root, request, sessions_dir, settings),
        settings.launch_stagger_seconds,
    )


def _dispatch_claude_code(
    repo_root: Path,
    requests: list[SessionRequest],
    sessions_dir: Path,
    settings: AdapterSettings,
) -> list[SessionDispatchResult]:
    return _launch_staggered(
        requests,
        lambda request: _run_claude_code_adapter(repo_root, request, sessions_dir, settings),
        settings.launch_stagger_seconds,
    )


def _dispatch_api(
    repo_root: Path,
    requests: list[SessionRequest],
    sessions_dir: Path,
    settings: AdapterSettings,
) -> list[SessionDispatchResult]:
    return _launch_staggered(
        requests,
        lambda request: _run_api_adapter(repo_root, request, sessions_dir, settings),
        settings.launch_stagger_seconds,
    )


# Single dispatch table keyed by ``worker.harness`` name. This -- not a
# separate if/elif chain -- is what ``dispatch_sessions`` consumes, and the
# assertion below fails import if it ever drifts from ``harnesses.py``'s
# registry (issue #1513: the previous if/elif chain and config.py's
# hand-maintained allowlist could silently diverge; a harness "valid" per
# config could be unreachable here, or vice versa).
_ADAPTER_DISPATCHERS: dict[
    str, Callable[[Path, list[SessionRequest], Path, AdapterSettings], list[SessionDispatchResult]]
] = {
    "manual": _dispatch_manual,
    "command": _dispatch_command,
    "devin-shell": _dispatch_devin_shell,
    "claude-code": _dispatch_claude_code,
    "api": _dispatch_api,
}

assert set(_ADAPTER_DISPATCHERS) == WORKER_HARNESSES, (
    "adapters._ADAPTER_DISPATCHERS must dispatch exactly the harnesses "
    "harnesses.WORKER_HARNESSES declares valid -- keep both in sync"
)


def dispatch_sessions(
    repo_root: Path,
    manifest_path: Path,
    results_path: Path,
    settings: AdapterSettings,
    requests: list[SessionRequest],
) -> list[SessionDispatchResult]:
    adapter = settings.adapter
    write_session_manifest(manifest_path, requests, adapter=adapter)
    # NOTE: constant-only substitution, not layout.sessions_dir_default(). That
    # helper requires a genuine state_root; manifest_path.parent only equals
    # the default dispatches dir when devin.session_manifest is at its default
    # value -- under an operator override it can point anywhere, so composing
    # through the helper here could change the resolved path.
    sessions_dir = settings.sessions_dir or manifest_path.parent / layout.SESSIONS_DIRNAME
    if settings.dry_run:
        results = [_dry_run_result(request, adapter) for request in requests]
    else:
        dispatcher = _ADAPTER_DISPATCHERS.get(adapter)
        if dispatcher is None:
            results = [
                _result(
                    request,
                    adapter=adapter,
                    ok=False,
                    error=f"Unsupported Devin adapter: {adapter}",
                )
                for request in requests
            ]
        else:
            results = dispatcher(repo_root, requests, sessions_dir, settings)
    write_session_results(results_path, results)
    return results


def _launch_staggered(
    requests: list[SessionRequest],
    launch: Callable[[SessionRequest], SessionDispatchResult],
    stagger_seconds: int,
) -> list[SessionDispatchResult]:
    """Launch each request in order, sleeping between consecutive launches.

    The sleep sits BETWEEN launches only -- never before the first, never
    after the last, and skipped entirely for a single request or when
    ``stagger_seconds`` is 0. This is orchestration-lane pacing, not adapter
    blocking: each individual ``launch`` call is still a non-blocking Popen
    (CLAUDE.md invariant), this just paces the loop that calls it so a burst
    of launches doesn't trip a provider's message rate limit (observed:
    Devin's "overall message rate limit" firing when 3 sessions launched
    within 6 seconds).
    """
    results: list[SessionDispatchResult] = []
    for index, request in enumerate(requests):
        if index > 0 and stagger_seconds > 0:
            time.sleep(stagger_seconds)
        results.append(launch(request))
    return results


def manifest_adapter_label(kinds: set[str]) -> str:
    """Derive the manifest ``adapter`` label from the set of adapter kinds in a pass.

    One kind → that kind (a homogeneous batch is labeled honestly, not as
    ``"mixed"``). More than one kind → ``"mixed"``. This is the single point
    of label derivation for session manifests (issue #626): the rescue tier's
    combined-manifest write (normal-tier worker harness + rescue's
    claude-code) goes through this helper so the label always reflects the
    actual partition.
    """
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


def write_session_manifest(
    path: Path, requests: list[SessionRequest], *, adapter: str = "manual"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "adapter": adapter,
        "instructions": _instructions(adapter),
        "sessions": [_request_dict(request) for request in requests],
    }
    _write_json(path, payload)


def write_session_results(path: Path, results: list[SessionDispatchResult]) -> None:
    payload = {"results": [result.to_dict() for result in results]}
    _write_json(path, payload)


def _instructions(adapter: str) -> list[str]:
    if adapter == "command":
        return [
            "Worker sessions are launched by the configured command adapter.",
            "Each command receives one issue prompt and must create exactly one worker session.",
            "Only successful command results are labeled in progress.",
        ]
    if adapter == "devin-shell":
        return [
            "Worker sessions were launched headless via the devin CLI (non-blocking).",
            "Per-session sidecar JSON and logs live under the sessions directory.",
            "Use doctor to probe the adapter and surface stale or failed sessions.",
        ]
    if adapter == "claude-code":
        return [
            "Claude Code workers were launched headless in isolated git worktrees.",
            "Per-worker sidecar JSON and logs live under the sessions directory.",
            "Never remove a worktree before deleting its .venv junction.",
        ]
    if adapter == "api":
        return [
            "API workers were launched headless via the Claude Code CLI against a",
            "configured Anthropic-compatible provider endpoint (adapter_kind: api).",
            "Per-worker sidecar JSON (issue-<n>.api.json) and logs live under the",
            "sessions directory; the provider env is injected into the child process",
            "only — the API key never appears in sidecars, logs, or argv.",
        ]
    if adapter == "mixed":
        return [
            "This manifest combines sessions launched by more than one worker",
            "harness in the same pass — normally the configured worker harness",
            "plus the rescue tier's claude-code reviewer-as-rescue-worker",
            "(issue #626).",
            "Each session's adapter is recorded in the dispatch results file, not in",
            "this manifest's top-level adapter field. Consult the results file for",
            "per-session adapter_kind, provider, and launch status.",
        ]
    return [
        "Open one Devin worker session per request.",
        "Paste the prompt file contents as the worker task.",
        "Keep each worker bound to exactly one GitHub issue.",
        "When an API adapter is available, replace this manifest consumer without changing orchestrator state.",
    ]


def _request_dict(request: SessionRequest) -> dict[str, Any]:
    return {
        "issue_number": request.issue_number,
        "issue_title": request.issue_title,
        "prompt_path": str(request.prompt_path),
        "branch_name": request.branch_name,
        "rework": request.rework,
        "recovery": request.recovery,
    }


def _manual_result(request: SessionRequest) -> SessionDispatchResult:
    return _result(request, adapter="manual", ok=True)


def _dry_run_result(request: SessionRequest, adapter: str) -> SessionDispatchResult:
    return _result(
        request,
        adapter=adapter,
        ok=True,
        error=None,
    )


def _run_devin_shell_adapter(
    repo_root: Path,
    request: SessionRequest,
    sessions_dir: Path,
    settings: AdapterSettings,
) -> SessionDispatchResult:
    from .devin_shell import DEFAULT_COMMAND_TEMPLATE, launch_devin_session

    try:
        record = launch_devin_session(
            request.issue_number,
            request.branch_name,
            request.prompt_path,
            repo_root=repo_root,
            sessions_dir=sessions_dir,
            worktrees_dir=settings.worktrees_dir,
            command_template=settings.shell_command or DEFAULT_COMMAND_TEMPLATE,
            worker_model=settings.worker_model,
            venv_source=settings.venv_source,
            worker_env=settings.worker_env,
            materialize_dirs=settings.materialize_dirs,
            rework=request.rework,
            recovery=request.recovery,
            base_ref=settings.base_ref,
            config=settings.config,
        )
        # Non-blocking launch: there is no returncode/stdout to report — liveness
        # and output live in the sidecar JSON and per-session log.
        ok = record.error is None and record.pid is not None
        return _result(
            request,
            adapter="devin-shell",
            ok=ok,
            command=list(record.command),
            error=record.error if not ok else None,
            reclaimed=record.reclaimed,
            pid=record.pid,
            process_start_time=record.process_start_time,
            failure_kind=record.failure_kind,
            worktree_path=record.worktree_path,
        )
    except Exception as exc:
        # Catch any unexpected exception and return as a failure result
        # (CLAUDE.md invariant: errors from external processes come back as values)
        return _result(
            request,
            adapter="devin-shell",
            ok=False,
            error=f"launch failed: {exc}",
        )


def _run_claude_code_adapter(
    repo_root: Path,
    request: SessionRequest,
    sessions_dir: Path,
    settings: AdapterSettings,
) -> SessionDispatchResult:
    from .claude_code import launch_claude_worker

    try:
        prompt_text = request.prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        return _result(request, adapter="claude-code", ok=False, error=str(exc))
    kwargs: dict[str, Any] = {}
    if settings.claude_command:
        kwargs["command_template"] = settings.claude_command
    if settings.tee_stream_json:
        kwargs["tee_stream_json"] = settings.tee_stream_json
    record = launch_claude_worker(
        request.issue_number,
        request.branch_name,
        prompt_text,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        worktrees_dir=settings.worktrees_dir,
        venv_source=settings.venv_source,
        env=settings.worker_env,
        materialize_dirs=settings.materialize_dirs,
        rework=request.rework,
        recovery=request.recovery,
        base_ref=settings.base_ref,
        config=settings.config,
        **kwargs,
    )
    ok = record.error is None and record.pid is not None
    return _result(
        request,
        adapter="claude-code",
        ok=ok,
        command=list(record.command),
        error=record.error if not ok else None,
        reclaimed=record.reclaimed,
        pid=record.pid,
        process_start_time=record.process_start_time,
        failure_kind=record.failure_kind,
        worktree_path=record.worktree_path,
    )


def _run_api_adapter(
    repo_root: Path,
    request: SessionRequest,
    sessions_dir: Path,
    settings: AdapterSettings,
) -> SessionDispatchResult:
    """Launch an api worker (Claude Code CLI + provider env).

    Mirrors ``_run_claude_code_adapter``: reads the prompt, delegates to
    ``api_worker.launch_api_worker`` (which resolves the active provider, builds
    the provider env, and calls ``claude_code.launch_claude_worker`` with
    ``adapter_kind="api"``), and maps the record into a SessionDispatchResult.
    ``tee_stream_json`` is force-enabled inside ``launch_api_worker`` regardless
    of ``settings.tee_stream_json`` (the budget ledger depends on events.jsonl).
    """
    from .api_worker import launch_api_worker

    api_worker_config = settings.api_worker_config
    if api_worker_config is None:
        return _result(
            request,
            adapter="api",
            ok=False,
            error=(
                "api adapter selected but no api_worker_config was resolved into "
                "AdapterSettings; cannot launch api worker"
            ),
        )
    try:
        prompt_text = request.prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        return _result(request, adapter="api", ok=False, error=str(exc))
    kwargs: dict[str, Any] = {}
    if settings.claude_command:
        kwargs["command_template"] = settings.claude_command
    record = launch_api_worker(
        request.issue_number,
        request.branch_name,
        prompt_text,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        api_worker_config=api_worker_config,
        worktrees_dir=settings.worktrees_dir,
        venv_source=settings.venv_source,
        worker_env=settings.worker_env,
        materialize_dirs=settings.materialize_dirs,
        rework=request.rework,
        recovery=request.recovery,
        base_ref=settings.base_ref,
        config=settings.config,
        **kwargs,
    )
    ok = record.error is None and record.pid is not None
    return _result(
        request,
        adapter="api",
        ok=ok,
        command=list(record.command),
        error=record.error if not ok else None,
        reclaimed=record.reclaimed,
        pid=record.pid,
        process_start_time=record.process_start_time,
        failure_kind=record.failure_kind,
        worktree_path=record.worktree_path,
    )


def _run_command_adapter(
    repo_root: Path,
    request: SessionRequest,
    dispatch_command: str | tuple[str, ...],
    command_timeout_seconds: int,
) -> SessionDispatchResult:
    try:
        command = _render_command(dispatch_command, request)
    except (KeyError, IndexError, ValueError) as exc:
        return _result(request, adapter="command", ok=False, error=str(exc))
    if not command:
        return _result(
            request,
            adapter="command",
            ok=False,
            error="devin.dispatch_command is required when worker.harness is command",
        )
    run = run_captured(
        command,
        cwd=repo_root,
        timeout_seconds=command_timeout_seconds,
        shell=isinstance(command, str),
    )
    return _result(
        request,
        adapter="command",
        ok=run.ok,
        command=command,
        returncode=run.returncode,
        stdout=run.stdout,
        stderr=run.stderr,
        error=None if run.ok else (run.error or "Dispatch command failed"),
    )


def _render_command(
    dispatch_command: str | tuple[str, ...], request: SessionRequest
) -> str | list[str] | None:
    values = {
        "issue_number": str(request.issue_number),
        "issue_title": request.issue_title,
        "prompt_path": str(request.prompt_path),
        "branch_name": request.branch_name,
    }
    if isinstance(dispatch_command, tuple):
        command = [str(part).format(**values) for part in dispatch_command]
        return command if command else None
    text = str(dispatch_command or "").strip()
    if not text:
        return None
    # String-form commands run through a shell. issue_title is attacker
    # controlled (anyone can title a GitHub issue), so interpolating it into
    # a shell string is command injection — refuse it. List-form commands
    # execute without a shell and may use every placeholder.
    if "{issue_title}" in text:
        raise ValueError(
            "devin.dispatch_command: {issue_title} is not allowed in string-form "
            "(shell) commands — use the list form, which runs without a shell"
        )
    return text.format(**values)


def _result(
    request: SessionRequest,
    *,
    adapter: str,
    ok: bool,
    command: str | list[str] | None = None,
    returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
    error: str | None = None,
    reclaimed: str | None = None,
    pid: int | None = None,
    process_start_time: float | None = None,
    failure_kind: str | None = None,
    worktree_path: str = "",
) -> SessionDispatchResult:
    return SessionDispatchResult(
        issue_number=request.issue_number,
        issue_title=request.issue_title,
        prompt_path=str(request.prompt_path),
        branch_name=request.branch_name,
        adapter=adapter,
        ok=ok,
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        error=error,
        reclaimed=reclaimed,
        pid=pid,
        process_start_time=process_start_time,
        failure_kind=failure_kind,
        worktree_path=worktree_path,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def cleanup_stale_session_tmp_files(sessions_dir: Path, min_age_seconds: float = 60.0) -> int:
    """Remove stranded ``.json.tmp`` files from the sessions directory.

    Issue #1393: the atomic session-record write (``_write_json`` /
    ``_write_json_atomic``) uses a tmp-file + ``replace()`` pattern.  If the
    process is interrupted between ``json.dump`` and ``tmp_path.replace()``
    (e.g. a watchdog kill during a launch-refusal write), the ``.json.tmp``
    file is left stranded — a torn record that defeats the atomic-write
    convention.  This sweep removes those stale tmp files at the start of
    each dispatch pass so they don't accumulate.

    Files younger than ``min_age_seconds`` (default 60s) are skipped so the
    sweep cannot race a legitimate in-flight atomic write — a writer that has
    closed its tmp file but not yet called ``tmp_path.replace()``.  The
    close→replace window is sub-second in practice; 60s is a generous margin
    that still reaps anything genuinely stranded from a prior pass.

    Returns the number of files removed.
    """
    if not sessions_dir.is_dir():
        return 0
    now = time.time()
    removed = 0
    for tmp_path in sessions_dir.glob("*.json.tmp"):
        try:
            stat = tmp_path.stat()
        except OSError:
            # File vanished between glob and stat — a concurrent writer
            # completed its replace(), or another sweeper removed it.
            continue
        age = now - stat.st_mtime
        if age < min_age_seconds:
            # Too young to safely call stranded — a legitimate writer may
            # be between close() and replace().  Leave it for a later pass.
            continue
        try:
            tmp_path.unlink()
            removed += 1
        except OSError:
            # Best-effort: a concurrent writer may hold the file, or the
            # file may have already been renamed.  Either way, skip it.
            continue
    return removed
