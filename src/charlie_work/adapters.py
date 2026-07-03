from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .subprocess_runner import run_captured


@dataclass(frozen=True)
class SessionRequest:
    issue_number: int
    issue_title: str
    prompt_path: Path
    branch_name: str


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
    # Extra env merged over the orchestrator's env in each claude-code worker
    # process (e.g. PYTEST_XDIST_AUTO_NUM_WORKERS to bound local test
    # parallelism). Empty means no overrides.
    worker_env: dict[str, str] = field(default_factory=dict)
    # devin-shell worker model; empty string means CLI default.
    worker_model: str = ""
    # dry_run: if True, adapters return synthetic results without launching
    # real worker processes or mutating worktrees.
    dry_run: bool = False


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
        }


def dispatch_sessions(
    repo_root: Path,
    manifest_path: Path,
    results_path: Path,
    settings: AdapterSettings,
    requests: list[SessionRequest],
) -> list[SessionDispatchResult]:
    adapter = settings.adapter
    write_session_manifest(manifest_path, requests, adapter=adapter)
    sessions_dir = settings.sessions_dir or manifest_path.parent / "sessions"
    if settings.dry_run:
        results = [_dry_run_result(request, adapter) for request in requests]
    elif adapter == "manual":
        results = [_manual_result(request) for request in requests]
    elif adapter == "command":
        results = [
            _run_command_adapter(
                repo_root, request, settings.dispatch_command, settings.command_timeout_seconds
            )
            for request in requests
        ]
    elif adapter == "devin-shell":
        results = [
            _run_devin_shell_adapter(repo_root, request, sessions_dir, settings)
            for request in requests
        ]
    elif adapter == "claude-code":
        results = [
            _run_claude_code_adapter(repo_root, request, sessions_dir, settings)
            for request in requests
        ]
    else:
        results = [
            _result(
                request,
                adapter=adapter,
                ok=False,
                error=f"Unsupported Devin adapter: {adapter}",
            )
            for request in requests
        ]
    write_session_results(results_path, results)
    return results


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

    record = launch_devin_session(
        request.issue_number,
        request.branch_name,
        request.prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        worktrees_dir=settings.worktrees_dir,
        command_template=settings.shell_command or DEFAULT_COMMAND_TEMPLATE,
        worker_model=settings.worker_model,
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
    record = launch_claude_worker(
        request.issue_number,
        request.branch_name,
        prompt_text,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        worktrees_dir=settings.worktrees_dir,
        venv_source=settings.venv_source,
        env=settings.worker_env,
        **kwargs,
    )
    ok = record.error is None and record.pid is not None
    return _result(
        request,
        adapter="claude-code",
        ok=ok,
        command=list(record.command),
        error=record.error if not ok else None,
    )


def _run_command_adapter(
    repo_root: Path,
    request: SessionRequest,
    dispatch_command: str | tuple[str, ...],
    command_timeout_seconds: int,
) -> SessionDispatchResult:
    try:
        command = _render_command(dispatch_command, request)
    except (KeyError, ValueError) as exc:
        return _result(request, adapter="command", ok=False, error=str(exc))
    if not command:
        return _result(
            request,
            adapter="command",
            ok=False,
            error="devin.dispatch_command is required when devin.adapter is command",
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
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)
