from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionRequest:
    issue_number: int
    issue_title: str
    prompt_path: Path
    branch_name: str


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
    adapter: str,
    dispatch_command: str | tuple[str, ...],
    command_timeout_seconds: int,
    requests: list[SessionRequest],
) -> list[SessionDispatchResult]:
    write_session_manifest(manifest_path, requests, adapter=adapter)
    if adapter == "manual":
        results = [_manual_result(request) for request in requests]
    elif adapter == "command":
        results = [
            _run_command_adapter(repo_root, request, dispatch_command, command_timeout_seconds)
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
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=command_timeout_seconds,
            shell=isinstance(command, str),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _result(
            request,
            adapter="command",
            ok=False,
            command=command,
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or ""),
            error=f"Dispatch command timed out after {command_timeout_seconds}s",
        )
    except OSError as exc:
        return _result(
            request,
            adapter="command",
            ok=False,
            command=command,
            error=str(exc),
        )
    return _result(
        request,
        adapter="command",
        ok=completed.returncode == 0,
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        error=None if completed.returncode == 0 else "Dispatch command failed",
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
    return text.format(**values) if text else None


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
