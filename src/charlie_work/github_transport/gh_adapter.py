"""`gh` CLI adapter (ADR-0006): the kill-switch transport and the per-call
fallback when the HTTP adapter cannot serve a request.

Every API request is rendered as exactly one ``gh api`` invocation -- never a
porcelain subcommand -- so this module is the single place that speaks gh's
command language. ``{owner}``/``{repo}`` arrive already filled in by the
guard, so the adapter never relies on gh's placeholder expansion. There is no
``--paginate``: pagination happens above the adapters (``pagination.py``), and
``--include`` combined with ``--paginate`` is ambiguous.

Bodies travel on stdin (``--input -``), never as ``-f``/``-F`` fields, which
sidesteps gh's type coercion of ``-F``, ``@file`` substitution and quoting
traps. ``--include`` makes gh print the HTTP status line and headers before
the body, so a non-2xx answer (gh exits 1) is still a ``Response`` and the
rate-limit headers reach the budget observer.

If gh printed no status line it failed before or without an HTTP exchange, and
its stderr is classified into a ``TransportFailure`` using the shared marker
sets (``failure_markers``). That text classification lives only here:
translation at the seam.

``_spawn`` is the sole ``subprocess.run`` in the GitHub layer. It keeps
``no_console_window_kwargs()``, ``encoding="utf-8"`` and ``errors="replace"``.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable

from ..subprocess_runner import no_console_window_kwargs
from .failure_markers import is_pre_connection_text, is_transport_class_text
from .outcome import (
    FailureKind,
    Outcome,
    Response,
    TransportFailure,
    graphql_errors_from_body,
    normalize_headers,
)
from .request import CliRequest, GraphQLRequest, Request, RestRequest

_STATUS_LINE_RE = re.compile(r"^HTTP/\S+\s+(\d{3})\b")
_HEADER_BODY_SPLIT_RE = re.compile(r"\r?\n\r?\n")

Spawn = Callable[[list[str], str | None, Path, float], "subprocess.CompletedProcess[str]"]


def _spawn(
    argv: list[str], stdin: str | None, cwd: Path, timeout: float
) -> "subprocess.CompletedProcess[str]":
    """The one subprocess call for GitHub (moved from `_run_via_gh_subprocess`)."""
    extra: dict[str, object] = {} if stdin is None else {"input": stdin}
    return subprocess.run(
        argv,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=timeout,
        **extra,
        **no_console_window_kwargs(),
    )


def render_argv(request: Request) -> tuple[list[str], str | None]:
    """The ``gh`` argv and stdin text for *request*."""
    if isinstance(request, CliRequest):
        return ["gh", *request.command.value], None
    if isinstance(request, GraphQLRequest):
        payload = {"query": request.document, "variables": json.loads(request.variables)}
        stdin = json.dumps(payload, ensure_ascii=False)
        return ["gh", "api", "--include", "graphql", "--input", "-"], stdin
    if isinstance(request, RestRequest):
        argv = [
            "gh",
            "api",
            "--include",
            "-X",
            request.method,
            "-H",
            f"Accept: {request.accept}",
        ]
        if request.body is not None:
            argv += ["--input", "-"]
        argv.append(request.target())
        return argv, request.body
    raise TypeError(f"GhAdapter cannot send {type(request).__name__}")


def parse_include_output(stdout: str) -> tuple[int, tuple[tuple[str, str], ...], str] | None:
    """Split ``gh api --include`` output into (status, headers, body).

    ``None`` when no HTTP status line was printed. If gh followed a redirect
    it prints each hop's header block; the last block before the body wins.
    """
    text = stdout.lstrip("\r\n")
    status: int | None = None
    headers: list[tuple[str, str]] = []
    body = text
    while True:
        match = _STATUS_LINE_RE.match(body)
        if match is None:
            break
        parts = _HEADER_BODY_SPLIT_RE.split(body, maxsplit=1)
        head = parts[0]
        body = parts[1] if len(parts) == 2 else ""
        status = int(match.group(1))
        headers = []
        for line in head.splitlines()[1:]:
            name, sep, value = line.partition(":")
            if sep:
                headers.append((name.strip(), value.strip()))
    if status is None:
        return None
    return status, normalize_headers(headers), body


class GhAdapter:
    """``Adapter`` implementation that shells out to ``gh api``."""

    name = "gh"

    def __init__(self, cwd: Path, *, spawn: Spawn | None = None) -> None:
        self._cwd = cwd
        self._spawn = spawn

    def send(self, request: Request, *, token: str | None, timeout: float) -> Outcome:
        del token  # gh resolves its own credentials
        try:
            argv, stdin = render_argv(request)
        except (TypeError, ValueError) as exc:
            return TransportFailure(FailureKind.ADAPTER_DEFECT, str(exc), "gh")
        spawn = self._spawn if self._spawn is not None else _spawn
        try:
            proc = spawn(argv, stdin, self._cwd, timeout)
        except FileNotFoundError as exc:
            return TransportFailure(FailureKind.CLI_MISSING, str(exc), "gh")
        except subprocess.TimeoutExpired:
            detail = f"gh command timed out after {timeout:g}s: {' '.join(argv)}"
            return TransportFailure(FailureKind.TIMEOUT, detail, "gh")
        except OSError as exc:
            return TransportFailure(
                FailureKind.ADAPTER_DEFECT, f"{type(exc).__name__}: {exc}", "gh"
            )
        if isinstance(request, CliRequest):
            return self._cli_response(proc)
        return self._api_outcome(request, proc)

    @staticmethod
    def _cli_response(proc: "subprocess.CompletedProcess[str]") -> Outcome:
        stdout = (proc.stdout or "").strip()
        if proc.returncode == 0:
            return Response(200, (), stdout, "gh", returncode=0)
        stderr = (proc.stderr or "").strip()
        return Response(0, (), stderr or stdout, "gh", returncode=proc.returncode)

    @staticmethod
    def _api_outcome(request: Request, proc: "subprocess.CompletedProcess[str]") -> Outcome:
        parsed = parse_include_output(proc.stdout or "")
        if parsed is None:
            return GhAdapter._classify_no_response(proc)
        status, headers, body = parsed
        errors = ()
        if isinstance(request, GraphQLRequest) and 200 <= status < 300:
            found = graphql_errors_from_body(body)
            if found is None:
                detail = "GraphQL response was not a JSON object"
                return TransportFailure(FailureKind.ADAPTER_DEFECT, detail, "gh")
            errors = found
        return Response(status, headers, body, "gh", errors, returncode=proc.returncode)

    @staticmethod
    def _classify_no_response(proc: "subprocess.CompletedProcess[str]") -> TransportFailure:
        text = (proc.stderr or "").strip() or (proc.stdout or "").strip()
        text = text or f"gh exited {proc.returncode}"
        if is_pre_connection_text(text):
            return TransportFailure(FailureKind.CONNECT, text, "gh")
        if is_transport_class_text(text):
            return TransportFailure(FailureKind.SENT_NO_RESPONSE, text, "gh")
        return TransportFailure(FailureKind.ADAPTER_DEFECT, text, "gh")
