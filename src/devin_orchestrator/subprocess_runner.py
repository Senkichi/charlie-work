"""One subprocess runner for every adapter and cross-family invocation.

Centralizes the Windows-safe capture contract: text mode with explicit UTF-8
decoding and ``errors="replace"`` (never the cp1252 default), and bytes-safe
handling of ``TimeoutExpired`` partial output. Callers get a plain result and
never an encoding crash.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value) if value else ""


def run_captured(
    command: list[str] | str,
    *,
    cwd: Path | str,
    timeout_seconds: int,
    shell: bool = False,
) -> RunResult:
    """Run ``command`` and capture output. Never raises for runtime failures —
    timeouts, missing binaries, and non-zero exits all come back as a result."""
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
            shell=shell,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return RunResult(
            returncode=None,
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
            timed_out=True,
            error=f"command timed out after {timeout_seconds}s",
        )
    except OSError as exc:
        return RunResult(returncode=None, stdout="", stderr="", error=str(exc))
    except subprocess.SubprocessError as exc:
        return RunResult(returncode=None, stdout="", stderr="", error=str(exc))
    return RunResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        error=None if completed.returncode == 0 else f"command exited {completed.returncode}",
    )
