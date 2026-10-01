"""Detached full-suite runner for the local merge gate (issue #1974).

The merge gate launches this module as a long-lived child --

    python -m charlie_work.local_suite_runner \
        --cwd <branch worktree> --log <suite.log> --result <suite-result.json> \
        --pid-file <suite-runner.json> --head-sha <sha> --base-sha <sha> \
        -- <suite argv...>

-- so the suite's outcome survives a supervisor crash or kill. The wrapper:

1. writes ``suite-runner.json`` (pid, process start-time fingerprint, claim
   fields) atomically *before* spawning the suite -- the crash-safety anchor a
   restarted supervisor uses to re-attach a gate whose state write never
   landed;
2. runs the suite argv with stdout+stderr streamed to ``suite.log``;
3. writes ``suite-result.json`` atomically on exit (returncode + ok + the same
   claim fields), because a bare pid can never yield the exit code once the
   launcher is gone.

The supervisor side reads the two files through :func:`suite_gate_paths`,
:func:`read_gate_result`, and :func:`read_gate_pid`, and spawns the wrapper
through :func:`launch_suite_gate`. Only the pid file is a liveness claim; a
result file is accepted only when its ``head_sha`` matches the head under
test, so a stale artifact can never be mistaken for the current gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from charlie_work.process_utils import CpuPriority, get_process_start_time, popen_worker
from charlie_work.subprocess_runner import hidden_console_kwargs

LOG_FILENAME = "suite.log"
RESULT_FILENAME = "suite-result.json"
RUNNER_STATUS_FILENAME = "suite-runner.json"
RUNNER_STDERR_FILENAME = "suite-runner-stderr.log"

_GATE_DIR_NAME = "local-merge-gate"


@dataclass(frozen=True)
class SuiteGatePaths:
    """The four on-disk artifacts one gate owns inside its dispatch dir."""

    gate_dir: Path
    log: Path
    result: Path
    pid_file: Path
    runner_stderr: Path


def suite_gate_paths(dispatches_dir: Path, pr_number: int) -> SuiteGatePaths:
    """The gate's artifact dir for one local record: ``<dispatches>/local-merge-gate/pr-<n>``."""
    gate_dir = dispatches_dir / _GATE_DIR_NAME / f"pr-{pr_number}"
    return SuiteGatePaths(
        gate_dir=gate_dir,
        log=gate_dir / LOG_FILENAME,
        result=gate_dir / RESULT_FILENAME,
        pid_file=gate_dir / RUNNER_STATUS_FILENAME,
        runner_stderr=gate_dir / RUNNER_STDERR_FILENAME,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Temp-file + ``replace()`` -- the repo's mandatory JSON write shape."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    """Parse a gate artifact; missing/unreadable/malformed all return None.

    A malformed artifact is deliberately indistinguishable from an absent one:
    the gate-side caller treats "no usable result" as not-green, never as a
    passed suite.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_gate_result(paths: SuiteGatePaths) -> dict[str, Any] | None:
    """The wrapper's terminal report, or None while absent/unparseable."""
    return _read_json(paths.result)


def read_gate_pid(paths: SuiteGatePaths) -> dict[str, Any] | None:
    """The wrapper's liveness claim, or None while absent/unparseable."""
    return _read_json(paths.pid_file)


def read_log_tail(log_path: Path, limit: int = 4000) -> str:
    """Bounded tail of the suite log for rework notes and event payloads."""
    try:
        data = log_path.read_bytes()
    except OSError:
        return ""
    return data[-limit:].decode("utf-8", errors="surrogateescape")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Supervisor side: spawn + argv shape
# ---------------------------------------------------------------------------


def gate_runner_argv(
    worktree_path: Path,
    suite_argv: Sequence[str],
    *,
    paths: SuiteGatePaths,
    head_sha: str,
    base_sha: str | None,
) -> list[str]:
    """The ``python -m`` invocation that runs one gate.

    ``--cwd`` pins the suite's working tree; the artifact paths stay under the
    lane's dispatch dir. ``--head-sha``/``--base-sha`` are echoed verbatim into
    both artifacts so the supervisor can reject stale files and re-check that
    the base did not move under the suite.
    """
    # ``__package__`` is "charlie_work" whether this module is imported or run
    # via ``-m``; building the dotted name from it keeps the spelling derived
    # from the module's own location instead of a literal.
    module_name = f"{__package__}.{Path(__file__).stem}"
    return [
        sys.executable,
        "-m",
        module_name,
        "--cwd",
        str(worktree_path),
        "--log",
        str(paths.log),
        "--result",
        str(paths.result),
        "--pid-file",
        str(paths.pid_file),
        "--head-sha",
        head_sha or "",
        "--base-sha",
        base_sha or "",
        "--",
        *suite_argv,
    ]


@dataclass(frozen=True)
class SuiteGateLaunch:
    """Outcome of the non-blocking gate spawn -- errors as values, never raised."""

    pid: int | None
    process_start_time: float | None
    paths: SuiteGatePaths
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.pid is not None


def launch_suite_gate(
    worktree_path: Path,
    suite_argv: Sequence[str],
    *,
    paths: SuiteGatePaths,
    head_sha: str,
    base_sha: str | None,
) -> SuiteGateLaunch:
    """Spawn the detached suite runner; return immediately with its pid.

    Stale artifacts from any earlier attempt are removed first so a fresh
    claim can never read another epoch's result. The wrapper's own stderr
    goes to ``suite-runner-stderr.log`` -- its stdout is discarded, but a
    crash before the result write should still leave a traceback somewhere.
    """
    try:
        paths.gate_dir.mkdir(parents=True, exist_ok=True)
        for stale in (paths.result, paths.pid_file, paths.runner_stderr):
            stale.unlink(missing_ok=True)
        argv = gate_runner_argv(
            worktree_path,
            suite_argv,
            paths=paths,
            head_sha=head_sha,
            base_sha=base_sha,
        )
        stderr_handle = paths.runner_stderr.open("wb")
        try:
            proc = popen_worker(
                argv,
                priority=CpuPriority.NORMAL,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_handle,
            )
        finally:
            stderr_handle.close()
    except OSError as exc:
        return SuiteGateLaunch(pid=None, process_start_time=None, paths=paths, error=str(exc))
    return SuiteGateLaunch(
        pid=proc.pid,
        process_start_time=get_process_start_time(proc.pid),
        paths=paths,
    )


# ---------------------------------------------------------------------------
# Child side: ``python -m charlie_work.local_suite_runner``
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split ``--`` off the tail: everything after it is the suite argv."""
    argv = list(argv)
    try:
        sep = argv.index("--")
    except ValueError as exc:
        raise SystemExit(
            "local_suite_runner: missing '--' separator before the suite argv"
        ) from exc
    parser = argparse.ArgumentParser(prog="charlie_work.local_suite_runner")
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--pid-file", required=True, dest="pid_file")
    parser.add_argument("--head-sha", default="", dest="head_sha")
    parser.add_argument("--base-sha", default="", dest="base_sha")
    return parser.parse_args(argv[:sep]), argv[sep + 1 :]


def main(argv: Sequence[str] | None = None) -> int:
    """Write the liveness claim, run the suite, write the result atomically.

    Exit code convention: the process result lives in ``suite-result.json``,
    not the exit status -- the supervisor never waits on this process. The
    wrapper returns 0 whenever it reached the point of writing a result;
    anything else means the claim write or the result write itself failed.
    """
    args, suite_argv = _parse_args(sys.argv[1:] if argv is None else argv)
    if not suite_argv:
        print("local_suite_runner: empty suite argv after '--'", file=sys.stderr)
        return 2
    cwd = Path(args.cwd)
    log_path = Path(args.log)
    result_path = Path(args.result)
    pid_path = Path(args.pid_file)
    started_at = _utc_now_iso()
    t0 = time.monotonic()
    pid = os.getpid()
    identity = {
        "pid": pid,
        "process_start_time": get_process_start_time(pid),
        "cwd": str(cwd),
        "suite_argv": list(suite_argv),
        "head_sha": args.head_sha or None,
        "base_sha": args.base_sha or None,
        "started_at": started_at,
    }
    try:
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(pid_path, identity)
    except OSError as exc:
        print(f"local_suite_runner: pid-file write failed: {exc}", file=sys.stderr)
        return 2

    error: str | None = None
    returncode: int | None = None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log_handle:
            proc = subprocess.run(
                list(suite_argv),
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                **hidden_console_kwargs(),
            )
        returncode = proc.returncode
    except OSError as exc:
        error = str(exc)

    result = {
        **identity,
        "returncode": returncode,
        "ok": returncode == 0,
        "error": error,
        "ended_at": _utc_now_iso(),
        "duration_seconds": round(time.monotonic() - t0, 3),
    }
    try:
        _write_json_atomic(result_path, result)
    except OSError as exc:
        print(f"local_suite_runner: result write failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
