"""Tee wiring for a claude-code workflow worker (issue #2814 rework round 4).

Extracted out of ``claude_code.py`` (file-size ratchet): everything between
"the launch command is rendered" and "the worker process exists" when
``tee_stream_json`` is ON — the synchronous pre-create of both tee
destinations, the tee splitter child spawn, and the stdin handoff that wires
the worker's ``stdout`` to the splitter. One domain module, so the launch
adapter stays within its ratchet cap and the tee contract has one home.

Result type: :class:`TeeLaunched` on success, :class:`TeeLaunchError` for a
pre-spawn failure the CALLER turns into its standard launch error record
(worktree teardown stays with the caller, which owns it). An ``OSError`` from
the WORKER spawn is re-raised after the splitter's stdin is released — the
caller's own spawn handler produces the "failed to launch claude" record.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .process_utils import CpuPriority

# The tee-splitter child script spawned by :func:`launch_tee`. Tests that
# spawn the script directly resolve it from ``charlie_work.tee_splitter.__file__``
# (round 3 review: no test-only public accessor for this path).
TEE_SPLITTER_PATH = Path(__file__).with_name("tee_splitter.py")


@dataclass(frozen=True)
class TeeLaunchError:
    """A tee wiring step failed before the worker was wired to its stdout.

    ``error`` is the exact error-record message the caller records; the
    caller also tears down the worktree — a launch that produced no worker
    process must not leave the checkout behind."""

    error: str


@dataclass(frozen=True)
class TeeLaunched:
    """The tee splitter child and the wired worker process.

    The launcher's lifetime is already irrelevant to the pipe (the splitter
    child owns the read end); the launcher only closes its OWN duplicate of
    the splitter's stdin handle right after the handoff."""

    splitter: subprocess.Popen
    process: subprocess.Popen


def launch_tee(
    *,
    command: tuple[str, ...],
    cwd: str,
    prompt_path: Path,
    feed_stdin: bool,
    log_path: Path,
    events_path: Path,
    worker_env: Mapping[str, str],
) -> TeeLaunched | TeeLaunchError:
    """Wire the tee splitter child and spawn the worker under it.

    The tee is a separate child process (``tee_splitter.py``) that owns the
    pipe's read end: the worker CLI's stdout is wired to the splitter's
    stdin, so the pipe survives the launching supervisor process exiting
    (self-deploy restart, fleet pause). With the previous
    ``subprocess.PIPE`` + launcher daemon-thread drain, every tee'd worker
    was a restart casualty — the supervisor's exit closed the pipe and the
    worker's next stdout write failed (measured on this host:
    ``OSError: [Errno 22] Invalid argument`` on Windows, a broken pipe on
    POSIX), killing any session unlucky enough to print.

    Both tee destinations are pre-created/truncated synchronously BEFORE the
    splitter child exists (round 3 review): previously only the splitter
    child opened the files stream-side, so an unopenable destination
    (permissions, path shape) failed inside the child AFTER the worker was
    already wired to its stdin — a silently dead worker, no log, nothing to
    drain, and no error record. An OSError surfaces here as a
    :class:`TeeLaunchError` — the same launch error record every other
    pre-spawn failure produces.

    The splitter's own stdout/stderr are DEVNULL (round 4 review): it must
    not inherit and hold the supervisor's output pipes for a worker's
    lifetime.
    """
    # The spawn seam resolves through claude_code's module namespace at CALL
    # time (the module import at the top is the pre-extraction contract):
    # tests that stub ``claude_code.popen_worker`` (e.g. the launch-failed
    # events suite) must reach the tee path too, or a stubbed launch really
    # spawns the splitter — and then the worker — through this module.
    from .claude_code import popen_worker

    try:
        for tee_dest in (log_path, events_path):
            tee_dest.open("w", encoding="utf-8").close()
    except OSError as exc:
        return TeeLaunchError(error=f"failed to create tee log files: {exc}")
    try:
        tee_splitter = popen_worker(
            (
                sys.executable,
                str(TEE_SPLITTER_PATH),
                str(log_path),
                str(events_path),
            ),
            priority=CpuPriority.BELOW_NORMAL,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=worker_env,
        )
    except OSError as exc:
        return TeeLaunchError(error=f"failed to launch tee splitter: {exc}")

    try:
        if feed_stdin:
            prompt_handle = prompt_path.open("r", encoding="utf-8")
            try:
                process = popen_worker(
                    command,
                    priority=CpuPriority.BELOW_NORMAL,
                    cwd=cwd,
                    stdin=prompt_handle,
                    stdout=tee_splitter.stdin,
                    stderr=subprocess.STDOUT,
                    env=worker_env,
                    start_new_session=(os.name != "nt"),
                )
            finally:
                prompt_handle.close()
        else:
            process = popen_worker(
                command,
                priority=CpuPriority.BELOW_NORMAL,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=tee_splitter.stdin,
                stderr=subprocess.STDOUT,
                env=worker_env,
                start_new_session=(os.name != "nt"),
            )
    except OSError:
        # Worker spawn failed after the splitter was up: release the
        # splitter's stdin so it reads EOF and exits; re-raise so the
        # caller's own spawn handler produces the "failed to launch claude"
        # error record and the worktree teardown.
        tee_splitter.stdin.close()
        raise

    # Hand the pipe write end to the worker process entirely: only the
    # worker holds it now, so the splitter keeps draining for the worker's
    # whole lifetime regardless of our lifecycle.
    tee_splitter.stdin.close()
    return TeeLaunched(splitter=tee_splitter, process=process)
