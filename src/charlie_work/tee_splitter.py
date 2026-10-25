"""Tee a worker CLI's stdout to the ``.log`` and the ``events.jsonl`` sidecar.

Spawned as a separate child process (issue #2814 rework) rather than drained
by a daemon thread in the launching supervisor process:

    [sys.executable, tee_splitter.py, <log_path>, <events_path>]

with the worker CLI's ``stdout`` wired to this process's ``stdin`` (a plain
OS pipe between the two children). Because the pipe's reader lives here, the
worker keeps working — and the log and events sidecar keep receiving — even
when the launching supervisor exits (a self-deploy restart or a fleet
pause). With the previous launcher-thread drain the supervisor's exit closed
the pipe and the worker's next stdout write failed with a broken pipe
(observed directly on this host: a worker child dies by
``OSError: [Errno 22] Invalid argument`` on Windows / SIGPIPE-equivalent on
POSIX), killing any session unlucky enough to print after a restart.

Reads stdin in binary and lines are flushed to both destinations as they
arrive, so a watchdog reading either file sees live activity exactly as the
old thread-based tee provided.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> None:
    # Two destinations: the plaintext .log, then the events.jsonl sidecar.
    # Opened without a `with` so both stay available for the whole stream —
    # they are closed together in the finally below. Binary passthrough,
    # like the non-tee path writes the .log: never re-encode the CLI's
    # bytes.
    destinations = [open(path, mode="wb") for path in argv[:2]]  # noqa: SIM115
    try:
        for line in sys.stdin.buffer:
            for destination in destinations:
                destination.write(line)
                destination.flush()
    finally:
        for destination in destinations:
            try:
                destination.close()
            except OSError:
                pass


if __name__ == "__main__":
    main(sys.argv[1:])
