"""Wait for a marker file a spawned worker writes, without racing its content.

Adapter launch tests spawn a fake worker with ``subprocess.Popen`` and then read
a marker file the child writes. They must poll, because the adapters return
immediately by design (CLAUDE.md: "Adapters must not block on worker
completion") — there is no handle to wait on.

The bug this module exists to remove: polling for ``path.exists()`` and then
reading is not the same as polling for the content. ``Path.write_text()`` in the
child creates the file first and writes to it second, so a test that stops
waiting the moment the path appears can legitimately observe a zero-byte or
half-written file. Under load — a busy CI runner, another job on the same host —
the create/write gap widens and the read loses:

    FAILED test_launch_claude_worker_prompt_path_placeholder_skips_stdin
        AssertionError: assert '' == 'prompt payload for argv'
    FAILED test_launch_api_worker_worker_env_merged_under_provider_env
        ValueError: not enough values to unpack (expected 2, got 1)

Both are the same race: existence observed, content not yet there. The second is
the nastier shape — a partially written probe parses as the wrong number of
fields rather than as an obviously empty read.

So wait on the thing the assertion actually depends on. ``read_worker_marker``
waits for a non-empty read (optionally for one exact expected value) and returns
the text, so callers assert on a value that was complete when it was read.
"""

from __future__ import annotations

import time
from pathlib import Path

DEFAULT_TIMEOUT_S = 10.0
_POLL_INTERVAL_S = 0.05


def read_worker_marker(
    path: Path,
    *,
    expected: str | None = None,
    reason: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Return the marker's text once the worker has finished writing it.

    Waits for the file to exist AND to have content. When ``expected`` is given,
    waits for exactly that text, which removes the partial-read race entirely:
    a prefix of the expected value keeps polling instead of failing.

    ``reason`` states the invariant being checked and is included in the failure
    message. Pass it instead of following this call with an
    ``assert text == expected`` — once ``expected`` is supplied that assert can
    never fail, so it documents the contract while testing nothing.

    Raises AssertionError with the observed state on timeout — never returns a
    value the caller would then compare against a half-written read.
    """
    deadline = time.monotonic() + timeout
    last: str | None = None

    while time.monotonic() < deadline:
        try:
            last = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            last = None
        except OSError:
            # Windows can briefly deny a read while the child holds the handle;
            # that is indistinguishable here from "not ready", so keep waiting
            # rather than turning a timing artifact into a hard failure.
            last = None
        else:
            if expected is None:
                if last != "":
                    return last
            elif last == expected:
                return last

        time.sleep(_POLL_INTERVAL_S)

    why = f"\n  invariant: {reason}" if reason else ""
    if last is None:
        raise AssertionError(
            f"worker never created {path} within {timeout}s "
            f"(parent dir exists: {path.parent.exists()}){why}"
        )
    if expected is None:
        raise AssertionError(f"worker left {path} empty after {timeout}s{why}")
    raise AssertionError(
        f"worker did not write the expected marker to {path} within {timeout}s:\n"
        f"  expected: {expected!r}\n"
        f"  observed: {last!r}{why}"
    )
