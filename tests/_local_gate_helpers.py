"""Shared driver helpers for the asynchronous local merge gate (issue #1974).

``_local_merge_approved`` no longer runs the suite inline: the first call
sync-merges and *launches* the detached ``local_suite_runner`` child, and a
later call resolves the result file it writes. Tests that exercise a
complete gate episode -- green merge, red rework, deferred final merge --
need the same launch -> poll-for-result -> resolve sequence, hoisted here
rather than duplicated across ``test_local_lane.py`` (which is at its
bound-member cap and cannot grow new helpers) and
``test_local_issues_merge_gate.py``.
"""

from __future__ import annotations

import time
from typing import Any

from charlie_work import local_suite_runner
from charlie_work.workflow import OrchestratorApp


def wait_for_gate_result(
    app: OrchestratorApp, pr_number: int, timeout_seconds: float = 90
) -> dict[str, Any]:
    """Poll the gate's ``suite-result.json`` until the runner reports."""
    paths = local_suite_runner.suite_gate_paths(app.paths.dispatches, pr_number)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = local_suite_runner.read_gate_result(paths)
        if result is not None:
            return result
        time.sleep(0.1)
    raise AssertionError(f"suite result for pr-{pr_number} never appeared at {paths.result}")


def drive_merge_gate(
    app: OrchestratorApp, pr_number: int, timeout_seconds: float = 90
) -> list[dict[str, Any]]:
    """Run one complete gate episode: launch pass, wait for the suite result,
    resolve pass. Returns the resolving call's results list."""
    launch = app._local_merge_approved()
    claim = next(
        (r for r in launch if r.get("pr") == pr_number),
        {},
    )
    # Pre-suite outcomes (conflict, missing suite command) resolve on the
    # launch pass itself -- nothing was spawned, nothing to wait on.
    if claim.get("outcome") == "suite_launched":
        wait_for_gate_result(app, pr_number, timeout_seconds)
    return app._local_merge_approved()
