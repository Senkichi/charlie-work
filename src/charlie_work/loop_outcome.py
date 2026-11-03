"""The headline ``(ok, message)`` of one ``loop`` pass, derived from its lanes.

Extracted byte-for-byte from the tail of ``orchestration.reap_loop._loop_body``
(issue #2763) so that module stays under the file-size cap. Pure: it reads the
five lane results and the pass's per-PR error list and decides nothing else.
Lives outside ``orchestration/`` because every top-level ``def`` there becomes an
``OrchestratorApp`` method.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .command_result import CommandResult


def loop_pass_outcome(
    intake: CommandResult,
    dispatch: CommandResult,
    dispatch_rework: CommandResult,
    dispatch_reviews: CommandResult,
    local_lane_result: CommandResult,
    errors: Sequence[Any],
) -> tuple[bool, str]:
    """``ok`` iff every lane is ok and no PR errored; ``message`` names the first failure."""
    ok = (
        intake.ok
        and dispatch.ok
        and dispatch_rework.ok
        and dispatch_reviews.ok
        and local_lane_result.ok
        and not errors
    )
    message = "loop complete"
    if errors:
        message = f"loop completed with {len(errors)} PR error(s)"
    elif not intake.ok:
        message = "loop completed with intake failures"
    elif not dispatch.ok:
        message = dispatch.message
    elif not dispatch_rework.ok:
        message = dispatch_rework.message
    elif not dispatch_reviews.ok:
        message = "loop completed with review dispatch failures"
    elif not local_lane_result.ok:
        message = "loop completed with local-lane failures"
    return ok, message
