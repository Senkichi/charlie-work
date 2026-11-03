"""``loop_pass_outcome``: the headline ok/message of a loop pass (extracted for #2763).

The block was moved byte-for-byte out of ``orchestration.reap_loop._loop_body``
to keep that module under the file-size cap; these pin every branch so the move
cannot drift. The all-ok case is the control for each failure case.
"""

from __future__ import annotations

import pytest

from charlie_work.command_result import CommandResult
from charlie_work.loop_outcome import loop_pass_outcome


def _res(ok: bool = True, message: str = "") -> CommandResult:
    return CommandResult(ok, message, {})


def _outcome(
    failing: dict[str, CommandResult] | None = None, *, errors: list[str] | None = None
) -> tuple[bool, str]:
    lanes = {
        "intake": _res(),
        "dispatch": _res(),
        "dispatch_rework": _res(),
        "dispatch_reviews": _res(),
        "local_lane_result": _res(),
    }
    lanes.update(failing or {})
    return loop_pass_outcome(**lanes, errors=errors or [])


def test_all_lanes_ok_and_no_errors_is_a_clean_pass() -> None:
    assert _outcome() == (True, "loop complete")


@pytest.mark.parametrize(
    ("lane", "message"),
    [
        ("intake", "loop completed with intake failures"),
        ("dispatch", "dispatch said no"),
        ("dispatch_rework", "rework said no"),
        ("dispatch_reviews", "loop completed with review dispatch failures"),
        ("local_lane_result", "loop completed with local-lane failures"),
    ],
)
def test_a_failed_lane_makes_the_pass_not_ok_and_names_itself(lane: str, message: str) -> None:
    failed = _res(False, message if lane in {"dispatch", "dispatch_rework"} else "ignored")

    assert _outcome({lane: failed}) == (False, message)


def test_pr_errors_win_over_a_failed_lane_and_are_counted() -> None:
    ok, message = _outcome({"intake": _res(False)}, errors=["a", "b"])

    assert (ok, message) == (False, "loop completed with 2 PR error(s)")


def test_the_first_failing_lane_in_order_names_the_message() -> None:
    ok, message = _outcome({"intake": _res(False), "dispatch": _res(False, "later")})

    assert (ok, message) == (False, "loop completed with intake failures")
