"""Reviewer-quota and provider-throttle state: probe_after deferral, throttle records, quota clearing.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from typing import Any
from charlie_work.state import is_throttled


def future_timestamp(*, days: int = 3650) -> str:
    """An ISO-8601 ``Z`` timestamp guaranteed to be in the future.

    Several state predicates decide behaviour by comparing a stored timestamp
    against ``datetime.now(UTC)`` -- ``is_reviewer_quota_exhausted`` (true only
    while ``throttled_until`` is future), ``is_reviewer_probe_ready``,
    ``is_throttled``. A test that hardcodes an absolute date to satisfy one of
    those preconditions is a time bomb: it passes until wall-clock time crosses
    the literal, then fails permanently, on every branch at once.

    That is not a theoretical concern -- a hardcoded ``2026-08-01T00:00:00Z``
    did exactly this, turning every open PR and main itself red at that instant
    and blocking the merge lane. Use this helper for any timestamp whose
    *futureness* is load-bearing.

    Timestamps that are merely round-tripped or compared for equality do not
    need this; an absolute literal is clearer there, and stable.
    """
    return (datetime.now(UTC) + timedelta(days=days)).isoformat().replace("+00:00", "Z")


def test_clear_reviewer_quota_drops_alerted_at() -> None:
    """clear_reviewer_quota must also pop alerted_at so a later exhaustion
    episode alerts again instead of staying silently suppressed."""
    from charlie_work.state import (
        clear_reviewer_quota,
        mark_reviewer_quota_alerted,
        set_reviewer_quota_exhausted,
    )

    state: dict[str, Any] = {"version": 1, "issues": {}, "prs": {}, "events": []}
    state = set_reviewer_quota_exhausted(
        state, throttled_until="2026-08-01T00:00:00Z", probe_after="2026-08-01T00:00:00Z"
    )
    state = mark_reviewer_quota_alerted(state)
    assert "alerted_at" in state["reviewer_quota"]

    cleared = clear_reviewer_quota(state)

    assert "alerted_at" not in cleared["reviewer_quota"]
    assert "throttled_until" not in cleared["reviewer_quota"]


def test_defer_reviewer_probe_after_bumps_past_probe_after() -> None:
    """A red flat probe must bump reviewer_quota.probe_after forward so
    dispatch_reviews's probe_mode gate defers instead of independently
    launching a real reviewer session into the same still-closed window
    (issue #663)."""
    from charlie_work.state import defer_reviewer_probe_after, set_reviewer_quota_exhausted

    state: dict[str, Any] = {"version": 1, "issues": {}, "prs": {}, "events": []}
    # Quota exhausted, probe_after in the past (ready to probe).
    state = set_reviewer_quota_exhausted(
        state,
        throttled_until="2099-01-01T00:00:00Z",
        probe_after="2020-01-01T00:00:00Z",
    )
    result = defer_reviewer_probe_after(state, "2026-08-01T00:30:00Z")
    assert result["reviewer_quota"]["probe_after"] == "2026-08-01T00:30:00Z"
    # throttled_until is untouched.
    assert result["reviewer_quota"]["throttled_until"] == "2099-01-01T00:00:00Z"


def test_defer_reviewer_probe_after_noop_when_quota_not_exhausted() -> None:
    """Must not write probe_after on a non-exhausted quota -- that would
    leave stale state for no reason (issue #663)."""
    from charlie_work.state import defer_reviewer_probe_after

    state: dict[str, Any] = {"version": 1, "issues": {}, "prs": {}, "events": []}
    result = defer_reviewer_probe_after(state, "2026-08-01T00:30:00Z")
    assert "probe_after" not in result.get("reviewer_quota", {})


def test_defer_reviewer_probe_after_never_moves_earlier() -> None:
    """If the reviewer quota's own exponential backoff already pushed
    probe_after further out than the flat probe's interval, the bump must
    not shorten it -- that would make dispatch_reviews probe more often,
    not less (issue #663)."""
    from charlie_work.state import defer_reviewer_probe_after, set_reviewer_quota_exhausted

    state: dict[str, Any] = {"version": 1, "issues": {}, "prs": {}, "events": []}
    state = set_reviewer_quota_exhausted(
        state,
        throttled_until="2099-01-01T00:00:00Z",
        probe_after="2026-08-01T04:00:00Z",
    )
    result = defer_reviewer_probe_after(state, "2026-08-01T00:30:00Z")
    assert result["reviewer_quota"]["probe_after"] == "2026-08-01T04:00:00Z"


def test_defer_reviewer_probe_after_overwrites_malformed_current() -> None:
    """A malformed current probe_after must not wedge the bump -- overwrite
    with the well-formed new value (issue #663)."""
    from charlie_work.state import defer_reviewer_probe_after, set_reviewer_quota_exhausted

    state: dict[str, Any] = {"version": 1, "issues": {}, "prs": {}, "events": []}
    state = set_reviewer_quota_exhausted(
        state,
        throttled_until="2099-01-01T00:00:00Z",
        probe_after="not-a-timestamp",
    )
    result = defer_reviewer_probe_after(state, "2026-08-01T00:30:00Z")
    assert result["reviewer_quota"]["probe_after"] == "2026-08-01T00:30:00Z"


def test_set_throttled_until_records_reason_and_adapter_kind() -> None:
    from charlie_work.state import empty_state, set_throttled_until

    state = set_throttled_until(
        empty_state(),
        "2026-08-01T00:00:00Z",
        reason="quota_exhausted",
        adapter_kind="claude-code",
    )

    assert state["throttled_until"] == "2026-08-01T00:00:00Z"
    assert state["throttle_reason"] == "quota_exhausted"
    assert state["throttle_adapter_kind"] == "claude-code"


def test_set_throttled_until_defaults_reason_and_adapter_kind_to_none() -> None:
    from charlie_work.state import empty_state, set_throttled_until

    state = set_throttled_until(empty_state(), "2026-08-01T00:00:00Z")

    assert state["throttle_reason"] is None
    assert state["throttle_adapter_kind"] is None


def test_quota_probe_arm_disarm_and_due_lifecycle() -> None:
    from datetime import UTC, datetime, timedelta

    from charlie_work.state import (
        arm_quota_probe,
        disarm_quota_probe,
        empty_state,
        is_quota_probe_armed,
        is_quota_probe_due,
    )

    state = empty_state()
    assert is_quota_probe_armed(state) is False
    assert is_quota_probe_due(state) is False  # unarmed is never "due"

    future = (datetime.now(UTC) + timedelta(minutes=15)).isoformat().replace("+00:00", "Z")
    state = arm_quota_probe(state, future)
    assert is_quota_probe_armed(state) is True
    assert is_quota_probe_due(state) is False

    past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    state = arm_quota_probe(state, past)
    assert is_quota_probe_due(state) is True

    state = disarm_quota_probe(state)
    assert is_quota_probe_armed(state) is False


def test_is_quota_probe_due_treats_malformed_timestamp_as_due() -> None:
    from charlie_work.state import arm_quota_probe, empty_state, is_quota_probe_due

    state = arm_quota_probe(empty_state(), "not-a-timestamp")

    assert is_quota_probe_due(state) is True


def test_any_quota_exhausted_indicator_gate() -> None:
    from datetime import UTC, datetime, timedelta

    from charlie_work.state import (
        any_quota_exhausted_indicator,
        empty_state,
        set_reviewer_quota_exhausted,
        set_throttled_until,
    )

    assert any_quota_exhausted_indicator(empty_state()) is False

    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    root_throttled = set_throttled_until(empty_state(), future, reason="rate_limited")
    assert any_quota_exhausted_indicator(root_throttled) is True

    reviewer_throttled = set_reviewer_quota_exhausted(
        empty_state(), throttled_until=future, probe_after=future
    )
    assert any_quota_exhausted_indicator(reviewer_throttled) is True


def test_clear_quota_throttles_clears_root_throttle_for_claude_code_or_unset_adapter() -> None:
    from charlie_work.state import clear_quota_throttles, empty_state, set_throttled_until

    for adapter_kind in (None, "claude-code"):
        state = set_throttled_until(
            empty_state(),
            "2026-08-01T00:00:00Z",
            reason="rate_limited",
            adapter_kind=adapter_kind,
        )

        cleared = clear_quota_throttles(state)

        assert cleared["throttled_until"] is None
        assert cleared["throttle_reason"] is None
        assert cleared["throttle_adapter_kind"] is None


def test_clear_quota_throttles_preserves_provider_auth_throttle() -> None:
    """A dead key does not self-heal within minutes -- see
    claude_code._classify_session_failure; a green probe must not mask it."""
    from charlie_work.state import clear_quota_throttles, empty_state, set_throttled_until

    state = set_throttled_until(
        empty_state(),
        "2026-08-01T00:00:00Z",
        reason="provider_auth",
        adapter_kind="claude-code",
    )

    cleared = clear_quota_throttles(state)

    assert cleared["throttled_until"] == "2026-08-01T00:00:00Z"
    assert cleared["throttle_reason"] == "provider_auth"


def test_clear_quota_throttles_preserves_non_claude_code_adapter_throttle() -> None:
    """A devin/api-adapter throttle is on a different credential the ambient
    Claude Code CLI probe provides no evidence about."""
    from charlie_work.state import clear_quota_throttles, empty_state, set_throttled_until

    for adapter_kind in ("devin", "api"):
        state = set_throttled_until(
            empty_state(),
            "2026-08-01T00:00:00Z",
            reason="rate_limited",
            adapter_kind=adapter_kind,
        )

        cleared = clear_quota_throttles(state)

        assert cleared["throttled_until"] == "2026-08-01T00:00:00Z"
        assert cleared["throttle_adapter_kind"] == adapter_kind


def test_clear_quota_throttles_always_clears_reviewer_quota_and_resets_probe_failures() -> None:
    from charlie_work.state import (
        clear_quota_throttles,
        empty_state,
        set_reviewer_quota_exhausted,
        set_throttled_until,
    )

    # The reviewer-quota throttle MUST be derived from now, never hardcoded.
    # ``clear_quota_throttles`` only resets ``consecutive_probe_failures`` when
    # ``is_reviewer_quota_exhausted(data) or cleared_root`` holds, and
    # ``is_reviewer_quota_exhausted`` is true only while ``throttled_until`` is
    # still in the future (``datetime.now(UTC) < throttle_time``). This test
    # deliberately uses a *devin* root throttle below so ``cleared_root`` is
    # False, which makes the exhaustion check the sole path to the reset --
    # so the instant a hardcoded date goes stale, the assertion below flips
    # from testing the reset to testing nothing, and fails.
    #
    # This is not hypothetical: this test pinned "2026-08-01T00:00:00Z" and
    # began failing on every PR and on main at exactly that instant, blocking
    # the merge lane until the date was made relative.
    future = future_timestamp(days=3650)
    # Opaque by contrast: the root throttle is only ever round-tripped and
    # compared for equality, never against the clock, so an absolute literal is
    # safe here and keeps the "untouched" assertion easy to read.
    root_throttle = "2026-08-01T00:00:00Z"

    state = set_reviewer_quota_exhausted(empty_state(), throttled_until=future, probe_after=future)
    state = {
        **state,
        "reviewer_quota": {**state["reviewer_quota"], "consecutive_probe_failures": 3},
    }
    # Also carry a devin-adapter root throttle, to confirm reviewer_quota
    # clears independently of what the root-throttle branch decides.
    state = set_throttled_until(state, root_throttle, reason="rate_limited", adapter_kind="devin")

    cleared = clear_quota_throttles(state)

    assert "throttled_until" not in cleared["reviewer_quota"]
    assert cleared["reviewer_quota"]["consecutive_probe_failures"] == 0
    # The devin adapter's root throttle must still be untouched.
    assert cleared["throttled_until"] == root_throttle


def test_clear_quota_throttles_records_last_probe_cleared_at() -> None:
    """Issue #662: ``clear_quota_throttles`` stamps ``last_probe_cleared_at``
    on reviewer_quota so the dead-reviewer reap sweep can tell a recovery
    happened. It is recorded even when reviewer_quota was never exhausted
    (a green probe clearing a root-only throttle still proves the provider
    recovered), and survives ``clear_reviewer_quota`` across episodes.
    """
    from charlie_work.state import (
        clear_quota_throttles,
        clear_reviewer_quota,
        empty_state,
        reviewer_quota_last_probe_cleared_at,
        set_reviewer_quota_exhausted,
        set_throttled_until,
    )

    # Reviewer-quota exhaustion present: marker recorded after clear.
    state = set_reviewer_quota_exhausted(
        empty_state(), throttled_until="2026-08-01T00:00:00Z", probe_after="2026-08-01T00:00:00Z"
    )
    cleared = clear_quota_throttles(state)
    assert reviewer_quota_last_probe_cleared_at(cleared) is not None
    assert "throttled_until" not in cleared["reviewer_quota"]

    # Root-only throttle (reviewer_quota never set): marker still recorded.
    root_only = set_throttled_until(
        empty_state(), "2026-08-01T00:00:00Z", reason="rate_limited", adapter_kind="claude-code"
    )
    cleared_root = clear_quota_throttles(root_only)
    assert reviewer_quota_last_probe_cleared_at(cleared_root) is not None
    assert "throttled_until" not in cleared_root["reviewer_quota"]

    # The marker survives a subsequent clear_reviewer_quota (new episode).
    re_exhausted = set_reviewer_quota_exhausted(
        cleared_root, throttled_until="2026-09-01T00:00:00Z", probe_after="2026-09-01T00:00:00Z"
    )
    re_cleared = clear_reviewer_quota(re_exhausted)
    assert reviewer_quota_last_probe_cleared_at(
        re_cleared
    ) == reviewer_quota_last_probe_cleared_at(cleared_root)


def test_is_throttled_checks_against_current_time(tmp_path: Path) -> None:
    """is_throttled should return True only when now < throttled_until."""
    from datetime import UTC, datetime, timedelta

    # Test with future timestamp
    future_time = datetime.now(UTC) + timedelta(hours=1)
    throttled_until = future_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    state = {"throttled_until": throttled_until}
    assert is_throttled(state) is True

    # Test with past timestamp
    past_time = datetime.now(UTC) - timedelta(hours=1)
    throttled_until = past_time.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    state = {"throttled_until": throttled_until}
    assert is_throttled(state) is False

    # Test with no throttled_until
    state = {"throttled_until": None}
    assert is_throttled(state) is False

    # Test with malformed timestamp
    state = {"throttled_until": "invalid-timestamp"}
    assert is_throttled(state) is False
