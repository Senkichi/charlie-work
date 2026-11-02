"""Issue #2101: the auto-deescalation sweep must also reset the issue-entry
windowed caps (``redispatch_at`` and its siblings) that gate the cleared
escalation reason.  Split out of ``test_issue_1683_review_dispatch_deescalation``
to keep that file under the file-size cap (#1442 ratchet)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state, state_lock
from charlie_work.unescalate_reset_fields import ISSUE_BUDGET_RESET_BY_ESCALATION_REASON

from _unescalate_fixtures import _app, _events


@pytest.mark.parametrize(
    ("reason", "field"),
    [
        ("redispatch_cap_exceeded", "redispatch_at"),
        ("worker_death_loop", "worker_death_at"),
        ("dispatch_blocked_environment", "blocked_environment_at"),
        ("dispatch_failed_cap_exceeded", "dispatch_failed_at"),
    ],
)
def test_sweep_resets_redispatch_at_for_redispatch_cap_exceeded(
    tmp_path: Path, reason: str, field: str
) -> None:
    """Issue #2101: the no-op rework cap gates on the issue-entry
    ``redispatch_at`` window, so a clear that leaves it full re-escalates on
    the next detection. The clear must drop it and report a real reset."""
    app = _app(tmp_path)
    now = datetime.now(UTC)
    full_window = [(now - timedelta(minutes=i)).isoformat() for i in range(5)]
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "escalated",
            "escalation_reason": reason,
        }
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": reason,
            "reason_class": "mechanical",
            "terminal_since": now.isoformat(),
            field: full_window,
        }
        save_state(app.paths.state_file, state)

    app._maybe_deescalate_mechanical()

    state = load_state(app.paths.state_file)
    issue_123 = state["issues"]["123"]
    assert issue_123["status"] == PASSIVE_OPEN_STATUS
    assert field not in issue_123
    cleared = _events(state, "deescalation_cleared")
    assert cleared[0]["payload"]["rework_budget_reset"] is True


def test_issue_budget_reset_fields_are_operator_door_fields() -> None:
    from charlie_work.unescalate_reset_fields import UNESCALATE_ISSUE_RESET_FIELDS

    for fields in ISSUE_BUDGET_RESET_BY_ESCALATION_REASON.values():
        assert set(fields) <= set(UNESCALATE_ISSUE_RESET_FIELDS)
