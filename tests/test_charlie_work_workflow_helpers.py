"""Small workflow.py free-function helpers: slugify and sink_census.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

from pathlib import Path
from charlie_work.state import PASSIVE_OPEN_STATUS
from charlie_work.workflow import (
    sink_census,
    slugify,
)


def test_slugify_makes_branch_safe_slug() -> None:
    assert slugify("Fix: Search / Windows path!!!") == "fix-search-windows-path"


def test_sink_census_counts_escalated_and_blocked_only(tmp_path: Path) -> None:
    """Issue #1083: ``sink_census`` reads the sink population from state.

    The sink is exactly the issues whose ``status`` is ``escalated`` or
    ``blocked`` -- the in-state mirror of the ``agent:human-needed`` label.
    Other terminal-ish statuses (``done``, ``merged``) and non-digit keys
    are excluded so the census matches the de-escalation sweep's own
    selection query.
    """
    state = {
        "issues": {
            "101": {"number": 101, "status": "escalated", "reason_class": "judgment"},
            "102": {"number": 102, "status": "blocked", "reason_class": "mechanical"},
            "103": {"number": 103, "status": "done"},
            "104": {"number": 104, "status": PASSIVE_OPEN_STATUS},
            "105": {"number": 105, "status": "rework_requested"},
            "not-an-issue": {"number": 0, "status": "escalated"},
        }
    }
    assert sink_census(state) == {101, 102}
    # An empty / malformed issues map is safe.
    assert sink_census({}) == set()
    assert sink_census({"issues": "not-a-dict"}) == set()
