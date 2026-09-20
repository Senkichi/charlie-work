"""Windowed redispatch accounting: escalated-edge clearing, escalation cap, timestamp pruning, known-call-site pinning.

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
from charlie_work.config import (
    OrchestratorConfig,
    WatchdogConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
)
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_redispatch_escalated_edge_clears_full_active_set(tmp_path: Path) -> None:
    """Test that redispatch_escalated edge clears all active labels (issue #165)."""
    from charlie_work.labels import _edges

    config = OrchestratorConfig()
    edges = _edges(config.labels)

    # Verify the edge exists
    assert "redispatch_escalated" in edges

    add, remove = edges["redispatch_escalated"]

    # Should add human_needed
    assert config.labels.human_needed in add

    # Should remove ALL other workflow labels (issue #215: terminal transitions clear siblings)
    assert set(remove) == config.labels.workflow_labels - {config.labels.human_needed}


def test_redispatch_within_window_does_not_escalate(tmp_path: Path) -> None:
    """Test that N-1 redispatches within the window does not escalate (issue #165)."""
    from datetime import UTC, datetime, timedelta

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            enabled=True,
            stall_minutes=20,
            redispatch_window_minutes=240,
            max_auto_redispatch=3,
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Setup state with 2 redispatches (within cap of 3)
    state = load_state(paths.state_file)
    now = datetime.now(UTC)
    state["issues"]["123"] = {
        "number": 123,
        "title": "Test issue",
        "url": "https://github.com/test/repo/issues/123",
        "status": "rework_requested",
        "redispatch_at": [
            (now - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
            (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
        ],
    }
    save_state(paths.state_file, state)

    # Test the counting logic directly
    entry = state["issues"]["123"]
    window_start = now - timedelta(minutes=config.watchdog.redispatch_window_minutes)
    prior = [
        t
        for t in entry.get("redispatch_at", [])
        if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start
    ]
    redispatch_at = prior + [now.isoformat().replace("+00:00", "Z")]

    # Should not escalate - only 2 redispatches, cap is 3
    assert len(redispatch_at) == 3  # 2 prior + 1 new
    assert len(redispatch_at) <= config.watchdog.max_auto_redispatch


def test_redispatch_exceeding_cap_escalates(tmp_path: Path) -> None:
    """Test that exceeding max_auto_redispatch triggers escalation (issue #165)."""
    from datetime import UTC, datetime, timedelta

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            enabled=True,
            stall_minutes=20,
            redispatch_window_minutes=240,
            max_auto_redispatch=3,
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Setup state with 3 redispatches (at cap of 3)
    state = load_state(paths.state_file)
    now = datetime.now(UTC)
    state["issues"]["123"] = {
        "number": 123,
        "title": "Test issue",
        "url": "https://github.com/test/repo/issues/123",
        "status": "rework_requested",
        "redispatch_at": [
            (now - timedelta(minutes=30)).isoformat().replace("+00:00", "Z"),
            (now - timedelta(minutes=20)).isoformat().replace("+00:00", "Z"),
            (now - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        ],
    }
    save_state(paths.state_file, state)

    # Test the counting logic directly
    entry = state["issues"]["123"]
    window_start = now - timedelta(minutes=config.watchdog.redispatch_window_minutes)
    prior = [
        t
        for t in entry.get("redispatch_at", [])
        if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start
    ]
    redispatch_at = prior + [now.isoformat().replace("+00:00", "Z")]

    # Should escalate - 4th redispatch exceeds cap of 3
    assert len(redispatch_at) == 4  # 3 prior + 1 new
    assert len(redispatch_at) > config.watchdog.max_auto_redispatch


def test_redispatch_timestamps_pruned_outside_window(tmp_path: Path) -> None:
    """Test that timestamps outside the window are pruned before counting (issue #165)."""
    from datetime import UTC, datetime, timedelta

    config = OrchestratorConfig(
        watchdog=WatchdogConfig(
            enabled=True,
            stall_minutes=20,
            redispatch_window_minutes=240,
            max_auto_redispatch=3,
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Setup state with old redispatches outside the window
    state = load_state(paths.state_file)
    now = datetime.now(UTC)
    state["issues"]["123"] = {
        "number": 123,
        "title": "Test issue",
        "url": "https://github.com/test/repo/issues/123",
        "status": "rework_requested",
        "redispatch_at": [
            (now - timedelta(minutes=300)).isoformat().replace("+00:00", "Z"),  # Outside window
            (now - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),  # Inside window
        ],
    }
    save_state(paths.state_file, state)

    # Test the counting logic directly
    entry = state["issues"]["123"]
    window_start = now - timedelta(minutes=config.watchdog.redispatch_window_minutes)
    prior = [
        t
        for t in entry.get("redispatch_at", [])
        if datetime.fromisoformat(t.replace("Z", "+00:00")) >= window_start
    ]
    redispatch_at = prior + [now.isoformat().replace("+00:00", "Z")]

    # Old timestamp should be pruned, only 2 remain (1 in window + 1 new)
    assert len(redispatch_at) == 2
    assert len(redispatch_at) <= config.watchdog.max_auto_redispatch


def test_redispatch_at_only_written_by_known_call_sites(tmp_path: Path) -> None:
    """Test that redispatch_at is only written by known call sites."""
    # This test verifies by code inspection that redispatch_at is only written in:
    # 1. dispatch_rework normal paths (success + no-op rework pre-dispatch) --
    #    workflow.py, OrchestratorApp.dispatch_rework.
    # 2. _classify_dead_sessions_and_update_throttle_state normal paths --
    #    dead_worker_reap.py (issue #1317: moved verbatim from workflow.py).
    # 3. _reap_restore_rework_requested (issue #315 review finding 2) --
    #    dead_worker_reap.py (issue #1317: moved verbatim from workflow.py).
    # Escalated paths now consolidate on _escalate_issue and pass redispatch_at
    # through issue_extra, so direct entry["redispatch_at"] assignments only
    # remain in the non-escalated branches below.
    #
    # issue #1283 Phase A hazard (not in the recon's original list -- found by
    # the full-suite run for this split): AST-based real-assignment count, not
    # a raw substring count. `_windowed_redispatch_at`'s own docstring quotes
    # ``entry["redispatch_at"]`` in prose ("Normalizes ``entry["redispatch_at"]``
    # to a list of strings..."), and `.count()` over raw source text counted
    # that quote as a 5th "call site" for as long as the docstring lived in
    # workflow.py (confirmed: pre-split workflow.py already had exactly 4 real
    # assignments + this 1 docstring quote = 5 -- the miscounting predates
    # this extraction). Moving the function (and its docstring) to
    # dispatch_selection.py dropped the naive count to 4 with zero change to
    # any real write site -- a false regression signal, the opposite failure
    # mode from AC7's hazard test but the same root cause (name/text search
    # over source that doesn't distinguish code from prose). All three files
    # are scanned and summed via real ast.Assign nodes so neither a docstring
    # quote nor a future extraction of one of the three named call sites can
    # produce a false pass or a false failure here.
    #
    # issue #1317: the dead-worker/session-reap extraction moved call sites 2
    # and 3 above out of workflow.py into dead_worker_reap.py verbatim -- the
    # writers still exist, only their module changed (confirmed real split:
    # workflow.py=2, dispatch_selection.py=0, dead_worker_reap.py=2, total
    # unchanged at 4). dead_worker_reap.py is added to the scan so this guard
    # keeps failing if a genuinely NEW, unknown writer appears in any of the
    # three files, rather than going blind to two known call sites because
    # they changed address.
    #
    # issue #1645 (L01 batch 2): the OrchestratorApp delegation extraction moved
    # call site 1 above (dispatch_rework normal paths) out of workflow.py into
    # orchestration/state_dispatch_rework.py verbatim (_dispatch_rework_impl) --
    # the two writers still exist, only their module changed (confirmed real
    # split: workflow.py=0, dispatch_selection.py=0, dead_worker_reap.py=2,
    # state_dispatch_rework.py=2, total unchanged at 4). state_dispatch_rework.py
    # is added to the scan for the same reason dead_worker_reap.py was: keep
    # failing on a genuinely NEW writer rather than going blind to two known
    # call sites because they changed address.
    import ast

    workflow_path = Path(__file__).parents[1] / "src" / "charlie_work" / "workflow.py"
    dispatch_selection_path = (
        Path(__file__).parents[1] / "src" / "charlie_work" / "dispatch_selection.py"
    )
    dead_worker_reap_path = (
        Path(__file__).parents[1] / "src" / "charlie_work" / "dead_worker_reap.py"
    )
    state_dispatch_rework_path = (
        Path(__file__).parents[1]
        / "src"
        / "charlie_work"
        / "orchestration"
        / "state_dispatch_rework.py"
    )

    def _count_redispatch_at_assignments(path: Path) -> int:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "entry"
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "redispatch_at"
                ):
                    count += 1
        return count

    # Any unexpected increase means a new call site is writing redispatch_at.
    redispatch_assignments = (
        _count_redispatch_at_assignments(workflow_path)
        + _count_redispatch_at_assignments(dispatch_selection_path)
        + _count_redispatch_at_assignments(dead_worker_reap_path)
        + _count_redispatch_at_assignments(state_dispatch_rework_path)
    )
    assert redispatch_assignments == 4, (
        'Expected 4 real entry["redispatch_at"] assignment statements across '
        "workflow.py, dispatch_selection.py, dead_worker_reap.py, and "
        "orchestration/state_dispatch_rework.py, found "
        f"{redispatch_assignments}"
    )


def test_windowed_redispatch_at_handles_corrupted_state(tmp_path: Path) -> None:
    """_windowed_redispatch_at must not crash when redispatch_at is corrupted
    (e.g., a string instead of a list). A string value would cause
    list("abc") → ['a', 'b', 'c'] which crashes datetime.fromisoformat.
    """
    from charlie_work.workflow import _windowed_redispatch_at

    # String instead of list — must return empty, not crash
    entry = {"redispatch_at": "2024-01-01T00:00:00Z"}
    result = _windowed_redispatch_at(entry, window_minutes=240)
    assert result == []

    # None
    entry = {"redispatch_at": None}
    result = _windowed_redispatch_at(entry, window_minutes=240)
    assert result == []

    # Missing key
    entry = {}
    result = _windowed_redispatch_at(entry, window_minutes=240)
    assert result == []

    # List with non-string entries
    entry = {"redispatch_at": [123, None, "not-a-date"]}
    result = _windowed_redispatch_at(entry, window_minutes=240)
    assert result == []

    # Valid list with recent timestamp
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    entry = {"redispatch_at": [now_iso]}
    result = _windowed_redispatch_at(entry, window_minutes=240)
    assert result == [now_iso]

    # Valid list with old timestamp (outside window)
    old_iso = (datetime.now(UTC) - timedelta(hours=10)).isoformat().replace("+00:00", "Z")
    entry = {"redispatch_at": [old_iso]}
    result = _windowed_redispatch_at(entry, window_minutes=240)
    assert result == []
