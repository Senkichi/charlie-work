"""Orphan-sweep redispatch caps: observation identity and history.

Split out of ``tests/test_charlie_work.py`` (issue #1547, Track-1 wave 1/8):
the ``test_orphan_sweep_redispatch_cap_*`` seam's bookkeeping half --
first-observation long-history handling, distinct-identity counting toward
the cap, and the api-worker-disabled variant. Trigger/reset conditions
live in ``tests/test_charlie_work_orphan_sweep.py``; shared fakes and
helpers in ``tests/_rework_dispatch_fixtures.py``.
"""

from __future__ import annotations

from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import (
    _wg,
)
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
)


def test_orphan_sweep_redispatch_cap_first_observation_with_long_history(
    tmp_path: Path,
) -> None:
    """Issue #1243 regression: an issue whose adapter_history already exceeds
    max_auto_redispatch from an earlier, unrelated PR cycle must NOT be
    escalated on the very first orphan-sweep pass. The first-observation
    branch resets orphan_redispatch_at to [now] (count=1), so the cap
    counts attempts since this failure mode started, not the issue's entire
    historical adapter_history length. With the timestamp-list counter
    (independent of adapter routing mode, appending only on a not-yet-counted
    dispatch identity), adapter_history is not read at all -- but the seed
    keeps a long adapter_history to prove the new counter does not depend
    on it.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Seed the issue with 5 adapter_history entries (more than
    # max_auto_redispatch=3) and NO orphan_redispatch_head_sha set -- this
    # is the first time the orphan-sweep cap code sees this issue. The
    # entries represent dispatches from an earlier, unrelated PR cycle.
    base_time = datetime.now(UTC)
    state = load_state(paths.state_file)
    state["issues"]["1243"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": base_time.isoformat().replace("+00:00", "Z"),
        "adapter_history": [
            {
                "ts": (base_time - timedelta(minutes=60 + i * 10))
                .isoformat()
                .replace("+00:00", "Z"),
                "kind": "claude-code",
                "provider": "",
                "reason": "policy:default",
            }
            for i in range(5)
        ],
    }
    save_state(paths.state_file, state)

    class FakeGitHubNoPR(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubNoPR()
    fake_gh.issues = [
        {
            "number": 1243,
            "title": "test issue",
            "url": "https://example.test/issues/1243",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    # First orphan-sweep pass through the cap code for this issue.
    with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
        )

    st = load_state(paths.state_file)
    entry = st["issues"]["1243"]

    # The cap must NOT fire on the first observation, even though
    # len(adapter_history)=5 > max_auto_redispatch=3. The first-observation
    # branch reset orphan_redispatch_at to [now], so redispatch_count=1.
    assert entry.get("status") == "dispatched"
    assert entry.get("escalation_reason") is None

    # The head fingerprint and timestamp list must be persisted so subsequent
    # passes can compare against them.
    assert entry.get("orphan_redispatch_head_sha") == "none:none"
    assert len(entry.get("orphan_redispatch_at", [])) == 1

    # No escalation event must have been emitted.
    escalated_events = [
        e for e in st.get("events", []) if e.get("kind") == "orphan_sweep_redispatch_escalated"
    ]
    assert len(escalated_events) == 0


def test_orphan_sweep_redispatch_cap_fires_with_api_worker_disabled(
    tmp_path: Path,
) -> None:
    """Issue #1243 regression: the redispatch cap must fire in the default
    configuration where ``api_worker.enabled`` is ``False`` (the production
    default). In that configuration, ``adapter_history`` never grows (the
    per-issue adapter selector that wrote it was deleted in Phase 2 Track B,
    PR #1517). The previous ``len(adapter_history)``-based counter therefore
    never incremented and the cap never fired -- leaving the #709 infinite
    loop unbounded.

    This test exercises the real orphan-sweep reclaim -> redispatch cycle
    with ``api_worker`` left at its default (disabled) config, proving:
    1. ``adapter_history`` stays empty throughout (the old counter's source
       never grows).
    2. The ``orphan_redispatch_at`` timestamp list increments once per
       distinct redispatch (a fresh ``dispatched_at`` each attempt).
    3. The cap fires after ``max_auto_redispatch + 1`` attempts.
    """
    from unittest.mock import patch

    # api_worker.enabled defaults to False -- no need to set it explicitly.
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    assert config.api_worker.enabled is False, "api_worker must be disabled by default"

    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1243"] = {
        "status": "dispatched",
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        # adapter_history is empty -- as it would be in production with
        # api_worker disabled (the per-issue adapter selector that wrote it
        # was deleted in Phase 2 Track B, PR #1517).
    }
    save_state(paths.state_file, state)

    class FakeGitHubNoPR(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubNoPR()
    fake_gh.issues = [
        {
            "number": 1243,
            "title": "test issue",
            "url": "https://example.test/issues/1243",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    def _run_sweep() -> None:
        with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
            _detect_and_handle_orphaned_workers(
                sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
            )

    def _simulate_dispatch(dispatch_index: int) -> None:
        """Simulate a real dispatch between sweeps WITHOUT touching
        adapter_history. In production with api_worker disabled, the
        per-issue adapter selector that wrote it was deleted in Phase 2
        Track B (PR #1517), so adapter_history stays empty. This helper
        resets the labels and orphan flags that a real dispatch would
        reset, and assigns a fresh ``dispatched_at`` (as a genuine
        redispatch always does), which changes the dead-dispatch identity
        the cap dedupes on.
        """
        st = load_state(paths.state_file)
        entry = st["issues"]["1243"]
        entry.pop("orphan_flagged_at", None)
        entry.pop("orphan_drift_fingerprint", None)
        entry.pop("orphan_drift_at", None)
        entry["dispatched_at"] = f"2026-08-14T00:0{dispatch_index}:00Z"
        fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
        fake_gh.labels_added = []
        fake_gh.labels_removed = []
        save_state(paths.state_file, st)

    max_attempts = config.watchdog.max_auto_redispatch

    # Run max_attempts sweeps that proceed (count 1..max_attempts), then
    # one more that must escalate (count max_attempts+1 > max_attempts).
    for attempt in range(1, max_attempts + 1):
        _run_sweep()
        st = load_state(paths.state_file)
        entry = st["issues"]["1243"]
        # adapter_history must stay empty -- the old counter's source never
        # grows in the default config.
        assert entry.get("adapter_history", []) == [], (
            f"adapter_history must stay empty with api_worker disabled (attempt {attempt})"
        )
        # The counter must grow once per distinct redispatch attempt.
        assert len(entry.get("orphan_redispatch_at", [])) == attempt, (
            f"orphan_redispatch_at must have {attempt} entries after attempt {attempt}"
        )
        assert entry.get("status") == "dispatched"
        # Simulate the reclaim's label change landing on GitHub, then the
        # next dispatch resetting labels to in_progress.
        fake_gh.issues[0]["labels"] = [{"name": config.labels.ready}]
        _simulate_dispatch(attempt + 1)

    # Final sweep: count = max_attempts + 1 > max_attempts -> ESCALATE.
    _run_sweep()
    st = load_state(paths.state_file)
    entry = st["issues"]["1243"]

    # adapter_history is STILL empty -- the cap fired without it ever growing.
    assert entry.get("adapter_history", []) == []

    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "orphan_sweep_redispatch_cap_exceeded"
    assert entry.get("reason_class") == "mechanical"

    escalated_events = [
        e for e in st.get("events", []) if e.get("kind") == "orphan_sweep_redispatch_escalated"
    ]
    assert len(escalated_events) == 1
    assert escalated_events[0]["payload"]["redispatch_count"] == max_attempts + 1
    assert escalated_events[0]["payload"]["reason"] == "orphan_sweep_redispatch_cap_exceeded"


def test_orphan_sweep_redispatch_cap_counts_distinct_identities_and_escalates(
    tmp_path: Path,
) -> None:
    """Issue #1243 round-3 fix: when each sweep pass observes a genuinely
    NEW dead dispatch (distinct ``worker_pid`` each time, head unchanged),
    the counter must still increment every pass and the cap must still fire
    once the count exceeds ``max_auto_redispatch`` -- proving the dedup fix
    only suppresses re-observation of the SAME dispatch, not real repeated
    redispatch attempts.
    """
    from unittest.mock import patch

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    state = load_state(paths.state_file)
    state["issues"]["1243"] = {
        "status": "dispatched",
        "worker_pid": 90001,
        "worker_process_start_time": 1234567890.0,
        "dispatched_at": "2026-08-14T00:00:01Z",
    }
    save_state(paths.state_file, state)

    class FakeGitHubNoPR(FakeGitHub):
        def pr_list(self):
            return []

    fake_gh = FakeGitHubNoPR()
    fake_gh.issues = [
        {
            "number": 1243,
            "title": "test issue",
            "url": "https://example.test/issues/1243",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    def _run_sweep() -> None:
        with patch("charlie_work.workflow._worker_pid_alive", return_value=False):
            _detect_and_handle_orphaned_workers(
                sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
            )

    def _simulate_redispatch(worker_pid: int) -> None:
        """A genuine redispatch always assigns a new worker_pid -- change the
        dead-dispatch identity so the next sweep counts it as a new attempt.
        """
        st = load_state(paths.state_file)
        entry = st["issues"]["1243"]
        entry.pop("orphan_flagged_at", None)
        entry.pop("orphan_drift_fingerprint", None)
        entry.pop("orphan_drift_at", None)
        entry["worker_pid"] = worker_pid
        fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
        fake_gh.labels_added = []
        fake_gh.labels_removed = []
        save_state(paths.state_file, st)

    max_attempts = config.watchdog.max_auto_redispatch

    for attempt in range(1, max_attempts + 1):
        _run_sweep()
        st = load_state(paths.state_file)
        entry = st["issues"]["1243"]
        assert entry.get("status") == "dispatched"
        assert len(entry.get("orphan_redispatch_at", [])) == attempt, (
            f"orphan_redispatch_at must have {attempt} entries after attempt {attempt}"
        )
        fake_gh.issues[0]["labels"] = [{"name": config.labels.ready}]
        _simulate_redispatch(90001 + attempt)

    # Final sweep: count = max_attempts + 1 > max_attempts -> ESCALATE.
    _run_sweep()
    st = load_state(paths.state_file)
    entry = st["issues"]["1243"]

    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "orphan_sweep_redispatch_cap_exceeded"
    assert entry.get("reason_class") == "mechanical"

    escalated_events = [
        e for e in st.get("events", []) if e.get("kind") == "orphan_sweep_redispatch_escalated"
    ]
    assert len(escalated_events) == 1
    assert escalated_events[0]["payload"]["issue_number"] == 1243
    assert escalated_events[0]["payload"]["redispatch_count"] == max_attempts + 1
    assert escalated_events[0]["payload"]["reason"] == "orphan_sweep_redispatch_cap_exceeded"
