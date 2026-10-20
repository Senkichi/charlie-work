"""Issue #2748: a dead-dispatched reap escalates no issue whose reclaim landed.

The timed dead-dispatched backstop fires ``orphan_drift_at`` minutes after the
arm, and a *failed* relabel pass arms both ``orphan_orphaned`` bookkeeping
stamps -- so when the NEXT pass's reclaim succeeds (the transient GitHub write
error recovers), the stale drift stamp keeps charging saturated-fleet queue
wait against the dead-worker timer and escalates a healthy ``ready`` issue to
the operator queue as a ``mechanical`` failure. These two tests pin the
``reap_flow`` skip on the landed label edge (``ready`` with no active label),
and its positive control -- with every relabel still failing, the backstop
escalates exactly as before.

Lives in its own module rather than ``test_charlie_work_orphaned_worker_
reclaim.py`` because that file is over its file-size ratchet cap.
"""

from __future__ import annotations

from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from _host_fixtures import host_probe
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
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


def _sweep_config() -> OrchestratorConfig:
    return OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, dead_dispatched_reap_minutes=60),
    )


def _seed_dispatched(paths, number: int, worker_pid: int) -> None:
    state = load_state(paths.state_file)
    state["issues"][str(number)] = {
        "status": "dispatched",
        "dispatched_at": (datetime.now(UTC) - timedelta(days=1))
        .isoformat()
        .replace("+00:00", "Z"),
        "worker_pid": worker_pid,
        "worker_process_start_time": 1784000000.0,
    }
    save_state(paths.state_file, state)


def _advance_drift_beyond_reap_window(state_file: Path, number: int) -> None:
    """Rewind ``orphan_drift_at`` (and its flag with it where asked) to 120
    minutes ago -- past the 60-minute reap window -- derived from the real
    clock, never a hard-coded date."""
    st = load_state(state_file)
    entry = st["issues"][str(number)]
    old_drift = (datetime.now(UTC) - timedelta(minutes=120)).isoformat().replace("+00:00", "Z")
    entry["orphan_drift_at"] = old_drift
    save_state(state_file, st)


def _reaped_events(state: dict, issue: int):
    return [
        e
        for e in state.get("events", [])
        if e.get("kind") == "dead_dispatched_worker_reaped"
        and e.get("payload", {}).get("issue_number") == issue
    ]


class _FlakyLabelGitHub(FakeGitHub):
    """Labels read back unchanged and ``remove_issue_label`` fails while
    ``fail_remove`` is set (FakeGitHub's label methods do not mutate
    ``self.issues``).

    The mutating variant flips ``fail_remove`` off to replay the
    interrupted-reclaim recovery, and the caller mirrors the landed labels on
    the fake the way a real successful label swap would leave them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fail_remove = True

    def remove_issue_label(self, number: int, label: str) -> bool:
        self.labels_removed.append((number, label))
        return not self.fail_remove


def test_reclaim_success_disarms_dead_dispatched_reap(tmp_path: Path, monkeypatch) -> None:
    """Two sweep passes: the first relabel write fails (arming the drift
    stamps), the second succeeds and returns the issue to ``ready``. Advancing
    the clock past ``dead_dispatched_reap_minutes`` with no dispatch in
    between must not reap the issue -- no ``dead_dispatched_worker_reaped``,
    no ``Escalate``, no operator-queue label, status still ``dispatched``."""
    config = _sweep_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_dispatched(paths, number=2748, worker_pid=27480)

    fake_gh = _FlakyLabelGitHub()
    fake_gh.issues = [
        {
            "number": 2748,
            "title": "dead-dispatched reap defect",
            "url": "https://example.test/issues/2748",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    def _run_sweep(monkeypatch) -> None:
        with host_probe(monkeypatch, alive=False):
            _detect_and_handle_orphaned_workers(
                sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
            )

    # Pass 1: the reclaim's label write fails. The failed pass falls through
    # to the drift branch, which arms both stamps.
    _run_sweep(monkeypatch=monkeypatch)
    entry = load_state(paths.state_file)["issues"]["2748"]
    assert entry.get("orphan_flagged_at") is not None
    drift_at_pass1 = entry["orphan_drift_at"]
    assert drift_at_pass1 is not None

    # Pass 2: the label write recovers -- the reclaim succeeds and returns
    # the issue to ready, restamping orphan_flagged_at with its own stamp.
    # The entry's status deliberately stays "dispatched" (issue #282
    # fingerprint preservation), and "orphan_drift_at" stays armed from pass 1.
    fake_gh.fail_remove = False
    fake_gh.labels_added = []
    fake_gh.labels_removed = []
    _run_sweep(monkeypatch=monkeypatch)
    entry = load_state(paths.state_file)["issues"]["2748"]
    assert (2748, config.labels.in_progress) in fake_gh.labels_removed
    assert (2748, config.labels.ready) in fake_gh.labels_added
    assert entry.get("orphan_flagged_at") is not None, (
        "the reclaim-success branch must restamp orphan_flagged_at"
    )
    assert entry.get("orphan_drift_at") == drift_at_pass1

    # Make the fake issue reflect pass 2's landed labels: ready only, no
    # active label -- a perfectly healthy dispatchable issue waiting for a
    # fleet slot.
    fake_gh.issues[0]["labels"] = [{"name": config.labels.ready}]
    fake_gh.labels_added = []
    fake_gh.labels_removed = []

    # Advance the clock past dead_dispatched_reap_minutes measured from the
    # pass-1 (failed-relabel) drift stamp -- the exact queue-wait window the
    # escalation report showed -- then run the sweep again.
    _advance_drift_beyond_reap_window(paths.state_file, 2748)
    _run_sweep(monkeypatch=monkeypatch)

    st = load_state(paths.state_file)
    entry = st["issues"]["2748"]
    assert entry.get("status") == "dispatched", (
        "a successfully reclaimed issue waiting for a slot must not be escalated"
    )
    assert "escalation_reason" not in entry
    assert _reaped_events(st, 2748) == []
    assert (2748, config.labels.operator_queue) not in fake_gh.labels_added, (
        "queue wait after a successful reclaim must not reach the operator-queue label"
    )
    drift_events = [
        e
        for e in st.get("events", [])
        if e.get("kind") == "orphaned_worker_drift"
        and e.get("payload", {}).get("issue_number") == 2748
    ]
    assert len(drift_events) == 1, "only pass 1's arming event; the quiet passes add none"


def test_failed_reclaims_still_reaped_after_grace(tmp_path: Path, monkeypatch) -> None:
    """Positive control: the same two-pass sequence with BOTH label writes
    failing must still reap after the grace window -- the backstop keeps
    working for a genuinely unreclaimed dead worker whose reclaim never got
    past GitHub."""
    config = _sweep_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_dispatched(paths, number=2749, worker_pid=27490)

    fake_gh = _FlakyLabelGitHub()
    fake_gh.issues = [
        {
            "number": 2749,
            "title": "still broken",
            "url": "https://example.test/issues/2749",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []

    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    def _run_sweep(monkeypatch) -> None:
        with host_probe(monkeypatch, alive=False):
            _detect_and_handle_orphaned_workers(
                sessions_dir, paths.state_file, config, fake_gh, write_gate=_wg(paths.state_file)
            )

    for _pass in range(2):
        _run_sweep(monkeypatch=monkeypatch)

    entry = load_state(paths.state_file)["issues"]["2749"]
    assert entry.get("orphan_flagged_at") is not None
    assert entry["orphan_drift_at"] == entry.get("orphan_flagged_at"), (
        "with every relabel failing, the drift branch stamps both -- the issue "
        "labels never reached ready, so the reap_flow skip cannot apply"
    )

    _advance_drift_beyond_reap_window(paths.state_file, 2749)
    st = load_state(paths.state_file)
    st["issues"]["2749"]["orphan_flagged_at"] = st["issues"]["2749"]["orphan_drift_at"]
    save_state(paths.state_file, st)
    _run_sweep(monkeypatch=monkeypatch)

    st = load_state(paths.state_file)
    entry = st["issues"]["2749"]
    assert entry.get("status") == "escalated"
    assert entry.get("escalation_reason") == "dead_dispatched_worker_reap"
    assert entry.get("reason_class") == "mechanical"
    reaped = _reaped_events(st, 2749)
    assert len(reaped) == 1
    assert reaped[0]["payload"]["reap_minutes"] == 60
    assert (2749, config.labels.operator_queue) in fake_gh.labels_added
