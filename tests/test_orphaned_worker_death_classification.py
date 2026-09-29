"""Issue #2002: the orphan sweep must classify a dead worker's sidecar log
before crediting its death into ``worker_death_at``.

The dead-session classifier (``dead_worker_reap._classify_dead_sessions_
and_update_throttle_state``) gates on ``is_worker_confirmed_dead`` -- a
verdict the Signal-1 inconclusive-probe deferral and the fresh-real-
activity veto can push to a later pass -- while the state.json-keyed
orphan sweep keys on a bare dead ``worker_pid``. A rate-limited rework
death can therefore reach the sweep's ``worker_death_at`` credit sites in
a pass where ``dead_worker_failure_kind`` was never stamped, and the
pre-#2002 gate (``is_provider_throttle_failure`` on the missing stamp)
credited it anyway -- the #1971 ``worker_death_loop`` escalation.

These tests run the real ``_detect_and_handle_orphaned_workers`` over the
shared dead-worker-with-PR bed plus a devin sidecar/log pair, so the
classify-at-credit seam (``dead_worker_classification``) is exercised
end-to-end rather than mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _orphan_sweep_fixtures import _dead_worker_rework_bed, _run_orphan_sweep
from charlie_work.state import load_state, save_state


def _sessions_dir(tmp_path: Path) -> Path:
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def _write_devin_sidecar(
    sessions_dir: Path,
    issue_number: int,
    *,
    pid: int,
    log_path: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the devin ``issue-<n>.json`` sidecar ``iter_workers`` reads.

    Same file ``launch_devin_session`` writes at dispatch -- for a rework
    dispatch the sidecar name is unchanged while ``log_path`` points at
    ``issue-<n>-rework.log``.
    """
    payload: dict[str, Any] = {
        "issue_number": issue_number,
        "branch": f"agent/issue-{issue_number}",
        "worktree_path": "",
        "prompt_path": "",
        "command": [],
        "pid": pid,
        "started_at": "2024-01-01T00:00:00Z",
        "log_path": str(log_path),
    }
    payload.update(extra or {})
    sidecar_path = sessions_dir / f"issue-{issue_number}.json"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar_path


def test_rate_limited_rework_death_classified_not_credited(tmp_path: Path) -> None:
    """AC1: a rate-limited rework death reaching the sweep in the same pass
    its sidecar log could be classified does NOT append ``worker_death_at``.

    Reproduce from the issue: the dead worker's devin log ends in
    ``Reached free model rate limit`` and the state entry carries no
    ``dead_worker_failure_kind`` stamp (the confirmed-dead classifier had
    not reached it). The sweep must classify the log itself, stamp the
    kind, arm the provider cooldown, and skip the credit.
    """
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    sessions_dir = _sessions_dir(tmp_path)
    log_path = sessions_dir / "issue-207-rework.log"
    log_path.write_text(
        "applying rework\nReached free model rate limit\n",
        encoding="utf-8",
    )
    sidecar_path = _write_devin_sidecar(sessions_dir, 207, pid=99999, log_path=log_path)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    # Recovery still runs -- the issue returns to rework_requested...
    assert entry.get("status") == "rework_requested"
    # ...but a provider-throttle death is never a credited death.
    assert not entry.get("worker_death_at")
    assert entry.get("dead_worker_failure_kind") == "rate_limited"
    # The fleet cooldown is armed even though the dead-session classifier
    # never ran this pass -- arming cannot be left to a later lane because
    # the sidecar now carries failure_kind and would return the stamp
    # without recomputing the window.
    assert state.get("throttled_until")
    # No credit -> no per-death kind record.
    assert not entry.get("worker_death_failure_kinds")
    # The sidecar itself was stamped, so the classifier lane's later pass
    # reaps it with the same classification instead of re-deriving it.
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar.get("failure_kind") == "rate_limited"
    # The recovery event records the resolved classification.
    recovered = [
        e
        for e in state.get("events", [])
        if e.get("kind") == "orphaned_worker_recovered"
        and e.get("payload", {}).get("issue_number") == 207
    ]
    assert len(recovered) == 1
    assert recovered[0]["payload"]["failure_kind"] == "rate_limited"


def test_unclassified_death_credited_and_records_null_kind(tmp_path: Path) -> None:
    """AC2 (unclassified arm): a death whose log shows no classification
    signature is still credited -- and the credit records its (null) kind."""
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    sessions_dir = _sessions_dir(tmp_path)
    log_path = sessions_dir / "issue-207-rework.log"
    log_path.write_text(
        "applying rework\nTraceback (most recent call last): boom\n",
        encoding="utf-8",
    )
    _write_devin_sidecar(sessions_dir, 207, pid=99999, log_path=log_path)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "rework_requested"
    death_at = entry.get("worker_death_at")
    assert isinstance(death_at, list) and len(death_at) == 1
    # The credit and its attribution share the same timestamp key.
    assert entry.get("worker_death_failure_kinds") == {death_at[0]: None}
    assert not state.get("throttled_until")


def test_stamped_death_credited_and_records_kind(tmp_path: Path) -> None:
    """AC2 (stamped arm): a death already stamped by another lane is
    credited (non-throttle kind) and the stamp is recorded alongside the
    timestamp so the attribution survives an unescalate."""
    config, paths, fake_gh, _dispatched_at = _dead_worker_rework_bed(tmp_path)
    state = load_state(paths.state_file)
    state["issues"]["207"]["dead_worker_failure_kind"] = "stalled"
    save_state(paths.state_file, state)
    # No sidecar for this worker -- the epoch stamp alone must suffice.
    _sessions_dir(tmp_path)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry.get("status") == "rework_requested"
    death_at = entry.get("worker_death_at")
    assert isinstance(death_at, list) and len(death_at) == 1
    assert entry.get("worker_death_failure_kinds") == {death_at[0]: "stalled"}
    recovered = [
        e
        for e in state.get("events", [])
        if e.get("kind") == "orphaned_worker_recovered"
        and e.get("payload", {}).get("issue_number") == 207
    ]
    assert len(recovered) == 1
    assert recovered[0]["payload"]["failure_kind"] == "stalled"


def test_stamp_surviving_unescalate_reset_fields(tmp_path: Path) -> None:
    """The per-death kinds map is forensic history, not cap state: it must
    NOT be among the fields ``unescalate`` pops when re-arming an issue."""
    from charlie_work.unescalate_reset_fields import UNESCALATE_ISSUE_RESET_FIELDS

    assert "worker_death_at" in UNESCALATE_ISSUE_RESET_FIELDS
    assert "dead_worker_failure_kind" in UNESCALATE_ISSUE_RESET_FIELDS
    assert "worker_death_failure_kinds" not in UNESCALATE_ISSUE_RESET_FIELDS
