"""Superseded-worker reap at the rework launch trigger (issue #1494).

Every producer of ``rework_requested`` — the janitor-gate/worktree-rescue
stall route this issue covers, the verdict reconcile, the stranded-commit
restore — flips the issue status without checking the prior worker's
liveness, and ``_check_worktree_writer_marker`` deliberately exempts a
marker owned by one of our own live sessions. A still-running prior worker
therefore admitted its replacement into the same worktree (#1337 ran two
workers against one worktree this way).

The fix enforces at the launch trigger: ``_reap_superseded_workers``
(superseded_worker_reap.py) kills every recorded prior-worker pid for the issue
via the fingerprinted ``write_gate.kill_process_tree`` before
``_dispatch_rework_impl`` / ``_local_dispatch_rework`` call
``dispatch_sessions``. A pid that survives (or is live with no
process-start fingerprint) blocks the launch as a
``prior_worker_still_alive`` blocked-environment failure.

This module carries the unit-level coverage — ``_reap_superseded_workers``
and the ``_reap_superseded_workers_for_launch`` batch partitioner. The
lane-integration coverage (``dispatch_rework``, ``_local_dispatch_rework``,
and the shared-helper guard) lives in the sibling module
``test_charlie_work_superseded_worker_reap_lanes.py``; both share
``tests/_superseded_worker_reap_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _superseded_worker_reap_fixtures import (
    PRIOR_PID,
    PRIOR_START,
    _gate,
    _patch_reap_internals,
    _worker_view,
)
from charlie_work.adapters import SessionRequest
from charlie_work.superseded_worker_reap import (
    _reap_superseded_workers,
    _reap_superseded_workers_for_launch,
)


# ---------------------------------------------------------------------------
# _reap_superseded_workers — unit coverage
# ---------------------------------------------------------------------------


def test_reap_superseded_workers_kills_live_sidecar_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live sidecar-recorded worker for the issue is killed via the
    fingerprinted process-tree kill, then confirmed dead on recheck."""
    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID, worktree_path="C:/wt/agent-123")],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123, {}, tmp_path / "sessions", write_gate=_gate(tmp_path)
    )

    assert survivors == []
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]
    assert [kind for kind, _payload in events] == ["superseded_worker_reaped"]
    assert events[0][1]["issue_number"] == 123
    assert events[0][1]["pid"] == PRIOR_PID


def test_reap_superseded_workers_dead_pid_no_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded-but-dead (or fingerprint-recycled) pid is already gone —
    no kill, no event, launch proceeds."""
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID)],
        alive={},  # is_pid_alive -> False
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123, {}, tmp_path / "sessions", write_gate=_gate(tmp_path)
    )

    assert survivors == []
    assert kill_calls == []
    assert events == []


def test_reap_superseded_workers_state_pid_without_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state-entry fallback covers a prior worker whose sidecar was
    already reaped: ``worker_pid``/``worker_process_start_time`` drive the
    fingerprinted kill."""
    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123,
        {"worker_pid": PRIOR_PID, "worker_process_start_time": PRIOR_START},
        tmp_path / "sessions",
        write_gate=_gate(tmp_path),
    )

    assert survivors == []
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]
    assert events[0][0] == "superseded_worker_reaped"
    assert events[0][1]["source"] == "state"


def test_reap_superseded_workers_last_known_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``last_known_worker_*`` is the fallback when ``worker_pid`` is absent —
    the same precedence ``_probe_recovery_liveness`` uses."""
    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123,
        {
            "last_known_worker_pid": PRIOR_PID,
            "last_known_worker_process_start_time": PRIOR_START,
        },
        tmp_path / "sessions",
        write_gate=_gate(tmp_path),
    )

    assert survivors == []
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]


def test_reap_superseded_workers_live_unfingerprinted_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live pid with no process_start_time fingerprint cannot be killed
    safely — kill_process_tree would have nothing to re-verify against —
    so it blocks the launch instead of killing on bare pid liveness."""
    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID, process_start_time=None)],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123, {}, tmp_path / "sessions", write_gate=_gate(tmp_path)
    )

    assert survivors == [PRIOR_PID]
    assert kill_calls == []
    assert [kind for kind, _p in events] == ["superseded_worker_reap_failed"]
    assert "no process_start_time" in events[0][1]["reason"]


def test_reap_superseded_workers_merges_state_fingerprint_onto_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sidecar/state fingerprint merge (the ``if pid in candidates``
    branch): a sidecar recorded before ``process_start_time`` capture
    carries no fingerprint, but the state entry for the same pid does.
    The merge must prefer the recorded fingerprint so a reaper-eligible
    worker is killed instead of failing closed as live-and-unfingerprinted
    the way the test above does. Removing the merge leaves the candidate
    unfingerprinted -> blocked, so this fails if the merge ``if`` is
    dropped."""
    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID, process_start_time=None)],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123,
        {"worker_pid": PRIOR_PID, "worker_process_start_time": PRIOR_START},
        tmp_path / "sessions",
        write_gate=_gate(tmp_path),
    )

    assert survivors == []
    # The kill ran against the STATE fingerprint, which the sidecar lacked.
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]
    assert [kind for kind, _p in events] == ["superseded_worker_reaped"]


def test_reap_superseded_workers_surviving_kill_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fingerprinted pid still alive after the guarded kill attempt is a
    survivor: launch must be refused."""
    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID)],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )
    # Kill attempt runs but the process survives it (refused identity
    # re-verification, or the pid genuinely outlived the signal).
    monkeypatch.setattr(
        "charlie_work.write_gate.kill_process_tree",
        lambda pid, st=None: kill_calls.append((pid, st)) or [],
    )

    survivors = _reap_superseded_workers(
        123, {}, tmp_path / "sessions", write_gate=_gate(tmp_path)
    )

    assert survivors == [PRIOR_PID]
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]
    assert [kind for kind, _p in events] == ["superseded_worker_reap_failed"]


def test_reap_superseded_workers_skips_other_issues_and_errored_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only this issue's non-errored sidecars are candidates: an errored
    sidecar's pid field can carry a diagnostic pid (the foreign/live worker
    a failed launch reported), not this issue's own worker."""
    alive = {PRIOR_PID: True, 999: True, 777: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[
            _worker_view(456, 999),  # a different issue's worker — untouched
            _worker_view(123, 777, error="launch failed"),  # diagnostic pid
        ],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123,
        {"worker_pid": PRIOR_PID, "worker_process_start_time": PRIOR_START},
        tmp_path / "sessions",
        write_gate=_gate(tmp_path),
    )

    assert survivors == []
    assert kill_calls == [(PRIOR_PID, PRIOR_START)]


def test_reap_superseded_workers_dry_run_never_kills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under a dry-run gate the underlying kill primitive is never invoked;
    a live prior worker is reported as a survivor so a real run would block."""
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID)],
        alive={PRIOR_PID: True},
        kill_calls=kill_calls,
        events=events,
    )

    survivors = _reap_superseded_workers(
        123, {}, tmp_path / "sessions", write_gate=_gate(tmp_path, dry_run=True)
    )

    assert survivors == [PRIOR_PID]
    assert kill_calls == []
    # The dry-run gate suppresses event emission too.
    assert events == []


def test_reap_superseded_workers_sweeps_orphans_after_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful tree kill is followed by an orphan sweep of the worker's
    recorded worktree — detached/daemonized children the tree kill leaves
    behind, same sweep the stall lane runs."""
    orphan_kills: list[int] = []
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID, worktree_path="C:/wt/agent-123")],
        alive={PRIOR_PID: True},
        kill_calls=kill_calls,
        events=events,
        orphans=[{"pid": 555, "name": "python.exe", "command_line": "x"}],
        orphan_kill_calls=orphan_kills,
    )

    survivors = _reap_superseded_workers(
        123, {}, tmp_path / "sessions", write_gate=_gate(tmp_path)
    )

    assert survivors == []
    assert orphan_kills == [555]
    assert events[0][1]["orphan_pids"] == [555]


def test_reap_superseded_workers_requires_gate(tmp_path: Path) -> None:
    """The helper is WriteGate-migrated: a missing gate is a TypeError, not
    a silent ungated kill."""
    with pytest.raises(TypeError):
        _reap_superseded_workers(123, {}, tmp_path / "sessions", write_gate=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _reap_superseded_workers_for_launch — batch partitioning
# ---------------------------------------------------------------------------


def test_reap_superseded_workers_for_launch_mixed_batch_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR #1926 review: the launch-lane wrapper must partition per request —
    a request whose prior worker survives the reap is blocked while a clean
    request in the SAME batch still launches. A ``continue`` turned into
    ``break`` would strand every request after the blocked one, so the
    launchable assertion is what pins the loop's per-request semantics."""
    alive = {PRIOR_PID: True}
    kill_calls: list[tuple[int, float | None]] = []
    events: list[tuple[str, dict]] = []
    _patch_reap_internals(
        monkeypatch,
        workers=[_worker_view(123, PRIOR_PID, process_start_time=None)],
        alive=alive,
        kill_calls=kill_calls,
        events=events,
    )

    blocked_request = SessionRequest(
        issue_number=123,
        issue_title="Blocked rework",
        prompt_path=tmp_path / "rework-123.md",
        branch_name="agent/issue-123-blocked",
        rework=True,
    )
    clean_request = SessionRequest(
        issue_number=7,
        issue_title="Clean rework",
        prompt_path=tmp_path / "rework-7.md",
        branch_name="agent/issue-7-clean",
        rework=True,
    )

    # Rescue-style branch mirrors _dispatch_rework_impl's adapter_label:
    # rescue-marked issues report "claude-code", the rest the configured
    # harness — here the BLOCKED request is the rescue one, so the label on
    # the synthetic result proves the callback was consulted.
    rescue_issue_numbers = {123}
    launchable, blocked = _reap_superseded_workers_for_launch(
        [blocked_request, clean_request],
        {},
        tmp_path / "sessions",
        repo_root=tmp_path,
        worktrees_dir=tmp_path / "worktrees",
        adapter_label=lambda request: (
            "claude-code" if request.issue_number in rescue_issue_numbers else "command"
        ),
        write_gate=_gate(tmp_path),
    )

    assert launchable == [clean_request]
    assert len(blocked) == 1
    result = blocked[0]
    assert result.issue_number == 123
    assert result.ok is False
    assert result.failure_kind == "prior_worker_still_alive"
    assert result.pid == PRIOR_PID
    assert result.adapter == "claude-code"
