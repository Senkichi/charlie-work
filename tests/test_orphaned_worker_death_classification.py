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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _orphan_sweep_fixtures import _dead_worker_rework_bed, _run_orphan_sweep
from _rework_dispatch_fixtures import _wg
from charlie_work.config import DevinConfig, OrchestratorConfig, WorkerRoleConfig
from charlie_work.dead_worker_classification import (
    _worker_view_for_entry,
    classify_and_credit_dead_worker,
    resolve_dead_worker_failure_kind,
)
from charlie_work.instrumentation import close_db
from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state
from charlie_work.workflow import OrchestratorApp


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


def test_resolve_persists_via_primitive_and_keeps_the_locked_entry_live(tmp_path: Path) -> None:
    """The classification is written by ``worker_fate.persist_failure`` (kind,
    ``classified_at`` and the cooldown together), yet the sweep's own ``entry``
    object stays the one stored in ``state["issues"]`` -- the caller keeps
    mutating it after this call, so identity must survive the pure rewrite.
    """
    sessions_dir = _sessions_dir(tmp_path)
    log_path = sessions_dir / "issue-207.log"
    log_path.write_text("Reached free model rate limit\n", encoding="utf-8")
    _write_devin_sidecar(sessions_dir, 207, pid=99999, log_path=log_path)
    entry: dict[str, Any] = {"status": "dispatched", "worker_pid": 99999}
    state: dict[str, Any] = {"issues": {"207": entry}}

    kind = resolve_dead_worker_failure_kind(
        entry,
        sessions_dir,
        207,
        state,
        OrchestratorConfig(devin=DevinConfig(), worker=WorkerRoleConfig(harness="devin-shell")),
        write_gate=_wg(tmp_path / "state.json"),
    )

    assert kind == "rate_limited"
    assert state["issues"]["207"] is entry
    assert entry["dead_worker_failure_kind"] == "rate_limited"
    assert entry["dead_worker_failure_classified_at"].endswith("Z")
    assert state["throttled_until"]
    assert state["throttle_reason"] == "rate_limited"
    assert state["throttle_adapter_kind"] == "devin"
    entry["later_mutation"] = True  # would be lost if the stored entry were a copy
    assert state["issues"]["207"]["later_mutation"] is True
    close_db(tmp_path / "state.json")


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


def _write_claude_sidecar(
    sessions_dir: Path,
    issue_number: int,
    *,
    adapter_kind: str,
    pid: int,
    log_path: Path,
) -> Path:
    """Write the ``issue-<n>.claude.json`` / ``issue-<n>.api.json`` sidecar."""
    suffix = {"claude-code": "claude", "api": "api", "opencode": "opencode"}[adapter_kind]
    payload = {
        "issue_number": issue_number,
        "branch": f"agent/issue-{issue_number}",
        "worktree_path": "",
        "prompt_path": "",
        "command": [],
        "pid": pid,
        "started_at": "2024-01-01T00:00:00Z",
        "log_path": str(log_path),
        "adapter_kind": adapter_kind,
    }
    sidecar_path = sessions_dir / f"issue-{issue_number}.{suffix}.json"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar_path


def _events(state: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [
        e
        for e in state.get("events", [])
        if e.get("kind") == kind and e.get("payload", {}).get("issue_number") == 207
    ]


# --- approved-rework restore site (real sweep) ---------------------------


def test_approved_rework_restore_rate_limited_death_not_credited(tmp_path: Path) -> None:
    config, paths, fake_gh, _ = _dead_worker_rework_bed(
        tmp_path, decision="approved", pr_state_status="rework_requested"
    )
    sessions_dir = _sessions_dir(tmp_path)
    log_path = sessions_dir / "issue-207-rework.log"
    log_path.write_text("applying\nReached free model rate limit\n", encoding="utf-8")
    _write_devin_sidecar(sessions_dir, 207, pid=99999, log_path=log_path)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == "rework_requested"
    assert not entry.get("worker_death_at")
    assert not entry.get("worker_death_failure_kinds")
    assert entry["dead_worker_failure_kind"] == "rate_limited"
    assert state.get("throttled_until")
    (event,) = _events(state, "orphaned_worker_recovered")
    assert event["payload"]["reason"] == "dead_worker_with_approved_rework"
    assert event["payload"]["failure_kind"] == "rate_limited"


def test_approved_rework_restore_unclassified_death_credited_with_null_kind(
    tmp_path: Path,
) -> None:
    config, paths, fake_gh, _ = _dead_worker_rework_bed(
        tmp_path, decision="approved", pr_state_status="rework_requested"
    )
    sessions_dir = _sessions_dir(tmp_path)
    log_path = sessions_dir / "issue-207-rework.log"
    log_path.write_text("applying\nTraceback: boom\n", encoding="utf-8")
    _write_devin_sidecar(sessions_dir, 207, pid=99999, log_path=log_path)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    death_at = entry["worker_death_at"]
    assert len(death_at) == 1
    assert entry["worker_death_failure_kinds"] == {death_at[0]: None}
    assert not state.get("throttled_until")
    (event,) = _events(state, "orphaned_worker_recovered")
    assert event["payload"]["reason"] == "dead_worker_with_approved_rework"
    assert event["payload"]["failure_kind"] is None
    assert event["payload"]["worker_death_at"] == death_at[0]


# --- unreviewed-PR advance site (real sweep) -----------------------------


def _unreviewed_pr_bed(tmp_path: Path) -> tuple[Any, Any, Any]:
    """Dead worker 207 with an open PR and no review verdict yet."""
    config, paths, fake_gh, _ = _dead_worker_rework_bed(tmp_path)
    state = load_state(paths.state_file)
    state["prs"]["100"] = {"reviewed_head_sha": None}
    save_state(paths.state_file, state)
    fake_gh.issues[0]["labels"] = [{"name": config.labels.in_progress}]
    fake_gh.prs[0].update(
        {
            "title": "Salvaged work for #207",
            "url": "https://example.test/pull/100",
            "baseRefName": "main",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #207\n\nTests: regression coverage added.",
            "labels": [],
            "state": "OPEN",
        }
    )
    # ``_dead_worker_rework_bed`` wrote a request_changes flat verdict.
    (paths.prs / "pr-100" / "review-decision.json").unlink()
    return config, paths, fake_gh


def test_unreviewed_pr_clean_exit_worker_quoting_rate_limit_is_not_log_classified(
    tmp_path: Path,
) -> None:
    """#656 false-positive class: a clean-exit worker that opened a PR and whose
    final prose quotes ``rate limit`` must not arm a fleet throttle, stamp the
    sidecar, or be attributed a kind -- it is still credited as unclassified."""
    config, paths, fake_gh = _unreviewed_pr_bed(tmp_path)
    sessions_dir = _sessions_dir(tmp_path)
    log_path = sessions_dir / "issue-207.log"
    log_path.write_text(
        "All done. The fix handles the case where we Reached free model rate limit.\n",
        encoding="utf-8",
    )
    sidecar_path = _write_devin_sidecar(sessions_dir, 207, pid=99999, log_path=log_path)
    (sessions_dir / "issue-207.devin.terminal.json").write_text(
        json.dumps(
            {
                "exit_code": 0,
                "duration_seconds": 12.0,
                # This dispatch's own record (rule 1: ended_at > dispatched_at;
                # the bed dispatched an hour ago).
                "ended_at": (datetime.now(UTC) - timedelta(minutes=5))
                .isoformat()
                .replace("+00:00", "Z"),
            }
        ),
        encoding="utf-8",
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == PASSIVE_OPEN_STATUS
    assert not state.get("throttled_until")
    assert "dead_worker_failure_kind" not in entry
    assert "failure_kind" not in json.loads(sidecar_path.read_text(encoding="utf-8"))
    death_at = entry["worker_death_at"]
    assert len(death_at) == 1
    assert entry["worker_death_failure_kinds"] == {death_at[0]: None}
    (event,) = _events(state, "orphaned_worker_advanced_to_pr_open")
    assert event["payload"]["exit_code"] == 0
    assert event["payload"]["failure_kind"] is None


def test_unreviewed_pr_advance_honors_existing_throttle_stamp(tmp_path: Path) -> None:
    config, paths, fake_gh = _unreviewed_pr_bed(tmp_path)
    state = load_state(paths.state_file)
    state["issues"]["207"]["dead_worker_failure_kind"] = "rate_limited"
    save_state(paths.state_file, state)
    _sessions_dir(tmp_path)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == PASSIVE_OPEN_STATUS
    assert not entry.get("worker_death_at")
    (event,) = _events(state, "orphaned_worker_advanced_to_pr_open")
    assert event["payload"]["failure_kind"] == "rate_limited"


def test_unreviewed_pr_advance_credits_stamped_kind(tmp_path: Path) -> None:
    config, paths, fake_gh = _unreviewed_pr_bed(tmp_path)
    state = load_state(paths.state_file)
    state["issues"]["207"]["dead_worker_failure_kind"] = "stalled"
    save_state(paths.state_file, state)
    _sessions_dir(tmp_path)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    entry = load_state(paths.state_file)["issues"]["207"]
    death_at = entry["worker_death_at"]
    assert len(death_at) == 1
    assert entry["worker_death_failure_kinds"] == {death_at[0]: "stalled"}


# --- adapter branches -----------------------------------------------------


def _run_claude_family_case(tmp_path: Path, adapter_kind: str, log_text: str) -> tuple[Any, Path]:
    config, paths, fake_gh, _ = _dead_worker_rework_bed(tmp_path)
    sessions_dir = _sessions_dir(tmp_path)
    log_path = sessions_dir / "issue-207-rework.claude.log"
    log_path.write_text(log_text, encoding="utf-8")
    sidecar_path = _write_claude_sidecar(
        sessions_dir, 207, adapter_kind=adapter_kind, pid=99999, log_path=log_path
    )
    _run_orphan_sweep(tmp_path, paths, config, fake_gh)
    return load_state(paths.state_file), sidecar_path


@pytest.mark.parametrize("adapter_kind", ["claude-code", "api"])
def test_claude_family_adapter_rate_limited_death_not_credited(
    tmp_path: Path, adapter_kind: str
) -> None:
    state, sidecar_path = _run_claude_family_case(
        tmp_path, adapter_kind, "working\nAPI Error: 429 rate limit exceeded\n"
    )
    entry = state["issues"]["207"]
    assert entry["dead_worker_failure_kind"] == "rate_limited"
    assert not entry.get("worker_death_at")
    assert state.get("throttled_until")
    assert json.loads(sidecar_path.read_text(encoding="utf-8"))["failure_kind"] == "rate_limited"


def test_opencode_provider_error_death_not_credited(tmp_path: Path) -> None:
    """opencode logs classify through its provider-error digest, not raw text."""
    log = (
        '{"type":"step_start"}\n'
        '{"type":"error","error":{"name":"APIError","data":{"message":"Usage limit reached",'
        '"statusCode":429,"isRetryable":true}}}\n'
    )
    state, sidecar_path = _run_claude_family_case(tmp_path, "opencode", log)
    entry = state["issues"]["207"]
    assert entry["dead_worker_failure_kind"] == "quota_exhausted"
    assert not entry.get("worker_death_at")
    assert state.get("throttled_until")
    assert json.loads(sidecar_path.read_text(encoding="utf-8"))["failure_kind"] == (
        "quota_exhausted"
    )


@pytest.mark.parametrize("adapter_kind", ["claude-code", "api", "opencode"])
def test_claude_family_adapter_unclassified_death_credited(
    tmp_path: Path, adapter_kind: str
) -> None:
    state, _ = _run_claude_family_case(tmp_path, adapter_kind, "working\nTraceback: boom\n")
    entry = state["issues"]["207"]
    death_at = entry["worker_death_at"]
    assert len(death_at) == 1
    assert entry["worker_death_failure_kinds"] == {death_at[0]: None}
    assert not state.get("throttled_until")


# --- _worker_view_for_entry matching -------------------------------------


def test_worker_view_for_entry_pid_mismatch_returns_none(tmp_path: Path) -> None:
    sessions_dir = _sessions_dir(tmp_path)
    log_path = sessions_dir / "issue-207.log"
    log_path.write_text("x", encoding="utf-8")
    _write_devin_sidecar(sessions_dir, 207, pid=11111, log_path=log_path)

    assert _worker_view_for_entry(sessions_dir, {"worker_pid": 99999}, 207) is None
    view = _worker_view_for_entry(sessions_dir, {"worker_pid": 11111}, 207)
    assert view is not None and view.pid == 11111
    # Legacy entry without a recorded pid accepts a lone sidecar.
    assert _worker_view_for_entry(sessions_dir, {}, 207) is not None


def test_worker_view_for_entry_multiple_sidecars_returns_none(tmp_path: Path) -> None:
    sessions_dir = _sessions_dir(tmp_path)
    log_path = sessions_dir / "issue-207.log"
    log_path.write_text("x", encoding="utf-8")
    _write_devin_sidecar(sessions_dir, 207, pid=99999, log_path=log_path)
    _write_claude_sidecar(
        sessions_dir, 207, adapter_kind="claude-code", pid=99999, log_path=log_path
    )

    assert _worker_view_for_entry(sessions_dir, {"worker_pid": 99999}, 207) is None


@pytest.mark.parametrize("case", ["pid_mismatch", "multiple_sidecars"])
def test_sweep_falls_back_to_unclassified_credit_when_sidecar_unmatchable(
    tmp_path: Path, case: str
) -> None:
    """Through the real sweep: an unmatchable sidecar set means the rate-limit
    log is never read, so the death is credited (unclassified), no throttle."""
    config, paths, fake_gh, _ = _dead_worker_rework_bed(tmp_path)
    sessions_dir = _sessions_dir(tmp_path)
    log_path = sessions_dir / "issue-207-rework.log"
    log_path.write_text("Reached free model rate limit\n", encoding="utf-8")
    if case == "pid_mismatch":
        _write_devin_sidecar(sessions_dir, 207, pid=11111, log_path=log_path)
    else:
        _write_devin_sidecar(sessions_dir, 207, pid=99999, log_path=log_path)
        _write_claude_sidecar(
            sessions_dir, 207, adapter_kind="claude-code", pid=99999, log_path=log_path
        )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    death_at = entry["worker_death_at"]
    assert len(death_at) == 1
    assert entry["worker_death_failure_kinds"] == {death_at[0]: None}
    assert not state.get("throttled_until")
    assert "dead_worker_failure_kind" not in entry


def test_classify_log_false_never_reads_sidecar(tmp_path: Path) -> None:
    sessions_dir = _sessions_dir(tmp_path)
    log_path = sessions_dir / "issue-207.log"
    log_path.write_text("Reached free model rate limit\n", encoding="utf-8")
    sidecar_path = _write_devin_sidecar(sessions_dir, 207, pid=99999, log_path=log_path)
    entry: dict[str, Any] = {"worker_pid": 99999}
    state: dict[str, Any] = {}

    kind = classify_and_credit_dead_worker(
        entry,
        sessions_dir,
        207,
        state,
        OrchestratorConfig(devin=DevinConfig(), worker=WorkerRoleConfig(harness="devin-shell")),
        write_gate=_wg(tmp_path / "state.json"),
        at="2026-01-01T00:00:00Z",
        classify_log=False,
    )

    assert kind is None
    assert entry["worker_death_at"] == ["2026-01-01T00:00:00Z"]
    assert not state
    assert "failure_kind" not in json.loads(sidecar_path.read_text(encoding="utf-8"))


# --- dead_worker_reap credit site ----------------------------------------


def test_reap_restore_records_failure_kind_alongside_death(tmp_path: Path) -> None:
    """``dead_worker_reap._reap_restore_rework_requested`` must pass the
    resolved ``failure_kind`` into ``_credit_worker_death`` so the credit and
    its attribution are recorded together."""
    from charlie_work.dead_worker_sweep.effects_rework import _reap_restore_rework_requested
    from charlie_work.worker import WorkerView

    config, paths, fake_gh, _ = _dead_worker_rework_bed(tmp_path)
    worker = WorkerView(
        adapter_kind="devin",
        issue_number=207,
        repo_key="",
        pid=99999,
        started_at="2024-01-01T00:00:00Z",
        process_start_time=1234567890.0,
        log_path=str(tmp_path / "issue-207.log"),
        worktree_path=str(tmp_path / "wt"),
        error=None,
        failure_kind="worker_died",
        reclaimed=None,
        branch="agent/issue-207",
    )
    _reap_restore_rework_requested(
        paths.state_file,
        fake_gh,
        config,
        {207: [fake_gh.prs[0]]},
        worker,
        failure_kind="worker_died",
        repo_root=tmp_path,
        write_gate=_wg(paths.state_file),
    )

    entry = load_state(paths.state_file)["issues"]["207"]
    death_at = entry["worker_death_at"]
    assert len(death_at) == 1
    assert entry["worker_death_failure_kinds"] == {death_at[0]: "worker_died"}


# --- unescalate survival --------------------------------------------------


def test_worker_death_failure_kinds_survives_real_unescalate(tmp_path: Path) -> None:
    """Credit a death, escalate, run the real ``unescalate``: the per-death
    kinds map survives while ``worker_death_at`` and the epoch stamp clear."""
    config, paths, fake_gh, _ = _dead_worker_rework_bed(tmp_path)
    state = load_state(paths.state_file)
    state["issues"]["207"]["dead_worker_failure_kind"] = "stalled"
    save_state(paths.state_file, state)
    _sessions_dir(tmp_path)
    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    death_at = entry["worker_death_at"]
    kinds = dict(entry["worker_death_failure_kinds"])
    assert kinds == {death_at[0]: "stalled"}
    entry["status"] = "escalated"
    entry["escalation_reason"] = "worker_death_loop"
    entry["escalation_reasons_seen"] = ["worker_death_loop"]
    entry["dead_worker_failure_kind"] = "stalled"
    save_state(paths.state_file, state)

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.unescalate(issue_number=207)
    assert result.ok is True

    entry = load_state(paths.state_file)["issues"]["207"]
    assert "worker_death_at" not in entry
    assert "dead_worker_failure_kind" not in entry
    assert entry["worker_death_failure_kinds"] == kinds
