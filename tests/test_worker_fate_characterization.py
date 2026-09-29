"""Characterization tests for the pre-refactor worker-fate decision sites.

Architecture-deepening candidate 1 ("Worker fate", `docs/superpowers/plans/
2026-09-29-architecture-deepening.md`) consolidates liveness and post-exit
fate classification -- today scattered across `workflow.py`,
`orphaned_worker_sweep.py`, `live_handoff_finalize.py`, `rework_outcome.py`,
`dead_worker_reap.py` and `worktree.py` -- into one module, resolving nine
named disagreements between sites (each a "flip rule" in the plan).

Every `test_flip<N>_<site>_<behaviour>` test below pins CURRENT (pre-flip)
behaviour at one (rule, site) pair, so the follow-up behaviour-flip commit
can diff against a known-good baseline instead of guessing what changed.
`test_preserve_*` tests pin non-disagreeing classification paths the new
module must keep agreeing on. None of these tests assert what the NEW
module should do -- only what the code does today.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from _fakes_github import FakeGitHub
from _orphan_sweep_fixtures import _dead_worker_rework_bed, _run_orphan_sweep, _write_outcome
from _salvage_fixtures import _SalvageTestGitHub, _salvage_labels
from _worktree_fixtures import _git, _init_bare_remote_and_clone, _init_repo

from charlie_work.config import (
    WORKER_OUTCOME_FILENAME,
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import resolved_layout, runtime_paths
from charlie_work.process_utils import write_worker_terminal_status
from charlie_work.state import load_state, save_state
from charlie_work.worktree import worktree_path_for_branch
from charlie_work.write_gate import WriteGate


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


class _FakeGitHubNoPR(FakeGitHub):
    def pr_list(self):
        return []


def _seed_no_pr_dispatched_issue(
    tmp_path: Path, config: OrchestratorConfig, issue_number: int, branch: str
) -> tuple[Path, Path]:
    """Seed a dead-PID ``dispatched`` issue with no open PR. Returns (state_file, sessions_dir)."""
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    state = load_state(paths.state_file)
    state["issues"][str(issue_number)] = {
        "status": "dispatched",
        "dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "worker_pid": 99999,
        "worker_process_start_time": 1234567890.0,
        "branch_name": branch,
    }
    save_state(paths.state_file, state)
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return paths.state_file, sessions_dir


def _no_pr_fake_gh(tmp_path: Path, config: OrchestratorConfig, issue_number: int) -> FakeGitHub:
    fake_gh = _FakeGitHubNoPR(repo_root=tmp_path)
    fake_gh.issues = [
        {
            "number": issue_number,
            "title": "test issue",
            "url": f"https://example.test/issues/{issue_number}",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    fake_gh.prs = []
    # A successful pr_create by default: only the A9 test drives a PR-open
    # path, but leaving the fake's default None-return in place would make
    # gh.pr_create "fail" and trigger real-time retry/backoff sleeps in
    # pr_create_retry -- pin a return value up front so no test pays that
    # cost by accident.
    fake_gh.pr_create_return = 5501
    return fake_gh


# ---------------------------------------------------------------------------
# Rule 1: only evidence from the current dispatch should count (outcome file
# / terminal record newer than dispatched_at, head_sha matching the live
# head). Two sites currently apply NO such freshness gate at all.
# ---------------------------------------------------------------------------


def test_flip1_a9_workflow_uses_stale_worker_outcome_from_prior_dispatch(tmp_path: Path) -> None:
    """FLIP 1: current behaviour; rule 1 will change this to reject stale evidence.

    `workflow.py`'s no-PR orphan lane pre-reads
    `worker_outcomes[issue_number] = terminal_outcome or worktree_outcome`
    once (no dispatched_at comparison anywhere), then the pushed-branch
    candidate check (`reported_push = ... worker_outcome.get("push_succeeded")
    is True and worker_outcome.get("pr_created") is False`) reuses that same
    value with no freshness gate either. A `.worker-outcome.json` left over
    from a PRIOR dispatch of this issue/branch -- written well before the
    CURRENT dispatch even started -- is still accepted as proof the current
    session pushed, and a PR is opened for it.
    """
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    issue_number = 9101
    branch = "agent/issue-9101-test"
    state_file, sessions_dir = _seed_no_pr_dispatched_issue(tmp_path, config, issue_number, branch)

    # dispatched_at (set just now, inside the seed helper) is well AFTER this
    # outcome file's mtime -- a leftover from a session that ran two hours
    # before the current dispatch began.
    worktrees_dir = resolved_layout(config, tmp_path).worktrees
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome_path = worktree_path / WORKER_OUTCOME_FILENAME
    outcome_path.write_text(
        json.dumps({"push_succeeded": True, "pr_created": False, "head_sha": "leftover999"}),
        encoding="utf-8",
    )
    stale_ts = (datetime.now(UTC) - timedelta(hours=2)).timestamp()
    os.utime(outcome_path, (stale_ts, stale_ts))

    fake_gh = _no_pr_fake_gh(tmp_path, config, issue_number)

    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch("charlie_work.workflow.remote_branch_head_sha", return_value=None),
        patch("charlie_work.workflow.remote_branch_ahead_count", return_value=(0, None)),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir, state_file, config, fake_gh, write_gate=_wg(state_file)
        )

    st = load_state(state_file)
    entry = st["issues"][str(issue_number)]
    # Current (disagreeing) behaviour: the stale outcome's push claim is
    # trusted -- a PR gets opened from it -- with no comparison against
    # dispatched_at at all.
    assert entry["status"] == "open_passive"
    assert entry.get("pr_number") is not None
    events = st.get("events", [])
    assert any(
        e.get("kind") == "worker_handoff_pr_opened"
        and e["payload"].get("reason") == "worker_handoff_clean_exit"
        for e in events
    )


def test_flip1_a5_live_handoff_finalize_ignores_dispatched_at(tmp_path: Path) -> None:
    """FLIP 1: current behaviour; rule 1 will change this to reject stale evidence.

    `collect_stale_live_handoff_pids` (`live_handoff_finalize.py`) takes no
    `dispatched_at` parameter at all and never compares the outcome file's
    age to when the current dispatch began -- only to `now`. A leftover
    outcome file from a PREVIOUS dispatch of this branch, written well
    before the current dispatch started, still counts as valid completion
    evidence once it is merely old enough relative to `now`.
    """
    from charlie_work.live_handoff_finalize import collect_stale_live_handoff_pids

    now = datetime.now(UTC)
    branch = "agent/issue-9102"
    worktrees_dir = tmp_path / "worktrees"
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome_path = worktree_path / WORKER_OUTCOME_FILENAME
    outcome_path.write_text(
        json.dumps({"push_succeeded": True, "pr_created": False, "head_sha": "deadbeef"}),
        encoding="utf-8",
    )
    # Written ~2 hours ago -- well before the CURRENT dispatch, which (per
    # the live_pid_entries below) started only 10 minutes ago.
    stale_ts = (now - timedelta(hours=2)).timestamp()
    os.utime(outcome_path, (stale_ts, stale_ts))

    live_pid_entries = {
        9102: {
            "branch_name": branch,
            "worker_pid": 4343,
            "dispatched_at": (now - timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        }
    }

    candidates = collect_stale_live_handoff_pids(
        live_pid_entries,
        worker_outcome_finalize_minutes=15,
        repo_root=tmp_path,
        worktrees_dir=worktrees_dir,
        now=now,
    )

    # Current (disagreeing) behaviour: no dispatched_at comparison exists in
    # this function at all -- the 2-hour-old leftover is accepted.
    assert 9102 in candidates
    assert candidates[9102]["worker_outcome"]["head_sha"] == "deadbeef"


# ---------------------------------------------------------------------------
# Rule 5: live PID + fresh outcome = completed immediately; the KILL waits
# for watchdog.worker_outcome_finalize_minutes, but ROUTING should not.
# Today the same threshold gates both.
# ---------------------------------------------------------------------------


def test_flip5_live_handoff_finalize_withholds_fresh_outcome_until_threshold(
    tmp_path: Path,
) -> None:
    """FLIP 5: current behaviour; rule 5 will change this to route immediately.

    `collect_stale_live_handoff_pids` skips (does not return as a candidate)
    any outcome whose age is `<= worker_outcome_finalize_minutes` --
    `if outcome_age <= timedelta(minutes=worker_outcome_finalize_minutes):
    continue`. A live PID with a fresh, on-target, confirmed-push outcome is
    thus NOT routed yet: the same threshold that governs when the stall
    watchdog is allowed to kill the process also gates when the completion
    evidence is even considered, rather than the kill waiting while routing
    fires immediately.
    """
    from charlie_work.live_handoff_finalize import collect_stale_live_handoff_pids

    now = datetime.now(UTC)
    branch = "agent/issue-9105"
    worktrees_dir = tmp_path / "worktrees"
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome_path = worktree_path / WORKER_OUTCOME_FILENAME
    outcome_path.write_text(
        json.dumps({"push_succeeded": True, "pr_created": False, "head_sha": "cafefeed"}),
        encoding="utf-8",
    )
    fresh_ts = (now - timedelta(minutes=2)).timestamp()
    os.utime(outcome_path, (fresh_ts, fresh_ts))

    live_pid_entries = {9105: {"branch_name": branch, "worker_pid": 4242}}

    candidates = collect_stale_live_handoff_pids(
        live_pid_entries,
        worker_outcome_finalize_minutes=15,
        repo_root=tmp_path,
        worktrees_dir=worktrees_dir,
        now=now,
    )

    # Current (disagreeing) behaviour: a fresh, on-target, confirmed-push
    # outcome is withheld from routing until it ages past the same
    # threshold the kill decision uses.
    assert candidates == {}


# ---------------------------------------------------------------------------
# Rule 2: a fresh `outcome: blocked` should beat push flags everywhere.
# The no-PR lane already honours this (test_worker_declared_blocked.py);
# the with-PR lane's `fresh_completed_worker_outcome` correctly refuses to
# treat `blocked` as *completed*, but the caller then cannot distinguish
# "no evidence at all" from "an explicit blocked declaration" -- both fall
# through to the SAME generic redispatch, silently retrying a worker that
# said it could not do the task instead of escalating.
# ---------------------------------------------------------------------------


def test_flip2_with_pr_fresh_blocked_outcome_falls_through_to_redispatch(tmp_path: Path) -> None:
    """FLIP 2: current behaviour; rule 2 will change this so blocked escalates
    on every lane, not just the no-PR one.

    A fresh, on-target outcome file (matching the live PR head, written
    after dispatched_at) that also carries `"outcome": "blocked"` on a dead
    worker whose branch already has an open PR is not distinguished from
    having no outcome evidence at all: `fresh_completed_worker_outcome`
    refuses it (correctly, it is not a completion), but
    `handle_dead_worker_with_pr` has no separate check for a blocked
    declaration the way the no-PR lane does -- it just resets the issue to
    `rework_requested` and redispatches, exactly as it would with zero
    signal from the worker.
    """
    config, paths, fake_gh, dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="request_changes"
    )
    _write_outcome(
        paths,
        tmp_path,
        {
            "outcome": "blocked",
            "reason_kind": "ambiguous_scope",
            "detail": "needs a human to disambiguate",
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "abc123",  # matches the bed's live PR head
        },
        mtime=datetime.now(UTC),  # fresh: after dispatched_at (~1h ago)
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    # Current (disagreeing) behaviour: the blocked declaration is ignored --
    # the dead worker on an open PR is auto-reset to rework_requested exactly
    # as it would be with no outcome file at all.
    assert entry["status"] == "rework_requested"
    events = state.get("events", [])
    assert any(
        e.get("kind") == "orphaned_worker_recovered"
        and e["payload"].get("reason") == "dead_worker_with_request_changes"
        for e in events
    )
    # worker_declared_blocked is emitted only by the no-PR lane
    # (workflow.py); the with-PR lane never inspects outcome["outcome"].
    assert not any(e.get("kind") == "worker_declared_blocked" for e in events)


# ---------------------------------------------------------------------------
# Rule 3: pushed = commits on the remote; local-only commits = stranded
# (salvage; park on no-remote repos). Today the SAME fact (local commits
# not on the remote branch) is classified two different, contradictory
# ways depending on which lane sees it first.
# ---------------------------------------------------------------------------


def test_flip3_dead_worker_reap_salvage_pushes_local_only_commits(
    tmp_path: Path, monkeypatch
) -> None:
    """FLIP 3: current behaviour; rule 3 folds this and the sibling test below
    into one "stranded" fate, resolving today's contradiction.

    `_attempt_salvage` (via the main sweep loop's `inspection.ahead_count >
    0` local-worktree check) pushes a branch with only local, never-pushed
    commits and opens a PR from it -- treating local-only commits as
    immediately salvageable with no cross-check against the refuse-to-reset
    guard the sibling test below applies to the identical fact.
    """
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    config = OrchestratorConfig()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=tmp_path)
    monkeypatch.setattr("charlie_work.dead_worker_reap.push_branch", lambda *a, **k: (True, None))

    from charlie_work.workflow import _attempt_salvage

    salvaged, error = _attempt_salvage(
        gh=gh,
        config=config,
        repo_root=tmp_path,
        worktree_path=tmp_path,
        branch="agent/issue-9103",
        base_ref="main",
        issue_number=9103,
        active_labels=active_labels,
        issue_labels=issue_labels,
        state_file=state_file,
        failure_kind="unpublished_work",
        issue_title="Local-only commits, never pushed",
        write_gate=_wg(state_file),
    )

    assert salvaged is True
    assert error is None
    assert len(gh.prs_created) == 1
    state = load_state(state_file)
    events = [e for e in state["events"] if e["kind"] == "session_salvaged"]
    assert len(events) == 1
    assert events[0]["payload"]["issue_number"] == 9103


def test_flip3_worktree_refuse_to_reset_treats_local_commits_as_unsafe(tmp_path: Path) -> None:
    """FLIP 3: current behaviour; contrast with the sibling test above.

    `_worktree_refuse_to_reset_reason` (backing
    `misc_worker_dispatch._worktree_still_unsafe`) classifies the SAME fact
    -- local commits on a branch that are not on the remote -- as a
    judgment-class escalation (`WORKTREE_UNSAFE_KIND_LOCAL_COMMITS`, in
    `config.DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS`) that is
    deliberately NEVER auto-cleared, the opposite action from the salvage
    lane's auto-push above.
    """
    from charlie_work.worktree import (
        WORKTREE_UNSAFE_KIND_LOCAL_COMMITS,
        _worktree_refuse_to_reset_reason,
        _worktree_unsafe_kind_from_reason,
    )

    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    branch = "agent/issue-9103"
    _git(repo_root, "checkout", "-b", branch)
    (repo_root / "work.txt").write_text("local work\n", encoding="utf-8")
    _git(repo_root, "add", "work.txt")
    _git(repo_root, "commit", "-m", "local-only commit")
    _git(repo_root, "checkout", "main")

    reason = _worktree_refuse_to_reset_reason(repo_root, branch, "main", None)

    assert reason is not None
    assert "local commit" in reason
    assert _worktree_unsafe_kind_from_reason(reason) == WORKTREE_UNSAFE_KIND_LOCAL_COMMITS


# ---------------------------------------------------------------------------
# Rule 4: a fresh outcome file should beat the exit code; the exit code
# decides only without one. Today `exit_code == 0` short-circuits BEFORE
# the fresh-outcome check ever runs, in both PR-linked branches.
# ---------------------------------------------------------------------------


def _seed_dead_worker_fresh_outcome_and_clean_exit(
    tmp_path: Path, paths, sessions_dir: Path
) -> None:
    write_worker_terminal_status(
        sessions_dir / "issue-207.claude.terminal.json",
        pid=99999,
        exit_code=0,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:05:00Z",
        duration_seconds=300.0,
    )
    _write_outcome(
        paths,
        tmp_path,
        {"push_succeeded": True, "pr_created": False, "head_sha": "abc123"},
        mtime=datetime.now(UTC),
    )


def test_flip4_request_changes_exit0_discards_fresh_outcome(tmp_path: Path) -> None:
    """FLIP 4: current behaviour; rule 4 will change this to prefer the
    fresh outcome file over the recorded exit code.

    `orphaned_worker_sweep.handle_dead_worker_with_pr`'s `request_changes`
    branch checks `if terminal_exit_code == 0:` BEFORE calling
    `handle_dead_worker_completed_outcome` (which is what consults
    `fresh_completed_worker_outcome`) -- so a terminal record reporting a
    clean exit short-circuits straight to the `dead_worker_clean_exit_no_op`
    drift event, even though a fresh, on-target, confirmed-push outcome
    file sits right there in the worktree.
    """
    config, paths, fake_gh, dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="request_changes"
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    _seed_dead_worker_fresh_outcome_and_clean_exit(tmp_path, paths, sessions_dir)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    # Current (disagreeing) behaviour: exit_code == 0 wins, the fresh
    # outcome is never consulted.
    assert entry["status"] == "dispatched"
    events = state.get("events", [])
    assert any(
        e.get("kind") == "orphaned_worker_drift"
        and e["payload"].get("reason") == "dead_worker_clean_exit_no_op"
        for e in events
    )
    assert not any(e.get("kind") == "orphaned_worker_recovered" for e in events)


def test_flip4_approved_rework_exit0_discards_fresh_outcome(tmp_path: Path) -> None:
    """FLIP 4: current behaviour; contrast/companion to the test above.

    The `approved` + `rework_requested`-PR-state branch of
    `handle_dead_worker_with_pr` has its own, separately-written
    `if terminal_exit_code == 0:` short-circuit ahead of
    `handle_dead_worker_completed_outcome` -- the same disagreement,
    duplicated at a second call site rather than shared.
    """
    config, paths, fake_gh, dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="approved", pr_state_status="rework_requested"
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    _seed_dead_worker_fresh_outcome_and_clean_exit(tmp_path, paths, sessions_dir)

    _run_orphan_sweep(tmp_path, paths, config, fake_gh)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == "dispatched"
    events = state.get("events", [])
    assert any(
        e.get("kind") == "orphaned_worker_drift"
        and e["payload"].get("reason") == "dead_worker_clean_exit_no_op"
        for e in events
    )
    assert not any(e.get("kind") == "orphaned_worker_recovered" for e in events)


# ---------------------------------------------------------------------------
# Rule 7: terminal-record precedence and the `terminal or worktree`
# empty-dict fall-through are replaced by rule 1. Two sites read the same
# two sources with opposite fall-through semantics for a present-but-empty
# terminal outcome.
# ---------------------------------------------------------------------------


def test_flip7_workflow_empty_terminal_outcome_falls_through_to_worktree(tmp_path: Path) -> None:
    """FLIP 7: current behaviour; rule 1 replaces this fall-through.

    `workflow.py`'s no-PR orphan lane computes
    `worker_outcomes[issue_number] = terminal_outcome or worktree_outcome`.
    `{}` is falsy in Python, so a terminal record that IS present but
    reports an empty `worker_outcome` dict silently falls through to
    whatever the worktree file happens to still contain -- here, a stale
    `blocked` declaration -- rather than being treated as authoritative
    (if empty) or triggering a "no signal" path.
    """
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20, max_auto_redispatch=3),
    )
    issue_number = 9107
    branch = "agent/issue-9107-test"
    state_file, sessions_dir = _seed_no_pr_dispatched_issue(tmp_path, config, issue_number, branch)

    write_worker_terminal_status(
        sessions_dir / f"issue-{issue_number}.claude.terminal.json",
        pid=99999,
        exit_code=1,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:05:00Z",
        duration_seconds=300.0,
        worker_outcome={},
    )

    worktrees_dir = resolved_layout(config, tmp_path).worktrees
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)
    (worktree_path / WORKER_OUTCOME_FILENAME).write_text(
        json.dumps(
            {
                "outcome": "blocked",
                "reason_kind": "cross_repo_scope",
                "detail": "targets a different repo",
            }
        ),
        encoding="utf-8",
    )

    fake_gh = _no_pr_fake_gh(tmp_path, config, issue_number)

    with (
        patch("charlie_work.workflow._worker_pid_alive", return_value=False),
        patch("charlie_work.workflow.remote_branch_head_sha", return_value=None),
        patch("charlie_work.workflow.remote_branch_ahead_count", return_value=(0, None)),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir, state_file, config, fake_gh, write_gate=_wg(state_file)
        )

    st = load_state(state_file)
    entry = st["issues"][str(issue_number)]
    # Current (disagreeing) behaviour: the present-but-empty terminal dict
    # is falsy, so `or` silently falls through to the worktree file's real
    # (blocked) content, and the sweep escalates on it.
    assert entry["status"] == "escalated"
    assert entry.get("escalation_reason") == "worker_declared_blocked"


def test_flip7_read_rework_outcome_returns_empty_terminal_dict_unlike_workflow(
    tmp_path: Path,
) -> None:
    """FLIP 7: current behaviour; contrast with the workflow.py site above.

    `rework_outcome._read_rework_outcome` reads the SAME two sources
    (durable terminal record, worktree fallback) but with
    `isinstance(outcome, dict): return outcome` -- it returns the
    present-but-empty terminal dict AS IS and never falls through to the
    worktree file's real content, the opposite polarity from
    `workflow.py`'s `terminal_outcome or worktree_outcome`.
    """
    from charlie_work.rework_outcome import _read_rework_outcome

    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    worktrees_dir = tmp_path / "worktrees"
    branch = "agent/issue-9108"
    write_worker_terminal_status(
        sessions_dir / "issue-9108.claude.terminal.json",
        pid=1,
        exit_code=1,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:05:00Z",
        duration_seconds=1.0,
        worker_outcome={},
    )
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)
    (worktree_path / WORKER_OUTCOME_FILENAME).write_text(
        json.dumps({"push_succeeded": True, "pr_created": False, "head_sha": "realcontent"}),
        encoding="utf-8",
    )

    result = _read_rework_outcome(sessions_dir, tmp_path, worktrees_dir, 9108, branch)

    assert result == {}


# ---------------------------------------------------------------------------
# Rule 8: `pr_created` should distinguish completed (PR exists) from
# pushed-without-PR consistently. Today `fresh_completed_worker_outcome`
# never inspects it at all.
# ---------------------------------------------------------------------------


def test_flip8_fresh_completed_outcome_ignores_pr_created_true(tmp_path: Path) -> None:
    """FLIP 8: current behaviour; rule 8 will make `pr_created` a distinguisher.

    `fresh_completed_worker_outcome` (rework_outcome.py) checks
    `push_succeeded`, `head_sha`, mtime freshness and the `blocked` sentinel
    -- but never reads `pr_created` at all. An outcome reporting
    `pr_created: True` (structurally unexpected -- workers carry no `gh`
    credential -- but not rejected) is still accepted as valid completed-
    outcome evidence exactly like `pr_created: False`.
    """
    from charlie_work.rework_outcome import fresh_completed_worker_outcome

    worktrees_dir = tmp_path / "worktrees"
    branch = "agent/issue-9109"
    dispatched_at = datetime.now(UTC) - timedelta(hours=1)
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome_path = worktree_path / WORKER_OUTCOME_FILENAME
    outcome_path.write_text(
        json.dumps({"push_succeeded": True, "pr_created": True, "head_sha": "abc999"}),
        encoding="utf-8",
    )
    fresh_ts = datetime.now(UTC).timestamp()
    os.utime(outcome_path, (fresh_ts, fresh_ts))

    result = fresh_completed_worker_outcome(
        worktree_path, live_head_sha="abc999", dispatched_at=dispatched_at
    )

    assert result is not None
    assert result["pr_created"] is True


# ---------------------------------------------------------------------------
# Rule 9: remote vs local ahead-count split per rule 3. The same-named
# "ahead count" means two different things at two different sites.
# ---------------------------------------------------------------------------


def _repo_with_pushed_and_local_commits(tmp_path: Path, branch: str) -> Path:
    _remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    _git(repo_root, "checkout", "-b", branch)
    for i in range(2):
        (repo_root / f"pushed_{i}.txt").write_text(f"{i}\n", encoding="utf-8")
        _git(repo_root, "add", f"pushed_{i}.txt")
        _git(repo_root, "commit", "-m", f"pushed commit {i}")
    _git(repo_root, "push", "-u", "origin", branch)
    (repo_root / "local_only.txt").write_text("local\n", encoding="utf-8")
    _git(repo_root, "add", "local_only.txt")
    _git(repo_root, "commit", "-m", "local-only commit, never pushed")
    return repo_root


def test_flip9_remote_branch_ahead_count_counts_only_pushed_commits(tmp_path: Path) -> None:
    """FLIP 9: current behaviour; rule 9 makes the remote/local split explicit.

    `remote_branch_ahead_count` counts commits on `origin/{branch}` only --
    the local-only, never-pushed commit is invisible to it.
    """
    from charlie_work.worktree import remote_branch_ahead_count

    branch = "agent/issue-9110"
    repo_root = _repo_with_pushed_and_local_commits(tmp_path, branch)

    ahead, error = remote_branch_ahead_count(repo_root, branch, "main")

    assert error is None
    assert ahead == 2


def test_flip9_local_worktree_ahead_count_counts_unpushed_commits_too(tmp_path: Path) -> None:
    """FLIP 9: current behaviour; contrast with the remote-only test above.

    `inspect_worktree_state`'s `ahead_count` is computed purely locally
    (`git merge-base main HEAD` + `git rev-list --count`), so it counts the
    local-only commit the remote-based function above cannot see -- the
    same branch reads as "2 ahead" and "3 ahead" depending on which site is
    asked.
    """
    from charlie_work.worktree import inspect_worktree_state

    branch = "agent/issue-9110"
    repo_root = _repo_with_pushed_and_local_commits(tmp_path, branch)

    inspection = inspect_worktree_state(repo_root, "main")

    assert inspection.ahead_count == 3


# ---------------------------------------------------------------------------
# Rule 6 (preserve): throttle kind is classified fresh at fate time and
# persisted as evidence; persisted dead_worker_failure_kind is read only
# through the fate. No two sites are documented as disagreeing on this
# today, but the exact mechanism -- a direct read of the persisted field --
# is what the new module's single fate accessor must keep behaving like.
# ---------------------------------------------------------------------------


def test_preserve_rule6_throttle_exemption_reads_failure_kind_directly(tmp_path: Path) -> None:
    """Preserve: the #654 timed backstop's throttle exemption reads
    `entry["dead_worker_failure_kind"]` directly (via
    `is_provider_throttle_failure`) and, while `state["throttled_until"]`
    is still in the future, returns immediately with NO side effects --
    it does not touch the rearm counter, `orphan_drift_at`, or emit any
    event. The new fate module must keep this exemption reachable the same
    way even once the field is read only through a fate accessor.
    """
    from charlie_work.orphaned_worker_sweep import maybe_reap_dead_dispatched_worker

    now = datetime.now(UTC)
    orphan_drift_at = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    entry: dict = {
        "dead_worker_failure_kind": "rate_limited",
        "orphan_drift_at": orphan_drift_at,
    }
    state: dict = {
        "issues": {},
        "throttled_until": (now + timedelta(minutes=30)).isoformat().replace("+00:00", "Z"),
    }
    sweep_events: list = []

    new_state, escalated = maybe_reap_dead_dispatched_worker(
        state=state,
        entry=entry,
        issue_number=9106,
        sessions_dir=tmp_path / "sessions",
        pr_data=None,
        dead_dispatched_reap_minutes=60,
        now=now,
        sweep_events=sweep_events,
        max_throttle_rearms=3,
    )

    assert escalated is False
    assert new_state is state
    # No side effects: the throttled_until-in-future branch returns before
    # ever touching the rearm counter or re-arming orphan_drift_at.
    assert "throttle_reap_rearm_count" not in entry
    assert entry["orphan_drift_at"] == orphan_drift_at
    assert sweep_events == []


# ---------------------------------------------------------------------------
# A2 (preserve): claude_code.is_worker_alive and devin_shell.is_session_alive
# are duplicate wrappers around process_utils.is_pid_alive. The new
# Adapter-liveness seam must keep them agreeing on the same input.
# ---------------------------------------------------------------------------


def test_preserve_a2_is_worker_alive_and_is_session_alive_agree(tmp_path: Path) -> None:
    """Preserve: the two adapter-specific liveness wrappers agree.

    `claude_code.is_worker_alive` and `devin_shell.is_session_alive` are
    separately-defined but byte-identical wrappers: `pid is None or pid <=
    0` short-circuits to `False`, otherwise both delegate to the same
    `process_utils.is_pid_alive`. The new module deletes the duplication
    (per the plan); this pins that they agree today, on the same inputs.
    """
    from charlie_work.claude_code import ClaudeWorkerRecord, is_worker_alive
    from charlie_work.devin_shell import SessionRecord, is_session_alive

    common = dict(
        issue_number=1,
        branch="agent/issue-1",
        worktree_path=str(tmp_path),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("echo", "hi"),
        started_at="2026-01-01T00:00:00Z",
        log_path=str(tmp_path / "log.txt"),
    )

    # pid is None -> both short-circuit to False without probing.
    assert is_worker_alive(ClaudeWorkerRecord(pid=None, **common)) is False
    assert is_session_alive(SessionRecord(pid=None, **common)) is False

    # pid <= 0 -> same short-circuit.
    assert is_worker_alive(ClaudeWorkerRecord(pid=0, **common)) is False
    assert is_session_alive(SessionRecord(pid=0, **common)) is False

    # A pid confirmed dead (no such process) -> both delegate to the SAME
    # process_utils.is_pid_alive and must agree with each other, whatever
    # this host's answer is.
    dead_pid = 999_999_937
    claude_dead = ClaudeWorkerRecord(pid=dead_pid, process_start_time=None, **common)
    devin_dead = SessionRecord(pid=dead_pid, process_start_time=None, **common)
    assert is_worker_alive(claude_dead) == is_session_alive(devin_dead)
