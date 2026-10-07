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

from _orphan_sweep_fixtures import _dead_worker_rework_bed, _run_orphan_sweep, _write_outcome
from _salvage_fixtures import _SalvageTestGitHub, _salvage_labels
from _worktree_fixtures import _git

from charlie_work.config import (
    WORKER_OUTCOME_FILENAME,
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.paths import resolved_layout
from charlie_work.process_utils import write_worker_terminal_status
from charlie_work.state import load_state
from charlie_work.worktree import worktree_path_for_branch
from _worker_fate_characterization_fixtures import (
    _no_pr_fake_gh,
    _seed_no_pr_dispatched_issue,
    _wg,
)
from _host_fixtures import host_probe


# ---------------------------------------------------------------------------
# Rule 1: only evidence from the current dispatch should count (outcome file
# / terminal record newer than dispatched_at, head_sha matching the live
# head). Two sites currently apply NO such freshness gate at all.
# ---------------------------------------------------------------------------


def test_flip1_a9_workflow_uses_stale_worker_outcome_from_prior_dispatch(
    tmp_path: Path, monkeypatch
) -> None:
    """FLIP 1: was `worker_outcomes[issue_number] = terminal_outcome or
    worktree_outcome` with no `dispatched_at` comparison anywhere (a
    `.worker-outcome.json` left over from a PRIOR dispatch of this
    issue/branch was accepted as proof the current session pushed, and a
    PR was opened from it); now `workflow.py`'s no-PR orphan lane builds
    real `worker_fate.TerminalEvidence`/`OutcomeEvidence` and arbitrates
    them through `resolve_fate`, whose freshness step (rule 1) rejects any
    candidate whose `written_at` does not postdate `dispatched_at` (rule
    1).
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
        host_probe(monkeypatch, alive=False),
        patch("charlie_work.workflow.remote_branch_head_sha", return_value=None),
        patch("charlie_work.workflow.remote_branch_ahead_count", return_value=(0, None)),
    ):
        _detect_and_handle_orphaned_workers(
            sessions_dir, state_file, config, fake_gh, write_gate=_wg(state_file)
        )

    st = load_state(state_file)
    entry = st["issues"][str(issue_number)]
    # New (rule 1) behaviour: the outcome predates dispatched_at, so
    # resolve_fate's freshness step rejects it -- no PR is opened and the
    # issue is left for the dead-PID lane to handle on its own terms.
    assert entry["status"] == "dispatched"
    assert entry.get("pr_number") is None
    events = st.get("events", [])
    assert not any(e.get("kind") == "worker_handoff_pr_opened" for e in events)


def test_flip1_a5_live_handoff_finalize_ignores_dispatched_at(tmp_path: Path) -> None:
    """FLIP 1: was no `dispatched_at` comparison at all in
    `collect_stale_live_handoff_pids` (only an age-vs-`now` check), so a
    leftover outcome file from a PREVIOUS dispatch of this branch still
    counted as valid completion evidence once merely old enough; now the
    function resolves each candidate's fate through `worker_fate`, whose
    freshness step (rule 1) rejects an outcome whose `written_at` predates
    the live entry's `dispatched_at`, regardless of how old it is by wall
    clock.
    """
    from charlie_work.dead_worker_sweep.live_handoff import collect_stale_live_handoff_pids

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
        sessions_dir=tmp_path / "sessions",
    )

    # New (rule 1) behaviour: the outcome predates the current dispatch by
    # ~1h50m, so it is rejected as a leftover from a prior run -- no
    # candidate is produced for it.
    assert candidates == {}


# ---------------------------------------------------------------------------
# Rule 5: live PID + fresh outcome = completed immediately; the KILL waits
# for watchdog.worker_outcome_finalize_minutes, but ROUTING should not.
# Today the same threshold gates both.
# ---------------------------------------------------------------------------


def test_flip5_live_handoff_finalize_withholds_fresh_outcome_until_threshold(
    tmp_path: Path,
) -> None:
    """FLIP 5: was `if outcome_age <= timedelta(minutes=
    worker_outcome_finalize_minutes): continue` -- a live PID with a
    fresh, on-target, confirmed-push outcome was NOT routed yet, because
    the same threshold that governs when the stall watchdog is allowed to
    kill the process also gated when the completion evidence was even
    considered; now a live PID with a declared push that survives rule 1's
    freshness gate (`written_at` after `dispatched_at`) routes immediately
    regardless of `worker_outcome_finalize_minutes` -- that threshold now
    only gates the kill decision elsewhere (rule 5).
    """
    from charlie_work.dead_worker_sweep.live_handoff import collect_stale_live_handoff_pids

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
        sessions_dir=tmp_path / "sessions",
    )

    # New (rule 5) behaviour: a fresh, on-target, declared-push outcome
    # routes immediately -- 2 minutes old is well inside the 15-minute
    # `worker_outcome_finalize_minutes` threshold, but that threshold no
    # longer gates this routing decision.
    assert 9105 in candidates
    assert candidates[9105]["worker_outcome"]["head_sha"] == "cafefeed"


def test_b9_live_handoff_finalize_omitted_pr_created_is_not_a_confirmed_no_pr(
    tmp_path: Path,
) -> None:
    """B9 (wf-review-opus.md): an outcome that OMITS ``pr_created`` entirely
    must not be finalized as if it had explicitly declared no PR.

    The legacy check required ``pr_created is False`` before routing. The
    worker-fate refactor loosened this to ``pr_created is not True``, which
    also admits ``None`` -- an incomplete self-report now gets finalized the
    same as an explicit ``false``. That widening was never one of the nine
    reviewed flips; this pins the restored strict gate.
    """
    from charlie_work.dead_worker_sweep.live_handoff import collect_stale_live_handoff_pids

    now = datetime.now(UTC)
    branch = "agent/issue-9106"
    worktrees_dir = tmp_path / "worktrees"
    worktree_path = worktree_path_for_branch(tmp_path, branch, worktrees_dir)
    worktree_path.mkdir(parents=True, exist_ok=True)
    outcome_path = worktree_path / WORKER_OUTCOME_FILENAME
    # pr_created is left out entirely -- not `false`, not `true`.
    outcome_path.write_text(
        json.dumps({"push_succeeded": True, "head_sha": "b01dface"}),
        encoding="utf-8",
    )
    fresh_ts = (now - timedelta(minutes=2)).timestamp()
    os.utime(outcome_path, (fresh_ts, fresh_ts))

    live_pid_entries = {9106: {"branch_name": branch, "worker_pid": 4242}}

    candidates = collect_stale_live_handoff_pids(
        live_pid_entries,
        worker_outcome_finalize_minutes=15,
        repo_root=tmp_path,
        worktrees_dir=worktrees_dir,
        now=now,
        sessions_dir=tmp_path / "sessions",
    )

    assert candidates == {}, "an omitted pr_created must not be treated as a confirmed no-PR push"


# ---------------------------------------------------------------------------
# Rule 2: a fresh `outcome: blocked` should beat push flags everywhere.
# The no-PR lane already honours this (test_worker_declared_blocked.py);
# the with-PR lane's `fresh_completed_worker_outcome` correctly refuses to
# treat `blocked` as *completed*, but the caller then cannot distinguish
# "no evidence at all" from "an explicit blocked declaration" -- both fall
# through to the SAME generic redispatch, silently retrying a worker that
# said it could not do the task instead of escalating.
# ---------------------------------------------------------------------------


def test_flip2_with_pr_fresh_blocked_outcome_falls_through_to_redispatch(
    tmp_path: Path, monkeypatch
) -> None:
    """FLIP 2: was a fresh, on-target `blocked` outcome on the with-PR lane
    silently indistinguishable from no evidence at all (auto-reset to
    `rework_requested` and redispatched); now the with-PR lane escalates it
    the same way the no-PR lane always has (rule 2).

    A fresh, on-target outcome file (matching the live PR head, written
    after dispatched_at) that also carries `"outcome": "blocked"` on a dead
    worker whose branch already has an open PR is now distinguished from
    having no outcome evidence at all: `handle_dead_worker_with_pr` reads
    `blocked_worker_outcome` directly (the same helper the no-PR lane uses)
    and escalates via `_escalate_issue` instead of falling through to the
    generic `rework_requested` reset.
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

    _run_orphan_sweep(tmp_path, paths, config, fake_gh, monkeypatch=monkeypatch)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    # New behaviour: the blocked declaration escalates the issue instead of
    # silently redispatching it.
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "worker_declared_blocked"
    # State and labels must move together in the SAME pass: the post-lock
    # `escalated` edge adds the operator-queue label and drops the active one.
    # (Only `_repair_escalated_labels` would otherwise converge it, capped per
    # pass -- a state/label split, the CLAUDE.md "state lives in labels AND
    # state.json" invariant.)
    assert (207, config.labels.operator_queue) in fake_gh.labels_added
    assert (207, config.labels.in_progress) in fake_gh.labels_removed
    events = state.get("events", [])
    assert any(
        e.get("kind") == "worker_declared_blocked"
        and e["payload"].get("reason_kind") == "ambiguous_scope"
        for e in events
    )
    # The generic no-signal reset path must not also fire.
    assert not any(
        e.get("kind") == "orphaned_worker_recovered"
        and e["payload"].get("reason") == "dead_worker_with_request_changes"
        for e in events
    )


def test_flip2_with_pr_fresh_blocked_outcome_and_clean_exit_still_escalates(
    tmp_path: Path, monkeypatch
) -> None:
    """B2 (wf-review-opus.md): FLIP 2 must fire on a clean (exit code 0)
    exit too -- the normal way a claude-code/api worker ends a blocked
    task is to write the outcome file and then exit 0, not crash. Before
    this fix, the blocked check lived only in the `else` of `terminal_
    exit_code == 0`, so this exact population (blocked declaration + exit
    0) took the exit-0 no-op drift branch and the blocked declaration was
    never read at all.
    """
    config, paths, fake_gh, dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="request_changes"
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
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
        {
            "outcome": "blocked",
            "reason_kind": "ambiguous_scope",
            "detail": "needs a human to disambiguate",
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "abc123",
        },
        mtime=datetime.now(UTC),
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh, monkeypatch=monkeypatch)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "worker_declared_blocked"
    # State and labels must move together in the SAME pass: the post-lock
    # `escalated` edge adds the operator-queue label and drops the active one.
    # (Only `_repair_escalated_labels` would otherwise converge it, capped per
    # pass -- a state/label split, the CLAUDE.md "state lives in labels AND
    # state.json" invariant.)
    assert (207, config.labels.operator_queue) in fake_gh.labels_added
    assert (207, config.labels.in_progress) in fake_gh.labels_removed
    events = state.get("events", [])
    assert any(
        e.get("kind") == "worker_declared_blocked"
        and e["payload"].get("reason_kind") == "ambiguous_scope"
        for e in events
    )
    # Neither the exit-0 no-op drift nor the generic reset must fire instead.
    assert not any(
        e.get("kind") == "orphaned_worker_drift"
        and e["payload"].get("reason") == "dead_worker_clean_exit_no_op"
        for e in events
    )
    assert not any(e.get("kind") == "orphaned_worker_recovered" for e in events)


def test_flip2_approved_rework_fresh_blocked_outcome_and_clean_exit_still_escalates(
    tmp_path: Path, monkeypatch
) -> None:
    """B2 companion for the second call site (the `approved` +
    `rework_requested`-PR-state branch of `handle_dead_worker_with_pr`,
    which duplicates the same blocked-check-inside-the-exit-0-else bug at
    its own, separately-written call site).
    """
    config, paths, fake_gh, dispatched_at = _dead_worker_rework_bed(
        tmp_path, decision="approved", pr_state_status="rework_requested"
    )
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
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
        {
            "outcome": "blocked",
            "reason_kind": "ambiguous_scope",
            "detail": "needs a human to disambiguate",
            "push_succeeded": True,
            "pr_created": False,
            "head_sha": "abc123",
        },
        mtime=datetime.now(UTC),
    )

    _run_orphan_sweep(tmp_path, paths, config, fake_gh, monkeypatch=monkeypatch)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "worker_declared_blocked"
    # State and labels must move together in the SAME pass: the post-lock
    # `escalated` edge adds the operator-queue label and drops the active one.
    # (Only `_repair_escalated_labels` would otherwise converge it, capped per
    # pass -- a state/label split, the CLAUDE.md "state lives in labels AND
    # state.json" invariant.)
    assert (207, config.labels.operator_queue) in fake_gh.labels_added
    assert (207, config.labels.in_progress) in fake_gh.labels_removed
    events = state.get("events", [])
    assert any(
        e.get("kind") == "worker_declared_blocked"
        and e["payload"].get("reason_kind") == "ambiguous_scope"
        for e in events
    )
    assert not any(
        e.get("kind") == "orphaned_worker_drift"
        and e["payload"].get("reason") == "dead_worker_clean_exit_no_op"
        for e in events
    )
    assert not any(e.get("kind") == "orphaned_worker_recovered" for e in events)


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
    monkeypatch.setattr(
        "charlie_work.dead_worker_sweep.effects_pr.push_branch", lambda *a, **k: (True, None)
    )

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
    """FLIP 3: was a permanent "unsafe" verdict for local-only commits, the
    opposite action from the salvage lane's auto-push above; now rule 3
    applies to both lanes.

    `_worktree_refuse_to_reset_reason` still reports the local-only commits
    (`WORKTREE_UNSAFE_KIND_LOCAL_COMMITS`) -- that read-only probe is
    unchanged -- but `misc_worker_dispatch._worktree_still_unsafe`, its
    de-escalation gate, now salvages them (ff-only push to the reachable
    origin) and clears only because the re-check is then clean (`None`).
    """
    from _unescalate_fixtures import _stranded_state, _stranded_worktree_bed

    from charlie_work.worktree import (
        WORKTREE_UNSAFE_KIND_LOCAL_COMMITS,
        _worktree_refuse_to_reset_reason,
        _worktree_unsafe_kind_from_reason,
    )

    branch = "agent/issue-9103"
    app, repo_root, wt_path, remote = _stranded_worktree_bed(tmp_path, branch)

    reason = _worktree_refuse_to_reset_reason(repo_root, branch, "", wt_path)
    assert reason is not None
    assert "local commit" in reason
    assert _worktree_unsafe_kind_from_reason(reason) == WORKTREE_UNSAFE_KIND_LOCAL_COMMITS

    assert app._worktree_still_unsafe(123, _stranded_state(branch)) is None

    # Salvaged, not merely relabelled: the branch is now on the remote.
    assert (
        _git(remote, "rev-parse", f"refs/heads/{branch}").stdout.strip()
        == _git(wt_path, "rev-parse", "HEAD").stdout.strip()
    )
    assert _worktree_refuse_to_reset_reason(repo_root, branch, "", wt_path) is None
